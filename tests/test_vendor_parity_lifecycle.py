"""Vendor-parity lifecycle behaviour pinned at the adapter boundary.

These four facts were closed at evidence level E3 (binary consumer/caller
trace of the installed reference build) and must hold in our integration:

1. the opacity reset fires on the natural step every 300 steps, so the first
   reset after the 500-step start lands on 600, not 599 or 601;
2. the split/clone boundary is a strict comparison of the largest axis with
   0.2 m: exactly 0.2 clones, anything above splits;
3. there is no opacity birth gate: a translucent parent with a high projected
   gradient is still cloned;
4. the reset caps probabilities above 0.2 to 0.2, leaves lower ones alone,
   and zeroes the optimizer moments of the opacity group.
"""

from __future__ import annotations

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from cloudstudio_3dgs.training.default_strategy_adapter import DefaultStrategyAdapter


def _adapter(**overrides):
    fields = dict(
        scene_scale=10.0,
        refine_start_iter=500,
        refine_stop_iter=20000,
        refine_every=100,
        reset_every=300,
        grow_grad2d=0.00015,
        prune_opa=0.1,
        split_scale_m=0.2,
        prune_scale_m=0.2,
        exact_mipmap_lifecycle=True,
        prune_opa_late=0.05,
        prune_switch_step=10000,
        reset_opacity_cap=0.2,
    )
    fields.update(overrides)
    return DefaultStrategyAdapter(**fields)


def _params(scales_m, opacities_prob):
    import torch

    n = len(scales_m)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(n, 3)),
            "scales": torch.nn.Parameter(torch.tensor(scales_m).log()),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)),
            "opacities": torch.nn.Parameter(torch.tensor(opacities_prob).logit()),
            "colors": torch.nn.Parameter(torch.zeros(n, 3)),
        }
    )


def _optimizers(params):
    import torch

    return {name: torch.optim.Adam([p], lr=1e-3) for name, p in params.items()}


def _info(n):
    """Minimal rasterizer info for one camera: the projected-gradient carrier
    with a zero gradient, so the post-backward step accumulates nothing and
    only the lifecycle bookkeeping (reset) is exercised."""
    import torch

    means2d = torch.zeros(1, n, 2, requires_grad=True)
    means2d.grad = torch.zeros(1, n, 2)
    return {
        "width": 100,
        "height": 100,
        "n_cameras": 1,
        "radii": torch.full((1, n, 2), 5, dtype=torch.int32),
        "gaussian_ids": None,
        "means2d": means2d,
    }


def _state(n, grad=0.001, radii=0.01):
    import torch

    return {
        "grad2d": torch.full((n,), grad),
        "count": torch.ones(n),
        "radii": torch.full((n,), radii),
        "scene_scale": 10.0,
    }


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class VendorParityLifecycleTests(unittest.TestCase):
    def test_reset_fires_on_natural_step_600_only(self) -> None:
        import torch

        adapter = _adapter()
        for step, expected in ((599, False), (600, True), (601, False), (900, True)):
            params = _params([[0.01] * 3] * 4, [0.9] * 4)
            optimizers = _optimizers(params)
            state = _state(4, grad=0.0)
            adapter._step_post_backward_mipmap(
                params=params, optimizers=optimizers, state=state, step=step, info=_info(4)
            )
            after = torch.sigmoid(params["opacities"].detach()).max().item()
            self.assertEqual(after <= 0.2 + 1e-6, expected, f"step {step}")

    def test_split_clone_boundary_is_strict_at_0p2_m(self) -> None:
        adapter = _adapter()
        params = _params([[0.2, 0.01, 0.01], [0.2000001, 0.01, 0.01], [0.05, 0.05, 0.05]], [0.5] * 3)
        clone_count, split_count = adapter._grow_mipmap(params, _optimizers(params), _state(3))
        self.assertEqual((clone_count, split_count), (2, 1))

    def test_no_opacity_birth_gate(self) -> None:
        adapter = _adapter()
        # Two eligible parents, one nearly transparent: both are cloned.
        params = _params([[0.05] * 3] * 2, [0.05, 0.9])
        clone_count, split_count = adapter._grow_mipmap(params, _optimizers(params), _state(2))
        self.assertEqual((clone_count, split_count), (2, 0))

    def test_reset_caps_only_above_0p2_and_zeroes_opacity_moments(self) -> None:
        import torch

        adapter = _adapter()
        # 0.15 sits above the 0.1 cull that runs on the same refine step and
        # below the 0.2 cap, so it must come through untouched.
        params = _params([[0.01] * 3] * 3, [0.9, 0.2, 0.15])
        optimizers = _optimizers(params)
        # Give the opacity optimizer a non-zero moment first.
        params["opacities"].grad = torch.ones_like(params["opacities"])
        optimizers["opacities"].step()
        moments_before = optimizers["opacities"].state[params["opacities"]]["exp_avg"].abs().sum().item()
        self.assertGreater(moments_before, 0.0)
        adapter._step_post_backward_mipmap(
            params=params, optimizers=optimizers, state=_state(3, grad=0.0), step=600, info=_info(3)
        )
        after = torch.sigmoid(params["opacities"].detach())
        self.assertAlmostEqual(float(after[0]), 0.2, places=5)
        self.assertLessEqual(float(after[1]), 0.2 + 1e-6)
        self.assertAlmostEqual(float(after[2]), 0.15, places=2)  # one 1e-3 Adam step moved it slightly
        opt_state = optimizers["opacities"].state[params["opacities"]]
        self.assertEqual(float(opt_state["exp_avg"].abs().sum()), 0.0)
        self.assertEqual(float(opt_state["exp_avg_sq"].abs().sum()), 0.0)

    def test_reset_keeps_adam_step_unless_the_audit_knob_restarts_it(self) -> None:
        import torch

        for restart in (False, True):
            adapter = _adapter(reset_adam_step=restart)
            params = _params([[0.01] * 3] * 3, [0.9, 0.2, 0.15])
            optimizers = _optimizers(params)
            params["opacities"].grad = torch.ones_like(params["opacities"])
            optimizers["opacities"].step()
            adapter._step_post_backward_mipmap(
                params=params, optimizers=optimizers, state=_state(3, grad=0.0), step=600, info=_info(3)
            )
            step = optimizers["opacities"].state[params["opacities"]]["step"]
            self.assertEqual(float(step), 0.0 if restart else 1.0)


if __name__ == "__main__":
    unittest.main()
