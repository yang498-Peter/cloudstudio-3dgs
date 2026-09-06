"""Cull-only phase after growth stops (opt-in, default vendor parity).

house0305 checkpoints carried 46-48% of their population below opacity 0.1,
almost all born in the last growth window and frozen there because the
recovered lifecycle does nothing after refine_stop_iter. This pins the knob:
off by default (nothing happens after refine stop), on it culls at full
strength on its own cadence without growing or resetting.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised on the CPU channel
    torch = None

from cloudstudio_3dgs.training.default_strategy_adapter import DefaultStrategyAdapter


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class PostRefineCullTests(unittest.TestCase):
    def _adapter(self, **overrides):
        settings = dict(
            scene_scale=10.0,
            refine_start_iter=500,
            refine_stop_iter=2000,
            refine_every=100,
            reset_every=300,
            grow_grad2d=0.00015,
            prune_opa=0.1,
            split_scale_m=0.2,
            prune_scale_m=0.2,
            exact_mipmap_lifecycle=True,
            lifecycle_execution_order="pre_optimizer_vendor",
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=100000,
            reset_opacity_cap=0.2,
            relaxed_cull_when_no_growth=True,
        )
        settings.update(overrides)
        return DefaultStrategyAdapter(**settings)

    def _population(self):
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(3, 3)),
                "scales": torch.nn.Parameter(torch.full((3, 3), 0.05).log()),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
                ),
                # Two live rows and one at 0.01: below prune_opa 0.1 but above
                # the relaxed x0.25 threshold, so a relaxed cull would spare it.
                "opacities": torch.nn.Parameter(
                    torch.tensor([0.3, 0.3, 0.03]).logit()
                ),
                "colors": torch.nn.Parameter(torch.zeros(3, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.zeros(3),
            "count": torch.zeros(3),
            "radii": torch.zeros(3),
            "scene_scale": 10.0,
        }
        return params, optimizers, state

    def _step(self, adapter, params, optimizers, state, step):
        adapter.step_post_backward(
            params=params, optimizers=optimizers, state=state, step=step, info={}
        )

    def test_default_does_nothing_after_refine_stop(self):
        adapter = self._adapter()
        params, optimizers, state = self._population()
        self._step(adapter, params, optimizers, state, 2100)
        self.assertEqual(len(params["means"]), 3)
        self.assertIsNone(adapter.last_lifecycle_event)

    def test_opt_in_culls_on_its_cadence_only(self):
        adapter = self._adapter(post_refine_cull_every=100)
        params, optimizers, state = self._population()
        self._step(adapter, params, optimizers, state, 2150)
        self.assertEqual(len(params["means"]), 3, "off-cadence step must not cull")
        self._step(adapter, params, optimizers, state, 2100)
        self.assertEqual(len(params["means"]), 2)
        event = adapter.last_lifecycle_event
        self.assertEqual(event["kind"], "post_refine_cull")
        self.assertEqual(event["cull_count"], 1)
        self.assertEqual(event["clone_count"], 0)
        self.assertFalse(event["opacity_reset"])
        self.assertAlmostEqual(event["cull_opacity_threshold"], 0.1)
        # Full-strength threshold: the relaxed no-growth branch would have kept
        # the 0.03 row (0.1 x 0.25 = 0.025).
        self.assertEqual(len(optimizers["means"].param_groups[0]["params"][0]), 2)

    def test_window_stops_culling_after_until(self):
        adapter = self._adapter(post_refine_cull_every=100, post_refine_cull_until=2100)
        params, optimizers, state = self._population()
        self._step(adapter, params, optimizers, state, 2200)
        self.assertEqual(len(params["means"]), 3, "past the window nothing may cull")
        self.assertIsNone(adapter.last_lifecycle_event)
        self._step(adapter, params, optimizers, state, 2100)
        self.assertEqual(len(params["means"]), 2, "the window's last step still culls")
        self.assertEqual(adapter.state_dict()["post_refine_cull_until"], 2100)
        with self.assertRaises(ValueError):
            self._adapter(post_refine_cull_until=2100)

    def test_knob_is_recorded_and_validated(self):
        adapter = self._adapter(post_refine_cull_every=250)
        self.assertEqual(adapter.state_dict()["post_refine_cull_every"], 250)
        self.assertIsNone(self._adapter().state_dict()["post_refine_cull_every"])
        with self.assertRaises(ValueError):
            self._adapter(post_refine_cull_every=0)


if __name__ == "__main__":
    unittest.main()
