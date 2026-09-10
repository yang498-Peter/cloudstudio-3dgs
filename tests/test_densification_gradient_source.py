"""Pin ``densification_gradient_source="rgb_only"`` before it becomes a control arm.

The trainer optimizes the total loss but scores densification from the
photometric loss alone, in one step, with two backward passes and a snapshot
of the means2d criterion gradient between them. Everything here runs on CPU
tensors against a tiny differentiable stand-in for the rasterizer: the
projected positions are a non-leaf tensor (retained like gsplat's means2d),
the "image" is a soft splat of colours, and a depth-like term shares that
projection so the geometry supervision has a real path into means2d - which
is exactly the leak the rgb_only path exists to keep out of the criterion.

The topology test drives the adapter's own preserve/restore helpers through
gsplat's pure-torch ``duplicate``/``remove`` ops, so gradient rows, Adam
moments and lineage are checked against the same index remap the trainer
relies on in the pre-optimizer vendor order.
"""

from __future__ import annotations

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised on the CPU channel
    HAS_TORCH = False

from cloudstudio_3dgs.training.densification_gradient import (
    CriterionGradientSnapshot,
    photometric_growth_loss,
    restore_criterion_gradients,
    snapshot_criterion_gradients,
    split_backward_for_growth_signal,
)
from cloudstudio_3dgs.training.optimization_audit import (
    AuditedLossTerm,
    component_gradient_audit,
)

RGB_L1_WEIGHT = 0.8
RGB_SSIM_WEIGHT = 0.2
RANGE_WEIGHT = 0.3
RANGE_STAGE = 0.5
REG_WEIGHT = 1e-2


class _Carrier:
    """Non-tensor handle so the stub can write onto the retained tensor."""

    def __init__(self, tensor):
        self.tensor = tensor


if HAS_TORCH:

    class _AbsgradRasterStub(torch.autograd.Function):
        """Mimics gsplat: every backward OVERWRITES ``means2d.absgrad``."""

        @staticmethod
        def forward(ctx, means2d, carrier):
            ctx.carrier = carrier
            return means2d.clone()

        @staticmethod
        def backward(ctx, grad):
            ctx.carrier.tensor.absgrad = grad.abs()
            return grad, None


def _scene(seed: int, *, n: int = 8, pixels: int = 16):
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape, scale=1.0):
        return torch.randn(*shape, generator=generator) * scale

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(draw(n, 3)),
            "scales": torch.nn.Parameter(draw(n, 3, scale=0.2) - 0.5),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)),
            "opacities": torch.nn.Parameter(draw(n)),
            "colors": torch.nn.Parameter(torch.rand(n, 3, generator=generator)),
        }
    )
    target = torch.rand(pixels, 3, generator=generator)
    pixel_xy = draw(pixels, 2)
    return params, target, pixel_xy


def _render(params, target, pixel_xy, *, absgrad: bool = False):
    """Differentiable stand-in: returns (means2d, l1, ssim, range_loss, reg)."""
    # Projection: a non-leaf that depends on all three coordinates, retained
    # the way DefaultStrategy.step_pre_backward retains gsplat's means2d.
    means2d = params["means"][:, :2] * 1.5 + params["means"][:, 2:3]
    means2d.retain_grad()
    used = _AbsgradRasterStub.apply(means2d, _Carrier(means2d)) if absgrad else means2d
    # The quaternion norm scales the footprint so every parameter group,
    # quats included, receives a gradient from both loss families.
    footprint = torch.exp(params["scales"][:, :2]).sum(-1) * params["quats"].norm(dim=-1)
    distance2 = (pixel_xy[:, None, :] - used[None]).square().sum(-1)  # (P,N)
    weight = torch.exp(-distance2 / footprint[None])
    alpha = torch.sigmoid(params["opacities"])[None] * weight  # (P,N)
    image = (alpha[..., None] * params["colors"][None]).sum(1)  # (P,3)
    l1 = (image - target).abs().mean()
    ssim = (image - target).square().mean()  # stand-in for the SSIM term
    # A depth-like supervision that shares the projection: this is the term
    # whose gradient must NOT reach the densification criterion.
    depth = (alpha * used.norm(dim=-1)[None]).sum(1)
    range_loss = (depth - 1.0).abs().mean()
    # Direct parameter regulariser with no path to means2d at all.
    reg = torch.exp(params["scales"]).mean() + torch.sigmoid(params["opacities"]).mean()
    return means2d, l1, ssim, range_loss, reg


def _growth_loss(l1, ssim):
    return photometric_growth_loss(
        l1=l1,
        ssim=ssim,
        rgb_l1_weight=RGB_L1_WEIGHT,
        rgb_ssim_weight=RGB_SSIM_WEIGHT,
        rgb_gradient_weight=0.0,
        rgb_gradient_l1=None,
        lidar_rgb_l1_weight=0.0,
        lidar_rgb_l1=None,
    )


def _losses(params, target, pixel_xy, *, absgrad: bool = False):
    means2d, l1, ssim, range_loss, reg = _render(
        params, target, pixel_xy, absgrad=absgrad
    )
    growth = _growth_loss(l1, ssim)
    total = growth + RANGE_WEIGHT * RANGE_STAGE * range_loss + REG_WEIGHT * reg
    return means2d, growth, total


def _split_backward(means2d, growth, total, *, require_absgrad: bool = False):
    holder: dict[str, CriterionGradientSnapshot] = {}

    def isolate() -> None:
        holder["snapshot"] = snapshot_criterion_gradients(
            means2d, require_absgrad=require_absgrad
        )

    def restore() -> None:
        restore_criterion_gradients(means2d, holder["snapshot"])

    split_backward_for_growth_signal(
        total_loss=total, growth_loss=growth, isolate=isolate, restore=restore
    )
    return holder["snapshot"]


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class SplitBackwardTests(unittest.TestCase):
    def test_leaf_gradients_match_a_single_total_backward(self) -> None:
        # (a) The optimizer must see exactly the total-loss gradient.
        reference_params, target, pixel_xy = _scene(11)
        _, _, reference_total = _losses(reference_params, target, pixel_xy)
        reference_total.backward()

        params, target, pixel_xy = _scene(11)
        means2d, growth, total = _losses(params, target, pixel_xy)
        _split_backward(means2d, growth, total)

        for name, parameter in params.items():
            expected = reference_params[name].grad
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(
                parameter.grad, expected, rtol=1e-5, atol=1e-6, msg=name
            )
            self.assertGreater(float(parameter.grad.abs().max()), 0.0, name)

    def test_means2d_growth_gradient_is_rgb_only_not_total(self) -> None:
        # (b) The criterion gradient is d(rgb)/d(means2d); the depth term that
        # shares the projection must not leak into it.
        probe_params, target, pixel_xy = _scene(23)
        probe_means2d, probe_growth, probe_total = _losses(probe_params, target, pixel_xy)
        rgb_only = torch.autograd.grad(probe_growth, probe_means2d, retain_graph=True)[0]
        total_grad = torch.autograd.grad(probe_total, probe_means2d)[0]
        # Sanity on the fixture: the geometry term really touches means2d.
        self.assertFalse(torch.allclose(rgb_only, total_grad, rtol=1e-3, atol=1e-6))

        params, target, pixel_xy = _scene(23)
        means2d, growth, total = _losses(params, target, pixel_xy)
        snapshot = _split_backward(means2d, growth, total)

        torch.testing.assert_close(means2d.grad, rgb_only, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(snapshot.grad, rgb_only, rtol=1e-5, atol=1e-7)
        self.assertFalse(torch.allclose(means2d.grad, total_grad, rtol=1e-3, atol=1e-6))
        self.assertIsNone(snapshot.absgrad)

    def test_absgrad_snapshot_survives_the_second_backward(self) -> None:
        # (c) gsplat overwrites .absgrad on every backward; only the snapshot
        # taken between the passes carries the photometric one through.
        probe_params, target, pixel_xy = _scene(37)
        probe_means2d, probe_growth, probe_total = _losses(
            probe_params, target, pixel_xy, absgrad=True
        )
        rgb_only_abs = torch.autograd.grad(probe_growth, probe_means2d, retain_graph=True)[0].abs()
        remainder_abs = torch.autograd.grad(
            probe_total - probe_growth, probe_means2d
        )[0].abs()
        self.assertFalse(torch.allclose(rgb_only_abs, remainder_abs, rtol=1e-3, atol=1e-6))

        # Control: the naive two-pass sequence without the snapshot ends with
        # the remainder's absgrad - the hazard the snapshot exists for.
        naive_params, target, pixel_xy = _scene(37)
        naive_means2d, naive_growth, naive_total = _losses(
            naive_params, target, pixel_xy, absgrad=True
        )
        naive_growth.backward(retain_graph=True)
        (naive_total - naive_growth).backward()
        torch.testing.assert_close(naive_means2d.absgrad, remainder_abs, rtol=1e-5, atol=1e-7)

        params, target, pixel_xy = _scene(37)
        means2d, growth, total = _losses(params, target, pixel_xy, absgrad=True)
        snapshot = _split_backward(means2d, growth, total, require_absgrad=True)

        torch.testing.assert_close(means2d.absgrad, rgb_only_abs, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(snapshot.absgrad, rgb_only_abs, rtol=1e-5, atol=1e-7)
        # The accumulating .grad is restored to the photometric one as well.
        check_params, target, pixel_xy = _scene(37)
        check_means2d, check_growth, _ = _losses(check_params, target, pixel_xy, absgrad=True)
        expected_grad = torch.autograd.grad(check_growth, check_means2d)[0]
        torch.testing.assert_close(means2d.grad, expected_grad, rtol=1e-5, atol=1e-7)

    def test_snapshot_fails_closed(self) -> None:
        means2d = torch.zeros(4, 2, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "no gradient on means2d"):
            snapshot_criterion_gradients(means2d, require_absgrad=False)
        means2d.grad = torch.ones(4, 2)
        with self.assertRaisesRegex(RuntimeError, "produced no means2d.absgrad"):
            snapshot_criterion_gradients(means2d, require_absgrad=True)
        snapshot = snapshot_criterion_gradients(means2d, require_absgrad=False)
        self.assertIsNone(snapshot.absgrad)
        # Restoring a snapshot without absgrad never plants a stale attribute.
        restore_criterion_gradients(means2d, snapshot)
        self.assertFalse(hasattr(means2d, "absgrad"))

    def test_photometric_growth_loss_skips_absent_terms(self) -> None:
        l1 = torch.tensor(2.0)
        ssim = torch.tensor(4.0)
        extra = torch.tensor(1.0)
        base = photometric_growth_loss(
            l1=l1, ssim=ssim, rgb_l1_weight=0.5, rgb_ssim_weight=0.25,
            rgb_gradient_weight=9.0, rgb_gradient_l1=None,
            lidar_rgb_l1_weight=9.0, lidar_rgb_l1=None,
        )
        self.assertAlmostEqual(float(base), 2.0)
        full = photometric_growth_loss(
            l1=l1, ssim=ssim, rgb_l1_weight=0.5, rgb_ssim_weight=0.25,
            rgb_gradient_weight=0.1, rgb_gradient_l1=extra,
            lidar_rgb_l1_weight=0.2, lidar_rgb_l1=extra,
        )
        self.assertAlmostEqual(float(full), 2.3)


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class ComponentGradientAuditTests(unittest.TestCase):
    def _audit(self, *, with_means2d: bool = True):
        params, target, pixel_xy = _scene(5)
        means2d, l1, ssim, range_loss, reg = _render(params, target, pixel_xy)
        components = {
            "rgb": _growth_loss(l1, ssim),
            "lidar_range": AuditedLossTerm(
                raw=range_loss, weight=RANGE_WEIGHT, stage_multiplier=RANGE_STAGE
            ),
            "geometry_reg": AuditedLossTerm(raw=reg, weight=REG_WEIGHT),
            "da2_depth": AuditedLossTerm(raw=None, weight=0.5, stage_multiplier=0.0),
        }
        report = component_gradient_audit(
            params, components, means2d=means2d if with_means2d else None
        )
        return params, means2d, components, report

    def test_audit_does_not_advance_the_global_rng(self) -> None:
        # (d) A telemetry read must never change what the split draws next.
        torch.manual_seed(3)
        torch.rand(10)
        before = torch.get_rng_state().clone()
        self._audit()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_terms_report_raw_effective_weight_and_means2d_status(self) -> None:
        params, means2d, components, report = self._audit()
        terms = report["terms"]
        self.assertTrue(report["means2d_probed"])

        range_term = terms["lidar_range"]
        self.assertTrue(range_term["present"])
        self.assertAlmostEqual(range_term["weight"], RANGE_WEIGHT)
        self.assertAlmostEqual(range_term["stage_multiplier"], RANGE_STAGE)
        self.assertAlmostEqual(range_term["effective_weight"], RANGE_WEIGHT * RANGE_STAGE)
        self.assertAlmostEqual(
            range_term["weighted_loss"],
            range_term["raw_loss"] * RANGE_WEIGHT * RANGE_STAGE,
            places=6,
        )
        self.assertEqual(range_term["means2d_gradient"], "measured")
        expected = torch.autograd.grad(
            components["lidar_range"].raw * RANGE_WEIGHT * RANGE_STAGE,
            means2d,
            retain_graph=True,
        )[0]
        self.assertAlmostEqual(
            range_term["means2d_gradient_l2"], float(expected.double().norm()), places=6
        )
        self.assertAlmostEqual(
            report["gradient_norms"]["lidar_range"]["means2d"]["l2"],
            float(expected.double().norm()),
            places=6,
        )

        # A direct regulariser has no path to means2d: N/A, not zero, not an error.
        reg_term = terms["geometry_reg"]
        self.assertTrue(reg_term["present"])
        self.assertEqual(reg_term["means2d_gradient"], "not_applicable")
        self.assertIsNone(reg_term["means2d_gradient_l2"])
        self.assertIsNone(report["gradient_norms"]["geometry_reg"]["means2d"])
        self.assertIsNone(report["gradient_norms"]["geometry_reg"]["means"])
        self.assertGreater(report["gradient_norms"]["geometry_reg"]["scales"]["l2"], 0.0)
        self.assertIsNone(report["pairwise_cosine"]["rgb__geometry_reg"]["means2d"])
        self.assertIsNone(report["pairwise_cosine"]["rgb__geometry_reg"]["means"])

        # An absent term is present=False with every gradient None.
        da2_term = terms["da2_depth"]
        self.assertFalse(da2_term["present"])
        self.assertIsNone(da2_term["raw_loss"])
        self.assertIsNone(da2_term["weighted_loss"])
        self.assertEqual(da2_term["effective_weight"], 0.0)
        self.assertTrue(all(v is None for v in report["gradient_norms"]["da2_depth"].values()))

        # Every parameter group is reported, colours included.
        self.assertIn("colors", report["gradient_norms"]["rgb"])
        self.assertIsNone(report["gradient_norms"]["lidar_range"]["colors"])

        # The rgb/range means2d cosine is a real number in [-1, 1].
        cosine = report["pairwise_cosine"]["rgb__lidar_range"]["means2d"]
        self.assertIsNotNone(cosine)
        self.assertGreaterEqual(cosine, -1.0)
        self.assertLessEqual(cosine, 1.0)

    def test_without_means2d_the_status_is_absent(self) -> None:
        _, _, _, report = self._audit(with_means2d=False)
        self.assertFalse(report["means2d_probed"])
        self.assertNotIn("means2d", report["gradient_norms"]["rgb"])
        self.assertEqual(report["terms"]["rgb"]["means2d_gradient"], "absent")
        self.assertEqual(report["terms"]["rgb"]["effective_weight"], 1.0)

    def test_legacy_tensor_components_keep_their_shape(self) -> None:
        means = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        report = component_gradient_audit(
            {"means": means},
            {"rgb": means.square().sum(), "range": -means.square().sum(), "normal": None},
        )
        self.assertAlmostEqual(report["pairwise_cosine"]["rgb__range"]["means"], -1.0)
        self.assertIsNone(report["pairwise_cosine"]["rgb__normal"]["means"])
        self.assertEqual(report["terms"]["rgb"]["weight"], 1.0)


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class TopologyGradientAlignmentTests(unittest.TestCase):
    """(e) Preserve -> duplicate/remove -> restore keeps rows aligned."""

    def _adapter(self):
        from cloudstudio_3dgs.training.default_strategy_adapter import (
            DefaultStrategyAdapter,
        )

        return DefaultStrategyAdapter(
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
            lifecycle_execution_order="pre_optimizer_vendor",
        )

    def test_gradients_adam_state_and_lineage_follow_the_same_remap(self) -> None:
        from gsplat.strategy.ops import duplicate, remove

        adapter = self._adapter()
        params, target, pixel_xy = _scene(41, n=6)
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        # One real step so Adam carries row-distinct moments, then a fresh
        # backward so the parameters hold the current step's gradient.
        _, _, total = _losses(params, target, pixel_xy)
        total.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means2d, growth, total = _losses(params, target, pixel_xy)
        _split_backward(means2d, growth, total)

        original = {name: p.detach().clone() for name, p in params.items()}
        original_grad = {name: p.grad.detach().clone() for name, p in params.items()}
        original_exp_avg = {
            name: optimizers[name].state[p]["exp_avg"].detach().clone()
            for name, p in params.items()
        }
        state = {
            "grad2d": torch.zeros(6),
            "count": torch.zeros(6),
            "radii": torch.zeros(6),
            "scene_scale": 10.0,
        }
        adapter._ensure_lineage(params, state)
        state["_cloudstudio_birth_step"] = torch.arange(6, dtype=torch.int32)
        state["_cloudstudio_birth_kind"] = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int8)

        preserved = adapter._preserve_current_step_gradients(params, state)
        # Append rows 1 and 3, then remove rows 0 and 4 of the enlarged set.
        duplicate(
            params=params,
            optimizers=optimizers,
            state=state,
            mask=torch.tensor([False, True, False, True, False, False]),
        )
        remove(
            params=params,
            optimizers=optimizers,
            state=state,
            mask=torch.tensor([True, False, False, False, True, False, False, False]),
        )
        # gsplat's topology ops leave the replacement Parameters without .grad.
        self.assertTrue(all(p.grad is None for p in params.values()))
        adapter.last_lifecycle_event = {}
        adapter._restore_current_step_gradients(params, state, preserved)

        provenance = torch.tensor([1, 2, 3, 5, 1, 3])
        self.assertEqual(len(params["means"]), 6)
        for name in params:
            torch.testing.assert_close(params[name].detach(), original[name][provenance])
            torch.testing.assert_close(params[name].grad, original_grad[name][provenance])
            exp_avg = optimizers[name].state[params[name]]["exp_avg"]
            # Survivors keep their own moments; the two appended clones start
            # from zero, as gsplat's duplicate defines them.
            torch.testing.assert_close(exp_avg[:4], original_exp_avg[name][provenance[:4]])
            self.assertEqual(float(exp_avg[4:].abs().sum()), 0.0)
            self.assertIs(optimizers[name].param_groups[0]["params"][0], params[name])
        self.assertEqual(state["_cloudstudio_birth_step"].tolist(), provenance.tolist())
        self.assertEqual(state["_cloudstudio_birth_kind"].tolist(), [1, 2, 0, 2, 1, 0])
        self.assertNotIn("_cloudstudio_current_step_gradient_source", state)
        self.assertTrue(adapter.last_lifecycle_event["current_step_gradient_remapped"])
        self.assertEqual(adapter.last_lifecycle_event["gradient_row_count"], 6)

    def test_restore_refuses_lost_or_mismatched_provenance(self) -> None:
        adapter = self._adapter()
        params, _, _ = _scene(43, n=3)
        for parameter in params.values():
            parameter.grad = torch.ones_like(parameter)
        state: dict = {}
        with self.assertRaisesRegex(RuntimeError, "lost gradient provenance"):
            adapter._restore_current_step_gradients(params, state, {})
        preserved = adapter._preserve_current_step_gradients(params, state)
        with self.assertRaisesRegex(RuntimeError, "stale pre-optimizer gradient provenance"):
            adapter._preserve_current_step_gradients(params, state)
        state["_cloudstudio_current_step_gradient_source"] = torch.tensor([0, 1])
        with self.assertRaisesRegex(RuntimeError, "provenance count mismatch"):
            adapter._restore_current_step_gradients(params, state, preserved)


if __name__ == "__main__":
    unittest.main()
