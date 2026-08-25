"""The reference-3DGS densification path must be selectable without disturbing MCMC.

The adapter exists so that gsplat's DefaultStrategy can be driven through the
attribute surface this backend and trainer already read from MCMC. Two things
therefore need guarding: that the surface is complete, and that selecting the
new path does not alter the old one.

The pre-backward hook gets its own test because omitting it fails SILENTLY -
DefaultStrategy scores Gaussians by a gradient that is only retained inside that
call, so a missing call leaves the criterion permanently inert rather than
raising. That is exactly the class of bug a test has to catch, because nothing
at runtime will.
"""

from __future__ import annotations

import unittest

from cloudstudio_3dgs.training.default_strategy_adapter import (
    DENSIFICATION_STRATEGIES,
    DefaultStrategyAdapter,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


def _minimal_config(**overrides):
    payload = {
        "run_id": "unit",
        "dataset_manifest": "d.json",
        "recording_root": ".",
        "mask_manifest": "m.json",
        "mask_root": ".",
        "split_manifest": "s.json",
        "output_dir": "out",
        "gsplat_lock": "lock.json",
        "device": "cuda:0",
        "max_steps": 100,
        "cap_max": 1000,
        # validate() enforces the full production contract, so these are present
        # to reach the densification checks rather than because this test cares
        # about person masking.
        "person_mask_manifest": "p.json",
        "person_mask_root": ".",
        "lidar_range_weight": 0.0,
        "initialization_ply": "init.ply",
    }
    payload.update(overrides)
    return payload


class DefaultStrategySelectionTests(unittest.TestCase):
    def test_the_existing_mcmc_path_remains_the_default(self):
        config = TrainerConfig.from_dict(_minimal_config())
        self.assertEqual(config.densification_strategy, "error_weighted_mcmc")

    def test_classic_densification_can_be_selected(self):
        config = TrainerConfig.from_dict(
            _minimal_config(densification_strategy="default_3dgs")
        )
        config.validate()
        self.assertEqual(config.densification_strategy, "default_3dgs")

    def test_an_unknown_strategy_is_refused_rather_than_ignored(self):
        config = TrainerConfig.from_dict(
            _minimal_config(densification_strategy="whatever")
        )
        with self.assertRaises(ValueError):
            config.validate()

    def test_published_refinements_pass_through_to_the_strategy(self):
        # absgrad is the AbsGS fix and revised_opacity the corrected split
        # opacity; both are upstream options, not local inventions.
        config = TrainerConfig.from_dict(
            _minimal_config(
                densification_strategy="default_3dgs",
                default_strategy={"absgrad": True, "revised_opacity": True},
            )
        )
        config.validate()
        self.assertEqual(
            config.default_strategy, {"absgrad": True, "revised_opacity": True}
        )


class DefaultStrategyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = DefaultStrategyAdapter(
            scene_scale=2.5,
            refine_start_iter=500,
            refine_stop_iter=7000,
            refine_every=100,
            absgrad=True,
        )

    def test_it_presents_every_attribute_the_backend_reads_from_mcmc(self):
        for name in ("min_opacity", "noise_injection_stop_iter", "refine_start_iter",
                     "refine_stop_iter", "refine_every", "cap_max"):
            self.assertTrue(hasattr(self.adapter, name), name)

    def test_min_opacity_maps_onto_the_upstream_prune_threshold(self):
        self.assertEqual(self.adapter.min_opacity, self.adapter.inner.prune_opa)

    def test_noise_injection_reports_stopped_because_classic_injects_none(self):
        # The backend uses this to decide whether to probe for a position delta;
        # reporting anything else would make it wait for noise that never comes.
        self.assertEqual(self.adapter.noise_injection_stop_iter, 0)

    def test_cap_max_is_none_because_classic_growth_is_threshold_driven(self):
        self.assertIsNone(self.adapter.cap_max)

    def test_refine_window_is_forwarded_not_defaulted(self):
        self.assertEqual(self.adapter.refine_start_iter, 500)
        self.assertEqual(self.adapter.refine_stop_iter, 7000)
        self.assertEqual(self.adapter.refine_every, 100)

    def test_state_carries_the_gradient_accumulator_the_criterion_needs(self):
        state = self.adapter.initialize_state()
        self.assertIn("grad2d", state)
        self.assertEqual(state["scene_scale"], 2.5)

    def test_step_post_backward_accepts_and_ignores_the_mcmc_lr_argument(self):
        # The trainer passes lr= unconditionally; only MCMC's noise term uses it.
        import inspect

        signature = inspect.signature(self.adapter.step_post_backward)
        self.assertIn("lr", signature.parameters)

    def test_pre_backward_is_exposed_because_omitting_it_fails_silently(self):
        # DefaultStrategy scores by a gradient retained only inside this call.
        # Without it the criterion never fires and nothing raises.
        self.assertTrue(callable(self.adapter.step_pre_backward))

    def test_state_dict_records_which_strategy_produced_a_run(self):
        recorded = self.adapter.state_dict()
        self.assertEqual(recorded["strategy"], "default_3dgs")
        self.assertTrue(recorded["absgrad"])
        self.assertEqual(recorded["grow_grad2d"], 0.0002)

    def test_the_strategy_names_are_the_two_that_exist(self):
        self.assertEqual(
            set(DENSIFICATION_STRATEGIES), {"error_weighted_mcmc", "default_3dgs"}
        )


class BackendWiringTests(unittest.TestCase):
    def test_backend_flags_whether_the_pre_backward_hook_is_required(self):
        from cloudstudio_3dgs.training.backend import GsplatBackend

        # Class-level default keeps instances built without __init__ (contract
        # tests do this) from claiming they need a hook they cannot serve.
        self.assertFalse(GsplatBackend.needs_pre_backward)
        self.assertTrue(hasattr(GsplatBackend, "strategy_pre_step"))

    def test_defaults_describe_mcmc_semantics(self):
        from cloudstudio_3dgs.training.backend import GsplatBackend

        # Both flags exist so the shared telemetry path can tell the two
        # strategies apart; MCMC neither prunes nor needs the absgrad channel.
        self.assertFalse(GsplatBackend.strategy_prunes)
        self.assertFalse(GsplatBackend.needs_absgrad)


class PruningInvariantTests(unittest.TestCase):
    """The count check encodes MCMC semantics that classic densification breaks."""

    def _event(self, before_count, after_count, **kwargs):
        from cloudstudio_3dgs.training.runtime_evidence import build_mcmc_step_event

        return build_mcmc_step_event(
            step=1000,
            before={"gaussian_count": before_count, "dead_gaussian_count": 0},
            after={"gaussian_count": after_count, "dead_gaussian_count": 0},
            refine_start_iter=500,
            refine_stop_iter=7000,
            refine_every=100,
            noise_injection_stop_iter=0,
            **kwargs,
        )

    def test_mcmc_still_rejects_a_falling_count(self):
        # MCMC relocates and appends but never removes, so this remains a bug
        # signal for the existing path and must not be relaxed for everyone.
        with self.assertRaises(RuntimeError):
            self._event(1000, 900)

    def test_a_pruning_strategy_is_allowed_to_reduce_the_count(self):
        event = self._event(1000, 900, strategy_prunes=True)
        self.assertEqual(event["pruned_gaussian_count"], 100)
        self.assertEqual(event["new_gaussian_count"], 0)

    def test_growth_is_reported_the_same_way_for_both(self):
        for prunes in (False, True):
            event = self._event(1000, 1200, strategy_prunes=prunes)
            self.assertEqual(event["new_gaussian_count"], 200)
            self.assertEqual(event["pruned_gaussian_count"], 0)


if __name__ == "__main__":
    unittest.main()


class MetricThresholdTests(unittest.TestCase):
    """Scale gates are normalised by scene extent, and the wrong quantity is silent."""

    def test_metric_thresholds_convert_through_scene_scale(self):
        adapter = DefaultStrategyAdapter(
            scene_scale=16.7, split_scale_m=0.03, prune_scale_m=0.20
        )
        recorded = adapter.state_dict()
        self.assertAlmostEqual(recorded["effective_split_scale_m"], 0.03, places=6)
        self.assertAlmostEqual(recorded["effective_prune_scale_m"], 0.20, places=6)

    def test_the_resolved_metres_are_recorded_not_just_the_ratios(self):
        # The run that passed median Gaussian size as scene_scale looked
        # identical in its config; only the resolved metres expose it.
        wrong = DefaultStrategyAdapter(scene_scale=0.0582).state_dict()
        right = DefaultStrategyAdapter(scene_scale=16.7).state_dict()
        self.assertLess(wrong["effective_split_scale_m"], 0.001)
        self.assertGreater(right["effective_split_scale_m"], 0.1)

    def test_a_non_positive_scene_scale_is_refused(self):
        with self.assertRaises(ValueError):
            DefaultStrategyAdapter(scene_scale=0.0)

    def test_screen_space_gates_are_exposed(self):
        # Upstream disables both behind refine_scale2d_stop_iter=0, so a
        # footprint problem cannot be addressed unless this is settable.
        adapter = DefaultStrategyAdapter(
            scene_scale=16.7, grow_scale2d=0.04, prune_scale2d=0.12,
            refine_scale2d_stop_iter=12000,
        )
        recorded = adapter.state_dict()
        self.assertEqual(recorded["grow_scale2d"], 0.04)
        self.assertEqual(recorded["prune_scale2d"], 0.12)
        self.assertEqual(recorded["refine_scale2d_stop_iter"], 12000)

    def test_pause_after_reset_is_exposed(self):
        adapter = DefaultStrategyAdapter(scene_scale=16.7, pause_refine_after_reset=938)
        self.assertEqual(adapter.state_dict()["pause_refine_after_reset"], 938)


class GradientSourceTests(unittest.TestCase):
    """rgb_only keeps the LiDAR terms out of the densification criterion.

    Both losses backprop through the same rasterization, so under a single
    backward means2d.grad differentiates the TOTAL loss and the criterion is no
    longer the published one. The knob exists to cut that leak; refusing it
    under MCMC exists because MCMC never reads means2d.grad and accepting it
    there would be a silent no-op.
    """

    def test_default_is_the_total_loss_for_compatibility(self):
        config = TrainerConfig.from_dict(_minimal_config())
        self.assertEqual(config.densification_gradient_source, "total_loss")

    def test_rgb_only_is_accepted_with_the_classic_strategy(self):
        config = TrainerConfig.from_dict(
            _minimal_config(
                densification_strategy="default_3dgs",
                densification_gradient_source="rgb_only",
            )
        )
        config.validate()
        self.assertEqual(config.densification_gradient_source, "rgb_only")

    def test_rgb_only_under_mcmc_is_refused_not_ignored(self):
        config = TrainerConfig.from_dict(
            _minimal_config(densification_gradient_source="rgb_only")
        )
        with self.assertRaises(ValueError):
            config.validate()

    def test_an_unknown_source_is_refused(self):
        config = TrainerConfig.from_dict(
            _minimal_config(
                densification_strategy="default_3dgs",
                densification_gradient_source="whatever",
            )
        )
        with self.assertRaises(ValueError):
            config.validate()


class GradientIsolationTests(unittest.TestCase):
    """The two-pass snapshot must survive both gradient semantics.

    means2d.grad ACCUMULATES across backward passes while gsplat OVERWRITES
    means2d.absgrad on each, so a snapshot between the passes is the only
    representation that stays photometric-only in both cases.
    """

    def _backend(self, needs_absgrad=False):
        from cloudstudio_3dgs.training.backend import GsplatBackend

        backend = GsplatBackend.__new__(GsplatBackend)
        backend.needs_absgrad = needs_absgrad
        return backend

    def test_grad_snapshot_survives_a_contaminating_second_pass(self):
        import torch

        means2d = torch.zeros(4, 2, requires_grad=True)
        means2d.grad = torch.full((4, 2), 0.5)
        info = {"means2d": means2d}
        backend = self._backend()
        backend.strategy_isolate_gradient(info)
        means2d.grad = means2d.grad + 1.0  # what the LiDAR pass does
        backend.strategy_restore_gradient(info)
        self.assertTrue(torch.equal(means2d.grad, torch.full((4, 2), 0.5)))

    def test_absgrad_snapshot_survives_the_overwrite_semantics(self):
        import torch

        means2d = torch.zeros(4, 2, requires_grad=True)
        means2d.grad = torch.zeros(4, 2)
        means2d.absgrad = torch.full((4, 2), 0.25)
        info = {"means2d": means2d}
        backend = self._backend(needs_absgrad=True)
        backend.strategy_isolate_gradient(info)
        means2d.absgrad = torch.full((4, 2), 9.0)  # gsplat overwrites per pass
        backend.strategy_restore_gradient(info)
        self.assertTrue(torch.equal(means2d.absgrad, torch.full((4, 2), 0.25)))

    def test_an_absgrad_strategy_without_absgrad_fails_closed(self):
        import torch

        means2d = torch.zeros(4, 2, requires_grad=True)
        means2d.grad = torch.zeros(4, 2)
        info = {"means2d": means2d}
        backend = self._backend(needs_absgrad=True)
        with self.assertRaises(RuntimeError):
            backend.strategy_isolate_gradient(info)

    def test_a_gradient_free_backward_fails_closed(self):
        import torch

        means2d = torch.zeros(4, 2, requires_grad=True)
        info = {"means2d": means2d}
        backend = self._backend()
        with self.assertRaises(RuntimeError):
            backend.strategy_isolate_gradient(info)
