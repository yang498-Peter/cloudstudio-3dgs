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

    def test_exact_mipmap_schedule_is_inclusive_at_step_500(self):
        adapter = DefaultStrategyAdapter(
            scene_scale=10.0,
            refine_start_iter=500,
            refine_stop_iter=2000,
            refine_every=100,
            reset_every=300,
            grow_grad2d=0.00015,
            prune_opa=0.1,
            split_scale_m=0.2,
            prune_scale_m=0.2,
            prune_scale2d=0.15,
            refine_scale2d_stop_iter=2000,
            exact_mipmap_lifecycle=True,
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
        )
        self.assertTrue(adapter.is_refine_step(500))
        self.assertFalse(adapter.is_refine_step(501))
        self.assertTrue(adapter.is_refine_step(600))

    def test_exact_mipmap_growth_requires_gradient_and_opacity(self):
        import torch

        adapter = DefaultStrategyAdapter(
            scene_scale=10.0,
            refine_start_iter=500,
            refine_stop_iter=2000,
            refine_every=100,
            reset_every=300,
            grow_grad2d=0.00015,
            prune_opa=0.1,
            split_scale_m=0.2,
            prune_scale_m=0.2,
            prune_scale2d=0.15,
            refine_scale2d_stop_iter=2000,
            exact_mipmap_lifecycle=True,
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(3, 3)),
                "scales": torch.nn.Parameter(
                    torch.tensor([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1]]).log()
                ),
                "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)),
                "opacities": torch.nn.Parameter(torch.tensor([0.2, 0.2, 0.1]).logit()),
                "colors": torch.nn.Parameter(torch.zeros(3, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.tensor([0.001, 0.001, 0.001]),
            "count": torch.ones(3),
            "radii": torch.zeros(3),
            "scene_scale": 10.0,
        }
        clone_count, split_count = adapter._grow_mipmap(
            params, optimizers, state
        )
        self.assertEqual((clone_count, split_count), (1, 1))
        self.assertEqual(len(params["means"]), 5)

    def test_capacity_conserving_clone_preserves_coincident_alpha(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            capacity_conserving_clone_opacity=True,
        )
        original_opacity = torch.tensor(0.64)
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(1, 3)),
                "scales": torch.nn.Parameter(torch.full((1, 3), 0.05).log()),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]])
                ),
                "opacities": torch.nn.Parameter(original_opacity[None].logit()),
                "colors": torch.nn.Parameter(torch.zeros(1, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.tensor([0.001]),
            "count": torch.ones(1),
            "radii": torch.zeros(1),
            "scene_scale": 10.0,
        }
        clone_count, split_count = adapter._grow_mipmap(
            params, optimizers, state
        )
        self.assertEqual((clone_count, split_count), (1, 0))
        child_opacity = torch.sigmoid(params["opacities"].detach())
        composited = 1.0 - torch.prod(1.0 - child_opacity)
        self.assertTrue(torch.allclose(composited, original_opacity, atol=1e-6))
        self.assertTrue(torch.allclose(child_opacity[0], child_opacity[1]))
        self.assertTrue(
            adapter._last_growth_event["capacity_conserving_clone_opacity"]
        )

    def test_pre_optimizer_vendor_lifecycle_remaps_current_gradients(self):
        import torch

        adapter = DefaultStrategyAdapter(
            scene_scale=10.0,
            refine_start_iter=500,
            refine_stop_iter=2000,
            refine_every=100,
            reset_every=300,
            grow_grad2d=0.00015,
            prune_opa=0.1,
            split_scale_m=0.2,
            prune_scale_m=0.2,
            prune_scale2d=0.15,
            refine_scale2d_stop_iter=2000,
            exact_mipmap_lifecycle=True,
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            revised_opacity=False,
            lifecycle_execution_order="pre_optimizer_vendor",
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(
                    torch.tensor(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
                    )
                ),
                "scales": torch.nn.Parameter(
                    torch.tensor(
                        [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1]]
                    ).log()
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
                ),
                "opacities": torch.nn.Parameter(
                    torch.tensor([0.2, 0.2, 0.05]).logit()
                ),
                "colors": torch.nn.Parameter(torch.zeros(3, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        # Materialize non-empty Adam moments before topology replacement.
        for parameter in params.values():
            parameter.grad = torch.ones_like(parameter)
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        row_gradient = torch.tensor([1.0, 2.0, 3.0])
        for parameter in params.values():
            view = (3,) + (1,) * (parameter.ndim - 1)
            parameter.grad = row_gradient.view(view).expand_as(parameter).clone()
        state = {
            "grad2d": torch.tensor([0.001, 0.001, 0.0]),
            "count": torch.ones(3),
            "radii": torch.zeros(3),
            "scene_scale": 10.0,
        }
        preserved = adapter._preserve_current_step_gradients(params, state)
        clone_count, split_count = adapter._grow_mipmap(
            params, optimizers, state
        )
        cull_count = adapter._prune_mipmap(
            params, optimizers, state, step=500
        )
        adapter._restore_current_step_gradients(params, state, preserved)

        self.assertEqual((clone_count, split_count, cull_count), (1, 1, 1))
        self.assertEqual(len(params["means"]), 4)
        # Split-first layout is rest[0], split children[1,1], clone child[0].
        self.assertTrue(
            torch.equal(
                params["means"].grad[:, 0],
                torch.tensor([1.0, 2.0, 2.0, 1.0]),
            )
        )
        before_step = params["means"].detach().clone()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertTrue(torch.all(params["means"].detach() != before_step))
        means_state = optimizers["means"].state[params["means"]]
        self.assertEqual(means_state["exp_avg"].shape[0], 4)
        self.assertTrue(torch.all(means_state["exp_avg"].abs().sum(dim=1) > 0.0))

    def test_pre_optimizer_vendor_requires_exact_lifecycle(self):
        with self.assertRaisesRegex(ValueError, "requires exact_mipmap_lifecycle"):
            DefaultStrategyAdapter(
                lifecycle_execution_order="pre_optimizer_vendor"
            )

    def test_pre_optimizer_vendor_accepts_signed_deferred_reset_profile(self):
        adapter = DefaultStrategyAdapter(
            exact_mipmap_lifecycle=True,
            lifecycle_execution_order="pre_optimizer_vendor",
            reset_every=3000,
            vendor_opacity_reset_profile="deferred_every3000_compatibility",
            growth_min_opacity=0.15,
            prune_opa=0.1,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            split_scale_m=0.2,
            prune_scale_m=0.2,
            reset_opacity_cap=0.2,
        )
        self.assertEqual(adapter.inner.reset_every, 3000)
        self.assertEqual(
            adapter.state_dict()["vendor_opacity_reset_profile"],
            "deferred_every3000_compatibility",
        )

    def test_pre_optimizer_vendor_rejects_reset_profile_interval_mismatch(self):
        with self.assertRaisesRegex(ValueError, "reset profile does not match"):
            DefaultStrategyAdapter(
                exact_mipmap_lifecycle=True,
                lifecycle_execution_order="pre_optimizer_vendor",
                reset_every=300,
                vendor_opacity_reset_profile=(
                    "deferred_every3000_compatibility"
                ),
                growth_min_opacity=0.15,
                prune_opa=0.1,
                prune_opa_late=0.05,
                prune_switch_step=1000,
                split_scale_m=0.2,
                prune_scale_m=0.2,
                reset_opacity_cap=0.2,
            )

    def test_capacity_cap_keeps_only_highest_gradient_births(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            capacity_cap=5,
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(3, 3)),
                "scales": torch.nn.Parameter(torch.full((3, 3), 0.05).log()),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
                ),
                "opacities": torch.nn.Parameter(torch.full((3,), 0.3).logit()),
                "colors": torch.nn.Parameter(torch.zeros(3, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.tensor([0.001, 0.003, 0.002]),
            "count": torch.ones(3),
            "radii": torch.zeros(3),
            "scene_scale": 10.0,
        }
        clone_count, split_count = adapter._grow_mipmap(
            params, optimizers, state
        )
        self.assertEqual((clone_count, split_count), (2, 0))
        self.assertEqual(len(params["means"]), 5)

    def test_lidar_birth_guard_blocks_unsupported_parent_and_projects_newborns(self):
        import numpy as np
        import torch

        from cloudstudio_3dgs.geometry.lidar_surface_field import build_surface_field
        from cloudstudio_3dgs.training.tangent_proposal import (
            ProposalConfig,
            TangentProposal,
        )

        xx, yy = np.meshgrid(
            np.linspace(-0.2, 0.2, 9), np.linspace(-0.2, 0.2, 9)
        )
        surface = np.column_stack(
            [xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)]
        )
        proposal = TangentProposal(
            build_surface_field(surface, knn=12),
            ProposalConfig(
                enabled=True,
                planarity_gate=0.6,
                support_gate=0.1,
                tangent_sigma_factor=0.25,
                normal_offset_factor=0.0,
                reject_unsupported_births=True,
            ),
            seed=7,
        )
        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            surface_birth_proposal=proposal,
        )
        params = torch.nn.ParameterDict(
            {
                # Row 0 is measured surface, row 1 is an unsupported floater.
                "means": torch.nn.Parameter(
                    torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
                ),
                "scales": torch.nn.Parameter(
                    torch.tensor([[0.05, 0.05, 0.02]] * 2).log()
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
                ),
                "opacities": torch.nn.Parameter(torch.tensor([0.3, 0.3]).logit()),
                "colors": torch.nn.Parameter(torch.zeros(2, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.tensor([0.001, 0.001]),
            "count": torch.ones(2),
            "radii": torch.zeros(2),
            "scene_scale": 10.0,
        }
        clone_count, split_count = adapter._grow_mipmap(
            params, optimizers, state
        )
        self.assertEqual((clone_count, split_count), (1, 0))
        self.assertEqual(len(params["means"]), 3)
        self.assertAlmostEqual(
            float(params["means"][-1, 2].detach()), 0.0, places=6
        )
        self.assertEqual(proposal.last_stats["applied"], 1)

    def test_observation_aware_opacity_cull_requires_streak_and_caps_each_event(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            opacity_cull_policy="observation_aware",
            opacity_cull_min_observations=4,
            opacity_cull_consecutive_events=2,
            opacity_cull_grace_after_reset_steps=200,
            opacity_cull_max_fraction=0.2,
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(10, 3)),
                "scales": torch.nn.Parameter(torch.full((10, 3), 0.05).log()),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 10)
                ),
                "opacities": torch.nn.Parameter(
                    torch.tensor([0.01] * 6 + [0.3] * 4).logit()
                ),
                "colors": torch.nn.Parameter(torch.zeros(10, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.zeros(10),
            "count": torch.full((10,), 4.0),
            "radii": torch.zeros(10),
            "scene_scale": 10.0,
        }

        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=500), 0
        )
        self.assertEqual(len(params["means"]), 10)
        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=600), 2
        )
        self.assertEqual(len(params["means"]), 8)
        self.assertEqual(
            adapter._last_cull_event["raw_opacity_candidate_count"], 6
        )
        self.assertEqual(adapter._last_cull_event["selected_opacity_count"], 2)
        self.assertEqual(adapter._last_cull_event["world_scale_count"], 0)

    def test_observation_aware_cull_honours_post_reset_grace(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            opacity_cull_policy="observation_aware",
            opacity_cull_min_observations=1,
            opacity_cull_consecutive_events=1,
            opacity_cull_grace_after_reset_steps=200,
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(2, 3)),
                "scales": torch.nn.Parameter(torch.full((2, 3), 0.05).log()),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
                ),
                "opacities": torch.nn.Parameter(torch.tensor([0.01, 0.3]).logit()),
                "colors": torch.nn.Parameter(torch.zeros(2, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.zeros(2),
            "count": torch.ones(2),
            "radii": torch.zeros(2),
            "scene_scale": 10.0,
            "_cloudstudio_last_opacity_reset_step": 600,
        }
        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=800), 0
        )
        self.assertTrue(adapter._last_cull_event["grace_active"])
        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=900), 1
        )

    def test_local_coverage_competition_preserves_one_representative_per_cell(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            opacity_cull_policy="local_coverage_competition",
            opacity_cull_min_observations=1,
            opacity_cull_consecutive_events=1,
            opacity_cull_max_fraction=1.0,
            opacity_cull_local_voxel_m=0.02,
        )
        params = torch.nn.ParameterDict(
            {
                # First three rows share one cell. The fourth is the only row
                # in its cell and must survive even though it is weak.
                "means": torch.nn.Parameter(
                    torch.tensor(
                        [
                            [0.001, 0.001, 0.001],
                            [0.002, 0.001, 0.001],
                            [0.003, 0.001, 0.001],
                            [0.041, 0.001, 0.001],
                        ]
                    )
                ),
                "scales": torch.nn.Parameter(torch.full((4, 3), 0.005).log()),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
                ),
                "opacities": torch.nn.Parameter(
                    torch.tensor([0.01, 0.02, 0.03, 0.01]).logit()
                ),
                "colors": torch.nn.Parameter(torch.zeros(4, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.zeros(4),
            "count": torch.ones(4),
            "radii": torch.zeros(4),
            "scene_scale": 10.0,
        }
        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=500), 2
        )
        remaining = torch.sigmoid(params["opacities"].detach())
        self.assertTrue(torch.isclose(remaining, torch.tensor(0.03)).any())
        self.assertTrue(torch.isclose(remaining, torch.tensor(0.01)).any())
        self.assertEqual(
            adapter._last_cull_event["local_competition_protected_count"], 2
        )
        self.assertEqual(
            adapter._last_cull_event["local_competition_cell_count"], 2
        )

    def test_local_coverage_can_protect_a_broad_thin_surface_carrier(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            opacity_cull_policy="local_coverage_competition",
            opacity_cull_min_observations=1,
            opacity_cull_consecutive_events=1,
            opacity_cull_max_fraction=1.0,
            opacity_cull_local_voxel_m=0.02,
            opacity_cull_local_protection="opacity_tangent_area",
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(
                    torch.tensor(
                        [
                            [0.001, 0.001, 0.001],
                            [0.002, 0.001, 0.001],
                        ]
                    )
                ),
                # Row zero has lower opacity but sixteen times the tangential
                # area, so it is the stronger surface-coverage carrier.
                "scales": torch.nn.Parameter(
                    torch.tensor(
                        [[0.02, 0.02, 0.001], [0.005, 0.005, 0.001]]
                    ).log()
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
                ),
                "opacities": torch.nn.Parameter(
                    torch.tensor([0.02, 0.03]).logit()
                ),
                "colors": torch.nn.Parameter(torch.zeros(2, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.zeros(2),
            "count": torch.ones(2),
            "radii": torch.zeros(2),
            "scene_scale": 10.0,
        }
        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=500), 1
        )
        remaining = torch.sigmoid(params["opacities"].detach())
        self.assertTrue(torch.isclose(remaining, torch.tensor(0.02)).any())
        self.assertEqual(
            adapter._last_cull_event["opacity_cull_local_protection"],
            "opacity_tangent_area",
        )

    def test_local_alpha_budget_preserves_enough_rows_for_surface_coverage(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            opacity_cull_policy="local_coverage_competition",
            opacity_cull_min_observations=1,
            opacity_cull_consecutive_events=1,
            opacity_cull_max_fraction=1.0,
            opacity_cull_local_voxel_m=0.02,
            opacity_cull_local_protection="opacity_tangent_area",
            opacity_cull_local_min_accumulated_alpha=0.15,
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(
                    torch.tensor(
                        [
                            [0.001, 0.001, 0.001],
                            [0.002, 0.001, 0.001],
                            [0.003, 0.001, 0.001],
                        ]
                    )
                ),
                "scales": torch.nn.Parameter(
                    torch.full((3, 3), 0.005).log()
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
                ),
                "opacities": torch.nn.Parameter(
                    torch.tensor([0.08, 0.08, 0.01]).logit()
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
            "count": torch.ones(3),
            "radii": torch.zeros(3),
            "scene_scale": 10.0,
        }
        # 1-(1-.08)^2 = .1536, so both coverage carriers survive and only
        # the third redundant row is removed.
        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=500), 1
        )
        remaining = torch.sigmoid(params["opacities"].detach())
        self.assertEqual(len(remaining), 2)
        self.assertTrue(torch.allclose(remaining, torch.full((2,), 0.08)))

    def test_cull_telemetry_separates_world_and_screen_scale_reasons(self):
        import torch

        adapter = DefaultStrategyAdapter(
            scene_scale=10.0,
            refine_start_iter=500,
            refine_stop_iter=2000,
            refine_every=100,
            reset_every=300,
            grow_grad2d=0.00015,
            prune_opa=0.1,
            split_scale_m=0.2,
            prune_scale_m=0.2,
            prune_scale2d=0.15,
            exact_mipmap_lifecycle=True,
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(3, 3)),
                "scales": torch.nn.Parameter(
                    torch.tensor(
                        [[0.05, 0.05, 0.05], [0.3, 0.05, 0.05], [0.05, 0.05, 0.05]]
                    ).log()
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
                ),
                "opacities": torch.nn.Parameter(torch.full((3,), 0.3).logit()),
                "colors": torch.nn.Parameter(torch.zeros(3, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.zeros(3),
            "count": torch.ones(3),
            "radii": torch.tensor([0.0, 0.0, 0.2]),
            "scene_scale": 10.0,
        }
        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=500), 2
        )
        self.assertEqual(adapter._last_cull_event["world_scale_count"], 1)
        self.assertEqual(adapter._last_cull_event["screen_scale_count"], 1)
        self.assertEqual(
            adapter._last_cull_event["forced_geometry_union_count"], 2
        )

    def test_large_footprint_priority_can_remove_blur_before_smaller_dead_points(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            opacity_cull_policy="observation_aware",
            opacity_cull_min_observations=1,
            opacity_cull_consecutive_events=1,
            opacity_cull_max_fraction=0.25,
            opacity_cull_priority="lowest_opacity_per_footprint",
            detail_split_policy="lidar_surface_screen_detail",
            detail_split_scale_m=0.02,
            detail_split_screen_radius=0.0035,
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(4, 3)),
                "scales": torch.nn.Parameter(
                    torch.tensor(
                        [[0.01] * 3, [0.05] * 3, [0.01] * 3, [0.01] * 3]
                    ).log()
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
                ),
                "opacities": torch.nn.Parameter(
                    torch.tensor([0.01, 0.02, 0.3, 0.3]).logit()
                ),
                "colors": torch.nn.Parameter(torch.zeros(4, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.zeros(4),
            "count": torch.ones(4),
            "radii": torch.tensor([0.001, 0.01, 0.001, 0.001]),
            "scene_scale": 10.0,
        }
        self.assertEqual(
            adapter._prune_mipmap(params, optimizers, state, step=500), 1
        )
        remaining_opacity = torch.sigmoid(params["opacities"].detach())
        self.assertTrue(
            torch.isclose(remaining_opacity, torch.tensor(0.01), atol=1e-6).any()
        )
        self.assertFalse(
            torch.isclose(remaining_opacity, torch.tensor(0.02), atol=1e-6).any()
        )
        self.assertEqual(
            adapter._last_cull_event["opacity_cull_priority"],
            "lowest_opacity_per_footprint",
        )

    def test_detail_split_requires_high_gradient_physical_and_screen_size(self):
        import torch

        adapter = DefaultStrategyAdapter(
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
            growth_min_opacity=0.15,
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            detail_split_policy="lidar_surface_screen_detail",
            detail_split_scale_m=0.02,
            detail_split_screen_radius=0.0035,
        )
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(3, 3)),
                "scales": torch.nn.Parameter(
                    torch.tensor([[0.05] * 3, [0.05] * 3, [0.01] * 3]).log()
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
                ),
                "opacities": torch.nn.Parameter(torch.full((3,), 0.3).logit()),
                "colors": torch.nn.Parameter(torch.zeros(3, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.full((3,), 0.001),
            "count": torch.ones(3),
            "radii": torch.tensor([0.01, 0.002, 0.01]),
            "scene_scale": 10.0,
        }

        clone_count, split_count = adapter._grow_mipmap(
            params, optimizers, state
        )
        self.assertEqual(clone_count, 2)
        self.assertEqual(split_count, 1)


    def test_revised_split_opacity_stays_finite_for_a_saturated_parent(self):
        """A parent whose sigmoid is exactly 1.0 in float32 used to give the
        children logit(1.0) = +inf through the library's revised-opacity
        formula; the next optimizer step then spread NaN through every
        opacity and the run failed closed at step 2900."""
        import torch

        adapter = DefaultStrategyAdapter(
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
            prune_opa_late=0.05,
            prune_switch_step=1000,
            reset_opacity_cap=0.2,
            revised_opacity=True,
            detail_split_policy="lidar_surface_screen_detail",
            detail_split_scale_m=0.02,
            detail_split_screen_radius=0.0035,
        )
        saturated = torch.tensor(40.0)
        self.assertEqual(float(torch.sigmoid(saturated)), 1.0)
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(3, 3)),
                "scales": torch.nn.Parameter(
                    torch.tensor([[0.05] * 3, [0.05] * 3, [0.01] * 3]).log()
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
                ),
                "opacities": torch.nn.Parameter(
                    torch.stack([saturated, torch.tensor(0.3).logit(), torch.tensor(0.3).logit()])
                ),
                "colors": torch.nn.Parameter(torch.zeros(3, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        state = {
            "grad2d": torch.full((3,), 0.001),
            "count": torch.ones(3),
            "radii": torch.tensor([0.01, 0.002, 0.01]),
            "scene_scale": 10.0,
        }
        state["_cloudstudio_current_step"] = 700
        clone_count, split_count = adapter._grow_mipmap(params, optimizers, state)
        self.assertEqual(split_count, 1)
        opacities = params["opacities"].detach()
        self.assertTrue(bool(torch.isfinite(opacities).all()))
        children = torch.sigmoid(opacities[-2:])
        self.assertTrue(bool((children < 1.0).all()))
        self.assertTrue(bool((children > 0.999).all()))
        # Lineage: post-optimizer order appends clones first, then the two
        # split children; the three original rows stay "init" with step -1.
        kind = state["_cloudstudio_birth_kind"].tolist()
        born = state["_cloudstudio_birth_step"].tolist()
        self.assertEqual(len(kind), 3 - 1 + clone_count + 2)
        self.assertEqual(kind[-2:], [2, 2])
        self.assertEqual(kind[-2 - clone_count : -2], [1] * clone_count)
        self.assertTrue(all(b == 700 for b in born[-2 - clone_count :]))
        self.assertTrue(all(b == -1 for b in born[: -2 - clone_count]))
        summary = adapter._last_growth_event["split_parent_opacity"]
        self.assertEqual(summary["frac_saturated"], 1.0)


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

    def test_exact_lifecycle_can_refine_on_inclusive_start_boundary(self):
        from cloudstudio_3dgs.training.runtime_evidence import build_mcmc_step_event

        event = build_mcmc_step_event(
            step=500,
            before={"gaussian_count": 1000, "dead_gaussian_count": 0},
            after={"gaussian_count": 1050, "dead_gaussian_count": 0},
            refine_start_iter=500,
            refine_stop_iter=5610,
            refine_every=100,
            noise_injection_stop_iter=0,
            strategy_prunes=True,
            refine_start_inclusive=True,
        )
        self.assertTrue(event["refine_triggered"])
        self.assertEqual(event["new_gaussian_count"], 50)


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

    def test_classic_lidar_birth_guard_does_not_require_error_weighted_mcmc(self):
        config = TrainerConfig.from_dict(
            _minimal_config(
                densification_strategy="default_3dgs",
                tangent_proposal={
                    "enabled": True,
                    "reject_unsupported_births": True,
                },
            )
        )
        config.validate()
        self.assertTrue(config.tangent_proposal.reject_unsupported_births)

    def test_hard_birth_guard_is_supported_on_mcmc_path(self):
        config = TrainerConfig.from_dict(
            _minimal_config(
                error_weighted_sampling={"enabled": True},
                tangent_proposal={
                    "enabled": True,
                    "reject_unsupported_births": True,
                },
            )
        )
        config.validate()
        self.assertNotEqual(config.densification_strategy, "default_3dgs")
        self.assertTrue(config.tangent_proposal.reject_unsupported_births)


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


class CapacityGuardTests(unittest.TestCase):
    """Threshold-driven growth has no upstream cap; the 2k smoke measured it
    accelerating (6k -> 32k per refine event). The guard turns a would-be CUDA
    OOM hours into an immediate diagnosis carrying the telemetry."""

    def _backend(self, hard_cap):
        from cloudstudio_3dgs.training.backend import GsplatBackend

        backend = GsplatBackend.__new__(GsplatBackend)
        backend.hard_cap = hard_cap
        return backend

    def test_a_count_past_the_cap_aborts_with_a_diagnosis(self):
        backend = self._backend(hard_cap=10)
        with self.assertRaises(RuntimeError) as caught:
            backend.enforce_capacity({"means": range(11)}, step=1200)
        self.assertIn("runaway", str(caught.exception))

    def test_a_count_at_the_cap_is_still_allowed(self):
        self._backend(hard_cap=10).enforce_capacity({"means": range(10)}, step=1200)

    def test_no_cap_means_no_guard(self):
        self._backend(hard_cap=None).enforce_capacity(
            {"means": range(10_000_000)}, step=1200
        )
