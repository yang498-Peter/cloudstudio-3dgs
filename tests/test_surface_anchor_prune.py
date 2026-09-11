"""Surface-anchor prune: far-from-LiDAR gaussians die, unsupported parents do not breed.

Pins (research/quality_recovery_v2/10_surface_anchor_prune.md):

* the distance twin is the exact nearest-anchor distance (brute-force oracle),
  bounded queries read ``inf`` exactly beyond the bound, box tests honour the
  margin;
* a cull event removes far rows and keeps near ones, with Adam moments and
  lineage columns re-indexed alongside the survivors;
* disabled, the adapter and the trainer contract are byte-identical;
* the adaptive-growth gate refuses an enabled arm;
* ``min_age_steps`` spares newborns, ``outside_box: prune`` removes rows
  outside the grown Tile box, the extra cadence acts after refine stop (and
  remaps the current step's gradient under the vendor order);
* ``reject_unsupported_parents`` blocks growth from far rows;
* the audit tool reproduces the table on a synthetic checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised on the CPU channel
    torch = None

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    ORDERED_STAGES,
    UPSTREAM_DATA_READY_STATUS,
    advance_adaptive_growth_gate,
    sign_gate,
)
from cloudstudio_3dgs.training.surface_anchor import (
    AUDIT_DISTANCE_THRESHOLDS_M,
    SurfaceAnchorPrune,
    SurfaceAnchorPruneConfig,
    audit_far_fraction,
    brute_force_nearest_distance,
    build_anchor_tree,
    nearest_surface_distance,
    outside_box_mask,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT_TOOL = ROOT / "tools" / "audit_surface_anchor.py"

# A 0.4 m square patch of "LiDAR" on z = 0 at 5 cm spacing.
_GRID = np.linspace(-0.2, 0.2, 9)
PLANE = np.array(
    [[x, y, 0.0] for x in _GRID for y in _GRID], dtype=np.float64
)
BOX = [[-0.25, -0.25, -0.1], [0.25, 0.25, 0.1]]


def _config(**overrides) -> SurfaceAnchorPruneConfig:
    settings = dict(enabled=True, max_distance_m=0.3, start_step=0)
    settings.update(overrides)
    return SurfaceAnchorPruneConfig(**settings)


class DistanceTwinTests(unittest.TestCase):
    def test_tree_distance_matches_brute_force_and_bound_reads_inf(self) -> None:
        rng = np.random.default_rng(20260911)
        anchors = rng.uniform(-1.0, 1.0, size=(500, 3))
        queries = np.concatenate(
            [
                anchors[:50] + rng.normal(0.0, 0.01, size=(50, 3)),
                rng.uniform(-3.0, 3.0, size=(150, 3)),
            ]
        )
        tree = build_anchor_tree(anchors)
        exact = nearest_surface_distance(queries, tree)
        oracle = brute_force_nearest_distance(queries, anchors)
        np.testing.assert_allclose(exact, oracle, rtol=0.0, atol=1e-12)
        bounded = nearest_surface_distance(queries, tree, max_distance_m=0.3)
        beyond = oracle > 0.3
        self.assertTrue(beyond.any() and (~beyond).any(), "fixture must straddle the bound")
        self.assertTrue(np.all(np.isinf(bounded[beyond])))
        np.testing.assert_allclose(bounded[~beyond], oracle[~beyond], atol=1e-12)
        self.assertEqual(nearest_surface_distance(np.zeros((0, 3)), tree).shape, (0,))

    def test_outside_box_honours_the_margin(self) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, -0.35]]
        )
        np.testing.assert_array_equal(
            outside_box_mask(points, BOX), [False, True, True, True]
        )
        np.testing.assert_array_equal(
            outside_box_mask(points, BOX, margin_m=0.3), [False, False, True, False]
        )
        with self.assertRaises(ValueError):
            outside_box_mask(points, [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

    def test_config_validation(self) -> None:
        SurfaceAnchorPruneConfig().validate()
        _config().validate()
        for bad in (
            dict(max_distance_m=0.0),
            dict(max_distance_m=-1.0),
            dict(start_step=-1),
            dict(every=0),
            dict(min_age_steps=-5),
            dict(outside_box="drop"),
            dict(enabled="yes"),
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                _config(**bad).validate()
        contract = _config(every=100, min_age_steps=100, outside_box="prune").to_dict()
        self.assertEqual(contract["every"], 100)
        self.assertEqual(contract["box_margin_m"], 0.3)
        self.assertEqual(contract["distance"], "exact_nearest_initialization_point_euclidean_m")

    def test_audit_table_counts_thresholds_and_box(self) -> None:
        tree = build_anchor_tree(PLANE)
        means = np.array(
            [[0.0, 0.0, 0.01], [0.1, 0.1, 0.07], [0.0, 0.0, 0.4], [0.0, 0.0, 2.0]]
        )
        opacity = np.array([0.5, 0.5, 0.01, 0.5])
        report = audit_far_fraction(means, tree, opacity=opacity, box=BOX)
        table = {row["threshold_m"]: row for row in report["far_fraction_table"]}
        self.assertEqual(list(table), list(AUDIT_DISTANCE_THRESHOLDS_M))
        self.assertAlmostEqual(table[0.05]["far_fraction"], 0.75)
        self.assertAlmostEqual(table[0.2]["far_fraction"], 0.5)
        self.assertAlmostEqual(table[1.0]["far_fraction"], 0.25)
        # The 0.4 m row is transparent, so the visible share beyond 0.2 is 1/3.
        self.assertAlmostEqual(table[0.2]["far_fraction_visible"], 1.0 / 3.0)
        self.assertAlmostEqual(report["z_quantiles_far_m"]["p50"], 1.2)
        self.assertAlmostEqual(report["outside_box_fraction"], 0.5)


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class AdapterPruneTests(unittest.TestCase):
    def _adapter(self, anchor=None, **overrides):
        from cloudstudio_3dgs.training.default_strategy_adapter import (
            DefaultStrategyAdapter,
        )

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
            prune_opa_late=0.05,
            prune_switch_step=100000,
            reset_opacity_cap=0.2,
            surface_anchor_prune=anchor,
        )
        settings.update(overrides)
        return DefaultStrategyAdapter(**settings)

    def _population(self, means, opacities=None, grad2d=None):
        count = len(means)
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.tensor(means, dtype=torch.float32)),
                "scales": torch.nn.Parameter(torch.full((count, 3), 0.05).log()),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count)
                ),
                "opacities": torch.nn.Parameter(
                    torch.tensor(
                        [0.5] * count if opacities is None else opacities
                    ).logit()
                ),
                "colors": torch.nn.Parameter(torch.zeros(count, 3)),
            }
        )
        optimizers = {
            name: torch.optim.Adam([parameter], lr=1e-3)
            for name, parameter in params.items()
        }
        # Materialize Adam moments with a distinct per-row signature so a
        # survivor can be recognised after the topology op.
        for name, parameter in params.items():
            rows = torch.arange(1, count + 1, dtype=torch.float32)
            view = (count,) + (1,) * (parameter.ndim - 1)
            parameter.grad = rows.view(view).expand_as(parameter).clone()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        state = {
            "grad2d": torch.zeros(count) if grad2d is None else torch.tensor(grad2d),
            "count": torch.ones(count),
            "radii": torch.zeros(count),
            "scene_scale": 10.0,
        }
        return params, optimizers, state

    def test_cull_event_removes_far_rows_and_keeps_state_aligned(self) -> None:
        anchor = SurfaceAnchorPrune(_config(), PLANE)
        adapter = self._adapter(anchor)
        # Rows: on the surface, 5 cm above it, a 1 m floater, a transparent
        # near row the opacity cull takes anyway.
        params, optimizers, state = self._population(
            [[0.0, 0.0, 0.0], [0.05, 0.05, 0.05], [0.0, 0.0, 1.0], [0.1, 0.0, 0.0]],
            opacities=[0.5, 0.5, 0.5, 0.01],
        )
        adapter._ensure_lineage(params, state)
        state["_cloudstudio_birth_step"][2] = 100
        state["_cloudstudio_birth_kind"][2] = 1
        exp_avg_before = optimizers["means"].state[params["means"]]["exp_avg"].clone()
        means_before = params["means"].detach().clone()

        culled = adapter._prune_mipmap(params, optimizers, state, step=700)

        self.assertEqual(culled, 2)
        self.assertEqual(len(params["means"]), 2)
        torch.testing.assert_close(params["means"].detach(), means_before[[0, 1]])
        exp_avg_after = optimizers["means"].state[params["means"]]["exp_avg"]
        torch.testing.assert_close(exp_avg_after, exp_avg_before[[0, 1]])
        self.assertEqual(len(state["_cloudstudio_birth_step"]), 2)
        self.assertEqual(state["_cloudstudio_birth_step"].tolist(), [-1, -1])
        event = adapter._last_surface_anchor_event
        self.assertEqual(event["candidates"], 1)
        self.assertEqual(event["pruned_far"], 1)
        self.assertEqual(event["pruned_outside"], 0)
        self.assertEqual(event["pruned"], 1)
        self.assertEqual(event["remaining"], 2)
        self.assertAlmostEqual(event["far_fraction"], 0.25)
        self.assertEqual(event["pruned_total"], 1)
        self.assertEqual(state["_cloudstudio_surface_anchor_pruned_total"], 1)
        self.assertEqual(adapter._last_cull_event["surface_anchor_count"], 1)
        self.assertEqual(adapter._last_cull_event["selected_opacity_count"], 1)

    def test_before_start_step_and_disabled_are_identical(self) -> None:
        rows = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]]
        outcomes = []
        for anchor in (None, SurfaceAnchorPrune(_config(start_step=800), PLANE)):
            adapter = self._adapter(anchor)
            params, optimizers, state = self._population(rows)
            adapter._ensure_lineage(params, state)
            culled = adapter._prune_mipmap(params, optimizers, state, step=700)
            outcomes.append((culled, params["means"].detach().clone(), dict(adapter._last_cull_event)))
        self.assertEqual(outcomes[0][0], 0)
        self.assertEqual(outcomes[1][0], 0)
        torch.testing.assert_close(outcomes[0][1], outcomes[1][1])
        self.assertEqual(outcomes[0][2], outcomes[1][2])
        self.assertIsNone(self._adapter(None)._last_surface_anchor_event)
        self.assertIsNone(self._adapter(None).state_dict()["surface_anchor_prune"])
        # An off-cadence step after refine stop is a no-op with and without it.
        for anchor in (None, SurfaceAnchorPrune(_config(every=100), PLANE)):
            adapter = self._adapter(anchor)
            params, optimizers, state = self._population(rows)
            adapter.step_post_backward(
                params=params, optimizers=optimizers, state=state, step=2150, info={}
            )
            self.assertEqual(len(params["means"]), 3)
            self.assertIsNone(adapter.last_lifecycle_event)

    def test_min_age_spares_newborns_until_they_age(self) -> None:
        anchor = SurfaceAnchorPrune(_config(min_age_steps=100), PLANE)
        adapter = self._adapter(anchor)
        params, optimizers, state = self._population(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.5]]
        )
        adapter._ensure_lineage(params, state)
        state["_cloudstudio_birth_step"][1] = 700  # born this event
        state["_cloudstudio_birth_step"][2] = 500  # two intervals old
        culled = adapter._prune_mipmap(params, optimizers, state, step=700)
        self.assertEqual(culled, 1)
        self.assertEqual(len(params["means"]), 2)
        self.assertAlmostEqual(float(params["means"][1, 2].detach()), 1.0, places=2)
        event = adapter._last_surface_anchor_event
        self.assertEqual(event["protected_young"], 1)
        self.assertEqual(event["candidates"], 2)
        # One interval later the same newborn is old enough.
        culled = adapter._prune_mipmap(params, optimizers, state, step=800)
        self.assertEqual(culled, 1)
        self.assertEqual(len(params["means"]), 1)
        self.assertEqual(state["_cloudstudio_surface_anchor_pruned_total"], 2)

    def test_outside_box_prune_removes_supported_rows_beyond_the_grown_box(self) -> None:
        # A second patch of anchors 1 m away in x supports rows that are near
        # a surface but outside the Tile box: only outside_box="prune" takes them.
        anchors = np.concatenate([PLANE, PLANE + np.array([1.0, 0.0, 0.0])])
        # x=1 is on the second patch but 0.45 m past the grown box; the third
        # row is 0.5 m from both patches yet inside the grown box (z 0.4 =
        # 0.1 + 0.3 is on the edge, not outside).
        rows = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.4]]
        keep = SurfaceAnchorPrune(_config(outside_box="keep"), anchors, box=BOX)
        adapter = self._adapter(keep)
        params, optimizers, state = self._population(rows)
        adapter._ensure_lineage(params, state)
        # "keep": only the far row goes; the supported row past the box stays.
        self.assertEqual(adapter._prune_mipmap(params, optimizers, state, step=700), 1)
        self.assertEqual(adapter._last_surface_anchor_event["pruned_outside"], 0)
        self.assertEqual(adapter._last_surface_anchor_event["outside_count"], 0)

        prune = SurfaceAnchorPrune(_config(outside_box="prune"), anchors, box=BOX)
        adapter = self._adapter(prune)
        params, optimizers, state = self._population(rows)
        adapter._ensure_lineage(params, state)
        culled = adapter._prune_mipmap(params, optimizers, state, step=700)
        self.assertEqual(culled, 2)
        self.assertEqual(len(params["means"]), 1)
        event = adapter._last_surface_anchor_event
        self.assertEqual(event["pruned_outside"], 1)
        self.assertEqual(event["pruned_far"], 1)
        self.assertEqual(event["outside_count"], 1)
        with self.assertRaises(ValueError):
            SurfaceAnchorPrune(_config(outside_box="prune"), anchors, box=None)

    def test_extra_cadence_acts_after_refine_stop_and_remaps_vendor_gradients(self) -> None:
        anchor = SurfaceAnchorPrune(_config(every=100), PLANE)
        adapter = self._adapter(
            anchor,
            lifecycle_execution_order="pre_optimizer_vendor",
            growth_min_opacity=0.15,
        )
        params, optimizers, state = self._population(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.1, 0.0]]
        )
        row_gradient = torch.tensor([1.0, 2.0, 3.0])
        for parameter in params.values():
            view = (3,) + (1,) * (parameter.ndim - 1)
            parameter.grad = row_gradient.view(view).expand_as(parameter).clone()
        adapter.step_post_backward(
            params=params, optimizers=optimizers, state=state, step=2200, info={}
        )
        self.assertEqual(len(params["means"]), 2)
        event = adapter.last_lifecycle_event
        self.assertEqual(event["kind"], "surface_anchor_prune")
        self.assertEqual(event["cull_count"], 1)
        self.assertEqual(event["surface_anchor_prune"]["pruned_far"], 1)
        self.assertTrue(event["current_step_gradient_remapped"])
        torch.testing.assert_close(params["means"].grad[:, 0], torch.tensor([1.0, 3.0]))
        self.assertNotIn("_cloudstudio_current_step_gradient_source", state)
        # A cull event on the same cadence does not double-prune.
        adapter = self._adapter(SurfaceAnchorPrune(_config(every=100), PLANE), post_refine_cull_every=100)
        params, optimizers, state = self._population([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        adapter.step_post_backward(
            params=params, optimizers=optimizers, state=state, step=2200, info={}
        )
        self.assertEqual(adapter.last_lifecycle_event["kind"], "post_refine_cull")
        self.assertEqual(adapter.last_lifecycle_event["surface_anchor_prune"]["pruned"], 1)
        self.assertEqual(len(params["means"]), 1)

    def test_reject_unsupported_parents_blocks_growth_from_far_rows(self) -> None:
        anchor = SurfaceAnchorPrune(
            _config(reject_unsupported_parents=True, start_step=10_000), PLANE
        )
        adapter = self._adapter(anchor)
        params, optimizers, state = self._population(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], grad2d=[0.001, 0.001]
        )
        adapter._ensure_lineage(params, state)
        clone_count, split_count = adapter._grow_mipmap(params, optimizers, state)
        self.assertEqual((clone_count, split_count), (1, 0))
        self.assertEqual(len(params["means"]), 3)
        self.assertAlmostEqual(float(params["means"][-1, 2].detach()), 0.0, places=2)
        growth = adapter._last_growth_event
        self.assertEqual(growth["surface_anchor_rejected_count"], 1)
        self.assertEqual(growth["selected_parent_count"], 1)
        # Without the flag the far row breeds.
        adapter = self._adapter(SurfaceAnchorPrune(_config(start_step=10_000), PLANE))
        params, optimizers, state = self._population(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], grad2d=[0.001, 0.001]
        )
        adapter._ensure_lineage(params, state)
        self.assertEqual(adapter._grow_mipmap(params, optimizers, state), (2, 0))
        self.assertEqual(adapter._last_growth_event["surface_anchor_rejected_count"], 0)


# ----------------------------------------------------------------------------
# trainer config, contract, gate
# ----------------------------------------------------------------------------


def _trainer_dict(**overrides) -> dict:
    config = {
        "run_id": "anchor-trainer",
        "trainer_preset": "custom",
        "dataset_manifest": "dataset.json",
        "recording_root": "recording",
        "mask_manifest": "masks.json",
        "mask_root": "masks",
        "split_manifest": "split.json",
        "initialization_ply": "sparse_pc.ply",
        "output_dir": "run",
        "gsplat_lock": "upstream/gsplat.lock.json",
        "require_person_masks": False,
        "lidar_range_weight": 0.0,
        "max_steps": 3000,
        "checkpoint_every": 500,
        "color_model": "sh",
        "sh_degree": 1,
        "sh_degree_interval": 0,
        "densification_strategy": "default_3dgs",
        "topology_policy": {"mode": "adaptive_growth"},
        "mcmc_refine_start_iter": 500,
        "mcmc_refine_stop_iter": 2100,
        "mcmc_refine_every": 100,
        "default_strategy": {
            "exact_mipmap_lifecycle": True,
            "refine_start_iter": 500,
            "refine_stop_iter": 2100,
            "refine_every": 100,
            "refine_scale2d_stop_iter": 2100,
            "reset_every": 300,
            "absgrad": True,
            "grow_grad2d": 0.00015,
            "split_scale_m": 0.2,
            "prune_scale_m": 0.2,
            "prune_opa": 0.1,
            "prune_opa_late": 0.05,
            "prune_switch_step": 1500,
            "prune_scale2d": 0.15,
            "reset_opacity_cap": 0.2,
        },
    }
    config.update(overrides)
    return config


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class TrainerConfigTests(unittest.TestCase):
    def test_contract_gains_the_key_only_when_enabled(self) -> None:
        from cloudstudio_3dgs.training.trainer import TrainerConfig

        # validate() of an exact-lifecycle config needs the signed renderer
        # mask files on disk; the real S1 configs are validated in
        # research/quality_recovery_v2/10_surface_anchor_prune.md. The contract
        # identity is a pure function of the dict and is pinned here.
        default = TrainerConfig.from_dict(_trainer_dict())
        explicit = TrainerConfig.from_dict(
            _trainer_dict(surface_anchor_prune={"enabled": False, "max_distance_m": 0.1})
        )
        explicit.surface_anchor_prune.validate()
        self.assertNotIn("surface_anchor_prune", default.contract_dict()["strategy"])
        self.assertEqual(default.contract_dict(), explicit.contract_dict())

        enabled = TrainerConfig.from_dict(
            _trainer_dict(
                surface_anchor_prune={
                    "enabled": True,
                    "max_distance_m": 0.3,
                    "start_step": 700,
                    "min_age_steps": 100,
                }
            )
        )
        enabled.surface_anchor_prune.validate()
        contract = enabled.contract_dict()
        self.assertEqual(
            contract["strategy"]["surface_anchor_prune"],
            enabled.surface_anchor_prune.to_dict(),
        )
        without = copy.deepcopy(contract)
        without["strategy"].pop("surface_anchor_prune")
        self.assertEqual(without, default.contract_dict())

    def test_validation_requires_the_classic_exact_lifecycle_and_tile_box(self) -> None:
        from cloudstudio_3dgs.training.trainer import TrainerConfig

        enabled = {"enabled": True, "start_step": 700}
        with self.assertRaisesRegex(ValueError, "exact_mipmap_lifecycle"):
            TrainerConfig.from_dict(
                _trainer_dict(
                    surface_anchor_prune=enabled,
                    densification_strategy="error_weighted_mcmc",
                    default_strategy={},
                )
            ).validate()
        with self.assertRaisesRegex(ValueError, "tile_inputs_manifest"):
            TrainerConfig.from_dict(
                _trainer_dict(surface_anchor_prune=dict(enabled, outside_box="prune"))
            ).validate()
        with self.assertRaisesRegex(ValueError, "before max_steps"):
            TrainerConfig.from_dict(
                _trainer_dict(surface_anchor_prune=dict(enabled, start_step=3000))
            ).validate()
        with self.assertRaisesRegex(ValueError, "must be an object"):
            TrainerConfig.from_dict(_trainer_dict(surface_anchor_prune=[]))
        with self.assertRaisesRegex(ValueError, "outside_box"):
            TrainerConfig.from_dict(
                _trainer_dict(surface_anchor_prune=dict(enabled, outside_box="x"))
            ).validate()


def _upstream_data_gate() -> dict:
    sha = {
        name: value * 64
        for name, value in (
            ("dataset", "1"),
            ("split", "2"),
            ("face", "3"),
            ("mask", "4"),
            ("da2", "5"),
            ("tile", "6"),
        )
    }
    return sign_gate(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "profile": GATE_PROFILE,
            "status": UPSTREAM_DATA_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": ["training implementation is not aligned"],
            "bindings": {
                "training_dataset_manifest_sha256": sha["dataset"],
                "split_manifest_sha256": sha["split"],
                "face4_train_manifest_sha256": sha["face"],
                "face4_val_manifest_sha256": sha["face"],
                "training_circle_mask_manifest_sha256": sha["mask"],
                "da2_train_manifest_sha256": sha["da2"],
                "spatial_tile_plan_manifest_sha256": sha["tile"],
            },
        }
    )


def _signed(config: dict) -> dict:
    signed = copy.deepcopy(config)
    signed.pop("config_manifest_sha256", None)
    signed["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(signed)
    ).hexdigest()
    return signed


# The parity arm tests/test_rgb_supervision_mask.py signs.
PARITY_ARM = {
    "run_id": "v26-boundary",
    "mipmap_tile_id": 1,
    "topology_policy": {"mode": "adaptive_growth"},
    "densification_strategy": "default_3dgs",
    "densification_gradient_source": "rgb_only",
    "mcmc_refine_start_iter": 500,
    "mcmc_refine_every": 100,
    "mcmc_refine_stop_iter": 5610,
    "mcmc_noise_injection_stop_iter": 0,
    "mcmc_noise_lr": 0.0,
    "max_steps": 7480,
    "controlled_stop_after_steps": 502,
    "factor": 1,
    "cap_max": 2200000,
    "sh_degree": 0,
    "da2_depth_weight": 0.0,
    "error_weighted_sampling": {"enabled": False},
    "default_strategy": {
        "exact_mipmap_lifecycle": True,
        "grow_grad2d": 0.00015,
        "growth_min_opacity": 0.15,
        "split_scale_m": 0.2,
        "prune_scale_m": 0.2,
        "prune_opa": 0.1,
        "prune_opa_late": 0.05,
        "prune_switch_step": 3740,
        "prune_scale2d": 0.15,
        "reset_every": 300,
        "reset_opacity_cap": 0.2,
        "absgrad": True,
        "revised_opacity": True,
    },
    "tangent_proposal": {"enabled": True, "reject_unsupported_births": True},
    "geometry_regularization": {
        "enabled": True,
        "opacity_sparsity_weight": 1e-4,
        "scale_upper_weight": 1e-4,
        "anisotropy_weight": 1e-4,
        "max_scale_ratio_to_reference": 8.0,
        "max_anisotropy": 10.0,
    },
}


class GateTests(unittest.TestCase):
    def test_adaptive_growth_gate_refuses_an_enabled_arm(self) -> None:
        gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), _signed(PARITY_ARM), stage="boundary"
        )
        self.assertTrue(gate["training_allowed"])
        disabled = dict(PARITY_ARM, surface_anchor_prune={"enabled": False})
        self.assertTrue(
            advance_adaptive_growth_gate(
                _upstream_data_gate(), _signed(disabled), stage="boundary"
            )["training_allowed"]
        )
        research = dict(
            PARITY_ARM, surface_anchor_prune={"enabled": True, "max_distance_m": 0.3}
        )
        with self.assertRaisesRegex(ValueError, "surface_anchor_prune is a research departure"):
            advance_adaptive_growth_gate(
                _upstream_data_gate(), _signed(research), stage="boundary"
            )
        tampered = _signed(research)
        tampered["max_steps"] = 7481
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            advance_adaptive_growth_gate(_upstream_data_gate(), tampered, stage="boundary")


# ----------------------------------------------------------------------------
# audit tool
# ----------------------------------------------------------------------------


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    records = np.zeros(len(xyz), dtype=np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))
    records["xyz"] = xyz.astype(np.float32)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        records.tofile(stream)


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class AuditToolTests(unittest.TestCase):
    def test_audit_tool_reproduces_the_table_on_a_synthetic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_ply(root / "init.ply", PLANE)
            means = torch.tensor(
                [[0.0, 0.0, 0.01], [0.1, 0.1, 0.07], [0.0, 0.0, 0.4], [0.0, 0.0, 2.0]]
            )
            payload = {
                "schema_version": 1,
                "step": 700,
                "params": {
                    "means": means,
                    "opacities": torch.tensor([0.5, 0.5, 0.01, 0.5]).logit(),
                },
            }
            torch.save(payload, root / "latest.pt")
            output = root / "report.json"
            environment = dict(os.environ, PYTHONPATH=str(ROOT))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(AUDIT_TOOL),
                    "--checkpoint",
                    str(root / "latest.pt"),
                    "--init-ply",
                    str(root / "init.ply"),
                    "--box",
                    json.dumps(BOX),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                env=environment,
                cwd=str(ROOT),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            table = {row["threshold_m"]: row for row in report["far_fraction_table"]}
            self.assertEqual(report["gaussian_count"], 4)
            self.assertEqual(report["anchor_count"], len(PLANE))
            self.assertEqual(report["checkpoint_step"], 700)
            self.assertAlmostEqual(table[0.2]["far_fraction"], 0.5)
            self.assertAlmostEqual(table[0.2]["far_fraction_visible"], 1.0 / 3.0)
            self.assertAlmostEqual(table[1.0]["far_fraction"], 0.25)
            self.assertAlmostEqual(report["outside_box_fraction"], 0.5)
            self.assertIn("far_all", completed.stdout)
            self.assertIn("outside training box: 0.500", completed.stdout)


if __name__ == "__main__":
    unittest.main()


class GrowthOnlyModeTests(unittest.TestCase):
    def test_cull_false_never_becomes_due_but_keeps_the_growth_gate(self):
        from cloudstudio_3dgs.training.surface_anchor import SurfaceAnchorPruneConfig

        config = SurfaceAnchorPruneConfig(enabled=True, start_step=0, every=100, reject_unsupported_parents=True, cull=False)
        config.validate()
        self.assertFalse(config.cull)
        self.assertEqual(config.to_dict()["cull"], False)
        with self.assertRaises(ValueError):
            SurfaceAnchorPruneConfig(enabled=True, reject_unsupported_parents=False, cull=False).validate()
        # a config that only gates growth must never schedule a removal
        import types
        fake = types.SimpleNamespace(config=config)
        from cloudstudio_3dgs.training.surface_anchor import SurfaceAnchorPrune

        self.assertFalse(SurfaceAnchorPrune.due(fake, 5000))
        self.assertFalse(SurfaceAnchorPrune.extra_due(fake, 5000))
        default = SurfaceAnchorPruneConfig(enabled=True, start_step=0, every=100)
        fake_default = types.SimpleNamespace(config=default)
        self.assertTrue(SurfaceAnchorPrune.due(fake_default, 5000))
        self.assertTrue(SurfaceAnchorPrune.extra_due(fake_default, 5000))

