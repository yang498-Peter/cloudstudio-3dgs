"""WP03 diagnostic sets: view selection, derived signed manifests, ROI boxes,
short-horizon arm configs (tools/build_diagnostic_set.py,
tools/make_diagnostic_arm_config.py).  Torch-free, no dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_diagnostic_set import (  # noqa: E402
    derive_tile_geometry_manifest,
    derive_tile_inputs_manifest,
    preset_dir_name,
    roi_bbox_in_crop,
    select_views,
    sign_tile_inputs_manifest,
)
from make_diagnostic_arm_config import (  # noqa: E402
    DEFAULT_PLAN,
    VARIANTS,
    build_diagnostic_arm,
    build_eval_config,
    diag_run_id,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402
from cloudstudio_3dgs.training.mipmap_tile_geometry import (  # noqa: E402
    sign_tile_geometry_manifest,
    verify_tile_geometry_manifest,
)
from cloudstudio_3dgs.training.schedule_audit import (  # noqa: E402
    RESEARCH_SCHEDULE_CONTRACT_V1,
    research_schedule_contract,
    resolved_schedule,
    validate_research_schedule_contract,
)
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest  # noqa: E402

DIAG_CONFIG_DIR = ROOT / "run_configs" / "house0305_tiles" / "diag_v2"


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------


def _row(image_id, camera, frame, support, rng_med, lapvar, *, n=100, rng_min=None, luma=100.0):
    return {
        "image_id": image_id,
        "camera_id": camera,
        "rig_frame_id": frame,
        "timestamp_ns": 0,
        "n_samples": n,
        "n_effective": int(round(support * n)),
        "support_fraction": support,
        "range_min_m": rng_med * 0.8 if rng_min is None else rng_min,
        "range_median_m": rng_med,
        "sharpness_lapvar_median": lapvar,
        "luma_median": luma,
        "effective_by_face": {"yaw_plus_35": int(round(support * n))},
    }


def _table():
    return [
        _row("img_a", "left", "rig_1", 0.90, 3.0, 800.0),
        _row("img_b", "right", "rig_1", 0.85, 2.5, 1500.0),   # sharpest near view
        _row("img_c", "left", "rig_2", 0.80, 6.0, 2500.0),    # sharp but far
        _row("img_d", "right", "rig_2", 0.75, 4.0, 600.0),
        _row("img_e", "left", "rig_3", 0.70, 1.5, 900.0),
        _row("img_f", "right", "rig_3", 0.55, 7.0, 300.0),
        _row("img_g", "left", "rig_4", 0.40, 2.0, 3000.0),    # below threshold
        _row("img_h", "right", "rig_4", 0.10, 9.0, 100.0),
        _row("img_i", "left", "rig_5", 0.00, float("nan"), float("nan"), n=100),
    ]


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _base_tile_inputs(root: Path) -> dict:
    tiles = []
    for tile_id, name, images in ((0, "Tile_0", ["img_a", "img_b"]), (1, "Tile_1", ["img_a", "img_b", "img_c"])):
        payload = f"ply-{name}".encode("ascii") * 100
        sha = _write(root / name / "initialization_full_lidar.ply", payload)
        views = []
        index = 0
        for image in images:
            for face in ("yaw_minus_35", "yaw_plus_35", "pitch_up_56"):
                index += 1
                views.append({"image_index": index, "x": 10 * index, "y": 0, "width": 200, "height": 300,
                              "pixel_load": 60000, "sample_id": f"{image}::{face}"})
        tiles.append({
            "tile_id": tile_id, "name": name,
            "core_box": [[0, 0, 0], [1, 1, 1]], "training_and_export_box": [[0, 0, 0], [1, 1, 1]],
            "view_count": len(views), "views": views,
            "initialization": {"kind": "full_lidar_roi_with_tile_halo", "path": f"{name}/initialization_full_lidar.ply",
                               "point_count": 123, "sha256": sha, "bytes": len(payload)},
            "recommended_training": {"resolution_level": 1, "steps": 20 * len(views), "stage_epochs": [5, 10, 5], "sh_degree": 1},
        })
    return sign_tile_inputs_manifest({
        "schema_version": 1, "kind": "lidar_adaptive_tile_training_inputs_v1",
        "tile_plan_manifest_sha256": "p" * 64,
        "source_point_cloud": {"path": "colorized.las", "sha256": "s" * 64},
        "halo_policy": "retain_full_training_and_export_box",
        "tile_count": 2, "tiles": tiles,
    })


def _base_tile_geometry(root: Path, tile_inputs: dict) -> dict:
    tiles = []
    for tile in tile_inputs["tiles"]:
        payload = f"npz-{tile['name']}".encode("ascii") * 50
        sha = _write(root / tile["name"] / "initialization_geometry_k7_k30.npz", payload)
        tiles.append({
            "tile_id": tile["tile_id"], "name": tile["name"],
            "initialization_ply_sha256": tile["initialization"]["sha256"],
            "point_count": tile["initialization"]["point_count"],
            "geometry": {"path": f"{tile['name']}/initialization_geometry_k7_k30.npz", "sha256": sha, "bytes": len(payload),
                         "arrays": {"normals": [123, 3]}},
            "report": {"algorithm": "synthetic"},
        })
    return sign_tile_geometry_manifest({
        "schema_version": 1, "kind": "mipmap_k7_k30_tile_initialization_geometry_v1",
        "tile_inputs_manifest_sha256": tile_inputs["tile_inputs_manifest_sha256"],
        "tile_count": 2, "tiles": tiles, "training_allowed": False,
        "next_required_artifact": "tile_face4_crop_consumption_manifest",
    })


def _base_arm() -> dict:
    """A tile1 R1-shaped base: 20-epoch max_steps truncated at 20000, window to 14000."""
    return {
        "run_id": "house0305-t1-R1d",
        "output_dir": "C:\\runs\\tile1_R1d_20k",
        "device": "cuda:0",
        "seed": 42,
        "trainer_preset": "custom",
        "mipmap_tile_id": 1,
        "tile_inputs_manifest": "C:\\runs\\tile_inputs_v9\\tile_inputs_manifest.json",
        "tile_inputs_root": "C:\\runs\\tile_inputs_v9",
        "initialization_ply": "C:\\runs\\tile_inputs_v9\\Tile_1\\initialization_full_lidar.ply",
        "initialization_geometry_manifest": "C:\\runs\\tile_geometry_v9\\tile_geometry_manifest.json",
        "initialization_geometry": "C:\\runs\\tile_geometry_v9\\Tile_1\\initialization_geometry_k7_k30.npz",
        "background_image_manifest": "C:\\runs\\tile_backgrounds_v9\\Tile_1\\background_manifest.json",
        "background_image_root": "C:\\runs\\tile_backgrounds_v9\\Tile_1",
        "metric_scale_calibration": {"mode": "precomputed", "knn_neighbors": 7, "scale_multiplier": 1.0},
        "max_steps": 36580,
        "checkpoint_every": 5000,
        "sh_degree": 1,
        "sh_degree_interval": 0,
        "color_model": "sh",
        "view_sampling_mode": "fisher_yates_without_replacement_per_epoch",
        "densification_strategy": "default_3dgs",
        "densification_gradient_source": "total_loss",
        "learning_rates": {"means": 1.6e-05, "scales": 0.005, "quats": 0.001, "opacities": 0.05, "colors": 0.0025},
        "means_lr_final_factor": 0.01,
        "mcmc_refine_start_iter": 500,
        "mcmc_refine_stop_iter": 14000,
        "mcmc_refine_every": 100,
        "default_strategy": {
            "exact_mipmap_lifecycle": True,
            "lifecycle_execution_order": "pre_optimizer_vendor",
            "prune_opa": 0.1, "prune_opa_late": 0.05, "prune_switch_step": 18290,
            "reset_every": 300, "refine_scale2d_stop_iter": 14000,
            "refine_start_iter": 500, "refine_stop_iter": 14000, "refine_every": 100,
            "vendor_cull_warmup_profile": "exact_0p10_to_0p05",
            "vendor_opacity_reset_profile": "exact_every300",
        },
        "controlled_stop_after_steps": 20000,
    }


def _selection(count=5, tile_id=1, region="indoor_door_leaf_Tile_1"):
    rows = select_views(_table(), count)["selected"]
    sample_ids = [f"{r['image_id']}::yaw_plus_35" for r in rows]
    return {
        "region": {"label": region, "tile": f"Tile_{tile_id}", "tile_id": tile_id, "status": "provisional",
                   "world_box": [[6.0, -3.0, 1.6], [7.6, -1.6, 3.2]]},
        "preset": "U1" if count == 5 else ("U0" if count == 1 else "DIAG"),
        "count": count,
        "policy": {"rule": "synthetic"},
        "tile_inputs_manifest_sha256": "i" * 64,
        "tile_geometry_manifest_sha256": "g" * 64,
        "view_count": len(sample_ids),
        "view_sample_ids": sample_ids,
        "selected": rows,
        "roi_in_crops": [{"sample_id": s, "image_id": s.split("::")[0], "camera_id": "left",
                          "roi": {"x0": 1, "y0": 2, "x1": 30, "y1": 40}} for s in sample_ids],
    }


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------


class SelectViewsTests(unittest.TestCase):
    def test_u0_is_sharpest_near_supported_view(self):
        out = select_views(_table(), 1)
        self.assertEqual(out["preset"], "U0")
        self.assertEqual([r["image_id"] for r in out["selected"]], ["img_b"])
        self.assertFalse(out["policy"]["threshold_relaxed"])
        self.assertFalse(out["policy"]["near_filter_relaxed"])
        # img_c is sharper but 6 m away; img_g is sharper and near but under-supported

    def test_u0_prefers_a_near_view_at_the_support_floor_over_a_far_full_view(self):
        table = [
            _row("img_far", "left", "rig_1", 0.95, 7.0, 3000.0),
            _row("img_near_a", "right", "rig_2", 0.35, 1.4, 900.0),
            _row("img_near_b", "right", "rig_3", 0.30, 1.5, 1200.0),
            _row("img_near_low", "left", "rig_4", 0.10, 1.2, 5000.0),  # under the floor
        ]
        out = select_views(table, 1)
        self.assertEqual([r["image_id"] for r in out["selected"]], ["img_near_b"])
        self.assertEqual(out["policy"]["near_pool_stage"], "near_and_support_floor")
        self.assertTrue(out["policy"]["near_filter_relaxed"])
        out = select_views(table[:1], 1)
        self.assertEqual(out["policy"]["near_pool_stage"], "far_fallback")

    def test_ranking_bins_support_so_range_decides_within_a_bin(self):
        table = [
            _row("img_x", "left", "rig_1", 0.52, 7.0, 800.0),
            _row("img_y", "left", "rig_2", 0.50, 3.0, 800.0),
            _row("img_z", "left", "rig_3", 0.61, 7.5, 800.0),
        ]
        self.assertEqual([r["image_id"] for r in select_views(table, 3)["selected"]], ["img_z", "img_y", "img_x"])

    def test_u1_alternates_cameras_best_first(self):
        out = select_views(_table(), 5)
        ids = [r["image_id"] for r in out["selected"]]
        self.assertEqual(ids, ["img_a", "img_b", "img_c", "img_d", "img_e"])
        self.assertEqual(out["policy"]["cameras_represented"], ["left", "right"])
        self.assertEqual([r["selection_rank"] for r in out["selected"]], [0, 1, 2, 3, 4])

    def test_u1_keeps_alternating_when_one_camera_runs_dry(self):
        table = [
            _row("img_a", "left", "rig_1", 0.9, 3.0, 800.0),
            _row("img_b", "right", "rig_1", 0.9, 3.0, 800.0),  # stereo partner of img_a is eligible
            _row("img_c", "left", "rig_2", 0.8, 3.0, 800.0),
            _row("img_d", "left", "rig_3", 0.7, 3.0, 800.0),
        ]
        out = select_views(table, 3)
        self.assertEqual([r["image_id"] for r in out["selected"]], ["img_a", "img_b", "img_c"])
        self.assertEqual(out["policy"]["short_by"], 0)
        self.assertEqual([r["image_id"] for r in select_views(table, 4)["selected"]], ["img_a", "img_b", "img_c", "img_d"])

    def test_diag_takes_every_candidate_when_short_and_balances_cameras(self):
        out = select_views(_table(), 40)
        self.assertEqual(out["preset"], "DIAG")
        ids = [r["image_id"] for r in out["selected"]]
        # relaxed down to the 0.2 floor: img_g (0.4) enters, img_h (0.1) never does
        self.assertEqual(sorted(ids), ["img_a", "img_b", "img_c", "img_d", "img_e", "img_f", "img_g"])
        self.assertEqual(ids[:2], ["img_a", "img_b"])  # best of each camera first
        self.assertTrue(out["policy"]["threshold_relaxed"])
        self.assertAlmostEqual(out["policy"]["min_support_fraction_applied"], 0.2)
        self.assertEqual(out["policy"]["short_by"], 33)

    def test_threshold_relaxes_to_fill_count_and_is_recorded(self):
        out = select_views(_table(), 6, min_support_fraction=0.9)
        self.assertTrue(out["policy"]["threshold_relaxed"])
        self.assertAlmostEqual(out["policy"]["min_support_fraction_applied"], 0.55)
        self.assertEqual(len(out["selected"]), 6)
        out = select_views(_table(), 8, min_support_fraction=0.9)
        self.assertAlmostEqual(out["policy"]["min_support_fraction_applied"], 0.2)  # floor, not img_h's 0.10
        self.assertEqual(len(out["selected"]), 7)
        self.assertNotIn("img_h", [r["image_id"] for r in out["selected"]])
        self.assertNotIn("img_i", [r["image_id"] for r in out["selected"]])  # zero effective never enters

    def test_no_effective_view_raises(self):
        with self.assertRaises(ValueError):
            select_views([_row("img_i", "left", "rig_5", 0.0, float("nan"), float("nan"))], 1)

    def test_preset_dir_names(self):
        self.assertEqual([preset_dir_name(c) for c in (1, 5, 40, 7)], ["U0_1", "U1_5", "DIAG_40", "N7_7"])


# ----------------------------------------------------------------------------
# derived manifests
# ----------------------------------------------------------------------------


class DerivedManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inputs_root = self.root / "tile_inputs_v9"
        self.base_inputs = _base_tile_inputs(self.inputs_root)
        self.geometry_root = self.root / "tile_geometry_v9"
        self.base_geometry = _base_tile_geometry(self.geometry_root, self.base_inputs)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tile_inputs_restricted_to_selected_images_and_signed(self):
        derived = derive_tile_inputs_manifest(
            self.base_inputs, tile_name="Tile_1", image_ids=["img_c", "img_a"], provenance={"preset": "U1"}
        )
        sha = verify_tile_inputs_manifest(derived, root=self.inputs_root, verify_artifacts=True)
        self.assertEqual(sha, derived["tile_inputs_manifest_sha256"])
        self.assertNotEqual(sha, self.base_inputs["tile_inputs_manifest_sha256"])
        self.assertEqual(derived["tile_count"], 1)
        tile = derived["tiles"][0]
        self.assertEqual(tile["name"], "Tile_1")
        ids = [v["sample_id"] for v in tile["views"]]
        self.assertEqual(len(ids), 6)
        self.assertTrue(all(s.split("::")[0] in {"img_a", "img_c"} for s in ids))
        self.assertEqual(ids, [s for s in [v["sample_id"] for v in self.base_inputs["tiles"][1]["views"]] if s in ids])  # order kept
        self.assertEqual(tile["view_count"], 6)
        self.assertEqual(tile["recommended_training"]["steps"], 120)
        self.assertEqual(tile["initialization"], self.base_inputs["tiles"][1]["initialization"])  # verbatim
        self.assertEqual(derived["tile_plan_manifest_sha256"], self.base_inputs["tile_plan_manifest_sha256"])
        self.assertEqual(derived["diagnostic"]["derived_from_tile_inputs_manifest_sha256"], self.base_inputs["tile_inputs_manifest_sha256"])
        self.assertEqual(derived["diagnostic"]["parent_image_ids"], ["img_c", "img_a"])
        self.assertEqual(derived["diagnostic"]["preset"], "U1")

    def test_tile_inputs_face_restriction(self):
        derived = derive_tile_inputs_manifest(
            self.base_inputs, tile_name="Tile_1", image_ids=["img_b"], provenance={},
            face_ids_by_image={"img_b": ["pitch_up_56"]},
        )
        self.assertEqual([v["sample_id"] for v in derived["tiles"][0]["views"]], ["img_b::pitch_up_56"])
        self.assertTrue(derived["diagnostic"]["faces_restricted"])

    def test_tile_inputs_rejects_unknown_image_duplicates_and_tampering(self):
        with self.assertRaises(ValueError):
            derive_tile_inputs_manifest(self.base_inputs, tile_name="Tile_1", image_ids=["img_zz"], provenance={})
        with self.assertRaises(ValueError):
            derive_tile_inputs_manifest(self.base_inputs, tile_name="Tile_1", image_ids=["img_a", "img_a"], provenance={})
        with self.assertRaises(ValueError):
            derive_tile_inputs_manifest(self.base_inputs, tile_name="Tile_9", image_ids=["img_a"], provenance={})
        tampered = copy.deepcopy(self.base_inputs)
        tampered["tiles"][1]["views"][0]["x"] += 1
        with self.assertRaises(ValueError):
            derive_tile_inputs_manifest(tampered, tile_name="Tile_1", image_ids=["img_a"], provenance={})
        derived = derive_tile_inputs_manifest(self.base_inputs, tile_name="Tile_1", image_ids=["img_a"], provenance={})
        derived["tiles"][0]["views"][0]["width"] += 1
        with self.assertRaises(ValueError):
            verify_tile_inputs_manifest(derived)

    def test_tile_geometry_bound_to_derived_inputs_and_verifies_with_relative_path(self):
        derived_inputs = derive_tile_inputs_manifest(self.base_inputs, tile_name="Tile_1", image_ids=["img_a"], provenance={})
        preset_dir = self.root / "diag_v2" / "region" / "U0_1"
        preset_dir.mkdir(parents=True)
        derived = derive_tile_geometry_manifest(
            self.base_geometry, tile_id=1, geometry_path="../../../tile_geometry_v9/Tile_1/initialization_geometry_k7_k30.npz",
            tile_inputs_manifest_sha256=derived_inputs["tile_inputs_manifest_sha256"], provenance={"preset": "U0"},
        )
        sha = verify_tile_geometry_manifest(derived, root=preset_dir, verify_artifacts=True)
        self.assertEqual(sha, derived["tile_geometry_manifest_sha256"])
        # the trainer's binding check (trainer.validate) compares exactly these two
        self.assertEqual(derived["tile_inputs_manifest_sha256"], derived_inputs["tile_inputs_manifest_sha256"])
        self.assertNotEqual(derived["tile_inputs_manifest_sha256"], self.base_geometry["tile_inputs_manifest_sha256"])
        self.assertEqual(derived["tile_count"], 1)
        tile = derived["tiles"][0]
        self.assertEqual(tile["tile_id"], 1)
        self.assertEqual(tile["geometry"]["sha256"], self.base_geometry["tiles"][1]["geometry"]["sha256"])
        self.assertEqual(tile["initialization_ply_sha256"], derived_inputs["tiles"][0]["initialization"]["sha256"])
        self.assertEqual(derived["diagnostic"]["derived_from_tile_geometry_manifest_sha256"], self.base_geometry["tile_geometry_manifest_sha256"])
        # the original v9 geometry manifest would NOT bind to the derived inputs
        self.assertNotEqual(self.base_geometry["tile_inputs_manifest_sha256"], derived_inputs["tile_inputs_manifest_sha256"])
        with self.assertRaises(ValueError):
            derive_tile_geometry_manifest(self.base_geometry, tile_id=7, geometry_path="x.npz", tile_inputs_manifest_sha256="i" * 64, provenance={})


# ----------------------------------------------------------------------------
# ROI box in the crop
# ----------------------------------------------------------------------------


class RoiBoxTests(unittest.TestCase):
    def setUp(self):
        self.face = FaceSpec(face_id="front", R_face=np.eye(3), K_face=np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1]]),
                             width=100, height=100, half_fov_deg=26.0)
        self.c2w = np.eye(4)

    def test_box_is_in_crop_coordinates(self):
        # points at z=1 with x in [0.1, 0.2], y in [-0.1, 0.0] -> face pixels u in [60, 70], v in [40, 50]
        xs = np.linspace(0.1, 0.2, 11)
        ys = np.linspace(-0.1, 0.0, 11)
        pts = np.array([[x, y, 1.0] for x in xs for y in ys])
        crop = {"x": 20, "y": 10, "width": 60, "height": 60}
        box = roi_bbox_in_crop(pts, self.c2w, self.face, crop, percentiles=(0.0, 100.0))
        self.assertIsNotNone(box)
        # pixel index = rint(u - 0.5) with half-to-even: u=60 -> 60, u=70 -> 70; box is [min, max+1) minus crop offset
        self.assertEqual((box["x0"], box["x1"]), (60 - 20, 70 - 20 + 1))
        self.assertEqual((box["y0"], box["y1"]), (40 - 10, 50 - 10 + 1))
        self.assertEqual(box["samples_in_crop"], 121)
        self.assertAlmostEqual(box["fraction_of_samples_in_crop"], 1.0)
        self.assertEqual(box["crop"], crop)

    def test_none_when_region_outside_crop_or_behind(self):
        pts = np.array([[0.15, -0.05, 1.0]])
        self.assertIsNone(roi_bbox_in_crop(pts, self.c2w, self.face, {"x": 0, "y": 0, "width": 30, "height": 30}))
        self.assertIsNone(roi_bbox_in_crop(np.array([[0.0, 0.0, -1.0]]), self.c2w, self.face, {"x": 0, "y": 0, "width": 100, "height": 100}))


# ----------------------------------------------------------------------------
# arm configs
# ----------------------------------------------------------------------------


class DiagnosticArmTests(unittest.TestCase):
    def _arm(self, label="R1", count=5, **kwargs):
        return build_diagnostic_arm(
            _base_arm(), selection=_selection(count=count),
            diag_tile_inputs_manifest="C:\\runs\\diag_v2\\indoor\\U1_5\\tile_inputs_manifest.json",
            diag_tile_geometry_manifest="C:\\runs\\diag_v2\\indoor\\U1_5\\tile_geometry_manifest.json",
            horizon=3000, label=label, output_dir="C:\\runs\\diag_v2\\indoor\\runs\\x",
            reference_scale_m=0.0061790370382368565, base_config_path="run_configs/x.json", base_config_sha256="b" * 64,
            checkpoint_every=3000, **kwargs,
        )

    def test_h3000_schedule_resolves_per_contract(self):
        arm = self._arm()
        self.assertEqual(arm["schedule_contract"], RESEARCH_SCHEDULE_CONTRACT_V1)
        record = validate_research_schedule_contract(arm, training_view_count=5)
        fields = record["resolved_fields"]
        self.assertEqual(fields["horizon_steps"], 3000)
        self.assertEqual(fields["refine_start_iter"], 500)
        self.assertEqual(fields["refine_stop_iter"], 2100)  # 14000/20000 = 0.7 of H, snapped to 100
        self.assertEqual(fields["refine_scale2d_stop_iter"], 2100)
        self.assertEqual(fields["prune_switch_step"], 1500)
        self.assertEqual(fields["reset_every"], 300)
        self.assertEqual(fields["sh_degree_interval"], 0)
        self.assertAlmostEqual(fields["means_lr_base"], 0.0032 * 0.0061790370382368565)
        self.assertNotIn("controlled_stop_after_steps", arm)
        self.assertEqual(arm["max_steps"], 3000)
        self.assertEqual(arm["mcmc_refine_stop_iter"], 2100)
        self.assertIsNone(arm["metric_scale_calibration"]["means_step_fraction"])
        self.assertEqual(record["event_summary"]["reset"]["count"], 5)  # 600, 900, 1200, 1500, 1800
        self.assertEqual(record["event_summary"]["grow"]["count"], 16)  # 500..2000
        self.assertTrue(record["late_threshold"]["ever_applied"])
        self.assertEqual(record["late_threshold"]["first_step"], 1500)
        self.assertEqual(record["violations"], [])
        schedule = resolved_schedule(arm, 5)
        self.assertEqual(schedule["mismatches"], [])
        self.assertEqual(schedule["views"]["configured_epochs"], 600.0)
        self.assertEqual(arm["checkpoint_every"], 3000)

    def test_naming_rebinding_and_diag_block(self):
        arm = self._arm()
        self.assertEqual(arm["run_id"], "diag_indoor_door_leaf_Tile_1_5_R1")
        self.assertEqual(diag_run_id("r", 40, "G1"), "diag_r_40_G1")
        self.assertEqual(arm["output_dir"], "C:\\runs\\diag_v2\\indoor\\runs\\x")
        self.assertEqual(arm["tile_inputs_manifest"], "C:\\runs\\diag_v2\\indoor\\U1_5\\tile_inputs_manifest.json")
        self.assertEqual(arm["initialization_geometry_manifest"], "C:\\runs\\diag_v2\\indoor\\U1_5\\tile_geometry_manifest.json")
        base = _base_arm()
        for key in ("tile_inputs_root", "initialization_ply", "initialization_geometry", "background_image_manifest", "background_image_root", "mipmap_tile_id"):
            self.assertEqual(arm[key], base[key], key)
        diag = arm["diag"]
        self.assertEqual(diag["region"]["label"], "indoor_door_leaf_Tile_1")
        self.assertEqual(diag["base_config_sha256"], "b" * 64)
        self.assertEqual(diag["view_count"], 5)
        self.assertEqual(diag["parent_image_ids"], ["img_a", "img_b", "img_c", "img_d", "img_e"])
        self.assertEqual(len(diag["view_sample_ids"]), 5)
        self.assertEqual(diag["variant"], "R1")
        self.assertEqual(diag["variant_fields"].keys(), {"checkpoint_every"})
        self.assertEqual(arm["schedule_contract_fields"]["arm"], "R1")
        self.assertEqual(arm["schedule_contract_fields"]["provenance"]["base_config_sha256"], "b" * 64)
        self.assertEqual(arm["default_strategy"]["lifecycle_execution_order"], "pre_optimizer_vendor")
        self.assertEqual(arm["densification_gradient_source"], "total_loss")

    def test_g0_g1_variants(self):
        g0 = self._arm("G0", count=40)
        g1 = self._arm("G1", count=40)
        self.assertEqual(g0["default_strategy"]["lifecycle_execution_order"], "post_optimizer_gsplat")
        self.assertEqual(g0["densification_gradient_source"], "total_loss")
        self.assertEqual(g1["default_strategy"]["lifecycle_execution_order"], "post_optimizer_gsplat")
        self.assertEqual(g1["densification_gradient_source"], "rgb_only")
        self.assertEqual(g1["run_id"], "diag_indoor_door_leaf_Tile_1_40_G1")
        # identical schedules: the pair differs only in the growth signal
        for key in ("max_steps", "mcmc_refine_stop_iter", "learning_rates"):
            self.assertEqual(g0[key], g1[key])
        self.assertEqual({k: v for k, v in g0["default_strategy"].items()}, g1["default_strategy"])
        self.assertEqual(sorted(VARIANTS), ["G0", "G1", "R1"])
        self.assertEqual(DEFAULT_PLAN["DIAG"], ("R1", "G0", "G1"))

    def test_rejects_wrong_tile_or_variant(self):
        with self.assertRaises(ValueError):
            build_diagnostic_arm(_base_arm(), selection=_selection(tile_id=0), diag_tile_inputs_manifest="a", diag_tile_geometry_manifest="b",
                                 horizon=3000, label="R1", output_dir="o", reference_scale_m=None, base_config_path="p", base_config_sha256="s")
        with self.assertRaises(ValueError):
            self._arm("Z9")

    def test_eval_config_carries_roi_map_and_no_diag_block(self):
        arm = self._arm(count=40)
        selection = _selection(count=40)
        ev = build_eval_config(arm, selection=selection, output_dir="C:\\runs\\eval")
        self.assertEqual(ev["run_id"], "diag_indoor_door_leaf_Tile_1_eval")
        self.assertNotIn("diag", ev)
        self.assertEqual(ev["tile_inputs_manifest"], arm["tile_inputs_manifest"])
        self.assertEqual(ev["mipmap_tile_id"], 1)
        roi = ev["diag_eval"]["roi_in_crops"]
        self.assertEqual(set(roi), set(selection["view_sample_ids"]))
        self.assertIn("panel i starts at x = i * (crop.width + 8)", ev["diag_eval"]["strip_layout"]["three_way_compare"])
        self.assertIn("diag_indoor_door_leaf_Tile_1_40_G1", ev["diag_eval"]["evaluates_arms"])


class GeneratedConfigsOnDiskTests(unittest.TestCase):
    """Every generated diag arm in the repo must still satisfy the contract."""

    def test_repo_diag_configs_satisfy_contract(self):
        if not DIAG_CONFIG_DIR.is_dir():
            self.skipTest("no generated diag configs")
        arms = [p for p in sorted(DIAG_CONFIG_DIR.glob("diag_*.json"))]
        if not arms:
            self.skipTest("no generated diag configs")
        for path in arms:
            config = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(path=path.name):
                self.assertEqual(config["schedule_contract"], RESEARCH_SCHEDULE_CONTRACT_V1)
                record = research_schedule_contract(config, training_view_count=config["diag"]["view_count"])
                self.assertEqual(record["violations"], [])
                self.assertEqual(path.stem, config["run_id"])
                self.assertEqual(config["run_id"], diag_run_id(config["diag"]["region"]["label"], config["diag"]["count"], config["diag"]["variant"]))
                self.assertEqual(config["schedule_contract_fields"]["resolved"], record["resolved_fields"])
                self.assertTrue(Path(config["output_dir"]).name == config["run_id"])


if __name__ == "__main__":
    unittest.main()
