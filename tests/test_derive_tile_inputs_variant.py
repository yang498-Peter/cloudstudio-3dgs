"""Face-excluded production Tile variants (tools/derive_tile_inputs_variant.py):
every parent image kept, the given faces dropped, derived manifests signed and
bound, the config a pure view-set change with the schedule pre-flight
recorded.  Torch-free, synthetic manifests only."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from derive_tile_inputs_variant import (  # noqa: E402
    CONFIG_ALLOWED_DIFF,
    VARIANT_KIND,
    derive_variant_config,
    derive_variant_manifests,
    main,
    plan_face_exclusion,
    roi_presence,
    schedule_preflight,
    top_level_diff,
)
from build_diagnostic_set import sign_tile_inputs_manifest  # noqa: E402
from test_diagnostic_set import _base_arm, _base_tile_geometry, _base_tile_inputs  # noqa: E402
from cloudstudio_3dgs.data.manifest import canonical_json_bytes  # noqa: E402
from cloudstudio_3dgs.training.mipmap_tile_geometry import (  # noqa: E402
    sign_tile_geometry_manifest,
    verify_tile_geometry_manifest,
)
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest  # noqa: E402

GEOMETRY_REL = "../tile_geometry_v9/Tile_1/initialization_geometry_k7_k30.npz"


def _with_pitch_up_only_parent(base_inputs: dict) -> dict:
    """Tile_1 gains ``img_d`` that only has a pitch_up_56 face (re-signed)."""
    payload = copy.deepcopy(base_inputs)
    tile = payload["tiles"][1]
    tile["views"].append({"image_index": 99, "x": 990, "y": 0, "width": 200, "height": 300,
                          "pixel_load": 60000, "sample_id": "img_d::pitch_up_56"})
    tile["view_count"] = len(tile["views"])
    return sign_tile_inputs_manifest(payload)


class VariantManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inputs_root = self.root / "tile_inputs_v9"
        self.base_inputs = _base_tile_inputs(self.inputs_root)
        self.geometry_root = self.root / "tile_geometry_v9"
        self.base_geometry = _base_tile_geometry(self.geometry_root, self.base_inputs)
        self.out_dir = self.root / "tile_inputs_v9_F3"
        self.out_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _derive(self, base_inputs=None, base_geometry=None, faces=("pitch_up_56",), **kwargs):
        return derive_variant_manifests(
            base_inputs or self.base_inputs, base_geometry or self.base_geometry,
            tile_name="Tile_1", exclude_face_ids=list(faces), geometry_path=GEOMETRY_REL, label="F3_nopitchup", **kwargs,
        )

    def test_drops_the_faces_of_every_parent_and_signs_bound_manifests(self):
        before_inputs = canonical_json_bytes(self.base_inputs)
        before_geometry = canonical_json_bytes(self.base_geometry)
        inputs, geometry, summary = self._derive(note="F3 at production scale")
        # inputs: every parent kept, only the excluded face gone, order kept
        self.assertEqual(verify_tile_inputs_manifest(inputs, root=self.inputs_root, verify_artifacts=True), inputs["tile_inputs_manifest_sha256"])
        self.assertEqual(inputs["tile_count"], 1)
        tile = inputs["tiles"][0]
        self.assertEqual(tile["name"], "Tile_1")
        base_ids = [v["sample_id"] for v in self.base_inputs["tiles"][1]["views"]]
        ids = [v["sample_id"] for v in tile["views"]]
        self.assertEqual(ids, [s for s in base_ids if not s.endswith("::pitch_up_56")])
        self.assertEqual(len(ids), 6)
        self.assertEqual(tile["view_count"], 6)
        self.assertEqual(sorted({s.split("::")[0] for s in ids}), ["img_a", "img_b", "img_c"])
        self.assertEqual(tile["initialization"], self.base_inputs["tiles"][1]["initialization"])
        self.assertEqual(tile["core_box"], self.base_inputs["tiles"][1]["core_box"])
        self.assertEqual(inputs["tile_plan_manifest_sha256"], self.base_inputs["tile_plan_manifest_sha256"])
        diagnostic = inputs["diagnostic"]
        self.assertEqual(diagnostic["kind"], VARIANT_KIND)
        self.assertEqual(diagnostic["variant"], "F3_nopitchup")
        self.assertEqual(diagnostic["excluded_face_ids"], ["pitch_up_56"])
        self.assertEqual(diagnostic["parent_image_ids"], ["img_a", "img_b", "img_c"])
        self.assertEqual(diagnostic["parent_images_dropped_by_exclusion"], [])
        self.assertEqual(diagnostic["view_count_before_exclusion"], 9)
        self.assertEqual(diagnostic["views_removed_by_exclusion"], 3)
        self.assertEqual(diagnostic["face_counts"], {"yaw_minus_35": 3, "yaw_plus_35": 3})
        self.assertEqual(diagnostic["note"], "F3 at production scale")
        self.assertEqual(diagnostic["derived_from_tile_inputs_manifest_sha256"], self.base_inputs["tile_inputs_manifest_sha256"])
        # geometry: bound to the derived inputs, npz verbatim through the relative path
        self.assertEqual(verify_tile_geometry_manifest(geometry, root=self.out_dir, verify_artifacts=True), geometry["tile_geometry_manifest_sha256"])
        self.assertEqual(geometry["tile_inputs_manifest_sha256"], inputs["tile_inputs_manifest_sha256"])
        self.assertNotEqual(geometry["tile_inputs_manifest_sha256"], self.base_geometry["tile_inputs_manifest_sha256"])
        self.assertEqual(geometry["tiles"][0]["geometry"]["sha256"], self.base_geometry["tiles"][1]["geometry"]["sha256"])
        self.assertEqual(geometry["tiles"][0]["geometry"]["path"], GEOMETRY_REL)
        self.assertEqual(geometry["diagnostic"]["variant"], "F3_nopitchup")
        # summary
        self.assertEqual(summary["view_count_before"], 9)
        self.assertEqual(summary["view_count_after"], 6)
        self.assertEqual(summary["views_removed"], 3)
        self.assertEqual(summary["face_counts_before"], {"yaw_minus_35": 3, "yaw_plus_35": 3, "pitch_up_56": 3})
        self.assertEqual(summary["face_counts_after"], {"yaw_minus_35": 3, "yaw_plus_35": 3})
        self.assertEqual((summary["parent_image_count_before"], summary["parent_image_count_after"]), (3, 3))
        self.assertEqual(summary["tile_inputs_manifest_sha256"], inputs["tile_inputs_manifest_sha256"])
        self.assertEqual(summary["tile_geometry_manifest_sha256"], geometry["tile_geometry_manifest_sha256"])
        self.assertEqual(summary["initialization_point_count"], 123)
        # inputs untouched
        self.assertEqual(canonical_json_bytes(self.base_inputs), before_inputs)
        self.assertEqual(canonical_json_bytes(self.base_geometry), before_geometry)

    def test_parent_left_with_zero_faces_is_dropped_and_reported(self):
        base = _with_pitch_up_only_parent(self.base_inputs)
        geometry = _base_tile_geometry(self.geometry_root, base)
        plan = plan_face_exclusion(base["tiles"][1], ["pitch_up_56"])
        self.assertEqual(plan["parent_image_ids"], ["img_a", "img_b", "img_c"])
        self.assertEqual(plan["parent_images_dropped"], ["img_d"])
        self.assertEqual(plan["tile_view_count_before"], 10)
        inputs, _geometry, summary = self._derive(base_inputs=base, base_geometry=geometry)
        self.assertEqual(verify_tile_inputs_manifest(inputs, root=self.inputs_root, verify_artifacts=True), inputs["tile_inputs_manifest_sha256"])
        ids = [v["sample_id"] for v in inputs["tiles"][0]["views"]]
        self.assertEqual(len(ids), 6)
        self.assertNotIn("img_d", {s.split("::")[0] for s in ids})
        self.assertEqual(inputs["diagnostic"]["parent_images_dropped_by_exclusion"], ["img_d"])
        self.assertEqual(inputs["diagnostic"]["tile_view_count_before_exclusion"], 10)
        self.assertEqual(summary["parent_images_dropped"], ["img_d"])
        self.assertEqual((summary["parent_image_count_before"], summary["parent_image_count_after"]), (4, 3))
        self.assertEqual((summary["view_count_before"], summary["view_count_after"], summary["views_removed"]), (10, 6, 4))
        self.assertEqual(summary["face_counts_before"]["pitch_up_56"], 4)

    def test_refuses_empty_exclusion_unbound_geometry_and_total_exclusion(self):
        with self.assertRaises(ValueError):
            self._derive(faces=())
        with self.assertRaises(ValueError):
            self._derive(faces=("yaw_minus_35", "yaw_plus_35", "pitch_up_56"))
        # geometry bound to other inputs (a stale production pair) is refused
        stale = copy.deepcopy(self.base_geometry)
        stale.pop("tile_geometry_manifest_sha256")
        stale["tile_inputs_manifest_sha256"] = "0" * 64
        stale = sign_tile_geometry_manifest(stale)
        with self.assertRaises(ValueError):
            self._derive(base_geometry=stale)
        with self.assertRaises(ValueError):
            derive_variant_manifests(self.base_inputs, self.base_geometry, tile_name="Tile_9", exclude_face_ids=["pitch_up_56"], geometry_path="x", label="v")
        # a face nobody has: nothing removed, still recorded
        inputs, _g, summary = self._derive(faces=("pitch_down_56",))
        self.assertEqual(summary["views_removed"], 0)
        self.assertEqual(inputs["diagnostic"]["excluded_face_ids"], ["pitch_down_56"])

    def test_roi_presence(self):
        inputs, _g, _s = self._derive()
        report = roi_presence(inputs["tiles"][0]["views"], ["img_a::yaw_minus_35", "img_b::pitch_up_56", "img_zz::yaw_plus_35"])
        self.assertEqual(report["present"], 1)
        self.assertEqual(report["missing"], ["img_b::pitch_up_56", "img_zz::yaw_plus_35"])
        self.assertFalse(report["all_present"])
        self.assertTrue(roi_presence(inputs["tiles"][0]["views"], ["img_c::yaw_plus_35"])["all_present"])


class VariantConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base_inputs = _base_tile_inputs(self.root / "tile_inputs_v9")
        self.base_geometry = _base_tile_geometry(self.root / "tile_geometry_v9", self.base_inputs)
        _inputs, _geometry, self.summary = derive_variant_manifests(
            self.base_inputs, self.base_geometry, tile_name="Tile_1", exclude_face_ids=["pitch_up_56"],
            geometry_path=GEOMETRY_REL, label="F3_nopitchup",
        )
        self.base = _base_arm()
        self.base["cap_max"] = 15_000_000
        self.base["topology_policy"] = {"mode": "adaptive_growth"}

    def tearDown(self):
        self.tmp.cleanup()

    def _config(self, base=None, **kwargs):
        return derive_variant_config(
            base or self.base, run_id="house0305-t1-F3-nopitchup", output_dir="C:\\runs\\tile1_F3_nopitchup_20k",
            tile_inputs_manifest="C:\\runs\\tile_inputs_v9_F3\\tile_inputs_manifest.json",
            tile_geometry_manifest="C:\\runs\\tile_inputs_v9_F3\\tile_geometry_manifest.json",
            summary=self.summary, base_config_path="run_configs/house0305_tiles/v9/tile1_R1d_20k.json",
            base_config_sha256="b" * 64, **kwargs,
        )

    def test_only_the_allowed_fields_change_and_the_schedule_is_verbatim(self):
        before = json.dumps(self.base, sort_keys=True)
        config = self._config(note="F3 at 20k")
        self.assertEqual(json.dumps(self.base, sort_keys=True), before)  # base not mutated
        changed = top_level_diff(self.base, config)
        self.assertEqual(changed, {"run_id", "output_dir", "tile_inputs_manifest", "initialization_geometry_manifest", "lineage"})
        self.assertTrue(changed <= CONFIG_ALLOWED_DIFF)
        self.assertEqual(config["run_id"], "house0305-t1-F3-nopitchup")
        self.assertEqual(config["output_dir"], "C:\\runs\\tile1_F3_nopitchup_20k")
        self.assertEqual(config["tile_inputs_manifest"], "C:\\runs\\tile_inputs_v9_F3\\tile_inputs_manifest.json")
        self.assertEqual(config["initialization_geometry_manifest"], "C:\\runs\\tile_inputs_v9_F3\\tile_geometry_manifest.json")
        for key in ("tile_inputs_root", "initialization_ply", "initialization_geometry", "background_image_manifest", "mipmap_tile_id",
                    "max_steps", "controlled_stop_after_steps", "default_strategy", "cap_max", "learning_rates", "checkpoint_every"):
            self.assertEqual(config[key], self.base[key], key)
        lineage = config["lineage"]
        self.assertEqual(lineage["base"], "tile1_R1d_20k")
        self.assertEqual(lineage["base_run_id"], "house0305-t1-R1d")
        self.assertEqual(lineage["base_config_sha256"], "b" * 64)
        self.assertEqual(lineage["data_variant"]["excluded_face_ids"], ["pitch_up_56"])
        self.assertEqual(lineage["data_variant"]["tile_inputs_manifest_sha256"], self.summary["tile_inputs_manifest_sha256"])
        self.assertEqual(lineage["data_variant"]["tile_geometry_manifest_sha256"], self.summary["tile_geometry_manifest_sha256"])
        self.assertEqual(lineage["rebound_fields"]["tile_inputs_manifest"]["base"], self.base["tile_inputs_manifest"])
        self.assertIn("tile_inputs_root", lineage["verbatim_fields"])
        self.assertIn("9 -> 6 views", lineage["single_change"])
        self.assertEqual(lineage["note"], "F3 at 20k")
        self.assertNotIn("base_lineage", lineage)
        # a base that already carries a lineage keeps it nested
        chained = dict(self.base, lineage={"base": "tile1_R1d_20k", "single_change": "x"})
        self.assertEqual(self._config(base=chained)["lineage"]["base_lineage"], chained["lineage"])

    def test_preflight_records_the_parity_rejection_without_touching_the_schedule(self):
        config = self._config()
        schedule = config["lineage"]["schedule"]
        # 36580 steps over 6 views: the production parity rule fails and the
        # parity-implied values are spelled out, the config itself unchanged
        self.assertEqual(schedule["check"], "max_steps_is_20_view_epochs")
        self.assertFalse(schedule["ok"])
        self.assertEqual(schedule["max_steps"], 36580)
        self.assertEqual(schedule["training_view_count"], 6)
        self.assertEqual(schedule["parity_max_steps"], 120)
        self.assertEqual(schedule["parity_prune_switch_step"], 60)
        self.assertEqual(schedule["current_prune_switch_step"], 18290)
        self.assertEqual(schedule["controlled_stop_after_steps"], 20000)
        self.assertIn("20 complete view epochs", schedule["trainer_error"])
        self.assertEqual(len(schedule["resolution_options"]), 2)
        self.assertEqual(config["max_steps"], 36580)
        self.assertEqual(config["default_strategy"]["prune_switch_step"], 18290)
        # parity holds -> ok; a research contract replaces the rule -> ok
        parity = dict(self.base, max_steps=120, default_strategy=dict(self.base["default_strategy"], prune_switch_step=60), controlled_stop_after_steps=100)
        ok = schedule_preflight(parity, 6)
        self.assertTrue(ok["ok"])
        self.assertNotIn("trainer_error", ok)
        contract = dict(self.base, schedule_contract="research_rescaled_horizon_v1")
        self.assertEqual(schedule_preflight(contract, 6)["check"], "max_steps_is_contract_horizon")
        self.assertTrue(schedule_preflight(contract, 6)["ok"])

    def test_rejects_wrong_tile_and_unexpected_changes(self):
        wrong = dict(self.base, mipmap_tile_id=0)
        with self.assertRaises(ValueError):
            self._config(base=wrong)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inputs_root = self.root / "tile_inputs_v9"
        self.base_inputs = _base_tile_inputs(self.inputs_root)
        self.geometry_root = self.root / "tile_geometry_v9"
        self.base_geometry = _base_tile_geometry(self.geometry_root, self.base_inputs)
        (self.inputs_root / "tile_inputs_manifest.json").write_text(json.dumps(self.base_inputs, indent=2), encoding="utf-8")
        (self.geometry_root / "tile_geometry_manifest.json").write_text(json.dumps(self.base_geometry, indent=2), encoding="utf-8")
        base = _base_arm()
        base.update({
            "tile_inputs_manifest": str(self.inputs_root / "tile_inputs_manifest.json"),
            "tile_inputs_root": str(self.inputs_root),
            "initialization_ply": str(self.inputs_root / "Tile_1" / "initialization_full_lidar.ply"),
            "initialization_geometry_manifest": str(self.geometry_root / "tile_geometry_manifest.json"),
            "initialization_geometry": str(self.geometry_root / "Tile_1" / "initialization_geometry_k7_k30.npz"),
            "topology_policy": {"mode": "adaptive_growth"},
        })
        self.base_config = self.root / "tile1_R1d_20k.json"
        self.base_config.write_text(json.dumps(base, indent=1) + "\n", encoding="utf-8")
        self.roi = self.root / "roi_ids.json"
        self.roi.write_text(json.dumps(["img_a::yaw_minus_35", "img_c::pitch_up_56"]), encoding="utf-8")
        self.out_dir = self.root / "tile_inputs_v9_F3"
        self.out_config = self.root / "tile1_F3_nopitchup_20k.json"
        self.copy_dir = self.root / "run_configs"

    def tearDown(self):
        self.tmp.cleanup()

    def _argv(self, *extra):
        return [
            "--base-config", str(self.base_config), "--exclude-faces", "pitch_up_56", "--label", "F3_nopitchup",
            "--out-dir", str(self.out_dir), "--run-id", "house0305-t1-F3-nopitchup",
            "--output-dir", str(self.root / "tile1_F3_nopitchup_20k"), "--out-config", str(self.out_config),
            "--copy-config-to", str(self.copy_dir), "--roi-ids", str(self.roi), *extra,
        ]

    def test_end_to_end_writes_verified_manifests_config_copy_and_summary(self):
        self.assertEqual(main(self._argv()), 0)
        inputs = json.loads((self.out_dir / "tile_inputs_manifest.json").read_text(encoding="utf-8"))
        geometry = json.loads((self.out_dir / "tile_geometry_manifest.json").read_text(encoding="utf-8"))
        # exactly the trainer's binding checks: inputs against the production root, geometry against the new dir
        self.assertEqual(verify_tile_inputs_manifest(inputs, root=self.inputs_root, verify_artifacts=True), inputs["tile_inputs_manifest_sha256"])
        self.assertEqual(verify_tile_geometry_manifest(geometry, root=self.out_dir, verify_artifacts=True), geometry["tile_geometry_manifest_sha256"])
        self.assertEqual(geometry["tile_inputs_manifest_sha256"], inputs["tile_inputs_manifest_sha256"])
        self.assertEqual(geometry["tiles"][0]["geometry"]["path"], GEOMETRY_REL)
        self.assertEqual((self.out_dir / geometry["tiles"][0]["geometry"]["path"]).resolve(), (self.geometry_root / "Tile_1" / "initialization_geometry_k7_k30.npz").resolve())
        config = json.loads(self.out_config.read_text(encoding="utf-8"))
        base = json.loads(self.base_config.read_text(encoding="utf-8"))
        self.assertEqual(top_level_diff(base, config), {"run_id", "output_dir", "tile_inputs_manifest", "initialization_geometry_manifest", "lineage"})
        self.assertEqual(config["tile_inputs_manifest"], str(self.out_dir / "tile_inputs_manifest.json"))
        self.assertEqual(config["initialization_geometry_manifest"], str(self.out_dir / "tile_geometry_manifest.json"))
        self.assertEqual(config["tile_inputs_root"], str(self.inputs_root))
        copied = self.copy_dir / self.out_config.name
        self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), hashlib.sha256(self.out_config.read_bytes()).hexdigest())
        summary = json.loads((self.out_dir / "variant.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["view_count_before"], summary["view_count_after"]), (9, 6))
        self.assertEqual(summary["roi_ids"]["present"], 1)
        self.assertEqual(summary["roi_ids"]["missing"], ["img_c::pitch_up_56"])
        self.assertEqual(summary["config"]["sha256"], hashlib.sha256(self.out_config.read_bytes()).hexdigest())
        self.assertEqual(summary["config"]["copy"], str(copied))
        self.assertFalse(summary["schedule_preflight"]["ok"])
        self.assertEqual(summary["schedule_preflight"]["parity_max_steps"], 120)
        self.assertEqual(summary["derived_from"]["base_config_sha256"], hashlib.sha256(self.base_config.read_bytes()).hexdigest())
        # the base manifests / config are not modified
        self.assertEqual(json.loads((self.inputs_root / "tile_inputs_manifest.json").read_text(encoding="utf-8")), self.base_inputs)
        self.assertEqual(json.loads(self.base_config.read_text(encoding="utf-8")), base)
        # a second run refuses to overwrite unless told to
        with self.assertRaises(SystemExit):
            main(self._argv())
        self.assertEqual(main(self._argv("--overwrite")), 0)


if __name__ == "__main__":
    unittest.main()
