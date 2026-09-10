"""Pipeline config loading and validation (tools/pipeline.py).

Every machine path the cmd scripts hard-coded must come from the JSON config,
and a typo in it must fail before any GPU time is spent.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.pipeline import (
    DEFAULTS,
    EXAMPLE_CONFIG,
    REQUIRED_PATH_KEYS,
    PipelineConfigError,
    load_pipeline_config,
    parse_pipeline_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def minimal_raw(base: Path) -> dict:
    raw = {key: str(base / key) for key in REQUIRED_PATH_KEYS}
    raw["schema_version"] = 1
    return raw


class PipelineConfigParseTests(unittest.TestCase):
    def test_minimal_config_gets_documented_defaults(self) -> None:
        base = Path("C:/host") if Path("C:/").exists() else Path("/host")
        config = parse_pipeline_config(minimal_raw(base))
        self.assertEqual(config.run_root, base / "run_root")
        self.assertEqual(config.repo_root, REPO_ROOT)
        self.assertIsNone(config.env_script)
        self.assertEqual(config.compare_frames, DEFAULTS["compare_frames"])
        self.assertEqual(config.battery_views, DEFAULTS["battery_views"])
        self.assertEqual(config.delivery_tiles, [1, 2, 3])
        self.assertEqual(config.identity_dir, REPO_ROOT / "research/quality_recovery_v1/identity")
        self.assertEqual(config.delivery_tile_arm("r1d", 2), "tile2_r1d_20k")
        self.assertEqual(config.delivery_ply_name("r1d"), "house0305_r1d_merged.ply")
        self.assertEqual(config.delivery_sky_name("r1d"), "house0305_r1d_sky.ply")
        self.assertEqual(config.env, {"PYTHONIOENCODING": "utf-8"})

    def test_missing_required_key_names_it(self) -> None:
        raw = minimal_raw(Path("/host"))
        del raw["reference_ply"]
        with self.assertRaises(PipelineConfigError) as caught:
            parse_pipeline_config(raw)
        self.assertIn("reference_ply", str(caught.exception))

    def test_unknown_key_is_rejected(self) -> None:
        raw = minimal_raw(Path("/host"))
        raw["referense_ply"] = "/typo"
        with self.assertRaises(PipelineConfigError) as caught:
            parse_pipeline_config(raw)
        self.assertIn("referense_ply", str(caught.exception))

    def test_underscore_keys_are_documentation(self) -> None:
        raw = minimal_raw(Path("/host"))
        raw["_doc"] = {"anything": "goes"}
        parse_pipeline_config(raw)

    def test_wrong_schema_version_is_rejected(self) -> None:
        raw = minimal_raw(Path("/host"))
        raw["schema_version"] = 2
        with self.assertRaises(PipelineConfigError):
            parse_pipeline_config(raw)

    def test_type_validation(self) -> None:
        cases = {
            "compare_frames": 0,
            "battery_views": "48",
            "export_min_opacity": "0.05",
            "harmonize_exposure": "yes",
            "delivery_tiles": [],
            "delivery_tile_arm_pattern": "tile{tile}_20k",
            "env": {"A": 1},
            "delivery_baselines": [],
            "python": "",
        }
        for key, value in cases.items():
            raw = minimal_raw(Path("/host"))
            raw[key] = value
            with self.subTest(key=key), self.assertRaises(PipelineConfigError) as caught:
                parse_pipeline_config(raw)
            self.assertIn(key, str(caught.exception))

    def test_relative_paths_resolve_against_config_dir_and_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            raw = minimal_raw(base)
            raw["run_root"] = "runs"
            raw["repo_root"] = "checkout"
            raw["identity_dir"] = "identity"
            raw["delivery_baselines"] = {"compare": ["delivery_f6/compare_matched"], "offtraj": {"F6": "delivery_f6/offtraj"}}
            source = base / "pipeline.json"
            source.write_text(json.dumps(raw), encoding="utf-8")
            config = load_pipeline_config(source)
            self.assertEqual(config.source, source)
            self.assertEqual(config.run_root, base / "runs")
            self.assertEqual(config.repo_root, base / "checkout")
            self.assertEqual(config.identity_dir, base / "checkout" / "identity")
            self.assertEqual(config.delivery_baselines["compare"], [base / "runs" / "delivery_f6" / "compare_matched"])
            self.assertEqual(config.delivery_baselines["offtraj"], {"F6": base / "runs" / "delivery_f6" / "offtraj"})
            self.assertEqual(config.arm_config("tile0_x"), base / "runs" / "tile0_x.json")
            self.assertEqual(config.arm_checkpoint("tile0_x"), base / "runs" / "tile0_x" / "checkpoints" / "latest.pt")
            self.assertEqual(config.delivery_dir("r1d"), base / "runs" / "delivery_r1d")

    def test_missing_paths_lists_inputs_that_do_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            raw = minimal_raw(base)
            for key in REQUIRED_PATH_KEYS:
                (base / key).write_text("x", encoding="utf-8")
            config = parse_pipeline_config(raw, source=base / "pipeline.json")
            self.assertEqual(config.missing_paths(), [])
            (base / "sky_ply").unlink()
            self.assertEqual([key for key, _ in config.missing_paths()], ["sky_ply"])


class PipelineConfigFileTests(unittest.TestCase):
    def test_missing_file_points_at_example(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(PipelineConfigError) as caught:
                load_pipeline_config(Path(temporary) / "pipeline.json")
        self.assertIn(EXAMPLE_CONFIG, str(caught.exception))

    def test_invalid_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pipeline.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(PipelineConfigError) as caught:
                load_pipeline_config(path)
        self.assertIn("not valid JSON", str(caught.exception))

    def test_committed_example_loads_with_machine_b_paths(self) -> None:
        config = load_pipeline_config(REPO_ROOT / EXAMPLE_CONFIG)
        self.assertEqual(config.run_root, Path("C:/Peter/3dgs-runs/house0305_sop"))
        self.assertEqual(config.repo_root, Path("C:/Peter/cloudstudio-3dgs-work"))
        self.assertEqual(config.python, Path("C:/Peter/cloudstudio-3dgs/.venv-train/Scripts/python.exe"))
        self.assertEqual(config.env_script, Path("C:/Peter/cloudstudio-3dgs-gate1/train/env_machine_b.cmd"))
        self.assertEqual(config.reference_ply, Path("C:/baidunetdiskdownload/house/USAgs.ply"))
        self.assertEqual(config.reference_alignment, Path("C:/Peter/3dgs-runs/probes/usa_gs_alignment.json"))
        self.assertEqual(config.sky_ply, Path("C:/Peter/3dgs-runs/exports/house0305_f5_sky_20260903.ply"))
        self.assertEqual(config.identity_dir, Path("C:/Peter/cloudstudio-3dgs-work/research/quality_recovery_v1/identity"))
        self.assertEqual(config.delivery_baselines["offtraj"]["G9sh1"], Path("C:/Peter/3dgs-runs/house0305_sop/delivery_g9/offtraj_matched_sh1"))
        self.assertEqual(config.compare_frames, 6)
        self.assertEqual(config.battery_views, 48)
        self.assertEqual(config.export_min_opacity, 0.05)


if __name__ == "__main__":
    unittest.main()
