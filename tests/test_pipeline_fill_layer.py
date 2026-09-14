"""Delivery fill layer reaches the merge command (tools/pipeline.py), CPU only.

Ownership masking leaves the merged delivery transparent wherever no tile was responsible
for a pixel, because the stand-in backdrop that covered those pixels during training does
not ship. Measured on the four-tile B5 delivery: alpha mean 0.712, alpha p05 0.189. The
fill layer puts coarse-prior gaussians back into voxels no delivery gaussian occupies and
recovers it (alpha 0.947, p05 0.733, PSNR 18.05 -> 19.11).

What is pinned:

* with no ``fill_checkpoint`` the merge argv is exactly what it was before this option
  existed - no fill flag appears, so existing deliveries are unchanged;
* with one, the checkpoint and both voxel-rule values are forwarded, and
  ``fill_min_opacity`` only appears when it is set;
* a ``fill_checkpoint`` naming a file that is not there fails config parsing rather than
  letting the merge run and quietly ship the coverage gap back;
* the voxel rule's numbers are validated, because a zero or negative voxel size would make
  the occupancy test meaningless.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from pipeline import PipelineConfigError, parse_pipeline_config  # noqa: E402

from test_pipeline_resume import DELIVERY_STEPS_AFTER_TILES, PipelineFixture  # noqa: E402,F401
from pipeline import deliver_steps, run_steps  # noqa: E402


class FillLayerConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self._tmp.name).resolve()
        (self.base / "runs").mkdir()
        (self.base / "repo" / "tools").mkdir(parents=True)
        (self.base / "exports").mkdir()
        (self.base / "sky.ply").write_bytes(b"sky")
        self.prior = self.base / "prior.pt"
        self.prior.write_bytes(b"prior")
        self.raw = {
            "run_root": str(self.base / "runs"),
            "repo_root": str(self.base / "repo"),
            "python": str(self.base / "python.exe"),
            "reference_ply": str(self.base / "ref.ply"),
            "reference_alignment": str(self.base / "align.json"),
            "tile_inputs_manifest": str(self.base / "tiles" / "manifest.json"),
            "tile_inputs_root": str(self.base / "tiles"),
            "exports_dir": str(self.base / "exports"),
            "delivery_eval_config": str(self.base / "runs" / "delivery_eval.json"),
            "sky_ply": str(self.base / "sky.ply"),
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_default_is_no_fill_layer(self) -> None:
        config = parse_pipeline_config(self.raw)
        self.assertIsNone(config.fill_checkpoint)
        self.assertIsNone(config.fill_min_opacity)
        self.assertEqual(config.fill_occupancy_voxel_m, 0.2)
        self.assertEqual(config.fill_occupancy_clearance_voxels, 0)

    def test_a_fill_checkpoint_is_accepted_and_resolved(self) -> None:
        config = parse_pipeline_config({**self.raw, "fill_checkpoint": str(self.prior)})
        self.assertEqual(config.fill_checkpoint, self.prior)

    def test_a_missing_fill_checkpoint_fails_the_config(self) -> None:
        with self.assertRaises(PipelineConfigError) as caught:
            parse_pipeline_config({**self.raw, "fill_checkpoint": str(self.base / "absent.pt")})
        self.assertIn("fill_checkpoint", str(caught.exception))

    def test_the_voxel_rule_numbers_are_validated(self) -> None:
        for key, value in (
            ("fill_occupancy_voxel_m", 0),
            ("fill_occupancy_voxel_m", -0.1),
            ("fill_occupancy_voxel_m", "0.2"),
            ("fill_occupancy_clearance_voxels", -1),
            ("fill_occupancy_clearance_voxels", 1.5),
            ("fill_min_opacity", "0.05"),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(PipelineConfigError):
                    parse_pipeline_config({**self.raw, key: value})


class FillLayerMergeArgvTests(PipelineFixture):
    TAG = "b5fill2"

    def setUp(self) -> None:
        super().setUp()
        self.prior = self.base / "prior.pt"
        self.prior.write_bytes(b"prior")
        self.write_arm_config("tile0_B5")
        self.plant_checkpoint("tile0_B5")
        for tile in (1, 2, 3):
            arm = self.config.delivery_tile_arm(self.TAG, tile)
            self.write_arm_config(arm)
            self.plant_checkpoint(arm)
        (self.run_root / "delivery_eval.json").write_text("{}", encoding="utf-8")

    def _merge_argv(self, **config_overrides):
        if config_overrides:
            self.config = parse_pipeline_config({**self.raw_config, **config_overrides})
            self.ctx = self.make_ctx()
        run_steps(deliver_steps(self.ctx, self.TAG, "tile0_B5"), status=lambda _: None)
        return dict(self.runner.calls)["merge_v28_tile_checkpoints.py"]

    def test_without_a_fill_checkpoint_no_fill_flag_appears(self) -> None:
        merge = self._merge_argv()
        self.assertNotIn("--fill-checkpoint", merge)
        self.assertNotIn("--fill-occupancy-voxel-m", merge)
        self.assertNotIn("--fill-min-opacity", merge)

    def test_the_fill_checkpoint_and_voxel_rule_are_forwarded(self) -> None:
        merge = self._merge_argv(
            fill_checkpoint=str(self.prior),
            fill_occupancy_voxel_m=0.35,
            fill_occupancy_clearance_voxels=1,
        )
        self.assertEqual(merge[merge.index("--fill-checkpoint") + 1], str(self.prior))
        self.assertEqual(merge[merge.index("--fill-occupancy-voxel-m") + 1], "0.35")
        self.assertEqual(merge[merge.index("--fill-occupancy-clearance-voxels") + 1], "1")
        self.assertNotIn("--fill-min-opacity", merge)

    def test_fill_min_opacity_is_forwarded_only_when_set(self) -> None:
        merge = self._merge_argv(fill_checkpoint=str(self.prior), fill_min_opacity=0.05)
        self.assertEqual(merge[merge.index("--fill-min-opacity") + 1], "0.05")


if __name__ == "__main__":
    unittest.main()
