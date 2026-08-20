from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import sparse_depth_npz_bytes
from cloudstudio_3dgs.evaluation.quality_report import (
    _safe_path,
    build_quality_report,
    sign_run_manifest,
    verify_quality_report,
    verify_run_manifest,
)
from cloudstudio_3dgs.evaluation.splits import SplitConfig, build_split_manifest
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap
from tests.test_splits import dataset_fixture


class QualityReportTests(unittest.TestCase):
    def test_artifact_paths_cannot_escape_run_root_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "forward slashes"):
                _safe_path(root, "..\\secret.png")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                _safe_path(root, "../secret.png")

    def test_report_is_masked_deterministic_and_always_writes_html(self) -> None:
        dataset = dataset_fixture(3)
        assignments = {"rig_000": "train", "rig_001": "val", "rig_002": "train"}
        split = build_split_manifest(
            dataset,
            SplitConfig(mode="manual", golden_rig_frames=1),
            manual=assignments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.ply"
            model.write_bytes(b"synthetic-model")
            frames = []
            for side in ("left", "right"):
                image_id = f"img_{side}_001"
                reference = np.full((32, 32, 3), 128, dtype=np.uint8)
                rendered = reference.copy()
                rendered[:8] = 0
                rendered[-8:] = 255
                rendered[:, :8] = 0
                rendered[:, -8:] = 255
                rendered[16, 16] = 100
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[8:24, 8:24] = 255
                reference_path = root / f"{image_id}_reference.png"
                rendered_path = root / f"{image_id}_rendered.png"
                mask_path = root / f"{image_id}_mask.png"
                Image.fromarray(reference).save(reference_path)
                Image.fromarray(rendered).save(rendered_path)
                Image.fromarray(mask).save(mask_path)

                pixel_index = np.array([16 * 32 + 16, 17 * 32 + 17], dtype=np.int32)
                lidar = SparseDepthMap(
                    (32, 32),
                    pixel_index,
                    np.array([2.0, 3.0], dtype=np.float32),
                    np.array([1.0, 0.5], dtype=np.float32),
                    np.array([0, 1], dtype=np.int64),
                    np.array([1, 1], dtype=np.int32),
                )
                cache_path = root / f"{image_id}_lidar.npz"
                cache_path.write_bytes(sparse_depth_npz_bytes(lidar))
                rendered_depth = np.ones((32, 32), dtype=np.float32)
                rendered_depth[16, 16] = 2.1
                rendered_depth[17, 17] = 2.8
                rendered_depth_path = root / f"{image_id}_depth.npy"
                np.save(rendered_depth_path, rendered_depth)
                frames.append(
                    {
                        "image_id": image_id,
                        "split": "val",
                        "reference_rgb_path": reference_path.name,
                        "rendered_rgb_path": rendered_path.name,
                        "combined_mask_path": mask_path.name,
                        "rendered_depth_path": rendered_depth_path.name,
                        "rendered_depth_semantics": "euclidean_ray_range_m",
                        "lidar_depth_cache_path": cache_path.name,
                    }
                )
            run = sign_run_manifest(
                {
                    "schema_version": 1,
                    "run_id": "synthetic-quality-run",
                    "dataset_manifest_sha256": dataset["manifest_sha256"],
                    "split_manifest_sha256": split["split_manifest_sha256"],
                    "frames": frames,
                    "training": {
                        "duration_seconds": 12.5,
                        "peak_vram_bytes": 1_024,
                        "gaussian_count": 5,
                        "model_path": model.name,
                    },
                }
            )
            first = build_quality_report(run, split, root, root / "report-a")
            second = build_quality_report(run, split, root, root / "report-b")
            json_a = (root / "report-a" / "quality_report.json").read_bytes()
            json_b = (root / "report-b" / "quality_report.json").read_bytes()
            html_a = (root / "report-a" / "quality_report.html").read_bytes()
            html_b = (root / "report-b" / "quality_report.html").read_bytes()
            assets_a = sorted(
                path.read_bytes() for path in (root / "report-a" / "quality_assets").iterdir()
            )
            assets_b = sorted(
                path.read_bytes() for path in (root / "report-b" / "quality_assets").iterdir()
            )

        self.assertEqual(verify_run_manifest(run), run["run_manifest_sha256"])
        self.assertEqual(verify_quality_report(first), first["quality_report_sha256"])
        self.assertEqual(first, second)
        self.assertEqual(json_a, json_b)
        self.assertEqual(html_a, html_b)
        self.assertEqual(assets_a, assets_b)
        self.assertEqual(first["status"], "PARTIAL")
        self.assertEqual(first["warnings"], ["image_metrics.lpips:NOT_RUN"])
        self.assertEqual(first["summary"]["frame_count"], 2)
        self.assertEqual(len(first["golden_views"]), 2)
        self.assertGreater(first["summary"]["image_metrics"]["psnr_db"]["mean"], 30.0)
        self.assertLess(first["summary"]["depth_metrics"]["mae_m"]["mean"], 0.2)
        self.assertEqual(first["resources"]["gaussian_count"]["value"], 5)
        self.assertEqual(first["resources"]["model_size_bytes"]["value"], 15)

    def test_run_split_mismatch_fails_before_reporting(self) -> None:
        dataset = dataset_fixture(2)
        split = build_split_manifest(
            dataset,
            SplitConfig(mode="manual"),
            manual={"rig_000": "train", "rig_001": "val"},
        )
        run = sign_run_manifest(
            {
                "schema_version": 1,
                "run_id": "bad-run",
                "dataset_manifest_sha256": dataset["manifest_sha256"],
                "split_manifest_sha256": "0" * 64,
                "frames": [{"image_id": "img_left_001"}],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "different split manifest"):
                build_quality_report(run, split, Path(temporary), Path(temporary) / "out")

    def test_partial_validation_selection_is_rejected(self) -> None:
        dataset = dataset_fixture(3)
        split = build_split_manifest(
            dataset,
            SplitConfig(mode="manual"),
            manual={"rig_000": "train", "rig_001": "val", "rig_002": "train"},
        )
        run = sign_run_manifest(
            {
                "schema_version": 1,
                "run_id": "cherry-picked-run",
                "dataset_manifest_sha256": dataset["manifest_sha256"],
                "split_manifest_sha256": split["split_manifest_sha256"],
                "frames": [{"image_id": "img_left_001"}],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "exactly the validation images"):
                build_quality_report(run, split, Path(temporary), Path(temporary) / "out")


if __name__ == "__main__":
    unittest.main()
