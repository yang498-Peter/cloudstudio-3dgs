from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import sparse_depth_npz_bytes
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.quality_report import (
    _golden_checkpoint_selection,
    _periodic_full_evaluation,
    _safe_path,
    build_quality_report,
    finalize_deferred_run_manifest,
    sign_run_manifest,
    verify_quality_report,
    verify_run_manifest,
)
from cloudstudio_3dgs.evaluation.splits import SplitConfig, build_split_manifest
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap
from cloudstudio_3dgs.training.golden_eval import GoldenEvaluationConfig
from tests.test_splits import dataset_fixture


class QualityReportTests(unittest.TestCase):
    def test_finalize_deferred_run_manifest_binds_source_and_evaluator(self) -> None:
        source = sign_run_manifest(
            {
                "schema_version": 1,
                "run_id": "face-stage",
                "final_evaluation_artifacts": {
                    "enabled": False,
                    "status": "DEFERRED",
                    "reason": "separate_3dgut_evaluation_required",
                },
                "frames": [],
            }
        )
        final = finalize_deferred_run_manifest(
            source,
            frames=[{"image_id": "camera_a"}],
            evaluation_runtime={"projection": "3DGUT"},
            checkpoint_sha256="checkpoint-sha",
        )
        self.assertEqual(verify_run_manifest(final), final["run_manifest_sha256"])
        self.assertEqual(
            final["final_evaluation_artifacts"],
            {
                "enabled": True,
                "status": "COMPLETE",
                "reason": None,
                "source_face_stage_run_manifest_sha256": source[
                    "run_manifest_sha256"
                ],
                "evaluation_runtime": {"projection": "3DGUT"},
                "checkpoint_sha256": "checkpoint-sha",
            },
        )

    def test_signed_face_stage_manifest_allows_only_explicit_deferred_frames(self) -> None:
        deferred = sign_run_manifest(
            {
                "schema_version": 1,
                "run_id": "face-stage",
                "final_evaluation_artifacts": {
                    "enabled": False,
                    "status": "DEFERRED",
                    "reason": "separate_3dgut_evaluation_required",
                },
                "frames": [],
            }
        )
        self.assertEqual(
            verify_run_manifest(deferred),
            deferred["run_manifest_sha256"],
        )

        incomplete = sign_run_manifest(
            {
                "schema_version": 1,
                "run_id": "invalid-face-stage",
                "frames": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "invalid or duplicate image IDs"):
            verify_run_manifest(incomplete)

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
        self.assertEqual(
            first["warnings"],
            [
                "golden_evaluation:NOT_RUN",
                "periodic_full_evaluation:NOT_RUN",
                "image_metrics.lpips:NOT_RUN",
            ],
        )
        self.assertEqual(first["golden_checkpoint_selection"]["status"], "NOT_RUN")
        self.assertEqual(first["summary"]["frame_count"], 2)
        self.assertEqual(len(first["golden_views"]), 2)
        self.assertGreater(first["summary"]["image_metrics"]["psnr_db"]["mean"], 30.0)
        self.assertLess(first["summary"]["depth_metrics"]["mae_m"]["mean"], 0.2)
        self.assertEqual(first["resources"]["gaussian_count"]["value"], 5)
        self.assertEqual(first["resources"]["model_size_bytes"]["value"], 15)

    def test_golden_checkpoint_history_is_verified_before_quality_use(self) -> None:
        config = GoldenEvaluationConfig()
        artifact_sha = hashlib.sha256(b"artifact").hexdigest()
        record = {
            "schema_version": 1,
            "algorithm_version": "golden_validation_v1",
            "completed_steps": 1000,
            "split_manifest_sha256": "split-sha",
            "image_ids": ["golden-left"],
            "frames": [
                {
                    "image_id": "golden-left",
                    "artifacts": {
                        "rendered_path": "evaluation/golden_rendered.png",
                        "rendered_sha256": artifact_sha,
                        "reference_path": "evaluation/golden_reference.png",
                        "reference_sha256": artifact_sha,
                        "mask_path": "evaluation/golden_mask.png",
                        "mask_sha256": artifact_sha,
                    },
                }
            ],
            "summary": {
                "selection_metric": "masked_rgb_psnr_db_mean",
                "psnr_db_mean": 24.0,
                "perfect_psnr_frame_count": 0,
                "ssim_mean": 0.7,
                "depth_mae_m_mean": None,
            },
        }
        record["golden_evaluation_sha256"] = hashlib.sha256(
            canonical_json_bytes(record)
        ).hexdigest()
        history = {
            "schema_version": 1,
            "configuration": config.to_dict(),
            "history": [record],
            "best": record,
        }
        history["golden_history_sha256"] = hashlib.sha256(
            canonical_json_bytes(history)
        ).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path = root / "evaluation" / "golden_history.json"
            history_path.parent.mkdir()
            history_path.write_text(json.dumps(history), encoding="utf-8")
            for name in ("golden_rendered.png", "golden_reference.png", "golden_mask.png"):
                (history_path.parent / name).write_bytes(b"artifact")
            checkpoint_path = root / "checkpoints" / "best_golden.pt"
            checkpoint_path.parent.mkdir()
            checkpoint_path.write_bytes(b"checkpoint")
            run = {
                "training": {"completed_steps": 1000},
                "golden_evaluation": {
                    "configuration": config.to_dict(),
                    "history_path": "evaluation/golden_history.json",
                    "history_sha256": history["golden_history_sha256"],
                    "evaluation_count": 1,
                    "best": record,
                    "best_checkpoint_path": "checkpoints/best_golden.pt",
                    "best_checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                }
            }
            selection, warnings = _golden_checkpoint_selection(run, root)
            self.assertEqual(selection["status"], "VERIFIED")
            self.assertEqual(warnings, [])
            history["history"][0]["summary"]["psnr_db_mean"] = 99.0
            history_path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                _golden_checkpoint_selection(run, root)

    def test_periodic_full_history_is_verified(self) -> None:
        record = {
            "schema_version": 1,
            "algorithm_version": "full_validation_v2",
            "evaluation_kind": "full",
            "completed_steps": 4000,
            "split_manifest_sha256": "split-sha",
            "image_ids": ["val"],
            "frames": [],
            "summary": {"psnr_db_mean": 20.0},
        }
        record["full_evaluation_sha256"] = hashlib.sha256(
            canonical_json_bytes(record)
        ).hexdigest()
        history = {"schema_version": 1, "history": [record]}
        history["full_evaluation_history_sha256"] = hashlib.sha256(
            canonical_json_bytes(history)
        ).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "evaluation" / "full_evaluation_history.json"
            path.parent.mkdir()
            path.write_text(json.dumps(history), encoding="utf-8")
            run = {
                "training": {"completed_steps": 4000},
                "golden_evaluation": {"configuration": GoldenEvaluationConfig().to_dict()},
                "periodic_full_evaluation": {
                    "history_path": "evaluation/full_evaluation_history.json",
                    "history_sha256": history["full_evaluation_history_sha256"],
                    "evaluation_count": 1,
                    "latest": record,
                }
            }
            report, warnings = _periodic_full_evaluation(run, root)
            self.assertEqual(report["status"], "VERIFIED")
            self.assertEqual(report["latest_completed_steps"], 4000)
            self.assertEqual(warnings, [])

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
