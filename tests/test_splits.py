from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.splits import (
    SplitConfig,
    build_split_manifest,
    verify_split_manifest,
)


def dataset_fixture(frame_count: int = 20) -> dict:
    images = []
    rig_frames = []
    for index in range(frame_count):
        image_ids = []
        for side, offset in (("left", -0.05), ("right", 0.05)):
            image_id = f"img_{side}_{index:03d}"
            image_ids.append(image_id)
            pose = np.eye(4)
            pose[:3, 3] = [index * 0.1, offset, (index // 5) * 2.5]
            images.append(
                {
                    "image_id": image_id,
                    "camera_id": side,
                    "side": side,
                    "timestamp_ns": 1_000_000_000 + index * 100_000_000,
                    "path_root": "recording",
                    "path": f"camera/{side}/{index:03d}.jpg",
                    "c2w": pose.tolist(),
                }
            )
        rig_frames.append(
            {
                "rig_frame_id": f"rig_{index:03d}",
                "timestamp_ns": 1_000_000_000 + index * 100_000_000,
                "left_image_id": image_ids[0],
                "right_image_id": image_ids[1],
                "image_ids": image_ids,
                "timestamp_delta_ns": 0,
            }
        )
    manifest = {"schema_version": 1, "images": images, "rig_frames": rig_frames}
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


class SplitManifestTests(unittest.TestCase):
    def test_checked_in_real_baseline_preserves_gpu_gates(self) -> None:
        baseline = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "baselines"
                / "gs2_evaluation.baseline.json"
            ).read_text(encoding="utf-8")
        )

        self.assertTrue(baseline["acceptance"]["paired_images_same_split"])
        self.assertTrue(baseline["acceptance"]["split_runs_byte_identical"])
        self.assertEqual(baseline["output"]["validation_images"], 124)
        self.assertEqual(
            baseline["historical_gpu_run_audit"][
                "old_index_modulo_split_mismatched_rig_frames"
            ],
            84,
        )
        self.assertFalse(
            baseline["historical_gpu_run_audit"]["formal_pr08_compatible"]
        )
        self.assertEqual(baseline["acceptance"]["new_split_gpu_training"], "not_run")
        self.assertEqual(baseline["acceptance"]["real_masked_lpips"], "not_run")

    def test_temporal_split_keeps_stereo_pair_together_and_is_deterministic(self) -> None:
        dataset = dataset_fixture()
        config = SplitConfig(
            mode="temporal_block",
            validation_fraction=0.2,
            temporal_block_count=5,
            seed=1,
            nearest_train_warning_m=0.2,
            golden_rig_frames=3,
        )
        first = build_split_manifest(dataset, config)
        second = build_split_manifest(dataset, config)

        self.assertEqual(first, second)
        self.assertTrue(first["summary"]["paired_images_same_split"])
        self.assertEqual(first["summary"]["val_rig_frames"], 4)
        self.assertEqual(first["summary"]["val_images"], 8)
        self.assertEqual(len(first["golden_views"]), 3)
        for frame in first["rig_frames"]:
            split_images = set(first["splits"][frame["split"]])
            self.assertTrue(set(frame["image_ids"]) <= split_images)
        self.assertEqual(verify_split_manifest(first), first["split_manifest_sha256"])

    def test_spatial_split_holds_out_whole_cells(self) -> None:
        dataset = dataset_fixture()
        result = build_split_manifest(
            dataset,
            SplitConfig(
                mode="spatial_block",
                validation_fraction=0.25,
                spatial_cell_m=2.0,
                seed=9,
            ),
        )
        cells: dict[tuple[int, int, int], set[str]] = {}
        for frame in result["rig_frames"]:
            cell = tuple(np.floor(np.asarray(frame["position_m"]) / 2.0).astype(int))
            cells.setdefault(cell, set()).add(frame["split"])

        self.assertTrue(all(len(splits) == 1 for splits in cells.values()))
        self.assertGreater(result["summary"]["train_rig_frames"], 0)
        self.assertGreater(result["summary"]["val_rig_frames"], 0)

    def test_manual_split_requires_every_rig_frame(self) -> None:
        dataset = dataset_fixture(4)
        assignments = {
            f"rig_{index:03d}": "val" if index == 2 else "train" for index in range(4)
        }
        result = build_split_manifest(
            dataset,
            SplitConfig(mode="manual"),
            manual=assignments,
        )
        self.assertEqual(result["summary"]["val_rig_frames"], 1)
        with self.assertRaisesRegex(ValueError, "manual split IDs differ"):
            build_split_manifest(
                dataset,
                SplitConfig(mode="manual"),
                manual={"rig_000": "train"},
            )

    def test_nearby_validation_views_create_explicit_leakage_warning(self) -> None:
        dataset = dataset_fixture(6)
        assignments = {
            f"rig_{index:03d}": "val" if index == 2 else "train" for index in range(6)
        }
        result = build_split_manifest(
            dataset,
            SplitConfig(mode="manual", nearest_train_warning_m=0.11),
            manual=assignments,
        )

        self.assertEqual(result["leakage"]["warning_count"], 1)
        self.assertLess(result["leakage"]["warnings"][0]["nearest_train_distance_m"], 0.11)
        result["summary"]["val_images"] = 999
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_split_manifest(result)

    def test_verifier_rejects_overlapping_image_lists(self) -> None:
        result = build_split_manifest(dataset_fixture(3), SplitConfig())
        image_id = result["splits"]["train"][0]
        result["splits"]["val"].append(image_id)
        unsigned = dict(result)
        unsigned.pop("split_manifest_sha256")
        result["split_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(unsigned)
        ).hexdigest()

        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_split_manifest(result)


if __name__ == "__main__":
    unittest.main()
