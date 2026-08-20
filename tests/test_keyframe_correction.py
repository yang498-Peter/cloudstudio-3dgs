from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cloudstudio_3dgs.poses.keyframe_correction import (
    PoseCorrectionConfig,
    build_corrected_pose_set,
    verify_pose_set_manifest,
    write_pose_set_outputs,
)
from tests.test_splits import dataset_fixture


GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def correction_at(index: int) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", index * 0.005).as_matrix()
    value[:3, 3] = [index * 0.01, index * -0.002, index * 0.001]
    return value


def transforms_fixture(
    dataset: dict,
    anchor_indexes: list[int],
    *,
    outlier_index: int | None = None,
) -> dict:
    images = {image["image_id"]: image for image in dataset["images"]}
    frames = []
    for index in anchor_indexes:
        correction = correction_at(index)
        if index == outlier_index:
            correction = correction.copy()
            correction[:3, 3] += [1.0, -0.8, 0.6]
            correction[:3, :3] = Rotation.from_euler("y", 0.8).as_matrix()
        rig = dataset["rig_frames"][index]
        for image_id in rig["image_ids"]:
            image = images[image_id]
            target_cv = correction @ np.asarray(image["c2w"], dtype=np.float64)
            target_gl = target_cv.copy()
            target_gl[:3, :3] = target_cv[:3, :3] @ GL_TO_CV
            frames.append(
                {
                    "file_path": image["path"].removeprefix("camera/").replace("/", "\\"),
                    "transform_matrix": target_gl.tolist(),
                }
            )
    return {"frames": frames}


class KeyframeCorrectionTests(unittest.TestCase):
    def test_checked_in_real_baseline_keeps_candidate_non_default(self) -> None:
        baseline = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "baselines"
                / "gs2_pose_correction.baseline.json"
            ).read_text(encoding="utf-8")
        )

        self.assertTrue(baseline["acceptance"]["rig_baseline_preserved"])
        self.assertTrue(baseline["acceptance"]["repeat_outputs_byte_identical"])
        self.assertFalse(baseline["acceptance"]["correction_curve_no_jump"])
        self.assertFalse(baseline["acceptance"]["candidate_accepted_as_default"])
        self.assertEqual(baseline["output"]["default_pose_set"], "imgpose")
        self.assertEqual(
            baseline["acceptance"]["real_low_resolution_lpips_improvement"],
            "not_run",
        )

    def test_linear_se3_correction_is_recovered_and_rig_baseline_is_fixed(self) -> None:
        dataset = dataset_fixture(13)
        transforms = transforms_fixture(dataset, [0, 3, 6, 9, 12])
        result = build_corrected_pose_set(
            dataset,
            transforms,
            transforms_sha256="1" * 64,
        )

        original = {image["image_id"]: image for image in dataset["images"]}
        for image in result["images"]:
            index = int(image["image_id"].rsplit("_", 1)[1])
            expected = correction_at(index) @ np.asarray(
                original[image["image_id"]]["c2w"], dtype=np.float64
            )
            np.testing.assert_allclose(image["c2w"], expected, atol=1e-10)
        self.assertEqual(
            verify_pose_set_manifest(result), result["pose_set_manifest_sha256"]
        )
        self.assertTrue(result["diagnostics"]["rig_baseline_preserved"])
        self.assertTrue(result["diagnostics"]["curve_is_smooth"])
        self.assertEqual(result["acceptance"]["default_pose_set"], "imgpose")
        self.assertFalse(result["summary"]["original_inputs_overwritten"])

    def test_temporal_outlier_is_filtered_before_interpolation(self) -> None:
        dataset = dataset_fixture(21)
        transforms = transforms_fixture(
            dataset,
            list(range(0, 21, 2)),
            outlier_index=10,
        )
        result = build_corrected_pose_set(
            dataset,
            transforms,
            transforms_sha256="2" * 64,
        )

        rejected = [anchor for anchor in result["anchors"] if not anchor["accepted"]]
        self.assertTrue(
            any(anchor["rig_frame_id"] == "rig_010" for anchor in rejected)
        )
        corrected = {
            image["image_id"]: np.asarray(image["c2w"])
            for image in result["images"]
        }
        original = {image["image_id"]: image for image in dataset["images"]}
        expected = correction_at(10) @ np.asarray(
            original["img_left_010"]["c2w"], dtype=np.float64
        )
        np.testing.assert_allclose(corrected["img_left_010"], expected, atol=1e-10)

    def test_default_changes_only_when_every_quality_gate_passes(self) -> None:
        dataset = dataset_fixture(13)
        transforms = transforms_fixture(dataset, [0, 3, 6, 9, 12])
        metrics = {
            "lidar_edge_error": {"baseline": 2.0, "candidate": 1.5, "unit": "px"},
            "low_resolution_lpips": {"baseline": 0.5, "candidate": 0.4},
            "building_double_edge_score": {"baseline": 0.2, "candidate": 0.2},
        }
        accepted = build_corrected_pose_set(
            dataset,
            transforms,
            transforms_sha256="3" * 64,
            acceptance_metrics=metrics,
        )
        self.assertEqual(
            accepted["acceptance"]["default_pose_set"], "keyframe_corrected"
        )

        metrics["low_resolution_lpips"]["candidate"] = 0.6
        rejected = build_corrected_pose_set(
            dataset,
            transforms,
            transforms_sha256="3" * 64,
            acceptance_metrics=metrics,
        )
        self.assertEqual(rejected["acceptance"]["default_pose_set"], "imgpose")

    def test_outputs_are_deterministic_and_include_curve_report(self) -> None:
        dataset = dataset_fixture(13)
        result = build_corrected_pose_set(
            dataset,
            transforms_fixture(dataset, [0, 3, 6, 9, 12]),
            transforms_sha256="4" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pose_set_outputs(root / "a", result)
            write_pose_set_outputs(root / "b", result)
            for name in (
                "pose_set_manifest.json",
                "pose_correction_curve.svg",
                "pose_correction_report.html",
            ):
                self.assertEqual((root / "a" / name).read_bytes(), (root / "b" / name).read_bytes())

    def test_incomplete_keyframe_rig_fails_closed(self) -> None:
        dataset = dataset_fixture(13)
        transforms = transforms_fixture(dataset, [0, 3, 6, 9, 12])
        transforms["frames"].pop()
        with self.assertRaisesRegex(ValueError, "one left and one right"):
            build_corrected_pose_set(
                dataset,
                transforms,
                transforms_sha256="5" * 64,
            )

    def test_tampered_manifest_fails_verification(self) -> None:
        dataset = dataset_fixture(13)
        result = build_corrected_pose_set(
            dataset,
            transforms_fixture(dataset, [0, 3, 6, 9, 12]),
            transforms_sha256="6" * 64,
        )
        result["summary"]["image_count"] = 1
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_pose_set_manifest(result)


if __name__ == "__main__":
    unittest.main()
