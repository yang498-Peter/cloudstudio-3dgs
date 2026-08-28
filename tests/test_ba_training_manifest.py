from __future__ import annotations

import hashlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cloudstudio_3dgs.ba.report import sign_ba_report
from cloudstudio_3dgs.ba.training_manifest import (
    build_ba_training_manifest,
    build_independent_at_training_manifest,
    directory_sha256,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest


def _signed(payload: dict, field: str) -> dict:
    result = dict(payload)
    result[field] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _dataset() -> dict:
    cameras = []
    for side in ("left", "right"):
        cameras.append(
            {
                "camera_id": side,
                "width": 32,
                "height": 32,
                "intrinsic": {"fl_x": 20.0, "fl_y": 21.0, "cx": 16.0, "cy": 16.0},
                "distortion": {
                    "camera_model": "OPENCV_FISHEYE",
                    "params": {"k1": 0.1, "k2": -0.01, "k3": 0.0, "k4": 0.0},
                },
            }
        )
    images = []
    for frame in range(2):
        for side in ("left", "right"):
            images.append(
                {
                    "image_id": f"img_{side}_{frame}",
                    "camera_id": side,
                    "path": f"camera/{side}/{frame}.png",
                    "pose_convention": "c2w_opencv",
                    "c2w": np.eye(4).tolist(),
                }
            )
    return _signed(
        {
            "schema_version": 1,
            "coordinate_frame": "s1_local",
            "source_hashes": {},
            "warnings": [],
            "cameras": cameras,
            "images": images,
        },
        "manifest_sha256",
    )


class _Rigid:
    def __init__(self, translation_x: float):
        self.translation_x = translation_x

    def matrix(self):
        return np.asarray(
            [[1.0, 0.0, 0.0, -self.translation_x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )


class _Image:
    def __init__(self, name: str, camera_id: int, translation_x: float):
        self.name = name
        self.camera_id = camera_id
        self.has_pose = True
        self._rigid = _Rigid(translation_x)

    def cam_from_world(self):
        return self._rigid


class _Camera:
    model_name = "OPENCV_FISHEYE"
    width = 32
    height = 32
    params_info = "fx, fy, cx, cy, k1, k2, k3, k4"

    def __init__(self, fx: float):
        self.params = np.asarray([fx, 21.5, 16.0, 16.0, 0.1, -0.01, 0.0, 0.0])


class _Model:
    def __init__(self):
        self.images = {
            1: _Image("left/0.png", 1, 1.0),
            2: _Image("right/0.png", 2, 1.1),
        }
        self._cameras = {1: _Camera(20.5), 2: _Camera(20.6)}

    def num_images(self):
        return len(self.images)

    def num_cameras(self):
        return len(self._cameras)

    def camera(self, camera_id: int):
        return self._cameras[camera_id]


class _AllImageModel(_Model):
    def __init__(self):
        super().__init__()
        for camera in self._cameras.values():
            camera.params = np.asarray(
                [20.0, 21.0, 16.0, 16.0, 0.1, -0.01, 0.0, 0.0]
            )
        self.images.update(
            {
                3: _Image("left/1.png", 1, 2.0),
                4: _Image("right/1.png", 2, 2.1),
            }
        )


class BaTrainingManifestTests(unittest.TestCase):
    def _inputs(self, model_dir: Path):
        dataset = _dataset()
        split = _signed(
            {
                "schema_version": 1,
                "dataset_manifest_sha256": dataset["manifest_sha256"],
                "splits": {
                    "train": ["img_left_0", "img_right_0"],
                    "val": ["img_left_1", "img_right_1"],
                },
                "rig_frames": [
                    {
                        "rig_frame_id": "rig_0",
                        "image_ids": ["img_left_0", "img_right_0"],
                        "split": "train",
                    },
                    {
                        "rig_frame_id": "rig_1",
                        "image_ids": ["img_left_1", "img_right_1"],
                        "split": "val",
                    },
                ],
                "golden_views": [
                    {
                        "rig_frame_id": "rig_1",
                        "image_ids": ["img_left_1", "img_right_1"],
                    }
                ],
            },
            "split_manifest_sha256",
        )
        report = sign_ba_report(
            {
                "schema_version": 1,
                "stage": "stage_2",
                "after_model_sha256": directory_sha256(model_dir),
                "gates": {"solver_success": {"status": "PASS"}},
                "candidate_accepted": True,
                "published_model": "after",
                "position_prior": {"stddev_m": 0.02},
            }
        )
        return dataset, split, report

    def test_accepted_candidate_replaces_train_pose_and_global_focal_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "images.bin").write_bytes(b"candidate")
            dataset, split, report = self._inputs(model_dir)
            fake_pycolmap = types.SimpleNamespace(Reconstruction=lambda _path: _Model())

            with patch.dict("sys.modules", {"pycolmap": fake_pycolmap}):
                derived = build_ba_training_manifest(dataset, split, report, model_dir)

            verify_dataset_manifest(derived)
            images = {item["image_id"]: item for item in derived["images"]}
            cameras = {item["camera_id"]: item for item in derived["cameras"]}
            self.assertAlmostEqual(images["img_left_0"]["c2w"][0][3], 1.0)
            self.assertAlmostEqual(images["img_right_0"]["c2w"][0][3], 1.1)
            self.assertEqual(images["img_left_0"]["pose_source"], "accepted_fixed_rig_ba")
            self.assertEqual(images["img_left_1"]["c2w"], np.eye(4).tolist())
            self.assertNotIn("pose_source", images["img_left_1"])
            self.assertAlmostEqual(cameras["left"]["intrinsic"]["fl_x"], 20.5)
            self.assertEqual(
                derived["training_lineage"]["base_dataset_manifest_sha256"],
                dataset["manifest_sha256"],
            )

    def test_rejected_report_cannot_publish_training_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "images.bin").write_bytes(b"candidate")
            dataset, split, report = self._inputs(model_dir)
            report = sign_ba_report(
                {
                    **{key: value for key, value in report.items() if key != "ba_report_sha256"},
                    "gates": {"solver_success": {"status": "FAIL"}},
                    "candidate_accepted": False,
                    "published_model": "before",
                }
            )

            with self.assertRaisesRegex(ValueError, "did not accept"):
                build_ba_training_manifest(dataset, split, report, model_dir)

    def test_model_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            model_file = model_dir / "images.bin"
            model_file.write_bytes(b"candidate")
            dataset, split, report = self._inputs(model_dir)
            model_file.write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "does not identify"):
                build_ba_training_manifest(dataset, split, report, model_dir)

    def test_converged_independent_at_replaces_all_image_poses(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "images.bin").write_bytes(b"independent-candidate")
            dataset, split, _report = self._inputs(model_dir)
            report = _signed(
                {
                    "schema_version": 1,
                    "algorithm_version": "independent_pos_prior_shared_kb4_at_v1",
                    "candidate_model_sha256": directory_sha256(model_dir),
                    "counts": {"images": 4},
                    "position_prior_sigma_xyz_m": [0.03, 0.03, 0.06],
                    "solver_usable": True,
                    "solver_converged": True,
                    "intrinsics_refined": False,
                },
                "report_sha256",
            )
            fake_pycolmap = types.SimpleNamespace(
                Reconstruction=lambda _path: _AllImageModel()
            )

            with patch.dict("sys.modules", {"pycolmap": fake_pycolmap}):
                derived = build_independent_at_training_manifest(
                    dataset, split, report, model_dir
                )

            verify_dataset_manifest(derived)
            images = {item["image_id"]: item for item in derived["images"]}
            self.assertAlmostEqual(images["img_left_0"]["c2w"][0][3], 1.0)
            self.assertAlmostEqual(images["img_left_1"]["c2w"][0][3], 2.0)
            self.assertTrue(
                all(
                    item["pose_source"] == "accepted_independent_pos_prior_at"
                    for item in images.values()
                )
            )
            self.assertEqual(
                derived["training_lineage"]["pose_policy"],
                "accepted_independent_at_all_images",
            )

    def test_independent_at_without_explicit_convergence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "images.bin").write_bytes(b"independent-candidate")
            dataset, split, _report = self._inputs(model_dir)
            report = _signed(
                {
                    "schema_version": 1,
                    "algorithm_version": "independent_pos_prior_shared_kb4_at_v1",
                    "candidate_model_sha256": directory_sha256(model_dir),
                    "counts": {"images": 4},
                    "position_prior_sigma_xyz_m": [0.03, 0.03, 0.06],
                    "solver_usable": True,
                    "solver_converged": False,
                    "intrinsics_refined": False,
                },
                "report_sha256",
            )

            with self.assertRaisesRegex(ValueError, "did not explicitly converge"):
                build_independent_at_training_manifest(
                    dataset, split, report, model_dir
                )


if __name__ == "__main__":
    unittest.main()
