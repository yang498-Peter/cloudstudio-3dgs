from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "converter"))

from s1_to_colmap import (
    compare_transform_intrinsics,
    load_calibration_intrinsics,
    load_imgpose_frames,
    quat_xyzw_to_rotmat,
)


class ImgPoseConversionTests(unittest.TestCase):
    def test_identity_c2w_pose_becomes_inverse_translation(self) -> None:
        template = {
            "file_path": "left\\keyframe.jpg",
            "w": 2912,
            "h": 2912,
            "fl_x": 788.0,
            "fl_y": 789.0,
            "cx": 1456.0,
            "cy": 1456.0,
            "k1": 0.08,
            "k2": -0.01,
            "k3": 0.0,
            "k4": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "ImgPose.txt").write_text(
                "index x y z roll pitch yaw qx qy qz qw timestamp\n"
                "left/1.jpg 1 2 3 0 0 0 0 0 0 1 123\n",
                encoding="utf-8",
            )
            frames = load_imgpose_frames(run_dir, {"left": template})

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["file_path"], "left/1.jpg")
        np.testing.assert_allclose(frames[0]["_r_w2c"], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(frames[0]["_t_w2c"], [-1, -2, -3], atol=1e-12)
        self.assertEqual(frames[0]["fl_x"], 788.0)

    def test_quaternion_is_normalized(self) -> None:
        np.testing.assert_allclose(quat_xyzw_to_rotmat([0, 0, 0, 2]), np.eye(3), atol=1e-12)

    def test_zero_quaternion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-length quaternion"):
            quat_xyzw_to_rotmat([0, 0, 0, 0])

    def test_recording_calibration_is_authoritative_over_transform_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recording = Path(tmp)
            (recording / "info").mkdir()
            cameras = []
            for side, focal in (("left", 700.0), ("right", 710.0)):
                cameras.append(
                    {
                        "name": side,
                        "type": "fisheye",
                        "width": 2912,
                        "height": 2912,
                        "intrinsic": {"fl_x": focal, "fl_y": focal + 1, "cx": 1450, "cy": 1451},
                        "distortion": {
                            "camera_model": "OPENCV_FISHEYE",
                            "params": {"k1": 0.1, "k2": 0.2, "k3": 0.3, "k4": 0.4},
                        },
                    }
                )
            (recording / "info" / "calibration.json").write_text(
                json.dumps({"cameras": cameras}), encoding="utf-8"
            )
            calibration = load_calibration_intrinsics(recording)
            transforms = {
                "frames": [
                    {"file_path": f"{side}/1.jpg", **{key: 999.0 for key in (
                        "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4"
                    )}}
                    for side in ("left", "right")
                ]
            }
            differences = compare_transform_intrinsics(calibration, transforms)

        self.assertEqual(calibration["left"]["fl_x"], 700.0)
        self.assertEqual(calibration["right"]["fl_x"], 710.0)
        self.assertEqual(differences["left"]["fl_x"], 299.0)


if __name__ == "__main__":
    unittest.main()
