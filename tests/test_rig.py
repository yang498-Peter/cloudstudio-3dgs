from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.data.schema import CameraRecord, ImageRecord
from cloudstudio_3dgs.geometry.rig import build_stereo_rig


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def matrix(rotation: np.ndarray, translation: list[float]) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def camera(side: str, camera_from_lidar: np.ndarray) -> CameraRecord:
    return CameraRecord(
        camera_id=side,
        side=side,
        camera_type="fisheye",
        width=2912,
        height=2912,
        intrinsic={"fl_x": 788.0, "fl_y": 789.0, "cx": 1456.0, "cy": 1456.0},
        distortion={
            "camera_model": "OPENCV_FISHEYE",
            "params": {"k1": 0.08, "k2": -0.01, "k3": 0.0, "k4": 0.0},
        },
        transform_from_lidar={
            "rotation": camera_from_lidar[:3, :3].tolist(),
            "position": camera_from_lidar[:3, 3].tolist(),
        },
    )


def image(side: str, timestamp_ns: int, world_from_camera: np.ndarray) -> ImageRecord:
    return ImageRecord(
        image_id=f"img_{side}_{timestamp_ns}",
        rig_frame_id=None,
        side=side,
        timestamp_ns=timestamp_ns,
        path_root="recording",
        path=f"camera/{side}/{timestamp_ns}.jpg",
        size_bytes=1,
        sha256="0" * 64,
        camera_id=side,
        pose_source="synthetic",
        pose_convention="c2w_opencv",
        c2w=world_from_camera.tolist(),
    )


class StereoRigTests(unittest.TestCase):
    def test_nontrivial_synthetic_rig_recovers_fixed_extrinsics(self) -> None:
        left_from_lidar = matrix(rotation_x(0.23), [0.02, -0.07, -0.03])
        right_from_lidar = matrix(rotation_y(-0.41), [-0.03, -0.06, 0.04])
        world_from_lidar = matrix(rotation_y(0.37) @ rotation_x(-0.12), [5.0, -2.0, 1.5])
        world_from_left = world_from_lidar @ np.linalg.inv(left_from_lidar)
        world_from_right = world_from_lidar @ np.linalg.inv(right_from_lidar)

        updated, frames, rig, diagnostics = build_stereo_rig(
            [camera("left", left_from_lidar), camera("right", right_from_lidar)],
            [
                image("left", 1_000_000_000, world_from_left),
                image("right", 1_000_000_300, world_from_right),
            ],
        )

        expected = left_from_lidar @ np.linalg.inv(right_from_lidar)
        np.testing.assert_allclose(rig["expected_right_to_left"], expected, atol=1e-12)
        self.assertEqual(len(frames), 1)
        self.assertEqual(updated[0].rig_frame_id, updated[1].rig_frame_id)
        self.assertLess(diagnostics["relative_translation_error_m"]["max"], 1e-12)
        self.assertLess(diagnostics["relative_rotation_error_rad"]["max"], 1e-12)

    def test_images_outside_tolerance_remain_explicitly_unpaired(self) -> None:
        identity = np.eye(4, dtype=np.float64)
        _updated, frames, _rig, diagnostics = build_stereo_rig(
            [camera("left", identity), camera("right", identity)],
            [image("left", 1_000, identity), image("right", 60_000_001, identity)],
            tolerance_ns=50_000_000,
        )

        self.assertEqual(frames, [])
        self.assertEqual(diagnostics["unpaired_left"], ["camera/left/1000.jpg"])
        self.assertEqual(diagnostics["unpaired_right"], ["camera/right/60000001.jpg"])


if __name__ == "__main__":
    unittest.main()
