from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.data.image_sample import CropWindow, prepare_image_sample
from cloudstudio_3dgs.geometry.kb4 import unproject_kb4
from cloudstudio_3dgs.geometry.lidar_projection import (
    DepthProjectionConfig,
    project_lidar_depth,
)


def camera_fixture() -> dict:
    return {
        "camera_id": "left",
        "camera_type": "fisheye",
        "width": 64,
        "height": 64,
        "intrinsic": {"fl_x": 20.0, "fl_y": 20.0, "cx": 31.5, "cy": 31.5},
        "distortion": {
            "camera_model": "OPENCV_FISHEYE",
            "params": {"k1": 0.02, "k2": -0.003, "k3": 0.0002, "k4": 0.0},
        },
    }


def rays_for_pixels(pixels: np.ndarray, camera: dict) -> np.ndarray:
    return unproject_kb4(
        pixels,
        camera["intrinsic"],
        camera["distortion"]["params"],
    )


class LidarProjectionTests(unittest.TestCase):
    def test_synthetic_sphere_ray_range_error_is_below_one_millimetre(self) -> None:
        camera = camera_fixture()
        pixels = np.array(
            [[32, 32], [24, 24], [40, 24], [24, 40], [40, 40]], dtype=np.float64
        )
        points = rays_for_pixels(pixels, camera) * 5.0
        result = project_lidar_depth(points, np.eye(4), camera)

        self.assertEqual(len(result.range_m), len(points))
        self.assertLess(float(np.max(np.abs(result.range_m - 5.0))), 0.001)

    def test_synthetic_plane_ray_range_error_is_below_one_millimetre(self) -> None:
        camera = camera_fixture()
        pixels = np.array(
            [[32, 32], [28, 28], [36, 28], [28, 36], [36, 36]], dtype=np.float64
        )
        rays = rays_for_pixels(pixels, camera)
        expected_range = 4.0 / rays[:, 2]
        points = rays * expected_range[:, None]
        result = project_lidar_depth(points, np.eye(4), camera)

        self.assertEqual(len(result.range_m), len(points))
        self.assertLess(
            float(np.max(np.abs(np.sort(result.range_m) - np.sort(expected_range)))),
            0.001,
        )

    def test_z_buffer_keeps_only_the_nearest_surface(self) -> None:
        camera = camera_fixture()
        pixel = np.array([[32.0, 32.0]])
        ray = rays_for_pixels(pixel, camera)[0]
        points = np.vstack([ray * 7.0, ray * 3.0, ray * 5.0])
        result = project_lidar_depth(points, np.eye(4), camera)

        self.assertEqual(len(result.range_m), 1)
        self.assertAlmostEqual(float(result.range_m[0]), 3.0, places=6)
        self.assertEqual(int(result.source_index[0]), 1)
        self.assertEqual(int(result.support_count[0]), 3)

    def test_dynamic_mask_and_crop_remove_supervision_before_cache(self) -> None:
        camera = camera_fixture()
        pixels = np.array([[10.0, 10.0], [20.0, 20.0]])
        points = rays_for_pixels(pixels, camera) * 4.0
        supervision = np.ones((64, 64), dtype=bool)
        supervision[10, 10] = False
        crop = CropWindow(8, 8, 16, 16)
        result = project_lidar_depth(
            points,
            np.eye(4),
            camera,
            supervision_mask=supervision,
            crop=crop,
        )

        self.assertEqual(result.shape, (16, 16))
        self.assertEqual(len(result.range_m), 1)
        self.assertEqual(int(result.pixel_index[0]), (20 - 8) * 16 + (20 - 8))

    def test_sparse_cache_can_share_factor_and_crop_with_image_loader(self) -> None:
        camera = camera_fixture()
        pixels = np.array([[16.0, 16.0], [24.0, 24.0], [32.0, 32.0]])
        points = rays_for_pixels(pixels, camera) * 6.0
        sparse = project_lidar_depth(points, np.eye(4), camera)
        depth, confidence, depth_valid = sparse.to_dense()
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        valid = np.ones((64, 64), dtype=bool)

        for factor in (1, 2, 4):
            sample = prepare_image_sample(
                image,
                valid,
                depth=depth,
                confidence=confidence,
                depth_valid_mask=depth_valid,
                crop=CropWindow(8, 8, 48, 48),
                factor=factor,
            )
            expected = (48 // factor, 48 // factor)
            self.assertEqual(sample.depth.shape, expected)
            self.assertEqual(sample.confidence.shape, expected)
            self.assertEqual(sample.mask.shape, expected)


if __name__ == "__main__":
    unittest.main()
