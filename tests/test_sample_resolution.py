"""Resolution-matched supervision (cloudstudio_3dgs/training/sample_resolution.py), CPU only.

What is pinned:

* a reduced pixel centre lies on the ray of the centre of its source block, so the camera the
  rasterizer sees and the photo it is compared with stay registered;
* colour is the block mean; a supervision mask survives only where the whole block was
  supervised; the sky label is a majority vote;
* sparse LiDAR keeps the nearest return of a block with that return's own confidence;
* the z-to-range factor is the one the reduced intrinsics imply;
* a face at or below the target focal length is returned untouched (never enlarged);
* pixel radii map to the same angle and never round a non-zero radius to zero.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.sample_resolution import (
    pixels,
    reduce_sample,
    reduced_shape,
    reduction_for_focal,
)


def make_sample(height: int = 12, width: int = 18, focal: float = 30.0) -> TrainingSample:
    rng = np.random.default_rng(11)
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    jj, ii = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    scale = np.sqrt(1.0 + ((jj - K[0, 2]) / focal) ** 2 + ((ii - K[1, 2]) / focal) ** 2).astype(np.float32)
    return TrainingSample(
        image_id="img::face",
        rig_frame_id="rig",
        camera_id="left",
        image=rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        rgb_mask=np.ones((height, width), dtype=bool),
        depth_range_m=np.zeros((height, width), dtype=np.float32),
        depth_confidence=np.zeros((height, width), dtype=np.float32),
        depth_mask=np.zeros((height, width), dtype=bool),
        depth_cache_path=None,
        c2w=np.eye(4, dtype=np.float32),
        K=K,
        radial_coeffs=np.zeros(4, dtype=np.float32),
        width=width,
        height=height,
        camera_model="pinhole",
        depth_to_range_scale=scale,
        mono_depth_range_m=np.full((height, width), 4.0, dtype=np.float32),
        mono_depth_mask=np.ones((height, width), dtype=bool),
        sky_mask=np.zeros((height, width), dtype=bool),
    )


class ReductionRuleTests(unittest.TestCase):
    def test_a_face_is_never_enlarged(self) -> None:
        self.assertEqual(reduction_for_focal(1039.7, 787.6), 1039.7 / 787.6)
        self.assertEqual(reduction_for_focal(500.0, 787.6), 1.0)
        self.assertEqual(reduction_for_focal(2308.9, None), 1.0)
        with self.assertRaises(ValueError):
            reduction_for_focal(2308.9, 0.0)

    def test_untouched_at_unit_reduction(self) -> None:
        sample = make_sample()
        self.assertIs(reduce_sample(sample, 1.0), sample)

    def test_pixel_radii_keep_their_angle_and_never_vanish(self) -> None:
        self.assertEqual(pixels(24, 3.0), 8)
        self.assertEqual(pixels(3, 2.93), 1)
        self.assertEqual(pixels(1, 2.93), 1)
        self.assertEqual(pixels(0, 2.93), 0)
        self.assertEqual(pixels(6, 1.0), 6)


class SensorFocalTests(unittest.TestCase):
    MANIFEST = {
        "cameras": [
            {"camera_id": "left", "intrinsic": {"fl_x": 787.64, "fl_y": 787.64}},
            {"camera_id": "right", "intrinsic": {"fl_x": 789.10, "fl_y": 789.10}},
        ]
    }

    def test_sensor_reads_the_largest_camera_focal_from_the_capture(self) -> None:
        from cloudstudio_3dgs.training.sample_resolution import resolve_training_focal_px

        self.assertEqual(resolve_training_focal_px("sensor", self.MANIFEST), 789.10)
        self.assertEqual(resolve_training_focal_px(640.0, None), 640.0)
        self.assertIsNone(resolve_training_focal_px(None, self.MANIFEST))

    def test_sensor_without_calibration_is_refused(self) -> None:
        from cloudstudio_3dgs.training.sample_resolution import resolve_training_focal_px

        with self.assertRaises(ValueError):
            resolve_training_focal_px("sensor", {"cameras": [{"camera_id": "left"}]})
        with self.assertRaises(ValueError):
            resolve_training_focal_px("sensor", None)
        with self.assertRaises(ValueError):
            resolve_training_focal_px("native", self.MANIFEST)
        with self.assertRaises(ValueError):
            resolve_training_focal_px(0.0, self.MANIFEST)


class ReduceSampleTests(unittest.TestCase):
    def test_colour_is_the_block_mean_and_the_camera_stays_registered(self) -> None:
        sample = make_sample()
        reduced = reduce_sample(sample, 3.0)
        self.assertEqual((reduced.height, reduced.width), reduced_shape(12, 18, 3.0))
        self.assertEqual(reduced.image.shape, (4, 6, 3))
        block = sample.image[3:6, 6:9].reshape(-1, 3).astype(np.float64).mean(axis=0)
        self.assertTrue(np.all(np.abs(reduced.image[1, 2].astype(np.float64) - block) <= 1.0))
        # The centre of reduced pixel (row 1, col 2) is the centre of source block rows 3-5, cols 6-8.
        ray_reduced = np.linalg.inv(reduced.K.astype(np.float64)) @ np.array([2.5, 1.5, 1.0])
        ray_source = np.linalg.inv(sample.K.astype(np.float64)) @ np.array([7.5, 4.5, 1.0])
        np.testing.assert_allclose(ray_reduced, ray_source, atol=1e-6)
        self.assertEqual(reduced.resolution_reduction, 3.0)
        self.assertAlmostEqual(
            float(reduced.depth_to_range_scale[1, 2]), float(np.linalg.norm(ray_reduced)), places=5
        )

    def test_a_partly_masked_block_is_not_supervised(self) -> None:
        sample = make_sample()
        mask = sample.rgb_mask.copy()
        mask[4, 7] = False  # one pixel of block (1, 2)
        reduced = reduce_sample(replace(sample, rgb_mask=mask), 3.0)
        self.assertFalse(bool(reduced.rgb_mask[1, 2]))
        self.assertEqual(int(reduced.rgb_mask.sum()), 4 * 6 - 1)
        self.assertFalse(bool(reduced.mono_depth_mask[1, 2]))

    def test_sparse_lidar_keeps_the_nearest_return_and_its_confidence(self) -> None:
        sample = make_sample()
        valid = np.zeros((12, 18), dtype=bool)
        ranges = np.zeros((12, 18), dtype=np.float32)
        confidence = np.zeros((12, 18), dtype=np.float32)
        # Two returns in block (1, 2): the wall behind at 9 m and a silhouette at 2 m.
        valid[3, 6] = valid[5, 8] = True
        ranges[3, 6], confidence[3, 6] = 9.0, 0.9
        ranges[5, 8], confidence[5, 8] = 2.0, 0.4
        # One lone return in block (3, 0).
        valid[10, 1] = True
        ranges[10, 1], confidence[10, 1] = 5.0, 0.7
        reduced = reduce_sample(
            replace(sample, depth_range_m=ranges, depth_confidence=confidence, depth_mask=valid), 3.0
        )
        self.assertEqual(int(reduced.depth_mask.sum()), 2)
        self.assertAlmostEqual(float(reduced.depth_range_m[1, 2]), 2.0)
        self.assertAlmostEqual(float(reduced.depth_confidence[1, 2]), 0.4, places=6)
        self.assertAlmostEqual(float(reduced.depth_range_m[3, 0]), 5.0)
        self.assertEqual(float(reduced.depth_range_m[0, 0]), 0.0)

    def test_monocular_depth_averages_valid_pixels_only(self) -> None:
        sample = make_sample()
        mono = np.full((12, 18), 4.0, dtype=np.float32)
        mono_mask = np.ones((12, 18), dtype=bool)
        mono[0:3, 0:3] = 100.0
        mono_mask[0:3, 0:2] = False  # two thirds of block (0, 0) invalid -> dropped
        mono[3:6, 0] = 100.0
        mono_mask[3:6, 0] = False  # one third of block (1, 0) invalid -> mean of the rest
        reduced = reduce_sample(replace(sample, mono_depth_range_m=mono, mono_depth_mask=mono_mask), 3.0)
        self.assertFalse(bool(reduced.mono_depth_mask[0, 0]))
        self.assertTrue(bool(reduced.mono_depth_mask[1, 0]))
        self.assertAlmostEqual(float(reduced.mono_depth_range_m[1, 0]), 4.0, places=4)

    def test_sky_label_is_a_majority_vote(self) -> None:
        sample = make_sample()
        sky = np.zeros((12, 18), dtype=bool)
        sky[0:3, 0:3] = True
        sky[0, 0] = False  # 8 of 9
        sky[3:6, 0] = True  # 3 of 9
        reduced = reduce_sample(replace(sample, sky_mask=sky), 3.0)
        self.assertTrue(bool(reduced.sky_mask[0, 0]))
        self.assertFalse(bool(reduced.sky_mask[1, 0]))

    def test_a_non_integer_reduction_uses_the_exact_per_axis_ratio(self) -> None:
        sample = make_sample(height=14, width=20, focal=40.0)
        reduced = reduce_sample(sample, 1.32)
        self.assertEqual((reduced.height, reduced.width), (11, 15))
        self.assertAlmostEqual(float(reduced.K[0, 0]), 40.0 * 15 / 20, places=5)
        self.assertAlmostEqual(float(reduced.K[1, 1]), 40.0 * 11 / 14, places=5)
        # Image corners map to image corners on both axes.
        self.assertAlmostEqual(float(reduced.K[0, 2]) / 15, float(sample.K[0, 2]) / 20, places=6)
        self.assertAlmostEqual(float(reduced.K[1, 2]) / 11, float(sample.K[1, 2]) / 14, places=6)

    def test_a_sliver_of_supervision_keeps_a_non_empty_mask(self) -> None:
        sample = make_sample()
        mask = np.zeros((12, 18), dtype=bool)
        mask[0:2, 0:3] = True  # never fills a whole 3x3 block
        reduced = reduce_sample(replace(sample, rgb_mask=mask), 3.0)
        self.assertTrue(bool(reduced.rgb_mask.any()))
        self.assertTrue(bool(reduced.rgb_mask[0, 0]))


if __name__ == "__main__":
    unittest.main()
