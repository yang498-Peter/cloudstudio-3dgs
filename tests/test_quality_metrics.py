from __future__ import annotations

import math
import unittest

import numpy as np

from cloudstudio_3dgs.evaluation.image_metrics import (
    masked_depth_metrics,
    masked_lpips_from_distance_map,
    masked_psnr,
    masked_ssim,
)


class QualityMetricTests(unittest.TestCase):
    def test_black_border_is_excluded_from_psnr_and_ssim(self) -> None:
        target = np.full((32, 32, 3), 0.5, dtype=np.float64)
        prediction = target.copy()
        prediction[:8] = 0.0
        prediction[-8:] = 1.0
        prediction[:, :8] = 0.0
        prediction[:, -8:] = 1.0
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True

        self.assertTrue(math.isinf(masked_psnr(prediction, target, mask)))
        self.assertAlmostEqual(masked_ssim(prediction, target, mask), 1.0, places=12)
        prediction[16, 16] = 0.0
        self.assertFalse(math.isinf(masked_psnr(prediction, target, mask)))
        self.assertLess(masked_ssim(prediction, target, mask), 1.0)

    def test_spatial_lpips_aggregation_ignores_masked_border(self) -> None:
        distances = np.full((8, 8), 10.0, dtype=np.float64)
        distances[2:6, 2:6] = 0.25
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True

        self.assertAlmostEqual(masked_lpips_from_distance_map(distances, mask), 0.25)
        with self.assertRaisesRegex(ValueError, "no valid pixels"):
            masked_lpips_from_distance_map(distances, np.zeros((8, 8), dtype=bool))

    def test_lidar_ray_range_mae_and_rmse_use_only_supervised_pixels(self) -> None:
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.array([[1.1, 2.0], [2.8, 100.0]])
        valid = np.array([[True, True], [True, False]])
        confidence = np.array([[1.0, 0.5], [1.0, 0.0]])
        result = masked_depth_metrics(
            prediction, target, valid, confidence=confidence
        )

        self.assertEqual(result["valid_pixels"], 3)
        self.assertAlmostEqual(result["mae_m"], 0.12)
        self.assertAlmostEqual(result["rmse_m"], math.sqrt(0.05 / 2.5))
        prediction[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "coverage.*below the gate"):
            masked_depth_metrics(prediction, target, valid)

        relaxed = masked_depth_metrics(
            prediction,
            target,
            valid,
            confidence=confidence,
            minimum_prediction_coverage=0.6,
        )
        self.assertEqual(relaxed["valid_pixels"], 2)
        self.assertEqual(relaxed["target_valid_pixels"], 3)
        self.assertEqual(relaxed["missing_prediction_pixels"], 1)
        self.assertAlmostEqual(relaxed["prediction_coverage_fraction"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
