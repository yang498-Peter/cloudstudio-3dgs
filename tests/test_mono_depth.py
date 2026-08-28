import io
import unittest

import numpy as np

from cloudstudio_3dgs.data.mono_depth import (
    AffineAlignmentConfig,
    fit_metric_affine_ransac,
    mono_depth_npz_bytes,
    sample_bilinear_at_source_pixels,
)


class MonoDepthTests(unittest.TestCase):
    def test_ransac_recovers_affine_with_outliers(self) -> None:
        rng = np.random.default_rng(7)
        mono = np.linspace(0.2, 4.0, 4000)
        metric = 2.5 * mono + 0.7
        metric += rng.normal(0.0, 0.002, size=len(metric))
        metric[::8] += 5.0
        result = fit_metric_affine_ransac(mono, metric, seed=31)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["scale"], 2.5, places=2)
        self.assertAlmostEqual(result["shift"], 0.7, places=2)
        self.assertGreater(result["inlier_ratio"], 0.8)

    def test_insufficient_pairs_are_invalid(self) -> None:
        result = fit_metric_affine_ransac(
            np.ones(20), np.ones(20), seed=1, config=AffineAlignmentConfig()
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "insufficient_positive_pairs")

    def test_resized_sampling_uses_pixel_centers(self) -> None:
        image = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        sampled = sample_bilinear_at_source_pixels(
            image,
            np.asarray([0.5]),
            np.asarray([0.5]),
            source_shape=(2, 2),
        )
        self.assertAlmostEqual(float(sampled[0]), 2.5, places=6)

    def test_cache_payload_is_deterministic(self) -> None:
        depth = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertEqual(mono_depth_npz_bytes(depth), mono_depth_npz_bytes(depth))
        with np.load(io.BytesIO(mono_depth_npz_bytes(np.asarray([[1e20]])))) as data:
            self.assertTrue(np.isfinite(data["relative_depth"]).all())
            self.assertEqual(float(data["relative_depth"][0, 0]), 65504.0)


if __name__ == "__main__":
    unittest.main()
