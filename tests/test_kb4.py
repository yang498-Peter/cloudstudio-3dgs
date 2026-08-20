from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.geometry.kb4 import project_kb4, unproject_kb4


class Kb4ProjectionTests(unittest.TestCase):
    def test_synthetic_pixel_roundtrip_is_below_point_zero_five_pixels(self) -> None:
        intrinsic = {"fl_x": 788.2, "fl_y": 789.1, "cx": 1453.9, "cy": 1450.8}
        distortion = {"k1": 0.081, "k2": -0.011, "k3": -0.003, "k4": 0.0002}
        rng = np.random.default_rng(20260820)
        pixels = np.column_stack(
            [rng.uniform(350, 2550, 1000), rng.uniform(350, 2550, 1000)]
        )
        rays = unproject_kb4(pixels, intrinsic, distortion)
        projected, _ranges, valid = project_kb4(rays, intrinsic, distortion)
        errors = np.linalg.norm(projected - pixels, axis=1)

        self.assertTrue(valid.all())
        self.assertLess(float(errors.max()), 0.05)

    def test_points_beyond_configured_fov_are_rejected(self) -> None:
        intrinsic = {"fl_x": 800.0, "fl_y": 800.0, "cx": 800.0, "cy": 800.0}
        distortion = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0}
        points = np.array([[0, 0, 1], [1, 0, -1]], dtype=np.float64)
        _pixels, _ranges, valid = project_kb4(
            points, intrinsic, distortion, max_theta_rad=np.radians(100)
        )

        self.assertEqual(valid.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
