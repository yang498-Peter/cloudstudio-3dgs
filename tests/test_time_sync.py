from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cloudstudio_3dgs.evaluation.time_sync import PoseTrajectory


class PoseTrajectoryTests(unittest.TestCase):
    def test_interpolates_translation_and_rotation(self) -> None:
        rotations = Rotation.from_euler("z", [[0.0], [90.0]], degrees=True)
        trajectory = PoseTrajectory(
            timestamps_ns=np.asarray([1_000_000_000, 2_000_000_000], dtype=np.int64),
            translations=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            rotations=rotations,
        )
        pose = trajectory.interpolate(1_250_000_000, offset_ms=250.0)
        np.testing.assert_allclose(pose[:3, 3], [1.0, 0.0, 0.0], atol=1e-12)
        angle = Rotation.from_matrix(pose[:3, :3]).as_euler("zxy", degrees=True)[0]
        self.assertAlmostEqual(angle, 45.0, places=9)

    def test_rejects_extrapolation(self) -> None:
        trajectory = PoseTrajectory(
            timestamps_ns=np.asarray([0, 1_000_000], dtype=np.int64),
            translations=np.zeros((2, 3)),
            rotations=Rotation.identity(2),
        )
        with self.assertRaisesRegex(ValueError, "outside trajectory"):
            trajectory.interpolate(0, offset_ms=-1.0)


if __name__ == "__main__":
    unittest.main()
