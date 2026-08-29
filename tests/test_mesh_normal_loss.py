from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from cloudstudio_3dgs.training.trainer import _rendered_range_normals_camera


class MeshNormalLossTests(unittest.TestCase):
    def test_constant_z_plane_produces_camera_z_normal(self) -> None:
        sample = SimpleNamespace(
            K=np.asarray(
                [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                np.float32,
            )
        )
        yy, xx = torch.meshgrid(
            torch.arange(5.0), torch.arange(5.0), indexing="ij"
        )
        ray_norm = torch.sqrt(
            ((xx + 0.5 - 2.0) / 100.0) ** 2
            + ((yy + 0.5 - 2.0) / 100.0) ** 2
            + 1.0
        )
        ray_range = 3.0 * ray_norm
        normals, valid = _rendered_range_normals_camera(torch, ray_range, sample)
        self.assertTrue(bool(valid[2, 2]))
        self.assertGreater(float(normals[2, 2, 2].abs()), 0.999)


if __name__ == "__main__":
    unittest.main()
