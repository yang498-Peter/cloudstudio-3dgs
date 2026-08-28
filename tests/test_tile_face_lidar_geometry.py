from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.data.tile_face_lidar_geometry import (
    filter_sparse_face_depth_to_world_box,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap


class TileFaceLidarGeometryTests(unittest.TestCase):
    def test_filter_requires_crop_and_world_box_membership(self) -> None:
        face = FaceSpec(
            face_id="front",
            R_face=np.eye(3),
            K_face=np.asarray([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
            width=5,
            height=5,
            half_fov_deg=45.0,
        )
        depth = SparseDepthMap(
            (5, 5),
            np.asarray([12, 13, 18], dtype=np.int32),
            np.asarray([2.0, 2.0, 8.0], dtype=np.float32),
            np.asarray([1.0, 0.8, 0.7], dtype=np.float32),
            np.asarray([3, 4, 5], dtype=np.int64),
            np.asarray([1, 1, 1], dtype=np.int32),
        )
        result = filter_sparse_face_depth_to_world_box(
            depth,
            face=face,
            c2w=np.eye(4),
            crop_xywh=(2, 2, 2, 2),
            world_box=np.asarray([[-0.1, -0.1, 1.0], [2.0, 2.0, 4.0]]),
        )
        np.testing.assert_array_equal(result.pixel_index, np.asarray([12, 13]))
        np.testing.assert_allclose(result.range_m, np.asarray([2.0, 2.0]))
        np.testing.assert_array_equal(result.source_index, np.asarray([-1, -1]))
        np.testing.assert_array_equal(result.support_count, np.asarray([0, 0]))

    def test_filter_applies_camera_to_world_transform(self) -> None:
        face = FaceSpec(
            face_id="front",
            R_face=np.eye(3),
            K_face=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            width=1,
            height=1,
            half_fov_deg=45.0,
        )
        depth = SparseDepthMap(
            (1, 1),
            np.asarray([0], dtype=np.int32),
            np.asarray([2.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([-1], dtype=np.int64),
            np.asarray([0], dtype=np.int32),
        )
        pose = np.eye(4)
        pose[:3, 3] = [10.0, 0.0, 0.0]
        result = filter_sparse_face_depth_to_world_box(
            depth,
            face=face,
            c2w=pose,
            crop_xywh=(0, 0, 1, 1),
            world_box=np.asarray([[9.5, -0.5, 1.5], [10.5, 0.5, 2.5]]),
        )
        self.assertEqual(len(result.pixel_index), 1)


if __name__ == "__main__":
    unittest.main()
