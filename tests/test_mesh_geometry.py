import copy
import unittest

import numpy as np

from cloudstudio_3dgs.data.mesh_geometry import (
    MESH_GEOMETRY_KIND,
    MESH_GEOMETRY_SCHEMA_VERSION,
    mesh_geometry_npz_bytes,
    sign_mesh_geometry_manifest,
    strict_mesh_admission_mask,
    verify_mesh_geometry_manifest,
)
from cloudstudio_3dgs.geometry.spatial_block_holdout import (
    build_spatial_block_holdout,
)


class MeshGeometryTests(unittest.TestCase):
    def test_npz_is_deterministic_and_invalid_pixels_are_zero(self) -> None:
        depth = np.array([[2.0, np.nan], [4.0, 5.0]], dtype=np.float32)
        normal = np.zeros((2, 2, 3), dtype=np.float32)
        normal[..., 2] = 2.0
        confidence = np.ones((2, 2), dtype=np.float32)
        valid = np.array([[True, True], [False, True]])
        first = mesh_geometry_npz_bytes(depth, normal, confidence, valid)
        second = mesh_geometry_npz_bytes(depth, normal, confidence, valid)
        self.assertEqual(first, second)
        import io

        with np.load(io.BytesIO(first), allow_pickle=False) as payload:
            np.testing.assert_array_equal(payload["valid"], [[1, 0], [0, 1]])
            self.assertEqual(float(payload["depth_range_m"][0, 1]), 0.0)
            np.testing.assert_allclose(payload["normal_camera"][0, 0], [0, 0, 1])

    def test_manifest_is_signed_and_fail_closed(self) -> None:
        manifest = sign_mesh_geometry_manifest(
            {
                "schema_version": MESH_GEOMETRY_SCHEMA_VERSION,
                "kind": MESH_GEOMETRY_KIND,
                "complete_face_cache": True,
                "records": [],
            }
        )
        self.assertEqual(verify_mesh_geometry_manifest(manifest), manifest["mesh_geometry_manifest_sha256"])
        damaged = copy.deepcopy(manifest)
        damaged["records"].append({"sample_id": "changed"})
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            verify_mesh_geometry_manifest(damaged)

    def test_npz_accepts_per_pixel_source_type(self) -> None:
        import io

        depth = np.ones((1, 2), dtype=np.float32)
        normal = np.zeros((1, 2, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        payload = mesh_geometry_npz_bytes(
            depth,
            normal,
            np.ones_like(depth),
            np.ones_like(depth, dtype=bool),
            source_type=np.asarray([[2, 3]], dtype=np.uint8),
        )
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            np.testing.assert_array_equal(loaded["source_type"], [[2, 3]])

    def test_strict_admission_rejects_unobservable_type_four(self) -> None:
        valid = np.asarray([[True, True, True, False]])
        source_type = np.asarray([[2, 3, 4, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(
            strict_mesh_admission_mask(valid, source_type),
            [[True, True, False, False]],
        )

    def test_spatial_holdout_keeps_whole_blocks_together(self) -> None:
        grid = np.stack(
            np.meshgrid(np.arange(12), np.arange(10), np.arange(2), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3).astype(np.float64)
        first = build_spatial_block_holdout(
            grid, block_size_m=2.0, holdout_fraction=0.2, seed=17
        )
        second = build_spatial_block_holdout(
            grid, block_size_m=2.0, holdout_fraction=0.2, seed=17
        )
        np.testing.assert_array_equal(first.holdout_mask, second.holdout_mask)
        for coordinate in np.unique(first.block_index, axis=0):
            members = np.all(first.block_index == coordinate, axis=1)
            self.assertIn(int(np.count_nonzero(first.holdout_mask[members])), (0, int(np.count_nonzero(members))))
        self.assertLess(abs(first.actual_fraction - 0.2), 0.05)


if __name__ == "__main__":
    unittest.main()
