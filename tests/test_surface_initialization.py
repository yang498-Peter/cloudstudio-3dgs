from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.training.surface_initialization import (
    SurfaceInitializationConfig,
    build_surface_aligned_initialization,
    cap_surface_initialization_scales,
    load_precomputed_surfel_geometry,
)


def _rotation_columns(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quaternions.T
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                axis=-1,
            ),
            np.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                axis=-1,
            ),
            np.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


class SurfaceInitializationTests(unittest.TestCase):
    def test_signed_precomputed_mode_accepts_arbitrary_thin_scales(self) -> None:
        with self.subTest("configuration"):
            SurfaceInitializationConfig(
                enabled=True, mode="signed_precomputed_surfel"
            ).validate()

    def test_precomputed_surfel_loader_does_not_require_half_scale(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.npz"
            np.savez_compressed(
                path,
                normals=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                eigenvalues=np.array([[0.0, 1.0, 1.0]], dtype=np.float32),
                scales_m=np.array([[0.01, 0.01, 0.0015]], dtype=np.float32),
                quaternions_wxyz=np.array(
                    [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
                ),
            )
            loaded = load_precomputed_surfel_geometry(path, expected_count=1)
            np.testing.assert_allclose(loaded[2], [[0.01, 0.01, 0.0015]])

    def test_sparse_scale_tail_is_clamped_without_changing_small_axes(self) -> None:
        scales = np.array(
            [[0.01, 0.02, 0.005], [1.4, 0.04, 0.02]], dtype=np.float32
        )
        capped, report = cap_surface_initialization_scales(
            scales, maximum_scale_m=0.08
        )
        np.testing.assert_allclose(capped[0], scales[0])
        np.testing.assert_allclose(capped[1], [0.08, 0.04, 0.02])
        self.assertEqual(report["clamped_gaussian_count"], 1)
        self.assertEqual(report["clamped_axis_count"], 1)
        self.assertAlmostEqual(report["after_max_m"], 0.08)

    def test_planar_points_start_as_normal_aligned_thin_gaussians(self) -> None:
        scales = np.array([0.02, 0.04, 0.08], dtype=np.float32)
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        eigenvalues = np.array(
            [
                [1e-6, 1e-3, 2e-3],
                [1e-6, 1e-3, 2e-3],
                [1e-6, 1e-3, 2e-3],
            ],
            dtype=np.float32,
        )
        config = SurfaceInitializationConfig(
            enabled=True,
            planarity_gate=0.6,
            normal_scale_ratio=0.08,
            minimum_normal_scale_m=0.0005,
        )

        anisotropic, quaternions, report = build_surface_aligned_initialization(
            scales,
            normals,
            eigenvalues,
            config=config,
        )

        expected_short = np.maximum(scales * 0.08, 0.0005)
        np.testing.assert_allclose(
            anisotropic[:, :2], scales[:, None].repeat(2, axis=1), rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(anisotropic[:, 2], expected_short, rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6)
        thin_axes = _rotation_columns(quaternions)[:, :, 2]
        alignment = np.abs(np.sum(thin_axes * normals, axis=1))
        np.testing.assert_allclose(alignment, 1.0, atol=1e-6)
        self.assertEqual(report["surface_aligned_count"], 3)
        self.assertAlmostEqual(report["surface_aligned_fraction"], 1.0)

    def test_nonplanar_points_remain_isotropic(self) -> None:
        scales = np.array([0.02, 0.04], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        eigenvalues = np.array(
            [
                [1.0, 1.0, 1.0],
                [0.8, 1.0, 1.2],
            ],
            dtype=np.float32,
        )

        anisotropic, quaternions, report = build_surface_aligned_initialization(
            scales,
            normals,
            eigenvalues,
            config=SurfaceInitializationConfig(enabled=True, planarity_gate=0.6),
        )

        np.testing.assert_allclose(anisotropic, scales[:, None].repeat(3, axis=1))
        np.testing.assert_allclose(
            quaternions,
            np.array([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        )
        self.assertEqual(report["surface_aligned_count"], 0)

    def test_invalid_geometry_fails_closed(self) -> None:
        scales = np.array([0.02, 0.04], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0]], dtype=np.float32)
        eigenvalues = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            build_surface_aligned_initialization(
                scales,
                normals,
                eigenvalues,
                config=SurfaceInitializationConfig(enabled=True),
            )


if __name__ == "__main__":
    unittest.main()
