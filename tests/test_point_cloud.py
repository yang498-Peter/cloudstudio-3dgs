from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cloudstudio_3dgs.data.point_cloud import (
    VoxelInitializationConfig,
    build_lidar_initialization,
    estimate_local_geometry,
    voxel_downsample_arrays,
)


def write_las(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.write(path)


class PointCloudInitializationTests(unittest.TestCase):
    def test_checked_in_configuration_and_baseline_respect_budget(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "lidar_init_8gb.json").read_text(encoding="utf-8")
        )
        baseline = json.loads(
            (root / "baselines" / "gs2_lidar_init.baseline.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(config["target_points"], 400_000)
        self.assertLess(config["target_points"], config["cap_max"])
        self.assertLessEqual(baseline["output"]["point_count"], config["target_points"])
        self.assertLess(baseline["output"]["point_count"], config["cap_max"])

    def test_array_voxelization_is_exactly_deterministic(self) -> None:
        rng = np.random.default_rng(123)
        xyz = rng.uniform(-2, 2, size=(2_000, 3))
        rgb = rng.integers(0, 256, size=(2_000, 3), dtype=np.uint8)

        first = voxel_downsample_arrays(
            xyz, rgb, voxel_size=0.25, edge_preservation_ratio=0.2, seed=7
        )
        second = voxel_downsample_arrays(
            xyz, rgb, voxel_size=0.25, edge_preservation_ratio=0.2, seed=7
        )

        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)
        keys = [tuple(int(value) for value in row) for row in first[2]]
        self.assertEqual(keys, sorted(keys))

    def test_budget_guard_rejects_target_equal_to_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be smaller than cap_max"):
            VoxelInitializationConfig(target_points=10, cap_max=10).validate()

    def test_detects_uint8_values_stored_in_uint16_las_fields(self) -> None:
        xyz = np.column_stack(
            [np.arange(30, dtype=float), np.zeros(30), np.zeros(30)]
        )
        rgb = np.tile(np.array([[12, 128, 250]], dtype=np.uint16), (30, 1))
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_las(run / "colorized.las", xyz, rgb)
            result = build_lidar_initialization(
                run,
                VoxelInitializationConfig(
                    target_points=31, cap_max=40, voxel_size=0.5, chunk_size=7
                ),
            )

        self.assertEqual(result.report["source"]["rgb_mode"], "uint8_in_uint16")
        self.assertTrue(np.all(result.rgb == np.array([12, 128, 250], dtype=np.uint8)))

    def test_scales_true_uint16_las_rgb_by_257(self) -> None:
        xyz = np.column_stack(
            [np.arange(20, dtype=float), np.zeros(20), np.zeros(20)]
        )
        rgb = np.tile(np.array([[0, 32_896, 65_535]], dtype=np.uint16), (20, 1))
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_las(run / "colorized.las", xyz, rgb)
            result = build_lidar_initialization(
                run,
                VoxelInitializationConfig(
                    target_points=21, cap_max=30, voxel_size=0.5, chunk_size=6
                ),
            )

        self.assertEqual(result.report["source"]["rgb_mode"], "uint16_scaled")
        self.assertTrue(np.all(result.rgb == np.array([0, 128, 255], dtype=np.uint8)))

    def test_auto_voxel_output_is_repeatable_and_below_budgets(self) -> None:
        rng = np.random.default_rng(99)
        xyz = rng.normal(size=(4_000, 3))
        rgb = rng.integers(0, 65_536, size=(4_000, 3), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_las(run / "scene_colorized.las", xyz, rgb)
            config = VoxelInitializationConfig(
                target_points=300,
                cap_max=350,
                voxel_size="auto",
                seed=42,
                chunk_size=509,
            )
            first = build_lidar_initialization(run, config)
            second = build_lidar_initialization(
                run,
                VoxelInitializationConfig(
                    target_points=300,
                    cap_max=350,
                    voxel_size="auto",
                    seed=42,
                    chunk_size=773,
                ),
            )

        np.testing.assert_array_equal(first.xyz, second.xyz)
        np.testing.assert_array_equal(first.rgb, second.rgb)
        self.assertEqual(first.report["output"], second.report["output"])
        self.assertEqual(first.report["coverage"], second.report["coverage"])
        self.assertEqual(
            first.report["auto_tuning_passes"], second.report["auto_tuning_passes"]
        )
        self.assertLessEqual(len(first.xyz), config.target_points)
        self.assertLess(len(first.xyz), config.cap_max)

    def test_voxel_coverage_beats_stride_for_density_ordered_cloud(self) -> None:
        dense = []
        for cell in range(10):
            for sample in range(100):
                dense.append([cell + sample / 1_000.0, 0.0, 0.0])
        sparse = [[cell + 0.1, 0.0, 0.0] for cell in range(10, 100)]
        xyz = np.asarray(dense + sparse, dtype=np.float64)
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint16)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_las(run / "colorized.las", xyz, rgb)
            result = build_lidar_initialization(
                run,
                VoxelInitializationConfig(
                    target_points=101, cap_max=120, voxel_size=1.0, chunk_size=137
                ),
            )

        coverage = result.report["coverage"]
        self.assertEqual(len(result.xyz), 100)
        self.assertGreater(coverage["voxel_coverage_ratio"], coverage["stride_coverage_ratio"])
        self.assertGreater(coverage["coverage_gain"], 0.5)

    def test_optional_pca_recovers_plane_normal(self) -> None:
        grid = np.linspace(-1.0, 1.0, 12)
        xx, yy = np.meshgrid(grid, grid)
        xyz = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        normals, eigenvalues, covariance = estimate_local_geometry(xyz, neighbors=12)

        self.assertEqual(normals.shape, xyz.shape)
        self.assertEqual(eigenvalues.shape, xyz.shape)
        self.assertEqual(covariance.shape, (len(xyz), 3, 3))
        np.testing.assert_allclose(np.abs(normals[:, 2]), 1.0, atol=1e-6)
        np.testing.assert_allclose(eigenvalues[:, 0], 0.0, atol=1e-9)
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            estimate_local_geometry(xyz, batch_size=0)


if __name__ == "__main__":
    unittest.main()
