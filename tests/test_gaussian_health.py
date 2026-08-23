"""Synthetic CPU tests for tools/gaussian_health.py.

Fixtures plant known pathologies (giant, needle, floaters, wall gaussians on a
z=0 LiDAR plane) and assert every metric hits exactly.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.gaussian_health import (
    SCHEMA_VERSION,
    compute_health,
    fit_planes_ransac,
    read_ply_xyz,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _grid_cloud(step: float = 0.1, extent: float = 4.0) -> np.ndarray:
    axis = np.arange(0.0, extent + step / 2, step)
    grid_x, grid_y = np.meshgrid(axis, axis)
    return np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1
    )


def _params(means, scales_m, quats, opacities_prob):
    means = np.asarray(means, dtype=np.float64)
    scales_m = np.asarray(scales_m, dtype=np.float64)
    quats = np.asarray(quats, dtype=np.float64)
    opacities = np.array([_logit(p) for p in opacities_prob], dtype=np.float64)
    return {
        "means": means,
        "scales": np.log(scales_m),
        "quats": quats,
        "opacities": opacities,
    }


class TestScaleOpacityFloater(unittest.TestCase):
    """One fixture with 1 giant, 1 needle, 1 degenerate, 5 floaters, opacity buckets."""

    @classmethod
    def setUpClass(cls):
        cls.lidar = _grid_cloud()
        identity = [1.0, 0.0, 0.0, 0.0]
        means, scales, quats, opac = [], [], [], []

        # 100 normal gaussians sitting exactly on grid points (NN distance 0).
        for i in range(100):
            means.append([(i % 10) * 0.1, (i // 10) * 0.1, 0.0])
            scales.append([0.05, 0.05, 0.05])
            quats.append(identity)
            opac.append(0.9)
        cls.normal_count = 100

        # 1 giant: max axis 25 m (> all four giant thresholds), ratio 25/15 < 10.
        means.append([2.0, 2.0, 0.0])
        scales.append([25.0, 20.0, 15.0])
        quats.append(identity)
        opac.append(0.8)

        # 1 needle: 0.4 / 0.002 = 200 (> 10/30/100), max axis < 0.5 (not a giant).
        means.append([2.0, 2.1, 0.0])
        scales.append([0.4, 0.4, 0.002])
        quats.append(identity)
        opac.append(0.8)

        # 1 degenerate: isotropic 1e-5 (ratio 1, so it never pollutes needle counts).
        means.append([2.0, 2.2, 0.0])
        scales.append([1e-5, 1e-5, 1e-5])
        quats.append(identity)
        opac.append(0.8)

        # 5 floaters: 2 m above the plane, opacity 0.8.
        for i in range(5):
            means.append([1.0 + i * 0.5, 1.0, 2.0])
            scales.append([0.1, 0.1, 0.1])
            quats.append(identity)
            opac.append(0.8)

        # Opacity buckets: 3 dead, 2 fog, 4 hard (hard ones on grid points).
        for i in range(3):
            means.append([3.0, 0.1 * i, 0.0])
            scales.append([0.05, 0.05, 0.05])
            quats.append(identity)
            opac.append(0.001)
        for i in range(2):
            means.append([3.1, 0.1 * i, 0.0])
            scales.append([0.05, 0.05, 0.05])
            quats.append(identity)
            opac.append(0.05)
        for i in range(4):
            means.append([3.2, 0.1 * i, 0.0])
            scales.append([0.05, 0.05, 0.05])
            quats.append(identity)
            opac.append(0.99)

        cls.total = len(means)
        cls.report = compute_health(
            _params(means, scales, quats, opac), cls.lidar, max_planes=1, seed=42
        )

    def test_schema_and_thresholds_echoed(self):
        self.assertEqual(self.report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(self.report["gaussian_count"], self.total)
        thresholds = self.report["thresholds"]
        self.assertEqual(thresholds["giant_max_axis_thresholds_m"], [0.5, 1.0, 5.0, 20.0])
        self.assertEqual(thresholds["needle_ratio_thresholds"], [10.0, 30.0, 100.0])
        self.assertEqual(thresholds["plane_inlier_threshold_m"], 0.03)

    def test_giant_counts(self):
        giant = self.report["scale"]["giant"]
        for key in ("gt_0.5m", "gt_1.0m", "gt_5.0m", "gt_20.0m"):
            self.assertEqual(giant[key]["count"], 1, key)
            self.assertAlmostEqual(giant[key]["opacity_median"], 0.8, places=6)
        self.assertAlmostEqual(self.report["scale"]["max_axis_m"]["max"], 25.0, places=6)

    def test_needle_counts(self):
        needle = self.report["scale"]["needle"]
        self.assertEqual(needle["ratio_gt_10"], 1)
        self.assertEqual(needle["ratio_gt_30"], 1)
        self.assertEqual(needle["ratio_gt_100"], 1)

    def test_degenerate_count(self):
        self.assertEqual(self.report["scale"]["degenerate_min_axis_lt_1e4_count"], 1)

    def test_opacity_buckets(self):
        op = self.report["opacity"]
        self.assertEqual(op["dead_lt_0_005"]["count"], 3)
        self.assertEqual(op["fog_0_005_to_0_1"]["count"], 2)
        self.assertEqual(op["hard_gt_0_95"]["count"], 4)
        self.assertAlmostEqual(
            op["dead_lt_0_005"]["fraction"], 3 / self.total, places=9
        )

    def test_floater_outliers(self):
        floater = self.report["floater"]
        # visible = everything except 3 dead + 2 fog (both < 0.1)
        self.assertEqual(floater["visible_count"], self.total - 5)
        outliers = floater["outliers"]
        self.assertEqual(outliers["gt_0.3m"]["count"], 5)
        self.assertEqual(outliers["gt_1.0m"]["count"], 5)
        self.assertEqual(outliers["gt_5.0m"]["count"], 0)
        self.assertAlmostEqual(outliers["gt_1.0m"]["mean_opacity"], 0.8, places=6)
        self.assertAlmostEqual(outliers["gt_1.0m"]["mean_max_axis_m"], 0.1, places=6)
        self.assertIsNone(outliers["gt_5.0m"]["mean_opacity"])
        # Non-floaters sit exactly on grid points, so p50 distance is 0.
        self.assertAlmostEqual(floater["distance_m"]["p50"], 0.0, places=9)


class TestWallMetrics(unittest.TestCase):
    lidar = None

    @classmethod
    def setUpClass(cls):
        cls.lidar = _grid_cloud()

    def test_ransac_finds_z0_plane(self):
        rng = np.random.default_rng(7)
        planes = fit_planes_ransac(self.lidar, max_planes=3, rng=rng)
        self.assertEqual(len(planes), 1)  # a single plane absorbs the whole cloud
        normal = planes[0]["normal"]
        angle = math.degrees(math.acos(min(abs(float(normal[2])), 1.0)))
        self.assertLess(angle, 5.0)
        self.assertEqual(planes[0]["inlier_count"], len(self.lidar))
        self.assertAlmostEqual(abs(planes[0]["offset"]), 0.0, places=6)

    def test_flat_wall_thickness_2cm(self):
        identity = [1.0, 0.0, 0.0, 0.0]
        means, scales, quats, opac = [], [], [], []
        for i in range(40):
            z_offset = 0.02 if i % 2 == 0 else -0.02  # RMS is exactly 0.02
            means.append([0.5 + (i % 8) * 0.4, 0.5 + (i // 8) * 0.6, z_offset])
            scales.append([0.05, 0.05, 0.005])  # shortest axis along z
            quats.append(identity)
            opac.append(0.9)
        report = compute_health(
            _params(means, scales, quats, opac), self.lidar, max_planes=1, seed=42
        )
        wall = report["wall"]
        self.assertEqual(wall["plane_count"], 1)
        plane = wall["planes"][0]
        self.assertEqual(plane["gaussian_count"], 40)
        self.assertAlmostEqual(plane["center_normal_rms_m"], 0.02, places=4)
        # sqrt(0.02^2 + 0.005^2) — deviation and normal-projected sigma combined
        expected_spread = math.sqrt(0.02**2 + 0.005**2)
        self.assertAlmostEqual(
            plane["effective_thickness_m"]["p50"], expected_spread, places=4
        )
        self.assertAlmostEqual(
            plane["effective_thickness_m"]["p95"], expected_spread, places=4
        )
        # Shortest axis parallel to plane normal: perfectly aligned.
        self.assertLess(plane["shortest_axis_angle_deg_p50"], 1.0)
        weighted = wall["weighted_by_lidar_inliers"]
        self.assertAlmostEqual(weighted["center_normal_rms_m"], 0.02, places=4)

    def test_tilted_wall_alignment_30deg(self):
        half = math.radians(30.0) / 2.0
        tilted = [math.cos(half), math.sin(half), 0.0, 0.0]  # 30 deg about x, wxyz
        means, scales, quats, opac = [], [], [], []
        for i in range(40):
            means.append([0.5 + (i % 8) * 0.4, 0.5 + (i // 8) * 0.6, 0.0])
            scales.append([0.05, 0.05, 0.005])
            quats.append(tilted)
            opac.append(0.9)
        report = compute_health(
            _params(means, scales, quats, opac), self.lidar, max_planes=1, seed=42
        )
        plane = report["wall"]["planes"][0]
        self.assertEqual(plane["gaussian_count"], 40)
        self.assertAlmostEqual(plane["shortest_axis_angle_deg_p50"], 30.0, delta=1.0)
        # Centers exactly on the plane: deviation RMS is 0, spread is pure sigma.
        self.assertAlmostEqual(plane["center_normal_rms_m"], 0.0, places=6)
        expected_sigma = math.sqrt(
            (0.05 * math.sin(math.radians(30))) ** 2
            + (0.005 * math.cos(math.radians(30))) ** 2
        )
        self.assertAlmostEqual(
            plane["effective_thickness_m"]["p50"], expected_sigma, places=4
        )


class TestUnnormalizedQuats(unittest.TestCase):
    def test_quat_normalization_is_applied(self):
        lidar = _grid_cloud()
        # Same 30-degree tilt but with quats scaled by 3 (checkpoint quats are raw).
        half = math.radians(30.0) / 2.0
        raw = [3.0 * math.cos(half), 3.0 * math.sin(half), 0.0, 0.0]
        means = [[1.0 + 0.2 * i, 1.0, 0.0] for i in range(10)]
        scales = [[0.05, 0.05, 0.005]] * 10
        report = compute_health(
            _params(means, scales, [raw] * 10, [0.9] * 10), lidar, max_planes=1, seed=42
        )
        plane = report["wall"]["planes"][0]
        self.assertAlmostEqual(plane["shortest_axis_angle_deg_p50"], 30.0, delta=1.0)


class TestCliSmoke(unittest.TestCase):
    def test_cli_roundtrip(self):
        import torch
        from cloudstudio_3dgs.data.point_cloud import write_binary_ply

        lidar = _grid_cloud(step=0.2, extent=2.0)
        identity = [1.0, 0.0, 0.0, 0.0]
        means = [[0.2 * (i % 5), 0.2 * (i // 5), 0.0] for i in range(25)]
        scales = [[0.05, 0.05, 0.01]] * 25
        params = _params(means, scales, [identity] * 25, [0.9] * 25)
        tensor_params = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in params.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_path = tmp_path / "ckpt.pt"
            ply_path = tmp_path / "lidar.ply"
            json_path = tmp_path / "health.json"
            torch.save({"params": tensor_params}, ckpt_path)
            write_binary_ply(
                ply_path,
                lidar.astype(np.float32),
                np.zeros_like(lidar, dtype=np.uint8),
            )
            # Reader round-trip against the canonical writer.
            self.assertTrue(np.allclose(read_ply_xyz(ply_path), lidar, atol=1e-6))
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "gaussian_health.py"),
                    "--checkpoint", str(ckpt_path),
                    "--lidar-ply", str(ply_path),
                    "--planes", "2",
                    "--output", str(json_path),
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Gaussian Health Report", result.stdout)
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(report["gaussian_count"], 25)
            self.assertEqual(report["lidar_point_count"], len(lidar))
            self.assertIn("thresholds", report)
            self.assertEqual(report["wall"]["plane_count"], 1)


if __name__ == "__main__":
    unittest.main()
