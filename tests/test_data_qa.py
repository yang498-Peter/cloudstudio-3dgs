from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from cloudstudio_3dgs.evaluation.data_qa import evaluate_gates, scan_las


class DataQaTests(unittest.TestCase):
    def test_las_statistics_include_bounds_rgb_and_deterministic_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.las"
            header = laspy.LasHeader(point_format=3, version="1.2")
            cloud = laspy.LasData(header)
            cloud.x = [0.0, 1.0, 2.0]
            cloud.y = [-1.0, 0.0, 1.0]
            cloud.z = [5.0, 6.0, 8.0]
            cloud.red = [0, 255, 65535]
            cloud.green = [0, 128, 65535]
            cloud.blue = [0, 64, 65535]
            cloud.write(path)

            metrics, points = scan_las(path, max_points=2)

        self.assertEqual(metrics["point_count"], 3)
        self.assertEqual(metrics["bounds_min"], [0.0, -1.0, 5.0])
        self.assertEqual(metrics["bounds_max"], [2.0, 1.0, 8.0])
        self.assertEqual(metrics["rgb"]["max"], [65535, 65535, 65535])
        self.assertAlmostEqual(metrics["rgb"]["black_fraction"], 1 / 3)
        self.assertEqual(len(points), 2)

    def test_threshold_evaluation_is_fail_closed(self) -> None:
        config = {
            "projection": {
                "minimum_visible_fraction_p50": 0.01,
                "maximum_edge_distance_p50_px": 8,
                "minimum_edge_frame_success_fraction": 1.0,
            },
            "timing": {"maximum_pair_delta_ms": 50, "maximum_frame_interval_ms": 1000},
            "trajectory": {"maximum_speed_mps": 15, "maximum_angular_speed_deg_s": 180},
            "rig": {
                "maximum_translation_error_p95_m": 0.01,
                "maximum_rotation_error_p95_rad": 0.01,
                "maximum_intrinsic_difference": 0.001,
            },
            "point_cloud": {"maximum_black_rgb_fraction": 0.9},
        }
        metrics = {
            "projection": {
                "visible_fraction": {"p50": 0.2},
                "edge_distance_px": {"p50": 2},
                "edge_frame_success_fraction": 1.0,
            },
            "timing": {"pair_delta_ms": {"max": 1}, "frame_interval_ms": {"max": 500}},
            "trajectory": {"speed_mps": {"max": 3}, "angular_speed_deg_s": {"max": 20}},
            "rig": {
                "relative_translation_error_m": {"p95": 0.001},
                "relative_rotation_error_rad": {"p95": 0.001},
                "calibration_vs_transforms": {"max_abs_difference": 0.0},
            },
            "point_cloud": {"rgb": {"black_fraction": 0.95}},
        }

        gates = evaluate_gates(metrics, config)

        self.assertEqual(len(gates), 11)
        self.assertEqual([gate["name"] for gate in gates if not gate["passed"]], ["point_cloud.black_rgb_fraction"])


if __name__ == "__main__":
    unittest.main()
