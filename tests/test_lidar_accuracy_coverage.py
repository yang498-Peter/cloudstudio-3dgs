from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.evaluation.lidar_accuracy_coverage import compare_tile_to_source
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap


def _depth(indices, ranges, confidences):
    count = len(indices)
    return SparseDepthMap(
        (3, 4),
        np.asarray(indices, dtype=np.int32),
        np.asarray(ranges, dtype=np.float32),
        np.asarray(confidences, dtype=np.float32),
        np.full(count, -1, dtype=np.int64),
        np.zeros(count, dtype=np.int32),
    )


class LidarAccuracyCoverageTests(unittest.TestCase):
    def test_reports_accuracy_coverage_confidence_and_edges(self) -> None:
        source = _depth([0, 1, 2, 5, 6], [1, 1, 3, 2, 2], [0.2, 0.6, 0.9, 0.9, 0.9])
        tile = _depth([0, 2, 6], [1.0, 3.06, 2.0], [0.2, 0.9, 0.9])
        report = compare_tile_to_source(source, tile, crop_xywh=(0, 0, 4, 2))
        self.assertEqual(report["candidate_pixels"], 5)
        self.assertEqual(report["retained_pixels"], 3)
        self.assertAlmostEqual(report["coverage_fraction"], 0.6)
        self.assertEqual(report["confidence"]["medium"]["retained_pixels"], 0)
        self.assertEqual(int(np.count_nonzero(report["errors_m"] > 0.05)), 1)
        self.assertGreater(report["edge"]["candidate_pixels"], 0)

    def test_rejects_tile_pixels_outside_source_crop(self) -> None:
        source = _depth([0, 1], [1, 1], [1, 1])
        tile = _depth([2], [1], [1])
        with self.assertRaisesRegex(ValueError, "outside the source crop"):
            compare_tile_to_source(source, tile, crop_xywh=(0, 0, 4, 1))


if __name__ == "__main__":
    unittest.main()
