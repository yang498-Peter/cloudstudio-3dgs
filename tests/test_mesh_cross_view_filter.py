import unittest

import numpy as np

from cloudstudio_3dgs.geometry.mesh_cross_view_filter import (
    SOURCE_CROSS_VIEW_SUPPORTED,
    SOURCE_LIDAR_ANCHOR,
    SOURCE_UNOBSERVABLE_RETAINED,
    CrossViewFilterConfig,
    classify_cross_view_support,
)


class MeshCrossViewFilterTests(unittest.TestCase):
    def test_policy_keeps_anchors_and_rejects_only_repeated_conflict(self) -> None:
        valid, confidence, source_type = classify_cross_view_support(
            source_valid=np.ones(5, dtype=bool),
            lidar_anchor=np.asarray([True, False, False, False, False]),
            observed=np.asarray([3, 3, 2, 0, 1]),
            consistent=np.asarray([0, 0, 1, 0, 1]),
            conflicts=np.asarray([3, 3, 1, 0, 0]),
            config=CrossViewFilterConfig(),
        )
        np.testing.assert_array_equal(valid, [True, False, True, True, True])
        self.assertEqual(int(source_type[0]), SOURCE_LIDAR_ANCHOR)
        self.assertEqual(int(source_type[2]), SOURCE_CROSS_VIEW_SUPPORTED)
        self.assertEqual(int(source_type[3]), SOURCE_UNOBSERVABLE_RETAINED)
        self.assertGreater(float(confidence[2]), float(confidence[3]))
        self.assertEqual(float(confidence[0]), 1.0)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            classify_cross_view_support(
                source_valid=np.ones(2, dtype=bool),
                lidar_anchor=np.ones(1, dtype=bool),
                observed=np.ones(2, dtype=np.int16),
                consistent=np.ones(2, dtype=np.int16),
                conflicts=np.zeros(2, dtype=np.int16),
                config=CrossViewFilterConfig(),
            )
