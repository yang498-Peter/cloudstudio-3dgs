import unittest

from cloudstudio_3dgs.training.mipmap_loss_schedule import (
    high_type2_loss_weights,
    high_type2_schedule_contract,
)


class MipMapLossScheduleTests(unittest.TestCase):
    def test_exact_epoch_boundaries(self) -> None:
        views = 10
        expected = {
            0: ("lidar_rgb_bootstrap", 0.0),
            5 * views: ("lidar_surface_growth", 0.0),
            10 * views: ("lidar_surface_growth", 0.04),
            15 * views: ("lidar_surface_polish", 0.04),
        }
        for step, values in expected.items():
            weights = high_type2_loss_weights(step, views)
            self.assertEqual(
                (
                    weights.stage,
                    weights.sky_opacity,
                ),
                values,
            )
            self.assertEqual((weights.rgb_l1, weights.rgb_dssim), (0.6, 0.4))
            self.assertEqual(weights.da2_depth, 0.0)
            self.assertEqual(weights.mesh_depth, 0.0)
            self.assertEqual(weights.mesh_normal, 0.0)
            self.assertEqual(weights.rendered_depth_normal_consistency, 0.0)
            self.assertEqual(weights.sparse_lidar_range, 0.05)
            self.assertEqual(weights.lidar_surface_normal, 0.01)
            self.assertEqual(weights.opacity_mean, 0.01)

    def test_schedule_is_fail_closed_and_tile_specific(self) -> None:
        contract = high_type2_schedule_contract(476)
        self.assertEqual(contract["total_steps"], 9520)
        self.assertEqual(contract["boundary_steps"], [0, 2380, 4760, 7140, 9520])
        self.assertFalse(contract["training_allowed"])
        self.assertEqual(
            contract["evidence_boundary"]["da2"],
            "DEFERRED_OPTIONAL_WEIGHT_ZERO",
        )
        with self.assertRaises(ValueError):
            high_type2_loss_weights(9520, 476)


if __name__ == "__main__":
    unittest.main()
