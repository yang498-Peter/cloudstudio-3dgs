"""Spatial hold-out must withhold whole images by camera cell, hit the
requested fraction, and be reproducible from its inputs."""

from __future__ import annotations

import unittest

from cloudstudio_3dgs.training.holdout import select_spatial_holdout


def _views(n_images: int, faces=("yaw_minus_35", "yaw_plus_35", "pitch_up_56")):
    views = []
    positions = {}
    for i in range(n_images):
        image_id = f"img_{i:04d}"
        # Cameras on a line, one per 0.5 m; 2 m cells hold four images each.
        positions[image_id] = [0.5 * i, 0.0, 1.5]
        for face in faces:
            views.append({"sample_id": f"{image_id}::{face}", "x": 0, "y": 0, "width": 8, "height": 8})
    return views, positions


class SpatialHoldoutTests(unittest.TestCase):
    def test_whole_images_are_withheld_and_fraction_is_met(self) -> None:
        views, positions = _views(40)
        record = select_spatial_holdout(views, positions, cell_m=2.0, fraction=0.1, seed=0)
        held = set(record["held_out_sample_ids"])
        self.assertGreaterEqual(len(held), 12)  # 10% of 120 views
        self.assertEqual(record["training_view_count"], 120 - len(held))
        for image_id in record["held_out_image_ids"]:
            for face in ("yaw_minus_35", "yaw_plus_35", "pitch_up_56"):
                self.assertIn(f"{image_id}::{face}", held)
        # Cells are withheld whole: four consecutive images per cell.
        self.assertEqual(len(record["held_out_image_ids"]) % 4, 0)
        self.assertEqual(record["held_out_cell_count"], len(record["held_out_image_ids"]) // 4)

    def test_guard_band_withholds_neighbours_and_reports_distances(self) -> None:
        views, positions = _views(40)
        plain = select_spatial_holdout(views, positions, cell_m=2.0, fraction=0.1, seed=0)
        guarded = select_spatial_holdout(views, positions, cell_m=2.0, fraction=0.1, seed=0, guard_m=1.0)
        self.assertEqual(plain["held_out_sample_ids"], guarded["held_out_sample_ids"])
        self.assertEqual(plain["guard_sample_ids"], [])
        self.assertGreater(len(guarded["guard_sample_ids"]), 0)
        # Cameras 0.5 m apart on a line: without a guard the nearest training
        # camera is 0.5 m away; with a 1 m guard it is at least 1 m away.
        self.assertAlmostEqual(plain["nearest_training_camera_m"]["min"], 0.5, places=6)
        self.assertGreaterEqual(guarded["nearest_training_camera_m"]["min"], 1.0)
        self.assertEqual(
            guarded["training_view_count"],
            120 - len(guarded["held_out_sample_ids"]) - len(guarded["guard_sample_ids"]),
        )
        self.assertAlmostEqual(guarded["actual_held_out_fraction"], len(guarded["held_out_sample_ids"]) / 120)

    def test_selection_is_deterministic_and_seed_sensitive(self) -> None:
        views, positions = _views(40)
        a = select_spatial_holdout(views, positions, cell_m=2.0, fraction=0.1, seed=0)
        b = select_spatial_holdout(views, positions, cell_m=2.0, fraction=0.1, seed=0)
        c = select_spatial_holdout(views, positions, cell_m=2.0, fraction=0.1, seed=1)
        self.assertEqual(a["held_out_sample_ids"], b["held_out_sample_ids"])
        self.assertNotEqual(a["held_out_sample_ids"], c["held_out_sample_ids"])

    def test_invalid_arguments_are_rejected(self) -> None:
        views, positions = _views(4)
        with self.assertRaises(ValueError):
            select_spatial_holdout(views, positions, cell_m=0.0, fraction=0.1, seed=0)
        with self.assertRaises(ValueError):
            select_spatial_holdout(views, positions, cell_m=2.0, fraction=1.0, seed=0)
        with self.assertRaises(ValueError):
            select_spatial_holdout(views, {}, cell_m=2.0, fraction=0.1, seed=0)


if __name__ == "__main__":
    unittest.main()
