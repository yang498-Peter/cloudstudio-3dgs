"""The battery's validation caches are derived by name (cloudstudio_3dgs/training/validation_paths.py).

The first SDK delivery of house0305 trained for seven hours and then died in the battery:
the evaluator turned the v9 training cache ``face4_train`` into ``face4_val_train`` with a
naive ``face4 -> face4_val`` replace. The rule now lives in one place and both spellings of
a training cache map to ``face4_val``.
"""

from __future__ import annotations

import unittest

from cloudstudio_3dgs.training.validation_paths import (
    derive_validation_paths,
    validation_face_path,
    validation_train_path,
)


class FacePathTests(unittest.TestCase):
    def test_v8_bare_face4_maps_to_face4_val(self) -> None:
        self.assertEqual(
            validation_face_path(r"C:\d\house0305_sop_v8\face4\face_manifest.json"),
            r"C:\d\house0305_sop_v8\face4_val\face_manifest.json",
        )

    def test_v9_face4_train_maps_to_face4_val_not_face4_val_train(self) -> None:
        self.assertEqual(
            validation_face_path(r"C:\d\house0305_sop_v9\face4_train\face_manifest.json"),
            r"C:\d\house0305_sop_v9\face4_val\face_manifest.json",
        )

    def test_an_already_validation_path_is_left_alone(self) -> None:
        path = r"C:\d\house0305_sop_v9\face4_val\face_manifest.json"
        self.assertEqual(validation_face_path(path), path)

    def test_train_suffix_maps_to_val(self) -> None:
        self.assertEqual(
            validation_train_path(r"C:\d\v9\face4_lidar_train_vis6\m.json"),
            r"C:\d\v9\face4_lidar_val_vis6\m.json",
        )


class DeriveTests(unittest.TestCase):
    def test_every_battery_key_is_derived_and_absent_keys_stay_absent(self) -> None:
        raw = {
            "face_cache_manifest": r"C:\d\v9\face4_train\face_manifest.json",
            "face_cache_root": r"C:\d\v9\face4_train",
            "renderer_mask_manifest": r"C:\d\v9\renderer_mask_train.json",
            "face_lidar_geometry_manifest": r"C:\d\v9\face4_lidar_train_vis6\face_lidar_geometry_manifest.json",
            "face_lidar_geometry_root": r"C:\d\v9\face4_lidar_train_vis6",
            "background_image_manifest": r"C:\r\tile_backgrounds_B1\Tile_0\background_manifest.json",
            "dataset_manifest": r"C:\d\v8\dataset_manifest.json",
        }
        out = derive_validation_paths(raw)
        self.assertEqual(out["face_cache_manifest"], r"C:\d\v9\face4_val\face_manifest.json")
        self.assertEqual(out["face_cache_root"], r"C:\d\v9\face4_val")
        self.assertEqual(out["renderer_mask_manifest"], r"C:\d\v9\renderer_mask_val.json")
        self.assertEqual(out["face_lidar_geometry_root"], r"C:\d\v9\face4_lidar_val_vis6")
        # a stand-in backdrop library has no _train in its name: the training artefact is used
        self.assertEqual(out["background_image_manifest"], raw["background_image_manifest"])
        self.assertNotIn("background_image_root", out, "not in the config, so not derived")
        self.assertNotIn("dataset_manifest", out, "the dataset manifest is shared, never derived")


if __name__ == "__main__":
    unittest.main()
