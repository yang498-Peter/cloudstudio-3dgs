from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.image_sample import (
    CropWindow,
    load_image_sample,
    prepare_image_sample,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import build_per_image_masks, verify_mask_manifest
from cloudstudio_3dgs.evaluation.image_metrics import masked_psnr


def signed_mask_fixture_manifest() -> dict:
    manifest = {
        "schema_version": 1,
        "cameras": [
            {
                "camera_id": "left",
                "camera_type": "fisheye",
                "width": 8,
                "height": 8,
                "intrinsic": {"fl_x": 2.0, "fl_y": 2.0, "cx": 3.5, "cy": 3.5},
                "distortion": {
                    "camera_model": "OPENCV_FISHEYE",
                    "params": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
                },
            }
        ],
        "images": [
            {
                "image_id": "img_left_001",
                "camera_id": "left",
                "path_root": "recording",
                "path": "camera/left/001.jpg",
            },
            {
                "image_id": "img_left_002",
                "camera_id": "left",
                "path_root": "recording",
                "path": "camera/left/002.jpg",
            },
        ],
    }
    import hashlib

    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


class ImageSampleTests(unittest.TestCase):
    def test_checked_in_real_mask_baseline_preserves_open_gates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads(
            (root / "baselines" / "gs2_masks.baseline.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(baseline["output"]["image_count"], 1_238)
        self.assertTrue(baseline["output"]["per_image_paths_unique"])
        self.assertEqual(baseline["acceptance"]["real_dynamic_region_replay"], "not_run")
        self.assertEqual(
            baseline["acceptance"]["trainer_consumption"],
            "contract_pass_real_cuda_not_run",
        )

    def test_same_camera_images_load_their_own_static_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((4, 4, 3), 127, dtype=np.uint8)
            valid = np.full((4, 4), 255, dtype=np.uint8)
            first_static = valid.copy()
            second_static = valid.copy()
            first_static[0, 0] = 0
            second_static[3, 3] = 0
            Image.fromarray(image).save(root / "first.jpg")
            Image.fromarray(image).save(root / "second.jpg")
            Image.fromarray(valid).save(root / "valid.png")
            Image.fromarray(first_static).save(root / "first_static.png")
            Image.fromarray(second_static).save(root / "second_static.png")

            first = load_image_sample(
                root / "first.jpg",
                root / "valid.png",
                static_mask_path=root / "first_static.png",
            )
            second = load_image_sample(
                root / "second.jpg",
                root / "valid.png",
                static_mask_path=root / "second_static.png",
            )

        self.assertFalse(first.mask[0, 0])
        self.assertTrue(first.mask[3, 3])
        self.assertTrue(second.mask[0, 0])
        self.assertFalse(second.mask[3, 3])
        self.assertFalse(np.array_equal(first.mask, second.mask))

    def test_factor_one_two_four_keep_all_modalities_aligned(self) -> None:
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        valid = np.ones((8, 8), dtype=bool)
        static = np.ones((8, 8), dtype=bool)
        static[::2, 1::2] = False
        depth = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        confidence = np.ones((8, 8), dtype=np.float32)

        for factor in (1, 2, 4):
            with self.subTest(factor=factor):
                sample = prepare_image_sample(
                    image,
                    valid,
                    static_mask=static,
                    depth=depth,
                    confidence=confidence,
                    factor=factor,
                )
                expected = (8 // factor, 8 // factor)
                self.assertEqual(sample.image.shape[:2], expected)
                self.assertEqual(sample.mask.shape, expected)
                self.assertEqual(sample.depth.shape, expected)
                self.assertEqual(sample.confidence.shape, expected)
                self.assertEqual(sample.depth_valid_mask.shape, expected)
        with self.assertRaisesRegex(ValueError, "divisible by factor"):
            prepare_image_sample(
                image,
                valid,
                factor=2,
                crop=CropWindow(x=0, y=0, width=7, height=8),
            )
        with self.assertRaisesRegex(ValueError, "does not match image shape"):
            prepare_image_sample(image, valid, static_mask=np.ones((7, 8), dtype=bool))

    def test_crop_is_pixel_exact_for_image_mask_depth_and_confidence(self) -> None:
        image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        valid = np.ones((6, 8), dtype=bool)
        static = (np.arange(48).reshape(6, 8) % 3) != 0
        depth = np.arange(48, dtype=np.float32).reshape(6, 8) + 1
        confidence = np.arange(48, dtype=np.float32).reshape(6, 8) / 48 + 0.1
        depth_valid = (np.arange(48).reshape(6, 8) % 4) != 0
        crop = CropWindow(x=2, y=1, width=4, height=4)

        sample = prepare_image_sample(
            image,
            valid,
            static_mask=static,
            depth_valid_mask=depth_valid,
            depth=depth,
            confidence=confidence,
            crop=crop,
        )
        rows, columns = slice(1, 5), slice(2, 6)
        np.testing.assert_array_equal(sample.image, image[rows, columns])
        np.testing.assert_array_equal(sample.static_mask, static[rows, columns])
        np.testing.assert_array_equal(sample.depth, depth[rows, columns])
        np.testing.assert_array_equal(sample.confidence, confidence[rows, columns])
        np.testing.assert_array_equal(
            sample.mask,
            valid[rows, columns] & static[rows, columns] & depth_valid[rows, columns],
        )

    def test_invalid_depth_and_confidence_are_composed_into_mask(self) -> None:
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        valid = np.ones((2, 3), dtype=bool)
        depth = np.array([[1.0, 0.0, np.nan], [2.0, 3.0, 4.0]], dtype=np.float32)
        confidence = np.array([[1.0, 1.0, 1.0], [0.0, np.nan, 0.5]], dtype=np.float32)
        sample = prepare_image_sample(
            image, valid, depth=depth, confidence=confidence
        )

        np.testing.assert_array_equal(
            sample.mask,
            np.array([[True, False, False], [False, False, True]]),
        )

    def test_masked_psnr_ignores_invalid_pixels(self) -> None:
        target = np.zeros((2, 2, 3), dtype=np.float64)
        prediction = target.copy()
        prediction[1, 1] = 1.0
        mask = np.array([[True, True], [True, False]])

        self.assertTrue(math.isinf(masked_psnr(prediction, target, mask)))
        mask[1, 1] = True
        self.assertAlmostEqual(masked_psnr(prediction, target, mask), 10 * math.log10(4.0))
        with self.assertRaisesRegex(ValueError, "no valid pixels"):
            masked_psnr(prediction, target, np.zeros((2, 2), dtype=bool))

    def test_mask_manifest_has_unique_per_image_paths_and_is_deterministic(self) -> None:
        source = signed_mask_fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build_per_image_masks(source, root / "first")
            second = build_per_image_masks(source, root / "second")
            first_bytes = (root / "first" / "mask_manifest.json").read_bytes()
            second_bytes = (root / "second" / "mask_manifest.json").read_bytes()
            for record in first["images"]:
                path = root / "first" / record["combined_mask_path"]
                with Image.open(path) as mask:
                    self.assertEqual(mask.size, (8, 8))
            (root / "first" / "preserve.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_per_image_masks(source, root / "first")
            forced = build_per_image_masks(source, root / "first", force=True)
            self.assertEqual(
                (root / "first" / "preserve.txt").read_text(encoding="utf-8"), "keep"
            )

        paths = [record["combined_mask_path"] for record in first["images"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(first["summary"]["per_image_paths_unique"])
        self.assertEqual(first, second)
        self.assertEqual(forced, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(all(record["static_mask_path"] is None for record in first["images"]))
        self.assertEqual(verify_mask_manifest(first), first["mask_manifest_sha256"])
        first["images"][0]["valid_fraction"] = 0.0
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_mask_manifest(first)

    def test_principal_point_circle_is_generated_and_signed(self) -> None:
        source = signed_mask_fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = build_per_image_masks(
                source,
                root / "masks",
                valid_radius_px=2.0,
            )
            with Image.open(root / "masks" / output["images"][0]["valid_mask_path"]) as opened:
                actual = np.asarray(opened, dtype=np.uint8) > 0

        yy, xx = np.mgrid[0:8, 0:8]
        expected = np.hypot(xx - 3.5, yy - 3.5) <= 2.0
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(output["valid_mask_profile"], "principal_point_circle_v1")
        self.assertIsNone(output["theta_max_deg"])
        self.assertEqual(output["valid_radius_px"], 2.0)
        self.assertEqual(verify_mask_manifest(output), output["mask_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
