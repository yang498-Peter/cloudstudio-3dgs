from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.ba.person_residual_audit import (
    PersonResidualAuditPolicy,
    audit_person_residuals,
)
from cloudstudio_3dgs.data.image_sample import CropWindow, load_image_sample
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import build_per_image_masks
from cloudstudio_3dgs.data.person_masks import (
    PersonMaskConfig,
    build_person_mask_review,
    build_person_masks,
    verify_person_mask_manifest,
    verify_person_mask_review,
)
from cloudstudio_3dgs.data.torchvision_person import load_person_model_lock


class FakePersonSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def segment(self, image: np.ndarray) -> list[dict]:
        self.calls += 1
        if self.calls == 2:
            return []
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[4:10, 5:11] = True
        return [{"mask": mask, "score": 0.93, "box_xyxy": [5.0, 4.0, 11.0, 10.0]}]


def _fixture(recording_root: Path) -> dict:
    images = []
    for index in range(2):
        relative = Path("camera") / "left" / f"{index:03d}.png"
        path = recording_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.full((16, 16, 3), 60 + index * 40, dtype=np.uint8)
        Image.fromarray(pixels).save(path, format="PNG", optimize=False)
        images.append(
            {
                "image_id": f"img_{index:03d}",
                "camera_id": "left",
                "path_root": "recording",
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "cameras": [
            {
                "camera_id": "left",
                "camera_type": "fisheye",
                "width": 16,
                "height": 16,
                "intrinsic": {"fl_x": 5.0, "fl_y": 5.0, "cx": 7.5, "cy": 7.5},
                "distortion": {
                    "camera_model": "OPENCV_FISHEYE",
                    "params": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
                },
            }
        ],
        "images": images,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


class PersonMaskTests(unittest.TestCase):
    def test_checked_in_torchvision_lock_is_exact_and_does_not_redistribute_weights(self) -> None:
        lock = load_person_model_lock(
            Path(__file__).resolve().parents[1] / "upstream" / "person_mask.lock.json"
        )
        self.assertEqual(
            lock["weights_sha256"],
            "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e",
        )
        self.assertEqual(lock["person_class_index"], 1)
        self.assertEqual(lock["weights_distribution"], "not_redistributed")

    def test_signed_person_layer_is_independent_complete_and_deterministic(self) -> None:
        identity = {
            "runtime": "torchvision",
            "version": "0.26.0+cu128",
            "architecture": "maskrcnn_resnet50_fpn_v2",
            "weights": "COCO_V1",
            "weights_sha256": "a" * 64,
            "person_class_index": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root / "recording"
            dataset = _fixture(recording)
            base = build_per_image_masks(dataset, root / "base")
            first = build_person_masks(
                dataset,
                base,
                recording,
                root / "first",
                segmenter=FakePersonSegmenter(),
                model_identity=identity,
                config=PersonMaskConfig(score_threshold=0.7, dilation_pixels=0),
            )
            second = build_person_masks(
                dataset,
                base,
                recording,
                root / "second",
                segmenter=FakePersonSegmenter(),
                model_identity=identity,
                config=PersonMaskConfig(score_threshold=0.7, dilation_pixels=0),
            )

            self.assertEqual(
                verify_person_mask_manifest(first), first["person_mask_manifest_sha256"]
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first["images"]), 2)
            self.assertEqual(first["base_mask_manifest_sha256"], base["mask_manifest_sha256"])
            self.assertEqual(first["images"][0]["person_instances"], 1)
            self.assertEqual(first["images"][1]["person_instances"], 0)
            self.assertEqual(
                len({record["person_mask_path"] for record in first["images"]}), 2
            )
            for record in first["images"]:
                self.assertTrue((root / "first" / record["person_mask_path"]).is_file())

            with self.assertRaisesRegex(FileExistsError, "published"):
                build_person_masks(
                    dataset,
                    base,
                    recording,
                    root / "first",
                    segmenter=FakePersonSegmenter(),
                    model_identity=identity,
                    config=PersonMaskConfig(score_threshold=0.7, dilation_pixels=0),
                    force=True,
                )

            review = build_person_mask_review(
                first,
                root / "first",
                {
                    "reviewer": {"type": "test", "identifier": "unit-test"},
                    "samples": [
                        {"image_id": item["image_id"], "status": "PASS"}
                        for item in first["review_samples"]
                    ],
                },
            )
            self.assertEqual(
                verify_person_mask_review(review),
                review["person_mask_review_sha256"],
            )
            self.assertEqual(review["status"], "PASS")

            first["images"][0]["person_pixels"] += 1
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_person_mask_manifest(first)

    def test_dynamic_person_layer_excludes_rgb_and_depth_with_same_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((8, 8, 3), 127, dtype=np.uint8)
            valid = np.full((8, 8), 255, dtype=np.uint8)
            person = np.zeros((8, 8), dtype=np.uint8)
            person[2:6, 2:6] = 255
            depth = np.ones((8, 8), dtype=np.float32)
            Image.fromarray(image).save(root / "image.png")
            Image.fromarray(valid).save(root / "valid.png")
            Image.fromarray(person).save(root / "person.png")
            np.savez(root / "depth.npz", range_m=depth, confidence=depth)

            sample = load_image_sample(
                root / "image.png",
                root / "valid.png",
                dynamic_mask_path=root / "person.png",
                depth_path=root / "depth.npz",
                confidence_path=root / "depth.npz",
                depth_key="range_m",
                confidence_key="confidence",
                factor=2,
                crop=CropWindow(0, 0, 8, 8),
            )

        self.assertFalse(sample.static_mask[1, 1])
        self.assertFalse(sample.mask[1, 1])
        self.assertFalse((sample.static_mask & sample.depth_valid_mask)[1, 1])
        self.assertTrue(sample.mask[0, 0])

    def test_high_residual_person_overlap_controls_ba_recommendation(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        observations = [
            {"image_id": "img", "xy": [6.0, 6.0], "error_px": 9.0},
            {"image_id": "img", "xy": [7.0, 7.0], "error_px": 8.0},
            {"image_id": "img", "xy": [8.0, 8.0], "error_px": 7.0},
            {"image_id": "img", "xy": [18.0, 18.0], "error_px": 6.0},
            {"image_id": "img", "xy": [2.0, 2.0], "error_px": 1.0},
        ]
        report = audit_person_residuals(
            observations,
            {"img": mask},
            PersonResidualAuditPolicy(
                high_residual_threshold_px=5.0,
                minimum_high_residual_observations=4,
                rerun_overlap_fraction=0.5,
            ),
        )
        self.assertEqual(report["decision"], "RERUN_MASKED_BA_RECOMMENDED")
        self.assertEqual(report["high_residual_observations"], 4)
        self.assertEqual(report["high_residual_on_person"], 3)
        self.assertEqual(report["high_residual_person_overlap_fraction"], 0.75)

        retained = audit_person_residuals(
            observations,
            {"img": np.zeros_like(mask)},
            PersonResidualAuditPolicy(
                high_residual_threshold_px=5.0,
                minimum_high_residual_observations=4,
                rerun_overlap_fraction=0.5,
            ),
        )
        self.assertEqual(retained["decision"], "RETAIN_CURRENT_BA")


if __name__ == "__main__":
    unittest.main()
