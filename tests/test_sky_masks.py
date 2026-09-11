import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.sky_masks import (
    ADE20K_SKY_LABEL_ID,
    SKY_MASK_KIND,
    SKY_MASK_MANIFEST_SHA_KEY,
    build_sky_mask_manifest,
    load_sky_mask,
    load_sky_mask_manifest,
    sign_sky_mask_manifest,
    sky_mask_path_for,
    sky_mask_records_by_key,
    verify_sky_mask_manifest,
)


MODEL = {
    "id": "nvidia/segformer-b4-finetuned-ade-512-512",
    "revision": "2641fd1e2893964d8d473d8cf65a906cb0bff071",
    "config_sha256": "c" * 64,
    "weights_sha256": "d" * 64,
    "license_note": "NVIDIA Source Code License-NC; research supervision data only, weights not shipped",
}
RULE = {
    "decision": "sky_probability >= 0.5",
    "sky_probability_threshold": 0.5,
    "model_input": "short side 512, aspect kept",
    "upsampling": "bilinear on probability, threshold after",
    "valid_mask": "AND face cache mask",
    "sky_fraction_denominator": "valid_pixels",
}


def _write_mask(root: Path, image_id: str, face_id: str, array: np.ndarray) -> dict:
    relative = sky_mask_path_for(image_id, face_id)
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)
    sky_pixels = int(np.count_nonzero(array))
    valid_pixels = int(array.size)
    return {
        "image_id": image_id,
        "camera_id": "left",
        "face_id": face_id,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "mask_path": relative,
        "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "valid_pixels": valid_pixels,
        "sky_pixels": sky_pixels,
        "sky_fraction": sky_pixels / valid_pixels,
    }


def _build(root: Path) -> dict:
    sky = np.zeros((4, 8), dtype=np.uint8)
    sky[:2] = 255  # top half sky -> fraction 0.5
    records = [
        _write_mask(root, "image_a", "pitch_up_56", sky),
        _write_mask(root, "image_a", "pitch_down_56", np.zeros((4, 8), dtype=np.uint8)),
    ]
    return build_sky_mask_manifest(
        split="train",
        source_face_manifest_sha256="a" * 64,
        source_identity={"split": "train"},
        model=MODEL,
        rule=RULE,
        records=records,
    )


class SkyMaskManifestTests(unittest.TestCase):
    def test_builds_signed_manifest_and_loads_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _build(root)
            self.assertEqual(manifest["kind"], SKY_MASK_KIND)
            self.assertEqual(manifest["label_ids"], [ADE20K_SKY_LABEL_ID])
            self.assertEqual(manifest["label_names"], ["sky"])
            self.assertEqual(
                verify_sky_mask_manifest(manifest), manifest[SKY_MASK_MANIFEST_SHA_KEY]
            )
            summary = manifest["summary"]
            self.assertEqual(summary["face_count"], 2)
            self.assertEqual(summary["image_count"], 1)
            self.assertAlmostEqual(summary["mean_sky_fraction"], 0.25)
            self.assertEqual(summary["faces_with_sky_gt_10pct"], 1)
            self.assertEqual(summary["faces_without_sky"], 1)
            self.assertEqual(summary["total_sky_pixels"], 16)
            self.assertEqual(summary["total_valid_pixels"], 64)

            path = root / "sky_mask_train.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_sky_mask_manifest(path, expected_face_manifest_sha256="a" * 64)
            lookup = sky_mask_records_by_key(loaded)
            sky = load_sky_mask(root, lookup[("image_a", "pitch_up_56")])
            self.assertEqual(sky.dtype, np.bool_)
            self.assertEqual(sky.shape, (4, 8))
            self.assertEqual(int(sky.sum()), 16)
            self.assertTrue(sky[:2].all())
            self.assertFalse(sky[2:].any())
            with self.assertRaisesRegex(ValueError, "different face cache"):
                load_sky_mask_manifest(path, expected_face_manifest_sha256="b" * 64)

    def test_rejects_tampering_and_inconsistency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _build(root)

            tampered = json.loads(json.dumps(manifest))
            tampered["masks"][0]["sky_pixels"] = 15
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                verify_sky_mask_manifest(tampered)

            resigned = sign_sky_mask_manifest(tampered)
            with self.assertRaisesRegex(ValueError, "sky_fraction does not match"):
                verify_sky_mask_manifest(resigned)

            wrong_label = sign_sky_mask_manifest({**manifest, "label_ids": [48]})
            with self.assertRaisesRegex(ValueError, "ADE20K sky"):
                verify_sky_mask_manifest(wrong_label)

            duplicate = json.loads(json.dumps(manifest))
            duplicate["masks"].append(dict(duplicate["masks"][0]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                verify_sky_mask_manifest(sign_sky_mask_manifest(duplicate))

            bad_summary = json.loads(json.dumps(manifest))
            bad_summary["summary"]["faces_with_sky_gt_10pct"] = 2
            with self.assertRaisesRegex(ValueError, "summary faces_with_sky_gt_10pct"):
                verify_sky_mask_manifest(sign_sky_mask_manifest(bad_summary))

            unbound = sign_sky_mask_manifest({**manifest, "source_face_manifest_sha256": ""})
            with self.assertRaisesRegex(ValueError, "not bound"):
                verify_sky_mask_manifest(unbound)

            no_model = sign_sky_mask_manifest({**manifest, "model": {"id": MODEL["id"]}})
            with self.assertRaisesRegex(ValueError, "model identity"):
                verify_sky_mask_manifest(no_model)

            unsigned = dict(manifest)
            unsigned.pop(SKY_MASK_MANIFEST_SHA_KEY)
            with self.assertRaisesRegex(ValueError, "unsigned"):
                verify_sky_mask_manifest(unsigned)

    def test_load_sky_mask_fails_closed_on_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _build(root)
            record = sky_mask_records_by_key(manifest)[("image_a", "pitch_up_56")]
            path = root / Path(*record["mask_path"].split("/"))
            Image.fromarray(np.full((4, 8), 255, dtype=np.uint8), mode="L").save(path)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_sky_mask(root, record)
            with self.assertRaisesRegex(ValueError, "pixel count"):
                load_sky_mask(root, record, verify_sha=False)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                load_sky_mask(root, record)

    def test_rejects_unsafe_mask_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _build(root)
            escaped = json.loads(json.dumps(manifest))
            escaped["masks"][0]["mask_path"] = "../outside_sky.png"
            with self.assertRaisesRegex(ValueError, "unsafe artifact path"):
                verify_sky_mask_manifest(sign_sky_mask_manifest(escaped))


if __name__ == "__main__":
    unittest.main()
