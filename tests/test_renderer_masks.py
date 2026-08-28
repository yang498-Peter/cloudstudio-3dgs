import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.renderer_masks import (
    build_renderer_mask_manifest,
    verify_renderer_mask_manifest,
)
from cloudstudio_3dgs.training.face_dataset import sign_face_manifest


class RendererMaskManifestTests(unittest.TestCase):
    def test_builds_signed_manifest_from_face_cache_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            faces = root / "faces"
            faces.mkdir()
            mask_path = faces / "image_front_mask.png"
            mask = np.array([[0, 255], [255, 255]], dtype=np.uint8)
            Image.fromarray(mask).save(mask_path)
            import hashlib

            mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            face = sign_face_manifest(
                {
                    "schema_version": 1,
                    "kind": "fisheye_face_cache",
                    "split": "train",
                    "source_identity": {
                        "person_mask_manifest_sha256": "a" * 64,
                    },
                    "cameras": {"left": {"faces": [{"face_id": "front"}]}},
                    "images": [
                        {
                            "image_id": "image",
                            "camera_id": "left",
                            "faces": [
                                {
                                    "face_id": "front",
                                    "mask_path": "faces/image_front_mask.png",
                                    "mask_sha256": mask_sha,
                                    "mask_true_pixels": 3,
                                }
                            ],
                        }
                    ],
                }
            )
            manifest = build_renderer_mask_manifest(face, root)
            self.assertEqual(
                verify_renderer_mask_manifest(manifest),
                manifest["renderer_mask_manifest_sha256"],
            )
            self.assertEqual(manifest["summary"]["face_sample_count"], 1)
            self.assertEqual(manifest["summary"]["total_keep_pixels"], 3)
            self.assertFalse(manifest["policy"]["multiclass_logits_recovered"])
            self.assertEqual(
                manifest["policy"]["label_33_semantics"], "UNKNOWN_NOT_INFERRED"
            )

            tampered = json.loads(json.dumps(manifest))
            tampered["masks"][0]["keep_pixels"] = 4
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                verify_renderer_mask_manifest(tampered)

    def test_requires_person_mask_lineage(self) -> None:
        face = sign_face_manifest(
            {
                "schema_version": 1,
                "kind": "fisheye_face_cache",
                "split": "train",
                "source_identity": {},
                "cameras": {"left": {"faces": [{"face_id": "front"}]}},
                "images": [{"image_id": "image", "camera_id": "left", "faces": []}],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "person mask manifest"):
                build_renderer_mask_manifest(face, Path(directory))


if __name__ == "__main__":
    unittest.main()
