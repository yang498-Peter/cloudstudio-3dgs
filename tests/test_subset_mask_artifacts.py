from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_mask_manifest
from tests.test_subset_dataset_manifest import synthetic_manifest
from tools.subset_dataset_manifest import subset_manifest
from tools.subset_mask_artifacts import person_mask_coverage, subset_mask_manifest


def _png_bytes(seed: int) -> bytes:
    # Content only needs to be stable and distinct per image for SHA checks.
    return b"\x89PNG" + bytes([seed % 256]) * 16


def synthetic_mask_manifest(dataset: dict, root: Path) -> dict:
    records = []
    for index, image in enumerate(dataset["images"]):
        relative = f"masks/{image['image_id']}.png"
        payload = _png_bytes(index)
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        records.append(
            {
                "image_id": image["image_id"],
                "camera_id": image["camera_id"],
                "source_image_path_root": "recording",
                "source_image_path": image["path"],
                "valid_mask_path": relative,
                "static_mask_path": None,
                "depth_valid_mask_path": None,
                "combined_mask_path": relative,
                "valid_mask_sha256": digest,
                "combined_mask_sha256": digest,
                "valid_fraction": 0.7,
            }
        )
    payload = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "path_root": "mask_output",
        "images": records,
        "summary": {"image_count": len(records), "camera_count": 2},
    }
    payload["mask_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    (root / "mask_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


class SubsetMaskArtifactsTests(unittest.TestCase):
    def test_mask_manifest_filtered_rebound_and_copied(self) -> None:
        parent = synthetic_manifest(6)
        subset, _ = subset_manifest(parent, count=6, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = synthetic_mask_manifest(parent, root / "parent_masks")
            out = root / "subset" / "masks"
            payload = subset_mask_manifest(
                subset, masks, mask_root=root / "parent_masks", output_dir=out
            )
            verify_mask_manifest(payload)
            kept = {image["image_id"] for image in subset["images"]}
            self.assertEqual({record["image_id"] for record in payload["images"]}, kept)
            self.assertEqual(payload["dataset_manifest_sha256"], subset["manifest_sha256"])
            self.assertEqual(
                payload["derived_from_mask_manifest_sha256"], masks["mask_manifest_sha256"]
            )
            self.assertEqual(payload["summary"]["image_count"], len(kept))
            for record in payload["images"]:
                copied = out / record["combined_mask_path"]
                self.assertTrue(copied.is_file())
                self.assertEqual(
                    hashlib.sha256(copied.read_bytes()).hexdigest(),
                    record["combined_mask_sha256"],
                )
            written = json.loads((out / "mask_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(written["mask_manifest_sha256"], payload["mask_manifest_sha256"])

    def test_mask_manifest_bound_elsewhere_is_rejected(self) -> None:
        parent = synthetic_manifest(4)
        subset, _ = subset_manifest(parent, count=4, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = synthetic_mask_manifest(parent, root / "parent_masks")
            masks.pop("mask_manifest_sha256")
            masks["dataset_manifest_sha256"] = "f" * 64
            masks["mask_manifest_sha256"] = hashlib.sha256(
                canonical_json_bytes(masks)
            ).hexdigest()
            with self.assertRaises(ValueError):
                subset_mask_manifest(
                    subset, masks, mask_root=root / "parent_masks", output_dir=root / "o"
                )

    def test_person_coverage_counts_missing(self) -> None:
        parent = synthetic_manifest(4, extra_unpaired=False)
        subset, _ = subset_manifest(parent, count=8, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "person_masks"
            source.mkdir()
            ids = [image["image_id"] for image in subset["images"]]
            for index, image_id in enumerate(ids[:3]):
                (source / f"{image_id}.png").write_bytes(_png_bytes(index))
            report = person_mask_coverage(
                subset, person_mask_dir=source, output_dir=root / "out"
            )
            self.assertEqual(report["summary"]["images"], 8)
            self.assertEqual(report["summary"]["present"], 3)
            self.assertEqual(report["summary"]["missing"], 5)
            self.assertEqual(set(report["missing_image_ids"]), set(ids[3:]))
            self.assertFalse(report["signed_manifest"])
            for entry in report["present"]:
                self.assertTrue(
                    (root / "out" / "person_masks" / f"{entry['image_id']}.png").is_file()
                )
            self.assertTrue((root / "out" / "person_mask_coverage.json").is_file())


if __name__ == "__main__":
    unittest.main()
