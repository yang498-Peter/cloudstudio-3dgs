from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest
from tools.subset_dataset_manifest import (
    PARENT_HASH_KEY,
    subset_manifest,
    uniform_indices,
)


def _image(index: int, side: str, rig_id: str | None, stamp: int) -> dict:
    return {
        "image_id": f"img_{side}_{index:03d}",
        "rig_frame_id": rig_id,
        "side": side,
        "camera_id": side,
        "timestamp_ns": stamp,
        "path_root": "recording",
        "path": f"camera/{side}/{stamp}.jpg",
        "size_bytes": 1,
        "sha256": "0" * 64,
        "pose_source": "ImgPose.txt",
        "pose_convention": "c2w_opencv",
        "c2w": [[1, 0, 0, index], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "split": None,
        "mask_path": None,
        "depth_path": None,
    }


def synthetic_manifest(rig_frames: int = 6, *, extra_unpaired: bool = True) -> dict:
    images, frames = [], []
    for index in range(rig_frames):
        stamp = 1_000_000_000 * (index + 1)
        rig_id = f"rig_{index:03d}"
        left = _image(index, "left", rig_id, stamp)
        right = _image(index, "right", rig_id, stamp + 5)
        # Interleave sides so filtering must not rely on ordering.
        images.extend([right, left])
        frames.append(
            {
                "rig_frame_id": rig_id,
                "timestamp_ns": stamp,
                "left_image_id": left["image_id"],
                "right_image_id": right["image_id"],
                "image_ids": [left["image_id"], right["image_id"]],
                "timestamp_delta_ns": -5,
            }
        )
    if extra_unpaired:
        images.append(_image(99, "left", None, 1_000_000_000 * (rig_frames + 1)))
    manifest = {
        "schema_version": 1,
        "coordinate_frame": "s1_local",
        "recording_id": "synthetic",
        "path_roots": {"recording": "recording_root", "run": "run_root"},
        "source_hashes": {"run:ImgPose.txt": "a" * 64},
        "cameras": [{"camera_id": "left"}, {"camera_id": "right"}],
        "images": images,
        "point_cloud": {"path": "colorized.las", "path_root": "run", "sha256": "b" * 64},
        "rig": {"rig_id": "mvp_s1_stereo"},
        "rig_diagnostics": {"pair_count": rig_frames, "unpaired_left": []},
        "unposed_images": ["camera/left/unposed.jpg"],
        "rig_frames": frames,
        "splits": {"train": [image["image_id"] for image in images]},
        "warnings": ["unposed_camera_images:1"],
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


class UniformIndicesTests(unittest.TestCase):
    def test_constant_stride_and_full_span(self) -> None:
        for total, keep in ((3043, 450), (10, 3), (7, 7), (5, 1)):
            for seed in (0, 1, 7):
                chosen, phase = uniform_indices(total, keep, seed)
                self.assertEqual(len(chosen), min(keep, total))
                self.assertEqual(chosen, sorted(set(chosen)))
                self.assertTrue(all(0 <= value < total for value in chosen))
                if keep < total:
                    self.assertLess(chosen[0], total / keep + 1)
                    self.assertGreater(chosen[-1], total - total / keep - 1)
                    self.assertGreaterEqual(phase, 0.0)

    def test_seed_changes_phase_not_uniformity(self) -> None:
        a, _ = uniform_indices(100, 10, 0)
        b, _ = uniform_indices(100, 10, 1)
        self.assertEqual(a, uniform_indices(100, 10, 0)[0])
        gaps_a = {y - x for x, y in zip(a, a[1:])}
        gaps_b = {y - x for x, y in zip(b, b[1:])}
        self.assertTrue(gaps_a <= {10, 11})
        self.assertTrue(gaps_b <= {10, 11})


class SubsetManifestTests(unittest.TestCase):
    def test_rig_frames_kept_whole_and_schema_preserved(self) -> None:
        parent = synthetic_manifest(6)
        subset, report = subset_manifest(parent, count=6, seed=0)

        self.assertEqual(set(subset), set(parent))
        verify_dataset_manifest(subset)
        self.assertNotEqual(subset["manifest_sha256"], parent["manifest_sha256"])
        self.assertEqual(subset["source_hashes"][PARENT_HASH_KEY], parent["manifest_sha256"])
        self.assertEqual(subset["source_hashes"]["run:ImgPose.txt"], "a" * 64)
        self.assertEqual(subset["cameras"], parent["cameras"])
        self.assertEqual(subset["unposed_images"], parent["unposed_images"])
        self.assertEqual(subset["warnings"][:-1], parent["warnings"])
        self.assertIn("subset_dataset:", subset["warnings"][-1])

        kept_ids = {image["image_id"] for image in subset["images"]}
        frame_ids = {frame["rig_frame_id"] for frame in subset["rig_frames"]}
        for frame in subset["rig_frames"]:
            self.assertTrue(set(frame["image_ids"]) <= kept_ids)
        for image in subset["images"]:
            if image["rig_frame_id"] is not None:
                self.assertIn(image["rig_frame_id"], frame_ids)
        self.assertEqual(subset["rig_diagnostics"]["pair_count"], len(subset["rig_frames"]))
        self.assertEqual(subset["rig_diagnostics"]["parent_pair_count"], 6)
        self.assertEqual(set(subset["splits"]["train"]), kept_ids)

        # 7 units (6 rig frames + 1 unpaired) hold 13 images; 6 requested
        # images map to round(6 * 7 / 13) = 3 units.
        self.assertEqual(report["selection"]["unit_count"], 7)
        self.assertEqual(report["selection"]["kept_unit_count"], 3)
        self.assertEqual(report["images"]["kept"], len(kept_ids))
        self.assertEqual(
            report["images"]["kept"] + report["images"]["dropped"],
            report["images"]["parent_posed"],
        )
        self.assertEqual(
            len(report["kept_units"]) + len(report["dropped_unit_ids"]), 7
        )
        self.assertEqual(report["rig_frames"]["kept"], len(subset["rig_frames"]))

    def test_deterministic_for_seed(self) -> None:
        parent = synthetic_manifest(12)
        first, _ = subset_manifest(parent, count=8, seed=3)
        second, _ = subset_manifest(parent, count=8, seed=3)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_count_at_or_above_total_keeps_everything(self) -> None:
        parent = synthetic_manifest(4, extra_unpaired=False)
        subset, report = subset_manifest(parent, count=50, seed=0)
        self.assertEqual(len(subset["images"]), 8)
        self.assertEqual(report["selection"]["dropped_unit_count"], 0)

    def test_image_only_manifest(self) -> None:
        parent = synthetic_manifest(5, extra_unpaired=False)
        parent["rig_frames"] = []
        for image in parent["images"]:
            image["rig_frame_id"] = None
        parent.pop("manifest_sha256")
        parent["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(parent)).hexdigest()
        subset, report = subset_manifest(parent, count=4, seed=0)
        self.assertEqual(report["selection"]["unit_structure"], "images")
        self.assertEqual(len(subset["images"]), 4)
        stamps = [image["timestamp_ns"] for image in subset["images"]]
        self.assertEqual(stamps, sorted(stamps))

    def test_rejects_tampered_parent(self) -> None:
        parent = synthetic_manifest(3)
        parent["recording_id"] = "tampered"
        with self.assertRaises(ValueError):
            subset_manifest(parent, count=2, seed=0)

    def test_cli_writes_manifest_and_report(self) -> None:
        parent = synthetic_manifest(6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dataset_manifest.json"
            source.write_text(json.dumps(parent), encoding="utf-8")
            out = root / "subset"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "subset_dataset_manifest.py"),
                    "--manifest",
                    str(source),
                    "--count",
                    "6",
                    "--seed",
                    "0",
                    "--output-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            written = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
            verify_dataset_manifest(written)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["subset_manifest_sha256"], written["manifest_sha256"])
            self.assertEqual(report["parent_manifest_path"], str(source.resolve()))
            second = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "subset_dataset_manifest.py"),
                    "--manifest",
                    str(source),
                    "--count",
                    "6",
                    "--output-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
