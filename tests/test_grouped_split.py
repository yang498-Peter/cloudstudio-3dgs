"""A grouped split must keep every image of a rig instant together, keep
stationary frames in train, hit its fractions, find loop closures, and be
deterministic. It must also refuse to write a file named split_manifest.json."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.propose_grouped_split import (
    detect_loop_closures,
    propose,
    rig_records,
    temporal_blocks,
)

ROOT = Path(__file__).resolve().parents[1]


def _c2w(x: float, y: float, z: float = 1.5) -> list[list[float]]:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m.tolist()


def loop_dataset(n_frames: int = 120, stationary_head: int = 4, period_s: float = 0.5) -> dict:
    """A circular walk of 30 m circumference done twice, 0.5 s per frame, so
    the second lap revisits the first at ~30 s gaps; the rig sits still for
    ``stationary_head`` frames before it starts."""
    images, frames = [], []
    radius = 30.0 / (2 * np.pi)
    for i in range(n_frames):
        k = 0 if i < stationary_head else i - stationary_head + 1
        angle = 2 * np.pi * (k / ((n_frames - stationary_head) / 2))
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        ids = [f"img_l{i:03d}", f"img_r{i:03d}"]
        for camera, image_id, dx in (("left", ids[0], 0.0), ("right", ids[1], 0.05)):
            images.append(
                {
                    "image_id": image_id,
                    "camera_id": camera,
                    "rig_frame_id": f"rig_{i:03d}",
                    "timestamp_ns": int(period_s * 1e9 * i) + (0 if camera == "left" else 100),
                    "c2w": _c2w(x + dx, y),
                }
            )
        frames.append(
            {
                "rig_frame_id": f"rig_{i:03d}",
                "image_ids": ids,
                "left_image_id": ids[0],
                "right_image_id": ids[1],
                "timestamp_ns": int(period_s * 1e9 * i),
            }
        )
    return {"images": images, "rig_frames": frames, "manifest_sha256": "0" * 64}


class GroupedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = loop_dataset()
        self.records = rig_records(self.dataset)
        self.env = {r["rig_frame_id"]: ("indoor" if i % 2 == 0 else "outdoor") for i, r in enumerate(self.records)}
        self.kwargs = dict(
            environment=self.env,
            val_fraction=0.10,
            test_fraction=0.07,
            seed=0,
            block_seconds=3.0,
            cell_m=2.0,
            loop_radius_m=1.0,
            loop_min_gap_s=20.0,
            min_revisit_blocks=2,
        )

    def test_stationary_detection_marks_whole_set_down_run(self) -> None:
        flags = [r["stationary"] for r in self.records[:6]]
        self.assertEqual(flags, [True, True, True, True, False, False])

    def test_loop_closures_found_on_second_lap_only(self) -> None:
        loops = detect_loop_closures(self.records, radius_m=1.0, min_gap_s=20.0)
        second_lap = [loops[r["rig_frame_id"]]["revisit"] for r in self.records[70:110]]
        self.assertTrue(all(second_lap))
        # A gap longer than the whole capture (60 s) leaves nothing to revisit.
        strict = detect_loop_closures(self.records, radius_m=1.0, min_gap_s=120.0)
        self.assertFalse(any(v["revisit"] for v in strict.values()))

    def test_blocks_are_contiguous_and_stationary_blocks_are_separate(self) -> None:
        blocks = temporal_blocks(self.records, block_seconds=3.0)
        self.assertEqual(blocks[0], [0, 1, 2, 3])
        flat = [i for b in blocks for i in b]
        self.assertEqual(flat, list(range(len(self.records))))
        for b in blocks[1:]:
            self.assertLessEqual(len(b), 7)  # 3 s at 0.5 s/frame, plus the closing frame

    def test_proposal_groups_fractions_and_stationary_pinning(self) -> None:
        proposal = propose(self.records, **self.kwargs)
        counts = proposal["counts"]
        total = sum(c["rig_frames"] for c in counts.values())
        self.assertEqual(total, len(self.records))
        self.assertGreaterEqual(counts["val"]["rig_frames"], 0.07 * total)
        self.assertLessEqual(counts["val"]["rig_frames"], 0.16 * total)
        self.assertGreaterEqual(counts["test"]["rig_frames"], 0.04 * total)
        self.assertEqual(counts["val"]["stationary_frames"], 0)
        self.assertEqual(counts["test"]["stationary_frames"], 0)
        # Both images of every rig frame share the split.
        for r in self.records:
            labels = {proposal["image_split"][i] for i in r["image_ids"]}
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels.pop(), proposal["rig_frame_split"][r["rig_frame_id"]])
        # Both environments are represented in validation, and loop closures were drawn.
        self.assertEqual(set(counts["val"]["by_environment"]), {"indoor", "outdoor"})
        self.assertGreaterEqual(counts["val"]["revisit_frames"], 1)
        # The train/val mapping folds test into val for the two-label builder.
        manual = proposal["manual_assignment_train_val"]
        self.assertEqual(set(manual.values()) <= {"train", "val"}, True)
        self.assertEqual(sum(1 for v in manual.values() if v == "val"), counts["val"]["rig_frames"] + counts["test"]["rig_frames"])

    def test_val_and_test_are_disjoint_blocks(self) -> None:
        proposal = propose(self.records, **self.kwargs)
        val_ids = {b["block_id"] for b in proposal["groups"]["val"]}
        test_ids = {b["block_id"] for b in proposal["groups"]["test"]}
        self.assertTrue(val_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids and test_ids)

    def test_deterministic_and_seed_sensitive(self) -> None:
        a = propose(self.records, **self.kwargs)["rig_frame_split"]
        b = propose(self.records, **self.kwargs)["rig_frame_split"]
        c = propose(self.records, **{**self.kwargs, "seed": 7})["rig_frame_split"]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_bad_fractions_rejected(self) -> None:
        with self.assertRaises(ValueError):
            propose(self.records, **{**self.kwargs, "val_fraction": 0.4, "test_fraction": 0.2})

    def test_cli_refuses_split_manifest_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "dataset_manifest.json"
            manifest.write_text(json.dumps(self.dataset), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "propose_grouped_split.py"),
                 "--dataset-manifest", str(manifest), "--output", str(Path(tmp) / "split_manifest.json")],
                capture_output=True, text=True, cwd=str(ROOT),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(tmp) / "split_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
