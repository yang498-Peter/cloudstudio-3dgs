from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cloudstudio_3dgs.ba.match_graph import (
    MatchGraphConfig,
    build_match_graph,
    verify_match_graph,
    write_hloc_pairs,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.splits import SplitConfig, build_split_manifest
from tests.test_splits import dataset_fixture


def loop_dataset(frame_count: int = 40) -> dict:
    dataset = dataset_fixture(frame_count)
    for image in dataset["images"]:
        index = int(image["image_id"].rsplit("_", 1)[1])
        angle = 2.0 * np.pi * index / frame_count
        side_offset = -0.05 if image["side"] == "left" else 0.05
        pose = np.eye(4)
        pose[:3, 3] = [2.0 * np.cos(angle), 2.0 * np.sin(angle) + side_offset, 0.0]
        image["c2w"] = pose.tolist()
    dataset.pop("manifest_sha256")
    dataset["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(dataset)
    ).hexdigest()
    return dataset


class BaMatchGraphTests(unittest.TestCase):
    def test_graph_is_training_only_deterministic_and_contains_loops(self) -> None:
        dataset = loop_dataset()
        assignments = {
            f"rig_{index:03d}": "val" if index in {5, 15, 25, 35} else "train"
            for index in range(40)
        }
        split = build_split_manifest(
            dataset,
            SplitConfig(mode="manual", golden_rig_frames=2),
            manual=assignments,
        )
        config = MatchGraphConfig(
            temporal_neighbor_rig_frames=2,
            loop_max_distance_m=0.8,
            loop_min_frame_gap=10,
            loop_neighbors_per_rig=2,
        )
        first = build_match_graph(dataset, split, config)
        second = build_match_graph(dataset, split, config)
        validation = set(split["splits"]["val"])

        self.assertEqual(first, second)
        self.assertEqual(verify_match_graph(first), first["match_graph_sha256"])
        self.assertEqual(first["summary"]["training_rig_frames"], 36)
        self.assertEqual(first["summary"]["reason_counts"]["stereo"], 36)
        self.assertGreater(first["summary"]["reason_counts"]["spatial_loop_left"], 0)
        self.assertTrue(
            all(
                pair["image_id_a"] not in validation
                and pair["image_id_b"] not in validation
                for pair in first["pairs"]
            )
        )
        tampered = dict(first)
        tampered["pairs"] = [dict(pair) for pair in first["pairs"]]
        leaked_pair = sorted(
            (next(iter(validation)), tampered["pairs"][0]["image_id_b"])
        )
        tampered["pairs"][0]["image_id_a"] = leaked_pair[0]
        tampered["pairs"][0]["image_id_b"] = leaked_pair[1]
        tampered.pop("match_graph_sha256")
        tampered["match_graph_sha256"] = hashlib.sha256(
            canonical_json_bytes(tampered)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "validation leakage"):
            verify_match_graph(tampered)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.txt"
            write_hloc_pairs(path, first)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), first["summary"]["pair_count"])
        self.assertTrue(all(len(line.split()) == 2 for line in lines))

    def test_invalid_loop_gap_fails_before_graph_construction(self) -> None:
        dataset = loop_dataset(8)
        split = build_split_manifest(dataset, SplitConfig())
        with self.assertRaisesRegex(ValueError, "exceed the temporal window"):
            build_match_graph(
                dataset,
                split,
                MatchGraphConfig(
                    temporal_neighbor_rig_frames=4,
                    loop_min_frame_gap=4,
                ),
            )


if __name__ == "__main__":
    unittest.main()
