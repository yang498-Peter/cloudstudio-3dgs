"""View membership must reproduce the battery's exact face picks and report a
hold-out violation for every battery parent image the split trains on."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_view_membership import (
    battery_face_cache_path,
    battery_picks,
    build_rows,
    classify_environment,
    lidar_environment_features,
    stationary_flags,
    rig_positions,
    tile_consumers,
    write_csv,
)

FACES = ("yaw_minus_35", "yaw_plus_35", "pitch_up_56", "pitch_down_56")


def _c2w(x: float, y: float, z: float) -> list[list[float]]:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m.tolist()


def synthetic_dataset(n_frames: int = 6, stationary_head: int = 2) -> dict:
    images, frames = [], []
    for i in range(n_frames):
        x = 0.0 if i < stationary_head else 0.5 * (i - stationary_head + 1)
        ids = [f"img_l{i}", f"img_r{i}"]
        for camera, image_id, dx in (("left", ids[0], 0.0), ("right", ids[1], 0.1)):
            images.append(
                {
                    "image_id": image_id,
                    "camera_id": camera,
                    "rig_frame_id": f"rig_{i}",
                    "timestamp_ns": 1_000_000_000 * i + (0 if camera == "left" else 100),
                    "c2w": _c2w(x + dx, 0.0, 1.5),
                    "pose_source": "synthetic_at",
                }
            )
        frames.append(
            {
                "rig_frame_id": f"rig_{i}",
                "image_ids": ids,
                "left_image_id": ids[0],
                "right_image_id": ids[1],
                "timestamp_ns": 1_000_000_000 * i,
            }
        )
    return {"images": images, "rig_frames": frames, "manifest_sha256": "0" * 64}


def synthetic_face_manifest(image_ids: list[str], empty: set[str] = frozenset()) -> dict:
    return {
        "images": [
            {
                "image_id": image_id,
                "camera_id": "left",
                "rig_frame_id": f"rig_{image_id[-1]}",
                "faces": [
                    {"face_id": face, "mask_true_pixels": 0 if f"{image_id}::{face}" in empty else 100}
                    for face in FACES
                ],
            }
            for image_id in image_ids
        ]
    }


class BatteryPickTests(unittest.TestCase):
    def test_stride_sampling_matches_probe_view_order(self) -> None:
        manifest = synthetic_face_manifest(["img_l0", "img_l1", "img_l2"])
        picks = battery_picks(manifest, 4)
        # 12 faces, stride 3: indexes 0, 3, 6, 9 -> one face per parent then wrap.
        self.assertEqual([p["dataset_index"] for p in picks], [0, 3, 6, 9])
        self.assertEqual([p["sample_id"] for p in picks][:2], ["img_l0::yaw_minus_35", "img_l0::pitch_down_56"])

    def test_empty_mask_faces_are_skipped_before_striding(self) -> None:
        manifest = synthetic_face_manifest(["img_l0", "img_l1"], empty={"img_l0::yaw_minus_35"})
        picks = battery_picks(manifest, 7)
        self.assertEqual(len(picks), 7)
        self.assertNotIn("img_l0::yaw_minus_35", [p["sample_id"] for p in picks])

    def test_more_views_than_faces_takes_every_face(self) -> None:
        manifest = synthetic_face_manifest(["img_l0"])
        self.assertEqual(len(battery_picks(manifest, 48)), 4)

    def test_face_cache_path_mirrors_probe_substitution(self) -> None:
        self.assertEqual(
            battery_face_cache_path({"face_cache_manifest": "D:/x/face4/face_manifest.json"}),
            Path("D:/x/face4_val/face_manifest.json"),
        )
        # The substitution is textual: a face4_train cache becomes a path that
        # does not exist, which is exactly the trap the tool must surface.
        self.assertEqual(
            battery_face_cache_path({"face_cache_manifest": "D:/x/face4_train/face_manifest.json"}),
            Path("D:/x/face4_val_train/face_manifest.json"),
        )


class MembershipRowTests(unittest.TestCase):
    def test_rows_report_split_battery_tiles_and_violations(self) -> None:
        dataset = synthetic_dataset()
        split = {"splits": {"train": ["img_l0", "img_r0", "img_l1", "img_r1", "img_l2", "img_r2"], "val": ["img_l3", "img_r3"]}}
        battery = battery_picks(synthetic_face_manifest(["img_l1", "img_l3"]), 4)
        tiles = {
            "tiles": [
                {"tile_id": "0", "views": [{"sample_id": "img_l1::yaw_minus_35"}, {"sample_id": "img_r1::pitch_up_56"}]},
                {"tile_id": "2", "views": [{"sample_id": "img_l1::pitch_down_56"}]},
            ]
        }
        rows, summary = build_rows(dataset, split, split_label="split_t", battery=battery, tile_inputs=tiles)
        by_id = {row["image_id"]: row for row in rows}
        self.assertEqual(len(rows), 12)
        self.assertEqual(by_id["img_l1"]["split_t"], "train")
        self.assertEqual(by_id["img_l3"]["split_t"], "val")
        self.assertEqual(by_id["img_l4"]["split_t"], "none")
        self.assertTrue(by_id["img_l1"]["battery_member"])
        self.assertEqual(by_id["img_l1"]["tile_ids"], "0|2")
        self.assertEqual(by_id["img_r1"]["tile_ids"], "0")
        self.assertEqual(by_id["img_l0"]["tile_ids"], "")
        self.assertEqual(summary["battery"]["parent_images"], 2)
        self.assertEqual(summary["battery"]["holdout_violations"], 1)
        self.assertEqual(summary["battery"]["violations"][0]["image_id"], "img_l1")
        self.assertEqual(summary["battery"]["held_out"], 1)
        self.assertEqual(by_id["img_l0"]["environment"], "stationary_unresolved")
        self.assertEqual(by_id["img_l4"]["environment"], "unknown")

    def test_stationary_flags_cover_both_ends_of_a_set_down_run(self) -> None:
        dataset = synthetic_dataset(n_frames=5, stationary_head=3)
        flags = stationary_flags(dataset, rig_positions(dataset))
        self.assertEqual([flags[f"rig_{i}"] for i in range(5)], [True, True, True, False, False])

    def test_tile_consumers_map_faces_to_parents(self) -> None:
        consumers = tile_consumers({"tiles": [{"tile_id": 1, "views": [{"sample_id": "a::f"}, {"sample_id": "a::g"}]}]})
        self.assertEqual(consumers, {"a": {"1"}})
        self.assertEqual(tile_consumers(None), {})

    def test_csv_roundtrip(self) -> None:
        dataset = synthetic_dataset(n_frames=3, stationary_head=0)
        rows, _ = build_rows(dataset, {"splits": {"train": []}}, split_label="s")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            write_csv(path, rows)
            text = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(text[0].split(",")[:3], ["image_id", "rig_frame_id", "camera"])
        self.assertEqual(len(text), 1 + len(rows))


class EnvironmentTests(unittest.TestCase):
    def test_rule_thresholds(self) -> None:
        self.assertEqual(classify_environment(0, 0), "outdoor")
        self.assertEqual(classify_environment(5000, 3), "covered")
        self.assertEqual(classify_environment(5000, 8), "indoor")
        self.assertEqual(classify_environment(199, 8), "outdoor")

    def test_features_from_a_synthetic_room_and_open_field(self) -> None:
        rng = np.random.default_rng(0)
        # Room 6x6 m around (0,0): ceiling at z=4, four walls, floor.
        n = 40000
        floor = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), np.zeros(n)])
        ceiling = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), np.full(n, 4.0)])
        walls = []
        for axis in (0, 1):
            for side in (-3.0, 3.0):
                w = np.column_stack([rng.uniform(-3, 3, n // 4), rng.uniform(0, 4, n // 4)])
                walls.append(np.insert(w, axis, side, axis=1) if axis == 0 else np.column_stack([w[:, 0], np.full(n // 4, side), w[:, 1]]))
        # Open field 100 m away: floor only.
        field = np.column_stack([rng.uniform(97, 103, n), rng.uniform(-3, 3, n), np.zeros(n)])
        xyz = np.vstack([floor, ceiling, *walls, field])
        features = lidar_environment_features(xyz, {"room": np.array([0.0, 0.0, 1.5]), "field": np.array([100.0, 0.0, 1.5])})
        self.assertEqual(classify_environment(**features["room"]), "indoor")
        self.assertEqual(classify_environment(**features["field"]), "outdoor")


if __name__ == "__main__":
    unittest.main()
