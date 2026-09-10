#!/usr/bin/env python3
"""One row per parent image: split, battery membership, Tile consumers, environment.

The question this answers is not "how many views are in the battery" but
"which parent images does the battery render, and does the split being
trained on withhold them". A held-out count of 48 says nothing about that:
the battery is sampled from whichever face cache the eval config's
``face_cache_manifest`` resolves to after the same ``face4 -> face4_val``
substitution ``tools/evaluate_probe_views.py`` performs, and that cache can
belong to an older split than the one the checkpoint trained on.

Membership is reported at the parent-image level because every Face4 face of
one fisheye exposure and both cameras of one rig instant share the pose
estimate; hold-out only means something at that granularity.

    python tools/build_view_membership.py \
        --dataset-manifest .../dataset_manifest.json \
        --split-manifest .../split_manifest.json --split-label split_v9 \
        --eval-config C:/Peter/3dgs-runs/house0305_sop/delivery_eval.json \
        --tile-inputs-manifest .../tile_inputs_manifest.json \
        [--lidar-las colorized.las] \
        --output-csv research/.../01_view_membership.csv \
        --output-summary research/.../01_view_membership_summary.json

Environment (indoor / covered / outdoor) is derived from the LiDAR cloud when
``--lidar-las`` is given: a dense return band 0.4-3.5 m above the camera inside
a 0.6 m cylinder means a roof overhead; returns at camera height in at least 7
of 8 azimuth sectors within 5 m means walls around. Roof+walls is indoor,
roof-only is covered (carport, awning), neither is outdoor. Without LiDAR
the column is ``unknown`` - it is not guessed from tile ids, which are
spatial strips and not room boundaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_SEPARATOR = "::"
FACE_CACHE_VAL_SUBSTITUTION = ("face4", "face4_val")

# LiDAR environment rule (see module docstring). Thresholds were read off the
# house0305 cloud where roof frames carry 5e3-1.7e5 returns in the cylinder
# and open-sky frames carry 0; 200 sits in the empty gap between them.
CEILING_MIN_POINTS = 200
CEILING_CYLINDER_RADIUS_M = 0.6
CEILING_BAND_M = (0.4, 3.5)
WALL_SEARCH_RADIUS_M = 5.0
WALL_BAND_M = (-0.3, 1.0)
WALL_SECTOR_MIN_POINTS = 300
WALL_SECTORS_FOR_INDOOR = 7
STATIONARY_STEP_M = 0.005


def battery_face_cache_path(eval_config: dict[str, Any]) -> Path:
    """Replicate evaluate_probe_views' default (non --tile-views) cache choice."""
    return Path(
        str(eval_config["face_cache_manifest"]).replace(*FACE_CACHE_VAL_SUBSTITUTION)
    )


def battery_picks(face_manifest: dict[str, Any], views: int) -> list[dict[str, Any]]:
    """The exact faces evaluate_probe_views scores: stride sampling over the
    FaceCacheDataset order (manifest image order, faces in manifest order,
    faces with an empty mask dropped)."""
    samples: list[dict[str, Any]] = []
    for image in face_manifest["images"]:
        for face in image["faces"]:
            if int(face.get("mask_true_pixels", -1)) == 0:
                continue
            samples.append(
                {
                    "image_id": str(image["image_id"]),
                    "rig_frame_id": str(image.get("rig_frame_id", "")),
                    "camera_id": str(image.get("camera_id", "")),
                    "face_id": str(face["face_id"]),
                    "sample_id": f"{image['image_id']}{SAMPLE_SEPARATOR}{face['face_id']}",
                }
            )
    stride = max(1, len(samples) // max(1, int(views)))
    picks = list(range(0, len(samples), stride))[: int(views)]
    return [dict(samples[index], dataset_index=index) for index in picks]


def tile_consumers(tile_inputs: dict[str, Any] | None) -> dict[str, set[str]]:
    """parent image_id -> set of tile ids whose training views include any face of it."""
    consumers: dict[str, set[str]] = {}
    if not tile_inputs:
        return consumers
    for tile in tile_inputs.get("tiles", []):
        tile_id = str(tile["tile_id"])
        for view in tile.get("views", []):
            parent = str(view["sample_id"]).split(SAMPLE_SEPARATOR, 1)[0]
            consumers.setdefault(parent, set()).add(tile_id)
    return consumers


def split_lookup(split_manifest: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for label, image_ids in split_manifest.get("splits", {}).items():
        for image_id in image_ids:
            lookup[str(image_id)] = str(label)
    return lookup


def rig_positions(dataset: dict[str, Any]) -> dict[str, np.ndarray]:
    images = {str(image["image_id"]): image for image in dataset["images"]}
    positions: dict[str, np.ndarray] = {}
    for frame in dataset.get("rig_frames", []):
        centres = [
            np.asarray(images[str(image_id)]["c2w"], dtype=np.float64)[:3, 3]
            for image_id in frame["image_ids"]
            if str(image_id) in images
        ]
        if centres:
            positions[str(frame["rig_frame_id"])] = np.mean(centres, axis=0)
    return positions


def classify_environment(ceiling_pts: int, wall_sectors: int) -> str:
    if ceiling_pts >= CEILING_MIN_POINTS and wall_sectors >= WALL_SECTORS_FOR_INDOOR:
        return "indoor"
    if ceiling_pts >= CEILING_MIN_POINTS:
        return "covered"
    return "outdoor"


def lidar_environment_features(
    xyz: np.ndarray, positions: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Per rig frame: overhead return count and enclosing wall sectors.

    ``xyz`` is an (N, 3) float array in the same frame as the camera poses.
    Pure numpy/scipy so it runs on a CPU-only host.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz[:, :2])
    features: dict[str, dict[str, Any]] = {}
    for rig_id, position in positions.items():
        index = tree.query_ball_point(position[:2], WALL_SEARCH_RADIUS_M)
        if not index:
            features[rig_id] = {"ceiling_pts": 0, "wall_sectors": 0}
            continue
        local = xyz[np.asarray(index)] - position
        horizontal = np.hypot(local[:, 0], local[:, 1])
        overhead = (
            (horizontal < CEILING_CYLINDER_RADIUS_M)
            & (local[:, 2] > CEILING_BAND_M[0])
            & (local[:, 2] < CEILING_BAND_M[1])
        )
        band = local[
            (local[:, 2] > WALL_BAND_M[0]) & (local[:, 2] < WALL_BAND_M[1]) & (horizontal > 0.5)
        ]
        azimuth = np.floor(
            (np.arctan2(band[:, 1], band[:, 0]) + np.pi) / (2.0 * np.pi / 8.0)
        ).astype(int) % 8
        sectors = np.bincount(azimuth, minlength=8)
        features[rig_id] = {
            "ceiling_pts": int(overhead.sum()),
            "wall_sectors": int((sectors >= WALL_SECTOR_MIN_POINTS).sum()),
        }
    return features


def load_las_xyz(path: Path) -> np.ndarray:
    import laspy  # optional dependency; only needed with --lidar-las

    las = laspy.read(str(path))
    return np.column_stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
    ).astype(np.float64)


def stationary_flags(dataset: dict[str, Any], positions: dict[str, np.ndarray]) -> dict[str, bool]:
    """True when the rig did not move from the previous rig frame (start/end
    of a capture); such frames are near-duplicates of their neighbours."""
    ordered = sorted(
        (frame for frame in dataset.get("rig_frames", []) if str(frame["rig_frame_id"]) in positions),
        key=lambda frame: int(frame["timestamp_ns"]),
    )
    ids = [str(frame["rig_frame_id"]) for frame in ordered]
    still_after = [
        float(np.linalg.norm(positions[ids[i + 1]] - positions[ids[i]])) < STATIONARY_STEP_M
        for i in range(len(ids) - 1)
    ]
    flags: dict[str, bool] = {}
    for i, rig_id in enumerate(ids):
        before = i > 0 and still_after[i - 1]
        after = i < len(still_after) and still_after[i]
        flags[rig_id] = bool(before or after)
    return flags


def build_rows(
    dataset: dict[str, Any],
    split_manifest: dict[str, Any],
    *,
    split_label: str,
    battery: Iterable[dict[str, Any]] = (),
    tile_inputs: dict[str, Any] | None = None,
    environment: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    splits = split_lookup(split_manifest)
    consumers = tile_consumers(tile_inputs)
    positions = rig_positions(dataset)
    stationary = stationary_flags(dataset, positions)
    battery_faces: dict[str, list[str]] = {}
    for pick in battery:
        battery_faces.setdefault(str(pick["image_id"]), []).append(str(pick["face_id"]))
    rig_of_image = {
        str(image_id): str(frame["rig_frame_id"])
        for frame in dataset.get("rig_frames", [])
        for image_id in frame["image_ids"]
    }
    timestamps = [int(image["timestamp_ns"]) for image in dataset["images"]]
    t0, t1 = min(timestamps), max(timestamps)
    span = max(1, t1 - t0)

    rows: list[dict[str, Any]] = []
    for image in sorted(dataset["images"], key=lambda item: (int(item["timestamp_ns"]), str(item["camera_id"]))):
        image_id = str(image["image_id"])
        rig_id = rig_of_image.get(image_id, str(image.get("rig_frame_id", "")))
        position = positions.get(rig_id)
        env = (environment or {}).get(rig_id)
        rows.append(
            {
                "image_id": image_id,
                "rig_frame_id": rig_id,
                "camera": str(image["camera_id"]),
                "timestamp_ns": int(image["timestamp_ns"]),
                "capture_fraction": round((int(image["timestamp_ns"]) - t0) / span, 4),
                split_label: splits.get(image_id, "none"),
                "battery_member": image_id in battery_faces,
                "battery_faces": "|".join(sorted(battery_faces.get(image_id, []))),
                "tile_ids": "|".join(sorted(consumers.get(image_id, set()), key=int)),
                # A rig that is set down (capture start/end) sees the operator
                # or the ground vehicle overhead; the LiDAR rule is not
                # trustworthy there, so the label is withheld rather than guessed.
                "environment": (
                    "stationary_unresolved"
                    if stationary.get(rig_id, False)
                    else classify_environment(env["ceiling_pts"], env["wall_sectors"]) if env else "unknown"
                ),
                "lidar_ceiling_pts": env["ceiling_pts"] if env else "",
                "lidar_wall_sectors": env["wall_sectors"] if env else "",
                "stationary": bool(stationary.get(rig_id, False)),
                "x_m": "" if position is None else round(float(position[0]), 4),
                "y_m": "" if position is None else round(float(position[1]), 4),
                "z_m": "" if position is None else round(float(position[2]), 4),
                "pose_source": str(image.get("pose_source", "")),
            }
        )

    battery_rows = [row for row in rows if row["battery_member"]]
    violations = [
        {"image_id": row["image_id"], "rig_frame_id": row["rig_frame_id"], "faces": row["battery_faces"], "tile_ids": row["tile_ids"]}
        for row in battery_rows
        if row[split_label] == "train"
    ]
    def count_by(key: str, subset: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in subset:
            counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
        return dict(sorted(counts.items()))

    summary = {
        "split_label": split_label,
        "images": len(rows),
        "rig_frames": len({row["rig_frame_id"] for row in rows}),
        "by_split": count_by(split_label, rows),
        "by_environment": count_by("environment", rows),
        "by_environment_and_split": {
            env: count_by(split_label, [row for row in rows if row["environment"] == env])
            for env in sorted({row["environment"] for row in rows})
        },
        "stationary_images": sum(1 for row in rows if row["stationary"]),
        "pose_sources": count_by("pose_source", rows),
        "battery": {
            "faces": sum(len(faces) for faces in battery_faces.values()),
            "parent_images": len(battery_faces),
            "rig_frames": len({row["rig_frame_id"] for row in battery_rows}),
            "by_split": count_by(split_label, battery_rows),
            "by_environment": count_by("environment", battery_rows),
            "holdout_violations": len(violations),
            "holdout_violation_rig_frames": len({item["rig_frame_id"] for item in violations}),
            "held_out": sum(1 for row in battery_rows if row[split_label] != "train"),
            "violations": violations,
        },
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-label", default="split_v9")
    parser.add_argument("--eval-config", type=Path, help="evaluate_probe_views config; the battery face cache is derived from it")
    parser.add_argument("--battery-face-manifest", type=Path, help="explicit face cache manifest for the battery (overrides --eval-config)")
    parser.add_argument("--battery-views", type=int, default=48)
    parser.add_argument("--tile-inputs-manifest", type=Path)
    parser.add_argument("--lidar-las", type=Path, help="LAS/LAZ cloud in the pose frame; enables the environment column")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    provenance: dict[str, Any] = {
        "dataset_manifest": str(args.dataset_manifest),
        "dataset_manifest_file_sha256": _sha256(args.dataset_manifest),
        "dataset_manifest_declared_sha256": dataset.get("manifest_sha256"),
        "split_manifest": str(args.split_manifest),
        "split_manifest_declared_sha256": split_manifest.get("split_manifest_sha256"),
        "split_configuration": split_manifest.get("configuration"),
    }

    battery: list[dict[str, Any]] = []
    battery_manifest_path = args.battery_face_manifest
    if battery_manifest_path is None and args.eval_config is not None:
        eval_config = json.loads(args.eval_config.read_text(encoding="utf-8"))
        battery_manifest_path = battery_face_cache_path(eval_config)
        provenance["eval_config"] = str(args.eval_config)
        provenance["eval_config_split_manifest"] = eval_config.get("split_manifest")
        provenance["eval_config_face_cache_manifest"] = eval_config.get("face_cache_manifest")
    if battery_manifest_path is not None:
        provenance["battery_face_manifest"] = str(battery_manifest_path)
        provenance["battery_face_manifest_exists"] = battery_manifest_path.exists()
        if battery_manifest_path.exists():
            face_manifest = json.loads(battery_manifest_path.read_text(encoding="utf-8"))
            battery = battery_picks(face_manifest, args.battery_views)
            provenance["battery_face_manifest_source_identity"] = face_manifest.get("source_identity")
            provenance["battery_face_cache_split"] = face_manifest.get("split")
            provenance["battery_face_cache_sample_count"] = sum(len(image["faces"]) for image in face_manifest["images"])

    tile_inputs = None
    if args.tile_inputs_manifest is not None:
        tile_inputs = json.loads(args.tile_inputs_manifest.read_text(encoding="utf-8"))
        provenance["tile_inputs_manifest"] = str(args.tile_inputs_manifest)
        provenance["tile_inputs_manifest_declared_sha256"] = tile_inputs.get("tile_inputs_manifest_sha256")

    environment = None
    if args.lidar_las is not None:
        positions = rig_positions(dataset)
        environment = lidar_environment_features(load_las_xyz(args.lidar_las), positions)
        provenance["lidar_las"] = str(args.lidar_las)
        provenance["environment_rule"] = {
            "ceiling_min_points": CEILING_MIN_POINTS,
            "ceiling_cylinder_radius_m": CEILING_CYLINDER_RADIUS_M,
            "ceiling_band_m": list(CEILING_BAND_M),
            "wall_search_radius_m": WALL_SEARCH_RADIUS_M,
            "wall_band_m": list(WALL_BAND_M),
            "wall_sector_min_points": WALL_SECTOR_MIN_POINTS,
            "wall_sectors_for_indoor": WALL_SECTORS_FOR_INDOOR,
        }

    rows, summary = build_rows(
        dataset,
        split_manifest,
        split_label=args.split_label,
        battery=battery,
        tile_inputs=tile_inputs,
        environment=environment,
    )
    summary["provenance"] = provenance
    summary["battery"]["picks"] = battery
    write_csv(args.output_csv, rows)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    b = summary["battery"]
    print(
        f"{len(rows)} images; battery {b['faces']} faces / {b['parent_images']} parents / "
        f"{b['rig_frames']} rig frames; in {args.split_label} train: {b['holdout_violations']} "
        f"(held out: {b['held_out']}) -> {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
