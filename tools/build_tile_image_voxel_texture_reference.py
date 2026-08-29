#!/usr/bin/env python3
"""Build a CloudStudio-native image-texture reference for one LiDAR Tile.

The audit follows the exact training inputs instead of borrowing the competitor
Tile numbering or coordinate frame.  Sparse LiDAR ranges are unprojected from
the signed Face4 sidecars through the accepted AT pose.  Image gradient and
local entropy are sampled at those same pixels, then reduced once per view and
0.5 m world voxel so dense LiDAR scans do not dominate the statistic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.trainer import load_initialization_ply


VOXEL_SIZE_M = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"artifact escapes root: {relative}") from error
    return candidate


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _encode_voxels(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64) + (1 << 20)
    if np.any(values < 0) or np.any(values >= (1 << 21)):
        raise ValueError("voxel index exceeds signed 21-bit encoding")
    return (values[:, 0] << 42) | (values[:, 1] << 21) | values[:, 2]


def _local_texture(
    gray: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = np.pad(gray, 2, mode="edge")
    rr = rows + 2
    cc = columns + 2
    gx = 0.5 * (padded[rr, cc + 1] - padded[rr, cc - 1])
    gy = 0.5 * (padded[rr + 1, cc] - padded[rr - 1, cc])
    gradient = np.sqrt(gx * gx + gy * gy)

    patches = np.stack(
        [padded[rr + dy, cc + dx] for dy in range(-2, 3) for dx in range(-2, 3)],
        axis=1,
    )
    bins = np.minimum((patches * 16.0).astype(np.int16), 15)
    entropy = np.zeros(len(rows), dtype=np.float32)
    for value in range(16):
        probability = np.mean(bins == value, axis=1)
        positive = probability > 0.0
        entropy[positive] -= probability[positive] * np.log2(probability[positive])
    return gradient.astype(np.float32), entropy, gray[rows, columns]


def _median_absolute_deviation(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    median = np.median(array)
    return float(np.median(np.abs(array - median)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--face-root", required=True, type=Path)
    parser.add_argument("--tile-inputs-manifest", required=True, type=Path)
    parser.add_argument("--tile-id", required=True, type=int)
    parser.add_argument("--lidar-geometry-manifest", required=True, type=Path)
    parser.add_argument("--lidar-geometry-root", required=True, type=Path)
    parser.add_argument("--initialization-ply", required=True, type=Path)
    parser.add_argument("--max-samples-per-view", type=int, default=20_000)
    parser.add_argument("--min-view-observations", type=int, default=3)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    face_manifest = json.loads(args.face_manifest.read_text(encoding="utf-8"))
    tile_manifest = json.loads(args.tile_inputs_manifest.read_text(encoding="utf-8"))
    geometry_manifest = json.loads(
        args.lidar_geometry_manifest.read_text(encoding="utf-8")
    )
    tile = next(
        (entry for entry in tile_manifest["tiles"] if int(entry["tile_id"]) == args.tile_id),
        None,
    )
    if tile is None:
        raise ValueError(f"Tile_{args.tile_id} is absent from tile inputs manifest")

    image_by_id = {str(entry["image_id"]): entry for entry in face_manifest["images"]}
    face_by_camera = {
        (str(camera_id), str(face["face_id"])): face
        for camera_id, camera in face_manifest["cameras"].items()
        for face in camera["faces"]
    }
    face_entry_by_sample = {
        f'{image["image_id"]}::{face["face_id"]}': face
        for image in face_manifest["images"]
        for face in image["faces"]
    }
    geometry_by_sample = {
        str(entry["sample_id"]): entry for entry in geometry_manifest["records"]
    }

    initial_xyz, _ = load_initialization_ply(args.initialization_ply)
    initial_keys, initial_counts = np.unique(
        _encode_voxels(np.floor(initial_xyz.astype(np.float64) / VOXEL_SIZE_M).astype(np.int64)),
        return_counts=True,
    )
    initial_count_by_key = {
        int(key): int(count) for key, count in zip(initial_keys, initial_counts, strict=True)
    }

    per_voxel_gradient: dict[int, list[float]] = defaultdict(list)
    per_voxel_entropy: dict[int, list[float]] = defaultdict(list)
    per_voxel_luma: dict[int, list[float]] = defaultdict(list)
    sampled_pixels = 0
    nonempty_views = 0
    skipped_missing_geometry = 0

    for view_index, view in enumerate(tile["views"]):
        sample_id = str(view["sample_id"])
        image_id, face_id = sample_id.split("::", 1)
        geometry = geometry_by_sample.get(sample_id)
        if geometry is None or not geometry.get("path"):
            skipped_missing_geometry += 1
            continue
        image_record = image_by_id[image_id]
        camera_id = str(image_record["camera_id"])
        face_spec = face_by_camera[(camera_id, face_id)]
        face_entry = face_entry_by_sample[sample_id]

        depth_path = _safe_artifact(args.lidar_geometry_root, str(geometry["path"]))
        with np.load(depth_path, allow_pickle=False) as payload:
            pixel_index = np.asarray(payload["pixel_index"], dtype=np.int64)
            ranges = np.asarray(payload["range_m"], dtype=np.float64)
            shape = tuple(int(value) for value in np.asarray(payload["shape"]).tolist())
        height, width = shape
        rows = pixel_index // width
        columns = pixel_index % width
        x0 = int(view["x"])
        y0 = int(view["y"])
        x1 = x0 + int(view["width"])
        y1 = y0 + int(view["height"])
        inside = (columns >= x0) & (columns < x1) & (rows >= y0) & (rows < y1)
        selected = np.flatnonzero(inside)
        if len(selected) == 0:
            continue
        if len(selected) > args.max_samples_per_view:
            positions = np.linspace(
                0, len(selected) - 1, args.max_samples_per_view, dtype=np.int64
            )
            selected = selected[positions]
        rows = rows[selected]
        columns = columns[selected]
        ranges = ranges[selected]

        rgb_path = _safe_artifact(args.face_root, str(face_entry["rgb_path"]))
        with Image.open(rgb_path) as source:
            crop = source.convert("L").crop((x0, y0, x1, y1))
            gray = np.asarray(crop, dtype=np.float32) / 255.0
        local_rows = rows - y0
        local_columns = columns - x0
        gradient, entropy, luma = _local_texture(gray, local_rows, local_columns)

        K = np.asarray(face_spec["K_face"], dtype=np.float64)
        directions = np.stack(
            [
                (columns.astype(np.float64) + 0.5 - K[0, 2]) / K[0, 0],
                (rows.astype(np.float64) + 0.5 - K[1, 2]) / K[1, 1],
                np.ones(len(rows), dtype=np.float64),
            ],
            axis=1,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        base_c2w = np.asarray(image_record["c2w"], dtype=np.float64)
        face_to_base = np.asarray(face_spec["R_face"], dtype=np.float64)
        rotation = base_c2w[:3, :3] @ face_to_base
        world = base_c2w[:3, 3] + (directions @ rotation.T) * ranges[:, None]
        keys = _encode_voxels(np.floor(world / VOXEL_SIZE_M).astype(np.int64))

        for key in np.unique(keys):
            if int(key) not in initial_count_by_key:
                continue
            mask = keys == key
            per_voxel_gradient[int(key)].append(float(np.median(gradient[mask])))
            per_voxel_entropy[int(key)].append(float(np.median(entropy[mask])))
            per_voxel_luma[int(key)].append(float(np.median(luma[mask])))
        sampled_pixels += int(len(selected))
        nonempty_views += 1
        if (view_index + 1) % 50 == 0:
            print(
                f"processed {view_index + 1}/{len(tile['views'])} views; "
                f"observed voxels={len(per_voxel_gradient)}",
                flush=True,
            )

    rows_out: list[dict[str, Any]] = []
    for key in sorted(per_voxel_gradient):
        observations = len(per_voxel_gradient[key])
        if observations < args.min_view_observations:
            continue
        rows_out.append(
            {
                "tile": args.tile_id,
                "voxel_key": key,
                "n_las": initial_count_by_key[key],
                "image_observation_count": observations,
                "gradient_median_depth": float(np.median(per_voxel_gradient[key])),
                "entropy_median_depth_bits": float(np.median(per_voxel_entropy[key])),
                "cross_view_luma_mad_depth_proxy": _median_absolute_deviation(
                    per_voxel_luma[key]
                ),
            }
        )
    if len(rows_out) < 25:
        raise ValueError(f"only {len(rows_out)} voxels have sufficient image observations")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows_out[0])
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    report = {
        "schema_version": 1,
        "kind": "cloudstudio_tile_image_voxel_texture_reference_v1",
        "tile_id": args.tile_id,
        "voxel_size_m": VOXEL_SIZE_M,
        "method": (
            "signed sparse LiDAR Face4 ranges -> accepted AT world voxel; "
            "same-pixel RGB gradient and 5x5 16-bin entropy; median per view then voxel"
        ),
        "input_identity": {
            "face_manifest": str(args.face_manifest.resolve()),
            "face_manifest_sha256": _sha256(args.face_manifest),
            "tile_inputs_manifest": str(args.tile_inputs_manifest.resolve()),
            "tile_inputs_manifest_sha256": _sha256(args.tile_inputs_manifest),
            "lidar_geometry_manifest": str(args.lidar_geometry_manifest.resolve()),
            "lidar_geometry_manifest_sha256": _sha256(args.lidar_geometry_manifest),
            "initialization_ply": str(args.initialization_ply.resolve()),
            "initialization_ply_sha256": _sha256(args.initialization_ply),
        },
        "counts": {
            "tile_views": int(len(tile["views"])),
            "nonempty_views": nonempty_views,
            "views_without_geometry": skipped_missing_geometry,
            "sampled_depth_pixels": sampled_pixels,
            "initialization_voxels": int(len(initial_keys)),
            "observed_voxels_before_minimum": int(len(per_voxel_gradient)),
            "reference_voxels": int(len(rows_out)),
            "reference_voxel_fraction": float(len(rows_out) / len(initial_keys)),
        },
        "sampling": {
            "max_samples_per_view": args.max_samples_per_view,
            "minimum_view_observations": args.min_view_observations,
        },
        "output_csv": str(args.output_csv.resolve()),
    }
    _write_json(args.output_json, report)
    print(
        f"texture reference -> {args.output_csv} ({len(rows_out)} voxels, "
        f"{sampled_pixels} sampled pixels)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
