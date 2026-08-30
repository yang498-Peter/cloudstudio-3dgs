"""Materialize signed LiDAR initialization inputs for an adaptive Tile plan."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.s1_reader import sha256_file
from cloudstudio_3dgs.pipeline.adaptive_tiling import verify_adaptive_tile_plan


TILE_INPUT_SCHEMA_VERSION = 1
TILE_INPUT_KIND = "lidar_adaptive_tile_training_inputs_v1"


def _write_ply_from_records(records_path: Path, destination: Path, count: int) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as output, records_path.open("rb") as records:
            output.write(header)
            for chunk in iter(lambda: records.read(8 * 1024 * 1024), b""):
                output.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_lidar_tile_inputs(
    tile_plan_path: Path,
    source_las: Path,
    output_root: Path,
    *,
    expected_point_cloud_sha256: str,
    force: bool = False,
    chunk_size: int = 1_000_000,
    voxel_decimate_m: float | None = None,
) -> dict[str, Any]:
    """Stream the LAS once and write one halo-inclusive initialization PLY per Tile.

    ``voxel_decimate_m`` keeps the first point per voxel of that size. It is a
    VRAM concession for single-Tile adaptive training on small cards - the
    reference recipe runs full density and splits SPACE instead - so the
    decimation is recorded in the signed manifest, never silent. The KNN
    footprint initialization downstream scales with local spacing, so thinner
    input yields proportionally larger starting footprints rather than holes.
    """

    import laspy

    if voxel_decimate_m is not None and (
        not math.isfinite(voxel_decimate_m) or voxel_decimate_m <= 0.0
    ):
        raise ValueError("voxel_decimate_m must be finite and positive")

    tile_plan_path = Path(tile_plan_path).resolve()
    source_las = Path(source_las).resolve()
    output_root = Path(output_root).resolve()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    tile_plan = json.loads(tile_plan_path.read_text(encoding="utf-8"))
    tile_plan_sha = verify_adaptive_tile_plan(tile_plan)
    actual_source_sha = sha256_file(source_las)
    if actual_source_sha != expected_point_cloud_sha256:
        raise ValueError("source LAS SHA256 does not match the accepted depth/gate binding")
    manifest_path = output_root / "tile_inputs_manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"refusing to replace Tile inputs: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    tiles = [tile for tile in tile_plan["tiles"] if not tile["low_support_discarded"]]
    boxes = [np.asarray(tile["training_and_export_box"], dtype=np.float64) for tile in tiles]
    record_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    record_paths = [output_root / f".{tile['name']}.records.tmp" for tile in tiles]
    streams = [path.open("wb") for path in record_paths]
    counts = [0 for _ in tiles]
    seen_voxels: list[set[int]] = [set() for _ in tiles]
    source_point_total = 0
    try:
        with laspy.open(source_las) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                xyz = np.column_stack([chunk.x, chunk.y, chunk.z]).astype(np.float64)
                source_point_total += len(xyz)
                dimensions = set(chunk.point_format.dimension_names)
                if {"red", "green", "blue"} <= dimensions:
                    source_rgb = np.column_stack([chunk.red, chunk.green, chunk.blue])
                    if int(source_rgb.max(initial=0)) <= 255:
                        rgb = source_rgb.astype(np.uint8)
                    else:
                        rgb = np.rint(source_rgb.astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
                for index, box in enumerate(boxes):
                    keep = np.all((xyz >= box[0]) & (xyz <= box[1]), axis=1)
                    selected = np.flatnonzero(keep)
                    if not len(selected):
                        continue
                    if voxel_decimate_m is not None:
                        cells = np.floor(
                            (xyz[selected] - box[0]) / voxel_decimate_m
                        ).astype(np.int64)
                        spans = (
                            np.floor((box[1] - box[0]) / voxel_decimate_m).astype(
                                np.int64
                            )
                            + 2
                        )
                        keys = (
                            cells[:, 0] * spans[1] + cells[:, 1]
                        ) * spans[2] + cells[:, 2]
                        registry = seen_voxels[index]
                        fresh_rows = []
                        for row, key in enumerate(keys.tolist()):
                            if key not in registry:
                                registry.add(key)
                                fresh_rows.append(row)
                        if not fresh_rows:
                            continue
                        selected = selected[np.asarray(fresh_rows, dtype=np.int64)]
                    records = np.empty(len(selected), dtype=record_dtype)
                    records["x"] = xyz[selected, 0]
                    records["y"] = xyz[selected, 1]
                    records["z"] = xyz[selected, 2]
                    records["red"] = rgb[selected, 0]
                    records["green"] = rgb[selected, 1]
                    records["blue"] = rgb[selected, 2]
                    streams[index].write(records.tobytes())
                    counts[index] += len(selected)
    finally:
        for stream in streams:
            stream.close()

    output_tiles: list[dict[str, Any]] = []
    try:
        for tile, records_path, count in zip(tiles, record_paths, counts):
            if count <= 0:
                raise ValueError(f"{tile['name']} contains no accepted LAS points")
            tile_root = output_root / tile["name"]
            tile_root.mkdir(parents=True, exist_ok=True)
            ply_path = tile_root / "initialization_full_lidar.ply"
            if ply_path.exists() and not force:
                raise FileExistsError(f"refusing to replace Tile initialization: {ply_path}")
            _write_ply_from_records(records_path, ply_path, count)
            output_tiles.append(
                {
                    "tile_id": int(tile["tile_id"]),
                    "name": str(tile["name"]),
                    "core_box": tile["core_box"],
                    "training_and_export_box": tile["training_and_export_box"],
                    "view_count": int(tile["valid_view_count"]),
                    "views": copy.deepcopy(tile["views"]),
                    "initialization": {
                        "kind": "full_lidar_roi_with_tile_halo",
                        "path": ply_path.relative_to(output_root).as_posix(),
                        "point_count": int(count),
                        "sha256": sha256_file(ply_path),
                        "bytes": ply_path.stat().st_size,
                    },
                    "recommended_training": {
                        "resolution_level": 1,
                        "steps": 20 * int(tile["valid_view_count"]),
                        "stage_epochs": [5, 10, 5],
                        "sh_degree": 1,
                        "cuda_empty_cache_interval_steps": 2,
                    },
                }
            )
    finally:
        for path in record_paths:
            path.unlink(missing_ok=True)

    payload: dict[str, Any] = {
        "schema_version": TILE_INPUT_SCHEMA_VERSION,
        "kind": TILE_INPUT_KIND,
        "tile_plan_manifest_sha256": tile_plan_sha,
        "source_point_cloud": {
            "path": str(source_las),
            "sha256": actual_source_sha,
        },
        "halo_policy": "retain_full_training_and_export_box",
        "tile_count": len(output_tiles),
        "tiles": output_tiles,
    }
    if voxel_decimate_m is not None:
        payload["voxel_decimation"] = {
            "voxel_m": float(voxel_decimate_m),
            "policy": "first_point_per_voxel",
            "source_point_total": int(source_point_total),
        }
    payload["tile_inputs_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return payload


def verify_tile_inputs_manifest(
    manifest: dict[str, Any],
    *,
    root: Path | None = None,
    verify_artifacts: bool = False,
) -> str:
    expected = str(manifest.get("tile_inputs_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("Tile input manifest is unsigned")
    unsigned = copy.deepcopy(manifest)
    unsigned.pop("tile_inputs_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("Tile input manifest signature mismatch")
    if manifest.get("schema_version") != TILE_INPUT_SCHEMA_VERSION or manifest.get("kind") != TILE_INPUT_KIND:
        raise ValueError("unsupported Tile input schema")
    if len(manifest.get("tiles", [])) != int(manifest.get("tile_count", -1)):
        raise ValueError("Tile input count is inconsistent")
    if verify_artifacts:
        if root is None:
            raise ValueError("Tile artifact verification requires a root")
        for tile in manifest["tiles"]:
            artifact = Path(root) / tile["initialization"]["path"]
            if not artifact.is_file() or sha256_file(artifact) != tile["initialization"]["sha256"]:
                raise ValueError(f"Tile initialization artifact mismatch: {artifact}")
    return expected
