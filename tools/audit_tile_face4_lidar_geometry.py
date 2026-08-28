#!/usr/bin/env python3
"""Fail-closed content audit for one Tile-scoped Face4 LiDAR sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.face_lidar_geometry import verify_face_lidar_geometry_manifest
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.s1_reader import sha256_file
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec
from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset, verify_face_manifest
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument("--tile-id", required=True, type=int)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--face-cache-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tile_inputs = _read(args.tile_inputs)
    tile_inputs_sha = verify_tile_inputs_manifest(
        tile_inputs, root=args.tile_inputs_root, verify_artifacts=True
    )
    matches = [tile for tile in tile_inputs["tiles"] if int(tile["tile_id"]) == args.tile_id]
    if len(matches) != 1:
        raise ValueError(f"Tile input manifest has no unique Tile {args.tile_id}")
    tile = matches[0]
    views = {str(view["sample_id"]): view for view in tile["views"]}

    face_manifest = _read(args.face_manifest)
    face_sha = verify_face_manifest(face_manifest)
    dataset = _read(args.dataset_manifest)
    images = {str(image["image_id"]): image for image in dataset["images"]}
    specs = {
        (str(camera_id), str(payload["face_id"])): FaceSpec.from_dict(payload)
        for camera_id, camera in face_manifest["cameras"].items()
        for payload in camera["faces"]
    }
    geometry = _read(args.geometry_manifest)
    geometry_sha = verify_face_lidar_geometry_manifest(geometry)
    if geometry.get("source_face_manifest_sha256") != face_sha:
        raise ValueError("Tile LiDAR geometry is bound to a different Face4 cache")
    if geometry.get("tile_inputs_manifest_sha256") != tile_inputs_sha:
        raise ValueError("Tile LiDAR geometry is bound to different Tile inputs")
    if int(geometry.get("tile_id", -1)) != args.tile_id:
        raise ValueError("Tile LiDAR geometry has a different Tile id")
    records = {str(record["sample_id"]): record for record in geometry["records"]}
    if set(records) != set(views):
        raise ValueError("Tile LiDAR geometry records differ from selected Tile views")

    box = np.asarray(tile["training_and_export_box"], dtype=np.float64)
    total_pixels = 0
    min_world = np.full(3, np.inf)
    max_world = np.full(3, -np.inf)
    for sample_id, view in views.items():
        record = records[sample_id]
        if int(record["valid_pixels"]) == 0:
            continue
        path = args.geometry_root / str(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Tile LiDAR artifact mismatch: {path}")
        sparse = load_sparse_depth(path)
        if len(sparse.pixel_index) != int(record["valid_pixels"]):
            raise ValueError(f"Tile LiDAR pixel count mismatch: {sample_id}")
        image_id, face_id = sample_id.rsplit("::", 1)
        image = images[image_id]
        face = specs[(str(image["camera_id"]), face_id)]
        yy, xx = np.divmod(sparse.pixel_index.astype(np.int64), face.width)
        if not np.all(
            (xx >= int(view["x"]))
            & (xx < int(view["x"]) + int(view["width"]))
            & (yy >= int(view["y"]))
            & (yy < int(view["y"]) + int(view["height"]))
        ):
            raise ValueError(f"Tile LiDAR pixels escape the crop: {sample_id}")
        rays = face.pixels_to_directions(np.column_stack([xx, yy]))
        camera_points = rays * sparse.range_m[:, None]
        pose = np.asarray(image["c2w"], dtype=np.float64)
        world = camera_points @ pose[:3, :3].T + pose[:3, 3]
        if not np.all((world >= box[0] - 1e-5) & (world <= box[1] + 1e-5)):
            raise ValueError(f"Tile LiDAR world points escape core+halo: {sample_id}")
        min_world = np.minimum(min_world, np.min(world, axis=0))
        max_world = np.maximum(max_world, np.max(world, axis=0))
        total_pixels += len(world)
    if total_pixels != int(geometry["tile_valid_pixels"]):
        raise ValueError("Tile LiDAR total pixel count differs from manifest")

    face_dataset = FaceCacheDataset(
        args.face_manifest,
        args.face_cache_root,
        verify_artifacts=True,
        tile_views=tile["views"],
        face_lidar_geometry_manifest_path=args.geometry_manifest,
        face_lidar_geometry_root=args.geometry_root,
    )
    index_by_id = {sample_id: index for index, sample_id in enumerate(face_dataset.image_ids)}
    ordered = sorted(records.values(), key=lambda item: int(item["valid_pixels"]))
    probes = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    probe_reports: list[dict[str, Any]] = []
    for record in probes:
        sample_id = str(record["sample_id"])
        sample = face_dataset[index_by_id[sample_id]]
        if sample.depth_mask is None or sample.depth_range_m is None:
            raise ValueError(f"Trainer-facing sample lacks LiDAR depth: {sample_id}")
        valid_pixels = int(np.count_nonzero(sample.depth_mask))
        if valid_pixels != int(record["valid_pixels"]):
            raise ValueError(f"Trainer-facing crop changes LiDAR support: {sample_id}")
        probe_reports.append(
            {
                "sample_id": sample_id,
                "shape": list(sample.depth_range_m.shape),
                "valid_pixels": valid_pixels,
                "range_min_m": float(np.min(sample.depth_range_m[sample.depth_mask])),
                "range_max_m": float(np.max(sample.depth_range_m[sample.depth_mask])),
            }
        )

    unsigned = {
        "schema_version": 1,
        "kind": "tile_face4_lidar_geometry_content_audit_v1",
        "status": "CONSUMPTION_READY",
        "tile_id": args.tile_id,
        "tile_name": tile["name"],
        "tile_inputs_manifest_sha256": tile_inputs_sha,
        "face_manifest_sha256": face_sha,
        "face_lidar_geometry_manifest_sha256": geometry_sha,
        "view_count": len(records),
        "nonempty_view_count": int(geometry["nonempty_view_count"]),
        "tile_valid_pixels": total_pixels,
        "crop_valid_pixels_before_tile_filter": int(
            geometry["crop_valid_pixels_before_tile_filter"]
        ),
        "retained_after_world_box_fraction": float(
            total_pixels / int(geometry["crop_valid_pixels_before_tile_filter"])
        ),
        "observed_world_bounds": [min_world.tolist(), max_world.tolist()],
        "required_world_bounds": box.tolist(),
        "trainer_facing_probes": probe_reports,
        "da2_used": False,
        "full_las_reprojection_performed": False,
        "training_allowed": False,
        "next_required_artifact": "short_trainer_consumption_smoke",
    }
    report = dict(unsigned)
    report["audit_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
