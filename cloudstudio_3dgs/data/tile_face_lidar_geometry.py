"""Materialize Tile-scoped Face4 LiDAR range supervision without reprojecting LAS."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth, sparse_depth_npz_bytes
from cloudstudio_3dgs.data.face_lidar_geometry import (
    sign_face_lidar_geometry_manifest,
    verify_face_lidar_geometry_manifest,
)
from cloudstudio_3dgs.data.s1_reader import sha256_file
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap
from cloudstudio_3dgs.training.face_dataset import verify_face_manifest
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest


def filter_sparse_face_depth_to_world_box(
    depth: SparseDepthMap,
    *,
    face: FaceSpec,
    c2w: np.ndarray,
    crop_xywh: tuple[int, int, int, int],
    world_box: np.ndarray,
) -> SparseDepthMap:
    """Keep pre-z-buffered Face4 ranges inside both one crop and one world box."""

    depth.validate()
    pose = np.asarray(c2w, dtype=np.float64)
    box = np.asarray(world_box, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("c2w must be a finite 4x4 matrix")
    if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[1] <= box[0]):
        raise ValueError("world_box must contain increasing 3D bounds")
    if depth.shape != (face.height, face.width):
        raise ValueError("sparse depth shape differs from Face4 geometry")
    x, y, width, height = (int(value) for value in crop_xywh)
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("crop must have nonnegative origin and positive size")
    if x + width > face.width or y + height > face.height:
        raise ValueError("crop exceeds Face4 raster")
    if not len(depth.pixel_index):
        return depth

    pixel_y, pixel_x = np.divmod(depth.pixel_index.astype(np.int64), face.width)
    keep = (
        (pixel_x >= x)
        & (pixel_x < x + width)
        & (pixel_y >= y)
        & (pixel_y < y + height)
    )
    selected = np.flatnonzero(keep)
    if len(selected):
        pixels = np.column_stack([pixel_x[selected], pixel_y[selected]])
        rays_camera = face.pixels_to_directions(pixels)
        points_camera = rays_camera * depth.range_m[selected, None]
        points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
        inside = np.all((points_world >= box[0]) & (points_world <= box[1]), axis=1)
        selected = selected[inside]
    result = SparseDepthMap(
        depth.shape,
        np.ascontiguousarray(depth.pixel_index[selected], dtype=np.int32),
        np.ascontiguousarray(depth.range_m[selected], dtype=np.float32),
        np.ascontiguousarray(depth.confidence[selected], dtype=np.float32),
        np.full(len(selected), -1, dtype=np.int64),
        np.zeros(len(selected), dtype=np.int32),
    )
    result.validate()
    return result


def _face_specs(manifest: dict[str, Any]) -> dict[tuple[str, str], FaceSpec]:
    return {
        (str(camera_id), str(payload["face_id"])): FaceSpec.from_dict(payload)
        for camera_id, camera in manifest["cameras"].items()
        for payload in camera["faces"]
    }


def materialize_tile_face_lidar_geometry(
    *,
    tile_inputs_path: Path,
    tile_inputs_root: Path,
    tile_id: int,
    face_manifest_path: Path,
    dataset_manifest_path: Path,
    source_geometry_manifest_path: Path,
    source_geometry_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create a signed range sidecar for one Tile from the reusable full-LAS cache."""

    tile_inputs_path = Path(tile_inputs_path).resolve()
    tile_inputs_root = Path(tile_inputs_root).resolve()
    face_manifest_path = Path(face_manifest_path).resolve()
    dataset_manifest_path = Path(dataset_manifest_path).resolve()
    source_geometry_manifest_path = Path(source_geometry_manifest_path).resolve()
    source_geometry_root = Path(source_geometry_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "face_lidar_geometry_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace Tile LiDAR geometry: {manifest_path}")

    tile_inputs = json.loads(tile_inputs_path.read_text(encoding="utf-8"))
    tile_inputs_sha = verify_tile_inputs_manifest(
        tile_inputs, root=tile_inputs_root, verify_artifacts=True
    )
    matches = [tile for tile in tile_inputs["tiles"] if int(tile["tile_id"]) == int(tile_id)]
    if len(matches) != 1:
        raise ValueError(f"Tile input manifest has no unique Tile {tile_id}")
    tile = matches[0]

    face_manifest = json.loads(face_manifest_path.read_text(encoding="utf-8"))
    face_manifest_sha = verify_face_manifest(face_manifest)
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    images = {str(item["image_id"]): item for item in dataset["images"]}
    specs = _face_specs(face_manifest)

    source = json.loads(source_geometry_manifest_path.read_text(encoding="utf-8"))
    source_sha = verify_face_lidar_geometry_manifest(source)
    if source.get("source_face_manifest_sha256") != face_manifest_sha:
        raise ValueError("source LiDAR geometry is bound to a different Face4 cache")
    if source.get("split") != face_manifest.get("split"):
        raise ValueError("source LiDAR geometry and Face4 cache use different splits")
    source_by_sample = {str(record["sample_id"]): record for record in source["records"]}
    if len(source_by_sample) != len(source["records"]):
        raise ValueError("source LiDAR geometry contains duplicate sample IDs")

    output_root.mkdir(parents=True, exist_ok=True)
    depth_root = output_root / "depth"
    depth_root.mkdir(parents=True, exist_ok=True)
    world_box = np.asarray(tile["training_and_export_box"], dtype=np.float64)
    records: list[dict[str, Any]] = []
    total_source_pixels = 0
    total_crop_pixels = 0
    total_tile_pixels = 0
    nonempty_count = 0
    for view in tile["views"]:
        sample_id = str(view["sample_id"])
        image_id, face_id = sample_id.rsplit("::", 1)
        image = images.get(image_id)
        source_record = source_by_sample.get(sample_id)
        if image is None or source_record is None:
            raise ValueError(f"missing dataset or LiDAR record for {sample_id}")
        camera_id = str(image["camera_id"])
        face = specs[(camera_id, face_id)]
        shape = [face.height, face.width]
        source_pixels = int(source_record["valid_pixels"])
        total_source_pixels += source_pixels
        output_depth: SparseDepthMap | None = None
        crop_pixels = 0
        if source_pixels:
            source_path = source_geometry_root / str(source_record["path"])
            if not source_path.is_file() or sha256_file(source_path) != source_record["sha256"]:
                raise ValueError(f"source LiDAR artifact mismatch: {source_path}")
            source_depth = load_sparse_depth(source_path)
            pixel_y, pixel_x = np.divmod(
                source_depth.pixel_index.astype(np.int64), face.width
            )
            crop_pixels = int(
                np.count_nonzero(
                    (pixel_x >= int(view["x"]))
                    & (pixel_x < int(view["x"]) + int(view["width"]))
                    & (pixel_y >= int(view["y"]))
                    & (pixel_y < int(view["y"]) + int(view["height"]))
                )
            )
            output_depth = filter_sparse_face_depth_to_world_box(
                source_depth,
                face=face,
                c2w=np.asarray(image["c2w"], dtype=np.float64),
                crop_xywh=(
                    int(view["x"]),
                    int(view["y"]),
                    int(view["width"]),
                    int(view["height"]),
                ),
                world_box=world_box,
            )
        total_crop_pixels += crop_pixels
        valid_pixels = 0 if output_depth is None else len(output_depth.pixel_index)
        total_tile_pixels += valid_pixels
        relative_path = None
        digest = None
        byte_count = 0
        if valid_pixels:
            nonempty_count += 1
            destination = depth_root / f"{sample_id.replace('::', '_')}.npz"
            payload = sparse_depth_npz_bytes(
                output_depth,
                include_provenance=False,
                confidence_encoding="uint8",
            )
            temporary = destination.with_name(destination.name + ".tmp")
            try:
                temporary.write_bytes(payload)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            relative_path = destination.relative_to(output_root).as_posix()
            digest = sha256_file(destination)
            byte_count = destination.stat().st_size
        records.append(
            {
                "sample_id": sample_id,
                "image_id": image_id,
                "face_id": face_id,
                "path": relative_path,
                "sha256": digest,
                "shape": shape,
                "valid_pixels": valid_pixels,
                "valid_fraction_of_crop": float(
                    valid_pixels / (int(view["width"]) * int(view["height"]))
                ),
                "source_valid_pixels": source_pixels,
                "crop_valid_pixels_before_tile_filter": crop_pixels,
                "bytes": byte_count,
            }
        )

    manifest = sign_face_lidar_geometry_manifest(
        {
            "schema_version": 1,
            "kind": "face4_sparse_lidar_geometry",
            "split": face_manifest["split"],
            "source_face_manifest_sha256": face_manifest_sha,
            "source_depth_manifest_sha256": source["source_depth_manifest_sha256"],
            "source_face_lidar_geometry_manifest_sha256": source_sha,
            "tile_inputs_manifest_sha256": tile_inputs_sha,
            "tile_id": int(tile["tile_id"]),
            "tile_name": str(tile["name"]),
            "tile_training_and_export_box": tile["training_and_export_box"],
            "selection_scope": "one_tile_selected_face4_views",
            "complete_face_cache": True,
            "expected_face_count": len(records),
            "depth_semantics": "euclidean_ray_range_m_sparse_real_lidar_only",
            "projection": "reuse_full_las_fisheye_zbuffer_face4_then_crop_world_box_filter",
            "quantization_note": "reuses the signed global Face4 range cache; does not reproject full LAS",
            "mesh_interpolation": False,
            "da2_used": False,
            "view_count": len(records),
            "nonempty_view_count": nonempty_count,
            "source_valid_pixels_with_view_overlap": total_source_pixels,
            "crop_valid_pixels_before_tile_filter": total_crop_pixels,
            "tile_valid_pixels": total_tile_pixels,
            "records": records,
        }
    )
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest
