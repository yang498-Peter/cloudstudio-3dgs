"""Deterministic sparse LiDAR depth cache generation and validation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import (
    verify_dataset_manifest,
    verify_mask_manifest,
)
from cloudstudio_3dgs.data.s1_reader import sha256_file
from cloudstudio_3dgs.geometry.lidar_projection import (
    DepthProjectionConfig,
    SparseDepthMap,
    project_lidar_depth,
)


DEPTH_MANIFEST_NAME = "depth_manifest.json"


def verify_depth_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("depth_manifest_sha256", ""))
    if not expected:
        raise ValueError("depth manifest has no depth_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("depth_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"depth manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    records = manifest.get("images", [])
    image_ids = [str(record.get("image_id", "")) for record in records]
    paths = [str(record.get("path", "")) for record in records]
    if not records or len(image_ids) != len(set(image_ids)) or not all(image_ids):
        raise ValueError("depth manifest contains invalid image IDs")
    if len(paths) != len(set(paths)) or not all(paths):
        raise ValueError("depth manifest contains invalid cache paths")
    return actual


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe cache path: {value!r}")
    return Path(*pure.parts)


def load_xyz_point_cloud(path: Path, *, max_points: int | None = None) -> np.ndarray:
    """Load PLY/NPY/NPZ/LAS points with optional deterministic stride limiting."""
    path = Path(path)
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "xyz" not in archive:
                raise KeyError(f"{path} does not contain 'xyz'")
            points = archive["xyz"]
    elif suffix == ".ply":
        with path.open("rb") as stream:
            header_lines: list[str] = []
            while True:
                raw = stream.readline()
                if not raw:
                    raise ValueError("PLY header has no end_header")
                try:
                    line = raw.decode("ascii").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ValueError("PLY header is not ASCII") from exc
                header_lines.append(line)
                if line == "end_header":
                    break
            if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
                raise ValueError("only binary_little_endian PLY is supported")
            vertex_lines = [line for line in header_lines if line.startswith("element vertex ")]
            if len(vertex_lines) != 1:
                raise ValueError("PLY must contain exactly one vertex element")
            count = int(vertex_lines[0].split()[2])
            expected_properties = [
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
            properties = [line for line in header_lines if line.startswith("property ")]
            if properties != expected_properties:
                raise ValueError("PLY properties do not match canonical XYZ/RGB layout")
            dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)], align=False)
            records = np.fromfile(stream, dtype=dtype, count=count)
            if len(records) != count:
                raise ValueError(f"PLY vertex payload is truncated: expected {count}, read {len(records)}")
            points = records["xyz"]
    elif suffix in {".las", ".laz"}:
        import laspy

        chunks: list[np.ndarray] = []
        offset = 0
        with laspy.open(path) as reader:
            total = int(reader.header.point_count)
            if max_points is None and total > 5_000_000:
                raise ValueError(
                    f"refusing to load {total:,} LAS points without a bound; "
                    "use the PR-04 voxel PLY or pass max_points explicitly"
                )
            stride = 1 if max_points is None else max(1, int(np.ceil(total / max_points)))
            for chunk in reader.chunk_iterator(2_000_000):
                indexes = np.arange(offset, offset + len(chunk), dtype=np.int64)
                selected = (indexes % stride) == 0
                chunks.append(
                    np.column_stack(
                        [chunk.x[selected], chunk.y[selected], chunk.z[selected]]
                    ).astype(np.float64, copy=False)
                )
                offset += len(chunk)
        points = np.concatenate(chunks, axis=0)
    else:
        raise ValueError(f"unsupported point-cloud cache input: {path}")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("point cloud must have shape [N, 3] and be nonempty")
    if not np.all(np.isfinite(points)):
        raise ValueError("point cloud contains non-finite coordinates")
    if max_points is not None and len(points) > max_points:
        stride = max(1, int(np.ceil(len(points) / max_points)))
        points = points[::stride][:max_points]
    if float(np.max(np.abs(points))) > 100_000.0:
        raise ValueError(
            "point cloud is not in a safe local coordinate frame; ECEF/global-scale "
            "coordinates must be rebased before depth projection"
        )
    return points


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"unsafe NPZ array name: {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                _npy_bytes(arrays[name]),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    return output.getvalue()


def sparse_depth_npz_bytes(depth: SparseDepthMap) -> bytes:
    depth.validate()
    return deterministic_npz_bytes(
        {
            "confidence": depth.confidence.astype("<f4", copy=False),
            "pixel_index": depth.pixel_index.astype("<i4", copy=False),
            "range_m": depth.range_m.astype("<f4", copy=False),
            "shape": np.asarray(depth.shape, dtype="<i4"),
            "source_index": depth.source_index.astype("<i8", copy=False),
            "support_count": depth.support_count.astype("<i4", copy=False),
        }
    )


def load_sparse_depth(path: Path) -> SparseDepthMap:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "shape",
            "pixel_index",
            "range_m",
            "confidence",
            "source_index",
            "support_count",
        }
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"depth cache is missing arrays: {', '.join(sorted(missing))}")
        shape_values = np.asarray(archive["shape"], dtype=np.int64)
        if shape_values.shape != (2,):
            raise ValueError("depth cache shape must contain height and width")
        result = SparseDepthMap(
            (int(shape_values[0]), int(shape_values[1])),
            np.asarray(archive["pixel_index"], dtype=np.int32),
            np.asarray(archive["range_m"], dtype=np.float32),
            np.asarray(archive["confidence"], dtype=np.float32),
            np.asarray(archive["source_index"], dtype=np.int64),
            np.asarray(archive["support_count"], dtype=np.int32),
        )
    result.validate()
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _distribution(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _select_images(images: list[dict[str, Any]], max_images: int | None) -> list[dict[str, Any]]:
    ordered = sorted(images, key=lambda item: (int(item["timestamp_ns"]), str(item["side"])))
    if max_images is None or max_images >= len(ordered):
        return ordered
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    indexes = np.linspace(0, len(ordered) - 1, max_images, dtype=int)
    return [ordered[int(index)] for index in indexes]


def build_depth_cache(
    dataset_manifest: dict[str, Any],
    mask_manifest: dict[str, Any],
    mask_root: Path,
    point_cloud_path: Path,
    output_dir: Path,
    *,
    config: DepthProjectionConfig = DepthProjectionConfig(),
    workers: int = 1,
    max_images: int | None = None,
    max_points: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate deterministic sparse caches, optionally in parallel threads."""
    dataset_sha = verify_dataset_manifest(dataset_manifest)
    mask_sha = verify_mask_manifest(mask_manifest)
    if mask_manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("mask manifest is bound to a different dataset manifest")
    if workers <= 0:
        raise ValueError("workers must be positive")
    config.validate()
    point_cloud_path = Path(point_cloud_path)
    point_cloud_sha = sha256_file(point_cloud_path)
    points = load_xyz_point_cloud(point_cloud_path, max_points=max_points)
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"depth output is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"depth output is not empty: {output_dir}; pass --force")
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = {str(camera["camera_id"]): camera for camera in dataset_manifest["cameras"]}
    mask_records = {str(record["image_id"]): record for record in mask_manifest["images"]}
    images_by_id = {str(image["image_id"]): image for image in dataset_manifest["images"]}
    if set(mask_records) != set(images_by_id):
        raise ValueError("dataset and mask manifests do not contain the same image IDs")
    selected = _select_images(list(images_by_id.values()), max_images)
    selected_ids = [str(image["image_id"]) for image in selected]
    cache_identity = {
        "schema_version": 1,
        "algorithm_version": "kb4_ray_zbuffer_v1",
        "dataset_manifest_sha256": dataset_sha,
        "mask_manifest_sha256": mask_sha,
        "point_cloud_sha256": point_cloud_sha,
        "point_cloud_points": len(points),
        "point_cloud_max_points": max_points,
        "projection": config.to_dict(),
        "selected_image_ids": selected_ids,
    }
    cache_key = hashlib.sha256(canonical_json_bytes(cache_identity)).hexdigest()

    masks_by_hash: dict[str, np.ndarray] = {}
    mask_for_image: dict[str, np.ndarray] = {}
    for image_id in selected_ids:
        record = mask_records[image_id]
        digest = str(record["combined_mask_sha256"])
        if digest not in masks_by_hash:
            relative = _safe_relative_path(str(record["combined_mask_path"]))
            path = Path(mask_root) / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing combined mask: {path}")
            payload = path.read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            if actual != digest:
                raise ValueError(f"combined mask SHA256 mismatch for {image_id}")
            with Image.open(io.BytesIO(payload)) as image:
                masks_by_hash[digest] = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        mask_for_image[image_id] = masks_by_hash[digest]

    staging = Path(tempfile.mkdtemp(prefix=".depth-build-", dir=output_dir))
    try:
        def build_one(image: dict[str, Any]) -> dict[str, Any]:
            image_id = str(image["image_id"])
            camera = cameras[str(image["camera_id"])]
            depth = project_lidar_depth(
                points,
                np.asarray(image["c2w"], dtype=np.float64),
                camera,
                supervision_mask=mask_for_image[image_id],
                config=config,
            )
            payload = sparse_depth_npz_bytes(depth)
            relative = f"depth/{image_id}.npz"
            destination = staging / _safe_relative_path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return {
                "image_id": image_id,
                "camera_id": image["camera_id"],
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": list(depth.shape),
                "valid_pixels": len(depth.pixel_index),
                "valid_fraction": len(depth.pixel_index) / (depth.shape[0] * depth.shape[1]),
                "range_m": _distribution(depth.range_m),
                "confidence": _distribution(depth.confidence),
                "combined_mask_sha256": mask_records[image_id]["combined_mask_sha256"],
            }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(build_one, selected))
        records.sort(key=lambda item: item["image_id"])
        depth_dir = output_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        for record in records:
            relative = _safe_relative_path(record["path"])
            source = staging / relative
            os.replace(source, output_dir / relative)

        valid_counts = np.asarray([record["valid_pixels"] for record in records])
        manifest: dict[str, Any] = {
            **cache_identity,
            "cache_key": cache_key,
            "coordinate_frame": "s1_local",
            "depth_semantics": "euclidean_ray_range_m",
            "z_buffer": "nearest_range_per_rounded_pixel",
            "confidence": "subpixel_alignment_times_log_support",
            "complete_dataset": len(selected) == len(images_by_id),
            "total_dataset_images": len(images_by_id),
            "images": records,
            "summary": {
                "image_count": len(records),
                "valid_pixels": _distribution(valid_counts),
            },
        }
        manifest["depth_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest()
        payload = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _atomic_write(output_dir / DEPTH_MANIFEST_NAME, payload)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)
