"""Deterministic local-coordinate LiDAR initialization for 3DGS."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .s1_reader import find_point_cloud, sha256_file


@dataclass(frozen=True)
class VoxelInitializationConfig:
    """Bounded settings for deterministic voxel-grid initialization."""

    target_points: int = 400_000
    cap_max: int = 1_000_000
    voxel_size: float | str = "auto"
    edge_preservation_ratio: float = 0.2
    seed: int = 0
    chunk_size: int = 2_000_000
    auto_tolerance: float = 0.90
    auto_max_passes: int = 8

    def validate(self) -> None:
        if self.target_points <= 0:
            raise ValueError("target_points must be positive")
        if self.cap_max <= 1:
            raise ValueError("cap_max must be greater than one")
        if self.target_points >= self.cap_max:
            raise ValueError(
                f"target_points ({self.target_points}) must be smaller than cap_max "
                f"({self.cap_max})"
            )
        if self.voxel_size != "auto" and float(self.voxel_size) <= 0.0:
            raise ValueError("voxel_size must be 'auto' or a positive number")
        if not 0.0 <= self.edge_preservation_ratio <= 1.0:
            raise ValueError("edge_preservation_ratio must be in [0, 1]")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 < self.auto_tolerance <= 1.0:
            raise ValueError("auto_tolerance must be in (0, 1]")
        if self.auto_max_passes <= 0:
            raise ValueError("auto_max_passes must be positive")


@dataclass(frozen=True)
class VoxelInitializationResult:
    xyz: np.ndarray
    rgb: np.ndarray
    report: dict[str, object]


@dataclass
class _Candidates:
    keys: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    source_index: np.ndarray
    score: np.ndarray
    counts: np.ndarray

    @classmethod
    def empty(cls) -> "_Candidates":
        return cls(
            keys=np.empty((0, 3), dtype=np.int64),
            xyz=np.empty((0, 3), dtype=np.float64),
            rgb=np.empty((0, 3), dtype=np.uint8),
            source_index=np.empty(0, dtype=np.int64),
            score=np.empty(0, dtype=np.float64),
            counts=np.empty(0, dtype=np.int64),
        )


def _percentile_from_histogram(histogram: np.ndarray, percentile: float) -> int:
    count = int(histogram.sum())
    if count == 0:
        return 0
    rank = max(1, math.ceil(percentile * count))
    return int(np.searchsorted(np.cumsum(histogram), rank, side="left"))


def _scan_las(path: Path, chunk_size: int) -> dict[str, object]:
    import laspy

    histogram = np.zeros(65_536, dtype=np.int64)
    point_count = 0
    has_rgb = None
    with laspy.open(path) as reader:
        bounds_min = np.asarray(reader.header.mins, dtype=np.float64)
        bounds_max = np.asarray(reader.header.maxs, dtype=np.float64)
        header_count = int(reader.header.point_count)
        for chunk in reader.chunk_iterator(chunk_size):
            dimensions = set(chunk.point_format.dimension_names)
            chunk_has_rgb = {"red", "green", "blue"} <= dimensions
            if has_rgb is None:
                has_rgb = chunk_has_rgb
            elif has_rgb != chunk_has_rgb:
                raise ValueError("LAS RGB dimensions changed between chunks")
            if chunk_has_rgb:
                for channel in (chunk.red, chunk.green, chunk.blue):
                    histogram += np.bincount(
                        np.asarray(channel, dtype=np.uint16), minlength=65_536
                    )
            point_count += len(chunk)
    if point_count != header_count:
        raise ValueError(
            f"LAS point count mismatch: header={header_count}, read={point_count}"
        )
    if point_count == 0:
        raise ValueError(f"point cloud is empty: {path}")

    if has_rgb:
        nonzero = np.flatnonzero(histogram)
        rgb_min = int(nonzero[0])
        rgb_max = int(nonzero[-1])
        rgb_median = _percentile_from_histogram(histogram, 0.50)
        rgb_p99 = _percentile_from_histogram(histogram, 0.99)
        rgb_mode = "uint8_in_uint16" if rgb_p99 <= 255 else "uint16_scaled"
    else:
        rgb_min = rgb_median = rgb_p99 = rgb_max = 180
        rgb_mode = "missing_default_180"

    return {
        "point_count": point_count,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "has_rgb": bool(has_rgb),
        "rgb_mode": rgb_mode,
        "rgb_source": {
            "min": rgb_min,
            "median": rgb_median,
            "p99": rgb_p99,
            "max": rgb_max,
        },
    }


def _convert_rgb(chunk: object, mode: str) -> np.ndarray:
    dimensions = set(chunk.point_format.dimension_names)
    if not {"red", "green", "blue"} <= dimensions:
        return np.full((len(chunk), 3), 180, dtype=np.uint8)
    source = np.column_stack([chunk.red, chunk.green, chunk.blue]).astype(
        np.uint16, copy=False
    )
    if mode == "uint8_in_uint16":
        return np.clip(source, 0, 255).astype(np.uint8)
    return np.rint(source.astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)


def _edge_cells(keys: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Choose a stable fraction of cells whose representative favors boundaries."""
    if ratio <= 0.0:
        return np.zeros(len(keys), dtype=bool)
    if ratio >= 1.0:
        return np.ones(len(keys), dtype=bool)
    unsigned = keys.astype(np.uint64, copy=False)
    mixed = (
        unsigned[:, 0] * np.uint64(0x9E3779B185EBCA87)
        ^ unsigned[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ unsigned[:, 2] * np.uint64(0x165667B19E3779F9)
        ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    )
    threshold = np.uint64(round(ratio * ((1 << 64) - 1)))
    return mixed <= threshold


def _reduce_candidates(candidates: _Candidates) -> _Candidates:
    if len(candidates.keys) == 0:
        return candidates
    order = np.lexsort(
        (
            candidates.source_index,
            candidates.score,
            candidates.keys[:, 2],
            candidates.keys[:, 1],
            candidates.keys[:, 0],
        )
    )
    keys = candidates.keys[order]
    starts = np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)]
    first = np.flatnonzero(starts)
    counts = np.add.reduceat(candidates.counts[order], first)
    chosen = order[first]
    return _Candidates(
        keys=candidates.keys[chosen],
        xyz=candidates.xyz[chosen],
        rgb=candidates.rgb[chosen],
        source_index=candidates.source_index[chosen],
        score=candidates.score[chosen],
        counts=counts,
    )


def _merge_candidates(left: _Candidates, right: _Candidates) -> _Candidates:
    if len(left.keys) == 0:
        return _reduce_candidates(right)
    combined = _Candidates(
        keys=np.concatenate([left.keys, right.keys]),
        xyz=np.concatenate([left.xyz, right.xyz]),
        rgb=np.concatenate([left.rgb, right.rgb]),
        source_index=np.concatenate([left.source_index, right.source_index]),
        score=np.concatenate([left.score, right.score]),
        counts=np.concatenate([left.counts, right.counts]),
    )
    return _reduce_candidates(combined)


def _chunk_candidates(
    xyz: np.ndarray,
    rgb: np.ndarray,
    source_index: np.ndarray,
    *,
    origin: np.ndarray,
    voxel_size: float,
    edge_preservation_ratio: float,
    seed: int,
) -> _Candidates:
    keys = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    centers = origin + (keys.astype(np.float64) + 0.5) * voxel_size
    distance_squared = np.einsum("ij,ij->i", xyz - centers, xyz - centers)
    edge = _edge_cells(keys, edge_preservation_ratio, seed)
    score = np.where(edge, -distance_squared, distance_squared)
    return _reduce_candidates(
        _Candidates(
            keys=keys,
            xyz=xyz,
            rgb=rgb,
            source_index=source_index,
            score=score,
            counts=np.ones(len(xyz), dtype=np.int64),
        )
    )


def voxel_downsample_arrays(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    voxel_size: float,
    origin: np.ndarray | None = None,
    edge_preservation_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Downsample arrays and return xyz, rgb, sorted voxel keys, occupancy."""
    points = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(rgb, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("xyz and rgb must both have shape [N, 3]")
    if len(points) == 0:
        raise ValueError("point cloud is empty")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    actual_origin = points.min(axis=0) if origin is None else np.asarray(origin, dtype=np.float64)
    reduced = _chunk_candidates(
        points,
        colors,
        np.arange(len(points), dtype=np.int64),
        origin=actual_origin,
        voxel_size=voxel_size,
        edge_preservation_ratio=edge_preservation_ratio,
        seed=seed,
    )
    return reduced.xyz, reduced.rgb, reduced.keys, reduced.counts


def _voxelize_las(
    path: Path,
    *,
    scan: dict[str, object],
    voxel_size: float,
    config: VoxelInitializationConfig,
) -> _Candidates:
    import laspy

    merged = _Candidates.empty()
    offset = 0
    origin = np.asarray(scan["bounds_min"], dtype=np.float64)
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(config.chunk_size):
            xyz = np.column_stack([chunk.x, chunk.y, chunk.z]).astype(
                np.float64, copy=False
            )
            rgb = _convert_rgb(chunk, str(scan["rgb_mode"]))
            indexes = np.arange(offset, offset + len(chunk), dtype=np.int64)
            reduced = _chunk_candidates(
                xyz,
                rgb,
                indexes,
                origin=origin,
                voxel_size=voxel_size,
                edge_preservation_ratio=config.edge_preservation_ratio,
                seed=config.seed,
            )
            merged = _merge_candidates(merged, reduced)
            offset += len(chunk)
    return merged


def _initial_auto_voxel_size(bounds_min: np.ndarray, bounds_max: np.ndarray, target: int) -> float:
    extent = np.maximum(bounds_max - bounds_min, 0.0)
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 0.0:
        return 1.0
    # S1 captures are surface-dominated, so a two-dimensional occupancy model
    # is a better starting estimate than bbox volume / target.
    return max(diagonal / math.sqrt(target), np.finfo(np.float64).eps)


def _auto_voxelize(
    path: Path,
    scan: dict[str, object],
    config: VoxelInitializationConfig,
) -> tuple[_Candidates, float, list[dict[str, float | int]]]:
    target = config.target_points
    size = _initial_auto_voxel_size(
        np.asarray(scan["bounds_min"]), np.asarray(scan["bounds_max"]), target
    )
    lower_size: float | None = None  # count above target
    upper_size: float | None = None  # count at or below target
    best: tuple[_Candidates, float] | None = None
    passes: list[dict[str, float | int]] = []

    for pass_index in range(config.auto_max_passes):
        candidates = _voxelize_las(path, scan=scan, voxel_size=size, config=config)
        count = len(candidates.keys)
        passes.append({"pass": pass_index + 1, "voxel_size": size, "points": count})
        if count <= target:
            if best is None or count > len(best[0].keys):
                best = (candidates, size)
            upper_size = size
            if count >= math.floor(target * config.auto_tolerance):
                return candidates, size, passes
        else:
            lower_size = size

        if lower_size is not None and upper_size is not None:
            size = math.sqrt(lower_size * upper_size)
        elif count > target:
            size *= max(1.05, math.sqrt(count / target))
        else:
            desired = max(1.0, target * 0.95)
            size *= min(0.95, math.sqrt(max(count, 1) / desired))

    if best is not None:
        return best[0], best[1], passes

    # A hard budget is more important than the pass hint. Continue increasing
    # the cell size until the invariant is satisfied.
    for pass_index in range(config.auto_max_passes, config.auto_max_passes + 8):
        size *= 1.5
        candidates = _voxelize_las(path, scan=scan, voxel_size=size, config=config)
        count = len(candidates.keys)
        passes.append({"pass": pass_index + 1, "voxel_size": size, "points": count})
        if count <= target:
            return candidates, size, passes
    raise RuntimeError("automatic voxel tuning could not satisfy target_points")


def _array_digest(xyz: np.ndarray, rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(xyz.astype("<f8", copy=False)).tobytes())
    digest.update(np.ascontiguousarray(rgb.astype(np.uint8, copy=False)).tobytes())
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    if len(values) == 0:
        return {"min": 0, "median": 0, "p95": 0, "max": 0}
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(np.max(values)),
    }


def _coverage_against_stride(
    path: Path,
    *,
    scan: dict[str, object],
    candidates: _Candidates,
    voxel_size: float,
    chunk_size: int,
) -> dict[str, float | int]:
    import laspy

    source_count = int(scan["point_count"])
    output_count = len(candidates.keys)
    stride = max(1, math.ceil(source_count / max(output_count, 1)))
    origin = np.asarray(scan["bounds_min"], dtype=np.float64)
    stride_keys: list[np.ndarray] = []
    offset = 0
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            local_indexes = np.arange(len(chunk), dtype=np.int64)
            chosen = ((offset + local_indexes) % stride) == 0
            if np.any(chosen):
                xyz = np.column_stack([chunk.x[chosen], chunk.y[chosen], chunk.z[chosen]])
                stride_keys.append(np.floor((xyz - origin) / voxel_size).astype(np.int64))
            offset += len(chunk)
    if stride_keys:
        unique_stride = np.unique(np.concatenate(stride_keys), axis=0)
        stride_cells = len(unique_stride)
    else:
        stride_cells = 0
    source_cells = max(output_count, 1)
    voxel_coverage = len(candidates.keys) / source_cells
    stride_coverage = stride_cells / source_cells
    return {
        "evaluation_voxel_size": voxel_size,
        "source_occupied_cells": len(candidates.keys),
        "voxel_occupied_cells": len(candidates.keys),
        "stride": stride,
        "stride_sample_points": math.ceil(source_count / stride),
        "stride_occupied_cells": stride_cells,
        "voxel_coverage_ratio": voxel_coverage,
        "stride_coverage_ratio": stride_coverage,
        "coverage_gain": voxel_coverage - stride_coverage,
    }


def build_lidar_initialization(
    run_dir: Path,
    config: VoxelInitializationConfig = VoxelInitializationConfig(),
) -> VoxelInitializationResult:
    """Build deterministic voxel representatives from a local S1 LAS/LAZ."""
    config.validate()
    path = find_point_cloud(Path(run_dir))
    source_stat = path.stat()
    source_sha256 = sha256_file(path)
    scan = _scan_las(path, config.chunk_size)
    final_stat = path.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        raise RuntimeError("point cloud changed while initialization was being built")
    if config.voxel_size == "auto":
        candidates, voxel_size, passes = _auto_voxelize(path, scan, config)
    else:
        voxel_size = float(config.voxel_size)
        candidates = _voxelize_las(
            path, scan=scan, voxel_size=voxel_size, config=config
        )
        passes = [{"pass": 1, "voxel_size": voxel_size, "points": len(candidates.keys)}]

    output_count = len(candidates.keys)
    if output_count >= config.cap_max:
        raise ValueError(
            f"voxel output ({output_count}) must be smaller than cap_max ({config.cap_max}); "
            "increase voxel_size or lower target_points"
        )
    if output_count > config.target_points:
        raise ValueError(
            f"voxel output ({output_count}) exceeds target_points ({config.target_points}); "
            "use voxel_size='auto' or increase voxel_size"
        )

    bounds_min = np.asarray(scan["bounds_min"], dtype=np.float64)
    bounds_max = np.asarray(scan["bounds_max"], dtype=np.float64)
    volume = float(np.prod(np.maximum(bounds_max - bounds_min, 0.0)))
    black_fraction = float(np.mean(np.all(candidates.rgb == 0, axis=1)))
    report: dict[str, object] = {
        "schema_version": 1,
        "algorithm": "deterministic_voxel_grid_v1",
        "source": {
            "file_name": path.name,
            "coordinate_frame": "s1_local",
            "size_bytes": source_stat.st_size,
            "sha256": source_sha256,
            "point_count": int(scan["point_count"]),
            "bounds_min": bounds_min.tolist(),
            "bounds_max": bounds_max.tolist(),
            "bbox_volume_m3": volume,
            "rgb_mode": scan["rgb_mode"],
            "rgb_statistics": scan["rgb_source"],
        },
        "configuration": {
            "target_points": config.target_points,
            "cap_max": config.cap_max,
            "requested_voxel_size": config.voxel_size,
            "edge_preservation_ratio": config.edge_preservation_ratio,
            "seed": config.seed,
            "chunk_size": config.chunk_size,
            "auto_tolerance": config.auto_tolerance,
            "auto_max_passes": config.auto_max_passes,
        },
        "output": {
            "point_count": output_count,
            "voxel_size": voxel_size,
            "point_count_below_cap": output_count < config.cap_max,
            "point_count_at_or_below_target": output_count <= config.target_points,
            "occupancy_points_per_voxel": _distribution(candidates.counts),
            "density_points_per_bbox_m3": output_count / volume if volume > 0 else None,
            "rgb_black_fraction": black_fraction,
            "sha256_xyz_rgb": _array_digest(candidates.xyz, candidates.rgb),
        },
        "auto_tuning_passes": passes,
        "coverage": _coverage_against_stride(
            path,
            scan=scan,
            candidates=candidates,
            voxel_size=voxel_size,
            chunk_size=config.chunk_size,
        ),
    }
    return VoxelInitializationResult(candidates.xyz, candidates.rgb, report)


def estimate_local_geometry(
    xyz: np.ndarray,
    *,
    neighbors: int = 16,
    batch_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate deterministic PCA normals, eigenvalues and covariance matrices."""
    from scipy.spatial import cKDTree

    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if len(points) < 3:
        raise ValueError("at least three points are required for local PCA")
    if neighbors < 3:
        raise ValueError("neighbors must be at least three")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    k = min(neighbors, len(points))
    tree = cKDTree(points)
    normals = np.empty((len(points), 3), dtype=np.float32)
    eigenvalues = np.empty((len(points), 3), dtype=np.float32)
    covariance = np.empty((len(points), 3, 3), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        _, indexes = tree.query(points[start:stop], k=k, workers=1)
        neighborhoods = points[indexes]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
        values, vectors = np.linalg.eigh(cov)
        normal = vectors[:, :, 0]
        major_axis = np.argmax(np.abs(normal), axis=1)
        signs = np.sign(normal[np.arange(len(normal)), major_axis])
        signs[signs == 0] = 1
        normal *= signs[:, None]
        normals[start:stop] = normal.astype(np.float32)
        eigenvalues[start:stop] = values.astype(np.float32)
        covariance[start:stop] = cov.astype(np.float32)
    return normals, eigenvalues, covariance


def write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write the canonical little-endian XYZ/RGB initialization PLY."""
    points = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("xyz and rgb must both have shape [N, 3]")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    records = np.empty(
        len(points), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)]
    )
    records["xyz"] = points
    records["rgb"] = colors
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        records.tofile(stream)
