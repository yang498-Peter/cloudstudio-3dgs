"""Deterministic mesh-supported coverage completion primitives."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


@dataclass(frozen=True)
class MeshCompletionConfig:
    alpha_floor: float = 0.9
    confidence_min: float = 0.8
    depth_tolerance_m: float = 0.05
    pixel_stride: int = 8
    voxel_size_m: float = 0.015
    max_completion_points: int = 600_000

    def validate(self) -> None:
        if not 0.0 < self.alpha_floor <= 1.0:
            raise ValueError("alpha_floor must be within (0, 1]")
        if not 0.0 <= self.confidence_min <= 1.0:
            raise ValueError("confidence_min must be within [0, 1]")
        if self.depth_tolerance_m <= 0.0:
            raise ValueError("depth_tolerance_m must be positive")
        if self.pixel_stride <= 0 or self.voxel_size_m <= 0.0:
            raise ValueError("pixel stride and voxel size must be positive")
        if self.max_completion_points <= 0:
            raise ValueError("max_completion_points must be positive")


def _safe_artifact(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError("mesh completion artifact paths must use forward slashes")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError("unsafe mesh completion artifact path")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("mesh completion artifact escapes its root")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_mesh_completion_initialization_manifest(
    manifest: dict,
    *,
    root: Path,
    verify_artifacts: bool = True,
) -> str:
    """Verify the signed mixed LiDAR/mesh completion initialization contract."""

    payload = deepcopy(manifest)
    expected = str(payload.pop("initialization_manifest_sha256", ""))
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if expected != actual:
        raise ValueError("mesh completion initialization manifest signature mismatch")
    if payload.get("kind") != "mesh_supported_surfel_completion_initialization_v1":
        raise ValueError("unexpected mesh completion initialization kind")
    counts = payload.get("counts", {})
    if int(counts.get("combined_gaussians", -1)) != int(
        counts.get("original_lidar_gaussians", -1)
    ) + int(counts.get("completion_surfels", -1)):
        raise ValueError("mesh completion initialization counts are inconsistent")
    artifacts = payload.get("artifacts", {})
    required = {
        "combined_ply": "combined_ply_sha256",
        "geometry": "geometry_sha256",
    }
    for path_key, sha_key in required.items():
        path = _safe_artifact(Path(root), str(artifacts.get(path_key, "")))
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_artifacts and _sha256_file(path) != str(artifacts.get(sha_key, "")):
            raise ValueError(f"mesh completion artifact SHA256 mismatch: {path.name}")
    return actual


def coverage_deficit_mask(
    *,
    rgb_mask: np.ndarray,
    mesh_valid: np.ndarray,
    mesh_confidence: np.ndarray,
    source_type: np.ndarray,
    rendered_alpha: np.ndarray,
    rendered_range_m: np.ndarray | None,
    mesh_range_m: np.ndarray,
    config: MeshCompletionConfig,
) -> np.ndarray:
    """Select high-authority mesh pixels not adequately represented by GS."""

    config.validate()
    shape = np.asarray(mesh_range_m).shape
    arrays = (
        rgb_mask,
        mesh_valid,
        mesh_confidence,
        source_type,
        rendered_alpha,
    )
    if any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("coverage-deficit arrays must share one shape")
    if rendered_range_m is not None and np.asarray(rendered_range_m).shape != shape:
        raise ValueError("rendered range shape differs from mesh range")

    authority = np.asarray(rgb_mask, dtype=bool)
    authority &= np.asarray(mesh_valid, dtype=bool)
    authority &= np.asarray(source_type) == 3
    authority &= np.asarray(mesh_confidence, dtype=np.float32) >= float(
        config.confidence_min
    )
    mesh_range = np.asarray(mesh_range_m, dtype=np.float32)
    authority &= np.isfinite(mesh_range) & (mesh_range > 0.0)

    alpha = np.asarray(rendered_alpha, dtype=np.float32)
    deficit = ~np.isfinite(alpha) | (alpha < float(config.alpha_floor))
    if rendered_range_m is not None:
        rendered = np.asarray(rendered_range_m, dtype=np.float32)
        range_missing = ~np.isfinite(rendered) | (rendered <= 0.0)
        range_wrong = np.abs(rendered - mesh_range) > float(config.depth_tolerance_m)
        deficit |= range_missing | range_wrong
    return authority & deficit


def deterministic_stride_mask(shape: tuple[int, int], stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be positive")
    yy, xx = np.indices(shape, dtype=np.int32)
    return (yy % stride == stride // 2) & (xx % stride == stride // 2)


def depth_boundary_mask(
    depth_range_m: np.ndarray,
    valid: np.ndarray,
    *,
    threshold_m: float = 0.1,
    dilation_pixels: int = 1,
) -> np.ndarray:
    """Mark invalid/depth-discontinuous pixels and a small safety dilation."""

    depth = np.asarray(depth_range_m, dtype=np.float32)
    usable = np.asarray(valid, dtype=bool)
    if depth.ndim != 2 or usable.shape != depth.shape:
        raise ValueError("depth and valid must share one 2D shape")
    if threshold_m <= 0.0 or dilation_pixels < 0:
        raise ValueError("threshold must be positive and dilation non-negative")
    usable &= np.isfinite(depth) & (depth > 0.0)
    boundary = ~usable
    horizontal = usable[:, 1:] & usable[:, :-1]
    jump = horizontal & (np.abs(depth[:, 1:] - depth[:, :-1]) > threshold_m)
    boundary[:, 1:] |= jump
    boundary[:, :-1] |= jump
    vertical = usable[1:, :] & usable[:-1, :]
    jump = vertical & (np.abs(depth[1:, :] - depth[:-1, :]) > threshold_m)
    boundary[1:, :] |= jump
    boundary[:-1, :] |= jump
    for _ in range(dilation_pixels):
        expanded = boundary.copy()
        expanded[1:, :] |= boundary[:-1, :]
        expanded[:-1, :] |= boundary[1:, :]
        expanded[:, 1:] |= boundary[:, :-1]
        expanded[:, :-1] |= boundary[:, 1:]
        boundary = expanded
    return boundary


def orient_normals_deterministically(normals: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("normals must have shape [N,3]")
    dominant = np.argmax(np.abs(values), axis=1)
    sign = values[np.arange(len(values)), dominant] < 0.0
    values[sign] *= -1.0
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm <= 1e-8) or not np.isfinite(values).all():
        raise ValueError("normals must be finite and non-zero")
    return values / norm


def merge_voxel_candidates(
    xyz: np.ndarray,
    normals: np.ndarray,
    rgb: np.ndarray,
    scores: np.ndarray,
    *,
    voxel_size_m: float,
    occupied_xyz: np.ndarray | None = None,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average candidate observations per voxel and remove LiDAR-occupied cells."""

    points = np.asarray(xyz, dtype=np.float32)
    normal_values = orient_normals_deterministically(normals)
    colors = np.asarray(rgb, dtype=np.float32)
    weight = np.asarray(scores, dtype=np.float32).reshape(-1)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or len(weight) != len(points)
    ):
        raise ValueError("candidate arrays have incompatible shapes")
    if voxel_size_m <= 0.0 or np.any(weight <= 0.0):
        raise ValueError("voxel size and scores must be positive")
    if not len(points):
        return (
            points,
            normal_values,
            colors.astype(np.uint8),
            weight,
            np.empty(0, dtype=np.int32),
        )

    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    sum_weight = np.bincount(inverse, weights=weight, minlength=len(unique))
    merged_xyz = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis] * weight, minlength=len(unique))
            / sum_weight
            for axis in range(3)
        ]
    ).astype(np.float32)
    merged_normal = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=normal_values[:, axis] * weight,
                minlength=len(unique),
            )
            / sum_weight
            for axis in range(3)
        ]
    ).astype(np.float32)
    merged_normal = orient_normals_deterministically(merged_normal)
    merged_rgb = np.column_stack(
        [
            np.bincount(inverse, weights=colors[:, axis] * weight, minlength=len(unique))
            / sum_weight
            for axis in range(3)
        ]
    )
    merged_score = (
        sum_weight * np.log1p(counts.astype(np.float32))
    ).astype(np.float32)

    keep = np.ones(len(unique), dtype=bool)
    if occupied_xyz is not None:
        occupied = np.asarray(occupied_xyz, dtype=np.float32)
        occupied_keys = np.unique(
            np.floor(occupied / float(voxel_size_m)).astype(np.int64), axis=0
        )
        occupied_set = {tuple(row) for row in occupied_keys.tolist()}
        keep = np.asarray(
            [tuple(row) not in occupied_set for row in unique.tolist()], dtype=bool
        )
    selected = np.flatnonzero(keep)
    if max_points is not None and len(selected) > int(max_points):
        order = np.lexsort((selected, -merged_score[selected]))
        selected = selected[order[: int(max_points)]]
    selected.sort()
    return (
        merged_xyz[selected],
        merged_normal[selected],
        np.clip(np.rint(merged_rgb[selected]), 0, 255).astype(np.uint8),
        merged_score[selected],
        counts[selected].astype(np.int32),
    )
