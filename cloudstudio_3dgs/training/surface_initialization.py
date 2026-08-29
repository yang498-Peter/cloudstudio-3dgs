"""LiDAR-PCA surface-aligned Gaussian initialization.

The dense LiDAR cloud already carries a measured local surface normal.  Starting
every Gaussian as an isotropic sphere with an identity quaternion discards that
information and asks photometric training to rediscover it.  This module keeps
rough/volumetric neighborhoods isotropic, while planar neighborhoods start as
thin surfels whose local z axis is aligned with the unoriented LiDAR normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SurfaceInitializationConfig:
    enabled: bool = False
    mode: str = "planar_surfel"
    planarity_gate: float = 0.6
    normal_scale_ratio: float = 0.08
    minimum_normal_scale_m: float = 0.0005
    maximum_scale_m: float | None = None

    def validate(self) -> None:
        if self.mode not in {"planar_surfel", "mipmap_k7_k30"}:
            raise ValueError(
                "surface initialization mode must be 'planar_surfel' or 'mipmap_k7_k30'"
            )
        if not 0.0 <= self.planarity_gate <= 1.0:
            raise ValueError("surface initialization planarity_gate must be within [0, 1]")
        if not 0.0 < self.normal_scale_ratio <= 1.0:
            raise ValueError("surface initialization normal_scale_ratio must be within (0, 1]")
        if self.minimum_normal_scale_m <= 0.0:
            raise ValueError("surface initialization minimum_normal_scale_m must be positive")
        if self.maximum_scale_m is not None and self.maximum_scale_m <= 0.0:
            raise ValueError("surface initialization maximum_scale_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "planarity_gate": self.planarity_gate,
            "normal_scale_ratio": self.normal_scale_ratio,
            "minimum_normal_scale_m": self.minimum_normal_scale_m,
            "maximum_scale_m": self.maximum_scale_m,
        }


def cap_surface_initialization_scales(
    scales: np.ndarray,
    *,
    maximum_scale_m: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp only the sparse KNN scale tail and report the exact intervention."""

    values = np.asarray(scales, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("surface initialization scales must be finite Nx3")
    if np.any(values <= 0.0):
        raise ValueError("surface initialization scales must be positive")
    before_max = float(values.max()) if len(values) else 0.0
    if maximum_scale_m is None:
        return values, {
            "enabled": False,
            "maximum_scale_m": None,
            "clamped_gaussian_count": 0,
            "clamped_axis_count": 0,
            "before_max_m": before_max,
            "after_max_m": before_max,
        }
    limit = float(maximum_scale_m)
    mask = values > limit
    clamped = np.minimum(values, limit).astype(np.float32, copy=False)
    return clamped, {
        "enabled": True,
        "maximum_scale_m": limit,
        "clamped_gaussian_count": int(np.count_nonzero(mask.any(axis=1))),
        "clamped_axis_count": int(np.count_nonzero(mask)),
        "before_max_m": before_max,
        "after_max_m": float(clamped.max()) if len(clamped) else 0.0,
    }


def load_initialization_geometry(
    path: Path,
    *,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load signed-by-the-trainer PCA arrays and fail closed on row mismatch."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if "normals" not in payload or "eigenvalues" not in payload:
            raise ValueError("initialization geometry must contain normals and eigenvalues")
        normals = np.asarray(payload["normals"], dtype=np.float32)
        eigenvalues = np.asarray(payload["eigenvalues"], dtype=np.float32)
    expected_shape = (int(expected_count), 3)
    if normals.shape != expected_shape or eigenvalues.shape != expected_shape:
        raise ValueError(
            "initialization geometry rows must exactly match initialization PLY "
            f"({normals.shape}, {eigenvalues.shape} != {expected_shape})"
        )
    if not np.isfinite(normals).all() or not np.isfinite(eigenvalues).all():
        raise ValueError("initialization geometry must be finite")
    return np.ascontiguousarray(normals), np.ascontiguousarray(eigenvalues)


def load_mipmap_k7_k30_geometry(
    path: Path,
    *,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the exact K=7/K=30 arrays consumed by the High/type-2 preset."""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "normals",
            "eigenvalues",
            "scales_m",
            "quaternions_wxyz",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(
                "MipMap initialization geometry is missing arrays: "
                + ", ".join(sorted(missing))
            )
        normals = np.asarray(payload["normals"], dtype=np.float32)
        eigenvalues = np.asarray(payload["eigenvalues"], dtype=np.float32)
        scales = np.asarray(payload["scales_m"], dtype=np.float32)
        quaternions = np.asarray(payload["quaternions_wxyz"], dtype=np.float32)
    expected_xyz = (int(expected_count), 3)
    if (
        normals.shape != expected_xyz
        or eigenvalues.shape != expected_xyz
        or scales.shape != expected_xyz
        or quaternions.shape != (int(expected_count), 4)
    ):
        raise ValueError("MipMap initialization geometry rows do not match the PLY")
    if not all(
        np.isfinite(array).all()
        for array in (normals, eigenvalues, scales, quaternions)
    ):
        raise ValueError("MipMap initialization geometry must be finite")
    if np.any(scales <= 0.0):
        raise ValueError("MipMap initialization scales must be positive")
    normal_norm = np.linalg.norm(normals, axis=1)
    quaternion_norm = np.linalg.norm(quaternions, axis=1)
    if np.any(normal_norm <= 1e-8) or not np.allclose(
        quaternion_norm, 1.0, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("MipMap initialization normals/quaternions are invalid")
    if not np.allclose(scales[:, 1], scales[:, 0], rtol=1e-5, atol=1e-8):
        raise ValueError("MipMap initialization tangent scales must be equal")
    if not np.allclose(scales[:, 2], 0.5 * scales[:, 0], rtol=1e-5, atol=1e-8):
        raise ValueError("MipMap initialization short scale must equal 0.5*d")
    return tuple(
        np.ascontiguousarray(array)
        for array in (normals, eigenvalues, scales, quaternions)
    )


def _z_axis_quaternions(normals: np.ndarray) -> np.ndarray:
    """Return normalized wxyz quaternions rotating local +z onto each normal."""

    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    # Normal direction is physically unoriented for a covariance.  Keep it in
    # the +z hemisphere so the closed-form shortest-arc quaternion never meets
    # its singular -z case and remains deterministic across platforms.
    flip = unit[:, 2] < 0.0
    unit[flip] *= -1.0
    quaternions = np.column_stack(
        [
            1.0 + unit[:, 2],
            -unit[:, 1],
            unit[:, 0],
            np.zeros(len(unit), dtype=np.float32),
        ]
    ).astype(np.float32)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def build_surface_aligned_initialization(
    isotropic_scales_m: np.ndarray,
    normals: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    config: SurfaceInitializationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build metric xyz scales and wxyz rotations for planar LiDAR points."""

    config.validate()
    scales = np.asarray(isotropic_scales_m, dtype=np.float32)
    normal_array = np.asarray(normals, dtype=np.float32)
    values = np.asarray(eigenvalues, dtype=np.float32)
    if scales.ndim != 1 or normal_array.shape != (len(scales), 3) or values.shape != (
        len(scales),
        3,
    ):
        raise ValueError("surface initialization expects scales [N], normals/eigenvalues [N, 3]")
    if (
        not np.isfinite(scales).all()
        or not np.isfinite(normal_array).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("surface initialization inputs must be finite")
    if np.any(scales <= 0.0):
        raise ValueError("surface initialization scales must be positive")
    normal_norm = np.linalg.norm(normal_array, axis=1)
    if np.any(normal_norm <= 1e-8):
        raise ValueError("surface initialization normals must be non-zero")

    nonnegative = np.maximum(values, 0.0)
    trace = nonnegative.sum(axis=1)
    planarity = np.zeros(len(scales), dtype=np.float32)
    valid_trace = trace > 1e-12
    planarity[valid_trace] = np.clip(
        1.0 - 3.0 * nonnegative[valid_trace, 0] / trace[valid_trace],
        0.0,
        1.0,
    )
    planar = planarity >= float(config.planarity_gate)

    output_scales = np.repeat(scales[:, None], 3, axis=1)
    output_quaternions = np.zeros((len(scales), 4), dtype=np.float32)
    output_quaternions[:, 0] = 1.0
    if np.any(planar):
        short = np.maximum(
            scales[planar] * float(config.normal_scale_ratio),
            float(config.minimum_normal_scale_m),
        )
        output_scales[planar, 2] = np.minimum(short, scales[planar])
        output_quaternions[planar] = _z_axis_quaternions(normal_array[planar])

    report = {
        "schema_version": 1,
        "algorithm": "lidar_pca_planar_surfel_v1",
        "configuration": config.to_dict(),
        "point_count": int(len(scales)),
        "surface_aligned_count": int(np.count_nonzero(planar)),
        "surface_aligned_fraction": float(np.mean(planar)),
        "planarity": _distribution(planarity),
        "tangent_scale_m": _distribution(output_scales[:, 0]),
        "normal_scale_m": _distribution(output_scales[:, 2]),
        "aspect_ratio": _distribution(
            output_scales[:, 0] / np.maximum(output_scales[:, 2], 1e-12)
        ),
    }
    return output_scales, output_quaternions, report
