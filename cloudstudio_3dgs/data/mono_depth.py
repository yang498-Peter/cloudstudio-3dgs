"""Deterministic DA2 relative-depth cache and MipMap-compatible alignment."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.depth_cache import deterministic_npz_bytes
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


MONO_DEPTH_SCHEMA_VERSION = 1
MONO_DEPTH_KIND = "face4_da2_relative_depth_cache"


@dataclass(frozen=True)
class AffineAlignmentConfig:
    minimum_pairs: int = 1001
    iterations: int = 2000
    slope_min: float = 0.01
    slope_max: float = 100.0
    relative_error: float = 0.01
    minimum_inlier_ratio: float = 0.05


def fit_metric_affine_ransac(
    mono_depth: np.ndarray,
    metric_depth: np.ndarray,
    *,
    seed: int,
    config: AffineAlignmentConfig = AffineAlignmentConfig(),
) -> dict[str, Any]:
    """Fit ``metric = scale * mono + shift`` with the recovered product rules."""
    mono = np.asarray(mono_depth, dtype=np.float64).reshape(-1)
    metric = np.asarray(metric_depth, dtype=np.float64).reshape(-1)
    keep = np.isfinite(mono) & (mono > 0.0)
    keep &= np.isfinite(metric) & (metric > 0.0)
    x = mono[keep]
    y = metric[keep]
    count = int(len(x))
    base: dict[str, Any] = {
        "valid": False,
        "pair_count": count,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "scale": None,
        "shift": None,
        "rmse_m": None,
    }
    if count < config.minimum_pairs:
        return {**base, "reason": "insufficient_positive_pairs"}
    rng = np.random.default_rng(int(seed))
    best = np.zeros(count, dtype=bool)
    best_count = 0
    for _ in range(config.iterations):
        i, j = rng.choice(count, size=2, replace=False)
        denominator = x[j] - x[i]
        if abs(denominator) <= np.finfo(np.float64).eps:
            continue
        scale = (y[j] - y[i]) / denominator
        if not (config.slope_min < scale < config.slope_max):
            continue
        shift = y[i] - scale * x[i]
        prediction = scale * x + shift
        relative = np.abs(prediction - y) / np.maximum(np.abs(y), 1e-12)
        inliers = relative < config.relative_error
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count > best_count:
            best = inliers
            best_count = inlier_count
    ratio = best_count / count
    if best_count < 2 or not (ratio > config.minimum_inlier_ratio):
        return {
            **base,
            "inlier_count": best_count,
            "inlier_ratio": ratio,
            "reason": "ransac_inlier_ratio_not_met",
        }
    design = np.column_stack([x[best], np.ones(best_count, dtype=np.float64)])
    scale, shift = np.linalg.lstsq(design, y[best], rcond=None)[0]
    if not (config.slope_min < scale < config.slope_max):
        return {
            **base,
            "inlier_count": best_count,
            "inlier_ratio": ratio,
            "reason": "refit_slope_out_of_range",
        }
    residual = scale * x[best] + shift - y[best]
    return {
        "valid": True,
        "reason": "ok",
        "pair_count": count,
        "inlier_count": best_count,
        "inlier_ratio": ratio,
        "scale": float(scale),
        "shift": float(shift),
        "rmse_m": float(np.sqrt(np.mean(residual * residual))),
    }


def fit_metric_affine_ransac_torch(
    mono_depth: np.ndarray,
    metric_depth: np.ndarray,
    *,
    seed: int,
    device: str = "cuda",
    candidate_batch_size: int = 32,
    config: AffineAlignmentConfig = AffineAlignmentConfig(),
) -> dict[str, Any]:
    """GPU-batched equivalent of :func:`fit_metric_affine_ransac`."""
    import torch

    mono = np.asarray(mono_depth, dtype=np.float64).reshape(-1)
    metric = np.asarray(metric_depth, dtype=np.float64).reshape(-1)
    keep = np.isfinite(mono) & (mono > 0.0)
    keep &= np.isfinite(metric) & (metric > 0.0)
    x = mono[keep]
    y = metric[keep]
    count = int(len(x))
    base: dict[str, Any] = {
        "valid": False,
        "pair_count": count,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "scale": None,
        "shift": None,
        "rmse_m": None,
    }
    if count < config.minimum_pairs:
        return {**base, "reason": "insufficient_positive_pairs"}
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    rng = np.random.default_rng(int(seed))
    pairs = np.asarray(
        [rng.choice(count, size=2, replace=False) for _ in range(config.iterations)],
        dtype=np.int64,
    )
    denominator = x[pairs[:, 1]] - x[pairs[:, 0]]
    scales = np.divide(
        y[pairs[:, 1]] - y[pairs[:, 0]],
        denominator,
        out=np.full(config.iterations, np.nan, dtype=np.float64),
        where=np.abs(denominator) > np.finfo(np.float64).eps,
    )
    shifts = y[pairs[:, 0]] - scales * x[pairs[:, 0]]
    valid_candidates = np.isfinite(scales)
    valid_candidates &= scales > config.slope_min
    valid_candidates &= scales < config.slope_max
    scales = scales[valid_candidates]
    shifts = shifts[valid_candidates]
    if not len(scales):
        return {**base, "reason": "ransac_inlier_ratio_not_met"}
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    best_count = 0
    best_scale = 0.0
    best_shift = 0.0
    with torch.inference_mode():
        for start in range(0, len(scales), candidate_batch_size):
            stop = min(start + candidate_batch_size, len(scales))
            st = torch.as_tensor(
                scales[start:stop], dtype=torch.float32, device=device
            )[:, None]
            bt = torch.as_tensor(
                shifts[start:stop], dtype=torch.float32, device=device
            )[:, None]
            prediction = st * xt[None, :] + bt
            relative = torch.abs(prediction - yt[None, :]) / torch.clamp(
                torch.abs(yt[None, :]), min=1e-12
            )
            counts = torch.count_nonzero(
                relative < config.relative_error, dim=1
            )
            local_value, local_index = torch.max(counts, dim=0)
            local_count = int(local_value.item())
            if local_count > best_count:
                candidate = start + int(local_index.item())
                best_count = local_count
                best_scale = float(scales[candidate])
                best_shift = float(shifts[candidate])
    ratio = best_count / count
    if best_count < 2 or not (ratio > config.minimum_inlier_ratio):
        return {
            **base,
            "inlier_count": best_count,
            "inlier_ratio": ratio,
            "reason": "ransac_inlier_ratio_not_met",
        }
    relative = np.abs(best_scale * x + best_shift - y) / np.maximum(
        np.abs(y), 1e-12
    )
    best = relative < config.relative_error
    refit_count = int(np.count_nonzero(best))
    design = np.column_stack([x[best], np.ones(refit_count, dtype=np.float64)])
    scale, shift = np.linalg.lstsq(design, y[best], rcond=None)[0]
    if not (config.slope_min < scale < config.slope_max):
        return {
            **base,
            "inlier_count": best_count,
            "inlier_ratio": ratio,
            "reason": "refit_slope_out_of_range",
        }
    residual = scale * x[best] + shift - y[best]
    return {
        "valid": True,
        "reason": "ok",
        "pair_count": count,
        "inlier_count": refit_count,
        "inlier_ratio": refit_count / count,
        "scale": float(scale),
        "shift": float(shift),
        "rmse_m": float(np.sqrt(np.mean(residual * residual))),
    }


def sample_bilinear_at_source_pixels(
    image: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    *,
    source_shape: tuple[int, int],
) -> np.ndarray:
    """Sample a resized DA2 raster at pixel centers from its source face."""
    values = np.asarray(image, dtype=np.float32)
    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)
    source_h, source_w = source_shape
    target_h, target_w = values.shape
    x = (source_x + 0.5) * target_w / source_w - 0.5
    y = (source_y + 0.5) * target_h / source_h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.clip(x0, 0, target_w - 1)
    y0 = np.clip(y0, 0, target_h - 1)
    x1 = np.clip(x0 + 1, 0, target_w - 1)
    y1 = np.clip(y0 + 1, 0, target_h - 1)
    wx = np.clip(x - x0, 0.0, 1.0)
    wy = np.clip(y - y0, 0.0, 1.0)
    return (
        values[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + values[y0, x1] * wx * (1.0 - wy)
        + values[y1, x0] * (1.0 - wx) * wy
        + values[y1, x1] * wx * wy
    ).astype(np.float32)


def mono_depth_npz_bytes(relative_depth: np.ndarray) -> bytes:
    relative = np.asarray(relative_depth, dtype=np.float32)
    if relative.ndim != 2 or not np.all(np.isfinite(relative)):
        raise ValueError("relative depth must be a finite HxW array")
    if np.any(relative < 0.0):
        raise ValueError("relative depth cannot be negative")
    storage = np.minimum(relative, np.finfo(np.float16).max)
    return deterministic_npz_bytes(
        {
            "relative_depth": storage.astype("<f2"),
            "shape": np.asarray(relative.shape, dtype="<i4"),
        }
    )


def sign_mono_depth_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("mono_depth_manifest_sha256", None)
    signed = copy.deepcopy(unsigned)
    signed["mono_depth_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def verify_mono_depth_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("mono_depth_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("mono depth manifest is unsigned")
    unsigned = copy.deepcopy(manifest)
    unsigned.pop("mono_depth_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("mono depth manifest signature mismatch")
    if int(manifest.get("schema_version", -1)) != MONO_DEPTH_SCHEMA_VERSION:
        raise ValueError("unsupported mono depth manifest schema")
    if manifest.get("kind") != MONO_DEPTH_KIND:
        raise ValueError("unexpected mono depth manifest kind")
    return actual
