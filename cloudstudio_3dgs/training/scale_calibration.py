"""Metric KNN scale initialization and MCMC LR/noise calibration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


@dataclass(frozen=True)
class MetricScaleCalibrationConfig:
    """Bind Gaussian scale, means LR and MCMC noise to local metric spacing.

    Upstream MCMC perturbs a mean by approximately ``covariance * normal *
    means_lr * noise_lr``. With isotropic scale ``s``, the nominal displacement
    is therefore ``s^2 * means_lr * noise_lr``. A fixed noise LR is not
    invariant when an unnormalised metric scene or its point spacing changes.
    """

    mode: str = "knn"
    knn_neighbors: int = 3
    scale_multiplier: float = 1.0
    clamp_min_ratio: float = 0.25
    clamp_max_ratio: float = 4.0
    means_step_fraction: float | None = 0.0032
    noise_std_fraction: float | None = 0.25

    def validate(self) -> None:
        if self.mode not in {"fixed", "knn"}:
            raise ValueError("metric scale mode must be 'fixed' or 'knn'")
        if self.knn_neighbors <= 0:
            raise ValueError("KNN scale neighbors must be positive")
        if self.scale_multiplier <= 0.0:
            raise ValueError("KNN scale multiplier must be positive")
        if self.clamp_min_ratio <= 0.0 or self.clamp_max_ratio < self.clamp_min_ratio:
            raise ValueError("KNN scale clamp ratios must be positive and ordered")
        if self.means_step_fraction is not None and self.means_step_fraction <= 0.0:
            raise ValueError("means step fraction must be positive or null")
        if self.noise_std_fraction is not None and self.noise_std_fraction <= 0.0:
            raise ValueError("MCMC noise standard-deviation fraction must be positive or null")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mode": self.mode,
            "knn_neighbors": self.knn_neighbors,
            "scale_multiplier": self.scale_multiplier,
            "clamp_min_ratio": self.clamp_min_ratio,
            "clamp_max_ratio": self.clamp_max_ratio,
            "means_step_fraction": self.means_step_fraction,
            "noise_std_fraction": self.noise_std_fraction,
        }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _knn_scales(points: np.ndarray, policy: MetricScaleCalibrationConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if len(points) <= policy.knn_neighbors:
        raise ValueError(
            f"KNN scale needs more than {policy.knn_neighbors} initialization points"
        )
    tree = cKDTree(points)
    distances, _ = tree.query(
        points,
        k=policy.knn_neighbors + 1,
        workers=1,
    )
    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float64)
    raw = np.sqrt(np.mean(np.square(neighbor_distances), axis=1))
    positive = raw[np.isfinite(raw) & (raw > 0.0)]
    if not len(positive):
        raise ValueError("KNN scale calibration found no positive neighbor spacing")
    replacement = float(np.median(positive))
    invalid = ~np.isfinite(raw) | (raw <= 0.0)
    raw[invalid] = replacement
    raw *= float(policy.scale_multiplier)
    reference = float(np.median(raw))
    minimum = reference * float(policy.clamp_min_ratio)
    maximum = reference * float(policy.clamp_max_ratio)
    clipped = np.clip(raw, minimum, maximum).astype(np.float32)
    return clipped, {
        "invalid_replaced_count": int(np.count_nonzero(invalid)),
        "clipped_count": int(np.count_nonzero((raw < minimum) | (raw > maximum))),
        "raw_scale_distribution_m": _distribution(raw),
        "clamp_min_m": minimum,
        "clamp_max_m": maximum,
    }


def build_metric_scale_calibration(
    xyz: np.ndarray,
    *,
    policy: MetricScaleCalibrationConfig,
    fixed_scale_m: float,
    configured_means_lr: float,
    configured_noise_lr: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-Gaussian metric scales and a signed effective-parameter report."""

    policy.validate()
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("metric scale calibration expects at least four XYZ points")
    if not np.all(np.isfinite(points)):
        raise ValueError("metric scale calibration XYZ must be finite")
    if fixed_scale_m <= 0.0 or configured_means_lr <= 0.0 or configured_noise_lr < 0.0:
        raise ValueError("fixed scale/means LR must be positive and noise LR non-negative")

    diagnostics: dict[str, Any]
    if policy.mode == "fixed":
        scales = np.full(len(points), float(fixed_scale_m), dtype=np.float32)
        diagnostics = {
            "invalid_replaced_count": 0,
            "clipped_count": 0,
            "raw_scale_distribution_m": _distribution(scales),
            "clamp_min_m": None,
            "clamp_max_m": None,
        }
    else:
        scales, diagnostics = _knn_scales(points, policy)

    reference_scale_m = float(np.median(scales))
    effective_means_lr = (
        float(configured_means_lr)
        if policy.means_step_fraction is None
        else reference_scale_m * float(policy.means_step_fraction)
    )
    effective_noise_lr = (
        float(configured_noise_lr)
        if policy.noise_std_fraction is None
        else float(policy.noise_std_fraction) / (reference_scale_m * effective_means_lr)
    )
    nominal_noise_std_m = (
        reference_scale_m * reference_scale_m * effective_means_lr * effective_noise_lr
    )
    unsigned = {
        "schema_version": 1,
        "policy": policy.to_dict(),
        "point_count": int(len(points)),
        "reference_scale_m": reference_scale_m,
        "scale_distribution_m": _distribution(scales),
        **diagnostics,
        "configured_fixed_scale_m": float(fixed_scale_m),
        "configured_means_lr": float(configured_means_lr),
        "configured_noise_lr": float(configured_noise_lr),
        "effective_means_lr_m": effective_means_lr,
        "effective_noise_lr": effective_noise_lr,
        "nominal_noise_std_m": nominal_noise_std_m,
        "nominal_noise_std_fraction": nominal_noise_std_m / reference_scale_m,
    }
    report = dict(unsigned)
    report["scale_calibration_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return scales, report


def verify_metric_scale_calibration_report(report: dict[str, Any]) -> str:
    """Verify the canonical report hash and its core dimensional invariants."""

    unsigned = dict(report)
    observed = unsigned.pop("scale_calibration_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise ValueError("metric scale calibration SHA256 mismatch")
    if unsigned.get("schema_version") != 1:
        raise ValueError("unsupported metric scale calibration schema")
    point_count = int(unsigned.get("point_count", 0))
    reference = float(unsigned.get("reference_scale_m", 0.0))
    means_lr = float(unsigned.get("effective_means_lr_m", 0.0))
    noise_lr = float(unsigned.get("effective_noise_lr", -1.0))
    nominal_m = float(unsigned.get("nominal_noise_std_m", -1.0))
    nominal_fraction = float(unsigned.get("nominal_noise_std_fraction", -1.0))
    if point_count < 4 or min(reference, means_lr, nominal_m, nominal_fraction) <= 0.0:
        raise ValueError("metric scale calibration contains non-positive effective values")
    if noise_lr < 0.0:
        raise ValueError("metric scale calibration noise LR is negative")
    recomputed_m = reference * reference * means_lr * noise_lr
    if not np.isclose(nominal_m, recomputed_m, rtol=1e-12, atol=0.0):
        raise ValueError("metric scale calibration nominal noise distance is inconsistent")
    if not np.isclose(nominal_fraction, nominal_m / reference, rtol=1e-12, atol=0.0):
        raise ValueError("metric scale calibration nominal noise fraction is inconsistent")
    return expected
