"""Metric geometry regularization for raw-fisheye Gaussian training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryRegularizationConfig:
    """Penalize fog, giant splats, and needle splats without fixing geometry."""

    enabled: bool = True
    opacity_sparsity_weight: float = 1e-4
    scale_upper_weight: float = 1e-4
    anisotropy_weight: float = 1e-4
    max_scale_ratio_to_reference: float = 8.0
    max_anisotropy: float = 10.0

    def validate(self) -> None:
        weights = (
            self.opacity_sparsity_weight,
            self.scale_upper_weight,
            self.anisotropy_weight,
        )
        if any(float(value) < 0.0 for value in weights):
            raise ValueError("geometry regularization weights must be non-negative")
        if self.max_scale_ratio_to_reference <= 1.0:
            raise ValueError("max_scale_ratio_to_reference must exceed one")
        if self.max_anisotropy <= 1.0:
            raise ValueError("max_anisotropy must exceed one")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": self.enabled,
            "opacity_sparsity_weight": self.opacity_sparsity_weight,
            "scale_upper_weight": self.scale_upper_weight,
            "anisotropy_weight": self.anisotropy_weight,
            "max_scale_ratio_to_reference": self.max_scale_ratio_to_reference,
            "max_anisotropy": self.max_anisotropy,
        }


def geometry_regularization_terms(
    params: Any,
    *,
    reference_scale_m: float,
    config: GeometryRegularizationConfig,
) -> dict[str, Any]:
    """Return differentiable, metric-space penalties and their weighted sum."""
    config.validate()
    if reference_scale_m <= 0.0 or not math.isfinite(reference_scale_m):
        raise ValueError("reference_scale_m must be finite and positive")
    required = {"opacities", "scales"}
    if not required <= set(params):
        raise ValueError("geometry regularization requires opacities and scales")
    torch = __import__("torch")
    scales = torch.exp(params["scales"])
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("geometry regularization expects [N,3] log scales")
    opacity = torch.sigmoid(params["opacities"]).reshape(-1)
    if opacity.shape[0] != scales.shape[0]:
        raise ValueError("opacity and scale counts differ")
    zero = scales.new_zeros(())
    if not config.enabled:
        return {
            "opacity_sparsity": zero,
            "scale_upper": zero,
            "anisotropy": zero,
            "total": zero,
        }
    # Mean opacity pushes unsupported translucent fog toward MCMC's prune path.
    opacity_sparsity = opacity.mean()
    max_scale = scales.max(dim=1).values
    min_scale = scales.min(dim=1).values.clamp_min(1e-12)
    # Soft barriers begin only outside metric bounds, preserving normal geometry.
    scale_upper = torch.relu(
        torch.log(max_scale / float(reference_scale_m))
        - math.log(config.max_scale_ratio_to_reference)
    ).square().mean()
    anisotropy = torch.relu(
        torch.log(max_scale / min_scale) - math.log(config.max_anisotropy)
    ).square().mean()
    total = (
        config.opacity_sparsity_weight * opacity_sparsity
        + config.scale_upper_weight * scale_upper
        + config.anisotropy_weight * anisotropy
    )
    return {
        "opacity_sparsity": opacity_sparsity,
        "scale_upper": scale_upper,
        "anisotropy": anisotropy,
        "total": total,
    }
