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
    scale_upper_tail_fraction: float = 1.0
    anisotropy_weight: float = 1e-4
    max_scale_ratio_to_reference: float = 8.0
    max_anisotropy: float = 10.0
    screen_clip_enabled: bool = False
    max_screen_fraction: float = 0.15
    screen_clip_hardness: float = 1.5
    screen_clip_opacity_bump: float = 3.0
    max_world_size_m: float | None = None

    def validate(self) -> None:
        weights = (
            self.opacity_sparsity_weight,
            self.scale_upper_weight,
            self.anisotropy_weight,
        )
        if any(float(value) < 0.0 for value in weights):
            raise ValueError("geometry regularization weights must be non-negative")
        if not 0.0 < self.scale_upper_tail_fraction <= 1.0:
            raise ValueError("scale_upper_tail_fraction must be within (0, 1]")
        if self.max_scale_ratio_to_reference <= 1.0:
            raise ValueError("max_scale_ratio_to_reference must exceed one")
        if self.max_anisotropy <= 1.0:
            raise ValueError("max_anisotropy must exceed one")
        if not 0.0 < self.max_screen_fraction < 1.0:
            raise ValueError("max_screen_fraction must be within (0, 1)")
        if self.screen_clip_hardness <= 1.0:
            raise ValueError("screen_clip_hardness must exceed one")
        if self.screen_clip_opacity_bump < 0.0:
            raise ValueError("screen_clip_opacity_bump must be non-negative")
        if self.max_world_size_m is not None and self.max_world_size_m <= 0.0:
            raise ValueError("max_world_size_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = {
            "enabled": self.enabled,
            "opacity_sparsity_weight": self.opacity_sparsity_weight,
            "scale_upper_weight": self.scale_upper_weight,
            "anisotropy_weight": self.anisotropy_weight,
            "max_scale_ratio_to_reference": self.max_scale_ratio_to_reference,
            "max_anisotropy": self.max_anisotropy,
            "screen_clip_enabled": self.screen_clip_enabled,
            "max_screen_fraction": self.max_screen_fraction,
            "screen_clip_hardness": self.screen_clip_hardness,
            "screen_clip_opacity_bump": self.screen_clip_opacity_bump,
            "max_world_size_m": self.max_world_size_m,
        }
        # Omit the compatibility value so existing Australian P5 contracts and
        # checkpoints remain byte-for-byte identifiable. Opt-in tail risk is
        # still explicit and signed.
        if self.scale_upper_tail_fraction != 1.0:
            result["scale_upper_tail_fraction"] = self.scale_upper_tail_fraction
        return result


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
            "scale_over_limit_fraction": zero,
            "scale_upper_tail_count": 0,
            "anisotropy": zero,
            "total": zero,
        }
    # Mean opacity pushes unsupported translucent fog toward MCMC's prune path.
    opacity_sparsity = opacity.mean()
    max_scale = scales.max(dim=1).values
    min_scale = scales.min(dim=1).values.clamp_min(1e-12)
    # Soft barriers begin only outside metric bounds, preserving normal geometry.
    scale_upper_per_gaussian = torch.relu(
        torch.log(max_scale / float(reference_scale_m))
        - math.log(config.max_scale_ratio_to_reference)
    ).square()
    tail_count = scales.shape[0]
    if config.scale_upper_tail_fraction < 1.0:
        tail_count = max(
            1,
            int(math.ceil(scales.shape[0] * config.scale_upper_tail_fraction)),
        )
        scale_upper = torch.topk(
            scale_upper_per_gaussian,
            k=tail_count,
            largest=True,
            sorted=False,
        ).values.mean()
    else:
        # Preserve the original reduction exactly for Australian P5.
        scale_upper = scale_upper_per_gaussian.mean()
    scale_over_limit_fraction = (
        max_scale
        > float(reference_scale_m) * config.max_scale_ratio_to_reference
    ).to(scales.dtype).mean()
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
        "scale_over_limit_fraction": scale_over_limit_fraction,
        "scale_upper_tail_count": int(tail_count),
        "anisotropy": anisotropy,
        "total": total,
    }


def clip_oversized_gaussians(
    params: Any,
    *,
    radii_px: Any,
    image_size_px: int,
    config: GeometryRegularizationConfig,
) -> dict[str, Any]:
    """Multiplicatively shrink splats that dominate the current view.

    A splat whose projected radius exceeds ``max_screen_fraction`` of the
    image's short side shrinks by log(oversize) per step (capped by
    ``screen_clip_hardness``), and under MCMC its opacity is bumped so the
    relocation sampler treats it as a split target instead of merely
    suppressing it. ``max_world_size_m`` is an unconditional metric fuse.
    Runs under no_grad on the parameter data, outside the loss graph.
    """
    config.validate()
    torch = __import__("torch")
    report = {"clipped_count": 0, "world_clamped_count": 0}
    if not config.enabled:
        return report
    with torch.no_grad():
        if config.screen_clip_enabled and radii_px is not None:
            radius = torch.as_tensor(radii_px, dtype=torch.float32)
            if radius.ndim >= 2 and radius.shape[-1] == 2:
                radius = radius.max(dim=-1).values
            radius = radius.reshape(-1).to(params["scales"].device)
            if radius.shape[0] != params["scales"].shape[0]:
                raise ValueError("radii do not match the gaussian count")
            fraction = radius / (float(config.max_screen_fraction) * float(image_size_px))
            oversize = fraction.clamp(max=float(config.screen_clip_hardness))
            selected = oversize > 1.0
            count = int(selected.sum())
            if count:
                shrink = torch.log(oversize[selected])
                params["scales"][selected] = (
                    params["scales"][selected] - shrink[:, None]
                )
                if config.screen_clip_opacity_bump > 0.0:
                    logits = params["opacities"][selected]
                    bumped = logits + config.screen_clip_opacity_bump * shrink
                    params["opacities"][selected] = torch.minimum(
                        bumped, torch.clamp_min(logits, 5.0)
                    )
            report["clipped_count"] = count
        if config.max_world_size_m is not None:
            bound = math.log(config.max_world_size_m)
            over = params["scales"] > bound
            report["world_clamped_count"] = int(over.any(dim=1).sum())
            params["scales"].clamp_(max=bound)
    return report
