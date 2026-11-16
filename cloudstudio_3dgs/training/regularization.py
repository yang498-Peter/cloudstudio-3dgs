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
    opacity_sparsity_scope: str = "all"
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
    world_shrink_factor: float | None = None
    # Hard ceiling on a single gaussian's opacity, applied every step with the
    # other in-place clamps. house0305 (2026-09-29): 34% of our delivered
    # gaussians sit above 0.9 against 2% in the competitor, whose solid surfaces
    # are stacks of 0.1-0.4 layers; opaque blades show their outlines, which is
    # what reads as "gaussian texture". None keeps every contract unchanged.
    max_opacity: float | None = None
    # Lower bound on every axis of a gaussian, in pixels of the finest view that has
    # rasterized it: sigma >= min_extent_px * depth / focal at that view. A training
    # view cannot tell a 0.2 px thick disk from a 0.5 px one seen face-on, so nothing in
    # the loss stops a disk thinning until, from an angle no photo covers, it is a
    # needle; Mip-Splatting (Yu et al. 2024, sec. 5.1) bounds the 3D extent by the
    # maximal sampling rate for that reason and uses variance 0.2 px^2, i.e. 0.447 px.
    # house0305: our delivered short axis is 0.26-0.29 mm at the median against 0.43 mm
    # in the reference, and 0.447 px of the 787.6 px/rad sensor is 0.57 mm at 1 m.
    # Applied as an in-place clamp with the other clamps, so the parameters are what
    # every downstream tool renders and nothing has to be baked at export. 0 = off.
    min_extent_px: float = 0.0

    def validate(self) -> None:
        weights = (
            self.opacity_sparsity_weight,
            self.scale_upper_weight,
            self.anisotropy_weight,
        )
        if any(float(value) < 0.0 for value in weights):
            raise ValueError("geometry regularization weights must be non-negative")
        if self.opacity_sparsity_scope not in {"all", "visible_current_view"}:
            raise ValueError(
                "opacity_sparsity_scope must be 'all' or 'visible_current_view'"
            )
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
        if self.max_opacity is not None and not 0.0 < float(self.max_opacity) < 1.0:
            raise ValueError("max_opacity must be within (0, 1)")
        if not 0.0 <= float(self.min_extent_px) <= 4.0:
            raise ValueError("min_extent_px must be within [0, 4]")
        if self.world_shrink_factor is not None:
            if not 0.0 < self.world_shrink_factor < 1.0:
                raise ValueError("world_shrink_factor must be within (0, 1)")
            if self.max_world_size_m is None:
                raise ValueError(
                    "world_shrink_factor requires max_world_size_m as its trigger"
                )

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
            "world_shrink_factor": self.world_shrink_factor,
        }
        if self.opacity_sparsity_scope != "all":
            result["opacity_sparsity_scope"] = self.opacity_sparsity_scope
        # Omit the compatibility value so existing Australian P5 contracts and
        # checkpoints remain byte-for-byte identifiable. Opt-in tail risk is
        # still explicit and signed.
        if self.scale_upper_tail_fraction != 1.0:
            result["scale_upper_tail_fraction"] = self.scale_upper_tail_fraction
        if self.max_opacity is not None:
            result["max_opacity"] = float(self.max_opacity)
        if self.min_extent_px > 0.0:
            result["min_extent_px"] = float(self.min_extent_px)
        return result


def geometry_regularization_terms(
    params: Any,
    *,
    reference_scale_m: float,
    config: GeometryRegularizationConfig,
    visibility_mask: Any | None = None,
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
            "opacity_sparsity_active_count": 0,
            "opacity_sparsity_active_fraction": zero,
            "scale_upper": zero,
            "scale_over_limit_fraction": zero,
            "scale_upper_tail_count": 0,
            "anisotropy": zero,
            "total": zero,
        }
    # Mean opacity pushes unsupported translucent fog toward the prune path.
    # The recovered vendor route applies it to every active Gaussian.  That is
    # a poor LiDAR-first warm-up when one cropped Tile view sees only a subset
    # of a nearly one-point-per-return initialization: an invisible surface
    # receives a negative opacity gradient on hundreds of unrelated views and
    # no compensating RGB gradient.  The opt-in visible scope preserves the
    # same loss on the current raster support while removing that sampling
    # bias; it does not add an alpha target or protect any local voxel.
    opacity_active_count = int(opacity.shape[0])
    opacity_active_fraction = opacity.new_ones(())
    if config.opacity_sparsity_scope == "visible_current_view":
        if visibility_mask is None:
            raise ValueError(
                "visible_current_view opacity sparsity requires a visibility mask"
            )
        visible = torch.as_tensor(
            visibility_mask,
            dtype=torch.bool,
            device=opacity.device,
        ).reshape(-1)
        if visible.shape != opacity.shape:
            raise ValueError("visibility mask and opacity counts differ")
        opacity_active_count = int(visible.sum().item())
        opacity_active_fraction = visible.to(opacity.dtype).mean()
        opacity_sparsity = (
            opacity[visible].mean() if opacity_active_count else zero
        )
    else:
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
        "opacity_sparsity_active_count": opacity_active_count,
        "opacity_sparsity_active_fraction": opacity_active_fraction,
        "scale_upper": scale_upper,
        "scale_over_limit_fraction": scale_over_limit_fraction,
        "scale_upper_tail_count": int(tail_count),
        "anisotropy": anisotropy,
        "total": total,
    }


#: Per-gaussian running maximum of focal / depth (pixels per metre) over the views that
#: rasterized it. Lives in the strategy state, so duplicate / split / remove carry it and
#: a checkpoint resumes it.
SAMPLING_RATE_KEY = "_cloudstudio_max_sampling_rate"
#: Depths nearer than this do not raise the sampling rate (a gaussian grazing the camera
#: would otherwise earn a floor of micrometres and keep it).
SAMPLING_RATE_MIN_DEPTH_M = 0.05


def observe_sampling_rate(
    state: dict[str, Any],
    *,
    radii: Any,
    depths: Any,
    focal_px: float,
    gaussian_count: int,
) -> Any:
    """Fold one view into each visible gaussian's maximal sampling rate.

    Must run while ``radii`` / ``depths`` still line up with the population, i.e.
    before the step's lifecycle event. Returns the state tensor.
    """
    torch = __import__("torch")
    if focal_px <= 0.0 or not math.isfinite(focal_px):
        raise ValueError("focal_px must be finite and positive")
    radius = torch.as_tensor(radii).detach()
    if radius.ndim >= 2 and radius.shape[-1] == 2:
        radius = radius.max(dim=-1).values
    radius = radius.reshape(-1)
    depth = torch.as_tensor(depths).detach().reshape(-1)
    if radius.shape[0] != gaussian_count or depth.shape[0] != gaussian_count:
        raise ValueError("radii / depths do not match the gaussian count")
    rate = state.get(SAMPLING_RATE_KEY)
    if rate is None or rate.shape[0] != gaussian_count:
        rate = torch.zeros(gaussian_count, dtype=torch.float32, device=depth.device)
        state[SAMPLING_RATE_KEY] = rate
    with torch.no_grad():
        visible = (radius > 0) & torch.isfinite(depth) & (depth > SAMPLING_RATE_MIN_DEPTH_M)
        observed = torch.where(
            visible, float(focal_px) / depth.clamp_min(SAMPLING_RATE_MIN_DEPTH_M).to(rate.dtype), rate
        )
        torch.maximum(rate, observed, out=rate)
    return rate


def apply_min_extent(
    params: Any,
    state: dict[str, Any],
    *,
    min_extent_px: float,
) -> dict[str, Any]:
    """Raise every axis below ``min_extent_px / max sampling rate`` to it, in place.

    Gaussians no view has rasterized yet have no rate and are left alone.
    """
    torch = __import__("torch")
    report = {"min_extent_raised_count": 0, "min_extent_observed_count": 0}
    rate = state.get(SAMPLING_RATE_KEY)
    if min_extent_px <= 0.0 or rate is None or rate.shape[0] != params["scales"].shape[0]:
        return report
    with torch.no_grad():
        observed = rate > 0
        floor = torch.where(
            observed,
            torch.log(float(min_extent_px) / rate.clamp_min(1e-12)),
            torch.full_like(rate, float("-inf")),
        ).to(params["scales"].dtype)
        below = params["scales"] < floor[:, None]
        report["min_extent_raised_count"] = int(below.any(dim=1).sum())
        report["min_extent_observed_count"] = int(observed.sum())
        torch.maximum(params["scales"], floor[:, None], out=params["scales"].data)
    return report


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
            if config.world_shrink_factor is None:
                params["scales"].clamp_(max=bound)
            else:
                # Per-axis clamping flattens the two long axes of an oversized
                # splat into each other, destroying the anisotropy the shape
                # regularizers are building. Scaling every axis of the offending
                # splat by a constant keeps its aspect ratio and walks it under
                # the bound over successive steps instead of snapping it.
                selected = over.any(dim=1)
                shrunk = int(selected.sum())
                if shrunk:
                    params["scales"][selected] += math.log(
                        config.world_shrink_factor
                    )
                report["world_shrunk_count"] = shrunk
        if config.max_opacity is not None:
            # Opacities are stored as logits; clamp the logit so no single
            # gaussian can exceed the ceiling. Adam state is left alone, as for
            # the scale clamps above.
            bound = math.log(float(config.max_opacity) / (1.0 - float(config.max_opacity)))
            over = params["opacities"] > bound
            capped = int(over.sum())
            report["opacity_capped_count"] = capped
            if capped:
                params["opacities"].clamp_(max=bound)
    return report
