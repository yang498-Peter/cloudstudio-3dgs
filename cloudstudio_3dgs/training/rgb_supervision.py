"""Photometric ownership mask: which pixels a Tile may be asked to paint.

The per-view backdrops of the Tile route are rendered from a sky dome only,
so photo pixels that show trees, neighbouring Tiles or anything else the Tile
has no geometry for can only be explained by the Tile's own Gaussians. Under
full-frame supervision those Gaussians are grown at arbitrary depth in front
of the real surfaces and occlude other views. ``rgb_supervision_mask``:

``all``
    Every renderer-mask pixel is supervised (the historical behaviour; the
    loss code path and the trainer contract are byte-identical).

``lidar_support``
    The photo-derived terms are computed only where the signed LiDAR support
    of the view, grown by ``dilation_radius_px`` (the same construction as
    the dilated alpha support: valid depth & confidence > 0, max-pooled), is
    true AND the renderer mask allows. Pixels outside get zero loss and zero
    gradient; the terms are normalised by the number of supervised pixels so
    the loss scale stays comparable across views.

The masked terms are the RGB L1, the SSIM (local or global), the RGB
gradient L1, the LiDAR-supported RGB L1 and the DA2 depth term. The DA2 term
is geometric, but its target is monocular depth predicted from the photo and
scale/shift-aligned on the LiDAR returns of the view; outside the LiDAR
support it is an extrapolated prediction of content the Tile cannot own, so
it is masked with the photometric terms rather than left to ask for a
surface where the RGB term has just been silenced. The LiDAR range and alpha
terms are untouched (they are already support-only by construction) and the
exposure gain is applied exactly as before.

Two twins on purpose, as for the alpha support: a torch version for the
trainer and a numpy version for CPU audits (``tools/estimate_rgb_supervision
_fraction.py``); ``tests/test_rgb_supervision_mask.py`` pins them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cloudstudio_3dgs.training.alpha_support import (
    lidar_alpha_support,
    lidar_alpha_support_numpy,
)


RGB_SUPERVISION_MASK_MODES: tuple[str, ...] = ("all", "lidar_support")

# Loss terms the mask restricts; recorded in the contract so a reader of a
# run manifest does not need this source to know what was silenced.
RGB_SUPERVISION_MASKED_TERMS: tuple[str, ...] = (
    "rgb_l1",
    "rgb_ssim",
    "rgb_gradient_l1",
    "lidar_rgb_l1",
    "da2_depth",
)


def rgb_supervision_mask_contract(*, mode: str, dilation_radius_px: int) -> dict[str, Any]:
    """The mask parameters as they enter the trainer contract (mode != all)."""
    if mode not in RGB_SUPERVISION_MASK_MODES:
        raise ValueError(f"unknown rgb_supervision_mask {mode!r}")
    return {
        "mode": mode,
        "source": "signed_lidar_depth_mask_confidence_max_dilated",
        "dilation_radius_px": int(dilation_radius_px),
        "combined_with": "rgb_mask & dilated_support",
        "masked_terms": list(RGB_SUPERVISION_MASKED_TERMS),
        "normalisation": "supervised_pixels",
        "empty_support_policy": "zero_photometric_loss_for_view",
        "exposure_gain": "applied_unchanged_before_masking",
    }


@dataclass(frozen=True)
class RgbSupervision:
    """``mask`` is the supervised-pixel mask, ``fraction`` its share of rgb_mask."""

    mask: Any
    supervised_pixels: int
    rgb_mask_pixels: int

    @property
    def fraction(self) -> float:
        if self.rgb_mask_pixels <= 0:
            return 0.0
        return float(self.supervised_pixels) / float(self.rgb_mask_pixels)


def rgb_supervision_mask(
    torch: Any,
    *,
    rgb_mask: Any,
    depth_mask: Any | None,
    confidence: Any | None,
    mode: str,
    dilation_radius_px: int,
) -> RgbSupervision:
    """Torch construction of the supervised-pixel mask for one view.

    ``mode == "all"`` returns ``rgb_mask`` itself (the same tensor object),
    so the default path stays byte-identical. In ``lidar_support`` a view
    without any signed LiDAR return (``depth_mask is None``) has an empty
    supervised set.
    """
    if mode not in RGB_SUPERVISION_MASK_MODES:
        raise ValueError(f"unknown rgb_supervision_mask {mode!r}")
    rgb_pixels = int(rgb_mask.sum().item())
    if mode == "all":
        return RgbSupervision(mask=rgb_mask, supervised_pixels=rgb_pixels, rgb_mask_pixels=rgb_pixels)
    if int(dilation_radius_px) <= 0:
        raise ValueError("lidar_support rgb supervision requires a positive dilation radius")
    if depth_mask is None or confidence is None:
        mask = torch.zeros_like(rgb_mask)
        return RgbSupervision(mask=mask, supervised_pixels=0, rgb_mask_pixels=rgb_pixels)
    support = lidar_alpha_support(
        torch,
        depth_mask=depth_mask,
        confidence=confidence,
        range_m=None,
        mode="dilated",
        dilation_radius_px=int(dilation_radius_px),
    ).support
    mask = rgb_mask & support
    return RgbSupervision(
        mask=mask, supervised_pixels=int(mask.sum().item()), rgb_mask_pixels=rgb_pixels
    )


def rgb_supervision_mask_numpy(
    *,
    rgb_mask: np.ndarray,
    depth_mask: np.ndarray | None,
    confidence: np.ndarray | None,
    mode: str,
    dilation_radius_px: int,
) -> RgbSupervision:
    """numpy twin of :func:`rgb_supervision_mask` (same mask, same counts)."""
    if mode not in RGB_SUPERVISION_MASK_MODES:
        raise ValueError(f"unknown rgb_supervision_mask {mode!r}")
    rgb_mask = np.asarray(rgb_mask, dtype=bool)
    rgb_pixels = int(rgb_mask.sum())
    if mode == "all":
        return RgbSupervision(mask=rgb_mask, supervised_pixels=rgb_pixels, rgb_mask_pixels=rgb_pixels)
    if int(dilation_radius_px) <= 0:
        raise ValueError("lidar_support rgb supervision requires a positive dilation radius")
    if depth_mask is None or confidence is None:
        mask = np.zeros_like(rgb_mask)
        return RgbSupervision(mask=mask, supervised_pixels=0, rgb_mask_pixels=rgb_pixels)
    support = lidar_alpha_support_numpy(
        depth_mask=np.asarray(depth_mask, dtype=bool),
        confidence=np.asarray(confidence, dtype=np.float32),
        range_m=None,
        mode="dilated",
        dilation_radius_px=int(dilation_radius_px),
    ).support
    mask = rgb_mask & support
    return RgbSupervision(mask=mask, supervised_pixels=int(mask.sum()), rgb_mask_pixels=rgb_pixels)
