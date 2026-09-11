"""LiDAR alpha-coverage support masks shared by the trainer and offline audits.

The alpha-coverage loss asks the rasterizer for alpha >= target on every pixel
the LiDAR "supports". Two constructions exist:

``dilated``
    The historical construction: the per-pixel LiDAR confidence is zero-filled
    where no signed return exists and max-pooled with a (2r+1)^2 window, and
    every pixel with a pooled confidence > 0 is supported. Every return grows a
    square of support regardless of what the neighbouring returns say, so at a
    depth discontinuity (door leaf against the corridor behind it, a pole in
    front of a wall) the near surface's silhouette is pushed r px outward over
    the far surface and vice versa.

``strict_visibility``
    The strict rule of ``tools/audit_observation_coverage.py``: a pixel is
    supported only when the returns in its window agree on one surface, i.e.
    the farthest return in the window lies within ``margin + tolerance * d``
    of the nearest return ``d``. Windows that mix a nearer and a farther
    surface are depth discontinuities and are rejected, and the rejection is
    grown by a small erosion radius so a window that by chance caught only
    the near surface's sparse returns cannot leak support across the edge.
    Interior pixels of a sparsely sampled surface keep their support because
    every return in their window agrees. The confidence weights are the same
    pooled confidence the dilated mode uses; only the mask changes.

Both modes are implemented twice on purpose: a torch version for the trainer
(``lidar_alpha_support``) and a numpy version for CPU-only audits
(``lidar_alpha_support_numpy``). ``tests/test_alpha_support.py`` pins them to
each other and pins ``dilated`` to the legacy inline construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ALPHA_SUPPORT_MODES: tuple[str, ...] = ("dilated", "strict_visibility")

# Strict band: the same numbers audit_observation_coverage.py uses for its
# "strict" view classification (STRICT_TOLERANCE / STRICT_MARGIN_M): a return
# supports a surface when it lies within 0.1 m + 3 % of range of it.
STRICT_VISIBILITY_TOLERANCE = 0.03
STRICT_VISIBILITY_MARGIN_M = 0.1
# Rejected discontinuity windows are grown by this radius before the support
# is taken, so sparse returns cannot leak a silhouette across an edge.
STRICT_VISIBILITY_EDGE_EROSION_PX = 3


def strict_visibility_contract(search_radius_px: int) -> dict[str, Any]:
    """The strict-mode parameters as they enter the run contract."""
    return {
        "search_radius_px": int(search_radius_px),
        "tolerance": STRICT_VISIBILITY_TOLERANCE,
        "margin_m": STRICT_VISIBILITY_MARGIN_M,
        "edge_erosion_px": STRICT_VISIBILITY_EDGE_EROSION_PX,
        "rule": (
            "window_returns_agree: farthest <= (1 + tolerance) * nearest + margin; "
            "disagreeing windows and their edge_erosion_px neighbourhood are rejected"
        ),
    }


@dataclass(frozen=True)
class AlphaSupport:
    """``support`` is the boolean mask, ``weights`` the pooled confidence.

    ``discontinuity`` marks windows whose returns disagree under the strict
    band (None in dilated mode); ``candidate`` is the dilated support the
    strict mask is a subset of.
    """

    support: Any
    weights: Any
    candidate: Any
    discontinuity: Any | None


def _max_pool(torch: Any, array: Any, radius: int) -> Any:
    """(2r+1)^2 max filter; padding behaves as -inf for floats, False for bools."""
    if radius <= 0:
        return array
    return torch.nn.functional.max_pool2d(
        array[None, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[0, 0]


def lidar_alpha_support(
    torch: Any,
    *,
    depth_mask: Any,
    confidence: Any,
    range_m: Any | None,
    mode: str,
    dilation_radius_px: int,
) -> AlphaSupport:
    """Torch construction of the alpha-coverage support for one view.

    ``mode == "dilated"`` reproduces the historical inline trainer code
    operation for operation (valid = depth_mask & finite & > 0, zero-fill,
    max-pool, > 0), so its outputs are bit-identical to that code.
    """
    if mode not in ALPHA_SUPPORT_MODES:
        raise ValueError(f"unknown lidar_alpha_support_mode {mode!r}")
    radius = int(dilation_radius_px)
    valid = depth_mask & torch.isfinite(confidence) & (confidence > 0.0)
    zero_filled = torch.where(valid, confidence, torch.zeros_like(confidence))
    if radius > 0:
        weights = _max_pool(torch, zero_filled, radius)
        candidate = weights > 0.0
    else:
        weights = zero_filled
        candidate = valid
    if mode == "dilated":
        return AlphaSupport(
            support=candidate, weights=weights, candidate=candidate, discontinuity=None
        )
    if range_m is None:
        raise ValueError("strict_visibility alpha support requires LiDAR ranges")
    ranges = range_m.to(dtype=confidence.dtype)
    strict_valid = valid & torch.isfinite(ranges) & (ranges > 0.0)
    neg_inf = torch.full_like(ranges, float("-inf"))
    nearest = -_max_pool(torch, torch.where(strict_valid, -ranges, neg_inf), radius)
    farthest = _max_pool(torch, torch.where(strict_valid, ranges, neg_inf), radius)
    has_return = torch.isfinite(farthest)
    agrees = has_return & (
        farthest <= nearest * (1.0 + STRICT_VISIBILITY_TOLERANCE) + STRICT_VISIBILITY_MARGIN_M
    )
    discontinuity = has_return & ~agrees
    erosion = STRICT_VISIBILITY_EDGE_EROSION_PX
    if erosion > 0:
        near_edge = _max_pool(torch, discontinuity.to(dtype=confidence.dtype), erosion) > 0.0
    else:
        near_edge = discontinuity
    support = agrees & ~near_edge
    return AlphaSupport(
        support=support, weights=weights, candidate=candidate, discontinuity=discontinuity
    )


# --------------------------------------------------------------------------
# numpy twin for CPU audits (no torch dependency)
# --------------------------------------------------------------------------


def _window_extreme(array: np.ndarray, radius: int, *, largest: bool) -> np.ndarray:
    """Separable (2r+1)^2 max (or min) filter with -inf (or +inf) padding."""
    if radius <= 0:
        return array.copy()
    fill = -np.inf if largest else np.inf
    reducer = np.maximum if largest else np.minimum
    out = array
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="constant", constant_values=fill)
        n = out.shape[axis]
        acc = None
        for shift in range(2 * radius + 1):
            window = np.take(padded, range(shift, shift + n), axis=axis)
            acc = window.copy() if acc is None else reducer(acc, window, out=acc)
        out = acc
    return out


def lidar_alpha_support_numpy(
    *,
    depth_mask: np.ndarray,
    confidence: np.ndarray,
    range_m: np.ndarray | None,
    mode: str,
    dilation_radius_px: int,
) -> AlphaSupport:
    """numpy twin of :func:`lidar_alpha_support` (same masks, same weights)."""
    if mode not in ALPHA_SUPPORT_MODES:
        raise ValueError(f"unknown lidar_alpha_support_mode {mode!r}")
    radius = int(dilation_radius_px)
    confidence = np.asarray(confidence, dtype=np.float32)
    depth_mask = np.asarray(depth_mask, dtype=bool)
    valid = depth_mask & np.isfinite(confidence) & (confidence > 0.0)
    zero_filled = np.where(valid, confidence, np.float32(0.0)).astype(np.float32)
    if radius > 0:
        weights = _window_extreme(zero_filled, radius, largest=True)
        candidate = weights > 0.0
    else:
        weights = zero_filled
        candidate = valid
    if mode == "dilated":
        return AlphaSupport(
            support=candidate, weights=weights, candidate=candidate, discontinuity=None
        )
    if range_m is None:
        raise ValueError("strict_visibility alpha support requires LiDAR ranges")
    ranges = np.asarray(range_m, dtype=np.float32)
    strict_valid = valid & np.isfinite(ranges) & (ranges > 0.0)
    nearest = -_window_extreme(
        np.where(strict_valid, -ranges, np.float32(-np.inf)).astype(np.float32),
        radius,
        largest=True,
    )
    farthest = _window_extreme(
        np.where(strict_valid, ranges, np.float32(-np.inf)).astype(np.float32),
        radius,
        largest=True,
    )
    has_return = np.isfinite(farthest)
    agrees = has_return & (
        farthest <= nearest * (1.0 + STRICT_VISIBILITY_TOLERANCE) + STRICT_VISIBILITY_MARGIN_M
    )
    discontinuity = has_return & ~agrees
    erosion = STRICT_VISIBILITY_EDGE_EROSION_PX
    if erosion > 0:
        near_edge = (
            _window_extreme(discontinuity.astype(np.float32), erosion, largest=True) > 0.0
        )
    else:
        near_edge = discontinuity
    support = agrees & ~near_edge
    return AlphaSupport(
        support=support, weights=weights, candidate=candidate, discontinuity=discontinuity
    )


def window_range_spread_numpy(
    *, valid: np.ndarray, range_m: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """(nearest, farthest) return in each window; +inf / -inf where empty.

    Audit helper: lets a caller classify a rejected pixel by *how much* the
    window's returns disagree (e.g. against the loose vis6 rule).
    """
    ranges = np.asarray(range_m, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    nearest = -_window_extreme(
        np.where(valid, -ranges, np.float32(-np.inf)).astype(np.float32),
        int(radius),
        largest=True,
    )
    farthest = _window_extreme(
        np.where(valid, ranges, np.float32(-np.inf)).astype(np.float32),
        int(radius),
        largest=True,
    )
    return nearest, farthest
