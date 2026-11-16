"""Supervise a face at the resolution the sensor resolved, not the resolution it was warped to.

The MipMap Face4 plan warps a 2912 px f-theta fisheye (787.6 px per radian on axis) onto
pinhole faces whose focal lengths were recovered from the reference product: 1039.7 px for
the two yaw faces and 2308.9 px for the two pitch faces. A pitch face is therefore the sensor
image interpolated 2.93x - nine pixels where the camera measured one - and what fills the
extra bandwidth is demosaic and JPEG texture that differs in every photo.

Training on it at full size costs the pixel work nine times over and asks the loss (SSIM most
of all, whose 11 px window then spans less than four sensor pixels) to reproduce structure no
second view agrees on. ``reduce_sample`` area-averages a decoded sample down to a requested
focal length, so every face is supervised at one angular resolution.

Everything the trainer reads from a sample is reduced together:

* colour: area average (the anti-aliased downsample; this is what removes per-photo noise);
* supervision masks: a reduced pixel is supervised only if every source pixel was, so a
  masked person or a foreign-tile return never bleeds into a kept pixel;
* sparse LiDAR: the nearest return that lands in the reduced pixel, with its own confidence.
  Averaging would invent a surface between a silhouette and what is behind it;
* monocular depth: mean over the valid source pixels, kept where most of them were valid;
* sky label: majority vote (the trainer erodes it afterwards);
* intrinsics: scaled by the exact per-axis ratio, so reduced pixel centres stay on the rays
  of the source pixel blocks; the z-to-range factor is recomputed from the new intrinsics.

Thresholds the trainer expresses in pixels are defined on the full-size face. ``pixels`` maps
one of them to the reduced grid so the same angle is covered.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

# Below this fraction of valid source pixels a reduced monocular-depth pixel is dropped.
_MONO_MIN_VALID_FRACTION = 0.5
# A reduced supervision pixel needs every source pixel; the tolerance absorbs float32 area
# sums over blocks of a few dozen pixels.
_FULL_COVERAGE = 1.0 - 1e-4


def reduction_for_focal(focal_px: float, target_focal_px: float | None) -> float:
    """How much to shrink a face of ``focal_px`` so it is trained at ``target_focal_px``.

    Never enlarges: a face already at or below the target is left alone.
    """
    if target_focal_px is None:
        return 1.0
    if target_focal_px <= 0.0:
        raise ValueError("training focal length must be positive")
    return max(1.0, float(focal_px) / float(target_focal_px))


def reduced_shape(height: int, width: int, reduction: float) -> tuple[int, int]:
    return max(1, int(round(height / reduction))), max(1, int(round(width / reduction)))


def pixels(full_size_pixels: int, reduction: float) -> int:
    """A pixel radius defined on the full-size face, on the reduced grid.

    Zero stays zero (the feature is off); anything else keeps at least one pixel so a guard
    band cannot vanish by rounding.
    """
    if full_size_pixels <= 0 or reduction <= 1.0:
        return int(full_size_pixels)
    return max(1, int(round(full_size_pixels / reduction)))


def _area(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _all_of(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return _area(mask.astype(np.float32), shape) >= _FULL_COVERAGE


def _nearest_return(
    range_m: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter sparse returns onto the reduced grid, keeping the nearest per pixel."""
    height, width = valid.shape
    out_h, out_w = shape
    rows, cols = np.nonzero(valid)
    out_range = np.zeros(shape, dtype=np.float32)
    out_confidence = np.zeros(shape, dtype=np.float32)
    out_valid = np.zeros(shape, dtype=bool)
    if rows.size == 0:
        return out_range, out_confidence, out_valid
    # Pixel centre (r + 0.5) of the source lands in reduced row floor((r + 0.5) * out_h / H).
    target_rows = np.minimum(((rows + 0.5) * (out_h / height)).astype(np.int64), out_h - 1)
    target_cols = np.minimum(((cols + 0.5) * (out_w / width)).astype(np.int64), out_w - 1)
    flat = target_rows * out_w + target_cols
    values = range_m[rows, cols].astype(np.float32)
    weights = confidence[rows, cols].astype(np.float32)
    # Sort by (cell, range) and keep the first entry of each cell: the nearest return.
    order = np.lexsort((values, flat))
    flat_sorted = flat[order]
    first = np.ones(flat_sorted.shape, dtype=bool)
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    chosen = order[first]
    cells = flat[chosen]
    out_range.reshape(-1)[cells] = values[chosen]
    out_confidence.reshape(-1)[cells] = weights[chosen]
    out_valid.reshape(-1)[cells] = True
    return out_range, out_confidence, out_valid


def _range_scale(K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """``||K^-1 [u + 0.5, v + 0.5, 1]||`` on the reduced grid (z-depth -> ray range)."""
    height, width = shape
    jj, ii = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    x = (jj - float(K[0, 2])) / float(K[0, 0])
    y = (ii - float(K[1, 2])) / float(K[1, 1])
    return np.sqrt(1.0 + x * x + y * y).astype(np.float32)


def reduce_sample(sample: Any, reduction: float) -> Any:
    """Return ``sample`` area-reduced by ``reduction`` (>= 1); 1 returns it untouched."""
    if reduction <= 1.0:
        return sample
    if getattr(sample, "camera_model", "pinhole") != "pinhole":
        raise ValueError("resolution reduction is defined for pinhole faces only")
    if getattr(sample, "mesh_depth_range_m", None) is not None:
        raise ValueError("resolution reduction does not carry mesh geometry supervision")
    height, width = int(sample.height), int(sample.width)
    shape = reduced_shape(height, width, reduction)
    if shape == (height, width):
        return sample
    scale_y = height / shape[0]
    scale_x = width / shape[1]

    image = _area(np.ascontiguousarray(sample.image), shape)
    rgb_mask = _all_of(np.asarray(sample.rgb_mask), shape)

    K = np.array(sample.K, dtype=np.float32, copy=True)
    K[0, 0] /= scale_x
    K[0, 2] /= scale_x
    K[1, 1] /= scale_y
    K[1, 2] /= scale_y

    fields: dict[str, Any] = {
        "image": image,
        "rgb_mask": rgb_mask,
        "K": K,
        "width": int(shape[1]),
        "height": int(shape[0]),
        "resolution_reduction": float(reduction),
    }
    if sample.depth_range_m is not None:
        depth_range, depth_confidence, depth_valid = _nearest_return(
            np.asarray(sample.depth_range_m),
            np.asarray(sample.depth_confidence),
            np.asarray(sample.depth_mask, dtype=bool),
            shape,
        )
        fields.update(
            depth_range_m=depth_range,
            depth_confidence=depth_confidence,
            depth_mask=depth_valid,
        )
    if sample.mono_depth_range_m is not None:
        mono_valid = np.asarray(sample.mono_depth_mask, dtype=bool)
        coverage = _area(mono_valid.astype(np.float32), shape)
        summed = _area(
            np.where(mono_valid, np.asarray(sample.mono_depth_range_m, dtype=np.float32), 0.0).astype(np.float32),
            shape,
        )
        mono_mask = coverage >= _MONO_MIN_VALID_FRACTION
        mono_range = np.where(mono_mask, summed / np.maximum(coverage, 1e-6), 0.0).astype(np.float32)
        fields.update(mono_depth_range_m=mono_range, mono_depth_mask=mono_mask & rgb_mask)
    if getattr(sample, "sky_mask", None) is not None:
        fields["sky_mask"] = _area(np.asarray(sample.sky_mask).astype(np.float32), shape) >= 0.5
    if sample.depth_to_range_scale is not None:
        fields["depth_to_range_scale"] = _range_scale(K, shape)
    if sample.sensor_pixel_coords is not None:
        fields["sensor_pixel_coords"] = _area(
            np.ascontiguousarray(sample.sensor_pixel_coords, dtype=np.float32), shape
        )
    if not rgb_mask.any():
        # A sliver of supervision can fall below one full reduced pixel. The sample still has
        # to carry a non-empty mask (an empty one is a hard error downstream), so keep the
        # reduced pixels that are mostly supervised instead.
        relaxed = _area(np.asarray(sample.rgb_mask).astype(np.float32), shape) >= 0.5
        if not relaxed.any():
            relaxed = _area(np.asarray(sample.rgb_mask).astype(np.float32), shape) > 0.0
        fields["rgb_mask"] = relaxed
        if "mono_depth_mask" in fields:
            fields["mono_depth_mask"] = mono_mask & relaxed
    return replace(sample, **fields)
