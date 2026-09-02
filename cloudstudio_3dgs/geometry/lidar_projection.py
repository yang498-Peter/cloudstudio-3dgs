"""KB4 LiDAR ray-range projection with deterministic pixel z-buffering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.image_sample import CropWindow
from cloudstudio_3dgs.geometry.kb4 import project_kb4


@dataclass(frozen=True)
class DepthProjectionConfig:
    min_range_m: float = 0.2
    max_range_m: float = 80.0
    max_theta_deg: float = 95.0
    confidence_support_points: int = 3
    confidence_subpixel_sigma_px: float = 0.75
    # Hidden-point rejection. A nearest-range z-buffer per exact pixel keeps
    # whatever point happens to land on that pixel, and on a sparse front
    # surface most pixels are reached first by a point from behind it (a
    # wall, a trunk, foliage). With a cell size > 0, every point is compared
    # against the nearest range in its 3x3 cell neighbourhood and dropped when
    # it lies farther than (1 + tolerance) x that range + margin. 0 disables.
    visibility_cell_px: int = 0
    visibility_tolerance: float = 0.2
    visibility_margin_m: float = 0.1

    def validate(self) -> None:
        if self.min_range_m < 0.0 or self.max_range_m <= self.min_range_m:
            raise ValueError("depth range bounds are invalid")
        if int(self.visibility_cell_px) < 0:
            raise ValueError("visibility_cell_px must be non-negative")
        if self.visibility_tolerance < 0.0 or self.visibility_margin_m < 0.0:
            raise ValueError("visibility tolerance and margin must be non-negative")
        if not 0.0 < self.max_theta_deg <= 180.0:
            raise ValueError("max_theta_deg must be in (0, 180]")
        if self.confidence_support_points <= 0:
            raise ValueError("confidence_support_points must be positive")
        if self.confidence_subpixel_sigma_px <= 0.0:
            raise ValueError("confidence_subpixel_sigma_px must be positive")

    def to_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "min_range_m": self.min_range_m,
            "max_range_m": self.max_range_m,
            "max_theta_deg": self.max_theta_deg,
            "confidence_support_points": self.confidence_support_points,
            "confidence_subpixel_sigma_px": self.confidence_subpixel_sigma_px,
        }
        if int(self.visibility_cell_px) > 0:
            payload["visibility_cell_px"] = int(self.visibility_cell_px)
            payload["visibility_tolerance"] = self.visibility_tolerance
            payload["visibility_margin_m"] = self.visibility_margin_m
        return payload


@dataclass(frozen=True)
class SparseDepthMap:
    shape: tuple[int, int]
    pixel_index: np.ndarray
    range_m: np.ndarray
    confidence: np.ndarray
    source_index: np.ndarray
    support_count: np.ndarray

    def validate(self) -> None:
        height, width = self.shape
        if min(height, width) <= 0:
            raise ValueError("depth map shape must be positive")
        length = len(self.pixel_index)
        for name, value in (
            ("range_m", self.range_m),
            ("confidence", self.confidence),
            ("source_index", self.source_index),
            ("support_count", self.support_count),
        ):
            if len(value) != length:
                raise ValueError(f"{name} length does not match pixel_index")
        if length:
            if np.any(self.pixel_index < 0) or np.any(self.pixel_index >= height * width):
                raise ValueError("depth pixel index is outside the image")
            if np.any(self.pixel_index[1:] <= self.pixel_index[:-1]):
                raise ValueError("depth pixel indexes must be strictly increasing")
            if not np.all(np.isfinite(self.range_m)) or np.any(self.range_m <= 0.0):
                raise ValueError("depth ranges must be finite and positive")
            if not np.all(np.isfinite(self.confidence)) or np.any(
                (self.confidence <= 0.0) | (self.confidence > 1.0)
            ):
                raise ValueError("depth confidence must be in (0, 1]")

    def to_dense(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.validate()
        height, width = self.shape
        depth = np.zeros((height, width), dtype=np.float32)
        confidence = np.zeros((height, width), dtype=np.float32)
        valid = np.zeros((height, width), dtype=bool)
        depth.flat[self.pixel_index] = self.range_m
        confidence.flat[self.pixel_index] = self.confidence
        valid.flat[self.pixel_index] = True
        return depth, confidence, valid


def visible_point_mask(
    pixel_xy: np.ndarray,
    ranges: np.ndarray,
    width: int,
    height: int,
    config: DepthProjectionConfig,
) -> np.ndarray:
    """Boolean keep-mask for the hidden-point test; all True when disabled."""
    cell = int(config.visibility_cell_px)
    keep = np.ones(len(ranges), dtype=bool)
    if cell <= 0 or not len(ranges):
        return keep
    cells_w = (int(width) + cell - 1) // cell
    cells_h = (int(height) + cell - 1) // cell
    cx = np.asarray(pixel_xy[:, 0], dtype=np.int64) // cell
    cy = np.asarray(pixel_xy[:, 1], dtype=np.int64) // cell
    nearest = np.full((cells_h, cells_w), np.inf, dtype=np.float64)
    np.minimum.at(nearest, (cy, cx), np.asarray(ranges, dtype=np.float64))
    # 3x3 min filter so a point near a cell border is judged against the
    # front surface on either side of it, not only its own cell.
    padded = np.pad(nearest, 1, mode="constant", constant_values=np.inf)
    neighbourhood = nearest.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            np.minimum(
                neighbourhood,
                padded[dy : dy + cells_h, dx : dx + cells_w],
                out=neighbourhood,
            )
    limit = (1.0 + float(config.visibility_tolerance)) * neighbourhood[cy, cx] + float(
        config.visibility_margin_m
    )
    keep &= np.asarray(ranges, dtype=np.float64) <= limit
    return keep


def project_camera_points_to_face(
    points_camera: np.ndarray,
    face: Any,
    *,
    source_index: np.ndarray | None = None,
    supervision_mask: np.ndarray | None = None,
    config: DepthProjectionConfig = DepthProjectionConfig(),
) -> SparseDepthMap:
    """Project exact camera-frame LiDAR points directly into one pinhole face.

    Pixel coordinates are rounded only in the destination Face4 raster.  This
    deliberately avoids reconstructing rays from an intermediate integer
    fisheye depth map.  ``source_index`` remains bound to the original LAS
    point so downstream audits can prove where every retained sample came from.
    """
    config.validate()
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape [N, 3]")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_camera contains non-finite coordinates")
    if source_index is None:
        sources = np.arange(len(points), dtype=np.int64)
    else:
        sources = np.asarray(source_index, dtype=np.int64)
        if sources.shape != (len(points),):
            raise ValueError("source_index must have shape [N]")
        if np.any(sources < 0):
            raise ValueError("source_index must identify original nonnegative points")

    width = int(face.width)
    height = int(face.height)
    if min(width, height) <= 0:
        raise ValueError("face dimensions must be positive")
    if supervision_mask is not None:
        allowed = np.asarray(supervision_mask, dtype=bool)
        if allowed.shape != (height, width):
            raise ValueError("supervision_mask shape does not match face")
    else:
        allowed = None
    if not len(points):
        return SparseDepthMap(
            (height, width),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )

    ranges = np.linalg.norm(points, axis=1)
    pixels, inside = face.directions_to_pixels(points)
    finite_pixels = np.isfinite(pixels).all(axis=1)
    rounded = np.zeros_like(pixels, dtype=np.int64)
    rounded[finite_pixels] = np.rint(pixels[finite_pixels]).astype(np.int64)
    valid = inside & finite_pixels & np.isfinite(ranges)
    valid &= (ranges >= config.min_range_m) & (ranges <= config.max_range_m)
    valid &= (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    selected = np.flatnonzero(valid).astype(np.int64)
    if not len(selected):
        return SparseDepthMap(
            (height, width),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )

    visible = visible_point_mask(rounded[selected], ranges[selected], width, height, config)
    selected = selected[visible]
    if not len(selected):
        return SparseDepthMap(
            (height, width),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )
    selected_pixels = pixels[selected]
    selected_xy = rounded[selected]
    pixel_index = selected_xy[:, 1] * width + selected_xy[:, 0]
    selected_range = ranges[selected]
    selected_sources = sources[selected]
    subpixel_error_squared = np.einsum(
        "ij,ij->i", selected_pixels - selected_xy, selected_pixels - selected_xy
    )
    order = np.lexsort((selected_sources, selected_range, pixel_index))
    ordered_pixels = pixel_index[order]
    starts = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    first = np.flatnonzero(starts)
    support_count = np.diff(np.r_[first, len(order)]).astype(np.int32)
    chosen = order[first]

    output_pixels = pixel_index[chosen].astype(np.int32)
    output_ranges = selected_range[chosen].astype(np.float32)
    output_sources = selected_sources[chosen].astype(np.int64, copy=False)
    sigma = config.confidence_subpixel_sigma_px
    spatial_confidence = np.exp(-subpixel_error_squared[chosen] / (2.0 * sigma * sigma))
    support_confidence = np.minimum(
        1.0,
        np.log1p(support_count) / np.log1p(config.confidence_support_points),
    )
    output_confidence = (spatial_confidence * support_confidence).astype(np.float32)

    if allowed is not None:
        allowed_output = allowed[selected_xy[chosen, 1], selected_xy[chosen, 0]]
        output_pixels = output_pixels[allowed_output]
        output_ranges = output_ranges[allowed_output]
        output_confidence = output_confidence[allowed_output]
        output_sources = output_sources[allowed_output]
        support_count = support_count[allowed_output]

    result = SparseDepthMap(
        (height, width),
        output_pixels,
        output_ranges,
        output_confidence,
        output_sources,
        support_count,
    )
    result.validate()
    return result


def _camera_parameters(camera: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    distortion = camera.get("distortion", {})
    if distortion.get("camera_model") != "OPENCV_FISHEYE":
        raise ValueError("LiDAR depth projection requires OPENCV_FISHEYE")
    return camera["intrinsic"], distortion["params"]


def project_lidar_depth(
    points_world: np.ndarray,
    world_from_camera: np.ndarray,
    camera: dict[str, Any],
    *,
    supervision_mask: np.ndarray | None = None,
    crop: CropWindow | None = None,
    config: DepthProjectionConfig = DepthProjectionConfig(),
) -> SparseDepthMap:
    """Project world points and keep the nearest Euclidean range per pixel."""
    config.validate()
    points = np.asarray(points_world, dtype=np.float64)
    pose = np.asarray(world_from_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N, 3]")
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("world_from_camera must be a finite 4x4 matrix")
    width = int(camera["width"])
    height = int(camera["height"])
    if min(width, height) <= 0:
        raise ValueError("camera dimensions must be positive")
    if supervision_mask is not None:
        allowed = np.asarray(supervision_mask, dtype=bool)
        if allowed.shape != (height, width):
            raise ValueError("supervision_mask shape does not match camera")
    else:
        allowed = None

    window = crop or CropWindow(0, 0, width, height)
    window.validate(width, height, 1)
    output_shape = (window.height, window.width)
    if len(points) == 0:
        return SparseDepthMap(
            output_shape,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )

    points_camera = (points - pose[:3, 3]) @ pose[:3, :3]
    intrinsic, distortion = _camera_parameters(camera)
    uv, ranges, valid = project_kb4(
        points_camera,
        intrinsic,
        distortion,
        min_range_m=config.min_range_m,
        max_range_m=config.max_range_m,
        max_theta_rad=np.deg2rad(config.max_theta_deg),
    )
    finite_uv = np.isfinite(uv).all(axis=1)
    rounded = np.zeros_like(uv, dtype=np.int64)
    rounded[finite_uv] = np.rint(uv[finite_uv]).astype(np.int64)
    valid &= finite_uv
    valid &= (
        (rounded[:, 0] >= window.x)
        & (rounded[:, 0] < window.x + window.width)
        & (rounded[:, 1] >= window.y)
        & (rounded[:, 1] < window.y + window.height)
    )
    source_index = np.flatnonzero(valid).astype(np.int64)
    if not len(source_index):
        return SparseDepthMap(
            output_shape,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )

    local_xy = rounded[source_index] - np.array([[window.x, window.y]], dtype=np.int64)
    visible = visible_point_mask(
        local_xy, ranges[source_index], window.width, window.height, config
    )
    source_index = source_index[visible]
    if not len(source_index):
        return SparseDepthMap(
            output_shape,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )
    selected_uv = uv[source_index]
    selected_xy = rounded[source_index]
    local_x = selected_xy[:, 0] - window.x
    local_y = selected_xy[:, 1] - window.y
    pixel_index = local_y * window.width + local_x
    selected_range = ranges[source_index]
    subpixel_error_squared = np.einsum(
        "ij,ij->i", selected_uv - selected_xy, selected_uv - selected_xy
    )
    order = np.lexsort((source_index, selected_range, pixel_index))
    ordered_pixels = pixel_index[order]
    starts = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    first = np.flatnonzero(starts)
    support_count = np.diff(np.r_[first, len(order)]).astype(np.int32)
    chosen = order[first]

    output_pixels = pixel_index[chosen].astype(np.int32)
    output_ranges = selected_range[chosen].astype(np.float32)
    output_sources = source_index[chosen]
    sigma = config.confidence_subpixel_sigma_px
    spatial_confidence = np.exp(-subpixel_error_squared[chosen] / (2.0 * sigma * sigma))
    support_confidence = np.minimum(
        1.0,
        np.log1p(support_count) / np.log1p(config.confidence_support_points),
    )
    output_confidence = (spatial_confidence * support_confidence).astype(np.float32)

    if allowed is not None:
        allowed_output = allowed[selected_xy[chosen, 1], selected_xy[chosen, 0]]
        output_pixels = output_pixels[allowed_output]
        output_ranges = output_ranges[allowed_output]
        output_confidence = output_confidence[allowed_output]
        output_sources = output_sources[allowed_output]
        support_count = support_count[allowed_output]

    result = SparseDepthMap(
        output_shape,
        output_pixels,
        output_ranges,
        output_confidence,
        output_sources,
        support_count,
    )
    result.validate()
    return result
