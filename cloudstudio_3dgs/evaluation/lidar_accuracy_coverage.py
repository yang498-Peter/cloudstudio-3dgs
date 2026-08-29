"""Accuracy times coverage metrics for Tile-scoped sparse LiDAR range."""

from __future__ import annotations

from typing import Any

import numpy as np

from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap


def _neighbor_depth_edge(
    pixel_index: np.ndarray,
    range_m: np.ndarray,
    *,
    width: int,
    threshold_m: float,
) -> np.ndarray:
    """Mark sparse pixels touching a four-neighbour range discontinuity."""

    index = np.asarray(pixel_index, dtype=np.int64)
    values = np.asarray(range_m, dtype=np.float64)
    order = np.argsort(index)
    index = index[order]
    values = values[order]
    edge = np.zeros(len(index), dtype=bool)
    x = index % int(width)
    for offset, allowed in (
        (-1, x > 0),
        (1, x + 1 < width),
        (-int(width), np.ones(len(index), dtype=bool)),
        (int(width), np.ones(len(index), dtype=bool)),
    ):
        target = index + offset
        position = np.searchsorted(index, target)
        valid = allowed & (position < len(index))
        matched = np.zeros(len(index), dtype=bool)
        matched[valid] = index[position[valid]] == target[valid]
        valid &= matched
        edge[valid] |= np.abs(values[valid] - values[position[valid]]) > float(
            threshold_m
        )
    restored = np.empty_like(edge)
    restored[order] = edge
    return restored


def compare_tile_to_source(
    source: SparseDepthMap,
    tile: SparseDepthMap,
    *,
    crop_xywh: tuple[int, int, int, int],
    edge_threshold_m: float = 0.10,
) -> dict[str, Any]:
    """Compare retained Tile rays to the authoritative source rays."""

    source.validate()
    tile.validate()
    if source.shape != tile.shape:
        raise ValueError("source and Tile sparse depths use different shapes")
    height, width = source.shape
    x0, y0, crop_width, crop_height = (int(value) for value in crop_xywh)
    if min(x0, y0) < 0 or min(crop_width, crop_height) <= 0:
        raise ValueError("crop must have a non-negative origin and positive size")
    if x0 + crop_width > width or y0 + crop_height > height:
        raise ValueError("crop exceeds source depth shape")

    source_order = np.argsort(source.pixel_index)
    source_index = source.pixel_index[source_order].astype(np.int64, copy=False)
    source_range = source.range_m[source_order].astype(np.float64, copy=False)
    source_confidence = source.confidence[source_order].astype(np.float64, copy=False)
    yy, xx = np.divmod(source_index, width)
    crop_mask = (
        (xx >= x0)
        & (xx < x0 + crop_width)
        & (yy >= y0)
        & (yy < y0 + crop_height)
    )
    candidate_index = source_index[crop_mask]
    candidate_range = source_range[crop_mask]
    candidate_confidence = source_confidence[crop_mask]
    candidate_edge = _neighbor_depth_edge(
        candidate_index,
        candidate_range,
        width=width,
        threshold_m=edge_threshold_m,
    )

    tile_order = np.argsort(tile.pixel_index)
    tile_index = tile.pixel_index[tile_order].astype(np.int64, copy=False)
    tile_range = tile.range_m[tile_order].astype(np.float64, copy=False)
    position = np.searchsorted(candidate_index, tile_index)
    if len(tile_index):
        if np.any(position >= len(candidate_index)) or not np.array_equal(
            candidate_index[position], tile_index
        ):
            raise ValueError("Tile depth contains pixels outside the source crop")
    error = np.abs(tile_range - candidate_range[position])
    retained_confidence = candidate_confidence[position]
    retained_edge = candidate_edge[position]

    strata = {
        "low": candidate_confidence < 0.5,
        "medium": (candidate_confidence >= 0.5) & (candidate_confidence < 0.8),
        "high": candidate_confidence >= 0.8,
    }
    confidence: dict[str, Any] = {}
    for name, mask in strata.items():
        retained = (
            retained_confidence < 0.5
            if name == "low"
            else (
                (retained_confidence >= 0.5) & (retained_confidence < 0.8)
                if name == "medium"
                else retained_confidence >= 0.8
            )
        )
        candidate_count = int(np.count_nonzero(mask))
        retained_count = int(np.count_nonzero(retained))
        confidence[name] = {
            "candidate_pixels": candidate_count,
            "retained_pixels": retained_count,
            "coverage_fraction": None
            if candidate_count == 0
            else retained_count / candidate_count,
            "error_max_m": None
            if retained_count == 0
            else float(np.max(error[retained])),
            "over_5cm_count": int(np.count_nonzero(error[retained] > 0.05)),
            "over_10cm_count": int(np.count_nonzero(error[retained] > 0.10)),
        }

    candidate_count = int(len(candidate_index))
    retained_count = int(len(tile_index))
    return {
        "candidate_pixels": candidate_count,
        "retained_pixels": retained_count,
        "coverage_fraction": None
        if candidate_count == 0
        else retained_count / candidate_count,
        "errors_m": error,
        "confidence": confidence,
        "edge": {
            "threshold_m": float(edge_threshold_m),
            "candidate_pixels": int(np.count_nonzero(candidate_edge)),
            "retained_pixels": int(np.count_nonzero(retained_edge)),
            "over_5cm_count": int(np.count_nonzero(error[retained_edge] > 0.05)),
            "over_10cm_count": int(np.count_nonzero(error[retained_edge] > 0.10)),
        },
        "nonedge": {
            "candidate_pixels": int(np.count_nonzero(~candidate_edge)),
            "retained_pixels": int(np.count_nonzero(~retained_edge)),
            "over_5cm_count": int(np.count_nonzero(error[~retained_edge] > 0.05)),
            "over_10cm_count": int(np.count_nonzero(error[~retained_edge] > 0.10)),
        },
    }
