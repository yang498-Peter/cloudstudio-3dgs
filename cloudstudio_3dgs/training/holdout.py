"""Spatial hold-out of training views inside one Tile.

The signed split manifest holds out one temporal block that sits entirely in
one Tile, so a Tile-level arm has no held-out views of its own. This picks
whole 2D cells of camera positions (every face of an image goes together, so
no face of a held-out image leaks into training) until the requested fraction
of the Tile's views is reached. The selection is a pure function of the view
list, the cell size, the fraction and the seed, so a later battery can
reconstruct it from the record the trainer writes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _cell_key(position: Any, cell_m: float) -> tuple[int, int]:
    p = np.asarray(position, dtype=np.float64)
    return (int(math.floor(p[0] / cell_m)), int(math.floor(p[1] / cell_m)))


def select_spatial_holdout(
    views: list[dict[str, Any]],
    position_by_image: dict[str, Any],
    *,
    cell_m: float,
    fraction: float,
    seed: int,
    guard_m: float = 0.0,
) -> dict[str, Any]:
    """Pick held-out views; with ``guard_m`` > 0, training views whose camera
    lies within that distance of any held-out camera are also withheld from
    training (a guard band) but are not scored, so a held-out camera cannot
    have a training camera centimetres away across a cell border."""
    if not (cell_m > 0.0) or not math.isfinite(cell_m):
        raise ValueError("holdout cell size must be positive")
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must be within (0, 1)")
    image_ids: list[str] = []
    views_by_image: dict[str, list[str]] = {}
    for view in views:
        sample_id = str(view["sample_id"])
        image_id = sample_id.split("::", 1)[0]
        if image_id not in views_by_image:
            views_by_image[image_id] = []
            image_ids.append(image_id)
        views_by_image[image_id].append(sample_id)
    cells: dict[tuple[int, int], list[str]] = {}
    for image_id in image_ids:
        if image_id not in position_by_image:
            raise ValueError(f"no camera position for held-out candidate {image_id}")
        cells.setdefault(_cell_key(position_by_image[image_id], cell_m), []).append(image_id)
    cell_keys = sorted(cells)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), len(views)]))
    order = rng.permutation(len(cell_keys))
    target = int(math.ceil(fraction * len(views)))
    held_images: list[str] = []
    held_views: list[str] = []
    for index in order:
        if len(held_views) >= target:
            break
        for image_id in cells[cell_keys[index]]:
            held_images.append(image_id)
            held_views.extend(views_by_image[image_id])
    held_set = set(held_views)
    held_image_set = set(held_images)
    positions = {i: np.asarray(position_by_image[i], dtype=np.float64) for i in image_ids}
    held_pos = np.stack([positions[i] for i in held_images]) if held_images else np.zeros((0, 3))
    guard_images: list[str] = []
    guard_views: list[str] = []
    for image_id in image_ids:
        if image_id in held_image_set or not len(held_pos):
            continue
        d = float(np.min(np.linalg.norm(held_pos - positions[image_id], axis=1)))
        if guard_m > 0.0 and d < guard_m:
            guard_images.append(image_id)
            guard_views.extend(views_by_image[image_id])
    guard_image_set = set(guard_images)
    train_images = [i for i in image_ids if i not in held_image_set and i not in guard_image_set]
    train_pos = np.stack([positions[i] for i in train_images]) if train_images else np.zeros((0, 3))
    nearest: list[float] = []
    for image_id in held_images:
        if len(train_pos):
            nearest.append(float(np.min(np.linalg.norm(train_pos - positions[image_id], axis=1))))
    nearest_arr = np.asarray(nearest) if nearest else np.zeros(0)
    held_cells = {_cell_key(positions[i], cell_m) for i in held_images}
    train_cells = {_cell_key(positions[i], cell_m) for i in train_images}
    adjacent = sum(
        1
        for (cx, cy) in held_cells
        if any(
            (cx + dx, cy + dy) in train_cells
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        )
    )
    return {
        "definition": "spatial-cell holdout v1.1 (whole images per 2D camera cell, optional guard band)",
        "guard_m": float(guard_m),
        "guard_image_ids": guard_images,
        "guard_sample_ids": guard_views,
        "actual_held_out_fraction": len(held_set) / max(1, len(views)),
        "held_out_image_count": len(held_images),
        "nearest_training_camera_m": (
            {
                "min": float(nearest_arr.min()),
                "p05": float(np.percentile(nearest_arr, 5)),
                "p50": float(np.percentile(nearest_arr, 50)),
                "p95": float(np.percentile(nearest_arr, 95)),
            }
            if len(nearest_arr)
            else None
        ),
        "held_out_cells_with_adjacent_training_cell": int(adjacent),
        "cell_m": float(cell_m),
        "fraction": float(fraction),
        "seed": int(seed),
        "total_view_count": len(views),
        "cell_count": len(cell_keys),
        "held_out_cell_count": len({_cell_key(position_by_image[i], cell_m) for i in held_images}),
        "held_out_image_ids": held_images,
        "held_out_sample_ids": held_views,
        "training_view_count": len(views) - len(held_set) - len(guard_views),
    }
