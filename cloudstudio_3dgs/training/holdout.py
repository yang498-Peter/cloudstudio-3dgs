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
) -> dict[str, Any]:
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
    return {
        "cell_m": float(cell_m),
        "fraction": float(fraction),
        "seed": int(seed),
        "total_view_count": len(views),
        "cell_count": len(cell_keys),
        "held_out_cell_count": len({_cell_key(position_by_image[i], cell_m) for i in held_images}),
        "held_out_image_ids": held_images,
        "held_out_sample_ids": held_views,
        "training_view_count": len(views) - len(held_set),
    }
