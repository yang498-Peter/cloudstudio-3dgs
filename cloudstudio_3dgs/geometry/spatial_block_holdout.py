"""Deterministic 3D block holdout for dense-surface leakage tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialBlockHoldout:
    construction_mask: np.ndarray
    holdout_mask: np.ndarray
    block_index: np.ndarray
    block_size_m: float
    target_fraction: float
    actual_fraction: float
    seed: int


def build_spatial_block_holdout(
    xyz: np.ndarray,
    *,
    block_size_m: float,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> SpatialBlockHoldout:
    """Hold out complete occupied 3D blocks before surface construction.

    Selection is deterministic and weighted by point count so the held-out
    point fraction, rather than merely the number of occupied voxels, tracks
    ``holdout_fraction``. A block is never split between construction and
    evaluation, preventing adjacent samples from leaking through triangulation.
    """

    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("xyz must be a non-empty [N, 3] array")
    if not np.all(np.isfinite(points)):
        raise ValueError("xyz must be finite")
    if block_size_m <= 0.0:
        raise ValueError("block_size_m must be positive")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be within (0, 1)")

    origin = points.min(axis=0)
    voxel = np.floor((points - origin) / float(block_size_m)).astype(np.int64)
    unique, inverse, counts = np.unique(
        voxel, axis=0, return_inverse=True, return_counts=True
    )
    scored: list[tuple[int, int]] = []
    seed_bytes = int(seed).to_bytes(8, "little", signed=True)
    for index, coordinate in enumerate(unique):
        digest = hashlib.sha256(seed_bytes + coordinate.astype("<i8").tobytes())
        scored.append((int.from_bytes(digest.digest()[:8], "little"), index))
    scored.sort()

    target = int(round(len(points) * float(holdout_fraction)))
    selected = np.zeros(len(unique), dtype=bool)
    selected_count = 0
    for _score, index in scored:
        if selected_count >= target:
            break
        selected[index] = True
        selected_count += int(counts[index])
    holdout = selected[inverse]
    construction = ~holdout
    if not np.any(construction) or not np.any(holdout):
        raise ValueError("holdout split produced an empty partition")
    return SpatialBlockHoldout(
        construction_mask=construction,
        holdout_mask=holdout,
        block_index=voxel,
        block_size_m=float(block_size_m),
        target_fraction=float(holdout_fraction),
        actual_fraction=float(np.mean(holdout)),
        seed=int(seed),
    )
