"""Deterministic LiDAR-first Face4 loss schedule for the snow task.

This module is a CPU-only schedule oracle.  It deliberately does not claim
    that deferred DA2 or mesh experiments are required. Consumers bind each
    non-zero weight to measured LiDAR or image supervision before the
matching supervision implementation before a training gate can advance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MipMapLossWeights:
    stage: str
    rgb_l1: float
    rgb_dssim: float
    da2_depth: float
    mesh_depth: float
    mesh_normal: float
    sparse_lidar_range: float
    lidar_surface_normal: float
    rendered_depth_normal_consistency: float
    opacity_mean: float
    sky_opacity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def high_type2_loss_weights(step: int, view_count: int) -> MipMapLossWeights:
    """Return the recovered loss weights for a zero-based training ``step``.

    Boundaries are expressed in complete view epochs ``V``:

    * ``0..5V``: measured LiDAR/RGB bootstrap before densification dominates;
    * ``5V..15V``: LiDAR-surface-constrained growth;
    * ``15V..20V``: surface polish with the same measured geometry authority.

    DA2 and all mesh-derived terms are exactly zero.  Sparse LiDAR range and
    direct KNN-PCA surface-normal anchoring remain active throughout, because
    split/clone can create geometry at every refinement stage.
    """

    step = int(step)
    view_count = int(view_count)
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    total_steps = 20 * view_count
    if step < 0 or step >= total_steps:
        raise ValueError(f"step must be within [0, {total_steps})")

    epoch = step // view_count
    if epoch < 5:
        stage = "lidar_rgb_bootstrap"
    elif epoch < 15:
        stage = "lidar_surface_growth"
    else:
        stage = "lidar_surface_polish"

    return MipMapLossWeights(
        stage=stage,
        rgb_l1=0.6,
        rgb_dssim=0.4,
        da2_depth=0.0,
        mesh_depth=0.0,
        mesh_normal=0.0,
        sparse_lidar_range=0.05,
        lidar_surface_normal=0.01,
        rendered_depth_normal_consistency=0.0,
        opacity_mean=0.01,
        sky_opacity=0.04 if epoch >= 10 else 0.0,
    )


def high_type2_schedule_contract(view_count: int) -> dict[str, Any]:
    """Compact, serializable boundary table for one Tile."""

    view_count = int(view_count)
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    boundary_steps = [0, 5 * view_count, 10 * view_count, 15 * view_count]
    return {
        "view_count": view_count,
        "total_steps": 20 * view_count,
        "stage_epochs": [5, 10, 5],
        "boundary_steps": boundary_steps + [20 * view_count],
        "weights_at_stage_start": [
            high_type2_loss_weights(step, view_count).to_dict()
            for step in boundary_steps
        ],
        "evidence_boundary": {
            "geometry_authority": "SPARSE_REAL_LIDAR_AND_KNN_PCA_SURFACE",
            "da2": "DEFERRED_OPTIONAL_WEIGHT_ZERO",
            "mesh": "DEFERRED_OPTIONAL_WEIGHT_ZERO",
            "independent_sky_optimizer": "REQUIRED_NOT_IMPLEMENTED_BY_THIS_MODULE",
        },
        "training_allowed": False,
    }
