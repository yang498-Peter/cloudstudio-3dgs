"""Unique core ownership for cut-then-merge Tile exports."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def _ordered_boxes(tiles: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(tiles, key=lambda tile: int(tile["tile_id"]))
    ids = np.asarray([int(tile["tile_id"]) for tile in ordered], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Tile ids must be unique")
    boxes = np.asarray([tile["core_box"] for tile in ordered], dtype=np.float64)
    if boxes.shape != (len(ordered), 2, 3) or not np.all(np.isfinite(boxes)):
        raise ValueError("each Tile must define one finite 3D core_box")
    if np.any(boxes[:, 1] <= boxes[:, 0]):
        raise ValueError("Tile core_box bounds must be strictly increasing")
    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            overlap = np.minimum(boxes[left, 1], boxes[right, 1]) - np.maximum(
                boxes[left, 0], boxes[right, 0]
            )
            if np.all(overlap > 1e-9):
                raise ValueError("Tile core boxes overlap in positive volume")
    global_min = boxes[:, 0].min(axis=0)
    global_max = boxes[:, 1].max(axis=0)
    partition_volume = np.prod(boxes[:, 1] - boxes[:, 0], axis=1).sum()
    global_volume = float(np.prod(global_max - global_min))
    if not np.isclose(partition_volume, global_volume, rtol=1e-9, atol=1e-8):
        raise ValueError("Tile core boxes do not form a gap-free rectangular partition")
    return ids, boxes


def assign_core_owners(
    points: np.ndarray,
    tiles: Sequence[dict[str, Any]],
    *,
    tolerance_m: float = 1e-6,
) -> np.ndarray:
    """Assign every point to exactly one core; shared boundaries use min Tile id."""

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("points must be a finite [N, 3] array")
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be non-negative")
    ids, boxes = _ordered_boxes(tiles)
    owners = np.full(len(xyz), -1, dtype=np.int64)
    for tile_id, box in zip(ids, boxes):
        inside = np.all(
            (xyz >= box[0] - float(tolerance_m))
            & (xyz <= box[1] + float(tolerance_m)),
            axis=1,
        )
        owners[(owners < 0) & inside] = tile_id
    if np.any(owners < 0):
        raise ValueError(f"{int(np.count_nonzero(owners < 0))} points have no core owner")
    return owners


def build_core_ownership_contract(
    *, tiles: Sequence[dict[str, Any]], tile_inputs_manifest_sha256: str
) -> dict[str, Any]:
    ids, boxes = _ordered_boxes(tiles)
    unsigned = {
        "schema_version": 1,
        "kind": "tile_core_ownership_contract_v1",
        "tile_inputs_manifest_sha256": str(tile_inputs_manifest_sha256),
        "tile_count": int(len(ids)),
        "tile_ids": ids.tolist(),
        "core_boxes": boxes.tolist(),
        "training_context": "core_plus_halo",
        "export_scope": "core_owner_only",
        "shared_boundary_rule": "minimum_tile_id",
        "merge_algorithm": "cut_each_tile_to_unique_core_owner_then_concatenate",
        "direct_halo_concatenation_allowed": False,
        "training_allowed": False,
        "next_required_artifact": "per_tile_core_cut_count_and_merged_uniqueness_audit",
    }
    result = dict(unsigned)
    result["ownership_contract_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return result
