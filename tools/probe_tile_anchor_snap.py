#!/usr/bin/env python3
"""Probe observation-guided anchors against exact per-Tile LiDAR geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from scipy.spatial import cKDTree

from cloudstudio_3dgs.data.depth_cache import load_xyz_point_cloud
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _tile_observation_points(
    *,
    points: np.ndarray,
    observation_image: np.ndarray,
    observation_point: np.ndarray,
    observation_xy: np.ndarray,
    view_ids: list[str],
    tile: dict[str, Any],
) -> tuple[np.ndarray, int]:
    id_to_index = {sample_id: index for index, sample_id in enumerate(view_ids)}
    starts = np.searchsorted(
        observation_image, np.arange(len(view_ids)), side="left"
    )
    ends = np.searchsorted(
        observation_image, np.arange(len(view_ids)), side="right"
    )
    box = np.asarray(tile["training_and_export_box"], dtype=np.float64)
    point_inside = np.all((points >= box[0]) & (points <= box[1]), axis=1)
    selected_parts: list[np.ndarray] = []
    observation_count = 0
    for view in tile["views"]:
        image_index = id_to_index[str(view["sample_id"])]
        start = int(starts[image_index])
        end = int(ends[image_index])
        if start == end:
            continue
        xy = observation_xy[start:end]
        point_index = observation_point[start:end]
        keep = point_inside[point_index]
        keep &= (xy[:, 0] >= int(view["x"])) & (
            xy[:, 0] < int(view["x"]) + int(view["width"])
        )
        keep &= (xy[:, 1] >= int(view["y"])) & (
            xy[:, 1] < int(view["y"]) + int(view["height"])
        )
        selected_parts.append(point_index[keep])
        observation_count += int(np.count_nonzero(keep))
    if not selected_parts:
        return np.empty(0, dtype=np.int64), 0
    return np.unique(np.concatenate(selected_parts)).astype(np.int64), observation_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--observation-manifest", required=True, type=Path)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument("--tile-geometry", required=True, type=Path)
    parser.add_argument("--tile-geometry-root", required=True, type=Path)
    parser.add_argument("--max-anchors-per-tile", type=int, default=100_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.max_anchors_per_tile <= 0:
        raise ValueError("max-anchors-per-tile must be positive")

    observation_manifest = _read(args.observation_manifest)
    tile_inputs = _read(args.tile_inputs)
    tile_geometry = _read(args.tile_geometry)
    geometry_by_id = {
        int(tile["tile_id"]): tile for tile in tile_geometry["tiles"]
    }
    with np.load(args.observations, allow_pickle=False) as archive:
        anchor_points = np.asarray(archive["points"], dtype=np.float64)
        observation_image = np.asarray(
            archive["train_observation_image"], dtype=np.int64
        )
        observation_point = np.asarray(
            archive["train_observation_point"], dtype=np.int64
        )
        observation_xy = np.asarray(
            archive["train_observation_xy"], dtype=np.float64
        )
    view_ids = [str(value) for value in observation_manifest["train_view_ids"]]
    reports: list[dict[str, Any]] = []
    for tile in tile_inputs["tiles"]:
        tile_id = int(tile["tile_id"])
        point_indexes, observation_count = _tile_observation_points(
            points=anchor_points,
            observation_image=observation_image,
            observation_point=observation_point,
            observation_xy=observation_xy,
            view_ids=view_ids,
            tile=tile,
        )
        if not len(point_indexes):
            raise ValueError(f"Tile_{tile_id} has no observation-guided anchors")
        if len(point_indexes) > args.max_anchors_per_tile:
            selection = np.linspace(
                0, len(point_indexes) - 1, args.max_anchors_per_tile, dtype=np.int64
            )
            probe_indexes = point_indexes[selection]
        else:
            probe_indexes = point_indexes
        initialization = tile["initialization"]
        ply_path = args.tile_inputs_root / str(initialization["path"])
        if _sha256_file(ply_path) != str(initialization["sha256"]):
            raise ValueError(f"Tile_{tile_id} initialization PLY SHA256 mismatch")
        exact_points = load_xyz_point_cloud(
            ply_path, max_points=int(initialization["point_count"])
        )
        geometry = geometry_by_id[tile_id]
        geometry_path = args.tile_geometry_root / str(geometry["geometry"]["path"])
        if _sha256_file(geometry_path) != str(geometry["geometry"]["sha256"]):
            raise ValueError(f"Tile_{tile_id} K7/K30 geometry SHA256 mismatch")
        with np.load(geometry_path, allow_pickle=False) as archive:
            normals = np.asarray(archive["normals"], dtype=np.float32)
            eigenvalues = np.asarray(archive["eigenvalues"], dtype=np.float32)
            scales = np.asarray(archive["scales_m"], dtype=np.float32)
        if len(exact_points) != len(normals):
            raise ValueError(f"Tile_{tile_id} point/geometry counts differ")

        tree = cKDTree(exact_points, compact_nodes=True, balanced_tree=True)
        distances, neighbors = tree.query(
            anchor_points[probe_indexes], k=2, workers=-1
        )
        nearest = neighbors[:, 0]
        displacement = anchor_points[probe_indexes] - exact_points[nearest]
        normal_distance = np.abs(
            np.einsum("ij,ij->i", displacement, normals[nearest])
        )
        values = eigenvalues[nearest]
        nonnegative = np.maximum(values, 0.0)
        trace = np.sum(nonnegative, axis=1)
        planarity = np.zeros(len(values), dtype=np.float64)
        valid_trace = trace > 1e-12
        planarity[valid_trace] = np.clip(
            1.0
            - 3.0 * nonnegative[valid_trace, 0] / trace[valid_trace],
            0.0,
            1.0,
        )
        ambiguity_gap = distances[:, 1] - distances[:, 0]
        accepted = (
            (distances[:, 0] <= 0.10)
            & (normal_distance <= 0.02)
            & (planarity >= 0.50)
        )
        reports.append(
            {
                "tile_id": tile_id,
                "name": tile["name"],
                "candidate_observation_count": observation_count,
                "unique_anchor_count": int(len(point_indexes)),
                "probed_anchor_count": int(len(probe_indexes)),
                "exact_lidar_point_count": int(len(exact_points)),
                "nearest_distance_m": _distribution(distances[:, 0]),
                "normal_distance_m": _distribution(normal_distance),
                "second_minus_first_distance_m": _distribution(ambiguity_gap),
                "planarity": _distribution(planarity),
                "local_tangent_scale_m": _distribution(scales[nearest, 0]),
                "within_1cm_fraction": float(np.mean(distances[:, 0] <= 0.01)),
                "within_2cm_fraction": float(np.mean(distances[:, 0] <= 0.02)),
                "within_5cm_fraction": float(np.mean(distances[:, 0] <= 0.05)),
                "within_10cm_fraction": float(np.mean(distances[:, 0] <= 0.10)),
                "accepted_fraction": float(np.mean(accepted)),
                "accepted_count": int(np.count_nonzero(accepted)),
            }
        )

    minimum_accepted = min(report["accepted_fraction"] for report in reports)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tile_observation_anchor_snap_probe_v1",
        "status": "PASS" if minimum_accepted >= 0.90 else "FAIL",
        "acceptance": {
            "max_nearest_distance_m": 0.10,
            "max_normal_distance_m": 0.02,
            "min_planarity": 0.50,
            "min_per_tile_accepted_fraction": 0.90,
        },
        "source_bindings": {
            "dataset_manifest_sha256": observation_manifest.get(
                "dataset_manifest_sha256"
            ),
            "point_cloud_sha256": observation_manifest.get("point_cloud_sha256"),
            "tile_inputs_manifest_sha256": tile_inputs.get(
                "tile_inputs_manifest_sha256"
            ),
            "tile_geometry_manifest_sha256": tile_geometry.get(
                "tile_geometry_manifest_sha256"
            ),
        },
        "max_anchors_per_tile": args.max_anchors_per_tile,
        "minimum_accepted_fraction": minimum_accepted,
        "tiles": reports,
    }
    payload["probe_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
