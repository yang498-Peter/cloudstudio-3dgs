#!/usr/bin/env python3
"""Compare snapped Tile anchors with partial full-LAS Face4 oracle caches."""

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

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth, load_xyz_point_cloud
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _empty_distribution() -> dict[str, float | None]:
    return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}


def _face_specs(face: dict[str, Any]) -> dict[tuple[str, str], FaceSpec]:
    return {
        (str(camera_id), str(payload["face_id"])): FaceSpec.from_dict(payload)
        for camera_id, camera in face["cameras"].items()
        for payload in camera["faces"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--observation-manifest", required=True, type=Path)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument("--tile-geometry", required=True, type=Path)
    parser.add_argument("--tile-geometry-root", required=True, type=Path)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--oracle-root", required=True, type=Path)
    parser.add_argument("--max-observations-per-tile", type=int, default=20_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    observation_manifest = _read(args.observation_manifest)
    tile_inputs = _read(args.tile_inputs)
    tile_geometry = _read(args.tile_geometry)
    face = _read(args.face_manifest)
    dataset = _read(args.dataset_manifest)
    specs = _face_specs(face)
    images = {str(item["image_id"]): item for item in dataset["images"]}
    geometry_by_id = {
        int(tile["tile_id"]): tile for tile in tile_geometry["tiles"]
    }
    view_ids = [str(value) for value in observation_manifest["train_view_ids"]]
    id_to_image_index = {sample_id: index for index, sample_id in enumerate(view_ids)}
    with np.load(args.observations, allow_pickle=False) as archive:
        anchor_points = np.asarray(archive["points"], dtype=np.float64)
        observation_image = np.asarray(archive["train_observation_image"], dtype=np.int64)
        observation_point = np.asarray(archive["train_observation_point"], dtype=np.int64)
        observation_xy = np.asarray(archive["train_observation_xy"], dtype=np.float64)
    starts = np.searchsorted(observation_image, np.arange(len(view_ids)), side="left")
    ends = np.searchsorted(observation_image, np.arange(len(view_ids)), side="right")

    reports: list[dict[str, Any]] = []
    all_range_delta: list[np.ndarray] = []
    all_pixel_shift: list[np.ndarray] = []
    all_snap_distance: list[np.ndarray] = []
    all_normal_distance: list[np.ndarray] = []
    all_anchor_range_delta: list[np.ndarray] = []
    all_signed_anchor_range_delta: list[np.ndarray] = []
    all_zbuffer_range_delta: list[np.ndarray] = []
    total_candidate = 0
    total_accepted = 0
    total_projected = 0
    total_oracle_hits = 0
    for tile in tile_inputs["tiles"]:
        tile_id = int(tile["tile_id"])
        box = np.asarray(tile["training_and_export_box"], dtype=np.float64)
        point_inside = np.all(
            (anchor_points >= box[0]) & (anchor_points <= box[1]), axis=1
        )
        candidate_points: list[np.ndarray] = []
        candidate_views: list[np.ndarray] = []
        candidate_xy: list[np.ndarray] = []
        selected_views: list[dict[str, Any]] = []
        for view in tile["views"]:
            sample_id = str(view["sample_id"])
            oracle_path = args.oracle_root / "depth" / f"{sample_id.replace('::', '_')}.npz"
            if not oracle_path.is_file():
                continue
            view_slot = len(selected_views)
            selected_views.append({**view, "oracle_path": oracle_path})
            image_index = id_to_image_index[sample_id]
            start = int(starts[image_index])
            end = int(ends[image_index])
            xy = observation_xy[start:end]
            point_index = observation_point[start:end]
            keep = point_inside[point_index]
            keep &= (xy[:, 0] >= int(view["x"])) & (
                xy[:, 0] < int(view["x"]) + int(view["width"])
            )
            keep &= (xy[:, 1] >= int(view["y"])) & (
                xy[:, 1] < int(view["y"]) + int(view["height"])
            )
            count = int(np.count_nonzero(keep))
            if count:
                candidate_points.append(point_index[keep])
                candidate_views.append(np.full(count, view_slot, dtype=np.int32))
                candidate_xy.append(xy[keep])
        point_index = np.concatenate(candidate_points)
        view_slot = np.concatenate(candidate_views)
        source_xy = np.concatenate(candidate_xy)
        if len(point_index) > args.max_observations_per_tile:
            selected = np.linspace(
                0,
                len(point_index) - 1,
                args.max_observations_per_tile,
                dtype=np.int64,
            )
            point_index = point_index[selected]
            view_slot = view_slot[selected]
            source_xy = source_xy[selected]
        total_candidate += len(point_index)

        initialization = tile["initialization"]
        exact_points = load_xyz_point_cloud(
            args.tile_inputs_root / str(initialization["path"]),
            max_points=int(initialization["point_count"]),
        )
        geometry = geometry_by_id[tile_id]
        with np.load(
            args.tile_geometry_root / str(geometry["geometry"]["path"]),
            allow_pickle=False,
        ) as archive:
            normals = np.asarray(archive["normals"], dtype=np.float32)
            eigenvalues = np.asarray(archive["eigenvalues"], dtype=np.float32)
        unique_points, inverse = np.unique(point_index, return_inverse=True)
        tree = cKDTree(exact_points, compact_nodes=True, balanced_tree=True)
        distances, nearest = tree.query(anchor_points[unique_points], k=1, workers=-1)
        displacement = anchor_points[unique_points] - exact_points[nearest]
        normal_distance = np.abs(
            np.einsum("ij,ij->i", displacement, normals[nearest])
        )
        values = np.maximum(eigenvalues[nearest], 0.0)
        trace = np.sum(values, axis=1)
        planarity = np.zeros(len(values), dtype=np.float64)
        valid_trace = trace > 1e-12
        planarity[valid_trace] = np.clip(
            1.0 - 3.0 * values[valid_trace, 0] / trace[valid_trace], 0.0, 1.0
        )
        accepted_unique = (
            (distances <= 0.10)
            & (normal_distance <= 0.02)
            & (planarity >= 0.50)
        )
        accepted = accepted_unique[inverse]
        total_accepted += int(np.count_nonzero(accepted))

        tile_deltas: list[np.ndarray] = []
        tile_shifts: list[np.ndarray] = []
        tile_zbuffer_deltas: list[np.ndarray] = []
        projected_count = 0
        oracle_hit_count = 0
        zbuffer_pixel_count = 0
        zbuffer_oracle_hit_count = 0
        for slot, view in enumerate(selected_views):
            selected = (view_slot == slot) & accepted
            if not np.any(selected):
                continue
            sample_id = str(view["sample_id"])
            image_id, face_id = sample_id.rsplit("::", 1)
            image = images[image_id]
            camera_id = str(image["camera_id"])
            pose = np.asarray(image["c2w"], dtype=np.float64)
            exact_index = nearest[inverse[selected]]
            world = exact_points[exact_index]
            camera = (world - pose[:3, 3]) @ pose[:3, :3]
            spec = specs[(camera_id, face_id)]
            pixels, inside = spec.directions_to_pixels(camera)
            rounded = np.rint(pixels).astype(np.int64)
            inside &= (
                (rounded[:, 0] >= int(view["x"]))
                & (rounded[:, 0] < int(view["x"]) + int(view["width"]))
                & (rounded[:, 1] >= int(view["y"]))
                & (rounded[:, 1] < int(view["y"]) + int(view["height"]))
            )
            if not np.any(inside):
                continue
            pixels = pixels[inside]
            rounded = rounded[inside]
            ranges = np.linalg.norm(camera[inside], axis=1).astype(np.float32)
            original_xy = source_xy[selected][inside]
            original_point_index = point_index[selected][inside]
            anchor_ranges = np.linalg.norm(
                anchor_points[original_point_index] - pose[:3, 3], axis=1
            ).astype(np.float32)
            signed_anchor_range_delta = ranges - anchor_ranges
            accepted_unique_index = inverse[selected][inside]
            pixel_index = (
                rounded[:, 1] * int(spec.width) + rounded[:, 0]
            ).astype(np.int32)
            oracle = load_sparse_depth(Path(view["oracle_path"]))
            positions = np.searchsorted(oracle.pixel_index, pixel_index)
            hit = positions < len(oracle.pixel_index)
            hit[hit] &= oracle.pixel_index[positions[hit]] == pixel_index[hit]
            if np.any(hit):
                delta = np.abs(ranges[hit] - oracle.range_m[positions[hit]])
                snap_distance = distances[accepted_unique_index[hit]]
                snapped_normal_distance = normal_distance[accepted_unique_index[hit]]
                tile_deltas.append(delta)
                tile_shifts.append(np.linalg.norm(pixels[hit] - original_xy[hit], axis=1))
                all_snap_distance.append(snap_distance)
                all_normal_distance.append(snapped_normal_distance)
                all_anchor_range_delta.append(
                    np.abs(signed_anchor_range_delta[hit])
                )
                all_signed_anchor_range_delta.append(signed_anchor_range_delta[hit])
                oracle_hit_count += int(np.count_nonzero(hit))
            order = np.lexsort((ranges, pixel_index))
            sorted_pixels = pixel_index[order]
            first = np.empty(len(order), dtype=bool)
            first[0] = True
            first[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
            winner = order[first]
            winner_pixels = pixel_index[winner]
            winner_ranges = ranges[winner]
            winner_positions = np.searchsorted(oracle.pixel_index, winner_pixels)
            winner_hit = winner_positions < len(oracle.pixel_index)
            winner_hit[winner_hit] &= (
                oracle.pixel_index[winner_positions[winner_hit]]
                == winner_pixels[winner_hit]
            )
            if np.any(winner_hit):
                zbuffer_delta = np.abs(
                    winner_ranges[winner_hit]
                    - oracle.range_m[winner_positions[winner_hit]]
                )
                tile_zbuffer_deltas.append(zbuffer_delta)
                zbuffer_oracle_hit_count += int(np.count_nonzero(winner_hit))
            zbuffer_pixel_count += len(winner_pixels)
            projected_count += len(pixel_index)
        total_projected += projected_count
        total_oracle_hits += oracle_hit_count
        delta_values = np.concatenate(tile_deltas) if tile_deltas else np.empty(0)
        shift_values = np.concatenate(tile_shifts) if tile_shifts else np.empty(0)
        zbuffer_delta_values = (
            np.concatenate(tile_zbuffer_deltas)
            if tile_zbuffer_deltas
            else np.empty(0)
        )
        if len(delta_values):
            all_range_delta.append(delta_values)
            all_pixel_shift.append(shift_values)
        if len(zbuffer_delta_values):
            all_zbuffer_range_delta.append(zbuffer_delta_values)
        reports.append(
            {
                "tile_id": tile_id,
                "name": tile["name"],
                "reference_view_count": len(selected_views),
                "candidate_observation_count": int(len(point_index)),
                "accepted_observation_count": int(np.count_nonzero(accepted)),
                "projected_inside_crop_count": projected_count,
                "oracle_hit_count": oracle_hit_count,
                "oracle_hit_fraction": (
                    float(oracle_hit_count / projected_count) if projected_count else 0.0
                ),
                "absolute_range_delta_m": (
                    _distribution(delta_values) if len(delta_values) else _empty_distribution()
                ),
                "source_to_exact_pixel_shift_px": (
                    _distribution(shift_values) if len(shift_values) else _empty_distribution()
                ),
                "over_10cm_fraction": (
                    float(np.mean(delta_values > 0.10)) if len(delta_values) else 1.0
                ),
                "zbuffer_pixel_count": zbuffer_pixel_count,
                "zbuffer_oracle_hit_count": zbuffer_oracle_hit_count,
                "zbuffer_oracle_hit_fraction": (
                    float(zbuffer_oracle_hit_count / zbuffer_pixel_count)
                    if zbuffer_pixel_count
                    else 0.0
                ),
                "zbuffer_absolute_range_delta_m": (
                    _distribution(zbuffer_delta_values)
                    if len(zbuffer_delta_values)
                    else _empty_distribution()
                ),
                "zbuffer_over_10cm_fraction": (
                    float(np.mean(zbuffer_delta_values > 0.10))
                    if len(zbuffer_delta_values)
                    else 1.0
                ),
            }
        )

    range_delta = np.concatenate(all_range_delta)
    pixel_shift = np.concatenate(all_pixel_shift)
    snap_distance = np.concatenate(all_snap_distance)
    normal_distance = np.concatenate(all_normal_distance)
    anchor_range_delta = np.concatenate(all_anchor_range_delta)
    signed_anchor_range_delta = np.concatenate(all_signed_anchor_range_delta)
    zbuffer_range_delta = np.concatenate(all_zbuffer_range_delta)
    filter_scenarios: list[dict[str, Any]] = []
    for name, keep in (
        ("base", np.ones(len(range_delta), dtype=bool)),
        ("snap_le_2cm", snap_distance <= 0.02),
        (
            "snap_le_2cm_normal_le_1cm",
            (snap_distance <= 0.02) & (normal_distance <= 0.01),
        ),
        (
            "snap_le_2cm_normal_le_1cm_shift_le_2_5px",
            (snap_distance <= 0.02)
            & (normal_distance <= 0.01)
            & (pixel_shift <= 2.5),
        ),
        (
            "snap_le_2cm_normal_le_1cm_shift_le_2px",
            (snap_distance <= 0.02)
            & (normal_distance <= 0.01)
            & (pixel_shift <= 2.0),
        ),
        ("anchor_range_delta_le_1cm", anchor_range_delta <= 0.01),
        ("anchor_range_delta_le_2cm", anchor_range_delta <= 0.02),
        ("anchor_range_delta_le_5cm", anchor_range_delta <= 0.05),
        (
            "snap_le_2cm_anchor_range_delta_le_2cm",
            (snap_distance <= 0.02) & (anchor_range_delta <= 0.02),
        ),
    ):
        retained = range_delta[keep]
        filter_scenarios.append(
            {
                "name": name,
                "retained_count": int(len(retained)),
                "retained_fraction": float(np.mean(keep)),
                "absolute_range_delta_m": _distribution(retained),
                "over_10cm_fraction": float(np.mean(retained > 0.10)),
            }
        )
    outlier = range_delta > 0.10
    status = "PASS"
    if (
        float(np.percentile(zbuffer_range_delta, 50)) > 0.002
        or float(np.percentile(zbuffer_range_delta, 95)) > 0.02
        or float(np.mean(zbuffer_range_delta > 0.10)) >= 0.005
    ):
        status = "FAIL"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tile_snapped_anchor_full_las_oracle_probe_v1",
        "status": status,
        "acceptance": {
            "max_range_delta_p50_m": 0.002,
            "max_range_delta_p95_m": 0.02,
            "max_over_10cm_fraction_exclusive": 0.005,
        },
        "total_candidate_observation_count": total_candidate,
        "total_accepted_observation_count": total_accepted,
        "total_projected_inside_crop_count": total_projected,
        "total_oracle_hit_count": total_oracle_hits,
        "overall_oracle_hit_fraction": float(total_oracle_hits / total_projected),
        "overall_absolute_range_delta_m": _distribution(range_delta),
        "overall_source_to_exact_pixel_shift_px": _distribution(pixel_shift),
        "overall_snap_distance_m": _distribution(snap_distance),
        "overall_normal_distance_m": _distribution(normal_distance),
        "overall_anchor_range_delta_m": _distribution(anchor_range_delta),
        "overall_signed_anchor_range_delta_m": _distribution(
            signed_anchor_range_delta
        ),
        "outlier_snap_distance_m": _distribution(snap_distance[outlier]),
        "outlier_normal_distance_m": _distribution(normal_distance[outlier]),
        "outlier_pixel_shift_px": _distribution(pixel_shift[outlier]),
        "outlier_anchor_range_delta_m": _distribution(anchor_range_delta[outlier]),
        "outlier_signed_anchor_range_delta_m": _distribution(
            signed_anchor_range_delta[outlier]
        ),
        "overall_over_10cm_fraction": float(np.mean(range_delta > 0.10)),
        "overall_zbuffer_absolute_range_delta_m": _distribution(
            zbuffer_range_delta
        ),
        "overall_zbuffer_over_10cm_fraction": float(
            np.mean(zbuffer_range_delta > 0.10)
        ),
        "filter_scenarios": filter_scenarios,
        "tiles": reports,
    }
    payload["probe_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
