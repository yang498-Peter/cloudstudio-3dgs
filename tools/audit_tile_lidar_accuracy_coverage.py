#!/usr/bin/env python3
"""Audit Tile LiDAR accuracy without hiding the coverage denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.face_lidar_geometry import verify_face_lidar_geometry_manifest
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.lidar_accuracy_coverage import compare_tile_to_source
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quantiles(values: np.ndarray, points: tuple[int, ...]) -> dict[str, float | None]:
    if not len(values):
        return {f"p{point}": None for point in points}
    return {
        f"p{point}": float(np.percentile(values, point)) for point in points
    }


def _empty_depth(shape: list[int]) -> SparseDepthMap:
    return SparseDepthMap(
        (int(shape[0]), int(shape[1])),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-id", required=True, type=int)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--tile-manifest", required=True, type=Path)
    parser.add_argument("--tile-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--edge-threshold-m", type=float, default=0.10)
    args = parser.parse_args()

    tile_inputs = _read(args.tile_inputs)
    matches = [
        tile for tile in tile_inputs["tiles"] if int(tile["tile_id"]) == args.tile_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Tile input manifest has no unique Tile {args.tile_id}")
    views = {str(view["sample_id"]): view for view in matches[0]["views"]}

    source_manifest = _read(args.source_manifest)
    source_sha = verify_face_lidar_geometry_manifest(source_manifest)
    tile_manifest = _read(args.tile_manifest)
    tile_sha = verify_face_lidar_geometry_manifest(tile_manifest)
    if tile_manifest.get("source_face_lidar_geometry_manifest_sha256") != source_sha:
        raise ValueError("Tile geometry is not derived from the supplied source geometry")
    if int(tile_manifest.get("tile_id", -1)) != args.tile_id:
        raise ValueError("Tile geometry manifest has a different Tile id")
    source_records = {
        str(record["sample_id"]): record for record in source_manifest["records"]
    }
    tile_records = {
        str(record["sample_id"]): record for record in tile_manifest["records"]
    }
    if set(tile_records) != set(views):
        raise ValueError("Tile geometry records differ from Tile views")

    total_candidate = 0
    total_retained = 0
    total_over_5cm = 0
    total_over_10cm = 0
    exact_error_max = 0.0
    sampled_errors: list[np.ndarray] = []
    confidence = {
        name: {"candidate_pixels": 0, "retained_pixels": 0, "over_5cm_count": 0, "over_10cm_count": 0, "error_max_m": 0.0}
        for name in ("low", "medium", "high")
    }
    edge = {
        name: {"candidate_pixels": 0, "retained_pixels": 0, "over_5cm_count": 0, "over_10cm_count": 0}
        for name in ("edge", "nonedge")
    }
    per_view: list[dict[str, Any]] = []
    for sample_id, view in views.items():
        source_record = source_records.get(sample_id)
        tile_record = tile_records[sample_id]
        if source_record is None:
            raise ValueError(f"source geometry lacks {sample_id}")
        source = (
            _empty_depth(source_record["shape"])
            if int(source_record["valid_pixels"]) == 0
            else load_sparse_depth(args.source_root / str(source_record["path"]))
        )
        tile = (
            _empty_depth(tile_record["shape"])
            if int(tile_record["valid_pixels"]) == 0
            else load_sparse_depth(args.tile_root / str(tile_record["path"]))
        )
        comparison = compare_tile_to_source(
            source,
            tile,
            crop_xywh=(
                int(view["x"]),
                int(view["y"]),
                int(view["width"]),
                int(view["height"]),
            ),
            edge_threshold_m=args.edge_threshold_m,
        )
        errors = comparison.pop("errors_m")
        total_candidate += comparison["candidate_pixels"]
        total_retained += comparison["retained_pixels"]
        over_5 = int(np.count_nonzero(errors > 0.05))
        over_10 = int(np.count_nonzero(errors > 0.10))
        total_over_5cm += over_5
        total_over_10cm += over_10
        if len(errors):
            exact_error_max = max(exact_error_max, float(np.max(errors)))
            stride = max(1, len(errors) // 4096)
            sampled_errors.append(errors[::stride][:4096])
        for name, values in comparison["confidence"].items():
            for key in (
                "candidate_pixels",
                "retained_pixels",
                "over_5cm_count",
                "over_10cm_count",
            ):
                confidence[name][key] += int(values[key])
            if values["error_max_m"] is not None:
                confidence[name]["error_max_m"] = max(
                    confidence[name]["error_max_m"], float(values["error_max_m"])
                )
        for name in ("edge", "nonedge"):
            for key in edge[name]:
                edge[name][key] += int(comparison[name][key])
        per_view.append(
            {
                "sample_id": sample_id,
                "candidate_pixels": comparison["candidate_pixels"],
                "retained_pixels": comparison["retained_pixels"],
                "coverage_fraction": comparison["coverage_fraction"],
                "error_p95_m": None
                if not len(errors)
                else float(np.percentile(errors, 95)),
                "error_max_m": None if not len(errors) else float(np.max(errors)),
                "over_5cm_count": over_5,
                "over_10cm_count": over_10,
            }
        )

    for values in confidence.values():
        values["coverage_fraction"] = (
            None
            if values["candidate_pixels"] == 0
            else values["retained_pixels"] / values["candidate_pixels"]
        )
    for values in edge.values():
        values["coverage_fraction"] = (
            None
            if values["candidate_pixels"] == 0
            else values["retained_pixels"] / values["candidate_pixels"]
        )
    coverage_values = np.asarray(
        [
            record["coverage_fraction"]
            for record in per_view
            if record["coverage_fraction"] is not None
        ],
        dtype=np.float64,
    )
    error_sample = (
        np.concatenate(sampled_errors) if sampled_errors else np.empty(0, dtype=np.float64)
    )
    unsigned = {
        "schema_version": 1,
        "kind": "tile_lidar_accuracy_coverage_audit_v1",
        "status": "ACCURACY_COVERAGE_READY"
        if exact_error_max <= 1e-6 and total_over_10cm == 0
        else "ACCURACY_FAIL",
        "tile_id": args.tile_id,
        "source_face_lidar_geometry_manifest_sha256": source_sha,
        "tile_face_lidar_geometry_manifest_sha256": tile_sha,
        "depth_semantics": "euclidean_ray_range_m",
        "candidate_definition": "authoritative_source_rays_inside_tile_view_crop_before_world_box_filter",
        "retained_definition": "candidate_rays_inside_tile_core_plus_halo_world_box",
        "candidate_pixels": total_candidate,
        "retained_pixels": total_retained,
        "coverage_fraction": None
        if total_candidate == 0
        else total_retained / total_candidate,
        "per_view_coverage": _quantiles(coverage_values, (5, 10, 50, 95, 99)),
        "accuracy_m": {
            **_quantiles(error_sample, (50, 95, 99)),
            "quantile_sample_count": int(len(error_sample)),
            "max": exact_error_max,
            "over_5cm_count": total_over_5cm,
            "over_5cm_fraction": 0.0
            if total_retained == 0
            else total_over_5cm / total_retained,
            "over_10cm_count": total_over_10cm,
            "over_10cm_fraction": 0.0
            if total_retained == 0
            else total_over_10cm / total_retained,
        },
        "confidence_strata": confidence,
        "depth_discontinuity_strata": {
            "threshold_m": float(args.edge_threshold_m),
            **edge,
        },
        "unavailable_strata": {
            "low_incidence_angle": "NOT_AVAILABLE_WITH_RANGE_ONLY_SIDECAR",
            "semantic_wall_corner": "NOT_AVAILABLE_WITHOUT_PIXEL_SEMANTICS_OR_NORMALS",
        },
        "per_view": per_view,
        "training_allowed": False,
        "next_required_artifact": "fixed_topology_directional_smoke",
    }
    report = dict(unsigned)
    report["audit_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {key: report[key] for key in report if key != "per_view"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "ACCURACY_COVERAGE_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
