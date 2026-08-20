#!/usr/bin/env python3
"""Build a deterministic, budget-safe S1 LiDAR initialization PLY."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.point_cloud import (
    VoxelInitializationConfig,
    build_lidar_initialization,
    estimate_local_geometry,
    write_binary_ply,
    write_report,
)


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a JSON object")
    unknown = set(payload) - {
        "target_points",
        "cap_max",
        "voxel_size",
        "edge_preservation_ratio",
        "seed",
        "chunk_size",
        "auto_tolerance",
        "auto_max_passes",
    }
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    return payload


def _replace_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="local-coordinate S1 solver run")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target-points", type=int)
    parser.add_argument("--cap-max", type=int)
    parser.add_argument("--voxel-size", help="'auto' or a positive size in metres")
    parser.add_argument("--edge-preservation-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--with-pca", action="store_true")
    parser.add_argument("--pca-neighbors", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    values = _load_config(args.config)
    overrides = {
        "target_points": args.target_points,
        "cap_max": args.cap_max,
        "edge_preservation_ratio": args.edge_preservation_ratio,
        "seed": args.seed,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    if args.voxel_size is not None:
        values["voxel_size"] = (
            "auto" if args.voxel_size == "auto" else float(args.voxel_size)
        )
    config = VoxelInitializationConfig(**values)

    args.output.mkdir(parents=True, exist_ok=True)
    destinations = [args.output / "sparse_pc.ply", args.output / "lidar_init_report.json"]
    if args.with_pca:
        destinations.append(args.output / "lidar_init_geometry.npz")
    existing = [path for path in destinations if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to replace existing output without --force: "
            + ", ".join(path.name for path in existing)
        )

    result = build_lidar_initialization(args.run, config)
    report = dict(result.report)
    report["local_geometry"] = {"computed": False}

    ply_temp = args.output / ".sparse_pc.ply.tmp"
    report_temp = args.output / ".lidar_init_report.json.tmp"
    geometry_temp = args.output / ".lidar_init_geometry.npz.tmp"
    try:
        write_binary_ply(ply_temp, result.xyz, result.rgb)
        if args.with_pca:
            normals, eigenvalues, covariance = estimate_local_geometry(
                result.xyz, neighbors=args.pca_neighbors
            )
            with geometry_temp.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    normals=normals,
                    eigenvalues=eigenvalues,
                    covariance=covariance,
                )
            report["local_geometry"] = {
                "computed": True,
                "neighbors": min(args.pca_neighbors, len(result.xyz)),
                "file": "lidar_init_geometry.npz",
            }
        write_report(report_temp, report)
        _replace_file(ply_temp, args.output / "sparse_pc.ply")
        if args.with_pca:
            _replace_file(geometry_temp, args.output / "lidar_init_geometry.npz")
        _replace_file(report_temp, args.output / "lidar_init_report.json")
    finally:
        for temporary in (ply_temp, report_temp, geometry_temp):
            temporary.unlink(missing_ok=True)

    output = report["output"]
    coverage = report["coverage"]
    print(
        f"wrote {output['point_count']:,} points at voxel {output['voxel_size']:.6g} m "
        f"(stride coverage {coverage['stride_coverage_ratio']:.3%}, "
        f"voxel coverage {coverage['voxel_coverage_ratio']:.3%}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
