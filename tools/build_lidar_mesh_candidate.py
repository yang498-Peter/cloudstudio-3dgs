"""Build and audit a conservative LiDAR surface candidate for D0-A.

This is an observable-output compatibility experiment, not a claim about the
vendor's unknown mesh topology algorithm. The validation path removes complete
3D blocks before reconstruction and measures the held-out samples against the
resulting surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.geometry.spatial_block_holdout import (
    build_spatial_block_holdout,
)
from cloudstudio_3dgs.training.trainer import load_initialization_ply


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {}
    q = np.quantile(finite, [0.5, 0.95, 0.99])
    return {"p50": float(q[0]), "p95": float(q[1]), "p99": float(q[2])}


def _mesh_from_bpa(
    xyz: np.ndarray,
    normals: np.ndarray,
    *,
    voxel_m: float,
    radii_m: list[float],
    max_edge_m: float,
):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    down = cloud.voxel_down_sample(float(voxel_m))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        down, o3d.utility.DoubleVector([float(value) for value in radii_m])
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles):
        tri_vertices = vertices[triangles]
        edges = np.stack(
            [
                np.linalg.norm(tri_vertices[:, 0] - tri_vertices[:, 1], axis=1),
                np.linalg.norm(tri_vertices[:, 1] - tri_vertices[:, 2], axis=1),
                np.linalg.norm(tri_vertices[:, 2] - tri_vertices[:, 0], axis=1),
            ],
            axis=1,
        )
        keep = np.max(edges, axis=1) <= float(max_edge_m)
        mesh.triangles = o3d.utility.Vector3iVector(triangles[keep])
        mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return down, mesh


def _heldout_distance(mesh, points: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
    import open3d as o3d

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    distances: list[np.ndarray] = []
    for start in range(0, len(points), batch_size):
        query = o3d.core.Tensor(
            np.asarray(points[start : start + batch_size], dtype=np.float32)
        )
        distances.append(scene.compute_distance(query).numpy())
    return np.concatenate(distances) if distances else np.empty(0, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization-ply", type=Path, required=True)
    parser.add_argument("--geometry-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-m", type=float, default=0.015)
    parser.add_argument("--holdout-block-m", type=float, default=0.25)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--no-holdout", action="store_true")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--radii-m", type=float, nargs="+", default=[0.015, 0.03, 0.06])
    parser.add_argument("--max-edge-m", type=float, default=0.08)
    args = parser.parse_args()

    import open3d as o3d

    started = time.perf_counter()
    xyz, _rgb = load_initialization_ply(args.initialization_ply)
    with np.load(args.geometry_npz, allow_pickle=False) as geometry:
        normals = np.asarray(geometry["normals"], dtype=np.float32)
    if normals.shape != xyz.shape:
        raise ValueError("geometry normals do not match initialization point count")
    holdout = None
    if not args.no_holdout:
        holdout = build_spatial_block_holdout(
            xyz,
            block_size_m=args.holdout_block_m,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        )
    construction_mask = (
        np.ones(len(xyz), dtype=bool)
        if holdout is None
        else holdout.construction_mask
    )
    construction_xyz = xyz[construction_mask]
    construction_normals = normals[construction_mask]
    down, mesh = _mesh_from_bpa(
        construction_xyz,
        construction_normals,
        voxel_m=args.voxel_m,
        radii_m=args.radii_m,
        max_edge_m=args.max_edge_m,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output / (
        "tile_lidar_surface_bpa_full.ply"
        if holdout is None
        else "tile_lidar_surface_bpa_holdout.ply"
    )
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False):
        raise RuntimeError("Open3D failed to write the mesh")
    heldout_distance = (
        np.empty(0, dtype=np.float32)
        if holdout is None
        else _heldout_distance(mesh, xyz[holdout.holdout_mask])
    )
    finite = heldout_distance[np.isfinite(heldout_distance)]
    report = {
        "schema_version": 1,
        "kind": "lidar_surface_candidate_block_holdout_audit",
        "status": "CANDIDATE_NOT_VENDOR_TOPOLOGY_EQUIVALENT",
        "algorithm": {
            "family": "Open3D ball pivoting",
            "open3d_version": o3d.__version__,
            "voxel_m": args.voxel_m,
            "radii_m": args.radii_m,
            "max_triangle_edge_m": args.max_edge_m,
            "vendor_mesh_topology": "UNKNOWN",
        },
        "sources": {
            "initialization_ply": str(args.initialization_ply.resolve()),
            "initialization_ply_sha256": _sha256(args.initialization_ply),
            "geometry_npz": str(args.geometry_npz.resolve()),
            "geometry_npz_sha256": _sha256(args.geometry_npz),
        },
        "holdout": None if holdout is None else {
            "block_size_m": holdout.block_size_m,
            "target_fraction": holdout.target_fraction,
            "actual_fraction": holdout.actual_fraction,
            "seed": holdout.seed,
            "construction_points": int(np.count_nonzero(holdout.construction_mask)),
            "heldout_points": int(np.count_nonzero(holdout.holdout_mask)),
        },
        "surface": {
            "downsampled_construction_points": len(down.points),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "mesh_path": mesh_path.name,
            "mesh_sha256": _sha256(mesh_path),
        },
        "heldout_unsigned_point_to_mesh_m": {
            "finite_count": int(len(finite)),
            "quantiles": _quantiles(finite),
            "gt_0p05_fraction": float(np.mean(finite > 0.05)) if len(finite) else None,
            "gt_0p10_fraction": float(np.mean(finite > 0.10)) if len(finite) else None,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": "per_view_mesh_depth_normal_raster_and_boundary_audit",
    }
    report_path = args.output / "mesh_candidate_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
