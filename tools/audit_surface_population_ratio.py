#!/usr/bin/env python3
"""Edge-to-plane population density ratio for gaussian surface models.

The target allocation puts many small gaussians where geometry is complex and
few on featureless planes. This audit turns that from an impression into one
number per model: voxelize the population, classify each voxel by how much its
members' surface normals agree (the resultant length of the summed unit
shortest-axis directions - planes agree, edges and corners disagree), then
compare member density between the two classes.

    R_rho = median(points per edge voxel) / median(points per planar voxel)

The same definition runs over any mix of delivery PLYs and training
checkpoints, so a reference delivery, an old export and a live run become
directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _load_positions_and_axes(path: Path, surface_rows: int | None):
    """Return (positions, quats, scales, opacities) from a .ply or .pt."""
    if path.suffix.lower() == ".pt":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        params = payload["params"] if "params" in payload else payload
        end = surface_rows if surface_rows else len(params["means"])
        return (
            params["means"][:end].detach().float().numpy(),
            params["quats"][:end].detach().float().numpy(),
            np.exp(params["scales"][:end].detach().float().numpy()),
            _sigmoid(params["opacities"][:end].detach().float().numpy().reshape(-1)),
        )
    from tools.build_three_way_compare import _load_ply_gaussians

    loaded = _load_ply_gaussians(path, None)
    end = surface_rows if surface_rows else len(loaded["means"])
    return (
        loaded["means"][:end].numpy(),
        loaded["quats"][:end].numpy(),
        np.exp(loaded["scales"][:end].numpy()),
        _sigmoid(loaded["opacities"][:end].numpy().reshape(-1)),
    )


def _shortest_axis_directions(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Unit direction of each gaussian's shortest axis in world space."""
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    columns = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1),
            np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1),
            np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1),
        ],
        axis=1,
    )  # (n, column, xyz)
    shortest = np.argmin(scales, axis=1)
    return columns[np.arange(len(q)), shortest]


def _voxel_keys(positions: np.ndarray, voxel_m: float) -> np.ndarray:
    indices = np.floor(positions / voxel_m).astype(np.int64)
    indices -= indices.min(axis=0, keepdims=True)
    spans = indices.max(axis=0) + 1
    return (indices[:, 0] * spans[1] + indices[:, 1]) * spans[2] + indices[:, 2]


def audit(path: Path, *, voxel_m: float, surface_rows: int | None,
          planar_resultant: float, edge_resultant: float, min_count: int,
          min_opacity: float = 0.0) -> dict:
    positions, quats, scales, opacities = _load_positions_and_axes(
        path, surface_rows
    )
    if min_opacity > 0.0:
        alive = opacities >= min_opacity
        positions, quats, scales = positions[alive], quats[alive], scales[alive]
    normals = _shortest_axis_directions(quats, scales)

    keys = _voxel_keys(positions, voxel_m)
    order = np.argsort(keys)
    keys_sorted = keys[order]
    normals_sorted = normals[order]
    unique_keys, starts, counts = np.unique(
        keys_sorted, return_index=True, return_counts=True
    )
    sums = np.add.reduceat(normals_sorted, starts, axis=0)
    # Antipodal normals describe the same plane; a plain vector sum would call
    # a wall with mixed orientations an "edge". Compare against the dominant
    # direction and fold the opposite hemisphere over before summing.
    dominant = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
    alignment = np.abs(np.einsum("ij,ij->i", normals_sorted, dominant[
        np.repeat(np.arange(len(unique_keys)), counts)
    ]))
    folded = np.add.reduceat(alignment, starts)
    resultant = folded / counts

    eligible = counts >= min_count
    planar = eligible & (resultant >= planar_resultant)
    edge = eligible & (resultant <= edge_resultant)

    def _density(mask: np.ndarray) -> dict:
        selected = counts[mask]
        if not len(selected):
            return {"voxels": 0, "median_per_voxel": None, "mean_per_voxel": None}
        return {
            "voxels": int(mask.sum()),
            "median_per_voxel": float(np.median(selected)),
            "mean_per_voxel": float(selected.mean()),
        }

    planar_stats = _density(planar)
    edge_stats = _density(edge)
    ratio = None
    if planar_stats["median_per_voxel"] and edge_stats["median_per_voxel"]:
        ratio = edge_stats["median_per_voxel"] / planar_stats["median_per_voxel"]
    return {
        "source": str(path),
        "gaussian_count": int(len(positions)),
        "min_opacity": min_opacity,
        "voxel_m": voxel_m,
        "occupied_voxels": int(len(unique_keys)),
        "eligible_voxels": int(eligible.sum()),
        "resultant_quartiles": [
            float(v) for v in np.percentile(resultant[eligible], [25, 50, 75])
        ] if eligible.any() else None,
        "planar": planar_stats,
        "edge": edge_stats,
        "edge_to_plane_density_ratio": ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True,
        help="NAME=PATH[:SURFACE_ROWS] - .ply or checkpoint .pt; the optional "
        "row bound drops an appended background layer from the audit",
    )
    parser.add_argument("--voxel-m", type=float, default=0.25)
    parser.add_argument("--planar-resultant", type=float, default=0.9)
    parser.add_argument("--edge-resultant", type=float, default=0.6)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument(
        "--min-opacity", type=float, default=0.0,
        help="count only gaussians at or above this opacity, so dead mass "
        "cannot masquerade as allocated capacity",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = []
    for item in args.source:
        name, _, rest = item.partition("=")
        location, _, bound = rest.rpartition(":")
        if not bound.isdigit():
            # No row bound - the colon we split on was the drive letter's.
            location, bound = rest, ""
        report = audit(
            Path(location),
            voxel_m=args.voxel_m,
            surface_rows=int(bound) if bound else None,
            planar_resultant=args.planar_resultant,
            edge_resultant=args.edge_resultant,
            min_count=args.min_count,
            min_opacity=args.min_opacity,
        )
        report["name"] = name
        results.append(report)
        ratio = report["edge_to_plane_density_ratio"]
        print(
            f"{name}: R_rho={ratio:.2f}" if ratio else f"{name}: R_rho=n/a",
            f"({report['gaussian_count']:,} gaussians, "
            f"edge {report['edge']['voxels']:,} vx @ "
            f"{report['edge']['median_per_voxel']}, "
            f"plane {report['planar']['voxels']:,} vx @ "
            f"{report['planar']['median_per_voxel']})",
        )
    if args.output:
        args.output.write_text(
            json.dumps({"schema_version": 1, "sources": results}, indent=1),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
