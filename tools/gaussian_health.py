"""Gaussian Health Metrics: quantify pathological gaussians in a 3DGS checkpoint.

Turns ad-hoc eyeballing of giant / needle / floater / wall-thickness pathologies
into a formal, reproducible report against the LiDAR initialization cloud.

Usage:
    python tools/gaussian_health.py --checkpoint ckpt.pt --lidar-ply sparse_pc.ply \
        [--planes 6] [--output health.json]

Importable API:
    compute_health(params_dict, lidar_xyz, max_planes=6, seed=0) -> dict

Checkpoint contract: torch.load(path)["params"] with means[N,3], scales[N,3]
(log domain; exp -> metric axis lengths), quats[N,4] (wxyz, not necessarily
normalized), opacities[N] (logit), plus color params (unused here).

CPU only by design: tensors are loaded with map_location="cpu" and converted to
numpy immediately.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCHEMA_VERSION = "gaussian-health-1.0"

# --- thresholds (all echoed into the JSON report) -------------------------
GIANT_MAX_AXIS_THRESHOLDS_M = (0.5, 1.0, 5.0, 20.0)
NEEDLE_RATIO_THRESHOLDS = (10.0, 30.0, 100.0)
DEGENERATE_MIN_AXIS_M = 1e-4

OPACITY_DEAD_MAX = 0.005
OPACITY_FOG_MAX = 0.1
OPACITY_HARD_MIN = 0.95

FLOATER_MIN_OPACITY = 0.1
FLOATER_DIST_THRESHOLDS_M = (0.3, 1.0, 5.0)

PLANE_RANSAC_ITERS = 400
PLANE_INLIER_THRESHOLD_M = 0.03
PLANE_MIN_INLIER_FRACTION = 0.05
PLANE_LIDAR_BAND_M = 0.05
WALL_GAUSSIAN_BAND_M = 0.15
WALL_MIN_OPACITY = 0.1
RANSAC_MAX_SAMPLE_POINTS = 200_000

THRESHOLDS = {
    "giant_max_axis_thresholds_m": list(GIANT_MAX_AXIS_THRESHOLDS_M),
    "needle_ratio_thresholds": list(NEEDLE_RATIO_THRESHOLDS),
    "degenerate_min_axis_m": DEGENERATE_MIN_AXIS_M,
    "opacity_dead_max": OPACITY_DEAD_MAX,
    "opacity_fog_max": OPACITY_FOG_MAX,
    "opacity_hard_min": OPACITY_HARD_MIN,
    "floater_min_opacity": FLOATER_MIN_OPACITY,
    "floater_dist_thresholds_m": list(FLOATER_DIST_THRESHOLDS_M),
    "plane_ransac_iters": PLANE_RANSAC_ITERS,
    "plane_inlier_threshold_m": PLANE_INLIER_THRESHOLD_M,
    "plane_min_inlier_fraction": PLANE_MIN_INLIER_FRACTION,
    "plane_lidar_band_m": PLANE_LIDAR_BAND_M,
    "wall_gaussian_band_m": WALL_GAUSSIAN_BAND_M,
    "wall_min_opacity": WALL_MIN_OPACITY,
}

_PLY_TYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_ply_xyz(path: Path) -> np.ndarray:
    """Read vertex x/y/z from a binary-little-endian or ascii PLY as float64 [N,3].

    Handles arbitrary extra scalar vertex properties (rgb, intensity, ...).
    List properties and non-vertex elements after vertex are not supported for
    binary files unless the vertex element comes first (true for our writer).
    """
    path = Path(path)
    with path.open("rb") as stream:
        magic = stream.readline().strip()
        if magic != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fmt = None
        vertex_count = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"unterminated PLY header: {path}")
            tokens = line.decode("ascii", errors="replace").strip().split()
            if not tokens or tokens[0] == "comment":
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                in_vertex = tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
                elif vertex_count is None:
                    raise ValueError("PLY vertex element must come first")
            elif tokens[0] == "property" and in_vertex:
                if tokens[1] == "list":
                    raise ValueError("list properties on vertex are unsupported")
                if tokens[1] not in _PLY_TYPE_MAP:
                    raise ValueError(f"unsupported PLY property type {tokens[1]}")
                properties.append((tokens[-1], _PLY_TYPE_MAP[tokens[1]]))
            elif tokens[0] == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        names = [name for name, _ in properties]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"PLY vertex missing property {axis!r}: {path}")
        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, "<" + kind) for name, kind in properties])
            payload = stream.read(vertex_count * dtype.itemsize)
            records = np.frombuffer(payload, dtype=dtype)
        elif fmt == "ascii":
            dtype = np.dtype([(name, kind) for name, kind in properties])
            rows = np.loadtxt(stream, dtype=np.float64, max_rows=vertex_count, ndmin=2)
            records = np.zeros(vertex_count, dtype=dtype)
            for index, (name, _) in enumerate(properties):
                records[name] = rows[:, index]
        else:
            raise ValueError(f"unsupported PLY format {fmt!r}: {path}")
        if len(records) != vertex_count:
            raise ValueError(
                f"PLY truncated: expected {vertex_count} vertices, got {len(records)}"
            )
        xyz = np.stack(
            [records["x"], records["y"], records["z"]], axis=1
        ).astype(np.float64)
        return xyz


def _to_numpy(value) -> np.ndarray:
    """Convert a torch tensor (any device) or array-like to a float64 ndarray."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _quats_to_rotmats(quats: np.ndarray) -> np.ndarray:
    """wxyz quaternions (not necessarily normalized) -> [N,3,3] rotation matrices."""
    q = np.asarray(quats, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    w, x, y, z = (q / norm).T
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - w * z)
    matrices[:, 0, 2] = 2 * (x * z + w * y)
    matrices[:, 1, 0] = 2 * (x * y + w * z)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - w * x)
    matrices[:, 2, 0] = 2 * (x * z - w * y)
    matrices[:, 2, 1] = 2 * (y * z + w * x)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def _percentiles(values: np.ndarray, points=(50, 95, 99)) -> dict:
    if values.size == 0:
        return {f"p{str(p).replace('.', '_')}": None for p in points}
    return {
        f"p{str(p).replace('.', '_')}": float(np.percentile(values, p))
        for p in points
    }


def _scale_metrics(scales_m: np.ndarray, opacity: np.ndarray) -> dict:
    max_axis = scales_m.max(axis=1)
    min_axis = scales_m.min(axis=1)
    ratio = max_axis / np.maximum(min_axis, 1e-12)
    giants = {}
    for threshold in GIANT_MAX_AXIS_THRESHOLDS_M:
        mask = max_axis > threshold
        giants[f"gt_{threshold}m"] = {
            "count": int(mask.sum()),
            "opacity_median": float(np.median(opacity[mask])) if mask.any() else None,
        }
    needles = {
        f"ratio_gt_{int(threshold)}": int((ratio > threshold).sum())
        for threshold in NEEDLE_RATIO_THRESHOLDS
    }
    return {
        "count": int(len(scales_m)),
        "max_axis_m": {
            **_percentiles(max_axis, (50, 95, 99, 99.9)),
            "max": float(max_axis.max()) if max_axis.size else None,
        },
        "giant": giants,
        "needle": needles,
        "degenerate_min_axis_lt_1e4_count": int(
            (min_axis < DEGENERATE_MIN_AXIS_M).sum()
        ),
    }


def _opacity_metrics(opacity: np.ndarray) -> dict:
    total = max(len(opacity), 1)
    dead = opacity < OPACITY_DEAD_MAX
    fog = (opacity >= OPACITY_DEAD_MAX) & (opacity < OPACITY_FOG_MAX)
    hard = opacity > OPACITY_HARD_MIN
    hist_counts, hist_edges = np.histogram(opacity, bins=10, range=(0.0, 1.0))
    return {
        "count": int(len(opacity)),
        **_percentiles(opacity, (5, 50, 95)),
        "mean": float(opacity.mean()) if opacity.size else None,
        "histogram": {
            "bin_edges": [float(edge) for edge in hist_edges],
            "counts": [int(count) for count in hist_counts],
        },
        "dead_lt_0_005": {"count": int(dead.sum()), "fraction": float(dead.sum() / total)},
        "fog_0_005_to_0_1": {"count": int(fog.sum()), "fraction": float(fog.sum() / total)},
        "hard_gt_0_95": {"count": int(hard.sum()), "fraction": float(hard.sum() / total)},
    }


def _floater_metrics(
    means: np.ndarray,
    opacity: np.ndarray,
    scales_m: np.ndarray,
    tree: cKDTree,
) -> dict:
    visible = opacity > FLOATER_MIN_OPACITY
    result = {
        "definition": (
            "visible gaussians (opacity > {:.2f}) by nearest-neighbor distance "
            "to the LiDAR cloud; outliers far from measured geometry are floaters"
        ).format(FLOATER_MIN_OPACITY),
        "visible_count": int(visible.sum()),
    }
    if not visible.any():
        result["distance_m"] = _percentiles(np.array([]))
        result["outliers"] = {}
        return result
    distances, _ = tree.query(means[visible], workers=-1)
    visible_opacity = opacity[visible]
    visible_max_axis = scales_m[visible].max(axis=1)
    outliers = {}
    for threshold in FLOATER_DIST_THRESHOLDS_M:
        mask = distances > threshold
        outliers[f"gt_{threshold}m"] = {
            "count": int(mask.sum()),
            "mean_opacity": float(visible_opacity[mask].mean()) if mask.any() else None,
            "mean_max_axis_m": float(visible_max_axis[mask].mean()) if mask.any() else None,
        }
    result["distance_m"] = _percentiles(distances, (50, 95, 99))
    result["outliers"] = outliers
    return result


def fit_planes_ransac(
    lidar_xyz: np.ndarray, max_planes: int, rng: np.random.Generator
) -> list[dict]:
    """Sequential RANSAC: fit up to max_planes dominant planes.

    Each accepted plane must claim >= PLANE_MIN_INLIER_FRACTION of the original
    cloud within PLANE_INLIER_THRESHOLD_M. Inliers are removed between rounds.
    Returns [{normal, offset, inlier_count, inlier_indices}] with unit normals
    and plane equation dot(normal, p) + offset = 0.
    """
    total = len(lidar_xyz)
    if total < 3:
        return []
    remaining = np.arange(total)
    planes = []
    min_inliers = max(int(math.ceil(total * PLANE_MIN_INLIER_FRACTION)), 3)
    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        points = lidar_xyz[remaining]
        # Cap the candidate pool so each iteration stays cheap on huge clouds.
        if len(points) > RANSAC_MAX_SAMPLE_POINTS:
            pool = rng.choice(len(points), RANSAC_MAX_SAMPLE_POINTS, replace=False)
            candidate_points = points[pool]
        else:
            candidate_points = points
        best_count = 0
        best_plane = None
        for _ in range(PLANE_RANSAC_ITERS):
            sample = rng.choice(len(candidate_points), 3, replace=False)
            a, b, c = candidate_points[sample]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            offset = -float(normal @ a)
            distances = np.abs(candidate_points @ normal + offset)
            count = int((distances <= PLANE_INLIER_THRESHOLD_M).sum())
            if count > best_count:
                best_count = count
                best_plane = (normal, offset)
        if best_plane is None:
            break
        normal, offset = best_plane
        # Refine on full remaining set: centroid + smallest covariance eigenvector.
        distances = np.abs(points @ normal + offset)
        inlier_mask = distances <= PLANE_INLIER_THRESHOLD_M
        if inlier_mask.sum() >= 3:
            inlier_points = points[inlier_mask]
            centroid = inlier_points.mean(axis=0)
            centered = inlier_points - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            refined_normal = vh[-1]
            refined_offset = -float(refined_normal @ centroid)
            refined_distances = np.abs(points @ refined_normal + refined_offset)
            refined_mask = refined_distances <= PLANE_INLIER_THRESHOLD_M
            if refined_mask.sum() >= inlier_mask.sum():
                normal, offset, inlier_mask = refined_normal, refined_offset, refined_mask
        inlier_count = int(inlier_mask.sum())
        if inlier_count < min_inliers:
            break
        planes.append(
            {
                "normal": normal,
                "offset": offset,
                "inlier_count": inlier_count,
                "inlier_indices": remaining[inlier_mask],
            }
        )
        remaining = remaining[~inlier_mask]
    return planes


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(normal @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _wall_metrics(
    means: np.ndarray,
    rotmats: np.ndarray,
    scales_m: np.ndarray,
    opacity: np.ndarray,
    lidar_xyz: np.ndarray,
    planes: list[dict],
) -> dict:
    per_plane = []
    for index, plane in enumerate(planes):
        normal = plane["normal"]
        offset = plane["offset"]
        inliers = lidar_xyz[plane["inlier_indices"]]
        band_mask = np.abs(inliers @ normal + offset) <= PLANE_LIDAR_BAND_M
        band_points = inliers[band_mask]
        entry = {
            "plane_index": index,
            "normal": [float(component) for component in normal],
            "offset": float(offset),
            "lidar_inlier_count": int(plane["inlier_count"]),
            "gaussian_count": 0,
            "center_normal_rms_m": None,
            "effective_thickness_m": {"p50": None, "p95": None},
            "shortest_axis_angle_deg_p50": None,
        }
        if len(band_points) < 3:
            per_plane.append(entry)
            continue
        u, v = _plane_basis(normal)
        band_u = band_points @ u
        band_v = band_points @ v
        u_min, u_max = float(band_u.min()), float(band_u.max())
        v_min, v_max = float(band_v.min()), float(band_v.max())
        signed = means @ normal + offset
        mean_u = means @ u
        mean_v = means @ v
        selected = (
            (opacity > WALL_MIN_OPACITY)
            & (np.abs(signed) <= WALL_GAUSSIAN_BAND_M)
            & (mean_u >= u_min)
            & (mean_u <= u_max)
            & (mean_v >= v_min)
            & (mean_v <= v_max)
        )
        count = int(selected.sum())
        entry["gaussian_count"] = count
        if count == 0:
            per_plane.append(entry)
            continue
        deviation = signed[selected]
        rotations = rotmats[selected]
        axis_scales = scales_m[selected]
        # sigma along the plane normal: || diag(scales) . R^T . n ||
        normal_in_local = np.einsum("nij,j->ni", rotations.transpose(0, 2, 1), normal)
        sigma_normal = np.linalg.norm(axis_scales * normal_in_local, axis=1)
        spread = np.sqrt(deviation**2 + sigma_normal**2)
        shortest_axis_index = np.argmin(axis_scales, axis=1)
        shortest_axes = rotations[np.arange(count), :, shortest_axis_index]
        cosines = np.clip(np.abs(shortest_axes @ normal), 0.0, 1.0)
        angles_deg = np.degrees(np.arccos(cosines))
        entry["center_normal_rms_m"] = float(np.sqrt(np.mean(deviation**2)))
        entry["effective_thickness_m"] = {
            "p50": float(np.percentile(spread, 50)),
            "p95": float(np.percentile(spread, 95)),
        }
        entry["shortest_axis_angle_deg_p50"] = float(np.percentile(angles_deg, 50))
        per_plane.append(entry)

    weighted = {"center_normal_rms_m": None, "effective_thickness_p50_m": None,
                "effective_thickness_p95_m": None, "shortest_axis_angle_deg_p50": None}
    usable = [entry for entry in per_plane if entry["gaussian_count"] > 0]
    if usable:
        weights = np.array([entry["lidar_inlier_count"] for entry in usable], dtype=np.float64)
        weights /= weights.sum()

        def _weighted(key_path):
            values = []
            for entry in usable:
                value = entry
                for key in key_path:
                    value = value[key]
                values.append(value)
            return float(np.dot(weights, np.array(values, dtype=np.float64)))

        weighted = {
            "center_normal_rms_m": _weighted(("center_normal_rms_m",)),
            "effective_thickness_p50_m": _weighted(("effective_thickness_m", "p50")),
            "effective_thickness_p95_m": _weighted(("effective_thickness_m", "p95")),
            "shortest_axis_angle_deg_p50": _weighted(("shortest_axis_angle_deg_p50",)),
        }
    return {
        "plane_count": len(per_plane),
        "planes": per_plane,
        "weighted_by_lidar_inliers": weighted,
    }


def compute_health(
    params_dict,
    lidar_xyz,
    max_planes: int = 6,
    seed: int = 0,
) -> dict:
    """Compute all gaussian health metrics.

    params_dict: mapping with means/scales/quats/opacities (torch tensors or arrays).
    lidar_xyz: [M,3] LiDAR initialization points (array-like).
    """
    means = _to_numpy(params_dict["means"]).reshape(-1, 3)
    scales_log = _to_numpy(params_dict["scales"]).reshape(-1, 3)
    quats = _to_numpy(params_dict["quats"]).reshape(-1, 4)
    opacity = _sigmoid(_to_numpy(params_dict["opacities"]).reshape(-1))
    lidar = np.asarray(lidar_xyz, dtype=np.float64).reshape(-1, 3)
    scales_m = np.exp(scales_log)
    rotmats = _quats_to_rotmats(quats)
    tree = cKDTree(lidar)
    rng = np.random.default_rng(seed)
    planes = fit_planes_ransac(lidar, max_planes, rng)
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": THRESHOLDS,
        "gaussian_count": int(len(means)),
        "lidar_point_count": int(len(lidar)),
        "ransac_seed": int(seed),
        "scale": _scale_metrics(scales_m, opacity),
        "opacity": _opacity_metrics(opacity),
        "floater": _floater_metrics(means, opacity, scales_m, tree),
        "wall": _wall_metrics(means, rotmats, scales_m, opacity, lidar, planes),
    }


def _format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_report(report: dict) -> str:
    """Render the health report as console tables."""
    lines = []
    lines.append(f"Gaussian Health Report ({report['schema_version']})")
    lines.append(
        f"gaussians={report['gaussian_count']}  lidar_points={report['lidar_point_count']}"
    )
    scale = report["scale"]
    lines.append("")
    lines.append("[scale] max-axis distribution (m)")
    dist = scale["max_axis_m"]
    lines.append(
        "  " + "  ".join(f"{key}={_format_value(dist[key])}" for key in dist)
    )
    lines.append("  giant counts (max axis):")
    for key, value in scale["giant"].items():
        lines.append(
            f"    {key:>8}: count={value['count']:<8} opacity_median={_format_value(value['opacity_median'])}"
        )
    lines.append(
        "  needle counts (max/min axis ratio): "
        + "  ".join(f"{key}={value}" for key, value in scale["needle"].items())
    )
    lines.append(
        f"  degenerate (min axis < {DEGENERATE_MIN_AXIS_M:g} m): "
        f"{scale['degenerate_min_axis_lt_1e4_count']}"
    )
    op = report["opacity"]
    lines.append("")
    lines.append("[opacity]")
    lines.append(
        f"  p5={_format_value(op['p5'])}  p50={_format_value(op['p50'])}  "
        f"p95={_format_value(op['p95'])}  mean={_format_value(op['mean'])}"
    )
    for key in ("dead_lt_0_005", "fog_0_005_to_0_1", "hard_gt_0_95"):
        bucket = op[key]
        lines.append(
            f"  {key:>16}: count={bucket['count']:<8} fraction={bucket['fraction']:.4f}"
        )
    floater = report["floater"]
    lines.append("")
    lines.append(f"[floater] visible (opacity>{FLOATER_MIN_OPACITY}) vs LiDAR NN distance")
    lines.append(f"  visible={floater['visible_count']}")
    dist = floater["distance_m"]
    lines.append(
        "  distance: " + "  ".join(f"{key}={_format_value(dist[key])}" for key in dist)
    )
    for key, value in floater.get("outliers", {}).items():
        lines.append(
            f"  {key:>8}: count={value['count']:<8} "
            f"mean_opacity={_format_value(value['mean_opacity'])} "
            f"mean_max_axis={_format_value(value['mean_max_axis_m'])} m"
        )
    wall = report["wall"]
    lines.append("")
    lines.append(f"[wall] {wall['plane_count']} RANSAC plane(s)")
    for entry in wall["planes"]:
        normal = ", ".join(f"{component:+.3f}" for component in entry["normal"])
        lines.append(
            f"  plane {entry['plane_index']}: n=({normal}) "
            f"lidar_inliers={entry['lidar_inlier_count']} gaussians={entry['gaussian_count']}"
        )
        lines.append(
            f"    center_rms={_format_value(entry['center_normal_rms_m'])} m  "
            f"thickness_p50={_format_value(entry['effective_thickness_m']['p50'])} m  "
            f"thickness_p95={_format_value(entry['effective_thickness_m']['p95'])} m  "
            f"axis_angle_p50={_format_value(entry['shortest_axis_angle_deg_p50'])} deg"
        )
    weighted = wall["weighted_by_lidar_inliers"]
    lines.append(
        "  weighted: "
        + "  ".join(f"{key}={_format_value(value)}" for key, value in weighted.items())
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantify gaussian pathologies of a 3DGS checkpoint against LiDAR."
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="torch checkpoint containing a 'params' dict")
    parser.add_argument("--lidar-ply", required=True, type=Path,
                        help="LiDAR initialization PLY (binary little-endian or ascii)")
    parser.add_argument("--planes", type=int, default=6,
                        help="maximum number of RANSAC planes (default 6)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RANSAC random seed (default 0)")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional path to write the JSON report")
    args = parser.parse_args(argv)

    import torch  # deferred so the numeric API stays importable without torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = checkpoint["params"] if "params" in checkpoint else checkpoint
    lidar_xyz = read_ply_xyz(args.lidar_ply)
    report = compute_health(params, lidar_xyz, max_planes=args.planes, seed=args.seed)
    print(render_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
