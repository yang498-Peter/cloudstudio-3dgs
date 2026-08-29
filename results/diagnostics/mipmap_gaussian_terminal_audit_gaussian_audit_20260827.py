"""Read-only terminal audit for the completed snow MipMap Gaussian task.

This script never writes into the MipMap task or source-data directories.  It
prints one UTF-8 JSON document to stdout so that every reported statistic can
be reproduced from the immutable task artifacts.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


TASK = Path(
    r"D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827"
)
RESULT = TASK / "result"
LAS_PATH = Path(
    r"G:\S1\USA\2026-02-24_16-21-11snow\process\2026-02-24_16-21-11snow_2"
    r"\2026-02-24_16-21-11snow_colorized.las"
)
SAMPLE_COUNT = 100_000
PCA_K = 24
PERCENTILES = [10, 50, 90, 95, 99]


def q(values, percentiles=PERCENTILES):
    values = np.asarray(values)
    if not len(values):
        return [None for _ in percentiles]
    return [float(v) for v in np.percentile(values, percentiles)]


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    out = np.empty_like(value)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def read_tile_metadata():
    with (RESULT / "task" / "tiles.json").open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    offset = np.asarray(payload["tile_system"]["offset"], dtype=np.float64)
    tiles = {}
    for tile in payload["tiles"]:
        boundary = np.asarray(tile["roi"]["boundary"], dtype=np.float64)
        lo = np.array(
            [boundary[:, 0].min(), boundary[:, 1].min(), tile["roi"]["min_z"]],
            dtype=np.float64,
        )
        hi = np.array(
            [boundary[:, 0].max(), boundary[:, 1].max(), tile["roi"]["max_z"]],
            dtype=np.float64,
        )
        tiles[tile["name"]] = {"lo": lo, "hi": hi, "max_memory_gb": tile["max_memory"]}
    return offset, tiles


def read_pnts_count(tile):
    root = RESULT / "3D" / "point-pnts" / tile
    files = sorted(root.rglob("*.pnts"))
    total = 0
    malformed = []
    declared_length_deltas = []
    for path in files:
        with path.open("rb") as stream:
            header = stream.read(28)
            if len(header) != 28:
                malformed.append(str(path))
                continue
            magic, version, byte_length, ft_json_len, _, _, _ = struct.unpack("<4s6I", header)
            if magic != b"pnts" or version != 1:
                malformed.append(str(path))
                continue
            # A subset of MipMap's PNTS files declares byteLength eight bytes
            # larger than the physical file.  Their Feature Table and payload
            # remain readable, so record this producer quirk separately rather
            # than dropping valid POINTS_LENGTH evidence.
            if byte_length != path.stat().st_size:
                declared_length_deltas.append(int(byte_length - path.stat().st_size))
            feature_json = stream.read(ft_json_len).rstrip(b" \t\r\n\x00")
            total += int(json.loads(feature_json.decode("utf-8"))["POINTS_LENGTH"])
    return {
        "file_count": len(files),
        "points_length": total,
        "malformed": malformed,
        "declared_byte_length_mismatch_file_count": len(declared_length_deltas),
        "declared_minus_physical_byte_length_unique": sorted(set(declared_length_deltas)),
    }


def read_pb(tile, level=0):
    path = (
        RESULT
        / "milestones"
        / "splats"
        / tile
        / f"gaussian_splat_level_{level}.pb.bin"
    )
    with path.open("rb") as stream:
        count = struct.unpack("<I", stream.read(4))[0]
    expected = 4 + count * 56
    if path.stat().st_size != expected:
        raise ValueError(f"PB size mismatch: {path}")
    records = np.memmap(path, dtype="<f4", mode="r", offset=4, shape=(count, 14))
    return path, count, records


def rotation_matrix_wxyz(raw_quat):
    quat = np.asarray(raw_quat, dtype=np.float64)
    norms = np.linalg.norm(quat, axis=1)
    quat = quat / np.maximum(norms[:, None], 1e-30)
    w, x, y, z = quat.T
    matrix = np.empty((len(quat), 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix, norms


def rotation_matrix_xyzw(raw_quat):
    raw_quat = np.asarray(raw_quat, dtype=np.float64)
    return rotation_matrix_wxyz(raw_quat[:, [3, 0, 1, 2]])


def pca_normals_for_anchors(las_xyz, las_tree, anchors):
    _, neighbors = las_tree.query(las_xyz[anchors], k=PCA_K, workers=-1)
    neighbor_xyz = las_xyz[neighbors]
    centered = neighbor_xyz - neighbor_xyz.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / float(PCA_K)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, :, 0]
    dominant = np.argmax(np.abs(normal), axis=1)
    signs = np.sign(normal[np.arange(len(normal)), dominant])
    signs[signs == 0] = 1
    normal *= signs[:, None]
    curvature = eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-30)
    planarity_support = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1e-30)
    reliable = (curvature < 0.02) & (planarity_support > 0.1)
    k_radius = np.linalg.norm(neighbor_xyz[:, -1] - las_xyz[anchors], axis=1)
    return normal, reliable, curvature, k_radius


def unique_voxels(xyz, voxel_size):
    keys = np.floor(np.asarray(xyz, dtype=np.float64) / voxel_size).astype(np.int32)
    return np.unique(keys, axis=0, return_counts=True)


def row_set(keys):
    return {tuple(map(int, row)) for row in keys}


def voxel_metrics(las_xyz, gs_xyz, voxel_size):
    las_keys, las_counts = unique_voxels(las_xyz, voxel_size)
    gs_keys, gs_counts = unique_voxels(gs_xyz, voxel_size)
    las_map = {tuple(map(int, key)): int(count) for key, count in zip(las_keys, las_counts)}
    gs_map = {tuple(map(int, key)): int(count) for key, count in zip(gs_keys, gs_counts)}
    shared_keys = sorted(set(las_map).intersection(gs_map))
    las_shared = np.asarray([las_map[key] for key in shared_keys], dtype=np.int64)
    gs_shared = np.asarray([gs_map[key] for key in shared_keys], dtype=np.int64)
    ratios = gs_shared / las_shared
    stable = las_shared >= 100
    gs_points_without_las = sum(count for key, count in gs_map.items() if key not in las_map)
    return {
        "voxel_size_m": voxel_size,
        "las_occupied_voxels": int(len(las_map)),
        "gs_occupied_voxels": int(len(gs_map)),
        "shared_voxels": int(len(shared_keys)),
        "las_voxels_without_gs_pct": float((len(las_map) - len(shared_keys)) / len(las_map) * 100),
        "gs_voxels_without_las_pct": float((len(gs_map) - len(shared_keys)) / len(gs_map) * 100),
        "gs_points_in_voxels_without_las_pct": float(gs_points_without_las / len(gs_xyz) * 100),
        "shared_ratio_p10_p50_p90_p95_p99": q(ratios),
        "shared_ratio_lt_0_5_pct": float(np.mean(ratios < 0.5) * 100),
        "shared_ratio_ge_2_pct": float(np.mean(ratios >= 2.0) * 100),
        "weighted_shared_ratio": float(gs_shared.sum() / las_shared.sum()),
        "stable_nlas_ge100_voxels": int(stable.sum()),
        "stable_ratio_p10_p50_p90_p95_p99": q(ratios[stable]),
    }


def parameter_metrics(records):
    raw_scale = np.asarray(records[:, 3:6], dtype=np.float64)
    scale_mm = np.exp(raw_scale) * 1000.0
    ordered_scale = np.sort(scale_mm, axis=1)
    opacity_logit = np.asarray(records[:, 9], dtype=np.float64)
    opacity = sigmoid(opacity_logit)
    raw_quat = np.asarray(records[:, 10:14], dtype=np.float64)
    quat_norm = np.linalg.norm(raw_quat, axis=1)
    return {
        "layout": "xyz, log_scale_0..2, f_dc_0..2, opacity_logit, rotation_0..3",
        "bytes_per_gaussian": 56,
        "scale_interpretation": "exp(log_scale), reported in millimetres",
        "short_axis_mm_p10_p50_p90_p95_p99": q(ordered_scale[:, 0]),
        "middle_axis_mm_p10_p50_p90_p95_p99": q(ordered_scale[:, 1]),
        "long_axis_mm_p10_p50_p90_p95_p99": q(ordered_scale[:, 2]),
        "aspect_long_over_short_p10_p50_p90_p95_p99": q(
            ordered_scale[:, 2] / np.maximum(ordered_scale[:, 0], 1e-30)
        ),
        "opacity_interpretation": "sigmoid(opacity_logit)",
        "opacity_probability_p10_p50_p90_p95_p99": q(opacity),
        "opacity_probability_min_max": [float(opacity.min()), float(opacity.max())],
        "opacity_lt_0_05_pct": float(np.mean(opacity < 0.05) * 100),
        "opacity_lt_0_10_pct": float(np.mean(opacity < 0.10) * 100),
        "rotation_norm_p10_p50_p90_p95_p99": q(quat_norm),
        "rotation_nonfinite_count": int((~np.isfinite(raw_quat)).any(axis=1).sum()),
        "sh_order": 0,
        "sh_evidence": "PB contains only f_dc_0..2 and no f_rest fields",
    }


def in_box(xyz, lo, hi):
    return np.all((xyz >= lo) & (xyz <= hi), axis=1)


def p2p_metrics(tile, records, las_xyz, las_tree, roi, las_voxel_sets):
    count = len(records)
    sample_indices = np.linspace(0, count - 1, min(SAMPLE_COUNT, count), dtype=np.int64)
    sample = np.asarray(records[sample_indices], dtype=np.float64)
    xyz = sample[:, :3]
    nearest, anchors = las_tree.query(xyz, k=1, workers=-1)
    anchors = anchors.astype(np.int64)
    normal, reliable, curvature, k_radius = pca_normals_for_anchors(
        las_xyz, las_tree, anchors
    )
    displacement = xyz - las_xyz[anchors]
    signed_normal = np.einsum("ij,ij->i", displacement, normal)
    normal_abs = np.abs(signed_normal)
    tangent = np.sqrt(np.maximum(nearest * nearest - normal_abs * normal_abs, 0.0))

    matrices_wxyz, quat_norm = rotation_matrix_wxyz(sample[:, 10:14])
    matrices_xyzw, _ = rotation_matrix_xyzw(sample[:, 10:14])
    shortest_index = np.argmin(sample[:, 3:6], axis=1)
    rows = np.arange(len(sample))
    shortest_axis_wxyz = matrices_wxyz[rows, :, shortest_index]
    shortest_axis_xyzw = matrices_xyzw[rows, :, shortest_index]
    angle_wxyz = np.degrees(
        np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", shortest_axis_wxyz, normal)), 0, 1))
    )
    angle_xyzw = np.degrees(
        np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", shortest_axis_xyzw, normal)), 0, 1))
    )

    lo = roi["lo"]
    hi = roi["hi"]
    boundary_xy = np.minimum.reduce(
        [xyz[:, 0] - lo[0], hi[0] - xyz[:, 0], xyz[:, 1] - lo[1], hi[1] - xyz[:, 1]]
    )
    same_las_voxel = {}
    for size, keys in las_voxel_sets.items():
        sample_keys = np.floor(xyz / size).astype(np.int32)
        same_las_voxel[size] = np.fromiter(
            (tuple(map(int, key)) in keys for key in sample_keys), dtype=bool, count=len(sample_keys)
        )

    bins = {}
    distance_mm = nearest * 1000.0
    bin_specs = [
        ("lt20mm", 0.0, 20.0),
        ("20to50mm", 20.0, 50.0),
        ("50to100mm", 50.0, 100.0),
        ("100to500mm", 100.0, 500.0),
        ("ge500mm", 500.0, None),
    ]
    for label, lower, upper in bin_specs:
        keep = distance_mm >= lower
        if upper is not None:
            keep &= distance_mm < upper
        reliable_keep = keep & reliable
        bins[label] = {
            "sample_count": int(keep.sum()),
            "sample_pct": float(keep.mean() * 100),
            "reliable_pct_within_bin": float(reliable[keep].mean() * 100) if keep.any() else None,
            "nearest_mm_p50": float(np.median(distance_mm[keep])) if keep.any() else None,
            "normal_abs_mm_p50_reliable": (
                float(np.median(normal_abs[reliable_keep]) * 1000) if reliable_keep.any() else None
            ),
            "tangent_mm_p50_reliable": (
                float(np.median(tangent[reliable_keep]) * 1000) if reliable_keep.any() else None
            ),
            "las_k24_radius_mm_p50": float(np.median(k_radius[keep]) * 1000) if keep.any() else None,
            "tile_xy_boundary_distance_m_p50": (
                float(np.median(boundary_xy[keep])) if keep.any() else None
            ),
            "same_0_5m_las_voxel_pct": (
                float(same_las_voxel[0.5][keep].mean() * 100) if keep.any() else None
            ),
            "same_1m_las_voxel_pct": (
                float(same_las_voxel[1.0][keep].mean() * 100) if keep.any() else None
            ),
        }

    far = nearest >= 0.020
    reliable_near = reliable
    return {
        "sampling": {
            "method": "deterministic evenly spaced indices over level-0 PB order",
            "sample_count": int(len(sample)),
            "pca_k": PCA_K,
            "reliable_rule": "lambda0/sum(lambda)<0.02 and lambda1/lambda2>0.1",
        },
        "sample_position_outside_tile_roi_count": int((~in_box(xyz, lo, hi)).sum()),
        "nearest_las_mm_p10_p50_p90_p95_p99": q(nearest * 1000.0),
        "nearest_las_ge20mm_pct": float(far.mean() * 100),
        "nearest_las_ge100mm_pct": float((nearest >= 0.100).mean() * 100),
        "reliable_pca_pct": float(reliable.mean() * 100),
        "normal_abs_mm_all_p10_p50_p90_p95_p99": q(normal_abs * 1000.0),
        "normal_abs_mm_reliable_p10_p50_p90_p95_p99": q(
            normal_abs[reliable_near] * 1000.0
        ),
        "normal_abs_lt5mm_reliable_pct": float(
            np.mean(normal_abs[reliable_near] < 0.005) * 100
        ),
        "normal_abs_ge20mm_reliable_pct": float(
            np.mean(normal_abs[reliable_near] >= 0.020) * 100
        ),
        "tangent_mm_all_p10_p50_p90_p95_p99": q(tangent * 1000.0),
        "tangent_mm_reliable_p10_p50_p90_p95_p99": q(tangent[reliable] * 1000.0),
        "far_ge20mm_with_normal_lt5mm_pct_conditional": (
            float(np.mean(normal_abs[far] < 0.005) * 100) if far.any() else None
        ),
        "shortest_axis_vs_normal_wxyz_deg_p10_p50_p90_p95_p99_reliable": q(
            angle_wxyz[reliable]
        ),
        "shortest_axis_vs_normal_wxyz_within15deg_pct_reliable": float(
            np.mean(angle_wxyz[reliable] <= 15.0) * 100
        ),
        "shortest_axis_vs_normal_wxyz_within30deg_pct_reliable": float(
            np.mean(angle_wxyz[reliable] <= 30.0) * 100
        ),
        "shortest_axis_vs_normal_xyzw_deg_p50_reliable": float(
            np.median(angle_xyzw[reliable])
        ),
        "sample_rotation_norm_p50": float(np.median(quat_norm)),
        "curvature_p10_p50_p90_p95_p99": q(curvature),
        "long_tail_distance_bins": bins,
    }


def grouped_by_anchor(anchor, signed, nearest, reliable, max_nn=None):
    keep = reliable.copy()
    if max_nn is not None:
        keep &= nearest < max_nn
    anchor = anchor[keep]
    signed = signed[keep]
    nearest = nearest[keep]
    order = np.argsort(anchor, kind="stable")
    anchor = anchor[order]
    signed = signed[order]
    nearest = nearest[order]
    starts = np.flatnonzero(np.r_[True, anchor[1:] != anchor[:-1]])
    ends = np.r_[starts[1:], len(anchor)]
    return {
        int(anchor[start]): (
            float(np.median(signed[start:end])),
            float(np.median(nearest[start:end])),
            int(end - start),
        )
        for start, end in zip(starts, ends)
    }


def join_groups(left, right):
    common = np.array(sorted(set(left).intersection(right)), dtype=np.int64)
    if not len(common):
        return np.empty((0, 6), dtype=np.float64)
    return np.asarray([left[int(item)] + right[int(item)] for item in common], dtype=np.float64)


def paired_metrics(joined):
    if not len(joined):
        return {"common_parents": 0}
    left = joined[:, 0]
    right = joined[:, 3]
    rng = np.random.default_rng(20260827)
    shuffled = right[rng.permutation(len(right))]
    diff_mm = np.abs(left - right) * 1000.0
    return {
        "common_parents": int(len(joined)),
        "same_side_pct": float(np.mean(left * right > 0) * 100),
        "same_side_shuffled_pct": float(np.mean(left * shuffled > 0) * 100),
        "signed_spearman": finite_float(spearmanr(left, right).statistic) if len(joined) > 1 else None,
        "absolute_spearman": (
            finite_float(spearmanr(np.abs(left), np.abs(right)).statistic)
            if len(joined) > 1
            else None
        ),
        "normal_abs_diff_mm_p10_p50_p90_p95_p99": q(diff_mm),
        "left_normal_abs_mm_p50_p90": q(np.abs(left) * 1000.0, [50, 90]),
        "right_normal_abs_mm_p50_p90": q(np.abs(right) * 1000.0, [50, 90]),
        "both_gt20mm_and_same_side_pct": float(
            np.mean((np.abs(left) > 0.020) & (np.abs(right) > 0.020) & (left * right > 0))
            * 100
        ),
    }


def overlap_metrics(tile_left, tile_right, focus_tile, gs_positions, tiles, las_xyz, las_tree):
    lo = np.maximum(tiles[tile_left]["lo"], tiles[tile_right]["lo"])
    hi = np.minimum(tiles[tile_left]["hi"], tiles[tile_right]["hi"])
    if np.any(hi < lo):
        raise ValueError(f"tiles do not overlap: {tile_left}, {tile_right}")
    strict_las = int(in_box(las_xyz, lo, hi).sum())
    data = {}
    all_anchor = []
    for tile in (tile_left, tile_right):
        xyz = gs_positions[tile][in_box(gs_positions[tile], lo, hi)]
        nearest, anchor = las_tree.query(xyz, k=1, workers=-1)
        anchor = anchor.astype(np.int64)
        data[tile] = {"xyz": xyz, "nearest": nearest, "anchor": anchor}
        all_anchor.append(anchor)

    unique_anchor = np.unique(np.concatenate(all_anchor))
    normals, reliable_anchor, _, _ = pca_normals_for_anchors(las_xyz, las_tree, unique_anchor)
    lookup = {int(anchor): index for index, anchor in enumerate(unique_anchor)}
    for tile in (tile_left, tile_right):
        item = data[tile]
        local = np.fromiter(
            (lookup[int(anchor)] for anchor in item["anchor"]),
            dtype=np.int64,
            count=len(item["anchor"]),
        )
        item["normal"] = normals[local]
        item["reliable"] = reliable_anchor[local]
        item["signed"] = np.einsum(
            "ij,ij->i", item["xyz"] - las_xyz[item["anchor"]], item["normal"]
        )

    groups_near = {
        tile: grouped_by_anchor(
            data[tile]["anchor"],
            data[tile]["signed"],
            data[tile]["nearest"],
            data[tile]["reliable"],
            max_nn=0.020,
        )
        for tile in (tile_left, tile_right)
    }
    joined_near = join_groups(groups_near[tile_left], groups_near[tile_right])
    groups_all = {
        tile: grouped_by_anchor(
            data[tile]["anchor"],
            data[tile]["signed"],
            data[tile]["nearest"],
            data[tile]["reliable"],
        )
        for tile in (tile_left, tile_right)
    }
    joined_all = join_groups(groups_all[tile_left], groups_all[tile_right])

    focus_column = 0 if focus_tile == tile_left else 3
    focus_abs = np.abs(joined_all[:, focus_column]) * 1000.0
    bins = {}
    for label, lower, upper in (
        ("20to50mm", 20, 50),
        ("50to100mm", 50, 100),
        ("ge100mm", 100, None),
    ):
        keep = focus_abs >= lower
        if upper is not None:
            keep &= focus_abs < upper
        bins[label] = paired_metrics(joined_all[keep])

    left_xyz = data[tile_left]["xyz"]
    right_xyz = data[tile_right]["xyz"]
    left_tree = cKDTree(left_xyz)
    right_tree = cKDTree(right_xyz)
    right_to_left_dist, right_to_left = left_tree.query(right_xyz, k=1, workers=-1)
    _, left_to_right = right_tree.query(left_xyz, k=1, workers=-1)
    right_indices = np.arange(len(right_xyz), dtype=np.int64)
    mutual = left_to_right[right_to_left] == right_indices

    return {
        "pair": [tile_left, tile_right],
        "strict_overlap_lo": lo.tolist(),
        "strict_overlap_hi": hi.tolist(),
        "overlap_width_xyz_m": (hi - lo).tolist(),
        "strict_overlap_las_count": strict_las,
        "left_overlap_gs_count": int(len(left_xyz)),
        "right_overlap_gs_count": int(len(right_xyz)),
        "left_reliable_pct": float(np.mean(data[tile_left]["reliable"]) * 100),
        "right_reliable_pct": float(np.mean(data[tile_right]["reliable"]) * 100),
        "near_surface_common_parent_lt20mm": paired_metrics(joined_near),
        "all_reliable_common_parent_count": int(len(joined_all)),
        "focus_tile_for_large_displacement_bins": focus_tile,
        "large_displacement_common_parent_bins": bins,
        "right_to_left_nearest_mm_p10_p50_p90_p95_p99": q(right_to_left_dist * 1000.0),
        "right_to_left_nearest_lt5mm_pct": float(np.mean(right_to_left_dist < 0.005) * 100),
        "right_to_left_nearest_lt20mm_pct": float(np.mean(right_to_left_dist < 0.020) * 100),
        "geometric_mutual_nearest_pairs_lt5mm_without_pca_filter": int(
            np.sum(mutual & (right_to_left_dist < 0.005))
        ),
        "geometric_mutual_nearest_pairs_lt20mm_without_pca_filter": int(
            np.sum(mutual & (right_to_left_dist < 0.020))
        ),
        "selection_warning": (
            "common-parent requires both final sets to hit the same nearest LAS anchor; "
            "mutual-nearest preferentially selects already-agreeing GS and is corroborative only"
        ),
    }


def main():
    offset, tiles = read_tile_metadata()
    las = laspy.read(LAS_PATH)
    las_xyz_world = np.column_stack(
        (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))
    )
    las_xyz = las_xyz_world - offset
    las_tree = cKDTree(las_xyz)

    output = {
        "audit_contract": {
            "task": str(TASK),
            "las": str(LAS_PATH),
            "coordinate_relation": "tile_local_xyz = las_world_xyz - tile_system.offset",
            "tile_system_offset": offset.tolist(),
            "read_only": True,
            "common_p2p_sample_count": SAMPLE_COUNT,
            "pca_k": PCA_K,
        },
        "tiles": {},
        "overlaps": {},
    }
    gs_positions = {}

    for tile, roi in tiles.items():
        las_mask = in_box(las_xyz, roi["lo"], roi["hi"])
        tile_las = las_xyz[las_mask]
        pnts = read_pnts_count(tile)
        levels = []
        records = None
        for level in range(6):
            path, count, current = read_pb(tile, level)
            levels.append({"level": level, "count": count, "bytes": path.stat().st_size})
            if level == 0:
                records = current
        assert records is not None
        gs_xyz = np.asarray(records[:, :3], dtype=np.float64)
        gs_positions[tile] = gs_xyz
        voxel_sets = {
            size: row_set(unique_voxels(tile_las, size)[0]) for size in (0.5, 1.0)
        }
        output["tiles"][tile] = {
            "roi_local_lo": roi["lo"].tolist(),
            "roi_local_hi": roi["hi"].tolist(),
            "estimated_max_memory_gb": float(roi["max_memory_gb"]),
            "pnts": pnts,
            "las_roi_count": int(len(tile_las)),
            "pnts_minus_las": int(pnts["points_length"] - len(tile_las)),
            "levels": levels,
            "level0_over_pnts": float(levels[0]["count"] / pnts["points_length"]),
            "level0_over_las": float(levels[0]["count"] / len(tile_las)),
            "parameter_metrics": parameter_metrics(records),
            "point_to_plane": p2p_metrics(
                tile, records, las_xyz, las_tree, roi, voxel_sets
            ),
            "voxel_0_5m": voxel_metrics(tile_las, gs_xyz, 0.5),
            "voxel_1m": voxel_metrics(tile_las, gs_xyz, 1.0),
        }

    output["overlaps"]["Tile_0__Tile_1"] = overlap_metrics(
        "Tile_0", "Tile_1", "Tile_1", gs_positions, tiles, las_xyz, las_tree
    )
    output["overlaps"]["Tile_0__Tile_2"] = overlap_metrics(
        "Tile_0", "Tile_2", "Tile_2", gs_positions, tiles, las_xyz, las_tree
    )
    output["overlaps"]["Tile_3__Tile_2"] = overlap_metrics(
        "Tile_3", "Tile_2", "Tile_2", gs_positions, tiles, las_xyz, las_tree
    )

    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
