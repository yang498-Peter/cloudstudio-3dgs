import json
import os
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


TASK = Path(r"D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827")
LAS_PATH = Path(
    r"G:\S1\USA\2026-02-24_16-21-11snow\process\2026-02-24_16-21-11snow_2"
    r"\2026-02-24_16-21-11snow_colorized.las"
)
OFFSET = np.array([2.767020873831839, -4.222884545523638, 1.4150408912260555])
PAIR = os.environ.get("MIPMAP_OVERLAP_PAIR", "02")
if PAIR == "02":
    TILE_A, TILE_B = "Tile_0", "Tile_2"
    OVERLAP_LO = np.array([0.018277978728065136, -53.256913594479855, -14.62071511425749])
    OVERLAP_HI = np.array([0.25115212003801624, -0.3875184733912409, 22.145557942960977])
elif PAIR == "23":
    TILE_A, TILE_B = "Tile_3", "Tile_2"
    OVERLAP_LO = np.array([0.018277978728065136, -0.5912247361185266, -14.62071511425749])
    OVERLAP_HI = np.array([62.280985021760216, -0.3875184733912409, 22.145557942960977])
else:
    raise ValueError(f"unsupported overlap pair: {PAIR}")


def read_gs(tile: str) -> np.ndarray:
    path = TASK / "result" / "milestones" / "splats" / tile / "gaussian_splat_level_0.pb.bin"
    return np.fromfile(path, dtype="<f4", offset=4).reshape(-1, 14)[:, :3].astype(np.float64)


def in_box(xyz: np.ndarray) -> np.ndarray:
    return np.all((xyz >= OVERLAP_LO) & (xyz <= OVERLAP_HI), axis=1)


def grouped(anchor: np.ndarray, signed: np.ndarray, nearest: np.ndarray, reliable: np.ndarray, max_nn=None):
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
    return np.array([left[int(a)] + right[int(a)] for a in common], dtype=np.float64)


def metrics(joined: np.ndarray):
    a = joined[:, 0]
    b = joined[:, 3]
    if len(joined) < 2:
        return {"parents": int(len(joined))}
    rng = np.random.default_rng(20260827)
    shuffled = b[rng.permutation(len(b))]
    diff = np.abs(a - b) * 1000.0
    return {
        "parents": int(len(joined)),
        "tile_a_positive_pct": float(np.mean(a > 0) * 100),
        "tile_b_positive_pct": float(np.mean(b > 0) * 100),
        "same_side_pct": float(np.mean(a * b > 0) * 100),
        "same_side_shuffled_pct": float(np.mean(a * shuffled > 0) * 100),
        "signed_spearman": float(spearmanr(a, b).statistic),
        "abs_spearman": float(spearmanr(np.abs(a), np.abs(b)).statistic),
        "abs_diff_mm_p10_p50_p90_p95": np.percentile(diff, [10, 50, 90, 95]).tolist(),
        "tile_a_abs_mm_p50_p90": np.percentile(np.abs(a) * 1000, [50, 90]).tolist(),
        "tile_b_abs_mm_p50_p90": np.percentile(np.abs(b) * 1000, [50, 90]).tolist(),
        "both_gt5mm_same_side_pct": float(np.mean((np.abs(a) > 0.005) & (np.abs(b) > 0.005) & (a * b > 0)) * 100),
        "both_gt20mm_same_side_pct": float(np.mean((np.abs(a) > 0.020) & (np.abs(b) > 0.020) & (a * b > 0)) * 100),
    }


las = laspy.read(LAS_PATH)
las_xyz = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))) - OFFSET
strict_las = int(in_box(las_xyz).sum())
tree = cKDTree(las_xyz)

tile_data = {}
all_anchors = []
for tile in (TILE_A, TILE_B):
    xyz_all = read_gs(tile)
    xyz = xyz_all[in_box(xyz_all)]
    nearest, anchor = tree.query(xyz, k=1, workers=-1)
    tile_data[tile] = {"xyz": xyz, "nearest": nearest, "anchor": anchor.astype(np.int64)}
    all_anchors.append(anchor.astype(np.int64))

unique_anchor = np.unique(np.concatenate(all_anchors))
_, neighbors = tree.query(las_xyz[unique_anchor], k=24, workers=-1)
neighbor_xyz = las_xyz[neighbors]
centered = neighbor_xyz - neighbor_xyz.mean(axis=1, keepdims=True)
cov = np.einsum("nki,nkj->nij", centered, centered) / 24.0
evals, evecs = np.linalg.eigh(cov)
normal = evecs[:, :, 0]
dominant = np.argmax(np.abs(normal), axis=1)
sign = np.sign(normal[np.arange(len(normal)), dominant])
sign[sign == 0] = 1
normal *= sign[:, None]
reliable_anchor = (evals[:, 0] / np.maximum(evals.sum(axis=1), 1e-30) < 0.02) & (
    evals[:, 1] / np.maximum(evals[:, 2], 1e-30) > 0.1
)
lookup = {int(anchor): i for i, anchor in enumerate(unique_anchor)}

for tile in (TILE_A, TILE_B):
    data = tile_data[tile]
    local_index = np.fromiter((lookup[int(a)] for a in data["anchor"]), dtype=np.int64)
    n = normal[local_index]
    data["signed"] = np.einsum("ij,ij->i", data["xyz"] - las_xyz[data["anchor"]], n)
    data["reliable"] = reliable_anchor[local_index]

g0_near = grouped(**{
    "anchor": tile_data[TILE_A]["anchor"],
    "signed": tile_data[TILE_A]["signed"],
    "nearest": tile_data[TILE_A]["nearest"],
    "reliable": tile_data[TILE_A]["reliable"],
}, max_nn=0.020)
g2_near = grouped(**{
    "anchor": tile_data[TILE_B]["anchor"],
    "signed": tile_data[TILE_B]["signed"],
    "nearest": tile_data[TILE_B]["nearest"],
    "reliable": tile_data[TILE_B]["reliable"],
}, max_nn=0.020)
near_join = join_groups(g0_near, g2_near)

g0_all = grouped(**{
    "anchor": tile_data[TILE_A]["anchor"],
    "signed": tile_data[TILE_A]["signed"],
    "nearest": tile_data[TILE_A]["nearest"],
    "reliable": tile_data[TILE_A]["reliable"],
})
g2_all = grouped(**{
    "anchor": tile_data[TILE_B]["anchor"],
    "signed": tile_data[TILE_B]["signed"],
    "nearest": tile_data[TILE_B]["nearest"],
    "reliable": tile_data[TILE_B]["reliable"],
})
all_join = join_groups(g0_all, g2_all)

large = {}
for threshold_mm in (20, 50, 100):
    subset = all_join[np.abs(all_join[:, 3]) >= threshold_mm / 1000.0]
    item = metrics(subset)
    if len(subset):
        item.update(
            {
                "tile_a_same_threshold_pct": float(
                    np.mean(np.abs(subset[:, 0]) >= threshold_mm / 1000.0) * 100
                ),
                "tile_a_nn_mm_p50": float(np.median(subset[:, 1]) * 1000),
                "tile_b_nn_mm_p50": float(np.median(subset[:, 4]) * 1000),
            }
        )
    large[str(threshold_mm)] = item

large_bins = {}
for label, low_mm, high_mm in (("20-50", 20, 50), ("50-100", 50, 100), (">=100", 100, None)):
    tile2_abs = np.abs(all_join[:, 3])
    mask = tile2_abs >= low_mm / 1000.0
    if high_mm is not None:
        mask &= tile2_abs < high_mm / 1000.0
    large_bins[label] = metrics(all_join[mask])

output = {
    "pair": [TILE_A, TILE_B],
    "overlap_width_xyz_m": (OVERLAP_HI - OVERLAP_LO).tolist(),
    "strict_overlap_las": strict_las,
    "tile_a_overlap_gs": int(len(tile_data[TILE_A]["xyz"])),
    "tile_b_overlap_gs": int(len(tile_data[TILE_B]["xyz"])),
    "tile_a_reliable_pct": float(np.mean(tile_data[TILE_A]["reliable"]) * 100),
    "tile_b_reliable_pct": float(np.mean(tile_data[TILE_B]["reliable"]) * 100),
    "near_surface_common_parent": metrics(near_join),
    "all_reliable_common_parents": int(len(all_join)),
    "tile2_large_signed_displacement_common_parent": large,
    "tile2_large_signed_displacement_bins": large_bins,
}
print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
