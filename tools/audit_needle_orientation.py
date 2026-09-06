#!/usr/bin/env python3
"""Needles and streaks: long-axis orientation against the LiDAR surface.

A gaussian whose long axis stands along the local surface normal reads as a
spike sticking out of the ground; one lying in the surface with a high axis
ratio reads as a streak. Both are invisible to Laplacian sharpness (spikes
even raise it), so the comparison against the reference is by count, not by
image score. Normals come from a PCA over the tile LiDAR, as in
audit_colocated_morphology.py.

    python tools/audit_needle_orientation.py --checkpoint RUN/checkpoints/latest.pt \
        --reference-ply USAgs.ply --alignment probes/usa_gs_alignment.json \
        --lidar-root RUN_ROOT/tile_inputs_v9 [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LONG_MM = 10.0       # a gaussian shorter than this cannot read as a needle
RATIO_MIN = 4.0      # elongation needed to read as a line rather than a blob
OUT_DEG = 30.0       # long axis within this angle of the normal = sticks out
IN_DEG = 60.0        # long axis beyond this angle from the normal = lies flat


def quat_to_R(q: np.ndarray) -> np.ndarray:  # w x y z
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1
    ).reshape(-1, 3, 3)


def load_checkpoint(path: Path):
    import torch

    p = torch.load(path, map_location="cpu", weights_only=False)["params"]
    return (p["means"].float().numpy(), torch.sigmoid(p["opacities"].float()).numpy().reshape(-1),
            np.exp(p["scales"].float().numpy()), p["quats"].float().numpy())


def load_reference(ply: Path, alignment: Path):
    from inspect_gaussian_ply import _read_ply

    v, _ = _read_ply(ply)
    A = np.asarray(json.loads(alignment.read_text(encoding="utf-8"))["transform"], dtype=np.float64)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1) @ A[:3, :3].T + A[:3, 3]
    opa = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)
    # Alignment rotation is ~identity for house0305 (0.05 deg); orientation
    # statistics ignore it, as the co-located audit does.
    return xyz.astype(np.float32), opa.astype(np.float32), sc.astype(np.float32), q.astype(np.float32)


def lidar_normals(lidar_root: Path, stride: int = 4, sub: int = 20):
    from scipy.spatial import cKDTree
    from inspect_gaussian_ply import _read_ply

    clouds = []
    for t in range(4):
        ply = lidar_root / f"Tile_{t}" / "initialization_full_lidar.ply"
        if ply.exists():
            v, _ = _read_ply(ply)
            clouds.append(np.stack([v["x"], v["y"], v["z"]], 1)[::stride])
    lid = np.concatenate(clouds).astype(np.float32)
    tree = cKDTree(lid)
    centers = lid[::sub]
    _, nb = tree.query(centers, k=16, workers=8)
    P = lid[nb] - centers[:, None]
    C = np.einsum("nki,nkj->nij", P, P)
    _, vecs = np.linalg.eigh(C)
    normals = vecs[:, :, 0]
    # Planarity of the neighbourhood: needles are only meaningful where the
    # LiDAR actually defines a surface (ground, walls), not in foliage.
    w = np.linalg.eigvalsh(C)
    planarity = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
    return cKDTree(centers), normals, planarity


def classify(name, xyz, opa, sc, q, ntree, normals, planarity, live=0.1):
    keep = opa > live
    xyz, opa, sc, q = xyz[keep], opa[keep], sc[keep], q[keep]
    d, j = ntree.query(xyz, workers=8)
    on_surface = (d < 0.15) & (planarity[j] > 0.5)
    R = quat_to_R(q)
    long_idx = np.argmax(sc, 1)
    long_axis = np.take_along_axis(R, long_idx[:, None, None].repeat(3, 1), 2)[:, :, 0]
    cosang = np.abs(np.einsum("ni,ni->n", long_axis, normals[j]))
    ang = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
    long_mm = sc.max(1) * 1000.0
    ratio = sc.max(1) / np.maximum(sc.min(1), 1e-9)
    elongated = (long_mm >= LONG_MM) & (ratio >= RATIO_MIN) & on_surface
    needle_out = elongated & (ang <= OUT_DEG)
    streak_in = elongated & (ang >= IN_DEG)
    n_surface = int(on_surface.sum())
    report = {
        "live_count": int(len(xyz)),
        "on_surface_count": n_surface,
        "elongated_on_surface": int(elongated.sum()),
        "needle_out_count": int(needle_out.sum()),
        "streak_in_count": int(streak_in.sum()),
        "needle_out_per_1k_surface": 1000.0 * float(needle_out.sum()) / max(1, n_surface),
        "streak_in_per_1k_surface": 1000.0 * float(streak_in.sum()) / max(1, n_surface),
        "needle_out_long_mm_p50": float(np.median(long_mm[needle_out])) if needle_out.any() else None,
        "needle_out_opacity_p50": float(np.median(opa[needle_out])) if needle_out.any() else None,
        "streak_in_long_mm_p50": float(np.median(long_mm[streak_in])) if streak_in.any() else None,
        "long_axis_vs_normal_deg_p50_on_surface": float(np.median(ang[on_surface])) if n_surface else None,
    }
    print(f"{name}: live {report['live_count']:,} on-surface {n_surface:,} | needles-out {report['needle_out_count']:,} "
          f"({report['needle_out_per_1k_surface']:.2f}/1k) streaks-in {report['streak_in_count']:,} "
          f"({report['streak_in_per_1k_surface']:.2f}/1k) | long-vs-normal p50 {report['long_axis_vs_normal_deg_p50_on_surface']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-ply", type=Path)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--lidar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ntree, normals, planarity = lidar_normals(args.lidar_root)
    report = {"ours": classify(args.checkpoint.parent.parent.name, *load_checkpoint(args.checkpoint), ntree, normals, planarity)}
    if args.reference_ply is not None:
        if args.alignment is None:
            parser.error("--reference-ply needs --alignment")
        report["reference"] = classify("reference", *load_reference(args.reference_ply, args.alignment), ntree, normals, planarity)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
        os.replace(tmp, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
