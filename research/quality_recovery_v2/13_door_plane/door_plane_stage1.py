"""Stage 1 of the door-plane consistency check (research/quality_recovery_v2/13_door_plane).

Fits the wall plane that carries the indoor door of ROI ``indoor_door_leaf_Tile_1`` from the
Tile_1 (+ Tile_2 halo) initialization clouds and writes in-plane occupancy maps so the door
opening can be located by eye and by column profile. CPU only, read-only on the datasets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
TILE_ROOT = Path(r"C:\Peter\3dgs-runs\house0305_sop\tile_inputs_v9")
PLY_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])


def read_ply(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line or line.strip() == b"end_header":
                break
        return np.fromfile(f, dtype=PLY_DT)


def load_union() -> tuple[np.ndarray, np.ndarray, dict]:
    parts = []
    info = {}
    for tile in ("Tile_1", "Tile_2"):
        a = read_ply(TILE_ROOT / tile / "initialization_full_lidar.ply")
        info[tile] = int(a.size)
        parts.append(a)
    a = np.concatenate(parts)
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    rgb = np.stack([a["r"], a["g"], a["b"]], 1)
    # halo overlap between the tiles duplicates points: dedupe on exact float32 coordinates
    key = np.ascontiguousarray(np.stack([a["x"], a["y"], a["z"]], 1)).view(np.dtype((np.void, 12))).ravel()
    _, first = np.unique(key, return_index=True)
    first.sort()
    info["union_raw"] = int(xyz.shape[0])
    info["union_dedup"] = int(first.size)
    return xyz[first], rgb[first], info


def fit_plane_lsq(p: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    c = p.mean(0)
    u, s, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2]
    d = (p - c) @ n
    return n, float(-(c @ n)), d  # plane: n.x + d0 = 0 ; signed distance = n.x + d0


def ransac_plane(p: np.ndarray, thresh: float, iters: int, rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray]:
    best = None
    best_n = -1
    for _ in range(iters):
        idx = rng.choice(p.shape[0], 3, replace=False)
        q = p[idx]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d0 = -(q[0] @ n)
        inl = np.abs(p @ n + d0) < thresh
        k = int(inl.sum())
        if k > best_n:
            best_n = k
            best = inl
    n, d0, _ = fit_plane_lsq(p[best])
    inl = np.abs(p @ n + d0) < thresh
    n, d0, _ = fit_plane_lsq(p[inl])
    return n, d0, inl


def main() -> None:
    rng = np.random.default_rng(42)
    xyz, rgb, info = load_union()
    print("clouds:", info)

    # generous neighbourhood of the ROI (x 6.0-7.6, y -3.0..-1.6, z 1.6-3.2)
    box = (xyz[:, 0] > 3.5) & (xyz[:, 0] < 11.0) & (xyz[:, 1] > -6.0) & (xyz[:, 1] < 1.0) & (xyz[:, 2] > -1.0) & (xyz[:, 2] < 4.5)
    P = xyz[box]
    C = rgb[box]
    print("neighbourhood points", P.shape[0])

    # coarse wall candidate: the dense line seen at y ~ -2.5 for x 5.2..9.6 at door height
    cand = (P[:, 0] > 5.0) & (P[:, 0] < 9.8) & (P[:, 1] > -2.9) & (P[:, 1] < -2.1) & (P[:, 2] > 0.0) & (P[:, 2] < 3.4)
    n, d0, inl = ransac_plane(P[cand], thresh=0.03, iters=3000, rng=rng)
    # orient the normal toward the carport / photo side (+y, where the ROI compare cameras sit)
    if n[1] < 0:
        n, d0 = -n, -d0
    dist = P @ n + d0  # >0 = in front (carport side), <0 = behind (interior)
    print("RANSAC wall plane n=%s d0=%.4f inliers=%d/%d (|d|<3cm)" % (np.round(n, 5), d0, inl.sum(), cand.sum()))
    print("  tilt from vertical: %.2f deg" % np.degrees(np.arccos(abs(n[2]))))
    resid = dist[cand][inl]
    print("  inlier residual rms %.4f m, p50 |d| %.4f" % (np.sqrt(np.mean(resid**2)), np.median(np.abs(resid))))

    # in-plane frame: u along the wall (horizontal), z world up
    up = np.array([0.0, 0.0, 1.0])
    e_u = np.cross(up, n)
    e_u /= np.linalg.norm(e_u)
    # make +u run with +x so the maps read left-to-right like the world x axis
    if e_u[0] < 0:
        e_u = -e_u
    origin = np.array([7.0, 0.0, 0.0])
    origin = origin - (origin @ n + d0) * n  # a point on the plane near the door
    u = (P - origin) @ e_u
    z = P[:, 2]
    print("in-plane axis e_u=%s origin=%s" % (np.round(e_u, 4), np.round(origin, 3)))

    # occupancy maps: on-plane (|d|<5cm), behind (0.3<-d<3.0), front (0.3<d<3.0)
    cell = 0.025
    u_lo, u_hi = -3.0, 3.0
    z_lo, z_hi = -0.8, 4.0
    nu = int(round((u_hi - u_lo) / cell))
    nz = int(round((z_hi - z_lo) / cell))

    def occ(mask: np.ndarray) -> np.ndarray:
        H, _, _ = np.histogram2d(z[mask], u[mask], bins=[nz, nu], range=[[z_lo, z_hi], [u_lo, u_hi]])
        return H

    layers = {
        "on_plane_5cm": occ(np.abs(dist) < 0.05),
        "behind_0p3_3m": occ((dist < -0.3) & (dist > -3.0)),
        "front_0p3_3m": occ((dist > 0.3) & (dist < 3.0)),
        "behind_0p1_0p3": occ((dist < -0.1) & (dist > -0.3)),
    }
    np.savez_compressed(OUT / "stage1_occupancy.npz", u_lo=u_lo, u_hi=u_hi, z_lo=z_lo, z_hi=z_hi, cell=cell, **layers)

    def to_img(H: np.ndarray, gain: float) -> np.ndarray:
        img = np.clip(np.log1p(H) * gain, 0, 255).astype(np.uint8)
        return np.flipud(img)  # z up on screen

    # composite: R = behind, G = on plane, B = front
    comp = np.stack([to_img(layers["behind_0p3_3m"], 40), to_img(layers["on_plane_5cm"], 40), to_img(layers["front_0p3_3m"], 40)], -1)
    # draw a 0.5 m grid so pixel -> (u,z) can be read off
    for k in range(int((u_hi - u_lo) / 0.5) + 1):
        col = int(round(k * 0.5 / cell))
        if col < nu:
            comp[:, col, :] = np.maximum(comp[:, col, :], 70)
    for k in range(int((z_hi - z_lo) / 0.5) + 1):
        row = nz - 1 - int(round(k * 0.5 / cell))
        if 0 <= row < nz:
            comp[row, :, :] = np.maximum(comp[row, :, :], 70)
    Image.fromarray(comp).save(OUT / "stage1_occupancy_rgb.png")
    Image.fromarray(to_img(layers["on_plane_5cm"], 40)).save(OUT / "stage1_on_plane.png")
    Image.fromarray(to_img(layers["behind_0p3_3m"], 40)).save(OUT / "stage1_behind.png")

    # column profile at a few z bands to locate the hole
    prof = {}
    for zb in [(0.2, 0.8), (0.8, 1.4), (1.4, 2.0), (2.0, 2.6), (2.6, 3.2)]:
        m = (np.abs(dist) < 0.05) & (z > zb[0]) & (z < zb[1])
        h, edges = np.histogram(u[m], bins=int((u_hi - u_lo) / 0.1), range=(u_lo, u_hi))
        prof["z%.1f-%.1f" % zb] = h.tolist()
        print("z %.1f-%.1f on-plane counts per 0.1 m u-bin (u from %.1f):" % (zb[0], zb[1], u_lo))
        print("   " + " ".join("%4d" % c for c in h))
    json.dump(
        {
            "plane_normal_toward_carport": n.tolist(),
            "plane_d0": d0,
            "e_u": e_u.tolist(),
            "origin_on_plane": origin.tolist(),
            "ransac_inliers": int(inl.sum()),
            "ransac_candidates": int(cand.sum()),
            "inlier_rms_m": float(np.sqrt(np.mean(resid**2))),
            "cloud_info": info,
            "column_profiles_0p1m": prof,
            "u_bins_start": u_lo,
        },
        open(OUT / "stage1_plane.json", "w", encoding="utf-8"),
        indent=1,
    )


if __name__ == "__main__":
    main()
