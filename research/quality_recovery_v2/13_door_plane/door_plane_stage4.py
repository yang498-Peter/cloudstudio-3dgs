"""Stage 4: fit the LiDAR door leaf (the thin line running from the left jamb into the
interior in the top-down maps), build its world rectangle, and map the in-leaf point
occupancy so the glass-pane region can be identified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from door_plane_stage1 import read_ply

OUT = Path(__file__).resolve().parent
TILE1_PLY = Path(r"C:\Peter\3dgs-runs\house0305_sop\tile_inputs_v9\Tile_1\initialization_full_lidar.ply")


def main() -> None:
    s2 = json.load(open(OUT / "stage2_door_opening.json", encoding="utf-8"))
    pl = s2["plane_strip_refit"]
    n = np.asarray(pl["normal"])
    d0 = float(pl["d0"])
    e_u, e_w, origin = (np.asarray(pl[k]) for k in ("e_u", "e_w", "origin"))
    op = s2["opening_in_plane"]
    a = read_ply(TILE1_PLY)
    T = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    rel = T - origin
    u, w, d = rel @ e_u, rel @ e_w, T @ n + d0

    # candidate leaf points: the thin line seen in the top-down map, away from the jamb and the wall
    cand = (u > 0.35) & (u < 0.75) & (d < -0.08) & (d > -1.10) & (w > op["w0"] + 0.1) & (w < op["w1"] - 0.1)
    uv = np.column_stack([u[cand], d[cand]])
    # robust line fit in the (u,d) top-down plane: RANSAC on 2D line
    rng = np.random.default_rng(7)
    best, best_k = None, -1
    for _ in range(4000):
        i, j = rng.choice(uv.shape[0], 2, replace=False)
        p, q = uv[i], uv[j]
        t = q - p
        if np.linalg.norm(t) < 0.3:
            continue
        t /= np.linalg.norm(t)
        nn = np.array([-t[1], t[0]])
        dist = (uv - p) @ nn
        k = int((np.abs(dist) < 0.02).sum())
        if k > best_k:
            best_k, best = k, (p, t, nn)
    p, t, nn = best
    inl = np.abs((uv - p) @ nn) < 0.02
    c = uv[inl].mean(0)
    _, _, vt = np.linalg.svd(uv[inl] - c, full_matrices=False)
    t = vt[0]
    if t[1] > 0:
        t = -t  # point into the interior (-d)
    nn = np.array([-t[1], t[0]])
    res = (uv[inl] - c) @ nn
    # hinge = intersection with the wall plane d=0; free edge = extreme along t
    s = (uv[inl] - c) @ t
    hinge = c + t * ((0.0 - c[1]) / t[1])
    free = c + t * s.max()
    length = float(np.linalg.norm(free - hinge))
    angle_deg = float(np.degrees(np.arctan2(-t[1], t[0])))  # angle of the leaf from the +u wall direction, into the interior
    open_angle_from_wall = float(np.degrees(np.arccos(np.clip(t @ np.array([1.0, 0.0]), -1, 1))))
    print("leaf line: hinge (u,d)=%s free (u,d)=%s length %.3f m  open angle from wall %.1f deg  inliers %d/%d  rms %.4f m" % (
        np.round(hinge, 3), np.round(free, 3), length, open_angle_from_wall, inl.sum(), uv.shape[0], np.sqrt(np.mean(res**2))))

    # world rectangle of the leaf: hinge line x w-range of the opening
    def world(uu: float, dd: float, ww: float) -> np.ndarray:
        return origin + uu * e_u + ww * e_w + dd * n

    corners = np.asarray([world(hinge[0], hinge[1], op["w0"]), world(free[0], free[1], op["w0"]), world(free[0], free[1], op["w1"]), world(hinge[0], hinge[1], op["w1"])])
    leaf_n = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    leaf_n /= np.linalg.norm(leaf_n)
    if leaf_n @ n < 0:  # orient roughly like the wall normal (toward the carport side)
        leaf_n = -leaf_n
    leaf_d0 = float(-(corners[0] @ leaf_n))

    # in-leaf occupancy: along-leaf coordinate s (0 at hinge) vs w, for points within 4 cm of the leaf plane
    dl = T @ leaf_n + leaf_d0
    near = np.abs(dl) < 0.04
    sl = ((T - corners[0]) @ (corners[1] - corners[0])) / length
    cell = 0.02
    m = near & (sl > -0.1) & (sl < length + 0.1) & (w > op["w0"] - 0.1) & (w < op["w1"] + 0.1)
    H, se, we = np.histogram2d(w[m], sl[m], bins=[int((op["w1"] - op["w0"] + 0.2) / cell), int((length + 0.2) / cell)], range=[[op["w0"] - 0.1, op["w1"] + 0.1], [-0.1, length + 0.1]])
    img = np.flipud(np.clip(np.log1p(H) * 40, 0, 255).astype(np.uint8))
    S = 4
    im = Image.fromarray(img).resize((img.shape[1] * S, img.shape[0] * S), Image.NEAREST).convert("RGB")
    dr = ImageDraw.Draw(im)
    for k in np.arange(0.0, length + 0.05, 0.1):
        x = int((k + 0.1) / cell * S)
        dr.line([(x, 0), (x, im.size[1])], fill=(255, 60, 60) if abs(k * 2 - round(k * 2)) < 1e-6 else (90, 0, 0))
    for k in np.arange(op["w0"], op["w1"] + 0.05, 0.1):
        y = im.size[1] - 1 - int((k - op["w0"] + 0.1) / cell * S)
        dr.line([(0, y), (im.size[0], y)], fill=(255, 60, 60) if abs(k * 2 - round(k * 2)) < 1e-6 else (90, 0, 0))
    dr.text((4, 4), "LiDAR leaf occupancy (|dist to leaf plane|<4cm): s along leaf from hinge (left) 0..%.2f m, w %.2f..%.2f (bottom..top); grid 0.1 m" % (length, op["w0"], op["w1"]), fill=(255, 255, 0))
    im.save(OUT / "stage4_leaf_occupancy.png")

    # row/column profiles on the leaf to find the glass panes (holes) — report per 10 cm bands
    prof = {}
    for wb in np.arange(op["w0"], op["w1"], 0.1):
        mm = m & (w >= wb) & (w < wb + 0.1) & (sl > 0.05) & (sl < length - 0.05)
        prof["w%.2f" % wb] = int(mm.sum())
    print("leaf points per 10 cm height band:", prof)

    json.dump(
        {
            "leaf_line_topdown": {"hinge_u_d": hinge.tolist(), "free_edge_u_d": free.tolist(), "length_m": length,
                                  "open_angle_from_wall_deg": open_angle_from_wall, "inliers": int(inl.sum()), "candidates": int(uv.shape[0]),
                                  "inlier_rms_m": float(np.sqrt(np.mean(res**2)))},
            "leaf_world_corners": corners.tolist(),
            "leaf_plane_normal": leaf_n.tolist(),
            "leaf_plane_d0": leaf_d0,
            "points_within_4cm_of_leaf_plane_in_rect": int(m.sum()),
            "leaf_points_per_10cm_band": prof,
        },
        open(OUT / "stage4_lidar_leaf.json", "w", encoding="utf-8"),
        indent=1,
    )
    print("wrote stage4_lidar_leaf.json; corners:", np.round(corners, 3).tolist())


if __name__ == "__main__":
    main()
