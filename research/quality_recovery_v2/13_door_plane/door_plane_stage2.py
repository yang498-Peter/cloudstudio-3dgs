"""Stage 2: locate the door opening on the wall plane, refit the plane from the wall strips
left/right of the opening, and histogram the signed plane distance of every initialization
point whose in-plane (u,w) coordinate falls inside the opening.

Sign convention: +d = carport / photo side (toward the ROI compare cameras), -d = interior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from door_plane_stage1 import fit_plane_lsq, load_union, read_ply

OUT = Path(__file__).resolve().parent
TILE1_PLY = Path(r"C:\Peter\3dgs-runs\house0305_sop\tile_inputs_v9\Tile_1\initialization_full_lidar.ply")
LAS = Path(r"C:\Peter\testdata\S1\house0305\colorized.las")


def frame_from_plane(n: np.ndarray, d0: float) -> dict:
    up = np.array([0.0, 0.0, 1.0])
    e_u = np.cross(up, n)
    e_u /= np.linalg.norm(e_u)
    if e_u[0] < 0:
        e_u = -e_u
    e_w = np.cross(n, e_u)  # in-plane vertical (~world up)
    if e_w[2] < 0:
        e_w = -e_w
    origin = np.array([7.0, 0.0, 0.0])
    origin = origin - (origin @ n + d0) * n
    return {"n": n, "d0": d0, "e_u": e_u, "e_w": e_w, "origin": origin}


def plane_coords(P: np.ndarray, fr: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = P - fr["origin"]
    return rel @ fr["e_u"], rel @ fr["e_w"], P @ fr["n"] + fr["d0"]


def detect_opening(u: np.ndarray, w: np.ndarray, d: np.ndarray, cell: float = 0.025) -> dict:
    """Find the rectangular hole in the on-plane occupancy around u~1.0, w~1.8."""
    on = np.abs(d) < 0.05
    u_lo, u_hi, w_lo, w_hi = -1.0, 3.0, -0.5, 3.5
    nu, nw = int((u_hi - u_lo) / cell), int((w_hi - w_lo) / cell)
    H, _, _ = np.histogram2d(u[on], w[on], bins=[nu, nw], range=[[u_lo, u_hi], [w_lo, w_hi]])
    # column profile over the mid-height band w 1.2..2.4
    r0, r1 = int((1.2 - w_lo) / cell), int((2.4 - w_lo) / cell)
    col = H[:, r0:r1].sum(1)
    wall_med = np.median(col[(col > 0)])
    c_seed = int((1.0 - u_lo) / cell)
    assert col[c_seed] < 0.02 * wall_med, "seed column is not empty: %s" % col[c_seed]
    c0 = c_seed
    while c0 > 0 and col[c0 - 1] < 0.02 * wall_med:
        c0 -= 1
    c1 = c_seed
    while c1 < nu - 1 and col[c1 + 1] < 0.02 * wall_med:
        c1 += 1
    u0, u1 = u_lo + c0 * cell, u_lo + (c1 + 1) * cell
    # row profile over the inner columns
    ci0, ci1 = int((u0 + 0.1 - u_lo) / cell), int((u1 - 0.1 - u_lo) / cell)
    row = H[ci0:ci1, :].sum(0)
    # reference: the wall strip left of the hole at the same rows, scaled to the inner column count
    cl0, cl1 = int((u0 - 0.6 - u_lo) / cell), int((u0 - 0.1 - u_lo) / cell)
    row_ref = H[cl0:cl1, :].sum(0) * (ci1 - ci0) / float(cl1 - cl0)
    ref = np.median(row_ref[int((1.0 - w_lo) / cell):int((2.6 - w_lo) / cell)])
    r_seed = int((1.8 - w_lo) / cell)
    assert row[r_seed] < 0.02 * ref
    rr0 = r_seed
    while rr0 > 0 and row[rr0 - 1] < 0.02 * ref:
        rr0 -= 1
    rr1 = r_seed
    while rr1 < nw - 1 and row[rr1 + 1] < 0.02 * ref:
        rr1 += 1
    w0, w1 = w_lo + rr0 * cell, w_lo + (rr1 + 1) * cell
    return {
        "u0": float(u0), "u1": float(u1), "w0": float(w0), "w1": float(w1),
        "width_m": float(u1 - u0), "height_m": float(w1 - w0),
        "wall_column_median_count_w1p2_2p4": float(wall_med),
        "hole_column_max_count": float(col[c0:c1 + 1].max()),
        "lintel_row_reference_count": float(ref),
        "hole_row_max_count": float(row[rr0:rr1 + 1].max()),
    }


def hist_report(d: np.ndarray, label: str) -> dict:
    edges = np.arange(-6.0, 4.0001, 0.05)
    h, _ = np.histogram(d, bins=edges)
    n = d.size
    rep = {
        "label": label,
        "n_points": int(n),
        "on_plane_abs_le_0p10": int((np.abs(d) <= 0.10).sum()),
        "on_plane_abs_le_0p05": int((np.abs(d) <= 0.05).sum()),
        "behind_gt_0p3": int((d < -0.3).sum()),
        "behind_0p1_to_0p3": int(((d < -0.1) & (d >= -0.3)).sum()),
        "front_gt_0p3": int((d > 0.3).sum()),
        "front_0p1_to_0p3": int(((d > 0.1) & (d <= 0.3)).sum()),
        "behind_percentiles_m": {p: float(-np.percentile(d[d < -0.3], q)) for p, q in (("p05", 95), ("p50", 50), ("p95", 5))} if (d < -0.3).any() else None,
        "hist_edges_m": edges.tolist(),
        "hist_counts": h.tolist(),
    }
    if n:
        rep["frac_on_plane_0p10"] = rep["on_plane_abs_le_0p10"] / n
        rep["frac_behind_gt_0p3"] = rep["behind_gt_0p3"] / n
        rep["frac_front_gt_0p3"] = rep["front_gt_0p3"] / n
    return rep


def main() -> None:
    s1 = json.load(open(OUT / "stage1_plane.json", encoding="utf-8"))
    n0 = np.asarray(s1["plane_normal_toward_carport"])
    d00 = float(s1["plane_d0"])
    xyz, rgb, info = load_union()
    box = (xyz[:, 0] > 3.5) & (xyz[:, 0] < 11.0) & (xyz[:, 1] > -6.0) & (xyz[:, 1] < 1.0) & (xyz[:, 2] > -1.0) & (xyz[:, 2] < 4.5)
    P = xyz[box]
    fr0 = frame_from_plane(n0, d00)
    u, w, d = plane_coords(P, fr0)
    op = detect_opening(u, w, d)
    print("opening (RANSAC frame):", json.dumps(op, indent=1))

    # --- refit from the wall strips left / right of the opening -------------------------
    strip_w = (w > op["w0"] + 0.15) & (w < op["w1"] - 0.15)
    left = (u > op["u0"] - 0.65) & (u < op["u0"] - 0.10) & strip_w & (np.abs(d) < 0.08)
    right = (u > op["u1"] + 0.10) & (u < op["u1"] + 0.65) & strip_w & (np.abs(d) < 0.08)
    strips = {}
    for name, m in (("left", left), ("right", right), ("both", left | right)):
        nn, dd, res = fit_plane_lsq(P[m])
        if nn[1] < 0:
            nn, dd, res = -nn, -dd, -res
        strips[name] = {
            "n_points": int(m.sum()),
            "normal": nn.tolist(),
            "d0": float(dd),
            "resid_rms_m": float(np.sqrt(np.mean(res**2))),
            "resid_p50_abs_m": float(np.median(np.abs(res))),
            "resid_p95_abs_m": float(np.percentile(np.abs(res), 95)),
        }
    nL, nR = np.asarray(strips["left"]["normal"]), np.asarray(strips["right"]["normal"])
    n, d0 = np.asarray(strips["both"]["normal"]), strips["both"]["d0"]
    fr = frame_from_plane(n, d0)
    door_c0 = fr0["origin"] + 0.5 * (op["u0"] + op["u1"]) * fr0["e_u"] + 0.5 * (op["w0"] + op["w1"]) * fr0["e_w"]
    plane_cmp = {
        "angle_left_vs_right_deg": float(np.degrees(np.arccos(np.clip(nL @ nR, -1, 1)))),
        "angle_ransac_vs_strips_deg": float(np.degrees(np.arccos(np.clip(n0 @ n, -1, 1)))),
        "offset_left_vs_right_at_door_centre_m": float((door_c0 @ nL + strips["left"]["d0"]) - (door_c0 @ nR + strips["right"]["d0"])),
        "offset_ransac_vs_strips_at_door_centre_m": float((door_c0 @ n0 + d00) - (door_c0 @ n + d0)),
        "tilt_from_vertical_deg": float(np.degrees(np.arcsin(abs(n[2])))),
    }
    print("strips:", json.dumps(strips, indent=1))
    print("plane comparison:", json.dumps(plane_cmp, indent=1))

    # --- redo coordinates in the refined frame and re-detect the opening ------------------
    u, w, d = plane_coords(P, fr)
    op = detect_opening(u, w, d)
    print("opening (strip-refit frame):", json.dumps(op, indent=1))
    inner = (u > op["u0"] + 0.03) & (u < op["u1"] - 0.03) & (w > op["w0"] + 0.03) & (w < op["w1"] - 0.03)
    z_of_w = lambda ww: float(fr["origin"][2] + ww * fr["e_w"][2])  # noqa: E731
    reports = {
        "opening_all_depths_tiles": hist_report(d[inner], "Tile_1+Tile_2 init points inside opening (u,w), any distance"),
        "opening_roi_zband_tiles": hist_report(d[inner & (P[:, 2] > 1.6) & (P[:, 2] < 3.2)], "same, restricted to ROI z 1.6-3.2"),
    }
    # how deep does the Tile_1 crop reach behind the plane at the door?  (y >= -5.0569)
    y_min_tile1 = -5.056892571428578
    depth_cap = None
    # walk straight back along -n from the door centre until y hits the crop
    c = fr["origin"] + 0.5 * (op["u0"] + op["u1"]) * fr["e_u"] + 0.5 * (op["w0"] + op["w1"]) * fr["e_w"]
    if n[1] > 0:
        depth_cap = float((c[1] - y_min_tile1) / n[1])
    print("Tile_1 crop reaches %.2f m behind the plane at the door centre" % depth_cap)

    # --- Tile_1 only (the actual R1d initialization file), same opening ---------------------
    a = read_ply(TILE1_PLY)
    T = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    ut, wt, dt = plane_coords(T, fr)
    inner_t = (ut > op["u0"] + 0.03) & (ut < op["u1"] - 0.03) & (wt > op["w0"] + 0.03) & (wt < op["w1"] - 0.03)
    reports["opening_all_depths_tile1_only"] = hist_report(dt[inner_t], "Tile_1 initialization_full_lidar.ply only, inside opening, any distance")
    # wall strips on the same file for a like-for-like 'how many points does a wall patch of door size carry'
    ref_patch = (ut > op["u0"] - 0.65 - (op["u1"] - op["u0"]) * 0) & (ut < op["u0"] - 0.10) & (wt > op["w0"]) & (wt < op["w1"])
    reports["left_wall_strip_tile1_same_height"] = hist_report(dt[ref_patch], "Tile_1 wall strip left of the opening (0.55 m wide, door height), any distance")

    # --- LAS (full recording, no tile crop) for depth beyond the Tile_1 crop ---------------
    las_rep = None
    try:
        import laspy

        L = laspy.read(str(LAS))
        Q = np.stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)], 1)
        ul, wl, dl = plane_coords(Q, fr)
        inner_l = (ul > op["u0"] + 0.03) & (ul < op["u1"] - 0.03) & (wl > op["w0"] + 0.03) & (wl < op["w1"] - 0.03)
        las_rep = hist_report(dl[inner_l], "colorized.las (full recording), inside opening, any distance")
        las_rep["las_points_total"] = int(Q.shape[0])
        reports["opening_all_depths_las_full"] = las_rep
        # also deeper histogram so we see how far the interior extends
        dd = dl[inner_l]
        edges = np.arange(-15.0, 4.0001, 0.25)
        h, _ = np.histogram(dd, bins=edges)
        las_rep["hist_0p25_edges_m"] = edges.tolist()
        las_rep["hist_0p25_counts"] = h.tolist()
    except Exception as exc:  # noqa: BLE001
        print("LAS read failed:", exc)

    for k, r in reports.items():
        print("%s: n=%d on(|d|<=0.10)=%d (%.1f%%) behind>0.3=%d (%.1f%%) front>0.3=%d (%.1f%%) behind p05/50/95=%s" % (
            k, r["n_points"], r["on_plane_abs_le_0p10"], 100 * r.get("frac_on_plane_0p10", 0), r["behind_gt_0p3"],
            100 * r.get("frac_behind_gt_0p3", 0), r["front_gt_0p3"], 100 * r.get("frac_front_gt_0p3", 0), r["behind_percentiles_m"]))

    out = {
        "plane_ransac_stage1": {"normal": n0.tolist(), "d0": d00},
        "plane_strip_refit": {"normal": n.tolist(), "d0": d0, "e_u": fr["e_u"].tolist(), "e_w": fr["e_w"].tolist(), "origin": fr["origin"].tolist(),
                              "sign_convention": "+d toward carport/photo side (+y), -d interior"},
        "strips": strips,
        "plane_comparison": plane_cmp,
        "opening_in_plane": op,
        "opening_world_corners": {
            k: (fr["origin"] + uu * fr["e_u"] + ww * fr["e_w"]).tolist()
            for k, (uu, ww) in {"bottom_left": (op["u0"], op["w0"]), "bottom_right": (op["u1"], op["w0"]),
                                "top_right": (op["u1"], op["w1"]), "top_left": (op["u0"], op["w1"])}.items()
        },
        "opening_z_range_world": [z_of_w(op["w0"]), z_of_w(op["w1"])],
        "tile1_crop_depth_behind_plane_at_door_m": depth_cap,
        "cloud_info": info,
        "reports": reports,
    }
    json.dump(out, open(OUT / "stage2_door_opening.json", "w", encoding="utf-8"), indent=1)

    # histogram figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    r = reports["opening_all_depths_tile1_only"]
    e = np.asarray(r["hist_edges_m"])
    axes[0].bar(e[:-1], r["hist_counts"], width=0.05, align="edge", color="tab:red")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title("Tile_1 init points inside the door opening (u %.2f-%.2f, w %.2f-%.2f): signed distance to wall plane, 5 cm bins  (+ = carport side, - = interior)" % (op["u0"], op["u1"], op["w0"], op["w1"]), fontsize=9)
    axes[0].set_ylabel("points")
    if las_rep:
        e2 = np.asarray(las_rep["hist_0p25_edges_m"])
        axes[1].bar(e2[:-1], las_rep["hist_0p25_counts"], width=0.25, align="edge", color="tab:purple")
        axes[1].axvline(0, color="k", lw=1)
        axes[1].set_title("colorized.las (no tile crop), same opening, 25 cm bins", fontsize=9)
    axes[1].set_xlabel("signed distance to wall plane [m]")
    axes[1].set_ylabel("points")
    fig.tight_layout()
    fig.savefig(OUT / "stage2_signed_distance_hist.png", dpi=110)
    print("wrote", OUT / "stage2_door_opening.json")


if __name__ == "__main__":
    main()
