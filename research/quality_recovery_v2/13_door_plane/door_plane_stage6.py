"""Stage 6: per-view line-of-sight test from the LiDAR cache. For every door-visible Tile_1
training sample, split the door quad into the LiDAR-leaf polygon and the 'gap' (quad minus
leaf) and report where the vis6 / vis6f returns lie relative to the wall plane in each part.
A gap whose returns are mostly this side of the plane (delta < -0.3 m) means a LiDAR surface
(another wall, a vehicle, furniture) sits between the camera and the doorway: the door is not
in view, whatever the photo shows.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-work")
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402

from door_plane_stage3 import BEHIND_M, DATASET, FACE_ROOT, ON_M, OUT, VIS6, VIS6F, load_sparse, quad_pixels  # noqa: E402


def region_stats(face: FaceSpec, c2w: np.ndarray, n: np.ndarray, d0: float, mask: np.ndarray, idx: np.ndarray, rng: np.ndarray, W: int) -> dict:
    vi, ui = np.divmod(idx, W)
    inq = mask[vi, ui] > 0
    out = {"px": int(mask.sum()), "n": int(inq.sum())}
    if not inq.any():
        return out
    px = np.column_stack([ui[inq] + 0.5, vi[inq] + 0.5]).astype(np.float64)
    d_w = face.pixels_to_directions(px) @ c2w[:3, :3].T
    t_plane = -(c2w[:3, 3] @ n + d0) / (d_w @ n)
    delta = rng[inq] - t_plane
    out.update({
        "delta_p50": float(np.median(delta)),
        "beyond": int((delta > BEHIND_M).sum()), "on": int((np.abs(delta) <= ON_M).sum()), "near": int((delta < -BEHIND_M).sum()),
    })
    return out


def main() -> None:
    s2 = json.load(open(OUT / "stage2_door_opening.json", encoding="utf-8"))
    pl = s2["plane_strip_refit"]
    n, d0 = np.asarray(pl["normal"]), float(pl["d0"])
    corners = np.asarray([s2["opening_world_corners"][k] for k in ("bottom_left", "bottom_right", "top_right", "top_left")])
    lj = json.load(open(OUT / "stage4_lidar_leaf.json", encoding="utf-8"))
    leaf = np.asarray(lj["leaf_world_corners"])
    leaf_n, leaf_d0 = np.asarray(lj["leaf_plane_normal"]), float(lj["leaf_plane_d0"])
    dm = json.load(open(DATASET, encoding="utf-8"))
    c2w_of = {im["image_id"]: np.asarray(im["c2w"], dtype=np.float64) for im in dm["images"]}
    cam_of = {im["image_id"]: im["camera_id"] for im in dm["images"]}
    fm = json.load(open(FACE_ROOT / "face_manifest.json", encoding="utf-8"))
    faces = {cam: {f["face_id"]: FaceSpec.from_dict(f) for f in payload["faces"]} for cam, payload in fm["cameras"].items()}
    views = list(csv.DictReader(open(OUT / "stage5_door_state_views.csv", encoding="utf-8")))

    out_rows = []
    for r in views:
        iid, fid = r["sample_id"].split("::")
        c2w = c2w_of[iid]
        face = faces[cam_of[iid]][fid]
        qp, _, _ = quad_pixels(face, c2w, corners)
        lp, _, lz = quad_pixels(face, c2w, leaf)
        row = dict(r)
        for tag, root in (("vis6", VIS6), ("vis6f", VIS6F)):
            path = root / (r["sample_id"].replace("::", "_") + ".npz")
            if not path.exists():
                continue
            idx, rng, shape = load_sparse(path)
            H, W = shape
            qmask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(qmask, [np.round(qp).astype(np.int32)], 1)
            lmask = np.zeros((H, W), np.uint8)
            if (lz > 0).all():
                cv2.fillPoly(lmask, [np.round(lp).astype(np.int32)], 1)
            k = np.ones((13, 13), np.uint8)
            gap = cv2.erode(qmask, k) & (1 - lmask)
            leaf_in_quad = cv2.erode(qmask & lmask, k)
            g = region_stats(face, c2w, n, d0, gap, idx, rng, W)
            for kk, v in g.items():
                row[f"{tag}_gap_{kk}"] = v
            if leaf_in_quad.any():
                l1 = region_stats(face, c2w, leaf_n, leaf_d0, leaf_in_quad, idx, rng, W)  # against the LEAF plane
                for kk, v in l1.items():
                    row[f"{tag}_leafpoly_vs_leafplane_{kk}"] = v
            # whole quad against the wall plane (same as stage 3, kept for one-table convenience)
            q = region_stats(face, c2w, n, d0, qmask, idx, rng, W)
            for kk, v in q.items():
                row[f"{tag}_quad_{kk}"] = v
        # LiDAR line of sight: doorway (gap) returns mostly this side of the plane => occluded
        gn = row.get("vis6f_gap_n", 0)
        if gn and gn >= 20:
            row["lidar_los"] = "occluded" if row["vis6f_gap_near"] > 0.5 * gn else "clear"
        else:
            row["lidar_los"] = "no_lidar"
        out_rows.append(row)

    cols = list(out_rows[0].keys())
    for r in out_rows:
        for c in r.keys():
            if c not in cols:
                cols.append(c)
    with open(OUT / "stage6_visibility.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in out_rows:
            wr.writerow({c: r.get(c, "") for c in cols})
    for side in ("carport", "interior"):
        sel = [r for r in out_rows if r["side"] == side]
        print(side, {k: sum(1 for r in sel if r["lidar_los"] == k) for k in ("clear", "occluded", "no_lidar")})
        for r in sel:
            print("  %3s cf%s d%s %-9s gap n=%s p50=%s beyond=%s on=%s near=%s | leafpoly n=%s on=%s beyond=%s near=%s" % (
                r["idx"], r["capture_fraction"], r["cam_to_door_centre_m"], r["lidar_los"], r.get("vis6f_gap_n"),
                None if r.get("vis6f_gap_delta_p50") is None else round(r["vis6f_gap_delta_p50"], 2), r.get("vis6f_gap_beyond"), r.get("vis6f_gap_on"), r.get("vis6f_gap_near"),
                r.get("vis6f_leafpoly_vs_leafplane_n"), r.get("vis6f_leafpoly_vs_leafplane_on"), r.get("vis6f_leafpoly_vs_leafplane_beyond"), r.get("vis6f_leafpoly_vs_leafplane_near")))


if __name__ == "__main__":
    main()
