"""Stage 7: roll-up. Combines the per-view LOS test (stage 6), the photo door-state labels
(time clusters checked by eye on the stage 5 sheets / stage 3b overlays) and the LiDAR
cache classification into the numbers quoted in 13_door_plane_consistency.md.
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

from door_plane_stage3 import DATASET, FACE_ROOT, OUT, TILE_INPUTS, VIS6, VIS6F, load_sparse, quad_pixels  # noqa: E402

# photo door state by capture-fraction interval (labelled by eye on stage5 sheets + stage3b overlays)
CLOSED_INTERVAL = (0.85, 0.90)   # img_4a55ba71 (cf 0.8665) closed; neighbours 0.846 (open) and 0.923 (open)
TILE1_CROP_DEPTH_M = 2.65


def main() -> None:
    s2 = json.load(open(OUT / "stage2_door_opening.json", encoding="utf-8"))
    pl = s2["plane_strip_refit"]
    n, d0 = np.asarray(pl["normal"]), float(pl["d0"])
    corners = np.asarray([s2["opening_world_corners"][k] for k in ("bottom_left", "bottom_right", "top_right", "top_left")])
    leaf = np.asarray(json.load(open(OUT / "stage4_lidar_leaf.json", encoding="utf-8"))["leaf_world_corners"])
    door_centre = corners.mean(0)
    dm = json.load(open(DATASET, encoding="utf-8"))
    c2w_of = {im["image_id"]: np.asarray(im["c2w"], dtype=np.float64) for im in dm["images"]}
    cam_of = {im["image_id"]: im["camera_id"] for im in dm["images"]}
    ts_of = {im["image_id"]: int(im["timestamp_ns"]) for im in dm["images"]}
    t0, t1 = min(ts_of.values()), max(ts_of.values())
    fm = json.load(open(FACE_ROOT / "face_manifest.json", encoding="utf-8"))
    faces = {cam: {f["face_id"]: FaceSpec.from_dict(f) for f in payload["faces"]} for cam, payload in fm["cameras"].items()}
    tiles = {t["name"]: t for t in json.load(open(TILE_INPUTS, encoding="utf-8"))["tiles"]}
    tile1_samples = [v["sample_id"] for v in tiles["Tile_1"]["views"]]

    vis = list(csv.DictReader(open(OUT / "stage6_visibility.csv", encoding="utf-8")))
    for r in vis:
        r["cf"] = float(r["capture_fraction"])
        r["closed_interval"] = CLOSED_INTERVAL[0] <= r["cf"] <= CLOSED_INTERVAL[1]

    # (1) every Tile_1 sample where any part of the door quad is inside the face and the camera is < 6 m:
    #     which of them fall in the closed interval (partially visible ones are not in stage 3-6 tables)
    partial = []
    for sample in tile1_samples:
        iid, fid = sample.split("::")
        if iid not in c2w_of:
            continue
        c2w = c2w_of[iid]
        face = faces[cam_of[iid]][fid]
        px, inside, zf = quad_pixels(face, c2w, corners)
        cf = (ts_of[iid] - t0) / (t1 - t0)
        dist = float(np.linalg.norm(c2w[:3, 3] - door_centre))
        if inside.any() and (zf > 0).all() and not inside.all():
            partial.append({"sample_id": sample, "cf": round(cf, 4), "dist_m": round(dist, 2), "side": "carport" if c2w[:3, 3] @ n + d0 > 0 else "interior",
                            "corners_inside": int(inside.sum()), "closed_interval": CLOSED_INTERVAL[0] <= cf <= CLOSED_INTERVAL[1]})
    partial_closed = [p for p in partial if p["closed_interval"]]
    print("partially visible door views:", len(partial), "of which in closed interval:", len(partial_closed))
    for p in sorted(partial_closed, key=lambda p: p["cf"]):
        print("   ", p)

    # (2) roll-up over fully visible views
    def roll(sel, tag):
        q = sum(int(r[f"{tag}_quad_px"]) for r in sel if r.get(f"{tag}_quad_px"))
        nl = sum(int(r[f"{tag}_quad_n"]) for r in sel if r.get(f"{tag}_quad_n"))
        beyond = sum(int(r[f"{tag}_quad_beyond"]) for r in sel if r.get(f"{tag}_quad_beyond"))
        on = sum(int(r[f"{tag}_quad_on"]) for r in sel if r.get(f"{tag}_quad_on"))
        near = sum(int(r[f"{tag}_quad_near"]) for r in sel if r.get(f"{tag}_quad_near"))
        lp_n = sum(int(r[f"{tag}_leafpoly_vs_leafplane_n"]) for r in sel if r.get(f"{tag}_leafpoly_vs_leafplane_n"))
        lp_on = sum(int(r[f"{tag}_leafpoly_vs_leafplane_on"]) for r in sel if r.get(f"{tag}_leafpoly_vs_leafplane_on"))
        lp_beyond = sum(int(r[f"{tag}_leafpoly_vs_leafplane_beyond"]) for r in sel if r.get(f"{tag}_leafpoly_vs_leafplane_beyond"))
        return {"views": len(sel), "quad_px": q, "lidar_px": nl, "coverage": nl / q if q else None,
                "beyond_plane_frac": beyond / nl if nl else None, "on_plane_frac": on / nl if nl else None, "near_frac": near / nl if nl else None,
                "leafpoly_lidar_px": lp_n, "leafpoly_on_leaf_frac": lp_on / lp_n if lp_n else None, "leafpoly_beyond_leaf_frac": lp_beyond / lp_n if lp_n else None}

    groups = {
        "carport_clear": [r for r in vis if r["side"] == "carport" and r["lidar_los"] == "clear"],
        "carport_clear_frontal_cf0p14_0p23": [r for r in vis if r["side"] == "carport" and r["lidar_los"] == "clear" and 0.14 <= r["cf"] <= 0.23],
        "interior_clear_open": [r for r in vis if r["side"] == "interior" and r["lidar_los"] == "clear" and not r["closed_interval"]],
        "interior_clear_closed": [r for r in vis if r["side"] == "interior" and r["lidar_los"] == "clear" and r["closed_interval"]],
        "occluded_any_side": [r for r in vis if r["lidar_los"] == "occluded"],
    }
    summary = {"closed_interval_cf": CLOSED_INTERVAL, "groups": {}}
    for g, sel in groups.items():
        summary["groups"][g] = {tag: roll(sel, tag) for tag in ("vis6", "vis6f")}
        summary["groups"][g]["sample_ids"] = [r["sample_id"] for r in sel] if len(sel) <= 12 else len(sel)
        print(g, len(sel), json.dumps(summary["groups"][g]["vis6f"]))

    # (3) how much of the doorway LiDAR (gap region, carport frontal views) lies beyond the Tile_1 crop depth
    deep = {"views": 0, "gap_px": 0, "gap_lidar": 0, "beyond_0p3": 0, "beyond_crop_2p65": 0, "delta_p50_list": []}
    for r in groups["carport_clear_frontal_cf0p14_0p23"]:
        iid, fid = r["sample_id"].split("::")
        c2w = c2w_of[iid]
        face = faces[cam_of[iid]][fid]
        qp, _, _ = quad_pixels(face, c2w, corners)
        lp, _, lz = quad_pixels(face, c2w, leaf)
        idx, rng, shape = load_sparse(VIS6F / (r["sample_id"].replace("::", "_") + ".npz"))
        H, W = shape
        qmask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(qmask, [np.round(qp).astype(np.int32)], 1)
        lmask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(lmask, [np.round(lp).astype(np.int32)], 1)
        gap = cv2.erode(qmask, np.ones((13, 13), np.uint8)) & (1 - lmask)
        vi, ui = np.divmod(idx, W)
        inq = gap[vi, ui] > 0
        if not inq.any():
            continue
        px = np.column_stack([ui[inq] + 0.5, vi[inq] + 0.5]).astype(np.float64)
        d_w = face.pixels_to_directions(px) @ c2w[:3, :3].T
        t_plane = -(c2w[:3, 3] @ n + d0) / (d_w @ n)
        delta = rng[inq] - t_plane
        deep["views"] += 1
        deep["gap_px"] += int(gap.sum())
        deep["gap_lidar"] += int(inq.sum())
        deep["beyond_0p3"] += int((delta > 0.3).sum())
        deep["beyond_crop_2p65"] += int((delta > TILE1_CROP_DEPTH_M).sum())
        deep["delta_p50_list"].append(float(np.median(delta)))
    deep["beyond_crop_frac_of_gap_lidar"] = deep["beyond_crop_2p65"] / deep["gap_lidar"] if deep["gap_lidar"] else None
    deep["beyond_0p3_frac_of_gap_lidar"] = deep["beyond_0p3"] / deep["gap_lidar"] if deep["gap_lidar"] else None
    deep["delta_p50_median_over_views"] = float(np.median(deep["delta_p50_list"])) if deep["delta_p50_list"] else None
    summary["carport_frontal_gap_region_vis6f"] = deep
    print("gap region (carport frontal, vis6f):", json.dumps({k: v for k, v in deep.items() if k != "delta_p50_list"}))

    # (4) pixel weight of the closed-door view(s) among all LOS-clear door views
    clear = [r for r in vis if r["lidar_los"] == "clear"]
    q_all = sum(int(r["vis6f_quad_px"]) for r in clear if r.get("vis6f_quad_px"))
    q_closed = sum(int(r["vis6f_quad_px"]) for r in groups["interior_clear_closed"] if r.get("vis6f_quad_px"))
    summary["closed_view_pixel_share_of_clear_door_pixels"] = q_closed / q_all if q_all else None
    summary["partially_visible"] = {"total": len(partial), "in_closed_interval": partial_closed}
    print("closed-door views: %d fully visible, quad px %d of %d clear door px (%.1f%%); partially visible in interval: %d" % (
        len(groups["interior_clear_closed"]), q_closed, q_all, 100 * q_closed / q_all if q_all else 0, len(partial_closed)))
    json.dump(summary, open(OUT / "stage7_summary.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
