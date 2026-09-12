"""Stage 5: time-sorted contact sheets of the door crop for every Tile_1 training view in
which the door opening is fully inside the face, with the opening (yellow) and the LiDAR
leaf (magenta) projected, so the photo door state (open at the LiDAR angle / closed /
other) can be labelled per view. Also a per-view gap-region luma so carport-side views
can be classified automatically (open -> dark interior, closed -> white leaf).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-work")
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402

from door_plane_stage3 import DATASET, FACE_ROOT, OUT, quad_pixels  # noqa: E402


def main() -> None:
    s2 = json.load(open(OUT / "stage2_door_opening.json", encoding="utf-8"))
    corners = np.asarray([s2["opening_world_corners"][k] for k in ("bottom_left", "bottom_right", "top_right", "top_left")])
    leaf = np.asarray(json.load(open(OUT / "stage4_lidar_leaf.json", encoding="utf-8"))["leaf_world_corners"])
    dm = json.load(open(DATASET, encoding="utf-8"))
    c2w_of = {im["image_id"]: np.asarray(im["c2w"], dtype=np.float64) for im in dm["images"]}
    cam_of = {im["image_id"]: im["camera_id"] for im in dm["images"]}
    ts_of = {im["image_id"]: int(im["timestamp_ns"]) for im in dm["images"]}
    t0, t1 = min(ts_of.values()), max(ts_of.values())
    fm = json.load(open(FACE_ROOT / "face_manifest.json", encoding="utf-8"))
    faces = {cam: {f["face_id"]: FaceSpec.from_dict(f) for f in payload["faces"]} for cam, payload in fm["cameras"].items()}
    rows = [r for r in csv.DictReader(open(OUT / "stage3_views.csv", encoding="utf-8")) if r["quad_fully_inside"] == "True"]
    for r in rows:
        r["capture_fraction"] = (ts_of[r["image_id"]] - t0) / (t1 - t0)
    rows.sort(key=lambda r: r["capture_fraction"])

    TH = 300
    out_rows = []
    thumbs = {"carport": [], "interior": []}
    for k, r in enumerate(rows):
        iid, fid = r["sample_id"].split("::")
        c2w = c2w_of[iid]
        face = faces[cam_of[iid]][fid]
        qp, _, _ = quad_pixels(face, c2w, corners)
        lp, _, lz = quad_pixels(face, c2w, leaf)
        rgb = np.asarray(Image.open(FACE_ROOT / "faces" / f"{iid}_{fid}_rgb.png").convert("RGB"))
        H, W = rgb.shape[:2]
        x0, y0 = qp.min(0)
        x1, y1 = qp.max(0)
        mx, my = 0.25 * (x1 - x0) + 20, 0.25 * (y1 - y0) + 20
        X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        X1, Y1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
        crop = rgb[Y0:Y1, X0:X1]
        # gap region = opening quad minus LiDAR-leaf polygon, eroded 6 px to avoid jamb edges
        qmask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(qmask, [np.round(qp).astype(np.int32)], 1)
        lmask = np.zeros((H, W), np.uint8)
        if (lz > 0).all():
            cv2.fillPoly(lmask, [np.round(lp).astype(np.int32)], 1)
        gap = cv2.erode(qmask, np.ones((13, 13), np.uint8)) & (1 - lmask)
        leafpx = cv2.erode(qmask & lmask, np.ones((13, 13), np.uint8))
        luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        gap_luma = float(luma[gap > 0].mean()) if gap.any() else float("nan")
        gap_p90 = float(np.percentile(luma[gap > 0], 90)) if gap.any() else float("nan")
        leaf_luma = float(luma[leafpx > 0].mean()) if leafpx.any() else float("nan")
        side = "carport" if float(r["cam_signed_dist_to_plane_m"]) > 0 else "interior"
        out_rows.append({
            "idx": k, "sample_id": r["sample_id"], "side": side, "capture_fraction": round(r["capture_fraction"], 4),
            "cam_to_door_centre_m": round(float(r["cam_to_door_centre_m"]), 2),
            "gap_pixels": int(gap.sum()), "gap_luma_mean": round(gap_luma, 1), "gap_luma_p90": round(gap_p90, 1),
            "leaf_region_pixels": int(leafpx.sum()), "leaf_region_luma_mean": round(leaf_luma, 1),
            "in_roi_list": r["in_roi_list"],
        })
        im = Image.fromarray(crop)
        s = TH / im.size[1]
        im = im.resize((max(1, int(im.size[0] * s)), TH), Image.BILINEAR)
        dr = ImageDraw.Draw(im)
        dr.polygon([((x - X0) * s, (y - Y0) * s) for x, y in qp], outline=(255, 255, 0))
        if (lz > 0).all():
            dr.polygon([((x - X0) * s, (y - Y0) * s) for x, y in lp], outline=(255, 0, 255))
        dr.text((2, 2), "%d cf%.3f %.1fm" % (k, r["capture_fraction"], float(r["cam_to_door_centre_m"])), fill=(0, 255, 255))
        dr.text((2, 14), "gap %.0f" % gap_luma, fill=(0, 255, 255))
        thumbs[side].append(im)

    with open(OUT / "stage5_door_state_views.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)

    for side, ims in thumbs.items():
        if not ims:
            continue
        cols = 10
        cw = max(i.size[0] for i in ims)
        n_rows = (len(ims) + cols - 1) // cols
        sheet_rows = []
        for start in range(0, len(ims), cols * 6):
            chunk = ims[start:start + cols * 6]
            nr = (len(chunk) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * cw, nr * TH), (20, 20, 20))
            for j, im in enumerate(chunk):
                sheet.paste(im, ((j % cols) * cw, (j // cols) * TH))
            sheet.save(OUT / f"stage5_sheet_{side}_{start // (cols * 6):02d}.png")
            sheet_rows.append(sheet.size)
        print(side, len(ims), "thumbs ->", sheet_rows)


if __name__ == "__main__":
    main()
