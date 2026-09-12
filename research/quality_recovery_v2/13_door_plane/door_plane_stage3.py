"""Stage 3: project the door opening and the initialization points inside it into the Face4
training views, overlay them on the face photos, and classify the face LiDAR cache
(vis6 = what R1d consumed, vis6f = hidden-point-filtered rebuild) at the door pixels
against the wall plane.

Outputs (research/quality_recovery_v2/13_door_plane/):
  overlay_<k>_<image>_<face>.png    photo | init points through the opening | LiDAR cache vs plane
  pick_<image>_<face>.png           gridded photo crop used to pick the window-pane corners
  stage3_views.csv / stage3_views.json  per-view door-pixel LiDAR classification
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

from door_plane_stage1 import read_ply  # noqa: E402

OUT = Path(__file__).resolve().parent
DATASET = Path(r"C:\Peter\3dgs-datasets\house0305_sop_v8\dataset_manifest.json")
FACE_ROOT = Path(r"C:\Peter\3dgs-datasets\house0305_sop_v9\face4_train")
VIS6 = Path(r"C:\Peter\3dgs-datasets\house0305_sop_v9\face4_lidar_train_vis6\depth")
VIS6F = Path(r"C:\Peter\3dgs-datasets\house0305_sop_v9\face4_lidar_train_vis6f\depth")
TILE_INPUTS = Path(r"C:\Peter\3dgs-runs\house0305_sop\tile_inputs_v9\tile_inputs_manifest.json")
ROI_LIST = Path(r"C:\Peter\3dgs-runs\house0305_sop\tile1_R1d_20k\compare_roi_compare_ids.roi.json")
TILE1_PLY = Path(r"C:\Peter\3dgs-runs\house0305_sop\tile_inputs_v9\Tile_1\initialization_full_lidar.ply")

BEHIND_M = 0.3   # LiDAR range further than the plane by more than this = "behind the leaf"
ON_M = 0.15      # |range - plane| <= this = "on the leaf"
DILATE_PX = 6    # R1d lidar_alpha_dilation_radius_px


def load_sparse(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    z = np.load(path)
    shape = tuple(int(v) for v in z["shape"])
    return z["pixel_index"].astype(np.int64), z["range_m"].astype(np.float64), shape


def quad_pixels(face: FaceSpec, c2w: np.ndarray, corners_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc = (corners_world - c2w[:3, 3]) @ c2w[:3, :3]
    px, inside = face.directions_to_pixels(pc)
    df = pc @ face.R_face
    return px, inside, df[:, 2]


def classify_view(face: FaceSpec, c2w: np.ndarray, plane_n: np.ndarray, plane_d0: float, quad_px: np.ndarray, cache_dir: Path, sample: str) -> dict | None:
    path = cache_dir / (sample.replace("::", "_") + ".npz")
    if not path.exists():
        return None
    idx, rng, shape = load_sparse(path)
    H, W = shape
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(quad_px).astype(np.int32)], 1)
    quad_area = int(mask.sum())
    vi, ui = np.divmod(idx, W)
    inq = mask[vi, ui] > 0
    out = {"quad_pixels": quad_area, "n_lidar_in_quad": int(inq.sum())}
    # alpha-floor footprint: LiDAR support dilated by DILATE_PX, intersected with the quad
    sup = np.zeros((H, W), np.uint8)
    sup[vi, ui] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
    dil = cv2.dilate(sup, k)
    out["quad_pixels_in_dilated_support"] = int(((dil > 0) & (mask > 0)).sum())
    if inq.sum() == 0:
        return out
    px = np.column_stack([ui[inq] + 0.5, vi[inq] + 0.5]).astype(np.float64)
    d_cam = face.pixels_to_directions(px)
    d_w = d_cam @ c2w[:3, :3].T
    c = c2w[:3, 3]
    denom = d_w @ plane_n
    t_plane = -(c @ plane_n + plane_d0) / denom
    r = rng[inq]
    delta = r - t_plane
    out.update({
        "t_plane_p50": float(np.median(t_plane)),
        "range_p05": float(np.percentile(r, 5)), "range_p50": float(np.median(r)), "range_p95": float(np.percentile(r, 95)),
        "delta_p05": float(np.percentile(delta, 5)), "delta_p50": float(np.median(delta)), "delta_p95": float(np.percentile(delta, 95)),
        "n_behind": int((delta > BEHIND_M).sum()),
        "n_on_leaf": int((np.abs(delta) <= ON_M).sum()),
        "n_front": int((delta < -BEHIND_M).sum()),
    })
    out["_pix"] = (ui[inq], vi[inq], delta)
    return out


def main() -> None:
    s2 = json.load(open(OUT / "stage2_door_opening.json", encoding="utf-8"))
    pl = s2["plane_strip_refit"]
    n = np.asarray(pl["normal"])
    d0 = float(pl["d0"])
    e_u, e_w, origin = (np.asarray(pl[k]) for k in ("e_u", "e_w", "origin"))
    op = s2["opening_in_plane"]
    corners = np.asarray([s2["opening_world_corners"][k] for k in ("bottom_left", "bottom_right", "top_right", "top_left")])
    door_centre = corners.mean(0)

    dm = json.load(open(DATASET, encoding="utf-8"))
    c2w_of = {im["image_id"]: np.asarray(im["c2w"], dtype=np.float64) for im in dm["images"]}
    cam_of = {im["image_id"]: im["camera_id"] for im in dm["images"]}
    fm = json.load(open(FACE_ROOT / "face_manifest.json", encoding="utf-8"))
    faces = {cam: {f["face_id"]: FaceSpec.from_dict(f) for f in payload["faces"]} for cam, payload in fm["cameras"].items()}
    tiles = {t["name"]: t for t in json.load(open(TILE_INPUTS, encoding="utf-8"))["tiles"]}
    tile1_samples = [v["sample_id"] for v in tiles["Tile_1"]["views"]]
    roi_frames = json.load(open(ROI_LIST, encoding="utf-8"))[0]["frames"]
    roi_samples = [f["sample_id"] for f in roi_frames]

    # initialization points inside the opening footprint (Tile_1 file = what R1d was seeded with)
    a = read_ply(TILE1_PLY)
    T = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    rel = T - origin
    u, w, d = rel @ e_u, rel @ e_w, T @ n + d0
    inner = (u > op["u0"] + 0.03) & (u < op["u1"] - 0.03) & (w > op["w0"] + 0.03) & (w < op["w1"] - 0.03)
    Pin, din = T[inner], d[inner]
    print("init points inside opening footprint:", Pin.shape[0])

    rows = []
    for sample in tile1_samples:
        iid, fid = sample.split("::")
        if iid not in c2w_of:
            continue
        c2w = c2w_of[iid]
        face = faces[cam_of[iid]][fid]
        px, inside, zf = quad_pixels(face, c2w, corners)
        cam_side = float(c2w[:3, 3] @ n + d0)
        dist = float(np.linalg.norm(c2w[:3, 3] - door_centre))
        row = {
            "sample_id": sample, "image_id": iid, "face_id": fid, "camera_id": cam_of[iid],
            "cam_signed_dist_to_plane_m": cam_side, "cam_to_door_centre_m": dist,
            "quad_fully_inside": bool(inside.all()), "quad_any_inside": bool(inside.any() and (zf > 0).all()),
            "in_roi_list": sample in roi_samples,
        }
        if not (inside.all() and (zf > 0).all()):
            rows.append(row)
            continue
        for tag, root in (("vis6", VIS6), ("vis6f", VIS6F)):
            r = classify_view(face, c2w, n, d0, px, root, sample)
            if r is None:
                continue
            pix = r.pop("_pix", None)
            for k, v in r.items():
                row[f"{tag}_{k}"] = v
            if tag == "vis6f":
                row["_pix_vis6f"] = pix
            if tag == "vis6":
                row["_pix_vis6"] = pix
        row["_quad_px"] = px
        rows.append(row)

    # ---------------- overlays for the five figure views ----------------
    figure_samples = roi_samples[:5]
    by_sample = {r["sample_id"]: r for r in rows}
    for k, sample in enumerate(figure_samples):
        r = by_sample.get(sample)
        if r is None or "_quad_px" not in r:
            print("figure view not fully visible:", sample, r and r.get("quad_any_inside"))
            continue
        iid, fid = sample.split("::")
        c2w = c2w_of[iid]
        face = faces[cam_of[iid]][fid]
        rgb = np.asarray(Image.open(FACE_ROOT / "faces" / f"{iid}_{fid}_rgb.png").convert("RGB"))
        H, W = rgb.shape[:2]
        qp = r["_quad_px"]
        x0, y0 = qp.min(0)
        x1, y1 = qp.max(0)
        mx, my = max(0.45 * (x1 - x0), 120), max(0.45 * (y1 - y0), 120)
        X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        X1, Y1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
        crop = rgb[Y0:Y1, X0:X1].copy()

        def panel(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
            im = Image.fromarray(crop.copy())
            dr = ImageDraw.Draw(im)
            dr.polygon([(float(x - X0), float(y - Y0)) for x, y in qp], outline=(255, 255, 0))
            dr.text((4, 4), title, fill=(255, 255, 0))
            return im, dr

        # panel 1: photo + opening quad
        p1, _ = panel(f"{iid[:12]} {fid}  photo + door opening (plane quad)")
        # panel 2: Tile_1 init points inside the opening footprint, coloured by plane distance
        pc = (Pin - c2w[:3, 3]) @ c2w[:3, :3]
        ppx, pin = face.directions_to_pixels(pc)
        p2, dr2 = panel("Tile_1 init points in opening footprint: green |d|<=0.1 on plane, red->yellow behind (0.1..3 m), blue in front")
        order = np.argsort(-np.abs(din))  # draw the far ones first
        nb = 0
        for j in order:
            if not pin[j]:
                continue
            x, y = ppx[j] - (X0, Y0)
            if not (0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]):
                continue
            dd = din[j]
            if abs(dd) <= 0.1:
                col = (0, 255, 0)
            elif dd < 0:
                t = min(1.0, (-dd - 0.1) / 2.9)
                col = (255, int(255 * t), 0)
                nb += 1
            else:
                col = (60, 120, 255)
            dr2.point((x, y), fill=col)
        # panel 3: vis6f LiDAR cache pixels vs plane
        p3, dr3 = panel(f"vis6f LiDAR cache in quad: red range>plane+{BEHIND_M} (behind leaf), green |range-plane|<={ON_M}, blue in front, orange between")
        pix = r.get("_pix_vis6f")
        if pix is not None:
            ui, vi, delta = pix
            for x, y, dd in zip(ui, vi, delta):
                if dd > BEHIND_M:
                    col = (255, 40, 40)
                elif abs(dd) <= ON_M:
                    col = (0, 255, 0)
                elif dd < -BEHIND_M:
                    col = (60, 120, 255)
                else:
                    col = (255, 160, 0)
                dr3.point((int(x) - X0, int(y) - Y0), fill=col)
        stats = "vis6f: n=%d behind=%d on=%d front=%d | t_plane p50 %.2f m, range p50 %.2f m" % (
            r.get("vis6f_n_lidar_in_quad", 0), r.get("vis6f_n_behind", 0), r.get("vis6f_n_on_leaf", 0), r.get("vis6f_n_front", 0),
            r.get("vis6f_t_plane_p50", float("nan")), r.get("vis6f_range_p50", float("nan")))
        dr3.text((4, 18), stats, fill=(255, 255, 0))
        dr2.text((4, 18), "points drawn behind plane: %d (of %d in footprint)" % (nb, Pin.shape[0]), fill=(255, 255, 0))
        Wc, Hc = p1.size
        sheet = Image.new("RGB", (3 * Wc + 8, Hc), (0, 0, 0))
        for i, p in enumerate((p1, p2, p3)):
            sheet.paste(p, (i * (Wc + 4), 0))
        sheet.save(OUT / f"overlay_{k}_{iid[:12]}_{fid}.png")
        print("wrote overlay", k, sample, crop.shape, stats)

        if k == 0:
            # gridded pick image at 2x for the window-pane corners (face pixel coordinates)
            S = 2
            big = Image.fromarray(crop).resize((crop.shape[1] * S, crop.shape[0] * S), Image.BICUBIC)
            dg = ImageDraw.Draw(big)
            step = 25
            for gx in range((X0 // step) * step, X1 + 1, step):
                xx = (gx - X0) * S
                dg.line([(xx, 0), (xx, big.size[1])], fill=(255, 0, 0) if gx % 100 == 0 else (110, 0, 0), width=1)
                if gx % 100 == 0:
                    dg.text((xx + 2, 2), str(gx), fill=(0, 255, 255))
            for gy in range((Y0 // step) * step, Y1 + 1, step):
                yy = (gy - Y0) * S
                dg.line([(0, yy), (big.size[0], yy)], fill=(255, 0, 0) if gy % 100 == 0 else (110, 0, 0), width=1)
                if gy % 100 == 0:
                    dg.text((2, yy + 2), str(gy), fill=(0, 255, 255))
            dg.polygon([((x - X0) * S, (y - Y0) * S) for x, y in qp], outline=(255, 255, 0))
            big.save(OUT / f"pick_{iid[:12]}_{fid}.png")
            json.dump({"sample_id": sample, "crop_origin_xy": [X0, Y0], "scale": S}, open(OUT / "pick_meta.json", "w"), indent=1)

    # ---------------- tables ----------------
    keep = [k for k in rows[0].keys()] if rows else []
    cols = sorted({k for r in rows for k in r.keys() if not k.startswith("_")}, key=lambda s: (s not in keep, s))
    with open(OUT / "stage3_views.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in cols})

    def agg(sel: list[dict], tag: str) -> dict:
        v = [r for r in sel if f"{tag}_n_lidar_in_quad" in r]
        q = sum(r[f"{tag}_quad_pixels"] for r in v)
        nl = sum(r[f"{tag}_n_lidar_in_quad"] for r in v)
        nb = sum(r.get(f"{tag}_n_behind", 0) for r in v)
        no = sum(r.get(f"{tag}_n_on_leaf", 0) for r in v)
        nf = sum(r.get(f"{tag}_n_front", 0) for r in v)
        nd = sum(r[f"{tag}_quad_pixels_in_dilated_support"] for r in v)
        per_view_behind = [r["%s_n_behind" % tag] / r["%s_n_lidar_in_quad" % tag] for r in v if r.get("%s_n_lidar_in_quad" % tag, 0) > 0]
        return {
            "views": len(v), "quad_pixels": q, "lidar_pixels_in_quad": nl,
            "coverage_frac": nl / q if q else None,
            "alpha_floor_footprint_frac_of_quad": nd / q if q else None,
            "behind_leaf": nb, "on_leaf": no, "in_front": nf,
            "behind_frac_of_lidar": nb / nl if nl else None, "on_frac_of_lidar": no / nl if nl else None, "front_frac_of_lidar": nf / nl if nl else None,
            "per_view_behind_frac_p10_p50_p90": [float(np.percentile(per_view_behind, p)) for p in (10, 50, 90)] if per_view_behind else None,
        }

    vis = [r for r in rows if "_quad_px" in r]
    carport = [r for r in vis if r["cam_signed_dist_to_plane_m"] > 0]
    interior = [r for r in vis if r["cam_signed_dist_to_plane_m"] < 0]
    roi = [r for r in vis if r["in_roi_list"]]
    summary = {
        "tile1_training_samples": len(rows),
        "door_quad_fully_inside_face": len(vis),
        "door_quad_partially_inside": int(sum(1 for r in rows if r["quad_any_inside"] and "_quad_px" not in r)),
        "carport_side_views": len(carport), "interior_side_views": len(interior), "roi_list_views_fully_visible": len(roi),
        "carport_cam_distance_p10_p50_p90_m": [float(np.percentile([r["cam_to_door_centre_m"] for r in carport], p)) for p in (10, 50, 90)] if carport else None,
        "interior_cam_distance_p10_p50_p90_m": [float(np.percentile([r["cam_to_door_centre_m"] for r in interior], p)) for p in (10, 50, 90)] if interior else None,
        "thresholds": {"behind_m": BEHIND_M, "on_m": ON_M, "dilate_px": DILATE_PX},
        "aggregate": {
            f"{grp}_{tag}": agg(sel, tag)
            for grp, sel in (("carport", carport), ("interior", interior), ("roi_list", roi), ("figure5", [by_sample[s] for s in figure_samples if s in by_sample and "_quad_px" in by_sample[s]]))
            for tag in ("vis6", "vis6f")
        },
    }
    json.dump(summary, open(OUT / "stage3_summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
