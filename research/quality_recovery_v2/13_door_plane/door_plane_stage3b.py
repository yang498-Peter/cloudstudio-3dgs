"""Stage 3b: overlays for arbitrary Tile_1 training samples (interior-side views, nearest
carport view) plus, optionally, the LiDAR leaf rectangle from stage 4 projected as a
magenta quad so the LiDAR leaf position can be compared with the photo leaf.

usage: python door_plane_stage3b.py <sample_id> [<sample_id> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-work")
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402

from door_plane_stage1 import read_ply  # noqa: E402
from door_plane_stage3 import BEHIND_M, DATASET, FACE_ROOT, ON_M, OUT, TILE1_PLY, VIS6F, classify_view, quad_pixels  # noqa: E402


def main(samples: list[str]) -> None:
    s2 = json.load(open(OUT / "stage2_door_opening.json", encoding="utf-8"))
    pl = s2["plane_strip_refit"]
    n = np.asarray(pl["normal"])
    d0 = float(pl["d0"])
    e_u, e_w, origin = (np.asarray(pl[k]) for k in ("e_u", "e_w", "origin"))
    op = s2["opening_in_plane"]
    corners = np.asarray([s2["opening_world_corners"][k] for k in ("bottom_left", "bottom_right", "top_right", "top_left")])
    leaf = None
    leaf_path = OUT / "stage4_lidar_leaf.json"
    if leaf_path.exists():
        lj = json.load(open(leaf_path, encoding="utf-8"))
        leaf = np.asarray(lj["leaf_world_corners"])
        leaf_n, leaf_d0 = np.asarray(lj["leaf_plane_normal"]), float(lj["leaf_plane_d0"])
    leaf_stats = {}

    dm = json.load(open(DATASET, encoding="utf-8"))
    c2w_of = {im["image_id"]: np.asarray(im["c2w"], dtype=np.float64) for im in dm["images"]}
    cam_of = {im["image_id"]: im["camera_id"] for im in dm["images"]}
    fm = json.load(open(FACE_ROOT / "face_manifest.json", encoding="utf-8"))
    faces = {cam: {f["face_id"]: FaceSpec.from_dict(f) for f in payload["faces"]} for cam, payload in fm["cameras"].items()}

    a = read_ply(TILE1_PLY)
    T = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    rel = T - origin
    u, w, d = rel @ e_u, rel @ e_w, T @ n + d0
    inner = (u > op["u0"] + 0.03) & (u < op["u1"] - 0.03) & (w > op["w0"] + 0.03) & (w < op["w1"] - 0.03)
    Pin, din = T[inner], d[inner]

    for sample in samples:
        iid, fid = sample.split("::")
        c2w = c2w_of[iid]
        face = faces[cam_of[iid]][fid]
        qp, inside, zf = quad_pixels(face, c2w, corners)
        side = float(c2w[:3, 3] @ n + d0)
        rgb = np.asarray(Image.open(FACE_ROOT / "faces" / f"{iid}_{fid}_rgb.png").convert("RGB"))
        H, W = rgb.shape[:2]
        qc = np.clip(qp, [0, 0], [W - 1, H - 1])
        x0, y0 = qc.min(0)
        x1, y1 = qc.max(0)
        mx, my = max(0.45 * (x1 - x0), 150), max(0.45 * (y1 - y0), 150)
        X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        X1, Y1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
        crop = rgb[Y0:Y1, X0:X1]
        r = classify_view(face, c2w, n, d0, qp, VIS6F, sample) or {}
        pix = r.pop("_pix", None)
        if leaf is not None:
            # LiDAR-cache ranges inside the projected LiDAR-leaf rectangle, against the leaf plane
            lp, lin, lz = quad_pixels(face, c2w, leaf)
            if (lz > 0).all():
                rl = classify_view(face, c2w, leaf_n, leaf_d0, lp, VIS6F, sample) or {}
                rl.pop("_pix", None)
                leaf_stats[sample] = {k: rl.get(k) for k in ("quad_pixels", "n_lidar_in_quad", "n_behind", "n_on_leaf", "n_front", "t_plane_p50", "range_p50", "delta_p50")}
                leaf_stats[sample]["leaf_quad_px"] = np.round(lp, 1).tolist()
                leaf_stats[sample]["opening_quad_px"] = np.round(qp, 1).tolist()

        def panel(title: str):
            im = Image.fromarray(crop.copy())
            dr = ImageDraw.Draw(im)
            dr.polygon([(float(x - X0), float(y - Y0)) for x, y in qp], outline=(255, 255, 0))
            if leaf is not None:
                lp, lin, lz = quad_pixels(face, c2w, leaf)
                if (lz > 0).all():
                    dr.polygon([(float(x - X0), float(y - Y0)) for x, y in lp], outline=(255, 0, 255))
            dr.text((4, 4), title, fill=(255, 255, 0))
            return im, dr

        p1, _ = panel(f"{iid[:12]} {fid}  cam side {side:+.2f} m ({'carport' if side > 0 else 'interior'}); yellow = opening, magenta = LiDAR leaf")
        p2, dr2 = panel("Tile_1 init points in opening footprint (green on plane, red/orange behind = interior side, blue carport side)")
        pc = (Pin - c2w[:3, 3]) @ c2w[:3, :3]
        ppx, pin = face.directions_to_pixels(pc)
        for j in np.argsort(-np.abs(din)):
            if not pin[j]:
                continue
            x, y = ppx[j] - (X0, Y0)
            if not (0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]):
                continue
            dd = din[j]
            col = (0, 255, 0) if abs(dd) <= 0.1 else ((255, int(255 * min(1.0, (-dd - 0.1) / 2.9)), 0) if dd < 0 else (60, 120, 255))
            dr2.point((x, y), fill=col)
        p3, dr3 = panel(f"vis6f cache: red = beyond plane by >{BEHIND_M} m, green = on plane (+-{ON_M}), blue = this side of plane, orange between")
        if pix is not None:
            ui, vi, delta = pix
            for x, y, dd in zip(ui, vi, delta):
                col = (255, 40, 40) if dd > BEHIND_M else ((0, 255, 0) if abs(dd) <= ON_M else ((60, 120, 255) if dd < -BEHIND_M else (255, 160, 0)))
                dr3.point((int(x) - X0, int(y) - Y0), fill=col)
        dr3.text((4, 18), "vis6f n=%d beyond=%d on=%d near=%d | t_plane p50 %.2f range p50 %.2f" % (
            r.get("n_lidar_in_quad", 0), r.get("n_behind", 0), r.get("n_on_leaf", 0), r.get("n_front", 0), r.get("t_plane_p50", float("nan")), r.get("range_p50", float("nan"))), fill=(255, 255, 0))
        Wc, Hc = p1.size
        sheet = Image.new("RGB", (3 * Wc + 8, Hc))
        for i, p in enumerate((p1, p2, p3)):
            sheet.paste(p, (i * (Wc + 4), 0))
        tag = "interior" if side < 0 else "carport"
        sheet.save(OUT / f"overlay_{tag}_{iid[:12]}_{fid}.png")
        print("wrote", tag, sample, "quad fully inside:", bool(inside.all()), {k: v for k, v in r.items() if k in ("n_lidar_in_quad", "n_behind", "n_on_leaf", "n_front", "t_plane_p50", "range_p50")})
        if sample in leaf_stats:
            print("   LiDAR-leaf polygon:", {k: v for k, v in leaf_stats[sample].items() if not k.endswith("_px")})
    if leaf_stats:
        prev = {}
        if (OUT / "stage3b_leaf_stats.json").exists():
            prev = json.load(open(OUT / "stage3b_leaf_stats.json", encoding="utf-8"))
        prev.update(leaf_stats)
        json.dump(prev, open(OUT / "stage3b_leaf_stats.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
