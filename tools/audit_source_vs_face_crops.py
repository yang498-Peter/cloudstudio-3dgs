#!/usr/bin/env python3
"""Source photo vs Face4 cache crops for one physical feature (WP03).

For every feature (a 3D point on a physical edge, e.g. a door frame) the tool
picks N parent photos that see it (Face4 mask valid, not occluded by the vis6
LiDAR geometry) and cuts the same feature from three representations:

  a) the ORIGINAL fisheye JPEG (PIL decode, ``convert("RGB")``), a physically
     matched window (same metres on the surface as the face crop),
  b) a FRESH KB4 -> pinhole remap computed here from the face plan in the face
     manifest (K_face / R_face, pixel centres at i + 0.5) using the repo KB4
     forward model ``cloudstudio_3dgs.geometry.kb4.project_kb4``; sampled twice,
     ``b_bilinear`` (own bilinear gather, same convention as
     ``cloudstudio_3dgs/data/face_warp.py``) and ``b_cubic`` (scipy cubic
     spline, to show what a higher-order single pass would give),
  c) the face cache PNG as stored (``faces/<image>_<face>_rgb.png``).

Per crop it measures the 10-90 % rise width across the strongest edges, the
noise (std of the flattest 16 px tile), and the saturation fraction, and it
records the decode / colour / resampling facts of each path. ``b_bilinear``
minus ``c`` documents whether the stored cache is the one-pass bilinear warp
the trainer assumes (up to uint8 rounding).

Selection of parent images, projection and occlusion reuse
``tools/audit_observation_coverage.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, JpegImagePlugin, features as pil_features

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from audit_observation_coverage import (  # noqa: E402
    STATUS_OCCLUDED,
    STATUS_SUPPORTED,
    CameraModel,
    ManifestLoaders,
    build_views,
    classify_support,
    face_footprint_px_per_m,
    fisheye_footprint_px_per_m,
    pixel_centre_to_index,
    world_to_camera,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402
from cloudstudio_3dgs.geometry.kb4 import project_kb4  # noqa: E402
from cloudstudio_3dgs.geometry.lidar_projection import DepthProjectionConfig  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114])


# ----------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------


def bilinear_gather(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Own bilinear gather in array-index coordinates (x = column, y = row)."""
    h, w = image.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    wx = np.clip(x - x0c, 0.0, 1.0)[..., None]
    wy = np.clip(y - y0c, 0.0, 1.0)[..., None]
    img = image.astype(np.float64)
    top = img[y0c, x0c] * (1 - wx) + img[y0c, x0c + 1] * wx
    bot = img[y0c + 1, x0c] * (1 - wx) + img[y0c + 1, x0c + 1] * wx
    return top * (1 - wy) + bot * wy


def cubic_gather(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    out = np.empty(x.shape + (image.shape[2],), dtype=np.float64)
    for c in range(image.shape[2]):
        out[..., c] = map_coordinates(image[..., c].astype(np.float64), [y, x], order=3, mode="nearest")
    return out


def remap_face_window(
    photo_rgb: np.ndarray,
    camera: CameraModel,
    face: FaceSpec,
    u0: int,
    v0: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fresh KB4->pinhole remap of the face window [v0:v0+size, u0:u0+size].

    Returns (bilinear, cubic, source_xy) where source_xy are the fisheye
    array-index coordinates that each face pixel centre maps to.
    """
    jj, ii = np.meshgrid(np.arange(u0, u0 + size) + 0.5, np.arange(v0, v0 + size) + 0.5)
    dirs_face = np.stack([(jj - face.K_face[0, 2]) / face.K_face[0, 0], (ii - face.K_face[1, 2]) / face.K_face[1, 1], np.ones_like(jj)], -1)
    dirs_cam = dirs_face.reshape(-1, 3) @ face.R_face.T
    uv, _, _ = project_kb4(dirs_cam, camera.intrinsic, camera.distortion)
    # pixel-centre coordinate -> array index (the face cache does the same, face_warp.py)
    x = (uv[:, 0] - 0.5).reshape(size, size)
    y = (uv[:, 1] - 0.5).reshape(size, size)
    return bilinear_gather(photo_rgb, x, y), cubic_gather(photo_rgb, x, y), np.stack([x, y], -1)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ LUMA


def _profile(gray: np.ndarray, cx: float, cy: float, nx: float, ny: float, half: float = 12.0, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(-half, half + step / 2, step)
    x = cx + t * nx
    y = cy + t * ny
    vals = bilinear_gather(gray[..., None], x, y)[..., 0]
    return t, vals


def edge_width_10_90(gray: np.ndarray, *, candidates: int = 5, border: int = 14, min_contrast: float = 20.0) -> dict[str, float]:
    """Median 10-90 % rise distance (px) across the strongest edges in the crop."""
    g = np.asarray(gray, dtype=np.float64)
    k = np.ones((3, 3)) / 9.0
    gs = g.copy()
    gs[1:-1, 1:-1] = sum(g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx] * k[dy + 1, dx + 1] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    gy, gx = np.gradient(gs)
    gm = np.hypot(gx, gy)
    gm[:border, :] = 0
    gm[-border:, :] = 0
    gm[:, :border] = 0
    gm[:, -border:] = 0
    widths: list[float] = []
    contrasts: list[float] = []
    taken: list[tuple[int, int]] = []
    order = np.argsort(gm, axis=None)[::-1]
    for flat in order[: 4000]:
        if len(widths) >= candidates or gm.flat[flat] <= 0:
            break
        y, x = np.unravel_index(flat, gm.shape)
        if any(abs(y - ty) < 16 and abs(x - tx) < 16 for ty, tx in taken):
            continue
        nx, ny = gx[y, x] / gm[y, x], gy[y, x] / gm[y, x]
        t, p = _profile(g, float(x), float(y), nx, ny)
        lo = float(np.mean(p[t <= -8]))
        hi = float(np.mean(p[t >= 8]))
        if abs(hi - lo) < min_contrast:
            taken.append((y, x))
            continue
        if hi < lo:  # orient so that the profile rises
            p = p[::-1]
            lo, hi = hi, lo
        delta = hi - lo
        above10 = np.flatnonzero(p >= lo + 0.1 * delta)
        above90 = np.flatnonzero(p >= lo + 0.9 * delta)
        if above10.size == 0 or above90.size == 0:
            taken.append((y, x))
            continue
        # last time the profile is below 10 % before the crossing, first time above 90 %
        t10 = t[above10[0]]
        t90 = t[above90[0]]
        widths.append(float(abs(t90 - t10)))
        contrasts.append(float(delta))
        taken.append((y, x))
    if not widths:
        return {"edge_width_px": float("nan"), "edge_contrast": float("nan"), "edge_count": 0}
    return {"edge_width_px": float(np.median(widths)), "edge_contrast": float(np.median(contrasts)), "edge_count": len(widths)}


def flat_patch_noise(gray: np.ndarray, tile: int = 16) -> dict[str, float]:
    g = np.asarray(gray, dtype=np.float64)
    gy, gx = np.gradient(g)
    energy = gx * gx + gy * gy
    h, w = g.shape
    best = None
    for y in range(0, h - tile + 1, tile // 2):
        for x in range(0, w - tile + 1, tile // 2):
            e = float(energy[y:y + tile, x:x + tile].mean())
            if best is None or e < best[0]:
                best = (e, y, x)
    _, y, x = best
    patch = g[y:y + tile, x:x + tile]
    # remove a fitted plane so a smooth gradient is not counted as noise
    yy, xx = np.mgrid[0:tile, 0:tile]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(tile * tile)])
    coef, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    resid = patch.ravel() - A @ coef
    return {"noise_std": float(resid.std()), "noise_patch_mean": float(patch.mean()), "noise_patch_xy": f"{x},{y}"}


def saturation(rgb: np.ndarray) -> dict[str, float]:
    a = np.asarray(rgb, dtype=np.float64)
    return {
        "sat_hi_frac": float((a.max(axis=-1) >= 250.0).mean()),
        "sat_lo_frac": float((a.max(axis=-1) <= 5.0).mean()),
        "mean_luma": float(to_gray(a).mean()),
    }


def anchor_edge_offset_px(gray: np.ndarray, anchor_xy: tuple[float, float], *, radius: int = 40, rel_threshold: float = 0.5) -> float:
    """Distance (px) from the projected LiDAR anchor to the nearest strong edge.

    The anchor is a LiDAR point *on* a physical edge, so with perfect camera /
    LiDAR / pose alignment a strong image edge passes through it. The offset
    is a per-image reprojection-misalignment proxy (also large when the object
    moved between passes). Uses gradient magnitude >= rel_threshold x the
    local maximum inside ``radius``.
    """
    g = np.asarray(gray, dtype=np.float64)
    gy, gx = np.gradient(g)
    gm = np.hypot(gx, gy)
    ax, ay = anchor_xy
    h, w = g.shape
    y0, y1 = max(int(ay) - radius, 1), min(int(ay) + radius + 1, h - 1)
    x0, x1 = max(int(ax) - radius, 1), min(int(ax) + radius + 1, w - 1)
    win = gm[y0:y1, x0:x1]
    if win.size == 0 or win.max() <= 0:
        return float("nan")
    ys, xs = np.nonzero(win >= rel_threshold * win.max())
    return float(np.min(np.hypot(xs + x0 - ax, ys + y0 - ay)))


def crop_metrics(rgb: np.ndarray, anchor_xy: tuple[float, float] | None = None) -> dict[str, float]:
    gray = to_gray(rgb)
    out: dict[str, float] = {}
    out.update(edge_width_10_90(gray))
    out.update(flat_patch_noise(gray))
    out.update(saturation(rgb))
    if anchor_xy is not None:
        out["anchor_edge_offset_px"] = anchor_edge_offset_px(gray, anchor_xy)
    return out


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------


def select_parent_images(
    point: np.ndarray,
    views,
    cameras,
    faces_by_camera,
    loaders: ManifestLoaders,
    cfg: DepthProjectionConfig,
    *,
    count: int,
) -> list[dict[str, Any]]:
    candidates = []
    for view in views:
        camera = cameras[view.camera_id]
        pc = world_to_camera(point[None, :], view.c2w)
        uv, ranges, valid = camera.project(pc, min_range_m=cfg.min_range_m, max_range_m=cfg.max_range_m)
        if not valid[0]:
            continue
        best_face = None
        for face in faces_by_camera[view.camera_id]:
            pix, inside = face.directions_to_pixels(pc)
            if not inside[0]:
                continue
            z_face = float((pc @ face.R_face)[0, 2] / np.linalg.norm(pc[0]))
            px = int(np.clip(pixel_centre_to_index(pix[0, 0]), 0, face.width - 1))
            py = int(np.clip(pixel_centre_to_index(pix[0, 1]), 0, face.height - 1))
            mask = loaders.face_mask(view.image_id, face.face_id)
            if mask is not None and not mask[py, px]:
                continue
            dense = loaders.face_geometry(view.image_id, face.face_id)
            status = int(classify_support(dense, np.array([px]), np.array([py]), ranges[:1],
                                          tolerance=cfg.visibility_tolerance, margin_m=cfg.visibility_margin_m)[0]) if dense is not None else 0
            if status == STATUS_OCCLUDED:
                continue
            if best_face is None or z_face > best_face["z_face"]:
                best_face = {"face": face, "px": px, "py": py, "status": status, "z_face": z_face,
                             "pix": (float(pix[0, 0]), float(pix[0, 1]))}
        if best_face is None:
            continue
        candidates.append({
            "view": view, "range_m": float(ranges[0]), "uv": (float(uv[0, 0]), float(uv[0, 1])),
            "foot_fish": float(fisheye_footprint_px_per_m(camera, pc)[0]),
            "foot_face": float(face_footprint_px_per_m(best_face["face"], pc)[0]),
            **best_face,
        })
    # supported first, then closest; distinct rig frames; alternate cameras when possible
    candidates.sort(key=lambda c: (0 if c["status"] == STATUS_SUPPORTED else 1, c["range_m"]))
    chosen: list[dict[str, Any]] = []
    used_frames: set[str] = set()
    want_cam = None
    pool = list(candidates)
    while pool and len(chosen) < count:
        pick = None
        for c in pool:
            if c["view"].rig_frame_id in used_frames:
                continue
            if want_cam is None or c["view"].camera_id == want_cam:
                pick = c
                break
        if pick is None:
            for c in pool:
                if c["view"].rig_frame_id not in used_frames:
                    pick = c
                    break
        if pick is None:
            break
        chosen.append(pick)
        used_frames.add(pick["view"].rig_frame_id)
        pool.remove(pick)
        want_cam = "right" if pick["view"].camera_id == "left" else "left"
    return chosen, len(candidates)


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def side_by_side(panels: Sequence[tuple[str, np.ndarray]], slot: int, path: Path) -> None:
    pad = 6
    label_h = 16
    canvas = Image.new("RGB", (len(panels) * (slot + pad) + pad, slot + label_h + 2 * pad), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    for i, (label, arr) in enumerate(panels):
        img = Image.fromarray(_to_uint8(arr))
        x = pad + i * (slot + pad)
        ox = x + max((slot - img.width) // 2, 0)
        oy = pad + label_h + max((slot - img.height) // 2, 0)
        if img.width > slot or img.height > slot:  # never resample: crop the centre for display
            left = max((img.width - slot) // 2, 0)
            top = max((img.height - slot) // 2, 0)
            img = img.crop((left, top, left + min(slot, img.width), top + min(slot, img.height)))
            ox, oy = x, pad + label_h
        canvas.paste(img, (ox, oy))
        draw.text((x, pad), label, fill=(240, 240, 240))
    canvas.save(path, format="PNG")


def decode_facts(photo_path: Path, face_png: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {"pil_version": Image.__version__, "jpeg_codec": pil_features.version_codec("jpg"),
                             "libjpeg_turbo": bool(pil_features.check_feature("libjpeg_turbo"))}
    with Image.open(photo_path) as im:
        facts.update({
            "photo_mode": im.mode, "photo_size": list(im.size),
            "photo_chroma_subsampling": {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(JpegImagePlugin.get_sampling(im), str(JpegImagePlugin.get_sampling(im))),
            "photo_has_icc_profile": "icc_profile" in im.info,
            "photo_exif_bytes": len(im.info.get("exif", b"")),
            "photo_exif_orientation": im.getexif().get(274),
            "photo_jfif": im.info.get("jfif_version"),
            "photo_luma_quant_row0": list(im.quantization[0])[:8] if getattr(im, "quantization", None) else None,
        })
    with Image.open(face_png) as im:
        facts.update({"face_png_mode": im.mode, "face_png_size": list(im.size), "face_png_has_icc": "icc_profile" in im.info,
                      "face_png_has_gamma": "gamma" in im.info})
    return facts


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--lidar-geometry-manifest", type=Path, default=None)
    parser.add_argument("--recording-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path, help="JSON: {features:[{label, region, point_world, ...}]}")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--images-per-feature", type=int, default=6)
    parser.add_argument("--crop", type=int, default=256)
    args = parser.parse_args(argv)

    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    face_manifest = json.loads(args.face_manifest.read_text(encoding="utf-8"))
    geometry_manifest = json.loads(args.lidar_geometry_manifest.read_text(encoding="utf-8")) if args.lidar_geometry_manifest else None
    pc = (geometry_manifest or {}).get("projection_config", {})
    cfg = DepthProjectionConfig(
        min_range_m=float(pc.get("min_range_m", 0.2)), max_range_m=float(pc.get("max_range_m", 80.0)),
        max_theta_deg=float(pc.get("max_theta_deg", 95.0)), visibility_cell_px=int(pc.get("visibility_cell_px", 6)),
        visibility_tolerance=float(pc.get("visibility_tolerance", 0.2)), visibility_margin_m=float(pc.get("visibility_margin_m", 0.1)),
    )
    cameras = {c["camera_id"]: CameraModel.from_manifest(c, max_theta_deg=cfg.max_theta_deg) for c in dataset_manifest["cameras"]}
    faces_by_camera = {cam: [FaceSpec.from_dict(f) for f in payload["faces"]] for cam, payload in face_manifest["cameras"].items()}
    views = build_views(face_manifest, dataset_manifest, warn=lambda m: print("WARNING:", m, file=sys.stderr))
    loaders = ManifestLoaders(face_manifest, args.face_manifest.parent, geometry_manifest,
                              args.lidar_geometry_manifest.parent if args.lidar_geometry_manifest else None,
                              dataset_manifest, args.recording_root)
    face_records = loaders.face_records
    features = json.loads(args.features.read_text(encoding="utf-8"))["features"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    facts_written = None
    half = args.crop // 2
    for feature in features:
        point = np.asarray(feature["point_world"], dtype=np.float64)
        chosen, n_candidates = select_parent_images(point, views, cameras, faces_by_camera, loaders, cfg, count=args.images_per_feature)
        print(f"[{feature['label']}] {n_candidates} candidate views, using {len(chosen)}", file=sys.stderr, flush=True)
        for pick in chosen:
            view = pick["view"]
            face: FaceSpec = pick["face"]
            camera = cameras[view.camera_id]
            photo_path = loaders.photo_paths[view.image_id]
            with Image.open(photo_path) as im:
                photo_rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
            face_png = args.face_manifest.parent / face_records[(view.image_id, face.face_id)]["rgb_path"]
            with Image.open(face_png) as im:
                face_rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
            if facts_written is None:
                facts_written = decode_facts(photo_path, face_png)
                facts_written["face_plan"] = {cam: [f.to_dict() for f in fl] for cam, fl in faces_by_camera.items()}
                (args.out_dir / "decode_and_resampling_facts.json").write_text(json.dumps(facts_written, indent=2), encoding="utf-8")
            # face window (clamped to the raster)
            u0 = int(np.clip(pick["px"] - half, 0, face.width - args.crop))
            v0 = int(np.clip(pick["py"] - half, 0, face.height - args.crop))
            c_crop = face_rgb[v0:v0 + args.crop, u0:u0 + args.crop]
            b_bil, b_cub, src_xy = remap_face_window(photo_rgb, camera, face, u0, v0, args.crop)
            # physically matched source window: same metres as the face crop
            ratio = pick["foot_fish"] / pick["foot_face"]
            a_size = max(int(round(args.crop * ratio)), 16)
            ax = int(np.clip(pixel_centre_to_index(pick["uv"][0]) - a_size // 2, 0, camera.width - a_size))
            ay = int(np.clip(pixel_centre_to_index(pick["uv"][1]) - a_size // 2, 0, camera.height - a_size))
            a_crop = photo_rgb[ay:ay + a_size, ax:ax + a_size]
            diff = np.abs(b_bil - c_crop)
            # how many source pixels each face pixel spans (local magnification)
            dx = np.diff(src_xy[..., 0], axis=1)
            dy = np.diff(src_xy[..., 1], axis=0)
            src_step = float(np.median(np.hypot(dx[:-1, :], dy[:, :-1])))
            stem = f"{feature['region']}__{feature['label']}__{view.image_id[:18]}_{view.camera_id}_{face.face_id}"
            side_by_side(
                [("a original (phys. matched)", a_crop), ("b fresh remap bilinear", b_bil), ("b fresh remap cubic", b_cub),
                 ("c face_cache PNG", c_crop), ("|b_bilinear - c| x16", np.clip(diff.mean(-1, keepdims=True).repeat(3, -1) * 16, 0, 255))],
                args.crop, args.out_dir / f"{stem}.png",
            )
            base = {
                "region": feature["region"], "feature": feature["label"], "point_world": " ".join(f"{v:.3f}" for v in point),
                "image_id": view.image_id, "camera": view.camera_id, "face": face.face_id, "timestamp_ns": view.timestamp_ns,
                "range_m": round(pick["range_m"], 3), "lidar_status": {1: "supported", 0: "no_lidar", 3: "lidar_behind"}.get(pick["status"], str(pick["status"])),
                "px_per_m_fisheye": round(pick["foot_fish"], 2), "px_per_m_face": round(pick["foot_face"], 2),
                "face_px_per_source_px": round(1.0 / src_step, 3) if src_step > 0 else float("nan"),
                "fisheye_px": f"{pick['uv'][0]:.1f},{pick['uv'][1]:.1f}", "face_px": f"{pick['pix'][0]:.1f},{pick['pix'][1]:.1f}",
                "face_window_u0v0": f"{u0},{v0}", "source_window_xy_size": f"{ax},{ay},{a_size}",
                "b_minus_c_max_abs": round(float(diff.max()), 3), "b_minus_c_mean_abs": round(float(diff.mean()), 4),
                "b_minus_c_frac_gt1": round(float((diff.max(-1) > 1.0).mean()), 5),
                "png": stem + ".png",
            }
            face_anchor = (pick["pix"][0] - 0.5 - u0, pick["pix"][1] - 0.5 - v0)
            fish_anchor = (pick["uv"][0] - 0.5 - ax, pick["uv"][1] - 0.5 - ay)
            for tag, arr, ppm, anchor in (("a", a_crop, pick["foot_fish"], fish_anchor), ("b_bilinear", b_bil, pick["foot_face"], face_anchor),
                                          ("b_cubic", b_cub, pick["foot_face"], face_anchor), ("c", c_crop, pick["foot_face"], face_anchor)):
                m = crop_metrics(arr, anchor)
                m["anchor_edge_offset_mm"] = m["anchor_edge_offset_px"] / ppm * 1000.0 if np.isfinite(m["anchor_edge_offset_px"]) else float("nan")
                row = dict(base)
                row.update({"variant": tag, "crop_size_px": arr.shape[0], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
                row["edge_width_mm"] = round(m["edge_width_px"] / ppm * 1000.0, 2) if np.isfinite(m["edge_width_px"]) else float("nan")
                rows.append(row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
