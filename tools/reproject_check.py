#!/usr/bin/env python3
"""Reproject the solver point cloud onto raw fisheye images to validate
calibration + per-image poses + the pose-matrix convention.

This is the Phase 1 hard gate from docs/RESEARCH_PLAN.md: if the projected
points do not hug image edges/structures (<~2px visually), everything
downstream is wasted effort.

The transform_matrix axis convention in transforms.json is undocumented, so we
render one overlay per candidate convention; the correct one is the overlay
where LiDAR points line up with image content. Conventions tested:

    c2w_cv   X_cam = R^T (X_w - t)            camera-to-world, OpenCV axes (x right, y down, z forward)
    c2w_gl   X_cam = F R^T (X_w - t)          camera-to-world, OpenGL axes (F = diag(1,-1,-1))
    w2c_cv   X_cam = R X_w + t                world-to-camera, OpenCV axes
    w2c_gl   X_cam = F (R X_w + t)            world-to-camera, OpenGL axes

Usage:
    python tools/reproject_check.py --run-dir <process/runName> --raw-dir <recording> \
        --out-dir experiments/reproject_gs2 [--frames 3] [--max-points 2000000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

CONVENTIONS = ("c2w_cv", "c2w_gl", "w2c_cv", "w2c_gl")
FLIP = np.diag([1.0, -1.0, -1.0])


def load_points(run_dir: Path, max_points: int) -> np.ndarray:
    """Load xyz from the solver point cloud, stride-subsampled to max_points."""
    import laspy

    for name in ("colorized.las", "uncolorized.las", "colorized.laz", "uncolorized.laz"):
        las_path = run_dir / name
        if las_path.exists():
            break
    else:
        raise FileNotFoundError(f"no LAS/LAZ point cloud in {run_dir}")

    chunks = []
    with laspy.open(las_path) as reader:
        total = reader.header.point_count
        stride = max(1, total // max_points)
        for chunk in reader.chunk_iterator(2_000_000):
            xyz = np.column_stack([chunk.x, chunk.y, chunk.z])[::stride]
            chunks.append(xyz.astype(np.float64))
    pts = np.concatenate(chunks)
    print(f"point cloud: {las_path.name}, {total:,} points -> subsampled {len(pts):,}")
    return pts


def project_kb4(pts_cam: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV fisheye (Kannala-Brandt k1..k4) projection of camera-frame points.

    Uses theta = atan2(sqrt(x^2+y^2), z) so directions beyond 90 deg off-axis
    still project (the lens FoV exceeds 180 deg). Returns (u, v, depth_mask).
    Same model as CloudStudio's process_s1_panoramas.py.
    """
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    rxy = np.sqrt(x * x + y * y)
    theta = np.arctan2(rxy, z)
    t2 = theta * theta
    theta_d = theta * (1 + frame["k1"] * t2 + frame["k2"] * t2**2 + frame["k3"] * t2**3 + frame["k4"] * t2**4)
    scale = np.where(rxy > 1e-9, theta_d / rxy, 0.0)
    u = frame["fl_x"] * x * scale + frame["cx"]
    v = frame["fl_y"] * y * scale + frame["cy"]
    # keep points in front hemisphere-ish and within a sane distance
    dist = np.sqrt(x * x + y * y + z * z)
    ok = (theta < np.radians(100)) & (dist > 0.3) & (dist < 60.0)
    return u, v, ok


def cam_points(pts_w: np.ndarray, mat: np.ndarray, convention: str) -> np.ndarray:
    r, t = mat[:3, :3], mat[:3, 3]
    if convention.startswith("c2w"):
        pc = (pts_w - t) @ r  # == R^T @ (X - t) row-wise
    else:
        pc = pts_w @ r.T + t
    if convention.endswith("_gl"):
        pc = pc @ FLIP
    return pc


def render_overlay(img_path: Path, pts_w: np.ndarray, frame: dict, convention: str, out_path: Path) -> float:
    mat = np.asarray(frame["transform_matrix"], dtype=np.float64)
    pc = cam_points(pts_w, mat, convention)
    u, v, ok = project_kb4(pc, frame)
    w, h = int(frame["w"]), int(frame["h"])
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    frac = float(ok.mean())

    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) * 0.35
    ui, vi = u[ok].astype(np.int32), v[ok].astype(np.int32)
    depth = np.linalg.norm(pc[ok], axis=1)
    # z-buffer per pixel so far points don't paint over near ones
    order = np.argsort(-depth)
    ui, vi, depth = ui[order], vi[order], depth[order]
    t = np.clip(depth / 15.0, 0, 1)[:, None]
    color = np.hstack([255 * (1 - t), 255 * np.minimum(2 * t, 2 - 2 * t), 255 * t])  # near red -> far blue
    img[vi, ui] = color
    # thicken dots 1px right/down for visibility at full res
    img[np.minimum(vi + 1, h - 1), ui] = color
    img[vi, np.minimum(ui + 1, w - 1)] = color

    out = Image.fromarray(img.astype(np.uint8)).resize((w // 2, h // 2), Image.BILINEAR)
    out.save(out_path, quality=90)
    return frac


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=3, help="frames per camera side to test")
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--conventions", nargs="*", default=list(CONVENTIONS), choices=CONVENTIONS)
    args = ap.parse_args()

    tf = json.loads((args.run_dir / "transforms.json").read_text(encoding="utf-8"))
    frames = tf["frames"]
    by_side = {"left": [], "right": []}
    for f in frames:
        side = str(f["file_path"]).replace("\\", "/").split("/")[0]
        if side in by_side:
            by_side[side].append(f)

    pts_w = load_points(args.run_dir, args.max_points)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for side, side_frames in by_side.items():
        if not side_frames:
            continue
        picks = [side_frames[i] for i in np.linspace(0, len(side_frames) - 1, min(args.frames, len(side_frames)), dtype=int)]
        for f in picks:
            rel = str(f["file_path"]).replace("\\", "/")
            img_path = args.raw_dir / "camera" / rel
            if not img_path.exists():
                print(f"SKIP missing image {img_path}")
                continue
            stem = Path(rel).stem[-8:]
            for conv in args.conventions:
                out_path = args.out_dir / f"{side}_{stem}_{conv}.jpg"
                frac = render_overlay(img_path, pts_w, f, conv, out_path)
                report.append((side, stem, conv, frac))
                print(f"{side}/{stem} {conv:7s} in-view {frac:6.1%} -> {out_path.name}")

    print("\nPer-convention mean in-view fraction (higher usually = plausible convention,")
    print("but the DECISIVE test is visual alignment in the overlays):")
    for conv in args.conventions:
        vals = [r[3] for r in report if r[2] == conv]
        if vals:
            print(f"  {conv:7s} {np.mean(vals):6.1%}")
    print(f"\nInspect the overlays in {args.out_dir} — points must hug edges/structures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
