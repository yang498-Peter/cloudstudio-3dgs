#!/usr/bin/env python3
"""Generate circular valid-region masks for the S1 fisheye images in a
COLMAP dataset (the square frames have black corners outside the lens image
circle; unmasked they poison 3DGS training with fake black supervision).

Mask radius per camera = min(
    analytic:  f * theta_d(theta_max)   (KB4 forward model, FoV cap),
    detected:  radial brightness falloff of the real image circle * margin
)

Output: <dataset>/masks/<image_name>.png (8-bit, 255 = valid pixel). One mask
is computed per COLMAP camera and replicated for its images (S1: left/right).

Usage:
    python tools/make_fisheye_masks.py --dataset G:\\3dgs-datasets\\gs2_keyframes \
        [--theta-max-deg 95] [--samples 8] [--margin 0.985]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def kb4_radius_px(params: list[float], theta: float) -> float:
    fx, fy, _cx, _cy, k1, k2, k3, k4 = params
    t2 = theta * theta
    theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    return 0.5 * (fx + fy) * theta_d


def detect_image_circle(img_paths: list[Path], cx: float, cy: float, w: int, h: int) -> float:
    """Estimate the lens image-circle radius from radial brightness falloff."""
    scale = 4
    acc = None
    for p in img_paths:
        im = np.asarray(Image.open(p).convert("L").resize((w // scale, h // scale)), dtype=np.float32)
        acc = im if acc is None else acc + im
    acc /= len(img_paths)
    yy, xx = np.mgrid[0 : h // scale, 0 : w // scale]
    r = np.sqrt((xx - cx / scale) ** 2 + (yy - cy / scale) ** 2).astype(np.int32)
    r_max = r.max()
    profile = np.bincount(r.ravel(), weights=acc.ravel(), minlength=r_max + 1)
    counts = np.bincount(r.ravel(), minlength=r_max + 1)
    profile = profile / np.maximum(counts, 1)
    # walk outward from mid-radius; circle edge = first radius whose local mean
    # stays below threshold (black surround is ~0-4, scene content is >>10)
    thresh = 6.0
    start = int(0.3 * r_max)
    below = np.where(profile[start:] < thresh)[0]
    edge = (start + below[0]) if len(below) else r_max
    return float(edge * scale)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--theta-max-deg", type=float, default=95.0,
                    help="half-angle FoV cap in degrees (95 => keep up to 190 deg full FoV; "
                         "use 80 for the 160-deg ablation)")
    ap.add_argument("--samples", type=int, default=8, help="images sampled per camera for circle detection")
    ap.add_argument("--margin", type=float, default=0.985, help="shrink factor on the detected circle")
    args = ap.parse_args()

    sparse = args.dataset / "sparse" / "0"
    cams: dict[int, list[float]] = {}
    sizes: dict[int, tuple[int, int]] = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cams[int(p[0])] = [float(v) for v in p[4:]]
        sizes[int(p[0])] = (int(p[2]), int(p[3]))

    by_cam: dict[int, list[str]] = {c: [] for c in cams}
    lines = (sparse / "images.txt").read_text().splitlines()
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) >= 10:
            by_cam[int(p[8])].append(p[9])

    mask_dir = args.dataset / "masks"
    mask_dir.mkdir(exist_ok=True)
    theta_max = np.radians(args.theta_max_deg)

    for cam_id, params in cams.items():
        w, h = sizes[cam_id]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        names = by_cam[cam_id]
        sample = [args.dataset / "images" / n
                  for n in np.array(names)[np.linspace(0, len(names) - 1, min(args.samples, len(names)), dtype=int)]]
        r_analytic = kb4_radius_px(params, theta_max)
        r_detected = detect_image_circle(sample, cx, cy, w, h) * args.margin
        r_mask = min(r_analytic, r_detected)
        theta_eff = np.degrees(r_mask / kb4_radius_px(params, 1.0))  # rough linear invert for report only
        print(f"camera {cam_id}: analytic r={r_analytic:.0f}px (theta_max {args.theta_max_deg} deg), "
              f"detected circle r={r_detected:.0f}px -> mask r={r_mask:.0f}px")

        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r_mask * r_mask).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        for n in names:
            mask_img.save(mask_dir / f"{Path(n).stem}.png")
        print(f"  wrote {len(names)} masks ({w}x{h})")

    print(f"masks -> {mask_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
