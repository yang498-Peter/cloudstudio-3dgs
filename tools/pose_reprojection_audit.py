"""Measure pose error in pixels, by patch correlation, without the hloc stack.

The repository has a full rig bundle adjustment pipeline (hloc + LightGlue +
ALIKED + pycolmap) with tests and gs2 baselines, and it has never been run on
this dataset - every ukgs image still carries pose_source "ImgPose.txt", raw
SLAM. But none of those three packages is installed on this machine, so the
question "how wrong are the poses" cannot be answered by running it today.

It can be answered directly. Take a textured patch in one image, use the LiDAR
range map to lift its centre to 3D, project that point into a second image using
the poses under test, and then search the neighbourhood for where the patch
actually matches. The offset between where the pose says it should land and
where it does is the pose error, in pixels, which is the same residual BA
minimises.

This succeeds where two earlier attempts failed. Those compared single-pixel
colours across many views, which is swamped by lighting and view-dependent
shading - both were insensitive even to a deliberate 0.4 deg rotation. A patch
carries structure, and normalised cross-correlation is invariant to brightness
and contrast, so the signal survives the things that drowned the earlier tests.

Reference points, for reading the result: gs2's real BA moved reprojection p50
from 1.59 px to 1.08 px. A gravel stone here is about 5 px across, and at
fl=778 px/rad one degree of rotation is 13.6 px.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Peter\cloudstudio-3dgs-gate1")
sys.path.insert(0, str(ROOT))
from cloudstudio_3dgs.training.dataset import S1TrainingDataset

PATCH = 31           # odd, so the patch has a centre pixel
SEARCH = 8           # +/- pixels searched around the predicted landing point
MIN_TEXTURE = 0.02   # reject flat patches: correlation is meaningless on them
MIN_PEAK = 0.7       # below this the match is not trustworthy; skip, do not guess
MAX_PAIRS = 40
RNG = np.random.default_rng(0)

# Published comparison points, so a result is readable without hunting for them.
GS2_BEFORE_BA_PX = 1.59
GS2_AFTER_BA_PX = 1.08


def fisheye_project(points_cam, K, dist):
    z = points_cam[..., 2]
    valid = z > 1e-6
    safe = np.where(valid, z, 1.0)
    x, y = points_cam[..., 0] / safe, points_cam[..., 1] / safe
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    t2 = theta * theta
    k1, k2, k3, k4 = dist
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(r > 1e-9, theta_d / np.where(r > 1e-9, r, 1.0), 1.0)
    return K[0, 0] * x * scale + K[0, 2], K[1, 1] * y * scale + K[1, 2], valid


def fisheye_unproject(u, v, K, dist, ray_range):
    """Pixel + measured range -> camera-frame 3D point (KB4 inverse by Newton)."""
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    theta_d = np.sqrt(x * x + y * y)
    theta = theta_d.copy()
    k1, k2, k3, k4 = dist
    for _ in range(10):
        t2 = theta * theta
        f = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4) - theta_d
        d = 1 + 3 * k1 * t2 + 5 * k2 * t2**2 + 7 * k3 * t2**3 + 9 * k4 * t2**4
        theta = theta - f / np.maximum(d, 1e-9)
    sin_theta = np.sin(theta)
    scale = np.where(theta_d > 1e-9, sin_theta / np.maximum(theta_d, 1e-9), 1.0)
    direction = np.stack([x * scale, y * scale, np.cos(theta)], axis=-1)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    return direction * ray_range[..., None]


def ncc(patch, window):
    """Normalised cross-correlation of one patch against every shift in window."""
    p = patch - patch.mean()
    p_norm = np.sqrt((p * p).sum())
    if p_norm < 1e-8:
        return None
    size = patch.shape[0]
    span = window.shape[0] - size + 1
    scores = np.full((span, span), -np.inf)
    for dy in range(span):
        for dx in range(span):
            w = window[dy:dy + size, dx:dx + size]
            w = w - w.mean()
            denominator = p_norm * np.sqrt((w * w).sum())
            if denominator > 1e-8:
                scores[dy, dx] = float((p * w).sum() / denominator)
    return scores


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, type=Path,
                        help="a trainer config; supplies the dataset and the "
                             "evaluation directory holding the LiDAR range maps")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ds = S1TrainingDataset(
        dataset_manifest_path=Path(cfg["dataset_manifest"]),
        recording_root=Path(cfg["recording_root"]),
        mask_manifest_path=Path(cfg["mask_manifest"]),
        mask_root=Path(cfg["mask_root"]),
        split_manifest_path=Path(cfg["split_manifest"]),
        split="val", factor=cfg["factor"], crop=None,
    )
    eval_dir = Path(cfg["output_dir"]) / "evaluation"

    frames = []
    for i in range(len(ds)):
        s = ds[i]
        path = eval_dir / f"{s.image_id}_range.npy"
        if not path.exists():
            continue
        frames.append((s, np.asarray(s.image, dtype=np.float32).mean(axis=2) / 255.0,
                       np.asarray(s.rgb_mask, dtype=bool), np.load(path)))
        if len(frames) >= args.max_frames:
            break
    print(f"frames with range maps: {len(frames)}", flush=True)

    half = PATCH // 2
    offsets, peaks = [], []
    pairs_done = 0
    for a_index in range(len(frames)):
        if pairs_done >= MAX_PAIRS:
            break
        sa, grey_a, mask_a, range_a = frames[a_index]
        Ka = np.asarray(sa.K, dtype=np.float64)
        dist_a = np.asarray(sa.radial_coeffs, dtype=np.float64).ravel()[:4]
        c2w_a = np.asarray(sa.c2w, dtype=np.float64)

        for b_index in range(a_index + 1, min(a_index + 4, len(frames))):
            if pairs_done >= MAX_PAIRS:
                break
            sb, grey_b, mask_b, _ = frames[b_index]
            Kb = np.asarray(sb.K, dtype=np.float64)
            dist_b = np.asarray(sb.radial_coeffs, dtype=np.float64).ravel()[:4]
            w2c_b = np.linalg.inv(np.asarray(sb.c2w, dtype=np.float64))

            height, width = grey_a.shape
            found = 0
            for _ in range(200):
                v = int(RNG.integers(half + SEARCH, height - half - SEARCH))
                u = int(RNG.integers(half + SEARCH, width - half - SEARCH))
                if not mask_a[v, u]:
                    continue
                depth = range_a[v, u]
                if not np.isfinite(depth) or depth <= 0:
                    continue
                patch = grey_a[v - half:v + half + 1, u - half:u + half + 1]
                if patch.std() < MIN_TEXTURE:
                    continue

                point_cam = fisheye_unproject(np.array([float(u)]), np.array([float(v)]),
                                              Ka, dist_a, np.array([float(depth)]))[0]
                world = c2w_a[:3, :3] @ point_cam + c2w_a[:3, 3]
                in_b = w2c_b[:3, :3] @ world + w2c_b[:3, 3]
                ub, vb, ok = fisheye_project(in_b[None, :], Kb, dist_b)
                if not ok[0]:
                    continue
                ub, vb = float(ub[0]), float(vb[0])
                ui, vi = int(round(ub)), int(round(vb))
                if not (half + SEARCH <= ui < width - half - SEARCH
                        and half + SEARCH <= vi < height - half - SEARCH):
                    continue
                if not mask_b[vi, ui]:
                    continue

                window = grey_b[vi - half - SEARCH:vi + half + SEARCH + 1,
                                ui - half - SEARCH:ui + half + SEARCH + 1]
                scores = ncc(patch, window)
                if scores is None:
                    continue
                best = np.unravel_index(np.argmax(scores), scores.shape)
                peak = float(scores[best])
                if peak < MIN_PEAK:     # no trustworthy match; skip rather than guess
                    continue
                dy = best[0] - SEARCH + (vi - vb)
                dx = best[1] - SEARCH + (ui - ub)
                offsets.append(np.hypot(dx, dy))
                peaks.append(peak)
                found += 1
                if found >= 12:
                    break
            if found:
                pairs_done += 1

    if not offsets:
        print("no confident matches found", flush=True)
        return 1

    offsets = np.asarray(offsets)
    print(f"\nconfident patch matches: {len(offsets)} over {pairs_done} image pairs")
    print(f"mean NCC peak: {np.mean(peaks):.3f}")
    print("\nreprojection offset (pixels) - where the pose says the patch lands "
          "vs where it does")
    for p in (25, 50, 75, 90, 95):
        print(f"  p{p:<3} {np.percentile(offsets, p):>7.2f} px")
    p50 = float(np.percentile(offsets, 50))
    if args.output is not None:
        args.output.write_text(json.dumps({
            "schema_version": "pose-reprojection-audit-1.0",
            "config": str(args.config),
            "matches": int(len(offsets)),
            "pairs": int(pairs_done),
            "mean_ncc_peak": float(np.mean(peaks)),
            "offset_px": {f"p{p}": float(np.percentile(offsets, p))
                          for p in (25, 50, 75, 90, 95)},
        }, indent=1), encoding="utf-8")
        print(f"report written to {args.output}")
    print(f"\ngs2 after real BA: {GS2_AFTER_BA_PX} px   "
          f"gs2 before: {GS2_BEFORE_BA_PX} px")
    print(f"a 5px gravel stone; 1 deg of rotation = 13.6 px at fl=778")
    if p50 > 3.0:
        print(f"\nVERDICT: p50 {p50:.2f} px is far above the ~1.6 px gs2 started from. "
              f"Pose error alone is enough to destroy few-pixel texture, and bundle "
              f"adjustment is the right next step - it needs hloc/lightglue/pycolmap "
              f"installed, which this machine lacks.")
    elif p50 > 1.5:
        print(f"\nVERDICT: p50 {p50:.2f} px is comparable to gs2 before BA. Worth "
              f"correcting - BA bought gs2 about 32% - but it is unlikely to be the "
              f"whole story on its own.")
    else:
        print(f"\nVERDICT: p50 {p50:.2f} px is already at or below what gs2 achieved "
              f"AFTER bundle adjustment. Pose is NOT the dominant cause of the blur; "
              f"do not spend the BA install on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
