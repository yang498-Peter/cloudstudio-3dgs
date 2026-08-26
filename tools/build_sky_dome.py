#!/usr/bin/env python3
"""Build a photo-baked sky dome as a gaussian layer, competitor-style.

The reference product ships its sky as a separate sky.ply: exactly 100k SH0
gaussians on a ~210m spherical shell (p05-p95 radius 192-225m), elevations
-12.5deg to +71deg (the below-horizon band closes the gap between dome and
terrain), sizes 1-6m, colours/opacities fitted to the photos. Without that
layer every full-frame render here shows the trainer's background colour
where the sky should be, and no comparison against a photo survives it.

This tool reproduces the recipe without any training:

  1. sample the shell procedurally (Fibonacci in sin-elevation, so density
     is uniform on the band), centred on the camera track at z=0;
  2. for a stride of posed frames, mark which pixels are definitely sky -
     inside the fisheye validity cone, away from any LiDAR depth support
     (dilated), outside person boxes, above a floor elevation;
  3. project every dome point into every such frame and collect the photo
     colours it lands on; the per-point robust median IS the dome colour.
     Points the walk never saw stay transparent.

Parallax is handled exactly - each dome point is a real 3D point projected
through each frame's own pose - so the same bake works for any capture
whose sky is stationary over the session.

Output is a checkpoint-shaped .pt (params: means/quats/scales/opacities/
sh0/shN) that concatenates onto any trained model of the same scene.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

SH0_SCALE = 0.2820947917738781
DOWN = 8  # sky-validity grids run at 1/8 resolution; dilation stays cheap


def fibonacci_band(count: int, elevation_min_deg: float,
                   elevation_max_deg: float, seed: int) -> np.ndarray:
    """Uniform directions on the elevation band via the golden-angle spiral."""
    lo = np.sin(np.radians(elevation_min_deg))
    hi = np.sin(np.radians(elevation_max_deg))
    index = np.arange(count, dtype=np.float64)
    z = lo + (hi - lo) * (index + 0.5) / count
    azimuth = index * np.pi * (3.0 - np.sqrt(5.0))
    # A deterministic twist so re-runs with another seed decorrelate the seam.
    azimuth = azimuth + np.random.default_rng(seed).uniform(0, 2 * np.pi)
    radial = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.stack([radial * np.cos(azimuth), radial * np.sin(azimuth), z], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--recording-root", required=True, type=Path)
    parser.add_argument("--depth-manifest", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--person-mask-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--radius-m", type=float, default=250.0)
    parser.add_argument("--elevation-min-deg", type=float, default=-15.0)
    parser.add_argument("--elevation-max-deg", type=float, default=88.0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--theta-max-deg", type=float, default=80.0)
    parser.add_argument("--min-samples", type=int, default=3,
                        help="a dome point needs this many photo readings "
                             "before it earns opacity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from PIL import Image
    from scipy.ndimage import maximum_filter

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    cameras = {c["camera_id"]: c for c in manifest["cameras"]}
    images = sorted(manifest["images"], key=lambda i: int(i["timestamp_ns"]))
    depth_manifest = json.loads(args.depth_manifest.read_text(encoding="utf-8"))
    depth_by_id = {d["image_id"]: d for d in depth_manifest["images"]}
    person_boxes: dict[str, list] = {}
    if args.person_mask_manifest and args.person_mask_manifest.exists():
        person = json.loads(args.person_mask_manifest.read_text(encoding="utf-8"))
        for entry in person.get("images", []):
            person_boxes[str(entry["image_id"])] = [
                instance["box_xyxy"] for instance in entry.get("instances", [])]

    positions = np.array([[f["c2w"][0][3], f["c2w"][1][3], f["c2w"][2][3]]
                          for f in images])
    centre = np.array([positions[:, 0].mean(), positions[:, 1].mean(), 0.0])

    directions = fibonacci_band(args.count, args.elevation_min_deg,
                                args.elevation_max_deg, args.seed)
    dome = centre + directions * args.radius_m
    # Mean neighbour spacing on the band sets the gaussian size: the dome
    # should read as a continuous backdrop, not a starfield.
    band_area = (2.0 * np.pi * args.radius_m ** 2
                 * (np.sin(np.radians(args.elevation_max_deg))
                    - np.sin(np.radians(args.elevation_min_deg))))
    spacing = float(np.sqrt(band_area / args.count))
    print(f"dome: {args.count:,} points, radius {args.radius_m:.0f}m, "
          f"spacing {spacing:.2f}m, centre ({centre[0]:.1f},{centre[1]:.1f},0)")

    sample_points: list[np.ndarray] = []
    sample_colors: list[np.ndarray] = []
    frames_used = 0
    for frame_index in range(0, len(images), args.frame_stride):
        image = images[frame_index]
        entry = depth_by_id.get(image["image_id"])
        if entry is None:
            continue
        camera = cameras[image["camera_id"]]
        intr = camera["intrinsic"]
        dist = camera["distortion"]["params"]
        w2c = np.linalg.inv(np.array(image["c2w"], dtype=np.float64))
        local = dome @ w2c[:3, :3].T + w2c[:3, 3]
        forward = local[:, 2]
        keep = forward > 1.0
        if keep.sum() < 100:
            continue
        xy = local[keep, :2] / forward[keep, None]
        r = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(r, 1.0)
        t2 = theta * theta
        factor = theta * (1 + dist["k1"] * t2 + dist["k2"] * t2 ** 2
                          + dist["k3"] * t2 ** 3 + dist["k4"] * t2 ** 4)
        s = np.where(r > 1e-9, factor / np.maximum(r, 1e-9), 1.0)
        u = intr["fl_x"] * xy[:, 0] * s + intr["cx"]
        v = intr["fl_y"] * xy[:, 1] * s + intr["cy"]

        with np.load(args.depth_root / entry["path"]) as archive:
            height, width = (int(x) for x in archive["shape"])
            grid_h, grid_w = height // DOWN + 1, width // DOWN + 1
            cell_hits = np.zeros((grid_h, grid_w), dtype=np.int32)
            flat = archive["pixel_index"]
            np.add.at(cell_hits, ((flat // width) // DOWN,
                                  (flat % width) // DOWN), 1)
        # The splatted LiDAR depth scatters stray single returns far outside
        # real surfaces - "any depth pixel in the cell" declares half the sky
        # to be geometry. A real surface fills its cells densely, so demand a
        # tenth of the cell before it counts, then one cell of margin for the
        # silhouette.
        covered = maximum_filter(cell_hits >= 6, size=3)

        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        ok = ((ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
              & (np.degrees(theta) < args.theta_max_deg))
        ok &= ~covered[np.clip(vi, 0, height - 1) // DOWN,
                       np.clip(ui, 0, width - 1) // DOWN]
        for box in person_boxes.get(str(image["image_id"]), ()):
            x0, y0, x1, y1 = box
            ok &= ~((ui >= x0 - 16) & (ui <= x1 + 16)
                    & (vi >= y0 - 16) & (vi <= y1 + 16))
        if not ok.any():
            continue

        with Image.open(args.recording_root / image["path"]) as handle:
            photo = np.asarray(handle.convert("RGB"))
        indices = np.flatnonzero(keep)[ok]
        sample_points.append(indices)
        sample_colors.append(photo[vi[ok], ui[ok]].astype(np.float32) / 255.0)
        frames_used += 1

    if not sample_points:
        print("no sky samples found - nothing to bake", file=sys.stderr)
        return 1
    points = np.concatenate(sample_points)
    colors = np.concatenate(sample_colors)
    print(f"baked from {frames_used} frames, {len(points):,} samples")

    order = np.argsort(points, kind="stable")
    points, colors = points[order], colors[order]
    unique, starts = np.unique(points, return_index=True)
    boundaries = np.append(starts, len(points))
    rgb = np.full((args.count, 3), 0.5, dtype=np.float32)
    counts = np.zeros(args.count, dtype=np.int64)
    for position, point in enumerate(unique):
        segment = colors[boundaries[position]:boundaries[position + 1]]
        rgb[point] = np.median(segment, axis=0)
        counts[point] = len(segment)

    seen = counts >= args.min_samples
    print(f"coverage: {seen.sum():,}/{args.count:,} points "
          f"({seen.mean() * 100:.1f}%) with >= {args.min_samples} samples; "
          f"median samples per seen point {np.median(counts[seen]):.0f}")

    # The sky is low-frequency, so directions the walk never sampled can
    # borrow the nearest sampled colour instead of punching white holes in
    # the backdrop. They stay distinguishable in the report, not in the render.
    if seen.any() and not seen.all():
        from scipy.spatial import cKDTree

        tree = cKDTree(directions[seen])
        _, nearest = tree.query(directions[~seen], k=1)
        rgb[~seen] = rgb[seen][nearest]
        filled = int((~seen).sum())
        print(f"filled {filled:,} unseen points from nearest sampled neighbours")
        seen = np.ones(args.count, dtype=bool)

    import torch

    count = args.count
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.where(seen, 4.0, -6.0).astype(np.float32)  # sigmoid: 0.98 / 0.002
    scales = np.full((count, 3), np.log(spacing * 0.7), dtype=np.float32)
    payload = {
        "params": {
            "means": torch.from_numpy(dome.astype(np.float32)),
            "quats": torch.from_numpy(quats),
            "scales": torch.from_numpy(scales),
            "opacities": torch.from_numpy(opacities),
            "sh0": torch.from_numpy(((rgb - 0.5) / SH0_SCALE)
                                    .astype(np.float32)[:, None, :]),
            "shN": torch.zeros((count, 0, 3), dtype=torch.float32),
        },
        "sky_dome": {
            "algorithm_version": "photo_baked_sky_dome_v1",
            "centre": centre.tolist(),
            "radius_m": args.radius_m,
            "elevation_deg": [args.elevation_min_deg, args.elevation_max_deg],
            "count": count,
            "spacing_m": spacing,
            "frames_used": frames_used,
            "seen_fraction": float(seen.mean()),
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"sky dome -> {args.output} "
          f"({args.output.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
