"""Per-frame pose error as a pixel shift, occlusion-cleaned.

Instrument 1 (raw LAS-vs-photo correlation) was blunted by occlusion. Here
every projected LAS point is visibility-checked against the frame's own
z-buffered depth map first (|projected range - depth pixel| < 0.5m), then the
photo is sampled on a grid of trial shifts. A well-posed frame peaks at
(0,0); a drifted pose peaks off-centre by the drift's pixel size, and the
correlation gained by shifting is the smoking gun.
"""
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

DATASETS = Path(r"C:\Peter\3dgs-datasets")
RECORDING = Path(r"C:\Peter\testdata\S1\house0305")
DEPTH_ROOT = DATASETS / "house0305_depth"
HOUSE = np.array([7.1, -5.4])
SEED = 42
EVERY = 12
SHIFTS = np.arange(-24, 25, 4)

manifest = json.loads(
    (DATASETS / "house0305_manifest" / "dataset_manifest.json").read_text(encoding="utf-8"))
cameras = {c["camera_id"]: c for c in manifest["cameras"]}
depth_manifest = json.loads(
    (DEPTH_ROOT / "depth_manifest.json").read_text(encoding="utf-8"))
depth_by_id = {d["image_id"]: d for d in depth_manifest["images"]}


def las_xyz_rgb(path: Path, stride: int):
    with path.open("rb") as stream:
        header = stream.read(400)
        fmt = header[104] & 0x3F
        offset = struct.unpack("<I", header[96:100])[0]
        record = struct.unpack("<H", header[105:107])[0]
        count = struct.unpack("<I", header[107:111])[0]
        if header[24] >= 1 and header[25] >= 4:
            big = struct.unpack("<Q", header[247:255])[0]
            if big:
                count = big
        scale = np.array(struct.unpack("<3d", header[131:155]))
        shift = np.array(struct.unpack("<3d", header[155:179]))
        stream.seek(offset)
        raw = np.frombuffer(stream.read(count * record), dtype=np.uint8).reshape(
            count, record)[::stride]
    xyz = raw[:, :12].copy().view("<i4").reshape(-1, 3).astype(np.float64)
    rgb_offset = {2: 20, 3: 28, 7: 30, 8: 30}.get(fmt)
    rgb = raw[:, rgb_offset:rgb_offset + 6].copy().view("<u2").reshape(-1, 3)
    return xyz * scale + shift, rgb.astype(np.float32) / 65535.0


rng = np.random.default_rng(SEED)
xyz, rgb = las_xyz_rgb(RECORDING / "colorized.las", stride=60)
las_gray = rgb.mean(axis=1)
print(f"LAS sample: {len(xyz):,} points")

images = sorted(manifest["images"], key=lambda i: int(i["timestamp_ns"]))
t0 = int(images[0]["timestamp_ns"])
span = int(images[-1]["timestamp_ns"]) - t0

print(f"\n{'idx':>4} {'t%':>4} {'bearing':>8} {'n_vis':>6} "
      f"{'corr@0':>7} {'best':>6} {'shift_px':>9}")
results = []
for index in range(0, len(images), EVERY):
    image = images[index]
    entry = depth_by_id.get(image["image_id"])
    if entry is None:
        continue
    camera = cameras[image["camera_id"]]
    intr, dist = camera["intrinsic"], camera["distortion"]["params"]
    c2w = np.array(image["c2w"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    local = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    forward = local[:, 2]
    rng_m = np.linalg.norm(local, axis=1)
    keep = (forward > 0.5) & (rng_m < 25.0)
    if keep.sum() < 500:
        continue
    xy = local[keep, :2] / forward[keep, None]
    r = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(r, 1.0)
    t2 = theta * theta
    factor = theta * (1 + dist["k1"] * t2 + dist["k2"] * t2**2
                      + dist["k3"] * t2**3 + dist["k4"] * t2**4)
    s = np.where(r > 1e-9, factor / np.maximum(r, 1e-9), 1.0)
    u = intr["fl_x"] * xy[:, 0] * s + intr["cx"]
    v = intr["fl_y"] * xy[:, 1] * s + intr["cy"]

    with np.load(DEPTH_ROOT / entry["path"]) as archive:
        height, width = archive["shape"]
        depth = np.zeros(height * width, dtype=np.float32)
        depth[archive["pixel_index"]] = archive["range_m"]
        depth = depth.reshape(height, width)
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    inside = ((ui >= 32) & (ui < width - 32) & (vi >= 32) & (vi < height - 32)
              & (np.degrees(theta) < 80))
    ui, vi = ui[inside], vi[inside]
    ranges = rng_m[keep][inside]
    gray = las_gray[keep][inside]
    depth_at = depth[vi, ui]
    visible = (depth_at > 0) & (np.abs(depth_at - ranges) < 0.5)
    if visible.sum() < 300:
        continue
    ui, vi, gray = ui[visible], vi[visible], gray[visible]

    with Image.open(RECORDING / image["path"]) as handle:
        photo = np.asarray(handle.convert("L")).astype(np.float32) / 255.0

    best = (-2.0, 0, 0)
    corr0 = None
    for dy in SHIFTS:
        for dx in SHIFTS:
            values = photo[vi + dy, ui + dx]
            c = float(np.corrcoef(values, gray)[0, 1])
            if dx == 0 and dy == 0:
                corr0 = c
            if c > best[0]:
                best = (c, dx, dy)
    magnitude = float(np.hypot(best[1], best[2]))
    p = c2w[:3, 3]
    bearing = float(np.degrees(np.arctan2(p[1] - HOUSE[1], p[0] - HOUSE[0])))
    fraction = (int(image["timestamp_ns"]) - t0) / span
    results.append((fraction, corr0, best[0], magnitude))
    print(f"{index:>4} {fraction*100:>3.0f}% {bearing:>+8.1f} {visible.sum():>6} "
          f"{corr0:>7.3f} {best[0]:>6.3f} ({best[1]:>+3d},{best[2]:>+3d})")

fractions = np.array([r[0] for r in results])
corr0 = np.array([r[1] for r in results])
bestc = np.array([r[2] for r in results])
mags = np.array([r[3] for r in results])
print(f"\nby capture-time decile: corr@0 / best / median shift px")
for decile in range(10):
    inside = (fractions >= decile / 10) & (fractions < (decile + 1) / 10)
    if inside.any():
        print(f"  {decile*10:>3}-{decile*10+10:<3}%  "
              f"{np.median(corr0[inside]):+.3f} / {np.median(bestc[inside]):+.3f} "
              f"/ {np.median(mags[inside]):5.1f}   n={inside.sum()}")
print(f"\noverall: frames with best-shift > 6px: "
      f"{(mags > 6).sum()}/{len(mags)}")
