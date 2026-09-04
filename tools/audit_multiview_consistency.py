"""Do two cameras agree on where the same LiDAR surface point appears?

This is exactly the consistency the trainer assumes. For a surface point whose
range is known in face A (from the signed LiDAR cache, so no depth estimation
is involved), back-project it to 3D with A's own K/c2w, project it into a
second face B that sees the same area, and measure the shift that best aligns
the two image patches. Perfect poses and extrinsics give a shift of zero; a
consistent non-zero shift is registration error, and it is the shift the
photometric loss can only resolve by blurring.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-work")
from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset

RUN = Path(r"C:\Peter\3dgs-runs\house0305_sop")
CONFIG = RUN / "tile0_G1_flatten033_t0_20k.json"
PATCH = 21          # patch half-size is PATCH//2
SEARCH = 8          # +/- 8 sensor px
PAIRS = 40          # face pairs to test
PER_PAIR = 40       # points per pair

raw = json.loads(CONFIG.read_text(encoding="utf-8"))
tiles = json.loads(Path(raw["tile_inputs_manifest"]).read_text(encoding="utf-8"))["tiles"]
tile = [t for t in tiles if int(t["tile_id"]) == int(raw.get("mipmap_tile_id", 0))][0]
dataset = FaceCacheDataset(
    Path(raw["face_cache_manifest"]),
    Path(raw["face_cache_root"]),
    verify_artifacts=False,
    dataset_manifest_path=Path(raw["dataset_manifest"]),
    renderer_mask_manifest_path=Path(raw["renderer_mask_manifest"]),
    face_lidar_geometry_manifest_path=Path(raw["face_lidar_geometry_manifest"]),
    face_lidar_geometry_root=Path(raw["face_lidar_geometry_root"]),
    tile_views=tile["views"],
)
print(f"faces {len(dataset)}")

samples = []
for index in range(0, len(dataset), max(1, len(dataset) // 120)):
    s = dataset[index]
    if s.depth_range_m is None:
        continue
    samples.append(s)
    if len(samples) >= 120:
        break
print(f"loaded {len(samples)} faces with LiDAR range")

grey = {}
for s in samples:
    grey[s.image_id] = cv2.cvtColor(np.asarray(s.image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

rng = np.random.default_rng(0)
half = PATCH // 2
rows = []
for i, a in enumerate(samples):
    Ka, c2wa = np.asarray(a.K, float), np.asarray(a.c2w, float)
    ranges = np.asarray(a.depth_range_m, dtype=np.float32)
    valid = np.argwhere(np.isfinite(ranges) & (ranges > 0.3))
    if len(valid) < 50:
        continue
    # z from euclidean range needs the per-pixel ray norm; rebuild it from K
    picked = valid[rng.choice(len(valid), size=min(PER_PAIR * 3, len(valid)), replace=False)]
    ya, xa = picked[:, 0].astype(float), picked[:, 1].astype(float)
    rays = np.stack([(xa - Ka[0, 2]) / Ka[0, 0], (ya - Ka[1, 2]) / Ka[1, 1], np.ones_like(xa)], 1)
    norms = np.linalg.norm(rays, axis=1)
    z = ranges[picked[:, 0], picked[:, 1]] / norms
    cam = rays * z[:, None]
    world = cam @ c2wa[:3, :3].T + c2wa[:3, 3]
    for b in samples[i + 1 : i + 6]:
        if b.image_id == a.image_id:
            continue
        Kb, c2wb = np.asarray(b.K, float), np.asarray(b.c2w, float)
        camb = (world - c2wb[:3, 3]) @ c2wb[:3, :3]
        front = camb[:, 2] > 0.3
        if front.sum() < 10:
            continue
        ub = Kb[0, 0] * camb[front, 0] / camb[front, 2] + Kb[0, 2]
        vb = Kb[1, 1] * camb[front, 1] / camb[front, 2] + Kb[1, 2]
        xa_f, ya_f = xa[front], ya[front]
        margin = half + SEARCH + 1
        ok = (
            (ub > margin) & (ub < b.width - margin) & (vb > margin) & (vb < b.height - margin)
            & (xa_f > margin) & (xa_f < a.width - margin) & (ya_f > margin) & (ya_f < a.height - margin)
        )
        if ok.sum() < 5:
            continue
        ga, gb = grey[a.image_id], grey[b.image_id]
        for xa_p, ya_p, ub_p, vb_p in list(zip(xa_f[ok], ya_f[ok], ub[ok], vb[ok]))[:PER_PAIR]:
            xa_i, ya_i, ub_i, vb_i = int(round(xa_p)), int(round(ya_p)), int(round(ub_p)), int(round(vb_p))
            template = ga[ya_i - half : ya_i + half + 1, xa_i - half : xa_i + half + 1]
            if template.std() < 12:            # featureless patch cannot localise
                continue
            window = gb[vb_i - half - SEARCH : vb_i + half + SEARCH + 1, ub_i - half - SEARCH : ub_i + half + SEARCH + 1]
            if window.shape[0] != PATCH + 2 * SEARCH or window.shape[1] != PATCH + 2 * SEARCH:
                continue
            result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            _, peak, _, loc = cv2.minMaxLoc(result)
            if peak < 0.5:                      # not the same surface / occluded
                continue
            dx = loc[0] - SEARCH + (ub_i - ub_p)
            dy = loc[1] - SEARCH + (vb_i - vb_p)
            rows.append({"a": a.image_id, "b": b.image_id, "dx": float(dx), "dy": float(dy), "peak": float(peak)})
        if len(rows) >= PAIRS * PER_PAIR:
            break
    if len(rows) >= PAIRS * PER_PAIR:
        break

dxs = np.array([r["dx"] for r in rows])
dys = np.array([r["dy"] for r in rows])
radial = np.hypot(dxs, dys)
print(f"\nmatched patches {len(rows)}  (NCC peak median {np.median([r['peak'] for r in rows]):.2f})")
print(f"dx  median {np.median(dxs):+.2f} px   dy median {np.median(dys):+.2f} px")
print(f"|disagreement|  median {np.median(radial):.2f} px   p75 {np.percentile(radial, 75):.2f}   p90 {np.percentile(radial, 90):.2f}")
print(f"systematic (median vector) {np.hypot(np.median(dxs), np.median(dys)):.2f} px   scatter (std) {np.hypot(dxs.std(), dys.std()):.2f} px")
json.dump(rows, open(Path(__file__).with_name("multiview_consistency.json"), "w"), indent=1)
