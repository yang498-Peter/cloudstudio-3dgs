"""Where do the training cameras actually stand, and which frames did training use?

The delivered H3 ply looks trained on one side of the house only. Three
suspects, checked in order of cheapness:
  1. the manifest's poses only cover one side (capture or pose-file truncation)
  2. poses cover everything but the training loader dropped frames
     (unposed / mask / split filtering)
  3. data is fine and the failure is elsewhere (export, opacity threshold)
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

DS = Path(r"C:\Peter\3dgs-datasets")
manifest = json.loads((DS / "house0305_manifest" / "dataset_manifest.json")
                      .read_text(encoding="utf-8"))

images = manifest["images"]
print(f"posed images: {len(images)}")
print(f"unposed images: {len(manifest.get('unposed_images') or [])}")
print(f"cameras: {Counter(i['camera_id'] for i in images)}")
print(f"splits field: {json.dumps(manifest.get('splits'))[:300]}")

positions = np.array([[f["c2w"][0][3], f["c2w"][1][3], f["c2w"][2][3]]
                      for f in images])
times = np.array([f["timestamp_ns"] for f in images], dtype=np.float64)
order = np.argsort(times)
positions = positions[order]
times = times[order]

span = times[-1] - times[0]
print(f"\ntrajectory: {span/1e9:.0f}s of capture")
print(f"position bbox: x [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}] "
      f"y [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}] "
      f"z [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")

# The scene (house) centre from the initialization cloud bounds would need a
# PLY read; the pose centroid is enough to see angular coverage around it.
centre = positions.mean(axis=0)
angles = np.degrees(np.arctan2(positions[:, 1] - centre[1],
                               positions[:, 0] - centre[0]))
hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
print("\nangular coverage around pose centroid (10-degree bins, count per bin):")
for i in range(0, 36, 6):
    row = " ".join(f"{hist[j]:4d}" for j in range(i, i + 6))
    print(f"  {(-180 + i*10):+4d}..{(-180 + (i+6)*10):+4d}: {row}")
empty = int((hist == 0).sum())
print(f"empty 10-degree bins: {empty}/36")

# Text map of the trajectory, top-down, with time along it: first third '1',
# middle '2', last '3' - shows whether the walk circles or doubles back.
W, H = 72, 30
x0, x1 = positions[:, 0].min(), positions[:, 0].max()
y0, y1 = positions[:, 1].min(), positions[:, 1].max()
grid = [[" "] * W for _ in range(H)]
for k, p in enumerate(positions):
    cx = int((p[0] - x0) / max(x1 - x0, 1e-9) * (W - 1))
    cy = int((p[1] - y0) / max(y1 - y0, 1e-9) * (H - 1))
    grid[H - 1 - cy][cx] = "123"[min(2, 3 * k // len(positions))]
print("\ntop-down trajectory (1=early, 2=mid, 3=late):")
for row in grid:
    print("".join(row))

# What did TRAINING actually consume? The run manifest records it.
for run in ("h3_dense", "p1_ppisp"):
    rm = Path(r"C:\Peter\3dgs-runs\probes") / run / "run_manifest.json"
    if rm.exists():
        payload = json.loads(rm.read_text(encoding="utf-8"))
        flat = json.dumps(payload)
        keys = {k: payload[k] for k in payload
                if any(s in k.lower() for s in ("frame", "image", "train", "view"))
                and not isinstance(payload[k], (dict, list))}
        print(f"\n{run} run_manifest scalar keys: {keys}")
        training = payload.get("training", {})
        print(f"  training block keys: {list(training)[:20]}")
        for key in ("train_image_count", "num_train_images", "train_frames",
                    "image_count", "dataset_image_count"):
            if key in training:
                print(f"  training.{key} = {training[key]}")

# Split manifest: does it carve away a big chunk?
split = json.loads((DS / "house0305_evaluation" / "split_manifest.json")
                   .read_text(encoding="utf-8"))
for key, value in split.items():
    if isinstance(value, list):
        print(f"\nsplit '{key}': {len(value)} entries")
    else:
        print(f"split {key}: {value}")

# Mask manifest coverage.
masks = json.loads((DS / "house0305_masks" / "mask_manifest.json")
                   .read_text(encoding="utf-8"))
entries = masks.get("masks") or masks.get("images") or []
print(f"\nmask manifest entries: {len(entries)}")
