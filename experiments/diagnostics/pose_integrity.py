"""Two checks that split the failure space.

1. Pose integrity: how many of the 886 frames share (near-)identical camera
   positions? A frozen/duplicated pose block would let one corner train
   correctly while mis-posed photos destroy everything else - matching the
   user's report exactly.
2. Why is the val split all in one corner - are its timestamps clustered at
   the start of the capture, or does the walk revisit that corner?
Then print train-split indices at spread bearings around the house for the
far-side render test.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

DS = Path(r"C:\Peter\3dgs-datasets")
HOUSE = np.array([7.1, -5.4])

manifest = json.loads((DS / "house0305_manifest" / "dataset_manifest.json")
                      .read_text(encoding="utf-8"))
images = manifest["images"]
by_id = {f["image_id"]: f for f in images}
print(f"images {len(images)}, unique ids {len(by_id)}")

positions = np.array([[f["c2w"][0][3], f["c2w"][1][3], f["c2w"][2][3]]
                      for f in images])
times = np.array([f["timestamp_ns"] for f in images], dtype=np.int64)

# Duplicate positions at 1mm resolution. Left and right cameras share a rig
# so pairs are expected; more than 2 per position is not.
keys = [tuple(np.round(p, 3)) for p in positions]
counts = Counter(keys)
histogram = Counter(counts.values())
print(f"frames per unique position (1mm): {dict(sorted(histogram.items()))}")
worst = counts.most_common(3)
print(f"most repeated positions: {worst}")

# Time coverage of the val split vs the capture.
split = json.loads((DS / "house0305_evaluation" / "split_manifest.json")
                   .read_text(encoding="utf-8"))
val_ids = set(split["splits"]["val"])
val_times = np.array(sorted(by_id[i]["timestamp_ns"] for i in val_ids
                            if i in by_id), dtype=np.int64)
t0 = times.min()
span = times.max() - t0
print(f"\ncapture span {span/1e9:.0f}s")
fractions = (val_times - t0) / span
hist, _ = np.histogram(fractions, bins=10, range=(0, 1))
print(f"val frames per capture-time decile: {hist.tolist()}")

# Where does the WALK sit in each decile? If the val corner is genuinely
# revisited all through the walk, position medians per decile show it.
frame_fraction = (times - t0) / span
for decile in range(10):
    inside = (frame_fraction >= decile / 10) & (frame_fraction < (decile + 1) / 10)
    p = positions[inside]
    if len(p):
        print(f"  decile {decile}: n={inside.sum():3d} "
              f"median xy ({np.median(p[:,0]):+6.1f},{np.median(p[:,1]):+6.1f}) "
              f"xy spread ({np.ptp(p[:,0]):.1f},{np.ptp(p[:,1]):.1f})")

# Train-split indices at spread bearings around the house.
train_ids = split["splits"]["train"]
bearings = []
for index, image_id in enumerate(train_ids):
    frame = by_id.get(image_id)
    if frame is None:
        continue
    p = np.array([frame["c2w"][0][3], frame["c2w"][1][3]])
    bearing = np.degrees(np.arctan2(p[1] - HOUSE[1], p[0] - HOUSE[0]))
    distance = np.linalg.norm(p - HOUSE)
    bearings.append((index, bearing, distance, frame["camera_id"], p))

print("\ntrain views chosen for the all-round render test:")
chosen = []
for target in (-150, -90, -30, 30, 90, 150):
    best = min(bearings, key=lambda b: abs((b[1] - target + 180) % 360 - 180)
               + 0.1 * abs(b[2] - 6))
    chosen.append(best[0])
    print(f"  bearing {target:+4d}: train index {best[0]:3d} "
          f"actual {best[1]:+7.1f} dist {best[2]:.1f}m cam {best[3]} "
          f"xy ({best[4][0]:+5.1f},{best[4][1]:+5.1f})")
print("indices:", " ".join(str(c) for c in chosen))
