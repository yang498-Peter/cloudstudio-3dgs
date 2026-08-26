"""Map WHERE the near-transparent gaussians live in the delivered H3 model.

The export kept 11.65M of 15.9M at min-opacity 0.005 - 27% of the model is
invisible. If that 27% is spatially concentrated on one side of the house,
training genuinely never supervised that side and the user's report is a data
coverage failure around the HOUSE (the earlier angular audit only proved
coverage around the pose centroid, which is not the same point).
"""
import json
from pathlib import Path

import numpy as np
import torch

PROBES = Path(r"C:\Peter\3dgs-runs\probes")
DS = Path(r"C:\Peter\3dgs-datasets")

payload = torch.load(PROBES / "h3_dense" / "checkpoints" / "latest.pt",
                     map_location="cpu", weights_only=False)
params = payload["params"]
means = params["means"].numpy()
opacity = torch.sigmoid(params["opacities"].float()).numpy().ravel()
print(f"h3 latest.pt: {len(means):,} gaussians, step {payload.get('step')}")
print(f"opacity: p05 {np.percentile(opacity,5):.4f} p50 {np.percentile(opacity,50):.4f}")
dead = opacity < 0.005
print(f"dead (<0.005): {dead.sum():,} ({dead.mean()*100:.1f}%)")

# House footprint: high cells in the z band above the ground.
z = means[:, 2]
print(f"z: p01 {np.percentile(z,1):.1f} p50 {np.percentile(z,50):.1f} "
      f"p99 {np.percentile(z,99):.1f}")
house_band = (z > 1.5) & (z < 9.0)
hx, hy = means[house_band, 0], means[house_band, 1]
house_centre = np.array([np.median(hx), np.median(hy)])
print(f"house-band count {house_band.sum():,}, centre xy "
      f"({house_centre[0]:.1f}, {house_centre[1]:.1f})")

# Top-down 1m grid over the full model: live fraction per cell, as a map.
x0, x1 = np.percentile(means[:, 0], [0.5, 99.5])
y0, y1 = np.percentile(means[:, 1], [0.5, 99.5])
W = 64
H = int(round(W * (y1 - y0) / (x1 - x0)))
H = max(24, min(H, 48))
ix = np.clip(((means[:, 0] - x0) / (x1 - x0) * (W - 1)).astype(int), 0, W - 1)
iy = np.clip(((means[:, 1] - y0) / (y1 - y0) * (H - 1)).astype(int), 0, H - 1)
total = np.zeros((H, W))
alive = np.zeros((H, W))
np.add.at(total, (iy, ix), 1)
np.add.at(alive, (iy, ix), (~dead).astype(float))
with np.errstate(invalid="ignore"):
    frac = alive / total
print(f"\nlive fraction map, x [{x0:.0f},{x1:.0f}] y [{y0:.0f},{y1:.0f}] "
      "(rows = north at top). '#'>80% live, '+'>50, '-'>20, '.'<=20, ' ' empty:")
for row in range(H - 1, -1, -1):
    line = []
    for col in range(W):
        if total[row, col] < 20:
            line.append(" ")
        elif frac[row, col] > 0.8:
            line.append("#")
        elif frac[row, col] > 0.5:
            line.append("+")
        elif frac[row, col] > 0.2:
            line.append("-")
        else:
            line.append(".")
    print("".join(line))

# Same map restricted to the house band - the structure the user looked at.
total_h = np.zeros((H, W))
alive_h = np.zeros((H, W))
np.add.at(total_h, (iy[house_band], ix[house_band]), 1)
np.add.at(alive_h, (iy[house_band], ix[house_band]),
          (~dead[house_band]).astype(float))
with np.errstate(invalid="ignore"):
    frac_h = alive_h / total_h
print("\nsame map, z 1.5-9m only (walls and roof):")
for row in range(H - 1, -1, -1):
    line = []
    for col in range(W):
        if total_h[row, col] < 10:
            line.append(" ")
        elif frac_h[row, col] > 0.8:
            line.append("#")
        elif frac_h[row, col] > 0.5:
            line.append("+")
        elif frac_h[row, col] > 0.2:
            line.append("-")
        else:
            line.append(".")
    print("".join(line))

# Cameras around the HOUSE centre, not the pose centroid.
manifest = json.loads((DS / "house0305_manifest" / "dataset_manifest.json")
                      .read_text(encoding="utf-8"))
images = manifest["images"]
positions = np.array([[f["c2w"][0][3], f["c2w"][1][3], f["c2w"][2][3]]
                      for f in images])
angles = np.degrees(np.arctan2(positions[:, 1] - house_centre[1],
                               positions[:, 0] - house_centre[0]))
hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
print("\ncamera bearing around HOUSE centre (10-deg bins):")
for i in range(0, 36, 6):
    row = " ".join(f"{hist[j]:4d}" for j in range(i, i + 6))
    print(f"  {(-180 + i*10):+4d}..{(-180 + (i+6)*10):+4d}: {row}")
print(f"empty bins: {int((hist==0).sum())}/36")

# Val views: index -> bearing, so far-side render views can be chosen.
split = json.loads((DS / "house0305_evaluation" / "split_manifest.json")
                   .read_text(encoding="utf-8"))
val_ids = split["splits"]["val"]
by_id = {f["image_id"]: f for f in images}
print(f"\nval views ({len(val_ids)}): index, bearing deg, xy")
for index, image_id in enumerate(val_ids):
    frame = by_id.get(image_id)
    if frame is None:
        continue
    p = [frame["c2w"][0][3], frame["c2w"][1][3]]
    bearing = np.degrees(np.arctan2(p[1] - house_centre[1],
                                    p[0] - house_centre[0]))
    marker = " <== current metric view" if index in (42, 60, 78) else ""
    if index % 4 == 0 or index in (42, 60, 78):
        print(f"  {index:3d}  {bearing:+7.1f}  ({p[0]:+6.1f},{p[1]:+6.1f})"
              f"{marker}")
