"""Band census: where exactly do the visible white gaussians sit, how big?

The mid-air census found essentially nothing free-floating (421 of 10.4M),
so the wisps are near-surface matter painted glare-white. This narrows the
question to: at what distance band from the LiDAR surface, and are they
oversized fuzzy blobs (scale) or normal-size white paint?
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-gate1")
sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-gate1\experiments\diagnostics")

from floater_census import las_xyz, SH0  # noqa: E402
from floater_census import CHECKPOINT as DEFAULT_CHECKPOINT  # noqa: E402
CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT


def main() -> int:
    import torch
    from scipy.spatial import cKDTree

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    params = payload["params"]
    means = params["means"].numpy()
    opacity = torch.sigmoid(params["opacities"].float()).numpy().ravel()
    rgb = np.clip(params["sh0"].numpy()[:, 0, :] * SH0 + 0.5, 0, 1)
    scales = np.exp(params["scales"].numpy()).max(axis=1)

    lidar = las_xyz(stride=20)
    tree = cKDTree(lidar)
    box = ((means[:, 0] > 2) & (means[:, 0] < 12)
           & (means[:, 1] > -10) & (means[:, 1] < 0)
           & (means[:, 2] > 0.2) & (means[:, 2] < 4.5))
    subset = np.flatnonzero(box)
    print(f"carport-area gaussians: {len(subset):,}")
    distance, _ = tree.query(means[subset], k=1, workers=8)
    bright = rgb[subset].mean(axis=1)
    white = (bright > 0.75) & (rgb[subset].std(axis=1) < 0.08)
    visible = opacity[subset] > 0.05

    for lo, hi, label in ((0, 0.05, "<5cm"), (0.05, 0.15, "5-15cm"),
                          (0.15, 0.30, "15-30cm"), (0.30, 9, ">30cm")):
        band = (distance >= lo) & (distance < hi)
        if band.sum() == 0:
            continue
        offenders = band & white & visible
        scale_p50 = (np.median(scales[subset][offenders]) * 1000
                     if offenders.sum() else 0)
        op_p50 = (np.median(opacity[subset][offenders])
                  if offenders.sum() else 0)
        print(f"{label:>8} n={band.sum():>9,}  visible-white "
              f"{offenders.sum():>7,} ({offenders.sum()/band.sum()*100:5.2f}%)  "
              f"scale p50 {scale_p50:4.0f}mm  opacity p50 {op_p50:.2f}")

    offenders = white & visible
    print(f"\ntotal visible-white in box: {offenders.sum():,} "
          f"({offenders.mean()*100:.2f}% of box)")
    if offenders.sum():
        print(f"  their scale p50 {np.median(scales[subset][offenders])*1000:.0f}mm "
              f"vs box-wide p50 {np.median(scales[subset])*1000:.0f}mm")
        print(f"  their opacity p50 {np.median(opacity[subset][offenders]):.2f} "
              f"vs box-wide p50 {np.median(opacity[subset]):.2f}")
        heights = means[subset][offenders, 2]
        print(f"  height p10/p50/p90: {np.percentile(heights, 10):.1f}/"
              f"{np.percentile(heights, 50):.1f}/"
              f"{np.percentile(heights, 90):.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
