"""Census of mid-air gaussians: are the white wisps glare-born floaters?

The user's observation kills the background-leak theory: the sky dome sits
behind everything at render time, so visible white fuzz must be gaussians
that are themselves white-ish and physically floating. The candidate
mechanism is veiling glare (22.9% spatially-varying photometric residual):
sun-facing views see a white haze over the carport that other views do not,
and the optimizer's compromise is translucent white matter hung in mid-air
along the sun-facing rays.

Test: every gaussian farther than FREE_SPACE_M from ANY LiDAR point is in
space the scanner demonstrably saw through - physically impossible matter.
If that population is disproportionately white and semi-transparent
(against the on-surface population), the glare-floater theory stands and
LiDAR-admission pruning (proven -72..85% floaters on the other campaign)
is the right medicine.
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-gate1")

RECORDING = Path(r"C:\Peter\testdata\S1\house0305")
CHECKPOINT = Path(r"C:\Peter\3dgs-runs\probes\q7_ba\checkpoints\latest.pt")
FREE_SPACE_M = 0.30
SH0 = 0.2820947917738781


def las_xyz(stride: int):
    with (RECORDING / "colorized.las").open("rb") as stream:
        header = stream.read(400)
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
    return raw[:, :12].copy().view("<i4").reshape(-1, 3).astype(np.float64) \
        * scale + shift


def main() -> int:
    import torch
    from scipy.spatial import cKDTree

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    params = payload["params"]
    means = params["means"].numpy()
    opacity = torch.sigmoid(params["opacities"].float()).numpy().ravel()
    rgb = np.clip(params["sh0"].numpy()[:, 0, :] * SH0 + 0.5, 0, 1)
    print(f"model: {len(means):,} gaussians (step {payload.get('step')})")

    lidar = las_xyz(stride=20)
    print(f"LiDAR reference: {len(lidar):,} points")
    tree = cKDTree(lidar)

    # Restrict to the scene interior so the sky-adjacent fringe does not
    # dominate: the walk's bounding region plus margin, below roof height.
    interior = ((means[:, 0] > -5) & (means[:, 0] < 20)
                & (means[:, 1] > -20) & (means[:, 1] < 8)
                & (means[:, 2] > 0.2) & (means[:, 2] < 6.0))
    subset = np.flatnonzero(interior)
    print(f"interior gaussians: {len(subset):,}")
    distance, _ = tree.query(means[subset], k=1, workers=8)

    floating = distance > FREE_SPACE_M
    surface = ~floating
    fl, sf = subset[floating], subset[surface]
    print(f"\nmid-air (> {FREE_SPACE_M*100:.0f}cm from any LiDAR point): "
          f"{len(fl):,} ({len(fl)/len(subset)*100:.1f}% of interior)")

    def describe(name, indices):
        if len(indices) == 0:
            print(f"  {name}: none")
            return
        bright = rgb[indices].mean(axis=1)
        whiteness = 1.0 - rgb[indices].std(axis=1) * 3
        visible = opacity[indices] > 0.05
        print(f"  {name}: n={len(indices):,}  opacity p50 "
              f"{np.median(opacity[indices]):.3f} (visible>0.05: "
              f"{visible.mean()*100:.0f}%)  brightness p50 "
              f"{np.median(bright):.2f}  bright&near-white share "
              f"{np.mean((bright > 0.7) & (whiteness > 0.6))*100:.0f}%")

    describe("mid-air ", fl)
    describe("surface ", sf)

    # The visible offenders specifically: mid-air AND visible AND bright.
    bright = rgb[fl].mean(axis=1)
    offenders = fl[(opacity[fl] > 0.05) & (bright > 0.7)]
    print(f"\nvisible bright mid-air offenders: {len(offenders):,}")
    if len(offenders):
        heights = means[offenders, 2]
        print(f"  height p10/p50/p90: {np.percentile(heights,10):.1f}/"
              f"{np.percentile(heights,50):.1f}/{np.percentile(heights,90):.1f} m")
        d2, _ = tree.query(means[offenders], k=1, workers=8)
        print(f"  distance-to-surface p50: {np.median(d2)*100:.0f}cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
