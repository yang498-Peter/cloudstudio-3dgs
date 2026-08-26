"""Cut identical small boxes from ours and the competitor: how is each built?

The user's question is structural: inside the SAME physical patch, how do
the two models arrange their gaussians - and do ours sit displaced or
wrong? LiDAR is the impartial referee for both (the competitor was
ICP-aligned onto the same colorized.las).

Per patch and per model:
    density        gaussians per m^3, and nearest-neighbour spacing
    offset         distance to the nearest LiDAR point: p50/p90, and the
                   fraction stranded beyond 10 cm - the direct answer to
                   "are we shifted or wrong"
    build          scale p10/50/90, opacity p50, visible share
    paint          brightness, near-white share (the glare question)
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-gate1")

RECORDING = Path(r"C:\Peter\testdata\S1\house0305")
OURS = Path(r"C:\Peter\3dgs-runs\probes\q7_ba\checkpoints\latest.pt")
COMPETITOR = Path(r"C:\Peter\3dgs-runs\probes\usa_full.pt")
SH0 = 0.2820947917738781

# Boxes chosen from the scene layout (house centre ~(7.1,-5.4)); occupancy
# is printed so an empty pick is visible immediately.
PATCHES = {
    "wall(house side)": ((4.0, 6.5), (-4.5, -2.0), (0.8, 2.3)),
    "ground(gravel)": ((3.0, 5.5), (-8.5, -6.0), (-0.1, 0.35)),
    "carport pillar zone": ((8.0, 10.5), (-3.5, -1.0), (0.3, 2.8)),
}


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


def load(checkpoint: Path):
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = payload["params"]
    return {
        "means": params["means"].numpy(),
        "opacity": torch.sigmoid(params["opacities"].float()).numpy().ravel(),
        "rgb": np.clip(params["sh0"].numpy()[:, 0, :] * SH0 + 0.5, 0, 1),
        "scale": np.exp(params["scales"].numpy()).max(axis=1),
    }


def main() -> int:
    from scipy.spatial import cKDTree

    lidar = las_xyz(stride=5)
    tree = cKDTree(lidar)
    models = {"OURS(q7)": load(OURS), "COMPETITOR": load(COMPETITOR)}

    for name, ((x0, x1), (y0, y1), (z0, z1)) in PATCHES.items():
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        in_box_lidar = np.sum(
            (lidar[:, 0] > x0) & (lidar[:, 0] < x1)
            & (lidar[:, 1] > y0) & (lidar[:, 1] < y1)
            & (lidar[:, 2] > z0) & (lidar[:, 2] < z1))
        print(f"\n=== {name}  box {volume:.1f} m^3, LiDAR pts inside "
              f"{in_box_lidar:,} ===")
        for tag, model in models.items():
            means = model["means"]
            box = ((means[:, 0] > x0) & (means[:, 0] < x1)
                   & (means[:, 1] > y0) & (means[:, 1] < y1)
                   & (means[:, 2] > z0) & (means[:, 2] < z1))
            subset = np.flatnonzero(box)
            if len(subset) == 0:
                print(f"{tag:>12}: EMPTY")
                continue
            pts = means[subset]
            offset, _ = tree.query(pts, k=1, workers=8)
            own = cKDTree(pts)
            nn, _ = own.query(pts, k=2, workers=8)
            spacing = nn[:, 1]
            opacity = model["opacity"][subset]
            visible = opacity > 0.05
            scale = model["scale"][subset]
            rgb = model["rgb"][subset]
            bright = rgb.mean(axis=1)
            white = (bright > 0.75) & (rgb.std(axis=1) < 0.08) & visible
            print(f"{tag:>12}: n={len(subset):>8,}  density "
                  f"{len(subset)/volume:>8,.0f}/m3  spacing p50 "
                  f"{np.median(spacing)*1000:4.0f}mm")
            print(f"{'':>12}  offset-to-LiDAR p50/p90 "
                  f"{np.median(offset)*100:4.1f}/{np.percentile(offset,90)*100:4.1f}cm"
                  f"  stranded>10cm {np.mean(offset>0.10)*100:4.1f}%")
            print(f"{'':>12}  scale p10/50/90 "
                  f"{np.percentile(scale,10)*1000:3.0f}/"
                  f"{np.percentile(scale,50)*1000:3.0f}/"
                  f"{np.percentile(scale,90)*1000:3.0f}mm  opacity p50 "
                  f"{np.median(opacity):.2f}  visible {visible.mean()*100:3.0f}%"
                  f"  white-share {white.mean()*100:4.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
