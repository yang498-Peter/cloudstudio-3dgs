"""Where exactly does the competitor beat us? A distribution, not a scalar.

Renders OUR model and the competitor at 12 train views spread around the
house, tiles each frame, and scores every tile's high-frequency energy
against the photo. The per-tile deficit (competitor - ours) is then
aggregated two ways:

  by geometry class   ray-elevation band x LiDAR-depth band, i.e. ground /
                      walls-and-objects / canopy, near / mid / far - the
                      actionable "what kind of surface do we lose on"
  by bearing          which side of the house loses most

Only LiDAR-supported pixels count (sky belongs to a separate layer on both
sides), person-masked pixels are already out of the rgb mask, and a tile
votes only when at least 40% of it is valid. The three worst views get a
heatmap PNG so the numbers can be eyeballed.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Peter\cloudstudio-3dgs-gate1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

RUNS = Path(r"C:\Peter\3dgs-runs")
PROBES = RUNS / "probes"
EXPORTS = RUNS / "exports"
OURS = PROBES / "q2_full_pose" / "checkpoints" / "latest.pt"
COMPETITOR = PROBES / "usa_full.pt"
CONFIG = RUNS / "probe_q2_config.json"
HOUSE = np.array([7.1, -5.4])
TILE = 56
N_VIEWS = 12
MIN_VALID = 0.4

ELEVATION_BANDS = ((-90, -15, "ground"), (-15, 10, "walls/objects"),
                   (10, 90, "canopy/high"))
DEPTH_BANDS = ((0, 5, "near<5m"), (5, 12, "mid5-12m"), (12, 99, "far>12m"))


def high_frequency(gray: np.ndarray) -> np.ndarray:
    from scipy.ndimage import laplace
    return np.abs(laplace(gray))


def main() -> int:
    import torch
    from sharpness_metrics import _load_backend
    from cloudstudio_3dgs.training.dataset import S1TrainingDataset

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    backend, torch = _load_backend(config)
    device = config.get("device", "cuda:0")

    dataset = S1TrainingDataset(
        dataset_manifest_path=Path(config["dataset_manifest"]),
        recording_root=Path(config["recording_root"]),
        mask_manifest_path=Path(config["mask_manifest"]),
        mask_root=Path(config["mask_root"]),
        split_manifest_path=Path(config["split_manifest"]),
        split="train",
        factor=config["factor"],
        crop=None,
        depth_manifest_path=Path(config["depth_manifest"]),
        depth_root=Path(config["depth_root"]),
    )

    # Spread the probe views in bearing around the house, 4-10m out.
    manifest = json.loads(Path(config["dataset_manifest"]).read_text(encoding="utf-8"))
    by_id = {f["image_id"]: f for f in manifest["images"]}
    split = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    candidates = []
    for index, image_id in enumerate(split["splits"]["train"]):
        frame = by_id.get(image_id)
        if frame is None:
            continue
        p = np.array([frame["c2w"][0][3], frame["c2w"][1][3]])
        bearing = float(np.degrees(np.arctan2(p[1] - HOUSE[1], p[0] - HOUSE[0])))
        distance = float(np.linalg.norm(p - HOUSE))
        candidates.append((index, bearing, distance))
    views = []
    for target in np.linspace(-180, 180, N_VIEWS, endpoint=False):
        best = min(candidates,
                   key=lambda c: abs((c[1] - target + 180) % 360 - 180)
                   + 0.15 * abs(c[2] - 6.5))
        if best[0] not in views:
            views.append(best[0])
    print(f"views: {views}")

    def render_all(checkpoint: Path):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        params = {k: v.to(device) for k, v in payload["params"].items()}
        del payload
        out = {}
        with torch.no_grad():
            for index in views:
                sample = dataset[index]
                rgb, _, _, _ = backend.render(
                    params, sample, with_range=False,
                    background_rgb=config["background_color"])
                out[index] = rgb.clamp(0, 1).cpu().numpy().astype(np.float16)
        del params
        torch.cuda.empty_cache()
        return out

    print("rendering ours...")
    ours = render_all(OURS)
    print("rendering competitor...")
    comp = render_all(COMPETITOR)

    class_sum = np.zeros((3, 3))
    class_cnt = np.zeros((3, 3), dtype=int)
    bearing_rows = []
    heat_maps = {}

    for index in views:
        sample = dataset[index]
        photo = np.asarray(sample.image, dtype=np.float32) / 255.0
        mask = np.asarray(sample.rgb_mask, dtype=bool)
        depth_mask = np.asarray(sample.depth_mask, dtype=bool)
        depth = np.asarray(sample.depth_range_m, dtype=np.float32)
        valid = mask & depth_mask

        gray_p = photo.mean(axis=2)
        gray_o = ours[index].astype(np.float32).mean(axis=2)
        gray_c = comp[index].astype(np.float32).mean(axis=2)
        hf_p, hf_o, hf_c = (high_frequency(g) for g in (gray_p, gray_o, gray_c))

        # Ray elevation per pixel (equidistant approx is fine for banding).
        height, width = gray_p.shape
        fl = float(sample.K[0, 0])
        cx, cy = float(sample.K[0, 2]), float(sample.K[1, 2])
        uu, vv = np.meshgrid(np.arange(width), np.arange(height))
        dx, dy = (uu - cx) / fl, (vv - cy) / fl
        r = np.sqrt(dx * dx + dy * dy) + 1e-9
        theta = r  # equidistant: r_norm = theta
        dir_cam = np.stack([np.sin(theta) * dx / r, np.sin(theta) * dy / r,
                            np.cos(theta)], axis=-1)
        rotation = np.asarray(sample.c2w, dtype=np.float64)[:3, :3]
        dir_world_z = dir_cam @ rotation.T[:, 2]
        elevation = np.degrees(np.arcsin(np.clip(dir_world_z, -1, 1)))

        grid_h, grid_w = height // TILE, width // TILE
        heat = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
        for gy in range(grid_h):
            for gx in range(grid_w):
                sl = (slice(gy * TILE, (gy + 1) * TILE),
                      slice(gx * TILE, (gx + 1) * TILE))
                v = valid[sl]
                if v.mean() < MIN_VALID:
                    continue
                base = float(hf_p[sl][v].mean())
                if base < 1e-4:
                    continue
                e_ours = float(hf_o[sl][v].mean()) / base
                e_comp = float(hf_c[sl][v].mean()) / base
                weakness = e_comp - e_ours
                heat[gy, gx] = weakness
                med_elev = float(np.median(elevation[sl][v]))
                d = depth[sl][depth_mask[sl] & mask[sl]]
                med_depth = float(np.median(d)) if len(d) else 8.0
                ei = next(i for i, (lo, hi, _) in enumerate(ELEVATION_BANDS)
                          if lo <= med_elev < hi)
                di = next(i for i, (lo, hi, _) in enumerate(DEPTH_BANDS)
                          if lo <= med_depth < hi)
                class_sum[ei, di] += weakness
                class_cnt[ei, di] += 1
        frame = by_id[sample.image_id]
        p = np.array([frame["c2w"][0][3], frame["c2w"][1][3]])
        bearing = float(np.degrees(np.arctan2(p[1] - HOUSE[1], p[0] - HOUSE[0])))
        mean_weak = float(np.nanmean(heat))
        bearing_rows.append((index, bearing, mean_weak))
        heat_maps[index] = (heat, mean_weak)
        print(f"view {index:4d} bearing {bearing:+7.1f}  "
              f"mean weakness {mean_weak:+.3f}")

    print("\nweakness (competitor HF - ours HF, photo-normalized; + = we are softer)")
    print(f"{'':>16}" + "".join(f"{label:>12}" for _, _, label in DEPTH_BANDS))
    for ei, (_, _, elabel) in enumerate(ELEVATION_BANDS):
        cells = []
        for di in range(3):
            if class_cnt[ei, di]:
                cells.append(f"{class_sum[ei, di] / class_cnt[ei, di]:+11.3f} ")
            else:
                cells.append(f"{'n/a':>12}")
        print(f"{elabel:>16}" + "".join(cells)
              + f"   (tiles {class_cnt[ei].sum()})")

    worst = sorted(bearing_rows, key=lambda r: -r[2])[:3]
    print(f"\nworst bearings: "
          + ", ".join(f"view {i} @{b:+.0f}deg ({w:+.3f})" for i, b, w in worst))

    from PIL import Image
    for index, bearing, _ in worst:
        heat, _ = heat_maps[index]
        sample = dataset[index]
        photo = np.asarray(sample.image, dtype=np.uint8)
        scale = np.clip((np.nan_to_num(heat, nan=0.0)) / 0.6, 0, 1)
        overlay = np.kron(scale, np.ones((TILE, TILE)))[:photo.shape[0], :photo.shape[1]]
        red = photo.astype(np.float32)
        red[..., 0] = np.clip(red[..., 0] + overlay * 200, 0, 255)
        red[..., 1] = red[..., 1] * (1 - overlay * 0.5)
        red[..., 2] = red[..., 2] * (1 - overlay * 0.5)
        Image.fromarray(red.astype(np.uint8)).save(
            EXPORTS / f"weakness_view{index}.png")
    print(f"heatmaps -> {EXPORTS}\\weakness_view*.png")

    report = {
        "views": bearing_rows,
        "class_matrix": {
            ELEVATION_BANDS[ei][2]: {
                DEPTH_BANDS[di][2]:
                    (class_sum[ei, di] / class_cnt[ei, di]
                     if class_cnt[ei, di] else None)
                for di in range(3)}
            for ei in range(3)},
        "tile_counts": class_cnt.tolist(),
    }
    (PROBES / "weakness_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
