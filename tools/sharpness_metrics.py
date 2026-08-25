"""Sharpness acceptance: is the render as detailed as the photo, and is the detail real?

The five acceptance tables (PSNR, SSIM, LPIPS, depth MAE, floater counts) all
average over the frame. A gravel bed collapsing into flat beige moves none of
them appreciably, so a campaign can improve every table for hours while the
renders stay visibly blurred - which is exactly what happened on 2026-08-25.
This module measures the thing those tables cannot see.

Two numbers, both scale-free and both 1.0 for a perfect render:

**energy** - mean gradient magnitude of the render over that of the photo, inside
the mask. It answers "is there as much detail as the photo has". Blur drives it
toward zero.

**agreement** - Pearson correlation between the two gradient magnitude maps. It
answers "is the detail in the RIGHT places". This exists because energy alone
cannot tell resolution from speckle: an arm that reached 0.829 energy scored
0.180 agreement, the lowest of any model measured, and its renders showed vivid
invented gravel that correlated with nothing in the scene.

Read them together:

    energy low,  agreement moderate -> blurred (edges attenuated but located)
    energy high, agreement low       -> invented detail, worse than blur
    energy high, agreement high      -> genuinely resolved

One honest caveat: a blurred render earns agreement cheaply, because a smoothed
edge still sits where the edge is. The metric is therefore biased in favour of
blur, and a modest agreement score from a sharp model is not automatically
damning - compare arms at similar energy before ranking them on agreement.

Usage::

    python tools/sharpness_metrics.py --config run.json --checkpoint best.pt \\
        [--split val] [--views 0 40 80] [--crops-dir out/] [--output report.json]

The crops are the point of the exercise as much as the numbers: they are written
at native resolution, taken from the highest-texture region of each view, so a
person can see what the score means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SCHEMA_VERSION = "sharpness-metrics-1.0"
DEFAULT_VIEWS = (0, 40, 80)
DEFAULT_CROP_PX = 320


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Per-pixel |grad| of the luminance-ish channel mean."""
    grey = image.mean(axis=2)
    gy, gx = np.gradient(grey)
    return np.hypot(gx, gy)


def energy_ratio(render: np.ndarray, reference: np.ndarray,
                 mask: np.ndarray) -> float:
    """Render detail over photo detail. 1.0 means as much texture as the photo."""
    inner = mask[1:-1, 1:-1]
    if not inner.any():
        return float("nan")
    a = gradient_magnitude(render)[1:-1, 1:-1][inner].mean()
    b = gradient_magnitude(reference)[1:-1, 1:-1][inner].mean()
    return float(a / b) if b > 0 else float("nan")


def agreement(render: np.ndarray, reference: np.ndarray,
              mask: np.ndarray) -> float:
    """Correlation of the gradient maps. 1.0 means the detail is where it belongs."""
    inner = mask[1:-1, 1:-1]
    if not inner.any():
        return float("nan")
    a = gradient_magnitude(render)[1:-1, 1:-1][inner]
    b = gradient_magnitude(reference)[1:-1, 1:-1][inner]
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def background_fraction(render: np.ndarray, mask: np.ndarray,
                        background: np.ndarray, tolerance: float = 0.02) -> float:
    """Share of masked pixels left showing the background - i.e. holes.

    Reported alongside sharpness because the two can be traded against each
    other silently. Shrinking Gaussians raises gradient energy whether the new
    edges are texture or the rims of gaps, and a model that has thinned its way
    to 15% background will look sharper on the energy number while being
    strictly worse. Coverage collapses when footprint is reduced without raising
    the count to match: covered area goes as N * r^2, so halving r needs four
    times the Gaussians just to stand still.
    """
    inner = mask
    if not inner.any():
        return float("nan")
    distance = np.abs(render - background[None, None, :]).max(axis=2)
    return float((distance[inner] < tolerance).mean())


def classify(energy: float, correlation: float, holes: float = 0.0) -> str:
    """The reading that matters, so a caller cannot rank on energy alone."""
    if not np.isfinite(energy) or not np.isfinite(correlation):
        return "unmeasured"
    if np.isfinite(holes) and holes > 0.10:
        return "holes"
    if energy > 0.6 and correlation > 0.6:
        return "resolved"
    if energy > 0.6 and correlation < 0.4:
        return "invented detail"
    if energy < 0.4 and correlation > 0.5:
        return "blurred"
    return "mixed"


def highest_texture_crop(reference: np.ndarray, mask: np.ndarray,
                         size: int = DEFAULT_CROP_PX) -> tuple[int, int]:
    """Locate the most textured masked window - where blur is visible to a person."""
    energy = gradient_magnitude(reference) * mask
    best, best_score = (0, 0), -1.0
    step = max(size // 2, 1)
    for y in range(0, max(energy.shape[0] - size, 1), step):
        for x in range(0, max(energy.shape[1] - size, 1), step):
            score = float(energy[y:y + size, x:x + size].mean())
            if score > best_score:
                best_score, best = score, (y, x)
    return best


def _load_backend(config: dict):
    import torch
    from cloudstudio_3dgs.training.backend import GsplatBackend

    backend = GsplatBackend(
        device=config.get("device", "cuda:0"),
        cap_max=config["cap_max"],
        lock_path=Path(config["gsplat_lock"]),
        mcmc_config={"noise_injection_stop_iter": 0},
    )
    backend.color_model = config.get("color_model", "sh")
    backend.sh_degree = int(config.get("sh_degree", 3))
    return backend, torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--views", type=int, nargs="*", default=list(DEFAULT_VIEWS))
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_PX)
    parser.add_argument("--crops-dir", type=Path, default=None,
                        help="write native-resolution photo|render crops here")
    parser.add_argument("--tag", default="model", help="prefix for crop filenames")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from PIL import Image
    from cloudstudio_3dgs.training.dataset import S1TrainingDataset

    config = json.loads(args.config.read_text(encoding="utf-8"))
    backend, torch = _load_backend(config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = config.get("device", "cuda:0")
    params = {k: v.to(device) for k, v in payload["params"].items()}

    dataset = S1TrainingDataset(
        dataset_manifest_path=Path(config["dataset_manifest"]),
        recording_root=Path(config["recording_root"]),
        mask_manifest_path=Path(config["mask_manifest"]),
        mask_root=Path(config["mask_root"]),
        split_manifest_path=Path(config["split_manifest"]),
        split=args.split,
        factor=config["factor"],
        crop=None,
    )

    print(f"Sharpness metrics ({SCHEMA_VERSION})")
    print(f"split={args.split}  frames={len(dataset)}  "
          f"gaussians={params['means'].shape[0]:,}")
    # Which model this actually is. best_golden.pt stops advancing as soon as
    # the golden gate starts refusing checkpoints, so a 16000-step run can hand
    # back a step-7000 model with nothing in the log saying so - and every
    # number below would then describe that earlier model.
    measured_step = payload.get("step", payload.get("completed_steps"))
    requested_steps = config.get("max_steps")
    if measured_step is not None:
        note = f"\ncheckpoint is step {measured_step:,}"
        if requested_steps:
            note += f" of {requested_steps:,} requested"
            if measured_step < requested_steps:
                note += (f"  <- DELIVERED MODEL IS EARLIER THAN THE RUN; the "
                         f"golden gate froze selection at "
                         f"{measured_step / requested_steps * 100:.0f}%")
        print(note)
    print(f"\n{'view':>6}{'energy':>10}{'agreement':>12}{'holes':>9}   reading")

    background = np.asarray(config["background_color"], dtype=np.float32)
    per_view = []
    for index in args.views:
        if index >= len(dataset):
            continue
        sample = dataset[index]
        reference = np.asarray(sample.image, dtype=np.float32) / 255.0
        mask = np.asarray(sample.rgb_mask, dtype=bool)
        with torch.no_grad():
            rgb, _, _, _ = backend.render(
                params, sample, with_range=False,
                background_rgb=config["background_color"],
            )
        render = rgb.clamp(0, 1).cpu().numpy()

        e = energy_ratio(render, reference, mask)
        c = agreement(render, reference, mask)
        h = background_fraction(render, mask, background)
        per_view.append({"view": index, "image_id": sample.image_id,
                         "energy": e, "agreement": c, "holes": h,
                         "reading": classify(e, c, h)})
        print(f"{index:>6}{e:>10.3f}{c:>12.3f}{h * 100:>8.1f}%   {classify(e, c, h)}")

        if args.crops_dir is not None:
            args.crops_dir.mkdir(parents=True, exist_ok=True)
            y, x = highest_texture_crop(reference, mask, args.crop_size)
            pair = np.concatenate([reference[y:y + args.crop_size, x:x + args.crop_size],
                                   render[y:y + args.crop_size, x:x + args.crop_size]],
                                  axis=1)
            Image.fromarray((pair * 255).astype(np.uint8)).save(
                args.crops_dir / f"sharpness_{args.tag}_view{index}.png")

    if not per_view:
        print("no views measured", file=sys.stderr)
        return 1

    energy_mean = float(np.mean([v["energy"] for v in per_view]))
    agreement_mean = float(np.mean([v["agreement"] for v in per_view]))
    holes_mean = float(np.mean([v["holes"] for v in per_view]))
    reading = classify(energy_mean, agreement_mean, holes_mean)
    print(f"\n{'mean':>6}{energy_mean:>10.3f}{agreement_mean:>12.3f}"
          f"{holes_mean * 100:>8.1f}%   {reading}")
    print("photo reference: energy 1.000, agreement 1.000, holes 0.0%")
    if holes_mean > 0.10:
        print(f"\nWARNING: {holes_mean * 100:.1f}% of the masked frame is background. "
              f"Sharpness numbers are not comparable against a covered model - hole "
              f"rims raise energy and depress agreement on their own. Raise cap_max "
              f"before reading anything else here.")

    if args.output is not None:
        args.output.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": measured_step,
            "requested_steps": requested_steps,
            "split": args.split,
            "energy": energy_mean,
            "agreement": agreement_mean,
            "holes": holes_mean,
            "reading": reading,
            "per_view": per_view,
        }, indent=1), encoding="utf-8")
        print(f"\nreport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
