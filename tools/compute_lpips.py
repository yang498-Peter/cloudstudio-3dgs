"""LPIPS perceptual metric over full validation sets of probe runs (offline, honest).

Complements ``tools/compare_validation_metrics.py`` (PSNR/SSIM/depth) with the
perceptual axis that Mip-Splatting-class papers report. Same directory
convention: ``<runs-root>/<run>/evaluation/{id}_rendered.png``,
``{id}_reference.png``, ``{id}_mask.png``.

Backends, in priority order (recorded per run as ``backend``/``calibrated``):

1. ``lpips`` package (``lpips.LPIPS``) -- the reference implementation with the
   authors' calibrated linear head. Reports ``calibrated: true``; values are
   comparable with published LPIPS numbers.
2. Torchvision-only fallback -- the same ImageNet backbone (alex/vgg) with
   unit-normalised features and *equal* layer weights instead of the authors'
   fitted 1x1 linear head. This is NOT standard LPIPS: it reports
   ``calibrated: false`` and ``backend: "torchvision_uncalibrated"``. Use it
   only for relative A/B comparison inside one sweep, never as a published
   LPIPS figure.
3. Neither available -> fail closed with an explicit error (see
   ``WeightsUnavailableError``); no silent degradation to a fake number.

Mask policy
-----------
LPIPS is a whole-image perceptual metric with a receptive field far larger than
one pixel, so masked pixels cannot simply be dropped the way masked PSNR/SSIM
drops them. Instead the invalid pixels (mask <= 127) are set to the SAME
constant (white, matching our white-background training) in BOTH the rendered
and the reference image. The invalid region is then pixel-identical between the
two inputs and contributes ~0 to the feature-space distance, while remaining
valid image content that leaks across the region boundary is still measured.
Recorded as ``mask_policy: "invalid_pixels_set_to_white_in_both"``, alongside
the per-frame valid-pixel fraction so a run with a shrinking valid area cannot
quietly look better.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_RUNS_ROOT = Path(r"C:\Peter\3dgs-runs\probes")
MASK_POLICY = "invalid_pixels_set_to_white_in_both"
# White background, matching the white-bg training/eval convention.
INVALID_FILL = 1.0
MASK_VALID_THRESHOLD = 127

# LPIPS input normalisation (Zhang et al.); identical in both backends so the
# fallback stays as close to the reference pipeline as the missing head allows.
_LPIPS_SHIFT = (-0.030, -0.088, -0.188)
_LPIPS_SCALE = (0.458, 0.448, 0.450)

# Layer taps used by the reference LPIPS implementation.
_ALEX_TAPS = (0, 1, 2, 3, 4)
_VGG_TAPS = (0, 1, 2, 3, 4)


class WeightsUnavailableError(RuntimeError):
    """Raised when no pretrained perceptual backbone can be obtained."""


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as float32 HxWx3 in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    """Load a validity mask as bool HxW (True == valid)."""
    return np.asarray(Image.open(path).convert("L")) > MASK_VALID_THRESHOLD


def apply_mask_policy(image: np.ndarray, mask: np.ndarray, fill: float = INVALID_FILL) -> np.ndarray:
    """Set invalid pixels to a constant so both inputs match there exactly.

    ``image`` is HxWx3 in [0, 1]; ``mask`` is HxW bool (True == valid). Applying
    this to the rendered and the reference frame with the same ``fill`` makes
    the invalid region identical in both, so it contributes ~0 to LPIPS.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got shape {image.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image {image.shape[:2]}")
    out = image.copy()
    out[~mask] = fill
    return out


def to_lpips_tensor(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """HxWx3 in [0, 1] -> 1x3xHxW in [-1, 1] (the LPIPS input convention)."""
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device)


class _UncalibratedLpips(torch.nn.Module):
    """Torchvision-backbone perceptual distance WITHOUT the fitted linear head.

    Unit-normalises each feature map over channels and averages the squared
    difference per layer, then sums layers with equal weight. Structurally this
    is LPIPS minus calibration, so it tracks LPIPS monotonically in practice but
    its absolute scale is arbitrary -- always reported as ``calibrated: false``.
    """

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        from torchvision import models

        if net == "alex":
            weights = models.AlexNet_Weights.IMAGENET1K_V1
            features = models.alexnet(weights=weights).features
            slices = [(0, 2), (2, 5), (5, 8), (8, 10), (10, 12)]
        elif net == "vgg":
            weights = models.VGG16_Weights.IMAGENET1K_V1
            features = models.vgg16(weights=weights).features
            slices = [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]
        else:
            raise ValueError(f"unsupported net for fallback backend: {net!r}")

        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(*[features[i] for i in range(start, stop)]) for start, stop in slices
        )
        self.register_buffer("shift", torch.tensor(_LPIPS_SHIFT).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor(_LPIPS_SCALE).view(1, 3, 1, 1))
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _unit_normalize(feat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(feat**2, dim=1, keepdim=True))
        return feat / (norm + eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (pred - self.shift) / self.scale
        target = (target - self.shift) / self.scale
        total = torch.zeros(pred.shape[0], device=pred.device)
        for block in self.blocks:
            pred = block(pred)
            target = block(target)
            diff = (self._unit_normalize(pred) - self._unit_normalize(target)) ** 2
            total = total + diff.sum(dim=1).mean(dim=(1, 2))
        return total


class LpipsScorer:
    """Wraps whichever perceptual backend is actually available on this box."""

    def __init__(self, net: str = "alex", device: str = "cpu", allow_fallback: bool = True) -> None:
        self.net = net
        self.device = device
        self.backend, self.calibrated, self._model = _build_model(net, device, allow_fallback)

    def distance(self, rendered: np.ndarray, reference: np.ndarray) -> float:
        """Perceptual distance between two HxWx3 [0, 1] images (lower is better)."""
        with torch.no_grad():
            value = self._model(
                to_lpips_tensor(rendered, self.device),
                to_lpips_tensor(reference, self.device),
            )
        return float(value.reshape(-1)[0].item())

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "net": self.net,
            "device": self.device,
            "calibrated": self.calibrated,
        }


def _build_model(net: str, device: str, allow_fallback: bool):
    """Return ``(backend_name, calibrated, model)`` or fail closed."""
    reasons = []
    try:
        import lpips as lpips_pkg

        model = lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return "lpips_package", True, model
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable here"
        reasons.append(f"lpips package unusable: {type(exc).__name__}: {exc}")

    if allow_fallback:
        try:
            model = _UncalibratedLpips(net=net).to(device).eval()
            return "torchvision_uncalibrated", False, model
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"torchvision fallback unusable: {type(exc).__name__}: {exc}")

    raise WeightsUnavailableError(
        "No pretrained perceptual backbone is available, so no LPIPS value can be "
        "produced (failing closed rather than reporting a meaningless number).\n"
        + "\n".join(f"  - {r}" for r in reasons)
        + "\nTo fix, on a machine with network access run:\n"
        f"  pip install lpips           # ships the calibrated v0.1 linear head\n"
        f"  python -c \"import torchvision.models as m; m.alexnet(weights='IMAGENET1K_V1')\"\n"
        "then copy the torch hub cache (TORCH_HOME, default %USERPROFILE%\\.cache\\torch)\n"
        "and the lpips package weights to this machine."
    )


def frame_ids(eval_dir: Path) -> list[str]:
    """Validation frame ids, matching compare_validation_metrics.py's convention."""
    return sorted({p.name.rsplit("_", 1)[0] for p in eval_dir.glob("*_rendered.png")})


def score_run(eval_dir: Path, scorer: LpipsScorer) -> dict:
    """Score every validation frame of one run; frames are processed one by one."""
    ids = frame_ids(eval_dir)
    if not ids:
        raise FileNotFoundError(f"no *_rendered.png found under {eval_dir}")

    per_frame: dict[str, float] = {}
    valid_fractions: dict[str, float] = {}
    for image_id in ids:
        reference = load_image_rgb(eval_dir / f"{image_id}_reference.png")
        rendered = load_image_rgb(eval_dir / f"{image_id}_rendered.png")
        mask_path = eval_dir / f"{image_id}_mask.png"
        if mask_path.exists():
            mask = load_mask(mask_path)
        else:
            mask = np.ones(reference.shape[:2], dtype=bool)
        if rendered.shape != reference.shape:
            raise ValueError(
                f"{image_id}: rendered {rendered.shape} != reference {reference.shape}"
            )
        valid_fractions[image_id] = float(mask.mean())
        per_frame[image_id] = scorer.distance(
            apply_mask_policy(rendered, mask),
            apply_mask_policy(reference, mask),
        )

    values = np.array([per_frame[i] for i in ids], dtype=np.float64)
    fractions = np.array([valid_fractions[i] for i in ids], dtype=np.float64)
    return {
        "frames": len(ids),
        "lpips_mean": float(values.mean()),
        "lpips_median": float(np.median(values)),
        "lpips_p90": float(np.percentile(values, 90)),
        "lpips_per_frame": per_frame,
        "valid_pixel_fraction_mean": float(fractions.mean()),
        "valid_pixel_fraction_min": float(fractions.min()),
        "valid_pixel_fraction_per_frame": valid_fractions,
        "mask_policy": MASK_POLICY,
        "invalid_fill": INVALID_FILL,
        **scorer.describe(),
    }


def print_table(results: dict) -> None:
    print(
        f"{'probe':<18} {'frames':>6} {'LPIPS':>7} {'med':>7} {'p90':>7} "
        f"{'valid':>6} {'backend':>24} {'calib':>6}"
    )
    for probe, r in results.items():
        print(
            f"{probe:<18} {r['frames']:6d} {r['lpips_mean']:7.4f} {r['lpips_median']:7.4f} "
            f"{r['lpips_p90']:7.4f} {r['valid_pixel_fraction_mean']:6.3f} "
            f"{r['backend']:>24} {str(r['calibrated']).lower():>6}"
        )
    if any(not r["calibrated"] for r in results.values()):
        print(
            "WARNING: uncalibrated backend in use -- these are NOT standard LPIPS "
            "values and must not be compared against published numbers."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device; defaults to cpu so acceptance runs never contend with training",
    )
    parser.add_argument("--net", default="alex", choices=("alex", "vgg"))
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="fail closed unless the calibrated lpips package is usable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scorer = LpipsScorer(net=args.net, device=args.device, allow_fallback=not args.no_fallback)
    results = {run: score_run(args.runs_root / run / "evaluation", scorer) for run in args.runs}
    print_table(results)
    output = args.output or (args.runs_root / "lpips_comparison.json")
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
