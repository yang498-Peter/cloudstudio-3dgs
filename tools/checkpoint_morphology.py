#!/usr/bin/env python3
"""Axis / opacity morphology of a checkpoint or a Gaussian PLY.

The delivery target is a set of per-gaussian numbers (short/mid/long axis,
axis ratios, opacity spread) measured on the reference delivery; every arm is
read against the same numbers on the same convention. The bracketed values in
the text report are those targets.

    python tools/checkpoint_morphology.py RUN/arm/checkpoints/latest.pt --label arm
    python tools/checkpoint_morphology.py delivery.ply --json morph.json

Checkpoints need torch; a PLY is read with the repository's own reader and
needs only numpy, so an exported delivery can be measured on a CPU host.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1

# Reference delivery, measured earlier on this exact convention.
REFERENCE_TARGETS: dict[str, float] = {
    "short_p50_mm": 0.43,
    "mid_p50_mm": 1.31,
    "long_p50_mm": 4.41,
    "long_p95_mm": 55.8,
    "max_min_p50": 10.2,
    "max_mid_p50": 3.06,
    "opacity_p50": 0.197,
    "opacity_frac_lt_0_1": 0.18,
}

# Quantiles on tens of millions of gaussians are memory-bound; a 2M sample
# reproduces the median to the third decimal.
SAMPLE_LIMIT = 1 << 21
SAMPLE_SEED = 0


def load_checkpoint(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """(step, linear scales (N,3), opacities in [0,1]) from a trainer checkpoint."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    params = payload["params"]
    scales = np.exp(params["scales"].detach().float().cpu().numpy())
    opacities = torch.sigmoid(params["opacities"].detach().float().flatten()).cpu().numpy()
    return int(payload.get("step", -1)), scales, opacities


def load_ply(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Same tensors from an exported 3DGS PLY (log scales, logit opacity)."""
    from tools.gaussian_health import read_ply_records

    records = read_ply_records(path)
    scales = np.exp(np.stack([records[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32))
    opacities = 1.0 / (1.0 + np.exp(-np.asarray(records["opacity"], dtype=np.float32)))
    return -1, scales, opacities


def load_morphology_inputs(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".ply":
        return load_ply(path)
    return load_checkpoint(path)


def compute_morphology(
    scales: np.ndarray,
    opacities: np.ndarray,
    *,
    sample_limit: int = SAMPLE_LIMIT,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """Axis and opacity statistics; quantiles on a fixed-seed subsample."""
    scales = np.asarray(scales, dtype=np.float32).reshape(-1, 3)
    opacities = np.asarray(opacities, dtype=np.float32).reshape(-1)
    count = int(scales.shape[0])
    if count == 0 or opacities.shape[0] != count:
        raise ValueError(f"need matching non-empty scales/opacities, got {scales.shape} and {opacities.shape}")
    sorted_axes = np.sort(scales, axis=1)
    short, mid, long = sorted_axes[:, 0], sorted_axes[:, 1], sorted_axes[:, 2]
    if count > sample_limit:
        index = np.random.default_rng(seed).choice(count, size=sample_limit, replace=False)
    else:
        index = np.arange(count)

    def quantile(values: np.ndarray, level: float) -> float:
        return float(np.quantile(values[index].astype(np.float64), level))

    tiny = np.maximum(short, 1e-9)
    return {
        "count": count,
        "sampled": int(index.shape[0]),
        "short_p50_mm": quantile(short, 0.5) * 1000.0,
        "mid_p50_mm": quantile(mid, 0.5) * 1000.0,
        "long_p50_mm": quantile(long, 0.5) * 1000.0,
        "long_p95_mm": quantile(long, 0.95) * 1000.0,
        "max_min_p50": quantile(long / tiny, 0.5),
        "max_mid_p50": quantile(long / np.maximum(mid, 1e-9), 0.5),
        "opacity_p50": quantile(opacities, 0.5),
        "opacity_p95": quantile(opacities, 0.95),
        "opacity_frac_lt_0_1": float(np.mean(opacities < 0.1)),
        "opacity_frac_gt_0_9": float(np.mean(opacities > 0.9)),
        "long_frac_gt_10mm": float(np.mean(long > 0.01)),
        "long_frac_gt_20mm": float(np.mean(long > 0.02)),
        "long_frac_gt_50mm": float(np.mean(long > 0.05)),
    }


def format_report(label: str, step: int, stats: dict[str, Any]) -> str:
    """The four-line block the arm ledger appends; targets in brackets."""
    ref = REFERENCE_TARGETS
    lines = [
        f"== {label}  step {step}  N={stats['count']:,}",
        (
            f"  short p50 {stats['short_p50_mm']:.3f}mm [{ref['short_p50_mm']}]   "
            f"mid p50 {stats['mid_p50_mm']:.3f}mm [{ref['mid_p50_mm']}]   "
            f"long p50 {stats['long_p50_mm']:.3f}mm [{ref['long_p50_mm']}]  "
            f"long p95 {stats['long_p95_mm']:.1f}mm [{ref['long_p95_mm']}]"
        ),
        (
            f"  max/min p50 {stats['max_min_p50']:.2f} [{ref['max_min_p50']}]   "
            f"max/mid p50 {stats['max_mid_p50']:.2f} [{ref['max_mid_p50']}]"
        ),
        (
            f"  opacity p50 {stats['opacity_p50']:.3f} [{ref['opacity_p50']}]  "
            f"p95 {stats['opacity_p95']:.3f}   "
            f"frac<0.1 {stats['opacity_frac_lt_0_1']:.3f} [{ref['opacity_frac_lt_0_1']}]   "
            f"frac>0.9 {stats['opacity_frac_gt_0_9']:.3f}"
        ),
        (
            f"  long>10mm {stats['long_frac_gt_10mm']:.3f}  "
            f"long>20mm {stats['long_frac_gt_20mm']:.3f}  "
            f"long>50mm {stats['long_frac_gt_50mm']:.3f}"
        ),
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", type=Path, help="checkpoint .pt or Gaussian .ply")
    parser.add_argument("--label", help="block label (default: the path)")
    parser.add_argument("--json", type=Path, help="also write the statistics as JSON here")
    parser.add_argument("--sample-limit", type=int, default=SAMPLE_LIMIT, help="quantile subsample size")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.path.exists():
        print(f"checkpoint_morphology: not found: {args.path}", file=sys.stderr)
        return 2
    label = args.label or str(args.path)
    step, scales, opacities = load_morphology_inputs(args.path)
    stats = compute_morphology(scales, opacities, sample_limit=args.sample_limit)
    sys.stdout.write(format_report(label, step, stats))
    if args.json is not None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "label": label,
            "path": str(args.path),
            "step": step,
            "reference_targets": REFERENCE_TARGETS,
            "stats": stats,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json.with_suffix(args.json.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=1), encoding="utf-8")
        os.replace(temporary, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
