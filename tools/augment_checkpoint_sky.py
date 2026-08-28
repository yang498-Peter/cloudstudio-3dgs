#!/usr/bin/env python3
"""Append a deterministic trainable far-field sky cap to a warm-start checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.sky_layer import (
    SkyLayerConfig,
    augment_checkpoint_with_sky,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--radius-m", type=float, default=100.0)
    parser.add_argument("--scale-m", type=float, default=0.85)
    parser.add_argument("--opacity", type=float, default=0.02)
    parser.add_argument("--rgb", type=float, nargs=3, default=(0.45, 0.58, 0.78))
    parser.add_argument("--min-world-z-direction", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = augment_checkpoint_with_sky(
        args.checkpoint,
        args.dataset_manifest,
        args.output,
        args.report,
        SkyLayerConfig(
            count=args.count,
            radius_m=args.radius_m,
            scale_m=args.scale_m,
            opacity=args.opacity,
            rgb=tuple(args.rgb),
            min_world_z_direction=args.min_world_z_direction,
        ),
        force=args.force,
    )
    print(
        f"appended {report['sky_gaussian_count']:,} sky gaussians -> "
        f"{report['total_gaussian_count']:,} total, "
        f"{report['output_checkpoint_bytes'] / 1e6:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
