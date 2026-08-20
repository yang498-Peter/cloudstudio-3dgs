#!/usr/bin/env python3
"""Build a deterministic Rig-frame train/validation split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.evaluation.splits import (
    SplitConfig,
    build_split_manifest,
    write_split_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=["temporal_block", "spatial_block", "manual"], default="temporal_block"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temporal-block-count", type=int, default=10)
    parser.add_argument("--spatial-cell-m", type=float, default=2.0)
    parser.add_argument("--nearest-train-warning-m", type=float, default=0.25)
    parser.add_argument("--golden-rig-frames", type=int, default=8)
    parser.add_argument("--manual", type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    manual = None
    if args.manual is not None:
        manual = json.loads(args.manual.read_text(encoding="utf-8"))
    result = build_split_manifest(
        dataset,
        SplitConfig(
            mode=args.mode,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            temporal_block_count=args.temporal_block_count,
            spatial_cell_m=args.spatial_cell_m,
            nearest_train_warning_m=args.nearest_train_warning_m,
            golden_rig_frames=args.golden_rig_frames,
        ),
        manual=manual,
    )
    write_split_manifest(args.output, result)
    print(
        f"split: {result['summary']['train_rig_frames']} train + "
        f"{result['summary']['val_rig_frames']} val rig frames, "
        f"leakage warnings={result['leakage']['warning_count']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
