#!/usr/bin/env python3
"""Safely rebind a raw checkpoint to a verified derived sampling identity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.sampling_rebind import (
    rebind_checkpoint_sampling_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--target-lineage-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = rebind_checkpoint_sampling_identity(
        args.source_checkpoint,
        args.target_lineage_checkpoint,
        args.output,
        args.report,
        force=args.force,
    )
    print(
        f"rebound step {report['source_step']} to face manifest "
        f"{report['target_face_manifest_sha256']}, "
        f"{report['output_checkpoint_bytes'] / 1e6:.1f} MB -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

