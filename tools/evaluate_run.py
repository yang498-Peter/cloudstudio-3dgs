#!/usr/bin/env python3
"""Generate quality_report.json/html for one signed 3DGS run manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.evaluation.quality_report import build_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "squeeze", "vgg"])
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    report = build_quality_report(
        run,
        split,
        args.run_manifest.parent,
        args.output,
        run_lpips_metric=args.lpips,
        lpips_net=args.lpips_net,
        lpips_device=args.lpips_device,
        force=args.force,
    )
    print(
        f"quality report: status={report['status']}, frames={report['summary']['frame_count']} "
        f"-> {args.output / 'quality_report.html'}"
    )
    return 2 if args.require_complete and report["status"] != "COMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
