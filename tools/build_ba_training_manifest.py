"""Build a signed Trainer dataset manifest from an accepted BA or independent AT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.ba.training_manifest import (
    build_ba_training_manifest,
    build_independent_at_training_manifest,
)
from cloudstudio_3dgs.data.manifest import write_manifest_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    report_group = parser.add_mutually_exclusive_group(required=True)
    report_group.add_argument("--ba-report", type=Path)
    report_group.add_argument("--independent-at-report", type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if args.independent_at_report is not None:
        report = json.loads(args.independent_at_report.read_text(encoding="utf-8"))
        derived = build_independent_at_training_manifest(
            dataset,
            split,
            report,
            args.candidate_model,
        )
    else:
        report = json.loads(args.ba_report.read_text(encoding="utf-8"))
        derived = build_ba_training_manifest(
            dataset,
            split,
            report,
            args.candidate_model,
        )
    destination = write_manifest_atomic(derived, args.output, force=args.force)
    lineage = derived["training_lineage"]
    source = (
        f"stage={lineage['ba_stage']}"
        if "ba_stage" in lineage
        else f"algorithm={lineage['independent_at_algorithm_version']}"
    )
    print(
        f"BA training manifest: {source}, "
        f"train={len(split['splits']['train'])}, val={len(split['splits']['val'])}, "
        f"sha256={derived['manifest_sha256']} -> {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
