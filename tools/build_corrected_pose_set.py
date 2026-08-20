#!/usr/bin/env python3
"""Build a non-destructive Rig-aware pose set from transforms keyframe corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.poses.keyframe_correction import (
    PoseCorrectionConfig,
    build_corrected_pose_set,
    write_pose_set_outputs,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--transforms", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acceptance-metrics", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    transforms = json.loads(args.transforms.read_text(encoding="utf-8"))
    metrics = (
        json.loads(args.acceptance_metrics.read_text(encoding="utf-8"))
        if args.acceptance_metrics is not None
        else None
    )
    result = build_corrected_pose_set(
        dataset,
        transforms,
        transforms_sha256=sha256_file(args.transforms),
        config=PoseCorrectionConfig(),
        acceptance_metrics=metrics,
    )
    write_pose_set_outputs(args.output, result, force=args.force)
    print(
        f"pose set: anchors={result['anchor_filter']['accepted_rig_frames']}, "
        f"images={result['summary']['image_count']}, "
        f"default={result['acceptance']['default_pose_set']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
