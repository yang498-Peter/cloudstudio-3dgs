#!/usr/bin/env python3
"""Build deterministic per-image sparse LiDAR ray-range caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.depth_cache import build_depth_cache
from cloudstudio_3dgs.geometry.lidar_projection import DepthProjectionConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--max-points",
        type=int,
        help="deterministic input limit; prefer a PR-04 voxel PLY instead of LAS stride",
    )
    parser.add_argument("--min-range-m", type=float, default=0.2)
    parser.add_argument("--max-range-m", type=float, default=80.0)
    parser.add_argument("--max-theta-deg", type=float, default=95.0)
    parser.add_argument(
        "--compact-provenance",
        action="store_true",
        help=(
            "omit source_index/support_count arrays and reconstruct their unused "
            "-1/0 sentinels while loading"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    masks = json.loads(args.mask_manifest.read_text(encoding="utf-8"))
    result = build_depth_cache(
        dataset,
        masks,
        args.mask_manifest.parent,
        args.point_cloud,
        args.output,
        config=DepthProjectionConfig(
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
            max_theta_deg=args.max_theta_deg,
        ),
        workers=args.workers,
        max_images=args.max_images,
        max_points=args.max_points,
        compact_provenance=args.compact_provenance,
        force=args.force,
    )
    print(
        f"depth cache: {result['summary']['image_count']} images, "
        f"complete={result['complete_dataset']}, key={result['cache_key']} -> "
        f"{args.output / 'depth_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
