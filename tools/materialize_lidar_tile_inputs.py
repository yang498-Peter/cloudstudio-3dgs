#!/usr/bin/env python3
"""Materialize full-LiDAR initialization PLYs for a signed adaptive Tile plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.tile_inputs import materialize_lidar_tile_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-plan", required=True, type=Path)
    parser.add_argument("--source-las", required=True, type=Path)
    parser.add_argument("--expected-las-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = materialize_lidar_tile_inputs(
        args.tile_plan,
        args.source_las,
        args.output,
        expected_point_cloud_sha256=args.expected_las_sha256,
        force=args.force,
    )
    print(
        f"Tile inputs: {manifest['tile_count']} tiles, "
        f"{sum(tile['initialization']['point_count'] for tile in manifest['tiles']):,} "
        f"halo-inclusive points, sha256={manifest['tile_inputs_manifest_sha256']}"
    )
    for tile in manifest["tiles"]:
        print(
            f"  {tile['name']}: {tile['initialization']['point_count']:,} points, "
            f"{tile['view_count']} views, {tile['recommended_training']['steps']} steps"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
