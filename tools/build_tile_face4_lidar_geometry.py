#!/usr/bin/env python3
"""Build one Tile's signed Face4 LiDAR range sidecar from the reusable cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.tile_face_lidar_geometry import (
    materialize_tile_face_lidar_geometry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument("--tile-id", required=True, type=int)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-geometry-manifest", required=True, type=Path)
    parser.add_argument("--source-geometry-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = materialize_tile_face_lidar_geometry(
        tile_inputs_path=args.tile_inputs,
        tile_inputs_root=args.tile_inputs_root,
        tile_id=args.tile_id,
        face_manifest_path=args.face_manifest,
        dataset_manifest_path=args.dataset_manifest,
        source_geometry_manifest_path=args.source_geometry_manifest,
        source_geometry_root=args.source_geometry_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "MATERIALIZED",
                "tile_id": manifest["tile_id"],
                "view_count": manifest["view_count"],
                "nonempty_view_count": manifest["nonempty_view_count"],
                "tile_valid_pixels": manifest["tile_valid_pixels"],
                "face_lidar_geometry_manifest_sha256": manifest[
                    "face_lidar_geometry_manifest_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
