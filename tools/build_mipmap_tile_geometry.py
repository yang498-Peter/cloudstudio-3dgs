"""Materialize signed MipMap-compatible K=7/K=30 Tile geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

from cloudstudio_3dgs.training.mipmap_tile_geometry import (
    materialize_mipmap_tile_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-inputs", type=Path, required=True)
    parser.add_argument("--tile-inputs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=-1)
    args = parser.parse_args()
    manifest = materialize_mipmap_tile_geometry(
        args.tile_inputs,
        args.tile_inputs_root,
        args.output,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(
        f"Tile geometry: tiles={manifest['tile_count']}, "
        f"training_allowed={manifest['training_allowed']}, "
        f"sha256={manifest['tile_geometry_manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
