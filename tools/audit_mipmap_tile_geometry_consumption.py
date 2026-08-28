"""Audit exact K=7/K=30 Tile geometry consumption without starting CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cloudstudio_3dgs.training.mipmap_tile_geometry import (
    audit_mipmap_tile_geometry_consumption,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--tile-inputs", type=Path, required=True)
    parser.add_argument("--tile-inputs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_mipmap_tile_geometry_consumption(
        args.geometry_manifest,
        args.geometry_root,
        args.tile_inputs,
        args.tile_inputs_root,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Tile geometry consumption: tiles={audit['tile_count']}, "
        f"points={audit['total_point_count']}, "
        f"sha256={audit['consumption_audit_sha256']}"
    )


if __name__ == "__main__":
    main()
