"""Audit real Tile Face4 crop and DA2 consumption without starting CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cloudstudio_3dgs.training.tile_face4_consumption import (
    audit_tile_face4_consumption,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-inputs", type=Path, required=True)
    parser.add_argument("--tile-inputs-root", type=Path, required=True)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--face-cache-root", type=Path, required=True)
    parser.add_argument("--renderer-mask-manifest", type=Path, required=True)
    parser.add_argument("--mono-depth-manifest", type=Path, required=True)
    parser.add_argument("--mono-depth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_tile_face4_consumption(
        tile_inputs_path=args.tile_inputs,
        tile_inputs_root=args.tile_inputs_root,
        face_manifest_path=args.face_manifest,
        face_cache_root=args.face_cache_root,
        renderer_mask_manifest_path=args.renderer_mask_manifest,
        mono_depth_manifest_path=args.mono_depth_manifest,
        mono_depth_root=args.mono_depth_root,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Tile Face4 consumption: tiles={report['tile_count']}, "
        f"views={report['total_view_instances_with_overlap']}, "
        f"sha256={report['tile_face4_consumption_audit_sha256']}"
    )


if __name__ == "__main__":
    main()
