#!/usr/bin/env python3
"""Build the signed core-only ownership contract used before Tile training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest
from cloudstudio_3dgs.training.tile_ownership import build_core_ownership_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.tile_inputs.read_text(encoding="utf-8"))
    manifest_sha = verify_tile_inputs_manifest(
        manifest, root=args.tile_inputs_root, verify_artifacts=True
    )
    contract = build_core_ownership_contract(
        tiles=manifest["tiles"], tile_inputs_manifest_sha256=manifest_sha
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
