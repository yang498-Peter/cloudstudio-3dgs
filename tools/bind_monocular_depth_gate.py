#!/usr/bin/env python3
"""Record completed monocular depth caches on a surface-route training gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.pipeline.mipmap_gate import (
    bind_monocular_depth_into_surface_gate,
    load_and_verify_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-gate", required=True, type=Path)
    parser.add_argument("--train-da2", required=True, type=Path)
    parser.add_argument("--val-da2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gate, _ = load_and_verify_gate(args.training_gate)
    updated = bind_monocular_depth_into_surface_gate(
        gate,
        json.loads(args.train_da2.read_text(encoding="utf-8")),
        json.loads(args.val_da2.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"{updated['status']} | deferred now {updated['deferred_stages']} | "
        f"sha {updated['gate_manifest_sha256'][:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
