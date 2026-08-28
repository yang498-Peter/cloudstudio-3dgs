"""Build the signed, non-training LiDAR-first Face4 parameter specification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.training.mipmap_type2_contract import (
    build_high_type2_parameter_spec,
)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-gate", type=Path, required=True)
    parser.add_argument("--tile-plan", type=Path, required=True)
    parser.add_argument("--tile-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace parameter specification: {args.output}")
    spec = build_high_type2_parameter_spec(
        _load(args.upstream_gate),
        _load(args.tile_plan),
        _load(args.tile_inputs),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"LiDAR-first parameter spec: status={spec['status']}, "
        f"training_allowed={spec['training_allowed']}, "
        f"tiles={len(spec['tiles'])}, sha256={spec['parameter_spec_sha256']}"
    )


if __name__ == "__main__":
    main()
