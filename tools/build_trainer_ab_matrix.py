#!/usr/bin/env python3
"""Build signed Gate 2 one-variable Trainer A/B configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.ab_matrix import (
    build_trainer_ab_matrix,
    verify_trainer_ab_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    manifest = build_trainer_ab_matrix(
        base,
        base_config_path=args.base_config,
        output_dir=args.output,
        experiment_id=args.experiment_id,
    )
    verify_trainer_ab_matrix(manifest, args.output)
    print(
        f"wrote {len(manifest['arms'])} signed A/B configs; "
        f"sha256={manifest['ab_matrix_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
