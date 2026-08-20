#!/usr/bin/env python3
"""Run the CloudStudio-owned raw-fisheye gsplat trainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.trainer import train_from_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    manifest = train_from_json(args.config)
    print(
        f"training complete: run={manifest['run_id']}, "
        f"steps={manifest['training']['completed_steps']}, "
        f"peak_vram={manifest['training']['peak_vram_bytes']} bytes, "
        f"sha256={manifest['run_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
