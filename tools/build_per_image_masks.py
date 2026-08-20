#!/usr/bin/env python3
"""CLI wrapper for the canonical per-image mask manifest builder."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.mask_manifest import main


if __name__ == "__main__":
    raise SystemExit(main())
