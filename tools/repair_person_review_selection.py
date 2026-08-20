#!/usr/bin/env python3
"""Fill duplicate review selections without rerunning person segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.person_masks import repair_person_review_selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-mask-manifest", type=Path, required=True)
    parser.add_argument("--person-mask-root", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.person_mask_manifest.read_text(encoding="utf-8"))
    repaired = repair_person_review_selection(
        manifest, args.recording_root, args.person_mask_root
    )
    print(
        "person review selection repaired: "
        f"samples={len(repaired['review_samples'])} "
        f"sha256={repaired['person_mask_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
