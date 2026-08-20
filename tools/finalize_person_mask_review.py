#!/usr/bin/env python3
"""Verify visual decisions and write a signed person-mask review report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.person_masks import build_person_mask_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-mask-manifest", type=Path, required=True)
    parser.add_argument("--person-mask-root", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"person mask review output already exists: {args.output}")
    manifest = json.loads(args.person_mask_manifest.read_text(encoding="utf-8"))
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    report = build_person_mask_review(manifest, args.person_mask_root, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "person mask review complete: "
        f"status={report['status']} reviewed={report['summary']['reviewed']} "
        f"sha256={report['person_mask_review_sha256']}"
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
