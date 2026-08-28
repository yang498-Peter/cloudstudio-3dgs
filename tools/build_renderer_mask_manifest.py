#!/usr/bin/env python3
"""Build a signed Face4 renderer-mask manifest without duplicating mask PNGs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.renderer_masks import build_renderer_mask_manifest


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--face-cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-verify-artifacts", action="store_true")
    args = parser.parse_args()

    face = json.loads(args.face_manifest.read_text(encoding="utf-8"))
    manifest = build_renderer_mask_manifest(
        face,
        args.face_cache_root,
        verify_artifacts=not args.no_verify_artifacts,
    )
    _atomic_json(args.output, manifest)
    summary = manifest["summary"]
    print(
        "Renderer masks: "
        f"split={manifest['split']}, images={summary['image_count']}, "
        f"faces={summary['face_sample_count']}, "
        f"sha256={manifest['renderer_mask_manifest_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
