#!/usr/bin/env python3
"""Advance a FACE4_BASE_READY gate after verified renderer masks are complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.pipeline.mipmap_gate import advance_renderer_mask_gate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-gate", required=True, type=Path)
    parser.add_argument("--train-renderer-mask", required=True, type=Path)
    parser.add_argument("--val-renderer-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    evidence = {
        "renderer_mask_train": {
            "path": str(args.train_renderer_mask.resolve()),
            "sha256": _sha256_file(args.train_renderer_mask),
        },
        "renderer_mask_val": {
            "path": str(args.val_renderer_mask.resolve()),
            "sha256": _sha256_file(args.val_renderer_mask),
        },
    }
    gate = advance_renderer_mask_gate(
        _read(args.frontend_gate),
        _read(args.train_renderer_mask),
        _read(args.val_renderer_mask),
        evidence=evidence,
    )
    _atomic_json(args.output, gate)
    print(
        f"MipMap gate: status={gate['status']}, "
        f"training_allowed={gate['training_allowed']}, "
        f"next={gate['next_required_stage']}, "
        f"sha256={gate['gate_manifest_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
