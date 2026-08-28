#!/usr/bin/env python3
"""Advance the signed MipMap-aligned gate after complete DA2 caches."""

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

from cloudstudio_3dgs.pipeline.mipmap_gate import advance_da2_depth_gate


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-gate", type=Path, required=True)
    parser.add_argument("--train-da2", type=Path, required=True)
    parser.add_argument("--val-da2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {
        "da2_train": {
            "path": str(args.train_da2.resolve()),
            "sha256": _sha256_file(args.train_da2),
        },
        "da2_val": {
            "path": str(args.val_da2.resolve()),
            "sha256": _sha256_file(args.val_da2),
        },
    }
    gate = advance_da2_depth_gate(
        _read(args.lidar_gate),
        _read(args.train_da2),
        _read(args.val_da2),
        evidence=evidence,
    )
    _atomic_json(args.output, gate)
    print(
        f"MipMap DA2 gate: status={gate['status']}, "
        f"next={gate['next_required_stage']}, "
        f"sha256={gate['gate_manifest_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
