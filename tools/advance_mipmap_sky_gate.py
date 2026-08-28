#!/usr/bin/env python3
"""Advance the signed MipMap-aligned gate after independent sky preparation."""

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

from cloudstudio_3dgs.pipeline.mipmap_gate import advance_independent_sky_gate


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
    parser.add_argument("--da2-gate", type=Path, required=True)
    parser.add_argument("--train-sky", type=Path, required=True)
    parser.add_argument("--val-sky", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in (
            ("sky_train_evidence", args.train_sky),
            ("sky_val_evidence", args.val_sky),
            ("sky_initialization", args.initialization),
        )
    }
    gate = advance_independent_sky_gate(
        _read(args.da2_gate),
        _read(args.train_sky),
        _read(args.val_sky),
        _read(args.initialization),
        evidence=evidence,
    )
    _atomic_json(args.output, gate)
    print(
        f"MipMap sky gate: status={gate['status']}, "
        f"next={gate['next_required_stage']}, "
        f"sha256={gate['gate_manifest_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
