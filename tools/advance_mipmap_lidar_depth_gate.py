#!/usr/bin/env python3
"""Advance the signed MipMap-aligned gate after complete LiDAR depth."""

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

from cloudstudio_3dgs.pipeline.mipmap_gate import advance_lidar_depth_gate


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer-gate", type=Path, required=True)
    parser.add_argument("--depth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    renderer_gate = _read(args.renderer_gate)
    depth_manifest = _read(args.depth_manifest)
    gate = advance_lidar_depth_gate(
        renderer_gate,
        depth_manifest,
        evidence={
            "lidar_depth": {
                "path": str(args.depth_manifest.resolve()),
                "sha256": _file_sha(args.depth_manifest),
            }
        },
    )
    _atomic_json(args.output, gate)
    print(
        f"MipMap LiDAR-depth gate: status={gate['status']}, "
        f"next={gate['next_required_stage']}, "
        f"sha256={gate['gate_manifest_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
