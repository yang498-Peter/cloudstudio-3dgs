#!/usr/bin/env python3
"""Run the remaining snow V28 Tile protocols serially and fail closed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument(
        "--tile-order", type=int, nargs="+", default=[2, 4, 0, 3]
    )
    args = parser.parse_args()
    review_stops = {0: 3332, 2: 3290, 3: 4249, 4: 3535}
    status_path = args.outputs_root / "full_area_v28_status.json"
    status = {
        "schema_version": 1,
        "kind": "snow_full_area_v28_status",
        "status": "RUNNING",
        "tile_order": args.tile_order,
        "completed_tiles": [],
        "started_unix": time.time(),
    }
    _write(status_path, status)
    for tile_id in args.tile_order:
        if tile_id not in review_stops:
            raise ValueError(f"Tile_{tile_id} has no prepared full-area protocol")
        config = (
            args.outputs_root
            / f"tile{tile_id}_full_area_protocol_v28b"
            / "fixed_topology_a0.config.json"
        )
        if not config.is_file():
            raise FileNotFoundError(f"prepared Tile config is missing: {config}")
        protocol_root = args.outputs_root / f"tile{tile_id}_v28_protocol"
        status.update({"active_tile": tile_id, "active_stage": "tile_protocol"})
        _write(status_path, status)
        command = [
            sys.executable,
            str(ROOT / "tools" / "run_tile_v28_protocol.py"),
            "--tile-id",
            str(tile_id),
            "--a0-config",
            str(config),
            "--upstream-gate",
            str(args.upstream_gate),
            "--extension",
            str(args.extension),
            "--output-root",
            str(protocol_root),
        ]
        code = subprocess.run(command, cwd=ROOT, check=False).returncode
        if code:
            status.update(
                {
                    "status": "FAIL",
                    "failed_tile": tile_id,
                    "return_code": int(code),
                    "failed_unix": time.time(),
                }
            )
            _write(status_path, status)
            return int(code)
        status["completed_tiles"].append(tile_id)
        _write(status_path, status)
    status.update(
        {
            "status": "TILES_PASS",
            "active_tile": None,
            "active_stage": "core_merge_pending",
            "completed_unix": time.time(),
        }
    )
    _write(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
