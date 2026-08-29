#!/usr/bin/env python3
"""Promote signed fixed-topology readiness into a narrowly scoped training gate."""

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

from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--promoted-config-dir", required=True, type=Path)
    args = parser.parse_args()

    upstream = _read_json(args.upstream_gate)
    readiness = _read_json(args.readiness)
    plan = _read_json(args.plan)
    arm_configs: dict[str, dict] = {}
    config_paths: dict[str, Path] = {}
    for arm in plan.get("arms", []):
        arm_name = str(arm["arm"])
        config_path = args.plan.parent / str(arm["path"])
        config = _read_json(config_path)
        arm_configs[arm_name] = config
        config_paths[arm_name] = config_path

    gate = advance_fixed_topology_evaluation_gate(
        upstream, readiness, plan, arm_configs
    )
    _atomic_json(args.output_gate, gate)

    for arm_name, config in arm_configs.items():
        promoted = dict(config)
        promoted["mipmap_pipeline_gate"] = args.output_gate.resolve().as_posix()
        output_config = args.promoted_config_dir / config_paths[arm_name].name
        _atomic_json(output_config, promoted)

    print(
        "fixed-topology evaluation promoted: "
        f"gate={args.output_gate}, sha256={gate['gate_manifest_sha256']}, "
        f"arms={','.join(sorted(arm_configs))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
