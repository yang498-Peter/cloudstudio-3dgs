#!/usr/bin/env python3
"""Authorize frozen-topology training on the surface route.

The evidence this gate demands is narrower than the High/type-2 contract's,
and deliberately so: that contract describes monocular depth, mesh
supervision, a bilateral grid and SIFT refinement, none of which this route
runs. What it requires instead is that the upstream data is signed, that every
point belongs to exactly one tile, and that a full-resolution smoke consumed
the real inputs and reported the population and peak the run will actually
carry. Freezing topology is what makes the shorter set sufficient.
"""

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

from cloudstudio_3dgs.pipeline.mipmap_gate import (  # noqa: E402
    promote_surface_frozen_training_gate,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-gate", type=Path, required=True)
    parser.add_argument("--ownership-contract", type=Path, required=True)
    parser.add_argument(
        "--fullres-smoke-manifest",
        type=Path,
        required=True,
        help="run_manifest.json from a factor-1 strict-fixed smoke",
    )
    parser.add_argument("--tile-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    smoke = _read(args.fullres_smoke_manifest)
    contract = smoke.get("trainer_contract", {})
    training = smoke.get("training", {})
    # Flatten the fields the gate checks so the check reads against one shape
    # regardless of where the manifest happens to nest them.
    smoke_view = {
        "factor": contract.get("factor"),
        "topology_policy": contract.get("topology_policy", {}),
        "peak_vram_bytes": training.get("peak_vram_bytes"),
        "gaussian_count": training.get("gaussian_count"),
    }

    plan = _read(args.tile_plan)
    tile = plan["tiles"][0]

    gate = promote_surface_frozen_training_gate(
        _read(args.upstream_gate),
        ownership_contract=_read(args.ownership_contract),
        fullres_smoke=smoke_view,
        initialization_count=int(smoke_view["gaussian_count"]),
        capacity_cap=int(contract.get("cap_max", 0)),
        evidence={
            "fullres_smoke": {
                "path": str(args.fullres_smoke_manifest.resolve()),
                "sha256": _sha256_file(args.fullres_smoke_manifest),
                "run_id": smoke.get("run_id"),
                "peak_vram_bytes": smoke_view["peak_vram_bytes"],
                "gaussian_count": smoke_view["gaussian_count"],
            },
            "core_ownership": {
                "path": str(args.ownership_contract.resolve()),
                "sha256": _sha256_file(args.ownership_contract),
            },
            "tile_plan": {
                "retained_tile_count": plan.get("retained_tile_count"),
                "halo_overlap_factor": plan.get("split_cost", {}).get(
                    "halo_overlap_factor"
                ),
                "anchor_count": tile.get("anchor_count"),
            },
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(gate, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    scope = gate["authorized_scope"]
    print(
        f"surface frozen gate: status={gate['status']}, "
        f"training_allowed={gate['training_allowed']}, "
        f"population={scope['initialization_count']:,}, "
        f"peak={scope['measured_peak_vram_bytes'] / 1024**3:.2f} GiB, "
        f"sha256={gate['gate_manifest_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
