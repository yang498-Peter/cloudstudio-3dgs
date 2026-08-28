"""Build a signed CPU-only LiDAR-first loss-schedule audit for every Tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.mipmap_loss_schedule import (
    high_type2_schedule_contract,
)
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-inputs", type=Path, required=True)
    parser.add_argument("--tile-inputs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tile_inputs = json.loads(args.tile_inputs.read_text(encoding="utf-8"))
    tile_inputs_sha = verify_tile_inputs_manifest(
        tile_inputs,
        root=args.tile_inputs_root,
        verify_artifacts=True,
    )
    tiles = []
    for tile in tile_inputs["tiles"]:
        schedule = high_type2_schedule_contract(int(tile["view_count"]))
        recommended = int(tile.get("recommended_training", {}).get("steps", -1))
        if recommended != schedule["total_steps"]:
            raise ValueError(
                f"Tile {tile['tile_id']} recommended steps differ from 20*V"
            )
        tiles.append(
            {
                "tile_id": int(tile["tile_id"]),
                "name": str(tile["name"]),
                **schedule,
            }
        )
    report = {
        "schema_version": 1,
        "kind": "lidar_first_face4_loss_schedule_audit",
        "status": "SCHEDULE_ORACLE_READY",
        "tile_inputs_manifest_sha256": tile_inputs_sha,
        "tiles": tiles,
        "training_allowed": False,
        "blocking_reasons": [
            "short GPU smoke has not verified sparse LiDAR range consumption",
            "short GPU smoke has not verified LiDAR-guarded split/clone",
            "independent sky optimizer is not implemented",
        ],
    }
    report["loss_schedule_audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"LiDAR-first schedule audit: tiles={len(tiles)}, "
        f"sha256={report['loss_schedule_audit_sha256']}"
    )


if __name__ == "__main__":
    main()
