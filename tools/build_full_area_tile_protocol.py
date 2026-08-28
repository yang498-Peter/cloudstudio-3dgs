#!/usr/bin/env python3
"""Build one snow Tile's signed A0 -> V27 protocol inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    V27_SNOW_TILE_PROFILES,
    advance_fixed_topology_evaluation_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sign(payload: dict, field: str) -> dict:
    signed = copy.deepcopy(payload)
    signed.pop(field, None)
    signed[field] = hashlib.sha256(canonical_json_bytes(signed)).hexdigest()
    return signed


def _replace_tile_path(value: object, tile_id: int) -> object:
    if not isinstance(value, str):
        return value
    return value.replace("/Tile_1", f"/Tile_{tile_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True, type=int)
    parser.add_argument("--base-a0-config", required=True, type=Path)
    parser.add_argument("--base-readiness", required=True, type=Path)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--controlled-stop", type=int)
    args = parser.parse_args()

    profile = V27_SNOW_TILE_PROFILES.get(args.tile_id)
    if profile is None:
        raise ValueError("tile-id must be one of 0, 1, 2, 3, 4")
    upstream = _read(args.upstream_gate)
    upstream_sha = verify_gate(upstream)
    base = _read(args.base_a0_config)
    base_readiness = _read(args.base_readiness)
    view_count = profile["view_count"]
    phase_a = 5 * view_count
    phase_b = 10 * view_count
    max_steps = profile["max_steps"]
    review_stop = profile["review_stop"]
    controlled_stop = args.controlled_stop or review_stop
    if not 0 < controlled_stop < max_steps:
        raise ValueError("controlled-stop must be between zero and max_steps")

    config = copy.deepcopy(base)
    for key in (
        "initialization_ply",
        "initialization_geometry",
        "face_lidar_geometry_manifest",
        "face_lidar_geometry_root",
    ):
        config[key] = _replace_tile_path(config.get(key), args.tile_id)
    config.update(
        {
            "run_id": f"snow-tile{args.tile_id}-fixed-topology-a0-review{review_stop}",
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_tile_id": args.tile_id,
            "mipmap_pipeline_gate": (
                args.output_root / "fixed_topology_a0_gate.json"
            ).resolve().as_posix(),
            "max_steps": max_steps,
            "controlled_stop_after_steps": controlled_stop,
            "checkpoint_every": view_count,
            "checkpoint_keep_every": review_stop,
            "cap_max": profile["cap_max"],
            "final_evaluation_artifacts": False,
            "fixed_topology_schedule": {
                "enabled": True,
                "phase_a_steps": phase_a,
                "phase_b_steps": phase_b,
                "phase_b_geometry_lr_scale": 0.1,
                "phase_c_geometry_lr_scale": 0.1,
                "phase_b_range_weight_scale": 1.0,
                "phase_c_range_weight_scale": 1.0,
                "phase_b_normal_weight_scale": 1.0,
                "phase_c_normal_weight_scale": 1.0,
                "audit_steps": [
                    1,
                    phase_a,
                    phase_a + 1,
                    phase_a + phase_b,
                    phase_a + phase_b + 1,
                    max_steps,
                ],
            },
        }
    )
    config.setdefault("surface_initialization", {})["maximum_scale_m"] = 0.08
    config.pop("resume_checkpoint", None)
    config.pop("warm_start_checkpoint", None)
    config.pop("config_manifest_sha256", None)
    for key in (
        "initialization_ply",
        "initialization_geometry",
        "initialization_geometry_manifest",
        "face_lidar_geometry_manifest",
        "face_lidar_geometry_root",
        "tile_inputs_manifest",
    ):
        value = config.get(key)
        if value is not None and not Path(value).exists():
            raise FileNotFoundError(f"{key} is missing: {value}")

    tile_manifest = _read(Path(config["tile_inputs_manifest"]))
    tile = next(
        item for item in tile_manifest["tiles"] if int(item["tile_id"]) == args.tile_id
    )
    initialization = tile["initialization"]
    if int(initialization["point_count"]) != profile["gaussian_count"]:
        raise ValueError("tile initialization count differs from signed V27 profile")
    if int(tile["view_count"]) != view_count:
        raise ValueError("tile view count differs from signed V27 profile")

    config = _sign(config, "config_manifest_sha256")
    config_path = args.output_root / "fixed_topology_a0.config.json"
    _write(config_path, config)

    plan = {
        "schema_version": 1,
        "kind": "fixed_topology_evaluation_plan_v1",
        "dataset": "snow-20260224",
        "tile_id": args.tile_id,
        "view_count": view_count,
        "epochs": {"phase_a": 5, "phase_b": 10, "phase_c": 5, "total": 20},
        "steps": {
            "phase_a": phase_a,
            "phase_b": phase_b,
            "phase_c": 5 * view_count,
            "total": max_steps,
            "controlled_review_stop": review_stop,
        },
        "matched_controls": [
            "authoritative_lidar_inputs",
            "accepted_independent_at_face4",
            "SH0",
            "seed_42",
            "full_resolution_factor_1",
            "fixed_topology_phase_schedule",
        ],
        "arms": [
            {
                "arm": "A0",
                "path": config_path.name,
                "sha256": _sha256(config_path),
                "topology_policy": {"mode": "strict_fixed"},
            }
        ],
        "training_allowed": False,
        "blocking_gates": ["signed_tile_a0_gate_not_yet_promoted"],
        "required_during_evaluation": [
            "geometry_freeze_audit",
            "core_owner_only_final_merge",
        ],
        "adaptive_growth_remains_blocked_by": [
            "bounded_mcmc_boundary602",
            "lidar_surface_relocation_admission",
        ],
    }
    plan = _sign(plan, "evaluation_plan_sha256")
    plan_path = args.output_root / "fixed_topology_a0_plan.json"
    _write(plan_path, plan)

    readiness = copy.deepcopy(base_readiness)
    readiness.pop("readiness_sha256", None)
    readiness.update(
        {
            "tile_id": args.tile_id,
            "upstream_gate_manifest_sha256": upstream_sha,
            "evaluation_plan_sha256": plan["evaluation_plan_sha256"],
            "training_allowed": False,
            "adaptive_growth_allowed": False,
            "reason": (
                "global LiDAR/Face4 implementation evidence reused; this Tile still "
                "requires an exact signed A0 arm promotion"
            ),
        }
    )
    readiness = _sign(readiness, "readiness_sha256")
    readiness_path = args.output_root / "fixed_topology_a0_readiness.json"
    _write(readiness_path, readiness)

    gate = advance_fixed_topology_evaluation_gate(
        upstream, readiness, plan, {"A0": config}
    )
    gate_path = args.output_root / "fixed_topology_a0_gate.json"
    _write(gate_path, gate)
    TrainerConfig.from_dict(config).validate()
    print(
        f"Tile_{args.tile_id} A0 ready: points={profile['gaussian_count']}, "
        f"views={view_count}, stop={controlled_stop}, gate={gate['gate_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
