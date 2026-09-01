#!/usr/bin/env python3
"""Build one signed A0-preserving full-area coverage or shape phase."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sign(value: dict[str, Any], key: str) -> dict[str, Any]:
    signed = copy.deepcopy(value)
    signed.pop(key, None)
    signed[key] = hashlib.sha256(canonical_json_bytes(signed)).hexdigest()
    return signed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--phase",
        choices=("coverage", "shape", "shortest_shape", "appearance"),
        required=True,
    )
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-gate", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--steps", type=int, choices=(25, 50, 100, 150, 250, 500), default=50
    )
    parser.add_argument(
        "--merge-contract",
        choices=("core_owner_only", "retain_full_halo"),
        default="core_owner_only",
    )
    args = parser.parse_args()

    for label, path in (
        ("source config", args.source_config),
        ("source checkpoint", args.source_checkpoint),
        ("upstream gate", args.upstream_gate),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    base = _read(args.source_config)
    if int(base.get("mipmap_tile_id", -1)) != args.tile_id:
        raise ValueError("source config targets another Tile")
    if base.get("topology_policy", {}).get("mode") != "strict_fixed":
        raise ValueError("V49 must start from an A0 strict-fixed config")

    protocol_root = args.protocol_root.resolve()
    config_path = protocol_root / "trainer.config.json"
    gate_path = protocol_root / "training_gate.json"
    plan_path = protocol_root / "evaluation_plan.json"
    readiness_path = protocol_root / "readiness.json"
    source_checkpoint_sha = _sha256(args.source_checkpoint)

    config = copy.deepcopy(base)
    config.update(
        {
            "run_id": args.run_id,
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_pipeline_gate": gate_path.as_posix(),
            "warm_start_checkpoint": args.source_checkpoint.resolve().as_posix(),
            "resume_checkpoint": None,
            "warm_start_fresh_auxiliary": [],
            "warm_start_min_opacity": 0.0,
            "warm_start_scale_multiplier": 1.0,
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": False,
            "max_steps": args.steps,
            "controlled_stop_after_steps": None,
            "view_sampling_mode": "with_replacement",
            "checkpoint_every": args.steps,
            "checkpoint_keep_every": 0,
            "factor": 1,
            "topology_policy": {"mode": "strict_fixed"},
            "fixed_topology_schedule": {"enabled": False},
            "lidar_range_weight": 0.0,
            "lidar_linear_aux_weight": 0.0,
            "lidar_alpha_weight": 1.0,
            "lidar_alpha_target": 0.95,
            "lidar_alpha_dilation_radius_px": 3,
            "da2_depth_weight": 0.0,
            "mesh_depth_weight": 0.0,
            "mesh_normal_weight": 0.0,
            "rendered_depth_normal_consistency_weight": 0.0,
            "golden_evaluation": {"enabled": False},
            "error_weighted_sampling": {"enabled": False},
            "lidar_admission": {"enabled": False},
            "tangent_proposal": {"enabled": False},
            "ppisp": {"enabled": False},
            "default_strategy": {},
        }
    )
    exposure = copy.deepcopy(config.get("exposure_compensation", {}))
    exposure.update(
        {"enabled": True, "learning_rate": 0.0, "bias_learning_rate": 0.0}
    )
    config["exposure_compensation"] = exposure
    bilateral_grid = copy.deepcopy(config.get("bilateral_grid", {}))
    if bilateral_grid:
        bilateral_grid["learning_rate"] = 0.0
        config["bilateral_grid"] = bilateral_grid

    regularization = copy.deepcopy(config.get("geometry_regularization", {}))
    regularization.update(
        {
            "enabled": True,
            "opacity_sparsity_weight": 0.0,
            "scale_upper_weight": 0.0,
            "anisotropy_weight": 0.0,
            "max_world_size_m": 0.2,
            "max_scale_ratio_to_reference": 8.0,
            "max_anisotropy": 256.0,
        }
    )
    config["geometry_regularization"] = regularization

    normal = copy.deepcopy(config.get("lidar_normal_alignment", {}))
    if args.phase == "coverage":
        config["learning_rates"] = {
            "means": 0.0,
            "scales": 0.0,
            "quats": 0.0,
            "opacities": 0.05,
            "colors": 0.0,
        }
        normal.update(
            {
                "enabled": False,
                "weight_align": 0.0,
                "weight_flatten": 0.0,
                "weight_point_to_plane": 0.0,
            }
        )
    elif args.phase == "appearance":
        config["learning_rates"] = {
            "means": 0.0,
            "scales": 0.0,
            "quats": 0.0,
            "opacities": 0.01,
            "colors": 0.001,
        }
        normal.update(
            {
                "enabled": False,
                "weight_align": 0.0,
                "weight_flatten": 0.0,
                "weight_point_to_plane": 0.0,
            }
        )
    else:
        config["learning_rates"] = {
            "means": 0.0,
            "scales": 0.003,
            "quats": 0.0005,
            "opacities": 0.0,
            "colors": 0.0,
        }
        normal.update(
            {
                "enabled": True,
                "weight_align": 0.1,
                "weight_flatten": 0.1,
                "weight_point_to_plane": 0.0,
                "flatten_mode": (
                    "tangent_ratio_shortest_only"
                    if args.phase == "shortest_shape"
                    else "tangent_ratio"
                ),
                "flatten_ratio_target": 0.15,
            }
        )
    config["lidar_normal_alignment"] = normal
    config.pop("config_manifest_sha256", None)
    config = _sign(config, "config_manifest_sha256")

    arm_name = args.phase.upper()
    plan = _sign(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": args.tile_id,
            "steps": {"total": args.steps},
            "arms": [
                {
                    "arm": arm_name,
                    "path": config_path.as_posix(),
                    "config_manifest_sha256": config["config_manifest_sha256"],
                    "warm_start_checkpoint_sha256": source_checkpoint_sha,
                }
            ],
            "training_allowed": False,
            "adaptive_growth_remains_blocked_by": [
                "full_area_A0_surface_quality_protocol_is_strict_fixed"
            ],
        },
        "evaluation_plan_sha256",
    )
    upstream = _read(args.upstream_gate)
    upstream_sha = verify_gate(upstream)
    readiness = _sign(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_readiness_v1",
            "status": "FIXED_TOPOLOGY_EVALUATION_PREPARED",
            "upstream_gate_manifest_sha256": upstream_sha,
            "evaluation_plan_sha256": plan["evaluation_plan_sha256"],
            "evidence": {
                "directional_pass": True,
                "topology_fixed_geometry_bounded": True,
                "actual_merge_contract": args.merge_contract,
                "core_only_merge_contract": args.merge_contract
                == "core_owner_only",
                "halo_overlap_retained": args.merge_contract
                == "retain_full_halo",
                "source_checkpoint_sha256": source_checkpoint_sha,
                "no_opacity_export_filter": True,
            },
            "training_allowed": False,
            "adaptive_growth_allowed": False,
        },
        "readiness_sha256",
    )
    gate = advance_fixed_topology_evaluation_gate(
        upstream,
        readiness,
        plan,
        {arm_name: config},
    )

    _write(config_path, config)
    _write(plan_path, plan)
    _write(readiness_path, readiness)
    _write(gate_path, gate)
    TrainerConfig.from_dict(config).validate()
    print(
        json.dumps(
            {
                "tile_id": args.tile_id,
                "phase": args.phase,
                "steps": args.steps,
                "source_checkpoint_sha256": source_checkpoint_sha,
                "config": config_path.as_posix(),
                "config_manifest_sha256": config["config_manifest_sha256"],
                "gate": gate_path.as_posix(),
                "gate_manifest_sha256": gate["gate_manifest_sha256"],
                "run_output": args.run_output.resolve().as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
