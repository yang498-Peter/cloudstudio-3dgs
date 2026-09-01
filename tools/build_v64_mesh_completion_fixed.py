#!/usr/bin/env python3
"""Build, but do not launch, the signed V64A fixed-topology completion gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.geometry.mesh_completion import (
    verify_mesh_completion_initialization_manifest,
)
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
    sign_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
VIEW_COUNT = 374
STEPS = 2 * VIEW_COUNT


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _signed(payload: dict, field: str) -> dict:
    result = copy.deepcopy(payload)
    result[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="v64a")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--opacity-lr", type=float, default=0.01)
    parser.add_argument("--initialization-variant", default="v64a")
    parser.add_argument(
        "--match-v63-training",
        action="store_true",
        help="retain V63b geometry LRs and competitor mesh-loss schedule",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.opacity_lr <= 0.0:
        raise ValueError("steps and opacity LR must be positive")
    tag = f"{args.variant}_mesh_completion_fixed{args.steps}"
    protocol = BASE / tag
    config_path = protocol / f"snow_tile1_{tag}.config.json"
    gate_path = protocol / "training_gate.json"
    initialization_root = BASE / f"{args.initialization_variant}_mesh_completion_initialization"
    initialization_manifest_path = (
        initialization_root / "mesh_completion_initialization_manifest.json"
    )
    initialization_manifest = _read(initialization_manifest_path)
    initialization_sha = verify_mesh_completion_initialization_manifest(
        initialization_manifest,
        root=initialization_root,
        verify_artifacts=True,
    )

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived["bindings"] = dict(derived["bindings"])
    derived["bindings"]["mesh_completion_initialization_manifest_sha256"] = (
        initialization_sha
    )
    derived.setdefault("evidence", {})["v64_mesh_completion_override"] = {
        "scope": "Tile_1 fixed-topology two-epoch completion admission",
        "range_only_births_allowed": False,
        "source_type_4_births_allowed": False,
        "geometry_frozen": not args.match_v63_training,
        "adaptive_growth_allowed": False,
    }
    derived = sign_gate(derived)

    config = _read(
        BASE
        / "v63b_scaleguard_boundary1872"
        / "snow_tile1_v63b_scaleguard_boundary1872.config.json"
    )
    learning_rates = (
        dict(config["learning_rates"])
        if args.match_v63_training
        else {
            "means": 0.0,
            "scales": 0.0,
            "quats": 0.0,
            "opacities": float(args.opacity_lr),
            "colors": 0.0025,
        }
    )
    if args.match_v63_training:
        learning_rates["opacities"] = float(args.opacity_lr)
    configured_max_steps = (
        20 * VIEW_COUNT if args.match_v63_training else args.steps + 1
    )
    artifacts = initialization_manifest["artifacts"]
    config.update(
        {
            "run_id": f"snow-tile1-{args.variant}-mesh-completion-fixed{args.steps}",
            "output_dir": str(BASE / f"training_tile1_{tag}"),
            "mipmap_pipeline_gate": str(gate_path),
            "initialization_ply": str(
                initialization_root / artifacts["combined_ply"]
            ),
            "initialization_geometry": str(
                initialization_root / artifacts["geometry"]
            ),
            "initialization_geometry_manifest": str(initialization_manifest_path),
            "cap_max": 1_200_000,
            "metric_scale_calibration": {
                "mode": "precomputed",
                "knn_neighbors": 7,
                "knn_reduction": "arithmetic_mean",
                "scale_multiplier": 1.0,
                "clamp_min_ratio": 0.25,
                "clamp_max_ratio": 4.0,
                "means_step_fraction": None,
                "noise_std_fraction": None,
            },
            "surface_initialization": {
                "enabled": True,
                "mode": "signed_precomputed_surfel",
                "planarity_gate": 0.6,
                "normal_scale_ratio": 0.15,
                "minimum_normal_scale_m": 0.001,
                "maximum_scale_m": 0.2,
            },
            "learning_rates": learning_rates,
            "topology_policy": {"mode": "strict_fixed"},
            "densification_strategy": "default_3dgs",
            "mcmc_noise_injection_stop_iter": 0,
            "mcmc_noise_lr": 0.0,
            "mono_depth_manifest": None,
            "mono_depth_root": None,
            "da2_depth_weight": 0.0,
            "mesh_depth_weight": 0.5 if args.match_v63_training else 0.0,
            "mesh_normal_weight": 0.05 if args.match_v63_training else 0.0,
            "competitor_loss_schedule_enabled": bool(args.match_v63_training),
            # Keep one unreachable scheduler step so the controlled-stop
            # contract can preserve the exact 748 completed updates.
            "max_steps": configured_max_steps,
            "checkpoint_every": VIEW_COUNT,
            "checkpoint_keep_every": VIEW_COUNT,
            "controlled_stop_after_steps": args.steps,
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": False,
            "fixed_topology_schedule": {
                "enabled": False,
                "phase_a_steps": 0,
                "phase_b_steps": 0,
                "phase_b_geometry_lr_scale": 1.0,
                "phase_c_geometry_lr_scale": 1.0,
                "phase_b_range_weight_scale": 1.0,
                "phase_c_range_weight_scale": 1.0,
                "phase_b_normal_weight_scale": 1.0,
                "phase_c_normal_weight_scale": 1.0,
                "audit_steps": [],
            },
            "config_manifest_sha256": None,
        }
    )
    unsigned = copy.deepcopy(config)
    unsigned.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    plan = _signed(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": 1,
            "view_count": VIEW_COUNT,
            "steps": {"total": configured_max_steps},
            "arms": [
                {
                    "arm": "MESH_COMPLETION_FIXED",
                    "config_manifest_sha256": config["config_manifest_sha256"],
                }
            ],
            "training_allowed": False,
            "adaptive_growth_remains_blocked_by": [
                "v64_fixed_topology_joint_gate_not_yet_passed",
                "mesh_completion_visual_quality_not_yet_reviewed",
            ],
        },
        "evaluation_plan_sha256",
    )
    readiness = _signed(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_readiness_v1",
            "status": "FIXED_TOPOLOGY_EVALUATION_PREPARED",
            "training_allowed": False,
            "adaptive_growth_allowed": False,
            "upstream_gate_manifest_sha256": derived["gate_manifest_sha256"],
            "evaluation_plan_sha256": plan["evaluation_plan_sha256"],
            "evidence": {
                "directional_pass": True,
                "phase_a_geometry_frozen": not args.match_v63_training,
                "topology_fixed_geometry_bounded": bool(args.match_v63_training),
                "actual_merge_contract": "retain_full_halo",
                "halo_overlap_retained": True,
                "scope": (
                    f"{args.variant} {args.steps}-step "
                    + (
                        "V63-matched mesh completion gate"
                        if args.match_v63_training
                        else "appearance-only surfel completion gate"
                    )
                ),
            },
        },
        "readiness_sha256",
    )
    gate = advance_fixed_topology_evaluation_gate(
        derived, readiness, plan, {"MESH_COMPLETION_FIXED": config}
    )
    _write(protocol / "derived_upstream_gate.json", derived)
    _write(protocol / "evaluation_plan.json", plan)
    _write(protocol / "readiness.json", readiness)
    _write(config_path, config)
    _write(gate_path, gate)
    TrainerConfig.from_dict(config).validate()
    print(config_path)
    print(gate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
