#!/usr/bin/env python3
"""Build matched A0/A1 fixed-topology configs without authorizing long training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--view-count", required=True, type=int)
    parser.add_argument("--phase-a-epochs", type=int, default=5)
    parser.add_argument("--phase-b-epochs", type=int, default=10)
    parser.add_argument("--phase-c-epochs", type=int, default=5)
    parser.add_argument("--geometry-lr-scale", type=float, default=0.1)
    parser.add_argument("--prune-threshold", type=float, default=0.01)
    args = parser.parse_args()
    if args.view_count <= 0:
        raise ValueError("view-count must be positive")
    epochs = args.phase_a_epochs + args.phase_b_epochs + args.phase_c_epochs
    if min(args.phase_a_epochs, args.phase_b_epochs, args.phase_c_epochs) <= 0:
        raise ValueError("all three phase epoch counts must be positive")

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    phase_a_steps = args.phase_a_epochs * args.view_count
    phase_b_steps = args.phase_b_epochs * args.view_count
    max_steps = epochs * args.view_count
    audit_steps = sorted(
        {
            1,
            phase_a_steps,
            phase_a_steps + 1,
            phase_a_steps + phase_b_steps,
            phase_a_steps + phase_b_steps + 1,
            max_steps,
        }
    )
    common = copy.deepcopy(base)
    common.update(
        {
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": True,
            "factor": 1,
            "max_steps": max_steps,
            "view_sampling_mode": "fisher_yates_without_replacement_per_epoch",
            "checkpoint_every": args.view_count,
            "checkpoint_keep_every": phase_a_steps,
            "cuda_empty_cache_interval_steps": 2,
            "means_lr_final_factor": 0.1,
            "color_model": "sh",
            "sh_degree": 0,
            "sh_degree_interval": 0,
            "lidar_admission": {"enabled": False},
            "tangent_proposal": {"enabled": False},
            "fixed_topology_schedule": {
                "enabled": True,
                "phase_a_steps": phase_a_steps,
                "phase_b_steps": phase_b_steps,
                "phase_b_geometry_lr_scale": args.geometry_lr_scale,
                "phase_c_geometry_lr_scale": args.geometry_lr_scale,
                "phase_b_range_weight_scale": 1.0,
                "phase_c_range_weight_scale": 1.0,
                "phase_b_normal_weight_scale": 1.0,
                "phase_c_normal_weight_scale": 1.0,
                "audit_steps": audit_steps,
            },
        }
    )

    configs = []
    for arm, topology in (
        ("A0", {"mode": "strict_fixed"}),
        (
            "A1",
            {
                "mode": "opacity_prune_only",
                "opacity_prune_step": phase_a_steps,
                "opacity_prune_threshold": args.prune_threshold,
            },
        ),
    ):
        config = copy.deepcopy(common)
        suffix = arm.lower()
        config["run_id"] = f"snow-tile1-fixed-topology-{suffix}-eval-v25a"
        config["output_dir"] = (
            "G:/cloudstudio-3dgs/outputs/snow-20260224-full-20260825/"
            f"training_tile1_fixed_topology_{suffix}_eval_v25a"
        )
        config["topology_policy"] = topology
        path = args.output_root / f"snow_tile1_fixed_topology_{suffix}_eval_v25a.json"
        digest = _write_json(path, config)
        configs.append(
            {
                "arm": arm,
                "path": path.name,
                "sha256": digest,
                "topology_policy": topology,
            }
        )

    unsigned = {
        "schema_version": 1,
        "kind": "fixed_topology_evaluation_plan_v1",
        "dataset": "snow-20260224",
        "tile_id": 1,
        "view_count": args.view_count,
        "epochs": {
            "phase_a": args.phase_a_epochs,
            "phase_b": args.phase_b_epochs,
            "phase_c": args.phase_c_epochs,
            "total": epochs,
        },
        "steps": {
            "phase_a": phase_a_steps,
            "phase_b": phase_b_steps,
            "phase_c": args.phase_c_epochs * args.view_count,
            "total": max_steps,
        },
        "matched_controls": [
            "authoritative_lidar_inputs",
            "train_and_validation_views",
            "SH0",
            "seed_42",
            "full_resolution_factor_1",
            "loss_weights",
            "phase_schedule",
            "checkpoint_and_evaluation_cadence",
        ],
        "arms": configs,
        "training_allowed": False,
        "blocking_gates": [
            "signed_fixed_topology_evaluation_gate_not_yet_promoted",
        ],
        "required_during_evaluation": [
            "multi_view_component_gradient_distribution_and_weight_selection",
            "phase_c_structural_gap_map",
            "per_tile_core_cut_and_merge_uniqueness_audit_before_final_merge",
        ],
        "adaptive_growth_remains_blocked_by": [
            "persistent_structural_gap_evidence",
            "texture_and_lidar_supported_birth_mask",
            "first_real_clone_split_cull_reset_boundary_smoke",
        ],
    }
    plan = dict(unsigned)
    plan["evaluation_plan_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    _write_json(args.output_root / "fixed_topology_evaluation_plan_v25a.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
