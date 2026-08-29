#!/usr/bin/env python3
"""Build one signed full-area raw-fisheye coverage or colour phase."""

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
    parser.add_argument("--phase", choices=("coverage", "color"), required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-gate", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--view-sampling-mode",
        choices=(
            "with_replacement",
            "fisher_yates_without_replacement_per_epoch",
        ),
        default="with_replacement",
    )
    parser.add_argument("--opacity-learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--background-profile",
        choices=("inherited", "black"),
        default="inherited",
        help=(
            "optionally composite against black during coverage calibration "
            "so bright snow and walls cannot reduce RGB loss by exposing a "
            "white canvas"
        ),
    )
    parser.add_argument(
        "--merge-contract",
        choices=("core_owner_only", "retain_full_halo"),
        default="core_owner_only",
        help="signed merge policy used by the supplied full-area checkpoint",
    )
    parser.add_argument("--lidar-alpha-weight", type=float, default=1.0)
    parser.add_argument("--lidar-alpha-target", type=float, default=0.95)
    parser.add_argument("--lidar-alpha-dilation-radius-px", type=int, default=3)
    parser.add_argument("--inherit-warm-start-auxiliary", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.steps <= 1_000:
        raise ValueError("steps must be between 10 and 1000")
    if not 0.0 < args.opacity_learning_rate <= 0.1:
        raise ValueError("opacity-learning-rate must be within (0, 0.1]")
    if not 0.0 < args.lidar_alpha_weight <= 10.0:
        raise ValueError("lidar-alpha-weight must be within (0, 10]")
    if not 0.5 <= args.lidar_alpha_target < 1.0:
        raise ValueError("lidar-alpha-target must be within [0.5, 1)")
    if not 0 <= args.lidar_alpha_dilation_radius_px <= 32:
        raise ValueError("lidar-alpha-dilation-radius-px must be within [0, 32]")

    for label, path in (
        ("source config", args.source_config),
        ("source checkpoint", args.source_checkpoint),
        ("upstream gate", args.upstream_gate),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    base = _read(args.source_config)
    if base.get("topology_policy", {}).get("mode") != "strict_fixed":
        raise ValueError("V50 must start from a strict-fixed full-area config")
    if not base.get("raw_fisheye_post_refine_face_manifest"):
        raise ValueError("V50 requires signed Face4 lineage for raw-fisheye training")

    protocol_root = args.protocol_root.resolve()
    config_path = protocol_root / "trainer.config.json"
    gate_path = protocol_root / "training_gate.json"
    plan_path = protocol_root / "evaluation_plan.json"
    readiness_path = protocol_root / "readiness.json"
    source_checkpoint_sha = _sha256(args.source_checkpoint)

    config = copy.deepcopy(base)
    config.pop("mipmap_tile_id", None)
    config.update(
        {
            "run_id": args.run_id,
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_pipeline_gate": gate_path.as_posix(),
            "warm_start_checkpoint": args.source_checkpoint.resolve().as_posix(),
            "resume_checkpoint": None,
            "warm_start_min_opacity": 0.0,
            "warm_start_scale_multiplier": 1.0,
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": True,
            "max_steps": args.steps,
            "controlled_stop_after_steps": None,
            "view_sampling_mode": args.view_sampling_mode,
            "checkpoint_every": args.steps,
            "checkpoint_keep_every": 0,
            "factor": 4,
            "cap_max": max(int(base.get("cap_max", 0)), 7_600_000),
            "topology_policy": {"mode": "strict_fixed"},
            "fixed_topology_schedule": {"enabled": False},
            "lidar_range_weight": 0.0,
            "lidar_linear_aux_weight": 0.0,
            "da2_depth_weight": 0.0,
            "golden_evaluation": {"enabled": False},
            "error_weighted_sampling": {"enabled": False},
            "lidar_admission": {"enabled": False},
            "tangent_proposal": {"enabled": False},
            "ppisp": {"enabled": False},
            "default_strategy": {},
            "cuda_empty_cache_interval_steps": 2,
        }
    )

    config["geometry_regularization"] = {
        "enabled": False,
        "opacity_sparsity_weight": 0.0,
        "scale_upper_weight": 0.0,
        "anisotropy_weight": 0.0,
        "max_scale_ratio_to_reference": 8.0,
        "max_anisotropy": 256.0,
    }
    if args.background_profile == "black":
        if args.phase != "coverage":
            raise ValueError("black background profile is coverage-only")
        config["background_color"] = [0.0, 0.0, 0.0]
    config["lidar_normal_alignment"] = {
        "enabled": False,
        "weight_align": 0.0,
        "weight_flatten": 0.0,
        "weight_point_to_plane": 0.0,
    }

    exposure = copy.deepcopy(config.get("exposure_compensation", {}))
    exposure["enabled"] = True
    if args.phase == "coverage":
        config["warm_start_fresh_auxiliary"] = (
            []
            if args.inherit_warm_start_auxiliary
            else ["exposure_log_gains"]
        )
        config["learning_rates"] = {
            "means": 0.0,
            "scales": 0.0,
            "quats": 0.0,
            "opacities": args.opacity_learning_rate,
            "colors": 0.0,
        }
        config["rgb_l1_weight"] = 0.6
        config["rgb_ssim_weight"] = 0.4
        config["lidar_alpha_weight"] = args.lidar_alpha_weight
        config["lidar_alpha_target"] = args.lidar_alpha_target
        config["lidar_alpha_dilation_radius_px"] = (
            args.lidar_alpha_dilation_radius_px
        )
        exposure["learning_rate"] = 0.0
    else:
        config["warm_start_fresh_auxiliary"] = []
        config["learning_rates"] = {
            "means": 0.0,
            "scales": 0.0,
            "quats": 0.0,
            "opacities": 0.0,
            "colors": 0.0005,
        }
        config["rgb_l1_weight"] = 0.8
        config["rgb_ssim_weight"] = 0.2
        config["lidar_alpha_weight"] = 0.0
        config["lidar_alpha_target"] = 0.95
        config["lidar_alpha_dilation_radius_px"] = 0
        exposure["learning_rate"] = 0.001
    config["exposure_compensation"] = exposure

    config.pop("config_manifest_sha256", None)
    config = _sign(config, "config_manifest_sha256")
    arm_name = f"FULL_AREA_RAW_FISHEYE_{args.phase.upper()}"
    plan = _sign(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": None,
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
                "full_area_surface_quality_protocol_is_strict_fixed"
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
                "phase_a_geometry_frozen": True,
                "core_only_merge_contract": args.merge_contract == "core_owner_only",
                "actual_merge_contract": args.merge_contract,
                "halo_overlap_retained": args.merge_contract == "retain_full_halo",
                "source_checkpoint_sha256": source_checkpoint_sha,
                "no_opacity_export_filter": True,
            },
            "training_allowed": False,
            "adaptive_growth_allowed": False,
        },
        "readiness_sha256",
    )
    gate = advance_fixed_topology_evaluation_gate(
        upstream, readiness, plan, {arm_name: config}
    )

    _write(config_path, config)
    _write(plan_path, plan)
    _write(readiness_path, readiness)
    _write(gate_path, gate)
    TrainerConfig.from_dict(config).validate()
    print(
        json.dumps(
            {
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
