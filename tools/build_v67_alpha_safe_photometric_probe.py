"""Build the signed Tile_1 V67 alpha-safe photometric probe."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mesh_geometry import verify_mesh_geometry_manifest
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
    sign_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
VIEW_COUNT = 374
TOTAL_STEPS = 20 * VIEW_COUNT
FIRST_POST_5V_COMPLETED_STEP = 5 * VIEW_COUNT + 2


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(payload: dict, field: str) -> dict:
    result = copy.deepcopy(payload)
    result[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return result


def _sign_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("config_manifest_sha256", None)
    result["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    protocol = BASE / "v67_alpha_safe_photometric_probe"
    source_protocol = BASE / "v64d_mesh_completion_fixed1872"
    source_config_path = (
        source_protocol / "snow_tile1_v64d_mesh_completion_fixed1872.config.json"
    )
    source_checkpoint = (
        BASE
        / "training_tile1_v64d_mesh_completion_fixed1872"
        / "checkpoints"
        / "latest.pt"
    )
    strict_mesh_root = BASE / "d0a_mesh_face4_tile1_strict23_p95_010_v66a"
    strict_mesh_path = strict_mesh_root / "mesh_geometry_manifest.json"

    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    checkpoint_sha = _sha256_file(source_checkpoint)
    mesh_manifest = _read(strict_mesh_path)
    mesh_sha = verify_mesh_geometry_manifest(mesh_manifest)
    admission = mesh_manifest.get("admission_policy", {})
    if admission.get("allowed_source_types") != [2, 3]:
        raise ValueError("V67 requires strict mesh source types 2 and 3")
    if admission.get("rejected_source_types") != [4]:
        raise ValueError("V67 requires source type 4 to be rejected")

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived["bindings"] = dict(derived["bindings"])
    derived["bindings"]["mesh_geometry_tile1_manifest_sha256"] = mesh_sha
    derived.setdefault("evidence", {})["v67_alpha_safe_photometric_probe"] = {
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_run": "snow-tile1-v64d-mesh-completion-fixed1872",
        "allowed_mesh_source_types": [2, 3],
        "rejected_mesh_source_types": [4],
        "scope": "one_epoch_probe_then_first_post_5v_boundary",
    }
    derived = sign_gate(derived)

    config = _read(source_config_path)
    run_id = "snow-tile1-v67a-alpha-safe-photometric374"
    config.update(
        {
            "run_id": run_id,
            "output_dir": str(BASE / f"training_{run_id}"),
            "mipmap_pipeline_gate": str(protocol / "training_gate.json"),
            "warm_start_checkpoint": str(source_checkpoint),
            "warm_start_min_opacity": 0.0,
            "warm_start_scale_multiplier": 1.0,
            "warm_start_fresh_auxiliary": [
                "exposure_log_gains",
                "exposure_biases",
                "bilateral_grid",
            ],
            # Preserve the production 20-epoch LR schedule while the signed
            # controlled stop bounds this probe to one complete view epoch.
            "max_steps": TOTAL_STEPS,
            "controlled_stop_after_steps": VIEW_COUNT,
            "checkpoint_every": VIEW_COUNT,
            "checkpoint_keep_every": VIEW_COUNT,
            "mesh_geometry_manifest": str(strict_mesh_path),
            "mesh_geometry_root": str(strict_mesh_root),
            "mono_depth_manifest": None,
            "mono_depth_root": None,
            "da2_depth_weight": 0.0,
            "background_color": [1.0, 1.0, 1.0],
            "ppisp": {"enabled": False},
            "exposure_compensation": {
                "enabled": True,
                "mode": "per_camera",
                "learning_rate": 0.005,
                "regularization_weight": 0.01,
                "max_abs_log_gain": 0.6931471805599453,
                "zero_mean_projection": False,
                "mean_anchor_weight": 0.0,
                "mean_anchor_beta": 0.1,
                "bias_enabled": True,
                "bias_learning_rate": 0.001,
                "bias_regularization_weight": 0.01,
                "max_abs_bias": 0.25,
            },
            "bilateral_grid": {
                "enabled": True,
                "learning_rate": 0.002,
                "grid_width": 16,
                "grid_height": 16,
                "grid_depth": 8,
                "tv_weight": 5.0,
                "warmup_fraction": 0.03333333333333333,
                "warmup_start_multiplier": 0.01,
                "final_lr_multiplier": 0.01,
            },
            "final_evaluation_artifacts": False,
        }
    )
    config = _sign_config(config)

    probe_validation_path = (
        ROOT
        / "results"
        / "diagnostics"
        / "snow-tile1-v67a-alpha-safe-photometric374-validation24"
        / "validation_summary.json"
    )
    reference_validation_path = (
        ROOT
        / "results"
        / "diagnostics"
        / "snow-tile1-v64d-step1872-validation24-white"
        / "validation_summary.json"
    )
    if not probe_validation_path.is_file() or not reference_validation_path.is_file():
        raise FileNotFoundError("V67A and V64D matched-view validation are required")
    probe_validation = _read(probe_validation_path)
    reference_validation = _read(reference_validation_path)
    probe_checks = {
        "psnr_ge_reference_minus_0p10db": float(probe_validation["psnr_mean_db"])
        >= float(reference_validation["psnr_mean_db"]) - 0.10,
        "alpha_mean_ge_reference_minus_0p02": float(probe_validation["alpha_mean"])
        >= float(reference_validation["alpha_mean"]) - 0.02,
        "alpha_p05_ge_reference_minus_0p02": float(probe_validation["alpha_p05_mean"])
        >= float(reference_validation["alpha_p05_mean"]) - 0.02,
        "lidar_alpha_p05_ge_reference_minus_0p02": float(
            probe_validation["lidar_alpha_p05_mean"]
        )
        >= float(reference_validation["lidar_alpha_p05_mean"]) - 0.02,
        "depth_mae_not_worse": float(probe_validation["depth_mae_mean_m"])
        <= float(reference_validation["depth_mae_mean_m"]),
    }
    if not all(probe_checks.values()):
        raise ValueError(f"V67A alpha-safe probe failed: {probe_checks}")

    five_v_config = copy.deepcopy(config)
    five_v_run_id = "snow-tile1-v67b-alpha-safe-strictmesh-5v"
    five_v_config.update(
        {
            "run_id": five_v_run_id,
            "output_dir": str(BASE / f"training_{five_v_run_id}"),
            "controlled_stop_after_steps": FIRST_POST_5V_COMPLETED_STEP,
        }
    )
    five_v_config = _sign_config(five_v_config)

    plan = _signed(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": 1,
            "view_count": VIEW_COUNT,
            "steps": {
                "total": TOTAL_STEPS,
                "controlled_stop_completed_step": FIRST_POST_5V_COMPLETED_STEP,
            },
            "arms": [
                {
                    "arm": "ALPHA_SAFE_PHOTOMETRIC",
                    "config_manifest_sha256": config["config_manifest_sha256"],
                    "warm_start_checkpoint_sha256": checkpoint_sha,
                },
                {
                    "arm": "ALPHA_SAFE_STRICTMESH_5V",
                    "config_manifest_sha256": five_v_config[
                        "config_manifest_sha256"
                    ],
                    "warm_start_checkpoint_sha256": checkpoint_sha,
                }
            ],
            "training_allowed": False,
            "adaptive_growth_allowed": False,
            "adaptive_growth_remains_blocked_by": [
                "v67_alpha_safe_5v_joint_gate_not_yet_passed"
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
                "topology_fixed_geometry_bounded": True,
                "actual_merge_contract": "retain_full_halo",
                "halo_overlap_retained": True,
                "source_checkpoint_sha256": checkpoint_sha,
                "photometric_affine_is_alpha_safe": True,
                "fresh_photometric_auxiliary": True,
                "probe_validation_sha256": _sha256_file(probe_validation_path),
                "reference_validation_sha256": _sha256_file(
                    reference_validation_path
                ),
                "probe_checks": probe_checks,
                "adaptive_growth_disabled": True,
                "sky_training": "INDEPENDENT_AND_NOT_STARTED",
            },
        },
        "readiness_sha256",
    )
    gate = advance_fixed_topology_evaluation_gate(
        derived,
        readiness,
        plan,
        {
            "ALPHA_SAFE_PHOTOMETRIC": config,
            "ALPHA_SAFE_STRICTMESH_5V": five_v_config,
        },
    )

    _write(protocol / "derived_upstream_gate.json", derived)
    _write(protocol / "evaluation_plan.json", plan)
    _write(protocol / "readiness.json", readiness)
    _write(protocol / "training_gate.json", gate)
    config_path = protocol / "snow_tile1_v67a_alpha_safe_photometric374.config.json"
    five_v_config_path = protocol / "snow_tile1_v67b_alpha_safe_5v.config.json"
    _write(config_path, config)
    _write(five_v_config_path, five_v_config)
    TrainerConfig.from_dict(config).validate()
    TrainerConfig.from_dict(five_v_config).validate()
    print(config_path)
    print(five_v_config_path)
    print(protocol / "training_gate.json")
    print(f"warm_start_checkpoint_sha256={checkpoint_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
