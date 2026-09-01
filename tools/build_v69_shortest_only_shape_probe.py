"""Build the signed V69 shortest-axis-only surface-shape probe."""

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
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
    sign_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
VIEW_COUNT = 374
TOTAL_STEPS = 20 * VIEW_COUNT
PROBE_STEPS = 50


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(payload: dict, field: str) -> dict:
    result = copy.deepcopy(payload)
    result[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return result


def main() -> int:
    protocol = BASE / "v69_shortest_only_shape_probe"
    source_protocol = BASE / "v67_alpha_safe_photometric_probe"
    source_config = _read(
        source_protocol / "snow_tile1_v67b_alpha_safe_5v.config.json"
    )
    source_checkpoint = (
        BASE
        / "training_snow-tile1-v67b-alpha-safe-strictmesh-5v"
        / "checkpoints"
        / "latest.pt"
    )
    source_checkpoint_sha = _sha256_file(source_checkpoint)

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived.setdefault("evidence", {})["v69_shortest_only_shape_probe"] = {
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha,
        "authorization_scope": "shortest_axis_and_quaternion_only_50_steps",
        "tangent_axis_gradient": "detached",
        "frozen": [
            "means",
            "opacities",
            "colors",
            "exposure_log_gains",
            "exposure_biases",
            "bilateral_grid",
            "topology",
        ],
    }
    derived = sign_gate(derived)

    run_id = "snow-tile1-v69a-shortest-only-shape50"
    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": run_id,
            "output_dir": str(BASE / f"training_{run_id}"),
            "mipmap_pipeline_gate": str(protocol / "training_gate.json"),
            "warm_start_checkpoint": str(source_checkpoint),
            "warm_start_min_opacity": 0.0,
            "warm_start_scale_multiplier": 1.0,
            "warm_start_fresh_auxiliary": [],
            "max_steps": TOTAL_STEPS,
            "controlled_stop_after_steps": PROBE_STEPS,
            "checkpoint_every": PROBE_STEPS,
            "checkpoint_keep_every": PROBE_STEPS,
            "lidar_range_weight": 0.0,
            "lidar_alpha_weight": 1.0,
            "lidar_alpha_target": 0.95,
            "lidar_alpha_dilation_radius_px": 3,
            "da2_depth_weight": 0.0,
            "mesh_depth_weight": 0.0,
            "rendered_depth_normal_consistency_weight": 0.0,
            "learning_rates": {
                "means": 0.0,
                "scales": 0.003,
                "quats": 0.0005,
                "opacities": 0.0,
                "colors": 0.0,
            },
            "final_evaluation_artifacts": False,
        }
    )
    config["exposure_compensation"].update(
        {"learning_rate": 0.0, "bias_learning_rate": 0.0}
    )
    config["bilateral_grid"]["learning_rate"] = 0.0
    config["geometry_regularization"].update(
        {
            "opacity_sparsity_weight": 0.0,
            "scale_upper_weight": 0.0,
            "anisotropy_weight": 0.0,
            "max_anisotropy": 256.0,
        }
    )
    config["lidar_normal_alignment"].update(
        {
            "enabled": True,
            "weight_align": 0.1,
            "weight_flatten": 0.1,
            "weight_point_to_plane": 0.0,
            "flatten_mode": "tangent_ratio_shortest_only",
            "flatten_ratio_target": 0.15,
        }
    )
    config.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest()

    continuation = copy.deepcopy(config)
    continuation.update(
        {
            "run_id": "snow-tile1-v69b-shortest-only-shape150",
            "output_dir": str(
                BASE / "training_snow-tile1-v69b-shortest-only-shape150"
            ),
            "controlled_stop_after_steps": 150,
            "checkpoint_every": 150,
            "checkpoint_keep_every": 150,
        }
    )
    continuation.pop("config_manifest_sha256", None)
    continuation["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(continuation)
    ).hexdigest()
    configs = {"SHORTEST50": config, "SHORTEST150": continuation}

    plan = _signed(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": 1,
            "view_count": VIEW_COUNT,
            "steps": {
                "total": TOTAL_STEPS,
                "controlled_stop_completed_step": 150,
            },
            "arms": [
                {
                    "arm": arm,
                    "config_manifest_sha256": arm_config[
                        "config_manifest_sha256"
                    ],
                    "warm_start_checkpoint_sha256": source_checkpoint_sha,
                }
                for arm, arm_config in configs.items()
            ],
            "training_allowed": False,
            "adaptive_growth_allowed": False,
            "adaptive_growth_remains_blocked_by": [
                "v69_joint_render_and_geometry_gate_not_yet_passed"
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
                "source_alpha_depth_gate_passed": True,
                "adaptive_growth_disabled": True,
            },
        },
        "readiness_sha256",
    )
    gate = advance_fixed_topology_evaluation_gate(
        derived, readiness, plan, configs
    )
    _write(protocol / "derived_upstream_gate.json", derived)
    _write(protocol / "evaluation_plan.json", plan)
    _write(protocol / "readiness.json", readiness)
    _write(protocol / "training_gate.json", gate)
    config_path = protocol / "snow_tile1_v69a_shortest_only_shape50.config.json"
    _write(config_path, config)
    continuation_path = (
        protocol / "snow_tile1_v69b_shortest_only_shape150.config.json"
    )
    _write(continuation_path, continuation)
    TrainerConfig.from_dict(config).validate()
    TrainerConfig.from_dict(continuation).validate()
    print(config_path)
    print(continuation_path)
    print(protocol / "training_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
