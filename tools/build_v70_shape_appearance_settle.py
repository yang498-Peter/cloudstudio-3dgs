"""Build a signed short appearance settle from the accepted V69B shape probe."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for entry in (ROOT, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from build_v69_shortest_only_shape_probe import (
    BASE,
    TOTAL_STEPS,
    VIEW_COUNT,
    _read,
    _sha256_file,
    _signed,
    _write,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
    sign_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


def main() -> int:
    protocol = BASE / "v70_shape_appearance_settle"
    source_protocol = BASE / "v69_shortest_only_shape_probe"
    source_config = _read(
        source_protocol / "snow_tile1_v69b_shortest_only_shape150.config.json"
    )
    source_checkpoint = (
        BASE
        / "training_snow-tile1-v69b-shortest-only-shape150"
        / "checkpoints"
        / "latest.pt"
    )
    source_checkpoint_sha = _sha256_file(source_checkpoint)

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived.setdefault("evidence", {})["v70_shape_appearance_settle"] = {
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha,
        "authorization_scope": "opacity_and_sh0_only_100_steps",
        "frozen": [
            "means",
            "scales",
            "quats",
            "exposure_log_gains",
            "exposure_biases",
            "bilateral_grid",
            "topology",
        ],
    }
    derived = sign_gate(derived)

    run_id = "snow-tile1-v70a-shape-appearance-settle100"
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
            "controlled_stop_after_steps": 100,
            "checkpoint_every": 100,
            "checkpoint_keep_every": 100,
            "lidar_range_weight": 0.0,
            "lidar_alpha_weight": 1.0,
            "lidar_alpha_target": 0.95,
            "lidar_alpha_dilation_radius_px": 3,
            "da2_depth_weight": 0.0,
            "mesh_depth_weight": 0.0,
            "mesh_normal_weight": 0.0,
            "rendered_depth_normal_consistency_weight": 0.0,
            "learning_rates": {
                "means": 0.0,
                "scales": 0.0,
                "quats": 0.0,
                "opacities": 0.01,
                "colors": 0.001,
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
        }
    )
    config["lidar_normal_alignment"]["enabled"] = False
    config.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest()

    plan = _signed(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": 1,
            "view_count": VIEW_COUNT,
            "steps": {
                "total": TOTAL_STEPS,
                "controlled_stop_completed_step": 100,
            },
            "arms": [
                {
                    "arm": "SETTLE100",
                    "config_manifest_sha256": config[
                        "config_manifest_sha256"
                    ],
                    "warm_start_checkpoint_sha256": source_checkpoint_sha,
                }
            ],
            "training_allowed": False,
            "adaptive_growth_allowed": False,
            "adaptive_growth_remains_blocked_by": [
                "v70_joint_render_and_alpha_gate_not_yet_passed"
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
        derived, readiness, plan, {"SETTLE100": config}
    )
    _write(protocol / "derived_upstream_gate.json", derived)
    _write(protocol / "evaluation_plan.json", plan)
    _write(protocol / "readiness.json", readiness)
    _write(protocol / "training_gate.json", gate)
    config_path = protocol / "snow_tile1_v70a_shape_appearance_settle100.config.json"
    _write(config_path, config)
    TrainerConfig.from_dict(config).validate()
    print(config_path)
    print(protocol / "training_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
