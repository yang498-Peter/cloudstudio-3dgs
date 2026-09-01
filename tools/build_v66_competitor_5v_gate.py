"""Build signed Tile_1 V66 configs for the competitor-aligned 5V gate."""

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
from cloudstudio_3dgs.data.mono_depth import verify_mono_depth_manifest
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
    protocol = BASE / "v66_competitor_strict_mesh_5v_gate"
    strict_mesh_root = BASE / "d0a_mesh_face4_tile1_strict23_p95_010_v66a"
    strict_mesh_path = strict_mesh_root / "mesh_geometry_manifest.json"
    mono_manifest_root = BASE / "d0a_da2_mesh_aligned_tile1_v62l"
    mono_path = mono_manifest_root / "mono_depth_manifest.json"
    # The aligned manifest reuses the immutable DA2 tensors; it adds only the
    # per-view RANSAC scale/shift admission metadata.
    mono_root = BASE / "da2_face4_v23k" / "train"
    holdout_path = (
        BASE
        / "d0a_mesh_candidate_tile1_proxyholdout10_v66b"
        / "mesh_candidate_audit.json"
    )

    mesh_manifest = _read(strict_mesh_path)
    mesh_sha = verify_mesh_geometry_manifest(mesh_manifest)
    admission = mesh_manifest.get("admission_policy", {})
    if admission.get("allowed_source_types") != [2, 3]:
        raise ValueError("strict mesh admission must allow only source types 2 and 3")
    if admission.get("rejected_source_types") != [4]:
        raise ValueError("strict mesh admission must reject source type 4")
    mono_sha = verify_mono_depth_manifest(_read(mono_path))
    holdout = _read(holdout_path)
    for name, result in holdout["heldout_category_audit"]["categories"].items():
        if float(result["quantiles_m"]["p95"]) > 0.10:
            raise ValueError(f"held-out mesh category fails 10 cm gate: {name}")

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived["bindings"] = dict(derived["bindings"])
    derived["bindings"].update(
        {
            "mesh_geometry_tile1_manifest_sha256": mesh_sha,
            "da2_train_manifest_sha256": mono_sha,
        }
    )
    derived.setdefault("evidence", {})["v66_strict_geometry"] = {
        "allowed_mesh_source_types": [2, 3],
        "rejected_mesh_source_types": [4],
        "view_p95_limit_m": 0.10,
        "views_enabled": admission["totals"]["views_enabled"],
        "views_disabled": admission["totals"]["views_disabled_p95"],
        "spatial_block_holdout_audit": str(holdout_path),
        "holdout_block_m": holdout["holdout"]["block_size_m"],
        "holdout_fraction": holdout["holdout"]["actual_fraction"],
        "category_status": holdout["heldout_category_audit"]["status"],
    }
    derived = sign_gate(derived)

    base_config = _read(
        BASE
        / "v63a_crossview_mesh_boundary1872"
        / "snow_tile1_v63a_crossview_mesh_boundary1872.config.json"
    )
    configs: dict[str, dict] = {}
    for arm, background in (
        ("WHITE", [1.0, 1.0, 1.0]),
        ("BLACK", [0.0, 0.0, 0.0]),
    ):
        config = copy.deepcopy(base_config)
        run_slug = f"snow-tile1-v66a-{arm.lower()}-strictmesh-5v"
        config.update(
            {
                "run_id": run_slug,
                "output_dir": str(BASE / f"training_{run_slug}"),
                "mipmap_pipeline_gate": str(protocol / "training_gate.json"),
                "mesh_geometry_manifest": str(strict_mesh_path),
                "mesh_geometry_root": str(strict_mesh_root),
                "mono_depth_manifest": str(mono_path),
                "mono_depth_root": str(mono_root),
                "da2_depth_weight": 0.5,
                "mesh_depth_weight": 0.5,
                "mesh_normal_weight": 0.05,
                "rendered_depth_normal_consistency_weight": 0.01,
                "competitor_loss_schedule_enabled": True,
                "max_steps": TOTAL_STEPS,
                "controlled_stop_after_steps": FIRST_POST_5V_COMPLETED_STEP,
                "checkpoint_every": VIEW_COUNT,
                "checkpoint_keep_every": VIEW_COUNT,
                "background_color": background,
                "pinhole_with_ut": True,
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
                "geometry_regularization": {
                    "enabled": True,
                    "opacity_sparsity_weight": 0.0,
                    "scale_upper_weight": 0.0,
                    "anisotropy_weight": 0.0,
                    "max_scale_ratio_to_reference": 8.0,
                    "max_anisotropy": 32.0,
                    "screen_clip_enabled": False,
                    "max_screen_fraction": 0.15,
                    "screen_clip_hardness": 1.5,
                    "screen_clip_opacity_bump": 0.0,
                    "max_world_size_m": 0.2,
                },
                "final_evaluation_artifacts": False,
                "config_manifest_sha256": None,
            }
        )
        configs[arm] = _sign_config(config)

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
                {"arm": name, "config_manifest_sha256": value["config_manifest_sha256"]}
                for name, value in configs.items()
            ],
            "training_allowed": False,
            "adaptive_growth_allowed": False,
            "adaptive_growth_remains_blocked_by": [
                "v66_fixed_topology_5v_joint_gate_not_yet_passed"
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
                "strict_mesh_source_types": [2, 3],
                "type4_excluded": True,
                "direct_gaussian_normal_raster": True,
                "per_camera_gain_bias": True,
                "bilateral_grid": True,
                "surface_only_training": True,
                "sky_training": "INDEPENDENT_AND_NOT_STARTED",
                "background_ab": ["WHITE", "BLACK"],
                "progressive_world_scale_shrink": 0.8,
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
    for arm, config in configs.items():
        config_path = protocol / f"snow_tile1_v66a_{arm.lower()}_5v.config.json"
        _write(config_path, config)
        TrainerConfig.from_dict(config).validate()
        print(config_path)
    print(protocol / "training_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
