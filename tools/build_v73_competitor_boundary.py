"""Build the signed V73 Tile_0 competitor-equivalent step-502 boundary.

The builder is deliberately fail closed.  It accepts only the four-Tile plan,
strict source-type 2/3 mesh supervision, and per-view mesh-aligned DA2.  Epoch
boundaries and the capacity cap are derived from the selected Tile instead of
reusing the old 374-view/971k-Gaussian constants.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mesh_geometry import verify_mesh_geometry_manifest
from cloudstudio_3dgs.data.mono_depth import verify_mono_depth_manifest
from cloudstudio_3dgs.pipeline.adaptive_tiling import verify_adaptive_tile_plan
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS,
    sign_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
TILE_ID = 0


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sign_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("config_manifest_sha256", None)
    result["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def _assert_holdout(path: Path) -> dict:
    audit = _read(path)
    categories = audit["heldout_category_audit"]["categories"]
    failures = {
        name: float(row["quantiles_m"]["p95"])
        for name, row in categories.items()
        if float(row["quantiles_m"]["p95"]) > 0.10
    }
    if failures:
        raise ValueError(f"held-out mesh categories exceed 10 cm P95: {failures}")
    return audit


def main() -> int:
    protocol = BASE / "v73_four_tile_competitor_equivalent" / "tile0_boundary502"
    plan_path = (
        BASE
        / "adaptive_tile_plan_lidar_visibility_4tile_v73"
        / "adaptive_tile_plan.json"
    )
    tile_inputs_root = BASE / "tile_training_inputs_lidar_4tile_v73"
    geometry_root = BASE / "tile_initialization_geometry_k7_k30_4tile_v73"
    lidar_root = BASE / "tile_face4_lidar_geometry_4tile_v73" / "Tile_0"
    mesh_root = BASE / "v73_four_tile_mesh" / "Tile_0" / "strict_source23_p95_010"
    mesh_path = mesh_root / "mesh_geometry_manifest.json"
    mono_root = BASE / "da2_face4_v23k" / "train"
    aligned_path = BASE / "v73_four_tile_mesh" / "Tile_0" / "da2_aligned"
    holdout_path = (
        BASE
        / "v73_four_tile_mesh"
        / "Tile_0"
        / "candidate_holdout10"
        / "mesh_candidate_audit.json"
    )

    plan_sha = verify_adaptive_tile_plan(_read(plan_path))
    tile_inputs = _read(tile_inputs_root / "tile_inputs_manifest.json")
    tile = next(row for row in tile_inputs["tiles"] if int(row["tile_id"]) == TILE_ID)
    view_count = int(tile["view_count"])
    initial_count = int(tile["initialization"]["point_count"])
    total_steps = 20 * view_count
    growth_stop = 15 * view_count
    prune_switch = 10 * view_count
    cap_max = int(math.ceil(initial_count * 1.5 / 1000.0) * 1000)

    mesh_manifest = _read(mesh_path)
    mesh_sha = verify_mesh_geometry_manifest(mesh_manifest)
    admission = mesh_manifest.get("admission_policy", {})
    if admission.get("allowed_source_types") != [2, 3]:
        raise ValueError("V73 mesh must allow only source types 2 and 3")
    if admission.get("rejected_source_types") != [4]:
        raise ValueError("V73 mesh must reject source type 4")
    if float(
        admission.get("max_view_absolute_range_error_p95_m", -1.0)
    ) != 0.10:
        raise ValueError("V73 mesh admission must use the 10 cm per-view P95 gate")
    mono_sha = verify_mono_depth_manifest(_read(aligned_path))
    holdout = _assert_holdout(holdout_path)

    upstream_path = (
        BASE
        / "v73_four_tile_competitor_equivalent"
        / "smoke_tile0_factor1_initcap"
        / "upstream_smoke_gate.json"
    )
    upstream = _read(upstream_path)
    verify_gate(upstream)

    config = _read(
        BASE
        / "v73_four_tile_competitor_equivalent"
        / "smoke_tile0_factor1_initcap"
        / "trainer.config.json"
    )
    config.update(
        {
            "run_id": "snow-tile0-v73-competitor-boundary502",
            "output_dir": str(protocol / "training_ewa"),
            "mipmap_pipeline_gate": str(protocol / "training_gate.json"),
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": False,
            "max_steps": total_steps,
            "controlled_stop_after_steps": 502,
            "checkpoint_every": 500,
            "checkpoint_keep_every": 500,
            "factor": 1,
            "pinhole_with_ut": False,
            "cap_max": cap_max,
            "densification_strategy": "default_3dgs",
            "densification_gradient_source": "total_loss",
            "mcmc_refine_start_iter": 500,
            "mcmc_refine_every": 100,
            "mcmc_refine_stop_iter": growth_stop,
            "mcmc_noise_injection_stop_iter": 0,
            "mcmc_noise_lr": 0.0,
            "topology_policy": {"mode": "adaptive_growth"},
            "fixed_topology_schedule": {"enabled": False},
            "default_strategy": {
                "exact_mipmap_lifecycle": True,
                "lifecycle_execution_order": "pre_optimizer_vendor",
                "grow_grad2d": 0.00015,
                "growth_min_opacity": 0.15,
                "split_scale_m": 0.2,
                "prune_scale_m": 0.2,
                "prune_opa": 0.10,
                "prune_opa_late": 0.05,
                "prune_switch_step": prune_switch,
                "prune_scale2d": 0.15,
                "refine_scale2d_stop_iter": growth_stop,
                "reset_every": 300,
                "reset_opacity_cap": 0.2,
                "absgrad": False,
                "revised_opacity": False,
                "capacity_conserving_clone_opacity": False,
                "detail_split_policy": "vendor_0_2m",
                "opacity_cull_policy": "immediate",
                "opacity_cull_min_observations": 0,
                "opacity_cull_consecutive_events": 1,
                "opacity_cull_grace_after_reset_steps": 0,
                "opacity_cull_max_fraction": 1.0,
                "opacity_cull_priority": "lowest_opacity",
                "opacity_cull_local_min_accumulated_alpha": 0.0,
                "vendor_cull_warmup_profile": "exact_0p10_to_0p05",
                "vendor_capacity_cull_profile": "disabled",
                "vendor_opacity_reset_profile": "exact_every300",
            },
            "tangent_proposal": {"enabled": False},
            "lidar_admission": {"enabled": False},
            "error_weighted_sampling": {"enabled": False},
            "lidar_range_weight": 0.0,
            "lidar_alpha_weight": 0.0,
            "da2_depth_weight": 0.5,
            "mono_depth_manifest": str(aligned_path),
            "mono_depth_root": str(mono_root),
            "mesh_depth_weight": 0.5,
            "mesh_normal_weight": 0.05,
            "mesh_geometry_manifest": str(mesh_path),
            "mesh_geometry_root": str(mesh_root),
            "rendered_depth_normal_consistency_weight": 0.01,
            "competitor_loss_schedule_enabled": True,
            "rig_pose_refinement": {"enabled": False},
            "color_model": "sh",
            "sh_degree": 0,
            "sh_degree_interval": 0,
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
            "ppisp": {"enabled": False},
            "bilateral_grid": {
                "enabled": True,
                "learning_rate": 0.002,
                "grid_width": 16,
                "grid_height": 16,
                "grid_depth": 8,
                "tv_weight": 5.0,
                "warmup_fraction": 1.0 / 30.0,
                "warmup_start_multiplier": 0.01,
                "final_lr_multiplier": 0.01,
            },
            "geometry_regularization": {
                "enabled": True,
                "opacity_sparsity_weight": 0.01,
                "opacity_sparsity_scope": "all",
                "scale_upper_weight": 0.0,
                "anisotropy_weight": 0.0,
                "max_scale_ratio_to_reference": 8.0,
                "max_anisotropy": 256.0,
                "screen_clip_enabled": False,
                "max_screen_fraction": 0.15,
                "screen_clip_hardness": 1.5,
                "screen_clip_opacity_bump": 0.0,
                "max_world_size_m": 0.2,
            },
            "lidar_normal_alignment": {
                "enabled": False,
                "weight_align": 0.0,
                "weight_flatten": 0.0,
                "weight_point_to_plane": 0.0,
            },
        }
    )
    config["surface_initialization"] = dict(config["surface_initialization"])
    config["surface_initialization"]["maximum_scale_m"] = 0.2
    config = _sign_config(config)

    gate = copy.deepcopy(upstream)
    gate.pop("gate_manifest_sha256", None)
    gate["status"] = ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS
    gate["training_allowed"] = True
    gate["blocking_reasons"] = []
    gate["next_required_stage"] = "adaptive_growth_boundary_evaluation"
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"].update(
        {
            "adaptive_growth_config_manifest_sha256": config[
                "config_manifest_sha256"
            ],
            "mesh_geometry_tile0_manifest_sha256": mesh_sha,
            "da2_train_manifest_sha256": mono_sha,
        }
    )
    gate.setdefault("evidence", {})["v73_dynamic_competitor_boundary"] = {
        "adaptive_tile_plan_manifest_sha256": plan_sha,
        "tile_id": TILE_ID,
        "view_count": view_count,
        "initial_gaussian_count": initial_count,
        "max_steps_20v": total_steps,
        "growth_stop_15v": growth_stop,
        "prune_switch_10v": prune_switch,
        "boundary_completed_step": 502,
        "capacity_cap_1p5x_rounded": cap_max,
        "allowed_mesh_source_types": [2, 3],
        "rejected_mesh_source_types": [4],
        "mesh_views_enabled": admission["totals"]["views_enabled"],
        "mesh_views_disabled_p95": admission["totals"]["views_disabled_p95"],
        "heldout_block_m": holdout["holdout"]["block_size_m"],
        "heldout_fraction": holdout["holdout"]["actual_fraction"],
        "quality_training_scope": "single_tile_first_lifecycle_boundary_only",
    }
    gate["adaptive_growth"] = {
        "profile": "v73_dynamic_four_tile_competitor_equivalent",
        "stage": "boundary",
        "tile_id": TILE_ID,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": 502,
        "mcmc_allowed": False,
        "capacity_cap": cap_max,
        "warm_start_checkpoint_sha256": None,
    }
    gate = sign_gate(gate)

    _write(protocol / "trainer.config.json", config)
    _write(protocol / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    print(protocol / "trainer.config.json")
    print(protocol / "training_gate.json")
    print(
        f"V={view_count} total={total_steps} growth_stop={growth_stop} "
        f"prune_switch={prune_switch} initial={initial_count} cap={cap_max}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
