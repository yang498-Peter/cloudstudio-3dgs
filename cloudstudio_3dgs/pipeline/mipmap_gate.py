"""Signed gate for the MipMap-aligned raw-fisheye -> Face4 route."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.depth_cache import verify_depth_manifest
from cloudstudio_3dgs.data.mono_depth import verify_mono_depth_manifest
from cloudstudio_3dgs.data.renderer_masks import verify_renderer_mask_manifest
from cloudstudio_3dgs.data.sky_background import (
    verify_sky_evidence_manifest,
    verify_sky_initialization_manifest,
)
from cloudstudio_3dgs.pipeline.adaptive_tiling import verify_adaptive_tile_plan


GATE_SCHEMA_VERSION = 2
GATE_PROFILE = "mipmap_aligned_face4_v2"
FRONTEND_READY_STATUS = "FACE4_BASE_READY"
RENDERER_MASK_READY_STATUS = "RENDERER_MASK_READY"
LIDAR_DEPTH_READY_STATUS = "LIDAR_DEPTH_READY"
DA2_DEPTH_READY_STATUS = "DA2_DEPTH_READY"
SKY_BACKGROUND_READY_STATUS = "SKY_BACKGROUND_READY"
UPSTREAM_DATA_READY_STATUS = "UPSTREAM_DATA_READY"
TRAINING_IMPLEMENTATION_READY_STATUS = "TRAINING_IMPLEMENTATION_READY"
FIXED_TOPOLOGY_EVALUATION_READY_STATUS = "FIXED_TOPOLOGY_EVALUATION_READY"
ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS = "ADAPTIVE_GROWTH_BOUNDARY_READY"
ADAPTIVE_GROWTH_EVALUATION_READY_STATUS = "ADAPTIVE_GROWTH_EVALUATION_READY"
# Compatibility import for callers. Old signed gates whose literal status is
# ``TRAINING_READY`` are intentionally not accepted by verify_training_gate.
TRAINING_READY_STATUS = TRAINING_IMPLEMENTATION_READY_STATUS
TRAINING_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION = 1
TRAINING_IMPLEMENTATION_CONTRACT_KIND = "mipmap_training_implementation_contract"
MIPMAP_HIGH_TYPE2_PRESET = "lidar_first_face4_snow_v1"
INDEPENDENT_AT_ALGORITHM = "independent_pos_prior_shared_single_focal_kb4_at_v2"

# The snow scene uses five LiDAR-first adaptive tiles. V27 keeps a fixed
# Gaussian capacity per tile, so its signed limits must follow the actual
# initialization count and view cadence instead of inheriting Tile_1 values.
V27_SNOW_TILE_PROFILES: dict[int, dict[str, int]] = {
    0: {
        "view_count": 476,
        "gaussian_count": 1_895_788,
        "cap_max": 1_895_789,
        "max_steps": 9_520,
        "review_stop": 3_332,
        "stabilization_stop": 3_808,
    },
    1: {
        "view_count": 374,
        "gaussian_count": 971_903,
        "cap_max": 971_904,
        "max_steps": 7_480,
        "review_stop": 2_618,
        "stabilization_stop": 2_992,
    },
    2: {
        "view_count": 470,
        "gaussian_count": 1_477_056,
        "cap_max": 1_477_057,
        "max_steps": 9_400,
        "review_stop": 3_290,
        "stabilization_stop": 3_760,
    },
    3: {
        "view_count": 607,
        "gaussian_count": 1_850_945,
        "cap_max": 1_850_946,
        "max_steps": 12_140,
        "review_stop": 4_249,
        "stabilization_stop": 4_856,
    },
    4: {
        "view_count": 505,
        "gaussian_count": 1_076_290,
        "cap_max": 1_076_291,
        "max_steps": 10_100,
        "review_stop": 3_535,
        "stabilization_stop": 4_040,
    },
}


MIPMAP_HIGH_TYPE2_REQUIRED_IMPLEMENTATION: dict[str, Any] = {
    "tile_face4_crop": {
        "consumed_by_dataset": True,
        "principal_point_shifted_by_crop_origin": True,
    },
    "face4_lidar_geometry": {
        "sparse_metric_range_consumed_by_dataset": True,
        "source_depth_manifest_bound": True,
        "mesh_interpolation": False,
        "da2_consumed": False,
    },
    "renderer_mask": {
        "manifest_consumed_by_dataset": True,
        "source_face_manifest_bound": True,
        "circle_fov_and_person_dynamic_mask": True,
    },
    "surface_initialization": {
        "scale_neighbor_query_k": 7,
        "scale_neighbor_count_excluding_self": 6,
        "scale_reduction": "arithmetic_mean_euclidean_distance",
        "normal_neighbor_count": 30,
        "linear_scale_axes": [1.0, 1.0, 0.5],
        "orientation": "shortest_arc_local_positive_z_to_pca_normal",
        "opacity_probability": 0.1,
        "opacity_storage": "logit",
    },
    "view_sampling": {
        "mode": "fisher_yates_without_replacement_per_epoch",
        "stage_epochs": [5, 10, 5],
    },
    "gaussian_management": {
        "mode": "classic_gradient_split_clone_cull",
        "mcmc_relocation": False,
        "redundancy_cull": False,
        "densify_start_step": 500,
        "densify_interval": 100,
        "xy_gradient_threshold": 0.00015,
        "candidate_opacity_threshold": 0.15,
        "clone_max_linear_axis": 0.2,
        "split_min_linear_axis_exclusive": 0.2,
        "split_children": 2,
        "split_scale_divisor": 1.6,
        "cull_opacity_first_half": 0.1,
        "cull_opacity_second_half": 0.05,
        "cull_world_scale": 0.2,
        "cull_screen_scale": 0.15,
        "opacity_reset_interval": 300,
        "opacity_reset_probability_cap": 0.2,
        "lidar_parent_planarity_support_gate": True,
        "newborn_tangent_surface_proposal": True,
        "unsupported_births_rejected": True,
    },
    "loss_schedule": {
        "rgb_l1_weight": 0.6,
        "rgb_ssim_weight": 0.4,
        "sparse_real_lidar_range_weight_positive": True,
        "lidar_surface_normal_weight_positive": True,
        "da2_weight": 0.0,
        "mesh_depth_weight": 0.0,
        "mesh_normal_weight": 0.0,
        "opacity_mean_weight": 0.01,
        "sky_opacity_second_half_weight": 0.04,
    },
    "deferred_optional_components": {
        "da2_required": False,
        "mesh_required": False,
        "bilateral_grid_required": False,
        "sift_pose_refinement_required": False,
    },
    "capacity": {
        "vendor_multiplier_c_resolved": True,
        "cap_formula_applied_per_tile": True,
    },
    "sky": {
        "independent_training": True,
        "gaussian_count": 100000,
        "sh_degree": 1,
        "merged_after_surface_tiles": True,
    },
    "tile_merge_and_delivery": {
        "halo_ownership_resolved": True,
        "cut_then_merge_consumed": True,
        "raw_fisheye_evaluation_passed": True,
        "ply_sog_lod_export_verified": True,
    },
}

ORDERED_STAGES = (
    "input_preflight",
    "time_sync_audit",
    "raw_circle_mask",
    "raw_person_mask",
    "masked_feature_matching",
    "known_pose_triangulation",
    "shared_single_focal_kb4_at",
    "accepted_training_manifest",
    "rebased_masks_and_split",
    "face4_rgb_and_person_mask",
    "renderer_dynamic_mask",
    "new_at_lidar_depth",
    "da2_monocular_depth",
    "independent_sky_background",
    "spatial_tile_plan",
    "tile_gaussian_training",
    "raw_fisheye_evaluation",
    "ply_sog_lod_export",
)


def sign_gate(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("gate_manifest_sha256", None)
    signed = copy.deepcopy(unsigned)
    signed["gate_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def sign_training_implementation_contract(
    payload: dict[str, Any],
) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("training_implementation_contract_sha256", None)
    signed = copy.deepcopy(unsigned)
    signed["training_implementation_contract_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def fixed_topology_evaluation_arm_fingerprint(config: dict[str, Any]) -> str:
    """Bind a fixed-topology arm without depending on its gate file path."""

    topology = config.get("topology_policy") or {}
    schedule = config.get("fixed_topology_schedule") or {}
    payload = {
        "run_id": config.get("run_id"),
        "mipmap_tile_id": config.get("mipmap_tile_id"),
        "topology_policy": {
            "mode": topology.get("mode", "adaptive_growth"),
            "opacity_prune_step": topology.get("opacity_prune_step"),
            "opacity_prune_threshold": topology.get("opacity_prune_threshold", 0.01),
        },
        "fixed_topology_schedule": {
            "enabled": schedule.get("enabled", False),
            "phase_a_steps": schedule.get("phase_a_steps", 0),
            "phase_b_steps": schedule.get("phase_b_steps", 0),
            "phase_b_geometry_lr_scale": schedule.get(
                "phase_b_geometry_lr_scale", 1.0
            ),
            "phase_c_geometry_lr_scale": schedule.get(
                "phase_c_geometry_lr_scale", 1.0
            ),
            "phase_b_range_weight_scale": schedule.get(
                "phase_b_range_weight_scale", 1.0
            ),
            "phase_c_range_weight_scale": schedule.get(
                "phase_c_range_weight_scale", 1.0
            ),
            "phase_b_normal_weight_scale": schedule.get(
                "phase_b_normal_weight_scale", 1.0
            ),
            "phase_c_normal_weight_scale": schedule.get(
                "phase_c_normal_weight_scale", 1.0
            ),
            "audit_steps": list(schedule.get("audit_steps", [])),
        },
        "max_steps": config.get("max_steps"),
        "factor": config.get("factor"),
        "color_model": config.get("color_model"),
        "sh_degree": config.get("sh_degree"),
        "da2_depth_weight": config.get("da2_depth_weight"),
        "lidar_admission_enabled": bool(
            (config.get("lidar_admission") or {}).get("enabled", False)
        ),
        "tangent_proposal_enabled": bool(
            (config.get("tangent_proposal") or {}).get("enabled", False)
        ),
        "densification_strategy": config.get("densification_strategy"),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_exact_subset(
    actual: Any,
    required: Any,
    *,
    path: str,
) -> None:
    if isinstance(required, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"training implementation contract {path} must be an object")
        for key, value in required.items():
            if key not in actual:
                raise ValueError(
                    f"training implementation contract is missing {path}.{key}"
                )
            _require_exact_subset(actual[key], value, path=f"{path}.{key}")
        return
    if actual != required:
        raise ValueError(
            f"training implementation contract mismatch at {path}: "
            f"expected={required!r}, actual={actual!r}"
        )


def verify_training_implementation_contract(contract: dict[str, Any]) -> str:
    expected = str(contract.get("training_implementation_contract_sha256", ""))
    if len(expected) != 64:
        raise ValueError("training implementation contract is unsigned")
    unsigned = copy.deepcopy(contract)
    unsigned.pop("training_implementation_contract_sha256", None)
    actual_sha = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual_sha != expected:
        raise ValueError("training implementation contract signature mismatch")
    if (
        int(contract.get("schema_version", -1))
        != TRAINING_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION
        or contract.get("kind") != TRAINING_IMPLEMENTATION_CONTRACT_KIND
    ):
        raise ValueError("unsupported training implementation contract")
    if contract.get("preset") != MIPMAP_HIGH_TYPE2_PRESET:
        raise ValueError("unexpected MipMap training implementation preset")
    _require_exact_subset(
        contract.get("implementation"),
        MIPMAP_HIGH_TYPE2_REQUIRED_IMPLEMENTATION,
        path="implementation",
    )
    verification = contract.get("verification", {})
    for required_gate in ("cpu_contract_tests_passed", "short_gpu_smoke_passed"):
        if verification.get(required_gate) is not True:
            raise ValueError(
                f"training implementation contract requires {required_gate}=true"
            )
    if contract.get("unresolved_blockers") != []:
        raise ValueError("training implementation contract still has unresolved blockers")
    for key in ("trainer_source_sha256", "trainer_config_sha256"):
        if len(str(contract.get(key, ""))) != 64:
            raise ValueError(f"training implementation contract requires {key}")
    return expected


def verify_gate(gate: dict[str, Any]) -> str:
    expected = str(gate.get("gate_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("MipMap pipeline gate is unsigned")
    unsigned = copy.deepcopy(gate)
    unsigned.pop("gate_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("MipMap pipeline gate signature mismatch")
    if int(gate.get("schema_version", -1)) != GATE_SCHEMA_VERSION:
        raise ValueError("unsupported MipMap pipeline gate schema")
    if gate.get("profile") != GATE_PROFILE:
        raise ValueError("unexpected MipMap pipeline gate profile")
    completed = tuple(str(value) for value in gate.get("completed_stages", []))
    if completed != ORDERED_STAGES[: len(completed)]:
        raise ValueError("MipMap pipeline stages are missing, reordered, or skipped")
    return expected


def load_and_verify_gate(path: Path) -> tuple[dict[str, Any], str]:
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    return gate, verify_gate(gate)


def advance_renderer_mask_gate(
    frontend_gate: dict[str, Any],
    train_manifest: dict[str, Any],
    val_manifest: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance a verified Face4-base gate by exactly one renderer-mask stage."""
    verify_gate(frontend_gate)
    if (
        frontend_gate.get("status") != FRONTEND_READY_STATUS
        or frontend_gate.get("training_allowed") is not False
        or tuple(frontend_gate.get("completed_stages", [])) != ORDERED_STAGES[:10]
    ):
        raise ValueError("renderer masks require an exact FACE4_BASE_READY gate")
    train_sha = verify_renderer_mask_manifest(train_manifest)
    val_sha = verify_renderer_mask_manifest(val_manifest)
    bindings = dict(frontend_gate.get("bindings", {}))
    expected = {
        "train": bindings.get("face4_train_manifest_sha256"),
        "val": bindings.get("face4_val_manifest_sha256"),
    }
    actual = {
        "train": train_manifest.get("source_face_manifest_sha256"),
        "val": val_manifest.get("source_face_manifest_sha256"),
    }
    if actual != expected:
        raise ValueError(
            "renderer masks are bound to different Face4 inputs: "
            f"expected={expected}, actual={actual}"
        )
    if train_manifest.get("split") != "train" or val_manifest.get("split") != "val":
        raise ValueError("renderer mask manifests must contain train and val splits")
    bindings.update(
        {
            "renderer_mask_train_manifest_sha256": train_sha,
            "renderer_mask_val_manifest_sha256": val_sha,
        }
    )
    payload = copy.deepcopy(frontend_gate)
    payload.pop("gate_manifest_sha256", None)
    payload.update(
        {
            "status": RENDERER_MASK_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:11]),
            "next_required_stage": ORDERED_STAGES[11],
            "blocking_reasons": [
                "LiDAR depth has not been rebuilt from the accepted AT",
                "DA2 depth, independent sky, and spatial Tile plan are not complete",
            ],
            "bindings": bindings,
        }
    )
    if evidence:
        payload.setdefault("evidence", {}).update(copy.deepcopy(evidence))
    return sign_gate(payload)


def advance_lidar_depth_gate(
    renderer_gate: dict[str, Any],
    depth_manifest: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance an exact renderer-mask gate by one complete LiDAR-depth stage."""
    verify_gate(renderer_gate)
    if (
        renderer_gate.get("status") != RENDERER_MASK_READY_STATUS
        or renderer_gate.get("training_allowed") is not False
        or tuple(renderer_gate.get("completed_stages", [])) != ORDERED_STAGES[:11]
    ):
        raise ValueError("LiDAR depth requires an exact RENDERER_MASK_READY gate")
    depth_sha = verify_depth_manifest(depth_manifest)
    bindings = dict(renderer_gate.get("bindings", {}))
    expected_dataset_sha = bindings.get("training_dataset_manifest_sha256")
    expected_mask_sha = bindings.get("training_circle_mask_manifest_sha256")
    if not expected_dataset_sha or not expected_mask_sha:
        raise ValueError("renderer gate is missing training dataset or circle-mask binding")
    if depth_manifest.get("dataset_manifest_sha256") != expected_dataset_sha:
        raise ValueError("LiDAR depth is bound to a different training dataset")
    if depth_manifest.get("mask_manifest_sha256") != expected_mask_sha:
        raise ValueError("LiDAR depth is bound to a different training circle mask")
    if (
        depth_manifest.get("complete_dataset") is not True
        or int(depth_manifest.get("total_dataset_images", 0)) <= 0
        or int(depth_manifest.get("summary", {}).get("image_count", 0))
        != int(depth_manifest.get("total_dataset_images", 0))
        or int(depth_manifest.get("point_cloud_points", 0)) <= 0
    ):
        raise ValueError("LiDAR depth manifest is incomplete or has no point cloud")
    bindings.update(
        {
            "lidar_depth_manifest_sha256": depth_sha,
            "lidar_depth_point_cloud_sha256": depth_manifest.get(
                "point_cloud_sha256"
            ),
        }
    )
    payload = copy.deepcopy(renderer_gate)
    payload.pop("gate_manifest_sha256", None)
    payload.update(
        {
            "status": LIDAR_DEPTH_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:12]),
            "next_required_stage": ORDERED_STAGES[12],
            "blocking_reasons": [
                "DA2 monocular depth has not been generated and scale-aligned",
                "independent sky and spatial Tile plan are not complete",
            ],
            "bindings": bindings,
        }
    )
    if evidence:
        payload.setdefault("evidence", {}).update(copy.deepcopy(evidence))
    return sign_gate(payload)


def advance_da2_depth_gate(
    lidar_gate: dict[str, Any],
    train_manifest: dict[str, Any],
    val_manifest: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance an exact LiDAR-depth gate by one complete DA2 stage."""
    verify_gate(lidar_gate)
    if (
        lidar_gate.get("status") != LIDAR_DEPTH_READY_STATUS
        or lidar_gate.get("training_allowed") is not False
        or tuple(lidar_gate.get("completed_stages", [])) != ORDERED_STAGES[:12]
    ):
        raise ValueError("DA2 depth requires an exact LIDAR_DEPTH_READY gate")
    train_sha = verify_mono_depth_manifest(train_manifest)
    val_sha = verify_mono_depth_manifest(val_manifest)
    bindings = dict(lidar_gate.get("bindings", {}))
    expected_dataset = bindings.get("training_dataset_manifest_sha256")
    expected_depth = bindings.get("lidar_depth_manifest_sha256")
    expected_faces = {
        "train": bindings.get("face4_train_manifest_sha256"),
        "val": bindings.get("face4_val_manifest_sha256"),
    }
    for split, manifest in (("train", train_manifest), ("val", val_manifest)):
        if manifest.get("split") != split:
            raise ValueError("DA2 manifests must contain train and val splits")
        if manifest.get("complete_face_cache") is not True:
            raise ValueError(f"DA2 {split} cache is incomplete")
        if manifest.get("dataset_manifest_sha256") != expected_dataset:
            raise ValueError(f"DA2 {split} cache is bound to a different dataset")
        if manifest.get("lidar_depth_manifest_sha256") != expected_depth:
            raise ValueError(f"DA2 {split} cache is bound to different LiDAR depth")
        if manifest.get("source_face_manifest_sha256") != expected_faces[split]:
            raise ValueError(f"DA2 {split} cache is bound to different Face4 inputs")
        summary = manifest.get("summary", {})
        if int(summary.get("face_count", 0)) != int(
            manifest.get("expected_face_count", -1)
        ):
            raise ValueError(f"DA2 {split} cache count is inconsistent")
    if train_manifest.get("model") != val_manifest.get("model"):
        raise ValueError("DA2 train and val caches use different models")
    if train_manifest.get("metric_alignment") != val_manifest.get(
        "metric_alignment"
    ):
        raise ValueError("DA2 train and val caches use different alignment rules")
    bindings.update(
        {
            "da2_train_manifest_sha256": train_sha,
            "da2_val_manifest_sha256": val_sha,
            "da2_checkpoint_sha256": train_manifest.get("model", {}).get(
                "checkpoint_sha256"
            ),
        }
    )
    payload = copy.deepcopy(lidar_gate)
    payload.pop("gate_manifest_sha256", None)
    payload.update(
        {
            "status": DA2_DEPTH_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:13]),
            "next_required_stage": ORDERED_STAGES[13],
            "blocking_reasons": [
                "independent sky/background training is not complete",
                "spatial Tile plan is not complete",
            ],
            "bindings": bindings,
        }
    )
    if evidence:
        payload.setdefault("evidence", {}).update(copy.deepcopy(evidence))
    return sign_gate(payload)


def advance_independent_sky_gate(
    da2_gate: dict[str, Any],
    train_evidence: dict[str, Any],
    val_evidence: dict[str, Any],
    initialization: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance an exact DA2 gate by one independent sky/background stage."""

    verify_gate(da2_gate)
    if (
        da2_gate.get("status") != DA2_DEPTH_READY_STATUS
        or da2_gate.get("training_allowed") is not False
        or tuple(da2_gate.get("completed_stages", [])) != ORDERED_STAGES[:13]
    ):
        raise ValueError("independent sky requires an exact DA2_DEPTH_READY gate")
    train_sha = verify_sky_evidence_manifest(train_evidence)
    val_sha = verify_sky_evidence_manifest(val_evidence)
    initialization_sha = verify_sky_initialization_manifest(initialization)
    bindings = dict(da2_gate.get("bindings", {}))
    expected_mono = {
        "train": bindings.get("da2_train_manifest_sha256"),
        "val": bindings.get("da2_val_manifest_sha256"),
    }
    expected_faces = {
        "train": bindings.get("face4_train_manifest_sha256"),
        "val": bindings.get("face4_val_manifest_sha256"),
    }
    for split, manifest in (("train", train_evidence), ("val", val_evidence)):
        if manifest.get("split") != split:
            raise ValueError("sky evidence manifests must contain train and val splits")
        if manifest.get("source_mono_depth_manifest_sha256") != expected_mono[split]:
            raise ValueError(f"sky {split} evidence is bound to different DA2 depth")
        if manifest.get("source_face_manifest_sha256") != expected_faces[split]:
            raise ValueError(f"sky {split} evidence is bound to different Face4 inputs")
        if manifest.get("dataset_manifest_sha256") != bindings.get(
            "training_dataset_manifest_sha256"
        ):
            raise ValueError(f"sky {split} evidence is bound to a different dataset")
        if int(manifest.get("summary", {}).get("accepted_view_count", 0)) <= 0:
            raise ValueError(f"sky {split} evidence accepted no views")
    if train_evidence.get("policy") != val_evidence.get("policy"):
        raise ValueError("sky train and val evidence use different policies")
    if initialization.get("source_sky_evidence_manifest_sha256") != train_sha:
        raise ValueError("sky initialization is not bound to training evidence")
    bindings.update(
        {
            "sky_train_evidence_manifest_sha256": train_sha,
            "sky_val_evidence_manifest_sha256": val_sha,
            "sky_initialization_manifest_sha256": initialization_sha,
            "sky_initialization_artifact_sha256": initialization.get("sha256"),
        }
    )
    payload = copy.deepcopy(da2_gate)
    payload.pop("gate_manifest_sha256", None)
    payload.update(
        {
            "status": SKY_BACKGROUND_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:14]),
            "next_required_stage": ORDERED_STAGES[14],
            "blocking_reasons": [
                "spatial Tile plan has not been recomputed from CloudStudio projections",
            ],
            "bindings": bindings,
        }
    )
    if evidence:
        payload.setdefault("evidence", {}).update(copy.deepcopy(evidence))
    return sign_gate(payload)


def advance_spatial_tile_gate(
    sky_gate: dict[str, Any],
    tile_plan: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance sky-ready inputs to upstream-data-ready, never to training-ready."""

    verify_gate(sky_gate)
    if (
        sky_gate.get("status") != SKY_BACKGROUND_READY_STATUS
        or sky_gate.get("training_allowed") is not False
        or tuple(sky_gate.get("completed_stages", [])) != ORDERED_STAGES[:14]
    ):
        raise ValueError("spatial tiling requires an exact SKY_BACKGROUND_READY gate")
    tile_sha = verify_adaptive_tile_plan(tile_plan)
    bindings = dict(sky_gate.get("bindings", {}))
    plan_bindings = tile_plan.get("source_bindings", {})
    required = {
        "training_dataset_manifest_sha256": bindings.get(
            "training_dataset_manifest_sha256"
        ),
        "face4_train_manifest_sha256": bindings.get("face4_train_manifest_sha256"),
        "sky_train_evidence_manifest_sha256": bindings.get(
            "sky_train_evidence_manifest_sha256"
        ),
    }
    if {key: plan_bindings.get(key) for key in required} != required:
        raise ValueError("spatial Tile plan is bound to different training inputs")
    if int(tile_plan.get("retained_tile_count", 0)) <= 0:
        raise ValueError("spatial Tile plan retains no trainable tiles")
    bindings["spatial_tile_plan_manifest_sha256"] = tile_sha
    payload = copy.deepcopy(sky_gate)
    payload.pop("gate_manifest_sha256", None)
    payload.update(
        {
            "status": UPSTREAM_DATA_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": [
                "training implementation has not passed the MipMap High/type-2 contract",
                "a short GPU smoke has not verified that the Trainer consumes the signed inputs",
            ],
            "bindings": bindings,
        }
    )
    if evidence:
        payload.setdefault("evidence", {}).update(copy.deepcopy(evidence))
    return sign_gate(payload)


def advance_training_implementation_gate(
    upstream_gate: dict[str, Any],
    implementation_contract: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Allow Tile training only after the implementation contract passes."""

    verify_gate(upstream_gate)
    if (
        upstream_gate.get("status") != UPSTREAM_DATA_READY_STATUS
        or upstream_gate.get("training_allowed") is not False
        or tuple(upstream_gate.get("completed_stages", [])) != ORDERED_STAGES[:15]
    ):
        raise ValueError(
            "training implementation requires an exact UPSTREAM_DATA_READY gate"
        )
    contract_sha = verify_training_implementation_contract(
        implementation_contract
    )
    bindings = dict(upstream_gate.get("bindings", {}))
    contract_bindings = implementation_contract.get("source_bindings", {})
    required_bindings = {
        key: bindings.get(key)
        for key in (
            "training_dataset_manifest_sha256",
            "face4_train_manifest_sha256",
            "da2_train_manifest_sha256",
            "spatial_tile_plan_manifest_sha256",
        )
    }
    if {key: contract_bindings.get(key) for key in required_bindings} != required_bindings:
        raise ValueError(
            "training implementation contract is bound to different upstream inputs"
        )
    bindings["training_implementation_contract_sha256"] = contract_sha
    payload = copy.deepcopy(upstream_gate)
    payload.pop("gate_manifest_sha256", None)
    payload.update(
        {
            "status": TRAINING_IMPLEMENTATION_READY_STATUS,
            "training_allowed": True,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": [],
            "bindings": bindings,
        }
    )
    if evidence:
        payload.setdefault("evidence", {}).update(copy.deepcopy(evidence))
    return sign_gate(payload)


def advance_fixed_topology_evaluation_gate(
    upstream_gate: dict[str, Any],
    readiness: dict[str, Any],
    evaluation_plan: dict[str, Any],
    arm_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Authorize only the signed fixed-topology A0/A1 evaluation arms."""

    upstream_sha = verify_gate(upstream_gate)
    if (
        upstream_gate.get("status") != UPSTREAM_DATA_READY_STATUS
        or upstream_gate.get("training_allowed") is not False
        or tuple(upstream_gate.get("completed_stages", [])) != ORDERED_STAGES[:15]
    ):
        raise ValueError(
            "fixed-topology evaluation requires an exact UPSTREAM_DATA_READY gate"
        )

    def verify_signed_payload(
        payload: dict[str, Any], *, hash_key: str, label: str
    ) -> str:
        expected = str(payload.get(hash_key, ""))
        if len(expected) != 64:
            raise ValueError(f"{label} is unsigned")
        unsigned = copy.deepcopy(payload)
        unsigned.pop(hash_key, None)
        actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if actual != expected:
            raise ValueError(f"{label} signature mismatch")
        return expected

    readiness_sha = verify_signed_payload(
        readiness,
        hash_key="readiness_sha256",
        label="fixed-topology readiness",
    )
    plan_sha = verify_signed_payload(
        evaluation_plan,
        hash_key="evaluation_plan_sha256",
        label="fixed-topology evaluation plan",
    )
    if (
        readiness.get("kind") != "fixed_topology_evaluation_readiness_v1"
        or readiness.get("status") != "FIXED_TOPOLOGY_EVALUATION_PREPARED"
        or readiness.get("training_allowed") is not False
        or readiness.get("adaptive_growth_allowed") is not False
    ):
        raise ValueError("fixed-topology readiness is not promotion eligible")
    if readiness.get("upstream_gate_manifest_sha256") != upstream_sha:
        raise ValueError("fixed-topology readiness is bound to another upstream gate")
    if readiness.get("evaluation_plan_sha256") != plan_sha:
        raise ValueError("fixed-topology readiness is bound to another evaluation plan")
    evidence = readiness.get("evidence", {})
    required_true = (
        "directional_pass",
        "phase_a_geometry_frozen",
        "core_only_merge_contract",
    )
    if any(evidence.get(key) is not True for key in required_true):
        raise ValueError("fixed-topology readiness is missing required PASS evidence")
    if (
        evaluation_plan.get("kind") != "fixed_topology_evaluation_plan_v1"
        or evaluation_plan.get("training_allowed") is not False
        or evaluation_plan.get("adaptive_growth_remains_blocked_by") in (None, [])
    ):
        raise ValueError("fixed-topology evaluation plan is not promotion eligible")

    permitted_modes = {"strict_fixed", "opacity_prune_only"}
    allowed_arms: list[dict[str, Any]] = []
    for arm in evaluation_plan.get("arms", []):
        arm_name = str(arm.get("arm", ""))
        config = arm_configs.get(arm_name)
        if config is None:
            raise ValueError(f"fixed-topology arm {arm_name!r} has no config")
        mode = str(config.get("topology_policy", {}).get("mode", ""))
        if mode not in permitted_modes:
            raise ValueError(f"fixed-topology arm {arm_name!r} enables {mode!r}")
        if config.get("mipmap_tile_id") != evaluation_plan.get("tile_id"):
            raise ValueError(f"fixed-topology arm {arm_name!r} targets another Tile")
        if config.get("max_steps") != evaluation_plan.get("steps", {}).get("total"):
            raise ValueError(f"fixed-topology arm {arm_name!r} has another step count")
        allowed_arms.append(
            {
                "arm": arm_name,
                "run_id": str(config.get("run_id", "")),
                "topology_mode": mode,
                "fingerprint_sha256": fixed_topology_evaluation_arm_fingerprint(config),
            }
        )
    if not allowed_arms:
        raise ValueError("fixed-topology evaluation plan has no arms")

    payload = copy.deepcopy(upstream_gate)
    payload.pop("gate_manifest_sha256", None)
    bindings = dict(payload.get("bindings", {}))
    bindings.update(
        {
            "fixed_topology_readiness_sha256": readiness_sha,
            "fixed_topology_evaluation_plan_sha256": plan_sha,
        }
    )
    payload.update(
        {
            "status": FIXED_TOPOLOGY_EVALUATION_READY_STATUS,
            "training_allowed": True,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": [],
            "bindings": bindings,
            "fixed_topology_evaluation": {
                "tile_id": evaluation_plan.get("tile_id"),
                "allowed_arms": allowed_arms,
                "adaptive_growth_allowed": False,
                "core_only_merge_required": True,
            },
        }
    )
    return sign_gate(payload)


def advance_adaptive_growth_gate(
    upstream_gate: dict[str, Any],
    signed_config: dict[str, Any],
    *,
    stage: str,
    boundary_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize one signed LiDAR-guarded classic-growth V26 arm."""

    upstream_sha = verify_gate(upstream_gate)
    if (
        upstream_gate.get("status") != UPSTREAM_DATA_READY_STATUS
        or upstream_gate.get("training_allowed") is not False
        or tuple(upstream_gate.get("completed_stages", [])) != ORDERED_STAGES[:15]
    ):
        raise ValueError("adaptive growth requires an exact UPSTREAM_DATA_READY gate")
    if stage not in {"boundary", "evaluation"}:
        raise ValueError("adaptive growth stage must be boundary or evaluation")
    config_sha = str(signed_config.get("config_manifest_sha256", ""))
    unsigned_config = copy.deepcopy(signed_config)
    unsigned_config.pop("config_manifest_sha256", None)
    if len(config_sha) != 64 or hashlib.sha256(
        canonical_json_bytes(unsigned_config)
    ).hexdigest() != config_sha:
        raise ValueError("adaptive growth config signature mismatch")

    topology = signed_config.get("topology_policy", {})
    proposal = signed_config.get("tangent_proposal", {})
    regularization = signed_config.get("geometry_regularization", {})
    strategy = signed_config.get("default_strategy", {})
    required_strategy = {
        "exact_mipmap_lifecycle": True,
        "grow_grad2d": 0.00015,
        "growth_min_opacity": 0.15,
        "split_scale_m": 0.2,
        "prune_scale_m": 0.2,
        "prune_opa": 0.1,
        "prune_opa_late": 0.05,
        "prune_switch_step": 3740,
        "prune_scale2d": 0.15,
        "reset_every": 300,
        "reset_opacity_cap": 0.2,
        "absgrad": True,
        "revised_opacity": True,
    }
    mismatched_strategy = {
        key: (expected, strategy.get(key))
        for key, expected in required_strategy.items()
        if strategy.get(key) != expected
    }
    if mismatched_strategy:
        raise ValueError(
            f"adaptive growth lifecycle parameters differ: {mismatched_strategy}"
        )
    required_regularization = {
        "enabled": True,
        "opacity_sparsity_weight": 1e-4,
        "scale_upper_weight": 1e-4,
        "anisotropy_weight": 1e-4,
        "max_scale_ratio_to_reference": 8.0,
        "max_anisotropy": 10.0,
    }
    if any(
        regularization.get(key) != expected
        for key, expected in required_regularization.items()
    ):
        raise ValueError("adaptive growth geometry regularization differs")
    if (
        topology.get("mode") != "adaptive_growth"
        or signed_config.get("densification_strategy") != "default_3dgs"
        or signed_config.get("densification_gradient_source") != "rgb_only"
        or signed_config.get("mcmc_refine_start_iter") != 500
        or signed_config.get("mcmc_refine_every") != 100
        or signed_config.get("mcmc_refine_stop_iter") != 5610
        or signed_config.get("max_steps") != 7480
        or signed_config.get("factor") != 1
        or signed_config.get("cap_max") != 2_200_000
        or signed_config.get("sh_degree") != 0
        or signed_config.get("da2_depth_weight") != 0.0
        or signed_config.get("mcmc_noise_lr") != 0.0
        or signed_config.get("mcmc_noise_injection_stop_iter") != 0
        or proposal.get("enabled") is not True
        or proposal.get("reject_unsupported_births") is not True
        or signed_config.get("error_weighted_sampling", {}).get("enabled", False)
    ):
        raise ValueError("adaptive growth config violates the V26 LiDAR-first scope")
    if stage == "boundary":
        if signed_config.get("controlled_stop_after_steps") != 502:
            raise ValueError("adaptive boundary gate requires controlled stop at 502")
        status = ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS
        boundary_sha = None
    else:
        if signed_config.get("controlled_stop_after_steps") is not None:
            raise ValueError("adaptive evaluation config must not retain a controlled stop")
        if not isinstance(boundary_report, dict):
            raise ValueError("adaptive evaluation requires a boundary report")
        boundary_sha = str(boundary_report.get("boundary_report_sha256", ""))
        unsigned_report = copy.deepcopy(boundary_report)
        unsigned_report.pop("boundary_report_sha256", None)
        if len(boundary_sha) != 64 or hashlib.sha256(
            canonical_json_bytes(unsigned_report)
        ).hexdigest() != boundary_sha:
            raise ValueError("adaptive boundary report signature mismatch")
        if (
            boundary_report.get("status") != "ADAPTIVE_GROWTH_BOUNDARY_PASS"
        ):
            raise ValueError("adaptive boundary report is not promotion eligible")
        for field in ("checkpoint_sha256", "source_trainer_config_sha256"):
            value = str(boundary_report.get(field, ""))
            if len(value) != 64:
                raise ValueError(f"adaptive boundary report has invalid {field}")
        status = ADAPTIVE_GROWTH_EVALUATION_READY_STATUS

    payload = copy.deepcopy(upstream_gate)
    payload.pop("gate_manifest_sha256", None)
    bindings = dict(payload.get("bindings", {}))
    bindings["adaptive_growth_config_manifest_sha256"] = config_sha
    if boundary_sha is not None:
        bindings["adaptive_growth_boundary_report_sha256"] = boundary_sha
    payload.update(
        {
            "status": status,
            "training_allowed": True,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": [],
            "bindings": bindings,
            "adaptive_growth": {
                "profile": "v26a_lidar_guarded_classic_growth",
                "stage": stage,
                "tile_id": signed_config.get("mipmap_tile_id"),
                "run_id": signed_config.get("run_id"),
                "controlled_stop_after_steps": signed_config.get(
                    "controlled_stop_after_steps"
                ),
                "mcmc_allowed": False,
                "capacity_cap": signed_config.get("cap_max"),
            },
        }
    )
    if boundary_sha is not None:
        payload["adaptive_growth"].update(
            {
                "resume_checkpoint_sha256": boundary_report.get(
                    "checkpoint_sha256"
                ),
                "resume_source_trainer_config_sha256": boundary_report.get(
                    "source_trainer_config_sha256"
                ),
            }
        )
    return sign_gate(payload)


def advance_adaptive_reallocation_gate(
    upstream_gate: dict[str, Any],
    signed_config: dict[str, Any],
    *,
    stage: str,
    boundary_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize the bounded A0-preserving MCMC reallocation arm.

    Unlike V26 classic growth this profile never prunes or increases capacity:
    it recycles at most two percent of the dense LiDAR cloud per event and
    admits relocation/addition sources only on trusted LiDAR surface patches.
    """

    verify_gate(upstream_gate)
    if (
        upstream_gate.get("status") != UPSTREAM_DATA_READY_STATUS
        or upstream_gate.get("training_allowed") is not False
        or tuple(upstream_gate.get("completed_stages", [])) != ORDERED_STAGES[:15]
    ):
        raise ValueError("adaptive reallocation requires an exact UPSTREAM_DATA_READY gate")
    if stage not in {"boundary", "evaluation", "continuation", "stabilization"}:
        raise ValueError(
            "adaptive reallocation stage must be boundary, evaluation, continuation, "
            "or stabilization"
        )
    config_sha = str(signed_config.get("config_manifest_sha256", ""))
    unsigned_config = copy.deepcopy(signed_config)
    unsigned_config.pop("config_manifest_sha256", None)
    if len(config_sha) != 64 or hashlib.sha256(
        canonical_json_bytes(unsigned_config)
    ).hexdigest() != config_sha:
        raise ValueError("adaptive reallocation config signature mismatch")

    topology = signed_config.get("topology_policy", {})
    proposal = signed_config.get("tangent_proposal", {})
    admission = signed_config.get("lidar_admission", {})
    sampling = signed_config.get("error_weighted_sampling", {})
    tile_id = signed_config.get("mipmap_tile_id")
    profile = V27_SNOW_TILE_PROFILES.get(tile_id)
    if profile is None:
        raise ValueError("adaptive reallocation targets an unknown snow Tile")
    common_required = (
        topology.get("mode") == "adaptive_growth"
        and signed_config.get("densification_strategy") == "error_weighted_mcmc"
        and signed_config.get("densification_gradient_source") == "total_loss"
        and signed_config.get("mcmc_refine_start_iter") == 500
        and signed_config.get("mcmc_refine_every") == 100
        and signed_config.get("mcmc_refine_stop_iter") == 1000
        and signed_config.get("max_steps") == profile["max_steps"]
        and signed_config.get("factor") == 1
        and signed_config.get("cap_max") == profile["cap_max"]
        and signed_config.get("checkpoint_every") == profile["view_count"]
        and signed_config.get("sh_degree") == 0
        and signed_config.get("da2_depth_weight") == 0.0
        and signed_config.get("mcmc_noise_lr") == 0.0
        and signed_config.get("mcmc_noise_injection_stop_iter") == 0
        and signed_config.get("mcmc_growth_rate") == 0.0
        and signed_config.get("mcmc_relocation_max_fraction") == 0.02
        and signed_config.get("mcmc_relocation_max_scale_m") == 0.08
        and signed_config.get("mcmc_relocation_max_anisotropy") == 8.0
        and bool(
            signed_config.get("warm_start_checkpoint")
            or signed_config.get("resume_checkpoint")
        )
    )
    if stage == "stabilization":
        learning_rates = signed_config.get("learning_rates", {})
        required = (
            common_required
            and sampling.get("enabled") is False
            and admission.get("enabled") is False
            and proposal.get("enabled") is False
            and signed_config.get("contribution", {}).get("enabled") is False
            and learning_rates.get("means") == 4e-06
            and learning_rates.get("scales") == 0.001
            and learning_rates.get("quats") == 0.0002
            and learning_rates.get("opacities") == 0.001
            and learning_rates.get("colors") == 0.0005
            and signed_config.get("post_refine_geometry_lr_scale") == 0.0
        )
    else:
        required = (
            common_required
            and sampling.get("enabled") is True
            and sampling.get("aggregation") == "contribution"
            and admission.get("enabled") is True
            and proposal.get("enabled") is True
            and proposal.get("reject_unsupported_births") is True
        )
    if not required:
        raise ValueError("adaptive reallocation config violates the V27 A0-safe scope")

    if stage == "boundary":
        if signed_config.get("controlled_stop_after_steps") != 602:
            raise ValueError("adaptive reallocation boundary requires stop at 602")
        status = ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS
        boundary_sha = None
    else:
        controlled_stop = signed_config.get("controlled_stop_after_steps")
        if stage == "evaluation" and controlled_stop != profile["review_stop"]:
            raise ValueError(
                "adaptive reallocation review must stop at "
                f"{profile['review_stop']} for Tile_{tile_id}"
            )
        if stage == "continuation" and controlled_stop is not None:
            raise ValueError("adaptive reallocation continuation must not stop early")
        if (
            stage == "stabilization"
            and controlled_stop != profile["stabilization_stop"]
        ):
            raise ValueError(
                "adaptive reallocation stabilization must stop at "
                f"{profile['stabilization_stop']} for Tile_{tile_id}"
            )
        if not isinstance(boundary_report, dict):
            raise ValueError("adaptive reallocation evaluation requires a boundary report")
        boundary_sha = str(boundary_report.get("boundary_report_sha256", ""))
        unsigned_report = copy.deepcopy(boundary_report)
        unsigned_report.pop("boundary_report_sha256", None)
        if len(boundary_sha) != 64 or hashlib.sha256(
            canonical_json_bytes(unsigned_report)
        ).hexdigest() != boundary_sha:
            raise ValueError("adaptive reallocation boundary signature mismatch")
        if boundary_report.get("status") != "ADAPTIVE_REALLOCATION_BOUNDARY_PASS":
            raise ValueError("adaptive reallocation boundary is not promotion eligible")
        for field in ("checkpoint_sha256", "source_trainer_config_sha256"):
            if len(str(boundary_report.get(field, ""))) != 64:
                raise ValueError(f"adaptive reallocation report has invalid {field}")
        status = ADAPTIVE_GROWTH_EVALUATION_READY_STATUS

    payload = copy.deepcopy(upstream_gate)
    payload.pop("gate_manifest_sha256", None)
    bindings = dict(payload.get("bindings", {}))
    bindings["adaptive_growth_config_manifest_sha256"] = config_sha
    if boundary_sha is not None:
        bindings["adaptive_growth_boundary_report_sha256"] = boundary_sha
    payload.update(
        {
            "status": status,
            "training_allowed": True,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": [],
            "bindings": bindings,
            "adaptive_growth": {
                "profile": (
                    "v27c_a0_safe_stabilization"
                    if stage == "stabilization"
                    else "v27a_a0_safe_mcmc_reallocation"
                ),
                "stage": stage,
                "tile_id": signed_config.get("mipmap_tile_id"),
                "run_id": signed_config.get("run_id"),
                "controlled_stop_after_steps": signed_config.get(
                    "controlled_stop_after_steps"
                ),
                "mcmc_allowed": stage != "stabilization",
                "capacity_cap": signed_config.get("cap_max"),
                "growth_rate": 0.0,
                "relocation_max_fraction": 0.02,
            },
        }
    )
    if boundary_sha is not None:
        payload["adaptive_growth"].update(
            {
                "resume_checkpoint_sha256": boundary_report.get("checkpoint_sha256"),
                "resume_source_trainer_config_sha256": boundary_report.get(
                    "source_trainer_config_sha256"
                ),
            }
        )
    return sign_gate(payload)


def verify_training_gate(
    path: Path,
    *,
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    face_manifest_sha256: str | None,
    allow_implementation_smoke: bool = False,
    fixed_topology_arm_fingerprint_sha256: str | None = None,
    adaptive_growth_config_manifest_sha256: str | None = None,
) -> str:
    """Reject training until implementation passed, except an explicit bounded smoke."""
    if face_manifest_sha256 is None:
        raise ValueError(
            "MipMap-aligned training requires a signed Face4 training manifest"
        )
    gate, gate_sha = load_and_verify_gate(path)
    implementation_ready = (
        gate.get("status") == TRAINING_IMPLEMENTATION_READY_STATUS
        and bool(gate.get("training_allowed", False))
    )
    fixed_topology_ready = (
        gate.get("status") == FIXED_TOPOLOGY_EVALUATION_READY_STATUS
        and bool(gate.get("training_allowed", False))
    )
    adaptive_growth_ready = (
        gate.get("status")
        in {
            ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS,
            ADAPTIVE_GROWTH_EVALUATION_READY_STATUS,
        }
        and bool(gate.get("training_allowed", False))
    )
    smoke_ready = (
        allow_implementation_smoke
        and gate.get("status") == UPSTREAM_DATA_READY_STATUS
        and gate.get("training_allowed") is False
    )
    if (
        not implementation_ready
        and not fixed_topology_ready
        and not adaptive_growth_ready
        and not smoke_ready
    ):
        next_stage = gate.get("next_required_stage", "unknown")
        raise ValueError(
            "MipMap-aligned training is blocked: data/implementation gate is only "
            f"{gate.get('status')!r}; next required stage is {next_stage!r}. "
            "UPSTREAM_DATA_READY does not authorize training"
        )
    if tuple(gate.get("completed_stages", [])) != ORDERED_STAGES[:15]:
        raise ValueError(
            "MipMap training gate must complete every stage through spatial_tile_plan"
        )
    bindings = gate.get("bindings", {})
    expected = {
        "training_dataset_manifest_sha256": str(dataset_manifest_sha256),
        "split_manifest_sha256": str(split_manifest_sha256),
        "face4_train_manifest_sha256": str(face_manifest_sha256),
    }
    actual = {key: bindings.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"MipMap training gate is bound to different inputs: "
            f"expected={expected}, actual={actual}"
        )
    if implementation_ready and len(
        str(bindings.get("training_implementation_contract_sha256", ""))
    ) != 64:
        raise ValueError(
            "MipMap training gate has no verified training implementation contract"
        )
    if fixed_topology_ready:
        allowed = gate.get("fixed_topology_evaluation", {}).get("allowed_arms", [])
        allowed_fingerprints = {
            str(arm.get("fingerprint_sha256", "")) for arm in allowed
        }
        if fixed_topology_arm_fingerprint_sha256 not in allowed_fingerprints:
            raise ValueError(
                "Trainer config is not an authorized fixed-topology evaluation arm"
            )
        if gate.get("fixed_topology_evaluation", {}).get(
            "adaptive_growth_allowed"
        ) is not False:
            raise ValueError("fixed-topology evaluation gate must block adaptive growth")
    if adaptive_growth_ready:
        expected_config_sha = str(
            bindings.get("adaptive_growth_config_manifest_sha256", "")
        )
        if (
            len(expected_config_sha) != 64
            or adaptive_growth_config_manifest_sha256 != expected_config_sha
        ):
            raise ValueError(
                "Trainer config is not the signed adaptive-growth evaluation arm"
            )
        adaptive = gate.get("adaptive_growth", {})
        if adaptive.get("mcmc_allowed") not in {False, True}:
            raise ValueError("adaptive-growth gate must declare whether MCMC is allowed")
    return gate_sha
