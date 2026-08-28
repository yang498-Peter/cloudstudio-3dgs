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
# Compatibility import for callers. Old signed gates whose literal status is
# ``TRAINING_READY`` are intentionally not accepted by verify_training_gate.
TRAINING_READY_STATUS = TRAINING_IMPLEMENTATION_READY_STATUS
TRAINING_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION = 1
TRAINING_IMPLEMENTATION_CONTRACT_KIND = "mipmap_training_implementation_contract"
MIPMAP_HIGH_TYPE2_PRESET = "lidar_first_face4_snow_v1"
INDEPENDENT_AT_ALGORITHM = "independent_pos_prior_shared_single_focal_kb4_at_v2"


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


def verify_training_gate(
    path: Path,
    *,
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    face_manifest_sha256: str | None,
    allow_implementation_smoke: bool = False,
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
    smoke_ready = (
        allow_implementation_smoke
        and gate.get("status") == UPSTREAM_DATA_READY_STATUS
        and gate.get("training_allowed") is False
    )
    if not implementation_ready and not smoke_ready:
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
    if (
        not allow_implementation_smoke
        and len(str(bindings.get("training_implementation_contract_sha256", ""))) != 64
    ):
        raise ValueError(
            "MipMap training gate has no verified training implementation contract"
        )
    return gate_sha
