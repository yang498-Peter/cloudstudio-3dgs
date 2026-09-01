"""Signed, fail-closed LiDAR-first Face4 training parameter contract.

The upstream pipeline gate proves that the accepted AT, Face4, depth, sky
evidence, and spatial Tile plan exist and share identities.  It does not prove
that the trainer consumes those products with the recovered closed-source
schedule.  This module deliberately keeps formal training blocked while that
second, independent parity layer is incomplete.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.adaptive_tiling import verify_adaptive_tile_plan
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    ORDERED_STAGES,
    UPSTREAM_DATA_READY_STATUS,
    verify_gate,
)
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest


PARAMETER_SPEC_SCHEMA_VERSION = 2
PARAMETER_SPEC_KIND = "lidar_first_face4_parameter_spec_v2"
PARAMETER_SPEC_STATUS = "PARAMETER_SPEC_READY"


BLOCKING_REQUIREMENTS = (
    "exact_lidar_k7_k30_initialization",
    "tile_face4_crop_and_intrinsics_consumer",
    "renderer_validity_and_dynamic_mask_consumer",
    "face4_sparse_metric_lidar_consumer",
    "lidar_surface_normal_anchor_consumer",
    "lidar_surface_birth_guard",
    "lidar_first_loss_schedule",
    "epoch_permutation_view_sampler",
    "gradient_split_clone_cull_reset_lifecycle",
    "independent_sh1_sky_trainer",
    "tile_merge_raw_fisheye_eval_and_export",
    "resolved_gaussian_capacity_policy",
)


def sign_parameter_spec(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("parameter_spec_sha256", None)
    signed = copy.deepcopy(unsigned)
    signed["parameter_spec_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def verify_parameter_spec(spec: dict[str, Any]) -> str:
    expected = str(spec.get("parameter_spec_sha256", ""))
    if len(expected) != 64:
        raise ValueError("MipMap parameter specification is unsigned")
    unsigned = copy.deepcopy(spec)
    unsigned.pop("parameter_spec_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("MipMap parameter specification signature mismatch")
    if int(spec.get("schema_version", -1)) != PARAMETER_SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported MipMap parameter specification schema")
    if spec.get("kind") != PARAMETER_SPEC_KIND:
        raise ValueError("unexpected MipMap parameter specification kind")
    if spec.get("status") != PARAMETER_SPEC_STATUS:
        raise ValueError("unexpected MipMap parameter specification status")
    if spec.get("training_allowed") is not False:
        raise ValueError("a research parameter specification may not allow training")
    requirements = spec.get("implementation_parity", {}).get("requirements", [])
    identifiers = tuple(str(item.get("id", "")) for item in requirements)
    if identifiers != BLOCKING_REQUIREMENTS:
        raise ValueError("parameter specification parity requirements are incomplete")
    if any(item.get("status") != "NOT_IMPLEMENTED_OR_NOT_BOUND" for item in requirements):
        raise ValueError("parameter specification must not claim implementation parity")
    return expected


def build_high_type2_parameter_spec(
    upstream_gate: dict[str, Any],
    tile_plan: dict[str, Any],
    tile_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Bind recovered parameters to exact snow Tile inputs without enabling training."""

    upstream_sha = verify_gate(upstream_gate)
    tile_plan_sha = verify_adaptive_tile_plan(tile_plan)
    tile_inputs_sha = verify_tile_inputs_manifest(tile_inputs)
    if (
        upstream_gate.get("status") != UPSTREAM_DATA_READY_STATUS
        or upstream_gate.get("training_allowed") is not False
        or tuple(upstream_gate.get("completed_stages", [])) != ORDERED_STAGES[:15]
    ):
        raise ValueError("parameter research requires an exact upstream input-ready gate")
    bound_plan = upstream_gate.get("bindings", {}).get(
        "spatial_tile_plan_manifest_sha256"
    )
    if bound_plan != tile_plan_sha:
        raise ValueError("upstream gate is bound to a different spatial Tile plan")
    if tile_inputs.get("tile_plan_manifest_sha256") != tile_plan_sha:
        raise ValueError("Tile inputs are bound to a different spatial Tile plan")

    plan_by_id = {int(tile["tile_id"]): tile for tile in tile_plan["tiles"]}
    tile_contracts: list[dict[str, Any]] = []
    for tile in tile_inputs["tiles"]:
        tile_id = int(tile["tile_id"])
        if tile_id not in plan_by_id:
            raise ValueError(f"Tile input references unknown Tile {tile_id}")
        plan_tile = plan_by_id[tile_id]
        view_count = int(tile["view_count"])
        if view_count != int(plan_tile["valid_view_count"]):
            raise ValueError(f"Tile {tile_id} view count differs from the spatial plan")
        point_count = int(tile["initialization"]["point_count"])
        if point_count <= 0 or view_count <= 0:
            raise ValueError(f"Tile {tile_id} has no initialization points or views")
        expected_steps = 20 * view_count
        recommended = tile.get("recommended_training", {})
        if int(recommended.get("steps", -1)) != expected_steps:
            raise ValueError(f"Tile {tile_id} does not carry the High 20*V schedule")
        tile_contracts.append(
            {
                "tile_id": tile_id,
                "name": str(tile["name"]),
                "view_count": view_count,
                "initial_point_count": point_count,
                "total_steps": expected_steps,
                "stage_step_boundaries": [
                    0,
                    5 * view_count,
                    15 * view_count,
                    20 * view_count,
                ],
                "initialization_sha256": tile["initialization"]["sha256"],
                "estimated_photo_memory_gib": float(
                    plan_tile["estimated_memory_gib"]
                ),
            }
        )

    requirements = [
        {
            "id": identifier,
            "status": "NOT_IMPLEMENTED_OR_NOT_BOUND",
            "required_for": "LONG_TRAINING_ALLOWED",
        }
        for identifier in BLOCKING_REQUIREMENTS
    ]
    payload: dict[str, Any] = {
        "schema_version": PARAMETER_SPEC_SCHEMA_VERSION,
        "kind": PARAMETER_SPEC_KIND,
        "status": PARAMETER_SPEC_STATUS,
        "training_allowed": False,
        "bindings": {
            "upstream_gate_sha256": upstream_sha,
            "tile_plan_manifest_sha256": tile_plan_sha,
            "tile_inputs_manifest_sha256": tile_inputs_sha,
            "training_dataset_manifest_sha256": upstream_gate.get("bindings", {}).get(
                "training_dataset_manifest_sha256"
            ),
            "face4_train_manifest_sha256": upstream_gate.get("bindings", {}).get(
                "face4_train_manifest_sha256"
            ),
            "renderer_mask_train_manifest_sha256": upstream_gate.get(
                "bindings", {}
            ).get("renderer_mask_train_manifest_sha256"),
            "da2_train_manifest_sha256": upstream_gate.get("bindings", {}).get(
                "da2_train_manifest_sha256"
            ),
            "sky_initialization_manifest_sha256": upstream_gate.get(
                "bindings", {}
            ).get("sky_initialization_manifest_sha256"),
        },
        "evidence_boundary": {
            "upstream_gate_semantics": "UPSTREAM_INPUTS_READY_ONLY",
            "parameter_values": "RECOVERED_STATIC_AND_TASK_EVIDENCE",
            "implementation_parity": "NOT_YET_PROVEN",
            "long_training": "BLOCKED",
        },
        "tiles": tile_contracts,
        "lidar_first_face4": {
            "decision": {
                "geometry_authority": "REAL_LIDAR",
                "da2": "OPTIONAL_DEFERRED_DISABLED",
                "mesh_depth": "OPTIONAL_DEFERRED_DISABLED",
                "mesh_normal": "OPTIONAL_DEFERRED_DISABLED",
                "reason": (
                    "DA2 adds model-biased relative depth and mesh only "
                    "interpolates existing LiDAR; neither blocks the measured "
                    "surface baseline"
                ),
            },
            "resolution_level": 1,
            "stage_epochs": [5, 10, 5],
            "view_sampling": "fresh_fisher_yates_permutation_without_replacement_per_epoch",
            "surface_training_sh_degree": 1,
            "surface_persisted_color": "SH_DC_ONLY",
            "sky_sh_degree": 1,
            "initialization": {
                "scale_knn_including_self": 7,
                "scale_distance": "mean_euclidean_distance_neighbors_1_to_6",
                "linear_scale_axes": [1.0, 1.0, 0.5],
                "normal_knn": 30,
                "rotation": "shortest_arc_quaternion_local_plus_z_to_unoriented_normal",
                "opacity_probability": 0.1,
                "color": "rgb_over_255_to_sh0",
            },
            "optimizer": {
                "adam_groups": "six_independent",
                "xyz_lr_initial": 0.000016,
                "xyz_lr_final": 0.0000016,
                "scale_lr": 0.005,
                "quaternion_lr": 0.001,
                "sh_dc_lr": 0.0025,
                "sh_rest_lr": 0.000125,
                "opacity_lr": 0.05,
                "adam_epsilon": 1e-15,
            },
            "loss": {
                "rgb_mean_l1": 0.6,
                "rgb_dssim": 0.4,
                "sparse_real_lidar_range": "positive_full_training",
                "lidar_surface_normal_alignment": "positive_full_training",
                "da2_mono_depth": 0.0,
                "mesh_depth": 0.0,
                "mesh_normal": 0.0,
                "opacity_mean": 0.01,
                "sky_opacity": "0.04_after_10V_when_available",
                "scale_regularizers": 0.0,
                "opacity_binarization": 0.0,
            },
            "lifecycle": {
                "mode": "gradient_split_clone_cull_reset",
                "mcmc_relocation": False,
                "redundancy_cull": False,
                "start_step": 500,
                "interval_steps": 100,
                "gradient_threshold": 0.00015,
                # growth_min_opacity, parent_surface_gate, newborn_position and
                # unsupported_birth_policy are REMOVED. A later pass over the
                # same evidence found the opacity>0.15 birth gate is computed
                # and never consumed, and the parent/newborn gates were never
                # active for this preset; enforcing them barred 21% of the
                # population from being born. A signed contract that still
                # asserts a retracted reading launders it back into evidence,
                # which is worse than having no contract at all.
                "clone_max_linear_scale_m": 0.2,
                "split_min_linear_scale_m": 0.2,
                "split_children": 2,
                "split_scale_divisor": 1.6,
                "cull_opacity_first_half": 0.1,
                "cull_opacity_second_half": 0.05,
                "cull_max_linear_scale_m": 0.2,
                "cull_max_screen_radius": 0.15,
                # NOT closed from the source evidence. What is recorded is a
                # reset parameter of 30 against a 100-step refine interval,
                # which reads as ~3000 steps rather than 300, and the
                # lowest-level audit leaves the interval explicitly unresolved.
                # Every measured population collapse lands at reset+100, so
                # this cadence is a prime suspect and must not carry an "exact"
                # label until the 300-vs-3000 comparison settles it.
                "opacity_reset_step_period": "UNRESOLVED_300_OR_3000",
                "opacity_reset_probability_cap": 0.2,
            },
            "deferred_optional_experiments": {
                "da2_low_weight_ab": "ONLY_IF_LIDAR_BASELINE_QUALITY_IS_INSUFFICIENT",
                "confidence_gated_local_mesh": "ONLY_IF_LIDAR_BASELINE_QUALITY_IS_INSUFFICIENT",
                "bilateral_grid": "NOT_REQUIRED_FOR_GEOMETRY_BASELINE",
                "sift_pose_refinement": "NOT_REQUIRED_AFTER_ACCEPTED_AT_BASELINE",
            },
            "capacity": {
                "formula": "max(3000000,min(500000*C,10*N_input))",
                "C": None,
                "status": "UNRESOLVED_VENDOR_MULTIPLIER",
            },
        },
        "implementation_parity": {
            "status": "BLOCKED",
            "requirements": requirements,
            "next_required_artifact": "exact_lidar_tile_initialization_geometry_manifest",
        },
    }
    return sign_parameter_spec(payload)


def load_and_verify_parameter_spec(path: Path) -> tuple[dict[str, Any], str]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return spec, verify_parameter_spec(spec)
