import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cloudstudio_3dgs.pipeline.mipmap_gate import (
    FRONTEND_READY_STATUS,
    DA2_DEPTH_READY_STATUS,
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    LIDAR_DEPTH_READY_STATUS,
    ORDERED_STAGES,
    MIPMAP_HIGH_TYPE2_PRESET,
    MIPMAP_HIGH_TYPE2_REQUIRED_IMPLEMENTATION,
    TRAINING_IMPLEMENTATION_CONTRACT_KIND,
    TRAINING_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION,
    TRAINING_IMPLEMENTATION_READY_STATUS,
    FIXED_TOPOLOGY_EVALUATION_READY_STATUS,
    ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS,
    ADAPTIVE_GROWTH_REVIEW_READY_STATUS,
    TRAINING_READY_STATUS,
    UPSTREAM_DATA_READY_STATUS,
    advance_training_implementation_gate,
    advance_fixed_topology_evaluation_gate,
    advance_adaptive_growth_gate,
    advance_lidar_depth_gate,
    advance_da2_depth_gate,
    advance_renderer_mask_gate,
    sign_gate,
    sign_training_implementation_contract,
    fixed_topology_evaluation_arm_fingerprint,
    verify_gate,
    verify_training_gate,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.renderer_masks import sign_renderer_mask_manifest
from cloudstudio_3dgs.data.mono_depth import sign_mono_depth_manifest
from cloudstudio_3dgs.training.trainer import TrainerConfig


DATASET_SHA = "1" * 64
SPLIT_SHA = "2" * 64
FACE_SHA = "3" * 64
MASK_SHA = "4" * 64
DA2_SHA = "5" * 64
TILE_SHA = "6" * 64


def _gate(*, ready: bool) -> dict:
    stage_count = 15 if ready else 10
    bindings = {
        "training_dataset_manifest_sha256": DATASET_SHA,
        "split_manifest_sha256": SPLIT_SHA,
        "face4_train_manifest_sha256": FACE_SHA,
        "face4_val_manifest_sha256": FACE_SHA,
        "training_circle_mask_manifest_sha256": MASK_SHA,
    }
    if ready:
        bindings["training_implementation_contract_sha256"] = "7" * 64
    return sign_gate(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "profile": GATE_PROFILE,
            "status": (
                TRAINING_IMPLEMENTATION_READY_STATUS
                if ready
                else FRONTEND_READY_STATUS
            ),
            "training_allowed": ready,
            "completed_stages": list(ORDERED_STAGES[:stage_count]),
            "next_required_stage": ORDERED_STAGES[stage_count],
            "bindings": bindings,
        }
    )


def _upstream_data_gate() -> dict:
    return sign_gate(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "profile": GATE_PROFILE,
            "status": UPSTREAM_DATA_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": ["training implementation is not aligned"],
            "bindings": {
                "training_dataset_manifest_sha256": DATASET_SHA,
                "split_manifest_sha256": SPLIT_SHA,
                "face4_train_manifest_sha256": FACE_SHA,
                "face4_val_manifest_sha256": FACE_SHA,
                "training_circle_mask_manifest_sha256": MASK_SHA,
                "da2_train_manifest_sha256": DA2_SHA,
                "spatial_tile_plan_manifest_sha256": TILE_SHA,
            },
        }
    )


def _implementation_contract(*, gpu_smoke: bool = True) -> dict:
    return sign_training_implementation_contract(
        {
            "schema_version": TRAINING_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION,
            "kind": TRAINING_IMPLEMENTATION_CONTRACT_KIND,
            "preset": MIPMAP_HIGH_TYPE2_PRESET,
            "trainer_source_sha256": "8" * 64,
            "trainer_config_sha256": "9" * 64,
            "source_bindings": {
                "training_dataset_manifest_sha256": DATASET_SHA,
                "face4_train_manifest_sha256": FACE_SHA,
                "da2_train_manifest_sha256": DA2_SHA,
                "spatial_tile_plan_manifest_sha256": TILE_SHA,
            },
            "implementation": copy.deepcopy(
                MIPMAP_HIGH_TYPE2_REQUIRED_IMPLEMENTATION
            ),
            "verification": {
                "cpu_contract_tests_passed": True,
                "short_gpu_smoke_passed": gpu_smoke,
            },
            "unresolved_blockers": [],
        }
    )


class MipMapPipelineGateTests(unittest.TestCase):
    def test_adaptive_boundary_gate_binds_signed_v26_config(self) -> None:
        config = {
            "run_id": "v26-boundary",
            "mipmap_tile_id": 1,
            "topology_policy": {"mode": "adaptive_growth"},
            "densification_strategy": "default_3dgs",
            "densification_gradient_source": "rgb_only",
            "mcmc_refine_start_iter": 500,
            "mcmc_refine_every": 100,
            "mcmc_refine_stop_iter": 5610,
            "mcmc_noise_injection_stop_iter": 0,
            "mcmc_noise_lr": 0.0,
            "max_steps": 7480,
            "controlled_stop_after_steps": 502,
            "factor": 1,
            "cap_max": 2200000,
            "sh_degree": 0,
            "da2_depth_weight": 0.0,
            "error_weighted_sampling": {"enabled": False},
            "default_strategy": {
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
            },
            "tangent_proposal": {
                "enabled": True,
                "reject_unsupported_births": True,
            },
            "geometry_regularization": {
                "enabled": True,
                "opacity_sparsity_weight": 1e-4,
                "scale_upper_weight": 1e-4,
                "anisotropy_weight": 1e-4,
                "max_scale_ratio_to_reference": 8.0,
                "max_anisotropy": 10.0,
            },
        }
        config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(config)
        ).hexdigest()
        gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), config, stage="boundary"
        )
        self.assertEqual(gate["status"], ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            verify_training_gate(
                path,
                dataset_manifest_sha256=DATASET_SHA,
                split_manifest_sha256=SPLIT_SHA,
                face_manifest_sha256=FACE_SHA,
                adaptive_growth_config_manifest_sha256=config[
                    "config_manifest_sha256"
                ],
            )

        review_config = copy.deepcopy(config)
        review_config["controlled_stop_after_steps"] = 1002
        review_config.pop("config_manifest_sha256")
        review_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(review_config)
        ).hexdigest()
        boundary_report = {
            "status": "ADAPTIVE_GROWTH_BOUNDARY_PASS",
            "checkpoint_sha256": "a" * 64,
            "source_trainer_config_sha256": "b" * 64,
        }
        boundary_report["boundary_report_sha256"] = hashlib.sha256(
            canonical_json_bytes(boundary_report)
        ).hexdigest()
        review_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(),
            review_config,
            stage="review",
            boundary_report=boundary_report,
        )
        self.assertEqual(
            review_gate["status"], ADAPTIVE_GROWTH_REVIEW_READY_STATUS
        )

        protected_config = copy.deepcopy(config)
        protected_config["run_id"] = "v34-cull-protected-boundary"
        protected_config["default_strategy"].update(
            {
                "opacity_cull_policy": "observation_aware",
                "opacity_cull_min_observations": 4,
                "opacity_cull_consecutive_events": 2,
                "opacity_cull_grace_after_reset_steps": 200,
                "opacity_cull_max_fraction": 0.05,
            }
        )
        protected_config.pop("config_manifest_sha256")
        protected_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(protected_config)
        ).hexdigest()
        protected_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), protected_config, stage="boundary"
        )
        self.assertEqual(
            protected_gate["cloudstudio_cull_enhancement"]["policy"],
            "observation_aware",
        )

        detail_config = copy.deepcopy(protected_config)
        detail_config["run_id"] = "v35-detail-split-boundary"
        detail_config["default_strategy"].update(
            {
                "detail_split_policy": "lidar_surface_screen_detail",
                "detail_split_scale_m": 0.02,
                "detail_split_screen_radius": 0.0035,
                "opacity_cull_priority": "lowest_opacity_per_footprint",
            }
        )
        detail_config.pop("config_manifest_sha256")
        detail_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(detail_config)
        ).hexdigest()
        detail_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), detail_config, stage="boundary"
        )
        self.assertEqual(
            detail_gate["cloudstudio_detail_split_enhancement"]["policy"],
            "lidar_surface_screen_detail",
        )

        vendor_gradient_config = copy.deepcopy(detail_config)
        vendor_gradient_config["run_id"] = "v35-vendor-gradient-boundary"
        vendor_gradient_config["default_strategy"].update(
            {"absgrad": False, "grow_grad2d": 0.00015}
        )
        vendor_gradient_config["geometry_regularization"].update(
            {"anisotropy_weight": 0.0, "max_anisotropy": 256.0}
        )
        vendor_gradient_config.pop("config_manifest_sha256")
        vendor_gradient_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(vendor_gradient_config)
        ).hexdigest()
        vendor_gradient_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), vendor_gradient_config, stage="boundary"
        )
        self.assertEqual(
            vendor_gradient_gate["projected_gradient_profile"],
            "vendor_plain_1p5e4",
        )
        self.assertEqual(
            vendor_gradient_gate["shape_regularization_profile"],
            "thin_surfel_unpenalized",
        )

        local_cull_config = copy.deepcopy(vendor_gradient_config)
        local_cull_config["run_id"] = "v36-local-coverage-boundary"
        local_cull_config["default_strategy"].update(
            {
                "opacity_cull_policy": "local_coverage_competition",
                "opacity_cull_min_observations": 4,
                "opacity_cull_consecutive_events": 2,
                "opacity_cull_grace_after_reset_steps": 100,
                "opacity_cull_max_fraction": 0.05,
                "opacity_cull_local_voxel_m": 0.02,
            }
        )
        local_cull_config.setdefault("lidar_normal_alignment", {}).update(
            {
                "enabled": True,
                "weight_point_to_plane": 0.01,
                "point_to_plane_huber_delta_m": 0.02,
            }
        )
        local_cull_config.pop("config_manifest_sha256")
        local_cull_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(local_cull_config)
        ).hexdigest()
        local_cull_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), local_cull_config, stage="boundary"
        )
        self.assertEqual(
            local_cull_gate["cloudstudio_cull_enhancement"]["policy"],
            "local_coverage_competition",
        )
        self.assertEqual(
            local_cull_gate["surface_motion_profile"],
            "soft_point_to_plane_2cm",
        )

        coverage_cull_config = copy.deepcopy(local_cull_config)
        coverage_cull_config["run_id"] = "v36b-coverage-weighted-boundary"
        coverage_cull_config["default_strategy"].update(
            {
                "opacity_cull_max_fraction": 0.02,
                "opacity_cull_local_protection": "opacity_tangent_area",
            }
        )
        coverage_cull_config["lidar_normal_alignment"].update(
            {"weight_flatten": 0.1, "flatten_target_m": 0.001}
        )
        coverage_cull_config.pop("config_manifest_sha256")
        coverage_cull_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(coverage_cull_config)
        ).hexdigest()
        coverage_cull_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), coverage_cull_config, stage="boundary"
        )
        self.assertEqual(
            coverage_cull_gate["cloudstudio_cull_enhancement"][
                "local_protection"
            ],
            "opacity_tangent_area",
        )
        self.assertEqual(
            coverage_cull_gate["cloudstudio_cull_enhancement"][
                "max_fraction_per_event"
            ],
            0.02,
        )
        self.assertEqual(
            coverage_cull_gate["flatten_profile"], "thin_surface_1mm"
        )

        ratio_flatten_config = copy.deepcopy(coverage_cull_config)
        ratio_flatten_config["run_id"] = "v37-ratio-flatten-boundary"
        ratio_flatten_config["lidar_normal_alignment"].update(
            {
                "weight_flatten": 0.01,
                "flatten_mode": "tangent_ratio",
                "flatten_ratio_target": 0.15,
            }
        )
        ratio_flatten_config.update(
            {"lidar_alpha_weight": 0.02, "lidar_alpha_target": 0.95}
        )
        ratio_flatten_config["default_strategy"][
            "opacity_cull_local_min_accumulated_alpha"
        ] = 0.5
        ratio_flatten_config.pop("config_manifest_sha256")
        ratio_flatten_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(ratio_flatten_config)
        ).hexdigest()
        ratio_flatten_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), ratio_flatten_config, stage="boundary"
        )
        self.assertEqual(
            ratio_flatten_gate["flatten_profile"],
            "thin_surface_ratio_0p15",
        )
        self.assertEqual(
            ratio_flatten_gate["lidar_alpha_profile"],
            "signed_lidar_alpha_floor_0p95",
        )
        self.assertEqual(
            ratio_flatten_gate["cloudstudio_cull_enhancement"][
                "local_min_accumulated_alpha"
            ],
            0.5,
        )

        vendor_pre_optimizer_config = copy.deepcopy(config)
        vendor_pre_optimizer_config["run_id"] = "v39-vendor-pre-optimizer"
        vendor_pre_optimizer_config["densification_gradient_source"] = "total_loss"
        vendor_pre_optimizer_config["default_strategy"].update(
            {
                "lifecycle_execution_order": "pre_optimizer_vendor",
                "absgrad": False,
                "revised_opacity": False,
                "detail_split_policy": "vendor_0_2m",
                "opacity_cull_policy": "immediate",
                "opacity_cull_min_observations": 0,
                "opacity_cull_consecutive_events": 1,
                "opacity_cull_grace_after_reset_steps": 0,
                "opacity_cull_max_fraction": 1.0,
                "opacity_cull_priority": "lowest_opacity",
                "opacity_cull_local_min_accumulated_alpha": 0.0,
            }
        )
        vendor_pre_optimizer_config["tangent_proposal"] = {
            "enabled": False,
            "reject_unsupported_births": False,
        }
        vendor_pre_optimizer_config["geometry_regularization"].update(
            {
                "opacity_sparsity_weight": 0.01,
                "scale_upper_weight": 0.0,
                "anisotropy_weight": 0.0,
                "max_anisotropy": 256.0,
            }
        )
        vendor_pre_optimizer_config["lidar_normal_alignment"] = {
            "weight_point_to_plane": 0.0,
            "point_to_plane_huber_delta_m": 0.02,
            "weight_flatten": 0.0,
            "flatten_mode": "absolute_m",
            "flatten_target_m": 0.02,
        }
        vendor_pre_optimizer_config["lidar_alpha_weight"] = 0.0
        vendor_pre_optimizer_config["lidar_alpha_target"] = 0.95
        vendor_pre_optimizer_config.pop("config_manifest_sha256")
        vendor_pre_optimizer_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(vendor_pre_optimizer_config)
        ).hexdigest()
        vendor_pre_optimizer_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(),
            vendor_pre_optimizer_config,
            stage="boundary",
        )
        self.assertEqual(
            vendor_pre_optimizer_gate["lifecycle_execution_order"],
            "pre_optimizer_vendor",
        )
        self.assertEqual(
            vendor_pre_optimizer_gate["adaptive_growth"]["profile"],
            "v39_vendor_pre_optimizer_classic_ppisp_compat",
        )

        warmup_safe_vendor_config = copy.deepcopy(vendor_pre_optimizer_config)
        warmup_safe_vendor_config["run_id"] = "v39b-vendor-cull-warmup-0p05"
        warmup_safe_vendor_config["default_strategy"].update(
            {
                "vendor_cull_warmup_profile": "compatibility_uniform_0p05",
                "prune_opa": 0.05,
                "prune_opa_late": 0.05,
            }
        )
        warmup_safe_vendor_config.pop("config_manifest_sha256")
        warmup_safe_vendor_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(warmup_safe_vendor_config)
        ).hexdigest()
        warmup_safe_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), warmup_safe_vendor_config, stage="boundary"
        )
        self.assertEqual(
            warmup_safe_gate["vendor_cull_warmup_profile"],
            "compatibility_uniform_0p05",
        )
        self.assertEqual(
            warmup_safe_gate["adaptive_growth"]["profile"],
            "v39b_vendor_pre_optimizer_cull0p05_ppisp_compat",
        )

        visible_opacity_config = copy.deepcopy(warmup_safe_vendor_config)
        visible_opacity_config["run_id"] = "v40-visible-opacity-sparsity"
        visible_opacity_config["geometry_regularization"][
            "opacity_sparsity_scope"
        ] = "visible_current_view"
        visible_opacity_config.pop("config_manifest_sha256")
        visible_opacity_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(visible_opacity_config)
        ).hexdigest()
        visible_opacity_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), visible_opacity_config, stage="boundary"
        )
        self.assertEqual(
            visible_opacity_gate["opacity_sparsity_profile"],
            "visible_current_view_lidar_compat",
        )
        self.assertEqual(
            visible_opacity_gate["adaptive_growth"]["profile"],
            "v40_vendor_pre_optimizer_visible_opacity_ppisp_compat",
        )

        deferred_reset_config = copy.deepcopy(visible_opacity_config)
        deferred_reset_config["run_id"] = "v41-deferred-opacity-reset"
        deferred_reset_config["default_strategy"].update(
            {
                "vendor_opacity_reset_profile": (
                    "deferred_every3000_compatibility"
                ),
                "reset_every": 3000,
            }
        )
        deferred_reset_config.pop("config_manifest_sha256")
        deferred_reset_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(deferred_reset_config)
        ).hexdigest()
        deferred_reset_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), deferred_reset_config, stage="boundary"
        )
        self.assertEqual(
            deferred_reset_gate["vendor_opacity_reset_profile"],
            "deferred_every3000_compatibility",
        )
        self.assertEqual(
            deferred_reset_gate["adaptive_growth"]["profile"],
            "v41_vendor_pre_optimizer_deferred_reset_ppisp_compat",
        )

        calibrated_gradient_config = copy.deepcopy(deferred_reset_config)
        calibrated_gradient_config["run_id"] = "v42-calibrated-gradient"
        calibrated_gradient_config["default_strategy"]["grow_grad2d"] = (
            0.000075
        )
        calibrated_gradient_config.pop("config_manifest_sha256")
        calibrated_gradient_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(calibrated_gradient_config)
        ).hexdigest()
        calibrated_gradient_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), calibrated_gradient_config, stage="boundary"
        )
        self.assertEqual(
            calibrated_gradient_gate["projected_gradient_profile"],
            "calibrated_plain_7p5e5",
        )

        calibrated_cull_config = copy.deepcopy(deferred_reset_config)
        calibrated_cull_config["run_id"] = "v42-calibrated-cull"
        calibrated_cull_config["default_strategy"].update(
            {
                "vendor_cull_warmup_profile": "calibrated_uniform_0p04",
                "prune_opa": 0.04,
                "prune_opa_late": 0.04,
            }
        )
        calibrated_cull_config.pop("config_manifest_sha256")
        calibrated_cull_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(calibrated_cull_config)
        ).hexdigest()
        calibrated_cull_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), calibrated_cull_config, stage="boundary"
        )
        self.assertEqual(
            calibrated_cull_gate["vendor_cull_warmup_profile"],
            "calibrated_uniform_0p04",
        )

        calibrated_opacity_config = copy.deepcopy(deferred_reset_config)
        calibrated_opacity_config["run_id"] = "v44-calibrated-opacity-sparsity"
        calibrated_opacity_config["geometry_regularization"][
            "opacity_sparsity_weight"
        ] = 0.001
        calibrated_opacity_config.pop("config_manifest_sha256")
        calibrated_opacity_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(calibrated_opacity_config)
        ).hexdigest()
        calibrated_opacity_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), calibrated_opacity_config, stage="boundary"
        )
        self.assertEqual(
            calibrated_opacity_gate["opacity_sparsity_profile"],
            "visible_current_view_calibrated_1e3",
        )
        self.assertEqual(
            calibrated_opacity_gate["adaptive_growth"]["profile"],
            "v44_vendor_pre_optimizer_reduced_opacity_ppisp_compat",
        )

        geometry_cull_only_config = copy.deepcopy(calibrated_opacity_config)
        geometry_cull_only_config["run_id"] = "v45-geometry-cull-only"
        geometry_cull_only_config["default_strategy"].update(
            {
                "vendor_cull_warmup_profile": (
                    "calibrated_geometry_only_0p00"
                ),
                "prune_opa": 0.0,
                "prune_opa_late": 0.0,
            }
        )
        geometry_cull_only_config.pop("config_manifest_sha256")
        geometry_cull_only_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(geometry_cull_only_config)
        ).hexdigest()
        geometry_cull_only_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), geometry_cull_only_config, stage="boundary"
        )
        self.assertEqual(
            geometry_cull_only_gate["vendor_cull_warmup_profile"],
            "calibrated_geometry_only_0p00",
        )
        self.assertEqual(
            geometry_cull_only_gate["adaptive_growth"]["profile"],
            "v45_vendor_pre_optimizer_geometry_cull_only_ppisp_compat",
        )

        screen_detail_config = copy.deepcopy(geometry_cull_only_config)
        screen_detail_config["run_id"] = "v46-screen-detail-split"
        screen_detail_config["default_strategy"].update(
            {
                "detail_split_policy": "lidar_surface_screen_detail",
                "detail_split_scale_m": 0.02,
                "detail_split_screen_radius": 0.0035,
                "revised_opacity": True,
            }
        )
        screen_detail_config.pop("config_manifest_sha256")
        screen_detail_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(screen_detail_config)
        ).hexdigest()
        screen_detail_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), screen_detail_config, stage="boundary"
        )
        self.assertEqual(
            screen_detail_gate["cloudstudio_detail_split_enhancement"][
                "policy"
            ],
            "lidar_surface_screen_detail",
        )
        self.assertEqual(
            screen_detail_gate["adaptive_growth"]["profile"],
            "v46_vendor_pre_optimizer_screen_detail_ppisp_compat",
        )

        lidar_alpha_config = copy.deepcopy(geometry_cull_only_config)
        lidar_alpha_config["run_id"] = "v47-lidar-alpha-surface-floor"
        lidar_alpha_config.update(
            {"lidar_alpha_weight": 0.02, "lidar_alpha_target": 0.95}
        )
        lidar_alpha_config.pop("config_manifest_sha256")
        lidar_alpha_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(lidar_alpha_config)
        ).hexdigest()
        lidar_alpha_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), lidar_alpha_config, stage="boundary"
        )
        self.assertEqual(
            lidar_alpha_gate["lidar_alpha_profile"],
            "signed_lidar_alpha_floor_0p95",
        )
        self.assertEqual(
            lidar_alpha_gate["adaptive_growth"]["profile"],
            "v47_vendor_pre_optimizer_lidar_alpha_ppisp_compat",
        )

        opacity_only_probe = copy.deepcopy(lidar_alpha_config)
        opacity_only_probe["run_id"] = "v47b-opacity-only-probe"
        opacity_only_probe["mcmc_refine_stop_iter"] = 602
        opacity_only_probe["default_strategy"]["refine_scale2d_stop_iter"] = 602
        opacity_only_probe["controlled_stop_after_steps"] = 702
        opacity_only_probe["lidar_range_weight"] = 0.0
        opacity_only_probe["learning_rates"] = {
            "means": 0.0,
            "scales": 0.0,
            "quats": 0.0,
            "opacities": 0.05,
            "colors": 0.0,
        }
        opacity_only_probe["geometry_regularization"][
            "opacity_sparsity_weight"
        ] = 0.0
        opacity_only_probe["ppisp"] = {"enabled": True, "learning_rate": 0.0}
        opacity_only_probe.pop("config_manifest_sha256")
        opacity_only_probe["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(opacity_only_probe)
        ).hexdigest()
        probe_report = {
            "status": "ADAPTIVE_GROWTH_BOUNDARY_PASS",
            "checkpoint_sha256": "8" * 64,
            "source_trainer_config_sha256": "9" * 64,
        }
        probe_report["boundary_report_sha256"] = hashlib.sha256(
            canonical_json_bytes(probe_report)
        ).hexdigest()
        opacity_only_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(),
            opacity_only_probe,
            stage="review",
            boundary_report=probe_report,
        )
        self.assertEqual(
            opacity_only_gate["opacity_sparsity_profile"],
            "disabled_surface_alpha_probe",
        )
        self.assertEqual(
            opacity_only_gate["adaptive_growth"]["profile"],
            "v47b_opacity_only_surface_alpha_probe",
        )
        self.assertEqual(
            opacity_only_gate["adaptive_growth"][
                "resume_allowed_lineage_differences"
            ],
            ["scale_calibration_sha256"],
        )

        shape_probe = copy.deepcopy(opacity_only_probe)
        shape_probe["run_id"] = "v48a-surface-shape-probe"
        shape_probe["controlled_stop_after_steps"] = 702
        shape_probe["lidar_alpha_weight"] = 1.0
        shape_probe["lidar_alpha_dilation_radius_px"] = 3
        shape_probe["learning_rates"] = {
            "means": 0.0,
            "scales": 0.001,
            "quats": 0.0002,
            "opacities": 0.0,
            "colors": 0.0,
        }
        shape_probe["lidar_normal_alignment"].update(
            {
                "enabled": True,
                "weight_align": 0.01,
                "weight_flatten": 0.01,
                "weight_point_to_plane": 0.0,
                "flatten_mode": "tangent_ratio",
                "flatten_ratio_target": 0.15,
            }
        )
        shape_probe.pop("config_manifest_sha256")
        shape_probe["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(shape_probe)
        ).hexdigest()
        shape_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(),
            shape_probe,
            stage="review",
            boundary_report=probe_report,
        )
        self.assertEqual(
            shape_gate["adaptive_growth"]["profile"],
            "v48a_scale_quaternion_surface_shape_probe",
        )
        self.assertEqual(shape_gate["flatten_profile"], "thin_surface_ratio_0p15")

        strong_shape_probe = copy.deepcopy(shape_probe)
        strong_shape_probe["run_id"] = "v48b-strong-surface-shape-probe"
        strong_shape_probe["learning_rates"].update(
            {"scales": 0.003, "quats": 0.0005}
        )
        strong_shape_probe["lidar_normal_alignment"].update(
            {"weight_align": 0.1, "weight_flatten": 0.1}
        )
        strong_shape_probe.pop("config_manifest_sha256")
        strong_shape_probe["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(strong_shape_probe)
        ).hexdigest()
        strong_shape_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(),
            strong_shape_probe,
            stage="review",
            boundary_report=probe_report,
        )
        self.assertEqual(
            strong_shape_gate["adaptive_growth"]["profile"],
            "v48b_strong_scale_quaternion_surface_shape_probe",
        )
        self.assertEqual(
            strong_shape_gate["flatten_profile"],
            "thin_surface_ratio_0p15_strong",
        )

        invalid_vendor_config = copy.deepcopy(vendor_pre_optimizer_config)
        invalid_vendor_config["default_strategy"][
            "opacity_cull_max_fraction"
        ] = 0.02
        invalid_vendor_config.pop("config_manifest_sha256")
        invalid_vendor_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(invalid_vendor_config)
        ).hexdigest()
        with self.assertRaisesRegex(
            ValueError, "forbids CloudStudio cull protections"
        ):
            advance_adaptive_growth_gate(
                _upstream_data_gate(), invalid_vendor_config, stage="boundary"
            )

        absgrad_config = copy.deepcopy(vendor_gradient_config)
        absgrad_config["run_id"] = "v35-absgrad-calibration-boundary"
        absgrad_config["default_strategy"].update(
            {"absgrad": True, "grow_grad2d": 0.0008}
        )
        absgrad_config.pop("config_manifest_sha256")
        absgrad_config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(absgrad_config)
        ).hexdigest()
        absgrad_gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), absgrad_config, stage="boundary"
        )
        self.assertEqual(
            absgrad_gate["projected_gradient_profile"], "absgrad_8e4"
        )

    def test_fixed_topology_evaluation_gate_authorizes_only_signed_arm(self) -> None:
        arm_config = {
            "run_id": "tile1-a0",
            "mipmap_tile_id": 1,
            "topology_policy": {"mode": "strict_fixed"},
            "fixed_topology_schedule": {
                "enabled": True,
                "phase_a_steps": 5,
                "phase_b_steps": 10,
                "audit_steps": [1, 5, 15, 20],
            },
            "max_steps": 20,
            "factor": 1,
            "color_model": "sh",
            "sh_degree": 0,
            "da2_depth_weight": 0.0,
            "lidar_admission": {"enabled": False},
            "tangent_proposal": {"enabled": False},
            "densification_strategy": "default_3dgs",
        }
        plan = {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "tile_id": 1,
            "steps": {"total": 20},
            "arms": [{"arm": "A0", "path": "a0.json"}],
            "training_allowed": False,
            "adaptive_growth_remains_blocked_by": ["gap evidence"],
        }
        plan["evaluation_plan_sha256"] = hashlib.sha256(
            canonical_json_bytes(plan)
        ).hexdigest()
        upstream = _upstream_data_gate()
        readiness = {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_readiness_v1",
            "status": "FIXED_TOPOLOGY_EVALUATION_PREPARED",
            "upstream_gate_manifest_sha256": upstream["gate_manifest_sha256"],
            "evaluation_plan_sha256": plan["evaluation_plan_sha256"],
            "evidence": {
                "directional_pass": True,
                "phase_a_geometry_frozen": True,
                "core_only_merge_contract": True,
            },
            "training_allowed": False,
            "adaptive_growth_allowed": False,
        }
        readiness["readiness_sha256"] = hashlib.sha256(
            canonical_json_bytes(readiness)
        ).hexdigest()
        gate = advance_fixed_topology_evaluation_gate(
            upstream, readiness, plan, {"A0": arm_config}
        )
        self.assertEqual(gate["status"], FIXED_TOPOLOGY_EVALUATION_READY_STATUS)
        self.assertTrue(gate["training_allowed"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            verify_training_gate(
                path,
                dataset_manifest_sha256=DATASET_SHA,
                split_manifest_sha256=SPLIT_SHA,
                face_manifest_sha256=FACE_SHA,
                fixed_topology_arm_fingerprint_sha256=(
                    fixed_topology_evaluation_arm_fingerprint(arm_config)
                ),
            )
            with self.assertRaisesRegex(ValueError, "not an authorized"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                    fixed_topology_arm_fingerprint_sha256="0" * 64,
                )

    def test_frontend_gate_is_valid_but_blocks_training(self) -> None:
        gate = _gate(ready=False)
        self.assertEqual(verify_gate(gate), gate["gate_manifest_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data/implementation gate is only"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                )

    def test_training_gate_requires_all_stages_and_exact_bindings(self) -> None:
        gate = _gate(ready=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            self.assertEqual(
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                ),
                gate["gate_manifest_sha256"],
            )
            with self.assertRaisesRegex(ValueError, "different inputs"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256="4" * 64,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                )

    def test_upstream_data_ready_does_not_authorize_training(self) -> None:
        gate = _upstream_data_gate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not authorize training"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                )

    def test_upstream_data_ready_allows_only_explicit_implementation_smoke(self) -> None:
        gate = _upstream_data_gate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            self.assertEqual(
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                    allow_implementation_smoke=True,
                ),
                gate["gate_manifest_sha256"],
            )

    def test_implementation_smoke_is_limited_to_two_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "run_id": "bounded-smoke",
                "trainer_preset": "custom",
                "dataset_manifest": str(root / "dataset.json"),
                "recording_root": str(root / "recording"),
                "mask_manifest": str(root / "mask.json"),
                "mask_root": str(root / "masks"),
                "split_manifest": str(root / "split.json"),
                "initialization_ply": str(root / "init.ply"),
                "output_dir": str(root / "output"),
                "gsplat_lock": str(root / "gsplat.lock.json"),
                "require_person_masks": False,
                "lidar_range_weight": 0.0,
                "implementation_smoke_only": True,
                "final_evaluation_artifacts": False,
                "factor": 4,
                "max_steps": 3,
                "checkpoint_every": 3,
            }
            with self.assertRaisesRegex(ValueError, "at most 2 steps"):
                TrainerConfig.from_dict(config).validate()

    def test_legacy_training_ready_literal_is_rejected(self) -> None:
        gate = _upstream_data_gate()
        gate["status"] = "TRAINING_READY"
        gate["training_allowed"] = True
        gate = sign_gate(gate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data/implementation gate is only"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                )

    def test_implementation_contract_requires_short_gpu_smoke(self) -> None:
        with self.assertRaisesRegex(ValueError, "short_gpu_smoke_passed=true"):
            advance_training_implementation_gate(
                _upstream_data_gate(),
                _implementation_contract(gpu_smoke=False),
            )

    def test_implementation_contract_cannot_omit_closed_source_blockers(self) -> None:
        contract = _implementation_contract()
        unsigned = copy.deepcopy(contract)
        unsigned.pop("training_implementation_contract_sha256")
        del unsigned["implementation"]["face4_lidar_geometry"]
        contract = sign_training_implementation_contract(unsigned)
        with self.assertRaisesRegex(ValueError, "face4_lidar_geometry"):
            advance_training_implementation_gate(_upstream_data_gate(), contract)

    def test_verified_implementation_contract_advances_training_gate(self) -> None:
        advanced = advance_training_implementation_gate(
            _upstream_data_gate(),
            _implementation_contract(),
        )
        self.assertEqual(advanced["status"], TRAINING_IMPLEMENTATION_READY_STATUS)
        self.assertTrue(advanced["training_allowed"])
        self.assertEqual(
            advanced["bindings"]["training_implementation_contract_sha256"],
            _implementation_contract()["training_implementation_contract_sha256"],
        )

    def test_training_requires_face4_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(_gate(ready=True)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires a signed Face4"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=None,
                )

    def test_renderer_mask_stage_advances_exactly_one_step(self) -> None:
        gate = _gate(ready=False)

        def renderer(split: str, face_sha: str) -> dict:
            return sign_renderer_mask_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_renderer_mask_cache",
                    "split": split,
                    "source_face_manifest_sha256": face_sha,
                    "policy": {
                        "profile": "mipmap_renderer_visibility_compat_v1",
                        "keep_expression": "face_cache_combined_mask != 0",
                        "competitor_reference_expression": "(seg != 255) & (seg != 33)",
                        "label_33_semantics": "UNKNOWN_NOT_INFERRED",
                    },
                    "masks": [
                        {
                            "image_id": f"{split}-image",
                            "face_id": "front",
                        }
                    ],
                    "summary": {
                        "face_sample_count": 1,
                        "empty_mask_count": 0,
                        "missing_mask_count": 0,
                    },
                }
            )

        train = renderer("train", FACE_SHA)
        val = renderer("val", FACE_SHA)
        advanced = advance_renderer_mask_gate(gate, train, val)
        self.assertEqual(advanced["status"], "RENDERER_MASK_READY")
        self.assertFalse(advanced["training_allowed"])
        self.assertEqual(advanced["next_required_stage"], "new_at_lidar_depth")
        self.assertEqual(len(advanced["completed_stages"]), 11)
        verify_gate(advanced)

    def test_lidar_depth_stage_requires_complete_bound_manifest(self) -> None:
        gate = _gate(ready=False)

        def renderer(split: str) -> dict:
            return sign_renderer_mask_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_renderer_mask_cache",
                    "split": split,
                    "source_face_manifest_sha256": FACE_SHA,
                    "policy": {
                        "profile": "mipmap_renderer_visibility_compat_v1",
                        "keep_expression": "face_cache_combined_mask != 0",
                        "competitor_reference_expression": "(seg != 255) & (seg != 33)",
                        "label_33_semantics": "UNKNOWN_NOT_INFERRED",
                    },
                    "masks": [{"image_id": split, "face_id": "front"}],
                    "summary": {
                        "face_sample_count": 1,
                        "empty_mask_count": 0,
                        "missing_mask_count": 0,
                    },
                }
            )

        renderer_gate = advance_renderer_mask_gate(
            gate, renderer("train"), renderer("val")
        )
        depth = {
            "schema_version": 1,
            "dataset_manifest_sha256": DATASET_SHA,
            "mask_manifest_sha256": MASK_SHA,
            "point_cloud_sha256": "5" * 64,
            "point_cloud_points": 100,
            "complete_dataset": True,
            "total_dataset_images": 1,
            "images": [{"image_id": "image", "path": "depth/image.npz"}],
            "summary": {"image_count": 1},
        }
        import hashlib

        from cloudstudio_3dgs.data.manifest import canonical_json_bytes

        depth["depth_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(depth)
        ).hexdigest()
        advanced = advance_lidar_depth_gate(renderer_gate, depth)
        self.assertEqual(advanced["status"], LIDAR_DEPTH_READY_STATUS)
        self.assertEqual(advanced["next_required_stage"], "da2_monocular_depth")
        self.assertEqual(len(advanced["completed_stages"]), 12)
        verify_gate(advanced)
        broken = dict(depth)
        broken["complete_dataset"] = False
        broken.pop("depth_manifest_sha256")
        broken["depth_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(broken)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            advance_lidar_depth_gate(renderer_gate, broken)

    def test_da2_stage_requires_complete_train_and_val(self) -> None:
        gate = _gate(ready=False)

        def renderer(split: str) -> dict:
            return sign_renderer_mask_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_renderer_mask_cache",
                    "split": split,
                    "source_face_manifest_sha256": FACE_SHA,
                    "policy": {
                        "profile": "mipmap_renderer_visibility_compat_v1",
                        "keep_expression": "face_cache_combined_mask != 0",
                        "competitor_reference_expression": "(seg != 255) & (seg != 33)",
                        "label_33_semantics": "UNKNOWN_NOT_INFERRED",
                    },
                    "masks": [{"image_id": split, "face_id": "front"}],
                    "summary": {
                        "face_sample_count": 1,
                        "empty_mask_count": 0,
                        "missing_mask_count": 0,
                    },
                }
            )

        renderer_gate = advance_renderer_mask_gate(
            gate, renderer("train"), renderer("val")
        )
        import hashlib

        from cloudstudio_3dgs.data.manifest import canonical_json_bytes

        depth = {
            "schema_version": 1,
            "dataset_manifest_sha256": DATASET_SHA,
            "mask_manifest_sha256": MASK_SHA,
            "point_cloud_sha256": "5" * 64,
            "point_cloud_points": 100,
            "complete_dataset": True,
            "total_dataset_images": 1,
            "images": [{"image_id": "image", "path": "depth/image.npz"}],
            "summary": {"image_count": 1},
        }
        depth["depth_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(depth)
        ).hexdigest()
        lidar_gate = advance_lidar_depth_gate(renderer_gate, depth)

        def mono(split: str) -> dict:
            return sign_mono_depth_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_da2_relative_depth_cache",
                    "split": split,
                    "source_face_manifest_sha256": FACE_SHA,
                    "dataset_manifest_sha256": DATASET_SHA,
                    "lidar_depth_manifest_sha256": depth["depth_manifest_sha256"],
                    "model": {"checkpoint_sha256": "6" * 64},
                    "metric_alignment": {"iterations": 2000},
                    "complete_face_cache": True,
                    "expected_face_count": 1,
                    "records": [{"sample_id": split}],
                    "summary": {"face_count": 1},
                }
            )

        advanced = advance_da2_depth_gate(lidar_gate, mono("train"), mono("val"))
        self.assertEqual(advanced["status"], DA2_DEPTH_READY_STATUS)
        self.assertEqual(advanced["next_required_stage"], "independent_sky_background")
        self.assertEqual(len(advanced["completed_stages"]), 13)
        verify_gate(advanced)

    def test_reordered_or_tampered_gate_is_rejected(self) -> None:
        gate = _gate(ready=False)
        gate["completed_stages"][1:3] = reversed(gate["completed_stages"][1:3])
        gate = sign_gate(gate)
        with self.assertRaisesRegex(ValueError, "missing, reordered, or skipped"):
            verify_gate(gate)
        gate = _gate(ready=False)
        gate["status"] = "TAMPERED"
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            verify_gate(gate)

    def test_trainer_rejects_shared_kb4_dataset_before_ready_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            split = root / "split.json"
            face = root / "face.json"
            gate_path = root / "gate.json"
            dataset.write_text(
                json.dumps(
                    {
                        "manifest_sha256": DATASET_SHA,
                        "training_lineage": {
                            "independent_at_algorithm_version": (
                                "independent_pos_prior_shared_single_focal_kb4_at_v2"
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            split.write_text(
                json.dumps({"split_manifest_sha256": SPLIT_SHA}), encoding="utf-8"
            )
            face.write_text(
                json.dumps({"face_manifest_sha256": FACE_SHA}), encoding="utf-8"
            )
            gate_path.write_text(json.dumps(_gate(ready=False)), encoding="utf-8")
            base = {
                "run_id": "shared-kb4-gate",
                "dataset_manifest": str(dataset),
                "recording_root": str(root / "recording"),
                "mask_manifest": str(root / "mask.json"),
                "mask_root": str(root / "masks"),
                "split_manifest": str(split),
                "initialization_ply": str(root / "initialization.ply"),
                "output_dir": str(root / "output"),
                "gsplat_lock": str(root / "gsplat.lock.json"),
                "require_person_masks": False,
                "lidar_range_weight": 0.0,
            }
            with self.assertRaisesRegex(ValueError, "requires mipmap_pipeline_gate"):
                TrainerConfig.from_dict(base).validate()
            with self.assertRaisesRegex(ValueError, "data/implementation gate is only"):
                TrainerConfig.from_dict(
                    {
                        **base,
                        "face_cache_manifest": str(face),
                        "face_cache_root": str(root / "faces"),
                        "mipmap_pipeline_gate": str(gate_path),
                    }
                ).validate()


if __name__ == "__main__":
    unittest.main()
