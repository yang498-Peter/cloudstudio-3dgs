"""CloudStudio-owned raw-fisheye gsplat trainer with no Viewer dependency."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import sparse_depth_npz_bytes
from cloudstudio_3dgs.data.image_sample import CropWindow
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mono_depth import verify_mono_depth_manifest
from cloudstudio_3dgs.data.mesh_geometry import verify_mesh_geometry_manifest
from cloudstudio_3dgs.data.face_lidar_geometry import (
    verify_face_lidar_geometry_manifest,
)
from cloudstudio_3dgs.data.renderer_masks import verify_renderer_mask_manifest
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap
from cloudstudio_3dgs.evaluation.quality_report import sign_run_manifest
from cloudstudio_3dgs.training.backend import (
    GsplatBackend,
    rendered_range_to_euclidean,
)
from cloudstudio_3dgs.training.checkpoint import (
    load_checkpoint,
    retain_checkpoint,
    save_checkpoint,
)
from cloudstudio_3dgs.training.contracts import build_coordinate_transform_manifest
from cloudstudio_3dgs.training.dataset import S1TrainingDataset, TrainingSample
from cloudstudio_3dgs.training.face_dataset import verify_face_manifest
from cloudstudio_3dgs.training.losses import (
    confidence_weighted_log_range_huber,
    confidence_weighted_range_l1,
    global_masked_rgb_ssim_loss,
    masked_rgb_gradient_l1,
    masked_rgb_l1,
    masked_rgb_ssim_loss,
)
from cloudstudio_3dgs.training.mipmap_loss_schedule import (
    competitor_high_type2_loss_weights,
)
from cloudstudio_3dgs.training.presets import (
    assert_trainer_preset_matches,
    expand_trainer_preset,
)
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    INDEPENDENT_AT_ALGORITHM,
    fixed_topology_evaluation_arm_fingerprint,
    load_and_verify_gate,
    verify_training_gate,
)
from cloudstudio_3dgs.training.exposure import (
    ExposureCompensationConfig,
    ExposureCompensator,
)
from cloudstudio_3dgs.training.golden_eval import (
    GoldenEvaluationConfig,
    evaluate_full_validation,
    evaluate_golden_views,
    is_golden_improvement,
)
from cloudstudio_3dgs.training.scale_calibration import (
    MetricScaleCalibrationConfig,
    build_metric_scale_calibration,
)
from cloudstudio_3dgs.training.surface_initialization import (
    SurfaceInitializationConfig,
    cap_surface_initialization_scales,
    build_surface_aligned_initialization,
    load_initialization_geometry,
)
from cloudstudio_3dgs.training.mipmap_tile_geometry import (
    load_mipmap_tile_geometry,
    verify_tile_geometry_manifest,
)
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest
from cloudstudio_3dgs.training.rig_pose import (
    RigPoseRefinementConfig,
    RigPoseRefiner,
    build_pose_refinement_report,
    disabled_pose_refinement_report,
)
from cloudstudio_3dgs.training.default_strategy_adapter import DENSIFICATION_STRATEGIES
from cloudstudio_3dgs.training.error_weighted_mcmc import ErrorScoreConfig
from cloudstudio_3dgs.training.contribution_attribution import (
    ContributionConfig,
    compute_contribution_scores,
)
from cloudstudio_3dgs.training.ppisp import PpispConfig, PpispCorrector
from cloudstudio_3dgs.training.lidar_normals import (
    LidarNormalAnchors,
    NormalAlignmentConfig,
    build_normal_field,
)
from cloudstudio_3dgs.training.lidar_admission import (
    AdmissionConfig,
    LidarAdmission,
    normal_field_from_surface_field,
    update_admission_telemetry,
)
from cloudstudio_3dgs.training.tangent_proposal import (
    ProposalConfig,
    TangentProposal,
    update_proposal_telemetry,
)
from cloudstudio_3dgs.training.regularization import (
    GeometryRegularizationConfig,
    clip_oversized_gaussians,
    geometry_regularization_terms,
)
from cloudstudio_3dgs.training.runtime_evidence import (
    append_mcmc_telemetry,
    build_mcmc_step_event,
    initialize_mcmc_telemetry,
    normalize_classic_strategy_telemetry,
    require_finite_training_tensors,
    snapshot_gaussians,
)
from cloudstudio_3dgs.training.topology_policy import (
    FixedTopologyScheduleConfig,
    TopologyPolicyConfig,
    topology_count_transition,
)
from cloudstudio_3dgs.training.optimization_audit import (
    component_gradient_audit,
    gradient_norms,
    parameter_update_norms,
    point_to_plane_drift_summary,
    shortest_axis_normals,
)


class ControlledTrainingInterruption(RuntimeError):
    """Acceptance-only stop raised after an atomic resumable checkpoint."""

    def __init__(self, *, completed_steps: int, checkpoint_path: Path) -> None:
        self.completed_steps = int(completed_steps)
        self.checkpoint_path = Path(checkpoint_path)
        super().__init__(
            f"controlled interruption after {self.completed_steps} steps: "
            f"{self.checkpoint_path}"
        )


def _current_view_visibility_mask(radii: Any, gaussian_count: int) -> Any:
    """Return one visibility bit per Gaussian from the current raster pass."""

    if radii is None:
        raise ValueError("current-view opacity sparsity requires raster radii")
    if gaussian_count <= 0:
        raise ValueError("gaussian_count must be positive")
    torch = __import__("torch")
    value = torch.as_tensor(radii).detach()
    if value.ndim >= 2 and value.shape[-1] == 2:
        value = value.max(dim=-1).values
    if value.numel() == gaussian_count:
        return value.reshape(gaussian_count) > 0
    if value.ndim >= 1 and value.shape[-1] == gaussian_count:
        return value.reshape(-1, gaussian_count).amax(dim=0) > 0
    raise ValueError(
        "raster radii cannot be mapped to one current-view visibility bit "
        f"per Gaussian: shape={tuple(value.shape)}, count={gaussian_count}"
    )


@dataclass(frozen=True)
class TrainerConfig:
    run_id: str
    dataset_manifest: Path
    recording_root: Path
    mask_manifest: Path
    mask_root: Path
    split_manifest: Path
    initialization_ply: Path
    output_dir: Path
    gsplat_lock: Path
    initialization_geometry: Path | None = None
    initialization_geometry_manifest: Path | None = None
    mipmap_tile_id: int | None = None
    person_mask_manifest: Path | None = None
    person_mask_root: Path | None = None
    depth_manifest: Path | None = None
    depth_root: Path | None = None
    resume_checkpoint: Path | None = None
    warm_start_checkpoint: Path | None = None
    warm_start_min_opacity: float = 0.0
    warm_start_scale_multiplier: float = 1.0
    warm_start_fresh_auxiliary: tuple[str, ...] = ()
    face_cache_manifest: Path | None = None
    face_cache_root: Path | None = None
    raw_fisheye_post_refine_face_manifest: Path | None = None
    renderer_mask_manifest: Path | None = None
    tile_inputs_manifest: Path | None = None
    tile_inputs_root: Path | None = None
    mono_depth_manifest: Path | None = None
    mono_depth_root: Path | None = None
    face_lidar_geometry_manifest: Path | None = None
    face_lidar_geometry_root: Path | None = None
    mesh_geometry_manifest: Path | None = None
    mesh_geometry_root: Path | None = None
    mipmap_pipeline_gate: Path | None = None
    config_manifest_sha256: str | None = None
    controlled_stop_after_steps: int | None = None
    implementation_smoke_only: bool = False
    topology_policy: TopologyPolicyConfig = field(default_factory=TopologyPolicyConfig)
    fixed_topology_schedule: FixedTopologyScheduleConfig = field(
        default_factory=FixedTopologyScheduleConfig
    )
    final_evaluation_artifacts: bool = True
    require_person_masks: bool = True
    trainer_preset: str = "custom"
    device: str = "cuda:0"
    seed: int = 42
    view_sampling_mode: str = "with_replacement"
    max_steps: int = 3_000
    checkpoint_every: int = 500
    checkpoint_keep_every: int = 0
    cuda_empty_cache_interval_steps: int = 0
    factor: int = 4
    crop: CropWindow | None = None
    cap_max: int = 1_000_000
    init_scale_m: float = 0.05
    metric_scale_calibration: MetricScaleCalibrationConfig = field(
        default_factory=MetricScaleCalibrationConfig
    )
    surface_initialization: SurfaceInitializationConfig = field(
        default_factory=SurfaceInitializationConfig
    )
    rgb_l1_weight: float = 0.8
    rgb_ssim_weight: float = 0.2
    rgb_gradient_weight: float = 0.0
    lidar_rgb_l1_weight: float = 0.0
    lidar_rgb_dilation_radius_px: int = 0
    rgb_ssim_mode: str = "local_gaussian"
    lidar_range_weight: float = 0.05
    lidar_alpha_weight: float = 0.0
    lidar_alpha_target: float = 0.95
    lidar_alpha_dilation_radius_px: int = 0
    da2_depth_weight: float = 0.0
    mesh_depth_weight: float = 0.0
    mesh_normal_weight: float = 0.0
    competitor_loss_schedule_enabled: bool = False
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    ssim_min_valid_fraction: float = 0.8
    lidar_range_loss_mode: str = "robust_log_huber"
    lidar_log_range_huber_delta: float = 0.05
    lidar_linear_aux_weight: float = 0.0
    decoupled_ssim: bool = False
    sh_regularization_weight: float = 0.0
    pinhole_rasterize_mode: str = "classic"
    pinhole_with_ut: bool = False
    color_model: str = "rgb_sigmoid"
    sh_degree: int = 2
    background_color: tuple[float, float, float] | None = None
    sh_degree_interval: int = 1000
    means_lr_final_factor: float = 1.0
    mcmc_refine_start_iter: int = 500
    mcmc_refine_stop_iter: int = 25_000
    mcmc_refine_every: int = 100
    mcmc_noise_injection_stop_iter: int = -1
    mcmc_noise_lr: float = 500_000.0
    mcmc_growth_rate: float = 0.05
    mcmc_relocation_max_fraction: float = 1.0
    mcmc_relocation_max_scale_m: float | None = None
    mcmc_relocation_max_anisotropy: float | None = None
    post_refine_geometry_lr_scale: float = 1.0
    # "error_weighted_mcmc" is this trainer's homegrown sampler; "default_3dgs"
    # is gsplat's reference implementation of Kerbl et al., which scores by the
    # projected-position gradient instead of an image-error map. Measurement on
    # 2026-08-25 traced the blur to where the error map places births.
    densification_strategy: str = "error_weighted_mcmc"
    default_strategy: dict[str, Any] = field(default_factory=dict)
    # Lifecycle-vs-optimizer ordering now lives in the strategy itself: pass
    # default_strategy={"lifecycle_execution_order": "pre_optimizer_vendor"}.
    # (Machine B's short-lived trainer-level lifecycle_order flag was retired
    # in favour of the snow machine's strategy-integrated mechanism.)
    # Initial gaussian opacity. The historical 0.1 sits exactly on the exact
    # MipMap lifecycle's prune gate; a coherent whole-machine transplant
    # starts at that machine's reset cap (0.2) instead.
    init_opacity: float = 0.1
    # Optional pre-baked far-shell sky layer (checkpoint-convention params)
    # concatenated into the trainable set at initialization, so sky pixels
    # have an owner and the white background stops pressing scene gaussians
    # toward transparency.
    sky_shell_checkpoint: Path | None = None
    # Parameter rows from this index on render but never optimize: their
    # gradients are zeroed every step. A baked far-field layer is a static
    # backdrop, not a training variable - left trainable, any pixel whose ray
    # reaches it (through under-opaque foreground) feeds it a constant
    # directional gradient that Adam normalizes into full-size steps, and the
    # layer drains to transparency within a few thousand steps. Freezing it
    # redirects that same gradient onto the foreground, which must become
    # opaque enough to occlude the backdrop - the desired direction.
    frozen_tail_start: int | None = None
    # Per-view prerendered backdrop compositing (see view_backgrounds.py).
    # When set, every sampled view composites onto its own stored image
    # instead of the constant background_color, and the far-field layer
    # stays out of the trainable set entirely.
    background_image_manifest: Path | None = None
    background_image_root: Path | None = None
    # What loss the densification criterion differentiates. "rgb_only" scores
    # births from L1+SSIM alone, the way Kerbl et al. trained; under "total_loss"
    # the LiDAR range and normal terms leak into means2d.grad through the shared
    # rasterization, and the criterion is no longer the published one.
    densification_gradient_source: str = "total_loss"
    rig_pose_refinement: RigPoseRefinementConfig = field(
        default_factory=RigPoseRefinementConfig
    )
    exposure_compensation: ExposureCompensationConfig = field(
        default_factory=ExposureCompensationConfig
    )
    geometry_regularization: GeometryRegularizationConfig = field(
        default_factory=GeometryRegularizationConfig
    )
    golden_evaluation: GoldenEvaluationConfig = field(
        default_factory=GoldenEvaluationConfig
    )
    error_weighted_sampling: ErrorScoreConfig = field(default_factory=ErrorScoreConfig)
    ppisp: PpispConfig = field(default_factory=PpispConfig)
    contribution: ContributionConfig = field(default_factory=ContributionConfig)
    contribution_every: int = 5
    lidar_normal_alignment: NormalAlignmentConfig = field(
        default_factory=NormalAlignmentConfig
    )
    lidar_admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    tangent_proposal: ProposalConfig = field(default_factory=ProposalConfig)
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {
            "means": 1.6e-4,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "colors": 2.5e-3,
        }
    )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainerConfig":
        value = expand_trainer_preset(value)
        paths = {
            key: None if value.get(key) is None else Path(value[key])
            for key in (
                "dataset_manifest",
                "recording_root",
                "mask_manifest",
                "mask_root",
                "split_manifest",
                "initialization_ply",
                "initialization_geometry",
                "initialization_geometry_manifest",
                "output_dir",
                "gsplat_lock",
                "person_mask_manifest",
                "person_mask_root",
                "depth_manifest",
                "depth_root",
                "resume_checkpoint",
                "warm_start_checkpoint",
                "face_cache_manifest",
                "face_cache_root",
                "raw_fisheye_post_refine_face_manifest",
                "renderer_mask_manifest",
                "tile_inputs_manifest",
                "tile_inputs_root",
                "mono_depth_manifest",
                "mono_depth_root",
                "face_lidar_geometry_manifest",
                "face_lidar_geometry_root",
                "mesh_geometry_manifest",
                "mesh_geometry_root",
                "mipmap_pipeline_gate",
                "sky_shell_checkpoint",
                "background_image_manifest",
                "background_image_root",
            )
        }
        crop_value = value.get("crop")
        crop = None if crop_value is None else CropWindow(**crop_value)
        pose_value = value.get("rig_pose_refinement", {})
        if not isinstance(pose_value, dict):
            raise ValueError("rig_pose_refinement must be an object")
        pose_refinement = RigPoseRefinementConfig(**pose_value)
        exposure_value = value.get("exposure_compensation", {})
        if not isinstance(exposure_value, dict):
            raise ValueError("exposure_compensation must be an object")
        exposure = ExposureCompensationConfig(**exposure_value)
        regularization_value = value.get("geometry_regularization", {})
        if not isinstance(regularization_value, dict):
            raise ValueError("geometry_regularization must be an object")
        regularization = GeometryRegularizationConfig(**regularization_value)
        golden_evaluation_value = value.get("golden_evaluation", {})
        if not isinstance(golden_evaluation_value, dict):
            raise ValueError("golden_evaluation must be an object")
        golden_evaluation = GoldenEvaluationConfig(**golden_evaluation_value)
        error_sampling_value = value.get("error_weighted_sampling", {})
        if not isinstance(error_sampling_value, dict):
            raise ValueError("error_weighted_sampling must be an object")
        error_weighted_sampling = ErrorScoreConfig(**error_sampling_value)
        ppisp_value = value.get("ppisp", {})
        if not isinstance(ppisp_value, dict):
            raise ValueError("ppisp must be an object")
        ppisp = PpispConfig(**ppisp_value)
        contribution_value = value.get("contribution", {})
        if not isinstance(contribution_value, dict):
            raise ValueError("contribution must be an object")
        contribution_config = ContributionConfig(**contribution_value)
        normal_value = value.get("lidar_normal_alignment", {})
        if not isinstance(normal_value, dict):
            raise ValueError("lidar_normal_alignment must be an object")
        lidar_normal_alignment = NormalAlignmentConfig(**normal_value)
        admission_value = value.get("lidar_admission", {})
        if not isinstance(admission_value, dict):
            raise ValueError("lidar_admission must be an object")
        lidar_admission = AdmissionConfig(**admission_value)
        proposal_value = value.get("tangent_proposal", {})
        if not isinstance(proposal_value, dict):
            raise ValueError("tangent_proposal must be an object")
        tangent_proposal = ProposalConfig(**proposal_value)
        scale_value = value.get("metric_scale_calibration", {})
        if not isinstance(scale_value, dict):
            raise ValueError("metric_scale_calibration must be an object")
        scale_calibration = MetricScaleCalibrationConfig(**scale_value)
        surface_initialization_value = value.get("surface_initialization", {})
        if not isinstance(surface_initialization_value, dict):
            raise ValueError("surface_initialization must be an object")
        surface_initialization = SurfaceInitializationConfig(
            **surface_initialization_value
        )
        topology_value = value.get("topology_policy", {})
        if not isinstance(topology_value, dict):
            raise ValueError("topology_policy must be an object")
        topology_policy = TopologyPolicyConfig(**topology_value)
        fixed_schedule_value = value.get("fixed_topology_schedule", {})
        if not isinstance(fixed_schedule_value, dict):
            raise ValueError("fixed_topology_schedule must be an object")
        audit_steps_value = fixed_schedule_value.get("audit_steps", [])
        if not isinstance(audit_steps_value, list):
            raise ValueError("fixed_topology_schedule.audit_steps must be a list")
        fixed_topology_schedule = FixedTopologyScheduleConfig(
            **{
                **fixed_schedule_value,
                "audit_steps": tuple(audit_steps_value),
            }
        )
        fresh_auxiliary_value = value.get("warm_start_fresh_auxiliary", [])
        if not isinstance(fresh_auxiliary_value, list) or not all(
            isinstance(name, str) and name for name in fresh_auxiliary_value
        ):
            raise ValueError("warm_start_fresh_auxiliary must be a list of non-empty strings")
        options = {
            key: value[key]
            for key in (
                "device",
                "require_person_masks",
                "seed",
                "view_sampling_mode",
                "max_steps",
                "checkpoint_every",
                "checkpoint_keep_every",
                "cuda_empty_cache_interval_steps",
                "factor",
                "cap_max",
                "init_scale_m",
                "rgb_l1_weight",
                "rgb_ssim_weight",
                "rgb_gradient_weight",
                "lidar_rgb_l1_weight",
                "lidar_rgb_dilation_radius_px",
                "rgb_ssim_mode",
                "lidar_range_weight",
                "lidar_alpha_weight",
                "lidar_alpha_target",
                "lidar_alpha_dilation_radius_px",
                "da2_depth_weight",
                "mesh_depth_weight",
                "mesh_normal_weight",
                "competitor_loss_schedule_enabled",
                "ssim_window_size",
                "ssim_sigma",
                "ssim_min_valid_fraction",
                "lidar_range_loss_mode",
                "lidar_log_range_huber_delta",
                "lidar_linear_aux_weight",
                "decoupled_ssim",
                "sh_regularization_weight",
                "pinhole_rasterize_mode",
                "pinhole_with_ut",
                "contribution_every",
                "color_model",
                "sh_degree",
                "background_color",
                "sh_degree_interval",
                "means_lr_final_factor",
                "mcmc_refine_start_iter",
                "mcmc_refine_stop_iter",
                "mcmc_refine_every",
                "mcmc_noise_injection_stop_iter",
                "mcmc_noise_lr",
                "mcmc_growth_rate",
                "mcmc_relocation_max_fraction",
                "mcmc_relocation_max_scale_m",
                "mcmc_relocation_max_anisotropy",
                "post_refine_geometry_lr_scale",
                "densification_strategy",
                "default_strategy",
                "densification_gradient_source",
                "init_opacity",
                "learning_rates",
                "warm_start_min_opacity",
                "warm_start_scale_multiplier",
                "frozen_tail_start",
                "final_evaluation_artifacts",
                "mipmap_tile_id",
                "implementation_smoke_only",
                "config_manifest_sha256",
                "controlled_stop_after_steps",
            )
            if key in value
        }
        return cls(
            run_id=str(value["run_id"]),
            trainer_preset=str(value["trainer_preset"]),
            crop=crop,
            rig_pose_refinement=pose_refinement,
            warm_start_fresh_auxiliary=tuple(fresh_auxiliary_value),
            metric_scale_calibration=scale_calibration,
            surface_initialization=surface_initialization,
            topology_policy=topology_policy,
            fixed_topology_schedule=fixed_topology_schedule,
            exposure_compensation=exposure,
            geometry_regularization=regularization,
            golden_evaluation=golden_evaluation,
            error_weighted_sampling=error_weighted_sampling,
            ppisp=ppisp,
            contribution=contribution_config,
            lidar_normal_alignment=lidar_normal_alignment,
            lidar_admission=lidar_admission,
            tangent_proposal=tangent_proposal,
            **paths,
            **options,
        )

    def validate(self) -> None:
        if not self.run_id or any(character in self.run_id for character in "\\/\0"):
            raise ValueError("run_id must be a non-empty path-safe name")
        if not self.device.startswith("cuda"):
            raise ValueError("3DGUT training requires an explicit CUDA device")
        if self.max_steps <= 0 or self.checkpoint_every <= 0:
            raise ValueError("max_steps and checkpoint_every must be positive")
        if self.controlled_stop_after_steps is not None and not (
            0 < int(self.controlled_stop_after_steps) < self.max_steps
        ):
            raise ValueError(
                "controlled_stop_after_steps must be between zero and max_steps"
            )
        self.topology_policy.validate(max_steps=self.max_steps)
        self.fixed_topology_schedule.validate(
            max_steps=self.max_steps, topology_mode=self.topology_policy.mode
        )
        if not self.topology_policy.strategy_enabled and (
            self.error_weighted_sampling.enabled
            or self.lidar_admission.enabled
            or self.tangent_proposal.enabled
        ):
            raise ValueError(
                "no-birth topology forbids densification sampling, admission, and proposal"
            )
        if (
            self.topology_policy.mode == "opacity_prune_only"
            and self.fixed_topology_schedule.enabled
            and int(self.topology_policy.opacity_prune_step or 0)
            < self.fixed_topology_schedule.phase_a_steps
        ):
            raise ValueError("opacity prune must occur after the appearance warm-up")
        if self.implementation_smoke_only:
            if self.max_steps > 2:
                raise ValueError("implementation_smoke_only is limited to at most 2 steps")
            if self.factor < 2 and not (
                self.factor == 1
                and self.max_steps == 1
                and self.topology_policy.mode == "strict_fixed"
            ):
                raise ValueError(
                    "factor=1 implementation smoke is limited to one strict-fixed step"
                )
            if self.final_evaluation_artifacts or self.golden_evaluation.enabled:
                raise ValueError(
                    "implementation_smoke_only forbids final or golden evaluation"
                )
            if self.resume_checkpoint is not None or (
                self.warm_start_checkpoint is not None
                and self.raw_fisheye_post_refine_face_manifest is None
            ):
                raise ValueError(
                    "implementation_smoke_only forbids resume and permits warm start "
                    "only for signed raw-fisheye post-refine lineage"
                )
            if self.checkpoint_every != self.max_steps:
                raise ValueError(
                    "implementation_smoke_only must checkpoint only at the final step"
                )
            if (
                self.topology_policy.strategy_enabled
                and int(self.default_strategy.get("refine_start_iter", 0))
                <= self.max_steps
            ):
                raise ValueError(
                    "implementation_smoke_only forbids densification during the smoke"
                )
        if self.view_sampling_mode not in {
            "with_replacement",
            "fisher_yates_without_replacement_per_epoch",
        }:
            raise ValueError("unsupported training view sampling mode")
        if self.cuda_empty_cache_interval_steps < 0:
            raise ValueError("cuda_empty_cache_interval_steps must be non-negative")
        if self.resume_checkpoint is not None and self.warm_start_checkpoint is not None:
            raise ValueError("resume_checkpoint and warm_start_checkpoint are mutually exclusive")
        if len(set(self.warm_start_fresh_auxiliary)) != len(
            self.warm_start_fresh_auxiliary
        ):
            raise ValueError("warm_start_fresh_auxiliary names must be unique")
        unsupported_fresh_auxiliary = set(self.warm_start_fresh_auxiliary) - {
            "rig_pose_deltas",
            "exposure_log_gains",
        }
        if unsupported_fresh_auxiliary:
            raise ValueError(
                "unsupported warm_start_fresh_auxiliary names: "
                + ", ".join(sorted(unsupported_fresh_auxiliary))
            )
        if self.warm_start_fresh_auxiliary:
            if self.warm_start_checkpoint is None:
                raise ValueError(
                    "warm_start_fresh_auxiliary requires warm_start_checkpoint"
                )
            if (
                "rig_pose_deltas" in self.warm_start_fresh_auxiliary
                and not self.rig_pose_refinement.enabled
            ):
                raise ValueError(
                    "fresh rig_pose_deltas requires rig_pose_refinement.enabled"
                )
            if (
                "exposure_log_gains" in self.warm_start_fresh_auxiliary
                and not self.exposure_compensation.enabled
            ):
                raise ValueError(
                    "fresh exposure_log_gains requires exposure compensation"
                )
        if (self.background_image_manifest is None) != (
            self.background_image_root is None
        ):
            raise ValueError(
                "background_image_manifest and background_image_root are a pair"
            )
        if self.background_image_manifest is not None:
            if not Path(self.background_image_manifest).is_file():
                raise ValueError(
                    "background image manifest is missing: "
                    f"{self.background_image_manifest}"
                )
            if not Path(self.background_image_root).is_dir():
                raise ValueError(
                    f"background image root is missing: {self.background_image_root}"
                )
        if self.frozen_tail_start is not None:
            if (
                not isinstance(self.frozen_tail_start, int)
                or isinstance(self.frozen_tail_start, bool)
                or self.frozen_tail_start < 0
            ):
                raise ValueError("frozen_tail_start must be a non-negative integer")
            if self.topology_policy.mode != "strict_fixed":
                raise ValueError(
                    "frozen_tail_start requires strict_fixed topology: any "
                    "relocation or pruning would scramble the frozen rows"
                )
        if not 0.0 <= float(self.warm_start_min_opacity) < 1.0:
            raise ValueError("warm_start_min_opacity must be within [0, 1)")
        if (
            not math.isfinite(float(self.warm_start_scale_multiplier))
            or self.warm_start_scale_multiplier <= 0.0
        ):
            raise ValueError("warm_start_scale_multiplier must be finite and positive")
        if self.warm_start_scale_multiplier != 1.0 and self.warm_start_checkpoint is None:
            raise ValueError("warm_start_scale_multiplier requires warm_start_checkpoint")
        if self.warm_start_min_opacity > 0.0:
            if self.warm_start_checkpoint is None:
                raise ValueError(
                    "warm_start_min_opacity requires warm_start_checkpoint"
                )
            if (
                self.lidar_normal_alignment.enabled
                or self.lidar_admission.enabled
                or self.tangent_proposal.enabled
            ):
                raise ValueError(
                    "warm-start opacity pruning currently requires normal alignment, "
                    "LiDAR admission, and tangent proposal to be disabled"
                )
        if self.checkpoint_keep_every < 0:
            raise ValueError("checkpoint_keep_every must be non-negative")
        if self.cap_max <= 4:
            raise ValueError("cap_max must be greater than four")
        if self.init_scale_m <= 0.0:
            raise ValueError("init_scale_m must be positive")
        self.metric_scale_calibration.validate()
        self.surface_initialization.validate()
        if self.color_model not in ("rgb_sigmoid", "sh"):
            raise ValueError("color_model must be 'rgb_sigmoid' or 'sh'")
        if not 0 <= self.sh_degree <= 3:
            raise ValueError("sh_degree must be within [0, 3]")
        if self.background_color is not None:
            values = tuple(float(v) for v in self.background_color)
            if len(values) != 3 or any(not 0.0 <= v <= 1.0 for v in values):
                raise ValueError("background_color must be three values in [0, 1]")
        if self.sh_degree_interval < 0:
            raise ValueError("sh_degree_interval must be non-negative")
        if not 0.0 < self.means_lr_final_factor <= 1.0:
            raise ValueError("means_lr_final_factor must be in (0, 1]")
        weights = (
            self.rgb_l1_weight,
            self.rgb_ssim_weight,
            self.rgb_gradient_weight,
            self.lidar_rgb_l1_weight,
            self.lidar_range_weight,
            self.lidar_linear_aux_weight,
            self.lidar_alpha_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if self.rgb_l1_weight + self.rgb_ssim_weight <= 0.0:
            raise ValueError("at least one RGB loss weight must be positive")
        if (
            not isinstance(self.lidar_rgb_dilation_radius_px, int)
            or isinstance(self.lidar_rgb_dilation_radius_px, bool)
            or not 0 <= self.lidar_rgb_dilation_radius_px <= 64
        ):
            raise ValueError("lidar_rgb_dilation_radius_px must be an integer within [0, 64]")
        if self.lidar_rgb_dilation_radius_px > 0 and self.lidar_rgb_l1_weight <= 0.0:
            raise ValueError(
                "lidar_rgb_dilation_radius_px requires positive lidar_rgb_l1_weight"
            )
        if self.rgb_ssim_mode not in {"local_gaussian", "global_moments"}:
            raise ValueError("rgb_ssim_mode must be local_gaussian or global_moments")
        if self.ssim_window_size <= 0 or self.ssim_window_size % 2 == 0:
            raise ValueError("ssim_window_size must be a positive odd integer")
        if self.ssim_sigma <= 0.0 or not 0.0 < self.ssim_min_valid_fraction <= 1.0:
            raise ValueError("SSIM sigma/valid fraction are outside the supported range")
        if self.lidar_range_loss_mode not in {"linear_l1", "robust_log_huber"}:
            raise ValueError("lidar_range_loss_mode must be linear_l1 or robust_log_huber")
        if self.lidar_log_range_huber_delta <= 0.0:
            raise ValueError("lidar_log_range_huber_delta must be positive")
        if not 0.0 < self.lidar_alpha_target <= 1.0:
            raise ValueError("lidar_alpha_target must be within (0, 1]")
        if (
            not isinstance(self.lidar_alpha_dilation_radius_px, int)
            or isinstance(self.lidar_alpha_dilation_radius_px, bool)
            or not 0 <= self.lidar_alpha_dilation_radius_px <= 16
        ):
            raise ValueError(
                "lidar_alpha_dilation_radius_px must be an integer within [0, 16]"
            )
        if self.lidar_alpha_dilation_radius_px > 0 and self.lidar_alpha_weight <= 0.0:
            raise ValueError(
                "lidar_alpha_dilation_radius_px requires positive lidar_alpha_weight"
            )
        if (
            self.lidar_linear_aux_weight > 0.0
            and self.lidar_range_loss_mode != "robust_log_huber"
        ):
            raise ValueError(
                "lidar_linear_aux_weight is only valid with robust_log_huber"
            )
        if self.sh_regularization_weight < 0.0:
            raise ValueError("sh_regularization_weight must be non-negative")
        if self.pinhole_rasterize_mode not in ("classic", "antialiased"):
            raise ValueError("pinhole_rasterize_mode must be 'classic' or 'antialiased'")
        if self.pinhole_with_ut and self.pinhole_rasterize_mode != "classic":
            raise ValueError("pinhole_with_ut requires classic rasterization")
        self.ppisp.validate()
        self.contribution.validate()
        if self.contribution_every <= 0:
            raise ValueError("contribution_every must be positive")
        self.lidar_normal_alignment.validate()
        self.lidar_admission.validate()
        if self.lidar_admission.enabled and not self.error_weighted_sampling.enabled:
            # Admission is a multiplier on the error-weighted sampling weights;
            # the plain-opacity fallback path never calls sampling_weights, so
            # enabling it alone would silently do nothing at all.
            raise ValueError(
                "lidar_admission requires error_weighted_sampling to be enabled"
            )
        self.tangent_proposal.validate()
        if (
            self.tangent_proposal.reject_unsupported_births
            and not self.tangent_proposal.enabled
        ):
            raise ValueError(
                "reject_unsupported_births requires tangent_proposal.enabled"
            )
        if (
            self.tangent_proposal.enabled
            and self.densification_strategy != "default_3dgs"
            and not self.error_weighted_sampling.enabled
        ):
            # The MCMC proposal runs inside the weighted _add_new_gs path.  The
            # classic split/clone adapter has its own explicit birth hook.
            raise ValueError(
                "MCMC tangent_proposal requires error_weighted_sampling to be enabled"
            )
        # In the MCMC route reject_unsupported_births hard-filters relocation
        # and addition source parents, then falls back to that supported parent
        # if a random tangent proposal itself misses the trusted surface patch.
        if self.ppisp.enabled and self.exposure_compensation.enabled:
            raise ValueError(
                "ppisp replaces scalar exposure compensation; enable only one"
            )
        if self.ppisp.enabled and self.decoupled_ssim:
            raise ValueError(
                "decoupled_ssim currently supports only the scalar exposure gain"
            )
        frozen_photometric_nuisance = (
            self.ppisp.enabled
            and self.ppisp.learning_rate == 0.0
            and not self.exposure_compensation.enabled
        ) or (
            self.exposure_compensation.enabled
            and self.exposure_compensation.learning_rate == 0.0
            and not self.ppisp.enabled
        )
        expected_lrs = {"means", "scales", "quats", "opacities", "colors"}
        if set(self.learning_rates) != expected_lrs:
            raise ValueError(f"learning_rates must contain exactly {sorted(expected_lrs)}")
        geometry_groups = ("means", "scales", "quats")
        if any(float(self.learning_rates[name]) < 0.0 for name in geometry_groups):
            raise ValueError("geometry learning rates must be non-negative")
        opacity_only_surface_alpha_probe = (
            float(self.learning_rates["colors"]) == 0.0
            and all(float(self.learning_rates[name]) == 0.0 for name in geometry_groups)
            and float(self.learning_rates["opacities"]) > 0.0
            and self.lidar_alpha_weight > 0.0
            and self.lidar_range_weight == 0.0
            and self.geometry_regularization.opacity_sparsity_weight == 0.0
            and frozen_photometric_nuisance
        )
        surface_shape_probe = (
            float(self.learning_rates["colors"]) == 0.0
            and float(self.learning_rates["means"]) == 0.0
            and float(self.learning_rates["opacities"]) == 0.0
            and (
                float(self.learning_rates["scales"]),
                float(self.learning_rates["quats"]),
                self.lidar_normal_alignment.weight_align,
                self.lidar_normal_alignment.weight_flatten,
                self.lidar_normal_alignment.flatten_ratio_target,
            )
            in {
                (0.001, 0.0002, 0.01, 0.01, 0.15),
                (0.003, 0.0005, 0.1, 0.1, 0.15),
            }
            and self.lidar_alpha_weight == 1.0
            and self.lidar_alpha_target == 0.95
            and self.lidar_alpha_dilation_radius_px == 3
            and self.lidar_range_weight == 0.0
            and self.geometry_regularization.opacity_sparsity_weight == 0.0
            and frozen_photometric_nuisance
            and self.lidar_normal_alignment.enabled
            and self.lidar_normal_alignment.weight_point_to_plane == 0.0
            and self.lidar_normal_alignment.flatten_mode == "tangent_ratio"
            and self.lidar_normal_alignment.flatten_ratio_target == 0.15
            and (
                self.topology_policy.mode == "strict_fixed"
                or (
                    self.mcmc_refine_stop_iter == 602
                    and self.default_strategy.get("refine_scale2d_stop_iter") == 602
                )
            )
        )
        if float(self.learning_rates["colors"]) <= 0.0 and not (
            opacity_only_surface_alpha_probe or surface_shape_probe
        ):
            raise ValueError("colour learning rate must be positive")
        if float(self.learning_rates["opacities"]) < 0.0:
            raise ValueError("opacity learning rate must be non-negative")
        if (
            float(self.learning_rates["opacities"]) == 0.0
            and self.topology_policy.mode != "strict_fixed"
            and not surface_shape_probe
        ):
            raise ValueError(
                "frozen opacity requires strict_fixed topology so births and culls "
                "cannot create an untrained opacity population"
            )
        if self.mcmc_refine_start_iter < 0 or self.mcmc_refine_stop_iter <= 0:
            raise ValueError("MCMC refine bounds must be non-negative/positive")
        if self.mcmc_refine_start_iter >= self.mcmc_refine_stop_iter:
            raise ValueError("MCMC refine start must be smaller than refine stop")
        if self.mcmc_refine_every <= 0 or self.mcmc_noise_lr < 0.0:
            raise ValueError("MCMC refine interval must be positive and noise LR non-negative")
        if not 0.0 <= self.mcmc_growth_rate <= 1.0:
            raise ValueError("mcmc_growth_rate must be within [0, 1]")
        if not 0.0 < self.mcmc_relocation_max_fraction <= 1.0:
            raise ValueError("mcmc_relocation_max_fraction must be within (0, 1]")
        if (
            self.mcmc_relocation_max_scale_m is not None
            and self.mcmc_relocation_max_scale_m <= 0.0
        ):
            raise ValueError("mcmc_relocation_max_scale_m must be positive")
        if (
            self.mcmc_relocation_max_anisotropy is not None
            and self.mcmc_relocation_max_anisotropy <= 1.0
        ):
            raise ValueError("mcmc_relocation_max_anisotropy must exceed one")
        if not 0.0 <= self.post_refine_geometry_lr_scale <= 1.0:
            raise ValueError("post_refine_geometry_lr_scale must be within [0, 1]")
        if self.mcmc_noise_injection_stop_iter < -1:
            raise ValueError("MCMC noise stop must be -1 or non-negative")
        if self.densification_strategy not in DENSIFICATION_STRATEGIES:
            raise ValueError(
                f"densification_strategy must be one of {list(DENSIFICATION_STRATEGIES)}"
            )
        if not 0.0 < float(self.init_opacity) < 1.0:
            raise ValueError("init_opacity must be within (0, 1)")
        if self.sky_shell_checkpoint is not None:
            if not self.sky_shell_checkpoint.is_file():
                raise FileNotFoundError(
                    f"sky shell checkpoint is missing: {self.sky_shell_checkpoint}"
                )
            if self.color_model != "sh":
                raise ValueError("sky_shell_checkpoint requires color_model='sh'")
        if self.densification_gradient_source not in ("total_loss", "rgb_only"):
            raise ValueError(
                "densification_gradient_source must be 'total_loss' or 'rgb_only'"
            )
        if (
            self.densification_gradient_source == "rgb_only"
            and self.densification_strategy != "default_3dgs"
        ):
            # The MCMC path never reads means2d.grad, so accepting the knob
            # there would be a silent no-op - the class of bug this campaign
            # keeps paying for.
            raise ValueError(
                "densification_gradient_source='rgb_only' requires "
                "densification_strategy='default_3dgs'"
            )
        if self.default_strategy.get("exact_mipmap_lifecycle") is True:
            if self.densification_strategy != "default_3dgs":
                raise ValueError("exact MipMap lifecycle requires classic 3DGS")
            if self.renderer_mask_manifest is None:
                raise ValueError(
                    "exact MipMap lifecycle requires a signed renderer mask manifest"
                )
            detail_split_policy = self.default_strategy.get(
                "detail_split_policy", "vendor_0_2m"
            )
            if detail_split_policy not in {
                "vendor_0_2m",
                "lidar_surface_screen_detail",
            }:
                raise ValueError("exact MipMap detail split policy is invalid")
            execution_order = self.default_strategy.get(
                "lifecycle_execution_order", "post_optimizer_gsplat"
            )
            if execution_order not in {
                "post_optimizer_gsplat",
                "pre_optimizer_vendor",
            }:
                raise ValueError("exact MipMap lifecycle execution order is invalid")
            expected_prune_opa = 0.1
            expected_prune_opa_late = 0.05
            vendor_cull_warmup_profile = None
            vendor_opacity_reset_profile = "exact_every300"
            expected_reset_every = 300
            if execution_order == "pre_optimizer_vendor":
                vendor_cull_warmup_profile = self.default_strategy.get(
                    "vendor_cull_warmup_profile", "exact_0p10_to_0p05"
                )
                vendor_cull_thresholds = {
                    "exact_0p10_to_0p05": (0.1, 0.05),
                    "compatibility_uniform_0p05": (0.05, 0.05),
                    "calibrated_uniform_0p04": (0.04, 0.04),
                    "calibrated_geometry_only_0p00": (0.0, 0.0),
                }
                expected_cull_thresholds = vendor_cull_thresholds.get(
                    vendor_cull_warmup_profile
                )
                if expected_cull_thresholds is None:
                    raise ValueError(
                        "pre-optimizer vendor lifecycle cull warm-up profile "
                        "is invalid"
                    )
                expected_prune_opa, expected_prune_opa_late = (
                    expected_cull_thresholds
                )
                vendor_opacity_reset_profile = self.default_strategy.get(
                    "vendor_opacity_reset_profile", "exact_every300"
                )
                vendor_reset_intervals = {
                    "exact_every300": 300,
                    "deferred_every3000_compatibility": 3000,
                }
                expected_reset_every = vendor_reset_intervals.get(
                    vendor_opacity_reset_profile
                )
                if expected_reset_every is None:
                    raise ValueError(
                        "pre-optimizer vendor opacity reset profile is invalid"
                    )
            expected_lifecycle = {
                "growth_min_opacity": 0.15,
                # Corrected reverse (2026-08-30): the split/clone size boundary
                # is 0.15 m; the world-size CULL limit stays 0.2 m. The two had
                # been conflated at 0.2 in earlier recoveries.
                "split_scale_m": 0.15,
                "prune_scale_m": 0.2,
                "prune_opa": expected_prune_opa,
                "prune_opa_late": expected_prune_opa_late,
                "prune_switch_step": self.max_steps // 2,
                "prune_scale2d": 0.15,
                "reset_every": expected_reset_every,
                "reset_opacity_cap": 0.2,
                "refine_scale2d_stop_iter": self.mcmc_refine_stop_iter,
            }
            mismatched = {
                key: (expected, self.default_strategy.get(key))
                for key, expected in expected_lifecycle.items()
                if self.default_strategy.get(key) != expected
            }
            if mismatched:
                raise ValueError(
                    f"exact MipMap lifecycle parameters differ: {mismatched}"
                )
            gradient_pair = (
                bool(self.default_strategy.get("absgrad")),
                self.default_strategy.get("grow_grad2d"),
            )
            allowed_gradient_pairs = {
                # Historical V26/V33/V34 compatibility only. This pair is no
                # longer emitted by the builder because gsplat documents that
                # AbsGrad requires a materially higher threshold.
                (True, 0.00015),
                # Recovered competitor/plain projected-gradient semantics.
                (False, 0.00015),
                # Plain-gradient calibration arms measured from the real
                # Tile_1 step-600 distribution (P95/P99 fall between these).
                (False, 0.000075),
                (False, 0.0001),
                # Bounded AbsGrad calibration arms.
                (True, 0.0004),
                (True, 0.0008),
                (True, 0.0012),
            }
            if gradient_pair not in allowed_gradient_pairs:
                raise ValueError(
                    "exact MipMap gradient semantics are not an approved profile"
                )
            if not isinstance(
                self.default_strategy.get(
                    "capacity_conserving_clone_opacity", False
                ),
                bool,
            ):
                raise ValueError(
                    "capacity-conserving clone opacity flag must be boolean"
                )
            if detail_split_policy == "lidar_surface_screen_detail":
                if self.default_strategy.get("detail_split_scale_m") != 0.02:
                    raise ValueError("detail split scale must be 0.02 m")
                if self.default_strategy.get("detail_split_screen_radius") != 0.0035:
                    raise ValueError(
                        "detail split screen radius must be 0.0035"
                    )
            cull_policy = self.default_strategy.get(
                "opacity_cull_policy", "immediate"
            )
            if cull_policy not in {
                "immediate",
                "observation_aware",
                "local_coverage_competition",
            }:
                raise ValueError("exact MipMap opacity cull policy is invalid")
            if cull_policy == "local_coverage_competition":
                if self.default_strategy.get("opacity_cull_local_voxel_m") != 0.02:
                    raise ValueError("local coverage cull voxel must be 0.02 m")
                if (
                    self.default_strategy.get(
                        "opacity_cull_grace_after_reset_steps"
                    )
                    != 100
                ):
                    raise ValueError(
                        "local coverage cull requires a 100-step reset grace"
                    )
            if (
                self.mcmc_refine_start_iter != 500
                or self.mcmc_refine_every != 100
                or self.mcmc_refine_stop_iter > self.max_steps
            ):
                raise ValueError(
                    "exact MipMap lifecycle requires start=500, interval=100, "
                    "and stop no later than max_steps"
                )
            if execution_order == "pre_optimizer_vendor":
                vendor_mismatches: dict[str, tuple[Any, Any]] = {}
                vendor_expected = {
                    "absgrad": False,
                    "revised_opacity": (
                        detail_split_policy
                        == "lidar_surface_screen_detail"
                    ),
                    "detail_split_policy": detail_split_policy,
                    "opacity_cull_policy": "immediate",
                    "prune_opa": expected_cull_thresholds[0],
                    "prune_opa_late": expected_cull_thresholds[1],
                    "vendor_opacity_reset_profile": (
                        vendor_opacity_reset_profile
                    ),
                }
                vendor_grow_grad2d = self.default_strategy.get("grow_grad2d")
                if vendor_grow_grad2d not in {0.000075, 0.0001, 0.00015}:
                    vendor_mismatches["grow_grad2d"] = (
                        "one of 7.5e-5, 1e-4, 1.5e-4",
                        vendor_grow_grad2d,
                    )
                for key, expected in vendor_expected.items():
                    default_value = None
                    if key == "opacity_cull_policy":
                        default_value = "immediate"
                    elif key == "vendor_opacity_reset_profile":
                        default_value = "exact_every300"
                    actual = self.default_strategy.get(
                        key,
                        default_value,
                    )
                    if actual != expected:
                        vendor_mismatches[key] = (expected, actual)
                if vendor_mismatches:
                    raise ValueError(
                        "pre-optimizer vendor lifecycle contains non-vendor "
                        f"strategy settings: {vendor_mismatches}"
                    )
                if self.densification_gradient_source != "total_loss":
                    raise ValueError(
                        "pre-optimizer vendor lifecycle requires the accumulated "
                        "full-loss projected gradient"
                    )
                if self.tangent_proposal.enabled or self.lidar_admission.enabled:
                    raise ValueError(
                        "pre-optimizer vendor lifecycle forbids CloudStudio birth guards"
                    )
                approved_vendor_alpha = {(0.0, 0.95), (0.02, 0.95)}
                if opacity_only_surface_alpha_probe or surface_shape_probe:
                    approved_vendor_alpha.update({(0.1, 0.95), (1.0, 0.95)})
                if (
                    self.lidar_alpha_weight,
                    self.lidar_alpha_target,
                ) not in approved_vendor_alpha:
                    raise ValueError(
                        "pre-optimizer vendor lifecycle permits only the signed "
                        "LiDAR alpha compatibility profile"
                    )
                if self.lidar_alpha_dilation_radius_px not in (
                    {0, 3}
                    if opacity_only_surface_alpha_probe or surface_shape_probe
                    else {0}
                ):
                    raise ValueError(
                        "pre-optimizer vendor lifecycle permits only the signed "
                        "surface-alpha dilation profile"
                    )
                cull_defaults = {
                    "opacity_cull_min_observations": 0,
                    "opacity_cull_consecutive_events": 1,
                    "opacity_cull_grace_after_reset_steps": 0,
                    "opacity_cull_max_fraction": 1.0,
                    "opacity_cull_priority": "lowest_opacity",
                    "opacity_cull_local_min_accumulated_alpha": 0.0,
                }
                cull_mismatches = {
                    key: (expected, self.default_strategy.get(key, expected))
                    for key, expected in cull_defaults.items()
                    if self.default_strategy.get(key, expected) != expected
                }
                if cull_mismatches:
                    raise ValueError(
                        "pre-optimizer vendor lifecycle contains CloudStudio cull "
                        f"protections: {cull_mismatches}"
                    )
        self.rig_pose_refinement.validate()
        self.exposure_compensation.validate()
        self.geometry_regularization.validate()
        self.golden_evaluation.validate()
        assert_trainer_preset_matches(
            self.trainer_preset,
            {
                "metric_scale_calibration": self.metric_scale_calibration.to_dict(),
                "color_model": self.color_model,
                "sh_degree": self.sh_degree,
                "sh_degree_interval": self.sh_degree_interval,
                "rgb_ssim_mode": self.rgb_ssim_mode,
                "decoupled_ssim": self.decoupled_ssim,
                "sh_regularization_weight": self.sh_regularization_weight,
                "lidar_range_loss_mode": self.lidar_range_loss_mode,
                "lidar_linear_aux_weight": self.lidar_linear_aux_weight,
                "means_lr_final_factor": self.means_lr_final_factor,
                "background_color": None
                if self.background_color is None
                else [float(value) for value in self.background_color],
                "exposure_compensation": self.exposure_compensation.to_dict(),
                "geometry_regularization": self.geometry_regularization.to_dict(),
            },
        )
        if (self.depth_manifest is None) != (self.depth_root is None):
            raise ValueError("depth_manifest and depth_root must be provided together")
        if (self.face_cache_manifest is None) != (self.face_cache_root is None):
            raise ValueError(
                "face_cache_manifest and face_cache_root must be provided together"
            )
        if self.raw_fisheye_post_refine_face_manifest is not None:
            if self.face_cache_manifest is not None:
                raise ValueError(
                    "raw-fisheye post-refine lineage cannot also consume Face4"
                )
            if self.warm_start_checkpoint is None:
                raise ValueError("raw-fisheye post-refine requires a warm start")
            if self.topology_policy.mode != "strict_fixed":
                raise ValueError("raw-fisheye post-refine requires strict topology")
            if any(
                float(self.learning_rates[name]) != 0.0
                for name in ("means", "scales", "quats")
            ):
                raise ValueError(
                    "raw-fisheye post-refine requires frozen geometry learning rates"
                )
            if not self.raw_fisheye_post_refine_face_manifest.is_file():
                raise FileNotFoundError(
                    "raw-fisheye post-refine Face4 lineage is missing: "
                    f"{self.raw_fisheye_post_refine_face_manifest}"
                )
            verify_face_manifest(
                json.loads(
                    self.raw_fisheye_post_refine_face_manifest.read_text(
                        encoding="utf-8"
                    )
                )
            )
        if self.renderer_mask_manifest is not None and self.face_cache_manifest is None:
            raise ValueError("renderer mask consumption requires face-cache training")
        if self.renderer_mask_manifest is not None:
            if not self.renderer_mask_manifest.is_file():
                raise FileNotFoundError(
                    f"renderer mask manifest is missing: {self.renderer_mask_manifest}"
                )
            if not self.face_cache_manifest.is_file():
                raise FileNotFoundError(
                    f"face cache manifest is missing: {self.face_cache_manifest}"
                )
            renderer_mask = json.loads(
                self.renderer_mask_manifest.read_text(encoding="utf-8")
            )
            verify_renderer_mask_manifest(renderer_mask)
            face_cache = json.loads(
                self.face_cache_manifest.read_text(encoding="utf-8")
            )
            if renderer_mask.get("source_face_manifest_sha256") != face_cache.get(
                "face_manifest_sha256"
            ):
                raise ValueError(
                    "renderer mask manifest is bound to different Face4 inputs"
                )
        if (self.tile_inputs_manifest is None) != (self.tile_inputs_root is None):
            raise ValueError(
                "tile_inputs_manifest and tile_inputs_root must be provided together"
            )
        if self.tile_inputs_manifest is not None and self.face_cache_manifest is None:
            raise ValueError("Tile Face4 crops require face-cache training")
        if (self.mono_depth_manifest is None) != (self.mono_depth_root is None):
            raise ValueError(
                "mono_depth_manifest and mono_depth_root must be provided together"
            )
        if (self.face_lidar_geometry_manifest is None) != (
            self.face_lidar_geometry_root is None
        ):
            raise ValueError(
                "Face4 LiDAR geometry manifest and root must be provided together"
            )
        if (self.mesh_geometry_manifest is None) != (self.mesh_geometry_root is None):
            raise ValueError(
                "mesh geometry manifest and root must be provided together"
            )
        if self.mesh_geometry_manifest is not None and self.face_cache_manifest is None:
            raise ValueError("mesh geometry requires face-cache training")
        if (
            self.face_lidar_geometry_manifest is not None
            and self.face_cache_manifest is None
        ):
            raise ValueError("Face4 LiDAR geometry requires face-cache training")
        if self.lidar_range_weight > 0.0 and self.face_cache_manifest is not None:
            if self.face_lidar_geometry_manifest is None:
                raise ValueError(
                    "positive LiDAR range weight on Face4 requires signed Face4 geometry"
                )
        if self.lidar_alpha_weight > 0.0 and self.face_cache_manifest is not None:
            if self.face_lidar_geometry_manifest is None:
                raise ValueError(
                    "positive LiDAR alpha weight on Face4 requires signed Face4 geometry"
                )
        if self.da2_depth_weight < 0.0:
            raise ValueError("DA2 depth weight must be non-negative")
        if self.da2_depth_weight > 0.0 and self.mono_depth_manifest is None:
            raise ValueError("positive DA2 depth weight requires a DA2 manifest and root")
        if self.mesh_depth_weight < 0.0 or self.mesh_normal_weight < 0.0:
            raise ValueError("mesh depth and normal weights must be non-negative")
        if (self.mesh_depth_weight > 0.0 or self.mesh_normal_weight > 0.0) and (
            self.mesh_geometry_manifest is None
        ):
            raise ValueError("positive mesh loss weight requires signed mesh geometry")
        if self.competitor_loss_schedule_enabled:
            if self.da2_depth_weight > 0.0 and self.mono_depth_manifest is None:
                raise ValueError(
                    "competitor loss schedule requires signed DA2 when its weight is positive"
                )
            if (
                self.mesh_depth_weight > 0.0 or self.mesh_normal_weight > 0.0
            ) and self.mesh_geometry_manifest is None:
                raise ValueError(
                    "competitor loss schedule requires signed mesh geometry when its weight is positive"
                )
            if self.rgb_l1_weight != 0.6 or self.rgb_ssim_weight != 0.4:
                raise ValueError("competitor loss schedule requires RGB weights [0.6, 0.4]")
        if not self.final_evaluation_artifacts:
            if self.face_cache_manifest is None and not (
                self.implementation_smoke_only
                and self.raw_fisheye_post_refine_face_manifest is not None
            ):
                raise ValueError(
                    "final_evaluation_artifacts may be disabled only for "
                    "face-cache training or a bounded raw-fisheye post-refine smoke"
                )
            if self.golden_evaluation.enabled:
                raise ValueError(
                    "golden_evaluation must be disabled when "
                    "final_evaluation_artifacts is false"
                )
        if (self.person_mask_manifest is None) != (self.person_mask_root is None):
            raise ValueError(
                "person_mask_manifest and person_mask_root must be provided together"
            )
        if self.require_person_masks and self.person_mask_manifest is None:
            raise ValueError(
                "production 3DGS training requires person_mask_manifest and person_mask_root"
            )
        if self.surface_initialization.enabled:
            if self.initialization_geometry is None:
                raise ValueError(
                    "surface_initialization requires initialization_geometry"
                )
            if not self.initialization_geometry.is_file():
                raise FileNotFoundError(
                    f"initialization geometry is missing: {self.initialization_geometry}"
                )
            if self.surface_initialization.mode == "mipmap_k7_k30":
                if self.metric_scale_calibration.mode != "precomputed":
                    raise ValueError(
                        "mipmap_k7_k30 initialization requires precomputed metric scales"
                    )
                if self.initialization_geometry_manifest is None:
                    raise ValueError(
                        "mipmap_k7_k30 initialization requires a signed geometry manifest"
                    )
                if self.mipmap_tile_id is None or self.mipmap_tile_id < 0:
                    raise ValueError(
                        "mipmap_k7_k30 initialization requires mipmap_tile_id"
                    )
                if not self.initialization_geometry_manifest.is_file():
                    raise FileNotFoundError(
                        "initialization geometry manifest is missing: "
                        f"{self.initialization_geometry_manifest}"
                    )
                geometry_manifest = json.loads(
                    self.initialization_geometry_manifest.read_text(encoding="utf-8")
                )
                verify_tile_geometry_manifest(
                    geometry_manifest,
                    root=self.initialization_geometry_manifest.parent,
                    verify_artifacts=True,
                )
                matches = [
                    tile
                    for tile in geometry_manifest["tiles"]
                    if int(tile["tile_id"]) == int(self.mipmap_tile_id)
                ]
                if len(matches) != 1:
                    raise ValueError("geometry manifest does not contain the selected Tile")
                selected = matches[0]
                expected_geometry = (
                    self.initialization_geometry_manifest.parent
                    / selected["geometry"]["path"]
                ).resolve()
                if self.initialization_geometry.resolve() != expected_geometry:
                    raise ValueError(
                        "initialization geometry path differs from the signed Tile artifact"
                    )
                if _sha256_file(self.initialization_ply) != selected.get(
                    "initialization_ply_sha256"
                ):
                    raise ValueError(
                        "initialization PLY differs from the geometry manifest binding"
                    )
                if self.tile_inputs_manifest is None or self.tile_inputs_root is None:
                    raise ValueError(
                        "mipmap_k7_k30 initialization requires signed Tile inputs"
                    )
                if not self.tile_inputs_manifest.is_file():
                    raise FileNotFoundError(
                        f"Tile inputs manifest is missing: {self.tile_inputs_manifest}"
                    )
                tile_inputs = json.loads(
                    self.tile_inputs_manifest.read_text(encoding="utf-8")
                )
                tile_inputs_sha = verify_tile_inputs_manifest(
                    tile_inputs,
                    root=self.tile_inputs_root,
                    verify_artifacts=True,
                )
                if geometry_manifest.get("tile_inputs_manifest_sha256") != tile_inputs_sha:
                    raise ValueError(
                        "Tile geometry manifest is bound to different Tile inputs"
                    )
                input_matches = [
                    tile
                    for tile in tile_inputs["tiles"]
                    if int(tile["tile_id"]) == int(self.mipmap_tile_id)
                ]
                if len(input_matches) != 1:
                    raise ValueError("Tile inputs do not contain the selected Tile")
                expected_ply = (
                    self.tile_inputs_root / input_matches[0]["initialization"]["path"]
                ).resolve()
                if self.initialization_ply.resolve() != expected_ply:
                    raise ValueError(
                        "initialization PLY path differs from the signed Tile input"
                    )
        elif self.metric_scale_calibration.mode == "precomputed":
            raise ValueError(
                "precomputed metric scales require enabled mipmap_k7_k30 initialization"
            )
        if (
            self.lidar_range_weight > 0.0
            or self.lidar_linear_aux_weight > 0.0
            or self.lidar_alpha_weight > 0.0
        ) and self.depth_manifest is None:
            raise ValueError(
                "positive LiDAR loss weight requires depth_manifest and depth_root"
            )
        if self.lidar_rgb_l1_weight > 0.0 and self.depth_manifest is None:
            raise ValueError(
                "positive LiDAR-supported RGB weight requires depth_manifest and depth_root"
            )
        required_paths = {
            "dataset_manifest": self.dataset_manifest,
            "recording_root": self.recording_root,
            "mask_manifest": self.mask_manifest,
            "mask_root": self.mask_root,
            "split_manifest": self.split_manifest,
            "initialization_ply": self.initialization_ply,
            "output_dir": self.output_dir,
            "gsplat_lock": self.gsplat_lock,
        }
        missing = [name for name, path in required_paths.items() if path is None]
        if missing:
            raise ValueError(f"trainer config is missing paths: {', '.join(sorted(missing))}")
        if self.warm_start_checkpoint is not None and not self.warm_start_checkpoint.is_file():
            raise FileNotFoundError(
                f"warm-start checkpoint is missing: {self.warm_start_checkpoint}"
            )
        if self.dataset_manifest.is_file():
            dataset = json.loads(self.dataset_manifest.read_text(encoding="utf-8"))
            lineage = dataset.get("training_lineage", {})
            if (
                lineage.get("independent_at_algorithm_version")
                == INDEPENDENT_AT_ALGORITHM
            ):
                if self.mipmap_pipeline_gate is None:
                    raise ValueError(
                        "shared-single-focal KB4 data requires mipmap_pipeline_gate; "
                        "the Face4/semantic/depth/Tile sequence may not be skipped"
                    )
                if not self.split_manifest.is_file():
                    raise FileNotFoundError(
                        f"split manifest is missing: {self.split_manifest}"
                    )
                split = json.loads(self.split_manifest.read_text(encoding="utf-8"))
                face_sha = None
                if self.face_cache_manifest is not None:
                    if not self.face_cache_manifest.is_file():
                        raise FileNotFoundError(
                            f"face cache manifest is missing: {self.face_cache_manifest}"
                        )
                    face = json.loads(
                        self.face_cache_manifest.read_text(encoding="utf-8")
                    )
                    face_sha = str(face.get("face_manifest_sha256", ""))
                elif self.raw_fisheye_post_refine_face_manifest is not None:
                    face = json.loads(
                        self.raw_fisheye_post_refine_face_manifest.read_text(
                            encoding="utf-8"
                        )
                    )
                    face_sha = verify_face_manifest(face)
                verify_training_gate(
                    self.mipmap_pipeline_gate,
                    dataset_manifest_sha256=str(dataset.get("manifest_sha256", "")),
                    split_manifest_sha256=str(
                        split.get("split_manifest_sha256", "")
                    ),
                    face_manifest_sha256=face_sha,
                    allow_implementation_smoke=self.implementation_smoke_only,
                    fixed_topology_arm_fingerprint_sha256=(
                        fixed_topology_evaluation_arm_fingerprint(
                            {
                                "run_id": self.run_id,
                                "mipmap_tile_id": self.mipmap_tile_id,
                                "topology_policy": self.topology_policy.to_dict(),
                                "fixed_topology_schedule": (
                                    self.fixed_topology_schedule.to_dict()
                                ),
                                "max_steps": self.max_steps,
                                "factor": self.factor,
                                "color_model": self.color_model,
                                "sh_degree": self.sh_degree,
                                "da2_depth_weight": self.da2_depth_weight,
                                "mesh_depth_weight": self.mesh_depth_weight,
                                "mesh_normal_weight": self.mesh_normal_weight,
                                "competitor_loss_schedule_enabled": (
                                    self.competitor_loss_schedule_enabled
                                ),
                                "lidar_admission": self.lidar_admission.to_dict(),
                                "tangent_proposal": self.tangent_proposal.to_dict(),
                                "densification_strategy": self.densification_strategy,
                            }
                        )
                        if self.topology_policy.mode != "adaptive_growth"
                        else None
                    ),
                    adaptive_growth_config_manifest_sha256=(
                        self.config_manifest_sha256
                        if self.topology_policy.mode == "adaptive_growth"
                        else None
                    ),
                    adaptive_growth_warm_start_checkpoint_sha256=(
                        _sha256_file(self.warm_start_checkpoint)
                        if self.topology_policy.mode == "adaptive_growth"
                        and self.warm_start_checkpoint is not None
                        else None
                    ),
                    fixed_topology_config_manifest_sha256=(
                        self.config_manifest_sha256
                        if self.topology_policy.mode != "adaptive_growth"
                        else None
                    ),
                    fixed_topology_warm_start_checkpoint_sha256=(
                        _sha256_file(self.warm_start_checkpoint)
                        if self.topology_policy.mode != "adaptive_growth"
                        and self.warm_start_checkpoint is not None
                        else None
                    ),
                )
                gate, _ = load_and_verify_gate(self.mipmap_pipeline_gate)
                if self.renderer_mask_manifest is not None:
                    renderer_mask = json.loads(
                        self.renderer_mask_manifest.read_text(encoding="utf-8")
                    )
                    renderer_mask_sha = verify_renderer_mask_manifest(renderer_mask)
                    if renderer_mask_sha != gate.get("bindings", {}).get(
                        "renderer_mask_train_manifest_sha256"
                    ):
                        raise ValueError(
                            "renderer mask manifest is bound to a different pipeline gate"
                        )
                if self.tile_inputs_manifest is not None:
                    tile_inputs = json.loads(
                        self.tile_inputs_manifest.read_text(encoding="utf-8")
                    )
                    if tile_inputs.get("tile_plan_manifest_sha256") != gate.get(
                        "bindings", {}
                    ).get("spatial_tile_plan_manifest_sha256"):
                        raise ValueError(
                            "Tile inputs are bound to a different pipeline gate"
                        )
                if self.mono_depth_manifest is not None:
                    mono = json.loads(
                        self.mono_depth_manifest.read_text(encoding="utf-8")
                    )
                    mono_sha = verify_mono_depth_manifest(mono)
                    if mono_sha != gate.get("bindings", {}).get(
                        "da2_train_manifest_sha256"
                    ):
                        raise ValueError(
                            "DA2 manifest is bound to a different pipeline gate"
                        )
                    if mono.get("source_face_manifest_sha256") != face_sha:
                        raise ValueError(
                            "DA2 manifest is bound to different Face4 inputs"
                        )
                if self.face_lidar_geometry_manifest is not None:
                    lidar_geometry = json.loads(
                        self.face_lidar_geometry_manifest.read_text(encoding="utf-8")
                    )
                    verify_face_lidar_geometry_manifest(lidar_geometry)
                    if lidar_geometry.get("source_face_manifest_sha256") != face_sha:
                        raise ValueError(
                            "Face4 LiDAR geometry is bound to different Face4 inputs"
                        )
                    if lidar_geometry.get("source_depth_manifest_sha256") != gate.get(
                        "bindings", {}
                    ).get("lidar_depth_manifest_sha256"):
                        raise ValueError(
                            "Face4 LiDAR geometry is bound to a different LiDAR depth gate"
                        )

    def contract_dict(self) -> dict[str, Any]:
        uses_lidar_linear_aux = self.lidar_linear_aux_weight > 0.0
        uses_lidar_rgb_boost = self.lidar_rgb_l1_weight > 0.0
        uses_lidar_alpha = self.lidar_alpha_weight > 0.0
        uses_rgb_gradient = self.rgb_gradient_weight > 0.0
        loss_weights = {
            "rgb_l1": self.rgb_l1_weight,
            "rgb_ssim": self.rgb_ssim_weight,
            "lidar_range": self.lidar_range_weight,
            "da2_depth": self.da2_depth_weight,
            "mesh_depth": self.mesh_depth_weight,
            "mesh_normal": self.mesh_normal_weight,
        }
        view_sampling_contract = {
            "mode": self.view_sampling_mode,
            "resume_identity": (
                "seed_epoch_and_step"
                if self.view_sampling_mode
                == "fisher_yates_without_replacement_per_epoch"
                else "torch_generator_state"
            ),
        }
        if self.lidar_rgb_l1_weight > 0.0:
            loss_weights["lidar_rgb_l1"] = self.lidar_rgb_l1_weight
        if uses_rgb_gradient:
            loss_weights["rgb_gradient_l1"] = self.rgb_gradient_weight
        if uses_lidar_alpha:
            loss_weights["lidar_alpha_coverage"] = self.lidar_alpha_weight
        lidar_range_contract = {
            "mode": self.lidar_range_loss_mode,
            "semantics": "euclidean_ray_range_m",
            "log_huber_delta": self.lidar_log_range_huber_delta,
            "confidence_weighted": True,
        }
        if uses_lidar_linear_aux:
            loss_weights["lidar_linear_aux"] = self.lidar_linear_aux_weight
            lidar_range_contract["linear_aux_weight"] = self.lidar_linear_aux_weight
        if self.densification_strategy == "default_3dgs":
            default_configuration = dict(self.default_strategy)
            default_configuration.setdefault(
                "refine_start_iter", self.mcmc_refine_start_iter
            )
            default_configuration.setdefault(
                "refine_stop_iter", self.mcmc_refine_stop_iter
            )
            default_configuration.setdefault("refine_every", self.mcmc_refine_every)
            strategy_contract = {
                "name": "DefaultStrategy",
                "implementation": "gsplat.strategy.DefaultStrategy",
                "gradient_source": self.densification_gradient_source,
                "configuration": dict(sorted(default_configuration.items())),
            }
        else:
            strategy_contract = {
                "name": "MCMC",
                "refine_start_iter": self.mcmc_refine_start_iter,
                "refine_stop_iter": self.mcmc_refine_stop_iter,
                "refine_every": self.mcmc_refine_every,
                "noise_injection_stop_iter": self.mcmc_noise_injection_stop_iter,
                "noise_lr": self.mcmc_noise_lr,
                "noise_std_fraction": self.metric_scale_calibration.noise_std_fraction,
                "growth_rate": self.mcmc_growth_rate,
                "relocation_max_fraction": self.mcmc_relocation_max_fraction,
                "relocation_max_scale_m": self.mcmc_relocation_max_scale_m,
                "relocation_max_anisotropy": self.mcmc_relocation_max_anisotropy,
            }
            if self.post_refine_geometry_lr_scale != 1.0:
                strategy_contract["post_refine_geometry_lr_scale"] = (
                    self.post_refine_geometry_lr_scale
                )
        if self.error_weighted_sampling.enabled:
            strategy_contract["error_weighted_sampling"] = (
                self.error_weighted_sampling.to_dict()
            )
        # Recorded only when active, unlike the older unconditional blocks:
        # contract_dict feeds trainer_config_sha256, which resume compares
        # against the checkpoint identity. An unconditional key would change the
        # hash of every existing config and invalidate every existing checkpoint
        # - including the L0 control arm, whose whole point is to stay identical
        # to a pre-WP-4 run.
        if self.lidar_admission.enabled:
            strategy_contract["lidar_admission"] = self.lidar_admission.to_dict()
        if self.tangent_proposal.enabled:
            # Same conditional rule and the same reason: an unconditional key
            # would rewrite trainer_config_sha256 for every pre-WP-5 config.
            strategy_contract["tangent_proposal"] = self.tangent_proposal.to_dict()
        if (
            self.topology_policy.mode != "adaptive_growth"
            or self.fixed_topology_schedule.enabled
        ):
            strategy_contract["topology_policy"] = self.topology_policy.to_dict()
            strategy_contract["active"] = self.topology_policy.strategy_enabled

        contract = {
            "schema_version": 2,
            "algorithm_version": (
                "cloudstudio_gsplat_trainer_v8_fixed_topology"
                if self.topology_policy.mode != "adaptive_growth"
                or self.fixed_topology_schedule.enabled
                else (
                    "cloudstudio_gsplat_trainer_v9_lidar_alpha"
                    if uses_lidar_alpha
                    else (
                        "cloudstudio_gsplat_trainer_v7"
                        if uses_rgb_gradient
                        else (
                            "cloudstudio_gsplat_trainer_v6"
                            if uses_lidar_rgb_boost
                            else (
                                "cloudstudio_gsplat_trainer_v5"
                                if uses_lidar_linear_aux
                                else "cloudstudio_gsplat_trainer_v4"
                            )
                        )
                    )
                )
            ),
            "trainer_preset": self.trainer_preset,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "evidence_scope": (
                "IMPLEMENTATION_SMOKE_ONLY_NOT_TRAINING_READY"
                if self.implementation_smoke_only
                else "TRAINING_RUN"
            ),
            "factor": self.factor,
            "crop": None
            if self.crop is None
            else {
                "x": self.crop.x,
                "y": self.crop.y,
                "width": self.crop.width,
                "height": self.crop.height,
            },
            "cap_max": self.cap_max,
            "init_scale_m": self.init_scale_m,
            "initialization": {
                "fixed_scale_m": self.init_scale_m,
                **self.metric_scale_calibration.to_dict(),
            },
            "loss_weights": loss_weights,
            "view_sampling": view_sampling_contract,
            "color_model": {
                "mode": self.color_model,
                "sh_degree": self.sh_degree if self.color_model == "sh" else None,
                "sh_degree_interval": self.sh_degree_interval
                if self.color_model == "sh"
                else None,
            },
            "background_compositing": {
                "color": None
                if self.background_color is None
                else [float(v) for v in self.background_color],
            },
            "means_lr_schedule": {
                "mode": "exponential_to_final_factor",
                "final_factor": self.means_lr_final_factor,
            },
            "cuda_memory_management": {
                "empty_cache_interval_steps": self.cuda_empty_cache_interval_steps,
                "enabled": self.cuda_empty_cache_interval_steps > 0,
                "tile_compatibility_value": 2,
            },
            "loss_contract": {
                "rgb_ssim": {
                    "mode": "mask_aware_local_gaussian"
                    if self.rgb_ssim_mode == "local_gaussian"
                    else "global_masked_moments",
                    "configuration_mode": self.rgb_ssim_mode,
                    "window_size": self.ssim_window_size,
                    "sigma": self.ssim_sigma,
                    "minimum_valid_fraction": self.ssim_min_valid_fraction,
                    "global_masked_ssim": "active"
                    if self.rgb_ssim_mode == "global_moments"
                    else "diagnostic_only",
                    "decoupled_exposure": self.decoupled_ssim,
                },
                "sh_regularization": {
                    "mode": "shN_l2",
                    "weight": self.sh_regularization_weight,
                },
                "lidar_range": lidar_range_contract,
                "lidar_alpha_coverage": {
                    "enabled": uses_lidar_alpha,
                    "source": (
                        "signed_lidar_depth_mask"
                        if self.lidar_alpha_dilation_radius_px == 0
                        else "signed_lidar_depth_mask_confidence_max_dilated"
                    ),
                    "target": self.lidar_alpha_target,
                    "dilation_radius_px": self.lidar_alpha_dilation_radius_px,
                    "loss": "confidence_weighted_squared_deficit",
                },
            },
            "dynamic_person_mask": {
                "required": self.require_person_masks,
                "rgb_composition": "base_rgb_mask & ~person_dynamic_mask",
                "depth_composition": "base_depth_valid & ~person_dynamic_mask",
            },
            "learning_rates": dict(sorted(self.learning_rates.items())),
            "optimizer": {
                "configured_learning_rates": dict(sorted(self.learning_rates.items())),
                "fixed_parameter_groups": sorted(
                    name
                    for name in ("means", "scales", "quats", "opacities")
                    if float(self.learning_rates[name]) == 0.0
                ),
                "means_step_fraction": self.metric_scale_calibration.means_step_fraction,
            },
            "face_split": {
                "enabled": self.face_cache_manifest is not None,
                "supervision": "pinhole_faces" if self.face_cache_manifest else "raw_fisheye",
                "validation": "raw_fisheye",
                "post_refine_face_lineage_sha256": (
                    verify_face_manifest(
                        json.loads(
                            self.raw_fisheye_post_refine_face_manifest.read_text(
                                encoding="utf-8"
                            )
                        )
                    )
                    if self.raw_fisheye_post_refine_face_manifest is not None
                    else None
                ),
                "pinhole_rasterize_mode": self.pinhole_rasterize_mode,
                "pinhole_with_ut": self.pinhole_with_ut,
            },
            "renderer": {
                "camera_model": "fisheye",
                "projection": "3DGUT",
                "with_ut": True,
                "with_eval3d": True,
                "range_mode": "RGB-Ed",
                "range_semantics": "euclidean_ray_range_m",
                "global_z_order": False,
                "packed": False,
            },
            "strategy": strategy_contract,
            "rig_pose_refinement": self.rig_pose_refinement.to_dict(),
            "exposure_compensation": self.exposure_compensation.to_dict(),
            "ppisp": self.ppisp.to_dict(),
            "contribution": {
                **self.contribution.to_dict(),
                "every": self.contribution_every,
            },
            "lidar_normal_alignment": self.lidar_normal_alignment.to_dict(),
            "geometry_regularization": self.geometry_regularization.to_dict(),
            "golden_evaluation": self.golden_evaluation.to_dict(),
            "viewer": False,
        }
        if (
            self.topology_policy.mode != "adaptive_growth"
            or self.fixed_topology_schedule.enabled
        ):
            contract["topology_policy"] = self.topology_policy.to_dict()
            contract["fixed_topology_schedule"] = (
                self.fixed_topology_schedule.to_dict()
            )
        if self.mipmap_pipeline_gate is not None:
            gate, gate_sha = load_and_verify_gate(self.mipmap_pipeline_gate)
            contract["mipmap_pipeline_gate"] = {
                "profile": gate["profile"],
                "status": gate["status"],
                "gate_manifest_sha256": gate_sha,
            }
        if not self.final_evaluation_artifacts:
            contract["face_split"]["final_raw_evaluation_artifacts"] = (
                "deferred_to_separate_3dgut_stage"
            )
        if self.checkpoint_keep_every > 0:
            contract["checkpoint_retention"] = {
                "keep_every_steps": self.checkpoint_keep_every,
                "path_pattern": "checkpoints/step_{step:08d}.pt",
            }
        if uses_lidar_rgb_boost:
            contract["loss_contract"]["lidar_rgb_support"] = {
                "source": "depth_mask",
                "dilation_radius_px": self.lidar_rgb_dilation_radius_px,
            }
        if uses_rgb_gradient:
            contract["loss_contract"]["rgb_gradient"] = {
                "mode": "masked_forward_difference_l1",
                "directions": ["horizontal", "vertical"],
            }
        if self.warm_start_checkpoint is not None:
            contract["warm_start"] = {
                "mode": "parameters_and_auxiliary_fresh_optimizers",
                "checkpoint_sha256": _sha256_file(self.warm_start_checkpoint),
                "min_opacity": float(self.warm_start_min_opacity),
                "fresh_auxiliary_names": list(self.warm_start_fresh_auxiliary),
            }
            if self.warm_start_scale_multiplier != 1.0:
                contract["warm_start"]["scale_multiplier"] = float(
                    self.warm_start_scale_multiplier
                )
        if self.surface_initialization.enabled:
            contract["initialization"]["surface_alignment"] = {
                **self.surface_initialization.to_dict(),
                "geometry_sha256": _sha256_file(self.initialization_geometry),
            }
            if self.surface_initialization.mode == "mipmap_k7_k30":
                geometry_manifest = json.loads(
                    self.initialization_geometry_manifest.read_text(encoding="utf-8")
                )
                contract["initialization"]["surface_alignment"].update(
                    {
                        "geometry_manifest_sha256": verify_tile_geometry_manifest(
                            geometry_manifest
                        ),
                        "mipmap_tile_id": int(self.mipmap_tile_id),
                    }
                )
        if self.tile_inputs_manifest is not None:
            tile_inputs = json.loads(
                self.tile_inputs_manifest.read_text(encoding="utf-8")
            )
            contract["tile_inputs"] = {
                "manifest_sha256": verify_tile_inputs_manifest(tile_inputs),
                "tile_id": self.mipmap_tile_id,
                "face4_crop_consumed": self.face_cache_manifest is not None,
                "principal_point_shifted_by_crop_origin": True,
            }
        if self.mono_depth_manifest is not None:
            mono = json.loads(self.mono_depth_manifest.read_text(encoding="utf-8"))
            contract["da2_depth"] = {
                "manifest_sha256": verify_mono_depth_manifest(mono),
                "valid_view_weight": self.da2_depth_weight,
                "invalid_alignment_policy": "skip_da2_loss_for_view",
            }
        if self.mesh_geometry_manifest is not None:
            mesh_geometry = json.loads(
                self.mesh_geometry_manifest.read_text(encoding="utf-8")
            )
            contract["mesh_geometry"] = {
                "manifest_sha256": verify_mesh_geometry_manifest(mesh_geometry),
                "depth_weight": self.mesh_depth_weight,
                "normal_weight": self.mesh_normal_weight,
                "depth_transform": "0.5*d_if_d_lt_1_else_1_minus_0.5_over_d",
                "source_separation": "not_fused_with_sparse_lidar_or_da2",
                "normal_consumer": (
                    "rendered_range_to_camera_normal_hemisphere_aligned_l1_proxy"
                ),
                "competitor_loss_schedule_enabled": (
                    self.competitor_loss_schedule_enabled
                ),
                "late_mesh_occlusion_filter": (
                    "rendered_range < mesh_range + 0.1 * scene_extent"
                    if self.competitor_loss_schedule_enabled
                    else None
                ),
            }
        if self.face_lidar_geometry_manifest is not None:
            lidar_geometry = json.loads(
                self.face_lidar_geometry_manifest.read_text(encoding="utf-8")
            )
            contract["face4_lidar_geometry"] = {
                "manifest_sha256": verify_face_lidar_geometry_manifest(
                    lidar_geometry
                ),
                "source_depth_manifest_sha256": lidar_geometry.get(
                    "source_depth_manifest_sha256"
                ),
                "sparse_metric_range_consumed_by_dataset": True,
                "mesh_interpolation": False,
            }
        if self.renderer_mask_manifest is not None:
            renderer_mask = json.loads(
                self.renderer_mask_manifest.read_text(encoding="utf-8")
            )
            contract["renderer_mask"] = {
                "manifest_sha256": verify_renderer_mask_manifest(renderer_mask),
                "manifest_consumed_by_dataset": True,
                "source_face_manifest_bound": True,
                "policy": renderer_mask.get("policy"),
            }
        return contract


def load_initialization_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the canonical binary little-endian XYZ/RGB PLY from PR-04."""
    path = Path(path)
    with path.open("rb") as stream:
        header: list[str] = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            line = raw.decode("ascii").rstrip("\r\n")
            header.append(line)
            if line == "end_header":
                break
        if header[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("initialization PLY must be binary_little_endian")
        vertex_lines = [line for line in header if line.startswith("element vertex ")]
        if len(vertex_lines) != 1:
            raise ValueError("initialization PLY must contain one vertex element")
        count = int(vertex_lines[0].split()[2])
        properties = [line for line in header if line.startswith("property ")]
        if properties != [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]:
            raise ValueError("initialization PLY properties do not match canonical XYZ/RGB")
        dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)], align=False)
        records = np.fromfile(stream, dtype=dtype, count=count)
        if len(records) != count or stream.read(1):
            raise ValueError("initialization PLY payload size does not match its header")
    xyz = np.asarray(records["xyz"], dtype=np.float32).copy()
    rgb = np.asarray(records["rgb"], dtype=np.uint8).copy()
    if not len(xyz) or not np.all(np.isfinite(xyz)):
        raise ValueError("initialization PLY contains no finite points")
    if float(np.max(np.abs(xyz))) > 100_000.0:
        raise ValueError("initialization PLY is not in a safe local coordinate frame")
    return xyz, rgb


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _append_monitor_event(path: Path, value: dict[str, Any]) -> None:
    """Append non-authoritative telemetry without touching checkpoint state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()


def _write_golden_history(
    path: Path,
    *,
    config: GoldenEvaluationConfig,
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist signed checkpoint-selection evidence independently of the model."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "configuration": config.to_dict(),
        "history": history,
        "best": best,
    }
    payload["golden_history_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    _atomic_json(path, payload)
    return payload


def _write_full_evaluation_history(
    path: Path,
    *,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "history": history,
    }
    payload["full_evaluation_history_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    _atomic_json(path, payload)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register_frozen_tail_hooks(params: dict[str, Any], start: int) -> int:
    """Zero every gradient row from ``start`` on, for all parameters.

    The hooks live on the Parameter objects, so they survive a warm start
    (which replaces ``.data`` on the same objects) and cost one in-place
    slice-fill per backward. Rows before ``start`` are untouched, and a
    parameter shorter than ``start`` passes through whole.
    """
    registered = 0
    for parameter in params.values():
        if not hasattr(parameter, "register_hook"):
            continue

        def _zero_tail(gradient, boundary=start):
            if len(gradient) > boundary:
                gradient[boundary:] = 0
            return gradient

        parameter.register_hook(_zero_tail)
        registered += 1
    return registered


def _warm_start_from_checkpoint(
    path: Path,
    *,
    expected_lineage: dict[str, Any],
    params: Any,
    auxiliary_params: dict[str, Any],
    fresh_auxiliary_names: tuple[str, ...] = (),
    scale_multiplier: float = 1.0,
    map_location: str,
) -> dict[str, Any]:
    """Load only model state from a compatible completed run.

    A warm start intentionally does not restore optimizer, sampler, strategy,
    evaluation history, or RNG state. It permits a new resolution/loss phase
    while requiring the same signed inputs, initialization, coordinate frame,
    and locked renderer.
    """
    import torch

    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported warm-start checkpoint schema")
    source_identity = payload.get("identity")
    if not isinstance(source_identity, dict):
        raise ValueError("warm-start checkpoint has no identity")
    nested_source_identity = source_identity.get("source_identity", {})
    if not isinstance(nested_source_identity, dict):
        raise ValueError("warm-start checkpoint source_identity is invalid")
    for key, expected in expected_lineage.items():
        actual = (
            source_identity[key]
            if key in source_identity
            else nested_source_identity.get(key)
        )
        if actual != expected:
            raise ValueError(f"warm-start checkpoint lineage mismatch for {key}")

    checkpoint_params = payload.get("params")
    if not isinstance(checkpoint_params, dict) or set(checkpoint_params) != set(params):
        raise ValueError("warm-start checkpoint parameter names do not match")
    topology_changed = False
    target_gaussian_count = len(next(iter(params.values())))
    source_gaussian_count = len(next(iter(checkpoint_params.values())))
    for name, parameter in params.items():
        value = checkpoint_params[name]
        if parameter.shape != value.shape:
            if parameter.ndim == 0 or parameter.shape[1:] != value.shape[1:]:
                raise ValueError(
                    f"warm-start checkpoint parameter shape mismatch for {name}"
                )
            topology_changed = True
        with torch.no_grad():
            # Fresh optimizers already reference this Parameter object. Replacing
            # its data (rather than the Parameter) permits a densified/pruned
            # topology to start a new phase without leaving optimizer groups
            # pointing at the old initialization row count.
            parameter.data = value.to(map_location).detach().clone()
    if not math.isfinite(float(scale_multiplier)) or scale_multiplier <= 0.0:
        raise ValueError("warm-start scale multiplier must be finite and positive")
    if scale_multiplier != 1.0:
        if "scales" not in params:
            raise ValueError("warm-start scale multiplier requires scales parameters")
        with torch.no_grad():
            params["scales"].add_(math.log(float(scale_multiplier)))

    checkpoint_auxiliary = payload.get("auxiliary_params", {})
    fresh_names = set(fresh_auxiliary_names)
    target_names = set(auxiliary_params)
    if not fresh_names <= target_names:
        raise ValueError("warm-start fresh auxiliary names are missing from target parameters")
    expected_source_names = target_names - fresh_names
    if (
        not isinstance(checkpoint_auxiliary, dict)
        or set(checkpoint_auxiliary) != expected_source_names
    ):
        raise ValueError("warm-start checkpoint auxiliary parameter names do not match")
    for name, parameter in auxiliary_params.items():
        if name in fresh_names:
            if not bool(torch.isfinite(parameter).all()) or bool(
                torch.count_nonzero(parameter).item()
            ):
                raise ValueError(
                    f"warm-start fresh auxiliary parameter {name} must be zero-initialized"
                )
            continue
        value = checkpoint_auxiliary[name]
        if parameter.shape != value.shape:
            raise ValueError(f"warm-start checkpoint auxiliary parameter shape mismatch for {name}")
        with torch.no_grad():
            parameter.copy_(value.to(map_location))

    return {
        "mode": "parameters_and_auxiliary_fresh_optimizers",
        "checkpoint_sha256": _sha256_file(path),
        "source_step": int(payload.get("step", 0)),
        "source_trainer_config_sha256": source_identity.get("trainer_config_sha256"),
        "fresh_auxiliary_names": sorted(fresh_names),
        "topology_changed": topology_changed,
        "target_gaussian_count_before_load": target_gaussian_count,
        "source_gaussian_count": source_gaussian_count,
        "scale_multiplier": float(scale_multiplier),
    }


def _retained_checkpoint_records(
    output_dir: Path,
    *,
    keep_every_steps: int,
    completed_steps: int,
) -> list[dict[str, Any]]:
    if keep_every_steps <= 0:
        return []
    records: list[dict[str, Any]] = []
    for step in range(keep_every_steps, completed_steps + 1, keep_every_steps):
        path = Path(output_dir) / "checkpoints" / f"step_{step:08d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"retained checkpoint is missing: {path}")
        records.append(
            {
                "step": step,
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256_file(path),
                "byte_count": path.stat().st_size,
            }
        )
    return records


def means_lr_for_step(
    base_learning_rate: float,
    final_factor: float,
    *,
    step: int,
    max_steps: int,
) -> float:
    """Australian parity schedule as a pure function for resume equivalence."""
    if base_learning_rate < 0.0:
        raise ValueError("base means learning rate must be non-negative")
    if not 0.0 < final_factor <= 1.0:
        raise ValueError("means LR final factor must be in (0, 1]")
    if step < 0 or max_steps <= 0:
        raise ValueError("means LR schedule requires non-negative step and max_steps")
    if base_learning_rate == 0.0:
        return 0.0
    return float(base_learning_rate * (final_factor ** (step / max(1, max_steps))))


def fisher_yates_epoch_order(
    count: int,
    *,
    seed: int,
    epoch: int,
) -> tuple[int, ...]:
    """Return one resume-stable, no-replacement Fisher-Yates permutation."""

    if count <= 0 or epoch < 0:
        raise ValueError("Fisher-Yates sampling requires positive count and epoch >= 0")
    # SeedSequence keeps each epoch independent while making the permutation a
    # pure function of (run seed, epoch).  A resumed run therefore does not
    # need a partially consumed permutation in its checkpoint.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    order = list(range(int(count)))
    for upper in range(count - 1, 0, -1):
        selected = int(rng.integers(0, upper + 1))
        order[upper], order[selected] = order[selected], order[upper]
    return tuple(order)


def active_sh_degree_for_step(config: TrainerConfig, step: int) -> int | None:
    """Australian progressive-SH schedule, derived solely from the step."""
    if step < 0:
        raise ValueError("SH schedule step must be non-negative")
    if config.color_model != "sh" or config.sh_degree_interval == 0:
        return None
    return min(config.sh_degree, step // config.sh_degree_interval)


def appearance_learning_rates(
    config: TrainerConfig, learning_rates: dict[str, float]
) -> dict[str, float]:
    """Map the configured RGB rate to Australian SH DC/rest optimizer rates."""
    effective = dict(learning_rates)
    if config.color_model == "sh":
        color_lr = float(effective.pop("colors"))
        effective["sh0"] = color_lr
        effective["shN"] = color_lr / 20.0
    return effective


def _tensor_sample(sample: TrainingSample, torch: Any, device: str) -> dict[str, Any]:
    result = {
        "rgb": torch.as_tensor(np.array(sample.image, copy=True), dtype=torch.float32, device=device) / 255.0,
        "rgb_mask": torch.as_tensor(np.array(sample.rgb_mask, copy=True), dtype=torch.bool, device=device),
    }
    if sample.depth_range_m is not None:
        result.update(
            {
                "range_m": torch.as_tensor(np.array(sample.depth_range_m, copy=True), dtype=torch.float32, device=device),
                "confidence": torch.as_tensor(np.array(sample.depth_confidence, copy=True), dtype=torch.float32, device=device),
                "depth_mask": torch.as_tensor(np.array(sample.depth_mask, copy=True), dtype=torch.bool, device=device),
            }
        )
    if sample.mono_depth_range_m is not None:
        result.update(
            {
                "da2_range_m": torch.as_tensor(
                    np.array(sample.mono_depth_range_m, copy=True),
                    dtype=torch.float32,
                    device=device,
                ),
                "da2_mask": torch.as_tensor(
                    np.array(sample.mono_depth_mask, copy=True),
                    dtype=torch.bool,
                    device=device,
                ),
            }
        )
    if sample.mesh_depth_range_m is not None:
        result.update(
            {
                "mesh_range_m": torch.as_tensor(
                    np.array(sample.mesh_depth_range_m, copy=True),
                    dtype=torch.float32,
                    device=device,
                ),
                "mesh_normal_camera": torch.as_tensor(
                    np.array(sample.mesh_normal_camera, copy=True),
                    dtype=torch.float32,
                    device=device,
                ),
                "mesh_confidence": torch.as_tensor(
                    np.array(sample.mesh_confidence, copy=True),
                    dtype=torch.float32,
                    device=device,
                ),
                "mesh_mask": torch.as_tensor(
                    np.array(sample.mesh_depth_mask, copy=True),
                    dtype=torch.bool,
                    device=device,
                ),
            }
        )
    return result


def _mipmap_compress_depth(torch: Any, depth: Any) -> Any:
    """Recovered distance compression used by the competitor depth terms."""
    safe = torch.clamp(depth, min=1e-8)
    return torch.where(safe < 1.0, 0.5 * safe, 1.0 - 0.5 / safe)


def _rendered_range_normals_camera(
    torch: Any, rendered_range: Any, sample: TrainingSample
) -> tuple[Any, Any]:
    """Recover differentiable camera-space normals from rendered ray range."""
    height, width = rendered_range.shape
    dtype = rendered_range.dtype
    device = rendered_range.device
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    K = torch.as_tensor(sample.K, dtype=dtype, device=device)
    ray = torch.stack(
        [
            (xx + 0.5 - K[0, 2]) / K[0, 0],
            (yy + 0.5 - K[1, 2]) / K[1, 1],
            torch.ones_like(xx),
        ],
        dim=-1,
    )
    ray = ray / torch.linalg.vector_norm(ray, dim=-1, keepdim=True).clamp_min(1e-8)
    points = ray * rendered_range[..., None]
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    interior = torch.linalg.cross(dx, dy, dim=-1)
    interior_norm = torch.linalg.vector_norm(interior, dim=-1, keepdim=True)
    interior = interior / interior_norm.clamp_min(1e-8)
    normals = torch.zeros_like(points)
    normals[1:-1, 1:-1] = interior
    valid_depth = torch.isfinite(rendered_range) & (rendered_range > 0.0)
    valid = torch.zeros_like(valid_depth)
    valid[1:-1, 1:-1] = (
        valid_depth[1:-1, 1:-1]
        & valid_depth[1:-1, 2:]
        & valid_depth[1:-1, :-2]
        & valid_depth[2:, 1:-1]
        & valid_depth[:-2, 1:-1]
        & (interior_norm[..., 0] > 1e-8)
    )
    return normals, valid


_VIEW_BACKGROUND_LIBRARIES: dict[tuple, Any] = {}


def _view_background_library(config: "TrainerConfig") -> Any | None:
    """Lazy per-(manifest, device) library shared by every loss call."""
    if config.background_image_manifest is None:
        return None
    key = (
        str(config.background_image_manifest),
        str(config.background_image_root),
        config.device,
    )
    library = _VIEW_BACKGROUND_LIBRARIES.get(key)
    if library is None:
        from cloudstudio_3dgs.training.view_backgrounds import ViewBackgroundLibrary

        library = ViewBackgroundLibrary(
            config.background_image_manifest,
            config.background_image_root,
            device=config.device,
        )
        _VIEW_BACKGROUND_LIBRARIES[key] = library
    return library


def _render_supervision_loss(
    *,
    backend: GsplatBackend,
    params: Any,
    sample: TrainingSample,
    tensors: dict[str, Any],
    config: TrainerConfig,
    c2w_override: Any | None = None,
    rgb_gain: Any | None = None,
    ppisp: Any | None = None,
    active_sh_degree: int | None = None,
    range_weight_scale: float = 1.0,
    da2_depth_weight_override: float | None = None,
    mesh_depth_weight_override: float | None = None,
    mesh_normal_weight_override: float | None = None,
    mesh_late_occlusion_scene_scale_m: float | None = None,
    background_library: Any | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    has_range = "range_m" in tensors and (
        config.lidar_range_weight > 0.0 or config.lidar_linear_aux_weight > 0.0
    )
    da2_weight = (
        config.da2_depth_weight
        if da2_depth_weight_override is None
        else float(da2_depth_weight_override)
    )
    mesh_depth_weight = (
        config.mesh_depth_weight
        if mesh_depth_weight_override is None
        else float(mesh_depth_weight_override)
    )
    mesh_normal_weight = (
        config.mesh_normal_weight
        if mesh_normal_weight_override is None
        else float(mesh_normal_weight_override)
    )
    has_da2 = "da2_range_m" in tensors and da2_weight > 0.0
    has_mesh_depth = "mesh_range_m" in tensors and mesh_depth_weight > 0.0
    has_mesh_normal = "mesh_normal_camera" in tensors and mesh_normal_weight > 0.0
    if background_library is None:
        background_library = _view_background_library(config)
    if background_library is not None:
        target_shape = tensors["rgb"].shape
        composite_background = background_library.background_for(
            sample.image_id,
            height=int(target_shape[0]),
            width=int(target_shape[1]),
            torch=backend.torch,
        )
    else:
        composite_background = config.background_color
    rendered, rendered_range, rendered_alpha, info = backend.render(
        params,
        sample,
        with_range=has_range or has_da2 or has_mesh_depth or has_mesh_normal,
        c2w_override=c2w_override,
        active_sh_degree=active_sh_degree,
        background_rgb=composite_background,
    )
    raw_rendered = rendered
    if ppisp is not None:
        # Per-camera ISP: exposure -> vignetting (on ORIGINAL sensor coords
        # for warped faces) -> color, applied after background compositing and
        # before the photometric losses. Validation renders stay at identity.
        coords = getattr(sample, "sensor_pixel_coords", None)
        pixel_coords = None
        if coords is not None:
            pixel_coords = ppisp.exposure_params.new_tensor(coords)
        rendered = ppisp.apply(
            rendered,
            sample.image_id.split("::")[0],
            pixel_coords=pixel_coords,
            resolution=getattr(sample, "sensor_resolution", None),
        )
    if rgb_gain is not None:
        # Per-frame auto-exposure compensation: scale the render toward the
        # frame's brightness for the photometric losses only. Geometry (range)
        # and validation renders stay at gain 1.0.
        rendered = rendered * rgb_gain
    l1 = masked_rgb_l1(rendered, tensors["rgb"], tensors["rgb_mask"])
    with backend.torch.no_grad():
        # Masked train-view PSNR, on the same pixels the L1 sees. One view is
        # noisy; the dashboard reads the trend, and it is the unit every
        # published comparison speaks in.
        masked_mse = (
            ((rendered - tensors["rgb"]) ** 2)
            .mean(dim=-1)[tensors["rgb_mask"]]
            .mean()
        )
        rgb_psnr = float(
            (-10.0 * backend.torch.log10(masked_mse.clamp_min(1e-10)))
            .detach()
            .cpu()
        )
    rgb_gradient_l1 = None
    if config.rgb_gradient_weight > 0.0:
        rgb_gradient_l1 = masked_rgb_gradient_l1(
            rendered, tensors["rgb"], tensors["rgb_mask"]
        )
    lidar_rgb_l1 = None
    if config.lidar_rgb_l1_weight > 0.0:
        if "depth_mask" not in tensors:
            raise ValueError("LiDAR-supported RGB loss requires depth supervision")
        lidar_rgb_mask = tensors["depth_mask"]
        if config.lidar_rgb_dilation_radius_px > 0:
            radius = config.lidar_rgb_dilation_radius_px
            lidar_rgb_mask = backend.torch.nn.functional.max_pool2d(
                lidar_rgb_mask[None, None].to(dtype=rendered.dtype),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[0, 0].to(dtype=backend.torch.bool)
        lidar_rgb_mask = tensors["rgb_mask"] & lidar_rgb_mask
        if bool(lidar_rgb_mask.any()):
            lidar_rgb_l1 = masked_rgb_l1(
                rendered, tensors["rgb"], lidar_rgb_mask
            )
        else:
            lidar_rgb_l1 = rendered.new_zeros(())
    if config.rgb_ssim_weight <= 0.0:
        ssim = rendered.new_zeros(())
    elif config.rgb_ssim_mode == "local_gaussian":
        decouple = config.decoupled_ssim and rgb_gain is not None
        ssim = masked_rgb_ssim_loss(
            # Decoupled: structure/contrast compares the RAW render so the
            # exposure gain can move brightness but never mask structure.
            raw_rendered if decouple else rendered,
            tensors["rgb"],
            tensors["rgb_mask"],
            window_size=config.ssim_window_size,
            sigma=config.ssim_sigma,
            min_valid_fraction=config.ssim_min_valid_fraction,
            luminance_gain=rgb_gain if decouple else None,
        )
    else:
        ssim = global_masked_rgb_ssim_loss(
            rendered, tensors["rgb"], tensors["rgb_mask"]
        )
    loss = config.rgb_l1_weight * l1 + config.rgb_ssim_weight * ssim
    if rgb_gradient_l1 is not None:
        loss = loss + config.rgb_gradient_weight * rgb_gradient_l1
    if lidar_rgb_l1 is not None:
        loss = loss + config.lidar_rgb_l1_weight * lidar_rgb_l1
    lidar_alpha_loss = None
    lidar_alpha_support_fraction = None
    if config.lidar_alpha_weight > 0.0:
        if "depth_mask" not in tensors or "confidence" not in tensors:
            raise ValueError("LiDAR alpha coverage requires depth supervision")
        if rendered_alpha is None:
            raise RuntimeError("rasterizer did not return alpha coverage")
        lidar_alpha_valid = (
            tensors["depth_mask"]
            & backend.torch.isfinite(tensors["confidence"])
            & (tensors["confidence"] > 0.0)
        )
        lidar_alpha_confidence = backend.torch.where(
            lidar_alpha_valid,
            tensors["confidence"],
            backend.torch.zeros_like(tensors["confidence"]),
        )
        if config.lidar_alpha_dilation_radius_px > 0:
            radius = config.lidar_alpha_dilation_radius_px
            lidar_alpha_confidence = backend.torch.nn.functional.max_pool2d(
                lidar_alpha_confidence[None, None],
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[0, 0]
            lidar_alpha_valid = lidar_alpha_confidence > 0.0
        lidar_alpha_mask = tensors["rgb_mask"] & lidar_alpha_valid
        lidar_alpha_support_fraction = lidar_alpha_mask.to(
            dtype=rendered.dtype
        ).mean()
        if bool(lidar_alpha_mask.any()):
            confidence = lidar_alpha_confidence[lidar_alpha_mask]
            deficit = backend.torch.relu(
                float(config.lidar_alpha_target)
                - rendered_alpha[lidar_alpha_mask]
            )
            lidar_alpha_loss = (
                confidence * deficit.square()
            ).sum() / confidence.sum().clamp_min(1e-8)
        else:
            lidar_alpha_loss = rendered.new_zeros(())
        loss = loss + config.lidar_alpha_weight * lidar_alpha_loss
    range_loss = None
    linear_range_aux_loss = None
    if has_range and getattr(sample, "camera_model", "fisheye") == "pinhole":
        # A face legitimately may catch only a handful of LiDAR rays, and the
        # edge-weight gate can empty the intersection entirely; skip the range
        # term for such samples instead of tripping the fail-closed check that
        # protects the dense fisheye path.
        torch_mod = backend.torch
        supervised = (
            tensors["depth_mask"]
            & torch_mod.isfinite(tensors["range_m"])
            & (tensors["range_m"] > 0.0)
            & (tensors["confidence"] > 0.0)
        )
        if not bool(supervised.any()):
            has_range = False
    if has_range or has_da2 or has_mesh_depth or has_mesh_normal:
        rendered_range = rendered_range_to_euclidean(
            backend.torch, rendered_range, sample, info
        )
    if has_range:
        assert rendered_range is not None
        if config.lidar_range_loss_mode == "linear_l1":
            range_loss = confidence_weighted_range_l1(
                rendered_range,
                tensors["range_m"],
                tensors["confidence"],
                tensors["depth_mask"],
            )
        else:
            range_loss = confidence_weighted_log_range_huber(
                rendered_range,
                tensors["range_m"],
                tensors["confidence"],
                tensors["depth_mask"],
                delta=config.lidar_log_range_huber_delta,
            )
        loss = loss + float(range_weight_scale) * config.lidar_range_weight * range_loss
        if config.lidar_linear_aux_weight > 0.0:
            linear_range_aux_loss = confidence_weighted_range_l1(
                rendered_range,
                tensors["range_m"],
                tensors["confidence"],
                tensors["depth_mask"],
            )
            loss = (
                loss
                + float(range_weight_scale)
                * config.lidar_linear_aux_weight
                * linear_range_aux_loss
            )
    da2_depth_loss = None
    if has_da2:
        assert rendered_range is not None
        da2_valid = (
            tensors["da2_mask"]
            & backend.torch.isfinite(tensors["da2_range_m"])
            & (tensors["da2_range_m"] > 0.0)
        )
        if bool(da2_valid.any()):
            da2_depth_loss = confidence_weighted_range_l1(
                rendered_range,
                tensors["da2_range_m"],
                backend.torch.ones_like(tensors["da2_range_m"]),
                da2_valid,
            )
            loss = loss + da2_weight * da2_depth_loss
    info["cloudstudio_da2_depth_loss"] = da2_depth_loss
    mesh_depth_loss = None
    if has_mesh_depth:
        assert rendered_range is not None
        mesh_valid = (
            tensors["mesh_mask"]
            & backend.torch.isfinite(tensors["mesh_range_m"])
            & (tensors["mesh_range_m"] > 0.0)
            & backend.torch.isfinite(tensors["mesh_confidence"])
            & (tensors["mesh_confidence"] > 0.0)
        )
        if mesh_late_occlusion_scene_scale_m is not None:
            mesh_valid &= rendered_range < (
                tensors["mesh_range_m"]
                + 0.1 * float(mesh_late_occlusion_scene_scale_m)
            )
        if bool(mesh_valid.any()):
            prediction = _mipmap_compress_depth(backend.torch, rendered_range)
            target = _mipmap_compress_depth(
                backend.torch, tensors["mesh_range_m"]
            )
            confidence = tensors["mesh_confidence"][mesh_valid]
            mesh_depth_loss = (
                confidence * backend.torch.abs(prediction[mesh_valid] - target[mesh_valid])
            ).sum() / confidence.sum().clamp_min(1e-8)
            loss = loss + mesh_depth_weight * mesh_depth_loss
    info["cloudstudio_mesh_depth_loss"] = mesh_depth_loss
    mesh_normal_loss = None
    if has_mesh_normal:
        assert rendered_range is not None
        rendered_normal, rendered_normal_valid = _rendered_range_normals_camera(
            backend.torch, rendered_range, sample
        )
        target_normal = tensors["mesh_normal_camera"]
        target_norm = backend.torch.linalg.vector_norm(
            target_normal, dim=-1, keepdim=True
        )
        target_normal = target_normal / target_norm.clamp_min(1e-8)
        normal_valid = (
            tensors["mesh_mask"]
            & rendered_normal_valid
            & backend.torch.isfinite(target_normal).all(dim=-1)
            & (target_norm[..., 0] > 1e-8)
        )
        if bool(normal_valid.any()):
            predicted = rendered_normal[normal_valid]
            target = target_normal[normal_valid]
            direct = backend.torch.abs(predicted - target).mean(dim=-1)
            flipped = backend.torch.abs(predicted + target).mean(dim=-1)
            hemisphere_l1 = backend.torch.minimum(direct, flipped)
            confidence = tensors["mesh_confidence"][normal_valid]
            mesh_normal_loss = (
                confidence * hemisphere_l1
            ).sum() / confidence.sum().clamp_min(1e-8)
            loss = loss + mesh_normal_weight * mesh_normal_loss
    info["cloudstudio_mesh_normal_loss"] = mesh_normal_loss
    info["cloudstudio_linear_range_aux_loss"] = (
        None if linear_range_aux_loss is None else linear_range_aux_loss.detach()
    )
    info["cloudstudio_lidar_rgb_l1"] = (
        lidar_rgb_l1
    )
    info["cloudstudio_rgb_gradient_l1"] = rgb_gradient_l1
    info["cloudstudio_rgb_psnr"] = rgb_psnr
    info["cloudstudio_lidar_rgb_support_fraction"] = (
        None
        if lidar_rgb_l1 is None
        else lidar_rgb_mask.to(dtype=rendered.dtype).mean().detach()
    )
    info["cloudstudio_lidar_alpha_loss"] = lidar_alpha_loss
    info["cloudstudio_lidar_alpha_support_fraction"] = (
        lidar_alpha_support_fraction
    )
    info["cloudstudio_rendered_range_tensor"] = rendered_range
    # Stashed for the error-weighted MCMC score update; detached, so it never
    # extends the autograd graph.
    info["cloudstudio_rendered_rgb"] = rendered.detach()
    return loss, l1, ssim, range_loss, info


def _range_directionality_audit(
    *,
    rendered_range: Any,
    tensors: dict[str, Any],
    config: TrainerConfig,
    perturbation_m: float = 0.01,
) -> dict[str, Any]:
    """Check that a range prediction moved toward LiDAR lowers the loss."""

    prediction = rendered_range.detach()
    target = tensors["range_m"]
    confidence = tensors["confidence"]
    mask = tensors["depth_mask"]
    residual = prediction - target
    direction = residual.sign()
    direction = direction.where(direction != 0.0, direction.new_ones(()))
    toward = prediction - direction * residual.abs().clamp_max(float(perturbation_m))
    away = prediction + direction * float(perturbation_m)

    def evaluate(value: Any) -> Any:
        if config.lidar_range_loss_mode == "linear_l1":
            return confidence_weighted_range_l1(value, target, confidence, mask)
        return confidence_weighted_log_range_huber(
            value,
            target,
            confidence,
            mask,
            delta=config.lidar_log_range_huber_delta,
        )

    baseline = float(evaluate(prediction).cpu())
    toward_loss = float(evaluate(toward).cpu())
    away_loss = float(evaluate(away).cpu())
    tolerance = 1e-10
    return {
        "perturbation_m": float(perturbation_m),
        "baseline_loss": baseline,
        "toward_surface_loss": toward_loss,
        "away_from_surface_loss": away_loss,
        "toward_nonincreasing": toward_loss <= baseline + tolerance,
        "away_nondecreasing": away_loss + tolerance >= baseline,
        "directional_pass": (
            toward_loss <= baseline + tolerance
            and away_loss + tolerance >= baseline
            and away_loss > toward_loss + tolerance
        ),
    }


def _compare_pose_candidate(
    *,
    backend: GsplatBackend,
    params: Any,
    dataset: S1TrainingDataset,
    refiner: RigPoseRefiner,
    config: TrainerConfig,
) -> tuple[float, float, int]:
    """Compare original and candidate poses on the same frozen model and samples."""
    torch = backend.torch
    before: list[Any] = []
    after: list[Any] = []
    indices = dataset.indices_for_rig_frames(
        config.rig_pose_refinement.evaluation_rig_frames
    )
    with torch.no_grad():
        for index in indices:
            sample = dataset[index]
            tensors = _tensor_sample(sample, torch, config.device)
            original_loss, _, _, _, _ = _render_supervision_loss(
                backend=backend,
                params=params,
                sample=sample,
                tensors=tensors,
                config=config,
            )
            refined_loss, _, _, _, _ = _render_supervision_loss(
                backend=backend,
                params=params,
                sample=sample,
                tensors=tensors,
                config=config,
                c2w_override=refiner.apply(sample.rig_frame_id, sample.c2w),
            )
            before.append(original_loss.detach())
            after.append(refined_loss.detach())
    if not before:
        raise ValueError("pose refinement comparison selected no training images")
    return (
        float(torch.stack(before).mean().cpu()),
        float(torch.stack(after).mean().cpu()),
        len(indices),
    )


def _save_evaluation_artifacts(
    *,
    backend: GsplatBackend,
    params: Any,
    dataset: S1TrainingDataset,
    output_dir: Path,
    background_rgb: Any | None = None,
) -> list[dict[str, Any]]:
    torch = backend.torch
    frames: list[dict[str, Any]] = []
    artifact_root = output_dir / "evaluation"
    artifact_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            # Face4 sample identities contain ``::``. Keep that signed identity
            # in memory, but make the artifact basename legal on Windows.
            artifact_id = sample.image_id.replace(":", "_")
            has_range = sample.depth_range_m is not None
            render_options = {"with_range": has_range}
            if background_rgb is not None:
                render_options["background_rgb"] = background_rgb
            rendered, rendered_range, rendered_alpha, render_info = backend.render(
                params, sample, **render_options
            )
            rendered_range = rendered_range_to_euclidean(
                backend.torch, rendered_range, sample, render_info
            )
            prefix = artifact_root / artifact_id
            reference_path = prefix.with_name(f"{artifact_id}_reference.png")
            render_path = prefix.with_name(f"{artifact_id}_rendered.png")
            mask_path = prefix.with_name(f"{artifact_id}_mask.png")
            alpha_path = prefix.with_name(f"{artifact_id}_alpha.npy")
            alpha_preview_path = prefix.with_name(f"{artifact_id}_alpha.png")
            Image.fromarray(sample.image).save(reference_path, format="PNG", optimize=False)
            rendered_u8 = (
                rendered.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
            )
            Image.fromarray(rendered_u8).save(render_path, format="PNG", optimize=False)
            Image.fromarray(sample.rgb_mask.astype(np.uint8) * 255).save(
                mask_path, format="PNG", optimize=False
            )
            alpha = rendered_alpha.detach().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            np.save(alpha_path, alpha)
            Image.fromarray(np.rint(alpha * 255.0).astype(np.uint8)).save(
                alpha_preview_path, format="PNG", optimize=False
            )
            frame: dict[str, Any] = {
                "image_id": sample.image_id,
                "split": "val",
                "reference_rgb_path": reference_path.relative_to(output_dir).as_posix(),
                "rendered_rgb_path": render_path.relative_to(output_dir).as_posix(),
                "combined_mask_path": mask_path.relative_to(output_dir).as_posix(),
                "rendered_alpha_path": alpha_path.relative_to(output_dir).as_posix(),
                "rendered_alpha_preview_path": alpha_preview_path.relative_to(
                    output_dir
                ).as_posix(),
                "rendered_alpha_semantics": "accumulated_foreground_opacity_0_to_1",
            }
            if has_range:
                assert rendered_range is not None and sample.depth_cache_path is not None
                rendered_depth_path = prefix.with_name(f"{artifact_id}_range.npy")
                lidar_path = prefix.with_name(f"{artifact_id}_lidar.npz")
                np.save(rendered_depth_path, rendered_range.detach().cpu().numpy().astype(np.float32))
                # Save the factor/crop-adjusted supervision the trainer actually
                # consumed, not the raw full-resolution cache: the quality
                # report compares it against the rendered range at render
                # resolution, and a 2912-basis cache next to a 728-basis
                # render fails its shape gate (first exposed by the real ukgs
                # run; the factor-1 synthetic fixture made the raw copy look
                # correct). Source indexes and support counts do not survive
                # the resample, so they carry sentinels and metrics ignore them.
                assert (
                    sample.depth_range_m is not None
                    and sample.depth_confidence is not None
                    and sample.depth_mask is not None
                )
                supervision_valid = (
                    sample.depth_mask
                    & np.isfinite(sample.depth_range_m)
                    & (sample.depth_range_m > 0.0)
                    & np.isfinite(sample.depth_confidence)
                    & (sample.depth_confidence > 0.0)
                )
                flat_index = np.flatnonzero(supervision_valid.reshape(-1)).astype(np.int32)
                adjusted = SparseDepthMap(
                    (int(sample.height), int(sample.width)),
                    flat_index,
                    sample.depth_range_m.reshape(-1)[flat_index].astype(np.float32),
                    np.clip(
                        sample.depth_confidence.reshape(-1)[flat_index].astype(np.float32),
                        1e-6,
                        1.0,
                    ),
                    np.full(len(flat_index), -1, dtype=np.int64),
                    np.zeros(len(flat_index), dtype=np.int32),
                )
                lidar_path.write_bytes(sparse_depth_npz_bytes(adjusted))
                frame.update(
                    {
                        "rendered_depth_path": rendered_depth_path.relative_to(output_dir).as_posix(),
                        "rendered_depth_semantics": "euclidean_ray_range_m",
                        "lidar_depth_cache_path": lidar_path.relative_to(output_dir).as_posix(),
                        "lidar_depth_cache_semantics": (
                            "factor_crop_mask_adjusted_euclidean_ray_range_m"
                        ),
                        "lidar_depth_valid_pixels": int(len(flat_index)),
                    }
                )
            frames.append(frame)
    return frames


def train(
    config: TrainerConfig,
    *,
    backend_factory: Any = GsplatBackend,
    controlled_stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Train and write a signed run manifest. No Viewer is created or imported."""
    config.validate()
    import torch

    if controlled_stop_after_steps is None:
        controlled_stop_after_steps = config.controlled_stop_after_steps
    if controlled_stop_after_steps is not None:
        controlled_stop_after_steps = int(controlled_stop_after_steps)
        if not 0 < controlled_stop_after_steps < config.max_steps:
            raise ValueError(
                "controlled_stop_after_steps must be between zero and max_steps"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 3DGUT gsplat trainer")
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and config.resume_checkpoint is None:
        raise FileExistsError(f"training output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_views = None
    if config.tile_inputs_manifest is not None:
        tile_inputs = json.loads(
            config.tile_inputs_manifest.read_text(encoding="utf-8")
        )
        selected_tiles = [
            tile
            for tile in tile_inputs["tiles"]
            if int(tile["tile_id"]) == int(config.mipmap_tile_id)
        ]
        if len(selected_tiles) != 1:
            raise ValueError("Tile inputs do not contain a unique selected Tile")
        tile_views = selected_tiles[0]["views"]

    if config.face_cache_manifest is not None:
        # Fisheye face-split training: supervision comes from pre-warped
        # zero-distortion pinhole faces at full source resolution, bypassing
        # the wide-FoV batch-state ceiling. Validation stays on the raw
        # fisheye path below so metrics remain comparable across presets.
        from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset

        if config.face_cache_root is None:
            raise ValueError("face_cache_root is required with face_cache_manifest")
        trainset = FaceCacheDataset(
            face_manifest_path=config.face_cache_manifest,
            cache_root=config.face_cache_root,
            dataset_manifest_path=config.dataset_manifest,
            tile_views=tile_views,
            renderer_mask_manifest_path=config.renderer_mask_manifest,
            mono_depth_manifest_path=config.mono_depth_manifest,
            mono_depth_root=config.mono_depth_root,
            face_lidar_geometry_manifest_path=(
                config.face_lidar_geometry_manifest
            ),
            face_lidar_geometry_root=config.face_lidar_geometry_root,
            mesh_geometry_manifest_path=config.mesh_geometry_manifest,
            mesh_geometry_root=config.mesh_geometry_root,
        )
    else:
        trainset = S1TrainingDataset(
            dataset_manifest_path=config.dataset_manifest,
            recording_root=config.recording_root,
            mask_manifest_path=config.mask_manifest,
            mask_root=config.mask_root,
            split_manifest_path=config.split_manifest,
            split="train",
            person_mask_manifest_path=config.person_mask_manifest,
            person_mask_root=config.person_mask_root,
            depth_manifest_path=config.depth_manifest,
            depth_root=config.depth_root,
            factor=config.factor,
            crop=config.crop,
        )
    valset = S1TrainingDataset(
        dataset_manifest_path=config.dataset_manifest,
        recording_root=config.recording_root,
        mask_manifest_path=config.mask_manifest,
        mask_root=config.mask_root,
        split_manifest_path=config.split_manifest,
        split="val",
        person_mask_manifest_path=config.person_mask_manifest,
        person_mask_root=config.person_mask_root,
        depth_manifest_path=config.depth_manifest,
        depth_root=config.depth_root,
        factor=config.factor,
        crop=config.crop,
    )
    train_dataset_sha = getattr(trainset, "dataset_sha256", None)
    if train_dataset_sha is not None and train_dataset_sha != valset.dataset_sha256:
        raise ValueError("train and validation datasets have different identities")
    if (
        config.view_sampling_mode
        == "fisher_yates_without_replacement_per_epoch"
        and config.topology_policy.mode == "adaptive_growth"
        and config.max_steps != 20 * len(trainset)
    ):
        raise ValueError(
            "MipMap epoch-permutation sampling requires exactly 20 complete view epochs"
        )
    if config.competitor_loss_schedule_enabled:
        if config.view_sampling_mode != "fisher_yates_without_replacement_per_epoch":
            raise ValueError("competitor loss schedule requires epoch permutations")
        if config.max_steps != 20 * len(trainset):
            raise ValueError("competitor loss schedule requires exactly 20 view epochs")
    coordinate = build_coordinate_transform_manifest(trainset.dataset_sha256)
    _atomic_json(output_dir / "coordinate_transform_manifest.json", coordinate)
    contract = config.contract_dict()
    config_sha256 = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    initialization_sha256 = _sha256_file(config.initialization_ply)
    xyz, rgb = load_initialization_ply(config.initialization_ply)
    geometry_tree = None
    if (
        config.golden_evaluation.max_floater_growth_ratio is not None
        or config.golden_evaluation.max_floater_count is not None
    ):
        # Shared metric reference for the checkpoint-selection floater guard.
        from scipy.spatial import cKDTree

        geometry_tree = cKDTree(np.asarray(xyz, dtype=np.float64))
    surface_field = None
    admission = None
    proposal = None
    if config.lidar_admission.enabled or config.tangent_proposal.enabled:
        # One continuous surface field (KNN-PCA normals, planarity, roughness,
        # spacing, confidence; ~1.2-2.2s for 390k points) shared by every
        # consumer: admission biases *which parent* densification clones, the
        # proposal decides *where the child lands*, and the normal prior reads
        # the same normals. Building a second field would double the cost and,
        # worse, let the three disagree about where the surface is.
        from cloudstudio_3dgs.geometry.lidar_surface_field import build_surface_field

        surface_field = build_surface_field(xyz)
    if config.lidar_admission.enabled:
        admission = LidarAdmission(surface_field, config.lidar_admission)
    if config.tangent_proposal.enabled:
        # Seeded off the run seed so the tangential scatter is reproducible
        # without coupling to torch's global RNG stream.
        proposal = TangentProposal(
            surface_field, config.tangent_proposal, seed=int(config.seed)
        )
    normal_anchors = None
    if config.lidar_normal_alignment.enabled:
        # LiDAR-normal geometry prior: KNN-PCA normals + planarity over the
        # metric initialization cloud (~1.3s for 390k points), anchoring each
        # gaussian's shortest axis to the measured surface orientation.
        if surface_field is not None and config.lidar_admission.share_normal_field:
            # The surface field already computed the same KNN-PCA with the same
            # planarity definition; adopt it (and its cKDTree) instead of
            # building a second one. Note this carries the surface field's knn
            # over to the alignment loss - see AdmissionConfig.share_normal_field.
            normal_field = normal_field_from_surface_field(surface_field)
        else:
            normal_field = build_normal_field(xyz)
        normal_anchors = LidarNormalAnchors(
            normal_field, config.lidar_normal_alignment
        )
    if len(xyz) >= config.cap_max and config.warm_start_checkpoint is None:
        # A warm start replaces the initialization population wholesale, so
        # the capacity budget binds the warm-started row count (validated
        # below once the checkpoint is applied), not the signed PLY's.
        raise ValueError(
            f"initialization has {len(xyz)} Gaussians but cap_max is {config.cap_max}"
        )
    exact_geometry: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
    if (
        config.surface_initialization.enabled
        and config.surface_initialization.mode == "mipmap_k7_k30"
    ):
        if config.initialization_geometry is None:
            raise ValueError("MipMap initialization geometry is unavailable")
        exact_geometry = load_mipmap_tile_geometry(
            config.initialization_geometry_manifest,
            config.initialization_geometry_manifest.parent,
            tile_id=int(config.mipmap_tile_id),
            expected_initialization_ply_sha256=initialization_sha256,
            expected_count=len(xyz),
        )
    initial_scales_m, scale_calibration = build_metric_scale_calibration(
        xyz,
        policy=config.metric_scale_calibration,
        fixed_scale_m=config.init_scale_m,
        configured_means_lr=float(config.learning_rates["means"]),
        configured_noise_lr=config.mcmc_noise_lr,
        precomputed_scales_m=(
            None if exact_geometry is None else exact_geometry[0]
        ),
    )
    initial_quaternions = None
    surface_initialization_report: dict[str, Any] = {"enabled": False}
    initialization_geometry_sha256 = None
    if config.surface_initialization.enabled:
        if config.initialization_geometry is None:
            raise ValueError("surface initialization geometry is unavailable")
        initialization_geometry_sha256 = _sha256_file(
            config.initialization_geometry
        )
        if exact_geometry is not None:
            exact_scales, exact_quats, exact_geometry_report = exact_geometry
            initial_scales_m = exact_scales
            initial_quaternions = exact_quats
            surface_initialization_report = {
                "schema_version": 1,
                "configuration": config.surface_initialization.to_dict(),
                "surface_aligned_count": int(len(xyz)),
                "surface_aligned_fraction": 1.0,
                "tangent_scale_median": float(np.median(exact_scales[:, 0])),
                "normal_scale_median": float(np.median(exact_scales[:, 2])),
                **exact_geometry_report,
            }
        else:
            geometry_normals, geometry_eigenvalues = load_initialization_geometry(
                config.initialization_geometry,
                expected_count=len(xyz),
            )
            (
                initial_scales_m,
                initial_quaternions,
                surface_initialization_report,
            ) = build_surface_aligned_initialization(
                initial_scales_m,
                geometry_normals,
                geometry_eigenvalues,
                config=config.surface_initialization,
            )
        initial_scales_m, scale_cap_report = cap_surface_initialization_scales(
            initial_scales_m,
            maximum_scale_m=config.surface_initialization.maximum_scale_m,
        )
        surface_initialization_report["scale_cap"] = scale_cap_report
        surface_initialization_report["geometry_sha256"] = (
            initialization_geometry_sha256
        )
        surface_unsigned = dict(surface_initialization_report)
        surface_initialization_report["surface_initialization_sha256"] = (
            hashlib.sha256(canonical_json_bytes(surface_unsigned)).hexdigest()
        )
        _atomic_json(
            output_dir / "surface_initialization_report.json",
            surface_initialization_report,
        )
    effective_learning_rates = appearance_learning_rates(
        config, config.learning_rates
    )
    reference_scale_m = float(scale_calibration["reference_scale_m"])
    # Scene EXTENT, distinct from reference_scale_m (median Gaussian size).
    # Upstream's scale gates are fractions of this, so conflating the two moves
    # every threshold by three orders of magnitude. p95 rather than max so a
    # handful of distant returns cannot inflate it.
    scene_extent_m = float(
        np.percentile(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), 95)
    )
    effective_learning_rates["means"] = float(scale_calibration["effective_means_lr_m"])
    backend = backend_factory(
        device=config.device,
        cap_max=config.cap_max,
        lock_path=config.gsplat_lock,
        mcmc_config={
            "refine_start_iter": config.mcmc_refine_start_iter,
            "refine_stop_iter": config.mcmc_refine_stop_iter,
            "refine_every": config.mcmc_refine_every,
            "noise_injection_stop_iter": config.mcmc_noise_injection_stop_iter,
            "noise_lr": float(scale_calibration["effective_noise_lr"]),
            "growth_rate": config.mcmc_growth_rate,
            "relocation_max_fraction": config.mcmc_relocation_max_fraction,
            "relocation_max_scale_m": config.mcmc_relocation_max_scale_m,
            "relocation_max_anisotropy": config.mcmc_relocation_max_anisotropy,
        },
        error_score_config=config.error_weighted_sampling,
        densification_strategy=config.densification_strategy,
        default_strategy_config={
            **config.default_strategy,
            # Upstream normalises its scale gates by the SCENE extent, not by
            # Gaussian size. Passing reference_scale_m here - the median initial
            # Gaussian scale, ~0.058 m - made the split gate 0.58 mm and the
            # prune gate 5.8 mm against 1-8 cm Gaussians, so everything split,
            # nothing cloned, and the arm read as evidence against the method.
            # Configs should set split_scale_m / prune_scale_m instead and let
            # the adapter convert; this remains the denominator for that.
            "scene_scale": scene_extent_m,
        },
    )
    classic_densification = config.densification_strategy == "default_3dgs"
    if admission is not None and not classic_densification:
        # The strategy consults this only inside _relocate_gs / _add_new_gs and
        # falls back to pure error weighting whenever it is stale.
        backend.strategy.admission_state = admission
    if proposal is not None and not classic_densification:
        # Read only inside _add_new_gs, and only for the rows it appends.
        backend.strategy.proposal_state = proposal
    if proposal is not None and classic_densification:
        # The LiDAR-first route keeps the recovered gradient split/clone/cull
        # lifecycle, but candidate parents and newborn positions are now
        # checked by the measured surface field instead of being free 3-D
        # perturbations.
        backend.strategy.surface_birth_proposal = proposal
    backend.pinhole_rasterize_mode = config.pinhole_rasterize_mode
    backend.pinhole_with_ut = config.pinhole_with_ut
    runtime_contract = {
        key: backend.runtime.get(key)
        for key in ("package", "version", "locked_commit", "source_kind", "commit", "wheel_sha256")
        if backend.runtime.get(key) is not None
    }
    checkpoint_identity = {
        **trainset.identity,
        "coordinate_transform_sha256": coordinate["coordinate_transform_sha256"],
        "trainer_config_sha256": config_sha256,
        "initialization_ply_sha256": initialization_sha256,
        "scale_calibration_sha256": scale_calibration["scale_calibration_sha256"],
        "gsplat_runtime": runtime_contract,
    }
    if initialization_geometry_sha256 is not None:
        checkpoint_identity["initialization_geometry_sha256"] = (
            initialization_geometry_sha256
        )
        checkpoint_identity["surface_initialization_sha256"] = (
            surface_initialization_report["surface_initialization_sha256"]
        )
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(config.seed)
    params, optimizers, strategy_state = backend.initialize(
        xyz,
        rgb,
        init_scales_m=initial_scales_m,
        init_quaternions=initial_quaternions,
        learning_rates=effective_learning_rates,
        color_model=config.color_model,
        sh_degree=config.sh_degree,
        init_opacity=config.init_opacity,
        sky_shell_path=config.sky_shell_checkpoint,
    )
    audit_initial_means = None
    audit_initial_normals = None
    if config.fixed_topology_schedule.audit_steps:
        audit_initial_means = np.ascontiguousarray(xyz, dtype=np.float32)
        audit_quaternions = (
            np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(xyz), 1))
            if initial_quaternions is None
            else np.asarray(initial_quaternions, dtype=np.float32)
        )
        audit_initial_normals = shortest_axis_normals(
            np.asarray(initial_scales_m, dtype=np.float32), audit_quaternions
        )
    pose_refiner = None
    pose_optimizer = None
    auxiliary_params: dict[str, Any] = {}
    auxiliary_optimizers: dict[str, Any] = {}
    if config.rig_pose_refinement.enabled:
        pose_refiner = RigPoseRefiner(
            trainset.rig_frame_ids,
            config=config.rig_pose_refinement,
            device=config.device,
            rig_frame_centers=trainset.rig_frame_centers(),
        )
        pose_optimizer = pose_refiner.make_optimizer()
        auxiliary_params["rig_pose_deltas"] = pose_refiner.deltas
        auxiliary_optimizers["rig_pose_deltas"] = pose_optimizer
    exposure = None
    exposure_optimizer = None
    if config.exposure_compensation.enabled:
        # Face samples ("base::face_id") share their base image's exposure:
        # every face of one capture saw the same physical auto-exposure.
        exposure = ExposureCompensator(
            getattr(trainset, "exposure_image_ids", trainset.image_ids),
            config=config.exposure_compensation,
            device=config.device,
            group_by_image=trainset.camera_id_by_image,
        )
        exposure_optimizer = exposure.make_optimizer()
        auxiliary_params["exposure_log_gains"] = exposure.log_gains
        auxiliary_optimizers["exposure_log_gains"] = exposure_optimizer
    ppisp = None
    ppisp_optimizer = None
    if config.ppisp.enabled:
        # Per-camera ISP correction (exposure/vignetting/color): the physical
        # generalisation of the scalar gain; mutual exclusion is validated.
        ppisp = PpispCorrector(
            getattr(trainset, "exposure_image_ids", trainset.image_ids),
            config=config.ppisp,
            device=config.device,
            camera_by_image=trainset.camera_id_by_image,
        )
        ppisp_optimizer = ppisp.make_optimizer()
        auxiliary_params["ppisp_exposure"] = ppisp.exposure_params
        auxiliary_params["ppisp_vignetting"] = ppisp.vignetting_params
        auxiliary_params["ppisp_color"] = ppisp.color_params
        if ppisp.crf_params is not None:
            auxiliary_params["ppisp_crf"] = ppisp.crf_params
        auxiliary_optimizers["ppisp"] = ppisp_optimizer
    completed_steps = 0
    screen_clip_total = 0
    world_clamp_total = 0
    # Anchor-liveness evidence: densification splits change the Gaussian count,
    # and a stale anchor table silently zeroes the normal term for the very
    # Gaussians a fresh split most needs constrained. The refresh is event-
    # driven (see the refine_triggered branch below); these counters prove it
    # held rather than assume it.
    normal_loss_steps_total = 0
    normal_loss_steps_stale = 0
    # A run can complete every step it was asked for and still DELIVER a much
    # earlier model: the golden gate refuses any checkpoint over the floater
    # budget, and best_golden.pt - which the acceptance pipeline and the
    # sharpness tool both read - simply stops advancing. Nothing warns. R1
    # trained 16000 steps and delivered step 7000. These counters put the
    # delivered-vs-trained gap in the manifest.
    golden_rejection_streak = 0
    golden_floater_rejections = 0
    # Last proposal batch already folded into the telemetry; see the fold site.
    proposal_batches = 0
    topology_events: list[dict[str, Any]] = []
    phase_history: list[dict[str, Any]] = []
    optimization_audits: list[dict[str, Any]] = []
    initial_topology_count = int(len(params["means"]))
    strategy_active = (
        config.topology_policy.strategy_enabled and not config.implementation_smoke_only
    )
    last_metrics: dict[str, Any] = {}
    initial_loss: float | None = None
    best_loss = float("inf")
    last_seen_lifecycle_event: dict[str, Any] | None = None
    mcmc_telemetry: dict[str, Any] | None = None
    golden_history: list[dict[str, Any]] = []
    best_golden: dict[str, Any] | None = None
    full_evaluation_history: list[dict[str, Any]] = []
    warm_start: dict[str, Any] | None = None
    if config.warm_start_checkpoint is not None:
        if "source_identity" in checkpoint_identity:
            warm_start_lineage = {
                key: value
                for key, value in checkpoint_identity.items()
                if key
                not in {
                    "trainer_config_sha256",
                    "scale_calibration_sha256",
                    "initialization_geometry_sha256",
                    "surface_initialization_sha256",
                }
            }
        else:
            lineage_keys = (
                "dataset_manifest_sha256",
                "mask_manifest_sha256",
                "person_mask_manifest_sha256",
                "split_manifest_sha256",
                "depth_manifest_sha256",
                "coordinate_transform_sha256",
                "initialization_ply_sha256",
                "gsplat_runtime",
            )
            if config.raw_fisheye_post_refine_face_manifest is not None:
                # A core-owner merge is intentionally multi-Tile, so it cannot
                # inherit one Tile's initialization hash. Raw depth and Face4
                # sidecars also have different manifests while sharing the
                # signed dataset, split, masks, coordinate frame and Face4
                # lineage verified above.
                lineage_keys = tuple(
                    key
                    for key in lineage_keys
                    if key
                    not in {
                        "depth_manifest_sha256",
                        "initialization_ply_sha256",
                    }
                )
            warm_start_lineage = {
                key: checkpoint_identity[key]
                for key in lineage_keys
                if key in checkpoint_identity
            }
        warm_start = _warm_start_from_checkpoint(
            config.warm_start_checkpoint,
            expected_lineage=warm_start_lineage,
            params=params,
            auxiliary_params=auxiliary_params,
            fresh_auxiliary_names=config.warm_start_fresh_auxiliary,
            scale_multiplier=config.warm_start_scale_multiplier,
            map_location=config.device,
        )
        if config.warm_start_min_opacity > 0.0:
            warm_start["opacity_pruning"] = backend.prune_below_opacity(
                params,
                optimizers,
                min_opacity=config.warm_start_min_opacity,
            )
        warm_started_rows = int(len(next(iter(params.values()))))
        if warm_started_rows >= config.cap_max:
            raise ValueError(
                f"warm start holds {warm_started_rows} Gaussians but cap_max "
                f"is {config.cap_max}"
            )
    if config.frozen_tail_start is not None:
        frozen_hooks = _register_frozen_tail_hooks(params, config.frozen_tail_start)
        print(
            f"frozen tail active: rows >= {config.frozen_tail_start:,} render "
            f"but never optimize ({frozen_hooks} parameter hooks)",
            flush=True,
        )
    if config.resume_checkpoint is not None:
        allowed_source_trainer_config_sha256 = None
        allowed_lineage_difference_keys: tuple[str, ...] = ()
        if config.mipmap_pipeline_gate is not None:
            resume_gate, _ = load_and_verify_gate(config.mipmap_pipeline_gate)
            adaptive_resume = resume_gate.get("adaptive_growth", {})
            if adaptive_resume.get("stage") in {
                "evaluation",
                "review",
                "continuation",
                "stabilization",
            }:
                expected_checkpoint_sha = adaptive_resume.get(
                    "resume_checkpoint_sha256"
                )
                if _sha256_file(config.resume_checkpoint) != expected_checkpoint_sha:
                    raise ValueError(
                        "adaptive evaluation resume checkpoint signature mismatch"
                    )
                allowed_source_trainer_config_sha256 = adaptive_resume.get(
                    "resume_source_trainer_config_sha256"
                )
                allowed_lineage_difference_keys = tuple(
                    str(key)
                    for key in adaptive_resume.get(
                        "resume_allowed_lineage_differences", []
                    )
                )
        completed_steps, strategy_state, sampler_state, training_state = load_checkpoint(
            config.resume_checkpoint,
            expected_identity=checkpoint_identity,
            params=params,
            optimizers=optimizers,
            map_location=config.device,
            auxiliary_params=auxiliary_params,
            auxiliary_optimizers=auxiliary_optimizers,
            allowed_source_trainer_config_sha256=(
                allowed_source_trainer_config_sha256
            ),
            allowed_lineage_difference_keys=allowed_lineage_difference_keys,
        )
        sampler.set_state(sampler_state.cpu())
        last_metrics = dict(training_state["last_metrics"])
        initial_loss = float(training_state["initial_loss"])
        best_loss = float(training_state["best_loss"])
        restored_golden = training_state.get("golden_evaluation", {})
        if not isinstance(restored_golden, dict):
            raise ValueError("checkpoint golden evaluation state is invalid")
        restored_history = restored_golden.get("history", [])
        if not isinstance(restored_history, list):
            raise ValueError("checkpoint golden evaluation history is invalid")
        golden_history = [dict(record) for record in restored_history]
        restored_best = restored_golden.get("best")
        if restored_best is not None and not isinstance(restored_best, dict):
            raise ValueError("checkpoint golden evaluation best record is invalid")
        best_golden = None if restored_best is None else dict(restored_best)
        restored_full_history = restored_golden.get("full_history", [])
        if not isinstance(restored_full_history, list):
            raise ValueError("checkpoint full evaluation history is invalid")
        full_evaluation_history = [dict(record) for record in restored_full_history]
        restored_telemetry = training_state.get("mcmc_telemetry")
        if restored_telemetry is None and config.mcmc_noise_injection_stop_iter != 0:
            raise ValueError("full-MCMC checkpoint has no MCMC telemetry state")
        if restored_telemetry is not None:
            mcmc_telemetry = dict(restored_telemetry)
            if config.densification_strategy == "default_3dgs":
                normalize_classic_strategy_telemetry(mcmc_telemetry)
        topology_events = [
            dict(event) for event in training_state.get("topology_events", [])
        ]
        phase_history = [
            dict(event) for event in training_state.get("phase_history", [])
        ]
        optimization_audits = [
            dict(event) for event in training_state.get("optimization_audits", [])
        ]
        error_score_state = getattr(backend, "error_score_state", None)
        restored_error_scores = training_state.get("error_weighted_sampling")
        if error_score_state is not None:
            if restored_error_scores is None:
                raise ValueError(
                    "error-weighted checkpoint has no resumable sampling state"
                )
            error_score_state.restore_checkpoint_state(
                restored_error_scores,
                expected_count=len(params["means"]),
            )
        if completed_steps >= config.max_steps:
            raise ValueError("checkpoint already reached or exceeded max_steps")

    if mcmc_telemetry is None:
        mcmc_telemetry = initialize_mcmc_telemetry(
            snapshot_gaussians(params, min_opacity=backend.strategy.min_opacity)
        )

    torch.cuda.reset_peak_memory_stats(config.device)
    started = time.perf_counter()
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    best_checkpoint_path = output_dir / "checkpoints" / "best_golden.pt"
    golden_history_path = output_dir / "evaluation" / "golden_history.json"
    monitor_progress_path = output_dir / "monitor" / "progress.jsonl"
    full_evaluation_history_path = (
        output_dir / "evaluation" / "full_evaluation_history.json"
    )

    def checkpoint_training_state() -> dict[str, Any]:
        error_score_state = getattr(backend, "error_score_state", None)
        return {
            "last_metrics": last_metrics,
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "mcmc_telemetry": mcmc_telemetry,
            "topology_events": topology_events,
            "phase_history": phase_history,
            "optimization_audits": optimization_audits,
            "golden_evaluation": {
                "history": golden_history,
                "best": best_golden,
                "full_history": full_evaluation_history,
            },
            "error_weighted_sampling": (
                None
                if error_score_state is None
                else error_score_state.checkpoint_state()
            ),
        }

    sampled_epoch = -1
    sampled_order: tuple[int, ...] = ()
    for step in range(completed_steps, config.max_steps):
        phase = config.fixed_topology_schedule.phase_for_step(step)
        if not phase_history or phase_history[-1]["name"] != phase["name"]:
            phase_history.append(
                {
                    "name": phase["name"],
                    "first_step": step + 1,
                    "geometry_lr_scale": phase["geometry_lr_scale"],
                    "range_weight_scale": phase["range_weight_scale"],
                    "normal_weight_scale": phase["normal_weight_scale"],
                    "emit_gap_analysis": phase["emit_gap_analysis"],
                }
            )
        prune_due = (
            config.topology_policy.mode == "opacity_prune_only"
            and step == int(config.topology_policy.opacity_prune_step or -1)
        )
        if prune_due:
            before_prune = int(len(params["means"]))
            prune_report = backend.prune_below_opacity(
                params,
                optimizers,
                min_opacity=config.topology_policy.opacity_prune_threshold,
            )
            topology_count_transition(
                mode=config.topology_policy.mode,
                before_count=before_prune,
                after_count=int(len(params["means"])),
                prune_due=True,
            )
            topology_events.append(
                {"completed_steps_before_event": step, "kind": "opacity_prune", **prune_report}
            )
            if normal_anchors is not None:
                normal_anchors.refresh(params["means"])
        step_topology_count = int(len(params["means"]))
        audit_due = step + 1 in config.fixed_topology_schedule.audit_steps
        audit_before = (
            {
                name: params[name].detach().clone()
                for name in ("means", "scales", "quats", "opacities")
            }
            if audit_due
            else None
        )
        if (
            config.view_sampling_mode
            == "fisher_yates_without_replacement_per_epoch"
        ):
            epoch = step // len(trainset)
            if epoch != sampled_epoch:
                sampled_order = fisher_yates_epoch_order(
                    len(trainset), seed=config.seed, epoch=epoch
                )
                sampled_epoch = epoch
            index = sampled_order[step % len(trainset)]
        else:
            index = int(
                torch.randint(len(trainset), (1,), generator=sampler).item()
            )
        sample = trainset[index]
        tensors = _tensor_sample(sample, torch, config.device)
        c2w_override = (
            None
            if pose_refiner is None
            else pose_refiner.apply(sample.rig_frame_id, sample.c2w)
        )
        effective_means_lr = float(effective_learning_rates["means"])
        if config.means_lr_final_factor < 1.0:
            # Deterministic function of the step (no scheduler state), so an
            # interrupted resume lands on exactly the same learning rate.
            decayed = means_lr_for_step(
                effective_learning_rates["means"],
                config.means_lr_final_factor,
                step=step,
                max_steps=config.max_steps,
            )
            effective_means_lr = decayed
        # ``Optimizer.load_state_dict`` restores the source checkpoint's
        # param-group learning rates. Re-apply every signed rate on every step
        # so a stabilization resume can actually freeze geometry and lower
        # opacity/colour rates instead of silently retaining the old values.
        geometry_lr_scale = float(phase["geometry_lr_scale"])
        if step >= config.mcmc_refine_stop_iter:
            geometry_lr_scale *= float(config.post_refine_geometry_lr_scale)
        for name, optimizer in optimizers.items():
            if name == "means":
                base_lr = effective_means_lr
                scale = geometry_lr_scale
            elif name in {"scales", "quats"}:
                base_lr = float(effective_learning_rates[name])
                scale = geometry_lr_scale
            elif name in effective_learning_rates:
                base_lr = float(effective_learning_rates[name])
                scale = 1.0
            else:
                continue
            for group in optimizer.param_groups:
                group["lr"] = base_lr * scale
        ppisp_effective_lr = None
        if ppisp_optimizer is not None:
            ppisp_effective_lr = config.ppisp.learning_rate_for_step(
                step=step, total_steps=config.max_steps
            )
            for group in ppisp_optimizer.param_groups:
                group["lr"] = ppisp_effective_lr
        active_sh_degree = active_sh_degree_for_step(config, step)
        competitor_weights = (
            competitor_high_type2_loss_weights(step, len(trainset))
            if config.competitor_loss_schedule_enabled
            else None
        )
        # The recovered vendor schedule supplies the temporal envelope, while
        # the configured nominal weights remain explicit source-admission
        # controls.  This lets a LiDAR-first run disable an unproven DA2 source
        # without silently disabling the verified mesh depth/normal schedule.
        scheduled_da2_weight = (
            None
            if competitor_weights is None
            else competitor_weights.da2_depth
            * (float(config.da2_depth_weight) / 0.5)
        )
        scheduled_mesh_depth_weight = (
            None
            if competitor_weights is None
            else competitor_weights.mesh_depth
            * (float(config.mesh_depth_weight) / 0.5)
        )
        scheduled_mesh_normal_weight = (
            None
            if competitor_weights is None
            else competitor_weights.mesh_normal
            * (float(config.mesh_normal_weight) / 0.05)
        )
        loss, l1, ssim, range_loss, info = _render_supervision_loss(
            backend=backend,
            params=params,
            sample=sample,
            tensors=tensors,
            config=config,
            c2w_override=c2w_override,
            rgb_gain=None
            if exposure is None
            else exposure.gain(sample.image_id.split("::")[0]),
            ppisp=ppisp,
            active_sh_degree=active_sh_degree,
            range_weight_scale=float(phase["range_weight_scale"]),
            da2_depth_weight_override=(
                scheduled_da2_weight
            ),
            mesh_depth_weight_override=(
                scheduled_mesh_depth_weight
            ),
            mesh_normal_weight_override=(
                scheduled_mesh_normal_weight
            ),
            mesh_late_occlusion_scene_scale_m=(
                scene_extent_m
                if competitor_weights is not None and step > 10 * len(trainset)
                else None
            ),
        )
        range_directionality = None
        if (
            audit_due
            and range_loss is not None
            and info.get("cloudstudio_rendered_range_tensor") is not None
        ):
            range_directionality = _range_directionality_audit(
                rendered_range=info["cloudstudio_rendered_range_tensor"],
                tensors=tensors,
                config=config,
            )
        pose_prior = None
        if pose_refiner is not None:
            pose_prior = pose_refiner.prior_loss()
            loss = loss + pose_prior
        if exposure is not None:
            loss = loss + exposure.prior_loss()
        if ppisp is not None:
            loss = loss + ppisp.regularization_loss()
        if config.sh_regularization_weight > 0.0 and "shN" in params:
            # Weak pull on the view-dependent SH bands: the 30k diagnosis
            # measured shN energy doubling over long training while validation
            # preferred lower degrees, i.e. the bands memorize per-view
            # residuals faster than they explain real view dependence.
            loss = loss + config.sh_regularization_weight * params["shN"].square().mean()
        opacity_visibility_mask = None
        if (
            config.geometry_regularization.opacity_sparsity_scope
            == "visible_current_view"
        ):
            opacity_visibility_mask = _current_view_visibility_mask(
                info.get("radii"), len(params["opacities"])
            )
        regularization = geometry_regularization_terms(
            params,
            reference_scale_m=reference_scale_m,
            config=config.geometry_regularization,
            visibility_mask=opacity_visibility_mask,
        )
        loss = loss + regularization["total"]
        normal_terms = None
        if normal_anchors is not None:
            if (
                normal_anchors.stale
                or step == 0
                or step % config.lidar_normal_alignment.refresh_every == 0
            ):
                normal_anchors.refresh(params["means"])
            normal_terms = normal_anchors.loss(params)
            loss = loss + float(phase["normal_weight_scale"]) * normal_terms["total"]
            normal_loss_steps_total += 1
            # loss() re-flags itself stale when the cached table no longer
            # matches the Gaussian count and contributes zero for that step.
            if normal_anchors.stale:
                normal_loss_steps_stale += 1
        component_audit = None
        if audit_due:
            rgb_component = config.rgb_l1_weight * l1 + config.rgb_ssim_weight * ssim
            if info["cloudstudio_rgb_gradient_l1"] is not None:
                rgb_component = (
                    rgb_component
                    + config.rgb_gradient_weight * info["cloudstudio_rgb_gradient_l1"]
                )
            if info["cloudstudio_lidar_rgb_l1"] is not None:
                rgb_component = (
                    rgb_component
                    + config.lidar_rgb_l1_weight * info["cloudstudio_lidar_rgb_l1"]
                )
            component_audit = component_gradient_audit(
                params,
                {
                    "rgb": rgb_component,
                    "range_config_weighted": None
                    if range_loss is None
                    else config.lidar_range_weight * range_loss,
                    "lidar_alpha_config_weighted": None
                    if info["cloudstudio_lidar_alpha_loss"] is None
                    else config.lidar_alpha_weight
                    * info["cloudstudio_lidar_alpha_loss"],
                    "normal_config_weighted": None
                    if normal_terms is None
                    else normal_terms["total"],
                },
            )
        require_finite_training_tensors(
            params=params,
            loss=loss,
            stage=f"step_{step}_loss",
            check_gradients=False,
            check_parameters=False,
        )
        # Must precede backward: classic densification scores each Gaussian by
        # the gradient of the loss with respect to its PROJECTED position, and
        # that gradient only survives the backward pass if the strategy is given
        # the chance to retain it here. A no-op under MCMC, which scores from
        # opacity instead, and silent rather than fatal if skipped - the
        # criterion would simply never fire.
        if strategy_active:
            backend.strategy_pre_step(
                params, optimizers, strategy_state, step=step, info=info
            )
        if (
            strategy_active
            and
            backend.needs_pre_backward
            and config.densification_gradient_source == "rgb_only"
        ):
            # Two-pass backward so the densification criterion sees the gradient
            # of L1+SSIM alone, as Kerbl et al. trained. One backward over the
            # total loss would let the LiDAR range and normal terms leak into
            # means2d.grad through the shared rasterization. Implementation
            # smoke runs explicitly forbid densification and use one backward,
            # because eval3d does not expose the classic means2d gradient.
            # The photometric
            # pass runs FIRST and its means2d gradients are snapshotted, because
            # .grad accumulates across passes while gsplat overwrites .absgrad
            # on each - only a snapshot survives both semantics. The optimizer
            # still steps on the full total-loss gradient (the two passes sum
            # in the leaf parameters).
            rgb_loss = config.rgb_l1_weight * l1 + config.rgb_ssim_weight * ssim
            if info["cloudstudio_rgb_gradient_l1"] is not None:
                rgb_loss = (
                    rgb_loss
                    + config.rgb_gradient_weight
                    * info["cloudstudio_rgb_gradient_l1"]
                )
            if info["cloudstudio_lidar_rgb_l1"] is not None:
                rgb_loss = (
                    rgb_loss
                    + config.lidar_rgb_l1_weight
                    * info["cloudstudio_lidar_rgb_l1"]
                )
            rest_loss = loss - rgb_loss
            rgb_loss.backward(retain_graph=True)
            backend.strategy_isolate_gradient(info)
            if rest_loss.requires_grad:
                rest_loss.backward()
            backend.strategy_restore_gradient(info)
        else:
            loss.backward()
        audited_gradients = gradient_norms(params) if audit_due else None
        refine_boundary = strategy_active and (
            backend.strategy.is_refine_step(step)
            if hasattr(backend.strategy, "is_refine_step")
            else (
                step < config.mcmc_refine_stop_iter
                and step > config.mcmc_refine_start_iter
                and step % config.mcmc_refine_every == 0
            )
        )
        checkpoint_boundary = (
            (step + 1) % config.checkpoint_every == 0
            or step + 1 == config.max_steps
        )
        if refine_boundary or checkpoint_boundary:
            require_finite_training_tensors(
                params=params,
                loss=loss,
                stage=f"step_{step}_backward",
                check_gradients=True,
            )
        lifecycle_execution_order = str(
            getattr(
                backend.strategy,
                "lifecycle_execution_order",
                "post_optimizer_gsplat",
            )
        )
        pre_optimizer_strategy_event = None
        if strategy_active and lifecycle_execution_order == "pre_optimizer_vendor":
            pre_optimizer_strategy_event = backend.strategy_post_step(
                params,
                optimizers,
                strategy_state,
                step=step,
                info=info,
            )
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if audit_due:
            assert audit_before is not None
            assert audit_initial_means is not None and audit_initial_normals is not None
            topology_changed_during_step = (
                len(audit_before["means"]) != len(params["means"])
            )
            parameter_updates = (
                {
                    "skipped": "pre_optimizer_topology_changed_row_identity",
                    "before_count": int(len(audit_before["means"])),
                    "after_count": int(len(params["means"])),
                }
                if topology_changed_during_step
                else parameter_update_norms(audit_before, params)
            )
            point_to_plane_drift = (
                {
                    "skipped": "pre_optimizer_topology_changed_row_identity",
                    "before_count": int(len(audit_initial_means)),
                    "after_count": int(len(params["means"])),
                }
                if len(audit_initial_means) != len(params["means"])
                else point_to_plane_drift_summary(
                    params["means"].detach().cpu().numpy(),
                    audit_initial_means,
                    audit_initial_normals,
                )
            )
            optimization_audits.append(
                {
                    "completed_step": step + 1,
                    "phase": phase["name"],
                    "gradient_norms": audited_gradients,
                    "component_gradients": component_audit,
                    "parameter_updates": parameter_updates,
                    "point_to_plane_drift": point_to_plane_drift,
                    "range_directionality": range_directionality,
                }
            )
        if pose_optimizer is not None:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        if exposure_optimizer is not None:
            exposure_optimizer.step()
            exposure_optimizer.zero_grad(set_to_none=True)
            exposure.project_zero_mean()
        if ppisp_optimizer is not None:
            ppisp_optimizer.step()
            ppisp_optimizer.zero_grad(set_to_none=True)
        if getattr(backend, "error_score_state", None) is not None:
            # Feed the per-pixel residual of this step's view into the
            # relocation/densification sampling scores while the projected
            # centers still match the gaussian count.
            error_map = (
                (info["cloudstudio_rendered_rgb"] - tensors["rgb"]).abs().mean(dim=-1)
            ) * tensors["rgb_mask"]
            radii = info["radii"].detach()
            radii = (
                radii.reshape(-1, 2) if radii.shape[-1] == 2 else radii.reshape(-1)
            )
            contribution = None
            if (
                getattr(config.error_weighted_sampling, "aggregation", "center")
                == "contribution"
                and step % max(1, config.contribution_every) == 0
            ):
                # One extra rasterization of a non-learnable scalar channel
                # gives the true alpha/transmittance-weighted error per
                # Gaussian, which the footprint proxy cannot see. Cadenced,
                # because the EMA already spans thousands of steps.
                contribution = compute_contribution_scores(
                    backend,
                    params,
                    sample,
                    error_map,
                    config=config.contribution,
                    c2w_override=c2w_override,
                )
            backend.error_score_state.update(
                info["means2d"].detach().reshape(-1, 2),
                radii,
                error_map,
                sample.height,
                sample.width,
                conics=info["conics"].detach().reshape(-1, 3),
                opacities=info["opacities"].detach().reshape(-1)
                if "opacities" in info
                else None,
                contribution=contribution,
            )
            # Advance the per-Gaussian lifecycle clock before the refinement
            # below can grow/relocate: ages tick, and rows born in this step
            # record it as their birth step.
            backend.error_score_state.on_step(step)
        # Before strategy_post_step: relocation/add would desynchronize this
        # step's projected radii from the gaussian count.
        clip_report = clip_oversized_gaussians(
            params,
            radii_px=info.get("radii"),
            image_size_px=min(sample.width, sample.height),
            config=config.geometry_regularization,
        )
        screen_clip_total += clip_report["clipped_count"]
        world_clamp_total += clip_report["world_clamped_count"]
        if admission is not None and (
            admission.stale or step % config.lidar_admission.refresh_every == 0
        ):
            # Refresh *before* strategy_post_step: the densification inside it
            # is the only consumer, and the means still line up with the count
            # the last refine produced. The stale branch covers the step after
            # each refine (count changed); the cadence branch covers positional
            # drift while the count held still.
            update_admission_telemetry(
                mcmc_telemetry,
                admission.refresh(
                    params["means"],
                    lifecycle=getattr(
                        getattr(backend, "error_score_state", None), "lifecycle", None
                    ),
                ),
            )
        if not strategy_active:
            mcmc_event = build_mcmc_step_event(
                step=step,
                before=None,
                after=None,
                refine_start_iter=config.max_steps,
                refine_stop_iter=config.max_steps + 1,
                refine_every=1,
                noise_injection_stop_iter=0,
                strategy_prunes=backend.strategy_prunes,
            )
            mcmc_event["strategy_skipped"] = True
            mcmc_event["strategy_skip_reason"] = (
                "implementation_smoke_only"
                if config.implementation_smoke_only
                else f"topology_policy={config.topology_policy.mode}"
            )
        elif pre_optimizer_strategy_event is not None:
            mcmc_event = pre_optimizer_strategy_event
        else:
            mcmc_event = backend.strategy_post_step(
                params,
                optimizers,
                strategy_state,
                step=step,
                info=info,
            )
        append_mcmc_telemetry(mcmc_telemetry, mcmc_event)
        topology_count_transition(
            mode=config.topology_policy.mode,
            before_count=step_topology_count,
            after_count=int(len(params["means"])),
            prune_due=False,
        )
        if mcmc_event["refine_triggered"] and normal_anchors is not None:
            # Relocation/densification changed the gaussian set; re-anchor
            # before the next normal-alignment loss uses stale indices.
            normal_anchors.refresh(params["means"])
        if mcmc_event["refine_triggered"] and admission is not None:
            # The strategy now maintains the cache event by event (extend on
            # grow, copy on relocate), so the wholesale invalidation is only the
            # fallback for refinements it could not follow - notably the
            # unweighted upstream sample_add branch, which keeps its parent
            # indices private. Anything still out of sync is re-queried lazily
            # by the stale branch above rather than eagerly here.
            if not admission.in_sync(len(params["means"])):
                admission.on_count_changed(len(params["means"]))
        if proposal is not None and proposal.propose_count != proposal_batches:
            # Fold in only when a batch actually ran: a refine that relocated
            # but grew nothing leaves last_stats untouched, and re-folding it
            # would double-count the previous batch.
            proposal_batches = proposal.propose_count
            update_proposal_telemetry(mcmc_telemetry, proposal.last_stats)
        if mcmc_event["refine_triggered"]:
            require_finite_training_tensors(
                params=params,
                loss=loss,
                stage=f"step_{step}_mcmc_refine",
                check_gradients=False,
            )
        last_metrics = {
            "loss": float(loss.detach().cpu()),
            "optimization_phase": phase["name"],
            "topology_mode": config.topology_policy.mode,
            "effective_geometry_lr_scale": geometry_lr_scale,
            "effective_lidar_range_weight": float(
                config.lidar_range_weight * phase["range_weight_scale"]
            ),
            "effective_lidar_alpha_weight": config.lidar_alpha_weight,
            "effective_lidar_normal_weight_scale": float(
                phase["normal_weight_scale"]
            ),
            "effective_da2_depth_weight": float(
                config.da2_depth_weight
                if competitor_weights is None
                else scheduled_da2_weight
            ),
            "effective_mesh_depth_weight": float(
                config.mesh_depth_weight
                if competitor_weights is None
                else scheduled_mesh_depth_weight
            ),
            "effective_mesh_normal_weight": float(
                config.mesh_normal_weight
                if competitor_weights is None
                else scheduled_mesh_normal_weight
            ),
            "ppisp_effective_lr": ppisp_effective_lr,
            "rgb_mask_true_pixels": int(tensors["rgb_mask"].sum().detach().cpu()),
            "rgb_mask_total_pixels": int(tensors["rgb_mask"].numel()),
            "rgb_mask_valid_fraction": float(
                tensors["rgb_mask"].float().mean().detach().cpu()
            ),
            "depth_mask_true_pixels": None
            if "depth_mask" not in tensors
            else int(tensors["depth_mask"].sum().detach().cpu()),
            "depth_mask_valid_fraction": None
            if "depth_mask" not in tensors
            else float(tensors["depth_mask"].float().mean().detach().cpu()),
            "mesh_mask_true_pixels": None
            if "mesh_mask" not in tensors
            else int(tensors["mesh_mask"].sum().detach().cpu()),
            "mesh_mask_valid_fraction": None
            if "mesh_mask" not in tensors
            else float(
                tensors["mesh_mask"].float().mean().detach().cpu()
            ),
            "rgb_l1": float(l1.detach().cpu()),
            "rgb_psnr": info.get("cloudstudio_rgb_psnr"),
            "rgb_gradient_l1": None
            if info["cloudstudio_rgb_gradient_l1"] is None
            else float(info["cloudstudio_rgb_gradient_l1"].detach().cpu()),
            "lidar_rgb_l1": None
            if info["cloudstudio_lidar_rgb_l1"] is None
            else float(info["cloudstudio_lidar_rgb_l1"].detach().cpu()),
            "lidar_rgb_support_fraction": None
            if info["cloudstudio_lidar_rgb_support_fraction"] is None
            else float(info["cloudstudio_lidar_rgb_support_fraction"].cpu()),
            "lidar_alpha_loss": None
            if info["cloudstudio_lidar_alpha_loss"] is None
            else float(info["cloudstudio_lidar_alpha_loss"].detach().cpu()),
            "lidar_alpha_support_fraction": None
            if info["cloudstudio_lidar_alpha_support_fraction"] is None
            else float(
                info["cloudstudio_lidar_alpha_support_fraction"].detach().cpu()
            ),
            "rgb_ssim_loss": float(ssim.detach().cpu()),
            "rgb_ssim_mode": config.rgb_ssim_mode,
            "rgb_local_ssim_loss": None
            if config.rgb_ssim_mode != "local_gaussian"
            else float(ssim.detach().cpu()),
            "rgb_global_ssim_loss": None
            if config.rgb_ssim_mode != "global_moments"
            else float(ssim.detach().cpu()),
            "lidar_range_loss": None
            if range_loss is None
            else float(range_loss.detach().cpu()),
            "lidar_range_loss_mode": config.lidar_range_loss_mode,
            "lidar_range_l1_m": None
            if range_loss is None or config.lidar_range_loss_mode != "linear_l1"
            else float(range_loss.detach().cpu()),
            "lidar_linear_aux_l1_m": None
            if info["cloudstudio_linear_range_aux_loss"] is None
            else float(info["cloudstudio_linear_range_aux_loss"].cpu()),
            "da2_depth_loss": None
            if info["cloudstudio_da2_depth_loss"] is None
            else float(info["cloudstudio_da2_depth_loss"].detach().cpu()),
            "mesh_depth_loss": None
            if info["cloudstudio_mesh_depth_loss"] is None
            else float(info["cloudstudio_mesh_depth_loss"].detach().cpu()),
            "mesh_normal_loss": None
            if info["cloudstudio_mesh_normal_loss"] is None
            else float(info["cloudstudio_mesh_normal_loss"].detach().cpu()),
            "rig_pose_prior": None
            if pose_prior is None
            else float(pose_prior.detach().cpu()),
            "geometry_regularization": float(regularization["total"].detach().cpu()),
            "opacity_sparsity": float(regularization["opacity_sparsity"].detach().cpu()),
            "opacity_sparsity_active_count": int(
                regularization["opacity_sparsity_active_count"]
            ),
            "opacity_sparsity_active_fraction": float(
                regularization["opacity_sparsity_active_fraction"].detach().cpu()
            ),
            "scale_upper": float(regularization["scale_upper"].detach().cpu()),
            "scale_over_limit_fraction": float(
                regularization["scale_over_limit_fraction"].detach().cpu()
            ),
            "scale_upper_tail_count": int(
                regularization["scale_upper_tail_count"]
            ),
            "anisotropy": float(regularization["anisotropy"].detach().cpu()),
            "lidar_normal_align_raw": None
            if normal_terms is None
            else float(normal_terms["align_raw"].detach().cpu()),
            "lidar_normal_flatten_raw": None
            if normal_terms is None
            else float(normal_terms["flatten_raw"].detach().cpu()),
            "lidar_point_to_plane_raw_m": None
            if normal_terms is None
            else float(normal_terms["point_to_plane_raw"].detach().cpu()),
        }
        lifecycle_event = getattr(
            getattr(backend, "strategy", None), "last_lifecycle_event", None
        )
        if lifecycle_event is not None:
            # The adapter refreshes this on refine steps only; carrying the
            # last-known event on every persisted record is what makes
            # birth/death visible on the dashboard instead of dying on the
            # strategy object (persistence is sparser than the refine cadence).
            last_seen_lifecycle_event = lifecycle_event
            growth = lifecycle_event.get("growth_diagnostics", {})
            cull = lifecycle_event.get("cull_reasons", {})
            quantiles = growth.get("gradient_quantiles", {})
            last_metrics.update(
                {
                    "lifecycle_cull_total": lifecycle_event.get("cull_count"),
                    "lifecycle_after_count": lifecycle_event.get("after_count"),
                    "lifecycle_clone_parents": growth.get("clone_parent_count"),
                    "lifecycle_split_parents": growth.get("split_parent_count"),
                    "lifecycle_growth_eligible": growth.get("selected_parent_count"),
                    "lifecycle_grad_threshold": growth.get("gradient_threshold"),
                    "lifecycle_grad_p90": quantiles.get("p90"),
                    "lifecycle_grad_p99": quantiles.get("p99"),
                    "lifecycle_frac_op_below_010": growth.get(
                        "fraction_opacity_below_0p10"
                    ),
                    "lifecycle_frac_op_above_015": growth.get(
                        "fraction_opacity_above_0p15"
                    ),
                    "lifecycle_grad_only_candidates": growth.get(
                        "gradient_only_candidate_count"
                    ),
                    "lifecycle_grad_sum_p99": growth.get(
                        "gradient_convention_audit", {}
                    ).get("window_sum_p99"),
                    "lifecycle_observations_p50": growth.get(
                        "gradient_convention_audit", {}
                    ).get("observations_p50"),
                    "lifecycle_cull_opacity": cull.get("selected_opacity_count"),
                    "lifecycle_cull_world_scale": cull.get("world_scale_count"),
                    "lifecycle_cull_screen_scale": cull.get("screen_scale_count"),
                    "lifecycle_cull_opacity_threshold": cull.get("opacity_threshold"),
                    "lifecycle_relaxed_cull": (
                        None
                        if cull.get("relaxed_cull_only") is None
                        else int(bool(cull.get("relaxed_cull_only")))
                    ),
                }
            )
        if initial_loss is None:
            initial_loss = last_metrics["loss"]
        best_loss = min(best_loss, last_metrics["loss"])
        completed = step + 1
        if (
            completed == 1
            or completed % 10 == 0
            or audit_due
            or lifecycle_event is not None
        ):
            _append_monitor_event(
                monitor_progress_path,
                {
                    "schema_version": 1,
                    "kind": "cloudstudio_training_progress",
                    "timestamp_unix": time.time(),
                    "run_id": config.run_id,
                    "completed_steps": completed,
                    "max_steps": config.max_steps,
                    "gaussian_count": int(len(params["means"])),
                    "phase": phase["name"],
                    "topology_mode": config.topology_policy.mode,
                    "metrics": last_metrics,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated(config.device)),
                    "latest_optimization_audit": (
                        optimization_audits[-1]
                        if audit_due and optimization_audits
                        else None
                    ),
                },
            )
        golden_artifact_due = config.golden_evaluation.enabled and (
            completed % config.golden_evaluation.artifact_every == 0
            or completed == config.max_steps
        )
        golden_due = config.golden_evaluation.enabled and (
            completed % config.golden_evaluation.every == 0
            or completed == config.max_steps
            or golden_artifact_due
        )
        golden_promoted = False
        if golden_due:
            golden_result = evaluate_golden_views(
                backend=backend,
                params=params,
                dataset=valset,
                split_manifest=valset.split_manifest,
                completed_steps=completed,
                background_rgb=config.background_color,
                artifact_output_dir=output_dir if golden_artifact_due else None,
                geometry_tree=geometry_tree,
            )
            golden_history.append(golden_result)
            golden_promoted = is_golden_improvement(
                golden_result,
                best_golden,
                min_psnr_improvement_db=config.golden_evaluation.min_psnr_improvement_db,
                max_depth_regression_m=config.golden_evaluation.max_depth_regression_m,
                max_floater_growth_ratio=(
                    config.golden_evaluation.max_floater_growth_ratio
                ),
                max_floater_count=config.golden_evaluation.max_floater_count,
            )
            if golden_promoted:
                best_golden = golden_result
                golden_rejection_streak = 0
            else:
                golden_rejection_streak += 1
                gate = config.golden_evaluation.max_floater_count
                floaters = golden_result["summary"].get("floater_count")
                if gate is not None and floaters is not None and floaters > gate:
                    golden_floater_rejections += 1
            _write_golden_history(
                golden_history_path,
                config=config.golden_evaluation,
                history=golden_history,
                best=best_golden,
            )
        full_evaluation_due = config.golden_evaluation.enabled and (
            completed % config.golden_evaluation.full_every == 0
            or completed == config.max_steps
        )
        if full_evaluation_due:
            full_evaluation_history.append(
                evaluate_full_validation(
                    backend=backend,
                    params=params,
                    dataset=valset,
                    split_manifest=valset.split_manifest,
                    completed_steps=completed,
                    background_rgb=config.background_color,
                )
            )
            _write_full_evaluation_history(
                full_evaluation_history_path,
                history=full_evaluation_history,
            )
        checkpoint_due = (
            completed % config.checkpoint_every == 0
            or completed == config.max_steps
            or completed == controlled_stop_after_steps
            or golden_due
            or full_evaluation_due
            or (
                config.checkpoint_keep_every > 0
                and completed % config.checkpoint_keep_every == 0
            )
        )
        if checkpoint_due:
            mcmc_telemetry["last_snapshot"] = snapshot_gaussians(
                params, min_opacity=backend.strategy.min_opacity
            )
            require_finite_training_tensors(
                params=params,
                loss=loss,
                stage=f"step_{step}_checkpoint",
                check_gradients=False,
            )
            save_checkpoint(
                checkpoint_path,
                step=completed,
                identity=checkpoint_identity,
                params=params,
                optimizers=optimizers,
                strategy_state=strategy_state,
                sampler_state=sampler.get_state(),
                training_state=checkpoint_training_state(),
                auxiliary_params=auxiliary_params,
                auxiliary_optimizers=auxiliary_optimizers,
            )
            if (
                config.checkpoint_keep_every > 0
                and completed % config.checkpoint_keep_every == 0
            ):
                retain_checkpoint(
                    checkpoint_path,
                    checkpoint_path.parent / f"step_{completed:08d}.pt",
                )
            if golden_promoted:
                save_checkpoint(
                    best_checkpoint_path,
                    step=completed,
                    identity=checkpoint_identity,
                    params=params,
                    optimizers=optimizers,
                    strategy_state=strategy_state,
                    sampler_state=sampler.get_state(),
                    training_state=checkpoint_training_state(),
                    auxiliary_params=auxiliary_params,
                    auxiliary_optimizers=auxiliary_optimizers,
                )
        if completed == controlled_stop_after_steps:
            torch.cuda.synchronize(config.device)
            raise ControlledTrainingInterruption(
                completed_steps=completed,
                checkpoint_path=checkpoint_path,
            )
        if (
            config.cuda_empty_cache_interval_steps > 0
            and step % config.cuda_empty_cache_interval_steps == 0
        ):
            # Release cached allocator blocks for serial tile runs. Live
            # tensors remain resident; non-tiled runs leave this disabled.
            torch.cuda.empty_cache()

    torch.cuda.synchronize(config.device)
    duration_seconds = time.perf_counter() - started
    peak_vram_bytes = int(torch.cuda.max_memory_allocated(config.device))
    mcmc_telemetry["last_snapshot"] = snapshot_gaussians(
        params, min_opacity=backend.strategy.min_opacity
    )
    pose_report = disabled_pose_refinement_report(config.rig_pose_refinement)
    if pose_refiner is not None:
        loss_before, loss_after, evaluated_images = _compare_pose_candidate(
            backend=backend,
            params=params,
            dataset=trainset,
            refiner=pose_refiner,
            config=config,
        )
        pose_report = build_pose_refinement_report(
            pose_refiner.rig_frame_ids,
            pose_refiner.snapshot(),
            loss_before=loss_before,
            loss_after=loss_after,
            config=config.rig_pose_refinement,
        )
        pose_report["comparison"]["evaluated_images"] = evaluated_images
        if not pose_report["candidate_accepted"]:
            pose_refiner.zero_()
        final_training_state = checkpoint_training_state()
        final_training_state["rig_pose_refinement"] = pose_report
        final_training_state["exposure_compensation"] = (
            None if exposure is None else exposure.report()
        )
        save_checkpoint(
            checkpoint_path,
            step=config.max_steps,
            identity=checkpoint_identity,
            params=params,
            optimizers=optimizers,
            strategy_state=strategy_state,
            sampler_state=sampler.get_state(),
            training_state=final_training_state,
            auxiliary_params=auxiliary_params,
            auxiliary_optimizers=auxiliary_optimizers,
        )
    golden_history_artifact = _write_golden_history(
        golden_history_path,
        config=config.golden_evaluation,
        history=golden_history,
        best=best_golden,
    )
    full_evaluation_artifact = _write_full_evaluation_history(
        full_evaluation_history_path,
        history=full_evaluation_history,
    )
    final_gaussian_count = len(params["means"])
    selected_model_path = checkpoint_path
    selected_model_step = config.max_steps
    selected_checkpoint_training_state = checkpoint_training_state()
    if best_golden is not None:
        (
            selected_model_step,
            _,
            _,
            selected_checkpoint_training_state,
        ) = load_checkpoint(
            best_checkpoint_path,
            expected_identity=checkpoint_identity,
            params=params,
            optimizers=optimizers,
            map_location=config.device,
            auxiliary_params=auxiliary_params,
            auxiliary_optimizers=auxiliary_optimizers,
        )
        if selected_model_step != int(best_golden["completed_steps"]):
            raise ValueError("best checkpoint step does not match golden selection")
        selected_model_path = best_checkpoint_path
    if config.final_evaluation_artifacts:
        frames = _save_evaluation_artifacts(
            backend=backend,
            params=params,
            dataset=valset,
            output_dir=output_dir,
            background_rgb=config.background_color,
        )
    else:
        frames = []
    retained_checkpoints = _retained_checkpoint_records(
        output_dir,
        keep_every_steps=config.checkpoint_keep_every,
        completed_steps=config.max_steps,
    )
    run_manifest = sign_run_manifest(
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "implementation_smoke_only": config.implementation_smoke_only,
            "dataset_manifest_sha256": trainset.dataset_sha256,
            "mask_manifest_sha256": trainset.mask_sha256,
            "person_mask_manifest_sha256": trainset.person_mask_sha256,
            "split_manifest_sha256": trainset.split_sha256,
            "depth_manifest_sha256": trainset.depth_sha256,
            "coordinate_transform_sha256": coordinate["coordinate_transform_sha256"],
            "trainer_config_sha256": config_sha256,
            "trainer_contract": contract,
            "gsplat_runtime": backend.runtime,
            "initialization_ply_sha256": initialization_sha256,
            "metric_scale_calibration": scale_calibration,
            "surface_initialization": surface_initialization_report,
            "rig_pose_refinement": pose_report,
            "exposure_compensation": None if exposure is None else exposure.report(),
            "ppisp": None if ppisp is None else ppisp.report(),
            "golden_evaluation": {
                "configuration": config.golden_evaluation.to_dict(),
                "history_path": golden_history_path.relative_to(output_dir).as_posix(),
                "history_sha256": golden_history_artifact["golden_history_sha256"],
                "evaluation_count": len(golden_history),
                "best": best_golden,
                "best_checkpoint_path": None
                if best_golden is None
                else best_checkpoint_path.relative_to(output_dir).as_posix(),
                "best_checkpoint_sha256": None
                if best_golden is None
                else _sha256_file(best_checkpoint_path),
            },
            "periodic_full_evaluation": {
                "history_path": full_evaluation_history_path.relative_to(
                    output_dir
                ).as_posix(),
                "history_sha256": full_evaluation_artifact[
                    "full_evaluation_history_sha256"
                ],
                "evaluation_count": len(full_evaluation_history),
                "latest": None
                if not full_evaluation_history
                else full_evaluation_history[-1],
            },
            "final_evaluation_artifacts": {
                "enabled": config.final_evaluation_artifacts,
                "status": "COMPLETE"
                if config.final_evaluation_artifacts
                else "DEFERRED",
                "reason": None
                if config.final_evaluation_artifacts
                else "separate_3dgut_evaluation_required",
            },
            "frames": frames,
            "training": {
                "status": (
                    "IMPLEMENTATION_SMOKE_COMPLETE"
                    if config.implementation_smoke_only
                    else "COMPLETE"
                ),
                "completed_steps": config.max_steps,
                "duration_seconds": duration_seconds,
                "peak_vram_bytes": peak_vram_bytes,
                # CAREFUL: params was reloaded from best_golden above, so this
                # counts the DELIVERED model. When the floater gate freezes
                # selection early the two diverge sharply - R5 delivered step
                # 1000 with 390,901 Gaussians while training ended at 1,894,580
                # - and reading the wrong one inverts conclusions about
                # capacity. final_gaussian_count is the end of training.
                "gaussian_count": len(params["means"]),
                "delivered_gaussian_count": len(params["means"]),
                "final_gaussian_count": final_gaussian_count,
                "model_path": selected_model_path.relative_to(output_dir).as_posix(),
                "model_sha256": _sha256_file(selected_model_path),
                "selected_checkpoint_step": selected_model_step,
                "selected_checkpoint_kind": "best_golden"
                if best_golden is not None
                else "latest_final",
                "latest_checkpoint_path": checkpoint_path.relative_to(
                    output_dir
                ).as_posix(),
                "retained_checkpoints": retained_checkpoints,
                "selected_checkpoint_last_metrics": selected_checkpoint_training_state.get(
                    "last_metrics"
                ),
                "screen_clip_events": screen_clip_total,
                "world_clamp_events": world_clamp_total,
                # Anchor-liveness evidence for the LiDAR normal term: any
                # nonzero stale count means Gaussians trained unconstrained
                # right after a refine event, which biases densification arms.
                "normal_loss_steps_total": normal_loss_steps_total,
                "normal_loss_steps_stale": normal_loss_steps_stale,
                "normal_loss_active_ratio": None
                if normal_loss_steps_total == 0
                else 1.0 - normal_loss_steps_stale / normal_loss_steps_total,
                # How much of the training the delivered model actually saw.
                # A ratio below 1.0 means the gate froze selection early and
                # every metric describes that earlier model, not this run.
                "delivered_step_fraction": None
                if not config.max_steps or selected_model_step is None
                else selected_model_step / config.max_steps,
                "golden_rejection_streak_final": golden_rejection_streak,
                "golden_floater_rejections": golden_floater_rejections,
                "last_metrics": last_metrics,
                "initial_loss": initial_loss,
                "best_loss": best_loss,
                "loss_improvement_fraction": None
                if initial_loss in (None, 0.0)
                else (initial_loss - last_metrics["loss"]) / initial_loss,
                "mcmc_operator_report": backend.runtime.get(
                    "mcmc_operator_report"
                ),
                # What actually densified this run, with the metric metres each
                # normalised gate resolved to - the scene_scale incident showed
                # the config alone cannot expose a wrong resolution.
                "densification": {
                    "active": strategy_active,
                    "topology_mode": config.topology_policy.mode,
                    "strategy": config.densification_strategy,
                    "gradient_source": config.densification_gradient_source,
                    "resolved": backend.strategy.state_dict()
                    if config.densification_strategy == "default_3dgs"
                    else None,
                },
                "topology": {
                    "policy": config.topology_policy.to_dict(),
                    "initial_gaussian_count": initial_topology_count,
                    "final_gaussian_count": final_gaussian_count,
                    "events": topology_events,
                },
                "optimization_phases": {
                    "configuration": config.fixed_topology_schedule.to_dict(),
                    "history": phase_history,
                    "audits": optimization_audits,
                },
                "mcmc_telemetry": mcmc_telemetry,
                "warm_start": warm_start,
            },
        }
    )
    _atomic_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest


def train_from_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = value.get("config_manifest_sha256")
    if expected is not None:
        if len(str(expected)) != 64:
            raise ValueError("trainer config manifest SHA256 is invalid")
        unsigned = dict(value)
        unsigned.pop("config_manifest_sha256", None)
        actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if actual != expected:
            raise ValueError("trainer config manifest signature mismatch")
    return train(TrainerConfig.from_dict(value))
