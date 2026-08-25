"""CloudStudio-owned raw-fisheye gsplat trainer with no Viewer dependency."""

from __future__ import annotations

import hashlib
import json
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
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap
from cloudstudio_3dgs.evaluation.quality_report import sign_run_manifest
from cloudstudio_3dgs.training.backend import GsplatBackend
from cloudstudio_3dgs.training.checkpoint import load_checkpoint, save_checkpoint
from cloudstudio_3dgs.training.contracts import build_coordinate_transform_manifest
from cloudstudio_3dgs.training.dataset import S1TrainingDataset, TrainingSample
from cloudstudio_3dgs.training.losses import (
    confidence_weighted_log_range_huber,
    confidence_weighted_range_l1,
    global_masked_rgb_ssim_loss,
    masked_rgb_l1,
    masked_rgb_ssim_loss,
)
from cloudstudio_3dgs.training.presets import (
    assert_trainer_preset_matches,
    expand_trainer_preset,
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
    initialize_mcmc_telemetry,
    require_finite_training_tensors,
    snapshot_gaussians,
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
    person_mask_manifest: Path | None = None
    person_mask_root: Path | None = None
    depth_manifest: Path | None = None
    depth_root: Path | None = None
    resume_checkpoint: Path | None = None
    face_cache_manifest: Path | None = None
    face_cache_root: Path | None = None
    require_person_masks: bool = True
    trainer_preset: str = "custom"
    device: str = "cuda:0"
    seed: int = 42
    max_steps: int = 3_000
    checkpoint_every: int = 500
    factor: int = 4
    crop: CropWindow | None = None
    cap_max: int = 1_000_000
    init_scale_m: float = 0.05
    metric_scale_calibration: MetricScaleCalibrationConfig = field(
        default_factory=MetricScaleCalibrationConfig
    )
    rgb_l1_weight: float = 0.8
    rgb_ssim_weight: float = 0.2
    rgb_ssim_mode: str = "local_gaussian"
    lidar_range_weight: float = 0.05
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    ssim_min_valid_fraction: float = 0.8
    lidar_range_loss_mode: str = "robust_log_huber"
    lidar_log_range_huber_delta: float = 0.05
    lidar_linear_aux_weight: float = 0.0
    decoupled_ssim: bool = False
    sh_regularization_weight: float = 0.0
    pinhole_rasterize_mode: str = "classic"
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
    # "error_weighted_mcmc" is this trainer's homegrown sampler; "default_3dgs"
    # is gsplat's reference implementation of Kerbl et al., which scores by the
    # projected-position gradient instead of an image-error map. Measurement on
    # 2026-08-25 traced the blur to where the error map places births.
    densification_strategy: str = "error_weighted_mcmc"
    default_strategy: dict[str, Any] = field(default_factory=dict)
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
                "output_dir",
                "gsplat_lock",
                "person_mask_manifest",
                "person_mask_root",
                "depth_manifest",
                "depth_root",
                "resume_checkpoint",
                "face_cache_manifest",
                "face_cache_root",
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
        options = {
            key: value[key]
            for key in (
                "device",
                "require_person_masks",
                "seed",
                "max_steps",
                "checkpoint_every",
                "factor",
                "cap_max",
                "init_scale_m",
                "rgb_l1_weight",
                "rgb_ssim_weight",
                "rgb_ssim_mode",
                "lidar_range_weight",
                "ssim_window_size",
                "ssim_sigma",
                "ssim_min_valid_fraction",
                "lidar_range_loss_mode",
                "lidar_log_range_huber_delta",
                "lidar_linear_aux_weight",
                "decoupled_ssim",
                "sh_regularization_weight",
                "pinhole_rasterize_mode",
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
                "densification_strategy",
                "default_strategy",
                "densification_gradient_source",
                "learning_rates",
            )
            if key in value
        }
        return cls(
            run_id=str(value["run_id"]),
            trainer_preset=str(value["trainer_preset"]),
            crop=crop,
            rig_pose_refinement=pose_refinement,
            metric_scale_calibration=scale_calibration,
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
        if self.cap_max <= 4:
            raise ValueError("cap_max must be greater than four")
        if self.init_scale_m <= 0.0:
            raise ValueError("init_scale_m must be positive")
        self.metric_scale_calibration.validate()
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
            self.lidar_range_weight,
            self.lidar_linear_aux_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if self.rgb_l1_weight + self.rgb_ssim_weight <= 0.0:
            raise ValueError("at least one RGB loss weight must be positive")
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
        if self.tangent_proposal.enabled and not self.error_weighted_sampling.enabled:
            # The proposal only runs inside the weighted _add_new_gs path; the
            # plain-opacity fallback calls upstream sample_add, which has no
            # hook for it, so enabling it alone would silently do nothing.
            raise ValueError(
                "tangent_proposal requires error_weighted_sampling to be enabled"
            )
        if self.ppisp.enabled and self.exposure_compensation.enabled:
            raise ValueError(
                "ppisp replaces scalar exposure compensation; enable only one"
            )
        if self.ppisp.enabled and self.decoupled_ssim:
            raise ValueError(
                "decoupled_ssim currently supports only the scalar exposure gain"
            )
        expected_lrs = {"means", "scales", "quats", "opacities", "colors"}
        if set(self.learning_rates) != expected_lrs:
            raise ValueError(f"learning_rates must contain exactly {sorted(expected_lrs)}")
        if any(float(value) <= 0.0 for value in self.learning_rates.values()):
            raise ValueError("all learning rates must be positive")
        if self.mcmc_refine_start_iter < 0 or self.mcmc_refine_stop_iter <= 0:
            raise ValueError("MCMC refine bounds must be non-negative/positive")
        if self.mcmc_refine_start_iter >= self.mcmc_refine_stop_iter:
            raise ValueError("MCMC refine start must be smaller than refine stop")
        if self.mcmc_refine_every <= 0 or self.mcmc_noise_lr < 0.0:
            raise ValueError("MCMC refine interval must be positive and noise LR non-negative")
        if self.mcmc_noise_injection_stop_iter < -1:
            raise ValueError("MCMC noise stop must be -1 or non-negative")
        if self.densification_strategy not in DENSIFICATION_STRATEGIES:
            raise ValueError(
                f"densification_strategy must be one of {list(DENSIFICATION_STRATEGIES)}"
            )
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
        if (self.person_mask_manifest is None) != (self.person_mask_root is None):
            raise ValueError(
                "person_mask_manifest and person_mask_root must be provided together"
            )
        if self.require_person_masks and self.person_mask_manifest is None:
            raise ValueError(
                "production 3DGS training requires person_mask_manifest and person_mask_root"
            )
        if (
            self.lidar_range_weight > 0.0 or self.lidar_linear_aux_weight > 0.0
        ) and self.depth_manifest is None:
            raise ValueError(
                "positive LiDAR loss weight requires depth_manifest and depth_root"
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

    def contract_dict(self) -> dict[str, Any]:
        uses_lidar_linear_aux = self.lidar_linear_aux_weight > 0.0
        loss_weights = {
            "rgb_l1": self.rgb_l1_weight,
            "rgb_ssim": self.rgb_ssim_weight,
            "lidar_range": self.lidar_range_weight,
        }
        lidar_range_contract = {
            "mode": self.lidar_range_loss_mode,
            "semantics": "euclidean_ray_range_m",
            "log_huber_delta": self.lidar_log_range_huber_delta,
            "confidence_weighted": True,
        }
        if uses_lidar_linear_aux:
            loss_weights["lidar_linear_aux"] = self.lidar_linear_aux_weight
            lidar_range_contract["linear_aux_weight"] = self.lidar_linear_aux_weight
        strategy_contract = {
            "name": "MCMC",
            "refine_start_iter": self.mcmc_refine_start_iter,
            "refine_stop_iter": self.mcmc_refine_stop_iter,
            "refine_every": self.mcmc_refine_every,
            "noise_injection_stop_iter": self.mcmc_noise_injection_stop_iter,
            "noise_lr": self.mcmc_noise_lr,
            "noise_std_fraction": self.metric_scale_calibration.noise_std_fraction,
        }
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

        return {
            "schema_version": 2,
            "algorithm_version": "cloudstudio_gsplat_trainer_v5"
            if uses_lidar_linear_aux
            else "cloudstudio_gsplat_trainer_v4",
            "trainer_preset": self.trainer_preset,
            "seed": self.seed,
            "max_steps": self.max_steps,
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
            },
            "dynamic_person_mask": {
                "required": self.require_person_masks,
                "rgb_composition": "base_rgb_mask & ~person_dynamic_mask",
                "depth_composition": "base_depth_valid & ~person_dynamic_mask",
            },
            "learning_rates": dict(sorted(self.learning_rates.items())),
            "optimizer": {
                "configured_learning_rates": dict(sorted(self.learning_rates.items())),
                "means_step_fraction": self.metric_scale_calibration.means_step_fraction,
            },
            "face_split": {
                "enabled": self.face_cache_manifest is not None,
                "supervision": "pinhole_faces" if self.face_cache_manifest else "raw_fisheye",
                "validation": "raw_fisheye",
                "pinhole_rasterize_mode": self.pinhole_rasterize_mode,
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


def means_lr_for_step(
    base_learning_rate: float,
    final_factor: float,
    *,
    step: int,
    max_steps: int,
) -> float:
    """Australian parity schedule as a pure function for resume equivalence."""
    if base_learning_rate <= 0.0:
        raise ValueError("base means learning rate must be positive")
    if not 0.0 < final_factor <= 1.0:
        raise ValueError("means LR final factor must be in (0, 1]")
    if step < 0 or max_steps <= 0:
        raise ValueError("means LR schedule requires non-negative step and max_steps")
    return float(base_learning_rate * (final_factor ** (step / max(1, max_steps))))


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
    return result


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
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    has_range = "range_m" in tensors and (
        config.lidar_range_weight > 0.0 or config.lidar_linear_aux_weight > 0.0
    )
    rendered, rendered_range, _, info = backend.render(
        params,
        sample,
        with_range=has_range,
        c2w_override=c2w_override,
        active_sh_degree=active_sh_degree,
        background_rgb=config.background_color,
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
    if has_range:
        assert rendered_range is not None
        depth_scale = getattr(sample, "depth_to_range_scale", None)
        if depth_scale is not None:
            # Pinhole faces render z-depth; LiDAR supervision is Euclidean
            # ray range. The per-pixel ||K^-1 [u,v,1]|| factor converts.
            rendered_range = rendered_range * backend.torch.as_tensor(
                depth_scale, dtype=rendered_range.dtype, device=rendered_range.device
            )
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
        loss = loss + config.lidar_range_weight * range_loss
        if config.lidar_linear_aux_weight > 0.0:
            linear_range_aux_loss = confidence_weighted_range_l1(
                rendered_range,
                tensors["range_m"],
                tensors["confidence"],
                tensors["depth_mask"],
            )
            loss = loss + config.lidar_linear_aux_weight * linear_range_aux_loss
    info["cloudstudio_linear_range_aux_loss"] = (
        None if linear_range_aux_loss is None else linear_range_aux_loss.detach()
    )
    # Stashed for the error-weighted MCMC score update; detached, so it never
    # extends the autograd graph.
    info["cloudstudio_rendered_rgb"] = rendered.detach()
    return loss, l1, ssim, range_loss, info


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
            has_range = sample.depth_range_m is not None
            render_options = {"with_range": has_range}
            if background_rgb is not None:
                render_options["background_rgb"] = background_rgb
            rendered, rendered_range, _, _ = backend.render(
                params, sample, **render_options
            )
            prefix = artifact_root / sample.image_id
            reference_path = prefix.with_name(f"{sample.image_id}_reference.png")
            render_path = prefix.with_name(f"{sample.image_id}_rendered.png")
            mask_path = prefix.with_name(f"{sample.image_id}_mask.png")
            Image.fromarray(sample.image).save(reference_path, format="PNG", optimize=False)
            rendered_u8 = (
                rendered.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
            )
            Image.fromarray(rendered_u8).save(render_path, format="PNG", optimize=False)
            Image.fromarray(sample.rgb_mask.astype(np.uint8) * 255).save(
                mask_path, format="PNG", optimize=False
            )
            frame: dict[str, Any] = {
                "image_id": sample.image_id,
                "split": "val",
                "reference_rgb_path": reference_path.relative_to(output_dir).as_posix(),
                "rendered_rgb_path": render_path.relative_to(output_dir).as_posix(),
                "combined_mask_path": mask_path.relative_to(output_dir).as_posix(),
            }
            if has_range:
                assert rendered_range is not None and sample.depth_cache_path is not None
                rendered_depth_path = prefix.with_name(f"{sample.image_id}_range.npy")
                lidar_path = prefix.with_name(f"{sample.image_id}_lidar.npz")
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
    if len(xyz) >= config.cap_max:
        raise ValueError(
            f"initialization has {len(xyz)} Gaussians but cap_max is {config.cap_max}"
        )
    initial_scales_m, scale_calibration = build_metric_scale_calibration(
        xyz,
        policy=config.metric_scale_calibration,
        fixed_scale_m=config.init_scale_m,
        configured_means_lr=float(config.learning_rates["means"]),
        configured_noise_lr=config.mcmc_noise_lr,
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
    backend.pinhole_rasterize_mode = config.pinhole_rasterize_mode
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
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(config.seed)
    params, optimizers, strategy_state = backend.initialize(
        xyz,
        rgb,
        init_scales_m=initial_scales_m,
        learning_rates=effective_learning_rates,
        color_model=config.color_model,
        sh_degree=config.sh_degree,
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
    last_metrics: dict[str, Any] = {}
    initial_loss: float | None = None
    best_loss = float("inf")
    mcmc_telemetry: dict[str, Any] | None = None
    golden_history: list[dict[str, Any]] = []
    best_golden: dict[str, Any] | None = None
    full_evaluation_history: list[dict[str, Any]] = []
    if config.resume_checkpoint is not None:
        completed_steps, strategy_state, sampler_state, training_state = load_checkpoint(
            config.resume_checkpoint,
            expected_identity=checkpoint_identity,
            params=params,
            optimizers=optimizers,
            map_location=config.device,
            auxiliary_params=auxiliary_params,
            auxiliary_optimizers=auxiliary_optimizers,
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

    for step in range(completed_steps, config.max_steps):
        index = int(torch.randint(len(trainset), (1,), generator=sampler).item())
        sample = trainset[index]
        tensors = _tensor_sample(sample, torch, config.device)
        c2w_override = (
            None
            if pose_refiner is None
            else pose_refiner.apply(sample.rig_frame_id, sample.c2w)
        )
        if config.means_lr_final_factor < 1.0:
            # Deterministic function of the step (no scheduler state), so an
            # interrupted resume lands on exactly the same learning rate.
            decayed = means_lr_for_step(
                effective_learning_rates["means"],
                config.means_lr_final_factor,
                step=step,
                max_steps=config.max_steps,
            )
            for group in optimizers["means"].param_groups:
                group["lr"] = decayed
        active_sh_degree = active_sh_degree_for_step(config, step)
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
        regularization = geometry_regularization_terms(
            params,
            reference_scale_m=reference_scale_m,
            config=config.geometry_regularization,
        )
        loss = loss + regularization["total"]
        if normal_anchors is not None:
            if (
                normal_anchors.stale
                or step == 0
                or step % config.lidar_normal_alignment.refresh_every == 0
            ):
                normal_anchors.refresh(params["means"])
            loss = loss + normal_anchors.loss(params)["total"]
            normal_loss_steps_total += 1
            # loss() re-flags itself stale when the cached table no longer
            # matches the Gaussian count and contributes zero for that step.
            if normal_anchors.stale:
                normal_loss_steps_stale += 1
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
        backend.strategy_pre_step(
            params, optimizers, strategy_state, step=step, info=info
        )
        if (
            backend.needs_pre_backward
            and config.densification_gradient_source == "rgb_only"
        ):
            # Two-pass backward so the densification criterion sees the gradient
            # of L1+SSIM alone, as Kerbl et al. trained. One backward over the
            # total loss would let the LiDAR range and normal terms leak into
            # means2d.grad through the shared rasterization. The photometric
            # pass runs FIRST and its means2d gradients are snapshotted, because
            # .grad accumulates across passes while gsplat overwrites .absgrad
            # on each - only a snapshot survives both semantics. The optimizer
            # still steps on the full total-loss gradient (the two passes sum
            # in the leaf parameters).
            rgb_loss = config.rgb_l1_weight * l1 + config.rgb_ssim_weight * ssim
            rest_loss = loss - rgb_loss
            rgb_loss.backward(retain_graph=True)
            backend.strategy_isolate_gradient(info)
            if rest_loss.requires_grad:
                rest_loss.backward()
            backend.strategy_restore_gradient(info)
        else:
            loss.backward()
        refine_boundary = (
            step < config.mcmc_refine_stop_iter
            and step > config.mcmc_refine_start_iter
            and step % config.mcmc_refine_every == 0
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
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
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
        mcmc_event = backend.strategy_post_step(
            params,
            optimizers,
            strategy_state,
            step=step,
            info=info,
        )
        append_mcmc_telemetry(mcmc_telemetry, mcmc_event)
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
            "rgb_l1": float(l1.detach().cpu()),
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
            "rig_pose_prior": None
            if pose_prior is None
            else float(pose_prior.detach().cpu()),
            "geometry_regularization": float(regularization["total"].detach().cpu()),
            "opacity_sparsity": float(regularization["opacity_sparsity"].detach().cpu()),
            "scale_upper": float(regularization["scale_upper"].detach().cpu()),
            "scale_over_limit_fraction": float(
                regularization["scale_over_limit_fraction"].detach().cpu()
            ),
            "scale_upper_tail_count": int(
                regularization["scale_upper_tail_count"]
            ),
            "anisotropy": float(regularization["anisotropy"].detach().cpu()),
        }
        if initial_loss is None:
            initial_loss = last_metrics["loss"]
        best_loss = min(best_loss, last_metrics["loss"])
        completed = step + 1
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
    frames = _save_evaluation_artifacts(
        backend=backend,
        params=params,
        dataset=valset,
        output_dir=output_dir,
        background_rgb=config.background_color,
    )
    run_manifest = sign_run_manifest(
        {
            "schema_version": 1,
            "run_id": config.run_id,
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
            "frames": frames,
            "training": {
                "status": "COMPLETE",
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
                    "strategy": config.densification_strategy,
                    "gradient_source": config.densification_gradient_source,
                    "resolved": backend.strategy.state_dict()
                    if config.densification_strategy == "default_3dgs"
                    else None,
                },
                "mcmc_telemetry": mcmc_telemetry,
            },
        }
    )
    _atomic_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest


def train_from_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return train(TrainerConfig.from_dict(value))
