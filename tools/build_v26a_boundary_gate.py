#!/usr/bin/env python3
"""Build a signed Tile_1 502-step classic-growth boundary arm and gate.

The optional PPISP profile keeps the proven V26 LiDAR-guarded lifecycle while
replacing its scalar exposure learner with the recovered per-view nuisance
model used by the competitor-cross-checked V33 route. The vendor lifecycle can
also bind an explicit 0.05 warm-up cull calibration; it is labelled as a
CloudStudio compatibility deviation rather than vendor-exact behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import advance_adaptive_growth_gate
from cloudstudio_3dgs.training.trainer import TrainerConfig


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument(
        "--run-id",
        default="snow-tile1-v26a-classic-lidar-boundary502",
    )
    parser.add_argument(
        "--ppisp-per-image",
        action="store_true",
        help="replace scalar exposure with scheduled no-CRF per-image PPISP",
    )
    parser.add_argument(
        "--observation-aware-cull",
        action="store_true",
        help=(
            "protect opacity culls with observation count, consecutive-event "
            "and post-reset grace gates, capped at five percent per event"
        ),
    )
    parser.add_argument(
        "--vendor-pre-optimizer-lifecycle",
        action="store_true",
        help=(
            "use the recovered backward -> Split/Clone/Cull -> Adam order and "
            "disable CloudStudio birth/cull/shape/alpha enhancements"
        ),
    )
    parser.add_argument(
        "--vendor-cull-warmup-0p05",
        action="store_true",
        help=(
            "keep the vendor pre-optimizer lifecycle but use the recovered "
            "0.05 base opacity threshold during warm-up instead of the "
            "vendor early 0.10 threshold"
        ),
    )
    parser.add_argument(
        "--vendor-cap-aware-cull",
        action="store_true",
        help=(
            "when the signed absolute Gaussian cap blocks births, use the "
            "recovered relaxed cull thresholds (opacity x0.25, world/screen x5)"
        ),
    )
    parser.add_argument(
        "--snow-tile1-cap-probe-985k",
        action="store_true",
        help=(
            "diagnostic only: cap Snow Tile_1 at 985,000 rows so the first "
            "birth event reaches the absolute budget and exercises relaxed cull"
        ),
    )
    parser.add_argument(
        "--cap-aware-scale-guard-0p2m",
        action="store_true",
        help=(
            "CloudStudio safety enhancement: retain the vendor cap-aware "
            "opacity cull but clamp every Gaussian maximum world axis to "
            "0.2 m instead of allowing the relaxed 1.0 m cull threshold"
        ),
    )
    parser.add_argument(
        "--cap-aware-near-cap-0p99",
        action="store_true",
        help=(
            "CloudStudio stability enhancement: keep relaxed capacity "
            "maintenance active while the post-growth population remains "
            "at or above 99 percent of the signed absolute cap"
        ),
    )
    parser.add_argument(
        "--vendor-geometry-cull-only",
        action="store_true",
        help=(
            "disable opacity culling while retaining the vendor world/screen "
            "geometry culls; intended for one-event coverage-safe probes"
        ),
    )
    parser.add_argument(
        "--capacity-conserving-clone-opacity",
        action="store_true",
        help=(
            "split the opacity budget between a cloned parent and child so "
            "their coincident composite alpha equals the pre-clone alpha"
        ),
    )
    parser.add_argument(
        "--visible-opacity-sparsity",
        action="store_true",
        help=(
            "apply the 0.01 opacity-mean loss only to Gaussians visible in "
            "the current cropped Tile view; keeps vendor topology order but "
            "removes the LiDAR initialization's unrelated-view opacity bias"
        ),
    )
    parser.add_argument(
        "--vendor-opacity-reset-profile",
        choices=("exact_every300", "deferred_every3000_compatibility"),
        default="exact_every300",
        help=(
            "bind the vendor opacity reset cadence; the deferred profile is "
            "a coverage-preserving diagnostic deviation"
        ),
    )
    parser.add_argument(
        "--vendor-opacity-sparsity-weight",
        choices=(0.01, 0.001, 0.0),
        default=0.01,
        type=float,
        help=(
            "bind the opacity sparsity strength; 0.001 is the signed PPISP "
            "calibration and 0 disables it for an alpha-protected growth probe"
        ),
    )
    parser.add_argument(
        "--optimization-profile",
        choices=("inherited", "surface_detail", "frozen_geometry_growth_signal"),
        default="inherited",
        help=(
            "optionally replace inherited learning rates with the bounded "
            "V45 surface-detail rates while preserving the auxiliary layout"
        ),
    )
    parser.add_argument(
        "--local-coverage-cull",
        action="store_true",
        help=(
            "use observation-aware culling but preserve the strongest local "
            "representative in each 2 cm world voxel"
        ),
    )
    parser.add_argument(
        "--coverage-weighted-local-protection",
        action="store_true",
        help=(
            "within each local cull voxel protect the row with the strongest "
            "opacity times tangential surface area"
        ),
    )
    parser.add_argument(
        "--local-alpha-budget",
        action="store_true",
        help=(
            "protect enough rows per 2 cm cell to retain at least 0.5 "
            "composited local alpha before opacity culling"
        ),
    )
    parser.add_argument(
        "--opacity-cull-max-fraction",
        type=float,
        choices=(0.02, 0.05),
        default=0.05,
        help="maximum fraction removed by the opacity gate at one event",
    )
    parser.add_argument(
        "--detail-split-2cm",
        action="store_true",
        help=(
            "split high-gradient LiDAR-supported gaussians above two centimetres "
            "only when their observed raster radius exceeds 0.35 percent, and "
            "prioritize low-opacity large-footprint culls"
        ),
    )
    parser.add_argument(
        "--gradient-profile",
        choices=(
            "legacy_absgrad_1p5e4",
            "vendor_plain_1p5e4",
            "calibrated_plain_1e4",
            "calibrated_plain_7p5e5",
            "absgrad_4e4",
            "absgrad_8e4",
            "absgrad_1p2e3",
        ),
        default="legacy_absgrad_1p5e4",
        help="bind projected-gradient semantics and its calibrated threshold",
    )
    parser.add_argument(
        "--thin-surfel-shape",
        action="store_true",
        help="disable the axis-ratio penalty that conflicts with competitor thin disks",
    )
    parser.add_argument(
        "--soft-point-to-plane",
        action="store_true",
        help=(
            "penalize only LiDAR-normal displacement with a 2 cm Huber tether; "
            "tangential center motion remains free"
        ),
    )
    parser.add_argument(
        "--lidar-alpha-coverage",
        action="store_true",
        help=(
            "penalize opacity deficits below 0.95 at signed LiDAR depth pixels"
        ),
    )
    parser.add_argument(
        "--preserve-inherited-lidar-alpha",
        action="store_true",
        help=(
            "retain an already signed LiDAR alpha profile from the base config "
            "during a vendor-order coverage-safe growth probe"
        ),
    )
    parser.add_argument(
        "--lidar-range-supervision",
        action="store_true",
        help=(
            "restore the signed 0.05 real-LiDAR range loss during the "
            "vendor-order growth warm-up"
        ),
    )
    parser.add_argument(
        "--flatten-target-m",
        type=float,
        default=None,
        help="optional shortest-axis target for the LiDAR planar prior",
    )
    parser.add_argument(
        "--flatten-ratio-target",
        type=float,
        default=None,
        help=(
            "optional shortest-axis over tangential-geometric-mean target "
            "for scale-invariant surface disks"
        ),
    )
    parser.add_argument(
        "--flatten-weight",
        type=float,
        default=None,
        help="optional planar shortest-axis loss weight",
    )
    args = parser.parse_args()
    if args.resume_checkpoint is not None and args.warm_start_checkpoint is not None:
        parser.error(
            "--resume-checkpoint and --warm-start-checkpoint are mutually exclusive"
        )
    if args.vendor_cull_warmup_0p05 and args.vendor_geometry_cull_only:
        parser.error(
            "--vendor-cull-warmup-0p05 and --vendor-geometry-cull-only are mutually exclusive"
        )
    if args.vendor_cull_warmup_0p05 and not args.vendor_pre_optimizer_lifecycle:
        parser.error(
            "--vendor-cull-warmup-0p05 requires "
            "--vendor-pre-optimizer-lifecycle"
        )
    if args.vendor_cap_aware_cull and not args.vendor_pre_optimizer_lifecycle:
        parser.error(
            "--vendor-cap-aware-cull requires "
            "--vendor-pre-optimizer-lifecycle"
        )
    if args.snow_tile1_cap_probe_985k and not args.vendor_cap_aware_cull:
        parser.error(
            "--snow-tile1-cap-probe-985k requires --vendor-cap-aware-cull"
        )
    if args.cap_aware_scale_guard_0p2m and not args.vendor_cap_aware_cull:
        parser.error(
            "--cap-aware-scale-guard-0p2m requires --vendor-cap-aware-cull"
        )
    if args.cap_aware_near_cap_0p99 and not args.vendor_cap_aware_cull:
        parser.error(
            "--cap-aware-near-cap-0p99 requires --vendor-cap-aware-cull"
        )
    if args.visible_opacity_sparsity and not args.vendor_pre_optimizer_lifecycle:
        parser.error(
            "--visible-opacity-sparsity requires "
            "--vendor-pre-optimizer-lifecycle"
        )
    if (
        args.vendor_opacity_reset_profile != "exact_every300"
        and not args.vendor_pre_optimizer_lifecycle
    ):
        parser.error(
            "a non-default vendor opacity reset profile requires "
            "--vendor-pre-optimizer-lifecycle"
        )
    if args.vendor_opacity_sparsity_weight not in {0.01, 0.001, 0.0}:
        parser.error("unsupported vendor opacity sparsity weight")
    if args.vendor_opacity_sparsity_weight == 0.001 and not (
        args.vendor_pre_optimizer_lifecycle and args.visible_opacity_sparsity
    ):
        parser.error(
            "a calibrated vendor opacity sparsity weight requires the vendor "
            "pre-optimizer lifecycle and visible opacity sparsity"
        )
    if args.vendor_opacity_sparsity_weight == 0.0 and not (
        args.vendor_pre_optimizer_lifecycle
    ):
        parser.error(
            "disabled vendor opacity sparsity requires the vendor "
            "pre-optimizer lifecycle"
        )

    config = _read(args.base_config)
    config.update(
        {
            "run_id": args.run_id,
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_pipeline_gate": args.output_gate.resolve().as_posix(),
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": False,
            "controlled_stop_after_steps": 502,
            "max_steps": 7480,
            "checkpoint_every": 374,
            "checkpoint_keep_every": 0,
            "factor": 1,
            # Face4 is a true pinhole camera.  The classic EWA path is
            # required here because DefaultStrategy/AbsGS consumes the
            # projected means2d gradient; eval3d/UT does not expose it.
            "pinhole_with_ut": False,
            "cap_max": (
                985_000 if args.snow_tile1_cap_probe_985k else 2_200_000
            ),
            "densification_strategy": "default_3dgs",
            "densification_gradient_source": "rgb_only",
            "mcmc_refine_start_iter": 500,
            "mcmc_refine_every": 100,
            "mcmc_refine_stop_iter": 5610,
            "mcmc_noise_injection_stop_iter": 0,
            "mcmc_noise_lr": 0.0,
            "topology_policy": {"mode": "adaptive_growth"},
            "fixed_topology_schedule": {"enabled": False},
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
                "refine_scale2d_stop_iter": 5610,
                "reset_every": 300,
                "reset_opacity_cap": 0.2,
                "absgrad": True,
                "revised_opacity": True,
                "capacity_conserving_clone_opacity": False,
            },
            "tangent_proposal": {
                "enabled": True,
                "mode": "tangent",
                "planarity_gate": 0.6,
                "support_gate": 0.1,
                "support_tangent_factor": 3.0,
                "sigma_perp_factor": 1.0,
                "tangent_sigma_factor": 0.5,
                "normal_offset_factor": 0.1,
                "init_shortest_axis": True,
                "thickness_factor": 0.5,
                "min_thickness_m": 0.001,
                "reject_unsupported_births": True,
            },
            "lidar_admission": {"enabled": False},
            "error_weighted_sampling": {"enabled": False},
            "geometry_regularization": {
                "enabled": True,
                "opacity_sparsity_weight": 0.0001,
                "scale_upper_weight": 0.0001,
                "anisotropy_weight": 0.0001,
                "max_scale_ratio_to_reference": 8.0,
                "max_anisotropy": 10.0,
            },
            "da2_depth_weight": 0.0,
            "rig_pose_refinement": {"enabled": False},
            "color_model": "sh",
            "sh_degree": 0,
            "sh_degree_interval": 0,
        }
    )
    if args.ppisp_per_image:
        config["exposure_compensation"] = {
            "enabled": False,
            "learning_rate": 0.005,
            "regularization_weight": 0.01,
            "max_abs_log_gain": 0.6931471805599453,
            "zero_mean_projection": False,
            "mean_anchor_weight": 0.0,
            "mean_anchor_beta": 0.1,
        }
        config["ppisp"] = {
            "enabled": True,
            "param_type": "no_crf",
            "mode": "per_image",
            "learning_rate": 0.002,
            "lr_schedule": "linear_warmup_exponential_decay",
            "warmup_fraction": 1.0 / 30.0,
            "warmup_start_multiplier": 0.01,
            "final_lr_multiplier": 0.01,
            "exposure_mean_weight": 1.0,
            "vig_center_weight": 0.02,
            "vig_channel_weight": 0.1,
            "vig_non_pos_weight": 0.01,
            "color_mean_weight": 1.0,
            "crf_channel_weight": 0.1,
        }
    if args.optimization_profile == "surface_detail":
        config["learning_rates"] = {
            "means": 1.6e-5,
            "scales": 0.005,
            "quats": 0.001,
            "opacities": 0.05,
            "colors": 0.0025,
        }
        if config.get("exposure_compensation", {}).get("enabled") is True:
            config["exposure_compensation"]["learning_rate"] = 0.005
    elif args.optimization_profile == "frozen_geometry_growth_signal":
        config["learning_rates"] = {
            "means": 0.0,
            "scales": 0.0,
            "quats": 0.0,
            "opacities": 0.01,
            "colors": 0.001,
        }
        if config.get("exposure_compensation", {}).get("enabled") is True:
            config["exposure_compensation"]["learning_rate"] = 0.001
    if args.lidar_range_supervision:
        config["lidar_range_weight"] = 0.05
    if args.observation_aware_cull or args.local_coverage_cull:
        config["default_strategy"].update(
            {
                "opacity_cull_policy": (
                    "local_coverage_competition"
                    if args.local_coverage_cull
                    else "observation_aware"
                ),
                "opacity_cull_min_observations": 4,
                "opacity_cull_consecutive_events": 2,
                "opacity_cull_grace_after_reset_steps": (
                    100 if args.local_coverage_cull else 200
                ),
                "opacity_cull_max_fraction": args.opacity_cull_max_fraction,
            }
        )
        if args.local_coverage_cull:
            config["default_strategy"]["opacity_cull_local_voxel_m"] = 0.02
            config["default_strategy"]["opacity_cull_local_protection"] = (
                "opacity_tangent_area"
                if args.coverage_weighted_local_protection
                else "opacity"
            )
            config["default_strategy"][
                "opacity_cull_local_min_accumulated_alpha"
            ] = 0.5 if args.local_alpha_budget else 0.0
    elif args.opacity_cull_max_fraction != 0.05:
        parser.error(
            "--opacity-cull-max-fraction requires an observation-aware cull profile"
        )
    if args.coverage_weighted_local_protection and not args.local_coverage_cull:
        parser.error(
            "--coverage-weighted-local-protection requires --local-coverage-cull"
        )
    if args.local_alpha_budget and not args.local_coverage_cull:
        parser.error("--local-alpha-budget requires --local-coverage-cull")
    if args.detail_split_2cm:
        if not (args.observation_aware_cull or args.local_coverage_cull):
            parser.error(
                "--detail-split-2cm requires an observation-aware cull profile"
            )
        config["default_strategy"].update(
            {
                "detail_split_policy": "lidar_surface_screen_detail",
                "detail_split_scale_m": 0.02,
                "detail_split_screen_radius": 0.0035,
                "opacity_cull_priority": "lowest_opacity_per_footprint",
            }
        )
    gradient_profiles = {
        "legacy_absgrad_1p5e4": (True, 0.00015),
        "vendor_plain_1p5e4": (False, 0.00015),
        "calibrated_plain_1e4": (False, 0.0001),
        "calibrated_plain_7p5e5": (False, 0.000075),
        "absgrad_4e4": (True, 0.0004),
        "absgrad_8e4": (True, 0.0008),
        "absgrad_1p2e3": (True, 0.0012),
    }
    absgrad, grow_grad2d = gradient_profiles[args.gradient_profile]
    config["default_strategy"].update(
        {"absgrad": absgrad, "grow_grad2d": grow_grad2d}
    )
    if args.thin_surfel_shape:
        config["geometry_regularization"].update(
            {"anisotropy_weight": 0.0, "max_anisotropy": 256.0}
        )
    if args.soft_point_to_plane:
        config.setdefault("lidar_normal_alignment", {}).update(
            {
                "enabled": True,
                "weight_point_to_plane": 0.01,
                "point_to_plane_huber_delta_m": 0.02,
            }
        )
    if args.lidar_alpha_coverage:
        config.update(
            {
                "lidar_alpha_weight": 0.02,
                "lidar_alpha_target": 0.95,
            }
        )
    flatten_targets = sum(
        value is not None
        for value in (args.flatten_target_m, args.flatten_ratio_target)
    )
    if flatten_targets > 1:
        parser.error("choose only one flatten target mode")
    if (flatten_targets == 0) != (args.flatten_weight is None):
        parser.error("a flatten target and --flatten-weight must be set together")
    if flatten_targets:
        target = (
            args.flatten_target_m
            if args.flatten_target_m is not None
            else args.flatten_ratio_target
        )
        if target <= 0.0 or args.flatten_weight < 0.0:
            parser.error("flatten target must be positive and weight non-negative")
        if args.flatten_ratio_target is not None and args.flatten_ratio_target >= 1.0:
            parser.error("flatten ratio target must be below one")
        config.setdefault("lidar_normal_alignment", {}).update(
            {
                "enabled": True,
                "weight_flatten": args.flatten_weight,
            }
        )
        if args.flatten_target_m is not None:
            config["lidar_normal_alignment"].update(
                {
                    "flatten_mode": "absolute_m",
                    "flatten_target_m": args.flatten_target_m,
                }
            )
        else:
            config["lidar_normal_alignment"].update(
                {
                    "flatten_mode": "tangent_ratio",
                    "flatten_ratio_target": args.flatten_ratio_target,
                }
            )
    if args.vendor_pre_optimizer_lifecycle:
        vendor_gradient_thresholds = {
            "legacy_absgrad_1p5e4": 0.00015,
            "vendor_plain_1p5e4": 0.00015,
            "calibrated_plain_1e4": 0.0001,
            "calibrated_plain_7p5e5": 0.000075,
        }
        if args.gradient_profile not in vendor_gradient_thresholds:
            parser.error(
                "vendor pre-optimizer lifecycle requires a plain-gradient profile"
            )
        incompatible = {
            "--observation-aware-cull": args.observation_aware_cull,
            "--local-coverage-cull": args.local_coverage_cull,
            "--coverage-weighted-local-protection": (
                args.coverage_weighted_local_protection
            ),
            "--local-alpha-budget": args.local_alpha_budget,
            "--detail-split-2cm": args.detail_split_2cm,
            "--thin-surfel-shape": args.thin_surfel_shape,
            "--soft-point-to-plane": args.soft_point_to_plane,
            "--flatten-target-m/--flatten-ratio-target": bool(flatten_targets),
        }
        selected = [name for name, enabled in incompatible.items() if enabled]
        if selected:
            parser.error(
                "--vendor-pre-optimizer-lifecycle is incompatible with "
                + ", ".join(selected)
            )
        config["densification_gradient_source"] = "total_loss"
        config["default_strategy"].update(
            {
                "lifecycle_execution_order": "pre_optimizer_vendor",
                "absgrad": False,
                "grow_grad2d": vendor_gradient_thresholds[
                    args.gradient_profile
                ],
                "revised_opacity": False,
                "detail_split_policy": "vendor_0_2m",
                "opacity_cull_policy": "immediate",
                "opacity_cull_min_observations": 0,
                "opacity_cull_consecutive_events": 1,
                "opacity_cull_grace_after_reset_steps": 0,
                "opacity_cull_max_fraction": 1.0,
                "opacity_cull_priority": "lowest_opacity",
                "opacity_cull_local_min_accumulated_alpha": 0.0,
                "vendor_cull_warmup_profile": (
                    "calibrated_geometry_only_0p00"
                    if args.vendor_geometry_cull_only
                    else (
                        "compatibility_uniform_0p05"
                        if args.vendor_cull_warmup_0p05
                        else "exact_0p10_to_0p05"
                    )
                ),
                "vendor_capacity_cull_profile": (
                    "cloudstudio_relaxed_near_cap_0p99"
                    if args.cap_aware_near_cap_0p99
                    else "exact_relaxed_at_cap"
                    if args.vendor_cap_aware_cull
                    else "disabled"
                ),
                "vendor_opacity_reset_profile": (
                    args.vendor_opacity_reset_profile
                ),
                "reset_every": (
                    3000
                    if args.vendor_opacity_reset_profile
                    == "deferred_every3000_compatibility"
                    else 300
                ),
            }
        )
        if args.vendor_geometry_cull_only:
            config["default_strategy"].update(
                {"prune_opa": 0.0, "prune_opa_late": 0.0}
            )
        elif args.vendor_cull_warmup_0p05:
            config["default_strategy"].update(
                {"prune_opa": 0.05, "prune_opa_late": 0.05}
            )
        config["tangent_proposal"] = {"enabled": False}
        config["lidar_admission"] = {"enabled": False}
        if args.lidar_alpha_coverage:
            config["lidar_alpha_weight"] = 0.02
            config["lidar_alpha_target"] = 0.95
            config["lidar_alpha_dilation_radius_px"] = 0
        elif not args.preserve_inherited_lidar_alpha:
            config["lidar_alpha_weight"] = 0.0
            config["lidar_alpha_dilation_radius_px"] = 0
        config["geometry_regularization"].update(
            {
                "opacity_sparsity_weight": (
                    args.vendor_opacity_sparsity_weight
                ),
                "scale_upper_weight": 0.0,
                "anisotropy_weight": 0.0,
                "max_anisotropy": 256.0,
                "screen_clip_enabled": False,
                "max_world_size_m": (
                    0.2 if args.cap_aware_scale_guard_0p2m else None
                ),
            }
        )
        if args.visible_opacity_sparsity:
            config["geometry_regularization"]["opacity_sparsity_scope"] = (
                "visible_current_view"
            )
        config.setdefault("lidar_normal_alignment", {}).update(
            {
                "weight_point_to_plane": 0.0,
                "weight_flatten": 0.0,
                "flatten_mode": "absolute_m",
                "flatten_target_m": 0.02,
            }
        )
    if args.capacity_conserving_clone_opacity:
        if not args.vendor_pre_optimizer_lifecycle:
            parser.error(
                "--capacity-conserving-clone-opacity currently requires "
                "--vendor-pre-optimizer-lifecycle"
            )
        config["default_strategy"][
            "capacity_conserving_clone_opacity"
        ] = True
    config.pop("config_manifest_sha256", None)
    if args.resume_checkpoint is not None:
        config["resume_checkpoint"] = args.resume_checkpoint.resolve().as_posix()
    else:
        config.pop("resume_checkpoint", None)
    if args.warm_start_checkpoint is not None:
        config["warm_start_checkpoint"] = (
            args.warm_start_checkpoint.resolve().as_posix()
        )
    else:
        config.pop("warm_start_checkpoint", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest()
    gate = advance_adaptive_growth_gate(
        _read(args.upstream_gate), config, stage="boundary"
    )
    _write(args.output_gate, gate)
    _write(args.output_config, config)
    TrainerConfig.from_dict(config).validate()
    print(
        "V26a boundary ready: "
        f"config_sha256={config['config_manifest_sha256']}, "
        f"gate_sha256={gate['gate_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
