#!/usr/bin/env python3
"""Build a signed Tile_1 502-step classic-growth boundary arm and gate.

The optional PPISP profile keeps the proven V26 LiDAR-guarded lifecycle while
replacing its scalar exposure learner with the recovered per-view nuisance
model used by the competitor-cross-checked V33 route.
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
    args = parser.parse_args()

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
            "cap_max": 2_200_000,
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
    if args.observation_aware_cull:
        config["default_strategy"].update(
            {
                "opacity_cull_policy": "observation_aware",
                "opacity_cull_min_observations": 4,
                "opacity_cull_consecutive_events": 2,
                "opacity_cull_grace_after_reset_steps": 200,
                "opacity_cull_max_fraction": 0.05,
            }
        )
    if args.detail_split_2cm:
        if not args.observation_aware_cull:
            parser.error("--detail-split-2cm requires --observation-aware-cull")
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
    config.pop("config_manifest_sha256", None)
    if args.resume_checkpoint is not None:
        config["resume_checkpoint"] = args.resume_checkpoint.resolve().as_posix()
    else:
        config.pop("resume_checkpoint", None)
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
