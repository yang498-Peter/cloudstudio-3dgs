#!/usr/bin/env python3
"""Build a signed A0-safe snow-Tile MCMC reallocation boundary arm."""

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
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    V27_SNOW_TILE_PROFILES,
    advance_adaptive_reallocation_gate,
)
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
    parser.add_argument("--warm-start-checkpoint", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--run-output", required=True, type=Path)
    args = parser.parse_args()

    config = _read(args.base_config)
    tile_id = int(config.get("mipmap_tile_id", -1))
    profile = V27_SNOW_TILE_PROFILES.get(tile_id)
    if profile is None:
        raise ValueError("base config does not target a supported snow Tile")
    run_id = args.run_id or f"snow-tile{tile_id}-v27a-a0-safe-mcmc-boundary602"
    config.update(
        {
            "run_id": str(run_id),
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_pipeline_gate": args.output_gate.resolve().as_posix(),
            "warm_start_checkpoint": args.warm_start_checkpoint.resolve().as_posix(),
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": False,
            "controlled_stop_after_steps": 602,
            "max_steps": profile["max_steps"],
            "checkpoint_every": profile["view_count"],
            "checkpoint_keep_every": 0,
            "factor": 1,
            "cap_max": profile["cap_max"],
            "pinhole_with_ut": True,
            "densification_strategy": "error_weighted_mcmc",
            "densification_gradient_source": "total_loss",
            "mcmc_refine_start_iter": 500,
            "mcmc_refine_every": 100,
            "mcmc_refine_stop_iter": 1000,
            "mcmc_noise_injection_stop_iter": 0,
            "mcmc_noise_lr": 0.0,
            "mcmc_growth_rate": 0.0,
            "mcmc_relocation_max_fraction": 0.02,
            "mcmc_relocation_max_scale_m": 0.08,
            "mcmc_relocation_max_anisotropy": 8.0,
            "topology_policy": {"mode": "adaptive_growth"},
            "fixed_topology_schedule": {"enabled": False},
            # A0 already contains good surface geometry. Fresh optimizers use
            # deliberately reduced rates so the only abrupt geometric action
            # is the bounded, auditable MCMC relocation event.
            "learning_rates": {
                "means": 4.0e-6,
                "scales": 1.0e-3,
                "quats": 2.0e-4,
                "opacities": 1.0e-2,
                "colors": 1.0e-3,
            },
            "means_lr_final_factor": 0.1,
            "error_weighted_sampling": {
                "enabled": True,
                "ema_decay": 0.95,
                "score_power": 0.4,
                "min_score_floor": 0.001,
                "aggregation": "contribution",
                "footprint_radius_px": 4,
            },
            "contribution": {
                "enabled": True,
                "error_map_mode": "l1",
                "ssim_weight": 0.5,
                "ssim_window": 11,
                "ssim_sigma": 1.5,
                "normalize": True,
                "eps": 1e-8,
            },
            "contribution_every": 5,
            "lidar_admission": {
                "enabled": True,
                "mode": "soft",
                "sigma_perp_factor": 1.0,
                "weight_floor": 0.05,
                "refresh_every": 500,
                "gate_tangent_factor": 3.0,
                "share_normal_field": False,
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
                "additive_births": True,
                "birth_opacity": 0.02,
                "reject_unsupported_births": True,
            },
            "exposure_compensation": {
                "enabled": True,
                "learning_rate": 1.0e-5,
                "regularization_weight": 0.01,
                "max_abs_log_gain": 0.6931471805599453,
                "zero_mean_projection": False,
                "mean_anchor_weight": 0.0,
                "mean_anchor_beta": 0.1,
            },
            "geometry_regularization": {
                "enabled": True,
                "opacity_sparsity_weight": 0.0001,
                "scale_upper_weight": 0.0001,
                "scale_upper_tail_fraction": 0.01,
                "anisotropy_weight": 0.0001,
                "max_scale_ratio_to_reference": 8.0,
                "max_anisotropy": 8.0,
                "screen_clip_enabled": False,
                "max_world_size_m": None,
            },
            "da2_depth_weight": 0.0,
            "rig_pose_refinement": {"enabled": False},
            "color_model": "sh",
            "sh_degree": 0,
            "sh_degree_interval": 0,
        }
    )
    config.pop("resume_checkpoint", None)
    config.pop("default_strategy", None)
    config.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest()
    gate = advance_adaptive_reallocation_gate(
        _read(args.upstream_gate), config, stage="boundary"
    )
    _write(args.output_gate, gate)
    _write(args.output_config, config)
    TrainerConfig.from_dict(config).validate()
    print(
        "V27a MCMC boundary ready: "
        f"config_sha256={config['config_manifest_sha256']}, "
        f"gate_sha256={gate['gate_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
