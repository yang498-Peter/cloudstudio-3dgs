#!/usr/bin/env python3
"""Build the signed Tile_1 V26a 502-step classic-growth boundary arm and gate."""

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
    args = parser.parse_args()

    config = _read(args.base_config)
    config.update(
        {
            "run_id": "snow-tile1-v26a-classic-lidar-boundary502",
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
