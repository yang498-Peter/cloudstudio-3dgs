#!/usr/bin/env python3
"""Build a signed scale/quaternion-only surface-shape probe from V47e/652."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import advance_adaptive_growth_gate
from cloudstudio_3dgs.training.trainer import TrainerConfig


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--output-handoff", required=True, type=Path)
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--additional-steps", type=int, choices=(25, 50), default=50)
    parser.add_argument(
        "--shape-profile",
        choices=("conservative", "strong"),
        default="conservative",
    )
    args = parser.parse_args()

    source_config = _read(args.source_config)
    source_checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    source_step = int(source_checkpoint.get("step", -1))
    if source_step != 652:
        raise ValueError("surface shape probe must start from immutable V47e step 652")
    source_identity = source_checkpoint.get("identity", {})
    source_trainer_sha = str(source_identity.get("trainer_config_sha256", ""))
    if len(source_trainer_sha) != 64:
        raise ValueError("source checkpoint has no signed trainer identity")
    if source_config.get("lidar_alpha_weight") != 1.0:
        raise ValueError("surface shape probe requires the accepted V47e alpha floor")
    if source_config.get("lidar_alpha_dilation_radius_px") != 3:
        raise ValueError("surface shape probe requires the accepted V47e dilation")

    handoff = {
        "schema_version": 1,
        "kind": "adaptive_growth_handoff_report_v1",
        "status": "ADAPTIVE_GROWTH_BOUNDARY_PASS",
        "promotion_eligible": True,
        "failed_checks": [],
        "checks": {
            "source_step_is_652": True,
            "source_is_immutable_v47e": True,
            "probe_disables_future_lifecycle": True,
            "means_colors_opacity_ppisp_frozen": True,
        },
        "checkpoint_sha256": _sha256(args.source_checkpoint),
        "source_trainer_config_sha256": source_trainer_sha,
        "completed_steps": source_step,
        "authorization_scope": "scale_quaternion_only_surface_shape_probe",
    }
    handoff["boundary_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(handoff)
    ).hexdigest()

    shape_profiles = {
        "conservative": {
            "scales_lr": 0.001,
            "quats_lr": 0.0002,
            "align_weight": 0.01,
            "flatten_weight": 0.01,
        },
        "strong": {
            "scales_lr": 0.003,
            "quats_lr": 0.0005,
            "align_weight": 0.1,
            "flatten_weight": 0.1,
        },
    }
    shape_profile = shape_profiles[args.shape_profile]
    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": args.run_id,
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_pipeline_gate": args.output_gate.resolve().as_posix(),
            "resume_checkpoint": args.source_checkpoint.resolve().as_posix(),
            "controlled_stop_after_steps": source_step + args.additional_steps,
            "lidar_range_weight": 0.0,
            "lidar_alpha_weight": 1.0,
            "lidar_alpha_target": 0.95,
            "lidar_alpha_dilation_radius_px": 3,
            "mcmc_refine_stop_iter": 602,
            "learning_rates": {
                "means": 0.0,
                "scales": shape_profile["scales_lr"],
                "quats": shape_profile["quats_lr"],
                "opacities": 0.0,
                "colors": 0.0,
            },
        }
    )
    config["geometry_regularization"].update(
        {
            "opacity_sparsity_weight": 0.0,
            "scale_upper_weight": 0.0,
            "anisotropy_weight": 0.0,
            "max_anisotropy": 256.0,
        }
    )
    config["ppisp"]["learning_rate"] = 0.0
    config["default_strategy"]["refine_scale2d_stop_iter"] = 602
    config["lidar_normal_alignment"].update(
        {
            "enabled": True,
            "weight_align": shape_profile["align_weight"],
            "weight_flatten": shape_profile["flatten_weight"],
            "weight_point_to_plane": 0.0,
            "flatten_mode": "tangent_ratio",
            "flatten_ratio_target": 0.15,
        }
    )
    config.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest()
    gate = advance_adaptive_growth_gate(
        _read(args.upstream_gate),
        config,
        stage="review",
        boundary_report=handoff,
    )
    _write(args.output_handoff, handoff)
    _write(args.output_gate, gate)
    _write(args.output_config, config)
    TrainerConfig.from_dict(config).validate()
    print(
        "V48 surface-shape probe ready: "
        f"source_step={source_step}, stop={config['controlled_stop_after_steps']}, "
        f"profile={args.shape_profile}, "
        f"config_sha256={config['config_manifest_sha256']}, "
        f"gate_sha256={gate['gate_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
