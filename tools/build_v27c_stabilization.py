#!/usr/bin/env python3
"""Build a scale-sanitized, geometry-frozen V27c stabilization arm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import advance_adaptive_reallocation_gate
from cloudstudio_3dgs.pipeline.mipmap_gate import V27_SNOW_TILE_PROFILES
from cloudstudio_3dgs.training.runtime_evidence import snapshot_gaussians
from cloudstudio_3dgs.training.trainer import TrainerConfig


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-sanitization-report", required=True, type=Path)
    parser.add_argument("--output-handoff-report", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--max-scale-m", type=float, default=0.08)
    parser.add_argument("--controlled-stop", type=int)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.max_scale_m <= 0.0:
        raise ValueError("max-scale-m must be positive")
    source_config = _read(args.source_config)
    tile_id = int(source_config.get("mipmap_tile_id", -1))
    profile = V27_SNOW_TILE_PROFILES.get(tile_id)
    if profile is None:
        raise ValueError("source config does not target a supported snow Tile")
    controlled_stop = args.controlled_stop or profile["stabilization_stop"]
    if controlled_stop != profile["stabilization_stop"]:
        raise ValueError("controlled stop differs from the signed Tile profile")
    run_id = args.run_id or (
        f"snow-tile{tile_id}-v27c-stabilization{controlled_stop}"
    )
    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    if int(checkpoint.get("step", -1)) != profile["review_stop"]:
        raise ValueError(
            "V27c stabilization must start from the signed seven-epoch review"
        )
    gaussian_count = int((checkpoint.get("params") or checkpoint.get("splats"))["means"].shape[0])
    if gaussian_count != profile["gaussian_count"]:
        raise ValueError("source checkpoint Gaussian count differs from Tile profile")
    params = checkpoint.get("params") or checkpoint.get("splats")
    if not isinstance(params, dict) or "scales" not in params:
        raise ValueError("source checkpoint has no Gaussian scales")

    before = snapshot_gaussians(params, min_opacity=0.005)
    linear_scales = torch.exp(params["scales"].detach())
    clamped_axes = linear_scales > float(args.max_scale_m)
    clamped_gaussians = clamped_axes.any(dim=-1)
    with torch.no_grad():
        params["scales"].clamp_(max=math.log(float(args.max_scale_m)))
    scale_optimizer = checkpoint.get("optimizers", {}).get("scales")
    if not isinstance(scale_optimizer, dict):
        raise ValueError("source checkpoint has no scale optimizer state")
    # The old Adam moments point in the direction that regrew metre-scale
    # splats. Geometry is frozen in V27c, and clearing these moments prevents a
    # later intentional unfreeze from immediately replaying that stale update.
    scale_optimizer["state"] = {}
    after = snapshot_gaussians(params, min_opacity=0.005)
    checkpoint.setdefault("training_state", {}).setdefault(
        "mcmc_telemetry", {}
    )["last_snapshot"] = after
    _save_checkpoint(args.output_checkpoint, checkpoint)

    sanitization = {
        "schema_version": 1,
        "kind": "v27c_scale_tail_sanitization_v1",
        "status": "PASS",
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "output_checkpoint": args.output_checkpoint.resolve().as_posix(),
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "completed_steps": int(checkpoint["step"]),
        "max_scale_m": float(args.max_scale_m),
        "clamped_gaussian_count": int(clamped_gaussians.sum().item()),
        "clamped_gaussian_fraction": float(clamped_gaussians.float().mean().item()),
        "clamped_axis_count": int(clamped_axes.sum().item()),
        "scale_optimizer_state_cleared": True,
        "before": before,
        "after": after,
    }
    sanitization["sanitization_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(sanitization)
    ).hexdigest()

    source_trainer_sha = str(
        checkpoint.get("identity", {}).get("trainer_config_sha256", "")
    )
    if len(source_trainer_sha) != 64:
        raise ValueError("source checkpoint has no trainer config identity")
    handoff = {
        "schema_version": 1,
        "kind": "adaptive_reallocation_stabilization_handoff_v1",
        "status": "ADAPTIVE_REALLOCATION_BOUNDARY_PASS",
        "promotion_eligible": True,
        "checkpoint_sha256": sanitization["output_checkpoint_sha256"],
        "source_trainer_config_sha256": source_trainer_sha,
        "completed_steps": int(checkpoint["step"]),
        "sanitization_report_sha256": sanitization[
            "sanitization_report_sha256"
        ],
    }
    handoff["boundary_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(handoff)
    ).hexdigest()

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": str(run_id),
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_pipeline_gate": args.output_gate.resolve().as_posix(),
            "resume_checkpoint": args.output_checkpoint.resolve().as_posix(),
            "controlled_stop_after_steps": controlled_stop,
            "checkpoint_keep_every": controlled_stop,
            "learning_rates": {
                "means": 4e-06,
                "scales": 0.001,
                "quats": 0.0002,
                "opacities": 0.001,
                "colors": 0.0005,
            },
            "post_refine_geometry_lr_scale": 0.0,
            "error_weighted_sampling": {"enabled": False},
            "contribution": {"enabled": False},
            "lidar_admission": {"enabled": False},
            "tangent_proposal": {"enabled": False},
        }
    )
    config.pop("warm_start_checkpoint", None)
    config.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest()
    gate = advance_adaptive_reallocation_gate(
        _read(args.upstream_gate),
        config,
        stage="stabilization",
        boundary_report=handoff,
    )
    _write_json(args.output_sanitization_report, sanitization)
    _write_json(args.output_handoff_report, handoff)
    _write_json(args.output_gate, gate)
    _write_json(args.output_config, config)
    TrainerConfig.from_dict(config).validate()
    print(
        "V27c stabilization ready: "
        f"clamped={sanitization['clamped_gaussian_count']}, "
        f"checkpoint_sha256={sanitization['output_checkpoint_sha256']}, "
        f"config_sha256={config['config_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
