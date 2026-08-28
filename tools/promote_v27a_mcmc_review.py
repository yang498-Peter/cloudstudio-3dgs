#!/usr/bin/env python3
"""Audit a V27 MCMC boundary and sign its bounded seven-epoch review arm."""

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
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    V27_SNOW_TILE_PROFILES,
    advance_adaptive_reallocation_gate,
)
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


def _peak_vram(progress_path: Path) -> int:
    peak = 0
    with progress_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                peak = max(peak, int(json.loads(line).get("peak_vram_bytes") or 0))
    return peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-config", required=True, type=Path)
    parser.add_argument("--boundary-checkpoint", required=True, type=Path)
    parser.add_argument("--boundary-progress", required=True, type=Path)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    boundary_config = _read(args.boundary_config)
    tile_id = int(boundary_config.get("mipmap_tile_id", -1))
    profile = V27_SNOW_TILE_PROFILES.get(tile_id)
    if profile is None:
        raise ValueError("boundary config does not target a supported snow Tile")
    gaussian_count = profile["gaussian_count"]
    relocation_cap = int(gaussian_count * 0.02)
    checkpoint = torch.load(
        args.boundary_checkpoint, map_location="cpu", weights_only=False
    )
    identity = checkpoint.get("identity", {})
    telemetry = checkpoint.get("training_state", {}).get("mcmc_telemetry", {})
    events = telemetry.get("events", [])
    if len(events) != 1:
        raise ValueError("boundary must contain exactly one MCMC refine event")
    event = events[0]
    relocation = event.get("adaptive_relocation", {})
    before = event.get("before", {})
    after = event.get("after", {})
    before_scale_p95 = float(before.get("scale_m", {}).get("p95", 1e9))
    # LiDAR sampling density differs per adaptive Tile. Preserve a 2 cm target
    # when the source already meets it, otherwise allow only the measured
    # source P95 (capped at 3 cm). The event must never worsen that baseline.
    scale_p95_limit_m = min(0.03, max(0.02, before_scale_p95))
    peak_vram = _peak_vram(args.boundary_progress)
    checks = {
        "completed_step_is_602": int(checkpoint.get("step", -1)) == 602,
        "first_refine_is_step_600": int(event.get("step", -1)) == 600,
        "finite_after_relocation": after.get("finite") is True,
        "capacity_is_exactly_preserved": int(before.get("gaussian_count", -1))
        == int(after.get("gaussian_count", -2))
        == gaussian_count,
        "true_relocation_count_is_capped": int(event.get("relocated_count", -1))
        == int(relocation.get("selected_count", -2))
        and 0 < int(relocation.get("selected_count", 0)) <= relocation_cap,
        "no_growth_or_prune": int(event.get("new_gaussian_count", -1)) == 0
        and int(event.get("pruned_gaussian_count", -1)) == 0,
        "all_shape_failures_fit_event_budget": int(
            relocation.get("oversized_count", relocation_cap + 1)
        )
        + int(relocation.get("anisotropy_failed_count", relocation_cap + 1))
        <= int(relocation.get("selected_count", 0)),
        "strict_surface_gate_active": relocation.get("strict_surface_gate") is True,
        "eligible_lidar_sources_exist": int(
            relocation.get("eligible_source_count", 0)
        )
        > 0,
        "all_relocations_received_surface_proposal": float(
            relocation.get("proposal", {}).get("applied_fraction", 0.0)
        )
        == 1.0,
        "proposal_fallback_below_2pct": float(
            relocation.get("proposal", {}).get("fallback_fraction", 1.0)
        )
        <= 0.02,
        "post_scale_p95_within_adaptive_lidar_baseline": float(
            after.get("scale_m", {}).get("p95", 1e9)
        )
        <= scale_p95_limit_m + 1e-6,
        "post_scale_max_below_8cm": float(
            after.get("scale_m", {}).get("max", 1e9)
        )
        <= 0.080001,
        "noise_disabled": int(telemetry.get("noise_injection_step_count", -1)) == 0,
        "peak_vram_below_7_5gib": peak_vram <= int(7.5 * 1024**3),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "kind": "adaptive_reallocation_boundary_report_v1",
        "status": (
            "ADAPTIVE_REALLOCATION_BOUNDARY_PASS"
            if not failed
            else "ADAPTIVE_REALLOCATION_BOUNDARY_FAIL"
        ),
        "promotion_eligible": not failed,
        "failed_checks": failed,
        "checks": checks,
        "boundary_config_manifest_sha256": boundary_config.get(
            "config_manifest_sha256"
        ),
        "checkpoint_sha256": _sha256(args.boundary_checkpoint),
        "source_trainer_config_sha256": identity.get("trainer_config_sha256"),
        "completed_steps": int(checkpoint.get("step", -1)),
        "peak_vram_bytes": peak_vram,
        "scale_p95_limit_m": scale_p95_limit_m,
        "mcmc_event": copy.deepcopy(event),
    }
    report["boundary_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    _write(args.output_report, report)
    if failed:
        print("V27 boundary FAIL: " + ", ".join(failed))
        return 2

    review = copy.deepcopy(boundary_config)
    review_stop = profile["review_stop"]
    run_id = args.run_id or (
        f"snow-tile{tile_id}-v27b-a0-safe-mcmc-review{review_stop}"
    )
    review.update(
        {
            "run_id": str(run_id),
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_pipeline_gate": args.output_gate.resolve().as_posix(),
            "resume_checkpoint": args.boundary_checkpoint.resolve().as_posix(),
            "controlled_stop_after_steps": review_stop,
            "checkpoint_keep_every": review_stop,
        }
    )
    review.pop("warm_start_checkpoint", None)
    review.pop("config_manifest_sha256", None)
    review["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(review)
    ).hexdigest()
    gate = advance_adaptive_reallocation_gate(
        _read(args.upstream_gate),
        review,
        stage="evaluation",
        boundary_report=report,
    )
    _write(args.output_gate, gate)
    _write(args.output_config, review)
    TrainerConfig.from_dict(review).validate()
    print(
        "V27 review ready: "
        f"boundary_report_sha256={report['boundary_report_sha256']}, "
        f"config_sha256={review['config_manifest_sha256']}, "
        f"gate_sha256={gate['gate_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
