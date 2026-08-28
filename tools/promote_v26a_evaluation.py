#!/usr/bin/env python3
"""Audit the V26a step-500 boundary and sign the resumed 7480-step arm."""

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


def _peak_vram(progress_path: Path) -> int:
    peak = 0
    with progress_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            peak = max(peak, int(record.get("peak_vram_bytes") or 0))
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
    parser.add_argument(
        "--run-id",
        default="snow-tile1-v26a-classic2d-lidar-eval7480",
    )
    parser.add_argument(
        "--controlled-review-stop",
        type=int,
        help="build a signed bounded continuation instead of the full evaluation arm",
    )
    args = parser.parse_args()

    boundary_config = _read(args.boundary_config)
    checkpoint = torch.load(
        args.boundary_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported boundary checkpoint schema")
    completed = int(checkpoint.get("step", -1))
    identity = checkpoint.get("identity", {})
    state = checkpoint.get("training_state", {})
    telemetry = state.get("mcmc_telemetry", {})
    events = telemetry.get("events", [])
    if len(events) != 1:
        raise ValueError("boundary must contain exactly one lifecycle event")
    event = events[0]
    lifecycle = event.get("classic_lifecycle", {})
    guard = lifecycle.get("surface_birth_guard", {})
    proposal = guard.get("proposal", {})
    after = event.get("after", {})
    initial_count = int(lifecycle.get("before_count", -1))
    final_count = int(lifecycle.get("after_count", -1))
    peak_vram = _peak_vram(args.boundary_progress)
    checks = {
        "completed_step_is_502": completed == 502,
        "first_lifecycle_is_step_500": int(event.get("step", -1)) == 500,
        "finite_after_lifecycle": after.get("finite") is True,
        "count_accounting_exact": final_count
        == initial_count
        + int(lifecycle.get("clone_count", 0))
        + int(lifecycle.get("split_parent_count", 0))
        - int(lifecycle.get("cull_count", 0)),
        "capacity_below_2_2m": 0 < final_count <= 2_200_000,
        "retains_at_least_90pct_of_initial_surface": (
            final_count >= int(0.9 * initial_count)
        ),
        "real_births_observed": int(guard.get("newborns", 0)) > 0,
        "all_newborn_proposals_applied": float(
            proposal.get("applied_fraction", 0.0)
        )
        == 1.0,
        "unsupported_candidates_rejected": int(guard.get("rejected_parents", 0))
        > 0,
        "guard_accounting_exact": int(guard.get("growth_candidates", -1))
        == int(guard.get("supported_parents", 0))
        + int(guard.get("rejected_parents", 0)),
        "fallback_fraction_below_2pct": float(
            proposal.get("fallback_fraction", 1.0)
        )
        <= 0.02,
        "parent_support_mean_above_90pct": float(
            proposal.get("support_mean", 0.0)
        )
        >= 0.9,
        "child_support_mean_above_90pct": float(
            proposal.get("child_support_mean", 0.0)
        )
        >= 0.9,
        "post_lifecycle_scale_p95_below_5cm": float(
            after.get("scale_m", {}).get("p95", 1e9)
        )
        <= 0.05,
        "post_lifecycle_scale_max_below_20cm": float(
            after.get("scale_m", {}).get("max", 1e9)
        )
        <= 0.200001,
        "peak_vram_below_7_5gib": peak_vram <= int(7.5 * 1024**3),
        "mcmc_noise_disabled": int(
            telemetry.get("noise_injection_step_count", -1)
        )
        == 0,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "kind": "adaptive_growth_boundary_report_v1",
        "status": (
            "ADAPTIVE_GROWTH_BOUNDARY_PASS"
            if not failed
            else "ADAPTIVE_GROWTH_BOUNDARY_FAIL"
        ),
        "promotion_eligible": not failed,
        "failed_checks": failed,
        "checks": checks,
        "boundary_config_manifest_sha256": boundary_config.get(
            "config_manifest_sha256"
        ),
        "checkpoint_sha256": _sha256(args.boundary_checkpoint),
        "source_trainer_config_sha256": identity.get("trainer_config_sha256"),
        "completed_steps": completed,
        "initial_gaussian_count": initial_count,
        "final_gaussian_count": final_count,
        "net_gaussian_change": final_count - initial_count,
        "classic_lifecycle": copy.deepcopy(lifecycle),
        "peak_vram_bytes": peak_vram,
        "post_lifecycle_snapshot": copy.deepcopy(after),
    }
    unsigned = copy.deepcopy(report)
    report["boundary_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    _write(args.output_report, report)
    if failed:
        print(f"V26a boundary FAIL: {', '.join(failed)}")
        return 2

    evaluation_config = copy.deepcopy(boundary_config)
    evaluation_config.update(
        {
            "run_id": args.run_id,
            "output_dir": args.run_output.resolve().as_posix(),
            "mipmap_pipeline_gate": args.output_gate.resolve().as_posix(),
            "resume_checkpoint": args.boundary_checkpoint.resolve().as_posix(),
            "checkpoint_keep_every": 2618,
        }
    )
    if args.controlled_review_stop is None:
        evaluation_config.pop("controlled_stop_after_steps", None)
        continuation_stage = "evaluation"
    else:
        evaluation_config["controlled_stop_after_steps"] = (
            args.controlled_review_stop
        )
        continuation_stage = "review"
    evaluation_config.pop("config_manifest_sha256", None)
    evaluation_config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(evaluation_config)
    ).hexdigest()
    gate = advance_adaptive_growth_gate(
        _read(args.upstream_gate),
        evaluation_config,
        stage=continuation_stage,
        boundary_report=report,
    )
    _write(args.output_gate, gate)
    _write(args.output_config, evaluation_config)
    TrainerConfig.from_dict(evaluation_config).validate()
    print(
        f"V26a {continuation_stage} ready: "
        f"boundary_report_sha256={report['boundary_report_sha256']}, "
        f"config_sha256={evaluation_config['config_manifest_sha256']}, "
        f"gate_sha256={gate['gate_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
