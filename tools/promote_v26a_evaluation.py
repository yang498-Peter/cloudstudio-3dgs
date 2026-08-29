#!/usr/bin/env python3
"""Audit a classic-growth boundary and sign a bounded continuation."""

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
    parser.add_argument(
        "--boundary-validation",
        type=Path,
        help="same-view validation summary for a vendor-order compatibility arm",
    )
    parser.add_argument(
        "--reference-validation",
        type=Path,
        help="A0 same-view validation summary used only as a bounded safety rail",
    )
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
    parser.add_argument(
        "--vendor-opacity-reset-profile",
        choices=("exact_every300", "deferred_every3000_compatibility"),
        default=None,
        help=(
            "override only the continuation reset cadence; use the deferred "
            "profile when the boundary is healthy but reset destroys coverage"
        ),
    )
    parser.add_argument(
        "--vendor-gradient-profile",
        choices=(
            "vendor_plain_1p5e4",
            "calibrated_plain_1e4",
            "calibrated_plain_7p5e5",
        ),
        default=None,
        help="override only the continuation plain-gradient threshold",
    )
    parser.add_argument(
        "--vendor-cull-profile",
        choices=(
            "exact_0p10_to_0p05",
            "compatibility_uniform_0p05",
            "calibrated_uniform_0p04",
            "calibrated_geometry_only_0p00",
        ),
        default=None,
        help="override only the continuation opacity cull threshold profile",
    )
    parser.add_argument(
        "--vendor-opacity-sparsity-weight",
        choices=(0.01, 0.001),
        default=None,
        type=float,
        help=(
            "override only the visible-current-view opacity sparsity weight; "
            "0.001 is the signed PPISP calibration arm"
        ),
    )
    parser.add_argument(
        "--vendor-detail-split-profile",
        choices=("vendor_0_2m", "calibrated_screen_detail_2cm"),
        default=None,
        help=(
            "override only how eligible parents are cloned or split; the "
            "calibrated arm splits >2 cm parents whose screen radius is >0.0035"
        ),
    )
    parser.add_argument(
        "--vendor-lidar-alpha-profile",
        choices=("disabled", "surface_floor_0p02_target_0p95"),
        default=None,
        help=(
            "override only the rendered alpha floor on signed Face4 LiDAR "
            "pixels; this does not constrain sky or pixels without LiDAR"
        ),
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
    last_metrics = state.get("last_metrics", {})
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
    vendor_pre_optimizer = (
        lifecycle.get("execution_order") == "pre_optimizer_vendor"
    )
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
    comparison = None
    if vendor_pre_optimizer:
        cull_reasons = lifecycle.get("cull_reasons", {})
        configured_cull_profile = boundary_config.get("default_strategy", {}).get(
            "vendor_cull_warmup_profile", "exact_0p10_to_0p05"
        )
        expected_boundary_cull = {
            "exact_0p10_to_0p05": 0.1,
            "compatibility_uniform_0p05": 0.05,
            "calibrated_uniform_0p04": 0.04,
            "calibrated_geometry_only_0p00": 0.0,
        }.get(configured_cull_profile)
        checks.update(
            {
                "vendor_gradient_remap_preserved": lifecycle.get(
                    "current_step_gradient_remapped"
                )
                is True,
                "vendor_cull_profile_matches_boundary": (
                    expected_boundary_cull is not None
                    and float(
                    lifecycle.get("cull_opacity_threshold", -1.0)
                    )
                    == expected_boundary_cull
                ),
                "retains_at_least_80pct_for_bounded_review": (
                    final_count >= int(0.8 * initial_count)
                ),
                "real_births_observed": (
                    int(lifecycle.get("clone_count", 0))
                    + int(lifecycle.get("split_child_count", 0))
                )
                > 0,
                "no_cloudstudio_cull_protection": (
                    cull_reasons.get("policy") == "immediate"
                ),
            }
        )
        if args.boundary_validation is None or args.reference_validation is None:
            raise ValueError(
                "vendor-order review requires boundary and reference validations"
            )
        candidate_validation = _read(args.boundary_validation)
        reference_validation = _read(args.reference_validation)
        if candidate_validation.get("frame_count") != reference_validation.get(
            "frame_count"
        ):
            raise ValueError("candidate/reference validation frame counts differ")
        comparison = {
            "candidate": {
                "path": args.boundary_validation.resolve().as_posix(),
                "psnr_mean_db": candidate_validation.get("psnr_mean_db"),
                "alpha_mean": candidate_validation.get("alpha_mean"),
                "alpha_p05_mean": candidate_validation.get("alpha_p05_mean"),
            },
            "reference": {
                "path": args.reference_validation.resolve().as_posix(),
                "psnr_mean_db": reference_validation.get("psnr_mean_db"),
                "alpha_mean": reference_validation.get("alpha_mean"),
                "alpha_p05_mean": reference_validation.get("alpha_p05_mean"),
            },
            "safety_margins": {
                "psnr_db": 0.5,
                "alpha_mean": 0.02,
                "alpha_p05_mean": 0.03,
            },
        }
        checks.update(
            {
                "same_view_psnr_within_0p5db_of_a0": float(
                    candidate_validation.get("psnr_mean_db", -1e9)
                )
                >= float(reference_validation.get("psnr_mean_db", 1e9)) - 0.5,
                "same_view_alpha_mean_within_0p02_of_a0": float(
                    candidate_validation.get("alpha_mean", -1e9)
                )
                >= float(reference_validation.get("alpha_mean", 1e9)) - 0.02,
                "same_view_alpha_p05_within_0p03_of_a0": float(
                    candidate_validation.get("alpha_p05_mean", -1e9)
                )
                >= float(reference_validation.get("alpha_p05_mean", 1e9)) - 0.03,
            }
        )
    else:
        checks.update(
            {
                "retains_at_least_90pct_of_initial_surface": (
                    final_count >= int(0.9 * initial_count)
                ),
                "real_births_observed": int(guard.get("newborns", 0)) > 0,
                "all_newborn_proposals_applied": float(
                    proposal.get("applied_fraction", 0.0)
                )
                == 1.0,
                "unsupported_candidates_rejected": int(
                    guard.get("rejected_parents", 0)
                )
                > 0,
                "guard_accounting_exact": int(
                    guard.get("growth_candidates", -1)
                )
                == int(guard.get("supported_parents", 0))
                + int(guard.get("rejected_parents", 0)),
                "fallback_fraction_below_5pct": float(
                    proposal.get("fallback_fraction", 1.0)
                )
                <= 0.05,
                "parent_support_mean_above_50pct": float(
                    proposal.get("support_mean", 0.0)
                )
                >= 0.5,
                "child_support_mean_above_50pct": float(
                    proposal.get("child_support_mean", 0.0)
                )
                >= 0.5,
                "soft_point_to_plane_drift_below_5mm": (
                    boundary_config.get("lidar_normal_alignment", {}).get(
                        "weight_point_to_plane", 0.0
                    )
                    == 0.01
                    and float(last_metrics.get("lidar_point_to_plane_raw_m", 1e9))
                    <= 0.005
                ),
            }
        )
    failed = sorted(key for key, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "kind": "adaptive_growth_boundary_report_v2",
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
        "same_view_reference_comparison": comparison,
        "authorization_scope": (
            "bounded_diagnostic_review_only"
            if vendor_pre_optimizer
            else "lidar_guarded_continuation"
        ),
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
    # A continuation resumes the exact boundary optimizer and sampler state;
    # retaining its original warm-start field would describe two mutually
    # exclusive checkpoint lineages and is rejected by the signed gate.
    evaluation_config.pop("warm_start_checkpoint", None)
    if args.vendor_opacity_reset_profile is not None:
        if not vendor_pre_optimizer:
            raise ValueError(
                "vendor opacity reset override requires a vendor-order boundary"
            )
        evaluation_config["default_strategy"][
            "vendor_opacity_reset_profile"
        ] = args.vendor_opacity_reset_profile
        evaluation_config["default_strategy"]["reset_every"] = (
            3000
            if args.vendor_opacity_reset_profile
            == "deferred_every3000_compatibility"
            else 300
        )
    if args.vendor_gradient_profile is not None:
        if not vendor_pre_optimizer:
            raise ValueError(
                "vendor gradient override requires a vendor-order boundary"
            )
        gradient_thresholds = {
            "vendor_plain_1p5e4": 0.00015,
            "calibrated_plain_1e4": 0.0001,
            "calibrated_plain_7p5e5": 0.000075,
        }
        evaluation_config["default_strategy"]["absgrad"] = False
        evaluation_config["default_strategy"]["grow_grad2d"] = (
            gradient_thresholds[args.vendor_gradient_profile]
        )
    if args.vendor_cull_profile is not None:
        if not vendor_pre_optimizer:
            raise ValueError(
                "vendor cull override requires a vendor-order boundary"
            )
        cull_thresholds = {
            "exact_0p10_to_0p05": (0.1, 0.05),
            "compatibility_uniform_0p05": (0.05, 0.05),
            "calibrated_uniform_0p04": (0.04, 0.04),
            "calibrated_geometry_only_0p00": (0.0, 0.0),
        }
        prune_opa, prune_opa_late = cull_thresholds[
            args.vendor_cull_profile
        ]
        evaluation_config["default_strategy"].update(
            {
                "vendor_cull_warmup_profile": args.vendor_cull_profile,
                "prune_opa": prune_opa,
                "prune_opa_late": prune_opa_late,
            }
        )
    if args.vendor_opacity_sparsity_weight is not None:
        if not vendor_pre_optimizer:
            raise ValueError(
                "vendor opacity sparsity override requires a vendor-order boundary"
            )
        if (
            evaluation_config.get("geometry_regularization", {}).get(
                "opacity_sparsity_scope", "all"
            )
            != "visible_current_view"
        ):
            raise ValueError(
                "vendor opacity sparsity override requires current-view scope"
            )
        evaluation_config["geometry_regularization"][
            "opacity_sparsity_weight"
        ] = args.vendor_opacity_sparsity_weight
    if args.vendor_detail_split_profile is not None:
        if not vendor_pre_optimizer:
            raise ValueError(
                "vendor detail split override requires a vendor-order boundary"
            )
        if args.vendor_detail_split_profile == "vendor_0_2m":
            evaluation_config["default_strategy"]["detail_split_policy"] = (
                "vendor_0_2m"
            )
        else:
            evaluation_config["default_strategy"].update(
                {
                    "detail_split_policy": "lidar_surface_screen_detail",
                    "detail_split_scale_m": 0.02,
                    "detail_split_screen_radius": 0.0035,
                    "revised_opacity": True,
                }
            )
    if args.vendor_lidar_alpha_profile is not None:
        if not vendor_pre_optimizer:
            raise ValueError(
                "vendor LiDAR alpha override requires a vendor-order boundary"
            )
        lidar_alpha_profiles = {
            "disabled": (0.0, 0.95),
            "surface_floor_0p02_target_0p95": (0.02, 0.95),
        }
        lidar_alpha_weight, lidar_alpha_target = lidar_alpha_profiles[
            args.vendor_lidar_alpha_profile
        ]
        evaluation_config.update(
            {
                "lidar_alpha_weight": lidar_alpha_weight,
                "lidar_alpha_target": lidar_alpha_target,
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
