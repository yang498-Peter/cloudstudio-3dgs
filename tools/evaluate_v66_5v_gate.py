"""Build a signed fail-closed V66 5V joint quality gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--scale-audit", type=Path, required=True)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--holdout-audit", type=Path, required=True)
    parser.add_argument("--normal-smoke", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gate-kind",
        default="v66_strict_mesh_5v_joint_gate_v1",
        help="signed report kind; defaults to the original V66 schema name",
    )
    args = parser.parse_args()

    validation = _read(args.validation)
    reference = _read(args.reference_validation)
    health = _read(args.health)
    scale = _read(args.scale_audit)
    mesh = _read(args.mesh_manifest)
    holdout = _read(args.holdout_audit)
    normal_smoke = _read(args.normal_smoke)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    admission = mesh["admission_policy"]
    category_p95 = {
        name: float(value["quantiles_m"]["p95"])
        for name, value in holdout["heldout_category_audit"]["categories"].items()
    }
    opacity_low_fraction = float(
        health["opacity"]["dead_lt_0_005"]["fraction"]
        + health["opacity"]["fog_0_005_to_0_1"]["fraction"]
    )
    wall_thickness_p95 = float(
        health["wall"]["weighted_by_lidar_inliers"]["effective_thickness_p95_m"]
    )
    checks = {
        "strict_mesh_allowed_source_types_exact_2_3": (
            admission.get("allowed_source_types") == [2, 3]
        ),
        "strict_mesh_source_type_4_excluded": (
            admission.get("rejected_source_types") == [4]
        ),
        "every_holdout_category_p95_le_0p10m": max(category_p95.values()) <= 0.10,
        "direct_normal_quaternion_gradient_nonzero": (
            float(normal_smoke["quaternion_gradient_norm"]) > 0.0
        ),
        "direct_normal_shortest_scale_gradient_nonzero": (
            float(normal_smoke["shortest_scale_gradient_abs_max"]) > 0.0
        ),
        "completed_first_post_5v_boundary": int(checkpoint["step"]) == 1872,
        "mesh_depth_schedule_reached_0p5": float(
            checkpoint["training_state"]["last_metrics"]["effective_mesh_depth_weight"]
        )
        == 0.5,
        "psnr_ge_reference_minus_0p10db": float(validation["psnr_mean_db"])
        >= float(reference["psnr_mean_db"]) - 0.10,
        "ssim_ge_reference_minus_0p01": float(validation["ssim_mean"])
        >= float(reference["ssim_mean"]) - 0.01,
        "alpha_mean_ge_reference_minus_0p03": float(validation["alpha_mean"])
        >= float(reference["alpha_mean"]) - 0.03,
        "alpha_p05_ge_reference_minus_0p05": float(validation["alpha_p05_mean"])
        >= float(reference["alpha_p05_mean"]) - 0.05,
        "lidar_alpha_p05_ge_reference_minus_0p05": float(
            validation["lidar_alpha_p05_mean"]
        )
        >= float(reference["lidar_alpha_p05_mean"]) - 0.05,
        "heldout_lidar_depth_mae_not_worse_than_reference": float(
            validation["depth_mae_mean_m"]
        )
        <= float(reference["depth_mae_mean_m"]),
        "wall_effective_thickness_p95_le_0p10m": wall_thickness_p95 <= 0.10,
        "median_axis_ratio_ge_5": float(scale["aspect_ratio"]["p50"]) >= 5.0,
        "opacity_lt_0p1_fraction_le_0p50": opacity_low_fraction <= 0.50,
        "no_visible_floater_over_0p3m": int(
            health["floater"]["outliers"]["gt_0.3m"]["count"]
        )
        == 0,
        "no_gaussian_over_0p2m": float(health["scale"]["max_axis_m"]["max"])
        <= 0.2,
        "full_ply_exists": args.ply.is_file() and args.ply.stat().st_size > 0,
    }
    numeric_pass = all(checks.values())
    payload = {
        "schema_version": 1,
        "kind": args.gate_kind,
        "status": "NUMERIC_PASS_VISUAL_PENDING" if numeric_pass else "BLOCKED",
        "long_training_allowed": False,
        "adaptive_growth_allowed": False,
        "white_background_arm_allowed": False,
        "checks": checks,
        "metrics": {
            "validation": {
                key: validation[key]
                for key in (
                    "frame_count",
                    "psnr_mean_db",
                    "ssim_mean",
                    "depth_mae_mean_m",
                    "alpha_mean",
                    "alpha_p05_mean",
                    "lidar_alpha_mean",
                    "lidar_alpha_p05_mean",
                )
            },
            "reference": {
                key: reference[key]
                for key in (
                    "frame_count",
                    "psnr_mean_db",
                    "ssim_mean",
                    "depth_mae_mean_m",
                    "alpha_mean",
                    "alpha_p05_mean",
                    "lidar_alpha_mean",
                    "lidar_alpha_p05_mean",
                )
            },
            "holdout_category_p95_m": category_p95,
            "opacity_lt_0p1_fraction": opacity_low_fraction,
            "wall_effective_thickness_p95_m": wall_thickness_p95,
            "shortest_axis_p50_m": scale["shortest_axis_m"]["p50"],
            "axis_ratio_p50": scale["aspect_ratio"]["p50"],
        },
        "evidence": {
            "validation": str(args.validation.resolve()),
            "reference_validation": str(args.reference_validation.resolve()),
            "health": str(args.health.resolve()),
            "scale_audit": str(args.scale_audit.resolve()),
            "mesh_manifest": str(args.mesh_manifest.resolve()),
            "holdout_audit": str(args.holdout_audit.resolve()),
            "normal_smoke": str(args.normal_smoke.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "ply": str(args.ply.resolve()),
        },
        "next_action": (
            "visual_review_before_any_continuation"
            if numeric_pass
            else (
                "calibrate_mesh_normal_shape_without_weakening_alpha"
                if checks["opacity_lt_0p1_fraction_le_0p50"]
                and (
                    not checks["wall_effective_thickness_p95_le_0p10m"]
                    or not checks["median_axis_ratio_ge_5"]
                )
                else "repair_opacity_and_normal_shape_before_repeating_5v"
            )
        ),
    }
    unsigned = copy.deepcopy(payload)
    payload["gate_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if numeric_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
