"""Build a signed joint RGB, alpha, and geometry gate for V63 boundaries."""

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


def _delta(candidate: dict, reference: dict, field: str) -> float:
    return float(candidate[field]) - float(reference[field])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--candidate-validation", type=Path, required=True)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = _read(args.reference_validation)
    candidate = _read(args.candidate_validation)
    mesh = _read(args.mesh_manifest)
    health = _read(args.health)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scale = torch.exp(checkpoint["params"]["scales"]).max(dim=1).values
    oversized_0p2m = int((scale > 0.2).sum().item())

    checks = {
        "cross_view_filter_signed": (
            mesh.get("cross_view_filter", {}).get("status")
            == "PRODUCTION_FILTER_APPLIED"
        ),
        "cross_view_retained_p05_ge_0p999": float(
            mesh.get("cross_view_filter", {}).get("retained_fraction_p05", 0.0)
        )
        >= 0.999,
        "psnr_delta_ge_minus_0p05_db": _delta(
            candidate, reference, "psnr_mean_db"
        )
        >= -0.05,
        "ssim_delta_ge_minus_0p002": _delta(candidate, reference, "ssim_mean")
        >= -0.002,
        "alpha_mean_delta_ge_minus_0p01": _delta(
            candidate, reference, "alpha_mean"
        )
        >= -0.01,
        "alpha_p05_delta_ge_minus_0p03": _delta(
            candidate, reference, "alpha_p05_mean"
        )
        >= -0.03,
        "lidar_alpha_p05_delta_ge_minus_0p02": _delta(
            candidate, reference, "lidar_alpha_p05_mean"
        )
        >= -0.02,
        "psnr_absolute_ge_18_db": float(candidate["psnr_mean_db"]) >= 18.0,
        "alpha_mean_absolute_ge_0p88": float(candidate["alpha_mean"]) >= 0.88,
        "alpha_p05_absolute_ge_0p45": float(candidate["alpha_p05_mean"])
        >= 0.45,
        "lidar_alpha_p05_absolute_ge_0p95": float(
            candidate["lidar_alpha_p05_mean"]
        )
        >= 0.95,
        "depth_mae_regression_le_2_percent": float(
            candidate["depth_mae_mean_m"]
        )
        <= float(reference["depth_mae_mean_m"]) * 1.02,
        "no_visible_floater_over_0p3m": int(
            health["floater"]["outliers"]["gt_0.3m"]["count"]
        )
        == 0,
        # MipMap's recovered lifecycle removes max-axis > 0.2 m splats. A
        # fixed-topology admission run must therefore not carry such splats
        # into a 7,480-step continuation where no Cull can remove them.
        "no_surface_gaussian_over_0p2m": oversized_0p2m == 0,
    }
    payload = {
        "schema_version": 1,
        "kind": "v63_dense_geometry_boundary_joint_gate_v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "long_training_allowed": all(checks.values()),
        "adaptive_growth_allowed": False,
        "completed_steps": int(checkpoint["step"]),
        "checks": checks,
        "metrics": {
            "psnr_delta_db": _delta(candidate, reference, "psnr_mean_db"),
            "ssim_delta": _delta(candidate, reference, "ssim_mean"),
            "depth_mae_delta_m": _delta(
                candidate, reference, "depth_mae_mean_m"
            ),
            "alpha_mean_delta": _delta(candidate, reference, "alpha_mean"),
            "alpha_p05_delta": _delta(candidate, reference, "alpha_p05_mean"),
            "lidar_alpha_p05_delta": _delta(
                candidate, reference, "lidar_alpha_p05_mean"
            ),
            "surface_gaussian_over_0p2m": oversized_0p2m,
            "max_scale_m": float(scale.max().item()),
        },
        "evidence": {
            "reference_validation": str(args.reference_validation.resolve()),
            "candidate_validation": str(args.candidate_validation.resolve()),
            "mesh_manifest": str(args.mesh_manifest.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "gaussian_health": str(args.health.resolve()),
        },
        "next_action": (
            "continue_to_7480_fixed_topology"
            if all(checks.values())
            else "repair_failed_gate_before_any_long_training"
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
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
