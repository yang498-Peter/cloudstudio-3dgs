"""Fail-closed aggregation of completed Gate 2 Trainer A/B runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.quality_report import (
    verify_quality_report,
    verify_run_manifest,
)
from cloudstudio_3dgs.training.ab_matrix import verify_trainer_ab_matrix


_SHARED_RUN_IDENTITIES = (
    "dataset_manifest_sha256",
    "mask_manifest_sha256",
    "person_mask_manifest_sha256",
    "split_manifest_sha256",
    "depth_manifest_sha256",
    "coordinate_transform_sha256",
    "initialization_ply_sha256",
)


def classify_metric_deltas(delta: dict[str, float]) -> str:
    """Classify strict no-regression metrics; positive deltas always mean better."""
    if not delta:
        raise ValueError("A/B metric delta set is empty")
    better = [value > 0.0 for value in delta.values()]
    worse = [value < 0.0 for value in delta.values()]
    if any(worse) and any(better):
        return "MIXED"
    if any(worse):
        return "REGRESSED"
    if any(better):
        return "IMPROVED"
    return "TIED"


def _measured_metrics(report: dict[str, Any], *, require_lpips: bool) -> dict[str, float]:
    image = report["summary"]["image_metrics"]
    psnr = image["psnr_db"].get("mean")
    ssim = image["ssim"].get("mean")
    lpips = image["lpips"]
    depth = report["summary"]["depth_metrics"]
    if psnr is None or ssim is None:
        raise ValueError("A/B quality report has no finite PSNR/SSIM")
    if not isinstance(depth, dict) or depth.get("mae_m", {}).get("mean") is None:
        raise ValueError("A/B quality report has no measured depth MAE")
    output = {
        "psnr_db_mean": float(psnr),
        "ssim_mean": float(ssim),
        "depth_mae_m_mean": float(depth["mae_m"]["mean"]),
    }
    lpips_mean = lpips.get("mean") if isinstance(lpips, dict) else None
    if require_lpips and lpips_mean is None:
        raise ValueError("formal A/B requires measured LPIPS")
    if lpips_mean is not None:
        output["lpips_mean"] = float(lpips_mean)
    return output


def _better_deltas(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    keys = set(candidate) & set(reference)
    result: dict[str, float] = {}
    for key in sorted(keys):
        raw = candidate[key] - reference[key]
        # Lower LPIPS and depth error are better; all stored deltas use a
        # single positive-is-better convention for unambiguous verdicts.
        result[key] = -raw if key in {"lpips_mean", "depth_mae_m_mean"} else raw
    return result


def build_trainer_ab_report(
    matrix: dict[str, Any],
    *,
    matrix_root: Path,
    require_lpips: bool = True,
    minimum_periodic_full_evaluations: int = 2,
) -> dict[str, Any]:
    """Verify every run/quality artifact and produce a signed comparison."""
    matrix_sha = verify_trainer_ab_matrix(matrix, matrix_root)
    if minimum_periodic_full_evaluations < 2:
        raise ValueError("formal Gate 2 requires at least two full-validation evaluations")
    arm_results: list[dict[str, Any]] = []
    shared_identity: dict[str, Any] | None = None
    for arm in matrix["arms"]:
        config_path = Path(matrix_root) / Path(*str(arm["config_path"]).split("/"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_root = Path(str(config["output_dir"]))
        run_path = run_root / "run_manifest.json"
        quality_path = run_root / "quality" / "quality_report.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"A/B run manifest is missing for arm {arm['arm']}")
        if not quality_path.is_file():
            raise FileNotFoundError(f"A/B quality report is missing for arm {arm['arm']}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        run_sha = verify_run_manifest(run)
        quality_sha = verify_quality_report(quality)
        if run.get("trainer_config_sha256") != arm["trainer_config_sha256"]:
            raise ValueError(f"A/B Trainer contract mismatch for arm {arm['arm']}")
        if quality.get("run_manifest_sha256") != run_sha:
            raise ValueError(f"A/B quality report is bound to another run for arm {arm['arm']}")
        if run.get("training", {}).get("status") != "COMPLETE":
            raise ValueError(f"A/B training is incomplete for arm {arm['arm']}")
        if int(run["training"].get("completed_steps", -1)) != int(config["max_steps"]):
            raise ValueError(f"A/B completed step count mismatch for arm {arm['arm']}")
        if quality.get("golden_checkpoint_selection", {}).get("status") != "VERIFIED":
            raise ValueError(f"A/B golden checkpoint evidence is unverified for arm {arm['arm']}")
        periodic = quality.get("periodic_full_evaluation", {})
        if periodic.get("status") != "VERIFIED" or int(
            periodic.get("evaluation_count", 0)
        ) < minimum_periodic_full_evaluations:
            raise ValueError(f"A/B periodic full validation is incomplete for arm {arm['arm']}")
        identity = {key: run.get(key) for key in _SHARED_RUN_IDENTITIES}
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError("A/B runs do not share dataset/mask/split/pose/initialization identity")
        metrics = _measured_metrics(quality, require_lpips=require_lpips)
        arm_results.append(
            {
                "arm": arm["arm"],
                "role": arm["role"],
                "trainer_preset": arm["trainer_preset"],
                "run_manifest_sha256": run_sha,
                "quality_report_sha256": quality_sha,
                "completed_steps": int(run["training"]["completed_steps"]),
                "peak_vram_bytes": int(run["training"]["peak_vram_bytes"]),
                "gaussian_count": int(run["training"]["gaussian_count"]),
                "metrics": metrics,
            }
        )

    reference_metrics = arm_results[0]["metrics"]
    for result in arm_results:
        deltas = _better_deltas(result["metrics"], reference_metrics)
        result["better_is_positive_delta"] = deltas
        result["verdict_vs_reference"] = classify_metric_deltas(deltas)
    candidate = next(result for result in arm_results if result["arm"] == "quality_candidate")
    candidate_pass = all(
        value >= 0.0 for value in candidate["better_is_positive_delta"].values()
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "gate2_trainer_ab_report_v1",
        "experiment_id": matrix["experiment_id"],
        "ab_matrix_sha256": matrix_sha,
        "shared_run_identity": shared_identity,
        "requirements": {
            "require_lpips": require_lpips,
            "minimum_periodic_full_evaluations": minimum_periodic_full_evaluations,
            "strict_no_regression_tolerance": 0.0,
        },
        "arms": arm_results,
        "gate2_quality_candidate": {
            "status": "PASS" if candidate_pass else "FAIL",
            "strictly_no_lower_than_legacy_reference": candidate_pass,
        },
    }
    report["ab_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def verify_trainer_ab_report(report: dict[str, Any], matrix: dict[str, Any]) -> str:
    expected = str(report.get("ab_report_sha256", ""))
    if not expected:
        raise ValueError("A/B report has no SHA256")
    unsigned = dict(report)
    unsigned.pop("ab_report_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(f"A/B report SHA256 mismatch: expected {expected}, computed {actual}")
    if report.get("ab_matrix_sha256") != matrix.get("ab_matrix_sha256"):
        raise ValueError("A/B report is bound to another matrix")
    return actual
