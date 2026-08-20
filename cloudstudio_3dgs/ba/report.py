"""Fail-closed before/after acceptance report for staged fixed-Rig BA."""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.geometry.rig import distribution, rotation_error_rad


@dataclass(frozen=True)
class RigBaPolicy:
    minimum_reprojection_p50_improvement_fraction: float = 0.30
    maximum_scale_drift_fraction: float = 0.005
    maximum_rig_translation_drift_m: float = 1e-6
    maximum_rig_rotation_drift_deg: float = 1e-4
    maximum_focal_relative_change: float = 0.05
    maximum_k1_k2_absolute_change: float = 0.02
    fixed_parameter_tolerance: float = 1e-10

    def validate(self) -> None:
        values = self.to_dict()
        for key, value in values.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} must be finite and non-negative")
        if not 0.0 < self.minimum_reprojection_p50_improvement_fraction < 1.0:
            raise ValueError("minimum reprojection improvement must be in (0, 1)")

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_reprojection_p50_improvement_fraction": (
                self.minimum_reprojection_p50_improvement_fraction
            ),
            "maximum_scale_drift_fraction": self.maximum_scale_drift_fraction,
            "maximum_rig_translation_drift_m": self.maximum_rig_translation_drift_m,
            "maximum_rig_rotation_drift_deg": self.maximum_rig_rotation_drift_deg,
            "maximum_focal_relative_change": self.maximum_focal_relative_change,
            "maximum_k1_k2_absolute_change": self.maximum_k1_k2_absolute_change,
            "fixed_parameter_tolerance": self.fixed_parameter_tolerance,
        }


def stage_options(stage: str) -> dict[str, bool]:
    if stage not in {"stage_1", "stage_2", "stage_3"}:
        raise ValueError("BA stage must be stage_1, stage_2, or stage_3")
    # PyCOLMAP exposes one switch for all distortion coefficients. Stage 3 may
    # solve with that switch, but _camera_changes only permits k1/k2 to survive
    # the publication gate; any k3/k4 movement rejects the candidate.
    return {
        "refine_rig_from_world": True,
        "refine_sensor_from_rig": False,
        "refine_focal_length": stage in {"stage_2", "stage_3"},
        "refine_principal_point": False,
        "refine_extra_params": stage == "stage_3",
    }


def _finite_errors(snapshot: dict[str, Any], label: str) -> list[float]:
    values = [float(value) for value in snapshot.get("reprojection_errors_px", [])]
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{label} reprojection errors must be finite and non-negative")
    return values


def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    return matrix


def _trajectory_scale(frames: dict[str, Any]) -> float:
    ordered = sorted(frames.values(), key=lambda item: int(item["timestamp_ns"]))
    centers = [np.asarray(item["center_m"], dtype=np.float64) for item in ordered]
    if any(center.shape != (3,) or not np.all(np.isfinite(center)) for center in centers):
        raise ValueError("BA snapshot contains an invalid Rig center")
    distances = [
        float(np.linalg.norm(centers[right] - centers[left]))
        for left in range(len(centers))
        for right in range(left + 1, len(centers))
        if float(np.linalg.norm(centers[right] - centers[left])) > 1e-9
    ]
    if not distances:
        raise ValueError("BA trajectory has no non-zero pair distance for scale audit")
    return float(np.median(distances))


def _camera_changes(
    before: dict[str, Any],
    after: dict[str, Any],
    stage: str,
    policy: RigBaPolicy,
) -> tuple[dict[str, Any], bool]:
    if set(before) != set(after) or not before:
        raise ValueError("before/after BA camera sets differ")
    report: dict[str, Any] = {}
    passed = True
    for camera_id in sorted(before):
        old = before[camera_id]
        new = after[camera_id]
        changes: dict[str, float] = {}
        for key in ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4"):
            old_value = float(old[key])
            new_value = float(new[key])
            if not np.isfinite(old_value) or not np.isfinite(new_value):
                raise ValueError(f"camera {camera_id} parameter {key} is not finite")
            changes[key] = new_value - old_value
        focal_relative = max(
            abs(changes["fl_x"]) / max(abs(float(old["fl_x"])), 1e-12),
            abs(changes["fl_y"]) / max(abs(float(old["fl_y"])), 1e-12),
        )
        fixed_keys = ["cx", "cy", "k3", "k4"]
        if stage == "stage_1":
            fixed_keys += ["fl_x", "fl_y", "k1", "k2"]
        elif stage == "stage_2":
            fixed_keys += ["k1", "k2"]
        fixed_max = max(abs(changes[key]) for key in fixed_keys)
        camera_passed = (
            fixed_max <= policy.fixed_parameter_tolerance
            and focal_relative <= policy.maximum_focal_relative_change
            and abs(changes["k1"]) <= policy.maximum_k1_k2_absolute_change
            and abs(changes["k2"]) <= policy.maximum_k1_k2_absolute_change
        )
        passed = passed and camera_passed
        report[camera_id] = {
            "changes": changes,
            "focal_maximum_relative_change": focal_relative,
            "fixed_parameter_maximum_absolute_change": fixed_max,
            "status": "PASS" if camera_passed else "FAIL",
        }
    return report, passed


def build_ba_report(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    stage: str,
    policy: RigBaPolicy = RigBaPolicy(),
) -> dict[str, Any]:
    policy.validate()
    options = stage_options(stage)
    before_errors = _finite_errors(before, "before")
    after_errors = _finite_errors(after, "after")
    before_p50 = float(np.percentile(before_errors, 50))
    after_p50 = float(np.percentile(after_errors, 50))
    if before_p50 <= 0.0:
        raise ValueError("before BA reprojection p50 must be positive")
    improvement = (before_p50 - after_p50) / before_p50
    reprojection_pass = (
        improvement >= policy.minimum_reprojection_p50_improvement_fraction
    )

    before_frames = before.get("rig_frames", {})
    after_frames = after.get("rig_frames", {})
    if set(before_frames) != set(after_frames) or not before_frames:
        raise ValueError("before/after BA Rig Frame sets differ")
    translation_drift: list[float] = []
    rotation_drift: list[float] = []
    for rig_frame_id in sorted(before_frames):
        old = _matrix(
            before_frames[rig_frame_id]["right_to_left"],
            f"before Rig baseline {rig_frame_id}",
        )
        new = _matrix(
            after_frames[rig_frame_id]["right_to_left"],
            f"after Rig baseline {rig_frame_id}",
        )
        translation_drift.append(float(np.linalg.norm(new[:3, 3] - old[:3, 3])))
        rotation_drift.append(float(np.degrees(rotation_error_rad(new, old))))
    rig_pass = (
        max(translation_drift) <= policy.maximum_rig_translation_drift_m
        and max(rotation_drift) <= policy.maximum_rig_rotation_drift_deg
    )

    before_scale = _trajectory_scale(before_frames)
    after_scale = _trajectory_scale(after_frames)
    scale_drift = abs(after_scale / before_scale - 1.0)
    scale_pass = scale_drift <= policy.maximum_scale_drift_fraction
    camera_report, camera_pass = _camera_changes(
        before.get("cameras", {}), after.get("cameras", {}), stage, policy
    )
    solver_pass = bool(after.get("solver_success", False))
    gates = {
        "solver_success": {"status": "PASS" if solver_pass else "FAIL"},
        "reprojection_p50_improvement": {
            "status": "PASS" if reprojection_pass else "FAIL",
            "before_px": before_p50,
            "after_px": after_p50,
            "improvement_fraction": improvement,
        },
        "rig_baseline_fixed": {
            "status": "PASS" if rig_pass else "FAIL",
            "translation_drift_m": distribution(translation_drift),
            "rotation_drift_deg": distribution(rotation_drift),
        },
        "scene_scale_fixed": {
            "status": "PASS" if scale_pass else "FAIL",
            "before_median_pair_distance_m": before_scale,
            "after_median_pair_distance_m": after_scale,
            "drift_fraction": scale_drift,
        },
        "camera_parameter_bounds": {
            "status": "PASS" if camera_pass else "FAIL",
            "cameras": camera_report,
        },
    }
    accepted = all(gate["status"] == "PASS" for gate in gates.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "fixed_rig_ba_acceptance_v1",
        "stage": stage,
        "stage_options": options,
        "policy": policy.to_dict(),
        "before_model_sha256": str(before.get("model_sha256", "")),
        "after_model_sha256": str(after.get("model_sha256", "")),
        "gates": gates,
        "reprojection_error_px": {
            "before": distribution(before_errors),
            "after": distribution(after_errors),
        },
        "candidate_accepted": accepted,
        "published_model": "after" if accepted else "before",
    }
    return sign_ba_report(report)


def sign_ba_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with a signature over all report fields except the signature."""
    signed = dict(report)
    signed.pop("ba_report_sha256", None)
    signed["ba_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(signed)
    ).hexdigest()
    return signed


def verify_ba_report(report: dict[str, Any]) -> str:
    expected = str(report.get("ba_report_sha256", ""))
    if not expected:
        raise ValueError("BA report has no ba_report_sha256")
    unsigned = dict(report)
    unsigned.pop("ba_report_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(f"BA report SHA256 mismatch: expected {expected}, computed {actual}")
    gate_pass = all(gate.get("status") == "PASS" for gate in report.get("gates", {}).values())
    if report.get("candidate_accepted") != gate_pass:
        raise ValueError("BA publication decision differs from gate results")
    if report.get("published_model") != ("after" if gate_pass else "before"):
        raise ValueError("BA published model differs from gate results")
    return actual


def _html_report(report: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{gate['status']}</td><td><code>{html.escape(str(gate))}</code></td></tr>"
        for name, gate in report["gates"].items()
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Rig BA Report</title>
<style>body{{font:14px system-ui;margin:2rem;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cbd5e1;padding:.45rem;text-align:left}}th{{background:#e2e8f0}}code{{white-space:pre-wrap;word-break:break-all}}</style></head>
<body><h1>Rig BA before/after 报告</h1><p>Stage：{report['stage']}；采用模型：<strong>{report['published_model']}</strong></p>
<p>Report SHA256：<code>{report['ba_report_sha256']}</code></p><table><thead><tr><th>Gate</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{rows}</tbody></table></body></html>
"""


def write_ba_report(output_dir: Path, report: dict[str, Any], *, force: bool = False) -> None:
    verify_ba_report(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"BA report output is not empty: {output_dir}; pass --force")
    payloads = {
        "ba_report.json": (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "ba_report.html": _html_report(report).encode("utf-8"),
    }
    for name, payload in payloads.items():
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=output_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output_dir / name)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
