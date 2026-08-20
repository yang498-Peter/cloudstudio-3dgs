"""Audit whether high BA reprojection residuals concentrate on person masks."""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.geometry.rig import distribution


@dataclass(frozen=True)
class PersonResidualAuditPolicy:
    high_residual_threshold_px: float = 5.0
    minimum_high_residual_observations: int = 20
    rerun_overlap_fraction: float = 0.3

    def validate(self) -> None:
        if not np.isfinite(self.high_residual_threshold_px) or self.high_residual_threshold_px <= 0.0:
            raise ValueError("high residual threshold must be finite and positive")
        if self.minimum_high_residual_observations <= 0:
            raise ValueError("minimum high residual observations must be positive")
        if not 0.0 < self.rerun_overlap_fraction <= 1.0:
            raise ValueError("rerun overlap fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "high_residual_threshold_px": self.high_residual_threshold_px,
            "minimum_high_residual_observations": self.minimum_high_residual_observations,
            "rerun_overlap_fraction": self.rerun_overlap_fraction,
        }


def audit_person_residuals(
    observations: list[dict[str, Any]],
    person_masks: dict[str, np.ndarray],
    policy: PersonResidualAuditPolicy = PersonResidualAuditPolicy(),
) -> dict[str, Any]:
    """Classify high residual observations without changing the BA model."""
    policy.validate()
    if not observations:
        raise ValueError("person residual audit requires observations")
    if not person_masks:
        raise ValueError("person residual audit requires person masks")
    masks: dict[str, np.ndarray] = {}
    for image_id, raw in person_masks.items():
        mask = np.asarray(raw, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"person mask for {image_id} must be two-dimensional")
        masks[str(image_id)] = mask

    normalized: list[dict[str, Any]] = []
    per_image: dict[str, list[dict[str, Any]]] = {}
    for raw in observations:
        image_id = str(raw.get("image_id", ""))
        if image_id not in masks:
            raise ValueError(f"observation references missing person mask: {image_id}")
        xy = np.asarray(raw.get("xy"), dtype=np.float64)
        error = float(raw.get("error_px", float("nan")))
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError("observation coordinates must be a finite pair")
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("observation residual must be finite and non-negative")
        x = int(np.rint(xy[0]))
        y = int(np.rint(xy[1]))
        height, width = masks[image_id].shape
        if x < 0 or y < 0 or x >= width or y >= height:
            raise ValueError(f"observation is outside person mask for {image_id}")
        record = {
            "image_id": image_id,
            "xy": [float(xy[0]), float(xy[1])],
            "error_px": error,
            "on_person": bool(masks[image_id][y, x]),
            "high_residual": error >= policy.high_residual_threshold_px,
        }
        normalized.append(record)
        per_image.setdefault(image_id, []).append(record)

    return audit_labeled_residuals(normalized, policy)


def audit_labeled_residuals(
    observations: Iterable[dict[str, Any]],
    policy: PersonResidualAuditPolicy = PersonResidualAuditPolicy(),
) -> dict[str, Any]:
    """Audit observations whose person overlap was classified while streaming masks."""
    policy.validate()
    errors = array("d")
    observations_on_person = 0
    high_observations = 0
    high_on_person = 0
    per_image: dict[str, dict[str, Any]] = {}
    for raw in observations:
        image_id = str(raw.get("image_id", ""))
        xy = np.asarray(raw.get("xy"), dtype=np.float64)
        error = float(raw.get("error_px", float("nan")))
        if not image_id:
            raise ValueError("labeled observation has no image ID")
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError("labeled observation coordinates must be a finite pair")
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("labeled observation residual must be finite and non-negative")
        on_person = bool(raw.get("on_person", False))
        high = error >= policy.high_residual_threshold_px
        errors.append(error)
        observations_on_person += int(on_person)
        high_observations += int(high)
        high_on_person += int(high and on_person)
        stats = per_image.setdefault(
            image_id,
            {
                "errors": array("d"),
                "observations_on_person": 0,
                "high_residual_observations": 0,
                "high_residual_on_person": 0,
            },
        )
        stats["errors"].append(error)
        stats["observations_on_person"] += int(on_person)
        stats["high_residual_observations"] += int(high)
        stats["high_residual_on_person"] += int(high and on_person)

    if not errors:
        raise ValueError("person residual audit requires labeled observations")
    overlap = 0.0 if not high_observations else high_on_person / high_observations
    if high_observations < policy.minimum_high_residual_observations:
        decision = "INSUFFICIENT_HIGH_RESIDUAL_OBSERVATIONS"
    elif overlap >= policy.rerun_overlap_fraction:
        decision = "RERUN_MASKED_BA_RECOMMENDED"
    else:
        decision = "RETAIN_CURRENT_BA"

    image_records: list[dict[str, Any]] = []
    for image_id, stats in sorted(per_image.items()):
        image_errors = stats["errors"]
        image_high = int(stats["high_residual_observations"])
        image_high_person = int(stats["high_residual_on_person"])
        image_records.append(
            {
                "image_id": image_id,
                "observations": len(image_errors),
                "observations_on_person": int(stats["observations_on_person"]),
                "high_residual_observations": image_high,
                "high_residual_on_person": image_high_person,
                "high_residual_person_overlap_fraction": (
                    0.0 if not image_high else image_high_person / image_high
                ),
                "reprojection_error_px": distribution(image_errors),
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "ba_person_residual_overlap_v1",
        "decision": decision,
        "policy": policy.to_dict(),
        "observations": len(errors),
        "observations_on_person": observations_on_person,
        "high_residual_observations": high_observations,
        "high_residual_on_person": high_on_person,
        "high_residual_person_overlap_fraction": overlap,
        "reprojection_error_px": distribution(errors),
        "images": image_records,
    }
    report["person_residual_audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report
