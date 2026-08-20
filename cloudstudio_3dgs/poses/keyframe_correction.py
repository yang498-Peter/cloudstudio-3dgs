"""Propagate solver keyframe corrections across a fixed stereo Rig timeline."""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest
from cloudstudio_3dgs.geometry.rig import distribution, rotation_error_rad


GL_TO_CV = np.diag([1.0, -1.0, -1.0])
POSE_MANIFEST_NAME = "pose_set_manifest.json"
CURVE_NAME = "pose_correction_curve.svg"
REPORT_NAME = "pose_correction_report.html"


@dataclass(frozen=True)
class PoseCorrectionConfig:
    minimum_anchor_rig_frames: int = 5
    outlier_mad_multiplier: float = 6.0
    translation_residual_floor_m: float = 0.002
    rotation_residual_floor_deg: float = 0.05
    maximum_pair_translation_disagreement_m: float = 0.01
    maximum_pair_rotation_disagreement_deg: float = 0.1
    maximum_step_translation_m: float = 0.05
    maximum_step_rotation_deg: float = 1.0

    def validate(self) -> None:
        if self.minimum_anchor_rig_frames < 3:
            raise ValueError("minimum_anchor_rig_frames must be at least three")
        for key, value in self.to_dict().items():
            if key == "minimum_anchor_rig_frames":
                continue
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} must be finite and positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_anchor_rig_frames": self.minimum_anchor_rig_frames,
            "outlier_mad_multiplier": self.outlier_mad_multiplier,
            "translation_residual_floor_m": self.translation_residual_floor_m,
            "rotation_residual_floor_deg": self.rotation_residual_floor_deg,
            "maximum_pair_translation_disagreement_m": (
                self.maximum_pair_translation_disagreement_m
            ),
            "maximum_pair_rotation_disagreement_deg": (
                self.maximum_pair_rotation_disagreement_deg
            ),
            "maximum_step_translation_m": self.maximum_step_translation_m,
            "maximum_step_rotation_deg": self.maximum_step_rotation_deg,
        }


def _validate_se3(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{label} rotation determinant is not one")
    return matrix


def _transforms_c2w_cv(value: Any, label: str) -> np.ndarray:
    matrix = _validate_se3(value, label).copy()
    matrix[:3, :3] = matrix[:3, :3] @ GL_TO_CV
    return _validate_se3(matrix, f"{label} converted to c2w_opencv")


def _normalise_image_path(value: str) -> str:
    normalised = value.replace("\\", "/").removeprefix("./")
    return normalised.removeprefix("camera/")


def _rotation_deg(matrix: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(matrix[:3, :3]).magnitude()))


def _average_transforms(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("cannot average an empty transform list")
    quaternions = Rotation.from_matrix(
        np.stack([value[:3, :3] for value in values])
    ).as_quat()
    reference = quaternions[0]
    for index in range(1, len(quaternions)):
        if float(np.dot(reference, quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    quaternion = np.mean(quaternions, axis=0)
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("correction rotations cannot be averaged")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    output[:3, 3] = np.mean(
        np.stack([value[:3, 3] for value in values]), axis=0
    )
    return output


def _interpolate_transform(
    left: np.ndarray,
    right: np.ndarray,
    fraction: float,
) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Slerp(
        [0.0, 1.0], Rotation.from_matrix([left[:3, :3], right[:3, :3]])
    )([fraction]).as_matrix()[0]
    output[:3, 3] = (1.0 - fraction) * left[:3, 3] + fraction * right[:3, 3]
    return output


def _robust_limit(values: list[float], floor: float, multiplier: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array)) if len(array) else 0.0
    mad = float(np.median(np.abs(array - median))) if len(array) else 0.0
    return max(floor, median + multiplier * 1.4826 * mad)


def _filter_anchor_outliers(
    anchors: list[dict[str, Any]],
    config: PoseCorrectionConfig,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    translation_residuals = [0.0] * len(anchors)
    rotation_residuals = [0.0] * len(anchors)
    for index in range(1, len(anchors) - 1):
        previous = anchors[index - 1]
        current = anchors[index]
        following = anchors[index + 1]
        span = following["timestamp_ns"] - previous["timestamp_ns"]
        if span <= 0:
            raise ValueError("anchor timestamps must be strictly increasing")
        fraction = (current["timestamp_ns"] - previous["timestamp_ns"]) / span
        predicted = _interpolate_transform(
            previous["correction"], following["correction"], fraction
        )
        residual = np.linalg.inv(predicted) @ current["correction"]
        translation_residuals[index] = float(np.linalg.norm(residual[:3, 3]))
        rotation_residuals[index] = _rotation_deg(residual)
    translation_limit = _robust_limit(
        translation_residuals[1:-1],
        config.translation_residual_floor_m,
        config.outlier_mad_multiplier,
    )
    rotation_limit = _robust_limit(
        rotation_residuals[1:-1],
        config.rotation_residual_floor_deg,
        config.outlier_mad_multiplier,
    )
    accepted: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        reasons: list[str] = []
        if anchor["pair_translation_disagreement_m"] > (
            config.maximum_pair_translation_disagreement_m
        ):
            reasons.append("pair_translation_disagreement")
        if anchor["pair_rotation_disagreement_deg"] > (
            config.maximum_pair_rotation_disagreement_deg
        ):
            reasons.append("pair_rotation_disagreement")
        if index not in {0, len(anchors) - 1}:
            if translation_residuals[index] > translation_limit:
                reasons.append("temporal_translation_outlier")
            if rotation_residuals[index] > rotation_limit:
                reasons.append("temporal_rotation_outlier")
        anchor["translation_residual_m"] = translation_residuals[index]
        anchor["rotation_residual_deg"] = rotation_residuals[index]
        anchor["accepted"] = not reasons
        anchor["rejection_reasons"] = reasons
        if not reasons:
            accepted.append(anchor)
    return accepted, {
        "translation_residual_limit_m": translation_limit,
        "rotation_residual_limit_deg": rotation_limit,
    }


def _interpolate_corrections(
    anchors: list[dict[str, Any]], timestamps_ns: np.ndarray
) -> tuple[list[np.ndarray], list[str]]:
    anchor_times = np.asarray([anchor["timestamp_ns"] for anchor in anchors], dtype=np.int64)
    origin = int(anchor_times[0])
    seconds = (anchor_times - origin).astype(np.float64) / 1_000_000_000.0
    query = (timestamps_ns - origin).astype(np.float64) / 1_000_000_000.0
    clipped = np.clip(query, seconds[0], seconds[-1])
    rotations = Slerp(
        seconds,
        Rotation.from_matrix(
            np.stack([anchor["correction"][:3, :3] for anchor in anchors])
        ),
    )(clipped).as_matrix()
    translations = np.column_stack(
        [
            np.interp(
                clipped,
                seconds,
                [anchor["correction"][axis, 3] for anchor in anchors],
            )
            for axis in range(3)
        ]
    )
    outputs: list[np.ndarray] = []
    sources: list[str] = []
    anchor_time_set = set(int(value) for value in anchor_times)
    for index, timestamp in enumerate(timestamps_ns):
        correction = np.eye(4, dtype=np.float64)
        correction[:3, :3] = rotations[index]
        correction[:3, 3] = translations[index]
        outputs.append(correction)
        timestamp_value = int(timestamp)
        if timestamp_value in anchor_time_set:
            sources.append("anchor")
        elif timestamp_value < int(anchor_times[0]) or timestamp_value > int(anchor_times[-1]):
            sources.append("extrapolated_constant")
        else:
            sources.append("interpolated")
    return outputs, sources


def _metric_gate(
    metrics: Mapping[str, Any] | None,
    key: str,
    *,
    strict_improvement: bool,
) -> dict[str, Any]:
    if metrics is None or key not in metrics:
        return {"status": "NOT_RUN", "baseline": None, "candidate": None}
    source = metrics[key]
    baseline = float(source["baseline"])
    candidate = float(source["candidate"])
    if not np.isfinite(baseline) or not np.isfinite(candidate) or min(baseline, candidate) < 0:
        raise ValueError(f"acceptance metric {key} must be finite and non-negative")
    passed = candidate < baseline if strict_improvement else candidate <= baseline
    return {
        "status": "PASS" if passed else "FAIL",
        "baseline": baseline,
        "candidate": candidate,
        "unit": str(source.get("unit", "score")),
    }


def _acceptance(
    metrics: Mapping[str, Any] | None,
    curve_is_smooth: bool,
) -> dict[str, Any]:
    gates = {
        "lidar_edge_error": _metric_gate(
            metrics, "lidar_edge_error", strict_improvement=True
        ),
        "low_resolution_lpips": _metric_gate(
            metrics, "low_resolution_lpips", strict_improvement=True
        ),
        "building_double_edge_score": _metric_gate(
            metrics, "building_double_edge_score", strict_improvement=False
        ),
        "correction_curve_no_jump": {
            "status": "PASS" if curve_is_smooth else "FAIL"
        },
    }
    accepted = all(gate["status"] == "PASS" for gate in gates.values())
    return {
        "gates": gates,
        "candidate_accepted_as_default": accepted,
        "default_pose_set": "keyframe_corrected" if accepted else "imgpose",
    }


def build_corrected_pose_set(
    dataset: dict[str, Any],
    transforms: dict[str, Any],
    *,
    transforms_sha256: str,
    config: PoseCorrectionConfig = PoseCorrectionConfig(),
    acceptance_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_sha = verify_dataset_manifest(dataset)
    config.validate()
    if len(transforms_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in transforms_sha256.lower()
    ):
        raise ValueError("transforms_sha256 must be a full SHA256 digest")
    images = {str(image["image_id"]): image for image in dataset["images"]}
    by_path: dict[str, dict[str, Any]] = {}
    for image in images.values():
        key = _normalise_image_path(str(image["path"]))
        if key in by_path:
            raise ValueError(f"dataset contains a duplicate image path: {key}")
        by_path[key] = image
    rig_frames = sorted(dataset["rig_frames"], key=lambda item: int(item["timestamp_ns"]))
    rig_by_id = {str(frame["rig_frame_id"]): frame for frame in rig_frames}
    rig_id_by_image: dict[str, str] = {}
    for frame in rig_frames:
        rig_frame_id = str(frame["rig_frame_id"])
        for image_id_value in frame["image_ids"]:
            image_id = str(image_id_value)
            if image_id in rig_id_by_image:
                raise ValueError(f"image {image_id} belongs to more than one Rig Frame")
            rig_id_by_image[image_id] = rig_frame_id

    grouped: dict[str, list[dict[str, Any]]] = {}
    matched_paths: set[str] = set()
    for index, frame in enumerate(transforms.get("frames", [])):
        path = _normalise_image_path(str(frame.get("file_path", "")))
        if path not in by_path:
            raise ValueError(f"transforms keyframe is not present in ImgPose dataset: {path}")
        if path in matched_paths:
            raise ValueError(f"transforms contains a duplicate keyframe path: {path}")
        matched_paths.add(path)
        image = by_path[path]
        rig_frame_id = rig_id_by_image.get(str(image["image_id"]), "")
        declared_rig_frame_id = str(image.get("rig_frame_id") or rig_frame_id)
        if declared_rig_frame_id != rig_frame_id:
            raise ValueError(f"keyframe {path} has inconsistent Rig Frame membership")
        if rig_frame_id not in rig_by_id:
            raise ValueError(f"keyframe {path} is not assigned to a complete Rig Frame")
        source = _validate_se3(image["c2w"], f"ImgPose {path}")
        target = _transforms_c2w_cv(
            frame.get("transform_matrix"), f"transforms frame {index}"
        )
        correction = target @ np.linalg.inv(source)
        grouped.setdefault(rig_frame_id, []).append(
            {
                "image_id": str(image["image_id"]),
                "side": str(image["side"]),
                "path": path,
                "correction": correction,
            }
        )
    anchors: list[dict[str, Any]] = []
    for rig_frame_id, matches in grouped.items():
        if len(matches) != 2 or {item["side"] for item in matches} != {"left", "right"}:
            raise ValueError(
                f"keyframe Rig {rig_frame_id} must contain one left and one right image"
            )
        matches.sort(key=lambda item: item["side"])
        disagreement = np.linalg.inv(matches[0]["correction"]) @ matches[1]["correction"]
        anchors.append(
            {
                "rig_frame_id": rig_frame_id,
                "timestamp_ns": int(rig_by_id[rig_frame_id]["timestamp_ns"]),
                "image_ids": sorted(item["image_id"] for item in matches),
                "correction": _average_transforms(
                    [item["correction"] for item in matches]
                ),
                "pair_translation_disagreement_m": float(
                    np.linalg.norm(disagreement[:3, 3])
                ),
                "pair_rotation_disagreement_deg": _rotation_deg(disagreement),
            }
        )
    anchors.sort(key=lambda item: item["timestamp_ns"])
    if len(anchors) < config.minimum_anchor_rig_frames:
        raise ValueError(
            f"only {len(anchors)} complete keyframe Rig anchors; "
            f"need {config.minimum_anchor_rig_frames}"
        )
    accepted, residual_limits = _filter_anchor_outliers(anchors, config)
    if len(accepted) < config.minimum_anchor_rig_frames:
        raise ValueError(
            f"only {len(accepted)} keyframe Rig anchors remain after robust filtering"
        )
    timestamps = np.asarray([int(frame["timestamp_ns"]) for frame in rig_frames], dtype=np.int64)
    corrections, sources = _interpolate_corrections(accepted, timestamps)

    corrected_images: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    step_translations: list[float] = []
    step_rotations: list[float] = []
    rig_translation_drift: list[float] = []
    rig_rotation_drift: list[float] = []
    previous: np.ndarray | None = None
    for frame, correction, source in zip(rig_frames, corrections, sources):
        image_ids = [str(value) for value in frame["image_ids"]]
        if len(image_ids) != 2 or any(image_id not in images for image_id in image_ids):
            raise ValueError(f"invalid Rig Frame image membership: {frame['rig_frame_id']}")
        corrected_by_id: dict[str, np.ndarray] = {}
        for image_id in image_ids:
            original = _validate_se3(images[image_id]["c2w"], f"image {image_id}")
            corrected = correction @ original
            corrected_by_id[image_id] = corrected
            corrected_images.append(
                {
                    "image_id": image_id,
                    "rig_frame_id": str(frame["rig_frame_id"]),
                    "side": str(images[image_id]["side"]),
                    "timestamp_ns": int(images[image_id]["timestamp_ns"]),
                    "source_pose_set": "imgpose",
                    "pose_convention": "c2w_opencv",
                    "c2w": corrected.tolist(),
                }
            )
        left_id = str(frame["left_image_id"])
        right_id = str(frame["right_image_id"])
        original_relative = np.linalg.inv(
            np.asarray(images[left_id]["c2w"], dtype=np.float64)
        ) @ np.asarray(images[right_id]["c2w"], dtype=np.float64)
        corrected_relative = np.linalg.inv(corrected_by_id[left_id]) @ corrected_by_id[right_id]
        relative_drift = np.linalg.inv(original_relative) @ corrected_relative
        rig_translation_drift.append(float(np.linalg.norm(relative_drift[:3, 3])))
        rig_rotation_drift.append(_rotation_deg(relative_drift))
        step_translation = 0.0
        step_rotation = 0.0
        if previous is not None:
            step = np.linalg.inv(previous) @ correction
            step_translation = float(np.linalg.norm(step[:3, 3]))
            step_rotation = _rotation_deg(step)
            step_translations.append(step_translation)
            step_rotations.append(step_rotation)
        curve.append(
            {
                "rig_frame_id": str(frame["rig_frame_id"]),
                "timestamp_ns": int(frame["timestamp_ns"]),
                "source": source,
                "translation_xyz_m": correction[:3, 3].tolist(),
                "translation_magnitude_m": float(np.linalg.norm(correction[:3, 3])),
                "rotation_xyzw": Rotation.from_matrix(correction[:3, :3]).as_quat().tolist(),
                "rotation_magnitude_deg": _rotation_deg(correction),
                "step_translation_m": step_translation,
                "step_rotation_deg": step_rotation,
            }
        )
        previous = correction
    curve_is_smooth = (
        max(step_translations, default=0.0) <= config.maximum_step_translation_m
        and max(step_rotations, default=0.0) <= config.maximum_step_rotation_deg
    )
    if max(rig_translation_drift, default=0.0) > 1e-9 or max(
        rig_rotation_drift, default=0.0
    ) > 1e-6:
        raise ValueError("propagated corrections changed the fixed stereo baseline")

    anchor_report = []
    for anchor in anchors:
        anchor_report.append(
            {
                key: value
                for key, value in anchor.items()
                if key != "correction"
            }
            | {
                "translation_xyz_m": anchor["correction"][:3, 3].tolist(),
                "rotation_xyzw": Rotation.from_matrix(
                    anchor["correction"][:3, :3]
                ).as_quat().tolist(),
            }
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "keyframe_se3_rig_interpolation_v1",
        "pose_set_id": "keyframe_corrected",
        "dataset_manifest_sha256": dataset_sha,
        "transforms_sha256": transforms_sha256,
        "configuration": config.to_dict(),
        "source_pose_sets": ["imgpose", "transforms_keyframes"],
        "anchors": anchor_report,
        "anchor_filter": {
            "matched_keyframe_images": len(matched_paths),
            "candidate_rig_frames": len(anchors),
            "accepted_rig_frames": len(accepted),
            "rejected_rig_frames": len(anchors) - len(accepted),
            **residual_limits,
        },
        "images": sorted(
            corrected_images, key=lambda item: (item["timestamp_ns"], item["side"])
        ),
        "correction_curve": curve,
        "diagnostics": {
            "correction_translation_m": distribution(
                [item["translation_magnitude_m"] for item in curve]
            ),
            "correction_rotation_deg": distribution(
                [item["rotation_magnitude_deg"] for item in curve]
            ),
            "step_translation_m": distribution(step_translations),
            "step_rotation_deg": distribution(step_rotations),
            "curve_is_smooth": curve_is_smooth,
            "rig_baseline_translation_drift_m": distribution(rig_translation_drift),
            "rig_baseline_rotation_drift_deg": distribution(rig_rotation_drift),
            "rig_baseline_preserved": True,
        },
        "acceptance": _acceptance(acceptance_metrics, curve_is_smooth),
        "summary": {
            "rig_frame_count": len(rig_frames),
            "image_count": len(corrected_images),
            "original_inputs_overwritten": False,
        },
    }
    result["pose_set_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def verify_pose_set_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("pose_set_manifest_sha256", ""))
    if not expected:
        raise ValueError("pose set manifest has no pose_set_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("pose_set_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"pose set manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    images = manifest.get("images", [])
    image_ids = [str(image.get("image_id", "")) for image in images]
    if not images or not all(image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError("pose set manifest contains invalid or duplicate images")
    rig_sides: dict[str, set[str]] = {}
    for image in images:
        _validate_se3(image.get("c2w"), f"corrected image {image.get('image_id')}")
        if image.get("pose_convention") != "c2w_opencv":
            raise ValueError("corrected pose convention must be c2w_opencv")
        rig_frame_id = str(image.get("rig_frame_id", ""))
        side = str(image.get("side", ""))
        if not rig_frame_id or side not in {"left", "right"}:
            raise ValueError("corrected image has invalid Rig membership")
        rig_sides.setdefault(rig_frame_id, set()).add(side)
    if any(sides != {"left", "right"} for sides in rig_sides.values()):
        raise ValueError("each corrected Rig Frame must contain left and right images")
    summary = manifest.get("summary", {})
    if summary.get("image_count") != len(images) or summary.get("rig_frame_count") != len(
        rig_sides
    ):
        raise ValueError("pose set summary counts are inconsistent")
    if len(manifest.get("correction_curve", [])) != len(rig_sides):
        raise ValueError("pose correction curve does not cover every Rig Frame")
    if manifest.get("diagnostics", {}).get("rig_baseline_preserved") is not True:
        raise ValueError("pose set does not preserve the fixed stereo baseline")
    if manifest.get("acceptance", {}).get("default_pose_set") not in {
        "imgpose",
        "keyframe_corrected",
    }:
        raise ValueError("pose set manifest contains an invalid default pose set")
    acceptance = manifest["acceptance"]
    gate_pass = all(
        gate.get("status") == "PASS" for gate in acceptance.get("gates", {}).values()
    )
    if acceptance.get("candidate_accepted_as_default") != gate_pass:
        raise ValueError("default pose decision is inconsistent with acceptance gates")
    expected_default = "keyframe_corrected" if gate_pass else "imgpose"
    if acceptance.get("default_pose_set") != expected_default:
        raise ValueError("default pose set is inconsistent with acceptance gates")
    return actual


def _curve_svg(manifest: dict[str, Any]) -> str:
    curve = manifest["correction_curve"]
    width, height, padding = 960, 520, 55
    plot_width = width - padding * 2
    panel_height = (height - padding * 3) / 2

    def points(key: str, top: float) -> str:
        values = np.asarray([float(item[key]) for item in curve], dtype=np.float64)
        maximum = max(float(np.max(values)), np.finfo(np.float64).eps)
        coordinates = []
        for index, value in enumerate(values):
            x = padding + plot_width * index / max(1, len(values) - 1)
            y = top + panel_height * (1.0 - float(value) / maximum)
            coordinates.append(f"{x:.3f},{y:.3f}")
        return " ".join(coordinates)

    translation_points = points("translation_magnitude_m", padding)
    rotation_points = points("rotation_magnitude_deg", padding * 2 + panel_height)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f8fafc"/>
<g stroke="#94a3b8" fill="none"><path d="M{padding} {padding}v{panel_height}h{plot_width}"/><path d="M{padding} {padding * 2 + panel_height}v{panel_height}h{plot_width}"/></g>
<polyline points="{translation_points}" fill="none" stroke="#0369a1" stroke-width="2"/>
<polyline points="{rotation_points}" fill="none" stroke="#b45309" stroke-width="2"/>
<g font-family="system-ui" font-size="15" fill="#0f172a"><text x="{padding}" y="28">Correction translation magnitude (m)</text><text x="{padding}" y="{padding + panel_height + 38}">Correction rotation magnitude (deg)</text><text x="{width - 210}" y="{height - 12}">Rig timeline →</text></g>
</svg>
"""


def _report_html(manifest: dict[str, Any]) -> str:
    diagnostics = manifest["diagnostics"]
    acceptance = manifest["acceptance"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(str(gate['status']))}</td>"
        f"<td>{html.escape(str(gate.get('baseline')))}</td>"
        f"<td>{html.escape(str(gate.get('candidate')))}</td>"
        "</tr>"
        for name, gate in acceptance["gates"].items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Pose Correction Report</title>
<style>body{{font:14px system-ui;margin:2rem;color:#172033}}table{{border-collapse:collapse}}th,td{{border:1px solid #cbd5e1;padding:.45rem}}th{{background:#e2e8f0}}img{{max-width:100%}}code{{word-break:break-all}}</style></head>
<body><h1>关键帧位姿修正传播报告</h1><p>默认位姿：<strong>{html.escape(acceptance['default_pose_set'])}</strong></p>
<p>Manifest SHA256：<code>{manifest['pose_set_manifest_sha256']}</code></p>
<ul><li>Anchor：{manifest['anchor_filter']['accepted_rig_frames']} / {manifest['anchor_filter']['candidate_rig_frames']}</li>
<li>平移修正 p50/p95/max：{diagnostics['correction_translation_m']['p50']:.6f} / {diagnostics['correction_translation_m']['p95']:.6f} / {diagnostics['correction_translation_m']['max']:.6f} m</li>
<li>旋转修正 p50/p95/max：{diagnostics['correction_rotation_deg']['p50']:.6f} / {diagnostics['correction_rotation_deg']['p95']:.6f} / {diagnostics['correction_rotation_deg']['max']:.6f} deg</li>
<li>曲线无突跳：{diagnostics['curve_is_smooth']}</li><li>Rig 基线保持：{diagnostics['rig_baseline_preserved']}</li></ul>
<img src="{CURVE_NAME}" alt="pose correction curve">
<h2>默认位姿门槛</h2><table><thead><tr><th>Gate</th><th>Status</th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>{rows}</tbody></table></body></html>
"""


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_pose_set_outputs(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    verify_pose_set_manifest(manifest)
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"pose output is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"pose output is not empty: {output_dir}; pass --force")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(output_dir / POSE_MANIFEST_NAME, payload)
    _atomic_write(output_dir / CURVE_NAME, _curve_svg(manifest).encode("utf-8"))
    _atomic_write(output_dir / REPORT_NAME, _report_html(manifest).encode("utf-8"))
