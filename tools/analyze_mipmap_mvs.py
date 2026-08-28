#!/usr/bin/env python3
"""Extract MipMap AT pose corrections and four-face undistortion geometry.

The tool is intentionally read-only with respect to a MipMap task.  It joins:

* ``task.json``: input poses, timestamps, camera IDs, and calibrations;
* ``result/report/report.json``: explicit per-image POS corrections;
* ``result/AT/mvs.xml``: optimized physical-camera poses;
* ``result/AT/mvs_undistort.xml``: derived zero-distortion views.

Outputs are analysis artifacts only: two CSV files and one JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class XmlPhoto:
    photo_id: int
    image_path: str
    center: np.ndarray
    rotation_w2c: np.ndarray


@dataclass(frozen=True)
class XmlGroup:
    index: int
    width: int
    height: int
    focal_px: float
    principal_x: float
    principal_y: float
    distortion: tuple[float, float, float, float]
    photos: tuple[XmlPhoto, ...]


def _matrix_from_photo(photo: ET.Element) -> np.ndarray:
    return np.asarray(
        [
            [float(photo.findtext(f"./Pose/Rotation/M_{row}{column}")) for column in range(3)]
            for row in range(3)
        ],
        dtype=np.float64,
    )


def _read_groups(path: Path) -> list[XmlGroup]:
    root = ET.parse(path).getroot()
    groups: list[XmlGroup] = []
    for group_index, group in enumerate(root.findall(".//Photogroup")):
        photos: list[XmlPhoto] = []
        for photo in group.findall("./Photo"):
            photos.append(
                XmlPhoto(
                    photo_id=int(photo.findtext("Id")),
                    image_path=str(photo.findtext("ImagePath")),
                    center=np.asarray(
                        [float(photo.findtext(f"./Pose/Center/{axis}")) for axis in "xyz"],
                        dtype=np.float64,
                    ),
                    rotation_w2c=_matrix_from_photo(photo),
                )
            )
        groups.append(
            XmlGroup(
                index=group_index,
                width=int(group.findtext("./ImageDimensions/Width")),
                height=int(group.findtext("./ImageDimensions/Height")),
                focal_px=float(group.findtext("FocalLengthPixels")),
                principal_x=float(group.findtext("./PrincipalPoint/x")),
                principal_y=float(group.findtext("./PrincipalPoint/y")),
                distortion=tuple(
                    float(group.findtext(f"./Distortion/K{index}")) for index in range(1, 5)
                ),
                photos=tuple(photos),
            )
        )
    return groups


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _flatten_matrix(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_{row}{column}": float(matrix[row, column])
        for row in range(3)
        for column in range(3)
    }


def _face_coverage_probe(face_specs: list[dict[str, Any]]) -> dict[str, float]:
    """Measure polar coverage of one physical camera's face set at 0.25 degrees."""
    theta_deg = np.arange(0.0, 95.0 + 0.25, 0.25, dtype=np.float64)
    azimuth = np.radians(np.arange(0.0, 360.0, 0.25, dtype=np.float64))
    coverage_by_theta: list[float] = []
    for theta_value in theta_deg:
        theta = math.radians(float(theta_value))
        directions = np.column_stack(
            [
                math.sin(theta) * np.cos(azimuth),
                math.sin(theta) * np.sin(azimuth),
                np.full_like(azimuth, math.cos(theta)),
            ]
        )
        covered = np.zeros(len(directions), dtype=bool)
        for face in face_specs:
            face_to_source = np.asarray(face["face_to_source_rotation"], dtype=np.float64)
            directions_face = directions @ face_to_source
            z = directions_face[:, 2]
            safe_z = np.where(np.abs(z) > 1e-12, z, 1.0)
            u = face["focal_px"] * directions_face[:, 0] / safe_z + face["principal_point"][0]
            v = face["focal_px"] * directions_face[:, 1] / safe_z + face["principal_point"][1]
            tolerance = 1e-6
            covered |= (
                (z > 0.0)
                & (u >= -tolerance)
                & (u <= face["width"] + tolerance)
                & (v >= -tolerance)
                & (v <= face["height"] + tolerance)
            )
        coverage_by_theta.append(float(np.mean(covered)))
    fully_covered = [
        float(theta) for theta, fraction in zip(theta_deg, coverage_by_theta) if fraction >= 1.0
    ]
    partly_covered = [
        float(theta) for theta, fraction in zip(theta_deg, coverage_by_theta) if fraction > 0.0
    ]
    at_least_one_percent = [
        float(theta) for theta, fraction in zip(theta_deg, coverage_by_theta) if fraction >= 0.01
    ]
    at_least_half = [
        float(theta) for theta, fraction in zip(theta_deg, coverage_by_theta) if fraction >= 0.5
    ]
    probes = {
        str(value): coverage_by_theta[int(round(value / 0.25))]
        for value in (70, 72, 75, 76, 77, 80, 85, 90, 95)
    }
    return {
        "probe_step_deg": 0.25,
        "max_fully_covered_polar_deg": max(fully_covered),
        "max_any_covered_polar_deg": max(partly_covered),
        "max_at_least_50pct_azimuth_covered_polar_deg": max(at_least_half),
        "max_at_least_1pct_azimuth_covered_polar_deg": max(at_least_one_percent),
        "azimuth_coverage_fraction_by_polar_deg": probes,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(task_root: Path, output_root: Path) -> dict[str, Any]:
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    report = json.loads(
        (task_root / "result" / "report" / "report.json").read_text(encoding="utf-8")
    )
    physical_groups = _read_groups(task_root / "result" / "AT" / "mvs.xml")
    face_groups = _read_groups(task_root / "result" / "AT" / "mvs_undistort.xml")

    if len(physical_groups) != len(task["camera_meta_data"]):
        raise ValueError("physical XML photogroup count does not match task camera count")
    if len(face_groups) % len(physical_groups) != 0:
        raise ValueError("derived photogroups cannot be evenly assigned to physical cameras")

    input_by_id = {int(item["id"]): item for item in task["image_meta_data"]}
    pos_diff_by_id = {
        int(item["id"]): np.asarray(item["pos_diff"], dtype=np.float64)
        for item in report["image_POS_diff"]
    }
    optimized_by_id = {
        photo.photo_id: photo for group in physical_groups for photo in group.photos
    }
    expected_ids = set(input_by_id)
    if set(pos_diff_by_id) != expected_ids or set(optimized_by_id) != expected_ids:
        raise ValueError("task, report, and mvs.xml image ID sets differ")

    pose_rows: list[dict[str, Any]] = []
    gauge_shifts: list[np.ndarray] = []
    correction_norms: dict[int, list[float]] = {group.index + 1: [] for group in physical_groups}
    rotation_corrections: dict[int, list[float]] = {
        group.index + 1: [] for group in physical_groups
    }
    for image_id in sorted(expected_ids):
        source = input_by_id[image_id]
        metadata = source["meta_data"]
        camera_id = int(metadata["camera_id"])
        raw_position = np.asarray(metadata["pos"], dtype=np.float64)
        position_diff = pos_diff_by_id[image_id]
        corrected_position = raw_position - position_diff
        raw_rotation = np.asarray(metadata["orientation"], dtype=np.float64).reshape(3, 3)
        optimized = optimized_by_id[image_id]
        delta_rotation = optimized.rotation_w2c @ raw_rotation.T
        rotation_deg = _rotation_angle_deg(delta_rotation)
        correction_norm = float(np.linalg.norm(position_diff))
        gauge_shifts.append(optimized.center - corrected_position)
        correction_norms[camera_id].append(correction_norm)
        rotation_corrections[camera_id].append(rotation_deg)

        row: dict[str, Any] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "timestamp": float(metadata["timestamp"]),
            "source_path": source["path"],
            "raw_x": float(raw_position[0]),
            "raw_y": float(raw_position[1]),
            "raw_z": float(raw_position[2]),
            "pos_diff_raw_minus_corrected_x": float(position_diff[0]),
            "pos_diff_raw_minus_corrected_y": float(position_diff[1]),
            "pos_diff_raw_minus_corrected_z": float(position_diff[2]),
            "pos_correction_norm_m": correction_norm,
            "corrected_x": float(corrected_position[0]),
            "corrected_y": float(corrected_position[1]),
            "corrected_z": float(corrected_position[2]),
            "mvs_center_x": float(optimized.center[0]),
            "mvs_center_y": float(optimized.center[1]),
            "mvs_center_z": float(optimized.center[2]),
            "rotation_correction_deg": rotation_deg,
        }
        row.update(_flatten_matrix("raw_w2c", raw_rotation))
        row.update(_flatten_matrix("optimized_w2c", optimized.rotation_w2c))
        row.update(_flatten_matrix("delta_w2c", delta_rotation))
        pose_rows.append(row)

    gauge_array = np.stack(gauge_shifts)
    gauge_shift = np.mean(gauge_array, axis=0)
    gauge_max_deviation = float(
        np.max(np.linalg.norm(gauge_array - gauge_shift[None, :], axis=1))
    )

    faces_per_camera = len(face_groups) // len(physical_groups)
    face_names = (
        "yaw_neg_35",
        "yaw_pos_35",
        "pitch_up_56",
        "pitch_down_56",
    )
    if faces_per_camera != len(face_names):
        face_names = tuple(f"face_{index}" for index in range(faces_per_camera))

    face_rows: list[dict[str, Any]] = []
    face_specs: list[dict[str, Any]] = []
    for group in face_groups:
        camera_zero_index = group.index // faces_per_camera
        camera_id = camera_zero_index + 1
        face_local_index = group.index % faces_per_camera
        face_name = face_names[face_local_index]
        physical_photos = physical_groups[camera_zero_index].photos
        group_relative_rotations: list[np.ndarray] = []
        center_errors: list[float] = []

        for derived in group.photos:
            physical = min(
                physical_photos,
                key=lambda candidate: float(np.linalg.norm(derived.center - candidate.center)),
            )
            center_error = float(np.linalg.norm(derived.center - physical.center))
            relative_source_to_face = derived.rotation_w2c @ physical.rotation_w2c.T
            face_to_source = relative_source_to_face.T
            group_relative_rotations.append(relative_source_to_face)
            center_errors.append(center_error)
            source = input_by_id[physical.photo_id]
            row = {
                "source_image_id": physical.photo_id,
                "camera_id": camera_id,
                "source_path": source["path"],
                "face_name": face_name,
                "derived_group": group.index,
                "derived_image_id": derived.photo_id,
                "derived_path": derived.image_path,
                "center_error_m": center_error,
                "width": group.width,
                "height": group.height,
                "focal_px": group.focal_px,
                "principal_x": group.principal_x,
                "principal_y": group.principal_y,
            }
            row.update(_flatten_matrix("face_to_source", face_to_source))
            face_rows.append(row)

        reference = group_relative_rotations[0]
        rotation_dispersion = np.asarray(
            [_rotation_angle_deg(relative @ reference.T) for relative in group_relative_rotations]
        )
        face_to_source = reference.T
        optical_axis_source = face_to_source[:, 2]
        face_specs.append(
            {
                "camera_id": camera_id,
                "derived_group": group.index,
                "face_name": face_name,
                "width": group.width,
                "height": group.height,
                "focal_px": group.focal_px,
                "principal_point": [group.principal_x, group.principal_y],
                "distortion": list(group.distortion),
                "pinhole_fov_deg": {
                    "horizontal": math.degrees(
                        2.0 * math.atan(group.width / (2.0 * group.focal_px))
                    ),
                    "vertical": math.degrees(
                        2.0 * math.atan(group.height / (2.0 * group.focal_px))
                    ),
                },
                "face_to_source_rotation": face_to_source.tolist(),
                "optical_axis_in_source": optical_axis_source.tolist(),
                "center_error_max_m": max(center_errors),
                "relative_rotation_dispersion_max_deg": float(np.max(rotation_dispersion)),
            }
        )

    initial_intrinsics = {int(item["id"]): item for item in report["initial_camera_parameters"]}
    optimized_intrinsics = {int(item["id"]): item for item in report["AT_camera_parameters"]}
    intrinsic_changes = []
    for camera_id in sorted(initial_intrinsics):
        before = np.asarray(initial_intrinsics[camera_id]["parameters"], dtype=np.float64)
        after = np.asarray(optimized_intrinsics[camera_id]["parameters"], dtype=np.float64)
        intrinsic_changes.append(
            {
                "camera_id": camera_id,
                "parameter_order": ["focal_px", "cx", "cy", "k1", "k2", "k3", "k4"],
                "before": before.tolist(),
                "after": after.tolist(),
                "delta_after_minus_before": (after - before).tolist(),
            }
        )

    stereo_pair_geometry: dict[str, Any] | None = None
    if len(physical_groups) == 2 and len(physical_groups[0].photos) == len(physical_groups[1].photos):
        raw_baselines: list[float] = []
        corrected_baselines: list[float] = []
        baseline_vector_changes: list[float] = []
        relative_rotation_changes: list[float] = []
        timestamp_deltas_ms: list[float] = []
        for left, right in zip(physical_groups[0].photos, physical_groups[1].photos):
            left_input = input_by_id[left.photo_id]["meta_data"]
            right_input = input_by_id[right.photo_id]["meta_data"]
            left_raw = np.asarray(left_input["pos"], dtype=np.float64)
            right_raw = np.asarray(right_input["pos"], dtype=np.float64)
            left_corrected = left_raw - pos_diff_by_id[left.photo_id]
            right_corrected = right_raw - pos_diff_by_id[right.photo_id]
            raw_baseline = right_raw - left_raw
            corrected_baseline = right_corrected - left_corrected
            raw_baselines.append(float(np.linalg.norm(raw_baseline)))
            corrected_baselines.append(float(np.linalg.norm(corrected_baseline)))
            baseline_vector_changes.append(float(np.linalg.norm(corrected_baseline - raw_baseline)))

            left_raw_rotation = np.asarray(left_input["orientation"], dtype=np.float64).reshape(3, 3)
            right_raw_rotation = np.asarray(right_input["orientation"], dtype=np.float64).reshape(3, 3)
            raw_relative = right_raw_rotation @ left_raw_rotation.T
            optimized_relative = right.rotation_w2c @ left.rotation_w2c.T
            relative_rotation_changes.append(
                _rotation_angle_deg(optimized_relative @ raw_relative.T)
            )
            timestamp_deltas_ms.append(
                abs(float(right_input["timestamp"]) - float(left_input["timestamp"])) * 1000.0
            )
        stereo_pair_geometry = {
            "pair_count": len(raw_baselines),
            "timestamp_delta_ms": _percentiles(np.asarray(timestamp_deltas_ms)),
            "raw_baseline_length_m": _percentiles(np.asarray(raw_baselines)),
            "corrected_baseline_length_m": _percentiles(np.asarray(corrected_baselines)),
            "baseline_vector_change_m": _percentiles(np.asarray(baseline_vector_changes)),
            "relative_rotation_change_deg": _percentiles(np.asarray(relative_rotation_changes)),
            "interpretation": "optimized image poses are not constrained to a fixed stereo rig",
        }

    source_paths = [Path(item["path"]) for item in task["image_meta_data"]]
    derived_paths = [Path(row["derived_path"]) for row in face_rows]
    source_bytes = sum(path.stat().st_size for path in source_paths if path.is_file())
    derived_bytes = sum(path.stat().st_size for path in derived_paths if path.is_file())
    one_camera_face_specs = [face for face in face_specs if face["camera_id"] == 1]

    all_norms = np.asarray(
        [value for values in correction_norms.values() for value in values], dtype=np.float64
    )
    all_rotations = np.asarray(
        [value for values in rotation_corrections.values() for value in values], dtype=np.float64
    )
    summary = {
        "task_root": str(task_root),
        "counts": {
            "physical_cameras": len(physical_groups),
            "source_images": len(expected_ids),
            "faces_per_source_image": faces_per_camera,
            "derived_images": len(face_rows),
            "mvs_tie_points": len(ET.parse(task_root / "result" / "AT" / "mvs.xml").getroot().findall(".//TiePoint")),
            "mvs_undistort_tie_points": len(
                ET.parse(task_root / "result" / "AT" / "mvs_undistort.xml")
                .getroot()
                .findall(".//TiePoint")
            ),
        },
        "pose_conventions": {
            "xml_rotation": "world_to_camera",
            "pos_diff": "raw_position_minus_corrected_position",
            "mvs_center_relation": "mvs_center = corrected_position + gauge_translation",
            "gauge_translation": gauge_shift.tolist(),
            "gauge_translation_max_deviation_m": gauge_max_deviation,
        },
        "position_correction_norm_m": {
            "all": _percentiles(all_norms),
            **{
                f"camera_{camera_id}": _percentiles(np.asarray(values))
                for camera_id, values in correction_norms.items()
            },
        },
        "rotation_correction_deg": {
            "all": _percentiles(all_rotations),
            **{
                f"camera_{camera_id}": _percentiles(np.asarray(values))
                for camera_id, values in rotation_corrections.items()
            },
        },
        "intrinsic_changes": intrinsic_changes,
        "face_specs": face_specs,
        "face_geometry": {
            "pixels_per_source_image": sum(
                int(face["width"]) * int(face["height"]) for face in one_camera_face_specs
            ),
            "source_image_pixels": int(task["image_meta_data"][0]["meta_data"]["width"])
            * int(task["image_meta_data"][0]["meta_data"]["height"]),
            "coverage_probe": _face_coverage_probe(one_camera_face_specs),
        },
        "stereo_pair_geometry": stereo_pair_geometry,
        "storage": {
            "source_image_bytes": source_bytes,
            "derived_face_bytes": derived_bytes,
            "derived_to_source_ratio": derived_bytes / source_bytes if source_bytes else None,
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "pose_corrections.csv", pose_rows)
    _write_csv(output_root / "face4_mapping.csv", face_rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.task_root.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
