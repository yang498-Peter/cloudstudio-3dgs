from __future__ import annotations

import bisect
import hashlib
from dataclasses import replace
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.schema import CameraRecord, ImageRecord, RigFrameRecord


DEFAULT_PAIR_TOLERANCE_NS = 50_000_000


def transform_matrix(transform: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(transform.get("rotation"), dtype=np.float64)
    position = np.asarray(transform.get("position"), dtype=np.float64)
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("transform_from_lidar must contain a 3x3 rotation and 3-vector position")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def rotation_error_rad(actual: np.ndarray, expected: np.ndarray) -> float:
    delta = expected[:3, :3].T @ actual[:3, :3]
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    skew = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=np.float64,
    )
    sine = float(np.linalg.norm(skew) / 2.0)
    return float(np.arctan2(sine, cosine))


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def pair_candidates(
    left: list[ImageRecord],
    right: list[ImageRecord],
    tolerance_ns: int,
) -> list[tuple[ImageRecord, ImageRecord]]:
    right_timestamps = [image.timestamp_ns for image in right]
    candidates: list[tuple[int, int, int, str, str, ImageRecord, ImageRecord]] = []
    for left_image in left:
        start = bisect.bisect_left(right_timestamps, left_image.timestamp_ns - tolerance_ns)
        end = bisect.bisect_right(right_timestamps, left_image.timestamp_ns + tolerance_ns)
        for right_image in right[start:end]:
            candidates.append(
                (
                    abs(right_image.timestamp_ns - left_image.timestamp_ns),
                    left_image.timestamp_ns,
                    right_image.timestamp_ns,
                    left_image.image_id,
                    right_image.image_id,
                    left_image,
                    right_image,
                )
            )
    used_left: set[str] = set()
    used_right: set[str] = set()
    pairs: list[tuple[ImageRecord, ImageRecord]] = []
    for *_key, left_image, right_image in sorted(candidates):
        if left_image.image_id in used_left or right_image.image_id in used_right:
            continue
        used_left.add(left_image.image_id)
        used_right.add(right_image.image_id)
        pairs.append((left_image, right_image))
    return sorted(pairs, key=lambda pair: (pair[0].timestamp_ns, pair[1].timestamp_ns))


def build_stereo_rig(
    cameras: list[CameraRecord],
    images: list[ImageRecord],
    *,
    tolerance_ns: int = DEFAULT_PAIR_TOLERANCE_NS,
) -> tuple[list[ImageRecord], list[RigFrameRecord], dict[str, Any], dict[str, Any]]:
    if tolerance_ns <= 0:
        raise ValueError("pairing tolerance must be positive")
    camera_by_side = {camera.side: camera for camera in cameras}
    if set(camera_by_side) != {"left", "right"}:
        raise ValueError("a stereo S1 rig requires exactly left and right cameras")
    left = sorted((image for image in images if image.side == "left"), key=lambda image: image.timestamp_ns)
    right = sorted((image for image in images if image.side == "right"), key=lambda image: image.timestamp_ns)
    pairs = pair_candidates(left, right, tolerance_ns)

    left_from_lidar = transform_matrix(camera_by_side["left"].transform_from_lidar)
    right_from_lidar = transform_matrix(camera_by_side["right"].transform_from_lidar)
    expected_right_to_left = left_from_lidar @ np.linalg.inv(right_from_lidar)
    replacements: dict[str, ImageRecord] = {}
    rig_frames: list[RigFrameRecord] = []
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    timestamp_deltas: list[float] = []
    observed_translations: list[list[float]] = []
    observed_rotation_errors: list[float] = []

    for left_image, right_image in pairs:
        identity = f"{left_image.image_id}\0{right_image.image_id}".encode("ascii")
        rig_frame_id = f"rig_{hashlib.sha256(identity).hexdigest()[:24]}"
        replacements[left_image.image_id] = replace(left_image, rig_frame_id=rig_frame_id)
        replacements[right_image.image_id] = replace(right_image, rig_frame_id=rig_frame_id)
        delta_ns = right_image.timestamp_ns - left_image.timestamp_ns
        rig_frames.append(
            RigFrameRecord(
                rig_frame_id=rig_frame_id,
                timestamp_ns=(left_image.timestamp_ns + right_image.timestamp_ns) // 2,
                left_image_id=left_image.image_id,
                right_image_id=right_image.image_id,
                image_ids=[left_image.image_id, right_image.image_id],
                timestamp_delta_ns=delta_ns,
            )
        )
        world_from_left = np.asarray(left_image.c2w, dtype=np.float64)
        world_from_right = np.asarray(right_image.c2w, dtype=np.float64)
        actual_right_to_left = np.linalg.inv(world_from_left) @ world_from_right
        translation_errors.append(
            float(np.linalg.norm(actual_right_to_left[:3, 3] - expected_right_to_left[:3, 3]))
        )
        rotation_errors.append(rotation_error_rad(actual_right_to_left, expected_right_to_left))
        timestamp_deltas.append(float(abs(delta_ns)))
        observed_translations.append(actual_right_to_left[:3, 3].tolist())
        observed_rotation_errors.append(rotation_error_rad(actual_right_to_left, expected_right_to_left))

    paired_ids = set(replacements)
    updated_images = [replacements.get(image.image_id, image) for image in images]
    unpaired_left = [image.path for image in left if image.image_id not in paired_ids]
    unpaired_right = [image.path for image in right if image.image_id not in paired_ids]
    translation_array = np.asarray(observed_translations, dtype=np.float64)
    translation_scatter = (
        np.std(translation_array, axis=0).tolist() if len(translation_array) else [0.0, 0.0, 0.0]
    )
    intrinsic_difference = {
        key: float(camera_by_side["right"].intrinsic[key] - camera_by_side["left"].intrinsic[key])
        for key in ("fl_x", "fl_y", "cx", "cy")
    }
    left_distortion = camera_by_side["left"].distortion["params"]
    right_distortion = camera_by_side["right"].distortion["params"]
    intrinsic_difference.update(
        {
            key: float(right_distortion[key] - left_distortion[key])
            for key in ("k1", "k2", "k3", "k4")
        }
    )
    diagnostics = {
        "pair_count": len(pairs),
        "unpaired_left": unpaired_left,
        "unpaired_right": unpaired_right,
        "timestamp_delta_ns": distribution(timestamp_deltas),
        "relative_translation_error_m": distribution(translation_errors),
        "relative_rotation_error_rad": distribution(rotation_errors),
        "relative_translation_scatter_xyz_m": translation_scatter,
        "relative_rotation_scatter_rad": float(np.std(observed_rotation_errors))
        if observed_rotation_errors
        else 0.0,
        "intrinsic_right_minus_left": intrinsic_difference,
    }
    rig = {
        "rig_id": "mvp_s1_stereo",
        "extrinsics_fixed": True,
        "pairing_tolerance_ns": tolerance_ns,
        "transform_convention": "camera_from_lidar",
        "left_from_lidar": left_from_lidar.tolist(),
        "right_from_lidar": right_from_lidar.tolist(),
        "expected_right_to_left": expected_right_to_left.tolist(),
    }
    return updated_images, rig_frames, rig, diagnostics
