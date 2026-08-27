"""Publish an accepted BA candidate as a signed Trainer dataset manifest."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.ba.report import verify_ba_report
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest
from cloudstudio_3dgs.evaluation.splits import verify_split_manifest


def directory_sha256(path: Path) -> str:
    """Hash a COLMAP model directory using stable relative paths and bytes."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"COLMAP model directory contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _normalise_name(value: str) -> str:
    return value.replace("\\", "/").removeprefix("camera/")


def _rigid_c2w(image: Any, name: str) -> list[list[float]]:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3] = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    c2w = np.linalg.inv(world_to_camera)
    if (
        not np.all(np.isfinite(c2w))
        or not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
        or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(c2w[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError(f"BA candidate image has an invalid rigid pose: {name}")
    return c2w.tolist()


def _camera_parameters(camera: Any, stable_id: str) -> dict[str, float]:
    if str(camera.model_name) not in {"OPENCV_FISHEYE", "PINHOLE"}:
        raise ValueError(
            f"BA candidate camera {stable_id} uses unsupported model {camera.model_name}"
        )
    names = [value.strip() for value in camera.params_info.split(",")]
    values = [float(value) for value in camera.params]
    if len(names) != len(values) or any(not np.isfinite(value) for value in values):
        raise ValueError(f"BA candidate camera {stable_id} has invalid parameters")
    raw = dict(zip(names, values))

    def required(*aliases: str) -> float:
        for alias in aliases:
            if alias in raw:
                return raw[alias]
        raise ValueError(
            f"BA candidate camera {stable_id} has no parameter {aliases[0]}"
        )

    result = {
        "fl_x": required("fx", "f"),
        "fl_y": required("fy", "f"),
        "cx": required("cx"),
        "cy": required("cy"),
    }
    if str(camera.model_name) == "OPENCV_FISHEYE":
        result.update({key: required(key) for key in ("k1", "k2", "k3", "k4")})
    return result


def verify_independent_at_report(report: dict[str, Any]) -> str:
    """Verify the signed product-style independent AT report."""
    expected = str(report.get("report_sha256", ""))
    if len(expected) != 64:
        raise ValueError("independent AT report is unsigned")
    unsigned = copy.deepcopy(report)
    unsigned.pop("report_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("independent AT report signature mismatch")
    if not report.get("solver_usable"):
        raise ValueError("independent AT solver did not produce a usable solution")
    if not report.get("solver_converged"):
        raise ValueError("independent AT solver did not explicitly converge")
    return expected


def build_independent_at_training_manifest(
    dataset: dict[str, Any],
    split_manifest: dict[str, Any],
    at_report: dict[str, Any],
    candidate_model_dir: Path,
) -> dict[str, Any]:
    """Publish a converged all-image independent AT model for Trainer.

    Unlike fixed-Rig BA publication, this product-style path replaces every
    image pose, including validation views, because the upstream AT solved all
    342 physical-camera images together.  A pose-only AT must preserve the base
    camera calibration exactly; a converged intrinsic-refining AT may publish
    its shared physical-camera parameters.
    """
    base_sha = verify_dataset_manifest(dataset)
    split_sha = verify_split_manifest(split_manifest)
    report_sha = verify_independent_at_report(at_report)
    if split_manifest.get("dataset_manifest_sha256") != base_sha:
        raise ValueError("split manifest is bound to a different base dataset")

    candidate_model_dir = candidate_model_dir.resolve()
    if not candidate_model_dir.is_dir():
        raise NotADirectoryError(
            f"independent AT candidate model is not a directory: {candidate_model_dir}"
        )
    model_sha = directory_sha256(candidate_model_dir)
    if at_report.get("candidate_model_sha256") != model_sha:
        raise ValueError("independent AT report does not identify the candidate model")

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError(
            "independent AT training manifest requires the optional pycolmap package"
        ) from exc
    model = pycolmap.Reconstruction(candidate_model_dir)
    image_records = {str(item["image_id"]): item for item in dataset.get("images", [])}
    model_images = {_normalise_name(image.name): image for image in model.images.values()}
    stable_id_by_name = {
        _normalise_name(str(item["path"])): str(item["image_id"])
        for item in image_records.values()
    }
    if len(stable_id_by_name) != len(image_records):
        raise ValueError("dataset repeats a normalized image path")
    if set(model_images) != set(stable_id_by_name):
        missing = sorted(set(stable_id_by_name) - set(model_images))
        extra = sorted(set(model_images) - set(stable_id_by_name))
        raise ValueError(
            "independent AT candidate image set differs from the dataset: "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    if int(at_report.get("counts", {}).get("images", -1)) != len(image_records):
        raise ValueError("independent AT report image count differs from the dataset")
    if any(not image.has_pose for image in model_images.values()):
        raise ValueError("independent AT candidate contains an image without a pose")

    stable_camera_to_numeric: dict[str, set[int]] = {}
    for name, model_image in model_images.items():
        stable_image_id = stable_id_by_name[name]
        stable_camera_id = str(image_records[stable_image_id]["camera_id"])
        stable_camera_to_numeric.setdefault(stable_camera_id, set()).add(
            int(model_image.camera_id)
        )
    dataset_camera_ids = {str(item["camera_id"]) for item in dataset.get("cameras", [])}
    if set(stable_camera_to_numeric) != dataset_camera_ids or any(
        len(values) != 1 for values in stable_camera_to_numeric.values()
    ):
        raise ValueError(
            "independent AT candidate must map one shared camera to each manifest camera"
        )

    derived = copy.deepcopy(dataset)
    derived.pop("manifest_sha256", None)
    derived_images = {str(item["image_id"]): item for item in derived["images"]}
    for name, image in model_images.items():
        record = derived_images[stable_id_by_name[name]]
        record["c2w"] = _rigid_c2w(image, name)
        record["pose_convention"] = "c2w_opencv"
        record["pose_source"] = "accepted_independent_pos_prior_at"

    intrinsics_refined = bool(at_report.get("intrinsics_refined"))
    base_cameras = {str(item["camera_id"]): item for item in dataset["cameras"]}
    derived_cameras = {str(item["camera_id"]): item for item in derived["cameras"]}
    for stable_id, numeric_ids in stable_camera_to_numeric.items():
        camera = model.camera(next(iter(numeric_ids)))
        base = base_cameras[stable_id]
        record = derived_cameras[stable_id]
        if (
            int(camera.width) != int(base["width"])
            or int(camera.height) != int(base["height"])
            or str(camera.model_name) != str(base["distortion"]["camera_model"])
        ):
            raise ValueError(f"independent AT camera contract changed for {stable_id}")
        parameters = _camera_parameters(camera, stable_id)
        expected = {
            **{key: float(base["intrinsic"][key]) for key in ("fl_x", "fl_y", "cx", "cy")},
            **{
                key: float(base["distortion"]["params"][key])
                for key in ("k1", "k2", "k3", "k4")
            },
        }
        if not intrinsics_refined and any(
            not np.isclose(parameters[key], expected[key], rtol=0.0, atol=1e-9)
            for key in expected
        ):
            raise ValueError("pose-only independent AT changed camera calibration")
        if intrinsics_refined:
            record["intrinsic"].update(
                {key: parameters[key] for key in ("fl_x", "fl_y", "cx", "cy")}
            )
            record["distortion"]["params"].update(
                {key: parameters[key] for key in ("k1", "k2", "k3", "k4")}
            )
            record["calibration_source"] = "accepted_independent_pos_prior_at"

    sigma = [float(value) for value in at_report.get("position_prior_sigma_xyz_m", [])]
    if len(sigma) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in sigma):
        raise ValueError("independent AT report has invalid position prior sigma")
    lineage = {
        "base_dataset_manifest_sha256": base_sha,
        "split_manifest_sha256": split_sha,
        "independent_at_report_sha256": report_sha,
        "independent_at_candidate_model_sha256": model_sha,
        "independent_at_algorithm_version": str(at_report.get("algorithm_version", "")),
        "position_prior_sigma_xyz_m": sigma,
        "pose_policy": "accepted_independent_at_all_images",
        "camera_policy": (
            "accepted_independent_at_shared_calibration"
            if intrinsics_refined
            else "base_calibration_pose_only"
        ),
    }
    derived["training_lineage"] = lineage
    source_hashes = dict(derived.get("source_hashes", {}))
    source_hashes.update(
        {
            "derived:base-dataset-manifest": base_sha,
            "derived:training-split-manifest": split_sha,
            "derived:independent-at-report": report_sha,
            "derived:independent-at-candidate-model": model_sha,
        }
    )
    derived["source_hashes"] = source_hashes
    warnings = list(derived.get("warnings", []))
    marker = "training_manifest_uses_converged_independent_at_for_all_images"
    if marker not in warnings:
        warnings.append(marker)
    derived["warnings"] = warnings
    derived["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(derived)
    ).hexdigest()
    verify_dataset_manifest(derived)
    return derived


def build_ba_training_manifest(
    dataset: dict[str, Any],
    split_manifest: dict[str, Any],
    ba_report: dict[str, Any],
    candidate_model_dir: Path,
) -> dict[str, Any]:
    """Create a signed dataset manifest that makes accepted BA visible to Trainer.

    Only training-image poses are replaced. Validation poses remain the original POS
    reference. Camera calibration is global, so accepted Stage 2 focal values apply
    to both splits and force rebuilding every derived mask/depth manifest.
    """
    base_sha = verify_dataset_manifest(dataset)
    split_sha = verify_split_manifest(split_manifest)
    report_sha = verify_ba_report(ba_report)
    if split_manifest.get("dataset_manifest_sha256") != base_sha:
        raise ValueError("split manifest is bound to a different base dataset")
    if not ba_report.get("candidate_accepted") or ba_report.get("published_model") != "after":
        raise ValueError("BA report did not accept and publish the candidate model")
    if str(ba_report.get("stage")) not in {"stage_1", "stage_2", "stage_3"}:
        raise ValueError("BA report has an unsupported stage")

    candidate_model_dir = candidate_model_dir.resolve()
    if not candidate_model_dir.is_dir():
        raise NotADirectoryError(
            f"BA candidate model is not a directory: {candidate_model_dir}"
        )
    model_sha = directory_sha256(candidate_model_dir)
    if ba_report.get("after_model_sha256") != model_sha:
        raise ValueError("BA report does not identify the candidate model directory")

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("BA training manifest requires the optional pycolmap package") from exc
    model = pycolmap.Reconstruction(candidate_model_dir)
    if model.num_images() <= 0 or model.num_cameras() <= 0:
        raise ValueError("BA candidate model has no images or cameras")

    image_records = {str(item["image_id"]): item for item in dataset.get("images", [])}
    train_ids = {str(value) for value in split_manifest.get("splits", {}).get("train", [])}
    val_ids = {str(value) for value in split_manifest.get("splits", {}).get("val", [])}
    if not train_ids or train_ids & val_ids or train_ids | val_ids != set(image_records):
        raise ValueError("split manifest must partition every dataset image into train/val")
    train_names = {
        _normalise_name(str(image_records[image_id]["path"])): image_id
        for image_id in train_ids
    }
    if len(train_names) != len(train_ids):
        raise ValueError("training split repeats a normalized image path")
    model_images = {_normalise_name(image.name): image for image in model.images.values()}
    if set(model_images) != set(train_names):
        missing = sorted(set(train_names) - set(model_images))
        extra = sorted(set(model_images) - set(train_names))
        raise ValueError(
            "BA candidate image set differs from the training split: "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    if any(not image.has_pose for image in model_images.values()):
        raise ValueError("BA candidate contains an image without an optimized pose")

    stable_camera_to_numeric: dict[str, set[int]] = {}
    for name, model_image in model_images.items():
        stable_id = str(image_records[train_names[name]]["camera_id"])
        stable_camera_to_numeric.setdefault(stable_id, set()).add(int(model_image.camera_id))
    dataset_camera_ids = {str(item["camera_id"]) for item in dataset.get("cameras", [])}
    if set(stable_camera_to_numeric) != dataset_camera_ids or any(
        len(values) != 1 for values in stable_camera_to_numeric.values()
    ):
        raise ValueError("BA candidate must map exactly one camera to each manifest camera")

    derived = copy.deepcopy(dataset)
    derived.pop("manifest_sha256", None)
    derived_images = {str(item["image_id"]): item for item in derived["images"]}
    for name, image in model_images.items():
        record = derived_images[train_names[name]]
        record["c2w"] = _rigid_c2w(image, name)
        record["pose_convention"] = "c2w_opencv"
        record["pose_source"] = "accepted_fixed_rig_ba"

    derived_cameras = {str(item["camera_id"]): item for item in derived["cameras"]}
    for stable_id, numeric_ids in stable_camera_to_numeric.items():
        camera = model.camera(next(iter(numeric_ids)))
        record = derived_cameras[stable_id]
        if (
            int(camera.width) != int(record["width"])
            or int(camera.height) != int(record["height"])
            or str(camera.model_name) != str(record["distortion"]["camera_model"])
        ):
            raise ValueError(f"BA candidate camera contract changed for {stable_id}")
        parameters = _camera_parameters(camera, stable_id)
        record["intrinsic"].update(
            {key: parameters[key] for key in ("fl_x", "fl_y", "cx", "cy")}
        )
        if str(camera.model_name) == "OPENCV_FISHEYE":
            record["distortion"]["params"].update(
                {key: parameters[key] for key in ("k1", "k2", "k3", "k4")}
            )
        record["calibration_source"] = "accepted_fixed_rig_ba"

    lineage = {
        "base_dataset_manifest_sha256": base_sha,
        "split_manifest_sha256": split_sha,
        "ba_report_sha256": report_sha,
        "ba_candidate_model_sha256": model_sha,
        "ba_stage": str(ba_report["stage"]),
        "position_prior_stddev_m": float(
            ba_report.get("position_prior", {}).get("stddev_m", 0.0)
        ),
        "pose_policy": "accepted_ba_train_original_pos_validation",
        "camera_policy": "accepted_ba_global_calibration",
    }
    derived["training_lineage"] = lineage
    source_hashes = dict(derived.get("source_hashes", {}))
    source_hashes.update(
        {
            "derived:base-dataset-manifest": base_sha,
            "derived:training-split-manifest": split_sha,
            "derived:accepted-ba-report": report_sha,
            "derived:accepted-ba-candidate-model": model_sha,
        }
    )
    derived["source_hashes"] = source_hashes
    warnings = list(derived.get("warnings", []))
    marker = "training_manifest_uses_accepted_ba_for_train_and_original_pos_for_validation"
    if marker not in warnings:
        warnings.append(marker)
    derived["warnings"] = warnings
    derived["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(derived)
    ).hexdigest()
    verify_dataset_manifest(derived)
    return derived
