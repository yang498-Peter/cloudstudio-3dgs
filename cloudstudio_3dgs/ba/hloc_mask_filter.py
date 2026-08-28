"""Filter HLoc local features through signed geometric and person masks."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.mask_manifest import (
    verify_dataset_manifest,
    verify_mask_manifest,
)
from cloudstudio_3dgs.data.person_masks import verify_person_mask_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if "\\" in value or pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe mask artifact path: {value!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"mask artifact escapes root: {value!r}")
    return resolved


def _read_mask(path: Path, expected_sha256: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing mask artifact: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"mask artifact SHA256 mismatch: {path}")
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L"), dtype=np.uint8) > 0


def filter_hloc_features_by_masks(
    source_features: Path,
    output_features: Path,
    *,
    dataset_manifest: dict[str, Any],
    mask_manifest: dict[str, Any],
    mask_root: Path,
    person_mask_manifest: dict[str, Any],
    person_mask_root: Path,
) -> dict[str, Any]:
    """Copy an HLoc feature H5 while removing invalid/dynamic keypoints."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("masked HLoc filtering requires h5py") from exc

    dataset_sha = verify_dataset_manifest(dataset_manifest)
    mask_sha = verify_mask_manifest(mask_manifest)
    person_sha = verify_person_mask_manifest(person_mask_manifest)
    if mask_manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("geometric mask manifest is bound to another dataset")
    if person_mask_manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("person mask manifest is bound to another dataset")
    if person_mask_manifest.get("base_mask_manifest_sha256") != mask_sha:
        raise ValueError("person mask manifest is not bound to the geometric masks")

    source_features = Path(source_features)
    output_features = Path(output_features)
    if not source_features.is_file():
        raise FileNotFoundError(f"source HLoc features do not exist: {source_features}")
    if output_features.exists():
        raise FileExistsError(f"filtered HLoc features already exist: {output_features}")
    output_features.parent.mkdir(parents=True, exist_ok=True)

    images = {str(item["image_id"]): item for item in dataset_manifest["images"]}
    masks = {str(item["image_id"]): item for item in mask_manifest["images"]}
    people = {
        str(item["image_id"]): item for item in person_mask_manifest["images"]
    }
    if set(images) != set(masks) or set(images) != set(people):
        raise ValueError("dataset, geometric masks, and person masks must cover identical images")

    records: list[dict[str, Any]] = []
    try:
        with h5py.File(source_features, "r") as source, h5py.File(
            output_features, "x"
        ) as target:
            for image_id, image in sorted(images.items(), key=lambda item: str(item[1]["path"])):
                feature_name = str(image["path"]).replace("\\", "/").removeprefix("camera/")
                if feature_name not in source:
                    raise ValueError(f"source HLoc features are missing {feature_name}")
                source_group = source[feature_name]
                if "keypoints" not in source_group:
                    raise ValueError(f"HLoc feature group has no keypoints: {feature_name}")
                keypoints = np.asarray(source_group["keypoints"], dtype=np.float64)
                if keypoints.ndim != 2 or keypoints.shape[1] != 2:
                    raise ValueError(f"invalid HLoc keypoint shape for {feature_name}")
                valid = _read_mask(
                    _safe_artifact(mask_root, str(masks[image_id]["combined_mask_path"])),
                    str(masks[image_id]["combined_mask_sha256"]),
                )
                person = _read_mask(
                    _safe_artifact(
                        person_mask_root, str(people[image_id]["person_mask_path"])
                    ),
                    str(people[image_id]["person_mask_sha256"]),
                )
                if valid.shape != person.shape:
                    raise ValueError(f"mask shape mismatch for {feature_name}")
                x = np.floor(keypoints[:, 0]).astype(np.int64)
                y = np.floor(keypoints[:, 1]).astype(np.int64)
                in_bounds = (
                    (x >= 0)
                    & (x < valid.shape[1])
                    & (y >= 0)
                    & (y < valid.shape[0])
                )
                keep = np.zeros(len(keypoints), dtype=bool)
                keep[in_bounds] = valid[y[in_bounds], x[in_bounds]] & ~person[
                    y[in_bounds], x[in_bounds]
                ]

                target_group = target.require_group(feature_name)
                for key, source_dataset in source_group.items():
                    data = np.asarray(source_dataset)
                    if key == "descriptors" and data.ndim >= 2 and data.shape[-1] == len(keep):
                        data = data[..., keep]
                    elif key != "image_size" and data.ndim >= 1 and data.shape[0] == len(keep):
                        data = data[keep]
                    target_group.create_dataset(key, data=data)
                for key, value in source_group.attrs.items():
                    target_group.attrs[key] = value
                records.append(
                    {
                        "image_id": image_id,
                        "feature_name": feature_name,
                        "keypoints_before": int(len(keep)),
                        "keypoints_after": int(np.count_nonzero(keep)),
                        "removed": int(np.count_nonzero(~keep)),
                    }
                )
    except BaseException:
        output_features.unlink(missing_ok=True)
        raise

    before = sum(int(item["keypoints_before"]) for item in records)
    after = sum(int(item["keypoints_after"]) for item in records)
    return {
        "algorithm_version": "hloc_feature_circle_and_person_filter_v1",
        "dataset_manifest_sha256": dataset_sha,
        "mask_manifest_sha256": mask_sha,
        "person_mask_manifest_sha256": person_sha,
        "source_features_sha256": _sha256_file(source_features),
        "filtered_features_sha256": _sha256_file(output_features),
        "coordinate_rule": "floor_hloc_keypoint_xy",
        "composition": "circle_valid & ~person_dynamic",
        "images": records,
        "summary": {
            "images": len(records),
            "keypoints_before": before,
            "keypoints_after": after,
            "removed": before - after,
            "removed_fraction": float((before - after) / before),
        },
    }
