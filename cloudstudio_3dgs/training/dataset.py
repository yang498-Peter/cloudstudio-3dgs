"""Manifest-bound raw-fisheye dataset used by the CloudStudio trainer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from cloudstudio_3dgs.data.depth_cache import verify_depth_manifest
from cloudstudio_3dgs.data.image_sample import CropWindow, load_image_sample
from cloudstudio_3dgs.data.mask_manifest import (
    verify_dataset_manifest,
    verify_mask_manifest,
)
from cloudstudio_3dgs.evaluation.splits import verify_split_manifest


SplitName = Literal["train", "val"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_artifact(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"artifact paths must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"artifact path escapes its root: {value!r}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingSample:
    image_id: str
    rig_frame_id: str
    camera_id: str
    image: np.ndarray
    rgb_mask: np.ndarray
    depth_range_m: np.ndarray | None
    depth_confidence: np.ndarray | None
    depth_mask: np.ndarray | None
    depth_cache_path: Path | None
    c2w: np.ndarray
    K: np.ndarray
    radial_coeffs: np.ndarray
    width: int
    height: int


class S1TrainingDataset:
    """Load one explicit split without importing gsplat example datasets.

    RGB supervision uses the complete per-image mask. Sparse LiDAR validity is
    tracked separately so enabling range loss never reduces RGB supervision to
    only the pixels that happen to contain LiDAR samples.
    """

    def __init__(
        self,
        *,
        dataset_manifest_path: Path,
        recording_root: Path,
        mask_manifest_path: Path,
        mask_root: Path,
        split_manifest_path: Path,
        split: SplitName,
        depth_manifest_path: Path | None = None,
        depth_root: Path | None = None,
        factor: int = 1,
        crop: CropWindow | None = None,
        verify_artifacts: bool = True,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if (depth_manifest_path is None) != (depth_root is None):
            raise ValueError("depth_manifest_path and depth_root must be provided together")

        self.dataset_manifest = _read_json(dataset_manifest_path)
        self.mask_manifest = _read_json(mask_manifest_path)
        self.split_manifest = _read_json(split_manifest_path)
        self.dataset_sha256 = verify_dataset_manifest(self.dataset_manifest)
        self.mask_sha256 = verify_mask_manifest(self.mask_manifest)
        self.split_sha256 = verify_split_manifest(self.split_manifest)
        if self.dataset_manifest.get("coordinate_frame") != "s1_local":
            raise ValueError("training dataset must use the s1_local coordinate frame")
        if self.mask_manifest.get("dataset_manifest_sha256") != self.dataset_sha256:
            raise ValueError("mask manifest is bound to a different dataset")
        if self.split_manifest.get("dataset_manifest_sha256") != self.dataset_sha256:
            raise ValueError("split manifest is bound to a different dataset")

        self.depth_manifest = None
        self.depth_sha256 = None
        if depth_manifest_path is not None:
            self.depth_manifest = _read_json(depth_manifest_path)
            self.depth_sha256 = verify_depth_manifest(self.depth_manifest)
            if self.depth_manifest.get("dataset_manifest_sha256") != self.dataset_sha256:
                raise ValueError("depth manifest is bound to a different dataset")
            if self.depth_manifest.get("mask_manifest_sha256") != self.mask_sha256:
                raise ValueError("depth manifest is bound to a different mask manifest")
            if self.depth_manifest.get("coordinate_frame") != "s1_local":
                raise ValueError("depth cache must use the s1_local coordinate frame")
            if self.depth_manifest.get("depth_semantics") != "euclidean_ray_range_m":
                raise ValueError("depth cache must contain Euclidean ray ranges")

        self.recording_root = Path(recording_root)
        self.mask_root = Path(mask_root)
        self.depth_root = None if depth_root is None else Path(depth_root)
        self.factor = factor
        self.crop = crop
        self.verify_artifacts = verify_artifacts
        self._verified_paths: set[Path] = set()

        cameras = {str(item["camera_id"]): item for item in self.dataset_manifest["cameras"]}
        if len(cameras) != len(self.dataset_manifest["cameras"]):
            raise ValueError("dataset manifest contains duplicate camera IDs")
        for camera in cameras.values():
            if camera.get("camera_type") != "fisheye":
                raise ValueError("PR-11 trainer only accepts raw fisheye cameras")
            if camera.get("distortion", {}).get("camera_model") != "OPENCV_FISHEYE":
                raise ValueError("raw fisheye cameras must use OPENCV_FISHEYE")
        self._cameras = cameras

        images = {str(item["image_id"]): item for item in self.dataset_manifest["images"]}
        masks = {str(item["image_id"]): item for item in self.mask_manifest["images"]}
        depths = (
            {}
            if self.depth_manifest is None
            else {str(item["image_id"]): item for item in self.depth_manifest["images"]}
        )
        if set(masks) != set(images):
            raise ValueError("mask manifest must cover every dataset image")
        if self.depth_manifest is not None and set(depths) != set(images):
            missing = sorted(set(images) - set(depths))
            extra = sorted(set(depths) - set(images))
            detail = []
            if missing:
                detail.append(f"missing={missing[:4]}")
            if extra:
                detail.append(f"unknown={extra[:4]}")
            raise ValueError(
                "depth manifest must cover every dataset image; " + ", ".join(detail)
            )
        for image_id, depth in depths.items():
            if str(depth.get("camera_id", "")) != str(images[image_id]["camera_id"]):
                raise ValueError(f"depth record camera mismatch for {image_id}")
            if str(depth.get("combined_mask_sha256", "")) != str(
                masks[image_id]["combined_mask_sha256"]
            ):
                raise ValueError(f"depth record mask identity mismatch for {image_id}")
        selected_ids = [str(value) for value in self.split_manifest["splits"][split]]
        if not selected_ids or len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"{split} split contains invalid image IDs")
        unknown = set(selected_ids) - set(images)
        if unknown:
            raise ValueError(f"{split} split references unknown images: {sorted(unknown)[:4]}")
        self._records = [
            (images[image_id], masks[image_id], depths.get(image_id))
            for image_id in selected_ids
        ]
        self.split = split

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "dataset_manifest_sha256": self.dataset_sha256,
            "mask_manifest_sha256": self.mask_sha256,
            "split_manifest_sha256": self.split_sha256,
            "depth_manifest_sha256": self.depth_sha256,
            "split": self.split,
            "factor": self.factor,
            "crop": None
            if self.crop is None
            else {
                "x": self.crop.x,
                "y": self.crop.y,
                "width": self.crop.width,
                "height": self.crop.height,
            },
        }

    def __len__(self) -> int:
        return len(self._records)

    def _verify(self, path: Path, expected: str, label: str) -> None:
        if not self.verify_artifacts or path in self._verified_paths:
            return
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} SHA256 mismatch: expected {expected}, computed {actual}")
        self._verified_paths.add(path)

    def __getitem__(self, index: int) -> TrainingSample:
        image_record, mask_record, depth_record = self._records[index]
        if image_record.get("path_root") != "recording":
            raise ValueError(f"unsupported image path_root for {image_record['image_id']}")
        image_path = _safe_artifact(self.recording_root, str(image_record["path"]))
        mask_path = _safe_artifact(self.mask_root, str(mask_record["combined_mask_path"]))
        self._verify(image_path, str(image_record["sha256"]), "source image")
        self._verify(mask_path, str(mask_record["combined_mask_sha256"]), "combined mask")

        depth_path = None
        if depth_record is not None:
            assert self.depth_root is not None
            depth_path = _safe_artifact(self.depth_root, str(depth_record["path"]))
            self._verify(depth_path, str(depth_record["sha256"]), "depth cache")

        sample = load_image_sample(
            image_path,
            mask_path,
            depth_path=depth_path,
            confidence_path=depth_path,
            factor=self.factor,
            crop=self.crop,
            depth_key="range_m",
            confidence_key="confidence",
        )
        rgb_mask = sample.valid_mask & sample.static_mask
        depth_mask = (
            None
            if sample.depth_valid_mask is None
            else rgb_mask & sample.depth_valid_mask
        )
        if not np.any(rgb_mask):
            raise ValueError(f"image {image_record['image_id']} has an empty RGB mask")

        camera = self._cameras[str(image_record["camera_id"])]
        crop = sample.crop
        intrinsic = camera["intrinsic"]
        K = np.asarray(
            [
                [float(intrinsic["fl_x"]) / self.factor, 0.0, (float(intrinsic["cx"]) - crop.x) / self.factor],
                [0.0, float(intrinsic["fl_y"]) / self.factor, (float(intrinsic["cy"]) - crop.y) / self.factor],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        params = camera["distortion"]["params"]
        radial = np.asarray([params[f"k{i}"] for i in range(1, 5)], dtype=np.float32)
        c2w = np.asarray(image_record["c2w"], dtype=np.float32)
        if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
            raise ValueError(f"image {image_record['image_id']} has an invalid c2w")
        height, width = sample.image.shape[:2]
        return TrainingSample(
            image_id=str(image_record["image_id"]),
            rig_frame_id=str(image_record["rig_frame_id"]),
            camera_id=str(image_record["camera_id"]),
            image=sample.image,
            rgb_mask=rgb_mask,
            depth_range_m=sample.depth,
            depth_confidence=sample.confidence,
            depth_mask=depth_mask,
            depth_cache_path=depth_path,
            c2w=c2w,
            K=K,
            radial_coeffs=radial,
            width=width,
            height=height,
        )
