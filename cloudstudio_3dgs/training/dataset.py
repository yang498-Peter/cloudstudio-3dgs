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
from cloudstudio_3dgs.data.person_masks import verify_person_mask_manifest
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
    # "fisheye" renders the raw KB4 image through 3DGUT; "pinhole" is a
    # zero-distortion face from the fisheye face-split pipeline.
    camera_model: str = "fisheye"
    # Pinhole faces render z-depth while LiDAR supervision is Euclidean ray
    # range; this per-pixel factor (||K^-1 [u,v,1]||) converts z to range.
    # None means the renderer already outputs ray range (the fisheye path).
    depth_to_range_scale: np.ndarray | None = None
    # Absolute pixel-center coordinates on the ORIGINAL fisheye sensor for
    # every pixel of this sample ([H, W, 2] x/y), plus that sensor's (w, h).
    # Faces need these for spatially varying corrections (PPISP vignetting):
    # the vignetting field lives on the sensor, not on the warped face.
    sensor_pixel_coords: np.ndarray | None = None
    sensor_resolution: tuple[int, int] | None = None
    # Metric-aligned DA2 supervision.  This stays separate from sparse LiDAR
    # range so schedules and evidence can distinguish the two sources.
    mono_depth_range_m: np.ndarray | None = None
    mono_depth_mask: np.ndarray | None = None
    mono_depth_cache_path: Path | None = None


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
        person_mask_manifest_path: Path | None = None,
        person_mask_root: Path | None = None,
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
        if (person_mask_manifest_path is None) != (person_mask_root is None):
            raise ValueError(
                "person_mask_manifest_path and person_mask_root must be provided together"
            )

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

        self.person_mask_manifest = None
        self.person_mask_sha256 = None
        if person_mask_manifest_path is not None:
            self.person_mask_manifest = _read_json(person_mask_manifest_path)
            self.person_mask_sha256 = verify_person_mask_manifest(
                self.person_mask_manifest
            )
            if (
                self.person_mask_manifest.get("dataset_manifest_sha256")
                != self.dataset_sha256
            ):
                raise ValueError("person mask manifest is bound to a different dataset")
            if (
                self.person_mask_manifest.get("base_mask_manifest_sha256")
                != self.mask_sha256
            ):
                raise ValueError(
                    "person mask manifest is bound to a different base mask manifest"
                )

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
        self.person_mask_root = (
            None if person_mask_root is None else Path(person_mask_root)
        )
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
        person_masks = (
            {}
            if self.person_mask_manifest is None
            else {
                str(item["image_id"]): item
                for item in self.person_mask_manifest["images"]
            }
        )
        depths = (
            {}
            if self.depth_manifest is None
            else {str(item["image_id"]): item for item in self.depth_manifest["images"]}
        )
        if set(masks) != set(images):
            raise ValueError("mask manifest must cover every dataset image")
        if self.person_mask_manifest is not None and set(person_masks) != set(images):
            missing = sorted(set(images) - set(person_masks))
            extra = sorted(set(person_masks) - set(images))
            detail = []
            if missing:
                detail.append(f"missing={missing[:4]}")
            if extra:
                detail.append(f"unknown={extra[:4]}")
            raise ValueError(
                "person mask manifest must cover every dataset image; "
                + ", ".join(detail)
            )
        for image_id, person in person_masks.items():
            if str(person.get("camera_id", "")) != str(
                images[image_id]["camera_id"]
            ):
                raise ValueError(f"person mask record camera mismatch for {image_id}")
            if str(person.get("source_image_sha256", "")) != str(
                images[image_id]["sha256"]
            ):
                raise ValueError(
                    f"person mask record source image mismatch for {image_id}"
                )
            if str(person.get("source_image_path_root", "")) != "recording" or str(
                person.get("source_image_path", "")
            ) != str(images[image_id]["path"]):
                raise ValueError(
                    f"person mask record source path mismatch for {image_id}"
                )
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
            (
                images[image_id],
                masks[image_id],
                person_masks.get(image_id),
                depths.get(image_id),
            )
            for image_id in selected_ids
        ]
        self.split = split

    @property
    def image_ids(self) -> list[str]:
        """Selected image IDs in split-manifest order."""
        return [str(record[0]["image_id"]) for record in self._records]

    @property
    def camera_id_by_image(self) -> dict[str, str]:
        """Physical camera for each selected image (exposure gains group by it)."""
        return {
            str(record[0]["image_id"]): str(record[0]["camera_id"])
            for record in self._records
        }

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "dataset_manifest_sha256": self.dataset_sha256,
            "mask_manifest_sha256": self.mask_sha256,
            "person_mask_manifest_sha256": self.person_mask_sha256,
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

    @property
    def rig_frame_ids(self) -> tuple[str, ...]:
        """Return selected Rig Frames once, preserving split-manifest order."""
        result: list[str] = []
        seen: set[str] = set()
        for image, _, _, _ in self._records:
            rig_frame_id = str(image.get("rig_frame_id") or "")
            if not rig_frame_id:
                raise ValueError(f"image {image['image_id']} has no Rig Frame")
            if rig_frame_id not in seen:
                result.append(rig_frame_id)
                seen.add(rig_frame_id)
        return tuple(result)

    def indices_for_rig_frames(self, maximum_rig_frames: int) -> tuple[int, ...]:
        if maximum_rig_frames <= 0:
            raise ValueError("maximum_rig_frames must be positive")
        rig_frame_ids = self.rig_frame_ids
        if len(rig_frame_ids) <= maximum_rig_frames:
            selected = set(rig_frame_ids)
        else:
            positions = np.linspace(
                0, len(rig_frame_ids) - 1, maximum_rig_frames, dtype=np.int64
            )
            selected = {rig_frame_ids[int(index)] for index in positions}
        return tuple(
            index
            for index, (image, _, _, _) in enumerate(self._records)
            if str(image.get("rig_frame_id") or "") in selected
        )

    def rig_frame_centers(self) -> dict[str, np.ndarray]:
        """Use the mean camera center as a stable rotation pivot for each Rig Frame."""
        grouped: dict[str, list[np.ndarray]] = {}
        for image, _, _, _ in self._records:
            rig_frame_id = str(image.get("rig_frame_id") or "")
            c2w = np.asarray(image.get("c2w"), dtype=np.float64)
            if not rig_frame_id or c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
                raise ValueError(f"image {image['image_id']} has no valid Rig pose")
            grouped.setdefault(rig_frame_id, []).append(c2w[:3, 3])
        if set(grouped) != set(self.rig_frame_ids):
            raise ValueError("training split Rig Frame centers are incomplete")
        return {
            rig_frame_id: np.mean(np.stack(centers), axis=0)
            for rig_frame_id, centers in grouped.items()
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
        image_record, mask_record, person_mask_record, depth_record = self._records[index]
        if image_record.get("path_root") != "recording":
            raise ValueError(f"unsupported image path_root for {image_record['image_id']}")
        image_path = _safe_artifact(self.recording_root, str(image_record["path"]))
        mask_path = _safe_artifact(self.mask_root, str(mask_record["combined_mask_path"]))
        self._verify(image_path, str(image_record["sha256"]), "source image")
        self._verify(mask_path, str(mask_record["combined_mask_sha256"]), "combined mask")

        person_mask_path = None
        if person_mask_record is not None:
            assert self.person_mask_root is not None
            person_mask_path = _safe_artifact(
                self.person_mask_root, str(person_mask_record["person_mask_path"])
            )
            self._verify(
                person_mask_path,
                str(person_mask_record["person_mask_sha256"]),
                "person dynamic mask",
            )

        depth_path = None
        if depth_record is not None:
            assert self.depth_root is not None
            depth_path = _safe_artifact(self.depth_root, str(depth_record["path"]))
            self._verify(depth_path, str(depth_record["sha256"]), "depth cache")

        sample = load_image_sample(
            image_path,
            mask_path,
            dynamic_mask_path=person_mask_path,
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
