"""``DatasetBundle``: the one shape every adapter produces and the SDK consumes.

A bundle is *evidence about a capture*, not a cache: images with poses,
intrinsics, an optional rig, an optional LiDAR cloud, an optional split and
optional capture timestamps.  Everything downstream (``caches.py``,
``tiling.py``, the trainer manifests) is derived from it.

The bundle is written to disk as a signed ``bundle_manifest.json`` with a
SHA256 per referenced file, using the same canonicalization as every other
signed manifest in this repository
(:func:`cloudstudio_3dgs.data.manifest.canonical_json_bytes`), so a bundle sha
can be compared with, and bound into, the trainer's own manifests.

Capabilities, not assumptions
-----------------------------
A generic dataset may lack LiDAR, timestamps, or a rig.  Rather than filling in
defaults, the bundle publishes a ``capabilities`` set; the cache planner refuses
the caches that need a capability the bundle does not have, naming it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from cloudstudio_3dgs.data.manifest import canonical_json_bytes

from .errors import BundleSignatureError, DatasetIncompleteError

BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_KIND = "sdk_dataset_bundle_v1"
BUNDLE_SHA_KEY = "bundle_manifest_sha256"

#: Capability tokens a bundle can advertise.  ``caches.py`` refuses by token.
CAPABILITY_LIDAR = "lidar_point_cloud"
CAPABILITY_TIMESTAMPS = "capture_timestamps"
CAPABILITY_RIG = "camera_rig"
CAPABILITY_SPLIT = "declared_split"
CAPABILITY_FISHEYE = "fisheye_source_images"
CAPABILITY_IMAGE_HASHES = "image_content_hashes"

_FISHEYE_MODELS = frozenset(
    {
        "OPENCV_FISHEYE",
        "FISHEYE",
        "KANNALA_BRANDT",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "THIN_PRISM_FISHEYE",
    }
)


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RigTransform:
    """A rigid transform, stored the way S1 calibration stores it."""

    rotation: tuple[tuple[float, float, float], ...]
    position: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rotation": [list(row) for row in self.rotation],
            "position": list(self.position),
        }

    @staticmethod
    def from_mapping(payload: Mapping[str, Any]) -> "RigTransform":
        rotation = tuple(tuple(float(v) for v in row) for row in payload["rotation"])
        position = tuple(float(v) for v in payload["position"])
        if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            raise DatasetIncompleteError("rig rotation must be a 3x3 matrix")
        if len(position) != 3:
            raise DatasetIncompleteError("rig position must be a 3-vector")
        return RigTransform(rotation=rotation, position=position)


@dataclass(frozen=True)
class CameraIntrinsics:
    """One physical camera.  ``transform_from_lidar`` is present only for rigs."""

    camera_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_model: str
    distortion: Mapping[str, float] = field(default_factory=dict)
    transform_from_lidar: RigTransform | None = None
    side: str | None = None

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise DatasetIncompleteError("camera_id must be a non-empty string")
        if self.width <= 0 or self.height <= 0:
            raise DatasetIncompleteError(
                f"camera {self.camera_id} has a non-positive image size"
            )
        if not (self.fx > 0.0 and self.fy > 0.0):
            raise DatasetIncompleteError(
                f"camera {self.camera_id} has a non-positive focal length"
            )

    @property
    def is_fisheye(self) -> bool:
        return self.camera_model.upper() in _FISHEYE_MODELS

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "camera_id": self.camera_id,
            "camera_type": "fisheye" if self.is_fisheye else "pinhole",
            "width": int(self.width),
            "height": int(self.height),
            "intrinsic": {
                "fl_x": float(self.fx),
                "fl_y": float(self.fy),
                "cx": float(self.cx),
                "cy": float(self.cy),
            },
            "distortion": {
                "camera_model": self.camera_model,
                "params": {str(k): float(v) for k, v in sorted(self.distortion.items())},
            },
        }
        if self.side is not None:
            payload["side"] = self.side
        if self.transform_from_lidar is not None:
            payload["transform_from_lidar"] = self.transform_from_lidar.to_dict()
        return payload


@dataclass(frozen=True)
class BundleImage:
    """One posed image.  ``c2w`` is a 4x4 camera-to-world matrix, OpenCV axes."""

    image_id: str
    camera_id: str
    path: str
    c2w: tuple[tuple[float, float, float, float], ...]
    timestamp_ns: int | None = None
    rig_frame_id: str | None = None
    split: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.image_id:
            raise DatasetIncompleteError("image_id must be a non-empty string")
        if "::" in self.image_id:
            # ``image_id::face_id`` is the sample key of every face cache.
            raise DatasetIncompleteError(
                f"image_id must not contain '::': {self.image_id}"
            )
        if len(self.c2w) != 4 or any(len(row) != 4 for row in self.c2w):
            raise DatasetIncompleteError(
                f"image {self.image_id} pose must be a 4x4 matrix"
            )
        if self.split not in (None, "train", "val", "golden"):
            raise DatasetIncompleteError(
                f"image {self.image_id} has an unsupported split: {self.split}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "camera_id": self.camera_id,
            "path": self.path,
            "c2w": [list(row) for row in self.c2w],
            "pose_convention": "c2w_opencv",
            "timestamp_ns": None if self.timestamp_ns is None else int(self.timestamp_ns),
            "rig_frame_id": self.rig_frame_id,
            "split": self.split,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class PointCloudRef:
    """The scene LiDAR cloud.  ``path`` is absolute; the manifest stores it as given."""

    path: Path
    format: str
    sha256: str | None = None
    size_bytes: int | None = None
    point_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "point_count": self.point_count,
        }


@dataclass(frozen=True)
class DatasetBundle:
    """Everything an adapter could establish about one capture."""

    dataset_id: str
    adapter: str
    source_root: Path
    images_root: Path
    cameras: tuple[CameraIntrinsics, ...]
    images: tuple[BundleImage, ...]
    point_cloud: PointCloudRef | None = None
    coordinate_frame: str = "unknown"
    pose_convention: str = "c2w_opencv"
    rig_frames: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    #: The adapter's native manifest payload, when it already produced one that
    #: the rest of the pipeline consumes verbatim (the S1 ``dataset_manifest``).
    native_dataset_manifest: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.cameras:
            raise DatasetIncompleteError(f"{self.adapter}: bundle has no cameras")
        if not self.images:
            raise DatasetIncompleteError(f"{self.adapter}: bundle has no posed images")
        known = {camera.camera_id for camera in self.cameras}
        unknown = sorted({image.camera_id for image in self.images} - known)
        if unknown:
            raise DatasetIncompleteError(
                f"{self.adapter}: images reference undeclared cameras: {', '.join(unknown)}"
            )
        seen: set[str] = set()
        for image in self.images:
            if image.image_id in seen:
                raise DatasetIncompleteError(
                    f"{self.adapter}: duplicate image_id {image.image_id}"
                )
            seen.add(image.image_id)

    # -- capabilities ----------------------------------------------------

    @property
    def capabilities(self) -> frozenset[str]:
        tokens: set[str] = set()
        if self.point_cloud is not None:
            tokens.add(CAPABILITY_LIDAR)
        if all(image.timestamp_ns is not None for image in self.images):
            tokens.add(CAPABILITY_TIMESTAMPS)
        if len(self.cameras) > 1 and all(
            camera.transform_from_lidar is not None for camera in self.cameras
        ):
            tokens.add(CAPABILITY_RIG)
        if any(image.split is not None for image in self.images):
            tokens.add(CAPABILITY_SPLIT)
        if any(camera.is_fisheye for camera in self.cameras):
            tokens.add(CAPABILITY_FISHEYE)
        if all(image.sha256 for image in self.images):
            tokens.add(CAPABILITY_IMAGE_HASHES)
        return frozenset(tokens)

    def camera(self, camera_id: str) -> CameraIntrinsics:
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        raise DatasetIncompleteError(f"unknown camera_id: {camera_id}")

    def image_path(self, image: BundleImage) -> Path:
        return self.images_root / image.path

    def counts(self) -> dict[str, int]:
        return {
            "cameras": len(self.cameras),
            "images": len(self.images),
            "rig_frames": len(self.rig_frames),
            "train_images": sum(1 for i in self.images if i.split == "train"),
            "val_images": sum(1 for i in self.images if i.split == "val"),
        }

    # -- manifest --------------------------------------------------------

    def to_manifest(self, *, hash_images: bool = True) -> dict[str, Any]:
        """Build the signed bundle manifest, hashing every referenced file."""

        images: list[dict[str, Any]] = []
        for image in self.images:
            row = image.to_dict()
            if row["sha256"] is None or row["size_bytes"] is None:
                absolute = self.image_path(image)
                if not absolute.is_file():
                    raise DatasetIncompleteError(
                        f"{self.adapter}: image file is missing: {absolute}"
                    )
                row["size_bytes"] = absolute.stat().st_size
                row["sha256"] = sha256_file(absolute) if hash_images else "not_computed"
            images.append(row)

        point_cloud: dict[str, Any] | None = None
        if self.point_cloud is not None:
            point_cloud = self.point_cloud.to_dict()
            if point_cloud["sha256"] is None:
                path = Path(point_cloud["path"])
                if not path.is_file():
                    raise DatasetIncompleteError(
                        f"{self.adapter}: point cloud is missing: {path}"
                    )
                point_cloud["sha256"] = sha256_file(path)
                point_cloud["size_bytes"] = path.stat().st_size

        warnings = list(self.warnings)
        if not hash_images:
            warnings.append("image_content_hashes_not_computed")

        payload: dict[str, Any] = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "dataset_id": self.dataset_id,
            "adapter": self.adapter,
            "source_root": str(self.source_root),
            "images_root": str(self.images_root),
            "coordinate_frame": self.coordinate_frame,
            "pose_convention": self.pose_convention,
            "capabilities": sorted(self.capabilities if hash_images else
                                   self.capabilities - {CAPABILITY_IMAGE_HASHES}),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "images": images,
            "point_cloud": point_cloud,
            "rig_frames": [dict(frame) for frame in self.rig_frames],
            "counts": self.counts(),
            "warnings": sorted(set(warnings)),
        }
        if self.native_dataset_manifest is not None:
            payload["native_dataset_manifest_sha256"] = str(
                self.native_dataset_manifest.get("manifest_sha256", "")
            )
        payload[BUNDLE_SHA_KEY] = hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest()
        return payload


def write_bundle_manifest(
    bundle: DatasetBundle,
    output_dir: Path,
    *,
    hash_images: bool = True,
    force: bool = False,
) -> Path:
    """Write ``bundle_manifest.json`` atomically; refuse to clobber silently."""

    output_dir = Path(output_dir)
    destination = output_dir / BUNDLE_MANIFEST_NAME
    if destination.exists() and not force:
        raise FileExistsError(
            f"refusing to replace an existing bundle manifest: {destination}"
        )
    payload = bundle.to_manifest(hash_images=hash_images)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def verify_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    verify_artifacts: bool = False,
) -> str:
    """Return the manifest sha, or raise :class:`BundleSignatureError`."""

    expected = str(manifest.get(BUNDLE_SHA_KEY, ""))
    if len(expected) != 64:
        raise BundleSignatureError("bundle manifest is unsigned")
    unsigned = {k: v for k, v in manifest.items() if k != BUNDLE_SHA_KEY}
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise BundleSignatureError(
            f"bundle manifest signature mismatch: expected {expected}, computed {actual}"
        )
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleSignatureError("unsupported bundle manifest schema_version")
    if manifest.get("kind") != BUNDLE_KIND:
        raise BundleSignatureError("manifest is not a dataset bundle manifest")
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise BundleSignatureError("bundle manifest lists no images")
    if manifest.get("counts", {}).get("images") != len(images):
        raise BundleSignatureError("bundle manifest image count is inconsistent")
    if verify_artifacts:
        images_root = Path(str(manifest["images_root"]))
        for row in images:
            absolute = images_root / str(row["path"])
            if not absolute.is_file():
                raise BundleSignatureError(f"bundle image is missing: {absolute}")
            if row.get("sha256") not in (None, "not_computed"):
                if sha256_file(absolute) != row["sha256"]:
                    raise BundleSignatureError(
                        f"bundle image content changed: {absolute}"
                    )
        point_cloud = manifest.get("point_cloud")
        if point_cloud:
            path = Path(str(point_cloud["path"]))
            if not path.is_file():
                raise BundleSignatureError(f"bundle point cloud is missing: {path}")
            if point_cloud.get("sha256") and sha256_file(path) != point_cloud["sha256"]:
                raise BundleSignatureError(f"bundle point cloud content changed: {path}")
    return expected


def load_bundle_manifest(path: Path, *, verify_artifacts: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_bundle_manifest(payload, verify_artifacts=verify_artifacts)
    return payload


def describe_capabilities(bundle: DatasetBundle) -> list[str]:
    """Human-readable lines for ``--dry-run`` output and error messages."""

    present = bundle.capabilities
    lines = []
    for token, explanation in (
        (CAPABILITY_LIDAR, "LiDAR cloud: depth, tiling and surface initialization"),
        (CAPABILITY_TIMESTAMPS, "timestamps: rig pairing and temporal split blocks"),
        (CAPABILITY_RIG, "camera rig: stereo rig frames and per-frame splits"),
        (CAPABILITY_SPLIT, "declared split: train/val already assigned"),
        (CAPABILITY_FISHEYE, "fisheye source: the Face4 pinhole cache applies"),
        (CAPABILITY_IMAGE_HASHES, "image hashes: cache bindings can be verified"),
    ):
        lines.append(f"[{'x' if token in present else ' '}] {token} - {explanation}")
    return lines


def require_capabilities(bundle: DatasetBundle, required: Iterable[str], *, what: str) -> None:
    missing = sorted(set(required) - bundle.capabilities)
    if missing:
        raise DatasetIncompleteError(
            f"{what} requires bundle capabilities this dataset does not have: "
            f"{', '.join(missing)}"
        )


__all__ = [
    "BUNDLE_KIND",
    "BUNDLE_MANIFEST_NAME",
    "BUNDLE_SCHEMA_VERSION",
    "BUNDLE_SHA_KEY",
    "BundleImage",
    "CAPABILITY_FISHEYE",
    "CAPABILITY_IMAGE_HASHES",
    "CAPABILITY_LIDAR",
    "CAPABILITY_RIG",
    "CAPABILITY_SPLIT",
    "CAPABILITY_TIMESTAMPS",
    "CameraIntrinsics",
    "DatasetBundle",
    "PointCloudRef",
    "RigTransform",
    "describe_capabilities",
    "load_bundle_manifest",
    "require_capabilities",
    "sha256_file",
    "verify_bundle_manifest",
    "write_bundle_manifest",
]
