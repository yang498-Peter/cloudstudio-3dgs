from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 1
COORDINATE_FRAME = "s1_local"


@dataclass(frozen=True)
class CameraRecord:
    camera_id: str
    side: str
    camera_type: str
    width: int
    height: int
    intrinsic: dict[str, float]
    distortion: dict[str, Any]
    transform_from_lidar: dict[str, Any]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    rig_frame_id: str | None
    side: str
    timestamp_ns: int
    path_root: str
    path: str
    size_bytes: int
    sha256: str
    camera_id: str
    pose_source: str
    pose_convention: str
    c2w: list[list[float]]
    split: str | None = None
    mask_path: str | None = None
    depth_path: str | None = None


@dataclass(frozen=True)
class RigFrameRecord:
    rig_frame_id: str
    timestamp_ns: int
    image_ids: list[str]


@dataclass
class DatasetManifest:
    recording_id: str
    source_hashes: dict[str, str]
    cameras: list[CameraRecord]
    images: list[ImageRecord]
    point_cloud: dict[str, Any]
    unposed_images: list[str] = field(default_factory=list)
    rig_frames: list[RigFrameRecord] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    coordinate_frame: str = COORDINATE_FRAME
    path_roots: dict[str, str] = field(
        default_factory=lambda: {"recording": "recording_root", "run": "run_root"}
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
