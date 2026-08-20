from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np

from .schema import CameraRecord, ImageRecord


INTRINSIC_KEYS = ("fl_x", "fl_y", "cx", "cy")
DISTORTION_KEYS = ("k1", "k2", "k3", "k4")
POINT_CLOUD_NAMES = ("colorized.las", "colorized.laz", "uncolorized.las")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative_image_path(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe image path in ImgPose.txt: {raw_path!r}")
    if path.parts[0] not in {"left", "right"} or len(path.parts) != 2:
        raise ValueError(f"expected left/<image> or right/<image>, got {raw_path!r}")
    return path.as_posix()


def quaternion_xyzw_to_matrix(values: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(values), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quaternion must contain exactly four values")
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("zero-length quaternion in ImgPose.txt")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_cameras(calibration_path: Path) -> list[CameraRecord]:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    raw_cameras = payload.get("cameras")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise ValueError("calibration.json has no cameras")

    cameras: list[CameraRecord] = []
    seen: set[str] = set()
    for raw in raw_cameras:
        side = str(raw.get("name", ""))
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported camera name in calibration.json: {side!r}")
        if side in seen:
            raise ValueError(f"duplicate camera in calibration.json: {side}")
        seen.add(side)
        intrinsic = raw.get("intrinsic", {})
        distortion = raw.get("distortion", {})
        params = distortion.get("params", {})
        missing = [key for key in INTRINSIC_KEYS if key not in intrinsic]
        missing += [key for key in DISTORTION_KEYS if key not in params]
        if missing:
            raise ValueError(f"camera {side} calibration missing: {', '.join(missing)}")
        cameras.append(
            CameraRecord(
                camera_id=side,
                side=side,
                camera_type=str(raw.get("type", "fisheye")),
                width=int(raw["width"]),
                height=int(raw["height"]),
                intrinsic={key: float(intrinsic[key]) for key in INTRINSIC_KEYS},
                distortion={
                    "camera_model": str(distortion.get("camera_model", "")),
                    "params": {key: float(params[key]) for key in DISTORTION_KEYS},
                },
                transform_from_lidar=raw.get("transform_from_lidar", {}),
            )
        )
    if seen != {"left", "right"}:
        raise ValueError("calibration.json must contain both left and right cameras")
    return sorted(cameras, key=lambda camera: camera.side)


def load_imgpose_images(
    imgpose_path: Path,
    recording_dir: Path,
    *,
    hash_images: bool = True,
) -> list[ImageRecord]:
    lines = imgpose_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("ImgPose.txt is empty")
    expected_header = "index x y z roll pitch yaw qx qy qz qw timestamp"
    if " ".join(lines[0].split()) != expected_header:
        raise ValueError("ImgPose.txt header does not match the supported S1 contract")

    images: list[ImageRecord] = []
    seen_paths: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 12:
            raise ValueError(
                f"ImgPose.txt line {line_number}: expected 12 fields, got {len(fields)}"
            )
        relative = normalize_relative_image_path(fields[0])
        if relative in seen_paths:
            raise ValueError(f"ImgPose.txt line {line_number}: duplicate image {relative}")
        seen_paths.add(relative)
        side = PurePosixPath(relative).parts[0]
        source_path = recording_dir / "camera" / Path(relative)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"ImgPose.txt line {line_number}: missing camera image {relative}"
            )
        try:
            position = np.asarray([float(value) for value in fields[1:4]], dtype=np.float64)
            rotation = quaternion_xyzw_to_matrix(float(value) for value in fields[7:11])
        except ValueError as exc:
            raise ValueError(f"ImgPose.txt line {line_number}: {exc}") from exc
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = position
        stem = PurePosixPath(relative).stem
        if not stem.isdigit():
            raise ValueError(f"ImgPose.txt line {line_number}: image name is not a ns timestamp")
        timestamp_ns = int(stem)
        path_digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        images.append(
            ImageRecord(
                image_id=f"img_{path_digest[:24]}",
                rig_frame_id=None,
                side=side,
                timestamp_ns=timestamp_ns,
                path_root="recording",
                path=f"camera/{relative}",
                size_bytes=source_path.stat().st_size,
                sha256=sha256_file(source_path) if hash_images else "not_computed",
                camera_id=side,
                pose_source="ImgPose.txt",
                pose_convention="c2w_opencv",
                c2w=c2w.tolist(),
            )
        )
    if not images:
        raise ValueError("ImgPose.txt contains no image records")
    return sorted(images, key=lambda image: (image.timestamp_ns, image.side, image.path))


def find_point_cloud(run_dir: Path) -> Path:
    for name in POINT_CLOUD_NAMES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    prefixed = sorted(
        path for path in run_dir.glob("*_colorized.las") if "ecef" not in path.name.lower()
    )
    if not prefixed:
        prefixed = sorted(
            path
            for path in run_dir.glob("*_uncolorized.las")
            if "ecef" not in path.name.lower()
        )
    if not prefixed:
        raise FileNotFoundError(f"no local-coordinate LAS/LAZ point cloud found in {run_dir}")
    return prefixed[0]


def list_camera_images(recording_dir: Path) -> list[str]:
    images: list[str] = []
    for side in ("left", "right"):
        camera_dir = recording_dir / "camera" / side
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"camera directory is missing: {camera_dir}")
        for path in camera_dir.iterdir():
            if path.is_file():
                images.append(f"{side}/{path.name}")
    if not images:
        raise ValueError("recording camera directories contain no images")
    return sorted(images)
