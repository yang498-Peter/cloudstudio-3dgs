"""A plain folder of images + a poses JSON + an optional point cloud.

Two JSON dialects are accepted, distinguished by their own content:

``sdk`` (native, preferred)::

    {
      "coordinate_frame": "site_local",
      "pose_convention": "c2w_opencv",
      "cameras": [{"camera_id": "cam0", "width": W, "height": H,
                   "fx": .., "fy": .., "cx": .., "cy": ..,
                   "model": "PINHOLE", "distortion": {"k1": ..}}],
      "frames":  [{"file_path": "images/0001.jpg", "camera_id": "cam0",
                   "transform_matrix": [[..]x4], "timestamp_ns": 0,
                   "split": "train", "rig_frame_id": "f0"}],
      "point_cloud": "cloud.ply"
    }

``nerfstudio`` (``transforms.json``): top-level ``fl_x``/``fl_y``/``cx``/``cy``/
``w``/``h`` plus ``frames[].transform_matrix``.  Those matrices are
camera-to-world in **OpenGL** axes (x right, y up, z backward); they are
converted to OpenCV here by negating the y and z basis columns, and the
conversion is recorded in ``bundle.warnings`` rather than assumed silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..bundle import BundleImage, CameraIntrinsics, DatasetBundle, PointCloudRef
from ..errors import DatasetIncompleteError

NAME = "pinhole_folder"
VERSION = "1"
REQUIRES = "a poses json (poses.json / transforms.json / cameras.json) + the images it names"

POSE_FILE_NAMES = ("poses.json", "transforms.json", "cameras.json")
_CLOUD_SUFFIXES = (".ply", ".las", ".laz")
_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetIncompleteError(f"{NAME}: {message}")


def pose_file(path: Path) -> Path | None:
    path = Path(path)
    for name in POSE_FILE_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def detect(path: Path) -> bool:
    path = Path(path)
    # An S1 run directory also carries transforms.json; it is not this adapter's
    # dataset, and letting both claim it would make detection ambiguous.
    if (path / "info" / "calibration.json").is_file() and (path / "camera").is_dir():
        return False
    if (path / "sparse" / "0" / "cameras.bin").is_file():
        return False
    return pose_file(path) is not None


def _dialect(payload: Mapping[str, Any]) -> str:
    if "cameras" in payload and "frames" in payload:
        return "sdk"
    if "frames" in payload and ("fl_x" in payload or "camera_angle_x" in payload):
        return "nerfstudio"
    raise DatasetIncompleteError(
        f"{NAME}: poses json has neither 'cameras'+'frames' (sdk dialect) nor "
        "top-level 'fl_x'/'camera_angle_x' with 'frames' (nerfstudio dialect)"
    )


def _sdk_cameras(payload: Mapping[str, Any]) -> list[CameraIntrinsics]:
    cameras: list[CameraIntrinsics] = []
    for index, row in enumerate(payload["cameras"]):
        for key in ("camera_id", "width", "height", "fx", "fy", "cx", "cy"):
            if key not in row:
                raise DatasetIncompleteError(
                    f"{NAME}: cameras[{index}] is missing required key '{key}'"
                )
        cameras.append(
            CameraIntrinsics(
                camera_id=str(row["camera_id"]),
                width=int(row["width"]),
                height=int(row["height"]),
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                camera_model=str(row.get("model", "PINHOLE")),
                distortion={
                    str(k): float(v) for k, v in dict(row.get("distortion", {})).items()
                },
            )
        )
    _require(bool(cameras), "poses json declares no cameras")
    return cameras


def _nerfstudio_camera(payload: Mapping[str, Any]) -> CameraIntrinsics:
    width = payload.get("w")
    height = payload.get("h")
    _require(
        width is not None and height is not None,
        "nerfstudio transforms.json must carry top-level 'w' and 'h'",
    )
    if "fl_x" in payload:
        fx = float(payload["fl_x"])
        fy = float(payload.get("fl_y", fx))
    else:
        angle = float(payload["camera_angle_x"])
        fx = fy = float(width) / (2.0 * np.tan(angle / 2.0))
    return CameraIntrinsics(
        camera_id=str(payload.get("camera_id", "cam0")),
        width=int(width),
        height=int(height),
        fx=fx,
        fy=fy,
        cx=float(payload.get("cx", float(width) / 2.0)),
        cy=float(payload.get("cy", float(height) / 2.0)),
        camera_model=str(payload.get("camera_model", "PINHOLE")),
        distortion={
            key: float(payload[key])
            for key in ("k1", "k2", "k3", "k4", "p1", "p2")
            if key in payload
        },
    )


def _matrix(frame: Mapping[str, Any], index: int) -> np.ndarray:
    if "transform_matrix" not in frame:
        raise DatasetIncompleteError(
            f"{NAME}: frames[{index}] has no 'transform_matrix'"
        )
    matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise DatasetIncompleteError(
            f"{NAME}: frames[{index}] transform_matrix is {matrix.shape}, expected 4x4"
        )
    if not np.all(np.isfinite(matrix)):
        raise DatasetIncompleteError(
            f"{NAME}: frames[{index}] transform_matrix is not finite"
        )
    return matrix


def load(
    path: Path,
    *,
    poses: Path | None = None,
    images_dir: Path | None = None,
    point_cloud: Path | None = None,
    dataset_id: str | None = None,
) -> DatasetBundle:
    root = Path(path).resolve()
    poses_path = Path(poses) if poses is not None else pose_file(root)
    if poses_path is None:
        raise DatasetIncompleteError(
            f"{NAME}: no poses json in {root}; expected one of "
            f"{', '.join(POSE_FILE_NAMES)} ({REQUIRES})"
        )
    _require(poses_path.is_file(), f"poses json is missing: {poses_path}")
    try:
        payload = json.loads(poses_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetIncompleteError(
            f"{NAME}: poses json is not valid JSON ({poses_path}): {exc}"
        ) from exc
    dialect = _dialect(payload)
    frames = payload.get("frames") or []
    _require(bool(frames), f"poses json lists no frames: {poses_path}")

    warnings: list[str] = []
    if dialect == "sdk":
        cameras = _sdk_cameras(payload)
        convention = str(payload.get("pose_convention", "c2w_opencv"))
        if convention not in ("c2w_opencv", "c2w_opengl"):
            raise DatasetIncompleteError(
                f"{NAME}: unsupported pose_convention '{convention}'; "
                "expected c2w_opencv or c2w_opengl"
            )
        coordinate_frame = str(payload.get("coordinate_frame", "unknown"))
    else:
        cameras = [_nerfstudio_camera(payload)]
        convention = "c2w_opengl"
        coordinate_frame = str(payload.get("coordinate_frame", "nerfstudio_world"))
    if convention == "c2w_opengl":
        warnings.append("poses_converted_from_opengl_c2w_to_opencv_c2w")

    images_root = Path(images_dir) if images_dir is not None else poses_path.parent
    _require(images_root.is_dir(), f"images directory is missing: {images_root}")

    known = {camera.camera_id for camera in cameras}
    images: list[BundleImage] = []
    missing: list[str] = []
    seen_ids: set[str] = set()
    for index, frame in enumerate(frames):
        if "file_path" not in frame:
            raise DatasetIncompleteError(f"{NAME}: frames[{index}] has no 'file_path'")
        relative = str(frame["file_path"]).replace("\\", "/")
        camera_id = str(frame.get("camera_id", cameras[0].camera_id))
        if camera_id not in known:
            raise DatasetIncompleteError(
                f"{NAME}: frames[{index}] references unknown camera_id '{camera_id}'"
            )
        absolute = images_root / relative
        if not absolute.is_file():
            missing.append(relative)
            continue
        matrix = _matrix(frame, index)
        if convention == "c2w_opengl":
            matrix = matrix @ _OPENGL_TO_OPENCV
        image_id = str(frame.get("image_id", Path(relative).stem))
        if image_id in seen_ids:
            image_id = f"{image_id}_{index}"
        seen_ids.add(image_id)
        images.append(
            BundleImage(
                image_id=image_id,
                camera_id=camera_id,
                path=relative,
                c2w=tuple(tuple(float(v) for v in row) for row in matrix),
                timestamp_ns=(
                    None if frame.get("timestamp_ns") is None else int(frame["timestamp_ns"])
                ),
                rig_frame_id=frame.get("rig_frame_id"),
                split=frame.get("split"),
            )
        )
    if missing:
        head = ", ".join(missing[:5])
        raise DatasetIncompleteError(
            f"{NAME}: {len(missing)} frames name images that do not exist under "
            f"{images_root} (first: {head}); pass images_dir= if they live elsewhere"
        )

    cloud_ref = None
    declared = payload.get("point_cloud")
    cloud_path = (
        Path(point_cloud)
        if point_cloud is not None
        else (poses_path.parent / str(declared) if declared else None)
    )
    if cloud_path is not None:
        _require(cloud_path.is_file(), f"point cloud is missing: {cloud_path}")
        cloud_ref = PointCloudRef(
            path=cloud_path.resolve(), format=cloud_path.suffix.lower().lstrip(".")
        )
    else:
        found = sorted(
            child
            for child in root.iterdir()
            if child.is_file() and child.suffix.lower() in _CLOUD_SUFFIXES
        )
        if found:
            cloud_ref = PointCloudRef(
                path=found[0].resolve(), format=found[0].suffix.lower().lstrip(".")
            )
        else:
            warnings.append("no_point_cloud_declared_or_found")
    if not any(image.timestamp_ns is not None for image in images):
        warnings.append("no_capture_timestamps_in_poses_json")

    return DatasetBundle(
        dataset_id=dataset_id or root.name,
        adapter=NAME,
        source_root=root,
        images_root=images_root,
        cameras=tuple(cameras),
        images=tuple(images),
        point_cloud=cloud_ref,
        coordinate_frame=coordinate_frame,
        pose_convention="c2w_opencv",
        warnings=tuple(warnings),
    )


__all__ = ["NAME", "POSE_FILE_NAMES", "REQUIRES", "VERSION", "detect", "load", "pose_file"]
