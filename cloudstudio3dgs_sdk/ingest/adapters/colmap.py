"""COLMAP sparse reconstruction (binary model) + an optional LiDAR cloud.

Layout::

    <root>/sparse/0/cameras.bin
    <root>/sparse/0/images.bin
    <root>/sparse/0/points3D.bin
    <root>/images/<name>              (or --images-dir)
    <root>/<anything>.ply|.las        optional LiDAR cloud, or pass point_cloud=

COLMAP stores world-to-camera (``qvec`` as w,x,y,z plus ``tvec``); the bundle
stores camera-to-world with OpenCV axes, which is the same axis convention, so
the mapping is a transpose and a sign, not a basis change.

The sparse ``points3D`` are SfM points, not LiDAR.  They are counted and
reported but they are *not* promoted to ``bundle.point_cloud``: the caches that
need metric LiDAR would then be planned against triangulated points and fail
much later.  :func:`export_sparse_point_cloud` makes that promotion explicit
when a caller really wants it.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ..bundle import BundleImage, CameraIntrinsics, DatasetBundle, PointCloudRef
from ..errors import DatasetIncompleteError

NAME = "colmap"
VERSION = "1"
REQUIRES = "sparse/0/{cameras,images,points3D}.bin + an images directory"

#: COLMAP camera model id -> (name, parameter count).
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

#: Distortion parameter names per model, after the four intrinsic entries.
_DISTORTION_NAMES: dict[str, tuple[str, ...]] = {
    "SIMPLE_PINHOLE": (),
    "PINHOLE": (),
    "SIMPLE_RADIAL": ("k1",),
    "RADIAL": ("k1", "k2"),
    "OPENCV": ("k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("k1", "k2", "k3", "k4"),
    "FULL_OPENCV": ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
    "FOV": ("omega",),
    "SIMPLE_RADIAL_FISHEYE": ("k1",),
    "RADIAL_FISHEYE": ("k1", "k2"),
    "THIN_PRISM_FISHEYE": ("k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"),
}

_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
_CLOUD_SUFFIXES = (".ply", ".las", ".laz")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetIncompleteError(f"{NAME}: {message}")


def model_dir(path: Path) -> Path | None:
    """Return the directory holding the binary model, or ``None``."""

    path = Path(path)
    for candidate in (path / "sparse" / "0", path / "sparse", path / "colmap" / "sparse" / "0", path):
        if all((candidate / name).is_file() for name in _MODEL_FILES):
            return candidate
    return None


def detect(path: Path) -> bool:
    return model_dir(path) is not None


def _read(handle: BinaryIO, fmt: str) -> tuple[Any, ...]:
    size = struct.calcsize(fmt)
    payload = handle.read(size)
    if len(payload) != size:
        raise DatasetIncompleteError(f"{NAME}: truncated binary model (wanted {size} bytes)")
    return struct.unpack(fmt, payload)


def read_cameras_binary(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    with Path(path).open("rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read(handle, "<iiQQ")
            if model_id not in CAMERA_MODELS:
                raise DatasetIncompleteError(
                    f"{NAME}: unsupported COLMAP camera model id {model_id} on camera {camera_id}"
                )
            model_name, parameter_count = CAMERA_MODELS[model_id]
            params = _read(handle, "<" + "d" * parameter_count)
            cameras[int(camera_id)] = {
                "camera_id": int(camera_id),
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": [float(value) for value in params],
            }
    return cameras


def read_images_binary(path: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            values = _read(handle, "<idddddddi")
            image_id = int(values[0])
            qvec = np.array(values[1:5], dtype=np.float64)  # w, x, y, z
            tvec = np.array(values[5:8], dtype=np.float64)
            camera_id = int(values[8])
            name_bytes = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise DatasetIncompleteError(f"{NAME}: truncated image name in {path}")
                if char == b"\x00":
                    break
                name_bytes += char
            (point_count,) = _read(handle, "<Q")
            handle.seek(point_count * 24, 1)
            images.append(
                {
                    "image_id": image_id,
                    "qvec_wxyz": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name_bytes.decode("utf-8"),
                    "point2D_count": int(point_count),
                }
            )
    return images


def read_points3d_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz[N,3] float64, rgb[N,3] uint8)`` from ``points3D.bin``."""

    xyz: list[list[float]] = []
    rgb: list[list[int]] = []
    with Path(path).open("rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            values = _read(handle, "<QdddBBBd")
            xyz.append([float(values[1]), float(values[2]), float(values[3])])
            rgb.append([int(values[4]), int(values[5]), int(values[6])])
            (track_length,) = _read(handle, "<Q")
            handle.seek(track_length * 8, 1)
    if not xyz:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def quaternion_wxyz_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in q)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm <= 0.0:
        raise DatasetIncompleteError(f"{NAME}: image carries a zero-length quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _intrinsics(record: dict[str, Any]) -> CameraIntrinsics:
    model = record["model"]
    params = record["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        distortion_values = params[3:]
    else:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        distortion_values = params[4:]
    names = _DISTORTION_NAMES.get(model, ())
    distortion = {
        name: float(value) for name, value in zip(names, distortion_values)
    }
    return CameraIntrinsics(
        camera_id=str(record["camera_id"]),
        width=int(record["width"]),
        height=int(record["height"]),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        camera_model=model,
        distortion=distortion,
    )


def _images_dir(root: Path, override: Path | None) -> Path:
    if override is not None:
        folder = Path(override)
        _require(folder.is_dir(), f"images directory is missing: {folder}")
        return folder
    for candidate in (root / "images", root / "image", root):
        if candidate.is_dir() and any(
            child.suffix.lower() in _IMAGE_SUFFIXES
            for child in candidate.iterdir()
            if child.is_file()
        ):
            return candidate
    raise DatasetIncompleteError(
        f"{NAME}: no images directory under {root}; pass images_dir=..."
    )


def _find_cloud(root: Path, override: Path | None) -> Path | None:
    if override is not None:
        cloud = Path(override)
        _require(cloud.is_file(), f"point cloud is missing: {cloud}")
        return cloud
    candidates = sorted(
        child
        for child in root.iterdir()
        if child.is_file() and child.suffix.lower() in _CLOUD_SUFFIXES
    )
    return candidates[0] if candidates else None


def export_sparse_point_cloud(path: Path, destination: Path) -> Path:
    """Write ``points3D.bin`` out as a binary PLY.

    This is the explicit "I have no LiDAR, use the SfM points" step.  It is a
    separate call, never a side effect of :func:`load`, because everything the
    plan says about scale, tiling and depth supervision changes when the cloud
    is triangulated rather than measured.
    """

    folder = model_dir(path)
    _require(folder is not None, f"{path} holds no COLMAP binary model")
    assert folder is not None
    xyz, rgb = read_points3d_binary(folder / "points3D.bin")
    _require(len(xyz) > 0, f"points3D.bin in {folder} is empty")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    records = np.zeros(
        len(xyz),
        dtype=np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        ),
    )
    records["x"], records["y"], records["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    records["red"], records["green"], records["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with destination.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)
    return destination


def load(
    path: Path,
    *,
    images_dir: Path | None = None,
    point_cloud: Path | None = None,
    coordinate_frame: str = "colmap_world",
    dataset_id: str | None = None,
) -> DatasetBundle:
    root = Path(path).resolve()
    folder = model_dir(root)
    if folder is None:
        text_model = any((root / "sparse" / "0" / name).is_file() for name in
                         ("cameras.txt", "images.txt", "points3D.txt"))
        if text_model:
            raise DatasetIncompleteError(
                f"{NAME}: {root} holds a COLMAP text model; convert it with "
                "`colmap model_converter --output_type BIN` and retry"
            )
        raise DatasetIncompleteError(
            f"{NAME}: {root} holds no COLMAP binary model ({REQUIRES})"
        )
    for name in _MODEL_FILES:
        _require((folder / name).is_file(), f"binary model file is missing: {folder / name}")

    camera_records = read_cameras_binary(folder / "cameras.bin")
    _require(bool(camera_records), f"cameras.bin in {folder} declares no cameras")
    image_records = read_images_binary(folder / "images.bin")
    _require(bool(image_records), f"images.bin in {folder} declares no registered images")

    images_root = _images_dir(root, images_dir)
    cameras = tuple(_intrinsics(camera_records[key]) for key in sorted(camera_records))

    images: list[BundleImage] = []
    missing: list[str] = []
    for record in sorted(image_records, key=lambda row: row["name"]):
        if record["camera_id"] not in camera_records:
            raise DatasetIncompleteError(
                f"{NAME}: image {record['name']} references camera "
                f"{record['camera_id']} which cameras.bin does not declare"
            )
        relative = record["name"].replace("\\", "/")
        if not (images_root / relative).is_file():
            missing.append(relative)
            continue
        rotation = quaternion_wxyz_to_rotation(record["qvec_wxyz"])
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = -rotation.T @ record["tvec"]
        images.append(
            BundleImage(
                image_id=Path(relative).stem.replace("::", "__"),
                camera_id=str(record["camera_id"]),
                path=relative,
                c2w=tuple(tuple(float(v) for v in row) for row in c2w),
            )
        )
    if missing:
        head = ", ".join(missing[:5])
        raise DatasetIncompleteError(
            f"{NAME}: {len(missing)} registered images are absent from {images_root} "
            f"(first: {head}); pass images_dir= pointing at the real image root"
        )

    warnings: list[str] = []
    cloud_path = _find_cloud(root, point_cloud)
    cloud_ref = None
    if cloud_path is not None:
        cloud_ref = PointCloudRef(
            path=cloud_path, format=cloud_path.suffix.lower().lstrip(".")
        )
    else:
        sparse_count = len(read_points3d_binary(folder / "points3D.bin")[0])
        warnings.append(f"no_lidar_cloud_sfm_points_only:{sparse_count}")
    warnings.append("no_capture_timestamps_in_colmap_model")

    return DatasetBundle(
        dataset_id=dataset_id or root.name,
        adapter=NAME,
        source_root=root,
        images_root=images_root,
        cameras=cameras,
        images=tuple(images),
        point_cloud=cloud_ref,
        coordinate_frame=coordinate_frame,
        pose_convention="c2w_opencv",
        warnings=tuple(warnings),
    )


__all__ = [
    "CAMERA_MODELS",
    "NAME",
    "REQUIRES",
    "VERSION",
    "detect",
    "export_sparse_point_cloud",
    "load",
    "model_dir",
    "quaternion_wxyz_to_rotation",
    "read_cameras_binary",
    "read_images_binary",
    "read_points3d_binary",
]
