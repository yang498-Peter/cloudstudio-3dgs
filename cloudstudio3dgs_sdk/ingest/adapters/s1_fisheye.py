"""MVP-S1 fisheye rig: the house0305 shape.

Layout (recording and run dir may be the same directory, and are on S1)::

    <recording>/info/calibration.json      left + right fisheye, transform_from_lidar
    <recording>/camera/left/<ns>.jpg       nanosecond-stamped frames
    <recording>/camera/right/<ns>.jpg
    <run>/ImgPose.txt                      per-image trajectory poses
    <run>/colorized.las                    colorized scene cloud (or .laz/uncolorized)
    <run>/transforms.json                  optional, only cross-checked

Bit-identity with today's inputs
--------------------------------
This adapter does not re-implement the S1 reader.  It calls
:func:`cloudstudio_3dgs.data.manifest.build_manifest` - the exact code that
produced ``house0305_sop_v8/dataset_manifest.json`` - and maps its output into
a bundle, keeping the payload in ``bundle.native_dataset_manifest``.  When a
signed ``dataset_manifest.json`` already exists next to the dataset (or is
passed in), it is verified and reused verbatim instead of rebuilt, so an
existing run's ``manifest_sha256`` survives ingestion unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..bundle import (
    BundleImage,
    CameraIntrinsics,
    DatasetBundle,
    PointCloudRef,
    RigTransform,
)
from ..errors import DatasetIncompleteError

NAME = "s1_fisheye"
VERSION = "1"
REQUIRES = "info/calibration.json + camera/{left,right}/ + ImgPose.txt + a point cloud"

_CALIBRATION = Path("info") / "calibration.json"
_POINT_CLOUD_SUFFIXES = (".las", ".laz")


def detect(path: Path) -> bool:
    path = Path(path)
    return (
        (path / _CALIBRATION).is_file()
        and (path / "camera" / "left").is_dir()
        and (path / "camera" / "right").is_dir()
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetIncompleteError(f"{NAME}: {message}")


def _check_inputs(recording_dir: Path, run_dir: Path) -> None:
    _require(
        (recording_dir / _CALIBRATION).is_file(),
        f"calibration is missing: {recording_dir / _CALIBRATION}",
    )
    for side in ("left", "right"):
        folder = recording_dir / "camera" / side
        _require(folder.is_dir(), f"camera folder is missing: {folder}")
        _require(
            any(folder.iterdir()),
            f"camera folder has no images: {folder}",
        )
    _require(
        (run_dir / "ImgPose.txt").is_file(),
        f"pose file is missing: {run_dir / 'ImgPose.txt'}",
    )
    clouds = [
        candidate
        for candidate in sorted(run_dir.iterdir())
        if candidate.is_file()
        and candidate.suffix.lower() in _POINT_CLOUD_SUFFIXES
        and "ecef" not in candidate.name.lower()
    ]
    _require(
        bool(clouds),
        f"no LAS/LAZ point cloud in the run directory: {run_dir}",
    )


def _splits_from_manifest(split_manifest: Mapping[str, Any]) -> dict[str, str]:
    """Map image_id -> split from a signed ``split_manifest.json``."""

    assignment: dict[str, str] = {}
    for frame in split_manifest.get("rig_frames", []):
        split = frame.get("split")
        if not split:
            continue
        for image_id in frame.get("image_ids", []):
            assignment[str(image_id)] = str(split)
    for golden in split_manifest.get("golden_views", []):
        for image_id in golden.get("image_ids", []):
            assignment[str(image_id)] = "golden"
    return assignment


def bundle_from_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    recording_root: Path,
    run_root: Path,
    dataset_id: str,
    splits: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """Map a signed S1 ``dataset_manifest`` payload onto a bundle."""

    cameras: list[CameraIntrinsics] = []
    for row in manifest.get("cameras", []):
        intrinsic = row["intrinsic"]
        distortion = row.get("distortion", {})
        transform = row.get("transform_from_lidar")
        cameras.append(
            CameraIntrinsics(
                camera_id=str(row["camera_id"]),
                width=int(row["width"]),
                height=int(row["height"]),
                fx=float(intrinsic["fl_x"]),
                fy=float(intrinsic["fl_y"]),
                cx=float(intrinsic["cx"]),
                cy=float(intrinsic["cy"]),
                camera_model=str(distortion.get("camera_model", "OPENCV_FISHEYE")),
                distortion=dict(distortion.get("params", {})),
                transform_from_lidar=(
                    RigTransform.from_mapping(transform) if transform else None
                ),
                side=row.get("side"),
            )
        )
    if not cameras:
        raise DatasetIncompleteError(f"{NAME}: dataset manifest declares no cameras")

    assignment = dict(splits or {})
    images: list[BundleImage] = []
    for row in manifest.get("images", []):
        images.append(
            BundleImage(
                image_id=str(row["image_id"]),
                camera_id=str(row["camera_id"]),
                path=str(row["path"]),
                c2w=tuple(tuple(float(v) for v in line) for line in row["c2w"]),
                timestamp_ns=(
                    None if row.get("timestamp_ns") is None else int(row["timestamp_ns"])
                ),
                rig_frame_id=row.get("rig_frame_id"),
                split=assignment.get(str(row["image_id"]), row.get("split")),
                sha256=(
                    row.get("sha256")
                    if row.get("sha256") not in (None, "not_computed")
                    else None
                ),
                size_bytes=row.get("size_bytes"),
            )
        )
    if not images:
        raise DatasetIncompleteError(f"{NAME}: dataset manifest declares no posed images")

    point_cloud = None
    cloud = manifest.get("point_cloud")
    if cloud:
        root = run_root if cloud.get("path_root") == "run" else recording_root
        absolute = root / str(cloud["path"])
        point_cloud = PointCloudRef(
            path=absolute,
            format=absolute.suffix.lower().lstrip(".") or "las",
            sha256=(
                cloud.get("sha256")
                if cloud.get("sha256") not in (None, "not_computed")
                else None
            ),
            size_bytes=cloud.get("size_bytes"),
        )

    return DatasetBundle(
        dataset_id=dataset_id,
        adapter=NAME,
        source_root=recording_root,
        images_root=recording_root,
        cameras=tuple(cameras),
        images=tuple(images),
        point_cloud=point_cloud,
        coordinate_frame=str(manifest.get("coordinate_frame", "s1_local")),
        pose_convention="c2w_opencv",
        rig_frames=tuple(dict(frame) for frame in manifest.get("rig_frames", [])),
        warnings=tuple(str(w) for w in manifest.get("warnings", [])),
        native_dataset_manifest=manifest,
    )


def load(
    path: Path,
    *,
    run_dir: Path | None = None,
    dataset_manifest: Path | None = None,
    split_manifest: Path | None = None,
    hash_images: bool = True,
    hash_point_cloud: bool = True,
    dataset_id: str | None = None,
) -> DatasetBundle:
    """Load an S1 capture.

    ``dataset_manifest`` - reuse an existing signed manifest instead of
    rebuilding.  When omitted, a ``dataset_manifest.json`` sitting next to the
    recording is reused automatically; that is what keeps house0305 identical.
    """

    from cloudstudio_3dgs.data.manifest import build_manifest
    from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest

    recording_dir = Path(path).resolve()
    run_root = Path(run_dir).resolve() if run_dir is not None else recording_dir
    if not detect(recording_dir):
        raise DatasetIncompleteError(
            f"{NAME}: {recording_dir} is not an S1 recording ({REQUIRES})"
        )
    _check_inputs(recording_dir, run_root)

    manifest_path = Path(dataset_manifest) if dataset_manifest else None
    if manifest_path is None:
        default = recording_dir / "dataset_manifest.json"
        manifest_path = default if default.is_file() else None
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise DatasetIncompleteError(
                f"{NAME}: dataset manifest is missing: {manifest_path}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_dataset_manifest(payload)
    else:
        payload = build_manifest(
            recording_dir,
            run_root,
            hash_images=hash_images,
            hash_point_cloud=hash_point_cloud,
        )

    splits: dict[str, str] = {}
    if split_manifest is not None:
        split_path = Path(split_manifest)
        if not split_path.is_file():
            raise DatasetIncompleteError(
                f"{NAME}: split manifest is missing: {split_path}"
            )
        splits = _splits_from_manifest(
            json.loads(split_path.read_text(encoding="utf-8"))
        )

    return bundle_from_dataset_manifest(
        payload,
        recording_root=recording_dir,
        run_root=run_root,
        dataset_id=dataset_id or recording_dir.name,
        splits=splits,
    )


__all__ = ["NAME", "REQUIRES", "VERSION", "bundle_from_dataset_manifest", "detect", "load"]
