#!/usr/bin/env python3
"""Build sparse real-LiDAR range supervision for an immutable Face4 cache.

The source Face4 RGB/mask cache is never rewritten.  Raw-fisheye sparse LiDAR
range is forward-splatted into each signed Face4 virtual camera, intersected
with that face's existing supervision mask, and stored in a separate signed
sidecar consumed by ``FaceCacheDataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import (
    load_xyz_point_cloud,
    load_sparse_depth,
    sparse_depth_npz_bytes,
    verify_depth_manifest,
)
from cloudstudio_3dgs.data.face_lidar_geometry import (
    FACE_LIDAR_GEOMETRY_KIND,
    FACE_LIDAR_GEOMETRY_SCHEMA_VERSION,
    sign_face_lidar_geometry_manifest,
    verify_face_lidar_geometry_manifest,
)
from cloudstudio_3dgs.data.face_warp import warp_sparse_depth_to_face
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec
from cloudstudio_3dgs.geometry.lidar_projection import (
    DepthProjectionConfig,
    SparseDepthMap,
    project_camera_points_to_face,
)
from cloudstudio_3dgs.training.face_dataset import (
    SAMPLE_ID_SEPARATOR,
    verify_face_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_artifact(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"artifact path must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    root = Path(root).resolve()
    path = (root / Path(*pure.parts)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path escapes root: {value!r}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _camera_calibration(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = camera["intrinsic"]
    distortion = camera["distortion"]["params"]
    K = np.array(
        [
            [intrinsic["fl_x"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fl_y"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    radial = np.array(
        [distortion["k1"], distortion["k2"], distortion["k3"], distortion["k4"]],
        dtype=np.float64,
    )
    return K, radial


def build_face4_lidar_geometry(
    *,
    face_manifest_path: Path,
    face_root: Path,
    dataset_manifest_path: Path,
    depth_manifest_path: Path,
    depth_root: Path,
    output_root: Path,
    point_cloud_path: Path | None = None,
    storage_profile: str = "audit",
    workers: int = 1,
) -> dict[str, Any]:
    face = _read(face_manifest_path)
    dataset = _read(dataset_manifest_path)
    depth = _read(depth_manifest_path)
    face_sha = verify_face_manifest(face)
    dataset_sha = verify_dataset_manifest(dataset)
    depth_sha = verify_depth_manifest(depth)
    source_identity = face.get("source_identity", {})
    if source_identity.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("Face4 cache and dataset manifest identities differ")
    if depth.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("LiDAR depth and dataset manifest identities differ")
    if depth.get("complete_dataset") is not True:
        raise ValueError("LiDAR depth manifest is incomplete")
    if storage_profile not in {"audit", "trainer_compact"}:
        raise ValueError("storage_profile must be 'audit' or 'trainer_compact'")

    direct_projection = point_cloud_path is not None
    if direct_projection and int(workers) != 1:
        raise ValueError(
            "full-LAS direct Face4 projection must use workers=1 to bound RAM; "
            "parallel Face4 images duplicate the full camera-frame point array"
        )
    points: np.ndarray | None = None
    point_source_index: np.ndarray | None = None
    projection_config = DepthProjectionConfig(**depth.get("projection", {}))
    if direct_projection:
        assert point_cloud_path is not None
        expected_cloud_sha = str(depth.get("point_cloud_sha256", ""))
        dataset_cloud_sha = str(dataset.get("point_cloud", {}).get("sha256", ""))
        actual_cloud_sha = _sha256_file(point_cloud_path)
        if not expected_cloud_sha or actual_cloud_sha != expected_cloud_sha:
            raise ValueError("full LAS SHA256 does not match LiDAR depth manifest")
        if dataset_cloud_sha != actual_cloud_sha:
            raise ValueError("full LAS SHA256 does not match dataset manifest")
        expected_points = int(depth.get("point_cloud_points", 0))
        if expected_points <= 0:
            raise ValueError("LiDAR depth manifest has no positive point count")
        points = load_xyz_point_cloud(point_cloud_path, max_points=expected_points)
        if len(points) != expected_points:
            raise ValueError(
                f"full LAS point count mismatch: expected {expected_points}, got {len(points)}"
            )
        point_source_index = np.arange(len(points), dtype=np.int64)

    cameras = {str(item["camera_id"]): item for item in dataset["cameras"]}
    dataset_images = {str(item["image_id"]): item for item in dataset["images"]}
    depth_by_image = {str(item["image_id"]): item for item in depth["images"]}
    if len(depth_by_image) != len(depth["images"]):
        raise ValueError("LiDAR depth manifest contains duplicate image IDs")
    face_specs = {
        (str(camera_id), str(payload["face_id"])): FaceSpec.from_dict(payload)
        for camera_id, camera in face["cameras"].items()
        for payload in camera["faces"]
    }
    output_root = Path(output_root)
    depth_output = output_root / "depth"
    depth_output.mkdir(parents=True, exist_ok=True)

    def process(image: dict[str, Any]) -> list[dict[str, Any]]:
        image_id = str(image["image_id"])
        camera_id = str(image["camera_id"])
        if image_id not in depth_by_image:
            raise ValueError(f"LiDAR depth does not cover Face4 image {image_id}")
        if image_id not in dataset_images:
            raise ValueError(f"dataset manifest does not cover Face4 image {image_id}")
        dataset_image = dataset_images[image_id]
        if str(dataset_image["camera_id"]) != camera_id:
            raise ValueError(f"camera identity differs for Face4 image {image_id}")
        source_record = depth_by_image[image_id]
        source_range: np.ndarray | None = None
        source_confidence: np.ndarray | None = None
        source_valid: np.ndarray | None = None
        K: np.ndarray | None = None
        radial: np.ndarray | None = None
        points_camera: np.ndarray | None = None
        if direct_projection:
            assert points is not None
            pose = np.asarray(dataset_image["c2w"], dtype=np.float64)
            points_camera = (points - pose[:3, 3]) @ pose[:3, :3]
        else:
            source_path = _safe_artifact(depth_root, str(source_record["path"]))
            if _sha256_file(source_path) != str(source_record["sha256"]):
                raise ValueError(f"LiDAR depth SHA256 mismatch for {image_id}")
            sparse = load_sparse_depth(source_path)
            source_range, source_confidence, source_valid = sparse.to_dense()
            K, radial = _camera_calibration(cameras[camera_id])
        records: list[dict[str, Any]] = []
        for face_entry in image["faces"]:
            face_id = str(face_entry["face_id"])
            sample_id = f"{image_id}{SAMPLE_ID_SEPARATOR}{face_id}"
            spec = face_specs[(camera_id, face_id)]
            mask_path = _safe_artifact(face_root, str(face_entry["mask_path"]))
            if _sha256_file(mask_path) != str(face_entry["mask_sha256"]):
                raise ValueError(f"Face4 mask SHA256 mismatch for {sample_id}")
            with Image.open(mask_path) as source:
                supervision_mask = np.asarray(source.convert("L"), dtype=np.uint8) > 0
            if direct_projection:
                assert points_camera is not None
                assert point_source_index is not None
                face_sparse = project_camera_points_to_face(
                    points_camera,
                    spec,
                    source_index=point_source_index,
                    supervision_mask=supervision_mask,
                    config=projection_config,
                )
                pixel_index = face_sparse.pixel_index
            else:
                assert source_range is not None
                assert source_confidence is not None
                assert source_valid is not None
                assert K is not None
                assert radial is not None
                face_range, face_confidence, face_valid = warp_sparse_depth_to_face(
                    source_range,
                    source_confidence,
                    source_valid,
                    K,
                    radial,
                    spec,
                )
                face_valid &= supervision_mask
                keep = face_valid & np.isfinite(face_range) & (face_range > 0.0)
                keep &= (
                    np.isfinite(face_confidence)
                    & (face_confidence > 0.0)
                    & (face_confidence <= 1.0)
                )
                pixel_index = np.flatnonzero(keep).astype(np.int32)
            relative: str | None = None
            artifact_sha: str | None = None
            if pixel_index.size:
                if not direct_projection:
                    count = int(pixel_index.size)
                    face_sparse = SparseDepthMap(
                        shape=(int(spec.height), int(spec.width)),
                        pixel_index=pixel_index,
                        range_m=face_range.reshape(-1)[pixel_index].astype(np.float32),
                        confidence=face_confidence.reshape(-1)[pixel_index].astype(np.float32),
                        source_index=np.full(count, -1, dtype=np.int64),
                        support_count=np.zeros(count, dtype=np.int32),
                    )
                payload = sparse_depth_npz_bytes(
                    face_sparse,
                    include_provenance=storage_profile == "audit",
                    confidence_encoding=(
                        "float32" if storage_profile == "audit" else "uint8"
                    ),
                )
                relative = f"depth/{image_id}_{face_id}.npz"
                artifact = _safe_artifact(output_root, relative)
                _atomic_write(artifact, payload)
                artifact_sha = hashlib.sha256(payload).hexdigest()
            records.append(
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "face_id": face_id,
                    "path": relative,
                    "sha256": artifact_sha,
                    "shape": [int(spec.height), int(spec.width)],
                    "valid_pixels": int(pixel_index.size),
                    "valid_fraction": float(
                        pixel_index.size / max(1, np.count_nonzero(supervision_mask))
                    ),
                }
            )
        return records

    images = list(face["images"])
    if int(workers) <= 1:
        batches = [process(image) for image in images]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            batches = list(executor.map(process, images))
    records = [record for batch in batches for record in batch]
    valid_total = sum(int(record["valid_pixels"]) for record in records)
    payload = {
        "schema_version": FACE_LIDAR_GEOMETRY_SCHEMA_VERSION,
        "kind": FACE_LIDAR_GEOMETRY_KIND,
        "split": face["split"],
        "source_face_manifest_sha256": face_sha,
        "source_depth_manifest_sha256": depth_sha,
        "dataset_manifest_sha256": dataset_sha,
        "complete_face_cache": True,
        "expected_face_count": len(records),
        "depth_semantics": "euclidean_ray_range_m_sparse_real_lidar_only",
        "projection": (
            "full_las_to_face4_direct_nearest_range_zbuffer"
            if direct_projection
            else "kb4_forward_splat_nearest_range_zbuffer"
        ),
        "intermediate_fisheye_raster": not direct_projection,
        "destination_pixel_quantizations": 1 if direct_projection else 2,
        "provenance_mode": (
            "explicit_original_las_source_index_and_face_support_count"
            if direct_projection and storage_profile == "audit"
            else "signed_global_las_identity_without_per_pixel_source_index"
            if direct_projection
            else "implicit_source_index_minus1_support_count_0"
        ),
        "storage_profile": storage_profile,
        "confidence_encoding": (
            "float32" if storage_profile == "audit" else "uint8_div_255"
        ),
        "source_depth_manifest_role": (
            "signed_projection_parameter_and_identity_anchor_only"
            if direct_projection
            else "source_sparse_range_cache"
        ),
        "source_point_cloud_sha256": depth.get("point_cloud_sha256"),
        "source_point_cloud_points": depth.get("point_cloud_points"),
        "face_supervision_mask_intersection": True,
        "mesh_interpolation": False,
        "records": records,
        "summary": {
            "face_count": len(records),
            "with_depth_count": sum(record["valid_pixels"] > 0 for record in records),
            "without_depth_count": sum(record["valid_pixels"] == 0 for record in records),
            "valid_pixel_count": valid_total,
        },
    }
    signed = sign_face_lidar_geometry_manifest(payload)
    verify_face_lidar_geometry_manifest(signed)
    _atomic_write(
        output_root / "face_lidar_geometry_manifest.json",
        (json.dumps(signed, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return signed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--face-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--depth-manifest", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument(
        "--point-cloud",
        type=Path,
        help=(
            "full LAS/PLY/NPY input; when provided, project exact 3D points "
            "directly to Face4 instead of warping integer fisheye depth"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--storage-profile",
        choices=("audit", "trainer_compact"),
        default="audit",
        help=(
            "audit preserves per-pixel LAS indexes; trainer_compact preserves "
            "float32 range but quantizes confidence and omits unused provenance"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest_path = args.output / "face_lidar_geometry_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace {manifest_path}")
    signed = build_face4_lidar_geometry(
        face_manifest_path=args.face_manifest,
        face_root=args.face_root,
        dataset_manifest_path=args.dataset_manifest,
        depth_manifest_path=args.depth_manifest,
        depth_root=args.depth_root,
        output_root=args.output,
        point_cloud_path=args.point_cloud,
        storage_profile=args.storage_profile,
        workers=args.workers,
    )
    summary = signed["summary"]
    print(
        f"Face4 LiDAR geometry {signed['split']}: "
        f"faces={summary['face_count']}, with_depth={summary['with_depth_count']}, "
        f"valid_pixels={summary['valid_pixel_count']}, sha256="
        f"{signed['face_lidar_geometry_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
