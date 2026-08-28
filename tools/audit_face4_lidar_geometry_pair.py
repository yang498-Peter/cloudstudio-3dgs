#!/usr/bin/env python3
"""Fail-closed audit for one same-generation train/val Face4 LiDAR pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.face_lidar_geometry import (
    verify_face_lidar_geometry_manifest,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec
from cloudstudio_3dgs.training.face_dataset import (
    SAMPLE_ID_SEPARATOR,
    verify_face_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"artifact path must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    if pure.parts[0] != "depth":
        raise ValueError(f"independent geometry must live under depth/: {value!r}")
    resolved_root = Path(root).resolve()
    path = (resolved_root / Path(*pure.parts)).resolve()
    if resolved_root not in path.parents:
        raise ValueError(f"artifact escapes geometry root: {value!r}")
    return path


def _face_shapes(face: dict[str, Any]) -> dict[str, tuple[int, int]]:
    specs = {
        (str(camera_id), str(payload["face_id"])): FaceSpec.from_dict(payload)
        for camera_id, camera in face["cameras"].items()
        for payload in camera["faces"]
    }
    result: dict[str, tuple[int, int]] = {}
    for image in face["images"]:
        camera_id = str(image["camera_id"])
        for entry in image["faces"]:
            face_id = str(entry["face_id"])
            sample_id = f"{image['image_id']}{SAMPLE_ID_SEPARATOR}{face_id}"
            spec = specs[(camera_id, face_id)]
            result[sample_id] = (int(spec.height), int(spec.width))
    return result


def _audit_split(
    *,
    split: str,
    face_manifest_path: Path,
    geometry_manifest_path: Path,
    geometry_root: Path,
) -> dict[str, Any]:
    face = _read(face_manifest_path)
    geometry = _read(geometry_manifest_path)
    face_sha = verify_face_manifest(face)
    geometry_sha = verify_face_lidar_geometry_manifest(geometry)
    if face.get("split") != split or geometry.get("split") != split:
        raise ValueError(f"{split} manifests carry the wrong split")
    if geometry.get("source_face_manifest_sha256") != face_sha:
        raise ValueError(f"{split} geometry is bound to a different Face4 cache")
    expected = _face_shapes(face)
    records = {str(record["sample_id"]): record for record in geometry["records"]}
    if set(records) != set(expected):
        raise ValueError(f"{split} geometry does not exactly cover the Face4 cache")

    storage_profile = str(geometry.get("storage_profile", ""))
    if storage_profile not in {"audit", "trainer_compact"}:
        raise ValueError(f"{split} geometry has no supported storage profile")
    valid_total = 0
    bytes_total = 0
    with_depth = 0
    source_min: int | None = None
    source_max: int | None = None
    support_min: int | None = None
    support_max: int | None = None
    for sample_id, record in records.items():
        shape = tuple(int(value) for value in record.get("shape", []))
        if shape != expected[sample_id]:
            raise ValueError(f"{split} shape mismatch for {sample_id}")
        count = int(record["valid_pixels"])
        fraction = float(record.get("valid_fraction", -1.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{split} invalid valid_fraction for {sample_id}")
        if count == 0:
            if fraction != 0.0:
                raise ValueError(f"{split} empty sample has nonzero valid_fraction")
            continue
        path = _artifact(geometry_root, str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"{split} artifact SHA256 mismatch for {sample_id}")
        sparse = load_sparse_depth(path)
        if sparse.shape != shape or len(sparse.pixel_index) != count:
            raise ValueError(f"{split} sparse payload mismatch for {sample_id}")
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
        if storage_profile == "trainer_compact":
            expected_names = {"shape", "pixel_index", "range_m", "confidence_u8"}
            if names != expected_names:
                raise ValueError(f"{split} compact arrays differ for {sample_id}")
            if not np.all(sparse.source_index == -1) or not np.all(
                sparse.support_count == 0
            ):
                raise ValueError(f"{split} compact provenance sentinels differ")
        else:
            expected_names = {
                "shape",
                "pixel_index",
                "range_m",
                "confidence",
                "source_index",
                "support_count",
            }
            if names != expected_names:
                raise ValueError(f"{split} audit arrays differ for {sample_id}")
            if np.any(sparse.source_index < 0) or np.any(sparse.support_count < 1):
                raise ValueError(f"{split} audit provenance is invalid")
            local_source_min = int(np.min(sparse.source_index))
            local_source_max = int(np.max(sparse.source_index))
            local_support_min = int(np.min(sparse.support_count))
            local_support_max = int(np.max(sparse.support_count))
            source_min = local_source_min if source_min is None else min(source_min, local_source_min)
            source_max = local_source_max if source_max is None else max(source_max, local_source_max)
            support_min = local_support_min if support_min is None else min(support_min, local_support_min)
            support_max = local_support_max if support_max is None else max(support_max, local_support_max)
        valid_total += count
        bytes_total += path.stat().st_size
        with_depth += 1

    summary = geometry.get("summary", {})
    if (
        int(summary.get("face_count", -1)) != len(records)
        or int(summary.get("with_depth_count", -1)) != with_depth
        or int(summary.get("without_depth_count", -1)) != len(records) - with_depth
        or int(summary.get("valid_pixel_count", -1)) != valid_total
    ):
        raise ValueError(f"{split} summary does not match audited artifacts")
    return {
        "split": split,
        "face_manifest_sha256": face_sha,
        "geometry_manifest_sha256": geometry_sha,
        "storage_profile": storage_profile,
        "face_count": len(records),
        "with_depth_count": with_depth,
        "without_depth_count": len(records) - with_depth,
        "valid_pixel_count": valid_total,
        "artifact_bytes": bytes_total,
        "source_index_min": source_min,
        "source_index_max": source_max,
        "support_count_min": support_min,
        "support_count_max": support_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val"):
        parser.add_argument(f"--{split}-face-manifest", required=True, type=Path)
        parser.add_argument(f"--{split}-geometry-manifest", required=True, type=Path)
        parser.add_argument(f"--{split}-geometry-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audited = [
        _audit_split(
            split=split,
            face_manifest_path=getattr(args, f"{split}_face_manifest"),
            geometry_manifest_path=getattr(args, f"{split}_geometry_manifest"),
            geometry_root=getattr(args, f"{split}_geometry_root"),
        )
        for split in ("train", "val")
    ]
    train_geometry = _read(args.train_geometry_manifest)
    val_geometry = _read(args.val_geometry_manifest)
    identity_keys = (
        "dataset_manifest_sha256",
        "source_depth_manifest_sha256",
        "source_point_cloud_sha256",
        "source_point_cloud_points",
        "projection",
        "intermediate_fisheye_raster",
        "destination_pixel_quantizations",
        "storage_profile",
        "confidence_encoding",
    )
    differing = [
        key
        for key in identity_keys
        if train_geometry.get(key) != val_geometry.get(key)
    ]
    if differing:
        raise ValueError(f"train/val geometry generations differ: {differing}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "face4_lidar_geometry_pair_audit",
        "status": "READY",
        "shared_identity": {key: train_geometry.get(key) for key in identity_keys},
        "splits": audited,
    }
    payload["audit_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
