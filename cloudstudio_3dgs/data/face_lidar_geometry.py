"""Signed sparse LiDAR geometry bound to an immutable Face4 RGB cache."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


FACE_LIDAR_GEOMETRY_SCHEMA_VERSION = 1
FACE_LIDAR_GEOMETRY_KIND = "face4_sparse_lidar_geometry"


def sign_face_lidar_geometry_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("face_lidar_geometry_manifest_sha256", None)
    signed = copy.deepcopy(unsigned)
    signed["face_lidar_geometry_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def verify_face_lidar_geometry_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("face_lidar_geometry_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("Face4 LiDAR geometry manifest is unsigned")
    unsigned = copy.deepcopy(manifest)
    unsigned.pop("face_lidar_geometry_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("Face4 LiDAR geometry manifest signature mismatch")
    if int(manifest.get("schema_version", -1)) != FACE_LIDAR_GEOMETRY_SCHEMA_VERSION:
        raise ValueError("unsupported Face4 LiDAR geometry manifest schema")
    if manifest.get("kind") != FACE_LIDAR_GEOMETRY_KIND:
        raise ValueError("unexpected Face4 LiDAR geometry manifest kind")
    if manifest.get("complete_face_cache") is not True:
        raise ValueError("Face4 LiDAR geometry manifest is incomplete")
    records = manifest.get("records", [])
    sample_ids = [str(record.get("sample_id", "")) for record in records]
    if not records or not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Face4 LiDAR geometry manifest has invalid sample IDs")
    if int(manifest.get("expected_face_count", -1)) != len(records):
        raise ValueError("Face4 LiDAR geometry manifest face count mismatch")
    for record in records:
        valid_pixels = int(record.get("valid_pixels", -1))
        path = record.get("path")
        sha256 = record.get("sha256")
        if valid_pixels < 0:
            raise ValueError("Face4 LiDAR geometry record has invalid pixel count")
        if valid_pixels == 0:
            if path is not None or sha256 is not None:
                raise ValueError("empty Face4 LiDAR geometry record carries an artifact")
        elif not path or not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError("non-empty Face4 LiDAR geometry record lacks an artifact")
    return expected
