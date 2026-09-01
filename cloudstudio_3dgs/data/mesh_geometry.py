"""Signed per-view LiDAR surface depth and normal sidecars."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.depth_cache import deterministic_npz_bytes
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


MESH_GEOMETRY_SCHEMA_VERSION = 1
MESH_GEOMETRY_KIND = "face4_lidar_surface_depth_normal"
STRICT_MESH_SOURCE_TYPES = (2, 3)


def strict_mesh_admission_mask(
    valid: np.ndarray,
    source_type: np.ndarray,
    *,
    allowed_source_types: tuple[int, ...] = STRICT_MESH_SOURCE_TYPES,
) -> np.ndarray:
    """Return the fail-closed Trainer mask for authoritative mesh pixels.

    Type 2 is a native LiDAR anchor and type 3 has cross-view support. Type 4
    is deliberately excluded: ``unobservable_or_occluded_retained`` is useful
    diagnostic output, but is not geometric training truth.
    """

    valid_array = np.asarray(valid, dtype=bool)
    source_array = np.asarray(source_type)
    if source_array.shape != valid_array.shape:
        raise ValueError("source_type must match the valid mask shape")
    if not allowed_source_types:
        raise ValueError("allowed_source_types must not be empty")
    allowed = np.asarray(tuple(int(value) for value in allowed_source_types))
    if np.any(allowed < 0) or np.any(allowed > 255):
        raise ValueError("allowed source types must fit uint8")
    return valid_array & np.isin(source_array, allowed)


def mesh_geometry_npz_bytes(
    depth_range_m: np.ndarray,
    normal_camera: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    *,
    source_type: int | np.ndarray = 1,
) -> bytes:
    """Serialize one dense Face4 surface observation deterministically.

    ``depth_range_m`` is Euclidean camera-ray range in metres. Normals are unit
    vectors in the Face4 camera coordinate system. Invalid pixels are written
    as zero so hashes do not depend on uninitialized values.
    """

    depth = np.asarray(depth_range_m, dtype=np.float32)
    normal = np.asarray(normal_camera, dtype=np.float32)
    confidence_array = np.asarray(confidence, dtype=np.float32)
    valid_array = np.asarray(valid, dtype=bool)
    if depth.ndim != 2:
        raise ValueError("depth_range_m must have shape [H, W]")
    if normal.shape != (*depth.shape, 3):
        raise ValueError("normal_camera must have shape [H, W, 3]")
    if confidence_array.shape != depth.shape or valid_array.shape != depth.shape:
        raise ValueError("confidence and valid must match the depth shape")
    if np.isscalar(source_type):
        if not 0 <= int(source_type) <= 255:
            raise ValueError("source_type must fit uint8")
        source_type_array = np.full(depth.shape, int(source_type), dtype=np.uint8)
    else:
        source_type_raw = np.asarray(source_type)
        if source_type_raw.shape != depth.shape:
            raise ValueError("source_type array must match the depth shape")
        if np.any(source_type_raw < 0) or np.any(source_type_raw > 255):
            raise ValueError("source_type array values must fit uint8")
        source_type_array = source_type_raw.astype(np.uint8)

    finite = np.isfinite(depth) & (depth > 0.0)
    finite &= np.all(np.isfinite(normal), axis=-1)
    finite &= np.isfinite(confidence_array)
    valid_array &= finite
    normal_length = np.linalg.norm(normal, axis=-1)
    valid_array &= normal_length > 1e-6

    safe_depth = np.where(valid_array, depth, 0.0).astype("<f4")
    safe_normal = np.zeros_like(normal, dtype=np.float32)
    safe_normal[valid_array] = (
        normal[valid_array] / normal_length[valid_array, None]
    )
    safe_confidence = np.where(
        valid_array, np.clip(confidence_array, 0.0, 1.0), 0.0
    ).astype(np.float32)
    source = np.zeros(depth.shape, dtype=np.uint8)
    source[valid_array] = source_type_array[valid_array]
    return deterministic_npz_bytes(
        {
            "depth_range_m": safe_depth,
            "normal_camera": safe_normal.astype("<f2"),
            "confidence": safe_confidence.astype("<f2"),
            "valid": valid_array.astype(np.uint8),
            "source_type": source,
            "shape": np.asarray(depth.shape, dtype="<i4"),
        }
    )


def sign_mesh_geometry_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("mesh_geometry_manifest_sha256", None)
    signed = copy.deepcopy(unsigned)
    signed["mesh_geometry_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def verify_mesh_geometry_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("mesh_geometry_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("mesh geometry manifest is unsigned")
    unsigned = copy.deepcopy(manifest)
    unsigned.pop("mesh_geometry_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("mesh geometry manifest signature mismatch")
    if int(manifest.get("schema_version", -1)) != MESH_GEOMETRY_SCHEMA_VERSION:
        raise ValueError("unsupported mesh geometry manifest schema")
    if manifest.get("kind") != MESH_GEOMETRY_KIND:
        raise ValueError("unexpected mesh geometry manifest kind")
    if manifest.get("complete_face_cache") is not True:
        raise ValueError("mesh geometry manifest is incomplete")
    return actual
