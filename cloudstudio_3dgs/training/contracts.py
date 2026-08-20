"""Signed identities for training coordinates and run inputs."""

from __future__ import annotations

import hashlib
from typing import Any

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def build_coordinate_transform_manifest(dataset_manifest_sha256: str) -> dict[str, Any]:
    """Declare that PR-11 trains directly in the metric S1 local frame."""
    if len(dataset_manifest_sha256) != 64:
        raise ValueError("dataset manifest SHA256 must contain 64 hexadecimal characters")
    try:
        bytes.fromhex(dataset_manifest_sha256)
    except ValueError as exc:
        raise ValueError("dataset manifest SHA256 is not hexadecimal") from exc
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "s1_local_identity_v1",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_frame": "s1_local",
        "model_frame": "s1_local",
        "units": "metres",
        "normalization_applied": False,
        "model_from_source": identity,
        "source_from_model": identity,
    }
    manifest["coordinate_transform_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


def verify_coordinate_transform_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("coordinate_transform_sha256", ""))
    if not expected:
        raise ValueError("coordinate transform manifest has no SHA256")
    unsigned = dict(manifest)
    unsigned.pop("coordinate_transform_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"coordinate transform SHA256 mismatch: expected {expected}, computed {actual}"
        )
    if manifest.get("source_frame") != "s1_local" or manifest.get("model_frame") != "s1_local":
        raise ValueError("PR-11 coordinate transform must preserve s1_local")
    if manifest.get("normalization_applied") is not False:
        raise ValueError("PR-11 must not normalize world coordinates")
    return actual
