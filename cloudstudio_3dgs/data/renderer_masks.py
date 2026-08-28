"""Signed renderer-mask manifests for the MipMap-aligned Face4 route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.face_dataset import verify_face_manifest


RENDERER_MASK_SCHEMA_VERSION = 1
RENDERER_MASK_KIND = "face4_renderer_mask_cache"
RENDERER_MASK_POLICY = "mipmap_renderer_visibility_compat_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"artifact paths must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"artifact path escapes its root: {value!r}")
    return resolved


def sign_renderer_mask_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("renderer_mask_manifest_sha256", None)
    signed = dict(unsigned)
    signed["renderer_mask_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def verify_renderer_mask_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("renderer_mask_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("renderer mask manifest is unsigned")
    unsigned = dict(manifest)
    unsigned.pop("renderer_mask_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("renderer mask manifest signature mismatch")
    if int(manifest.get("schema_version", -1)) != RENDERER_MASK_SCHEMA_VERSION:
        raise ValueError("unsupported renderer mask manifest schema")
    if manifest.get("kind") != RENDERER_MASK_KIND:
        raise ValueError("unexpected renderer mask manifest kind")
    policy = manifest.get("policy", {})
    if (
        policy.get("profile") != RENDERER_MASK_POLICY
        or policy.get("keep_expression") != "face_cache_combined_mask != 0"
        or policy.get("competitor_reference_expression")
        != "(seg != 255) & (seg != 33)"
        or policy.get("label_33_semantics") != "UNKNOWN_NOT_INFERRED"
    ):
        raise ValueError("renderer mask policy is incomplete or unsupported")
    records = manifest.get("masks", [])
    keys = [
        (str(record.get("image_id", "")), str(record.get("face_id", "")))
        for record in records
    ]
    if not records or len(keys) != len(set(keys)) or any(not all(key) for key in keys):
        raise ValueError("renderer mask manifest has invalid or duplicate records")
    summary = manifest.get("summary", {})
    if (
        int(summary.get("face_sample_count", -1)) != len(records)
        or int(summary.get("empty_mask_count", -1)) != 0
        or int(summary.get("missing_mask_count", -1)) != 0
    ):
        raise ValueError("renderer mask manifest summary is not training-ready")
    return expected


def build_renderer_mask_manifest(
    face_manifest: dict[str, Any],
    face_cache_root: Path,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Bind every Face4 combined mask to the recovered renderer-mask contract.

    This is deliberately not advertised as recovered SegFormer classes. The
    static audit proves only the competitor's final boolean renderer rule;
    label 33's class name remains unknown. Our combined mask is stronger and
    auditable: circle/FoV validity AND the separately signed person mask.
    """
    face_sha = verify_face_manifest(face_manifest)
    root = Path(face_cache_root)
    records: list[dict[str, Any]] = []
    total_keep_pixels = 0
    image_ids: set[str] = set()
    missing_count = 0
    empty_count = 0
    for image in face_manifest.get("images", []):
        image_id = str(image["image_id"])
        image_ids.add(image_id)
        for face in image.get("faces", []):
            relative = str(face["mask_path"])
            path = _safe_artifact(root, relative)
            if not path.is_file():
                missing_count += 1
                raise FileNotFoundError(f"Face4 renderer mask is missing: {path}")
            expected_sha = str(face["mask_sha256"])
            if verify_artifacts:
                actual_sha = _sha256_file(path)
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"Face4 renderer mask SHA256 mismatch: {path}"
                    )
            with Image.open(path) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8)
            keep_pixels = int(np.count_nonzero(mask))
            if keep_pixels <= 0:
                empty_count += 1
                raise ValueError(f"Face4 renderer mask is empty: {path}")
            if keep_pixels != int(face.get("mask_true_pixels", -1)):
                raise ValueError(
                    f"Face4 renderer mask pixel count mismatch: {path}"
                )
            total_keep_pixels += keep_pixels
            records.append(
                {
                    "image_id": image_id,
                    "camera_id": str(image["camera_id"]),
                    "face_id": str(face["face_id"]),
                    "mask_path": relative,
                    "mask_sha256": expected_sha,
                    "width": int(mask.shape[1]),
                    "height": int(mask.shape[0]),
                    "keep_pixels": keep_pixels,
                }
            )
    source_identity = dict(face_manifest.get("source_identity", {}))
    if not source_identity.get("person_mask_manifest_sha256"):
        raise ValueError("Face4 cache is not bound to a person mask manifest")
    payload = {
        "schema_version": RENDERER_MASK_SCHEMA_VERSION,
        "kind": RENDERER_MASK_KIND,
        "split": str(face_manifest.get("split", "")),
        "source_face_manifest_sha256": face_sha,
        "source_identity": source_identity,
        "policy": {
            "profile": RENDERER_MASK_POLICY,
            "consumer": "renderer_forward_visibility",
            "keep_expression": "face_cache_combined_mask != 0",
            "competitor_reference_expression": "(seg != 255) & (seg != 33)",
            "excluded_sources": [
                "fisheye_circle_invalid",
                "face_fov_invalid",
                "face_weight_below_minimum",
                "person_dynamic",
            ],
            "multiclass_logits_recovered": False,
            "label_33_semantics": "UNKNOWN_NOT_INFERRED",
        },
        "masks": records,
        "summary": {
            "image_count": len(image_ids),
            "face_sample_count": len(records),
            "total_keep_pixels": total_keep_pixels,
            "empty_mask_count": empty_count,
            "missing_mask_count": missing_count,
        },
    }
    return sign_renderer_mask_manifest(payload)
