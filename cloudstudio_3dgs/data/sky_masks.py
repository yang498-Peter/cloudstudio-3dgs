"""Signed SegFormer sky-mask manifests for the MipMap-aligned Face4 route.

Research supervision data for the "sky is special" training arm: one uint8
PNG per Face4 face (255 = sky AND face-valid, 0 otherwise) plus a signed
manifest that binds the masks to the face cache
(``source_face_manifest_sha256``), to the model identity (weights and config
hashes) and to the decision rule that produced them.

The segmentation model lives only in ``tools/build_sky_masks.py``; this
module has no ML dependency so a trainer can verify and load the cache with
numpy + Pillow alone. The manifest layout deliberately mirrors
:mod:`cloudstudio_3dgs.data.renderer_masks`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


SKY_MASK_SCHEMA_VERSION = 1
SKY_MASK_KIND = "face4_sky_mask_cache"
SKY_MASK_MANIFEST_SHA_KEY = "sky_mask_manifest_sha256"
# ADE20K index 2 is "sky". The id is bound explicitly because name matching
# is a trap on this label set: index 48 is "skyscraper".
ADE20K_SKY_LABEL_ID = 2
SKY_MASK_LABEL_IDS = [ADE20K_SKY_LABEL_ID]
SKY_MASK_LABEL_NAMES = ["sky"]
SKY_MASK_VALUE = 255
SKY_FRACTION_DENOMINATOR = "valid_pixels"
SKY_FRACTION_REPORT_THRESHOLD = 0.10

_RECORD_KEYS = (
    "image_id",
    "camera_id",
    "face_id",
    "width",
    "height",
    "mask_path",
    "mask_sha256",
    "valid_pixels",
    "sky_pixels",
    "sky_fraction",
)
_REQUIRED_MODEL_KEYS = ("id", "revision", "config_sha256", "weights_sha256", "license_note")
_REQUIRED_RULE_KEYS = (
    "decision",
    "sky_probability_threshold",
    "model_input",
    "upsampling",
    "valid_mask",
    "sky_fraction_denominator",
)


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


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def sky_mask_path_for(image_id: str, face_id: str) -> str:
    """Relative artifact path (forward slashes) for one face's sky mask."""
    return f"faces/{image_id}_{face_id}_sky.png"


def sign_sky_mask_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop(SKY_MASK_MANIFEST_SHA_KEY, None)
    signed = dict(unsigned)
    signed[SKY_MASK_MANIFEST_SHA_KEY] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def summarize_sky_mask_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    fractions = [float(record["sky_fraction"]) for record in records]
    return {
        "face_count": len(records),
        "image_count": len({str(record["image_id"]) for record in records}),
        "total_valid_pixels": int(sum(int(r["valid_pixels"]) for r in records)),
        "total_sky_pixels": int(sum(int(r["sky_pixels"]) for r in records)),
        "mean_sky_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "faces_with_sky_gt_10pct": int(
            sum(1 for f in fractions if f > SKY_FRACTION_REPORT_THRESHOLD)
        ),
        "faces_without_sky": int(sum(1 for r in records if int(r["sky_pixels"]) == 0)),
    }


def _validate_record(record: dict[str, Any]) -> None:
    missing = [key for key in _RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f"sky mask record is missing {missing}")
    width, height = int(record["width"]), int(record["height"])
    valid_pixels, sky_pixels = int(record["valid_pixels"]), int(record["sky_pixels"])
    if width <= 0 or height <= 0:
        raise ValueError("sky mask record has a non-positive size")
    if not 0 <= sky_pixels <= valid_pixels <= width * height:
        raise ValueError("sky mask record pixel counts are inconsistent")
    expected_fraction = sky_pixels / valid_pixels if valid_pixels else 0.0
    if abs(float(record["sky_fraction"]) - expected_fraction) > 1e-9:
        raise ValueError("sky mask record sky_fraction does not match its counts")
    if not _is_sha256(record["mask_sha256"]):
        raise ValueError("sky mask record has an invalid mask_sha256")
    _safe_artifact(Path("."), str(record["mask_path"]))


def verify_sky_mask_manifest(manifest: dict[str, Any]) -> str:
    """Fail-closed structural + signature check; returns the manifest SHA256."""
    expected = str(manifest.get(SKY_MASK_MANIFEST_SHA_KEY, ""))
    if not _is_sha256(expected):
        raise ValueError("sky mask manifest is unsigned")
    unsigned = dict(manifest)
    unsigned.pop(SKY_MASK_MANIFEST_SHA_KEY, None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("sky mask manifest signature mismatch")
    if int(manifest.get("schema_version", -1)) != SKY_MASK_SCHEMA_VERSION:
        raise ValueError("unsupported sky mask manifest schema")
    if manifest.get("kind") != SKY_MASK_KIND:
        raise ValueError("unexpected sky mask manifest kind")
    if not str(manifest.get("split", "")):
        raise ValueError("sky mask manifest has no split")
    if not _is_sha256(manifest.get("source_face_manifest_sha256", "")):
        raise ValueError("sky mask manifest is not bound to a face manifest")
    if list(manifest.get("label_ids", [])) != SKY_MASK_LABEL_IDS or list(
        manifest.get("label_names", [])
    ) != SKY_MASK_LABEL_NAMES:
        raise ValueError("sky mask manifest label binding is not ADE20K sky (id 2)")
    model = manifest.get("model", {})
    if not isinstance(model, dict) or any(not str(model.get(k, "")) for k in _REQUIRED_MODEL_KEYS):
        raise ValueError("sky mask manifest model identity is incomplete")
    rule = manifest.get("rule", {})
    if not isinstance(rule, dict) or any(k not in rule for k in _REQUIRED_RULE_KEYS):
        raise ValueError("sky mask manifest rule is incomplete")
    threshold = float(rule["sky_probability_threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("sky mask probability threshold must lie in (0, 1]")
    if rule.get("sky_fraction_denominator") != SKY_FRACTION_DENOMINATOR:
        raise ValueError("sky mask manifest uses an unsupported sky_fraction denominator")
    records = manifest.get("masks", [])
    keys = [
        (str(record.get("image_id", "")), str(record.get("face_id", "")))
        for record in records
    ]
    if not records or len(keys) != len(set(keys)) or any(not all(key) for key in keys):
        raise ValueError("sky mask manifest has invalid or duplicate records")
    for record in records:
        _validate_record(record)
    summary = manifest.get("summary", {})
    expected_summary = summarize_sky_mask_records(records)
    for key, value in expected_summary.items():
        if key not in summary:
            raise ValueError(f"sky mask manifest summary is missing {key}")
        if isinstance(value, float):
            if abs(float(summary[key]) - value) > 1e-9:
                raise ValueError(f"sky mask manifest summary {key} is inconsistent")
        elif int(summary[key]) != value:
            raise ValueError(f"sky mask manifest summary {key} is inconsistent")
    return expected


def build_sky_mask_manifest(
    *,
    split: str,
    source_face_manifest_sha256: str,
    source_identity: dict[str, Any],
    model: dict[str, Any],
    rule: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble and sign a manifest from per-face records; verifies before returning."""
    ordered = [dict(record) for record in records]
    payload = {
        "schema_version": SKY_MASK_SCHEMA_VERSION,
        "kind": SKY_MASK_KIND,
        "split": str(split),
        "source_face_manifest_sha256": str(source_face_manifest_sha256),
        "source_identity": dict(source_identity),
        "model": dict(model),
        "label_ids": list(SKY_MASK_LABEL_IDS),
        "label_names": list(SKY_MASK_LABEL_NAMES),
        "rule": dict(rule),
        "masks": ordered,
        "summary": summarize_sky_mask_records(ordered),
    }
    signed = sign_sky_mask_manifest(payload)
    verify_sky_mask_manifest(signed)
    return signed


def load_sky_mask_manifest(
    path: Path,
    *,
    expected_face_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Read, verify and (optionally) bind a sky mask manifest to a face cache."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_sky_mask_manifest(manifest)
    if expected_face_manifest_sha256 is not None and (
        manifest["source_face_manifest_sha256"] != expected_face_manifest_sha256
    ):
        raise ValueError("sky mask manifest is bound to a different face cache")
    return manifest


def sky_mask_records_by_key(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(record["image_id"]), str(record["face_id"])): record
        for record in manifest.get("masks", [])
    }


def load_sky_mask(
    root: Path,
    record: dict[str, Any],
    *,
    verify_sha: bool = True,
) -> np.ndarray:
    """Load one face's sky mask as a boolean (H, W) array; fail closed on drift."""
    path = _safe_artifact(Path(root), str(record["mask_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"sky mask is missing: {path}")
    if verify_sha and _sha256_file(path) != str(record["mask_sha256"]):
        raise ValueError(f"sky mask SHA256 mismatch: {path}")
    with Image.open(path) as source:
        mask = np.asarray(source.convert("L"), dtype=np.uint8)
    if mask.shape != (int(record["height"]), int(record["width"])):
        raise ValueError(f"sky mask size does not match its record: {path}")
    sky = mask != 0
    if int(np.count_nonzero(sky)) != int(record["sky_pixels"]):
        raise ValueError(f"sky mask pixel count does not match its record: {path}")
    return sky
