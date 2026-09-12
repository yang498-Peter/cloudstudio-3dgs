"""Signed precomputed Tile-ownership mask cache for Face4 Tile training.

``FaceCacheDataset`` derives, per cropped Tile view, the pair
``(owned, foreign_region)`` through
:func:`cloudstudio_3dgs.training.face_dataset.tile_ownership_masks`: LiDAR
returns are unprojected against the Tile's ``training_and_export_box`` grown
by ``margin_m``, and the dilated (``dilation_px``) neighbourhood of foreign
returns that is not also the dilated neighbourhood of owned returns leaves
the photometric / DA2 masks while the foreign returns leave the range mask.
Computed on the fly that is a full-resolution dilation on the CPU for every
sample every step (3.2x slower training at DIAG scale).

This module stores the pair per sample as packed bits inside one ``.npz``
plus a signed manifest bound to every input the on-the-fly path reads: the
Face4 cache, the renderer mask manifest (or none: the face cache mask is the
supervision mask then), the Face4 LiDAR geometry manifest (or none: the face
cache carries the depth), the Tile inputs manifest / tile id / box, the
margin and the dilation. Any of those changing invalidates the cache. The
consumer (``FaceCacheDataset``) fails closed on a missing record or artifact
and verifies the artifact SHA and the recorded pixel counts on load.

No ML dependency; layout mirrors :mod:`cloudstudio_3dgs.data.sky_masks`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


TILE_OWNERSHIP_SCHEMA_VERSION = 1
TILE_OWNERSHIP_KIND = "face4_tile_ownership_mask_cache"
TILE_OWNERSHIP_MANIFEST_SHA_KEY = "tile_ownership_manifest_sha256"
TILE_OWNERSHIP_FUNCTION = "cloudstudio_3dgs.training.face_dataset.tile_ownership_masks"
# Row-major bit packing of the boolean (H, W) masks; ``count=H*W`` on unpack.
TILE_OWNERSHIP_ENCODING = "npz_packbits_row_major_v1"

_CROP_KEYS = ("x", "y", "width", "height")
_RECORD_KEYS = (
    "sample_id",
    "image_id",
    "camera_id",
    "face_id",
    "crop",
    "width",
    "height",
    "path",
    "sha256",
    "ownership_applied",
    "crop_pixels",
    "rgb_mask_pixels",
    "depth_mask_pixels",
    "owned_pixels",
    "foreign_pixels",
    "foreign_region_pixels",
    "rgb_dropped_pixels",
    "rgb_kept_pixels",
    "rgb_mask_preserved",
    "lidar_owned_fraction",
    "rgb_dropped_fraction",
)
_REQUIRED_RULE_KEYS = ("function", "margin_m", "dilation_px", "encoding")


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


def _is_optional_sha256(value: Any) -> bool:
    return value is None or _is_sha256(value)


def tile_ownership_mask_path_for(image_id: str, face_id: str) -> str:
    """Relative artifact path (forward slashes) for one sample's mask pair."""
    return f"masks/{image_id}_{face_id}.npz"


def tile_ownership_rule(margin_m: float, dilation_px: int) -> dict[str, Any]:
    """The decision rule the cache was built with; bound into the manifest."""
    return {
        "function": TILE_OWNERSHIP_FUNCTION,
        "margin_m": float(margin_m),
        "dilation_px": int(dilation_px),
        "encoding": TILE_OWNERSHIP_ENCODING,
        "inputs": (
            "depth_range_m / depth_mask (= supervision mask AND LiDAR valid), "
            "K and c2w of the cropped face exactly as FaceCacheDataset.__getitem__ "
            "builds them (crop applied, principal point shifted)"
        ),
        "owned": "depth_mask AND unprojected point inside training_and_export_box grown by margin_m",
        "foreign_region": (
            "dilate(depth_mask AND NOT inside, dilation_px) AND NOT dilate(owned, dilation_px)"
        ),
        "applied_when": "depth present AND depth_mask has at least one return (else no record artifact)",
        "consumer": (
            "rgb_mask &= ~foreign_region unless that empties it (then the photometric "
            "mask is preserved); mono depth mask likewise; depth_mask = owned"
        ),
    }


def sign_tile_ownership_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop(TILE_OWNERSHIP_MANIFEST_SHA_KEY, None)
    signed = dict(unsigned)
    signed[TILE_OWNERSHIP_MANIFEST_SHA_KEY] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def ownership_record_counts(
    rgb_mask: np.ndarray,
    depth_mask: np.ndarray | None,
    owned: np.ndarray | None,
    foreign_region: np.ndarray | None,
) -> dict[str, Any]:
    """Pixel accounting of one sample, in the consumer's own terms.

    ``rgb_dropped_pixels`` is the number of supervision pixels the dataset
    actually removes: zero when the ownership pass would empty the mask, in
    which case the dataset keeps the photometric mask
    (``rgb_mask_preserved``); the foreign returns still leave the range mask.
    """
    rgb_mask = np.asarray(rgb_mask, dtype=bool)
    crop_pixels = int(rgb_mask.size)
    rgb_mask_pixels = int(np.count_nonzero(rgb_mask))
    if depth_mask is None or owned is None or foreign_region is None:
        return {
            "ownership_applied": False,
            "crop_pixels": crop_pixels,
            "rgb_mask_pixels": rgb_mask_pixels,
            "depth_mask_pixels": 0
            if depth_mask is None
            else int(np.count_nonzero(np.asarray(depth_mask, dtype=bool))),
            "owned_pixels": 0,
            "foreign_pixels": 0,
            "foreign_region_pixels": 0,
            "rgb_dropped_pixels": 0,
            "rgb_kept_pixels": rgb_mask_pixels,
            "rgb_mask_preserved": False,
            "lidar_owned_fraction": 0.0,
            "rgb_dropped_fraction": 0.0,
        }
    depth_mask = np.asarray(depth_mask, dtype=bool)
    owned = np.asarray(owned, dtype=bool)
    foreign_region = np.asarray(foreign_region, dtype=bool)
    if depth_mask.shape != rgb_mask.shape or owned.shape != rgb_mask.shape or foreign_region.shape != rgb_mask.shape:
        raise ValueError("ownership masks and the supervision mask differ in shape")
    depth_mask_pixels = int(np.count_nonzero(depth_mask))
    owned_pixels = int(np.count_nonzero(owned))
    masked_rgb = rgb_mask & ~foreign_region
    kept = int(np.count_nonzero(masked_rgb))
    preserved = kept == 0
    dropped = 0 if preserved else rgb_mask_pixels - kept
    kept = rgb_mask_pixels if preserved else kept
    return {
        "ownership_applied": True,
        "crop_pixels": crop_pixels,
        "rgb_mask_pixels": rgb_mask_pixels,
        "depth_mask_pixels": depth_mask_pixels,
        "owned_pixels": owned_pixels,
        "foreign_pixels": depth_mask_pixels - owned_pixels,
        "foreign_region_pixels": int(np.count_nonzero(foreign_region)),
        "rgb_dropped_pixels": dropped,
        "rgb_kept_pixels": kept,
        "rgb_mask_preserved": preserved,
        "lidar_owned_fraction": (owned_pixels / depth_mask_pixels) if depth_mask_pixels else 0.0,
        "rgb_dropped_fraction": (dropped / rgb_mask_pixels) if rgb_mask_pixels else 0.0,
    }


def pack_ownership_pair(owned: np.ndarray, foreign_region: np.ndarray) -> dict[str, np.ndarray]:
    owned = np.asarray(owned, dtype=bool)
    foreign_region = np.asarray(foreign_region, dtype=bool)
    if owned.ndim != 2 or owned.shape != foreign_region.shape:
        raise ValueError("ownership pair must be two boolean (H, W) arrays of one shape")
    return {
        "shape": np.asarray(owned.shape, dtype=np.int64),
        "owned_packed": np.packbits(owned.ravel(order="C")),
        "foreign_region_packed": np.packbits(foreign_region.ravel(order="C")),
    }


def write_ownership_pair(path: Path, owned: np.ndarray, foreign_region: np.ndarray) -> str:
    """Atomically write the packed pair; returns the file SHA256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **pack_ownership_pair(owned, foreign_region))
    os.replace(temporary, path)
    return _sha256_file(path)


def unpack_ownership_pair(payload: Any, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    required = {"shape", "owned_packed", "foreign_region_packed"}
    if not required.issubset(set(payload.files)):
        raise ValueError("tile ownership cache is missing required arrays")
    shape = tuple(int(v) for v in np.asarray(payload["shape"]).tolist())
    if len(shape) != 2 or shape != tuple(int(v) for v in expected_shape):
        raise ValueError(
            f"tile ownership cache shape {shape} differs from the expected {tuple(expected_shape)}"
        )
    count = shape[0] * shape[1]
    owned = np.unpackbits(np.asarray(payload["owned_packed"], dtype=np.uint8), count=count)
    foreign = np.unpackbits(np.asarray(payload["foreign_region_packed"], dtype=np.uint8), count=count)
    return (
        np.ascontiguousarray(owned.reshape(shape).astype(bool)),
        np.ascontiguousarray(foreign.reshape(shape).astype(bool)),
    )


def load_ownership_pair(
    root: Path,
    record: dict[str, Any],
    *,
    verify_sha: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one record's (owned, foreign_region); fail closed on any drift."""
    if not record.get("path"):
        raise ValueError(
            f"tile ownership record {record.get('sample_id')!r} carries no artifact"
        )
    path = _safe_artifact(Path(root), str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"tile ownership mask is missing: {path}")
    if verify_sha and _sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"tile ownership mask SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        owned, foreign_region = unpack_ownership_pair(
            payload, (int(record["height"]), int(record["width"]))
        )
    if int(np.count_nonzero(owned)) != int(record["owned_pixels"]) or int(
        np.count_nonzero(foreign_region)
    ) != int(record["foreign_region_pixels"]):
        raise ValueError(f"tile ownership mask pixel counts differ from the record: {path}")
    return owned, foreign_region


def summarize_tile_ownership_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    applied = [r for r in records if bool(r["ownership_applied"])]
    owned_fractions = [float(r["lidar_owned_fraction"]) for r in applied]
    dropped_fractions = [float(r["rgb_dropped_fraction"]) for r in applied]
    total_rgb = int(sum(int(r["rgb_mask_pixels"]) for r in records))
    total_dropped = int(sum(int(r["rgb_dropped_pixels"]) for r in records))
    total_depth = int(sum(int(r["depth_mask_pixels"]) for r in records))
    total_owned = int(sum(int(r["owned_pixels"]) for r in records))
    return {
        "sample_count": len(records),
        "image_count": len({str(r["image_id"]) for r in records}),
        "samples_with_ownership": len(applied),
        "samples_without_lidar_support": len(records) - len(applied),
        "samples_with_rgb_mask_preserved": int(sum(1 for r in applied if bool(r["rgb_mask_preserved"]))),
        "total_crop_pixels": int(sum(int(r["crop_pixels"]) for r in records)),
        "total_rgb_mask_pixels": total_rgb,
        "total_rgb_dropped_pixels": total_dropped,
        "total_depth_mask_pixels": total_depth,
        "total_owned_pixels": total_owned,
        "pooled_rgb_dropped_fraction": (total_dropped / total_rgb) if total_rgb else 0.0,
        "pooled_lidar_owned_fraction": (total_owned / total_depth) if total_depth else 0.0,
        "median_rgb_dropped_fraction": float(np.median(dropped_fractions)) if dropped_fractions else 0.0,
        "median_lidar_owned_fraction": float(np.median(owned_fractions)) if owned_fractions else 0.0,
    }


def _validate_crop(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != set(_CROP_KEYS):
        raise ValueError("tile ownership record crop must be null or {x, y, width, height}")
    if any(int(value[k]) < 0 for k in _CROP_KEYS) or int(value["width"]) <= 0 or int(value["height"]) <= 0:
        raise ValueError("tile ownership record crop is invalid")


def _validate_record(record: dict[str, Any]) -> None:
    missing = [key for key in _RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f"tile ownership record is missing {missing}")
    for key in ("sample_id", "image_id", "camera_id", "face_id"):
        if not str(record[key]):
            raise ValueError(f"tile ownership record has an empty {key}")
    if str(record["sample_id"]) != f"{record['image_id']}::{record['face_id']}":
        raise ValueError("tile ownership record sample_id is not image_id::face_id")
    _validate_crop(record["crop"])
    width, height = int(record["width"]), int(record["height"])
    if width <= 0 or height <= 0:
        raise ValueError("tile ownership record has a non-positive size")
    if record["crop"] is not None and (
        int(record["crop"]["width"]) != width or int(record["crop"]["height"]) != height
    ):
        raise ValueError("tile ownership record size differs from its crop")
    crop_pixels = int(record["crop_pixels"])
    rgb = int(record["rgb_mask_pixels"])
    depth = int(record["depth_mask_pixels"])
    owned = int(record["owned_pixels"])
    foreign = int(record["foreign_pixels"])
    region = int(record["foreign_region_pixels"])
    dropped = int(record["rgb_dropped_pixels"])
    kept = int(record["rgb_kept_pixels"])
    if crop_pixels != width * height:
        raise ValueError("tile ownership record crop_pixels differs from width*height")
    if not 0 <= owned <= depth <= rgb <= crop_pixels or foreign != depth - owned:
        raise ValueError("tile ownership record LiDAR pixel counts are inconsistent")
    if not 0 <= region <= crop_pixels or not 0 <= dropped <= rgb or dropped + kept != rgb:
        raise ValueError("tile ownership record supervision pixel counts are inconsistent")
    applied = bool(record["ownership_applied"])
    preserved = bool(record["rgb_mask_preserved"])
    if applied != (depth > 0):
        raise ValueError("tile ownership record ownership_applied disagrees with depth_mask_pixels")
    if not applied and (owned or region or dropped or preserved):
        raise ValueError("tile ownership record without LiDAR support carries ownership counts")
    if preserved and dropped != 0:
        raise ValueError("tile ownership record preserved mask cannot drop pixels")
    expected_owned = (owned / depth) if depth else 0.0
    expected_dropped = (dropped / rgb) if rgb else 0.0
    if abs(float(record["lidar_owned_fraction"]) - expected_owned) > 1e-9:
        raise ValueError("tile ownership record lidar_owned_fraction does not match its counts")
    if abs(float(record["rgb_dropped_fraction"]) - expected_dropped) > 1e-9:
        raise ValueError("tile ownership record rgb_dropped_fraction does not match its counts")
    if applied:
        if not record["path"] or not _is_sha256(record["sha256"]):
            raise ValueError("tile ownership record with LiDAR support lacks a signed artifact")
        _safe_artifact(Path("."), str(record["path"]))
    elif record["path"] is not None or record["sha256"] is not None:
        raise ValueError("tile ownership record without LiDAR support carries an artifact")


def verify_tile_ownership_manifest(manifest: dict[str, Any]) -> str:
    """Fail-closed structural + signature check; returns the manifest SHA256."""
    expected = str(manifest.get(TILE_OWNERSHIP_MANIFEST_SHA_KEY, ""))
    if not _is_sha256(expected):
        raise ValueError("tile ownership manifest is unsigned")
    unsigned = dict(manifest)
    unsigned.pop(TILE_OWNERSHIP_MANIFEST_SHA_KEY, None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("tile ownership manifest signature mismatch")
    if int(manifest.get("schema_version", -1)) != TILE_OWNERSHIP_SCHEMA_VERSION:
        raise ValueError("unsupported tile ownership manifest schema")
    if manifest.get("kind") != TILE_OWNERSHIP_KIND:
        raise ValueError("unexpected tile ownership manifest kind")
    if not str(manifest.get("split", "")):
        raise ValueError("tile ownership manifest has no split")
    if not _is_sha256(manifest.get("source_face_manifest_sha256", "")):
        raise ValueError("tile ownership manifest is not bound to a face manifest")
    for key in ("renderer_mask_manifest_sha256", "face_lidar_geometry_manifest_sha256"):
        if key not in manifest or not _is_optional_sha256(manifest[key]):
            raise ValueError(f"tile ownership manifest {key} must be null or a SHA256")
    if not _is_sha256(manifest.get("tile_inputs_manifest_sha256", "")):
        raise ValueError("tile ownership manifest is not bound to Tile inputs")
    tile_id = manifest.get("tile_id")
    if isinstance(tile_id, bool) or not isinstance(tile_id, int) or tile_id < 0:
        raise ValueError("tile ownership manifest tile_id must be a non-negative integer")
    box = np.asarray(manifest.get("training_and_export_box"), dtype=np.float64)
    if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[0] > box[1]):
        raise ValueError("tile ownership manifest box must be a finite [[min xyz], [max xyz]]")
    rule = manifest.get("rule", {})
    if not isinstance(rule, dict) or any(k not in rule for k in _REQUIRED_RULE_KEYS):
        raise ValueError("tile ownership manifest rule is incomplete")
    if rule["function"] != TILE_OWNERSHIP_FUNCTION or rule["encoding"] != TILE_OWNERSHIP_ENCODING:
        raise ValueError("tile ownership manifest rule names an unsupported function or encoding")
    if float(rule["margin_m"]) < 0.0 or not np.isfinite(float(rule["margin_m"])):
        raise ValueError("tile ownership manifest margin_m must be finite and non-negative")
    if isinstance(rule["dilation_px"], bool) or int(rule["dilation_px"]) < 0:
        raise ValueError("tile ownership manifest dilation_px must be a non-negative integer")
    records = manifest.get("records", [])
    sample_ids = [str(record.get("sample_id", "")) for record in records]
    if not records or len(sample_ids) != len(set(sample_ids)) or not all(sample_ids):
        raise ValueError("tile ownership manifest has invalid or duplicate records")
    for record in records:
        _validate_record(record)
    summary = manifest.get("summary", {})
    expected_summary = summarize_tile_ownership_records(records)
    for key, value in expected_summary.items():
        if key not in summary:
            raise ValueError(f"tile ownership manifest summary is missing {key}")
        if isinstance(value, float):
            if abs(float(summary[key]) - value) > 1e-9:
                raise ValueError(f"tile ownership manifest summary {key} is inconsistent")
        elif int(summary[key]) != value:
            raise ValueError(f"tile ownership manifest summary {key} is inconsistent")
    return expected


def build_tile_ownership_manifest(
    *,
    split: str,
    source_face_manifest_sha256: str,
    renderer_mask_manifest_sha256: str | None,
    face_lidar_geometry_manifest_sha256: str | None,
    tile_inputs_manifest_sha256: str,
    tile_id: int,
    training_and_export_box: Any,
    margin_m: float,
    dilation_px: int,
    source_identity: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble and sign a manifest from per-sample records; verifies before returning."""
    ordered = [dict(record) for record in records]
    box = np.asarray(training_and_export_box, dtype=np.float64)
    payload = {
        "schema_version": TILE_OWNERSHIP_SCHEMA_VERSION,
        "kind": TILE_OWNERSHIP_KIND,
        "split": str(split),
        "source_face_manifest_sha256": str(source_face_manifest_sha256),
        "renderer_mask_manifest_sha256": renderer_mask_manifest_sha256,
        "face_lidar_geometry_manifest_sha256": face_lidar_geometry_manifest_sha256,
        "tile_inputs_manifest_sha256": str(tile_inputs_manifest_sha256),
        "tile_id": int(tile_id),
        "training_and_export_box": box.tolist(),
        "rule": tile_ownership_rule(margin_m, dilation_px),
        "source_identity": dict(source_identity),
        "records": ordered,
        "summary": summarize_tile_ownership_records(ordered),
    }
    signed = sign_tile_ownership_manifest(payload)
    verify_tile_ownership_manifest(signed)
    return signed


def load_tile_ownership_manifest(
    path: Path,
    *,
    expected_face_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Read, verify and (optionally) bind a tile ownership manifest to a face cache."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_tile_ownership_manifest(manifest)
    if expected_face_manifest_sha256 is not None and (
        manifest["source_face_manifest_sha256"] != expected_face_manifest_sha256
    ):
        raise ValueError("tile ownership manifest is bound to a different face cache")
    return manifest


def tile_ownership_records_by_sample(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["sample_id"]): record for record in manifest.get("records", [])}
