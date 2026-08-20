"""Signed per-image person masks kept separate from geometric valid masks."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import (
    SAFE_IMAGE_ID,
    verify_dataset_manifest,
    verify_mask_manifest,
)


PERSON_MASK_MANIFEST_NAME = "person_mask_manifest.json"
PERSON_MASK_REVIEW_NAME = "person_mask_review.json"


class PersonSegmenter(Protocol):
    def segment(self, image: np.ndarray) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PersonMaskConfig:
    score_threshold: float = 0.65
    mask_threshold: float = 0.5
    dilation_pixels: int = 12
    review_frames_per_camera: int = 12

    def validate(self) -> None:
        if not 0.0 < self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in (0, 1]")
        if not 0.0 < self.mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in (0, 1)")
        if self.dilation_pixels < 0:
            raise ValueError("dilation_pixels must be non-negative")
        if self.review_frames_per_camera <= 0:
            raise ValueError("review_frames_per_camera must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_threshold": self.score_threshold,
            "mask_threshold": self.mask_threshold,
            "dilation_pixels": self.dilation_pixels,
            "review_frames_per_camera": self.review_frames_per_camera,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_source(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"source image paths must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe source image path: {value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"source image escapes recording root: {value!r}")
    return resolved


def _png_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(
        stream, format="PNG", optimize=False, compress_level=9
    )
    return stream.getvalue()


def _write_atomic(path: Path, payload: bytes) -> None:
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


def _validate_model_identity(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "runtime",
        "version",
        "architecture",
        "weights",
        "weights_sha256",
        "person_class_index",
    }
    if set(value) < required:
        raise ValueError(f"person model identity is missing {sorted(required - set(value))}")
    digest = str(value["weights_sha256"])
    if len(digest) != 64:
        raise ValueError("person model weights SHA256 must contain 64 characters")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError("person model weights SHA256 is not hexadecimal") from exc
    return dict(sorted(value.items()))


def _combine_instances(
    instances: list[dict[str, Any]],
    shape: tuple[int, int],
    config: PersonMaskConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    combined = np.zeros(shape, dtype=bool)
    accepted: list[dict[str, Any]] = []
    for instance in instances:
        score = float(instance.get("score", float("nan")))
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("person segmentation score must be finite and in [0, 1]")
        if score < config.score_threshold:
            continue
        raw = np.asarray(instance.get("mask"))
        if raw.shape != shape:
            raise ValueError("person instance mask shape does not match source image")
        mask = raw >= config.mask_threshold if raw.dtype != np.bool_ else raw
        combined |= mask
        box = [float(value) for value in instance.get("box_xyxy", [])]
        if len(box) != 4 or not np.all(np.isfinite(box)):
            raise ValueError("person instance box must contain four finite coordinates")
        accepted.append({"score": score, "box_xyxy": box, "pixels": int(mask.sum())})
    if config.dilation_pixels and np.any(combined):
        combined = distance_transform_edt(~combined) <= config.dilation_pixels
    return combined, accepted


def _review_ids(
    records: list[dict[str, Any]], frames_per_camera: int
) -> list[str]:
    selected: set[str] = set()
    cameras = sorted({str(record["camera_id"]) for record in records})
    for camera_id in cameras:
        camera_records = sorted(
            (
                record
                for record in records
                if str(record["camera_id"]) == camera_id
            ),
            key=lambda record: str(record["source_image_path"]),
        )
        detected = sorted(
            camera_records,
            key=lambda record: (-float(record["person_fraction"]), str(record["image_id"])),
        )
        top_count = frames_per_camera // 2
        selected.update(str(record["image_id"]) for record in detected[:top_count])
        uniform_count = frames_per_camera - top_count
        if uniform_count:
            indexes = np.linspace(
                0, len(camera_records) - 1, min(uniform_count, len(camera_records)), dtype=np.int64
            )
            selected.update(str(camera_records[int(index)]["image_id"]) for index in indexes)
    return sorted(selected)


def _overlay_bytes(image: np.ndarray, person: np.ndarray) -> bytes:
    pixels = np.asarray(image, dtype=np.uint8).copy()
    red = np.zeros_like(pixels)
    red[..., 0] = 255
    pixels[person] = np.rint(0.45 * pixels[person] + 0.55 * red[person]).astype(np.uint8)
    return _png_bytes(pixels)


def verify_person_mask_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("person_mask_manifest_sha256", ""))
    if not expected:
        raise ValueError("person mask manifest has no SHA256")
    unsigned = dict(manifest)
    unsigned.pop("person_mask_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"person mask manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    records = manifest.get("images", [])
    image_ids = [str(record.get("image_id", "")) for record in records]
    paths = [str(record.get("person_mask_path", "")) for record in records]
    if not records or not all(image_ids) or not all(paths):
        raise ValueError("person mask manifest contains incomplete image records")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("person mask manifest contains duplicate image IDs")
    if len(paths) != len(set(paths)):
        raise ValueError("person mask manifest contains shared artifact paths")
    for record in records:
        for key in ("person_mask_path", "source_image_path"):
            value = str(record.get(key, ""))
            pure = PurePosixPath(value)
            if (
                "\\" in value
                or pure.is_absolute()
                or not pure.parts
                or ".." in pure.parts
            ):
                raise ValueError(f"person mask manifest has unsafe {key}: {value!r}")
        for key in ("person_mask_sha256", "source_image_sha256"):
            digest = str(record.get(key, ""))
            if len(digest) != 64:
                raise ValueError(f"person mask manifest has invalid {key}")
            try:
                bytes.fromhex(digest)
            except ValueError as exc:
                raise ValueError(f"person mask manifest has invalid {key}") from exc
    _validate_model_identity(dict(manifest.get("model_identity", {})))
    return actual


def verify_person_mask_review(report: dict[str, Any]) -> str:
    expected = str(report.get("person_mask_review_sha256", ""))
    if not expected:
        raise ValueError("person mask review has no SHA256")
    unsigned = dict(report)
    unsigned.pop("person_mask_review_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"person mask review SHA256 mismatch: expected {expected}, computed {actual}"
        )
    samples = report.get("samples", [])
    if not samples or len({str(item.get("image_id", "")) for item in samples}) != len(
        samples
    ):
        raise ValueError("person mask review samples are empty or duplicated")
    if any(item.get("status") not in {"PASS", "FAIL"} for item in samples):
        raise ValueError("person mask review sample status must be PASS or FAIL")
    return actual


def build_person_mask_review(
    person_manifest: dict[str, Any],
    person_mask_root: Path,
    decisions: dict[str, Any],
) -> dict[str, Any]:
    """Bind explicit visual decisions to every selected review overlay."""
    person_sha = verify_person_mask_manifest(person_manifest)
    expected = {
        str(record["image_id"]): record
        for record in person_manifest.get("review_samples", [])
    }
    provided = {
        str(record.get("image_id", "")): record
        for record in decisions.get("samples", [])
    }
    if not expected or set(provided) != set(expected):
        missing = sorted(set(expected) - set(provided))
        extra = sorted(set(provided) - set(expected))
        raise ValueError(
            "review decisions must cover every selected overlay; "
            f"missing={missing[:4]}, unknown={extra[:4]}"
        )
    reviewer = decisions.get("reviewer")
    if not isinstance(reviewer, dict) or not str(reviewer.get("type", "")) or not str(
        reviewer.get("identifier", "")
    ):
        raise ValueError("person mask review requires explicit reviewer type and identifier")
    samples: list[dict[str, Any]] = []
    root = Path(person_mask_root)
    for image_id, overlay_record in sorted(expected.items()):
        overlay = _safe_source(root, str(overlay_record["overlay_path"]))
        if not overlay.is_file():
            raise FileNotFoundError(f"missing person review overlay: {overlay}")
        overlay_sha = _sha256_file(overlay)
        if overlay_sha != str(overlay_record["overlay_sha256"]):
            raise ValueError(f"person review overlay SHA256 mismatch for {image_id}")
        decision = provided[image_id]
        status = str(decision.get("status", ""))
        if status not in {"PASS", "FAIL"}:
            raise ValueError(f"invalid person review status for {image_id}")
        samples.append(
            {
                "image_id": image_id,
                "overlay_path": str(overlay_record["overlay_path"]),
                "overlay_sha256": overlay_sha,
                "status": status,
                "note": str(decision.get("note", "")),
            }
        )
    failed = [sample for sample in samples if sample["status"] == "FAIL"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "person_mask_visual_review_v1",
        "person_mask_manifest_sha256": person_sha,
        "reviewer": {
            "type": str(reviewer["type"]),
            "identifier": str(reviewer["identifier"]),
        },
        "status": "FAIL" if failed else "PASS",
        "samples": samples,
        "summary": {
            "reviewed": len(samples),
            "passed": len(samples) - len(failed),
            "failed": len(failed),
        },
    }
    report["person_mask_review_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def build_person_masks(
    dataset_manifest: dict[str, Any],
    base_mask_manifest: dict[str, Any],
    recording_root: Path,
    output_dir: Path,
    *,
    segmenter: PersonSegmenter,
    model_identity: dict[str, Any],
    config: PersonMaskConfig = PersonMaskConfig(),
    force: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Generate an independent dynamic-person layer without rewriting base masks."""
    config.validate()
    dataset_sha256 = verify_dataset_manifest(dataset_manifest)
    base_sha256 = verify_mask_manifest(base_mask_manifest)
    if base_mask_manifest.get("dataset_manifest_sha256") != dataset_sha256:
        raise ValueError("base mask manifest is bound to a different dataset")
    identity = _validate_model_identity(model_identity)
    dataset_images = {
        str(record["image_id"]): record for record in dataset_manifest["images"]
    }
    base_ids = {str(record["image_id"]) for record in base_mask_manifest["images"]}
    if set(dataset_images) != base_ids:
        raise ValueError("base mask manifest must cover every dataset image")

    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"person mask output is not a directory: {output_dir}")
    published_manifest = output_dir / PERSON_MASK_MANIFEST_NAME
    if published_manifest.is_file() and force:
        raise FileExistsError(
            "refusing to overwrite a published person-mask manifest; use a fresh output directory"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"person mask output is not empty: {output_dir}; pass --force"
        )
    (output_dir / "person_masks").mkdir(parents=True, exist_ok=True)
    recording_root = Path(recording_root)
    records: list[dict[str, Any]] = []
    source_paths: dict[str, Path] = {}
    ordered_images = sorted(
        dataset_images.items(), key=lambda item: str(item[1]["path"])
    )
    for image_index, (image_id, image_record) in enumerate(ordered_images, start=1):
        if not SAFE_IMAGE_ID.fullmatch(image_id):
            raise ValueError(f"unsafe image ID for person mask path: {image_id!r}")
        if image_record.get("path_root") != "recording":
            raise ValueError(f"unsupported image path root for {image_id}")
        source = _safe_source(recording_root, str(image_record["path"]))
        if not source.is_file():
            raise FileNotFoundError(f"missing source image: {source}")
        actual_source_sha = _sha256_file(source)
        if actual_source_sha != str(image_record.get("sha256", "")):
            raise ValueError(f"source image SHA256 mismatch for {image_id}")
        with Image.open(source) as opened:
            pixels = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        person, instances = _combine_instances(segmenter.segment(pixels), pixels.shape[:2], config)
        relative = f"person_masks/{image_id}.png"
        payload = _png_bytes(person.astype(np.uint8) * 255)
        _write_atomic(output_dir / Path(relative), payload)
        source_paths[image_id] = source
        scores = [float(instance["score"]) for instance in instances]
        records.append(
            {
                "image_id": image_id,
                "camera_id": str(image_record["camera_id"]),
                "source_image_path_root": "recording",
                "source_image_path": str(image_record["path"]),
                "source_image_sha256": actual_source_sha,
                "person_mask_path": relative,
                "person_mask_sha256": hashlib.sha256(payload).hexdigest(),
                "person_instances": len(instances),
                "person_pixels": int(person.sum()),
                "person_fraction": float(person.mean()),
                "score_min": None if not scores else min(scores),
                "score_max": None if not scores else max(scores),
                "instances": instances,
            }
        )
        if progress is not None:
            progress(image_index, len(ordered_images), image_id)

    records.sort(key=lambda record: str(record["image_id"]))
    review_ids = _review_ids(records, config.review_frames_per_camera)
    records_by_id = {str(record["image_id"]): record for record in records}
    review: list[dict[str, Any]] = []
    for image_id in review_ids:
        with Image.open(source_paths[image_id]) as opened:
            pixels = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        mask_record = records_by_id[image_id]
        with Image.open(output_dir / str(mask_record["person_mask_path"])) as opened:
            person = np.asarray(opened, dtype=np.uint8) > 0
        payload = _overlay_bytes(pixels, person)
        relative = f"review/{image_id}.png"
        _write_atomic(output_dir / Path(relative), payload)
        review.append(
            {
                "image_id": image_id,
                "overlay_path": relative,
                "overlay_sha256": hashlib.sha256(payload).hexdigest(),
                "review_status": "PENDING_MANUAL_REVIEW",
            }
        )

    fractions = [float(record["person_fraction"]) for record in records]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "independent_person_dynamic_mask_v1",
        "dataset_manifest_sha256": dataset_sha256,
        "base_mask_manifest_sha256": base_sha256,
        "composition": "base_rgb_mask & ~person_dynamic_mask",
        "depth_composition": "base_depth_valid & ~person_dynamic_mask",
        "model_identity": identity,
        "config": config.to_dict(),
        "path_root": "person_mask_output",
        "images": records,
        "review_samples": review,
        "summary": {
            "image_count": len(records),
            "images_with_person": sum(
                int(record["person_instances"] > 0) for record in records
            ),
            "person_instances": sum(int(record["person_instances"]) for record in records),
            "person_fraction_min": min(fractions),
            "person_fraction_p50": float(np.percentile(fractions, 50)),
            "person_fraction_p95": float(np.percentile(fractions, 95)),
            "person_fraction_max": max(fractions),
            "manual_review": "PENDING",
        },
    }
    manifest["person_mask_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_atomic(
        output_dir / PERSON_MASK_MANIFEST_NAME,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return manifest
