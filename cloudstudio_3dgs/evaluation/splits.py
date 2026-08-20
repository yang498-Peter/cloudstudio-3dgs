"""Deterministic Rig-frame train/validation split contracts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "temporal_block"
    validation_fraction: float = 0.1
    seed: int = 0
    temporal_block_count: int = 10
    spatial_cell_m: float = 2.0
    nearest_train_warning_m: float = 0.25
    golden_rig_frames: int = 8

    def validate(self) -> None:
        if self.mode not in {"temporal_block", "spatial_block", "manual"}:
            raise ValueError("split mode must be temporal_block, spatial_block, or manual")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.temporal_block_count < 2:
            raise ValueError("temporal_block_count must be at least two")
        if self.spatial_cell_m <= 0.0:
            raise ValueError("spatial_cell_m must be positive")
        if self.nearest_train_warning_m < 0.0:
            raise ValueError("nearest_train_warning_m must be non-negative")
        if self.golden_rig_frames <= 0:
            raise ValueError("golden_rig_frames must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "validation_fraction": self.validation_fraction,
            "seed": self.seed,
            "temporal_block_count": self.temporal_block_count,
            "spatial_cell_m": self.spatial_cell_m,
            "nearest_train_warning_m": self.nearest_train_warning_m,
            "golden_rig_frames": self.golden_rig_frames,
        }


def _distribution(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _rig_records(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    images = {str(image["image_id"]): image for image in dataset["images"]}
    records: list[dict[str, Any]] = []
    covered: set[str] = set()
    for frame in sorted(dataset["rig_frames"], key=lambda item: int(item["timestamp_ns"])):
        image_ids = [str(value) for value in frame["image_ids"]]
        if len(image_ids) != 2 or set(image_ids) != {
            str(frame["left_image_id"]),
            str(frame["right_image_id"]),
        }:
            raise ValueError(f"rig frame {frame['rig_frame_id']} is not a complete stereo pair")
        if any(image_id not in images for image_id in image_ids):
            raise ValueError(f"rig frame {frame['rig_frame_id']} references an unknown image")
        if covered.intersection(image_ids):
            raise ValueError("an image appears in more than one rig frame")
        covered.update(image_ids)
        left = images[str(frame["left_image_id"])]
        right = images[str(frame["right_image_id"])]
        position = 0.5 * (
            np.asarray(left["c2w"], dtype=np.float64)[:3, 3]
            + np.asarray(right["c2w"], dtype=np.float64)[:3, 3]
        )
        records.append(
            {
                "rig_frame_id": str(frame["rig_frame_id"]),
                "timestamp_ns": int(frame["timestamp_ns"]),
                "image_ids": sorted(image_ids),
                "left_image_id": str(frame["left_image_id"]),
                "right_image_id": str(frame["right_image_id"]),
                "position_m": position.tolist(),
            }
        )
    if covered != set(images):
        missing = sorted(set(images) - covered)
        raise ValueError(
            f"formal split requires every image in a complete rig frame; unpaired images: {missing[:4]}"
        )
    if len(records) < 2:
        raise ValueError("at least two rig frames are required for a train/validation split")
    return records


def _temporal_assignment(records: list[dict[str, Any]], config: SplitConfig) -> set[str]:
    count = len(records)
    block_count = min(config.temporal_block_count, count)
    blocks = np.array_split(np.arange(count), block_count)
    target = max(1, round(count * config.validation_fraction))
    ordered_blocks = [
        (config.seed + offset) % block_count for offset in range(block_count)
    ]
    validation: list[int] = []
    for block_index in ordered_blocks:
        if len(validation) >= target:
            break
        validation.extend(int(value) for value in blocks[block_index])
    # Keep the selected blocks intact; overshooting is preferable to cutting a
    # time block and creating a false independent-view boundary.
    return {records[index]["rig_frame_id"] for index in sorted(set(validation))}


def _spatial_assignment(records: list[dict[str, Any]], config: SplitConfig) -> set[str]:
    cells: dict[tuple[int, int, int], list[str]] = {}
    for record in records:
        position = np.asarray(record["position_m"], dtype=np.float64)
        cell = tuple(np.floor(position / config.spatial_cell_m).astype(np.int64).tolist())
        cells.setdefault(cell, []).append(record["rig_frame_id"])
    ranked = sorted(
        cells,
        key=lambda cell: hashlib.sha256(
            f"{config.seed}:{cell[0]}:{cell[1]}:{cell[2]}".encode("ascii")
        ).hexdigest(),
    )
    target = max(1, round(len(records) * config.validation_fraction))
    validation: set[str] = set()
    for cell in ranked:
        if len(validation) >= target:
            break
        validation.update(cells[cell])
    return validation


def _manual_assignment(
    records: list[dict[str, Any]], manual: Mapping[str, str] | None
) -> set[str]:
    if manual is None:
        raise ValueError("manual split mode requires an assignment mapping")
    expected = {record["rig_frame_id"] for record in records}
    if set(manual) != expected:
        missing = sorted(expected - set(manual))
        unknown = sorted(set(manual) - expected)
        raise ValueError(f"manual split IDs differ; missing={missing[:4]}, unknown={unknown[:4]}")
    invalid = sorted({value for value in manual.values() if value not in {"train", "val"}})
    if invalid:
        raise ValueError(f"manual split contains invalid labels: {invalid}")
    return {frame_id for frame_id, split in manual.items() if split == "val"}


def _golden_frames(validation: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    chosen_count = min(count, len(validation))
    indexes = np.linspace(0, len(validation) - 1, chosen_count, dtype=int)
    return [
        {
            "rig_frame_id": validation[int(index)]["rig_frame_id"],
            "image_ids": validation[int(index)]["image_ids"],
        }
        for index in indexes
    ]


def build_split_manifest(
    dataset: dict[str, Any],
    config: SplitConfig = SplitConfig(),
    *,
    manual: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    dataset_sha = verify_dataset_manifest(dataset)
    config.validate()
    records = _rig_records(dataset)
    if config.mode == "temporal_block":
        validation_ids = _temporal_assignment(records, config)
    elif config.mode == "spatial_block":
        validation_ids = _spatial_assignment(records, config)
    else:
        validation_ids = _manual_assignment(records, manual)
    if not validation_ids or len(validation_ids) == len(records):
        raise ValueError("split must contain at least one train and one validation rig frame")

    output_records: list[dict[str, Any]] = []
    train_positions: list[list[float]] = []
    validation: list[dict[str, Any]] = []
    for record in records:
        split = "val" if record["rig_frame_id"] in validation_ids else "train"
        output = {**record, "split": split}
        output_records.append(output)
        if split == "train":
            train_positions.append(record["position_m"])
        else:
            validation.append(output)

    tree = cKDTree(np.asarray(train_positions, dtype=np.float64))
    validation_positions = np.asarray(
        [record["position_m"] for record in validation], dtype=np.float64
    )
    nearest_distance, _ = tree.query(validation_positions, k=1, workers=1)
    leakage_warnings = [
        {
            "rig_frame_id": record["rig_frame_id"],
            "nearest_train_distance_m": float(distance),
            "threshold_m": config.nearest_train_warning_m,
        }
        for record, distance in zip(validation, nearest_distance)
        if distance < config.nearest_train_warning_m
    ]
    train_images = sorted(
        image_id
        for record in output_records
        if record["split"] == "train"
        for image_id in record["image_ids"]
    )
    validation_images = sorted(
        image_id
        for record in output_records
        if record["split"] == "val"
        for image_id in record["image_ids"]
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "rig_frame_split_v1",
        "dataset_manifest_sha256": dataset_sha,
        "configuration": config.to_dict(),
        "rig_frames": output_records,
        "splits": {"train": train_images, "val": validation_images},
        "golden_views": _golden_frames(validation, config.golden_rig_frames),
        "leakage": {
            "nearest_train_distance_m": _distribution(nearest_distance),
            "warning_count": len(leakage_warnings),
            "warnings": leakage_warnings,
        },
        "summary": {
            "rig_frame_count": len(records),
            "train_rig_frames": len(records) - len(validation),
            "val_rig_frames": len(validation),
            "train_images": len(train_images),
            "val_images": len(validation_images),
            "paired_images_same_split": True,
        },
    }
    manifest["split_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


def verify_split_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("split_manifest_sha256", ""))
    if not expected:
        raise ValueError("split manifest has no split_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("split_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"split manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    train_images = {str(value) for value in manifest["splits"]["train"]}
    validation_images = {str(value) for value in manifest["splits"]["val"]}
    if train_images & validation_images:
        raise ValueError("train and validation image lists overlap")
    seen_images: set[str] = set()
    for frame in manifest.get("rig_frames", []):
        if frame.get("split") not in {"train", "val"} or len(frame.get("image_ids", [])) != 2:
            raise ValueError("split manifest contains an invalid rig frame")
        for image_id in frame["image_ids"]:
            if image_id in seen_images:
                raise ValueError("split manifest assigns an image more than once")
            seen_images.add(image_id)
            if image_id not in manifest["splits"][frame["split"]]:
                raise ValueError("rig-frame and image split assignments differ")
    if seen_images != train_images | validation_images:
        raise ValueError("split manifest image lists are inconsistent")
    golden_images = {
        str(image_id)
        for item in manifest.get("golden_views", [])
        for image_id in item.get("image_ids", [])
    }
    if not golden_images or not golden_images <= validation_images:
        raise ValueError("golden views must be a non-empty subset of validation images")
    return actual


def write_split_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
