"""Deterministic training-only stereo, temporal, and spatial loop match graph."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest
from cloudstudio_3dgs.evaluation.splits import verify_split_manifest


@dataclass(frozen=True)
class MatchGraphConfig:
    temporal_neighbor_rig_frames: int = 4
    loop_max_distance_m: float = 1.5
    loop_min_frame_gap: int = 30
    loop_neighbors_per_rig: int = 4

    def validate(self) -> None:
        if self.temporal_neighbor_rig_frames <= 0:
            raise ValueError("temporal_neighbor_rig_frames must be positive")
        if not np.isfinite(self.loop_max_distance_m) or self.loop_max_distance_m <= 0:
            raise ValueError("loop_max_distance_m must be finite and positive")
        if self.loop_min_frame_gap <= self.temporal_neighbor_rig_frames:
            raise ValueError("loop_min_frame_gap must exceed the temporal window")
        if self.loop_neighbors_per_rig <= 0:
            raise ValueError("loop_neighbors_per_rig must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "temporal_neighbor_rig_frames": self.temporal_neighbor_rig_frames,
            "loop_max_distance_m": self.loop_max_distance_m,
            "loop_min_frame_gap": self.loop_min_frame_gap,
            "loop_neighbors_per_rig": self.loop_neighbors_per_rig,
        }


def _rig_position(frame: dict[str, Any], images: dict[str, dict[str, Any]]) -> np.ndarray:
    left = np.asarray(images[str(frame["left_image_id"])]["c2w"], dtype=np.float64)
    right = np.asarray(images[str(frame["right_image_id"])]["c2w"], dtype=np.float64)
    return 0.5 * (left[:3, 3] + right[:3, 3])


def _add_pair(
    pairs: dict[tuple[str, str], set[str]],
    first: str,
    second: str,
    pair_type: str,
) -> None:
    if first == second:
        raise ValueError("match graph cannot contain a self-pair")
    key = tuple(sorted((first, second)))
    pairs.setdefault(key, set()).add(pair_type)


def build_match_graph(
    dataset: dict[str, Any],
    split_manifest: dict[str, Any],
    config: MatchGraphConfig = MatchGraphConfig(),
) -> dict[str, Any]:
    dataset_sha = verify_dataset_manifest(dataset)
    split_sha = verify_split_manifest(split_manifest)
    if split_manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("dataset and split manifests have different identities")
    config.validate()
    images = {str(image["image_id"]): image for image in dataset["images"]}
    validation_images = {str(value) for value in split_manifest["splits"]["val"]}
    training_images = {str(value) for value in split_manifest["splits"]["train"]}
    frames = sorted(dataset["rig_frames"], key=lambda item: int(item["timestamp_ns"]))
    indexed_frames: list[tuple[int, dict[str, Any]]] = []
    for index, frame in enumerate(frames):
        image_ids = {str(value) for value in frame["image_ids"]}
        if len(image_ids) != 2 or not image_ids <= set(images):
            raise ValueError(f"Rig Frame {frame['rig_frame_id']} is not a complete pair")
        if image_ids <= training_images:
            indexed_frames.append((index, frame))
        elif not image_ids <= validation_images:
            raise ValueError("split manifest divides or omits a Rig Frame")
    if len(indexed_frames) < 2:
        raise ValueError("match graph requires at least two training Rig Frames")

    pairs: dict[tuple[str, str], set[str]] = {}
    for _index, frame in indexed_frames:
        _add_pair(
            pairs,
            str(frame["left_image_id"]),
            str(frame["right_image_id"]),
            "stereo",
        )
    for left_position, (left_index, left_frame) in enumerate(indexed_frames):
        for right_index, right_frame in indexed_frames[left_position + 1 :]:
            gap = right_index - left_index
            if gap > config.temporal_neighbor_rig_frames:
                break
            _add_pair(
                pairs,
                str(left_frame["left_image_id"]),
                str(right_frame["left_image_id"]),
                "temporal_left",
            )
            _add_pair(
                pairs,
                str(left_frame["right_image_id"]),
                str(right_frame["right_image_id"]),
                "temporal_right",
            )

    positions = np.stack([_rig_position(frame, images) for _index, frame in indexed_frames])
    tree = cKDTree(positions)
    for local_index, (frame_index, frame) in enumerate(indexed_frames):
        candidates = tree.query_ball_point(
            positions[local_index], config.loop_max_distance_m, workers=1
        )
        ranked = sorted(
            (
                (
                    float(np.linalg.norm(positions[other] - positions[local_index])),
                    indexed_frames[other][0],
                    other,
                )
                for other in candidates
                if other != local_index
                and abs(indexed_frames[other][0] - frame_index) >= config.loop_min_frame_gap
            ),
            key=lambda item: (item[0], item[1]),
        )[: config.loop_neighbors_per_rig]
        for distance_m, _other_frame_index, other in ranked:
            other_frame = indexed_frames[other][1]
            _add_pair(
                pairs,
                str(frame["left_image_id"]),
                str(other_frame["left_image_id"]),
                "spatial_loop_left",
            )
            _add_pair(
                pairs,
                str(frame["right_image_id"]),
                str(other_frame["right_image_id"]),
                "spatial_loop_right",
            )
            key_left = tuple(
                sorted((str(frame["left_image_id"]), str(other_frame["left_image_id"])))
            )
            key_right = tuple(
                sorted((str(frame["right_image_id"]), str(other_frame["right_image_id"])))
            )
            # Store a stable diagnostic without allowing it to affect identity ordering.
            pairs[key_left].add(f"loop_distance_m:{distance_m:.9f}")
            pairs[key_right].add(f"loop_distance_m:{distance_m:.9f}")

    records = [
        {
            "image_id_a": key[0],
            "image_id_b": key[1],
            "path_a": str(images[key[0]]["path"]),
            "path_b": str(images[key[1]]["path"]),
            "reasons": sorted(value for value in reasons if not value.startswith("loop_distance_m:")),
            "loop_distance_m": next(
                (
                    float(value.split(":", 1)[1])
                    for value in sorted(reasons)
                    if value.startswith("loop_distance_m:")
                ),
                None,
            ),
        }
        for key, reasons in sorted(pairs.items())
    ]
    if any(
        record["image_id_a"] in validation_images
        or record["image_id_b"] in validation_images
        for record in records
    ):
        raise ValueError("validation leakage detected in match graph")
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "rig_match_graph_v1",
        "dataset_manifest_sha256": dataset_sha,
        "split_manifest_sha256": split_sha,
        "configuration": config.to_dict(),
        "feature_contract": {
            "extractor": "ALIKED",
            "matcher": "LightGlue",
            "orchestrator": "HLoc",
            "optional_runtime": True,
        },
        "split_image_ids": {
            "train": sorted(training_images),
            "val": sorted(validation_images),
        },
        "pairs": records,
        "summary": {
            "training_rig_frames": len(indexed_frames),
            "training_images": len(training_images),
            "validation_images_used": 0,
            "pair_count": len(records),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
    }
    manifest["match_graph_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


def verify_match_graph(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("match_graph_sha256", ""))
    if not expected:
        raise ValueError("match graph has no match_graph_sha256")
    unsigned = dict(manifest)
    unsigned.pop("match_graph_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(f"match graph SHA256 mismatch: expected {expected}, computed {actual}")
    pairs = manifest.get("pairs", [])
    identities = [(item.get("image_id_a"), item.get("image_id_b")) for item in pairs]
    if not pairs or any(not a or not b or a >= b for a, b in identities):
        raise ValueError("match graph contains an invalid canonical pair")
    if len(identities) != len(set(identities)):
        raise ValueError("match graph contains duplicate pairs")
    if manifest.get("summary", {}).get("validation_images_used") != 0:
        raise ValueError("match graph is not training-only")
    split_image_ids = manifest.get("split_image_ids", {})
    training = {str(value) for value in split_image_ids.get("train", [])}
    validation = {str(value) for value in split_image_ids.get("val", [])}
    if not training or not validation or training & validation:
        raise ValueError("match graph has invalid train/validation identities")
    pair_images = {str(value) for identity in identities for value in identity}
    if pair_images != training or pair_images & validation:
        raise ValueError("validation leakage detected in signed match graph")
    if manifest.get("summary", {}).get("training_images") != len(training):
        raise ValueError("match graph training image count differs from signed identities")
    return actual


def write_match_graph(path: Path, manifest: dict[str, Any]) -> None:
    verify_match_graph(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
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


def write_hloc_pairs(path: Path, manifest: dict[str, Any]) -> None:
    """Write HLoc's two-column pair contract without invoking optional GPU code."""
    verify_match_graph(manifest)
    lines: list[str] = []
    for pair in manifest["pairs"]:
        first = str(pair["path_a"]).removeprefix("camera/")
        second = str(pair["path_b"]).removeprefix("camera/")
        if any(character.isspace() for character in first + second):
            raise ValueError("HLoc pair paths cannot contain whitespace")
        lines.append(f"{first} {second}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
