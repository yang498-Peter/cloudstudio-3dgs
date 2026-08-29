"""Audit how MipMap's undistorted virtual views map to parent raw photos.

This read-only utility compares two MVSBlock protobuf files using descriptors
embedded in divide_engine.exe.  It measures whether virtual undistorted images
form camera-centre groups, whether their points are preserved from the raw
block, and whether the undistorted point/view observation graph is a subset of
the raw parent-photo graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf import message_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_mipmap_mvs as mvs_decoder  # noqa: E402


def load_block(engine_path: Path, mvs_path: Path) -> Any:
    pool = mvs_decoder._build_pool(engine_path)
    descriptor = pool.FindMessageTypeByName("mipmap.engine.message.MVSBlock")
    block_class = message_factory.GetMessageClass(descriptor)
    block = block_class()
    block.ParseFromString(mvs_path.read_bytes())
    return block


def point_array(block: Any) -> np.ndarray:
    return np.asarray(
        [[point.point.x, point.point.y, point.point.z] for point in block.point],
        dtype=np.float32,
    )


def camera_centres(block: Any) -> np.ndarray:
    matrices = np.asarray(
        [image.projection_matrix for image in block.image], dtype=np.float64
    ).reshape(-1, 3, 4)
    return np.asarray(
        [
            np.linalg.solve(matrix[:, :3], -matrix[:, 3])
            for matrix in matrices
        ],
        dtype=np.float64,
    )


def coordinate_key(point: np.ndarray) -> bytes:
    return np.asarray(point, dtype="<f4").tobytes()


def nearest_parent_images(
    raw_centres: np.ndarray, undistorted_centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.empty(len(undistorted_centres), dtype=np.int64)
    distance = np.empty(len(undistorted_centres), dtype=np.float64)
    for start in range(0, len(undistorted_centres), 256):
        stop = min(start + 256, len(undistorted_centres))
        delta = (
            undistorted_centres[start:stop, None, :]
            - raw_centres[None, :, :]
        )
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        indices = np.argmin(squared, axis=1)
        parent[start:stop] = indices
        distance[start:stop] = np.sqrt(
            squared[np.arange(stop - start), indices]
        )
    return parent, distance


def metadata_parent_images(raw: Any, undistorted: Any) -> tuple[np.ndarray, dict[str, int]]:
    raw_metadata = {int(row.id): row.meta_data for row in raw.image_meta_data}
    undistorted_metadata = {
        int(row.id): row.meta_data for row in undistorted.image_meta_data
    }
    raw_by_key: dict[tuple[int, float], list[int]] = defaultdict(list)
    for index, image in enumerate(raw.image):
        metadata = raw_metadata.get(int(image.img_id))
        if metadata is None:
            continue
        raw_by_key[(int(metadata.camera_id), float(metadata.timestamp))].append(index)

    parent = np.full(len(undistorted.image), -1, dtype=np.int64)
    missing_metadata = 0
    missing_key = 0
    ambiguous_key = 0
    inherited_raw_dimensions = 0
    for index, image in enumerate(undistorted.image):
        metadata = undistorted_metadata.get(int(image.img_id))
        if metadata is None:
            missing_metadata += 1
            continue
        if int(metadata.width) == 2912 and int(metadata.height) == 2912:
            inherited_raw_dimensions += 1
        matches = raw_by_key.get(
            (int(metadata.camera_id), float(metadata.timestamp)), []
        )
        if not matches:
            missing_key += 1
            continue
        parent[index] = matches[0]
        ambiguous_key += max(len(matches) - 1, 0)
    return parent, {
        "missing_undistorted_metadata": missing_metadata,
        "missing_parent_key": missing_key,
        "ambiguous_extra_parent_keys": ambiguous_key,
        "undistorted_metadata_with_2912x2912_source_dimensions": inherited_raw_dimensions,
    }


def summarize(engine_path: Path, raw_path: Path, undistorted_path: Path) -> dict[str, Any]:
    raw = load_block(engine_path, raw_path)
    undistorted = load_block(engine_path, undistorted_path)
    raw_points = point_array(raw)
    undistorted_points = point_array(undistorted)
    raw_centres = camera_centres(raw)
    undistorted_centres = camera_centres(undistorted)

    parent_image, parent_distance = nearest_parent_images(
        raw_centres, undistorted_centres
    )
    metadata_parent, metadata_stats = metadata_parent_images(raw, undistorted)
    group_sizes = Counter(int(index) for index in parent_image)
    metadata_mapped = metadata_parent >= 0
    metadata_centre_agreement = int(
        np.count_nonzero(
            metadata_mapped & (metadata_parent == parent_image)
        )
    )

    raw_point_by_key: dict[bytes, list[int]] = defaultdict(list)
    for index, point in enumerate(raw_points):
        raw_point_by_key[coordinate_key(point)].append(index)

    undistorted_to_raw_point = np.full(len(undistorted_points), -1, dtype=np.int64)
    ambiguous_point_matches = 0
    for index, point in enumerate(undistorted_points):
        matches = raw_point_by_key.get(coordinate_key(point), [])
        if matches:
            undistorted_to_raw_point[index] = matches[0]
            ambiguous_point_matches += max(len(matches) - 1, 0)

    raw_pairs = {
        (int(observation.pnt_index), int(observation.img_index))
        for observation in raw.observation
    }
    mapped_undistorted_pairs: set[tuple[int, int]] = set()
    unmapped_observations = 0
    for observation in undistorted.observation:
        raw_point = int(undistorted_to_raw_point[int(observation.pnt_index)])
        if raw_point < 0:
            unmapped_observations += 1
            continue
        mapped_undistorted_pairs.add(
            (raw_point, int(parent_image[int(observation.img_index)]))
        )

    shared_pairs = raw_pairs & mapped_undistorted_pairs
    raw_only_pairs = raw_pairs - mapped_undistorted_pairs
    undistorted_only_pairs = mapped_undistorted_pairs - raw_pairs

    return {
        "method": "MipMap raw/undistorted parent-view graph audit",
        "evidence_boundary": (
            "Image parentage is matched exactly by source camera ID plus timestamp and "
            "cross-checked by nearest camera centre; observation-pair equality is measured "
            "directly from the protobufs."
        ),
        "input": {
            "engine": str(engine_path),
            "raw_mvs": str(raw_path),
            "undistorted_mvs": str(undistorted_path),
        },
        "counts": {
            "raw_images": len(raw.image),
            "undistorted_images": len(undistorted.image),
            "raw_points": len(raw.point),
            "undistorted_points": len(undistorted.point),
            "raw_observations": len(raw.observation),
            "undistorted_observations": len(undistorted.observation),
        },
        "parent_image_mapping": {
            "unique_parent_images_used": len(group_sizes),
            "group_size_histogram": {
                str(size): count
                for size, count in sorted(Counter(group_sizes.values()).items())
            },
            "maximum_camera_centre_distance_m": float(parent_distance.max(initial=0.0)),
            "p99_camera_centre_distance_m": float(
                np.quantile(parent_distance, 0.99) if len(parent_distance) else 0.0
            ),
            "metadata_exact_matches": int(np.count_nonzero(metadata_mapped)),
            "metadata_and_camera_centre_agree": metadata_centre_agreement,
            **metadata_stats,
        },
        "point_mapping": {
            "exact_float32_matches": int(np.count_nonzero(undistorted_to_raw_point >= 0)),
            "unmatched_undistorted_points": int(
                np.count_nonzero(undistorted_to_raw_point < 0)
            ),
            "ambiguous_extra_matches": int(ambiguous_point_matches),
        },
        "observation_parent_graph": {
            "raw_unique_pairs": len(raw_pairs),
            "mapped_undistorted_unique_pairs": len(mapped_undistorted_pairs),
            "shared_pairs": len(shared_pairs),
            "raw_only_pairs": len(raw_only_pairs),
            "undistorted_only_pairs": len(undistorted_only_pairs),
            "unmapped_undistorted_observations": unmapped_observations,
            "undistorted_pair_precision_against_raw": (
                len(shared_pairs) / len(mapped_undistorted_pairs)
                if mapped_undistorted_pairs
                else None
            ),
            "raw_pair_recall_in_undistorted": (
                len(shared_pairs) / len(raw_pairs) if raw_pairs else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("raw_mvs", type=Path)
    parser.add_argument("undistorted_mvs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.engine, args.raw_mvs, args.undistorted_mvs)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
