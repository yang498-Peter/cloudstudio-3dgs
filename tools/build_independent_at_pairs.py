#!/usr/bin/env python3
"""Build deterministic all-image pairs for a product-style independent AT run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest


def _normalise_path(value: str) -> str:
    return value.replace("\\", "/").removeprefix("camera/")


def _atomic_write(path: Path, payload: bytes) -> None:
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


def build_pairs(
    dataset: dict[str, Any],
    *,
    temporal_neighbors: int = 4,
    loop_max_distance_m: float = 1.5,
    loop_min_frame_gap: int = 30,
    loop_neighbors: int = 4,
) -> dict[str, Any]:
    dataset_sha = verify_dataset_manifest(dataset)
    if temporal_neighbors <= 0 or loop_neighbors <= 0:
        raise ValueError("neighbor counts must be positive")
    if loop_min_frame_gap <= temporal_neighbors:
        raise ValueError("loop_min_frame_gap must exceed the temporal window")
    if not np.isfinite(loop_max_distance_m) or loop_max_distance_m <= 0.0:
        raise ValueError("loop_max_distance_m must be finite and positive")

    images = {str(image["image_id"]): image for image in dataset["images"]}
    frames = sorted(dataset["rig_frames"], key=lambda item: int(item["timestamp_ns"]))
    if len(frames) < 2:
        raise ValueError("independent AT requires at least two stereo frames")
    covered = {
        str(image_id) for frame in frames for image_id in frame.get("image_ids", [])
    }
    if covered != set(images):
        raise ValueError("all-image AT requires every image in one complete stereo frame")

    pairs: dict[tuple[str, str], set[str]] = {}

    def add(first: str, second: str, reason: str) -> None:
        if first == second:
            raise ValueError("AT graph cannot contain a self-pair")
        pairs.setdefault(tuple(sorted((first, second))), set()).add(reason)

    for frame in frames:
        add(str(frame["left_image_id"]), str(frame["right_image_id"]), "stereo")
    for left_index, left_frame in enumerate(frames):
        for right_index in range(left_index + 1, min(len(frames), left_index + temporal_neighbors + 1)):
            right_frame = frames[right_index]
            add(
                str(left_frame["left_image_id"]),
                str(right_frame["left_image_id"]),
                "temporal_left",
            )
            add(
                str(left_frame["right_image_id"]),
                str(right_frame["right_image_id"]),
                "temporal_right",
            )

    positions = np.stack(
        [
            0.5
            * (
                np.asarray(images[str(frame["left_image_id"])]["c2w"], dtype=np.float64)[:3, 3]
                + np.asarray(images[str(frame["right_image_id"])]["c2w"], dtype=np.float64)[:3, 3]
            )
            for frame in frames
        ]
    )
    tree = cKDTree(positions)
    loop_distance_by_pair: dict[tuple[str, str], float] = {}
    for frame_index, frame in enumerate(frames):
        candidates = tree.query_ball_point(
            positions[frame_index], loop_max_distance_m, workers=1
        )
        ranked = sorted(
            (
                float(np.linalg.norm(positions[other] - positions[frame_index])),
                other,
            )
            for other in candidates
            if other != frame_index and abs(other - frame_index) >= loop_min_frame_gap
        )[:loop_neighbors]
        for distance_m, other in ranked:
            other_frame = frames[other]
            for side in ("left", "right"):
                first = str(frame[f"{side}_image_id"])
                second = str(other_frame[f"{side}_image_id"])
                key = tuple(sorted((first, second)))
                add(first, second, f"spatial_loop_{side}")
                loop_distance_by_pair[key] = min(
                    distance_m, loop_distance_by_pair.get(key, distance_m)
                )

    records = []
    reason_counts: dict[str, int] = {}
    for (first, second), reasons in sorted(pairs.items()):
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        records.append(
            {
                "image_id_a": first,
                "image_id_b": second,
                "path_a": _normalise_path(str(images[first]["path"])),
                "path_b": _normalise_path(str(images[second]["path"])),
                "reasons": sorted(reasons),
                "loop_distance_m": loop_distance_by_pair.get((first, second)),
            }
        )
    pair_images = {value for pair in pairs for value in pair}
    if pair_images != set(images):
        raise ValueError("AT graph does not cover every source image")
    manifest = {
        "schema_version": 1,
        "algorithm_version": "independent_all_image_at_pairs_v1",
        "dataset_manifest_sha256": dataset_sha,
        "configuration": {
            "temporal_neighbors": temporal_neighbors,
            "loop_max_distance_m": loop_max_distance_m,
            "loop_min_frame_gap": loop_min_frame_gap,
            "loop_neighbors": loop_neighbors,
        },
        "pairs": records,
        "summary": {
            "rig_frames": len(frames),
            "images": len(images),
            "pairs": len(records),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--temporal-neighbors", type=int, default=4)
    parser.add_argument("--loop-max-distance-m", type=float, default=1.5)
    parser.add_argument("--loop-min-frame-gap", type=int, default=30)
    parser.add_argument("--loop-neighbors", type=int, default=4)
    args = parser.parse_args()
    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = build_pairs(
        dataset,
        temporal_neighbors=args.temporal_neighbors,
        loop_max_distance_m=args.loop_max_distance_m,
        loop_min_frame_gap=args.loop_min_frame_gap,
        loop_neighbors=args.loop_neighbors,
    )
    _atomic_write(
        args.output,
        (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    lines = [f"{pair['path_a']} {pair['path_b']}\n" for pair in result["pairs"]]
    _atomic_write(args.pairs, "".join(lines).encode("utf-8"))
    print(
        f"Independent AT pairs: images={result['summary']['images']}, "
        f"pairs={result['summary']['pairs']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
