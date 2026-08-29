"""Recover MipMap splitter geometry from a completed MVS block.

This is a read-only audit helper.  It mirrors the root-box construction that
was recovered from divide_engine.exe: camera centres plus sparse points with at
least three observations, a 20 percent per-side expansion, and float32 storage.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from google.protobuf import message_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_mipmap_mvs as mvs_decode


def load_block(engine: Path, source: Path) -> object:
    pool = mvs_decode._build_pool(engine)
    descriptor = pool.FindMessageTypeByName("mipmap.engine.message.MVSBlock")
    message_class = message_factory.GetMessageClass(descriptor)
    block = message_class()
    block.ParseFromString(source.read_bytes())
    return block


def camera_center(image: object) -> np.ndarray:
    projection = np.asarray(image.projection_matrix, dtype=np.float64).reshape(3, 4)
    rotation = projection[:, :3]
    translation = projection[:, 3]
    return np.linalg.solve(rotation, -translation)


def expanded_supported_box(block: object) -> dict[str, object]:
    support = Counter(int(row.pnt_index) for row in block.observation)
    supported_ids = [index for index in range(len(block.point)) if support[index] >= 3]
    points = np.asarray(
        [
            [block.point[index].point.x, block.point[index].point.y, block.point[index].point.z]
            for index in supported_ids
        ],
        dtype=np.float64,
    )
    centres = np.asarray([camera_center(image) for image in block.image], dtype=np.float64)

    # The engine converts both sources to float before updating its Box3f.
    points_f32 = points.astype(np.float32)
    centres_f32 = centres.astype(np.float32)
    samples = np.concatenate([points_f32, centres_f32], axis=0)
    minimum_f32 = samples.min(axis=0).astype(np.float32)
    maximum_f32 = samples.max(axis=0).astype(np.float32)

    # The recovered routine converts the box to double and expands each side
    # by 20 percent of the original extent, then converts back to float.
    minimum_f64 = minimum_f32.astype(np.float64)
    maximum_f64 = maximum_f32.astype(np.float64)
    padding = (maximum_f64 - minimum_f64) * 0.2
    expanded_minimum = (minimum_f64 - padding).astype(np.float32)
    expanded_maximum = (maximum_f64 + padding).astype(np.float32)

    candidates = {
        axis: np.linspace(
            expanded_minimum[index], expanded_maximum[index], 64, dtype=np.float32
        ).astype(float).tolist()
        for index, axis in enumerate("XYZ")
    }
    return {
        "image_count": len(block.image),
        "point_count": len(block.point),
        "supported_point_count_ge_3": len(supported_ids),
        "camera_center_minimum_f32": centres_f32.min(axis=0).astype(float).tolist(),
        "camera_center_maximum_f32": centres_f32.max(axis=0).astype(float).tolist(),
        "supported_point_minimum_f32": points_f32.min(axis=0).astype(float).tolist(),
        "supported_point_maximum_f32": points_f32.max(axis=0).astype(float).tolist(),
        "union_minimum_f32": minimum_f32.astype(float).tolist(),
        "union_maximum_f32": maximum_f32.astype(float).tolist(),
        "expanded_minimum_f32": expanded_minimum.astype(float).tolist(),
        "expanded_maximum_f32": expanded_maximum.astype(float).tolist(),
        "candidate_positions": candidates,
    }


def closest_candidates(candidates: dict[str, list[float]], cuts: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cut in cuts:
        choices = []
        for axis, values in candidates.items():
            array = np.asarray(values)
            index = int(np.argmin(np.abs(array - cut)))
            choices.append(
                {
                    "axis": axis,
                    "index": index,
                    "candidate": float(array[index]),
                    "absolute_error": abs(float(array[index]) - cut),
                }
            )
        rows.append({"cut": cut, "closest_by_axis": choices})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("mvs", type=Path)
    parser.add_argument("--cut", type=float, action="append", default=[])
    args = parser.parse_args()

    result = expanded_supported_box(load_block(args.engine, args.mvs))
    result["source"] = str(args.mvs)
    if args.cut:
        result["closest_candidates"] = closest_candidates(
            result["candidate_positions"], args.cut
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
