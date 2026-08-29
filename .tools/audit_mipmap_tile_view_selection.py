"""Audit how MipMap expands a spatial Tile into a contextual MVS block.

The tool is read-only.  It compares the emitted per-Tile image/point sets with
simple observation-graph closures built from the undistorted source MVS.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf import message_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_mipmap_mvs as mvs_decode
import replay_mipmap_adaptive_tiling as replay


def load_message(engine: Path, source: Path) -> object:
    pool = mvs_decode._build_pool(engine)
    descriptor = pool.FindMessageTypeByName("mipmap.engine.message.MVSBlock")
    message_class = message_factory.GetMessageClass(descriptor)
    block = message_class()
    block.ParseFromString(source.read_bytes())
    return block


def set_metrics(predicted: set[int], actual: set[int]) -> dict[str, float | int]:
    intersection = len(predicted & actual)
    union = len(predicted | actual)
    return {
        "predicted": len(predicted),
        "actual": len(actual),
        "intersection": intersection,
        "precision": intersection / len(predicted) if predicted else 0.0,
        "recall": intersection / len(actual) if actual else 0.0,
        "jaccard": intersection / union if union else 1.0,
    }


def coordinate_key(point: object) -> tuple[float, float, float]:
    return (float(point.point.x), float(point.point.y), float(point.point.z))


def audit(
    engine: Path, source_mvs: Path, tiles_path: Path, block_directory: Path
) -> dict[str, Any]:
    source_message = load_message(engine, source_mvs)
    data = replay.load_mvs(engine, source_mvs)
    source_path_to_index = {
        str(image.img_path): index for index, image in enumerate(source_message.image)
    }
    source_point_keys = {coordinate_key(point) for point in source_message.point}
    rows = []
    for tile in replay.runtime_tiles(tiles_path):
        block_path = block_directory / f"{tile['name']}.pb.bin"
        block = load_message(engine, block_path)
        actual_images = {
            source_path_to_index[str(image.img_path)]
            for image in block.image
            if str(image.img_path) in source_path_to_index
        }

        variants: dict[str, dict[str, float | int]] = {}
        for name, box in (("core", tile["core"]), ("exported", tile["expanded"])):
            selected_points = replay.point_mask(data["points"], box)
            selected_observations = (
                selected_points[data["observation_point"]]
                & data["observation_projection_valid"]
            )
            any_observation_images = set(
                map(int, np.unique(data["observation_image"][selected_observations]))
            )
            variants[f"{name}_any_observation"] = set_metrics(
                any_observation_images, actual_images
            )

        actual_image_mask = np.isin(
            data["observation_image"], np.asarray(sorted(actual_images), dtype=np.int64)
        )
        graph_point_ids = set(
            map(int, np.unique(data["observation_point"][actual_image_mask]))
        )
        block_point_counter = Counter(coordinate_key(point) for point in block.point)
        block_point_keys = set(block_point_counter)
        unique_coordinates = np.asarray(list(block_point_keys), dtype=np.float64)
        multiplicities = np.asarray(
            [block_point_counter[tuple(point)] for point in unique_coordinates],
            dtype=np.int64,
        )
        in_core = replay.point_mask(unique_coordinates, tile["core"])
        in_exported = replay.point_mask(unique_coordinates, tile["expanded"])
        duplicated = multiplicities > 1
        rows.append(
            {
                "tile": tile["name"],
                "actual_block": {
                    "images": len(block.image),
                    "mapped_source_images": len(actual_images),
                    "points": len(block.point),
                    "observations": len(block.observation),
                },
                "image_set_hypotheses": variants,
                "closure_from_actual_images": {
                    "source_observations": int(np.count_nonzero(actual_image_mask)),
                    "source_unique_points": len(graph_point_ids),
                    "block_points_found_exactly_in_source": len(
                        block_point_keys & source_point_keys
                    ),
                    "block_unique_point_coordinates": len(block_point_keys),
                },
                "block_point_duplication": {
                    "duplicated_unique_coordinates": int(np.count_nonzero(duplicated)),
                    "duplicated_inside_core": int(
                        np.count_nonzero(duplicated & in_core)
                    ),
                    "duplicated_inside_exported_roi": int(
                        np.count_nonzero(duplicated & in_exported)
                    ),
                    "duplicated_outside_exported_roi": int(
                        np.count_nonzero(duplicated & ~in_exported)
                    ),
                    "unique_coordinates_inside_core": int(np.count_nonzero(in_core)),
                    "unique_coordinates_inside_exported_roi": int(
                        np.count_nonzero(in_exported)
                    ),
                    "maximum_multiplicity": int(multiplicities.max(initial=0)),
                },
            }
        )
    return {
        "method": "MipMap Tile view-selection graph audit",
        "source_mvs": str(source_mvs),
        "tiles": str(tiles_path),
        "block_directory": str(block_directory),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("source_mvs", type=Path)
    parser.add_argument("tiles", type=Path)
    parser.add_argument("block_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.engine, args.source_mvs, args.tiles, args.block_directory)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
