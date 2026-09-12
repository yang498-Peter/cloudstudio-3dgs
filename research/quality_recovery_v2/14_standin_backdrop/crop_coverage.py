#!/usr/bin/env python3
"""Which photo pixels does no Tile ever train on? (CPU, manifests only)

A Tile's views are pixel rectangles around its own LiDAR returns
(tile_inputs_v9). Content outside every Tile's crop of a face - canopy far
above any wall, the far yard - is never supervised by anyone, so no Tile
grows gaussians for it and the merged model cannot show it. The B0 coarse
prior trains on whole faces, which is why it can stand in there. This
measures the gap per face and per Face4 face id from the signed manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUNS = Path("C:/Peter/3dgs-runs/house0305_sop")
DATASETS = Path("C:/Peter/3dgs-datasets/house0305_sop_v9")
HERE = Path(__file__).resolve().parent
SCALE = 8  # raster the crops at 1/8 resolution; exact for axis-aligned rectangles up to rounding


def main() -> int:
    tiles = json.loads((RUNS / "tile_inputs_v9" / "tile_inputs_manifest.json").read_text(encoding="utf-8"))["tiles"]
    faces = json.loads((DATASETS / "face4_train" / "face_manifest.json").read_text(encoding="utf-8"))
    dataset = json.loads(
        Path("C:/Peter/3dgs-datasets/house0305_sop_v8/dataset_manifest.json").read_text(encoding="utf-8")
    )
    camera_of = {str(r["image_id"]): str(r["camera_id"]) for r in dataset["images"]}
    face_size = {}
    for camera_id, entry in faces["cameras"].items():
        for face in entry["faces"]:
            face_size[(camera_id, face["face_id"])] = (int(face["width"]), int(face["height"]))

    crops: dict[str, list[tuple[int, int, int, int, int]]] = {}
    for tile in tiles:
        for view in tile["views"]:
            crops.setdefault(str(view["sample_id"]), []).append(
                (int(tile["tile_id"]), int(view["x"]), int(view["y"]), int(view["width"]), int(view["height"]))
            )
    all_samples = [
        f"{image['image_id']}::{face['face_id']}"
        for image in faces["images"]
        for face in image["faces"]
    ]

    per_face_id: dict[str, dict[str, float]] = {}
    uncovered_pixels = 0
    total_pixels = 0
    never_in_any_tile = 0
    rows = []
    for sample_id in all_samples:
        base, face_id = sample_id.split("::", 1)
        width, height = face_size[(camera_of[base], face_id)]
        grid = np.zeros((height // SCALE, width // SCALE), dtype=bool)
        entries = crops.get(sample_id, [])
        if not entries:
            never_in_any_tile += 1
        for _, x, y, w, h in entries:
            grid[y // SCALE:(y + h) // SCALE, x // SCALE:(x + w) // SCALE] = True
        covered = float(grid.mean())
        rows.append((sample_id, len(entries), covered))
        bucket = per_face_id.setdefault(face_id, {"faces": 0, "uncovered_sum": 0.0, "fully_covered": 0})
        bucket["faces"] += 1
        bucket["uncovered_sum"] += 1.0 - covered
        bucket["fully_covered"] += int(covered >= 0.999)
        uncovered_pixels += (1.0 - covered) * width * height
        total_pixels += width * height

    uncovered = np.array([1.0 - r[2] for r in rows])
    report = {
        "kind": "tile_crop_photo_coverage_v1",
        "face_views": len(all_samples),
        "face_views_in_no_tile": never_in_any_tile,
        "pixel_fraction_outside_every_tile_crop": uncovered_pixels / total_pixels,
        "view_uncovered_fraction_percentiles": {
            "p50": float(np.percentile(uncovered, 50)),
            "p90": float(np.percentile(uncovered, 90)),
            "p95": float(np.percentile(uncovered, 95)),
            "mean": float(uncovered.mean()),
        },
        "views_with_more_than_half_uncovered": int(np.count_nonzero(uncovered > 0.5)),
        "per_face_id": {
            face_id: {
                "faces": b["faces"],
                "mean_uncovered_fraction": b["uncovered_sum"] / b["faces"],
                "fully_covered_faces": b["fully_covered"],
            }
            for face_id, b in sorted(per_face_id.items())
        },
        "tile1_views": len([r for r in rows if any(e[0] == 1 for e in crops.get(r[0], []))]),
    }
    (HERE / "crop_coverage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
