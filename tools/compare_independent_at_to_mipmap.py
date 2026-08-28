#!/usr/bin/env python3
"""Compare a CloudStudio independent-pose AT candidate with MipMap poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def _normalise_name(value: str) -> str:
    return value.replace("\\", "/").removeprefix("camera/")


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("comparison distribution requires finite values")
    return {
        "mean": float(np.mean(array)),
        "rmse": float(np.sqrt(np.mean(array * array))),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _matrix_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            [float(row[f"{prefix}_{axis}{column}"]) for column in range(3)]
            for axis in range(3)
        ],
        dtype=np.float64,
    )


def _reprojection_errors(reconstruction: Any) -> list[float]:
    errors: list[float] = []
    for point in reconstruction.points3D.values():
        for element in point.track.elements:
            image = reconstruction.image(element.image_id)
            projected = image.project_point(point.xyz)
            if projected is None:
                continue
            observed = np.asarray(image.point2D(element.point2D_idx).xy, dtype=np.float64)
            errors.append(float(np.linalg.norm(np.asarray(projected) - observed)))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--at-report", required=True, type=Path)
    parser.add_argument("--mipmap-poses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"comparison output already exists: {args.output}")
    args.output.mkdir(parents=True)

    import pycolmap

    reconstruction = pycolmap.Reconstruction(args.candidate_model)
    at_report = json.loads(args.at_report.read_text(encoding="utf-8"))
    ours_by_name = {
        _normalise_name(str(item["name"])): item for item in at_report["per_image"]
    }
    model_by_name = {
        _normalise_name(str(image.name)): image for image in reconstruction.images.values()
    }
    with args.mipmap_poses.open("r", encoding="utf-8", newline="") as stream:
        mipmap_rows = list(csv.DictReader(stream))

    per_image: list[dict[str, Any]] = []
    for mipmap in mipmap_rows:
        name = _normalise_name(str(mipmap["source_path"]).split("/camera/", 1)[-1])
        if name not in model_by_name or name not in ours_by_name:
            raise ValueError(f"candidate AT is missing MipMap image: {name}")
        image = model_by_name[name]
        ours = ours_by_name[name]
        ours_center = np.asarray(image.projection_center(), dtype=np.float64)
        mipmap_center = np.asarray(
            [float(mipmap[f"corrected_{axis}"]) for axis in "xyz"], dtype=np.float64
        )
        ours_w2c = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)[:3, :3]
        mipmap_w2c = _matrix_from_row(mipmap, "optimized_w2c")
        ours_correction = np.asarray(ours["raw_minus_corrected_m"], dtype=np.float64)
        mipmap_correction = np.asarray(
            [float(mipmap[f"pos_diff_raw_minus_corrected_{axis}"]) for axis in "xyz"],
            dtype=np.float64,
        )
        per_image.append(
            {
                "name": name,
                "camera_id": int(mipmap["camera_id"]),
                "ours_position_correction_m": float(
                    ours["position_correction_norm_m"]
                ),
                "mipmap_position_correction_m": float(mipmap["pos_correction_norm_m"]),
                "correction_vector_difference_m": float(
                    np.linalg.norm(ours_correction - mipmap_correction)
                ),
                "final_center_difference_m": float(
                    np.linalg.norm(ours_center - mipmap_center)
                ),
                "ours_rotation_correction_deg": float(ours["rotation_correction_deg"]),
                "mipmap_rotation_correction_deg": float(
                    mipmap["rotation_correction_deg"]
                ),
                "final_rotation_difference_deg": _rotation_angle_deg(
                    ours_w2c @ mipmap_w2c.T
                ),
            }
        )

    if len(per_image) != reconstruction.num_reg_images():
        raise ValueError("candidate and MipMap image counts differ")

    def values(key: str, camera_id: int | None = None) -> list[float]:
        return [
            float(item[key])
            for item in per_image
            if camera_id is None or item["camera_id"] == camera_id
        ]

    metric_names = (
        "ours_position_correction_m",
        "mipmap_position_correction_m",
        "correction_vector_difference_m",
        "final_center_difference_m",
        "ours_rotation_correction_deg",
        "mipmap_rotation_correction_deg",
        "final_rotation_difference_deg",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "image_count": len(per_image),
        "point_count": reconstruction.num_points3D(),
        "observation_count": reconstruction.compute_num_observations(),
        "candidate_reprojection_error_px": _distribution(
            _reprojection_errors(reconstruction)
        ),
        "all": {name: _distribution(values(name)) for name in metric_names},
        "by_camera": {
            str(camera_id): {
                name: _distribution(values(name, camera_id)) for name in metric_names
            }
            for camera_id in (1, 2)
        },
    }
    import hashlib

    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "per_image_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    print(
        "AT comparison: "
        f"images={len(per_image)}, center_p50="
        f"{100.0 * report['all']['final_center_difference_m']['p50']:.3f}cm, "
        f"rotation_p50={report['all']['final_rotation_difference_deg']['p50']:.3f}deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
