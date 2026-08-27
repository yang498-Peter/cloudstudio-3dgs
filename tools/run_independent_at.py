#!/usr/bin/env python3
"""Run product-style all-image AT without a hard stereo Rig constraint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.ba.pycolmap_adapter import (
    reconstruction_snapshot,
    run_independent_pose_bundle_adjustment,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest
from cloudstudio_3dgs.geometry.rig import distribution, rotation_error_rad


def _normalise_name(value: str) -> str:
    return value.replace("\\", "/").removeprefix("camera/")


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"COLMAP model directory contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _angle_deg(after_w2c: np.ndarray, before_w2c: np.ndarray) -> float:
    return float(np.degrees(rotation_error_rad(after_w2c, before_w2c)))


def _camera_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for camera_id in sorted(before):
        old = before[camera_id]
        new = after[camera_id]
        output[camera_id] = {
            "before": old,
            "after": new,
            "delta_after_minus_before": {
                key: float(new[key]) - float(old[key])
                for key in ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4")
            },
        }
    return output


def _snapshot_distribution(snapshot: dict[str, Any]) -> dict[str, float]:
    return distribution([float(value) for value in snapshot["reprojection_errors_px"]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--position-sigma-x", type=float, default=0.03)
    parser.add_argument("--position-sigma-y", type=float, default=0.03)
    parser.add_argument("--position-sigma-z", type=float, default=0.06)
    parser.add_argument("--pose-iterations", type=int, default=150)
    parser.add_argument("--full-iterations", type=int, default=250)
    parser.add_argument(
        "--skip-intrinsics",
        action="store_true",
        help="publish the converged pose-and-points stage without intrinsic refinement",
    )
    parser.add_argument("--mipmap-summary", type=Path)
    args = parser.parse_args()

    if not args.model.is_dir():
        raise NotADirectoryError(f"input model does not exist: {args.model}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {args.manifest}")
    if args.output.exists():
        if not args.output.is_dir():
            raise NotADirectoryError(f"output is not a directory: {args.output}")
        if any(args.output.iterdir()):
            raise FileExistsError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_sha = verify_dataset_manifest(dataset)
    import pycolmap

    reconstruction = pycolmap.Reconstruction(args.model)
    if reconstruction.num_reg_images() != len(dataset["images"]):
        raise ValueError(
            "independent all-image AT requires every manifest image registered in the model"
        )
    if reconstruction.num_frames() != reconstruction.num_reg_images():
        raise ValueError("input model does not have one independently movable frame per image")
    if any(len(rig.sensor_ids()) != 1 for rig in reconstruction.rigs.values()):
        raise ValueError("input model already contains a multi-sensor Rig")

    model_images = {
        _normalise_name(image.name): image for image in reconstruction.images.values()
    }
    stable_id_by_model_id: dict[int, str] = {}
    position_priors: dict[int, np.ndarray] = {}
    for record in dataset["images"]:
        name = _normalise_name(str(record["path"]))
        if name not in model_images:
            raise ValueError(f"COLMAP model is missing manifest image {name}")
        image = model_images[name]
        c2w = np.asarray(record["c2w"], dtype=np.float64)
        if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
            raise ValueError(f"manifest image has invalid c2w pose: {name}")
        stable_id_by_model_id[int(image.image_id)] = str(record["image_id"])
        position_priors[int(image.image_id)] = c2w[:3, 3]
    selected_ids = set(stable_id_by_model_id)
    included_stable_ids = set(stable_id_by_model_id.values())

    before_dir = args.output / "before_model"
    before_dir.mkdir()
    reconstruction.write(before_dir)
    before = reconstruction_snapshot(
        reconstruction,
        dataset,
        model_sha256=_directory_sha256(before_dir),
        solver_success=True,
        included_image_ids=included_stable_ids,
    )
    before_pose = {
        image_id: (
            reconstruction.image(image_id).projection_center().copy(),
            np.asarray(reconstruction.image(image_id).cam_from_world().matrix())[:3, :3].copy(),
        )
        for image_id in selected_ids
    }

    sigma = (
        args.position_sigma_x,
        args.position_sigma_y,
        args.position_sigma_z,
    )
    summaries = []
    pose_summary = run_independent_pose_bundle_adjustment(
        reconstruction,
        image_ids=selected_ids,
        position_priors_by_image_id=position_priors,
        position_prior_stddev_xyz_m=sigma,
        refine_intrinsics=False,
        max_num_iterations=args.pose_iterations,
    )
    summaries.append(
        {
            "stage": "pose_points",
            "is_solution_usable": bool(pose_summary.is_solution_usable()),
            "brief_report": pose_summary.brief_report(),
        }
    )
    if pose_summary.is_solution_usable() and not args.skip_intrinsics:
        full_summary = run_independent_pose_bundle_adjustment(
            reconstruction,
            image_ids=selected_ids,
            position_priors_by_image_id=position_priors,
            position_prior_stddev_xyz_m=sigma,
            refine_intrinsics=True,
            max_num_iterations=args.full_iterations,
        )
        summaries.append(
            {
                "stage": "pose_points_shared_intrinsics",
                "is_solution_usable": bool(full_summary.is_solution_usable()),
                "brief_report": full_summary.brief_report(),
            }
        )
    candidate_dir = args.output / "candidate_model"
    candidate_dir.mkdir()
    reconstruction.write(candidate_dir)
    solver_usable = all(item["is_solution_usable"] for item in summaries)
    solver_converged = all(
        "Termination: CONVERGENCE" in item["brief_report"] for item in summaries
    )
    after = reconstruction_snapshot(
        reconstruction,
        dataset,
        model_sha256=_directory_sha256(candidate_dir),
        solver_success=solver_usable,
        included_image_ids=included_stable_ids,
    )

    position_corrections = []
    rotation_corrections = []
    per_image = []
    for image_id in sorted(selected_ids):
        image = reconstruction.image(image_id)
        old_center, old_rotation = before_pose[image_id]
        new_center = image.projection_center()
        new_rotation = np.asarray(image.cam_from_world().matrix())[:3, :3]
        raw_minus_corrected = old_center - new_center
        position_norm = float(np.linalg.norm(raw_minus_corrected))
        rotation_deg = _angle_deg(new_rotation, old_rotation)
        position_corrections.append(position_norm)
        rotation_corrections.append(rotation_deg)
        per_image.append(
            {
                "image_id": stable_id_by_model_id[image_id],
                "model_image_id": image_id,
                "name": image.name,
                "raw_minus_corrected_m": raw_minus_corrected.tolist(),
                "position_correction_norm_m": position_norm,
                "rotation_correction_deg": rotation_deg,
            }
        )

    raw_baselines = []
    corrected_baselines = []
    baseline_changes = []
    relative_rotation_changes = []
    records = {str(item["image_id"]): item for item in dataset["images"]}
    names = {
        str(item["image_id"]): model_images[_normalise_name(str(item["path"]))]
        for item in dataset["images"]
    }
    for frame in dataset["rig_frames"]:
        left_id = str(frame["left_image_id"])
        right_id = str(frame["right_image_id"])
        left_model_id = int(names[left_id].image_id)
        right_model_id = int(names[right_id].image_id)
        left_old_center, left_old_rotation = before_pose[left_model_id]
        right_old_center, right_old_rotation = before_pose[right_model_id]
        left_new = reconstruction.image(left_model_id)
        right_new = reconstruction.image(right_model_id)
        raw_vector = right_old_center - left_old_center
        corrected_vector = right_new.projection_center() - left_new.projection_center()
        raw_baselines.append(float(np.linalg.norm(raw_vector)))
        corrected_baselines.append(float(np.linalg.norm(corrected_vector)))
        baseline_changes.append(float(np.linalg.norm(corrected_vector - raw_vector)))
        old_relative = right_old_rotation @ left_old_rotation.T
        new_relative = (
            np.asarray(right_new.cam_from_world().matrix())[:3, :3]
            @ np.asarray(left_new.cam_from_world().matrix())[:3, :3].T
        )
        relative_rotation_changes.append(_angle_deg(new_relative, old_relative))

    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "independent_pos_prior_shared_kb4_at_v1",
        "dataset_manifest_sha256": dataset_sha,
        "input_model_sha256": before["model_sha256"],
        "candidate_model_sha256": after["model_sha256"],
        "counts": {
            "images": reconstruction.num_reg_images(),
            "points3D": reconstruction.num_points3D(),
            "observations": reconstruction.compute_num_observations(),
        },
        "position_prior_sigma_xyz_m": list(sigma),
        "solver": summaries,
        "solver_usable": solver_usable,
        "solver_converged": solver_converged,
        "intrinsics_refined": not args.skip_intrinsics,
        "reprojection_error_px": {
            "before": _snapshot_distribution(before),
            "after": _snapshot_distribution(after),
        },
        "position_correction_norm_m": distribution(position_corrections),
        "rotation_correction_deg": distribution(rotation_corrections),
        "camera_parameters": _camera_changes(before["cameras"], after["cameras"]),
        "stereo_pair_geometry": {
            "raw_baseline_length_m": distribution(raw_baselines),
            "corrected_baseline_length_m": distribution(corrected_baselines),
            "baseline_vector_change_m": distribution(baseline_changes),
            "relative_rotation_change_deg": distribution(relative_rotation_changes),
        },
        "per_image": per_image,
    }
    if args.mipmap_summary is not None:
        mipmap = json.loads(args.mipmap_summary.read_text(encoding="utf-8"))
        report["mipmap_reference"] = {
            "position_correction_norm_m": mipmap.get("position_correction_norm_m", {}).get("all"),
            "rotation_correction_deg": mipmap.get("rotation_correction_deg", {}).get("all"),
            "intrinsic_changes": mipmap.get("intrinsic_changes"),
            "stereo_pair_geometry": mipmap.get("stereo_pair_geometry"),
        }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    (args.output / "at_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Independent AT: images={report['counts']['images']}, points={report['counts']['points3D']}, "
        f"solver_usable={solver_usable}, reprojection_p50="
        f"{report['reprojection_error_px']['after']['p50']:.6f}px -> {args.output}"
    )
    return 0 if solver_usable else 2


if __name__ == "__main__":
    raise SystemExit(main())
