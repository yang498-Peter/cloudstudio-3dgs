#!/usr/bin/env python3
"""Run staged PyCOLMAP BA on a pre-triangulated HLoc model and audit publication gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.ba.match_graph import verify_match_graph
from cloudstudio_3dgs.ba.pycolmap_adapter import (
    apply_fixed_stereo_rig,
    reconstruction_snapshot,
    run_bundle_adjustment_stage,
)
from cloudstudio_3dgs.ba.report import (
    build_ba_report,
    sign_ba_report,
    write_ba_report,
)
from cloudstudio_3dgs.ba.runtime_lock import verify_signed_runtime_manifest
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"COLMAP model directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def hloc_pairs_sha256(graph: dict) -> str:
    lines = []
    for pair in graph["pairs"]:
        first = str(pair["path_a"]).replace("\\", "/").removeprefix("camera/")
        second = str(pair["path_b"]).replace("\\", "/").removeprefix("camera/")
        if any(character.isspace() for character in first + second):
            raise ValueError("HLoc pair paths cannot contain whitespace")
        lines.append(f"{first} {second}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--match-graph", required=True, type=Path)
    parser.add_argument(
        "--triangulation-manifest",
        type=Path,
        help="defaults to triangulation_runtime_manifest.json beside the model directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--through-stage", choices=["stage_1", "stage_2", "stage_3"], default="stage_1"
    )
    parser.add_argument(
        "--position-prior-stddev-m",
        type=float,
        default=0.05,
        help="1-sigma Cartesian POS prior applied to every selected camera center",
    )
    args = parser.parse_args()

    if not args.model.is_dir():
        raise NotADirectoryError(f"COLMAP model is not a directory: {args.model}")
    if args.output.resolve().is_relative_to(args.model.resolve()):
        raise ValueError("BA output cannot be inside the immutable input model")
    if args.output.exists():
        if not args.output.is_dir():
            raise NotADirectoryError(f"BA output is not a directory: {args.output}")
        if any(args.output.iterdir()):
            raise FileExistsError(f"BA output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    graph = json.loads(args.match_graph.read_text(encoding="utf-8"))
    dataset_sha = verify_dataset_manifest(dataset)
    verify_match_graph(graph)
    if graph["dataset_manifest_sha256"] != dataset_sha:
        raise ValueError("match graph and dataset manifest have different identities")
    model_sha = directory_sha256(args.model)
    triangulation_manifest_path = (
        args.triangulation_manifest
        if args.triangulation_manifest is not None
        else args.model.parent / "triangulation_runtime_manifest.json"
    )
    if not triangulation_manifest_path.is_file():
        raise FileNotFoundError(
            f"triangulation runtime manifest does not exist: {triangulation_manifest_path}"
        )
    triangulation_manifest = json.loads(
        triangulation_manifest_path.read_text(encoding="utf-8")
    )
    triangulation_sha = verify_signed_runtime_manifest(
        triangulation_manifest, "triangulation_manifest_sha256"
    )
    if triangulation_manifest.get("output", {}).get("sfm_model_sha256") != model_sha:
        raise ValueError("triangulation manifest does not identify the input COLMAP model")
    if (
        triangulation_manifest.get("inputs", {}).get("pairs_sha256")
        != hloc_pairs_sha256(graph)
    ):
        raise ValueError("triangulation manifest and match graph use different HLoc pairs")

    import pycolmap

    reconstruction = pycolmap.Reconstruction(args.model)
    stable_training_ids = {
        str(pair[key])
        for pair in graph["pairs"]
        for key in ("image_id_a", "image_id_b")
    }
    apply_fixed_stereo_rig(
        reconstruction, dataset, included_image_ids=stable_training_ids
    )
    names_by_stable_id = {
        str(image["image_id"]): str(image["path"])
        .replace("\\", "/")
        .removeprefix("camera/")
        for image in dataset["images"]
    }
    model_image_ids = {
        image.name.replace("\\", "/"): int(image.image_id)
        for image in reconstruction.images.values()
    }
    selected_model_ids = {
        model_image_ids[names_by_stable_id[stable_id]] for stable_id in stable_training_ids
    }
    position_priors: dict[int, np.ndarray] = {}
    for image_record in dataset["images"]:
        if str(image_record["image_id"]) not in stable_training_ids:
            continue
        name = str(image_record["path"]).replace("\\", "/").removeprefix("camera/")
        if name not in model_image_ids:
            raise ValueError(f"COLMAP model is missing training image {name}")
        c2w = np.asarray(image_record["c2w"], dtype=np.float64)
        if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
            raise ValueError(f"dataset image has invalid POS c2w: {name}")
        position_priors[model_image_ids[name]] = c2w[:3, 3]
    before_dir = args.output / "before_model"
    before_dir.mkdir()
    reconstruction.write(before_dir)
    before = reconstruction_snapshot(
        reconstruction,
        dataset,
        model_sha256=directory_sha256(before_dir),
        solver_success=True,
        included_image_ids=stable_training_ids,
    )

    stages = ["stage_1", "stage_2", "stage_3"]
    summaries = []
    for stage in stages[: stages.index(args.through_stage) + 1]:
        summary = run_bundle_adjustment_stage(
            reconstruction,
            stage,
            image_ids=selected_model_ids,
            position_priors_by_image_id=position_priors,
            position_prior_stddev_m=args.position_prior_stddev_m,
        )
        summaries.append(summary)
        if not summary.is_solution_usable():
            break
    candidate_dir = args.output / "candidate_model"
    candidate_dir.mkdir()
    reconstruction.write(candidate_dir)
    solver_success = bool(summaries) and all(
        summary.is_solution_usable() for summary in summaries
    )
    after = reconstruction_snapshot(
        reconstruction,
        dataset,
        model_sha256=directory_sha256(candidate_dir),
        solver_success=solver_success,
        included_image_ids=stable_training_ids,
    )
    report = build_ba_report(before, after, stage=args.through_stage)
    report["solver_summaries"] = [summary.brief_report() for summary in summaries]
    report["position_prior"] = {
        "coordinate_system": "CARTESIAN",
        "image_count": len(position_priors),
        "stddev_m": args.position_prior_stddev_m,
    }
    report["triangulation_manifest_sha256"] = triangulation_sha
    report = sign_ba_report(report)
    write_ba_report(args.output / "report", report)
    print(
        f"Rig BA: stage={args.through_stage}, accepted={report['candidate_accepted']}, "
        f"published={report['published_model']} -> {args.output}"
    )
    return 0 if report["candidate_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
