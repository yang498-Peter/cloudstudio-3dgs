#!/usr/bin/env python3
"""Triangulate ALIKED+LightGlue matches against a checked known-pose model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.ba.runtime_lock import (
    collect_runtime_evidence,
    load_runtime_lock,
    runtime_lock_sha256,
    verify_signed_runtime_manifest,
)
from cloudstudio_3dgs.ba.pycolmap_adapter import (
    build_reference_model_from_manifest,
    build_training_reference_model,
)
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    reference_source = parser.add_mutually_exclusive_group(required=True)
    reference_source.add_argument("--reference-model", type=Path)
    reference_source.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--matches", required=True, type=Path)
    parser.add_argument("--feature-runtime-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        default=REPOSITORY_ROOT / "upstream" / "rig_ba.lock.json",
    )
    parser.add_argument(
        "--allow-unverified-vcs",
        action="store_true",
        help="allow wheel installs without a provable VCS commit; evidence remains UNVERIFIED",
    )
    args = parser.parse_args()

    directories = {"image directory": args.image_dir}
    if args.reference_model is not None:
        directories["reference model"] = args.reference_model
    for label, path in directories.items():
        if not path.is_dir():
            raise NotADirectoryError(f"{label} is not a directory: {path}")
    files = {
        "pairs": args.pairs,
        "features": args.features,
        "matches": args.matches,
        "feature runtime manifest": args.feature_runtime_manifest,
        "runtime lock": args.runtime_lock,
    }
    if args.dataset_manifest is not None:
        files["dataset manifest"] = args.dataset_manifest
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    protected_directories = {"image directory": args.image_dir}
    if args.reference_model is not None:
        protected_directories["reference model"] = args.reference_model
    for label, path in protected_directories.items():
        if args.output.resolve().is_relative_to(path.resolve()):
            raise ValueError(f"triangulation output cannot be inside the {label}")
    if args.output.exists():
        if not args.output.is_dir():
            raise NotADirectoryError(f"triangulation output is not a directory: {args.output}")
        if any(args.output.iterdir()):
            raise FileExistsError(f"triangulation output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    lock = load_runtime_lock(args.runtime_lock)
    runtime = collect_runtime_evidence(
        lock, allow_unverified_vcs=args.allow_unverified_vcs
    )
    feature_runtime = json.loads(
        args.feature_runtime_manifest.read_text(encoding="utf-8")
    )
    feature_runtime_sha = verify_signed_runtime_manifest(
        feature_runtime, "runtime_manifest_sha256"
    )
    expected_inputs = {
        "pair_file_sha256": sha256_file(args.pairs),
        "features_sha256": sha256_file(args.features),
        "matches_sha256": sha256_file(args.matches),
        "runtime_lock_sha256": runtime_lock_sha256(lock),
    }
    for field, expected in expected_inputs.items():
        if feature_runtime.get(field) != expected:
            raise ValueError(
                f"feature runtime {field} does not match triangulation input"
            )
    if feature_runtime.get("runtime") != runtime:
        raise ValueError("feature extraction and triangulation runtime evidence differ")
    try:
        import pycolmap
        from hloc import triangulation
    except ImportError as exc:
        raise RuntimeError("locked HLoc triangulation runtime is unavailable") from exc
    pair_lines = [line.split() for line in args.pairs.read_text(encoding="utf-8").splitlines()]
    if not pair_lines or any(len(pair) != 2 for pair in pair_lines):
        raise ValueError("HLoc pairs must contain exactly two image names per line")
    training_names = {name for pair in pair_lines for name in pair}
    dataset_sha = None
    if args.dataset_manifest is not None:
        dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        dataset_sha = verify_dataset_manifest(dataset)
        reference = build_reference_model_from_manifest(dataset)
        reference_model_dir = args.output / "manifest_reference_model"
        reference_model_dir.mkdir()
        reference.write(reference_model_dir)
    else:
        reference_model_dir = args.reference_model
        reference = pycolmap.Reconstruction(reference_model_dir)
    training_reference = build_training_reference_model(reference, training_names)
    training_reference_dir = args.output / "training_reference_model"
    training_reference_dir.mkdir()
    training_reference.write(training_reference_dir)
    reconstruction = triangulation.main(
        args.output / "sfm",
        training_reference_dir,
        args.image_dir,
        args.pairs,
        args.features,
        args.matches,
        skip_geometric_verification=False,
        estimate_two_view_geometries=False,
        min_match_score=None,
        verbose=False,
        mapper_options=None,
    )
    if reconstruction.num_reg_images() < 2 or reconstruction.num_points3D() < 1:
        raise RuntimeError("HLoc triangulation produced no usable registered model")
    inputs = {
        "reference_model_sha256": directory_sha256(reference_model_dir),
        "training_reference_model_sha256": directory_sha256(
            training_reference_dir
        ),
        "pairs_sha256": sha256_file(args.pairs),
        "features_sha256": sha256_file(args.features),
        "matches_sha256": sha256_file(args.matches),
        "feature_runtime_manifest_sha256": feature_runtime_sha,
    }
    if args.dataset_manifest is not None:
        inputs["dataset_manifest_sha256"] = dataset_sha
        inputs["dataset_manifest_file_sha256"] = sha256_file(args.dataset_manifest)
    manifest = {
        "schema_version": 1,
        "algorithm_version": "hloc_known_pose_triangulation_v1",
        "runtime_lock_sha256": runtime_lock_sha256(lock),
        "runtime": runtime,
        "inputs": inputs,
        "policy": {
            "reference_source": (
                "dataset_manifest"
                if args.dataset_manifest is not None
                else "colmap_model"
            ),
            "skip_geometric_verification": False,
            "estimate_two_view_geometries": False,
            "verification": "known_pose_epipolar",
        },
        "output": {
            "sfm_model_sha256": directory_sha256(args.output / "sfm"),
            "registered_images": int(reconstruction.num_reg_images()),
            "training_images": len(training_names),
            "points3D": int(reconstruction.num_points3D()),
        },
    }
    manifest["triangulation_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    atomic_write(
        args.output / "triangulation_runtime_manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    print(
        f"HLoc triangulation: images={manifest['output']['registered_images']}, "
        f"points={manifest['output']['points3D']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
