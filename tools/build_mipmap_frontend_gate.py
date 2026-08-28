#!/usr/bin/env python3
"""Verify and sign the mandatory MipMap-aligned front-half pipeline chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.ba.training_manifest import (
    directory_sha256,
    verify_independent_at_report,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import (
    verify_dataset_manifest,
    verify_mask_manifest,
)
from cloudstudio_3dgs.data.person_masks import verify_person_mask_manifest
from cloudstudio_3dgs.evaluation.splits import verify_split_manifest
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    FRONTEND_READY_STATUS,
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    INDEPENDENT_AT_ALGORITHM,
    ORDERED_STAGES,
    sign_gate,
)
from cloudstudio_3dgs.training.face_dataset import verify_face_manifest


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_signed(payload: dict[str, Any], field: str) -> str:
    expected = str(payload.get(field, ""))
    if len(expected) != 64:
        raise ValueError(f"artifact is unsigned: missing {field}")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(f"artifact signature mismatch: {field}")
    return expected


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--time-sync-report", required=True, type=Path)
    parser.add_argument("--raw-circle-mask", required=True, type=Path)
    parser.add_argument("--raw-person-mask", required=True, type=Path)
    parser.add_argument("--feature-runtime", required=True, type=Path)
    parser.add_argument("--triangulation-runtime", required=True, type=Path)
    parser.add_argument("--at-report", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--training-dataset", required=True, type=Path)
    parser.add_argument("--training-circle-mask", required=True, type=Path)
    parser.add_argument("--training-person-mask", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--face4-train", required=True, type=Path)
    parser.add_argument("--face4-val", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = _read(args.raw_dataset)
    raw_sha = verify_dataset_manifest(raw)
    raw_image_count = len(raw.get("images", []))
    if raw_image_count <= 0 or len(raw.get("cameras", [])) != 2:
        raise ValueError(
            "MipMap-aligned profile requires a non-empty dual-fisheye dataset"
        )

    sync = _read(args.time_sync_report)
    if sync.get("schema_version") != "camera-time-sync-render-sweep-1.0":
        raise ValueError("time-sync report uses an unsupported schema")
    if sync.get("base_dataset_manifest_sha256") != raw_sha:
        raise ValueError("time-sync report is bound to another raw dataset")
    if abs(float(sync.get("best_offset_ms", float("inf")))) > 1e-9:
        raise ValueError(
            "time-sync audit selected a non-zero offset; correct the timestamps and rerun"
        )

    raw_mask = _read(args.raw_circle_mask)
    raw_mask_sha = verify_mask_manifest(raw_mask)
    if raw_mask.get("dataset_manifest_sha256") != raw_sha:
        raise ValueError("raw circle mask is bound to another dataset")
    if raw_mask.get("valid_mask_profile") != "principal_point_circle_v1" or not (
        abs(float(raw_mask.get("valid_radius_px", 0.0)) - 1200.0) <= 1e-9
    ):
        raise ValueError("raw mask must be the fixed 1200 px principal-point circle")

    raw_person = _read(args.raw_person_mask)
    raw_person_sha = verify_person_mask_manifest(raw_person)
    if (
        raw_person.get("dataset_manifest_sha256") != raw_sha
        or raw_person.get("base_mask_manifest_sha256") != raw_mask_sha
        or raw_person.get("composition") != "circle_valid & ~person_dynamic_mask"
    ):
        raise ValueError("raw person masks are not composed with the raw circle mask")

    feature = _read(args.feature_runtime)
    feature_sha = _verify_signed(feature, "runtime_manifest_sha256")
    feature_filter = feature.get("feature_filter", {})
    if (
        not feature.get("cuda_used")
        or int(feature.get("image_count", 0)) != raw_image_count
        or feature_filter.get("dataset_manifest_sha256") != raw_sha
        or feature_filter.get("mask_manifest_sha256") != raw_mask_sha
        or feature_filter.get("person_mask_manifest_sha256") != raw_person_sha
        or feature_filter.get("composition") != "circle_valid & ~person_dynamic"
    ):
        raise ValueError("masked feature runtime identity is incomplete or mismatched")

    triangulation = _read(args.triangulation_runtime)
    triangulation_sha = _verify_signed(
        triangulation, "triangulation_manifest_sha256"
    )
    if (
        triangulation.get("inputs", {}).get("feature_runtime_manifest_sha256")
        != feature_sha
        or int(triangulation.get("output", {}).get("registered_images", 0))
        != raw_image_count
    ):
        raise ValueError("triangulation is not the complete masked-feature product")

    at = _read(args.at_report)
    at_sha = verify_independent_at_report(at)
    candidate_sha = directory_sha256(args.candidate_model.resolve())
    if (
        at.get("algorithm_version") != INDEPENDENT_AT_ALGORITHM
        or at.get("dataset_manifest_sha256") != raw_sha
        or at.get("candidate_model_sha256") != candidate_sha
        or at.get("triangulation_identity", {}).get(
            "triangulation_manifest_sha256"
        )
        != triangulation_sha
        or at.get("triangulation_identity", {}).get(
            "feature_runtime_manifest_sha256"
        )
        != feature_sha
        or not at.get("intrinsic_outer_converged")
    ):
        raise ValueError("AT did not converge from the verified masked triangulation")

    training = _read(args.training_dataset)
    training_sha = verify_dataset_manifest(training)
    lineage = training.get("training_lineage", {})
    if (
        lineage.get("base_dataset_manifest_sha256") != raw_sha
        or lineage.get("independent_at_report_sha256") != at_sha
        or lineage.get("independent_at_candidate_model_sha256") != candidate_sha
        or lineage.get("independent_at_algorithm_version") != INDEPENDENT_AT_ALGORITHM
    ):
        raise ValueError("training dataset is not published from the accepted AT")

    training_mask = _read(args.training_circle_mask)
    training_mask_sha = verify_mask_manifest(training_mask)
    if (
        training_mask.get("dataset_manifest_sha256") != training_sha
        or training_mask.get("valid_mask_profile") != "principal_point_circle_v1"
        or abs(float(training_mask.get("valid_radius_px", 0.0)) - 1200.0) > 1e-9
    ):
        raise ValueError("training circle masks were not rebuilt after AT")
    training_person = _read(args.training_person_mask)
    training_person_sha = verify_person_mask_manifest(training_person)
    if (
        training_person.get("dataset_manifest_sha256") != training_sha
        or training_person.get("base_mask_manifest_sha256") != training_mask_sha
    ):
        raise ValueError("training person masks were not rebound after AT")
    split = _read(args.split_manifest)
    split_sha = verify_split_manifest(split)
    if split.get("dataset_manifest_sha256") != training_sha:
        raise ValueError("train/validation split was not rebuilt after AT")
    train_image_count = len(split.get("splits", {}).get("train", []))
    val_image_count = len(split.get("splits", {}).get("val", []))
    if (
        train_image_count <= 0
        or val_image_count <= 0
        or train_image_count + val_image_count != raw_image_count
    ):
        raise ValueError("train/validation split does not cover every raw image once")

    face_train = _read(args.face4_train)
    face_train_sha = verify_face_manifest(face_train)
    face_val = _read(args.face4_val)
    face_val_sha = verify_face_manifest(face_val)
    common_identity = {
        "dataset_manifest_sha256": training_sha,
        "mask_manifest_sha256": training_mask_sha,
        "person_mask_manifest_sha256": training_person_sha,
        "split_manifest_sha256": split_sha,
    }
    for label, face, expected_split, images in (
        ("train", face_train, "train", train_image_count),
        ("val", face_val, "val", val_image_count),
    ):
        identity = face.get("source_identity", {})
        if (
            face.get("face_plan") != "mipmap_face4"
            or face.get("split") != expected_split
            or any(identity.get(key) != value for key, value in common_identity.items())
            or int(face.get("summary", {}).get("image_count", 0)) != images
            or int(face.get("summary", {}).get("face_sample_count", 0)) != images * 4
            or int(face.get("summary", {}).get("skipped_count", -1)) != 0
        ):
            raise ValueError(f"{label} Face4 cache is incomplete or mismatched")

    evidence_paths = {
        "raw_dataset": args.raw_dataset,
        "time_sync_report": args.time_sync_report,
        "raw_circle_mask": args.raw_circle_mask,
        "raw_person_mask": args.raw_person_mask,
        "feature_runtime": args.feature_runtime,
        "triangulation_runtime": args.triangulation_runtime,
        "at_report": args.at_report,
        "candidate_model": args.candidate_model,
        "training_dataset": args.training_dataset,
        "training_circle_mask": args.training_circle_mask,
        "training_person_mask": args.training_person_mask,
        "split_manifest": args.split_manifest,
        "face4_train": args.face4_train,
        "face4_val": args.face4_val,
    }
    evidence = {}
    for name, path in evidence_paths.items():
        resolved = path.resolve()
        evidence[name] = {
            "path": str(resolved),
            "sha256": directory_sha256(resolved)
            if resolved.is_dir()
            else _file_sha(resolved),
        }
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "profile": GATE_PROFILE,
        "status": FRONTEND_READY_STATUS,
        "training_allowed": False,
        "completed_stages": list(ORDERED_STAGES[:10]),
        "next_required_stage": ORDERED_STAGES[10],
        "blocking_reasons": [
            "Face4 renderer dynamic mask has not been verified",
            "LiDAR depth has not been rebuilt from the accepted AT",
            "DA2 depth, independent sky, and spatial Tile plan are not complete",
        ],
        "bindings": {
            "raw_dataset_manifest_sha256": raw_sha,
            "training_dataset_manifest_sha256": training_sha,
            "training_circle_mask_manifest_sha256": training_mask_sha,
            "training_person_mask_manifest_sha256": training_person_sha,
            "split_manifest_sha256": split_sha,
            "face4_train_manifest_sha256": face_train_sha,
            "face4_val_manifest_sha256": face_val_sha,
            "at_report_sha256": at_sha,
            "candidate_model_sha256": candidate_sha,
        },
        "evidence": evidence,
    }
    gate = sign_gate(payload)
    _atomic_json(args.output, gate)
    print(
        f"MipMap front-half gate: status={gate['status']}, "
        f"training_allowed={gate['training_allowed']}, "
        f"next={gate['next_required_stage']}, "
        f"sha256={gate['gate_manifest_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
