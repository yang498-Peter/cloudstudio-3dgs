#!/usr/bin/env python3
"""Derive the mask artifacts of a subset dataset from its parent's artifacts.

Given a subset manifest written by ``subset_dataset_manifest.py`` this tool

* filters the parent's signed geometric mask manifest to the kept images,
  copies their PNGs, rebinds ``dataset_manifest_sha256`` to the subset and
  re-signs the result so downstream verifiers accept it;
* reports person-mask coverage from a directory of per-image PNGs (a
  ``person_masks/<image_id>.png`` layout, with or without a manifest), copying
  the PNGs that exist. No signed person manifest is produced: that needs the
  model identity and per-instance scores only ``build_person_masks`` records.

Geometric masks are per-camera circle templates, so copying them is exact;
they still have to be rebuilt after AT because the accepted training manifest
changes the dataset signature (SOP step ``rebased_masks_and_split``).

    python tools/subset_mask_artifacts.py \
        --dataset-manifest OUT/dataset_manifest.json \
        --mask-manifest PARENT_MASKS/mask_manifest.json \
        --person-mask-dir PARENT_PERSON/person_masks \
        --output-dir OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import (
    MASK_MANIFEST_NAME,
    verify_dataset_manifest,
    verify_mask_manifest,
)
from tools.subset_dataset_manifest import PARENT_HASH_KEY

PERSON_COVERAGE_NAME = "person_mask_coverage.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _copy_verified(source: Path, destination: Path, expected_sha256: str | None) -> str:
    digest = _sha256_file(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"mask PNG differs from its manifest record: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != digest:
            raise FileExistsError(f"existing subset mask differs: {destination}")
    else:
        shutil.copyfile(source, destination)
    return digest


def subset_mask_manifest(
    dataset_manifest: dict[str, Any],
    mask_manifest: dict[str, Any],
    *,
    mask_root: Path,
    output_dir: Path,
    copy_files: bool = True,
) -> dict[str, Any]:
    """Return the re-signed subset mask manifest (also written to disk)."""
    dataset_sha = verify_dataset_manifest(dataset_manifest)
    parent_mask_sha = verify_mask_manifest(mask_manifest)
    parent_dataset_sha = dataset_manifest.get("source_hashes", {}).get(PARENT_HASH_KEY)
    bound = mask_manifest.get("dataset_manifest_sha256")
    if bound not in (dataset_sha, parent_dataset_sha):
        raise ValueError(
            "mask manifest is bound to neither the subset nor its parent dataset"
        )
    kept = {str(image["image_id"]) for image in dataset_manifest.get("images", [])}
    records = []
    for record in mask_manifest.get("images", []):
        image_id = str(record["image_id"])
        if image_id not in kept:
            continue
        record = dict(record)
        if copy_files:
            for path_key, sha_key in (
                ("valid_mask_path", "valid_mask_sha256"),
                ("combined_mask_path", "combined_mask_sha256"),
                ("static_mask_path", None),
                ("depth_valid_mask_path", None),
            ):
                relative = record.get(path_key)
                if not relative:
                    continue
                _copy_verified(
                    mask_root / relative,
                    output_dir / relative,
                    record.get(sha_key) if sha_key else None,
                )
        records.append(record)
    missing = kept - {str(record["image_id"]) for record in records}
    if missing:
        raise ValueError(
            f"{len(missing)} subset images have no record in the parent mask manifest"
        )
    payload = dict(mask_manifest)
    payload.pop("mask_manifest_sha256", None)
    payload["dataset_manifest_sha256"] = dataset_sha
    payload["derived_from_mask_manifest_sha256"] = parent_mask_sha
    payload["images"] = records
    summary = dict(payload.get("summary") or {})
    summary["image_count"] = len(records)
    summary["per_image_paths_unique"] = len(
        {record["combined_mask_path"] for record in records}
    ) == len(records)
    payload["summary"] = summary
    payload["mask_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    verify_mask_manifest(payload)
    _write_json(output_dir / MASK_MANIFEST_NAME, payload)
    return payload


def person_mask_coverage(
    dataset_manifest: dict[str, Any],
    *,
    person_mask_dir: Path,
    output_dir: Path | None,
) -> dict[str, Any]:
    """Report which kept images have a ``person_masks/<image_id>.png``."""
    dataset_sha = verify_dataset_manifest(dataset_manifest)
    present, missing = [], []
    per_camera: dict[str, dict[str, int]] = {}
    for image in dataset_manifest.get("images", []):
        image_id = str(image["image_id"])
        camera = str(image["camera_id"])
        bucket = per_camera.setdefault(camera, {"present": 0, "missing": 0})
        source = person_mask_dir / f"{image_id}.png"
        if source.is_file():
            digest = _sha256_file(source)
            if output_dir is not None:
                _copy_verified(source, output_dir / "person_masks" / f"{image_id}.png", digest)
            present.append({"image_id": image_id, "person_mask_sha256": digest})
            bucket["present"] += 1
        else:
            missing.append(image_id)
            bucket["missing"] += 1
    total = len(present) + len(missing)
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset_sha,
        "person_mask_source_dir": str(person_mask_dir.resolve()),
        "signed_manifest": False,
        "note": (
            "coverage of pre-existing PNGs only; a signed person_mask_manifest.json "
            "must come from build_person_masks against the AT-accepted manifest"
        ),
        "summary": {
            "images": total,
            "present": len(present),
            "missing": len(missing),
            "present_fraction": (len(present) / total) if total else 0.0,
            "per_camera": per_camera,
        },
        "present": present,
        "missing_image_ids": missing,
    }
    if output_dir is not None:
        _write_json(output_dir / PERSON_COVERAGE_NAME, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--mask-manifest", type=Path, help="parent mask_manifest.json")
    parser.add_argument(
        "--person-mask-dir", type=Path, help="directory holding <image_id>.png person masks"
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="subset dataset directory")
    parser.add_argument(
        "--no-copy", action="store_true", help="write manifests/reports without copying PNGs"
    )
    args = parser.parse_args()
    if args.mask_manifest is None and args.person_mask_dir is None:
        parser.error("pass --mask-manifest and/or --person-mask-dir")

    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if args.mask_manifest is not None:
        masks = json.loads(args.mask_manifest.read_text(encoding="utf-8"))
        payload = subset_mask_manifest(
            dataset,
            masks,
            mask_root=args.mask_manifest.parent,
            output_dir=args.output_dir / "masks",
            copy_files=not args.no_copy,
        )
        print(
            f"masks: {payload['summary']['image_count']} records, "
            f"sha256={payload['mask_manifest_sha256']} -> {args.output_dir / 'masks' / MASK_MANIFEST_NAME}"
        )
    if args.person_mask_dir is not None:
        report = person_mask_coverage(
            dataset,
            person_mask_dir=args.person_mask_dir,
            output_dir=None if args.no_copy else args.output_dir / "person_masks",
        )
        if args.no_copy:
            _write_json(args.output_dir / "person_masks" / PERSON_COVERAGE_NAME, report)
        summary = report["summary"]
        print(
            f"person masks: {summary['present']}/{summary['images']} present, "
            f"{summary['missing']} missing -> {args.output_dir / 'person_masks' / PERSON_COVERAGE_NAME}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
