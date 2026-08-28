#!/usr/bin/env python3
"""Bind existing immutable person-mask artifacts to a new geometric mask layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import (
    verify_dataset_manifest,
    verify_mask_manifest,
)
from cloudstudio_3dgs.data.person_masks import (
    PERSON_MASK_MANIFEST_NAME,
    verify_person_mask_manifest,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-mask-manifest", required=True, type=Path)
    parser.add_argument("--base-mask-manifest", required=True, type=Path)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help=(
            "optional pose/intrinsic-refined dataset; immutable person artifacts "
            "are rebound only when image IDs, paths, SHA256 values, and cameras match"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        if not args.output.is_dir():
            raise NotADirectoryError(f"output is not a directory: {args.output}")
        if any(args.output.iterdir()):
            raise FileExistsError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.person_mask_manifest.read_text(encoding="utf-8"))
    base = json.loads(args.base_mask_manifest.read_text(encoding="utf-8"))
    source_sha = verify_person_mask_manifest(source)
    base_sha = verify_mask_manifest(base)
    target_dataset_sha = str(base.get("dataset_manifest_sha256", ""))
    if args.dataset_manifest is None:
        if source.get("dataset_manifest_sha256") != target_dataset_sha:
            raise ValueError("person and geometric masks are bound to different datasets")
    else:
        target_dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        verified_target_sha = verify_dataset_manifest(target_dataset)
        if verified_target_sha != target_dataset_sha:
            raise ValueError("geometric masks are bound to a different target dataset")
        target_images = {
            str(item["image_id"]): item for item in target_dataset.get("images", [])
        }
        if len(target_images) != len(target_dataset.get("images", [])):
            raise ValueError("target dataset repeats an image ID")
        for record in source.get("images", []):
            image_id = str(record["image_id"])
            target = target_images.get(image_id)
            if target is None:
                raise ValueError(f"target dataset omits person-mask image {image_id}")
            expected = {
                "source_image_path": str(target["path"]),
                "source_image_sha256": str(target["sha256"]),
                "camera_id": str(target["camera_id"]),
            }
            actual = {key: str(record.get(key, "")) for key in expected}
            if actual != expected:
                raise ValueError(
                    f"person-mask source identity differs for {image_id}: "
                    f"expected={expected}, actual={actual}"
                )
        if set(target_images) != {
            str(item["image_id"]) for item in source.get("images", [])
        }:
            raise ValueError("person masks do not cover the complete target dataset")
    source_ids = {str(item["image_id"]) for item in source["images"]}
    base_ids = {str(item["image_id"]) for item in base["images"]}
    if source_ids != base_ids:
        raise ValueError("person and geometric masks do not cover identical images")

    output = deepcopy(source)
    output.pop("person_mask_manifest_sha256", None)
    output["algorithm_version"] = "independent_person_dynamic_mask_rebound_v2"
    output["dataset_manifest_sha256"] = target_dataset_sha
    output["source_person_mask_manifest_sha256"] = source_sha
    output["base_mask_manifest_sha256"] = base_sha
    output["composition"] = "circle_valid & ~person_dynamic_mask"
    output["depth_composition"] = "circle_depth_valid & ~person_dynamic_mask"
    output["person_mask_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(output)
    ).hexdigest()
    _atomic_write(
        args.output / PERSON_MASK_MANIFEST_NAME,
        (json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    print(
        f"Rebound person masks: images={len(output['images'])}, "
        f"sha256={output['person_mask_manifest_sha256']} -> "
        f"{args.output / PERSON_MASK_MANIFEST_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
