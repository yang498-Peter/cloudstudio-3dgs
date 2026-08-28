#!/usr/bin/env python3
"""Extract signed sparse Face4 LiDAR geometry from a depth-augmented cache.

The RGB Face4 manifest remains immutable so renderer masks, Tile crops, and
other image-only artifacts do not need to be rebuilt merely because LiDAR
supervision was added later.  ``tools/build_face_cache.py`` can populate the
depth NPZ files efficiently; this tool binds only those files to the original
RGB manifest and the accepted raw-fisheye LiDAR depth manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.depth_cache import (
    load_sparse_depth,
    verify_depth_manifest,
)
from cloudstudio_3dgs.data.face_lidar_geometry import (
    FACE_LIDAR_GEOMETRY_KIND,
    FACE_LIDAR_GEOMETRY_SCHEMA_VERSION,
    sign_face_lidar_geometry_manifest,
    verify_face_lidar_geometry_manifest,
)
from cloudstudio_3dgs.training.face_dataset import (
    SAMPLE_ID_SEPARATOR,
    verify_face_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(manifest: dict[str, Any]) -> dict[str, tuple[dict, dict]]:
    return {
        f"{image['image_id']}{SAMPLE_ID_SEPARATOR}{face['face_id']}": (image, face)
        for image in manifest["images"]
        for face in image["faces"]
    }


def _assert_rgb_identity(original: dict, augmented: dict) -> None:
    for key in ("face_plan", "fov_deg", "split", "cameras"):
        if original.get(key) != augmented.get(key):
            raise ValueError(f"augmented cache changed Face4 RGB geometry: {key}")
    left = _index(original)
    right = _index(augmented)
    if set(left) != set(right):
        raise ValueError("augmented cache changed the Face4 sample set")
    for sample_id in left:
        image_a, face_a = left[sample_id]
        image_b, face_b = right[sample_id]
        for key in ("camera_id", "rig_frame_id", "c2w"):
            if image_a.get(key) != image_b.get(key):
                raise ValueError(f"augmented cache changed {sample_id} {key}")
        for key in (
            "rgb_path",
            "rgb_sha256",
            "mask_path",
            "mask_sha256",
            "mask_true_pixels",
            "warp_valid_pixels",
        ):
            if face_a.get(key) != face_b.get(key):
                raise ValueError(f"augmented cache changed {sample_id} {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-face-manifest", required=True, type=Path)
    parser.add_argument("--augmented-face-manifest", required=True, type=Path)
    parser.add_argument("--face-root", required=True, type=Path)
    parser.add_argument("--source-depth-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    original = _read(args.source_face_manifest)
    augmented = _read(args.augmented_face_manifest)
    depth = _read(args.source_depth_manifest)
    face_sha = verify_face_manifest(original)
    verify_face_manifest(augmented)
    depth_sha = verify_depth_manifest(depth)
    _assert_rgb_identity(original, augmented)

    records: list[dict[str, Any]] = []
    with_depth = 0
    valid_total = 0
    for sample_id, (image, face) in _index(augmented).items():
        relative = face.get("depth_path")
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "image_id": str(image["image_id"]),
            "face_id": str(face["face_id"]),
            "path": None,
            "sha256": None,
            "valid_pixels": 0,
        }
        if relative:
            artifact = args.face_root / Path(str(relative))
            sparse = load_sparse_depth(artifact)
            valid_pixels = int(len(sparse.pixel_index))
            if valid_pixels <= 0:
                raise ValueError(f"empty sparse depth artifact: {artifact}")
            actual_sha = _sha256_file(artifact)
            if actual_sha != str(face.get("depth_sha256")):
                raise ValueError(f"depth SHA256 mismatch for {sample_id}")
            record.update(
                path=Path(str(relative)).as_posix(),
                sha256=actual_sha,
                valid_pixels=valid_pixels,
            )
            with_depth += 1
            valid_total += valid_pixels
        records.append(record)

    payload = {
        "schema_version": FACE_LIDAR_GEOMETRY_SCHEMA_VERSION,
        "kind": FACE_LIDAR_GEOMETRY_KIND,
        "split": original["split"],
        "source_face_manifest_sha256": face_sha,
        "source_depth_manifest_sha256": depth_sha,
        "dataset_manifest_sha256": original.get("source_identity", {}).get(
            "dataset_manifest_sha256"
        ),
        "complete_face_cache": True,
        "expected_face_count": len(records),
        "depth_semantics": "euclidean_ray_range_m_sparse_real_lidar_only",
        "mesh_interpolation": False,
        "records": records,
        "summary": {
            "face_count": len(records),
            "with_depth_count": with_depth,
            "without_depth_count": len(records) - with_depth,
            "valid_pixel_count": valid_total,
        },
    }
    signed = sign_face_lidar_geometry_manifest(payload)
    verify_face_lidar_geometry_manifest(signed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(signed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Face4 LiDAR geometry: faces={len(records)}, with_depth={with_depth}, "
        f"valid_pixels={valid_total}, sha256="
        f"{signed['face_lidar_geometry_manifest_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
