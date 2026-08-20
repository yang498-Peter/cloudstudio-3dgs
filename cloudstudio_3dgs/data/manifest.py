from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .s1_reader import (
    find_point_cloud,
    list_camera_images,
    load_cameras,
    load_imgpose_images,
    sha256_file,
)
from .schema import DatasetManifest


MANIFEST_NAME = "dataset_manifest.json"


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_manifest(
    recording_dir: Path,
    run_dir: Path,
    *,
    hash_images: bool = True,
    hash_point_cloud: bool = True,
) -> dict[str, Any]:
    recording_dir = recording_dir.resolve()
    run_dir = run_dir.resolve()
    calibration_path = recording_dir / "info" / "calibration.json"
    imgpose_path = run_dir / "ImgPose.txt"
    for required in (calibration_path, imgpose_path):
        if not required.is_file():
            raise FileNotFoundError(f"required S1 input is missing: {required}")

    point_cloud_path = find_point_cloud(run_dir)
    cameras = load_cameras(calibration_path)
    images = load_imgpose_images(
        imgpose_path,
        recording_dir,
        hash_images=hash_images,
    )
    raw_image_paths = list_camera_images(recording_dir)
    posed_by_relative = {
        image.path.removeprefix("camera/"): image for image in images
    }
    unposed_images = sorted(set(raw_image_paths) - set(posed_by_relative))
    image_set_digest = hashlib.sha256()
    for relative in raw_image_paths:
        image = posed_by_relative.get(relative)
        content_hash = image.sha256 if image else (
            sha256_file(recording_dir / "camera" / Path(relative))
            if hash_images
            else "not_computed"
        )
        image_set_digest.update(relative.encode("utf-8"))
        image_set_digest.update(b"\0")
        image_set_digest.update(content_hash.encode("ascii"))
        image_set_digest.update(b"\n")

    point_cloud_hash = sha256_file(point_cloud_path) if hash_point_cloud else "not_computed"
    point_cloud_relative = point_cloud_path.relative_to(run_dir).as_posix()
    source_hashes = {
        "recording:info/calibration.json": sha256_file(calibration_path),
        "run:ImgPose.txt": sha256_file(imgpose_path),
        f"run:{point_cloud_relative}": point_cloud_hash,
        "recording:camera-image-set": image_set_digest.hexdigest(),
    }
    warnings = ["rig_pairing_pending_pr02"]
    if not hash_images:
        warnings.append("image_content_hashes_not_computed")
    if not hash_point_cloud:
        warnings.append("point_cloud_content_hash_not_computed")
    if unposed_images:
        warnings.append(f"unposed_camera_images:{len(unposed_images)}")
    manifest = DatasetManifest(
        recording_id=recording_dir.name,
        source_hashes=source_hashes,
        cameras=cameras,
        images=images,
        point_cloud={
            "path_root": "run",
            "path": point_cloud_relative,
            "size_bytes": point_cloud_path.stat().st_size,
            "sha256": point_cloud_hash,
            "coordinate_frame": "s1_local",
        },
        unposed_images=[f"camera/{path}" for path in unposed_images],
        warnings=warnings,
    ).to_dict()
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


def write_manifest_atomic(
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    force: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --force to overwrite the manifest"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / MANIFEST_NAME
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{MANIFEST_NAME}.", suffix=".tmp", dir=output_dir
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a canonical MVP S1 dataset manifest")
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-image-hashes", action="store_true")
    parser.add_argument("--skip-point-cloud-hash", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        args.recording,
        args.run,
        hash_images=not args.skip_image_hashes,
        hash_point_cloud=not args.skip_point_cloud_hash,
    )
    destination = write_manifest_atomic(manifest, args.output, force=args.force)
    print(
        f"manifest: {len(manifest['images'])} images, {len(manifest['cameras'])} cameras, "
        f"sha256={manifest['manifest_sha256']} -> {destination}"
    )
    if args.skip_image_hashes or args.skip_point_cloud_hash:
        print("WARNING: one or more content hashes were skipped; this manifest is not archival-grade")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
