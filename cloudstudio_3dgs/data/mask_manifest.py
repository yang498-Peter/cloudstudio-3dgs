"""Deterministic per-image mask artifacts for MVP S1 datasets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .manifest import canonical_json_bytes


MASK_MANIFEST_NAME = "mask_manifest.json"
SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def verify_dataset_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("manifest_sha256", ""))
    if not expected:
        raise ValueError("dataset manifest has no manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"dataset manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    return actual


def verify_mask_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("mask_manifest_sha256", ""))
    if not expected:
        raise ValueError("mask manifest has no mask_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("mask_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"mask manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    records = manifest.get("images", [])
    image_ids = [str(record.get("image_id", "")) for record in records]
    paths = [str(record.get("combined_mask_path", "")) for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("mask manifest contains duplicate image IDs")
    if len(paths) != len(set(paths)):
        raise ValueError("mask manifest contains shared combined-mask paths")
    if not records or not all(image_ids) or not all(paths):
        raise ValueError("mask manifest contains incomplete image records")
    return actual


def generate_fisheye_valid_mask(
    camera: dict[str, Any],
    *,
    theta_max_deg: float = 95.0,
    radius_px: float | None = None,
) -> np.ndarray:
    if camera.get("camera_type") != "fisheye":
        raise ValueError(f"camera {camera.get('camera_id')} is not fisheye")
    distortion = camera.get("distortion", {})
    if distortion.get("camera_model") != "OPENCV_FISHEYE":
        raise ValueError(
            f"camera {camera.get('camera_id')} does not use OPENCV_FISHEYE"
        )
    if not 0.0 < theta_max_deg <= 180.0:
        raise ValueError("theta_max_deg must be in (0, 180]")

    width = int(camera["width"])
    height = int(camera["height"])
    intrinsic = camera["intrinsic"]
    params = distortion["params"]
    fx = float(intrinsic["fl_x"])
    fy = float(intrinsic["fl_y"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])
    if min(width, height) <= 0 or min(fx, fy) <= 0.0:
        raise ValueError("camera dimensions and focal lengths must be positive")

    yy, xx = np.mgrid[0:height, 0:width]
    if radius_px is not None:
        radius_px = float(radius_px)
        if not np.isfinite(radius_px) or radius_px <= 0.0:
            raise ValueError("radius_px must be finite and positive")
        return np.hypot(xx - cx, yy - cy) <= radius_px

    theta = np.deg2rad(theta_max_deg)
    theta2 = theta * theta
    distorted_radius = theta * (
        1.0
        + float(params["k1"]) * theta2
        + float(params["k2"]) * theta2**2
        + float(params["k3"]) * theta2**3
        + float(params["k4"]) * theta2**4
    )
    if not np.isfinite(distorted_radius) or distorted_radius <= 0.0:
        raise ValueError("camera distortion produces an invalid FoV radius")
    normalized_radius = np.hypot((xx - cx) / fx, (yy - cy) / fy)
    return normalized_radius <= distorted_radius


def _png_bytes(mask: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(
        stream, format="PNG", optimize=False, compress_level=9
    )
    return stream.getvalue()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
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


def build_per_image_masks(
    dataset_manifest: dict[str, Any],
    output_dir: Path,
    *,
    theta_max_deg: float = 95.0,
    valid_radius_px: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write one independently addressable valid mask for every posed image."""
    dataset_sha256 = verify_dataset_manifest(dataset_manifest)
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"mask output is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"mask output is not empty: {output_dir}; pass --force to replace owned files"
        )
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    cameras = {str(camera["camera_id"]): camera for camera in dataset_manifest["cameras"]}
    if len(cameras) != len(dataset_manifest["cameras"]):
        raise ValueError("dataset manifest contains duplicate camera IDs")
    templates: dict[str, tuple[bytes, float, str]] = {}
    for camera_id, camera in sorted(cameras.items()):
        mask = generate_fisheye_valid_mask(
            camera,
            theta_max_deg=theta_max_deg,
            radius_px=valid_radius_px,
        )
        payload = _png_bytes(mask)
        templates[camera_id] = (
            payload,
            float(np.mean(mask)),
            hashlib.sha256(payload).hexdigest(),
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for image in sorted(dataset_manifest["images"], key=lambda item: str(item["image_id"])):
        image_id = str(image["image_id"])
        if image_id in seen:
            raise ValueError(f"duplicate image_id in dataset manifest: {image_id}")
        if not SAFE_IMAGE_ID.fullmatch(image_id):
            raise ValueError(f"unsafe image_id for mask path: {image_id!r}")
        seen.add(image_id)
        camera_id = str(image["camera_id"])
        if camera_id not in templates:
            raise ValueError(f"image {image_id} references unknown camera {camera_id}")
        payload, valid_fraction, digest = templates[camera_id]
        relative = f"masks/{image_id}.png"
        destination = output_dir / Path(relative)
        if force or not destination.exists():
            _write_bytes_atomic(destination, payload)
        elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"existing mask differs: {destination}")
        records.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "source_image_path_root": image["path_root"],
                "source_image_path": image["path"],
                "valid_mask_path": relative,
                "static_mask_path": None,
                "depth_valid_mask_path": None,
                "combined_mask_path": relative,
                "valid_mask_sha256": digest,
                "combined_mask_sha256": digest,
                "valid_fraction": valid_fraction,
                "static_mask_policy": "identity_pending_pr06",
                "depth_valid_policy": "not_applied_pending_pr07",
            }
        )

    if not records:
        raise ValueError("dataset manifest contains no images")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset_sha256,
        "mask_composition": "fisheye_valid & static_mask & optional_depth_valid",
        "valid_mask_profile": (
            "principal_point_circle_v1"
            if valid_radius_px is not None
            else "analytic_kb4_theta_cap_v1"
        ),
        "theta_max_deg": None if valid_radius_px is not None else theta_max_deg,
        "valid_radius_px": valid_radius_px,
        "path_root": "mask_output",
        "images": records,
        "summary": {
            "image_count": len(records),
            "camera_count": len(cameras),
            "per_image_paths_unique": len({record["combined_mask_path"] for record in records})
            == len(records),
            "static_masks": "identity_pending_pr06",
            "depth_valid_masks": "not_applied_pending_pr07",
        },
    }
    payload["mask_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    manifest_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(output_dir / MASK_MANIFEST_NAME, manifest_bytes)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--theta-max-deg", type=float, default=95.0)
    parser.add_argument(
        "--valid-radius-px",
        type=float,
        help="circular valid-region radius in full-resolution source pixels",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = build_per_image_masks(
        manifest,
        args.output,
        theta_max_deg=args.theta_max_deg,
        valid_radius_px=args.valid_radius_px,
        force=args.force,
    )
    print(
        f"mask manifest: {output['summary']['image_count']} per-image masks, "
        f"sha256={output['mask_manifest_sha256']} -> {args.output / MASK_MANIFEST_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
