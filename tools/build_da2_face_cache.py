#!/usr/bin/env python3
"""Build signed Face4 DA2 caches with MipMap-compatible metric alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from cloudstudio_3dgs.ba.training_manifest import directory_sha256
from cloudstudio_3dgs.data.depth_cache import load_sparse_depth, verify_depth_manifest
from cloudstudio_3dgs.data.face_warp import warp_sparse_depth_to_face
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest
from cloudstudio_3dgs.data.mono_depth import (
    MONO_DEPTH_KIND,
    MONO_DEPTH_SCHEMA_VERSION,
    fit_metric_affine_ransac,
    fit_metric_affine_ransac_torch,
    mono_depth_npz_bytes,
    sample_bilinear_at_source_pixels,
    sign_mono_depth_manifest,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec
from cloudstudio_3dgs.training.face_dataset import verify_face_manifest


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _camera(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = camera["intrinsic"]
    distortion = camera["distortion"]
    if distortion.get("camera_model") != "OPENCV_FISHEYE":
        raise ValueError("DA2 Face4 cache requires OPENCV_FISHEYE cameras")
    K = np.asarray(
        [
            [intrinsic["fl_x"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fl_y"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    params = distortion["params"]
    radial = np.asarray([params[f"k{index}"] for index in range(1, 5)])
    return K, radial


def _face(record: dict[str, Any]) -> FaceSpec:
    return FaceSpec(
        face_id=str(record["face_id"]),
        R_face=np.asarray(record["R_face"], dtype=np.float64),
        K_face=np.asarray(record["K_face"], dtype=np.float64),
        width=int(record["width"]),
        height=int(record["height"]),
        half_fov_deg=float(record["half_fov_deg"]),
    )


def _load_model(source: Path, checkpoint: Path, device: str):
    sys.path.insert(0, str(source.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--face-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--depth-manifest", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-faces", type=int)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = _read(args.dataset_manifest)
    dataset_sha = verify_dataset_manifest(dataset)
    depth_manifest = _read(args.depth_manifest)
    depth_sha = verify_depth_manifest(depth_manifest)
    face_manifest = _read(args.face_manifest)
    face_sha = verify_face_manifest(face_manifest)
    identity = face_manifest.get("source_identity", {})
    if identity.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("Face4 manifest is bound to a different dataset")
    if depth_manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("LiDAR depth is bound to a different dataset")
    if depth_manifest.get("complete_dataset") is not True:
        raise ValueError("LiDAR depth must cover the complete dataset")

    cameras = {str(item["camera_id"]): item for item in dataset["cameras"]}
    face_cameras = face_manifest["cameras"]
    depths = {str(item["image_id"]): item for item in depth_manifest["images"]}
    expected = sum(len(item["faces"]) for item in face_manifest["images"])
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for image in face_manifest["images"]:
        for face in image["faces"]:
            selected.append((image, face))
    if args.max_faces is not None:
        if args.max_faces <= 0:
            raise ValueError("max-faces must be positive")
        selected = selected[: args.max_faces]

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "depth"
    record_dir = args.output / "records"
    cache_dir.mkdir(exist_ok=True)
    record_dir.mkdir(exist_ok=True)
    checkpoint_sha = _sha256_file(args.checkpoint)
    source_sha = directory_sha256(args.model_source)
    model = _load_model(args.model_source, args.checkpoint, args.device)
    face_specs: dict[tuple[str, str], FaceSpec] = {}
    for camera_id, camera_faces in face_cameras.items():
        for record in camera_faces["faces"]:
            face_specs[(str(camera_id), str(record["face_id"]))] = _face(record)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (image_record, face_record) in enumerate(selected, start=1):
        image_id = str(image_record["image_id"])
        camera_id = str(image_record["camera_id"])
        face_id = str(face_record["face_id"])
        sample_id = f"{image_id}__{face_id}"
        relative_path = f"depth/{sample_id}.npz"
        destination = args.output / relative_path
        sidecar_path = record_dir / f"{sample_id}.json"
        if destination.is_file() and sidecar_path.is_file():
            existing = _read(sidecar_path)
            if (
                existing.get("sample_id") == sample_id
                and existing.get("cache_format_version") == 2
                and existing.get("source_rgb_sha256") == face_record["rgb_sha256"]
                and existing.get("source_mask_sha256") == face_record["mask_sha256"]
                and existing.get("checkpoint_sha256") == checkpoint_sha
                and existing.get("source_directory_sha256") == source_sha
                and existing.get("sha256") == _sha256_file(destination)
            ):
                records.append(existing)
                print(
                    f"DA2 {index}/{len(selected)} {sample_id}: resume "
                    f"align={existing['alignment']['valid']}",
                    flush=True,
                )
                continue
        rgb_path = args.face_root / str(face_record["rgb_path"])
        mask_path = args.face_root / str(face_record["mask_path"])
        if _sha256_file(rgb_path) != face_record["rgb_sha256"]:
            raise ValueError(f"Face4 RGB SHA mismatch for {sample_id}")
        if _sha256_file(mask_path) != face_record["mask_sha256"]:
            raise ValueError(f"Face4 mask SHA mismatch for {sample_id}")
        raw = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        face_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None or face_mask is None:
            raise FileNotFoundError(f"cannot decode Face4 inputs for {sample_id}")
        tensor, _ = model.image2tensor(raw, args.input_size)
        with torch.inference_mode():
            prediction = model(tensor.to(args.device))[0]
        disparity = prediction.detach().float().cpu().numpy()
        relative = np.divide(
            1.0,
            disparity,
            out=np.zeros_like(disparity, dtype=np.float32),
            where=disparity > 0.0,
        )

        depth_record = depths[image_id]
        sparse = load_sparse_depth(args.depth_root / str(depth_record["path"]))
        dense, confidence, valid = sparse.to_dense()
        camera_K, radial = _camera(cameras[camera_id])
        face_spec = face_specs[(camera_id, face_id)]
        face_range, _face_confidence, face_valid = warp_sparse_depth_to_face(
            dense, confidence, valid, camera_K, radial, face_spec
        )
        face_valid &= face_mask > 0
        ys, xs = np.nonzero(face_valid & np.isfinite(face_range) & (face_range > 0.0))
        mono_pairs = sample_bilinear_at_source_pixels(
            relative,
            xs.astype(np.float64),
            ys.astype(np.float64),
            source_shape=(face_spec.height, face_spec.width),
        )
        seed = int.from_bytes(
            hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "little"
        )
        alignment = (
            fit_metric_affine_ransac_torch(
                mono_pairs, face_range[ys, xs], seed=seed, device=args.device
            )
            if args.device.startswith("cuda")
            else fit_metric_affine_ransac(
                mono_pairs, face_range[ys, xs], seed=seed
            )
        )
        payload = mono_depth_npz_bytes(relative)
        _atomic_bytes(destination, payload)
        record = {
            "cache_format_version": 2,
            "sample_id": sample_id,
            "image_id": image_id,
            "camera_id": camera_id,
            "face_id": face_id,
            "path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_rgb_sha256": face_record["rgb_sha256"],
            "source_mask_sha256": face_record["mask_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "source_directory_sha256": source_sha,
            "source_shape": [face_spec.height, face_spec.width],
            "inference_shape": list(relative.shape),
            "positive_pixels": int(np.count_nonzero(relative > 0.0)),
            "alignment": alignment,
        }
        _atomic_json(sidecar_path, record)
        records.append(record)
        elapsed = time.perf_counter() - started
        print(
            f"DA2 {index}/{len(selected)} {sample_id}: "
            f"align={alignment['valid']} pairs={alignment['pair_count']} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    records.sort(key=lambda item: item["sample_id"])
    valid_count = sum(bool(item["alignment"]["valid"]) for item in records)
    manifest = sign_mono_depth_manifest(
        {
            "schema_version": MONO_DEPTH_SCHEMA_VERSION,
            "kind": MONO_DEPTH_KIND,
            "split": face_manifest["split"],
            "source_face_manifest_sha256": face_sha,
            "dataset_manifest_sha256": dataset_sha,
            "lidar_depth_manifest_sha256": depth_sha,
            "model": {
                "family": "Depth Anything V2 Small",
                "encoder": "vits",
                "license": "Apache-2.0",
                "checkpoint_sha256": checkpoint_sha,
                "source_directory_sha256": source_sha,
                "input_size": args.input_size,
                "native_output": True,
                "positive_output_transform": "reciprocal",
                "storage": "float16_clamped_to_65504",
            },
            "metric_alignment": {
                "relation": "lidar_range_m = scale * da2_relative_depth + shift",
                "minimum_positive_pairs": 1001,
                "ransac_iterations": 2000,
                "slope_range_open": [0.01, 100.0],
                "inlier_relative_error": 0.01,
                "minimum_inlier_ratio_open": 0.05,
                "final_refit": "affine_ordinary_least_squares_on_best_inliers",
            },
            "complete_face_cache": len(records) == expected,
            "expected_face_count": expected,
            "records": records,
            "summary": {
                "face_count": len(records),
                "valid_alignment_count": valid_count,
                "invalid_alignment_count": len(records) - valid_count,
            },
        }
    )
    _atomic_json(args.output / "mono_depth_manifest.json", manifest)
    print(
        f"DA2 cache: {len(records)}/{expected}, aligned={valid_count}, "
        f"complete={manifest['complete_face_cache']}, "
        f"sha={manifest['mono_depth_manifest_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
