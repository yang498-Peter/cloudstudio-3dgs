#!/usr/bin/env python3
"""Build the offline fisheye face-split training cache.

For every training image the S1 dataset already composes (RGB + combined mask
+ optional person mask + optional sparse LiDAR range), warp it onto the
pinhole faces planned by ``cloudstudio_3dgs.geometry.fisheye_faces`` and store
per-(image, face) artifacts plus a signed ``face_manifest.json`` that
``cloudstudio_3dgs.training.face_dataset.FaceCacheDataset`` replays.

Artifacts under ``--output``:
    faces/{image_id}_{face_id}_rgb.png    uint8 RGB, bilinear warp
    faces/{image_id}_{face_id}_mask.png   0/255; warped source mask (conservative
                                          4-neighbor semantics) AND face FoV
                                          validity AND face_weight > 0.02
    faces/{image_id}_{face_id}_depth.npz  SparseDepthMap-compatible layout
                                          (source_index=-1 / support_count=0
                                          sentinels; forward-splat, z-buffered)
    face_manifest.json                    signed manifest (schema below)

The continuous seam-fusion ``face_weight`` ramp is NOT cached: it is pure face
geometry, so readers recompute it from the manifest's FaceSpec when blending.

Fail-closed: a missing/corrupt source artifact aborts the run (raised by
S1TrainingDataset); a face whose warp output is entirely invalid, or whose
final supervision mask is empty, is skipped and listed under ``skipped``.

Idempotent: re-running against the same output root skips (image, face)
entries whose artifact files already exist (RGB warp is the expensive part);
the manifest is rebuilt and re-signed every run. A pre-existing manifest with
a different FoV/face plan aborts instead of mixing caches.

CPU multiprocessing only (Windows spawn-safe); never imports torch.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import sparse_depth_npz_bytes
from cloudstudio_3dgs.data.face_warp import (
    FaceWarpGrid,
    build_face_warp_grid,
    warp_image_to_face,
    warp_mask_to_face,
    warp_sparse_depth_to_face,
)
from cloudstudio_3dgs.geometry.fisheye_faces import (
    FaceSpec,
    face_weight,
    plan_fisheye_faces,
)
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap
from cloudstudio_3dgs.training.dataset import S1TrainingDataset, TrainingSample
from cloudstudio_3dgs.training.face_dataset import (
    FACE_CACHE_SCHEMA_VERSION,
    FACE_MANIFEST_NAME,
    SAMPLE_ID_SEPARATOR,
    sign_face_manifest,
    verify_face_manifest,
)


MIN_FACE_WEIGHT = 0.02
FACES_SUBDIR = "faces"

# Sentinels for face depth caches: the fisheye source point index is
# meaningless after the face re-projection, and no support statistic exists
# (mirrors the trainer's adjusted-cache convention around
# ``sparse_depth_npz_bytes``).
DEPTH_SOURCE_INDEX_SENTINEL = -1
DEPTH_SUPPORT_COUNT_SENTINEL = 0


def _check_id_component(value: str, label: str) -> str:
    value = str(value)
    if not value or any(ch in value for ch in "/\\:*?\"<>|") or ".." in value:
        raise ValueError(f"{label} is not filename-safe: {value!r}")
    if SAMPLE_ID_SEPARATOR in value:
        raise ValueError(f"{label} must not contain {SAMPLE_ID_SEPARATOR!r}: {value!r}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _png_bytes(array: np.ndarray, mode: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _face_depth_arrays(
    sample: TrainingSample, face: FaceSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Forward-splat the sample's sparse range map onto ``face``; None if the
    sample carries no depth."""
    if sample.depth_range_m is None:
        return None
    depth = np.asarray(sample.depth_range_m, dtype=np.float64)
    if sample.depth_confidence is None:
        confidence = np.ones_like(depth)
    else:
        confidence = np.asarray(sample.depth_confidence, dtype=np.float64)
    if sample.depth_mask is not None:
        valid = np.asarray(sample.depth_mask, dtype=bool)
    else:
        valid = np.isfinite(depth) & (depth > 0.0)
    valid = valid & np.isfinite(confidence) & (confidence > 0.0)
    return warp_sparse_depth_to_face(
        depth,
        confidence,
        valid,
        np.asarray(sample.K, dtype=np.float64),
        np.asarray(sample.radial_coeffs, dtype=np.float64),
        face,
    )


def _sparse_depth_bytes(
    face_range: np.ndarray, face_conf: np.ndarray, face_valid: np.ndarray
) -> bytes | None:
    keep = face_valid & np.isfinite(face_range) & (face_range > 0.0)
    keep &= np.isfinite(face_conf) & (face_conf > 0.0) & (face_conf <= 1.0)
    pixel_index = np.flatnonzero(keep).astype(np.int32)
    if pixel_index.size == 0:
        return None
    count = pixel_index.size
    sparse = SparseDepthMap(
        shape=(int(face_range.shape[0]), int(face_range.shape[1])),
        pixel_index=pixel_index,
        range_m=face_range.reshape(-1)[pixel_index].astype(np.float32),
        confidence=face_conf.reshape(-1)[pixel_index].astype(np.float32),
        source_index=np.full(count, DEPTH_SOURCE_INDEX_SENTINEL, dtype=np.int64),
        support_count=np.full(count, DEPTH_SUPPORT_COUNT_SENTINEL, dtype=np.int32),
    )
    return sparse_depth_npz_bytes(sparse)


def _face_weight_keep_mask(
    face: FaceSpec, min_face_weight: float
) -> np.ndarray:
    jj, ii = np.meshgrid(
        np.arange(face.width, dtype=np.float64) + 0.5,
        np.arange(face.height, dtype=np.float64) + 0.5,
    )
    pixels = np.stack([jj.reshape(-1), ii.reshape(-1)], axis=1)
    weights = face_weight(face, pixels).reshape(face.height, face.width)
    return weights > min_face_weight


def process_sample(
    sample: TrainingSample,
    faces: Sequence[FaceSpec],
    output_root: Path,
    *,
    fov_deg: float,
    grids: dict[str, FaceWarpGrid] | None = None,
    min_face_weight: float = MIN_FACE_WEIGHT,
    skip_existing: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Produce every face artifact for one training sample.

    ``grids`` is a per-camera cache (face_id -> FaceWarpGrid) that the caller
    keeps alive across images of the same camera; it is filled lazily.
    Returns the manifest image record and the skipped-face entries.
    """
    output_root = Path(output_root)
    faces_dir = output_root / FACES_SUBDIR
    faces_dir.mkdir(parents=True, exist_ok=True)
    if grids is None:
        grids = {}
    image_id = _check_id_component(sample.image_id, "image_id")
    max_theta_rad = float(np.radians(fov_deg / 2.0))
    K = np.asarray(sample.K, dtype=np.float64)
    radial = np.asarray(sample.radial_coeffs, dtype=np.float64)

    face_entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for face in faces:
        face_id = _check_id_component(face.face_id, "face_id")
        stem = f"{image_id}_{face_id}"
        rgb_rel = f"{FACES_SUBDIR}/{stem}_rgb.png"
        mask_rel = f"{FACES_SUBDIR}/{stem}_mask.png"
        depth_rel = f"{FACES_SUBDIR}/{stem}_depth.npz"
        rgb_path = faces_dir / f"{stem}_rgb.png"
        mask_path = faces_dir / f"{stem}_mask.png"
        depth_path = faces_dir / f"{stem}_depth.npz"

        grid = grids.get(face_id)
        if grid is None:
            grid = build_face_warp_grid(K, radial, face, max_theta_rad=max_theta_rad)
            grids[face_id] = grid

        entry: dict[str, Any] = {
            "face_id": face_id,
            "rgb_path": rgb_rel,
            "mask_path": mask_rel,
            "depth_path": None,
            "depth_sha256": None,
        }

        if skip_existing and rgb_path.is_file() and mask_path.is_file():
            # Idempotent fast path: trust the finished RGB/mask files (the
            # expensive part) and only refresh depth if it is missing.
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8) > 0
            entry["rgb_sha256"] = _sha256_file(rgb_path)
            entry["mask_sha256"] = _sha256_file(mask_path)
            entry["mask_true_pixels"] = int(np.count_nonzero(mask))
            src_h, src_w = np.asarray(sample.image).shape[:2]
            x0 = np.floor(grid.u).astype(np.int64)
            y0 = np.floor(grid.v).astype(np.int64)
            inb = (x0 >= 0) & (x0 + 1 <= src_w - 1) & (y0 >= 0) & (y0 + 1 <= src_h - 1)
            entry["warp_valid_pixels"] = int(np.count_nonzero(grid.fov_valid & inb))
            if depth_path.is_file():
                entry["depth_path"] = depth_rel
                entry["depth_sha256"] = _sha256_file(depth_path)
            else:
                splat = _face_depth_arrays(sample, face)
                if splat is not None:
                    payload = _sparse_depth_bytes(*splat)
                    if payload is not None:
                        _atomic_write(depth_path, payload)
                        entry["depth_path"] = depth_rel
                        entry["depth_sha256"] = _sha256_bytes(payload)
            face_entries.append(entry)
            continue

        face_mask, geom_valid = warp_mask_to_face(
            np.asarray(sample.rgb_mask, dtype=bool),
            K,
            radial,
            face,
            grid=grid,
            max_theta_rad=max_theta_rad,
        )
        if not np.any(geom_valid):
            skipped.append(
                {"image_id": image_id, "face_id": face_id, "reason": "warp_all_invalid"}
            )
            continue
        final_mask = face_mask & _face_weight_keep_mask(face, min_face_weight)
        if not np.any(final_mask):
            skipped.append(
                {"image_id": image_id, "face_id": face_id, "reason": "empty_mask"}
            )
            continue

        face_image, _valid = warp_image_to_face(
            np.asarray(sample.image),
            K,
            radial,
            face,
            interpolation="bilinear",
            grid=grid,
            max_theta_rad=max_theta_rad,
        )
        face_rgb = np.clip(np.rint(face_image), 0.0, 255.0).astype(np.uint8)
        rgb_payload = _png_bytes(face_rgb, "RGB")
        mask_payload = _png_bytes(final_mask.astype(np.uint8) * 255, "L")
        _atomic_write(rgb_path, rgb_payload)
        _atomic_write(mask_path, mask_payload)
        entry["rgb_sha256"] = _sha256_bytes(rgb_payload)
        entry["mask_sha256"] = _sha256_bytes(mask_payload)
        entry["mask_true_pixels"] = int(np.count_nonzero(final_mask))
        entry["warp_valid_pixels"] = int(np.count_nonzero(geom_valid))

        splat = _face_depth_arrays(sample, face)
        if splat is not None:
            payload = _sparse_depth_bytes(*splat)
            if payload is not None:
                _atomic_write(depth_path, payload)
                entry["depth_path"] = depth_rel
                entry["depth_sha256"] = _sha256_bytes(payload)
        face_entries.append(entry)

    record = {
        "image_id": image_id,
        "camera_id": str(sample.camera_id),
        "rig_frame_id": str(sample.rig_frame_id),
        "c2w": np.asarray(sample.c2w, dtype=np.float64).tolist(),
        "faces": face_entries,
    }
    return record, skipped


# ------------------------------- multiprocessing --------------------------------

_WORKER: dict[str, Any] = {}


def _dataset_from_kwargs(kwargs: Mapping[str, Any]) -> S1TrainingDataset:
    def path_or_none(key: str) -> Path | None:
        value = kwargs.get(key)
        return None if value is None else Path(value)

    return S1TrainingDataset(
        dataset_manifest_path=Path(kwargs["dataset_manifest_path"]),
        recording_root=Path(kwargs["recording_root"]),
        mask_manifest_path=Path(kwargs["mask_manifest_path"]),
        mask_root=Path(kwargs["mask_root"]),
        split_manifest_path=Path(kwargs["split_manifest_path"]),
        split=kwargs["split"],
        person_mask_manifest_path=path_or_none("person_mask_manifest_path"),
        person_mask_root=path_or_none("person_mask_root"),
        depth_manifest_path=path_or_none("depth_manifest_path"),
        depth_root=path_or_none("depth_root"),
        factor=1,
        verify_artifacts=bool(kwargs["verify_artifacts"]),
    )


def _worker_init(
    dataset_kwargs: dict[str, Any],
    faces_serialized: dict[str, list[dict[str, Any]]],
    output_root: str,
    fov_deg: float,
    skip_existing: bool,
) -> None:
    _WORKER["dataset"] = _dataset_from_kwargs(dataset_kwargs)
    _WORKER["faces"] = {
        camera_id: [FaceSpec.from_dict(payload) for payload in payloads]
        for camera_id, payloads in faces_serialized.items()
    }
    _WORKER["grids"] = {}
    _WORKER["output_root"] = Path(output_root)
    _WORKER["fov_deg"] = float(fov_deg)
    _WORKER["skip_existing"] = bool(skip_existing)


def _worker_process(index: int) -> dict[str, Any]:
    dataset: S1TrainingDataset = _WORKER["dataset"]
    sample = dataset[index]
    camera_id = str(sample.camera_id)
    faces = _WORKER["faces"].get(camera_id)
    if faces is None:
        raise ValueError(f"no planned faces for camera {camera_id!r}")
    grids = _WORKER["grids"].setdefault(camera_id, {})
    record, skipped = process_sample(
        sample,
        faces,
        _WORKER["output_root"],
        fov_deg=_WORKER["fov_deg"],
        grids=grids,
        skip_existing=_WORKER["skip_existing"],
    )
    return {"record": record, "skipped": skipped}


# ------------------------------------ main ---------------------------------------


def _camera_calibration(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    intrinsic = camera["intrinsic"]
    K = np.array(
        [
            [float(intrinsic["fl_x"]), 0.0, float(intrinsic["cx"])],
            [0.0, float(intrinsic["fl_y"]), float(intrinsic["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    params = camera["distortion"]["params"]
    radial = np.asarray([float(params[f"k{i}"]) for i in range(1, 5)], dtype=np.float64)
    return K, radial, (int(camera["width"]), int(camera["height"]))


def _check_existing_manifest(
    output: Path, fov_deg: float, faces_serialized: dict[str, list[dict[str, Any]]]
) -> None:
    manifest_path = output / FACE_MANIFEST_NAME
    if not manifest_path.is_file():
        return
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_face_manifest(existing)
    same_fov = float(existing.get("fov_deg", -1.0)) == float(fov_deg)
    existing_faces = {
        camera_id: entry["faces"] for camera_id, entry in existing.get("cameras", {}).items()
    }
    if not same_fov or json.dumps(existing_faces, sort_keys=True) != json.dumps(
        faces_serialized, sort_keys=True
    ):
        raise SystemExit(
            f"existing {manifest_path} was built with a different FoV/face plan; "
            "use a fresh --output directory instead of mixing caches"
        )


def build_manifest_payload(
    *,
    fov_deg: float,
    split: str,
    source_identity: Mapping[str, Any],
    faces_serialized: Mapping[str, list[dict[str, Any]]],
    records: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
    min_face_weight: float = MIN_FACE_WEIGHT,
) -> dict[str, Any]:
    """Assemble the unsigned face-cache manifest payload (shared with tests)."""
    face_count = sum(len(record["faces"]) for record in records)
    return {
        "schema_version": FACE_CACHE_SCHEMA_VERSION,
        "kind": "fisheye_face_cache",
        "fov_deg": float(fov_deg),
        "split": str(split),
        "min_face_weight": float(min_face_weight),
        "pixel_convention": "pixel_center_plus_half",
        "mask_semantics": "warped_source_mask_and_fov_valid_and_face_weight_gt_min",
        "depth_semantics": "euclidean_ray_range_m",
        "depth_sentinels": {
            "source_index": DEPTH_SOURCE_INDEX_SENTINEL,
            "support_count": DEPTH_SUPPORT_COUNT_SENTINEL,
        },
        "source_identity": dict(source_identity),
        "cameras": {
            camera_id: {"faces": list(payloads)}
            for camera_id, payloads in faces_serialized.items()
        },
        "images": list(records),
        "skipped": sorted(
            skipped, key=lambda item: (item["image_id"], item["face_id"])
        ),
        "summary": {
            "image_count": len(records),
            "face_sample_count": face_count,
            "skipped_count": len(skipped),
            "with_depth_count": sum(
                1
                for record in records
                for entry in record["faces"]
                if entry["depth_path"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--recording-root", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="face cache root")
    parser.add_argument("--person-mask-manifest", type=Path)
    parser.add_argument("--person-mask-root", type=Path)
    parser.add_argument("--depth-manifest", type=Path)
    parser.add_argument("--depth-root", type=Path)
    parser.add_argument("--fov-deg", type=float, default=190.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (multiprocessing.cpu_count() or 2) - 2),
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip SHA256 verification of source artifacts in workers",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="rebuild every artifact even when its files already exist",
    )
    args = parser.parse_args()

    dataset_kwargs = {
        "dataset_manifest_path": str(args.dataset_manifest),
        "recording_root": str(args.recording_root),
        "mask_manifest_path": str(args.mask_manifest),
        "mask_root": str(args.mask_root),
        "split_manifest_path": str(args.split_manifest),
        "split": args.split,
        "person_mask_manifest_path": None
        if args.person_mask_manifest is None
        else str(args.person_mask_manifest),
        "person_mask_root": None
        if args.person_mask_root is None
        else str(args.person_mask_root),
        "depth_manifest_path": None
        if args.depth_manifest is None
        else str(args.depth_manifest),
        "depth_root": None if args.depth_root is None else str(args.depth_root),
        "verify_artifacts": not args.no_verify,
    }
    dataset = _dataset_from_kwargs(dataset_kwargs)

    cameras = {str(c["camera_id"]): c for c in dataset.dataset_manifest["cameras"]}
    used_camera_ids = sorted(set(dataset.camera_id_by_image.values()))
    faces_serialized: dict[str, list[dict[str, Any]]] = {}
    for camera_id in used_camera_ids:
        K, radial, image_size = _camera_calibration(cameras[camera_id])
        faces = plan_fisheye_faces(args.fov_deg, K, radial, image_size)
        faces_serialized[camera_id] = [face.to_dict() for face in faces]
        print(
            f"camera {camera_id}: {len(faces)} faces, "
            f"sizes {sorted({face.width for face in faces})}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _check_existing_manifest(output, args.fov_deg, faces_serialized)

    skip_existing = not args.no_skip_existing
    total = len(dataset)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if args.workers <= 1:
        _worker_init(
            dataset_kwargs, faces_serialized, str(output), args.fov_deg, skip_existing
        )
        results = map(_worker_process, range(total))
        _finish = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(
                dataset_kwargs,
                faces_serialized,
                str(output),
                args.fov_deg,
                skip_existing,
            ),
        )
        results = pool.imap(_worker_process, range(total), chunksize=2)
        _finish = pool
    try:
        for done, result in enumerate(results, start=1):
            records.append(result["record"])
            skipped.extend(result["skipped"])
            if done % 25 == 0 or done == total:
                print(f"[{done}/{total}] images cached", flush=True)
    finally:
        if _finish is not None:
            _finish.close()
            _finish.join()

    payload = build_manifest_payload(
        fov_deg=args.fov_deg,
        split=args.split,
        source_identity=dataset.identity,
        faces_serialized=faces_serialized,
        records=records,
        skipped=skipped,
    )
    face_count = payload["summary"]["face_sample_count"]
    manifest = sign_face_manifest(payload)
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(output / FACE_MANIFEST_NAME, text.encode("utf-8"))
    print(
        f"face cache: {len(records)} images x faces -> {face_count} samples, "
        f"{len(skipped)} skipped, sha256={manifest['face_manifest_sha256']} -> "
        f"{output / FACE_MANIFEST_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
