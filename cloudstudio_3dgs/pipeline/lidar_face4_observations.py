"""Build adaptive-tiling observations from accepted LiDAR depth and Face4."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import verify_depth_manifest
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.geometry.kb4 import unproject_kb4
from cloudstudio_3dgs.pipeline.adaptive_tiling import ProjectedObservationTable
from cloudstudio_3dgs.training.face_dataset import verify_face_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lidar_face4_projected_observations(
    dataset_manifest_path: Path,
    depth_manifest_path: Path,
    depth_root: Path,
    face_manifests: Iterable[tuple[Path, Path]],
    output_path: Path,
    *,
    samples_per_raw_view: int = 5_000,
    force: bool = False,
) -> dict[str, Any]:
    """Use visible LiDAR z-buffer samples, not visual SfM points, as anchors."""

    if samples_per_raw_view < 100:
        raise ValueError("LiDAR tile planning requires at least 100 samples per raw view")
    dataset_manifest_path = Path(dataset_manifest_path).resolve()
    depth_manifest_path = Path(depth_manifest_path).resolve()
    depth_root = Path(depth_root).resolve()
    output_path = Path(output_path).resolve()
    output_manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for path in (output_path, output_manifest_path):
        if path.exists() and not force:
            raise FileExistsError(f"refusing to replace LiDAR Face4 observations: {path}")
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    depth = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
    depth_sha = verify_depth_manifest(depth)
    if depth.get("dataset_manifest_sha256") != dataset.get("manifest_sha256"):
        raise ValueError("LiDAR depth is bound to a different accepted dataset")
    images = {str(row["image_id"]): row for row in dataset["images"]}
    cameras = {str(row["camera_id"]): row for row in dataset["cameras"]}

    samples: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    face_bindings: dict[str, str] = {}
    for face_manifest_path, face_root in face_manifests:
        face = json.loads(Path(face_manifest_path).read_text(encoding="utf-8"))
        face_sha = verify_face_manifest(face)
        split = str(face.get("split", ""))
        face_bindings[split] = face_sha
        specs = {
            (str(camera_id), str(spec["face_id"])): spec
            for camera_id, payload in face["cameras"].items()
            for spec in payload["faces"]
        }
        for image in face["images"]:
            for face_row in image["faces"]:
                face_id = str(face_row["face_id"])
                sample_id = f"{image['image_id']}::{face_id}"
                samples.append(
                    (
                        sample_id,
                        image,
                        {
                            **face_row,
                            "spec": specs[(str(image["camera_id"]), face_id)],
                            "root": str(Path(face_root).resolve()),
                        },
                        split,
                    )
                )
    samples.sort(key=lambda row: (row[3] != "train", row[0]))
    samples_by_image: dict[str, list[tuple[int, dict[str, Any], str]]] = {}
    all_sizes: list[list[int]] = []
    train_indices: list[int] = []
    for index, (sample_id, image, face, split) in enumerate(samples):
        samples_by_image.setdefault(str(image["image_id"]), []).append((index, face, split))
        spec = face["spec"]
        all_sizes.append([int(spec["width"]), int(spec["height"])])
        if split == "train":
            train_indices.append(index)
    train_remap = {source: target for target, source in enumerate(train_indices)}

    points: list[np.ndarray] = []
    all_xy: list[np.ndarray] = []
    all_image: list[np.ndarray] = []
    all_point: list[np.ndarray] = []
    train_xy: list[np.ndarray] = []
    train_image: list[np.ndarray] = []
    train_point: list[np.ndarray] = []
    point_offset = 0
    sampled_raw_pixels = 0
    for depth_row in depth["images"]:
        image_id = str(depth_row["image_id"])
        source = images[image_id]
        camera = cameras[str(source["camera_id"])]
        with np.load(depth_root / depth_row["path"], allow_pickle=False) as payload:
            pixel_index = np.asarray(payload["pixel_index"], dtype=np.int64)
            ranges = np.asarray(payload["range_m"], dtype=np.float64)
            shape = tuple(int(value) for value in payload["shape"])
        count = min(samples_per_raw_view, len(pixel_index))
        selection = np.linspace(0, len(pixel_index) - 1, count, dtype=np.int64)
        selected_pixels = pixel_index[selection]
        selected_ranges = ranges[selection]
        height, width = shape
        pixel_xy = np.column_stack(
            [selected_pixels % width + 0.5, selected_pixels // width + 0.5]
        )
        rays = unproject_kb4(
            pixel_xy,
            camera["intrinsic"],
            camera["distortion"]["params"],
        )
        c2w = np.asarray(source["c2w"], dtype=np.float64)
        world = (rays * selected_ranges[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]
        points.append(world)
        global_indices = np.arange(point_offset, point_offset + count, dtype=np.int64)
        point_offset += count
        sampled_raw_pixels += count
        for derived_index, face, split in samples_by_image[image_id]:
            spec = face["spec"]
            parent = (world - c2w[:3, 3]) @ c2w[:3, :3]
            face_coordinates = parent @ np.asarray(spec["R_face"], dtype=np.float64)
            positive = face_coordinates[:, 2] > 1e-6
            K = np.asarray(spec["K_face"], dtype=np.float64)
            uv = np.column_stack(
                [
                    K[0, 0] * face_coordinates[:, 0] / np.maximum(face_coordinates[:, 2], 1e-12) + K[0, 2],
                    K[1, 1] * face_coordinates[:, 1] / np.maximum(face_coordinates[:, 2], 1e-12) + K[1, 2],
                ]
            )
            face_width, face_height = int(spec["width"]), int(spec["height"])
            keep = positive & (uv[:, 0] >= 0) & (uv[:, 0] < face_width) & (uv[:, 1] >= 0) & (uv[:, 1] < face_height)
            mask_path = Path(face["root"]) / face["mask_path"]
            with Image.open(mask_path) as opened:
                mask = np.asarray(opened, dtype=np.uint8) > 0
            chosen = np.flatnonzero(keep)
            if len(chosen):
                x = np.floor(uv[chosen, 0]).astype(np.int64)
                y = np.floor(uv[chosen, 1]).astype(np.int64)
                chosen = chosen[mask[y, x]]
            if not len(chosen):
                continue
            all_xy.append(uv[chosen])
            all_image.append(np.full(len(chosen), derived_index, dtype=np.int64))
            all_point.append(global_indices[chosen])
            if split == "train":
                train_xy.append(uv[chosen])
                train_image.append(np.full(len(chosen), train_remap[derived_index], dtype=np.int64))
                train_point.append(global_indices[chosen])

    def joined(rows: list[np.ndarray], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.concatenate(rows) if rows else np.empty(shape, dtype=dtype)

    xyz = np.concatenate(points).astype(np.float64)
    arrays = {
        "points": xyz,
        "all_observation_xy": joined(all_xy, (0, 2), np.float64),
        "all_observation_image": joined(all_image, (0,), np.int64),
        "all_observation_point": joined(all_point, (0,), np.int64),
        "all_image_sizes": np.asarray(all_sizes, dtype=np.int64),
        "train_observation_xy": joined(train_xy, (0, 2), np.float64),
        "train_observation_image": joined(train_image, (0,), np.int64),
        "train_observation_point": joined(train_point, (0,), np.int64),
        "train_image_sizes": np.asarray(all_sizes, dtype=np.int64)[train_indices],
    }
    all_table = ProjectedObservationTable(xyz, arrays["all_observation_xy"], arrays["all_observation_image"], arrays["all_observation_point"], arrays["all_image_sizes"]).validated()
    train_table = ProjectedObservationTable(xyz, arrays["train_observation_xy"], arrays["train_observation_image"], arrays["train_observation_point"], arrays["train_image_sizes"]).validated()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "accepted_lidar_depth_reprojected_to_face4_v1",
        "dataset_manifest_sha256": dataset.get("manifest_sha256"),
        "lidar_depth_manifest_sha256": depth_sha,
        "point_cloud_sha256": depth.get("point_cloud_sha256"),
        "face_manifest_sha256_by_split": face_bindings,
        "sampling": {
            "method": "uniform_over_sorted_valid_zbuffer_pixels",
            "maximum_samples_per_raw_view": samples_per_raw_view,
            "sampled_raw_pixels": sampled_raw_pixels,
        },
        "point_count": len(xyz),
        "all_view_count": len(all_sizes),
        "train_view_count": len(train_indices),
        "all_observation_count": len(all_table.observation_xy),
        "train_observation_count": len(train_table.observation_xy),
        "all_observation_table_sha256": all_table.sha256(),
        "train_observation_table_sha256": train_table.sha256(),
        "all_view_ids": [row[0] for row in samples],
        "train_view_ids": [samples[index][0] for index in train_indices],
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "bytes": output_path.stat().st_size,
    }
    manifest["face4_observation_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    output_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_lidar_face4_projected_observations(path: Path) -> tuple[ProjectedObservationTable, ProjectedObservationTable]:
    with np.load(Path(path), allow_pickle=False) as payload:
        points = np.asarray(payload["points"])
        all_table = ProjectedObservationTable(points, np.asarray(payload["all_observation_xy"]), np.asarray(payload["all_observation_image"]), np.asarray(payload["all_observation_point"]), np.asarray(payload["all_image_sizes"])).validated()
        train_table = ProjectedObservationTable(points, np.asarray(payload["train_observation_xy"]), np.asarray(payload["train_observation_image"]), np.asarray(payload["train_observation_point"]), np.asarray(payload["train_image_sizes"])).validated()
    return all_table, train_table
