"""Reproject accepted AT tracks into derived Face4 views for spatial tiling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.adaptive_tiling import ProjectedObservationTable
from cloudstudio_3dgs.training.face_dataset import verify_face_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _face_samples(manifest: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for image in manifest["images"]:
        camera_id = str(image["camera_id"])
        specs = {
            str(spec["face_id"]): spec
            for spec in manifest["cameras"][camera_id]["faces"]
        }
        for face in image["faces"]:
            face_id = str(face["face_id"])
            result[f"{image['image_id']}::{face_id}"] = {
                "image": image,
                "face": face,
                "spec": specs[face_id],
                "root": root,
            }
    return result


def build_face4_projected_observations(
    candidate_model_path: Path,
    dataset_manifest_path: Path,
    face_manifests: Iterable[tuple[Path, Path]],
    output_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Project only AT-observed 3D tracks into Face4, preserving overlap views."""

    import pycolmap

    candidate_model_path = Path(candidate_model_path).resolve()
    dataset_manifest_path = Path(dataset_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for path in (output_path, manifest_path):
        if path.exists() and not force:
            raise FileExistsError(f"refusing to replace Face4 observations: {path}")
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    image_id_by_name = {
        str(image["path"]).removeprefix("camera/"): str(image["image_id"])
        for image in dataset["images"]
    }
    sample_sets: list[tuple[str, dict[str, Any], str]] = []
    face_bindings: dict[str, str] = {}
    for face_manifest_path, face_root in face_manifests:
        face_manifest_path = Path(face_manifest_path).resolve()
        face_root = Path(face_root).resolve()
        face = json.loads(face_manifest_path.read_text(encoding="utf-8"))
        face_sha = verify_face_manifest(face)
        split = str(face.get("split", ""))
        if split in face_bindings:
            raise ValueError(f"duplicate Face4 split: {split}")
        face_bindings[split] = face_sha
        for sample_id, payload in _face_samples(face, face_root).items():
            sample_sets.append((sample_id, payload, split))
    if not sample_sets:
        raise ValueError("no Face4 samples were supplied")
    sample_sets.sort(key=lambda row: (row[2] != "train", row[0]))
    samples_by_image: dict[str, list[tuple[int, dict[str, Any], str]]] = {}
    image_sizes = []
    train_image_indices: list[int] = []
    for index, (sample_id, payload, split) in enumerate(sample_sets):
        image_id = sample_id.split("::", 1)[0]
        samples_by_image.setdefault(image_id, []).append((index, payload, split))
        spec = payload["spec"]
        image_sizes.append([int(spec["width"]), int(spec["height"])])
        if split == "train":
            train_image_indices.append(index)
    train_remap = {source: target for target, source in enumerate(train_image_indices)}

    reconstruction = pycolmap.Reconstruction(str(candidate_model_path))
    point_ids = sorted(int(value) for value in reconstruction.points3D)
    point_index = {point_id: index for index, point_id in enumerate(point_ids)}
    points = np.asarray(
        [reconstruction.points3D[point_id].xyz for point_id in point_ids],
        dtype=np.float64,
    )
    all_xy: list[np.ndarray] = []
    all_image: list[np.ndarray] = []
    all_point: list[np.ndarray] = []
    train_xy: list[np.ndarray] = []
    train_image: list[np.ndarray] = []
    train_point: list[np.ndarray] = []
    mask_cache: dict[Path, np.ndarray] = {}
    missing_names: list[str] = []
    for image in reconstruction.images.values():
        image_id = image_id_by_name.get(str(image.name))
        if image_id is None:
            missing_names.append(str(image.name))
            continue
        rows = samples_by_image.get(image_id, [])
        observed_ids = [int(point.point3D_id) for point in image.points2D if point.has_point3D()]
        if not observed_ids:
            continue
        indices = np.asarray([point_index[value] for value in observed_ids], dtype=np.int64)
        xyz = points[indices]
        for derived_index, payload, split in rows:
            source = payload["image"]
            spec = payload["spec"]
            c2w = np.asarray(source["c2w"], dtype=np.float64)
            parent = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
            face_coordinates = parent @ np.asarray(spec["R_face"], dtype=np.float64)
            positive = face_coordinates[:, 2] > 1e-6
            K = np.asarray(spec["K_face"], dtype=np.float64)
            uv = np.empty((len(xyz), 2), dtype=np.float64)
            uv[:, 0] = K[0, 0] * face_coordinates[:, 0] / np.maximum(face_coordinates[:, 2], 1e-12) + K[0, 2]
            uv[:, 1] = K[1, 1] * face_coordinates[:, 1] / np.maximum(face_coordinates[:, 2], 1e-12) + K[1, 2]
            width, height = int(spec["width"]), int(spec["height"])
            keep = positive & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            mask_path = (Path(payload["root"]) / payload["face"]["mask_path"]).resolve()
            mask = mask_cache.get(mask_path)
            if mask is None:
                with Image.open(mask_path) as source_mask:
                    mask = np.asarray(source_mask, dtype=np.uint8) > 0
                mask_cache[mask_path] = mask
            valid_indices = np.flatnonzero(keep)
            if len(valid_indices):
                x = np.clip(np.floor(uv[valid_indices, 0]).astype(np.int64), 0, width - 1)
                y = np.clip(np.floor(uv[valid_indices, 1]).astype(np.int64), 0, height - 1)
                valid_indices = valid_indices[mask[y, x]]
            if not len(valid_indices):
                continue
            selected_xy = uv[valid_indices]
            selected_points = indices[valid_indices]
            all_xy.append(selected_xy)
            all_image.append(np.full(len(valid_indices), derived_index, dtype=np.int64))
            all_point.append(selected_points)
            if split == "train":
                train_xy.append(selected_xy)
                train_image.append(np.full(len(valid_indices), train_remap[derived_index], dtype=np.int64))
                train_point.append(selected_points)
    if missing_names:
        raise ValueError(f"candidate model images are absent from dataset manifest: {missing_names[:3]}")

    def concatenate(rows: list[np.ndarray], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.concatenate(rows, axis=0) if rows else np.empty(shape, dtype=dtype)

    arrays = {
        "points": points,
        "all_observation_xy": concatenate(all_xy, (0, 2), np.float64),
        "all_observation_image": concatenate(all_image, (0,), np.int64),
        "all_observation_point": concatenate(all_point, (0,), np.int64),
        "all_image_sizes": np.asarray(image_sizes, dtype=np.int64),
        "train_observation_xy": concatenate(train_xy, (0, 2), np.float64),
        "train_observation_image": concatenate(train_image, (0,), np.int64),
        "train_observation_point": concatenate(train_point, (0,), np.int64),
        "train_image_sizes": np.asarray(image_sizes, dtype=np.int64)[train_image_indices],
    }
    all_table = ProjectedObservationTable(
        points, arrays["all_observation_xy"], arrays["all_observation_image"],
        arrays["all_observation_point"], arrays["all_image_sizes"]
    ).validated()
    train_table = ProjectedObservationTable(
        points, arrays["train_observation_xy"], arrays["train_observation_image"],
        arrays["train_observation_point"], arrays["train_image_sizes"]
    ).validated()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "accepted_at_tracks_reprojected_to_face4_v1",
        "candidate_model_path": str(candidate_model_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "face_manifest_sha256_by_split": face_bindings,
        "point_count": len(points),
        "all_view_count": len(image_sizes),
        "train_view_count": len(train_image_indices),
        "all_observation_count": len(all_table.observation_xy),
        "train_observation_count": len(train_table.observation_xy),
        "all_view_ids": [row[0] for row in sample_sets],
        "train_view_ids": [sample_sets[index][0] for index in train_image_indices],
        "all_observation_table_sha256": all_table.sha256(),
        "train_observation_table_sha256": train_table.sha256(),
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "bytes": output_path.stat().st_size,
    }
    payload["face4_observation_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def load_face4_projected_observations(path: Path) -> tuple[ProjectedObservationTable, ProjectedObservationTable]:
    with np.load(Path(path), allow_pickle=False) as payload:
        points = np.asarray(payload["points"])
        all_table = ProjectedObservationTable(
            points,
            np.asarray(payload["all_observation_xy"]),
            np.asarray(payload["all_observation_image"]),
            np.asarray(payload["all_observation_point"]),
            np.asarray(payload["all_image_sizes"]),
        ).validated()
        train_table = ProjectedObservationTable(
            points,
            np.asarray(payload["train_observation_xy"]),
            np.asarray(payload["train_observation_image"]),
            np.asarray(payload["train_observation_point"]),
            np.asarray(payload["train_image_sizes"]),
        ).validated()
    return all_table, train_table
