#!/usr/bin/env python3
"""Read-only MipMap voxel/image correlation audit for the completed snow task.

The MipMap customer task and all source data are read-only.  This script writes
only a UTF-8 JSON summary and a UTF-8 CSV under ``results/diagnostics``.

The saved undistorted JPEGs were deleted by MipMap at successful completion
(``keep_undistort_images=false``).  Instead of pretending they still exist, the
script reconstructs only the required 11x11 local patches.  It uses the exact
zero-distortion face camera from ``mvs_undistort.xml`` and maps those face pixels
back to the original fisheye image with the optimized physical camera in
``mvs.xml``.  The physical fisheye projection is independently checked against
the saved AT tie-point measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr


TASK_ROOT_DEFAULT = Path(
    r"D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827"
)
OUTPUT_ROOT_DEFAULT = Path(r"G:\cloudstudio-3dgs\results\diagnostics")
VOXEL_SIZE_M = 0.5
MIN_LAS_PER_VOXEL = 100
PCA_K = 24
GS_SAMPLE_PER_TILE = 75_000
PATCH_SIZE = 11
MIN_NORMAL_SAMPLES_PER_VOXEL = 5
MIN_IMAGE_OBS_FOR_TEXTURE = 3


@dataclass(frozen=True)
class PhysicalPhoto:
    photo_id: int
    camera_id: int
    image_path: Path
    center: np.ndarray
    rotation_w2c: np.ndarray
    width: int
    height: int
    focal_px: float
    cx: float
    cy: float
    distortion: np.ndarray


@dataclass(frozen=True)
class FaceView:
    group_index: int
    derived_photo_id: int
    source: PhysicalPhoto
    center: np.ndarray
    rotation_w2c: np.ndarray
    width: int
    height: int
    focal_px: float
    cx: float
    cy: float
    near_depth: float
    far_depth: float


def matrix_from_photo(photo: ET.Element) -> np.ndarray:
    return np.asarray(
        [
            [float(photo.findtext(f"./Pose/Rotation/M_{r}{c}")) for c in range(3)]
            for r in range(3)
        ],
        dtype=np.float64,
    )


def center_from_photo(photo: ET.Element) -> np.ndarray:
    return np.asarray(
        [float(photo.findtext(f"./Pose/Center/{axis}")) for axis in "xyz"],
        dtype=np.float64,
    )


def load_cameras(task_root: Path) -> tuple[dict[int, PhysicalPhoto], dict[int, list[FaceView]], ET.Element]:
    physical_root = ET.parse(task_root / "result" / "AT" / "mvs.xml").getroot()
    face_root = ET.parse(task_root / "result" / "AT" / "mvs_undistort.xml").getroot()
    physical_groups = physical_root.findall(".//Photogroup")
    face_groups = face_root.findall(".//Photogroup")
    if len(physical_groups) != 2 or len(face_groups) != 8:
        raise ValueError(
            f"unexpected camera layout: physical={len(physical_groups)}, face={len(face_groups)}"
        )

    physical_by_id: dict[int, PhysicalPhoto] = {}
    physical_by_camera: dict[int, list[PhysicalPhoto]] = defaultdict(list)
    for group_index, group in enumerate(physical_groups):
        camera_id = group_index + 1
        width = int(group.findtext("./ImageDimensions/Width"))
        height = int(group.findtext("./ImageDimensions/Height"))
        focal = float(group.findtext("FocalLengthPixels"))
        cx = float(group.findtext("./PrincipalPoint/x"))
        cy = float(group.findtext("./PrincipalPoint/y"))
        distortion = np.asarray(
            [float(group.findtext(f"./Distortion/K{i}")) for i in range(1, 5)],
            dtype=np.float64,
        )
        for photo in group.findall("./Photo"):
            item = PhysicalPhoto(
                photo_id=int(photo.findtext("Id")),
                camera_id=camera_id,
                image_path=Path(str(photo.findtext("ImagePath"))),
                center=center_from_photo(photo),
                rotation_w2c=matrix_from_photo(photo),
                width=width,
                height=height,
                focal_px=focal,
                cx=cx,
                cy=cy,
                distortion=distortion,
            )
            physical_by_id[item.photo_id] = item
            physical_by_camera[camera_id].append(item)

    views_by_source: dict[int, list[FaceView]] = defaultdict(list)
    for group_index, group in enumerate(face_groups):
        camera_id = group_index // 4 + 1
        width = int(group.findtext("./ImageDimensions/Width"))
        height = int(group.findtext("./ImageDimensions/Height"))
        focal = float(group.findtext("FocalLengthPixels"))
        cx = float(group.findtext("./PrincipalPoint/x"))
        cy = float(group.findtext("./PrincipalPoint/y"))
        physical_candidates = physical_by_camera[camera_id]
        candidate_centers = np.stack([item.center for item in physical_candidates])
        center_tree = cKDTree(candidate_centers)
        for photo in group.findall("./Photo"):
            center = center_from_photo(photo)
            center_error, physical_index = center_tree.query(center, k=1)
            if float(center_error) > 1e-7:
                raise ValueError(f"derived/source center mismatch: {center_error} m")
            source = physical_candidates[int(physical_index)]
            view = FaceView(
                group_index=group_index,
                derived_photo_id=int(photo.findtext("Id")),
                source=source,
                center=center,
                rotation_w2c=matrix_from_photo(photo),
                width=width,
                height=height,
                focal_px=focal,
                cx=cx,
                cy=cy,
                near_depth=float(photo.findtext("NearDepth")),
                far_depth=float(photo.findtext("FarDepth")),
            )
            views_by_source[source.photo_id].append(view)

    counts = {photo_id: len(views) for photo_id, views in views_by_source.items()}
    if set(counts.values()) != {4} or set(counts) != set(physical_by_id):
        raise ValueError("each physical source image must map to exactly four face views")
    return physical_by_id, views_by_source, physical_root


def fisheye_project(directions: np.ndarray, photo: PhysicalPhoto) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-space directions with the XML's 4-coefficient KB model."""
    xyz = np.asarray(directions, dtype=np.float64)
    radial = np.linalg.norm(xyz[..., :2], axis=-1)
    theta = np.arctan2(radial, xyz[..., 2])
    theta2 = theta * theta
    k1, k2, k3, k4 = photo.distortion
    theta_distorted = theta * (
        1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
    )
    radial_safe = np.where(radial > 1e-15, radial, 1.0)
    scale = theta_distorted / radial_safe
    u = photo.focal_px * xyz[..., 0] * scale + photo.cx
    v = photo.focal_px * xyz[..., 1] * scale + photo.cy
    u = np.where(radial > 1e-15, u, photo.cx)
    v = np.where(radial > 1e-15, v, photo.cy)
    return u, v


def validate_fisheye_projection(
    physical_root: ET.Element,
    physical_by_id: dict[int, PhysicalPhoto],
    max_measurements: int = 10_000,
) -> dict[str, Any]:
    errors: list[float] = []
    for tie_point in physical_root.findall(".//TiePoint"):
        position = np.asarray(
            [float(tie_point.findtext(f"./Position/{axis}")) for axis in "xyz"],
            dtype=np.float64,
        )
        for measurement in tie_point.findall("./Measurement"):
            photo_id = int(measurement.findtext("PhotoId"))
            photo = physical_by_id[photo_id]
            q = photo.rotation_w2c @ (position - photo.center)
            u, v = fisheye_project(q[None, :], photo)
            measured = np.asarray(
                [float(measurement.findtext("x")), float(measurement.findtext("y"))]
            )
            errors.append(float(np.linalg.norm(np.asarray([u[0], v[0]]) - measured)))
            if len(errors) >= max_measurements:
                break
        if len(errors) >= max_measurements:
            break
    values = np.asarray(errors, dtype=np.float64)
    return {
        "measurement_count": int(len(values)),
        "pixel_error": percentile_summary(values),
        "formula": "q=R(X-C); theta=atan2(norm(qxy),qz); theta_d=theta*(1+k1*t2+k2*t4+k3*t6+k4*t8)",
    }


def percentile_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": None, "p10": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def read_pnts_positions(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(28)
        magic, version, byte_length, ft_json_len, ft_bin_len, bt_json_len, bt_bin_len = struct.unpack(
            "<4s6I", header
        )
        if magic != b"pnts" or version != 1:
            raise ValueError(f"invalid PNTS header: {path}")
        feature = json.loads(stream.read(ft_json_len).decode("utf-8").strip())
        feature_binary = stream.read(ft_bin_len)
        structured_length = stream.tell() + bt_json_len + bt_bin_len
        actual_length = path.stat().st_size
        if (
            actual_length < structured_length
            or actual_length - structured_length > 15
            or abs(byte_length - actual_length) > 15
        ):
            raise ValueError(f"PNTS byte length mismatch: {path}")
    count = int(feature["POINTS_LENGTH"])
    if "POSITION" in feature:
        offset = int(feature["POSITION"].get("byteOffset", 0))
        xyz = np.frombuffer(feature_binary, dtype="<f4", count=count * 3, offset=offset)
        xyz = xyz.reshape(count, 3).astype(np.float64)
    elif "POSITION_QUANTIZED" in feature:
        offset = int(feature["POSITION_QUANTIZED"].get("byteOffset", 0))
        raw = np.frombuffer(feature_binary, dtype="<u2", count=count * 3, offset=offset)
        scale = np.asarray(feature["QUANTIZED_VOLUME_SCALE"], dtype=np.float64)
        origin = np.asarray(feature["QUANTIZED_VOLUME_OFFSET"], dtype=np.float64)
        xyz = raw.reshape(count, 3).astype(np.float64) / 65535.0 * scale + origin
    else:
        raise ValueError(f"PNTS has no supported position field: {path}")
    if "RTC_CENTER" in feature:
        xyz += np.asarray(feature["RTC_CENTER"], dtype=np.float64)
    return xyz


def read_tile_las_proxy(task_root: Path, tile: int) -> tuple[np.ndarray, dict[str, Any]]:
    tile_root = task_root / "result" / "3D" / "point-pnts" / f"Tile_{tile}"
    paths = sorted(tile_root.rglob("*.pnts"), key=lambda item: item.as_posix())
    if not paths:
        raise FileNotFoundError(f"no PNTS files for Tile_{tile}")
    chunks = [read_pnts_positions(path) for path in paths]
    points = np.concatenate(chunks, axis=0)
    return points, {
        "pnts_file_count": len(paths),
        "point_count": int(len(points)),
        "source": str(tile_root),
        "interpretation": "validated in the live audit as the exact LAS ROI subset (0-4 point boundary differences)",
    }


def read_gs_positions(task_root: Path, tile: int) -> tuple[np.memmap, dict[str, Any]]:
    path = (
        task_root
        / "result"
        / "milestones"
        / "splats"
        / f"Tile_{tile}"
        / "gaussian_splat_level_0.pb.bin"
    )
    with path.open("rb") as stream:
        count = struct.unpack("<I", stream.read(4))[0]
    expected = 4 + count * 56
    if path.stat().st_size != expected:
        raise ValueError(f"unexpected GS PB size for {path}")
    records = np.memmap(path, dtype="<f4", mode="r", offset=4, shape=(count, 14))
    return records, {"path": str(path), "gaussian_count": int(count), "record_bytes": 56}


def encode_voxels(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64) + (1 << 20)
    if np.any(values < 0) or np.any(values >= (1 << 21)):
        raise ValueError("voxel index exceeds signed 21-bit encoding")
    return (values[:, 0] << 42) | (values[:, 1] << 21) | values[:, 2]


def build_stable_voxels(
    tile: int,
    las_xyz: np.ndarray,
    gs_records: np.ndarray,
) -> tuple[list[dict[str, Any]], cKDTree, dict[str, Any]]:
    las_indices = np.floor(las_xyz / VOXEL_SIZE_M).astype(np.int64)
    las_keys = encode_voxels(las_indices)
    unique_las, inverse, las_counts = np.unique(las_keys, return_inverse=True, return_counts=True)
    sums = np.column_stack(
        [np.bincount(inverse, weights=las_xyz[:, axis]) for axis in range(3)]
    )
    centroids = sums / las_counts[:, None]
    second_moments = np.empty((len(unique_las), 3, 3), dtype=np.float64)
    for axis_a in range(3):
        for axis_b in range(axis_a, 3):
            value = np.bincount(
                inverse,
                weights=las_xyz[:, axis_a] * las_xyz[:, axis_b],
            ) / las_counts
            second_moments[:, axis_a, axis_b] = value
            second_moments[:, axis_b, axis_a] = value
    voxel_covariance = second_moments - np.einsum("ni,nj->nij", centroids, centroids)
    voxel_eigenvalues = np.linalg.eigvalsh(voxel_covariance[selected := np.flatnonzero(las_counts >= MIN_LAS_PER_VOXEL)])
    voxel_curvature = voxel_eigenvalues[:, 0] / np.maximum(
        np.sum(voxel_eigenvalues, axis=1), 1e-18
    )

    gs_xyz = np.asarray(gs_records[:, :3], dtype=np.float64)
    gs_keys = encode_voxels(np.floor(gs_xyz / VOXEL_SIZE_M).astype(np.int64))
    unique_gs, gs_counts = np.unique(gs_keys, return_counts=True)
    positions = np.searchsorted(unique_gs, unique_las)
    matches = (positions < len(unique_gs)) & (unique_gs[np.minimum(positions, len(unique_gs) - 1)] == unique_las)
    stable_gs_counts = np.zeros(len(unique_las), dtype=np.int64)
    stable_gs_counts[matches] = gs_counts[positions[matches]]

    tree = cKDTree(las_xyz)
    _, neighbors = tree.query(centroids[selected], k=PCA_K, workers=-1)
    local = las_xyz[neighbors]
    centered = local - np.mean(local, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / PCA_K
    eigenvalues = np.linalg.eigvalsh(covariance)
    curvature = eigenvalues[:, 0] / np.maximum(np.sum(eigenvalues, axis=1), 1e-18)
    planarity_support = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1e-18)

    rows: list[dict[str, Any]] = []
    for local_index, voxel_index in enumerate(selected):
        n_las = int(las_counts[voxel_index])
        n_gs = int(stable_gs_counts[voxel_index])
        center = centroids[voxel_index]
        rows.append(
            {
                "tile": tile,
                "voxel_key": int(unique_las[voxel_index]),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "n_las": n_las,
                "n_gs": n_gs,
                "gs_las_ratio": float(n_gs / n_las),
                "surface_curvature": float(voxel_curvature[local_index]),
                "surface_curvature_voxel_covariance": float(voxel_curvature[local_index]),
                "surface_curvature_pca24_centroid": float(curvature[local_index]),
                "pca24_planarity_support": float(planarity_support[local_index]),
                "normal_displacement_median_mm": math.nan,
                "normal_displacement_p90_mm": math.nan,
                "normal_sample_count": 0,
            }
        )
    facts = {
        "las_occupied_voxels": int(len(unique_las)),
        "gs_occupied_voxels": int(len(unique_gs)),
        "stable_las_voxels_n_ge_100": int(len(rows)),
        "stable_voxels_with_zero_gs": int(sum(row["n_gs"] == 0 for row in rows)),
    }
    return rows, tree, facts


def add_normal_displacement(
    rows: list[dict[str, Any]],
    tree: cKDTree,
    las_xyz: np.ndarray,
    gs_records: np.ndarray,
) -> dict[str, Any]:
    count = len(gs_records)
    sample_count = min(GS_SAMPLE_PER_TILE, count)
    sample_indices = np.linspace(0, count - 1, sample_count, dtype=np.int64)
    gs_xyz = np.asarray(gs_records[sample_indices, :3], dtype=np.float64)
    _, anchors = tree.query(gs_xyz, k=1, workers=-1)
    _, neighbor_indices = tree.query(las_xyz[anchors], k=PCA_K, workers=-1)
    local = las_xyz[neighbor_indices]
    center = np.mean(local, axis=1)
    centered = local - center[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / PCA_K
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    curvature = eigenvalues[:, 0] / np.maximum(np.sum(eigenvalues, axis=1), 1e-18)
    support = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1e-18)
    reliable = (curvature < 0.02) & (support > 0.1)
    displacement_mm = np.abs(np.einsum("ni,ni->n", gs_xyz - las_xyz[anchors], normals)) * 1000.0
    sample_keys = encode_voxels(np.floor(gs_xyz / VOXEL_SIZE_M).astype(np.int64))

    row_keys = np.asarray([row["voxel_key"] for row in rows], dtype=np.int64)
    key_order = np.argsort(row_keys)
    sorted_keys = row_keys[key_order]
    locations = np.searchsorted(sorted_keys, sample_keys)
    matched = (locations < len(sorted_keys)) & (
        sorted_keys[np.minimum(locations, len(sorted_keys) - 1)] == sample_keys
    ) & reliable
    row_indices = key_order[locations[matched]]
    values = displacement_mm[matched]
    order = np.argsort(row_indices, kind="stable")
    row_indices = row_indices[order]
    values = values[order]
    unique_rows, starts, counts = np.unique(row_indices, return_index=True, return_counts=True)
    for row_index, start, group_count in zip(unique_rows, starts, counts):
        group = values[start : start + group_count]
        rows[int(row_index)]["normal_displacement_median_mm"] = float(np.median(group))
        rows[int(row_index)]["normal_displacement_p90_mm"] = float(np.percentile(group, 90))
        rows[int(row_index)]["normal_sample_count"] = int(group_count)
    return {
        "deterministic_gs_sample_count": int(sample_count),
        "reliable_sample_count": int(np.sum(reliable)),
        "reliable_fraction": float(np.mean(reliable)),
        "stable_voxels_with_at_least_5_normal_samples": int(
            sum(row["normal_sample_count"] >= MIN_NORMAL_SAMPLES_PER_VOXEL for row in rows)
        ),
    }


def project_face(points: np.ndarray, view: FaceView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = (points - view.center) @ view.rotation_w2c.T
    safe_z = np.where(np.abs(q[:, 2]) > 1e-12, q[:, 2], 1.0)
    u = view.focal_px * q[:, 0] / safe_z + view.cx
    v = view.focal_px * q[:, 1] / safe_z + view.cy
    return u, v, q[:, 2]


def decode_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return image


def patch_metrics(
    points: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    view: FaceView,
    image: np.ndarray,
    classify_mask: np.ndarray,
    batch_size: int = 2_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = PATCH_SIZE // 2
    dx, dy = np.meshgrid(
        np.arange(-half, half + 1, dtype=np.float64),
        np.arange(-half, half + 1, dtype=np.float64),
    )
    offsets_x = dx.reshape(-1)
    offsets_y = dy.reshape(-1)
    relative_face_to_source = view.source.rotation_w2c @ view.rotation_w2c.T
    gradients = np.full(len(points), np.nan, dtype=np.float64)
    entropies = np.full(len(points), np.nan, dtype=np.float64)
    luma = np.full(len(points), np.nan, dtype=np.float64)
    valid_output = np.zeros(len(points), dtype=bool)

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        uu = u[start:stop, None] + offsets_x[None, :]
        vv = v[start:stop, None] + offsets_y[None, :]
        face_directions = np.stack(
            [
                (uu - view.cx) / view.focal_px,
                (vv - view.cy) / view.focal_px,
                np.ones_like(uu),
            ],
            axis=-1,
        )
        source_directions = face_directions @ relative_face_to_source.T
        map_x, map_y = fisheye_project(source_directions, view.source)
        source_valid = (
            np.all(map_x >= 1.0, axis=1)
            & np.all(map_x <= image.shape[1] - 2.0, axis=1)
            & np.all(map_y >= 1.0, axis=1)
            & np.all(map_y <= image.shape[0] - 2.0, axis=1)
        )
        center_index = (PATCH_SIZE * PATCH_SIZE) // 2
        mask_x = np.clip(
            np.rint(map_x[:, center_index] * classify_mask.shape[1] / image.shape[1]).astype(int),
            0,
            classify_mask.shape[1] - 1,
        )
        mask_y = np.clip(
            np.rint(map_y[:, center_index] * classify_mask.shape[0] / image.shape[0]).astype(int),
            0,
            classify_mask.shape[0] - 1,
        )
        source_valid &= classify_mask[mask_y, mask_x] == 0
        if not np.any(source_valid):
            continue
        valid_local = np.flatnonzero(source_valid)
        patches = cv2.remap(
            image,
            map_x[valid_local].astype(np.float32).reshape(-1, PATCH_SIZE),
            map_y[valid_local].astype(np.float32).reshape(-1, PATCH_SIZE),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        ).reshape(-1, PATCH_SIZE, PATCH_SIZE).astype(np.float32) / 255.0
        gx = (patches[:, 1:-1, 2:] - patches[:, 1:-1, :-2]) * 0.5
        gy = (patches[:, 2:, 1:-1] - patches[:, :-2, 1:-1]) * 0.5
        gradient = np.mean(np.sqrt(gx * gx + gy * gy), axis=(1, 2))
        quantized = np.minimum((patches * 16.0).astype(np.uint8), 15)
        histogram = np.stack(
            [np.sum(quantized == bin_index, axis=(1, 2)) for bin_index in range(16)],
            axis=1,
        ).astype(np.float64)
        probability = histogram / (PATCH_SIZE * PATCH_SIZE)
        entropy = -np.sum(
            np.where(probability > 0, probability * np.log2(np.maximum(probability, 1e-15)), 0.0),
            axis=1,
        )
        global_indices = start + valid_local
        gradients[global_indices] = gradient
        entropies[global_indices] = entropy
        luma[global_indices] = patches[:, half, half]
        valid_output[global_indices] = True
    return gradients, entropies, luma, valid_output


def grouped_statistics(indices: np.ndarray, values: np.ndarray, row_count: int) -> dict[str, np.ndarray]:
    output = {
        "median": np.full(row_count, np.nan, dtype=np.float64),
        "p90": np.full(row_count, np.nan, dtype=np.float64),
        "mad": np.full(row_count, np.nan, dtype=np.float64),
        "count": np.zeros(row_count, dtype=np.int64),
    }
    if not len(indices):
        return output
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    sorted_values = values[order]
    unique, starts, counts = np.unique(sorted_indices, return_index=True, return_counts=True)
    for row_index, start, count in zip(unique, starts, counts):
        group = sorted_values[start : start + count]
        median = float(np.median(group))
        output["median"][row_index] = median
        output["p90"][row_index] = float(np.percentile(group, 90))
        output["mad"][row_index] = float(np.median(np.abs(group - median)))
        output["count"][row_index] = int(count)
    return output


def add_image_metrics(
    task_root: Path,
    rows: list[dict[str, Any]],
    views_by_source: dict[int, list[FaceView]],
) -> dict[str, Any]:
    points = np.asarray(
        [[row["center_x"], row["center_y"], row["center_z"]] for row in rows],
        dtype=np.float64,
    )
    row_count = len(rows)
    frustum_count = np.zeros(row_count, dtype=np.int64)
    depth_count = np.zeros(row_count, dtype=np.int64)
    effective_count = np.zeros(row_count, dtype=np.int64)
    depth_effective_count = np.zeros(row_count, dtype=np.int64)
    all_obs_index: list[np.ndarray] = []
    all_obs_gradient: list[np.ndarray] = []
    all_obs_entropy: list[np.ndarray] = []
    all_obs_luma: list[np.ndarray] = []
    all_obs_distance: list[np.ndarray] = []
    depth_obs_index: list[np.ndarray] = []
    depth_obs_gradient: list[np.ndarray] = []
    depth_obs_entropy: list[np.ndarray] = []
    depth_obs_luma: list[np.ndarray] = []
    depth_obs_distance: list[np.ndarray] = []
    decoded_images = 0
    missing_images: list[str] = []

    for source_number, source_id in enumerate(sorted(views_by_source), start=1):
        views = views_by_source[source_id]
        source = views[0].source
        candidate_all: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        candidate_depth: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        source_frustum = np.zeros(row_count, dtype=bool)
        source_depth = np.zeros(row_count, dtype=bool)
        half = PATCH_SIZE // 2 + 1
        for view_index, view in enumerate(views):
            u, v, depth = project_face(points, view)
            valid = (
                (depth > 0.0)
                & (u >= 0.0)
                & (u < view.width)
                & (v >= 0.0)
                & (v < view.height)
            )
            gated = valid & (depth >= view.near_depth) & (depth <= view.far_depth)
            source_frustum |= valid
            source_depth |= gated
            patch_valid = (
                (depth > 0.0)
                & (u >= half)
                & (u < view.width - half)
                & (v >= half)
                & (v < view.height - half)
            )
            margin = np.minimum.reduce(
                [u - half, view.width - half - u, v - half, view.height - half - v]
            ) / min(view.width, view.height)
            candidate_all.append((patch_valid, margin, u, v))
            candidate_depth.append(
                (patch_valid & (depth >= view.near_depth) & (depth <= view.far_depth), margin, u, v)
            )
        frustum_count += source_frustum
        depth_count += source_depth
        if not np.any(source_frustum):
            continue
        if not source.image_path.is_file():
            missing_images.append(str(source.image_path))
            continue
        mask_path = task_root / "result" / "milestones" / "classify" / f"{source.photo_id}.tif"
        if not mask_path.is_file():
            missing_images.append(str(mask_path))
            continue
        image = decode_image(source.image_path, cv2.IMREAD_GRAYSCALE)
        classify_mask = decode_image(mask_path, cv2.IMREAD_GRAYSCALE)
        decoded_images += 1

        for depth_mode, candidates in ((False, candidate_all), (True, candidate_depth)):
            valid_stack = np.stack([candidate[0] for candidate in candidates], axis=1)
            score_stack = np.stack(
                [np.where(candidate[0], candidate[1], -np.inf) for candidate in candidates], axis=1
            )
            selected_view = np.argmax(score_stack, axis=1)
            selected_valid = np.any(valid_stack, axis=1)
            for view_index, view in enumerate(views):
                chosen_rows = np.flatnonzero(selected_valid & (selected_view == view_index))
                if not len(chosen_rows):
                    continue
                _, _, u, v = candidates[view_index]
                gradient, entropy, luma, valid_patch = patch_metrics(
                    points[chosen_rows],
                    u[chosen_rows],
                    v[chosen_rows],
                    view,
                    image,
                    classify_mask,
                )
                accepted = np.flatnonzero(valid_patch)
                if not len(accepted):
                    continue
                accepted_rows = chosen_rows[accepted]
                distance = np.linalg.norm(points[accepted_rows] - source.center, axis=1)
                if depth_mode:
                    depth_effective_count[accepted_rows] += 1
                    depth_obs_index.append(accepted_rows)
                    depth_obs_gradient.append(gradient[accepted])
                    depth_obs_entropy.append(entropy[accepted])
                    depth_obs_luma.append(luma[accepted])
                    depth_obs_distance.append(distance)
                else:
                    effective_count[accepted_rows] += 1
                    all_obs_index.append(accepted_rows)
                    all_obs_gradient.append(gradient[accepted])
                    all_obs_entropy.append(entropy[accepted])
                    all_obs_luma.append(luma[accepted])
                    all_obs_distance.append(distance)
        if source_number % 25 == 0:
            print(
                f"images {source_number}/{len(views_by_source)} decoded={decoded_images} "
                f"depth_observations={sum(len(item) for item in depth_obs_index)}",
                flush=True,
            )

    def concatenate(items: list[np.ndarray], dtype: Any = np.float64) -> np.ndarray:
        return np.concatenate(items).astype(dtype, copy=False) if items else np.empty(0, dtype=dtype)

    all_indices = concatenate(all_obs_index, np.int64)
    depth_indices = concatenate(depth_obs_index, np.int64)
    all_gradient_stats = grouped_statistics(all_indices, concatenate(all_obs_gradient), row_count)
    all_entropy_stats = grouped_statistics(all_indices, concatenate(all_obs_entropy), row_count)
    all_luma_stats = grouped_statistics(all_indices, concatenate(all_obs_luma), row_count)
    all_distance_stats = grouped_statistics(all_indices, concatenate(all_obs_distance), row_count)
    depth_gradient_stats = grouped_statistics(depth_indices, concatenate(depth_obs_gradient), row_count)
    depth_entropy_stats = grouped_statistics(depth_indices, concatenate(depth_obs_entropy), row_count)
    depth_luma_stats = grouped_statistics(depth_indices, concatenate(depth_obs_luma), row_count)
    depth_distance_stats = grouped_statistics(depth_indices, concatenate(depth_obs_distance), row_count)

    for index, row in enumerate(rows):
        row.update(
            {
                "frustum_source_view_count_proxy": int(frustum_count[index]),
                "depth_gated_source_view_count_proxy": int(depth_count[index]),
                "effective_image_observation_count": int(effective_count[index]),
                "depth_effective_image_observation_count": int(depth_effective_count[index]),
                "gradient_median_all": float(all_gradient_stats["median"][index]),
                "gradient_p90_all": float(all_gradient_stats["p90"][index]),
                "entropy_median_all_bits": float(all_entropy_stats["median"][index]),
                "entropy_p90_all_bits": float(all_entropy_stats["p90"][index]),
                "camera_distance_median_all_m": float(all_distance_stats["median"][index]),
                "gradient_median_depth": float(depth_gradient_stats["median"][index]),
                "gradient_p90_depth": float(depth_gradient_stats["p90"][index]),
                "entropy_median_depth_bits": float(depth_entropy_stats["median"][index]),
                "entropy_p90_depth_bits": float(depth_entropy_stats["p90"][index]),
                "camera_distance_median_depth_m": float(depth_distance_stats["median"][index]),
                "cross_view_luma_mad_depth_proxy": float(depth_luma_stats["mad"][index]),
            }
        )
    return {
        "physical_source_images_in_xml": int(len(views_by_source)),
        "decoded_source_images": int(decoded_images),
        "missing_input_count": int(len(missing_images)),
        "missing_inputs": missing_images[:20],
        "total_frustum_source_voxel_observations": int(np.sum(frustum_count)),
        "total_depth_gated_source_voxel_observations": int(np.sum(depth_count)),
        "total_effective_image_observations": int(np.sum(effective_count)),
        "total_depth_effective_image_observations": int(np.sum(depth_effective_count)),
    }


def spearman(values_x: np.ndarray, values_y: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values_x) & np.isfinite(values_y)
    x = values_x[finite]
    y = values_y[finite]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"n": int(len(x)), "rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {"n": int(len(x)), "rho": float(result.statistic), "p_value": float(result.pvalue)}


def stratified_rank_correlation(
    values_x: np.ndarray, values_y: np.ndarray, tiles: np.ndarray
) -> dict[str, Any]:
    finite = np.isfinite(values_x) & np.isfinite(values_y)
    x_rank_parts: list[np.ndarray] = []
    y_rank_parts: list[np.ndarray] = []
    for tile in np.unique(tiles[finite]):
        selected = finite & (tiles == tile)
        if np.sum(selected) < 3:
            continue
        x_rank = rankdata(values_x[selected], method="average")
        y_rank = rankdata(values_y[selected], method="average")
        x_rank_parts.append((x_rank - np.mean(x_rank)) / max(np.std(x_rank), 1e-15))
        y_rank_parts.append((y_rank - np.mean(y_rank)) / max(np.std(y_rank), 1e-15))
    if not x_rank_parts:
        return {"n": 0, "rho": None}
    x = np.concatenate(x_rank_parts)
    y = np.concatenate(y_rank_parts)
    return {"n": int(len(x)), "rho": float(np.corrcoef(x, y)[0, 1])}


def correlation_table(rows: list[dict[str, Any]], response: str) -> dict[str, Any]:
    predictors = [
        "gradient_median_depth",
        "gradient_p90_depth",
        "entropy_median_depth_bits",
        "entropy_p90_depth_bits",
        "depth_effective_image_observation_count",
        "frustum_source_view_count_proxy",
        "depth_gated_source_view_count_proxy",
        "camera_distance_median_depth_m",
        "n_las",
        "surface_curvature",
        "cross_view_luma_mad_depth_proxy",
    ]
    tiles = np.asarray([row["tile"] for row in rows], dtype=np.int64)
    y = np.asarray([row[response] for row in rows], dtype=np.float64)
    minimum_observations = np.asarray(
        [row["depth_effective_image_observation_count"] for row in rows], dtype=np.int64
    ) >= MIN_IMAGE_OBS_FOR_TEXTURE
    if response.startswith("normal_displacement"):
        minimum_observations &= np.asarray(
            [row["normal_sample_count"] for row in rows], dtype=np.int64
        ) >= MIN_NORMAL_SAMPLES_PER_VOXEL
    table: dict[str, Any] = {}
    for predictor in predictors:
        x = np.asarray([row[predictor] for row in rows], dtype=np.float64)
        mask = minimum_observations & np.isfinite(x) & np.isfinite(y)
        by_tile = {
            f"Tile_{tile}": spearman(x[mask & (tiles == tile)], y[mask & (tiles == tile)])
            for tile in range(4)
        }
        table[predictor] = {
            "pooled": spearman(x[mask], y[mask]),
            "pooled_within_tile_rank": stratified_rank_correlation(
                x[mask], y[mask], tiles[mask]
            ),
            "by_tile": by_tile,
        }
    return table


def binned_table(rows: list[dict[str, Any]], predictor: str) -> list[dict[str, Any]]:
    x = np.asarray([row[predictor] for row in rows], dtype=np.float64)
    ratio = np.asarray([row["gs_las_ratio"] for row in rows], dtype=np.float64)
    displacement = np.asarray(
        [row["normal_displacement_median_mm"] for row in rows], dtype=np.float64
    )
    normal_count = np.asarray([row["normal_sample_count"] for row in rows], dtype=np.int64)
    observations = np.asarray(
        [row["depth_effective_image_observation_count"] for row in rows], dtype=np.int64
    )
    finite = np.isfinite(x) & np.isfinite(ratio) & (observations >= MIN_IMAGE_OBS_FOR_TEXTURE)
    if np.sum(finite) < 10:
        return []
    edges = np.unique(np.quantile(x[finite], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(edges) < 2:
        return []
    bins = np.searchsorted(edges[1:-1], x, side="right")
    output: list[dict[str, Any]] = []
    for bin_index in range(len(edges) - 1):
        selected = finite & (bins == bin_index)
        normal_selected = selected & np.isfinite(displacement) & (
            normal_count >= MIN_NORMAL_SAMPLES_PER_VOXEL
        )
        output.append(
            {
                "bin": bin_index + 1,
                "n": int(np.sum(selected)),
                "predictor_min": float(np.min(x[selected])),
                "predictor_median": float(np.median(x[selected])),
                "predictor_max": float(np.max(x[selected])),
                "gs_las_ratio_p25": float(np.percentile(ratio[selected], 25)),
                "gs_las_ratio_median": float(np.median(ratio[selected])),
                "gs_las_ratio_p75": float(np.percentile(ratio[selected], 75)),
                "normal_displacement_n": int(np.sum(normal_selected)),
                "normal_displacement_median_mm": (
                    float(np.median(displacement[normal_selected]))
                    if np.any(normal_selected)
                    else None
                ),
            }
        )
    return output


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonify(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(task_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.time()
    physical_by_id, views_by_source, physical_root = load_cameras(task_root)
    projection_validation = validate_fisheye_projection(physical_root, physical_by_id)
    print("fisheye validation", projection_validation, flush=True)

    rows: list[dict[str, Any]] = []
    tile_facts: dict[str, Any] = {}
    for tile in range(4):
        print(f"Tile_{tile}: reading PNTS/LAS proxy and GS", flush=True)
        las_xyz, las_facts = read_tile_las_proxy(task_root, tile)
        gs_records, gs_facts = read_gs_positions(task_root, tile)
        tile_rows, tree, voxel_facts = build_stable_voxels(tile, las_xyz, gs_records)
        displacement_facts = add_normal_displacement(tile_rows, tree, las_xyz, gs_records)
        rows.extend(tile_rows)
        tile_facts[f"Tile_{tile}"] = {
            "las": las_facts,
            "gs": gs_facts,
            "voxel": voxel_facts,
            "normal_displacement": displacement_facts,
        }
        del tree, las_xyz, gs_records
        print(f"Tile_{tile}: stable voxels={len(tile_rows)}", flush=True)

    print(f"image audit for {len(rows)} stable tile-voxels", flush=True)
    image_facts = add_image_metrics(task_root, rows, views_by_source)

    density_correlations = correlation_table(rows, "gs_las_ratio")
    displacement_correlations = correlation_table(rows, "normal_displacement_median_mm")
    binned_predictors = [
        "gradient_median_depth",
        "entropy_median_depth_bits",
        "depth_effective_image_observation_count",
        "camera_distance_median_depth_m",
        "n_las",
        "surface_curvature",
        "cross_view_luma_mad_depth_proxy",
    ]
    bins = {predictor: binned_table(rows, predictor) for predictor in binned_predictors}

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "mipmap_image_voxel_regression_audit.voxels.csv"
    json_path = output_root / "mipmap_image_voxel_regression_audit.summary.json"
    write_csv(csv_path, rows)
    summary = {
        "task_root": str(task_root),
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "runtime_seconds": float(time.time() - started),
        "method": {
            "voxel_size_m": VOXEL_SIZE_M,
            "stable_voxel_rule": f"N_LAS >= {MIN_LAS_PER_VOXEL}; includes N_GS=0",
            "las_source": "hierarchical PNTS positions previously validated point-for-point/count-for-count as each Tile's LAS ROI",
            "gaussian_source": "per-Tile final level-0 PB, position fields 0..2",
            "normal_displacement": f"deterministic {GS_SAMPLE_PER_TILE} GS/tile; nearest LAS anchor; PCA k={PCA_K}; reliable when curvature<0.02 and lambda1/lambda2>0.1; voxel median absolute point-to-plane",
            "surface_curvature": "primary geometry-complexity feature is lambda_min/sum(lambda) from all LAS points inside each 0.5 m voxel; a separate centroid-neighborhood PCA24 curvature is retained in the CSV",
            "undistorted_image_reconstruction": f"local {PATCH_SIZE}x{PATCH_SIZE} patches in native mvs_undistort.xml face pixels remapped to optimized original fisheye image; no full derived image is synthesized or saved",
            "gradient": "mean central-difference grayscale magnitude over the inner 9x9 of an 11x11 native face-pixel patch, intensity normalized to [0,1]",
            "entropy": "16-bin Shannon entropy over the 11x11 grayscale patch, bits",
            "view_counting": "deduplicated by physical source image; among overlapping four faces choose the face with largest normalized border margin",
            "effective_observation": "face projection valid, reconstructed fisheye patch in bounds, and MipMap classify mask center=0",
            "depth_gate": "per-face NearDepth <= camera-forward z <= FarDepth from mvs_undistort.xml",
            "correlation_filter": f"at least {MIN_IMAGE_OBS_FOR_TEXTURE} depth-effective source observations; normal response also needs at least {MIN_NORMAL_SAMPLES_PER_VOXEL} reliable sampled GS in voxel",
        },
        "projection_validation": projection_validation,
        "tile_facts": tile_facts,
        "image_facts": image_facts,
        "row_counts": {
            "all_stable_tile_voxels": len(rows),
            "with_depth_texture_observations_ge_3": int(
                sum(row["depth_effective_image_observation_count"] >= MIN_IMAGE_OBS_FOR_TEXTURE for row in rows)
            ),
            "with_normal_samples_ge_5": int(
                sum(row["normal_sample_count"] >= MIN_NORMAL_SAMPLES_PER_VOXEL for row in rows)
            ),
            "with_both": int(
                sum(
                    row["depth_effective_image_observation_count"] >= MIN_IMAGE_OBS_FOR_TEXTURE
                    and row["normal_sample_count"] >= MIN_NORMAL_SAMPLES_PER_VOXEL
                    for row in rows
                )
            ),
        },
        "spearman": {
            "density_response_gs_las_ratio": density_correlations,
            "normal_displacement_response_mm": displacement_correlations,
            "note": "pooled_within_tile_rank centers standardized ranks inside each Tile before pooling, reducing between-Tile confounding",
        },
        "quintile_bins": bins,
        "limitations": [
            "MipMap deleted the saved undistorted JPEGs because keep_undistort_images=false; local patches are deterministically reconstructed from the optimized physical/face cameras and original source image.",
            "Frustum and Near/FarDepth counts do not perform mesh/LiDAR z-buffer occlusion and are visibility proxies, not true visible-surface counts.",
            "No renderer output or per-pixel training residual was saved. cross_view_luma_mad_depth_proxy is only a robust cross-view intensity-dispersion proxy and must not be called photometric loss/residual.",
            "Gradient/entropy are observational image features at the projected LAS voxel centroid. Correlation is not proof that the optimizer used either feature explicitly.",
            "The moving-object/classify mask is sampled at its saved quarter resolution; a center pixel is used rather than requiring the entire 11x11 remapped patch to be unmasked.",
            "N_GS/N_LAS versus N_LAS contains denominator coupling. Both per-Tile and within-Tile-rank results should be preferred over naive pooled correlation.",
            "Normal displacement uses a deterministic GS sample, not every Gaussian. Voxel summaries require at least five reliable sampled GS for the normal-response correlations.",
            "Tile overlap means some physical space appears as multiple independent tile-voxel rows; per-Tile and within-Tile-rank results make this explicit.",
        ],
        "evidence": {
            "voxel_csv": str(csv_path),
            "summary_json": str(json_path),
            "mvs_physical": str(task_root / "result" / "AT" / "mvs.xml"),
            "mvs_undistort": str(task_root / "result" / "AT" / "mvs_undistort.xml"),
            "classify_masks": str(task_root / "result" / "milestones" / "classify"),
            "tile_las_proxy": str(task_root / "result" / "3D" / "point-pnts"),
            "tile_final_gs": str(task_root / "result" / "milestones" / "splats"),
        },
    }
    json_path.write_text(
        json.dumps(jsonify(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    args = parser.parse_args()
    summary = run(args.task_root.resolve(), args.output_root.resolve())
    print(
        json.dumps(
            {
                "runtime_seconds": summary["runtime_seconds"],
                "row_counts": summary["row_counts"],
                "image_facts": summary["image_facts"],
                "evidence": summary["evidence"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
