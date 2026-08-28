"""MipMap-compatible K=7 scale and K=30 normal Tile initialization geometry."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.s1_reader import sha256_file
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest


TILE_GEOMETRY_SCHEMA_VERSION = 1
TILE_GEOMETRY_KIND = "mipmap_k7_k30_tile_initialization_geometry_v1"


def _load_canonical_xyz_rgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("rb") as stream:
        header: list[str] = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            line = raw.decode("ascii").rstrip("\r\n")
            header.append(line)
            if line == "end_header":
                break
        if header[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("Tile initialization PLY must be binary_little_endian")
        vertex = [line for line in header if line.startswith("element vertex ")]
        if len(vertex) != 1:
            raise ValueError("Tile initialization PLY must contain one vertex element")
        count = int(vertex[0].split()[2])
        expected_properties = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        properties = [line for line in header if line.startswith("property ")]
        if properties != expected_properties:
            raise ValueError("Tile initialization PLY properties are not canonical XYZ/RGB")
        dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)], align=False)
        records = np.fromfile(stream, dtype=dtype, count=count)
        if len(records) != count or stream.read(1):
            raise ValueError("Tile initialization PLY payload does not match its header")
    xyz = np.asarray(records["xyz"], dtype=np.float32).copy()
    rgb = np.asarray(records["rgb"], dtype=np.uint8).copy()
    if len(xyz) < 30 or not np.isfinite(xyz).all():
        raise ValueError("Tile initialization needs at least 30 finite points")
    return xyz, rgb


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p50": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def compute_mipmap_tile_geometry(
    xyz: np.ndarray,
    *,
    batch_size: int = 20_000,
    workers: int = -1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return deterministic K=7/K=30 geometry in the input point order."""

    from scipy.spatial import cKDTree

    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30:
        raise ValueError("MipMap Tile geometry expects at least 30 XYZ points")
    if not np.isfinite(points).all():
        raise ValueError("MipMap Tile geometry XYZ must be finite")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tree = cKDTree(points)
    count = len(points)
    normals = np.empty((count, 3), dtype=np.float32)
    eigenvalues = np.empty((count, 3), dtype=np.float32)
    scales = np.empty((count, 3), dtype=np.float32)
    quaternions = np.empty((count, 4), dtype=np.float32)
    nonpositive_scale_count = 0
    degenerate_normal_count = 0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        distances, indexes = tree.query(
            points[start:stop],
            k=30,
            workers=workers,
        )
        tangent_scale = np.mean(np.asarray(distances[:, 1:7]), axis=1)
        invalid_scale = ~np.isfinite(tangent_scale) | (tangent_scale <= 0.0)
        nonpositive_scale_count += int(np.count_nonzero(invalid_scale))
        if np.any(invalid_scale):
            positive = tangent_scale[~invalid_scale]
            if not len(positive):
                raise ValueError("K=7 initialization found no positive neighbor spacing")
            tangent_scale[invalid_scale] = float(np.median(positive))

        neighborhoods = points[indexes]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / 29.0
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, :, 0]
        normal_norm = np.linalg.norm(normal, axis=1)
        invalid_normal = ~np.isfinite(normal_norm) | (normal_norm <= 1e-8)
        degenerate_normal_count += int(np.count_nonzero(invalid_normal))
        if np.any(invalid_normal):
            normal[invalid_normal] = np.asarray([0.0, 0.0, 1.0])
            normal_norm[invalid_normal] = 1.0
        normal = normal / normal_norm[:, None]
        # PCA normals are unoriented and x/y scales are equal. Choosing the +Z
        # hemisphere is deterministic and covariance-equivalent to the opposite sign.
        normal[normal[:, 2] < 0.0] *= -1.0
        quaternion = np.column_stack(
            [
                1.0 + normal[:, 2],
                -normal[:, 1],
                normal[:, 0],
                np.zeros(len(normal), dtype=np.float64),
            ]
        )
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)

        normals[start:stop] = normal.astype(np.float32)
        eigenvalues[start:stop] = np.maximum(values, 0.0).astype(np.float32)
        scales[start:stop, 0] = tangent_scale.astype(np.float32)
        scales[start:stop, 1] = tangent_scale.astype(np.float32)
        scales[start:stop, 2] = (0.5 * tangent_scale).astype(np.float32)
        quaternions[start:stop] = quaternion.astype(np.float32)

    geometry = {
        "normals": normals,
        "eigenvalues": eigenvalues,
        "scales_m": scales,
        "quaternions_wxyz": quaternions,
    }
    trace = eigenvalues.sum(axis=1)
    planarity = np.zeros(count, dtype=np.float32)
    valid_trace = trace > 1e-12
    planarity[valid_trace] = np.clip(
        1.0 - 3.0 * eigenvalues[valid_trace, 0] / trace[valid_trace],
        0.0,
        1.0,
    )
    report = {
        "algorithm": "mipmap_compatible_k7_mean_distance_k30_pca_v1",
        "evidence_boundary": "ALGORITHM_COMPATIBLE_NOT_VENDOR_BIT_EXACT",
        "point_count": count,
        "scale_knn_including_self": 7,
        "scale_neighbor_reduction": "mean_euclidean_distance_neighbors_1_to_6",
        "normal_knn": 30,
        "linear_scale_axes": [1.0, 1.0, 0.5],
        "rotation": "shortest_arc_local_plus_z_to_unoriented_pca_normal_wxyz",
        "nonpositive_scale_replacement_count": nonpositive_scale_count,
        "degenerate_normal_fallback_count": degenerate_normal_count,
        "tangent_scale_m": _distribution(scales[:, 0]),
        "normal_scale_m": _distribution(scales[:, 2]),
        "planarity": _distribution(planarity),
    }
    return geometry, report


def sign_tile_geometry_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("tile_geometry_manifest_sha256", None)
    signed = copy.deepcopy(unsigned)
    signed["tile_geometry_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def verify_tile_geometry_manifest(
    manifest: dict[str, Any],
    *,
    root: Path | None = None,
    verify_artifacts: bool = False,
) -> str:
    expected = str(manifest.get("tile_geometry_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("Tile geometry manifest is unsigned")
    unsigned = copy.deepcopy(manifest)
    unsigned.pop("tile_geometry_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("Tile geometry manifest signature mismatch")
    if (
        int(manifest.get("schema_version", -1)) != TILE_GEOMETRY_SCHEMA_VERSION
        or manifest.get("kind") != TILE_GEOMETRY_KIND
    ):
        raise ValueError("unsupported Tile geometry manifest schema")
    if int(manifest.get("tile_count", -1)) != len(manifest.get("tiles", [])):
        raise ValueError("Tile geometry count is inconsistent")
    if verify_artifacts:
        if root is None:
            raise ValueError("Tile geometry artifact verification requires a root")
        for tile in manifest["tiles"]:
            path = Path(root) / tile["geometry"]["path"]
            if not path.is_file() or sha256_file(path) != tile["geometry"]["sha256"]:
                raise ValueError(f"Tile geometry artifact mismatch: {path}")
    return expected


def load_mipmap_tile_geometry(
    manifest_path: Path,
    root: Path,
    *,
    tile_id: int,
    expected_initialization_ply_sha256: str,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one exact-order Tile geometry artifact after full identity checks."""

    manifest_path = Path(manifest_path)
    root = Path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = verify_tile_geometry_manifest(
        manifest,
        root=root,
        verify_artifacts=True,
    )
    matches = [
        tile for tile in manifest["tiles"] if int(tile["tile_id"]) == int(tile_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"Tile geometry manifest has no unique Tile {tile_id}")
    tile = matches[0]
    if tile.get("initialization_ply_sha256") != expected_initialization_ply_sha256:
        raise ValueError("Tile geometry is bound to a different initialization PLY")
    if int(tile.get("point_count", -1)) != int(expected_count):
        raise ValueError("Tile geometry row count differs from initialization PLY")
    geometry_path = root / tile["geometry"]["path"]
    with np.load(geometry_path, allow_pickle=False) as payload:
        required = {"normals", "eigenvalues", "scales_m", "quaternions_wxyz"}
        if set(payload.files) != required:
            raise ValueError("Tile geometry NPZ arrays do not match the contract")
        normals = np.asarray(payload["normals"], dtype=np.float32)
        eigenvalues = np.asarray(payload["eigenvalues"], dtype=np.float32)
        scales = np.asarray(payload["scales_m"], dtype=np.float32)
        quaternions = np.asarray(payload["quaternions_wxyz"], dtype=np.float32)
    count = int(expected_count)
    if (
        normals.shape != (count, 3)
        or eigenvalues.shape != (count, 3)
        or scales.shape != (count, 3)
        or quaternions.shape != (count, 4)
    ):
        raise ValueError("Tile geometry NPZ row shapes differ from initialization PLY")
    if not all(
        np.isfinite(array).all()
        for array in (normals, eigenvalues, scales, quaternions)
    ):
        raise ValueError("Tile geometry contains non-finite values")
    if np.any(scales <= 0.0):
        raise ValueError("Tile geometry scales must be positive")
    if not np.allclose(scales[:, 0], scales[:, 1], rtol=2e-6, atol=1e-9):
        raise ValueError("Tile geometry tangent scales differ")
    if not np.allclose(scales[:, 2], 0.5 * scales[:, 0], rtol=2e-6, atol=1e-9):
        raise ValueError("Tile geometry normal scale is not exactly half the tangent scale")
    quaternion_norm = np.linalg.norm(quaternions, axis=1)
    if not np.allclose(quaternion_norm, 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError("Tile geometry quaternions are not normalized")
    report = copy.deepcopy(tile["report"])
    report.update(
        {
            "enabled": True,
            "tile_id": int(tile_id),
            "tile_geometry_manifest_sha256": manifest_sha,
            "geometry_sha256": tile["geometry"]["sha256"],
            "initialization_ply_sha256": expected_initialization_ply_sha256,
        }
    )
    return np.ascontiguousarray(scales), np.ascontiguousarray(quaternions), report


def explicit_mipmap_scale_calibration_report(
    scales_m: np.ndarray,
    *,
    configured_means_lr: float,
    configured_noise_lr: float,
) -> dict[str, Any]:
    """Describe externally materialized K=7 scales without recomputing K=3 RMS."""

    scales = np.asarray(scales_m, dtype=np.float64)
    if scales.ndim != 2 or scales.shape[1] != 3 or np.any(scales <= 0.0):
        raise ValueError("explicit MipMap scales must have shape [N,3] and be positive")
    reference = float(np.median(scales[:, 0]))
    nominal_noise_std_m = (
        reference * reference * float(configured_means_lr) * float(configured_noise_lr)
    )
    unsigned = {
        "schema_version": 1,
        "policy": {
            "mode": "external_mipmap_k7_k30",
            "scale_knn_including_self": 7,
            "scale_neighbor_reduction": "mean_euclidean_distance_neighbors_1_to_6",
            "linear_scale_axes": [1.0, 1.0, 0.5],
            "means_step_fraction": None,
            "noise_std_fraction": None,
        },
        "point_count": int(len(scales)),
        "reference_scale_m": reference,
        "scale_distribution_m": _distribution(scales[:, 0]),
        "invalid_replaced_count": 0,
        "clipped_count": 0,
        "raw_scale_distribution_m": _distribution(scales[:, 0]),
        "clamp_min_m": None,
        "clamp_max_m": None,
        "configured_fixed_scale_m": None,
        "configured_means_lr": float(configured_means_lr),
        "configured_noise_lr": float(configured_noise_lr),
        "effective_means_lr_m": float(configured_means_lr),
        "effective_noise_lr": float(configured_noise_lr),
        "nominal_noise_std_m": nominal_noise_std_m,
        "nominal_noise_std_fraction": nominal_noise_std_m / reference,
    }
    report = copy.deepcopy(unsigned)
    report["scale_calibration_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return report


def audit_mipmap_tile_geometry_consumption(
    geometry_manifest_path: Path,
    geometry_root: Path,
    tile_inputs_path: Path,
    tile_inputs_root: Path,
) -> dict[str, Any]:
    """Exercise the Trainer-facing loader for every independently bound Tile."""

    geometry_manifest_path = Path(geometry_manifest_path).resolve()
    geometry_root = Path(geometry_root).resolve()
    tile_inputs_path = Path(tile_inputs_path).resolve()
    tile_inputs_root = Path(tile_inputs_root).resolve()
    tile_inputs = json.loads(tile_inputs_path.read_text(encoding="utf-8"))
    tile_inputs_sha = verify_tile_inputs_manifest(
        tile_inputs,
        root=tile_inputs_root,
        verify_artifacts=True,
    )
    geometry_manifest = json.loads(
        geometry_manifest_path.read_text(encoding="utf-8")
    )
    geometry_manifest_sha = verify_tile_geometry_manifest(
        geometry_manifest,
        root=geometry_root,
        verify_artifacts=True,
    )
    if geometry_manifest.get("tile_inputs_manifest_sha256") != tile_inputs_sha:
        raise ValueError("Tile geometry is bound to a different Tile input manifest")
    input_by_id = {int(tile["tile_id"]): tile for tile in tile_inputs["tiles"]}
    if len(input_by_id) != len(tile_inputs["tiles"]):
        raise ValueError("Tile input manifest contains duplicate Tile ids")
    audited_tiles: list[dict[str, Any]] = []
    total_rows = 0
    for geometry_tile in geometry_manifest["tiles"]:
        tile_id = int(geometry_tile["tile_id"])
        input_tile = input_by_id.get(tile_id)
        if input_tile is None:
            raise ValueError(f"Tile input manifest has no Tile {tile_id}")
        initialization = input_tile["initialization"]
        if geometry_tile.get("name") != input_tile.get("name"):
            raise ValueError(f"Tile {tile_id} name differs between manifests")
        scales, quaternions, report = load_mipmap_tile_geometry(
            geometry_manifest_path,
            geometry_root,
            tile_id=tile_id,
            expected_initialization_ply_sha256=initialization["sha256"],
            expected_count=int(initialization["point_count"]),
        )
        rows = len(scales)
        total_rows += rows
        audited_tiles.append(
            {
                "tile_id": tile_id,
                "name": input_tile["name"],
                "status": "CONSUMPTION_READY",
                "point_count": rows,
                "initialization_ply_sha256": initialization["sha256"],
                "geometry_sha256": report["geometry_sha256"],
                "tangent_scale_p50_m": float(np.median(scales[:, 0])),
                "normal_scale_p50_m": float(np.median(scales[:, 2])),
                "quaternion_norm_max_abs_error": float(
                    np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0))
                ),
            }
        )
    if len(audited_tiles) != int(tile_inputs["tile_count"]):
        raise ValueError("Tile geometry does not cover every Tile input")
    unsigned = {
        "schema_version": 1,
        "kind": "mipmap_tile_geometry_consumption_audit_v1",
        "status": "CONSUMPTION_READY",
        "tile_inputs_manifest_sha256": tile_inputs_sha,
        "tile_geometry_manifest_sha256": geometry_manifest_sha,
        "tile_count": len(audited_tiles),
        "total_point_count": total_rows,
        "tiles": audited_tiles,
        "training_allowed": False,
        "next_required_artifact": "tile_face4_crop_consumption_manifest",
    }
    audit = copy.deepcopy(unsigned)
    audit["consumption_audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return audit


def materialize_mipmap_tile_geometry(
    tile_inputs_path: Path,
    tile_inputs_root: Path,
    output_root: Path,
    *,
    batch_size: int = 20_000,
    workers: int = -1,
) -> dict[str, Any]:
    tile_inputs_path = Path(tile_inputs_path).resolve()
    tile_inputs_root = Path(tile_inputs_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "tile_geometry_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace Tile geometry: {manifest_path}")
    tile_inputs = json.loads(tile_inputs_path.read_text(encoding="utf-8"))
    tile_inputs_sha = verify_tile_inputs_manifest(
        tile_inputs,
        root=tile_inputs_root,
        verify_artifacts=True,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_tiles: list[dict[str, Any]] = []
    for tile in tile_inputs["tiles"]:
        initialization = tile["initialization"]
        ply_path = tile_inputs_root / initialization["path"]
        if sha256_file(ply_path) != initialization["sha256"]:
            raise ValueError(f"Tile {tile['tile_id']} initialization SHA256 mismatch")
        xyz, _rgb = _load_canonical_xyz_rgb_ply(ply_path)
        if len(xyz) != int(initialization["point_count"]):
            raise ValueError(f"Tile {tile['tile_id']} initialization count mismatch")
        geometry, report = compute_mipmap_tile_geometry(
            xyz,
            batch_size=batch_size,
            workers=workers,
        )
        tile_root = output_root / str(tile["name"])
        tile_root.mkdir(parents=True, exist_ok=True)
        geometry_path = tile_root / "initialization_geometry_k7_k30.npz"
        temporary = geometry_path.with_name(geometry_path.name + ".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, **geometry)
            os.replace(temporary, geometry_path)
        finally:
            temporary.unlink(missing_ok=True)
        output_tiles.append(
            {
                "tile_id": int(tile["tile_id"]),
                "name": str(tile["name"]),
                "initialization_ply_sha256": initialization["sha256"],
                "point_count": len(xyz),
                "geometry": {
                    "path": geometry_path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(geometry_path),
                    "bytes": geometry_path.stat().st_size,
                    "arrays": {
                        name: list(value.shape) for name, value in geometry.items()
                    },
                },
                "report": report,
            }
        )
    manifest = sign_tile_geometry_manifest(
        {
            "schema_version": TILE_GEOMETRY_SCHEMA_VERSION,
            "kind": TILE_GEOMETRY_KIND,
            "tile_inputs_manifest_sha256": tile_inputs_sha,
            "tile_count": len(output_tiles),
            "tiles": output_tiles,
            "training_allowed": False,
            "next_required_artifact": "tile_face4_crop_consumption_manifest",
        }
    )
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest
