#!/usr/bin/env python3
"""Build a signed LiDAR + mesh-supported surfel initialization for V64."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.point_cloud import write_binary_ply
from cloudstudio_3dgs.geometry.mesh_completion import (
    MeshCompletionConfig,
    coverage_deficit_mask,
    depth_boundary_mask,
    deterministic_stride_mask,
    merge_voxel_candidates,
)
from cloudstudio_3dgs.training.backend import (
    GsplatBackend,
    rendered_range_to_euclidean,
)
from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset
from cloudstudio_3dgs.training.trainer import load_initialization_ply


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.ply")
    write_binary_ply(temporary, xyz, rgb)
    temporary.replace(path)


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(array.min()),
        "p50": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _z_axis_quaternions(normals: np.ndarray) -> np.ndarray:
    unit = np.asarray(normals, dtype=np.float32).copy()
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    unit[unit[:, 2] < 0.0] *= -1.0
    quaternions = np.column_stack(
        [
            1.0 + unit[:, 2],
            -unit[:, 1],
            unit[:, 0],
            np.zeros(len(unit), dtype=np.float32),
        ]
    ).astype(np.float32)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions


def _tile_views(config: dict) -> list[dict]:
    manifest = _read(Path(config["tile_inputs_manifest"]))
    matches = [
        tile
        for tile in manifest["tiles"]
        if int(tile["tile_id"]) == int(config["mipmap_tile_id"])
    ]
    if len(matches) != 1:
        raise ValueError("config does not bind exactly one Tile")
    return list(matches[0]["views"])


def _backproject(
    sample: object,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(selected)
    K = np.asarray(sample.K, dtype=np.float64)
    rays = np.column_stack(
        [
            (xx.astype(np.float64) + 0.5 - K[0, 2]) / K[0, 0],
            (yy.astype(np.float64) + 0.5 - K[1, 2]) / K[1, 1],
            np.ones(len(xx), dtype=np.float64),
        ]
    )
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    ranges = np.asarray(sample.mesh_depth_range_m, dtype=np.float64)[yy, xx]
    camera_points = rays * ranges[:, None]
    c2w = np.asarray(sample.c2w, dtype=np.float64)
    world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    camera_normals = np.asarray(sample.mesh_normal_camera, dtype=np.float64)[yy, xx]
    world_normals = camera_normals @ c2w[:3, :3].T
    colors = np.asarray(sample.image, dtype=np.uint8)[yy, xx]
    return (
        world_points.astype(np.float32),
        world_normals.astype(np.float32),
        colors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--voxel-size-m", type=float, default=0.015)
    parser.add_argument("--max-completion-points", type=int, default=600_000)
    parser.add_argument("--alpha-floor", type=float, default=0.9)
    parser.add_argument("--confidence-min", type=float, default=0.8)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.05)
    parser.add_argument("--boundary-threshold-m", type=float, default=0.1)
    parser.add_argument("--boundary-dilation-pixels", type=int, default=1)
    parser.add_argument(
        "--allow-range-only",
        action="store_true",
        help=(
            "allow births where alpha is already sufficient but rendered range "
            "differs from mesh; disabled by default to prevent double surfaces"
        ),
    )
    args = parser.parse_args()

    config = _read(args.config)
    completion = MeshCompletionConfig(
        alpha_floor=args.alpha_floor,
        confidence_min=args.confidence_min,
        depth_tolerance_m=args.depth_tolerance_m,
        pixel_stride=args.pixel_stride,
        voxel_size_m=args.voxel_size_m,
        max_completion_points=args.max_completion_points,
    )
    completion.validate()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = FaceCacheDataset(
        face_manifest_path=Path(config["face_cache_manifest"]),
        cache_root=Path(config["face_cache_root"]),
        dataset_manifest_path=Path(config["dataset_manifest"]),
        tile_views=_tile_views(config),
        renderer_mask_manifest_path=Path(config["renderer_mask_manifest"]),
        face_lidar_geometry_manifest_path=Path(
            config["face_lidar_geometry_manifest"]
        ),
        face_lidar_geometry_root=Path(config["face_lidar_geometry_root"]),
        mesh_geometry_manifest_path=Path(config["mesh_geometry_manifest"]),
        mesh_geometry_root=Path(config["mesh_geometry_root"]),
    )

    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backend = GsplatBackend(
        device=config.get("device", "cuda:0"),
        cap_max=int(config["cap_max"]),
        lock_path=Path(config["gsplat_lock"]),
        mcmc_config={"noise_injection_stop_iter": 0},
    )
    backend.color_model = config.get("color_model", "sh")
    backend.sh_degree = int(config.get("sh_degree", 0))
    backend.pinhole_rasterize_mode = config.get("pinhole_rasterize_mode", "classic")
    backend.pinhole_with_ut = bool(config.get("pinhole_with_ut", False))
    params = {
        name: value.to(config.get("device", "cuda:0"))
        for name, value in payload["params"].items()
    }
    background_rgb = config.get("background_color", [1.0, 1.0, 1.0])

    xyz_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    frame_records: list[dict] = []
    raw_deficit_total = 0
    boundary_rejected_total = 0
    sampled_total = 0
    deficit_categories = {
        "raw_alpha_low": 0,
        "raw_range_missing": 0,
        "raw_range_mismatch": 0,
        "raw_range_only": 0,
        "sampled_alpha_low": 0,
        "sampled_range_only": 0,
    }
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            if sample.mesh_depth_range_m is None or sample.mesh_geometry_cache_path is None:
                raise ValueError(f"selected sample has no mesh geometry: {sample.image_id}")
            _rendered, rendered_range, rendered_alpha, render_info = backend.render(
                params,
                sample,
                with_range=True,
                background_rgb=background_rgb,
            )
            rendered_range = rendered_range_to_euclidean(
                torch, rendered_range, sample, render_info
            )
            alpha = rendered_alpha.detach().cpu().numpy().astype(np.float32)
            range_m = rendered_range.detach().cpu().numpy().astype(np.float32)
            with np.load(sample.mesh_geometry_cache_path, allow_pickle=False) as sidecar:
                source_type = np.asarray(sidecar["source_type"], dtype=np.uint8)
                mesh_valid = np.asarray(sidecar["valid"], dtype=bool)
            raw_deficit = coverage_deficit_mask(
                rgb_mask=sample.rgb_mask,
                mesh_valid=mesh_valid,
                mesh_confidence=sample.mesh_confidence,
                source_type=source_type,
                rendered_alpha=alpha,
                rendered_range_m=range_m,
                mesh_range_m=sample.mesh_depth_range_m,
                config=completion,
            )
            boundary = depth_boundary_mask(
                sample.mesh_depth_range_m,
                mesh_valid,
                threshold_m=args.boundary_threshold_m,
                dilation_pixels=args.boundary_dilation_pixels,
            )
            alpha_low = np.isfinite(alpha) & (alpha < completion.alpha_floor)
            range_missing = ~np.isfinite(range_m) | (range_m <= 0.0)
            range_mismatch = (
                ~range_missing
                & (
                    np.abs(range_m - np.asarray(sample.mesh_depth_range_m))
                    > completion.depth_tolerance_m
                )
            )
            range_only = raw_deficit & ~alpha_low & (range_missing | range_mismatch)
            safe = raw_deficit & ~boundary
            birth_safe = safe if args.allow_range_only else safe & alpha_low
            selected = birth_safe & deterministic_stride_mask(
                birth_safe.shape, completion.pixel_stride
            )
            raw_count = int(raw_deficit.sum())
            safe_count = int(safe.sum())
            selected_count = int(selected.sum())
            deficit_categories["raw_alpha_low"] += int((raw_deficit & alpha_low).sum())
            deficit_categories["raw_range_missing"] += int(
                (raw_deficit & range_missing).sum()
            )
            deficit_categories["raw_range_mismatch"] += int(
                (raw_deficit & range_mismatch).sum()
            )
            deficit_categories["raw_range_only"] += int(range_only.sum())
            deficit_categories["sampled_alpha_low"] += int(
                (selected & alpha_low).sum()
            )
            deficit_categories["sampled_range_only"] += int(
                (selected & range_only).sum()
            )
            raw_deficit_total += raw_count
            boundary_rejected_total += raw_count - safe_count
            sampled_total += selected_count
            if selected_count:
                points, normals, colors = _backproject(sample, selected)
                selected_alpha = alpha[selected]
                selected_range = range_m[selected]
                target_range = np.asarray(sample.mesh_depth_range_m)[selected]
                confidence = np.asarray(sample.mesh_confidence)[selected]
                alpha_gap = np.clip(
                    (completion.alpha_floor - selected_alpha)
                    / completion.alpha_floor,
                    0.0,
                    1.0,
                )
                range_gap = np.ones(selected_count, dtype=np.float32)
                rendered_valid = np.isfinite(selected_range) & (selected_range > 0.0)
                range_gap[rendered_valid] = np.clip(
                    np.abs(selected_range[rendered_valid] - target_range[rendered_valid])
                    / completion.depth_tolerance_m,
                    0.0,
                    1.0,
                )
                scores = confidence * (0.05 + np.maximum(alpha_gap, range_gap))
                xyz_parts.append(points)
                normal_parts.append(normals)
                rgb_parts.append(colors)
                score_parts.append(scores.astype(np.float32))
            frame_records.append(
                {
                    "sample_id": sample.image_id,
                    "raw_deficit_pixels": raw_count,
                    "boundary_safe_pixels": safe_count,
                    "birth_admitted_pixels": int(birth_safe.sum()),
                    "stride_sampled_pixels": selected_count,
                }
            )
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                print(
                    f"mesh completion probe {index + 1}/{len(dataset)}: "
                    f"raw={raw_deficit_total:,} sampled={sampled_total:,}",
                    flush=True,
                )

    if not xyz_parts:
        raise RuntimeError("mesh completion produced no trusted candidates")
    candidate_xyz = np.concatenate(xyz_parts)
    candidate_normals = np.concatenate(normal_parts)
    candidate_rgb = np.concatenate(rgb_parts)
    candidate_scores = np.concatenate(score_parts)

    original_ply = Path(config["initialization_ply"])
    original_geometry = Path(config["initialization_geometry"])
    lidar_xyz, lidar_rgb = load_initialization_ply(original_ply)
    completion_xyz, completion_normals, completion_rgb, merged_scores, counts = (
        merge_voxel_candidates(
            candidate_xyz,
            candidate_normals,
            candidate_rgb,
            candidate_scores,
            voxel_size_m=completion.voxel_size_m,
            occupied_xyz=lidar_xyz,
            max_points=completion.max_completion_points,
        )
    )
    if not len(completion_xyz):
        raise RuntimeError("all mesh completion candidates overlap LiDAR voxels")
    combined_xyz = np.concatenate([lidar_xyz, completion_xyz])
    combined_rgb = np.concatenate([lidar_rgb, completion_rgb])
    with np.load(original_geometry, allow_pickle=False) as geometry:
        lidar_normals = np.asarray(geometry["normals"], dtype=np.float32)
        lidar_eigenvalues = np.asarray(geometry["eigenvalues"], dtype=np.float32)
        lidar_scales = np.asarray(geometry["scales_m"], dtype=np.float32)
        lidar_quaternions = np.asarray(
            geometry["quaternions_wxyz"], dtype=np.float32
        )
    if lidar_normals.shape != lidar_xyz.shape or lidar_eigenvalues.shape != lidar_xyz.shape:
        raise ValueError("original geometry rows do not match original LiDAR PLY")
    completion_eigenvalues = np.tile(
        np.array([[0.0, 1.0, 1.0]], dtype=np.float32),
        (len(completion_xyz), 1),
    )
    combined_normals = np.concatenate([lidar_normals, completion_normals])
    combined_eigenvalues = np.concatenate(
        [lidar_eigenvalues, completion_eigenvalues]
    )
    from scipy.spatial import cKDTree

    neighbour_distance = cKDTree(combined_xyz).query(
        completion_xyz, k=8, workers=-1
    )[0][:, 1:]
    completion_tangent_scale = np.clip(
        neighbour_distance.mean(axis=1), 0.003, 0.05
    ).astype(np.float32)
    completion_scales = np.repeat(completion_tangent_scale[:, None], 3, axis=1)
    completion_scales[:, 2] = np.maximum(
        0.001, 0.15 * completion_tangent_scale
    )
    completion_quaternions = _z_axis_quaternions(completion_normals)
    combined_scales = np.concatenate([lidar_scales, completion_scales])
    combined_quaternions = np.concatenate(
        [lidar_quaternions, completion_quaternions]
    )

    completion_ply = args.output / "mesh_completion_only.ply"
    combined_ply = args.output / "initialization_lidar_plus_mesh_completion.ply"
    geometry_path = args.output / "initialization_geometry_planar_surfel.npz"
    _atomic_ply(completion_ply, completion_xyz, completion_rgb)
    _atomic_ply(combined_ply, combined_xyz, combined_rgb)
    geometry_temporary = geometry_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        geometry_temporary,
        normals=combined_normals.astype(np.float32),
        eigenvalues=combined_eigenvalues.astype(np.float32),
        scales_m=combined_scales.astype(np.float32),
        quaternions_wxyz=combined_quaternions.astype(np.float32),
    )
    geometry_temporary.replace(geometry_path)

    calibration_reports = [
        ROOT
        / "outputs/snow-20260224-full-20260825/d0a_mesh_candidate_tile1_block05_v62b/mesh_candidate_audit.json",
        ROOT
        / "outputs/snow-20260224-full-20260825/d0a_mesh_candidate_tile1_block10_v62c/mesh_candidate_audit.json",
        ROOT
        / "outputs/snow-20260224-full-20260825/d0a_mesh_candidate_tile1_v62a/mesh_candidate_audit.json",
    ]
    report = {
        "schema_version": 1,
        "kind": "mesh_supported_surfel_completion_initialization_v1",
        "status": "CANDIDATE_NOT_TRAINING_PROMOTED",
        "dataset": "snow-20260224",
        "tile_id": int(config["mipmap_tile_id"]),
        "algorithm": {
            "authority": "source_type_3_cross_view_supported_only",
            "source_type_4_allowed": False,
            "confidence_min": completion.confidence_min,
            "confidence_contract": "float16 0.83349609375 means two-view support",
            "deficit": {
                "alpha_floor": completion.alpha_floor,
                "range_tolerance_m": completion.depth_tolerance_m,
                "range_only_births_allowed": bool(args.allow_range_only),
                "birth_contract": (
                    "trusted_mesh_and_alpha_deficit"
                    if not args.allow_range_only
                    else "trusted_mesh_and_alpha_or_range_deficit"
                ),
            },
            "depth_boundary": {
                "threshold_m": args.boundary_threshold_m,
                "dilation_pixels": args.boundary_dilation_pixels,
            },
            "sampling": {
                "pixel_stride": completion.pixel_stride,
                "voxel_size_m": completion.voxel_size_m,
                "max_completion_points": completion.max_completion_points,
                "occupied_lidar_voxels_excluded": True,
            },
        },
        "source": {
            "diagnostic_config": args.config.resolve().as_posix(),
            "diagnostic_config_sha256": _sha256(args.config),
            "diagnostic_checkpoint": args.checkpoint.resolve().as_posix(),
            "diagnostic_checkpoint_sha256": _sha256(args.checkpoint),
            "mesh_geometry_manifest": Path(config["mesh_geometry_manifest"])
            .resolve()
            .as_posix(),
            "mesh_geometry_manifest_sha256": _sha256(
                Path(config["mesh_geometry_manifest"])
            ),
            "original_initialization_ply": original_ply.resolve().as_posix(),
            "original_initialization_ply_sha256": _sha256(original_ply),
            "original_geometry_sha256": _sha256(original_geometry),
            "held_out_spatial_block_calibration": [
                {"path": path.resolve().as_posix(), "sha256": _sha256(path)}
                for path in calibration_reports
            ],
        },
        "counts": {
            "view_count": len(dataset),
            "raw_deficit_pixels": raw_deficit_total,
            "depth_boundary_rejected_pixels": boundary_rejected_total,
            "stride_sampled_observations": sampled_total,
            "candidate_observations": int(len(candidate_xyz)),
            "completion_surfels": int(len(completion_xyz)),
            "original_lidar_gaussians": int(len(lidar_xyz)),
            "combined_gaussians": int(len(combined_xyz)),
            "deficit_categories": deficit_categories,
        },
        "completion": {
            "multi_view_observation_count": _distribution(counts),
            "selection_score": _distribution(merged_scores),
            "bounds_min": completion_xyz.min(axis=0).astype(float).tolist(),
            "bounds_max": completion_xyz.max(axis=0).astype(float).tolist(),
            "tangent_scale_m": _distribution(completion_scales[:, 0]),
            "normal_scale_m": _distribution(completion_scales[:, 2]),
            "aspect_ratio": _distribution(
                completion_scales[:, 0] / completion_scales[:, 2]
            ),
        },
        "artifacts": {
            "completion_ply": completion_ply.name,
            "completion_ply_sha256": _sha256(completion_ply),
            "combined_ply": combined_ply.name,
            "combined_ply_sha256": _sha256(combined_ply),
            "geometry": geometry_path.name,
            "geometry_sha256": _sha256(geometry_path),
        },
        "per_view": frame_records,
    }
    report["initialization_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    manifest_path = args.output / "mesh_completion_initialization_manifest.json"
    _atomic_json(manifest_path, report)
    print(manifest_path)
    print(
        f"completion={len(completion_xyz):,} combined={len(combined_xyz):,} "
        f"sampled_observations={len(candidate_xyz):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
