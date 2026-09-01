"""Write a signed production cross-view-filtered Face4 mesh sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.mesh_geometry import (
    MESH_GEOMETRY_KIND,
    MESH_GEOMETRY_SCHEMA_VERSION,
    mesh_geometry_npz_bytes,
    sign_mesh_geometry_manifest,
    verify_mesh_geometry_manifest,
)
from cloudstudio_3dgs.geometry.mesh_cross_view_filter import (
    CrossViewFilterConfig,
    classify_cross_view_support,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _camera_records(mesh_manifest: dict, face_manifest: dict) -> dict[str, dict]:
    images = {str(item["image_id"]): item for item in face_manifest["images"]}
    specs = {
        (str(camera_id), str(face["face_id"])): face
        for camera_id, group in face_manifest["cameras"].items()
        for face in group["faces"]
    }
    result: dict[str, dict] = {}
    for record in mesh_manifest["records"]:
        sample_id = str(record["sample_id"])
        image_id, face_id = sample_id.split("::", 1)
        image = images[image_id]
        spec = specs[(str(image["camera_id"]), face_id)]
        face_to_base = np.eye(4, dtype=np.float64)
        face_to_base[:3, :3] = np.asarray(spec["R_face"], dtype=np.float64)
        c2w = np.asarray(image["c2w"], dtype=np.float64) @ face_to_base
        result[sample_id] = {
            "sample_id": sample_id,
            "image_id": image_id,
            "face_id": face_id,
            "c2w": c2w,
            "w2c": np.linalg.inv(c2w),
            "K": np.asarray(spec["K_face"], dtype=np.float64),
            "crop": record["crop"],
            "path": str(record["path"]),
            "record": record,
        }
    return result


def _neighbor_map(cameras: dict[str, dict], count: int) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for sample_id, source in cameras.items():
        candidates = [
            target
            for target in cameras.values()
            if target["face_id"] == source["face_id"]
            and target["image_id"] != source["image_id"]
        ]
        candidates.sort(
            key=lambda target: float(
                np.linalg.norm(target["c2w"][:3, 3] - source["c2w"][:3, 3])
            )
        )
        result[sample_id] = [str(item["sample_id"]) for item in candidates[:count]]
    return result


def _stable_target_mask(depth: np.ndarray, valid: np.ndarray, edge_m: float) -> np.ndarray:
    safe = np.where(valid, depth, 0.0).astype(np.float32)
    local_max = cv2.dilate(safe, np.ones((3, 3), dtype=np.uint8))
    large = np.where(valid, safe, np.finfo(np.float32).max)
    local_min = cv2.erode(large, np.ones((3, 3), dtype=np.uint8))
    discontinuity = valid & ((local_max - local_min) > float(edge_m))
    invalid_boundary = cv2.dilate((~valid).astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    boundary = cv2.dilate(
        (discontinuity | invalid_boundary).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
    ) > 0
    return valid & ~boundary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--face-lidar-manifest", type=Path, required=True)
    parser.add_argument("--face-lidar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument("--absolute-threshold-m", type=float, default=0.05)
    parser.add_argument("--relative-threshold", type=float, default=0.005)
    parser.add_argument("--edge-threshold-m", type=float, default=0.10)
    parser.add_argument("--cache-views", type=int, default=8)
    args = parser.parse_args()
    if args.neighbors < 2:
        raise ValueError("production filtering requires at least two neighbors")

    started = time.perf_counter()
    mesh_manifest = _read(args.mesh_manifest)
    source_manifest_sha = verify_mesh_geometry_manifest(mesh_manifest)
    face_manifest = _read(args.face_manifest)
    lidar_manifest = _read(args.face_lidar_manifest)
    cameras = _camera_records(mesh_manifest, face_manifest)
    neighbors = _neighbor_map(cameras, args.neighbors)
    lidar_records = {
        str(item["sample_id"]): item for item in lidar_manifest["records"]
    }
    cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def arrays(sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if sample_id in cache:
            value = cache.pop(sample_id)
            cache[sample_id] = value
            return value
        with np.load(
            args.mesh_root / cameras[sample_id]["path"], allow_pickle=False
        ) as payload:
            value = (
                np.asarray(payload["depth_range_m"], dtype=np.float32),
                np.asarray(payload["normal_camera"], dtype=np.float32),
                np.asarray(payload["confidence"], dtype=np.float32),
                np.asarray(payload["valid"], dtype=bool),
            )
        cache[sample_id] = value
        while len(cache) > args.cache_views:
            cache.popitem(last=False)
        return value

    config = CrossViewFilterConfig()
    output_records: list[dict] = []
    totals = {
        "input_valid": 0,
        "output_valid": 0,
        "rejected_conflict": 0,
        "lidar_anchor": 0,
        "supported": 0,
        "unobservable_retained": 0,
    }
    for index, (sample_id, camera) in enumerate(cameras.items(), start=1):
        depth, normal, _old_confidence, source_valid = arrays(sample_id)
        shape = depth.shape
        yy, xx = np.nonzero(source_valid)
        crop = camera["crop"]
        u = xx.astype(np.float64) + float(crop["x"]) + 0.5
        v = yy.astype(np.float64) + float(crop["y"]) + 0.5
        K = camera["K"]
        rays = np.stack(
            [
                (u - K[0, 2]) / K[0, 0],
                (v - K[1, 2]) / K[1, 1],
                np.ones_like(u),
            ],
            axis=1,
        )
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        points_camera = rays * depth[yy, xx, None]
        points_world = (
            points_camera @ camera["c2w"][:3, :3].T
            + camera["c2w"][:3, 3]
        )
        observed_flat = np.zeros(len(xx), dtype=np.int16)
        consistent_flat = np.zeros(len(xx), dtype=np.int16)
        conflict_flat = np.zeros(len(xx), dtype=np.int16)
        point_h = np.concatenate(
            [points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1
        )
        for target_id in neighbors[sample_id]:
            target = cameras[target_id]
            target_depth, _target_normal, _target_confidence, target_valid = arrays(
                target_id
            )
            stable = _stable_target_mask(
                target_depth, target_valid, args.edge_threshold_m
            )
            point_target = point_h @ target["w2c"].T
            z = point_target[:, 2]
            projected_u = (
                target["K"][0, 0]
                * point_target[:, 0]
                / np.maximum(z, 1e-12)
                + target["K"][0, 2]
            )
            projected_v = (
                target["K"][1, 1]
                * point_target[:, 1]
                / np.maximum(z, 1e-12)
                + target["K"][1, 2]
            )
            target_crop = target["crop"]
            tx = np.rint(
                projected_u - float(target_crop["x"]) - 0.5
            ).astype(np.int64)
            ty = np.rint(
                projected_v - float(target_crop["y"]) - 0.5
            ).astype(np.int64)
            inside = z > 0.0
            inside &= tx >= 0
            inside &= ty >= 0
            inside &= tx < int(target_crop["width"])
            inside &= ty < int(target_crop["height"])
            selected = np.flatnonzero(inside)
            if not len(selected):
                continue
            selected = selected[stable[ty[selected], tx[selected]]]
            if not len(selected):
                continue
            expected = np.linalg.norm(point_target[selected, :3], axis=1)
            actual = target_depth[ty[selected], tx[selected]]
            tolerance = np.maximum(
                float(args.absolute_threshold_m),
                float(args.relative_threshold) * expected,
            )
            delta = actual - expected
            observed_flat[selected] += 1
            consistent_flat[selected] += np.abs(delta) <= tolerance
            conflict_flat[selected] += delta > tolerance

        observed = np.zeros(shape, dtype=np.int16)
        consistent = np.zeros(shape, dtype=np.int16)
        conflicts = np.zeros(shape, dtype=np.int16)
        observed[yy, xx] = observed_flat
        consistent[yy, xx] = consistent_flat
        conflicts[yy, xx] = conflict_flat

        sparse = load_sparse_depth(
            args.face_lidar_root / str(lidar_records[sample_id]["path"])
        )
        _lidar_depth, _lidar_confidence, lidar_valid_full = sparse.to_dense()
        lidar_anchor = lidar_valid_full[
            int(crop["y"]) : int(crop["y"]) + int(crop["height"]),
            int(crop["x"]) : int(crop["x"]) + int(crop["width"]),
        ]
        filtered_valid, confidence, source_type = classify_cross_view_support(
            source_valid=source_valid,
            lidar_anchor=lidar_anchor,
            observed=observed,
            consistent=consistent,
            conflicts=conflicts,
            config=config,
        )
        payload = mesh_geometry_npz_bytes(
            depth,
            normal,
            confidence,
            filtered_valid,
            source_type=source_type,
        )
        relative_path = f"depth/{sample_id.replace('::', '__')}.npz"
        _atomic_bytes(args.output / relative_path, payload)
        input_count = int(np.count_nonzero(source_valid))
        output_count = int(np.count_nonzero(filtered_valid))
        record = dict(camera["record"])
        record.update(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "mesh_valid_pixels_before_filter": input_count,
                "mesh_valid_pixels": output_count,
                "mesh_valid_fraction": float(
                    output_count / max(int(record["rgb_mask_pixels"]), 1)
                ),
                "cross_view": {
                    "neighbors": neighbors[sample_id],
                    "observable_pixels": int(np.count_nonzero(observed)),
                    "consistent_pixels": int(np.count_nonzero(consistent)),
                    "conflicting_pixels": int(np.count_nonzero(conflicts)),
                    "rejected_pixels": input_count - output_count,
                    "retained_fraction": float(output_count / max(input_count, 1)),
                },
            }
        )
        output_records.append(record)
        totals["input_valid"] += input_count
        totals["output_valid"] += output_count
        totals["rejected_conflict"] += input_count - output_count
        totals["lidar_anchor"] += int(np.count_nonzero(source_type == 2))
        totals["supported"] += int(np.count_nonzero(source_type == 3))
        totals["unobservable_retained"] += int(np.count_nonzero(source_type == 4))
        print(
            f"cross-view filter {index}/{len(cameras)} {sample_id}: "
            f"retained={output_count / max(input_count, 1):.5f} "
            f"rejected={input_count - output_count}",
            flush=True,
        )

    retained_fractions = np.asarray(
        [item["cross_view"]["retained_fraction"] for item in output_records],
        dtype=np.float64,
    )
    output_manifest = sign_mesh_geometry_manifest(
        {
            "schema_version": MESH_GEOMETRY_SCHEMA_VERSION,
            "kind": MESH_GEOMETRY_KIND,
            "split": mesh_manifest["split"],
            "tile_id": mesh_manifest["tile_id"],
            "source_face_manifest_sha256": mesh_manifest[
                "source_face_manifest_sha256"
            ],
            "source_tile_inputs_manifest_sha256": mesh_manifest.get(
                "source_tile_inputs_manifest_sha256"
            ),
            "source_face_lidar_geometry_manifest_sha256": mesh_manifest[
                "source_face_lidar_geometry_manifest_sha256"
            ],
            "source_mesh_geometry_manifest_sha256": source_manifest_sha,
            "mesh": mesh_manifest["mesh"],
            "depth_semantics": mesh_manifest["depth_semantics"],
            "normal_semantics": mesh_manifest["normal_semantics"],
            "confidence_semantics": (
                "1=native_lidar_anchor; 0.6..1=cross_view_support; "
                "0.2=unobservable_or_occluded_retained"
            ),
            "source_type_table": {
                "0": "invalid",
                "2": "native_lidar_anchor",
                "3": "lidar_mesh_cross_view_supported",
                "4": "lidar_mesh_unobservable_or_occluded_retained",
            },
            "cross_view_filter": {
                "status": "PRODUCTION_FILTER_APPLIED",
                "neighbor_policy": (
                    f"nearest_{args.neighbors}_different_physical_images_same_face_id"
                ),
                "absolute_threshold_m": args.absolute_threshold_m,
                "relative_threshold": args.relative_threshold,
                "edge_threshold_m": args.edge_threshold_m,
                "policy": config.__dict__,
                "totals": totals,
                "retained_fraction_p50": float(np.median(retained_fractions)),
                "retained_fraction_p05": float(
                    np.quantile(retained_fractions, 0.05)
                ),
            },
            "complete_face_cache": len(output_records) == len(mesh_manifest["records"]),
            "expected_face_count": mesh_manifest["expected_face_count"],
            "records": output_records,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "mesh_geometry_manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path, flush=True)
    print(output_manifest["mesh_geometry_manifest_sha256"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
