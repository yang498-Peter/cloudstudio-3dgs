"""Rasterize a LiDAR mesh into selected Tile Face4 crops and audit geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "p50": None, "p95": None, "gt_0p05": None, "gt_0p10": None}
    q = np.quantile(values, [0.5, 0.95])
    return {
        "count": int(len(values)),
        "p50": float(q[0]),
        "p95": float(q[1]),
        "gt_0p05": float(np.mean(values > 0.05)),
        "gt_0p10": float(np.mean(values > 0.10)),
    }


def _select_evenly(records: list[dict], count: int) -> list[dict]:
    if count >= len(records):
        return records
    indexes = np.linspace(0, len(records) - 1, count, dtype=np.int64)
    return [records[int(index)] for index in indexes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--face-root", type=Path, required=True)
    parser.add_argument("--tile-inputs-manifest", type=Path, required=True)
    parser.add_argument("--tile-id", type=int, default=1)
    parser.add_argument("--face-lidar-manifest", type=Path, required=True)
    parser.add_argument("--face-lidar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=8)
    args = parser.parse_args()

    import open3d as o3d

    started = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if not len(mesh.triangles):
        raise ValueError("mesh has no triangles")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    face_manifest = _read(args.face_manifest)
    tile_manifest = _read(args.tile_inputs_manifest)
    lidar_manifest = _read(args.face_lidar_manifest)
    tile = next(item for item in tile_manifest["tiles"] if int(item["tile_id"]) == args.tile_id)
    selected = _select_evenly(tile["views"], args.max_views)
    images = {str(item["image_id"]): item for item in face_manifest["images"]}
    face_specs = {
        (str(camera_id), str(face["face_id"])): face
        for camera_id, group in face_manifest["cameras"].items()
        for face in group["faces"]
    }
    lidar_records = {str(item["sample_id"]): item for item in lidar_manifest["records"]}
    args.output.mkdir(parents=True, exist_ok=True)
    depth_dir = args.output / "depth"
    depth_dir.mkdir(exist_ok=True)
    reports: list[dict] = []

    for number, view in enumerate(selected, start=1):
        sample_id = str(view["sample_id"])
        image_id, face_id = sample_id.split("::", 1)
        image = images[image_id]
        face_entry = next(item for item in image["faces"] if str(item["face_id"]) == face_id)
        spec = face_specs[(str(image["camera_id"]), face_id)]
        x, y = int(view["x"]), int(view["y"])
        width, height = int(view["width"]), int(view["height"])
        K = np.asarray(spec["K_face"], dtype=np.float64)
        K[0, 2] -= x
        K[1, 2] -= y
        face_to_base = np.eye(4, dtype=np.float64)
        face_to_base[:3, :3] = np.asarray(spec["R_face"], dtype=np.float64)
        c2w = np.asarray(image["c2w"], dtype=np.float64) @ face_to_base
        w2c = np.linalg.inv(c2w)

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(K.astype(np.float32)),
            extrinsic_matrix=o3d.core.Tensor(w2c.astype(np.float32)),
            width_px=width,
            height_px=height,
        )
        cast = scene.cast_rays(rays)
        t_hit = cast["t_hit"].numpy()
        ray_array = rays.numpy()
        valid = np.isfinite(t_hit) & (t_hit > 0.0)
        safe_t = np.where(valid, t_hit, 0.0)
        hit = ray_array[..., :3] + safe_t[..., None] * ray_array[..., 3:]
        origin = ray_array[..., :3]
        depth_range = np.linalg.norm(hit - origin, axis=-1).astype(np.float32)
        normal_world = cast["primitive_normals"].numpy()
        normal_camera = np.einsum("ij,hwj->hwi", w2c[:3, :3], normal_world).astype(np.float32)

        mask_path = args.face_root / str(face_entry["mask_path"])
        face_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if face_mask is None:
            raise FileNotFoundError(mask_path)
        crop_mask = face_mask[y : y + height, x : x + width] > 0
        valid &= crop_mask
        confidence = valid.astype(np.float32)

        lidar_record = lidar_records[sample_id]
        sparse = load_sparse_depth(args.face_lidar_root / str(lidar_record["path"]))
        lidar_depth, _lidar_confidence, lidar_valid = sparse.to_dense()
        lidar_depth = lidar_depth[y : y + height, x : x + width]
        lidar_valid = lidar_valid[y : y + height, x : x + width] & crop_mask
        overlap = valid & lidar_valid
        error = np.abs(depth_range[overlap] - lidar_depth[overlap])

        relative_path = f"depth/{sample_id.replace('::', '__')}.npz"
        payload = mesh_geometry_npz_bytes(
            depth_range, normal_camera, confidence, valid, source_type=1
        )
        destination = args.output / relative_path
        _atomic_bytes(destination, payload)
        record = {
            "sample_id": sample_id,
            "path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "crop": {"x": x, "y": y, "width": width, "height": height},
            "rgb_mask_pixels": int(np.count_nonzero(crop_mask)),
            "mesh_valid_pixels": int(np.count_nonzero(valid)),
            "mesh_valid_fraction": float(np.count_nonzero(valid) / max(np.count_nonzero(crop_mask), 1)),
            "lidar_valid_pixels": int(np.count_nonzero(lidar_valid)),
            "overlap_pixels": int(np.count_nonzero(overlap)),
            "absolute_range_error_m": _summary(error),
        }
        reports.append(record)
        print(
            f"mesh probe {number}/{len(selected)} {sample_id}: "
            f"coverage={record['mesh_valid_fraction']:.3f} "
            f"overlap={record['overlap_pixels']} p95={record['absolute_range_error_m']['p95']}",
            flush=True,
        )

    aggregate_errors = [
        item["absolute_range_error_m"]["p95"]
        for item in reports
        if item["absolute_range_error_m"]["p95"] is not None
    ]
    report = {
        "schema_version": 1,
        "kind": "face4_lidar_mesh_depth_normal_probe",
        "status": "PROBE_NOT_TRAINER_INPUT",
        "tile_id": args.tile_id,
        "mesh": {"path": str(args.mesh.resolve()), "sha256": _sha256(args.mesh)},
        "camera_convention": "face_c2w=base_c2w@R_face; OpenCV pinhole rays",
        "depth_semantics": "euclidean_camera_ray_range_m",
        "normal_semantics": "unit_vector_in_face_camera_coordinates",
        "confidence_semantics": "binary_mesh_hit_probe_only",
        "selected_view_count": len(reports),
        "summary": {
            "mesh_valid_fraction_p50": float(np.median([item["mesh_valid_fraction"] for item in reports])),
            "per_view_range_error_p95_p50_m": float(np.median(aggregate_errors)) if aggregate_errors else None,
        },
        "records": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "mesh_face4_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    mesh_manifest = sign_mesh_geometry_manifest(
        {
            "schema_version": MESH_GEOMETRY_SCHEMA_VERSION,
            "kind": MESH_GEOMETRY_KIND,
            "split": face_manifest["split"],
            "tile_id": args.tile_id,
            "source_face_manifest_sha256": face_manifest["face_manifest_sha256"],
            "source_tile_inputs_manifest_sha256": tile_manifest.get(
                "tile_inputs_manifest_sha256"
            ),
            "source_face_lidar_geometry_manifest_sha256": lidar_manifest[
                "face_lidar_geometry_manifest_sha256"
            ],
            "mesh": {
                "path": str(args.mesh.resolve()),
                "sha256": _sha256(args.mesh),
                "topology_algorithm": "OPEN3D_BPA_CANDIDATE_NOT_VENDOR_CONFIRMED",
            },
            "depth_semantics": "euclidean_camera_ray_range_m",
            "normal_semantics": "unit_vector_in_face_camera_coordinates",
            "confidence_semantics": "binary_mesh_hit_before_cross_view_filter",
            "source_type_table": {"0": "invalid", "1": "lidar_mesh"},
            "complete_face_cache": len(reports) == len(tile["views"]),
            "expected_face_count": len(tile["views"]),
            "records": reports,
        }
    )
    (args.output / "mesh_geometry_manifest.json").write_text(
        json.dumps(mesh_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
