"""Build a signed Tile DA2 manifest aligned to LiDAR-mesh depth.

The relative DA2 cache is reused byte-for-byte.  Only the per-view metric
alignment is recomputed, at the native DA2 raster, from every valid DA2 pixel
whose source-pixel centre has a valid rasterized LiDAR-mesh depth sample.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.mesh_geometry import verify_mesh_geometry_manifest
from cloudstudio_3dgs.data.mono_depth import (
    fit_metric_affine_ransac,
    fit_metric_affine_ransac_torch,
    sign_mono_depth_manifest,
    verify_mono_depth_manifest,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_da2_mesh_pairs(
    relative: np.ndarray,
    metric_crop: np.ndarray,
    valid_crop: np.ndarray,
    *,
    source_shape: tuple[int, int],
    crop: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all common DA2-native and nearest mesh-raster samples."""
    relative = np.asarray(relative, dtype=np.float32)
    metric_crop = np.asarray(metric_crop, dtype=np.float32)
    valid_crop = np.asarray(valid_crop, dtype=bool)
    if metric_crop.shape != valid_crop.shape:
        raise ValueError("mesh depth and valid mask shapes differ")
    source_h, source_w = (int(source_shape[0]), int(source_shape[1]))
    target_h, target_w = relative.shape
    source_x = (np.arange(target_w, dtype=np.float64) + 0.5) * source_w / target_w - 0.5
    source_y = (np.arange(target_h, dtype=np.float64) + 0.5) * source_h / target_h - 0.5
    crop_x = int(crop["x"])
    crop_y = int(crop["y"])
    mesh_x = np.rint(source_x - crop_x).astype(np.int64)
    mesh_y = np.rint(source_y - crop_y).astype(np.int64)
    inside_x = (mesh_x >= 0) & (mesh_x < metric_crop.shape[1])
    inside_y = (mesh_y >= 0) & (mesh_y < metric_crop.shape[0])
    target_x = np.nonzero(inside_x)[0]
    target_y = np.nonzero(inside_y)[0]
    if not len(target_x) or not len(target_y):
        return np.empty(0, np.float32), np.empty(0, np.float32)
    yy, xx = np.meshgrid(target_y, target_x, indexing="ij")
    my = mesh_y[yy]
    mx = mesh_x[xx]
    keep = valid_crop[my, mx]
    keep &= np.isfinite(metric_crop[my, mx]) & (metric_crop[my, mx] > 0.0)
    keep &= np.isfinite(relative[yy, xx]) & (relative[yy, xx] > 0.0)
    return relative[yy, xx][keep], metric_crop[my, mx][keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--mono-manifest", type=Path, required=True)
    parser.add_argument("--mono-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    mesh_manifest = _read(args.mesh_manifest)
    mesh_sha = verify_mesh_geometry_manifest(mesh_manifest)
    mono_manifest = _read(args.mono_manifest)
    mono_sha = verify_mono_depth_manifest(mono_manifest)
    mono_records = {str(item["sample_id"]): item for item in mono_manifest["records"]}
    records: list[dict] = []
    for index, mesh_record in enumerate(mesh_manifest["records"]):
        sample_id = str(mesh_record["sample_id"])
        mono_id = sample_id.replace("::", "__")
        source = mono_records[mono_id]
        mono_path = args.mono_root / str(source["path"])
        mesh_path = args.mesh_root / str(mesh_record["path"])
        if _sha256(mono_path) != source["sha256"]:
            raise ValueError(f"DA2 cache SHA mismatch: {mono_id}")
        if _sha256(mesh_path) != mesh_record["sha256"]:
            raise ValueError(f"mesh cache SHA mismatch: {sample_id}")
        with np.load(mono_path, allow_pickle=False) as payload:
            relative = np.asarray(payload["relative_depth"], dtype=np.float32)
        with np.load(mesh_path, allow_pickle=False) as payload:
            metric = np.asarray(payload["depth_range_m"], dtype=np.float32)
            valid = np.asarray(payload["valid"], dtype=bool)
        mono_values, metric_values = native_da2_mesh_pairs(
            relative,
            metric,
            valid,
            source_shape=tuple(source["source_shape"]),
            crop=mesh_record["crop"],
        )
        seed_bytes = hashlib.sha256(
            f"{args.seed}:{sample_id}".encode("utf-8")
        ).digest()
        fit_seed = int.from_bytes(seed_bytes[:8], "little")
        alignment = (
            fit_metric_affine_ransac_torch(
                mono_values,
                metric_values,
                seed=fit_seed,
                device=args.device,
            )
            if args.device.startswith("cuda")
            else fit_metric_affine_ransac(
                mono_values, metric_values, seed=fit_seed
            )
        )
        record = copy.deepcopy(source)
        record["sample_id"] = mono_id
        record["alignment"] = alignment
        record["alignment_source"] = {
            "kind": "lidar_mesh_native_da2_grid",
            "mesh_sample_id": sample_id,
            "mesh_record_sha256": mesh_record["sha256"],
            "mesh_crop": mesh_record["crop"],
            "sampling": "all_positive_da2_native_pixels_with_nearest_valid_mesh_raster_hit",
        }
        records.append(record)
        print(
            f"DA2 mesh-native {index + 1}/{len(mesh_manifest['records'])} "
            f"{sample_id}: valid={alignment['valid']} "
            f"pairs={alignment['pair_count']} ratio={alignment['inlier_ratio']:.4f}",
            flush=True,
        )

    valid_count = sum(bool(item["alignment"]["valid"]) for item in records)
    output = sign_mono_depth_manifest(
        {
            "schema_version": mono_manifest["schema_version"],
            "kind": mono_manifest["kind"],
            "split": mono_manifest["split"],
            "scope": {"kind": "tile", "tile_id": mesh_manifest["tile_id"]},
            "source_face_manifest_sha256": mono_manifest["source_face_manifest_sha256"],
            "dataset_manifest_sha256": mono_manifest.get("dataset_manifest_sha256"),
            "lidar_depth_manifest_sha256": mono_manifest.get("lidar_depth_manifest_sha256"),
            "source_mono_depth_manifest_sha256": mono_sha,
            "source_mesh_geometry_manifest_sha256": mesh_sha,
            "model": mono_manifest["model"],
            "metric_alignment": {
                **mono_manifest["metric_alignment"],
                "metric_source": "rasterized_lidar_mesh_candidate",
                "pair_grid": "native_da2_output",
                "pair_sampling": "all_common_valid_pixels",
                "invalid_view_policy": "disable_da2_loss_for_view",
            },
            "complete_face_cache": len(records) == int(mesh_manifest["expected_face_count"]),
            "expected_face_count": int(mesh_manifest["expected_face_count"]),
            "records": sorted(records, key=lambda item: item["sample_id"]),
            "summary": {
                "face_count": len(records),
                "valid_alignment_count": valid_count,
                "invalid_alignment_count": len(records) - valid_count,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"signed Tile DA2 mesh alignment: {valid_count}/{len(records)} "
        f"sha={output['mono_depth_manifest_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
