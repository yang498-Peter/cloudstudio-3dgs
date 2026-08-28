#!/usr/bin/env python3
"""Audit real Face4 sparse-LiDAR consumption for every adaptive Tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.face_lidar_geometry import (  # noqa: E402
    verify_face_lidar_geometry_manifest,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes  # noqa: E402
from cloudstudio_3dgs.data.renderer_masks import (  # noqa: E402
    verify_renderer_mask_manifest,
)
from cloudstudio_3dgs.training.face_dataset import (  # noqa: E402
    FaceCacheDataset,
    verify_face_manifest,
)
from cloudstudio_3dgs.training.tile_inputs import (  # noqa: E402
    verify_tile_inputs_manifest,
)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--face-root", required=True, type=Path)
    parser.add_argument("--renderer-mask-manifest", required=True, type=Path)
    parser.add_argument("--lidar-geometry-manifest", required=True, type=Path)
    parser.add_argument("--lidar-geometry-root", required=True, type=Path)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument("--gpu-smoke", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    face = _read(args.face_manifest)
    renderer = _read(args.renderer_mask_manifest)
    lidar = _read(args.lidar_geometry_manifest)
    tile_inputs = _read(args.tile_inputs)
    face_sha = verify_face_manifest(face)
    renderer_sha = verify_renderer_mask_manifest(renderer)
    lidar_sha = verify_face_lidar_geometry_manifest(lidar)
    tile_sha = verify_tile_inputs_manifest(
        tile_inputs, root=args.tile_inputs_root, verify_artifacts=True
    )
    if renderer.get("source_face_manifest_sha256") != face_sha:
        raise ValueError("renderer mask is bound to a different Face4 cache")
    if lidar.get("source_face_manifest_sha256") != face_sha:
        raise ValueError("LiDAR geometry is bound to a different Face4 cache")

    lidar_by_sample = {record["sample_id"]: record for record in lidar["records"]}
    tile_reports = []
    for tile in tile_inputs["tiles"]:
        dataset = FaceCacheDataset(
            args.face_manifest,
            args.face_root,
            tile_views=tile["views"],
            renderer_mask_manifest_path=args.renderer_mask_manifest,
            face_lidar_geometry_manifest_path=args.lidar_geometry_manifest,
            face_lidar_geometry_root=args.lidar_geometry_root,
        )
        probe_index = next(
            (
                index
                for index, sample_id in enumerate(dataset.image_ids)
                if int(lidar_by_sample[sample_id]["valid_pixels"]) > 0
            ),
            None,
        )
        if probe_index is None:
            raise ValueError(f"Tile {tile['tile_id']} has no LiDAR-supervised view")
        sample = dataset[probe_index]
        valid_after_crop = int(np.count_nonzero(sample.depth_mask))
        if valid_after_crop <= 0 or sample.depth_range_m is None:
            raise ValueError(
                f"Tile {tile['tile_id']} probe lost all LiDAR pixels after crop"
            )
        tile_reports.append(
            {
                "tile_id": int(tile["tile_id"]),
                "view_count": len(dataset),
                "probe_sample_id": sample.image_id,
                "crop_width": int(sample.width),
                "crop_height": int(sample.height),
                "lidar_pixels_after_crop": valid_after_crop,
                "depth_cache_path": str(sample.depth_cache_path),
            }
        )

    gpu_smoke_binding = None
    if args.gpu_smoke is not None:
        gpu_smoke = _read(args.gpu_smoke)
        expected_smoke_sha = str(gpu_smoke.get("gpu_smoke_sha256", ""))
        unsigned_smoke = dict(gpu_smoke)
        unsigned_smoke.pop("gpu_smoke_sha256", None)
        actual_smoke_sha = hashlib.sha256(
            canonical_json_bytes(unsigned_smoke)
        ).hexdigest()
        if (
            expected_smoke_sha != actual_smoke_sha
            or gpu_smoke.get("status") != "PASS"
        ):
            raise ValueError("GPU smoke is unsigned, tampered, or not PASS")
        gpu_smoke_binding = expected_smoke_sha

    report = {
        "schema_version": 1,
        "kind": "lidar_first_face4_training_readiness_audit",
        "status": (
            "LIDAR_GEOMETRY_AND_BIRTH_GUARD_COMPONENT_SMOKE_PASS"
            if gpu_smoke_binding is not None
            else "LIDAR_GEOMETRY_AND_BIRTH_GUARD_READY_FOR_GPU_SMOKE"
        ),
        "training_allowed": False,
        "bindings": {
            "face_manifest_sha256": face_sha,
            "renderer_mask_manifest_sha256": renderer_sha,
            "face_lidar_geometry_manifest_sha256": lidar_sha,
            "source_depth_manifest_sha256": lidar[
                "source_depth_manifest_sha256"
            ],
            "tile_inputs_manifest_sha256": tile_sha,
            "gpu_component_smoke_sha256": gpu_smoke_binding,
        },
        "geometry_policy": {
            "authority": "REAL_LIDAR",
            "sparse_metric_range_consumed": True,
            "mesh_interpolation": False,
            "da2_consumed": False,
        },
        "birth_policy": {
            "strategy": "classic_gradient_split_clone_cull",
            "parent_gate": "lidar_planarity_and_support",
            "newborn_position": "seeded_local_tangent_surface",
            "unsupported_births": "REJECT_BEFORE_GROWTH",
        },
        "tiles": tile_reports,
        "blocking_reasons": [
            (
                "full Trainer raster smoke has not run"
                if gpu_smoke_binding is not None
                else "short real GPU smoke has not run"
            ),
            "independent sky trainer is not complete",
            "Tile merge/raw-fisheye evaluation/export is not complete",
        ],
    }
    report["readiness_audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"LiDAR-first readiness: tiles={len(tile_reports)}, "
        f"sha256={report['readiness_audit_sha256']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
