#!/usr/bin/env python3
"""Bind all Tile Face4 LiDAR manifests, audits, and component smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.face_lidar_geometry import verify_face_lidar_geometry_manifest
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_named_signature(payload: dict[str, Any], field: str) -> str:
    expected = str(payload.get(field, ""))
    if len(expected) != 64:
        raise ValueError(f"artifact lacks {field}")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(f"{field} mismatch")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--smoke-suffix", default="v24e")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    tile_inputs = _read(args.tile_inputs)
    tile_inputs_sha = verify_tile_inputs_manifest(
        tile_inputs, root=args.tile_inputs_root, verify_artifacts=True
    )
    entries: list[dict[str, Any]] = []
    total_views = 0
    total_pixels = 0
    total_bytes = 0
    for tile in tile_inputs["tiles"]:
        tile_id = int(tile["tile_id"])
        tile_root = args.geometry_root / f"Tile_{tile_id}"
        manifest_path = tile_root / "face_lidar_geometry_manifest.json"
        audit_path = tile_root / "content_audit.json"
        smoke_path = args.smoke_root / (
            f"lidar_first_face4_gpu_smoke_tile{tile_id}_{args.smoke_suffix}.json"
        )
        manifest = _read(manifest_path)
        manifest_sha = verify_face_lidar_geometry_manifest(manifest)
        if (
            int(manifest.get("tile_id", -1)) != tile_id
            or manifest.get("tile_inputs_manifest_sha256") != tile_inputs_sha
            or manifest.get("tile_name") != tile["name"]
        ):
            raise ValueError(f"Tile {tile_id} geometry binding mismatch")
        audit = _read(audit_path)
        audit_sha = _verify_named_signature(audit, "audit_sha256")
        if (
            audit.get("status") != "CONSUMPTION_READY"
            or audit.get("face_lidar_geometry_manifest_sha256") != manifest_sha
        ):
            raise ValueError(f"Tile {tile_id} content audit is not ready")
        smoke = _read(smoke_path)
        smoke_sha = _verify_named_signature(smoke, "gpu_smoke_sha256")
        if smoke.get("status") != "PASS" or int(smoke.get("tile_id", -1)) != tile_id:
            raise ValueError(f"Tile {tile_id} component smoke did not pass")
        bytes_for_tile = sum(
            path.stat().st_size for path in (tile_root / "depth").glob("*.npz")
        )
        total_views += int(manifest["view_count"])
        total_pixels += int(manifest["tile_valid_pixels"])
        total_bytes += bytes_for_tile
        entries.append(
            {
                "tile_id": tile_id,
                "tile_name": tile["name"],
                "view_count": int(manifest["view_count"]),
                "nonempty_view_count": int(manifest["nonempty_view_count"]),
                "tile_valid_pixels": int(manifest["tile_valid_pixels"]),
                "depth_bytes": bytes_for_tile,
                "geometry_manifest": {
                    "path": manifest_path.relative_to(args.geometry_root).as_posix(),
                    "sha256": manifest_sha,
                },
                "content_audit": {
                    "path": audit_path.relative_to(args.geometry_root).as_posix(),
                    "sha256": audit_sha,
                },
                "component_gpu_smoke": {
                    "path": smoke_path.relative_to(args.smoke_root).as_posix(),
                    "sha256": smoke_sha,
                    "peak_cuda_memory_mib": int(smoke["peak_cuda_memory_mib"]),
                },
            }
        )

    unsigned = {
        "schema_version": 1,
        "kind": "tile_face4_lidar_geometry_bundle_v1",
        "status": "CONSUMPTION_READY_COMPONENT_SMOKE_PASS",
        "tile_inputs_manifest_sha256": tile_inputs_sha,
        "tile_count": len(entries),
        "total_view_instances_with_overlap": total_views,
        "total_tile_valid_pixels_with_overlap": total_pixels,
        "total_depth_bytes": total_bytes,
        "tiles": entries,
        "da2_used": False,
        "mesh_used": False,
        "full_las_reprojection_performed": False,
        "training_allowed": False,
        "next_required_artifact": "short_tile_trainer_raster_smoke",
    }
    bundle = dict(unsigned)
    bundle["bundle_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": bundle["status"],
                "tile_count": bundle["tile_count"],
                "total_view_instances_with_overlap": total_views,
                "total_tile_valid_pixels_with_overlap": total_pixels,
                "total_depth_gib": total_bytes / (1024**3),
                "bundle_sha256": bundle["bundle_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
