"""Build a signed mesh sidecar admitted only by source types 2 and 3.

The source cache remains untouched. A whole view is disabled when its measured
LiDAR-overlap absolute range error P95 exceeds the configured threshold.
"""

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

from cloudstudio_3dgs.data.mesh_geometry import (
    STRICT_MESH_SOURCE_TYPES,
    mesh_geometry_npz_bytes,
    sign_mesh_geometry_manifest,
    strict_mesh_admission_mask,
    verify_mesh_geometry_manifest,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _view_p95(record: dict) -> float | None:
    value = record.get("absolute_range_error_m", {}).get("p95")
    return None if value is None else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-view-p95-m", type=float, default=0.10)
    args = parser.parse_args()
    if args.max_view_p95_m <= 0.0:
        raise ValueError("max-view-p95-m must be positive")

    source = _read(args.source_manifest)
    source_sha = verify_mesh_geometry_manifest(source)
    records: list[dict] = []
    totals = {
        "source_valid": 0,
        "admitted_valid": 0,
        "source_type_2": 0,
        "source_type_3": 0,
        "source_type_4_rejected": 0,
        "views_enabled": 0,
        "views_disabled_p95": 0,
        "views_disabled_missing_p95": 0,
    }
    disabled_views: list[dict] = []
    for index, record in enumerate(source["records"], start=1):
        source_path = args.source_root / str(record["path"])
        with np.load(source_path, allow_pickle=False) as payload:
            required = {
                "depth_range_m",
                "normal_camera",
                "confidence",
                "valid",
                "source_type",
            }
            if not required.issubset(payload.files):
                raise ValueError(f"mesh sidecar lacks source_type: {source_path}")
            depth = np.asarray(payload["depth_range_m"], dtype=np.float32)
            normal = np.asarray(payload["normal_camera"], dtype=np.float32)
            confidence = np.asarray(payload["confidence"], dtype=np.float32)
            source_valid = np.asarray(payload["valid"], dtype=bool)
            source_type = np.asarray(payload["source_type"], dtype=np.uint8)

        admitted = strict_mesh_admission_mask(source_valid, source_type)
        p95 = _view_p95(record)
        disable_reason = None
        if p95 is None:
            disable_reason = "missing_lidar_overlap_p95"
            totals["views_disabled_missing_p95"] += 1
        elif p95 > args.max_view_p95_m:
            disable_reason = "lidar_overlap_p95_exceeds_threshold"
            totals["views_disabled_p95"] += 1
        else:
            totals["views_enabled"] += 1
        if disable_reason is not None:
            admitted[:] = False
            disabled_views.append(
                {
                    "sample_id": str(record["sample_id"]),
                    "reason": disable_reason,
                    "absolute_range_error_p95_m": p95,
                }
            )

        encoded = mesh_geometry_npz_bytes(
            depth,
            normal,
            confidence,
            admitted,
            source_type=source_type,
        )
        relative = f"depth/{str(record['sample_id']).replace('::', '__')}.npz"
        _atomic_bytes(args.output / relative, encoded)
        admitted_count = int(np.count_nonzero(admitted))
        output_record = dict(record)
        output_record.update(
            {
                "path": relative,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "mesh_valid_pixels": admitted_count,
                "mesh_valid_fraction": float(
                    admitted_count / max(int(record.get("rgb_mask_pixels", 0)), 1)
                ),
                "mesh_depth_enabled": disable_reason is None,
                "mesh_depth_disable_reason": disable_reason,
                "admission_source_types": list(STRICT_MESH_SOURCE_TYPES),
            }
        )
        records.append(output_record)
        totals["source_valid"] += int(np.count_nonzero(source_valid))
        totals["admitted_valid"] += admitted_count
        totals["source_type_2"] += int(np.count_nonzero(source_valid & (source_type == 2)))
        totals["source_type_3"] += int(np.count_nonzero(source_valid & (source_type == 3)))
        totals["source_type_4_rejected"] += int(
            np.count_nonzero(source_valid & (source_type == 4))
        )
        print(
            f"strict mesh {index}/{len(source['records'])} "
            f"{record['sample_id']}: admitted={admitted_count} "
            f"enabled={disable_reason is None}",
            flush=True,
        )

    output = dict(source)
    output.update(
        {
            "source_mesh_geometry_manifest_sha256": source_sha,
            "confidence_semantics": (
                "1=native_lidar_anchor; 0.6..1=cross_view_support; "
                "source_type_4_always_invalid"
            ),
            "admission_policy": {
                "status": "STRICT_SOURCE_TYPE_AND_VIEW_P95_APPLIED",
                "allowed_source_types": list(STRICT_MESH_SOURCE_TYPES),
                "rejected_source_types": [4],
                "max_view_absolute_range_error_p95_m": args.max_view_p95_m,
                "missing_view_p95_policy": "disable_mesh_depth_for_view",
                "totals": totals,
                "disabled_views": disabled_views,
            },
            "records": records,
            "complete_face_cache": len(records) == int(source["expected_face_count"]),
        }
    )
    output.pop("mesh_geometry_manifest_sha256", None)
    signed = sign_mesh_geometry_manifest(output)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "mesh_geometry_manifest.json"
    manifest_path.write_text(
        json.dumps(signed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path, flush=True)
    print(signed["mesh_geometry_manifest_sha256"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
