"""Audit per-view DA2 affine calibration against rasterized LiDAR mesh depth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.mono_depth import (
    fit_metric_affine_ransac,
    fit_metric_affine_ransac_torch,
    sample_bilinear_at_source_pixels,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-probe", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--mono-manifest", type=Path, required=True)
    parser.add_argument("--mono-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=100_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    probe = _read(args.mesh_probe)
    mono_manifest = _read(args.mono_manifest)
    mono_records = {str(item["sample_id"]): item for item in mono_manifest["records"]}
    reports: list[dict] = []
    for index, mesh_record in enumerate(probe["records"]):
        sample_id = str(mesh_record["sample_id"])
        mono_id = sample_id.replace("::", "__")
        mono_record = mono_records[mono_id]
        with np.load(args.mesh_root / str(mesh_record["path"]), allow_pickle=False) as payload:
            metric = np.asarray(payload["depth_range_m"], dtype=np.float32)
            valid = np.asarray(payload["valid"], dtype=bool)
        with np.load(args.mono_root / str(mono_record["path"]), allow_pickle=False) as payload:
            relative = np.asarray(payload["relative_depth"], dtype=np.float32)
        ys, xs = np.nonzero(valid)
        rng = np.random.default_rng(args.seed + index)
        if len(xs) > args.max_pairs:
            chosen = rng.choice(len(xs), args.max_pairs, replace=False)
            ys, xs = ys[chosen], xs[chosen]
        crop = mesh_record["crop"]
        source_x = xs.astype(np.float64) + float(crop["x"])
        source_y = ys.astype(np.float64) + float(crop["y"])
        mono_values = sample_bilinear_at_source_pixels(
            relative,
            source_x,
            source_y,
            source_shape=tuple(mono_record["source_shape"]),
        )
        metric_values = metric[ys, xs]
        fit = (
            fit_metric_affine_ransac_torch(
                mono_values,
                metric_values,
                seed=args.seed + index,
                device=args.device,
            )
            if args.device.startswith("cuda")
            else fit_metric_affine_ransac(
                mono_values, metric_values, seed=args.seed + index
            )
        )
        reports.append(
            {
                "sample_id": sample_id,
                "sampled_mesh_pairs": int(len(xs)),
                "mesh_alignment": fit,
                "old_sparse_lidar_alignment": mono_record.get("alignment"),
            }
        )
        print(
            f"DA2 mesh align {index + 1}/{len(probe['records'])} {sample_id}: "
            f"valid={fit['valid']} pairs={fit['pair_count']} "
            f"ratio={fit['inlier_ratio']:.4f} rmse={fit['rmse_m']}",
            flush=True,
        )
    valid_count = sum(bool(item["mesh_alignment"]["valid"]) for item in reports)
    output = {
        "schema_version": 1,
        "kind": "da2_mesh_affine_alignment_probe",
        "status": "PASS_PROBE" if valid_count == len(reports) else "FAIL_PROBE",
        "metric_relation": "mesh_range_m = scale * da2_relative_depth + shift",
        "pair_sampling": {
            "max_pairs_per_view": args.max_pairs,
            "seed": args.seed,
            "note": "bounded probe only; production alignment must use every valid mesh/mono pair",
        },
        "valid_count": valid_count,
        "view_count": len(reports),
        "records": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"alignment probe: {valid_count}/{len(reports)}", flush=True)
    return 0 if valid_count == len(reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
