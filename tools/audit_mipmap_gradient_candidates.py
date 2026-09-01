"""Compare legacy and recovered MipMap densification candidates by texture voxel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


def _metric(values: np.ndarray, density: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(values) & np.isfinite(density)
    if int(valid.sum()) < 3:
        return {"rho": None, "pvalue": None, "voxel_count": int(valid.sum())}
    rho, pvalue = spearmanr(values[valid], density[valid])
    return {
        "rho": float(rho),
        "pvalue": float(pvalue),
        "voxel_count": int(valid.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.00015)
    parser.add_argument("--opacity-floor", type=float, default=0.15)
    parser.add_argument("--max-distance-m", type=float, default=0.45)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    means = payload["params"]["means"].detach().cpu().numpy()
    opacity = torch.sigmoid(payload["params"]["opacities"].flatten()).cpu().numpy()
    state = payload["strategy_state"]
    legacy = (
        state["grad2d"] / state["count"].clamp_min(1.0)
    ).detach().cpu().numpy()
    weighted = (
        state["_cloudstudio_mipmap_grad_sum"]
        / state["_cloudstudio_mipmap_weight_sum"].clamp_min(1e-8)
    ).detach().cpu().numpy()
    legacy_mask = (legacy > args.threshold) & (opacity > args.opacity_floor)
    weighted_mask = (weighted > args.threshold) & (opacity > args.opacity_floor)

    rows: list[dict[str, str]] = []
    with Path(args.reference_csv).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["tile"]) == args.tile:
                rows.append(row)
    centers = np.asarray(
        [[float(row[f"center_{axis}"]) for axis in "xyz"] for row in rows],
        dtype=np.float64,
    )
    distances, indexes = cKDTree(centers).query(means, k=1)
    assigned = distances <= args.max_distance_m
    legacy_counts = np.bincount(
        indexes[assigned & legacy_mask], minlength=len(rows)
    ).astype(np.float64)
    weighted_counts = np.bincount(
        indexes[assigned & weighted_mask], minlength=len(rows)
    ).astype(np.float64)
    las = np.asarray([float(row["n_las"]) for row in rows])
    legacy_density = legacy_counts / np.maximum(las, 1.0)
    weighted_density = weighted_counts / np.maximum(las, 1.0)
    metrics = {
        name: np.asarray([float(row[name]) for row in rows])
        for name in (
            "gradient_median_all",
            "gradient_p90_all",
            "entropy_median_all_bits",
            "cross_view_luma_mad_depth_proxy",
        )
    }

    correlations = {
        name: {
            "legacy": _metric(values, legacy_density),
            "mipmap_equivalent": _metric(values, weighted_density),
        }
        for name, values in metrics.items()
    }
    quintiles: dict[str, dict[str, list[float]]] = {}
    for name, values in metrics.items():
        finite = np.isfinite(values)
        edges = np.quantile(values[finite], [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(values, edges)
        quintiles[name] = {
            "legacy_candidate_density_median": [
                float(np.median(legacy_density[(bins == index) & finite]))
                for index in range(5)
            ],
            "mipmap_candidate_density_median": [
                float(np.median(weighted_density[(bins == index) & finite]))
                for index in range(5)
            ],
        }

    report = {
        "schema_version": 1,
        "kind": "mipmap_gradient_candidate_texture_audit_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "completed_steps": int(payload["step"]),
        "threshold": args.threshold,
        "opacity_floor": args.opacity_floor,
        "tile": args.tile,
        "reference_voxel_count": len(rows),
        "gaussian_count": int(len(means)),
        "assigned_gaussian_fraction": float(assigned.mean()),
        "candidate_counts": {
            "legacy": int(legacy_mask.sum()),
            "mipmap_equivalent": int(weighted_mask.sum()),
            "overlap": int((legacy_mask & weighted_mask).sum()),
            "mipmap_only": int((weighted_mask & ~legacy_mask).sum()),
        },
        "correlations": correlations,
        "texture_quintiles": quintiles,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["candidate_counts"], ensure_ascii=False))
    for name, result in correlations.items():
        print(name, result)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
