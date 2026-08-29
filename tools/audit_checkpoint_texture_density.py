#!/usr/bin/env python3
"""Compare checkpoint density redistribution with recovered image texture.

The reference CSV is the read-only MipMap 0.5 m voxel/image audit.  This tool
does not reuse the competitor Gaussian counts as supervision.  It only reuses
the already measured per-voxel image gradient/entropy and compares them with
the change from CloudStudio's common LiDAR initialization to each checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.trainer import load_initialization_ply


VOXEL_SIZE_M = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _encode_voxels(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64) + (1 << 20)
    if np.any(values < 0) or np.any(values >= (1 << 21)):
        raise ValueError("voxel index exceeds signed 21-bit encoding")
    return (values[:, 0] << 42) | (values[:, 1] << 21) | values[:, 2]


def _counts_by_key(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = _encode_voxels(np.floor(xyz / VOXEL_SIZE_M).astype(np.int64))
    return np.unique(keys, return_counts=True)


def _lookup_counts(
    row_keys: np.ndarray, unique_keys: np.ndarray, unique_counts: np.ndarray
) -> np.ndarray:
    positions = np.searchsorted(unique_keys, row_keys)
    valid = positions < len(unique_keys)
    matched = np.zeros(len(row_keys), dtype=bool)
    matched[valid] = unique_keys[positions[valid]] == row_keys[valid]
    counts = np.zeros(len(row_keys), dtype=np.int64)
    counts[matched] = unique_counts[positions[matched]]
    return counts


def _spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3 or np.unique(x[finite]).size < 2:
        return {"n": int(finite.sum()), "rho": None, "p_value": None}
    result = spearmanr(x[finite], y[finite])
    return {
        "n": int(finite.sum()),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _quintiles(
    predictor: np.ndarray, response: np.ndarray
) -> list[dict[str, float | int]]:
    finite = np.isfinite(predictor) & np.isfinite(response)
    order = np.argsort(predictor[finite], kind="stable")
    x = predictor[finite][order]
    y = response[finite][order]
    bins = np.array_split(np.arange(len(x)), 5)
    result: list[dict[str, float | int]] = []
    for index, selected in enumerate(bins, start=1):
        result.append(
            {
                "bin": index,
                "n": int(len(selected)),
                "predictor_median": float(np.median(x[selected])),
                "response_p25": float(np.quantile(y[selected], 0.25)),
                "response_median": float(np.median(y[selected])),
                "response_p75": float(np.quantile(y[selected], 0.75)),
            }
        )
    return result


def _read_reference(path: Path, tile: int | None) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if tile is None or int(row["tile"]) == tile:
                rows.append(row)
    if not rows:
        raise ValueError(f"reference CSV has no rows for tile={tile}")
    fields = (
        "gradient_median_depth",
        "entropy_median_depth_bits",
        "cross_view_luma_mad_depth_proxy",
        "n_las",
    )
    # Halo voxels may occur in more than one competitor Tile.  For an "all"
    # audit, collapse them to one texture record instead of giving boundary
    # voxels extra weight merely because the competitor partition overlaps.
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["voxel_key"]), []).append(row)
    row_keys = np.asarray(sorted(grouped), dtype=np.int64)
    result: dict[str, np.ndarray] = {"voxel_key": row_keys}
    for field in fields:
        result[field] = np.asarray(
            [
                float(np.median([float(row[field]) for row in grouped[int(key)]]))
                for key in row_keys
            ],
            dtype=np.float64,
        )
    return result


def _parse_tile(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("tile must be an integer or 'all'") from error


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("arm name must not be empty")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--initialization-ply", required=True, type=Path)
    parser.add_argument("--tile", type=_parse_tile, default=1)
    parser.add_argument("--arm", action="append", required=True, type=_parse_arm)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference = _read_reference(args.reference_csv, args.tile)
    row_keys = reference["voxel_key"]
    initial_xyz, _ = load_initialization_ply(args.initialization_ply)
    initial_keys, initial_values = _counts_by_key(initial_xyz.astype(np.float64))
    initial_counts = _lookup_counts(row_keys, initial_keys, initial_values)
    stable = initial_counts > 0
    if int(stable.sum()) < 25:
        raise ValueError("too few reference voxels overlap the initialization")

    reports: dict[str, Any] = {}
    for name, checkpoint_path in args.arm:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        means = payload["params"]["means"].detach().cpu().numpy().astype(np.float64)
        opacity = torch.sigmoid(payload["params"]["opacities"].detach()).cpu().numpy().reshape(-1)
        keys, values = _counts_by_key(means)
        counts = _lookup_counts(row_keys, keys, values)
        delta_fraction = np.zeros(len(row_keys), dtype=np.float64)
        delta_fraction[stable] = (
            counts[stable].astype(np.float64) - initial_counts[stable]
        ) / initial_counts[stable]
        retained_fraction = np.zeros(len(row_keys), dtype=np.float64)
        retained_fraction[stable] = counts[stable] / initial_counts[stable]

        correlations = {
            field: _spearman(reference[field][stable], delta_fraction[stable])
            for field in (
                "gradient_median_depth",
                "entropy_median_depth_bits",
                "cross_view_luma_mad_depth_proxy",
            )
        }
        reports[name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "completed_steps": int(payload.get("step", -1)),
            "gaussian_count": int(len(means)),
            "overlapping_reference_voxels": int(stable.sum()),
            "voxel_change": {
                "delta_fraction_p25": float(np.quantile(delta_fraction[stable], 0.25)),
                "delta_fraction_p50": float(np.quantile(delta_fraction[stable], 0.50)),
                "delta_fraction_p75": float(np.quantile(delta_fraction[stable], 0.75)),
                "retained_fraction_min": float(np.min(retained_fraction[stable])),
            },
            "opacity": {
                "below_0_005_fraction": float(np.mean(opacity < 0.005)),
                "below_0_1_fraction": float(np.mean(opacity < 0.1)),
                "above_0_1_count": int(np.sum(opacity > 0.1)),
                "p50": float(np.quantile(opacity, 0.5)),
            },
            "spearman_delta_fraction": correlations,
            "gradient_quintiles_delta_fraction": _quintiles(
                reference["gradient_median_depth"][stable], delta_fraction[stable]
            ),
            "entropy_quintiles_delta_fraction": _quintiles(
                reference["entropy_median_depth_bits"][stable], delta_fraction[stable]
            ),
        }
        del payload, means, opacity

    output = {
        "schema_version": 1,
        "kind": "cloudstudio_checkpoint_texture_density_audit_v1",
        "reference_tile": "all" if args.tile is None else args.tile,
        "initialization_voxel_count": int(len(initial_keys)),
        "overlapping_reference_voxel_count": int(stable.sum()),
        "overlapping_initialization_voxel_fraction": float(stable.sum() / len(initial_keys)),
        "voxel_size_m": VOXEL_SIZE_M,
        "reference_csv": str(args.reference_csv.resolve()),
        "reference_csv_sha256": _sha256(args.reference_csv),
        "initialization_ply": str(args.initialization_ply.resolve()),
        "initialization_ply_sha256": _sha256(args.initialization_ply),
        "interpretation": (
            "response is (checkpoint voxel count - common LiDAR initialization "
            "count) / initialization count; competitor GS counts are not used"
        ),
        "arms": reports,
    }
    _write_json(args.output, output)
    print(f"texture-density audit -> {args.output}")
    for name, report in reports.items():
        gradient = report["spearman_delta_fraction"]["gradient_median_depth"]["rho"]
        luma = report["spearman_delta_fraction"]["cross_view_luma_mad_depth_proxy"]["rho"]
        print(f"{name}: gradient rho={gradient:.4f}, luma-MAD rho={luma:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
