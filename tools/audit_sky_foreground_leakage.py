#!/usr/bin/env python3
"""Fail closed when an independent sky layer changes measured foreground pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def _atomic_json(path: Path, value: dict) -> None:
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


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def audit(
    surface_evaluation: Path,
    composed_evaluation: Path,
    *,
    maximum_mean_rgb_delta: float,
    maximum_fraction_over_0_05: float,
    minimum_foreground_alpha: float,
) -> dict:
    surface_evaluation = Path(surface_evaluation)
    composed_evaluation = Path(composed_evaluation)
    image_ids = sorted(
        path.name.rsplit("_", 1)[0]
        for path in surface_evaluation.glob("*_rendered.png")
    )
    if not image_ids:
        raise FileNotFoundError("surface evaluation contains no rendered frames")

    records = []
    all_delta = []
    all_alpha = []
    for image_id in image_ids:
        surface_path = surface_evaluation / f"{image_id}_rendered.png"
        composed_path = composed_evaluation / f"{image_id}_rendered.png"
        alpha_path = surface_evaluation / f"{image_id}_alpha.npy"
        lidar_path = surface_evaluation / f"{image_id}_lidar.npz"
        for path in (surface_path, composed_path, alpha_path, lidar_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing leakage input: {path}")
        surface = _rgb(surface_path)
        composed = _rgb(composed_path)
        alpha = np.asarray(np.load(alpha_path), dtype=np.float32)
        sparse = load_sparse_depth(lidar_path)
        _, _, lidar_valid = sparse.to_dense()
        if surface.shape != composed.shape or alpha.shape != lidar_valid.shape:
            raise ValueError(f"leakage input shape mismatch for {image_id}")
        foreground = lidar_valid & np.isfinite(alpha)
        if not foreground.any():
            continue
        delta = np.mean(np.abs(composed - surface), axis=2)[foreground]
        foreground_alpha = alpha[foreground]
        all_delta.append(delta)
        all_alpha.append(foreground_alpha)
        records.append(
            {
                "image_id": image_id,
                "foreground_pixels": int(len(delta)),
                "foreground_alpha_mean": float(foreground_alpha.mean()),
                "foreground_alpha_below_minimum_fraction": float(
                    np.mean(foreground_alpha < minimum_foreground_alpha)
                ),
                "sky_rgb_delta_mean": float(delta.mean()),
                "sky_rgb_delta_p95": float(np.quantile(delta, 0.95)),
                "sky_rgb_delta_over_0_05_fraction": float(np.mean(delta > 0.05)),
            }
        )
    if not all_delta:
        raise ValueError("leakage audit selected no LiDAR-supported foreground pixels")
    delta = np.concatenate(all_delta)
    alpha = np.concatenate(all_alpha)
    checks = {
        "foreground_alpha_mean_at_least_minimum": float(alpha.mean())
        >= minimum_foreground_alpha,
        "sky_mean_rgb_delta_at_most_limit": float(delta.mean())
        <= maximum_mean_rgb_delta,
        "sky_rgb_delta_over_0_05_fraction_at_most_limit": float(
            np.mean(delta > 0.05)
        )
        <= maximum_fraction_over_0_05,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "schema_version": 1,
        "kind": "sky_foreground_leakage_audit_v1",
        "status": "PASS" if not failed else "FAIL",
        "promotion_eligible": not failed,
        "failed_checks": failed,
        "checks": checks,
        "thresholds": {
            "minimum_foreground_alpha": minimum_foreground_alpha,
            "maximum_mean_rgb_delta": maximum_mean_rgb_delta,
            "maximum_fraction_over_0_05": maximum_fraction_over_0_05,
        },
        "summary": {
            "frame_count": len(records),
            "foreground_pixels": int(len(delta)),
            "foreground_alpha_mean": float(alpha.mean()),
            "foreground_alpha_p05": float(np.quantile(alpha, 0.05)),
            "foreground_alpha_below_minimum_fraction": float(
                np.mean(alpha < minimum_foreground_alpha)
            ),
            "sky_rgb_delta_mean": float(delta.mean()),
            "sky_rgb_delta_p95": float(np.quantile(delta, 0.95)),
            "sky_rgb_delta_over_0_05_fraction": float(np.mean(delta > 0.05)),
        },
        "frames": records,
    }
    payload["audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-evaluation", required=True, type=Path)
    parser.add_argument("--composed-evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-foreground-alpha", type=float, default=0.90)
    parser.add_argument("--maximum-mean-rgb-delta", type=float, default=0.01)
    parser.add_argument("--maximum-fraction-over-0-05", type=float, default=0.02)
    args = parser.parse_args()
    if not 0.0 < args.minimum_foreground_alpha <= 1.0:
        raise ValueError("minimum foreground alpha must be within (0, 1]")
    if args.maximum_mean_rgb_delta < 0.0:
        raise ValueError("maximum mean RGB delta must be non-negative")
    if not 0.0 <= args.maximum_fraction_over_0_05 <= 1.0:
        raise ValueError("maximum delta fraction must be within [0, 1]")
    report = audit(
        args.surface_evaluation,
        args.composed_evaluation,
        maximum_mean_rgb_delta=args.maximum_mean_rgb_delta,
        maximum_fraction_over_0_05=args.maximum_fraction_over_0_05,
        minimum_foreground_alpha=args.minimum_foreground_alpha,
    )
    _atomic_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "failed_checks", "summary")}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
