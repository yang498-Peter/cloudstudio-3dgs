#!/usr/bin/env python3
"""Audit metric KNN scale and effective MCMC LR/noise without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.scale_calibration import (
    MetricScaleCalibrationConfig,
    build_metric_scale_calibration,
    verify_metric_scale_calibration_report,
)
from cloudstudio_3dgs.training.trainer import load_initialization_ply


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"metric scale evidence already exists: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("knn", "fixed"), default="knn")
    parser.add_argument("--fixed-scale-m", type=float, default=0.05)
    parser.add_argument("--knn-neighbors", type=int, default=3)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--clamp-min-ratio", type=float, default=0.25)
    parser.add_argument("--clamp-max-ratio", type=float, default=4.0)
    parser.add_argument("--configured-means-lr", type=float, default=1.6e-4)
    parser.add_argument("--configured-noise-lr", type=float, default=500_000.0)
    parser.add_argument("--means-step-fraction", type=float, default=0.0032)
    parser.add_argument("--noise-std-fraction", type=float, default=0.25)
    parser.add_argument(
        "--explicit-lr-noise",
        action="store_true",
        help="preserve configured means/noise LR instead of metric calibration",
    )
    args = parser.parse_args()

    xyz, _ = load_initialization_ply(args.ply)
    policy = MetricScaleCalibrationConfig(
        mode=args.mode,
        knn_neighbors=args.knn_neighbors,
        scale_multiplier=args.scale_multiplier,
        clamp_min_ratio=args.clamp_min_ratio,
        clamp_max_ratio=args.clamp_max_ratio,
        means_step_fraction=None if args.explicit_lr_noise else args.means_step_fraction,
        noise_std_fraction=None if args.explicit_lr_noise else args.noise_std_fraction,
    )
    _, calibration = build_metric_scale_calibration(
        xyz,
        policy=policy,
        fixed_scale_m=args.fixed_scale_m,
        configured_means_lr=args.configured_means_lr,
        configured_noise_lr=args.configured_noise_lr,
    )
    verify_metric_scale_calibration_report(calibration)
    evidence = {
        "schema_version": 1,
        "evidence_type": "cloudstudio_metric_scale_calibration",
        "source": {
            "file_name": args.ply.name,
            "size_bytes": args.ply.stat().st_size,
            "sha256": hashlib.sha256(args.ply.read_bytes()).hexdigest(),
        },
        "calibration": calibration,
        "training": "NOT_RUN",
    }
    _atomic_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
