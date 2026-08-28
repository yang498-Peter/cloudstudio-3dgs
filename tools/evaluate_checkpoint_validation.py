#!/usr/bin/env python3
"""Render and score a checkpoint on the signed raw-fisheye validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.image_sample import CropWindow
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.image_metrics import masked_psnr, masked_ssim
from cloudstudio_3dgs.training.backend import GsplatBackend
from cloudstudio_3dgs.training.dataset import S1TrainingDataset
from cloudstudio_3dgs.training.trainer import _save_evaluation_artifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(output_dir: Path) -> list[dict]:
    evaluation = output_dir / "evaluation"
    image_ids = sorted(
        path.name.rsplit("_", 1)[0]
        for path in evaluation.glob("*_rendered.png")
    )
    records: list[dict] = []
    for image_id in image_ids:
        rendered = np.asarray(
            Image.open(evaluation / f"{image_id}_rendered.png"), dtype=np.float32
        ) / 255.0
        reference = np.asarray(
            Image.open(evaluation / f"{image_id}_reference.png"), dtype=np.float32
        ) / 255.0
        mask = np.asarray(Image.open(evaluation / f"{image_id}_mask.png")) > 127
        record = {
            "image_id": image_id,
            "psnr_db": float(masked_psnr(rendered, reference, mask)),
            "ssim": float(masked_ssim(rendered, reference, mask)),
            "valid_fraction": float(mask.mean()),
            "depth_mae_m": None,
            "depth_rmse_m": None,
            "depth_coverage": None,
            "alpha_mean": None,
            "alpha_p05": None,
            "alpha_below_0_95_fraction": None,
            "lidar_alpha_mean": None,
            "lidar_alpha_p05": None,
            "lidar_alpha_below_0_95_fraction": None,
        }
        alpha_path = evaluation / f"{image_id}_alpha.npy"
        if alpha_path.is_file():
            alpha = np.asarray(np.load(alpha_path), dtype=np.float32)
            if alpha.shape != mask.shape:
                raise ValueError(
                    f"alpha shape mismatch for {image_id}: {alpha.shape} != {mask.shape}"
                )
            valid_alpha = alpha[mask & np.isfinite(alpha)]
            if valid_alpha.size:
                record["alpha_mean"] = float(valid_alpha.mean())
                record["alpha_p05"] = float(np.quantile(valid_alpha, 0.05))
                record["alpha_below_0_95_fraction"] = float(
                    np.mean(valid_alpha < 0.95)
                )
        range_path = evaluation / f"{image_id}_range.npy"
        lidar_path = evaluation / f"{image_id}_lidar.npz"
        if range_path.is_file() and lidar_path.is_file():
            rendered_range = np.load(range_path)
            sparse = load_sparse_depth(lidar_path)
            target, _, valid = sparse.to_dense()
            supervised = valid & mask
            if alpha_path.is_file():
                lidar_alpha = alpha[supervised & np.isfinite(alpha)]
                if lidar_alpha.size:
                    record["lidar_alpha_mean"] = float(lidar_alpha.mean())
                    record["lidar_alpha_p05"] = float(
                        np.quantile(lidar_alpha, 0.05)
                    )
                    record["lidar_alpha_below_0_95_fraction"] = float(
                        np.mean(lidar_alpha < 0.95)
                    )
            covered = supervised & np.isfinite(rendered_range) & (rendered_range > 0.0)
            record["depth_coverage"] = float(
                covered.sum() / max(1, int(supervised.sum()))
            )
            if bool(covered.any()):
                error = np.abs(rendered_range[covered] - target[covered])
                record["depth_mae_m"] = float(error.mean())
                record["depth_rmse_m"] = float(np.sqrt(np.square(error).mean()))
        records.append(record)
    return records


def _mean(records: list[dict], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return None if not values else float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--mask-manifest", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--person-mask-manifest", type=Path, default=None)
    parser.add_argument("--person-mask-root", type=Path, default=None)
    parser.add_argument("--surface-gaussian-count", type=int, default=None)
    parser.add_argument("--max-surface-scale-m", type=float, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    factor = int(config["factor"] if args.factor is None else args.factor)
    crop_value = config.get("crop")
    crop = None if crop_value is None else CropWindow(**crop_value)
    dataset = S1TrainingDataset(
        dataset_manifest_path=Path(config["dataset_manifest"]),
        recording_root=Path(config["recording_root"]),
        mask_manifest_path=(
            Path(config["mask_manifest"])
            if args.mask_manifest is None
            else args.mask_manifest
        ),
        mask_root=Path(config["mask_root"]) if args.mask_root is None else args.mask_root,
        split_manifest_path=Path(config["split_manifest"]),
        split="val",
        person_mask_manifest_path=(
            Path(config["person_mask_manifest"])
            if args.person_mask_manifest is None
            else args.person_mask_manifest
        ),
        person_mask_root=(
            Path(config["person_mask_root"])
            if args.person_mask_root is None
            else args.person_mask_root
        ),
        depth_manifest_path=Path(config["depth_manifest"]),
        depth_root=Path(config["depth_root"]),
        factor=factor,
        crop=crop,
    )
    import torch

    backend = GsplatBackend(
        device=config.get("device", "cuda:0"),
        cap_max=int(config["cap_max"]),
        lock_path=Path(config["gsplat_lock"]),
        mcmc_config={"noise_injection_stop_iter": 0},
    )
    backend.color_model = config.get("color_model", "sh")
    backend.sh_degree = int(config.get("sh_degree", 0))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = {
        name: value.to(config.get("device", "cuda:0"))
        for name, value in payload["params"].items()
    }
    if (args.surface_gaussian_count is None) != (
        args.max_surface_scale_m is None
    ):
        raise ValueError(
            "surface Gaussian count and maximum scale must be provided together"
        )
    if args.surface_gaussian_count is not None:
        if not 0 < args.surface_gaussian_count <= len(params["scales"]):
            raise ValueError("surface Gaussian count is outside checkpoint bounds")
        if args.max_surface_scale_m <= 0.0:
            raise ValueError("maximum surface scale must be positive")
        with torch.no_grad():
            params["scales"][: args.surface_gaussian_count].clamp_(
                max=math.log(args.max_surface_scale_m)
            )
    args.output.mkdir(parents=True, exist_ok=True)
    frames = _save_evaluation_artifacts(
        backend=backend,
        params=params,
        dataset=dataset,
        output_dir=args.output,
        background_rgb=config.get("background_color"),
    )
    records = _score(args.output)
    report = {
        "schema_version": 1,
        "kind": "checkpoint_raw_fisheye_validation_v1",
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "completed_steps": int(payload.get("step", -1)),
        "factor": factor,
        "surface_scale_cap_m": args.max_surface_scale_m,
        "frame_count": len(frames),
        "psnr_mean_db": _mean(records, "psnr_db"),
        "psnr_median_db": float(np.median([item["psnr_db"] for item in records])),
        "ssim_mean": _mean(records, "ssim"),
        "depth_mae_mean_m": _mean(records, "depth_mae_m"),
        "depth_rmse_mean_m": _mean(records, "depth_rmse_m"),
        "depth_coverage_mean": _mean(records, "depth_coverage"),
        "alpha_mean": _mean(records, "alpha_mean"),
        "alpha_p05_mean": _mean(records, "alpha_p05"),
        "alpha_below_0_95_fraction_mean": _mean(
            records, "alpha_below_0_95_fraction"
        ),
        "lidar_alpha_mean": _mean(records, "lidar_alpha_mean"),
        "lidar_alpha_p05_mean": _mean(records, "lidar_alpha_p05"),
        "lidar_alpha_below_0_95_fraction_mean": _mean(
            records, "lidar_alpha_below_0_95_fraction"
        ),
        "frames": records,
    }
    report["validation_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    output_path = args.output / "validation_summary.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"validated {len(records)} frames: PSNR={report['psnr_mean_db']:.3f}, "
        f"SSIM={report['ssim_mean']:.4f}, depth={report['depth_mae_mean_m']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
