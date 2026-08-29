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
from cloudstudio_3dgs.training.backend import (
    GsplatBackend,
    rendered_range_to_euclidean,
)
from cloudstudio_3dgs.training.dataset import S1TrainingDataset
from cloudstudio_3dgs.training.trainer import _save_evaluation_artifacts


class _EvaluationSubset:
    """Read-only deterministic view over selected dataset rows."""

    def __init__(self, dataset: object, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> object:
        return self.dataset[self.indices[index]]


def _stratified_indices(count: int, maximum: int | None) -> list[int]:
    if count <= 0:
        return []
    if maximum is None or maximum >= count:
        return list(range(count))
    if maximum <= 0:
        raise ValueError("maximum evaluation frame count must be positive")
    if maximum == 1:
        return [count // 2]
    # Evenly span the signed Tile view order.  This is stable across model
    # versions and avoids a random probe accidentally over-sampling one camera
    # segment or one Face4 direction.
    return np.linspace(0, count - 1, num=maximum, dtype=np.int64).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(
    output_dir: Path, *, compute_ssim: bool = True
) -> list[dict]:
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
            "ssim": (
                float(masked_ssim(rendered, reference, mask))
                if compute_ssim
                else None
            ),
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


def _render_metrics(
    *,
    backend: GsplatBackend,
    params: dict,
    dataset: object,
    background_rgb: object | None = None,
    compute_ssim: bool = True,
) -> list[dict]:
    """Render validation frames and score them without writing frame artifacts."""
    torch = backend.torch
    records: list[dict] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            has_range = sample.depth_range_m is not None
            render_options = {"with_range": has_range}
            if background_rgb is not None:
                render_options["background_rgb"] = background_rgb
            rendered, rendered_range, rendered_alpha, render_info = backend.render(
                params, sample, **render_options
            )
            rendered_range = rendered_range_to_euclidean(
                backend.torch, rendered_range, sample, render_info
            )

            # Match the artifact-scoring path exactly: RGB is first quantized
            # to an 8-bit PNG, whereas alpha/range retain float precision.
            rendered_u8 = (
                rendered.detach()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            rendered_rgb = rendered_u8.astype(np.float32) / 255.0
            reference_rgb = np.asarray(sample.image, dtype=np.float32) / 255.0
            mask = np.asarray(sample.rgb_mask, dtype=bool)
            alpha = (
                rendered_alpha.detach()
                .clamp(0.0, 1.0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            valid_alpha = alpha[mask & np.isfinite(alpha)]
            record = {
                "image_id": sample.image_id,
                "psnr_db": float(masked_psnr(rendered_rgb, reference_rgb, mask)),
                "ssim": (
                    float(masked_ssim(rendered_rgb, reference_rgb, mask))
                    if compute_ssim
                    else None
                ),
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
            if valid_alpha.size:
                record["alpha_mean"] = float(valid_alpha.mean())
                record["alpha_p05"] = float(np.quantile(valid_alpha, 0.05))
                record["alpha_below_0_95_fraction"] = float(
                    np.mean(valid_alpha < 0.95)
                )

            if has_range:
                assert rendered_range is not None
                assert sample.depth_range_m is not None
                assert sample.depth_mask is not None
                rendered_range_np = (
                    rendered_range.detach().cpu().numpy().astype(np.float32)
                )
                target = np.asarray(sample.depth_range_m, dtype=np.float32)
                supervised = np.asarray(sample.depth_mask, dtype=bool) & mask
                lidar_alpha = alpha[supervised & np.isfinite(alpha)]
                if lidar_alpha.size:
                    record["lidar_alpha_mean"] = float(lidar_alpha.mean())
                    record["lidar_alpha_p05"] = float(
                        np.quantile(lidar_alpha, 0.05)
                    )
                    record["lidar_alpha_below_0_95_fraction"] = float(
                        np.mean(lidar_alpha < 0.95)
                    )
                covered = (
                    supervised
                    & np.isfinite(rendered_range_np)
                    & (rendered_range_np > 0.0)
                )
                record["depth_coverage"] = float(
                    covered.sum() / max(1, int(supervised.sum()))
                )
                if bool(covered.any()):
                    error = np.abs(rendered_range_np[covered] - target[covered])
                    record["depth_mae_m"] = float(error.mean())
                    record["depth_rmse_m"] = float(
                        np.sqrt(np.square(error).mean())
                    )
            records.append(record)
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"metrics-only validation: {index + 1}/{len(dataset)}")
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
    parser.add_argument("--face-manifest", type=Path, default=None)
    parser.add_argument("--face-root", type=Path, default=None)
    parser.add_argument("--renderer-mask-manifest", type=Path, default=None)
    parser.add_argument(
        "--background-rgb",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help=(
            "override checkpoint compositing background for a transparency "
            "challenge render; values must be within [0, 1]"
        ),
    )
    parser.add_argument(
        "--skip-ssim",
        action="store_true",
        help="compute fast PSNR/alpha/depth metrics without Gaussian-window SSIM",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="reuse already-rendered artifacts in the output directory",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="render and score in memory without writing per-frame artifacts",
    )
    parser.add_argument(
        "--use-config-tile-views",
        action="store_true",
        help="evaluate only the selected Tile views/crops bound by the config",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "evaluate a deterministic stratified subset of at most this many "
            "frames; omit only for promotion-grade full validation"
        ),
    )
    args = parser.parse_args()
    if args.score_only and args.metrics_only:
        raise ValueError("score-only and metrics-only are mutually exclusive")
    if args.background_rgb is not None and any(
        value < 0.0 or value > 1.0 for value in args.background_rgb
    ):
        raise ValueError("background RGB values must be within [0, 1]")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    factor = int(config["factor"] if args.factor is None else args.factor)
    crop_value = config.get("crop")
    crop = None if crop_value is None else CropWindow(**crop_value)
    face_options = (
        args.face_manifest,
        args.face_root,
        args.renderer_mask_manifest,
    )
    if any(value is not None for value in face_options):
        if any(value is None for value in face_options):
            raise ValueError(
                "face manifest, root and renderer-mask manifest are required together"
            )
        from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset

        tile_views = None
        if args.use_config_tile_views:
            tile_inputs = json.loads(
                Path(config["tile_inputs_manifest"]).read_text(encoding="utf-8")
            )
            matches = [
                tile
                for tile in tile_inputs["tiles"]
                if int(tile["tile_id"]) == int(config["mipmap_tile_id"])
            ]
            if len(matches) != 1:
                raise ValueError("config does not bind one selected Tile")
            tile_views = matches[0]["views"]
        dataset = FaceCacheDataset(
            face_manifest_path=args.face_manifest,
            cache_root=args.face_root,
            dataset_manifest_path=Path(config["dataset_manifest"]),
            tile_views=tile_views,
            renderer_mask_manifest_path=args.renderer_mask_manifest,
            face_lidar_geometry_manifest_path=(
                Path(config["face_lidar_geometry_manifest"])
                if config.get("face_lidar_geometry_manifest")
                else None
            ),
            face_lidar_geometry_root=(
                Path(config["face_lidar_geometry_root"])
                if config.get("face_lidar_geometry_root")
                else None
            ),
        )
        evaluation_space = (
            "face4_pinhole_tile_views"
            if args.use_config_tile_views
            else "face4_pinhole"
        )
        factor = 1
        crop = None
    else:
        if args.use_config_tile_views:
            raise ValueError("Tile view evaluation requires Face4 inputs")
        dataset = S1TrainingDataset(
            dataset_manifest_path=Path(config["dataset_manifest"]),
            recording_root=Path(config["recording_root"]),
            mask_manifest_path=(
                Path(config["mask_manifest"])
                if args.mask_manifest is None
                else args.mask_manifest
            ),
            mask_root=(
                Path(config["mask_root"])
                if args.mask_root is None
                else args.mask_root
            ),
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
        evaluation_space = "raw_fisheye"
    full_frame_count = len(dataset)
    evaluation_indices = _stratified_indices(full_frame_count, args.max_frames)
    if len(evaluation_indices) != full_frame_count:
        dataset = _EvaluationSubset(dataset, evaluation_indices)
    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (args.surface_gaussian_count is None) != (
        args.max_surface_scale_m is None
    ):
        raise ValueError(
            "surface Gaussian count and maximum scale must be provided together"
        )
    if args.surface_gaussian_count is not None and args.score_only:
        raise ValueError("score-only cannot apply a temporary surface scale cap")
    if args.surface_gaussian_count is not None:
        if not 0 < args.surface_gaussian_count <= len(payload["params"]["scales"]):
            raise ValueError("surface Gaussian count is outside checkpoint bounds")
        if args.max_surface_scale_m <= 0.0:
            raise ValueError("maximum surface scale must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_background = (
        config.get("background_color")
        if args.background_rgb is None
        else [float(value) for value in args.background_rgb]
    )
    frames = None
    records = None
    if not args.score_only:
        backend = GsplatBackend(
            device=config.get("device", "cuda:0"),
            cap_max=int(config["cap_max"]),
            lock_path=Path(config["gsplat_lock"]),
            mcmc_config={"noise_injection_stop_iter": 0},
        )
        backend.color_model = config.get("color_model", "sh")
        backend.sh_degree = int(config.get("sh_degree", 0))
        # Match Trainer exactly. Face4 production runs use the UT/eval3d
        # pinhole path; silently falling back to classic EWA either changes
        # metrics or fails when the locked runtime intentionally contains only
        # the 3DGUT operators.
        backend.pinhole_rasterize_mode = config.get(
            "pinhole_rasterize_mode", "classic"
        )
        backend.pinhole_with_ut = bool(config.get("pinhole_with_ut", False))
        params = {
            name: value.to(config.get("device", "cuda:0"))
            for name, value in payload["params"].items()
        }
        if args.surface_gaussian_count is not None:
            with torch.no_grad():
                params["scales"][: args.surface_gaussian_count].clamp_(
                    max=math.log(args.max_surface_scale_m)
                )
        if args.metrics_only:
            records = _render_metrics(
                backend=backend,
                params=params,
                dataset=dataset,
                background_rgb=evaluation_background,
                compute_ssim=not args.skip_ssim,
            )
        else:
            frames = _save_evaluation_artifacts(
                backend=backend,
                params=params,
                dataset=dataset,
                output_dir=args.output,
                background_rgb=evaluation_background,
            )
    if records is None:
        records = _score(args.output, compute_ssim=not args.skip_ssim)
    report = {
        "schema_version": 1,
        "kind": "checkpoint_raw_fisheye_validation_v1",
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "completed_steps": int(payload.get("step", -1)),
        "evaluation_space": evaluation_space,
        "artifact_policy": (
            "metrics_only_no_per_frame_artifacts"
            if args.metrics_only
            else "saved_per_frame_artifacts"
        ),
        "factor": factor,
        "background_rgb": evaluation_background,
        "background_source": (
            "checkpoint_config"
            if args.background_rgb is None
            else "evaluation_override"
        ),
        "surface_scale_cap_m": args.max_surface_scale_m,
        "frame_count": len(records) if frames is None else len(frames),
        "full_frame_count": full_frame_count,
        "sampling_policy": (
            "full"
            if len(evaluation_indices) == full_frame_count
            else "deterministic_stratified_signed_view_order"
        ),
        "sampling_indices": evaluation_indices,
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
    ssim_label = (
        "SKIPPED" if report["ssim_mean"] is None else f"{report['ssim_mean']:.4f}"
    )
    print(
        f"validated {len(records)} frames: PSNR={report['psnr_mean_db']:.3f}, "
        f"SSIM={ssim_label}, depth={report['depth_mae_mean_m']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
