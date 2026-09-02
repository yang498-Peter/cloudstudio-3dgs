#!/usr/bin/env python3
"""Held-out view battery for a lifecycle probe checkpoint.

Masked PSNR answers "does it look right", foreground alpha answers "or is it
going transparent to fake it", and LiDAR range MAE answers "is the surface
where the scanner measured it". A probe verdict needs all three together -
population history alone cannot distinguish healthy circulation from decay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pooled_p05(torch, chunks):
    # torch.quantile refuses inputs beyond ~16M elements; 48 full faces of
    # foreground alpha exceed that, so take the 5th percentile by rank.
    pooled = torch.cat(chunks)
    k = max(1, int(0.05 * pooled.numel()))
    return float(torch.kthvalue(pooled, k).values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--reference-ply",
        type=Path,
        help="score a delivery PLY instead of a checkpoint, so the reference "
        "and our runs are read on the identical battery",
    )
    parser.add_argument(
        "--reference-alignment",
        type=Path,
        help="JSON carrying the rigid transform that brings the PLY into our frame",
    )
    parser.add_argument("--views", type=int, default=48)
    parser.add_argument(
        "--tile-views",
        action="store_true",
        help="score on the selected Tile's own cropped training views instead "
        "of the held-out faces; a single-Tile checkpoint rendered on the full "
        "scene shows background wherever a neighbouring Tile owns the pixels, "
        "so the held-out battery cannot compare Tile-level arms to each other "
        "or to a whole-scene delivery",
    )
    parser.add_argument(
        "--tile-owned",
        action="store_true",
        help="with --tile-views: score only pixels the Tile owns (foreign "
        "LiDAR returns and their neighbourhood leave the mask), the same "
        "rule tile_ownership_masking applies in training",
    )
    parser.add_argument(
        "--holdout-from",
        type=Path,
        help="with --tile-views: score only the views listed as held out in "
        "this run directory's holdout_views.json (views the run never trained on)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tile_owned and not args.tile_views:
        parser.error("--tile-owned needs --tile-views")
    if args.holdout_from is not None and not args.tile_views:
        parser.error("--holdout-from needs --tile-views")
    if (args.checkpoint is None) == (args.reference_ply is None):
        parser.error("pass exactly one of --checkpoint or --reference-ply")

    import torch

    from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset
    from cloudstudio_3dgs.training.view_backgrounds import ViewBackgroundLibrary
    from cloudstudio_3dgs.training.trainer import rendered_range_to_euclidean
    from tools.sharpness_metrics import _load_backend

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    backend, torch_mod = _load_backend(raw)
    device = raw.get("device", "cuda:0")

    if args.tile_views:
        if not raw.get("tile_inputs_manifest"):
            parser.error("--tile-views needs a config bound to a Tile inputs manifest")
        tile_inputs = json.loads(
            Path(raw["tile_inputs_manifest"]).read_text(encoding="utf-8")
        )
        selected = [
            tile
            for tile in tile_inputs["tiles"]
            if int(tile["tile_id"]) == int(raw.get("mipmap_tile_id", 0))
        ]
        if len(selected) != 1:
            parser.error("Tile inputs do not contain a unique selected Tile")
        # Training faces, cropped exactly as the trainer crops them; the
        # held-out split has no Tile crops, so a Tile-level score is by
        # construction a training-view score and must be read as such -
        # unless --holdout-from restricts it to views the run withheld.
        tile_view_list = selected[0]["views"]
        if args.holdout_from is not None:
            record = args.holdout_from
            if record.is_dir():
                record = record / "holdout_views.json"
            held = set(json.loads(record.read_text(encoding="utf-8"))["held_out_sample_ids"])
            tile_view_list = [v for v in tile_view_list if str(v["sample_id"]) in held]
            if not tile_view_list:
                parser.error("no Tile views match the hold-out record")
        dataset = FaceCacheDataset(
            Path(raw["face_cache_manifest"]),
            Path(raw["face_cache_root"]),
            verify_artifacts=False,
            dataset_manifest_path=Path(raw["dataset_manifest"]),
            tile_views=tile_view_list,
            renderer_mask_manifest_path=Path(raw["renderer_mask_manifest"]),
            face_lidar_geometry_manifest_path=Path(
                raw["face_lidar_geometry_manifest"]
            ),
            face_lidar_geometry_root=Path(raw["face_lidar_geometry_root"]),
            tile_ownership_box=(
                selected[0]["training_and_export_box"] if args.tile_owned else None
            ),
            tile_ownership_margin_m=float(raw.get("tile_ownership_margin_m", 0.5)),
            tile_ownership_dilation_px=int(raw.get("tile_ownership_dilation_px", 15)),
        )
        backgrounds = None
        if raw.get("background_image_manifest"):
            backgrounds = ViewBackgroundLibrary(
                Path(raw["background_image_manifest"]),
                Path(raw["background_image_root"]),
                device=device,
            )
    else:
        dataset = FaceCacheDataset(
            Path(raw["face_cache_manifest"].replace("face4", "face4_val")),
            Path(raw["face_cache_root"].replace("face4", "face4_val")),
            verify_artifacts=False,
            dataset_manifest_path=Path(raw["dataset_manifest"]),
            renderer_mask_manifest_path=Path(
                raw["renderer_mask_manifest"].replace("_train", "_val")
            ),
            face_lidar_geometry_manifest_path=Path(
                raw["face_lidar_geometry_manifest"].replace("_train", "_val")
            ),
            face_lidar_geometry_root=Path(
                raw["face_lidar_geometry_root"].replace("_train", "_val")
            ),
        )
        backgrounds = None
        if raw.get("background_image_manifest"):
            backgrounds = ViewBackgroundLibrary(
                Path(raw["background_image_manifest"].replace("_train", "_val")),
                Path(raw["background_image_root"].replace("_train", "_val")),
                device=device,
            )

    if args.reference_ply is not None:
        import numpy as _np

        from tools.build_three_way_compare import _load_ply_gaussians

        transform = None
        if args.reference_alignment is not None:
            transform = _np.asarray(
                json.loads(args.reference_alignment.read_text(encoding="utf-8"))[
                    "transform"
                ],
                dtype=_np.float64,
            )
        loaded = _load_ply_gaussians(args.reference_ply, transform)
        params = {
            key: value.to(device) for key, value in loaded.items()
        }
        step = -1
        source = str(args.reference_ply)
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        params = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in payload["params"].items()
        }
        step = int(payload.get("step", 0))
        source = str(args.checkpoint)

    stride = max(1, len(dataset) // args.views)
    picks = list(range(0, len(dataset), stride))[: args.views]

    # Full-mask numbers include sky and, on downward faces, a plain backdrop;
    # the foreground numbers restrict to a 15 px neighbourhood of LiDAR
    # returns so the two models are compared on surfaces both can represent.
    fg_radius = 15
    psnrs, alpha_means, alpha_p05s, range_maes = [], [], [], []
    fg_psnrs, fg_alpha_all = [], []
    per_face: dict[str, list[float]] = {}
    for index in picks:
        sample = dataset[index]
        target = (
            torch.as_tensor(
                np.array(sample.image, copy=True), dtype=torch.float32, device=device
            )
            / 255.0
        )
        mask = torch.as_tensor(np.array(sample.rgb_mask), device=device)
        background = (1.0, 1.0, 1.0)
        if backgrounds is not None:
            background = backgrounds.background_for(
                sample.image_id,
                height=int(target.shape[0]),
                width=int(target.shape[1]),
                torch=torch,
            )
        with torch.no_grad():
            rendered, rendered_range, alpha, info = backend.render(
                params,
                sample,
                with_range=sample.depth_range_m is not None,
                background_rgb=background,
            )
            sq = ((rendered - target) ** 2).mean(dim=-1)
            mse = sq[mask].mean()
            psnr = float(-10.0 * torch.log10(mse.clamp_min(1e-10)))
            psnrs.append(psnr)
            alpha_map = alpha[0] if alpha.ndim == 3 else alpha
            foreground_alpha = alpha_map[mask]
            alpha_means.append(float(foreground_alpha.mean()))
            alpha_p05s.append(float(torch.quantile(foreground_alpha.float(), 0.05)))
            face_kind = sample.image_id.split("::")[-1].rsplit("_", 1)[0]
            per_face.setdefault(face_kind, []).append(psnr)
            if sample.depth_mask is not None:
                support = torch.as_tensor(
                    np.array(sample.depth_mask), device=device
                )
                fg = torch.nn.functional.max_pool2d(
                    support[None, None].float(),
                    kernel_size=2 * fg_radius + 1,
                    stride=1,
                    padding=fg_radius,
                )[0, 0] > 0.0
                fg &= mask
                if bool(fg.any()):
                    fg_mse = sq[fg].mean()
                    fg_psnrs.append(
                        float(-10.0 * torch.log10(fg_mse.clamp_min(1e-10)))
                    )
                    fg_alpha_all.append(alpha_map[fg].float().flatten().cpu())
            if sample.depth_range_m is not None and rendered_range is not None:
                depth_mask = torch.as_tensor(
                    np.array(sample.depth_mask), device=device
                )
                measured = torch.as_tensor(
                    np.array(sample.depth_range_m), dtype=torch.float32, device=device
                )
                euclidean = rendered_range_to_euclidean(
                    torch, rendered_range, sample, info
                )
                if depth_mask.any():
                    range_maes.append(
                        float((euclidean[depth_mask] - measured[depth_mask]).abs().mean())
                    )

    report = {
        "source": source,
        "step": step,
        "gaussian_count": int(len(params["means"])),
        "views": len(picks),
        "view_set": (
            "tile_holdout_crops" if args.holdout_from is not None
            else "tile_training_crops" if args.tile_views
            else "held_out_faces"
        ),
        "pixel_scope": "tile_owned" if args.tile_owned else "full_mask",
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_p10": float(np.percentile(psnrs, 10)),
        "alpha_mean": float(np.mean(alpha_means)),
        "alpha_p05": float(np.mean(alpha_p05s)),
        "lidar_range_mae_m": float(np.mean(range_maes)) if range_maes else None,
        "lidar_views": len(range_maes),
        "foreground_radius_px": fg_radius,
        "foreground_views": len(fg_psnrs),
        "psnr_fg_mean": float(np.mean(fg_psnrs)) if fg_psnrs else None,
        "psnr_fg_p10": float(np.percentile(fg_psnrs, 10)) if fg_psnrs else None,
        "alpha_fg_p05_pooled": (
            _pooled_p05(torch, fg_alpha_all) if fg_alpha_all else None
        ),
        "psnr_by_face": {
            kind: {
                "views": len(values),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
            }
            for kind, values in sorted(per_face.items())
        },
    }
    print(json.dumps(report, indent=1))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
