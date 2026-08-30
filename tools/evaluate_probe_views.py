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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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

    psnrs, alpha_means, alpha_p05s, range_maes = [], [], [], []
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
            mse = ((rendered - target) ** 2).mean(dim=-1)[mask].mean()
            psnrs.append(float(-10.0 * torch.log10(mse.clamp_min(1e-10))))
            foreground_alpha = alpha[0][mask] if alpha.ndim == 3 else alpha[mask]
            alpha_means.append(float(foreground_alpha.mean()))
            alpha_p05s.append(float(torch.quantile(foreground_alpha.float(), 0.05)))
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
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_p10": float(np.percentile(psnrs, 10)),
        "alpha_mean": float(np.mean(alpha_means)),
        "alpha_p05": float(np.mean(alpha_p05s)),
        "lidar_range_mae_m": float(np.mean(range_maes)) if range_maes else None,
        "lidar_views": len(range_maes),
    }
    print(json.dumps(report, indent=1))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
