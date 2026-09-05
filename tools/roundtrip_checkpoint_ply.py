#!/usr/bin/env python3
"""Checkpoint -> PLY -> checkpoint -> same-backend render, with controls.

Every delivered PLY is judged in a viewer we do not control, so the first
question is whether the file even carries what the checkpoint holds. This
tool exports a checkpoint, imports the PLY back, compares the tensors, then
renders both parameter sets through the identical backend on the same views
and reports the image difference. Negative controls (DC-only SH, a shifted
camera, a different background, a repeated render) show that the difference
metric is sensitive to the things it must catch and how large the raster's
own run-to-run noise is, so "roundtrip is clean" is a measured statement.

    python tools/roundtrip_checkpoint_ply.py --config eval.json \
        --checkpoint merged.pt --output research/.../wp01_roundtrip --frames 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Tolerance for a float32 PLY roundtrip of float32 parameters: the writer and
# reader both keep float32, so anything above a few ulps means a domain error.
TENSOR_TOLERANCE = 1e-6


def _model_sh_degree(params) -> int:
    total = int(params["sh0"].shape[-2] + params["shN"].shape[-2])
    return max(0, int(round(math.sqrt(total))) - 1)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _image_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "frac_pixels_over_1_255": float((diff.max(axis=-1) > 1.0 / 255.0).mean()),
        "psnr": _psnr(a, b),
    }


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _compare_params(original: dict, reimported: dict) -> dict:
    import torch

    report = {}
    for key in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        lhs = original[key].detach().float().cpu()
        rhs = reimported[key].detach().float().cpu()
        entry = {"shape_original": list(lhs.shape), "shape_reimported": list(rhs.shape)}
        if lhs.shape != rhs.shape:
            entry["max_abs"] = None
            entry["pass"] = False
            report[key] = entry
            continue
        if key == "quats":
            # The importer normalises; the rasterizer does too, so compare
            # directions rather than raw magnitudes.
            lhs = lhs / lhs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rhs = rhs / rhs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if lhs.numel() == 0:
            entry["max_abs"] = 0.0
        else:
            entry["max_abs"] = float((lhs - rhs).abs().max())
        entry["pass"] = entry["max_abs"] <= TENSOR_TOLERANCE
        report[key] = entry
    report["all_pass"] = all(value["pass"] for value in report.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--shift-m", type=float, default=0.035,
                        help="camera translation for the pose control")
    parser.add_argument("--background", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()

    import torch
    from PIL import Image

    from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset
    from export_gaussian_ply import export_checkpoint_ply
    from import_gaussian_ply import import_ply
    from sharpness_metrics import _load_backend

    args.output.mkdir(parents=True, exist_ok=True)
    raw = json.loads(args.config.read_text(encoding="utf-8"))

    # Stage 1: tensor roundtrip through the exporter and importer.
    ply_path = args.output / "roundtrip.ply"
    reimported_path = args.output / "reimported.pt"
    export_report = export_checkpoint_ply(args.checkpoint, ply_path, min_opacity=0.0)
    import_report = import_ply(ply_path, reimported_path)
    original_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original = original_payload["params"]
    reimported = torch.load(reimported_path, map_location="cpu", weights_only=False)["params"]
    tensor_report = _compare_params(original, reimported)
    sh_degree = _model_sh_degree(original)

    # Stage 2: render both through one backend on the same views. The backend
    # takes the model's degree, not the config's: delivery_eval*.json said 0
    # and rendered every SH1 checkpoint DC-only until this control caught it.
    backend, torch_mod = _load_backend(raw, sh_degree=sh_degree)
    tile_views = None
    if raw.get("tile_inputs_manifest"):
        tile_inputs = json.loads(Path(raw["tile_inputs_manifest"]).read_text(encoding="utf-8"))
        selected = [
            tile for tile in tile_inputs["tiles"]
            if int(tile["tile_id"]) == int(raw.get("mipmap_tile_id", 0))
        ]
        if len(selected) != 1:
            raise ValueError("Tile inputs do not contain a unique selected Tile")
        tile_views = selected[0]["views"]
    dataset = FaceCacheDataset(
        Path(raw["face_cache_manifest"]),
        Path(raw["face_cache_root"]),
        verify_artifacts=False,
        dataset_manifest_path=Path(raw["dataset_manifest"]),
        renderer_mask_manifest_path=Path(raw["renderer_mask_manifest"]),
        tile_views=tile_views,
    )
    device = raw.get("device", "cuda:0")
    backgrounds = None
    if raw.get("background_image_manifest"):
        from cloudstudio_3dgs.training.view_backgrounds import ViewBackgroundLibrary

        backgrounds = ViewBackgroundLibrary(
            Path(raw["background_image_manifest"]),
            Path(raw["background_image_root"]),
            device=device,
        )

    def to_device(params):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in params.items()}

    ours = to_device(original)
    back = to_device(reimported)

    def render(params, sample, *, background, degree, c2w=None):
        with torch.no_grad():
            rendered, _, _, _ = backend.render(
                params, sample, with_range=False, background_rgb=background,
                active_sh_degree=degree, c2w_override=c2w,
            )
        return rendered.detach().clamp(0.0, 1.0).cpu().numpy()

    stride = max(1, len(dataset) // args.frames)
    picks = list(range(0, len(dataset), stride))[: args.frames]
    frames = []
    controls = None
    for order, index in enumerate(picks):
        sample = dataset[index]
        photo = np.asarray(sample.image, dtype=np.uint8)
        background = tuple(args.background)
        if backgrounds is not None:
            background = backgrounds.background_for(
                sample.image_id, height=photo.shape[0], width=photo.shape[1], torch=torch,
            )
        a = render(ours, sample, background=background, degree=sh_degree)
        b = render(back, sample, background=background, degree=sh_degree)
        stats = _image_stats(a, b)
        stats.update({"index": int(index), "image_id": str(sample.image_id)})
        frames.append(stats)
        amplified = _to_uint8(np.abs(a - b) * 20.0)
        panel = np.concatenate([photo, _to_uint8(a), _to_uint8(b), amplified], axis=1)
        Image.fromarray(panel).save(args.output / f"roundtrip_{order:02d}.png")

        if controls is None:
            # Controls on the first view only; each must move the metric more
            # than the roundtrip does, and the repeat render bounds the noise.
            repeat = render(ours, sample, background=background, degree=sh_degree)
            dc_only = render(back, sample, background=background, degree=0)
            c2w = np.array(sample.c2w, dtype=np.float64).copy()
            c2w[:3, 3] += c2w[:3, 0] * args.shift_m
            shifted = render(ours, sample, background=background, degree=sh_degree, c2w=c2w)
            other_background = (0.0, 0.0, 0.0) if backgrounds is None else tuple(args.background)
            recoloured = render(ours, sample, background=other_background, degree=sh_degree)
            controls = {
                "repeat_render": _image_stats(a, repeat),
                "dc_only_sh": _image_stats(a, dc_only) if sh_degree > 0 else None,
                "camera_shift_m": args.shift_m,
                "camera_shift": _image_stats(a, shifted),
                "background_change": _image_stats(a, recoloured),
            }
            panel = np.concatenate(
                [_to_uint8(a), _to_uint8(dc_only), _to_uint8(shifted), _to_uint8(recoloured)],
                axis=1,
            )
            Image.fromarray(panel).save(args.output / "controls_00.png")

    noise_floor = max(controls["repeat_render"]["max_abs"], 1.0 / 255.0)
    render_pass = all(f["max_abs"] <= noise_floor for f in frames)
    sensitivity = {
        "dc_only_exceeds_roundtrip": (
            controls["dc_only_sh"] is not None
            and controls["dc_only_sh"]["mean_abs"] > max(f["mean_abs"] for f in frames)
        ),
        "shift_exceeds_roundtrip": controls["camera_shift"]["mean_abs"] > max(f["mean_abs"] for f in frames),
        "background_exceeds_roundtrip": controls["background_change"]["mean_abs"] > max(f["mean_abs"] for f in frames),
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "gaussian_count": int(original["means"].shape[0]),
        "sh_degree": sh_degree,
        "export": export_report,
        "import": import_report,
        "tensor_roundtrip": tensor_report,
        "render_roundtrip": {"frames": frames, "noise_floor_max_abs": noise_floor, "pass": render_pass},
        "controls": controls,
        "sensitivity": sensitivity,
        "verdict": {
            "tensor": "PASS" if tensor_report["all_pass"] else "FAIL",
            "render": "PASS" if render_pass else "FAIL",
            "controls": "PASS" if all(sensitivity.values()) else "FAIL",
        },
    }
    temporary = args.output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(temporary, args.output / "report.json")
    print(json.dumps(report["verdict"]))
    print("tensor max_abs:", {k: v.get("max_abs") for k, v in tensor_report.items() if isinstance(v, dict)})
    print("render frames:", [(f["index"], round(f["max_abs"], 5), round(f["psnr"], 2)) for f in frames])
    print("controls:", {k: (round(v["max_abs"], 4), round(v["psnr"], 2)) for k, v in controls.items() if isinstance(v, dict)})
    return 0 if all(value == "PASS" for value in report["verdict"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
