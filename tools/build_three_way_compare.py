#!/usr/bin/env python3
"""Photo, our render and the reference render at the same camera poses.

Loss curves say whether training is descending; they do not say whether the
surface is there. This puts the ground-truth photo beside our latest
checkpoint and beside the reference delivery rendered from the identical pose,
so the three can be read against each other directly.

The reference PLY lives in its own frame, so it is brought into ours with the
signed rigid transform from the point-cloud alignment; the alignment's own
residual is printed so a viewer knows how much of any offset is registration
rather than reconstruction.
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


def _load_ply_gaussians(path: Path, transform: np.ndarray | None):
    """Read a 3DGS PLY into the tensors the rasterizer wants."""
    import torch

    with path.open("rb") as handle:
        header = b""
        while b"end_header\n" not in header:
            chunk = handle.read(1)
            if not chunk:
                raise ValueError(f"truncated PLY header: {path}")
            header += chunk
        text = header.decode("ascii", errors="replace")
        count = next(
            int(line.split()[-1])
            for line in text.splitlines()
            if line.startswith("element vertex")
        )
        names = [
            line.split()[-1]
            for line in text.splitlines()
            if line.startswith("property")
        ]
        data = np.fromfile(handle, dtype=np.float32, count=count * len(names))
    data = data.reshape(count, len(names))
    index = {name: i for i, name in enumerate(names)}

    means = data[:, [index["x"], index["y"], index["z"]]].astype(np.float64)
    if transform is not None:
        means = means @ transform[:3, :3].T + transform[:3, 3]
    scales = data[:, [index[f"scale_{i}"] for i in range(3)]]
    quats = data[:, [index[f"rot_{i}"] for i in range(4)]]
    opacities = data[:, index["opacity"]]
    sh0 = data[:, [index[f"f_dc_{i}"] for i in range(3)]]

    rest = [name for name in names if name.startswith("f_rest_")]
    shN = (
        torch.from_numpy(
            data[:, [index[name] for name in rest]]
            .reshape(count, len(rest) // 3, 3)
            .copy()
        )
        if rest
        else torch.zeros((count, 0, 3), dtype=torch.float32)
    )
    return {
        "means": torch.from_numpy(means.astype(np.float32)),
        "scales": torch.from_numpy(scales.copy()),
        "quats": torch.from_numpy(quats.copy()),
        "opacities": torch.from_numpy(opacities.copy()),
        "sh0": torch.from_numpy(sh0.copy()).unsqueeze(1),
        "shN": shN,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-ply", type=Path)
    parser.add_argument(
        "--reference-alignment",
        type=Path,
        help="JSON carrying the rigid transform that brings the reference into our frame",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--background", type=float, nargs=3, default=(1.0, 1.0, 1.0)
    )
    args = parser.parse_args()

    import torch
    from PIL import Image

    from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset
    from cloudstudio_3dgs.training.trainer import TrainerConfig
    from tools.sharpness_metrics import _load_backend

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    backend, torch_mod = _load_backend(raw)

    dataset = FaceCacheDataset(
        Path(raw["face_cache_manifest"]),
        Path(raw["face_cache_root"]),
        verify_artifacts=False,
        dataset_manifest_path=Path(raw["dataset_manifest"]),
        renderer_mask_manifest_path=Path(raw["renderer_mask_manifest"]),
    )

    device = raw.get("device", "cuda:0")

    backgrounds = None
    if raw.get("background_image_manifest"):
        # Render both models over the same baked backdrop the training sees;
        # a white void where the training composites sky would make every
        # strip lie about the current state.
        from cloudstudio_3dgs.training.view_backgrounds import (
            ViewBackgroundLibrary,
        )

        backgrounds = ViewBackgroundLibrary(
            Path(raw["background_image_manifest"]),
            Path(raw["background_image_root"]),
            device=device,
        )

    def to_device(params):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in params.items()
        }

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ours = to_device(checkpoint["params"])
    step = int(checkpoint.get("step", 0))

    reference = None
    residual_m = None
    if args.reference_ply is not None:
        transform = None
        if args.reference_alignment is not None:
            payload = json.loads(
                args.reference_alignment.read_text(encoding="utf-8")
            )
            transform = np.asarray(payload["transform"], dtype=np.float64)
            residual_m = payload.get("median_nn_distance_m")
        reference = to_device(_load_ply_gaussians(args.reference_ply, transform))

    stride = max(1, len(dataset) // args.frames)
    picks = list(range(0, len(dataset), stride))[: args.frames]
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for order, index in enumerate(picks):
        sample = dataset[index]
        photo = np.asarray(sample.image, dtype=np.uint8)
        panels = [("photo", photo)]

        view_background = tuple(args.background)
        if backgrounds is not None:
            view_background = backgrounds.background_for(
                sample.image_id,
                height=photo.shape[0],
                width=photo.shape[1],
                torch=torch,
            )
        for label, params in (("ours", ours), ("reference", reference)):
            if params is None:
                continue
            with torch.no_grad():
                rendered, _, _, _ = backend.render(
                    params, sample, with_range=False,
                    background_rgb=view_background,
                )
            image = (
                rendered.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0
            ).astype(np.uint8)
            panels.append((label, image))

        heights = [p.shape[0] for _, p in panels]
        widths = [p.shape[1] for _, p in panels]
        strip = np.full(
            (max(heights), sum(widths) + 8 * (len(panels) - 1), 3), 24, np.uint8
        )
        x = 0
        for _, panel in panels:
            strip[: panel.shape[0], x : x + panel.shape[1]] = panel
            x += panel.shape[1] + 8
        name = f"compare_{order:02d}_{sample.image_id[:18]}.png"
        Image.fromarray(strip).save(args.output / name)
        rows.append({"file": name, "image_id": sample.image_id,
                     "panels": [label for label, _ in panels]})
        print(f"  {name}  ({' | '.join(label for label, _ in panels)})")

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "run_id": raw.get("run_id"),
        "checkpoint_run": args.checkpoint.resolve().parent.parent.name,
        "step": step,
        "gaussian_count": int(len(ours["means"])),
        "reference_ply": str(args.reference_ply) if args.reference_ply else None,
        "reference_alignment_median_nn_m": residual_m,
        "background_rgb": list(args.background),
        "frames": rows,
    }
    (args.output / "compare_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(f"\nstep {step:,}, {len(ours['means']):,} gaussians -> {args.output}")
    if residual_m is not None:
        print(f"reference alignment residual (median nn): {residual_m*100:.1f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
