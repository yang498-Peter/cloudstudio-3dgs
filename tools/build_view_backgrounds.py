#!/usr/bin/env python3
"""Prerender the static far-field layer into per-view background images.

Each training/validation view gets the baked dome rendered once from its own
pose, composited over the configured constant colour, downsampled (the layer
is low-frequency by construction) and stored as PNG. Training then composites
``final = render + (1 - alpha) * background_view`` and the far field never
enters the trainable set - see cloudstudio_3dgs/training/view_backgrounds.py.
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
    parser.add_argument("--config", type=Path, required=True,
                        help="trainer config supplying dataset and renderer lock")
    parser.add_argument("--dome", type=Path, required=True,
                        help="checkpoint-convention .pt of the static layer")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument(
        "--background", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        help="colour behind the layer where even it is transparent",
    )
    parser.add_argument(
        "--limit", type=int,
        help="render only the first N views (equivalence smoke runs)",
    )
    parser.add_argument(
        "--verify-against", type=Path,
        help="directory of previously baked images; every rendered view is "
        "compared pixel-for-pixel and any mismatch is an error",
    )
    parser.add_argument("--save-threads", type=int, default=8)
    args = parser.parse_args()

    import hashlib
    import torch
    from PIL import Image

    from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset
    from cloudstudio_3dgs.training.view_backgrounds import (
        write_view_background_manifest,
    )
    from tools.sharpness_metrics import _load_backend

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    backend, torch_mod = _load_backend(raw)
    device = raw.get("device", "cuda:0")

    suffix = "" if args.split == "train" else "_val"
    dataset = FaceCacheDataset(
        Path(raw["face_cache_manifest"].replace("face4", f"face4{suffix}"))
        if suffix
        else Path(raw["face_cache_manifest"]),
        Path(raw["face_cache_root"].replace("face4", f"face4{suffix}"))
        if suffix
        else Path(raw["face_cache_root"]),
        verify_artifacts=False,
        dataset_manifest_path=Path(raw["dataset_manifest"]),
        renderer_mask_manifest_path=Path(
            raw["renderer_mask_manifest"].replace("_train", f"_{args.split}")
        ),
    )

    payload = torch.load(args.dome, map_location="cpu", weights_only=False)
    params = payload["params"] if "params" in payload else payload
    dome = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in params.items()
    }
    if "shN" not in dome:
        dome["shN"] = torch.zeros(
            (len(dome["means"]), 0, 3), dtype=torch.float32, device=device
        )

    digest = hashlib.sha256()
    with args.dome.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    args.output.mkdir(parents=True, exist_ok=True)
    views: dict[str, dict] = {}

    # The render itself is milliseconds of GPU over 100k gaussians; loading
    # the full training sample (photo + masks + hash checks) was two orders
    # of magnitude more, and PNG encoding is CPU-bound. So: camera-only
    # samples, and encoding/saving on worker threads.
    from concurrent.futures import ThreadPoolExecutor

    def encode_and_save(name: str, image: np.ndarray) -> None:
        picture = Image.fromarray(image)
        if args.downsample > 1:
            picture = picture.resize(
                (picture.width // args.downsample, picture.height // args.downsample),
                Image.BILINEAR,
            )
        picture.save(args.output / name)
        if args.verify_against is not None:
            with Image.open(args.verify_against / name) as previous:
                reference = np.asarray(previous.convert("RGB"), dtype=np.int16)
            fresh = np.asarray(picture.convert("RGB"), dtype=np.int16)
            if reference.shape != fresh.shape:
                raise ValueError(f"verification shape mismatch for {name}")
            worst = int(np.abs(reference - fresh).max())
            if worst > 1:
                raise ValueError(
                    f"verification mismatch for {name}: max |delta| = {worst}"
                )

    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    with ThreadPoolExecutor(max_workers=args.save_threads) as pool:
        pending = []
        for index in range(total):
            sample = dataset.camera_sample(index)
            with torch.no_grad():
                rendered, _, _, _ = backend.render(
                    dome, sample, with_range=False,
                    background_rgb=tuple(args.background),
                )
            image = (
                rendered.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0
            ).astype(np.uint8)
            name = sample.image_id.replace(":", "_") + ".png"
            pending.append(pool.submit(encode_and_save, name, image))
            views[sample.image_id] = {
                "file": name,
                "height": sample.height // max(1, args.downsample),
                "width": sample.width // max(1, args.downsample),
            }
            if index % 200 == 0:
                print(f"  {index + 1}/{total}", flush=True)
        for task in pending:
            task.result()

    if args.limit is not None:
        print(f"limit={args.limit}: smoke run, no manifest written")
        return 0

    manifest_path = args.output / f"view_background_manifest_{args.split}.json"
    signature = write_view_background_manifest(
        manifest_path,
        views=views,
        metadata={
            "split": args.split,
            "dome_source": str(args.dome),
            "dome_sha256": digest.hexdigest(),
            "downsample": args.downsample,
            "background_rgb": list(args.background),
        },
    )
    print(f"{len(views)} views -> {manifest_path} ({signature[:8]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
