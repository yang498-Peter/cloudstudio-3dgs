#!/usr/bin/env python3
"""Render raw fisheye validation frames for a deferred face-training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.image_sample import CropWindow
from cloudstudio_3dgs.evaluation.quality_report import (
    finalize_deferred_run_manifest,
    verify_run_manifest,
)
from cloudstudio_3dgs.training.backend import GsplatBackend
from cloudstudio_3dgs.training.dataset import S1TrainingDataset
from cloudstudio_3dgs.training.trainer import (
    _atomic_json,
    _save_evaluation_artifacts,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    source_sha256 = verify_run_manifest(source)
    if source.get("final_evaluation_artifacts", {}).get("status") != "DEFERRED":
        raise ValueError("run manifest is not awaiting deferred evaluation")

    run_root = args.run_manifest.parent
    model_path = run_root / Path(source["training"]["model_path"])
    checkpoint_sha256 = _sha256_file(model_path)
    if checkpoint_sha256 != source["training"]["model_sha256"]:
        raise ValueError("selected checkpoint SHA256 does not match run manifest")

    import torch

    backend = GsplatBackend(
        device=config.get("device", "cuda:0"),
        cap_max=int(config["cap_max"]),
        lock_path=Path(config["gsplat_lock"]),
        mcmc_config={"noise_injection_stop_iter": 0},
    )
    backend.color_model = config.get("color_model", "sh")
    backend.sh_degree = int(config.get("sh_degree", 3))
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    device = config.get("device", "cuda:0")
    params = {name: value.to(device) for name, value in payload["params"].items()}

    crop_value = config.get("crop")
    crop = None if crop_value is None else CropWindow(**crop_value)
    dataset = S1TrainingDataset(
        dataset_manifest_path=Path(config["dataset_manifest"]),
        recording_root=Path(config["recording_root"]),
        mask_manifest_path=Path(config["mask_manifest"]),
        mask_root=Path(config["mask_root"]),
        split_manifest_path=Path(config["split_manifest"]),
        split="val",
        person_mask_manifest_path=Path(config["person_mask_manifest"]),
        person_mask_root=Path(config["person_mask_root"]),
        depth_manifest_path=Path(config["depth_manifest"]),
        depth_root=Path(config["depth_root"]),
        factor=int(config["factor"]),
        crop=crop,
    )
    frames = _save_evaluation_artifacts(
        backend=backend,
        params=params,
        dataset=dataset,
        output_dir=run_root,
        background_rgb=config.get("background_color"),
    )
    evaluator_runtime = {
        **backend.runtime,
        "camera_model": "fisheye",
        "projection": "3DGUT",
        "frame_count": len(frames),
    }
    final = finalize_deferred_run_manifest(
        source,
        frames=frames,
        evaluation_runtime=evaluator_runtime,
        checkpoint_sha256=checkpoint_sha256,
    )
    backup = run_root / "face_stage_run_manifest.json"
    if backup.exists():
        existing = json.loads(backup.read_text(encoding="utf-8"))
        if verify_run_manifest(existing) != source_sha256:
            raise ValueError("existing face-stage manifest backup does not match source")
    else:
        _atomic_json(backup, source)
    _atomic_json(args.run_manifest, final)
    print(
        f"finalized {len(frames)} raw fisheye frames: "
        f"source={source_sha256}, final={final['run_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
