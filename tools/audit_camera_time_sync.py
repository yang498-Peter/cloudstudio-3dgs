#!/usr/bin/env python3
"""Sweep camera timestamp offsets by rerendering signed validation views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.evaluation.image_metrics import masked_psnr, masked_ssim
from cloudstudio_3dgs.evaluation.time_sync import trajectories_from_manifest
from cloudstudio_3dgs.training.dataset import S1TrainingDataset
from tools.sharpness_metrics import _load_backend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    from cloudstudio_3dgs.training.trainer import _atomic_json as write_json

    write_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-dataset-manifest", required=True, type=Path)
    parser.add_argument("--offset-ms", required=True, type=float, nargs="+")
    parser.add_argument("--factor", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--maximum-rig-frames", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.maximum_rig_frames <= 0:
        parser.error("--maximum-rig-frames must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_manifest = json.loads(args.base_dataset_manifest.read_text(encoding="utf-8"))
    trajectories = trajectories_from_manifest(base_manifest)
    records = {
        str(image["image_id"]): image for image in base_manifest.get("images", [])
    }

    backend, torch = _load_backend(config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = {
        name: value.to(config.get("device", "cuda:0"))
        for name, value in payload["params"].items()
    }
    dataset = S1TrainingDataset(
        dataset_manifest_path=Path(config["dataset_manifest"]),
        recording_root=Path(config["recording_root"]),
        mask_manifest_path=Path(config["mask_manifest"]),
        mask_root=Path(config["mask_root"]),
        split_manifest_path=Path(config["split_manifest"]),
        split="val",
        person_mask_manifest_path=Path(config["person_mask_manifest"]),
        person_mask_root=Path(config["person_mask_root"]),
        factor=args.factor,
        crop=None,
    )
    requested_offsets = sorted(set(float(value) for value in args.offset_ms))
    minimum_offset_ns = int(round(min(requested_offsets) * 1_000_000.0))
    maximum_offset_ns = int(round(max(requested_offsets) * 1_000_000.0))
    indexes = []
    for index in dataset.indices_for_rig_frames(args.maximum_rig_frames + 2):
        sample = dataset[index]
        record = records[sample.image_id]
        trajectory = trajectories[str(record["side"])]
        timestamp_ns = int(record["timestamp_ns"])
        if (
            timestamp_ns + minimum_offset_ns >= int(trajectory.timestamps_ns[0])
            and timestamp_ns + maximum_offset_ns <= int(trajectory.timestamps_ns[-1])
        ):
            indexes.append(index)
    if not indexes:
        raise ValueError("no validation frames support the complete offset sweep")
    background = config.get("background_color")
    results = []
    with torch.no_grad():
        for offset_ms in requested_offsets:
            frames = []
            for index in indexes:
                sample = dataset[index]
                record = records[sample.image_id]
                c2w = trajectories[str(record["side"])].interpolate(
                    int(record["timestamp_ns"]), offset_ms
                )
                rendered, _, _, _ = backend.render(
                    params,
                    sample,
                    with_range=False,
                    c2w_override=c2w,
                    background_rgb=background,
                )
                prediction = rendered.detach().clamp(0.0, 1.0).cpu().numpy()
                reference = np.asarray(sample.image, dtype=np.float32) / 255.0
                frames.append(
                    {
                        "image_id": sample.image_id,
                        "psnr_db": float(
                            masked_psnr(prediction, reference, sample.rgb_mask)
                        ),
                        "ssim": float(
                            masked_ssim(prediction, reference, sample.rgb_mask)
                        ),
                    }
                )
            results.append(
                {
                    "offset_ms": offset_ms,
                    "frame_count": len(frames),
                    "psnr_db_mean": float(np.mean([item["psnr_db"] for item in frames])),
                    "ssim_mean": float(np.mean([item["ssim"] for item in frames])),
                    "frames": frames,
                }
            )

    baseline = next((item for item in results if item["offset_ms"] == 0.0), None)
    if baseline is None:
        raise ValueError("offset sweep must include 0 ms")
    best = max(results, key=lambda item: item["psnr_db_mean"])
    report = {
        "schema_version": "camera-time-sync-render-sweep-1.0",
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "base_dataset_manifest": str(args.base_dataset_manifest.resolve()),
        "base_dataset_manifest_sha256": str(base_manifest.get("manifest_sha256")),
        "factor": args.factor,
        "maximum_rig_frames": args.maximum_rig_frames,
        "results": results,
        "best_offset_ms": best["offset_ms"],
        "best_psnr_improvement_db": float(
            best["psnr_db_mean"] - baseline["psnr_db_mean"]
        ),
        "best_ssim_change": float(best["ssim_mean"] - baseline["ssim_mean"]),
    }
    _atomic_json(args.output, report)
    print(
        f"best_offset_ms={report['best_offset_ms']:.3f} "
        f"psnr_gain_db={report['best_psnr_improvement_db']:.6f} "
        f"ssim_change={report['best_ssim_change']:.6f} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
