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


def _select_offset(
    results: list[dict], baseline: dict, alpha: float
) -> tuple[dict, dict]:
    """Adopt a non-zero offset only when it beats zero by a real margin.

    Every offset renders the same rig frames, so the comparison against zero is
    paired per frame. Taking the argmax of the mean alone adopts whichever
    offset noise happens to favour, which on a flat sweep is a coin toss; a
    paired test asks the question the audit actually cares about, namely whether
    the timestamps are systematically wrong.
    """
    import numpy as np
    from scipy import stats

    base_by_image = {item["image_id"]: item["psnr_db"] for item in baseline["frames"]}
    candidates = []
    for item in results:
        if item["offset_ms"] == 0.0:
            continue
        paired = [
            (frame["psnr_db"], base_by_image[frame["image_id"]])
            for frame in item["frames"]
            if frame["image_id"] in base_by_image
        ]
        if len(paired) < 3:
            continue
        shifted = np.array([value for value, _ in paired], dtype=np.float64)
        zero = np.array([value for _, value in paired], dtype=np.float64)
        delta = shifted - zero
        if float(delta.std()) == 0.0:
            # A constant shift has no variance for the t statistic to divide
            # by. Zero spread around a non-zero mean is the most consistent
            # evidence there is, not the least, so decide it by its sign.
            p_value = 1.0 if float(delta.mean()) == 0.0 else 0.0
        else:
            p_value = float(stats.ttest_rel(shifted, zero).pvalue)
        candidates.append(
            {
                "offset_ms": item["offset_ms"],
                "paired_frames": len(paired),
                "mean_psnr_delta_db": float(delta.mean()),
                "p_value": p_value,
                "significant": bool(p_value < alpha and delta.mean() > 0.0),
            }
        )

    leader = max(candidates, key=lambda c: c["mean_psnr_delta_db"], default=None)
    adopted = baseline
    reason = "no non-zero offset was measured"
    if leader is not None:
        if leader["significant"]:
            adopted = next(
                item for item in results if item["offset_ms"] == leader["offset_ms"]
            )
            reason = (
                f"offset {leader['offset_ms']} ms improves PSNR by "
                f"{leader['mean_psnr_delta_db']:.6f} dB with p={leader['p_value']:.4f}"
            )
        else:
            reason = (
                f"best candidate {leader['offset_ms']} ms gains only "
                f"{leader['mean_psnr_delta_db']:.6f} dB at p={leader['p_value']:.4f}, "
                f"which does not clear alpha={alpha}; keeping 0 ms"
            )
    selection = {
        "rule": "paired_t_test_against_zero",
        "significance_alpha": alpha,
        "candidates": candidates,
        "reason": reason,
    }
    return adopted, selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-dataset-manifest", required=True, type=Path)
    parser.add_argument("--offset-ms", required=True, type=float, nargs="+")
    parser.add_argument("--factor", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--maximum-rig-frames", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=0.05,
        help="a non-zero offset is adopted only below this paired-test p-value",
    )
    parser.add_argument(
        "--reanalyze",
        type=Path,
        help=(
            "recompute the selection from an existing report's per-frame "
            "measurements instead of re-rendering"
        ),
    )
    args = parser.parse_args()
    if args.reanalyze is not None:
        source = json.loads(args.reanalyze.read_text(encoding="utf-8"))
        results = source["results"]
        baseline = next(
            (item for item in results if item["offset_ms"] == 0.0), None
        )
        if baseline is None:
            raise ValueError("source report has no 0 ms offset")
        best, selection = _select_offset(results, baseline, args.significance_alpha)
        report = dict(source)
        report["best_offset_ms"] = best["offset_ms"]
        report["best_psnr_improvement_db"] = float(
            best["psnr_db_mean"] - baseline["psnr_db_mean"]
        )
        report["best_ssim_change"] = float(
            best["ssim_mean"] - baseline["ssim_mean"]
        )
        report["selection"] = selection
        report["reanalyzed_from"] = str(args.reanalyze.resolve())
        _atomic_json(args.output, report)
        print(
            f"best_offset_ms={report['best_offset_ms']:.3f} "
            f"({selection['reason']}) -> {args.output}"
        )
        return 0
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
    best, selection = _select_offset(results, baseline, args.significance_alpha)
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
        "selection": selection,
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
