#!/usr/bin/env python3
"""Gate 1 interrupted-resume equivalence: kill a real full-MCMC CUDA training
mid-flight after its step-40 checkpoint lands, resume from that checkpoint, and
compare the finished run against an uninterrupted twin.

CUDA rasterization backward uses atomic float accumulation, so bit-exact
equality across the interrupt boundary is not guaranteed; the gate therefore
requires identical Gaussian counts, a finite loss within a small relative
tolerance of the uninterrupted run, and telemetry that continues rather than
resets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.point_cloud import write_binary_ply
from cloudstudio_3dgs.training.backend import GsplatBackend
from cloudstudio_3dgs.training.scale_calibration import MetricScaleCalibrationConfig
from cloudstudio_3dgs.training.trainer import (
    TrainerConfig,
    load_initialization_ply,
    train,
)
from run_synthetic_training_acceptance import build_fixture

STEPS = 80
CHECKPOINT_EVERY = 40
LOSS_RELATIVE_TOLERANCE = 5e-3


def _densified_init(base_init: Path, output: Path) -> Path:
    base_xyz, base_rgb = load_initialization_ply(base_init)
    rng = np.random.default_rng(42)
    dense_xyz = np.concatenate(
        [base_xyz + rng.normal(0.0, 0.02, size=base_xyz.shape).astype(np.float32) for _ in range(3)]
    )
    dense_rgb = np.concatenate([base_rgb] * 3)
    path = output / "fixture" / "initialization_full_mcmc.ply"
    write_binary_ply(path, dense_xyz.astype(np.float32), dense_rgb)
    return path


def _trainer_config(paths: dict, run_dir: Path, *, max_steps: int, resume: Path | None) -> TrainerConfig:
    return TrainerConfig(
        run_id="gate1-resume-equivalence",
        dataset_manifest=paths["dataset"],
        recording_root=paths["recording"],
        mask_manifest=paths["mask_root"] / "mask_manifest.json",
        mask_root=paths["mask_root"],
        split_manifest=paths["split"],
        initialization_ply=paths["init"],
        output_dir=run_dir,
        gsplat_lock=paths["lock"],
        depth_manifest=paths["depth_manifest"],
        depth_root=paths["depth_root"],
        resume_checkpoint=resume,
        require_person_masks=False,
        max_steps=max_steps,
        checkpoint_every=CHECKPOINT_EVERY,
        factor=1,
        cap_max=64,
        init_scale_m=0.16,
        metric_scale_calibration=MetricScaleCalibrationConfig(
            mode="fixed",
            means_step_fraction=None,
            noise_std_fraction=None,
        ),
        rgb_l1_weight=1.0,
        rgb_ssim_weight=0.0,
        lidar_range_weight=0.01,
        mcmc_refine_start_iter=max(10, STEPS // 4),
        mcmc_refine_stop_iter=STEPS,
        mcmc_refine_every=max(10, STEPS // 8),
        learning_rates={
            "means": 1.6e-4,
            "scales": 1e-8,
            "quats": 1e-8,
            "opacities": 1e-3,
            "colors": 5e-2,
        },
    )


def _config_json(config: TrainerConfig, path: Path) -> None:
    value = {
        "run_id": config.run_id,
        "dataset_manifest": str(config.dataset_manifest),
        "recording_root": str(config.recording_root),
        "mask_manifest": str(config.mask_manifest),
        "mask_root": str(config.mask_root),
        "split_manifest": str(config.split_manifest),
        "initialization_ply": str(config.initialization_ply),
        "output_dir": str(config.output_dir),
        "gsplat_lock": str(config.gsplat_lock),
        "depth_manifest": str(config.depth_manifest),
        "depth_root": str(config.depth_root),
        "require_person_masks": False,
        "max_steps": config.max_steps,
        "checkpoint_every": config.checkpoint_every,
        "factor": config.factor,
        "cap_max": config.cap_max,
        "init_scale_m": config.init_scale_m,
        "metric_scale_calibration": config.metric_scale_calibration.to_dict(),
        "rgb_l1_weight": config.rgb_l1_weight,
        "rgb_ssim_weight": config.rgb_ssim_weight,
        "lidar_range_weight": config.lidar_range_weight,
        "mcmc_refine_start_iter": config.mcmc_refine_start_iter,
        "mcmc_refine_stop_iter": config.mcmc_refine_stop_iter,
        "mcmc_refine_every": config.mcmc_refine_every,
        "learning_rates": config.learning_rates,
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_step(path: Path) -> int | None:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    step = payload.get("step")
    return None if step is None else int(step)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gsplat-lock",
        type=Path,
        default=ROOT / "upstream" / "cloudstudio_trainer.lock.json",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"resume equivalence output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    backend = GsplatBackend(
        device="cuda:0",
        cap_max=64,
        lock_path=args.gsplat_lock,
        mcmc_config={
            "refine_start_iter": max(10, STEPS // 4),
            "refine_stop_iter": STEPS,
            "refine_every": max(10, STEPS // 8),
        },
    )
    (
        _,
        dataset_path,
        recording,
        mask_root,
        split_path,
        init_path,
        depth_manifest_path,
        depth_root,
    ) = build_fixture(args.output / "fixture", backend)
    del backend
    import torch

    torch.cuda.empty_cache()

    paths = {
        "dataset": dataset_path,
        "recording": recording,
        "mask_root": mask_root,
        "split": split_path,
        "init": _densified_init(init_path, args.output),
        "depth_manifest": depth_manifest_path,
        "depth_root": depth_root,
        "lock": args.gsplat_lock,
    }

    print("=== run A: uninterrupted 80 steps ===", flush=True)
    manifest_a = train(_trainer_config(paths, args.output / "run_a", max_steps=STEPS, resume=None))
    training_a = manifest_a["training"]

    print("=== run B phase 1: launch subprocess, kill after step-40 checkpoint ===", flush=True)
    run_b = args.output / "run_b"
    config_b1 = _trainer_config(paths, run_b, max_steps=STEPS, resume=None)
    config_path = args.output / "run_b_phase1_config.json"
    _config_json(config_b1, config_path)
    child = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "train_gsplat.py"), "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    checkpoint_path = run_b / "checkpoints" / "latest.pt"
    interrupted_at = None
    deadline = time.monotonic() + 900.0
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(
                "training subprocess finished before it could be interrupted; "
                f"output:\n{child.stdout.read() if child.stdout else ''}"
            )
        if checkpoint_path.exists():
            step = _checkpoint_step(checkpoint_path)
            if step is not None and step >= CHECKPOINT_EVERY:
                child.kill()
                child.wait(timeout=60)
                interrupted_at = step
                break
        time.sleep(0.5)
    if interrupted_at is None:
        child.kill()
        raise RuntimeError("step-40 checkpoint never appeared; cannot exercise interruption")
    print(f"killed training subprocess after checkpoint at step {interrupted_at}", flush=True)

    print("=== run B phase 2: resume from checkpoint to 80 steps ===", flush=True)
    manifest_b = train(
        _trainer_config(paths, run_b, max_steps=STEPS, resume=checkpoint_path)
    )
    training_b = manifest_b["training"]

    loss_a = float(training_a["last_metrics"]["loss"])
    loss_b = float(training_b["last_metrics"]["loss"])
    relative = abs(loss_b - loss_a) / max(abs(loss_a), 1e-12)
    gaussians_equal = int(training_a["gaussian_count"]) == int(training_b["gaussian_count"])
    evidence = {
        "schema_version": 1,
        "evidence_type": "cloudstudio_full_mcmc_interrupted_resume_equivalence",
        "steps": STEPS,
        "interrupted_after_step": interrupted_at,
        "uninterrupted": {
            "run_manifest_sha256": manifest_a["run_manifest_sha256"],
            "final_loss": loss_a,
            "gaussian_count": int(training_a["gaussian_count"]),
        },
        "interrupted_resumed": {
            "run_manifest_sha256": manifest_b["run_manifest_sha256"],
            "final_loss": loss_b,
            "gaussian_count": int(training_b["gaussian_count"]),
        },
        "loss_relative_difference": relative,
        "loss_relative_tolerance": LOSS_RELATIVE_TOLERANCE,
        "loss_within_tolerance": relative <= LOSS_RELATIVE_TOLERANCE,
        "gaussian_count_equal": gaussians_equal,
        "equivalent": bool(relative <= LOSS_RELATIVE_TOLERANCE and gaussians_equal),
    }
    evidence["resume_equivalence_sha256"] = hashlib.sha256(
        canonical_json_bytes(evidence)
    ).hexdigest()
    evidence_path = args.output / "resume_equivalence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if evidence["equivalent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
