#!/usr/bin/env python3
"""Exercise the locked CUDA renderer with degree-2 spherical harmonics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.backend import GsplatBackend
from cloudstudio_3dgs.training.dataset import TrainingSample


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _runtime_identity(runtime: dict) -> dict:
    return {
        key: runtime.get(key)
        for key in ("package", "version", "locked_commit", "source_kind", "commit", "clean")
    }


def build_evidence(lock_path: Path) -> dict:
    import torch

    backend = GsplatBackend(
        device="cuda:0",
        cap_max=64,
        lock_path=lock_path,
        mcmc_config={"noise_injection_stop_iter": 0},
        appearance_mode="sh",
        maximum_sh_degree=3,
        sh_rest_lr_scale=0.05,
    )
    xyz = np.asarray(
        [
            [-0.3, -0.3, 2.0],
            [0.3, -0.3, 2.0],
            [-0.3, 0.3, 2.0],
            [0.3, 0.3, 2.0],
        ],
        dtype=np.float32,
    )
    rgb = np.asarray(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
        dtype=np.uint8,
    )
    params, _, _ = backend.initialize(
        xyz,
        rgb,
        init_scale_m=0.1,
        learning_rates={
            name: 1e-4
            for name in ("means", "scales", "quats", "opacities", "colors")
        },
    )
    backend.set_training_step(2_000, interval=1_000)
    sample = TrainingSample(
        image_id="gate2_sh_cuda",
        rig_frame_id="gate2_sh_cuda",
        camera_id="left",
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        rgb_mask=np.ones((128, 128), dtype=bool),
        depth_range_m=None,
        depth_confidence=None,
        depth_mask=None,
        depth_cache_path=None,
        c2w=np.eye(4, dtype=np.float32),
        K=np.asarray(
            [[100.0, 0.0, 63.5], [0.0, 100.0, 63.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        radial_coeffs=np.zeros(4, dtype=np.float32),
        width=128,
        height=128,
    )
    rendered, _, alpha, _ = backend.render(params, sample, with_range=False)
    rendered.square().mean().backward()
    checks = {
        "render_finite": bool(torch.isfinite(rendered).all()),
        "covered_pixels": int((alpha > 1e-5).sum().item()),
        "sh0_gradient_finite": bool(torch.isfinite(params["sh0"].grad).all()),
        "sh0_gradient_max_abs": float(params["sh0"].grad.abs().max().item()),
        "shN_gradient_finite": bool(torch.isfinite(params["shN"].grad).all()),
        "shN_gradient_max_abs": float(params["shN"].grad.abs().max().item()),
    }
    passed = (
        checks["render_finite"]
        and checks["covered_pixels"] > 0
        and checks["sh0_gradient_finite"]
        and checks["sh0_gradient_max_abs"] > 0.0
        and checks["shN_gradient_finite"]
        and checks["shN_gradient_max_abs"] > 0.0
    )
    evidence = {
        "schema_version": 1,
        "evidence_type": "cloudstudio_gate2_sh_cuda_smoke",
        "gate_status": "PASS_RUNTIME_ONLY" if passed else "FAIL",
        "scope": {
            "appearance_mode": "sh",
            "maximum_sh_degree": 3,
            "active_sh_degree": backend.active_sh_degree,
            "point_count": int(len(xyz)),
            "image_size": [128, 128],
            "real_scene_quality": "NOT_RUN",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "runtime": _runtime_identity(backend.runtime),
        "checks": checks,
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(evidence)
    ).hexdigest()
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gsplat-lock",
        type=Path,
        default=ROOT / "upstream" / "cloudstudio_trainer.lock.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"evidence already exists: {args.output}")
    evidence = build_evidence(args.gsplat_lock)
    _atomic_json(args.output, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if evidence["gate_status"] == "PASS_RUNTIME_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
