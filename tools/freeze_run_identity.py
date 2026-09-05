#!/usr/bin/env python3
"""Freeze the identity of a training run or a bare checkpoint.

A quality comparison is only as good as the claim "these two runs differ in
exactly X". That claim needs the resolved config, every input manifest, the
initialisation, the checkpoint and the compiled rasterizer pinned by hash, not
by memory. This tool writes one JSON per run so later arms can be diffed
against it mechanically.

    python tools/freeze_run_identity.py --run RUN_DIR --output identity.json
    python tools/freeze_run_identity.py --checkpoint merged.pt --output id.json

Hashing multi-gigabyte checkpoints takes minutes; that is the point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONFIG_INPUT_KEYS = (
    "dataset_manifest",
    "face_cache_manifest",
    "face_lidar_geometry_manifest",
    "tile_inputs_manifest",
    "initialization_ply",
    "initialization_geometry_manifest",
    "background_image_manifest",
    "depth_manifest",
    "mono_depth_manifest",
    "mask_manifest",
    "person_mask_manifest",
    "renderer_mask_manifest",
    "gsplat_lock",
)

CONFIG_RESOLVED_KEYS = (
    "seed",
    "cap_max",
    "max_steps",
    "controlled_stop_after_steps",
    "sh_degree",
    "color_model",
    "background_color",
    "default_strategy",
    "densification_strategy",
    "densification_gradient_source",
    "geometry_regularization",
    "learning_rates",
    "lidar_alpha_weight",
    "lidar_alpha_target",
    "lidar_range_weight",
    "lidar_range_loss_mode",
    "lidar_normal_alignment",
    "da2_depth_weight",
    "exposure_compensation",
    "pinhole_rasterize_mode",
    "pinhole_with_ut",
    "mipmap_tile_id",
    "mipmap_pipeline_gate",
    "trainer_preset",
    "factor",
)


def sha256_file(path: Path, chunk: int = 1 << 24) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size": size}


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def repo_identity(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "head": _git(["rev-parse", "HEAD"], root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty_files": _git(["status", "--porcelain"], root).splitlines(),
    }


def runtime_identity() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info
    info["torch"] = torch.__version__
    info["cuda_runtime"] = torch.version.cuda
    info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import gsplat
    except ImportError:
        info["gsplat"] = None
        return info
    package = Path(gsplat.__file__).parent
    record: dict[str, Any] = {
        "version": gsplat.__version__,
        "package": str(package),
        "git_head": _git(["rev-parse", "HEAD"], package.parent),
        "git_dirty_files": _git(["status", "--porcelain"], package.parent).splitlines(),
    }
    try:
        from gsplat.cuda import _backend

        record["extension"] = sha256_file(Path(_backend._C.__file__))
    except Exception as error:  # CPU hosts have no compiled extension
        record["extension"] = {"error": repr(error)}
    info["gsplat"] = record
    return info


def checkpoint_identity(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"file": sha256_file(path)}
    try:
        import torch
    except ImportError:
        record["payload"] = "NOT_RUN: torch unavailable"
        return record
    payload = torch.load(path, map_location="cpu", weights_only=False)
    params = payload.get("params") or payload.get("splats") or {}
    shapes = {
        key: list(value.shape)
        for key, value in params.items()
        if hasattr(value, "shape")
    }
    sh_rest = shapes.get("shN", [0, 0, 3])
    strategy_state = payload.get("strategy_state")
    record["payload"] = {
        "schema_version": payload.get("schema_version"),
        "step": payload.get("step"),
        "gaussian_count": shapes.get("means", [0])[0],
        "sh_rest_coeffs": sh_rest[1] if len(sh_rest) > 1 else 0,
        "param_shapes": shapes,
        "identity": payload.get("identity"),
        "has_torch_rng_state": payload.get("torch_rng_state") is not None,
        "has_cuda_rng_state": payload.get("cuda_rng_state") is not None,
        "has_sampler_state": payload.get("sampler_state") is not None,
        "strategy_state_keys": (
            sorted(strategy_state.keys())
            if isinstance(strategy_state, dict)
            else None
        ),
        "sky_layer": payload.get("sky_layer"),
        "imported": payload.get("imported"),
    }
    return record


def run_identity(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_as_run.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} missing; pass --checkpoint for bare checkpoints"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "config_as_run": sha256_file(config_path),
        "resolved": {key: config.get(key) for key in CONFIG_RESOLVED_KEYS},
        "inputs": {},
    }
    for key in CONFIG_INPUT_KEYS:
        value = config.get(key)
        if not value:
            continue
        path = Path(value)
        record["inputs"][key] = (
            sha256_file(path) if path.exists() else {"path": value, "missing": True}
        )
    for name in (
        "surface_initialization_report.json",
        "coordinate_transform_manifest.json",
    ):
        side = run_dir / name
        if side.exists():
            record["inputs"][name] = sha256_file(side)
    latest = run_dir / "checkpoints" / "latest.pt"
    if latest.exists():
        record["checkpoint"] = checkpoint_identity(latest)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, help="training run directory")
    parser.add_argument(
        "--checkpoint", type=Path, help="bare checkpoint (merged, imported)"
    )
    parser.add_argument(
        "--extra-file",
        type=Path,
        action="append",
        default=[],
        help="additional artifact to hash (exported PLY, report)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run is None and args.checkpoint is None:
        parser.error("--run or --checkpoint is required")
    record: dict[str, Any] = {
        "schema_version": 1,
        "repo": repo_identity(ROOT),
        "runtime": runtime_identity(),
    }
    if args.run is not None:
        record["run"] = run_identity(args.run)
    if args.checkpoint is not None:
        record["checkpoint"] = checkpoint_identity(args.checkpoint)
    record["extra_files"] = [sha256_file(path) for path in args.extra_file]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
