#!/usr/bin/env python3
"""Resume-safe serial runner for one snow Tile's A0 -> V27 -> V27c protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.pipeline.mipmap_gate import V27_SNOW_TILE_PROFILES


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_matches(path: Path, *, step: int, count: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        params = payload.get("params") or payload.get("splats")
        return (
            int(payload.get("step", -1)) == step
            and isinstance(params, dict)
            and int(params["means"].shape[0]) == count
        )
    except (EOFError, OSError, RuntimeError, KeyError, TypeError):
        return False


def _run(command: list[str]) -> int:
    return int(subprocess.run(command, cwd=ROOT, check=False).returncode)


def _training_command(extension: Path, config: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools" / "run_with_prebuilt_gsplat.py"),
        "--extension",
        str(extension),
        str(ROOT / "tools" / "train_gsplat.py"),
        "--config",
        str(config),
    ]


def _run_controlled_training(
    *, extension: Path, config: Path, checkpoint: Path, step: int, count: int
) -> None:
    if _checkpoint_matches(checkpoint, step=step, count=count):
        return
    _run(_training_command(extension, config))
    if not _checkpoint_matches(checkpoint, step=step, count=count):
        raise RuntimeError(
            f"controlled training did not produce step={step}, count={count}: {checkpoint}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True, type=int)
    parser.add_argument("--a0-config", required=True, type=Path)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    profile = V27_SNOW_TILE_PROFILES.get(args.tile_id)
    if profile is None:
        raise ValueError("tile-id must be one of 0, 1, 2, 3, 4")
    a0 = _read(args.a0_config)
    if int(a0.get("mipmap_tile_id", -1)) != args.tile_id:
        raise ValueError("A0 config targets another Tile")
    count = profile["gaussian_count"]
    review_stop = profile["review_stop"]
    stabilization_stop = profile["stabilization_stop"]
    a0_output = Path(a0["output_dir"])
    a0_checkpoint = a0_output / "checkpoints" / "latest.pt"
    status_path = args.output_root / "protocol_status.json"

    status = {
        "schema_version": 1,
        "kind": "snow_tile_v28_protocol_status",
        "tile_id": args.tile_id,
        "gaussian_count": count,
        "view_count": profile["view_count"],
        "started_unix": time.time(),
        "stage": "a0",
        "status": "RUNNING",
    }
    _write(status_path, status)
    _run_controlled_training(
        extension=args.extension,
        config=args.a0_config,
        checkpoint=a0_checkpoint,
        step=review_stop,
        count=count,
    )

    boundary_root = args.output_root / "v27a_boundary"
    boundary_output = args.output_root / "training_v27a_boundary602"
    boundary_config = boundary_root / "boundary.config.json"
    boundary_gate = boundary_root / "boundary.gate.json"
    if not boundary_config.is_file() or not boundary_gate.is_file():
        code = _run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_v27a_mcmc_reallocation_gate.py"),
                "--base-config",
                str(args.a0_config),
                "--upstream-gate",
                str(args.upstream_gate),
                "--warm-start-checkpoint",
                str(a0_checkpoint),
                "--output-config",
                str(boundary_config),
                "--output-gate",
                str(boundary_gate),
                "--run-output",
                str(boundary_output),
            ]
        )
        if code:
            raise RuntimeError("failed to build V27 boundary arm")
    status.update({"stage": "v27a_boundary602"})
    _write(status_path, status)
    boundary_checkpoint = boundary_output / "checkpoints" / "latest.pt"
    _run_controlled_training(
        extension=args.extension,
        config=boundary_config,
        checkpoint=boundary_checkpoint,
        step=602,
        count=count,
    )

    review_root = args.output_root / "v27b_review"
    review_output = args.output_root / f"training_v27b_review{review_stop}"
    review_report = review_root / "boundary_report.json"
    review_config = review_root / "review.config.json"
    review_gate = review_root / "review.gate.json"
    if not review_config.is_file() or not review_gate.is_file():
        code = _run(
            [
                sys.executable,
                str(ROOT / "tools" / "promote_v27a_mcmc_review.py"),
                "--boundary-config",
                str(boundary_config),
                "--boundary-checkpoint",
                str(boundary_checkpoint),
                "--boundary-progress",
                str(boundary_output / "monitor" / "progress.jsonl"),
                "--upstream-gate",
                str(args.upstream_gate),
                "--output-report",
                str(review_report),
                "--output-config",
                str(review_config),
                "--output-gate",
                str(review_gate),
                "--run-output",
                str(review_output),
            ]
        )
        if code:
            raise RuntimeError("V27 boundary audit failed")
    status.update({"stage": f"v27b_review{review_stop}"})
    _write(status_path, status)
    review_checkpoint = review_output / "checkpoints" / "latest.pt"
    _run_controlled_training(
        extension=args.extension,
        config=review_config,
        checkpoint=review_checkpoint,
        step=review_stop,
        count=count,
    )

    stabilization_root = args.output_root / "v27c_stabilization"
    stabilization_output = (
        args.output_root / f"training_v27c_stabilization{stabilization_stop}"
    )
    sanitized_checkpoint = (
        stabilization_root / f"step_{review_stop:08d}_scale008_sanitized.pt"
    )
    stabilization_config = stabilization_root / "stabilization.config.json"
    stabilization_gate = stabilization_root / "stabilization.gate.json"
    if not stabilization_config.is_file() or not stabilization_gate.is_file():
        code = _run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_v27c_stabilization.py"),
                "--source-config",
                str(review_config),
                "--source-checkpoint",
                str(review_checkpoint),
                "--upstream-gate",
                str(args.upstream_gate),
                "--output-checkpoint",
                str(sanitized_checkpoint),
                "--output-sanitization-report",
                str(stabilization_root / "scale_tail_sanitization.json"),
                "--output-handoff-report",
                str(stabilization_root / "stabilization_handoff.json"),
                "--output-config",
                str(stabilization_config),
                "--output-gate",
                str(stabilization_gate),
                "--run-output",
                str(stabilization_output),
            ]
        )
        if code:
            raise RuntimeError("failed to build V27c stabilization arm")
    status.update({"stage": f"v27c_stabilization{stabilization_stop}"})
    _write(status_path, status)
    stabilization_checkpoint = stabilization_output / "checkpoints" / "latest.pt"
    _run_controlled_training(
        extension=args.extension,
        config=stabilization_config,
        checkpoint=stabilization_checkpoint,
        step=stabilization_stop,
        count=count,
    )

    ply = args.output_root / "exports" / (
        f"snow_tile{args.tile_id}_v28_step{stabilization_stop}_sh0_full.ply"
    )
    if not ply.is_file():
        code = _run(
            [
                sys.executable,
                str(ROOT / "tools" / "export_gaussian_ply.py"),
                "--checkpoint",
                str(stabilization_checkpoint),
                "--output",
                str(ply),
                "--min-opacity",
                "0",
            ]
        )
        if code:
            raise RuntimeError("failed to export final Tile PLY")
    status.update(
        {
            "stage": "complete",
            "status": "PASS",
            "completed_unix": time.time(),
            "final_checkpoint": stabilization_checkpoint.resolve().as_posix(),
            "final_checkpoint_sha256": _sha256(stabilization_checkpoint),
            "final_ply": ply.resolve().as_posix(),
            "final_ply_sha256": _sha256(ply),
        }
    )
    _write(status_path, status)
    print(f"Tile_{args.tile_id} V28 protocol PASS: {ply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
