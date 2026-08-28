#!/usr/bin/env python3
"""Export the V27 review checkpoint and immediately relay into uninterrupted polish.

The review arm is deliberately signed to stop at step 2618.  This supervisor
waits for that atomic checkpoint, signs a no-early-stop continuation against
the exact checkpoint hash, starts the continuation, and then exports the full
SH0 PLY while training proceeds in the new process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import advance_adaptive_reallocation_gate
from cloudstudio_3dgs.training.trainer import TrainerConfig


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


def _latest_completed_step(progress_path: Path) -> int:
    if not progress_path.is_file():
        return -1
    latest = -1
    with progress_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                latest = int(json.loads(line).get("completed_steps", latest))
    return latest


def _wait_for_checkpoint(
    *, progress_path: Path, checkpoint_path: Path, target_step: int, poll_seconds: float
) -> dict:
    import torch

    while True:
        # Progress telemetry is intentionally sampled every ten steps.  A
        # controlled boundary such as 2618 can therefore have a complete
        # checkpoint while the last progress row is 2610.  Treat telemetry as
        # a cheap proximity hint and make the checkpoint step authoritative.
        if _latest_completed_step(progress_path) < target_step - 10:
            time.sleep(poll_seconds)
            continue
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError):
            time.sleep(poll_seconds)
            continue
        if int(payload.get("step", -1)) == target_step:
            return payload
        time.sleep(poll_seconds)


def _launch_continuation(
    *, extension: Path, config_path: Path, output_dir: Path
) -> tuple[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "continuation.log"
    command = [
        sys.executable,
        str(ROOT / "tools" / "run_with_prebuilt_gsplat.py"),
        "--extension",
        str(extension),
        str(ROOT / "tools" / "train_gsplat.py"),
        "--config",
        str(config_path),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    log_stream = log_path.open("w", encoding="utf-8", newline="\n")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    finally:
        log_stream.close()
    return int(process.pid), log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-config", required=True, type=Path)
    parser.add_argument("--review-progress", required=True, type=Path)
    parser.add_argument("--review-checkpoint", required=True, type=Path)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--output-handoff-report", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--continuation-output", required=True, type=Path)
    parser.add_argument("--ply-output", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--status-output", required=True, type=Path)
    parser.add_argument("--target-step", type=int, default=2618)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--run-id", default="snow-tile1-v27b-a0-safe-mcmc-continuation7480"
    )
    args = parser.parse_args()

    review = _read(args.review_config)
    checkpoint = _wait_for_checkpoint(
        progress_path=args.review_progress,
        checkpoint_path=args.review_checkpoint,
        target_step=args.target_step,
        poll_seconds=args.poll_seconds,
    )
    identity = checkpoint.get("identity", {})
    source_config_sha = str(identity.get("trainer_config_sha256", ""))
    expected_source_config_sha = hashlib.sha256(
        canonical_json_bytes(TrainerConfig.from_dict(review).contract_dict())
    ).hexdigest()
    if source_config_sha != expected_source_config_sha:
        raise ValueError("step-2618 checkpoint does not belong to the review config")

    checkpoint_params = checkpoint.get("params") or checkpoint.get("splats")
    if not isinstance(checkpoint_params, dict) or "means" not in checkpoint_params:
        raise ValueError("step-2618 checkpoint has no Gaussian parameter dictionary")
    handoff_report = {
        "schema_version": 1,
        "kind": "adaptive_reallocation_continuation_handoff_v1",
        "status": "ADAPTIVE_REALLOCATION_BOUNDARY_PASS",
        "promotion_eligible": True,
        "checkpoint_sha256": _sha256(args.review_checkpoint),
        "source_trainer_config_sha256": source_config_sha,
        "completed_steps": int(checkpoint.get("step", -1)),
        "gaussian_count": int(checkpoint_params["means"].shape[0]),
    }
    handoff_report["boundary_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(handoff_report)
    ).hexdigest()

    continuation = copy.deepcopy(review)
    continuation.update(
        {
            "run_id": str(args.run_id),
            "output_dir": args.continuation_output.resolve().as_posix(),
            "mipmap_pipeline_gate": args.output_gate.resolve().as_posix(),
            "resume_checkpoint": args.review_checkpoint.resolve().as_posix(),
            "checkpoint_keep_every": 2618,
        }
    )
    continuation.pop("controlled_stop_after_steps", None)
    continuation.pop("warm_start_checkpoint", None)
    continuation.pop("config_manifest_sha256", None)
    continuation["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(continuation)
    ).hexdigest()
    gate = advance_adaptive_reallocation_gate(
        _read(args.upstream_gate),
        continuation,
        stage="continuation",
        boundary_report=handoff_report,
    )

    _write(args.output_handoff_report, handoff_report)
    _write(args.output_gate, gate)
    _write(args.output_config, continuation)
    TrainerConfig.from_dict(continuation).validate()

    continuation_pid, continuation_log = _launch_continuation(
        extension=args.extension.resolve(),
        config_path=args.output_config.resolve(),
        output_dir=args.continuation_output.resolve(),
    )

    export_command = [
        sys.executable,
        str(ROOT / "tools" / "export_gaussian_ply.py"),
        "--checkpoint",
        str(args.review_checkpoint),
        "--output",
        str(args.ply_output),
        "--min-opacity",
        "0",
        "--layer",
        "all",
    ]
    exported = subprocess.run(
        export_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    status = {
        "schema_version": 1,
        "kind": "v27_step2618_relay_status_v1",
        "status": "CONTINUATION_STARTED_AND_PLY_EXPORTED",
        "target_step": args.target_step,
        "checkpoint": args.review_checkpoint.resolve().as_posix(),
        "checkpoint_sha256": handoff_report["checkpoint_sha256"],
        "gaussian_count": handoff_report["gaussian_count"],
        "ply": args.ply_output.resolve().as_posix(),
        "ply_bytes": args.ply_output.stat().st_size,
        "continuation_pid": continuation_pid,
        "continuation_config": args.output_config.resolve().as_posix(),
        "continuation_log": continuation_log.resolve().as_posix(),
        "export_stdout": exported.stdout.strip(),
    }
    _write(args.status_output, status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
