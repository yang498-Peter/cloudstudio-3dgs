"""Where a work root stands: stages, steps, the training in flight and what is left.

A delivery is ~40 hours of unattended work. ``python -m cloudstudio3dgs_sdk status --work W``
answers "how far is it, is anything still running, and what would a re-run do" from the files
the run already writes - stage sidecars, the pipeline's arm job states, the trainer's
``monitor/progress.jsonl``, the GPU lease and the logs - without starting anything. It judges a
step done exactly as the stage driver does (:meth:`Project.step_done`), so what it calls pending
is what the next ``run`` would execute.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cloudstudio3dgs_sdk.plan import STAGES, Plan, PlannedStep

#: How much of a trainer's progress.jsonl to read for its rate: ~1.5 KB per record, one record
#: per 10 steps, so ~1700 steps of history - enough for a steady rate, cheap to read.
PROGRESS_TAIL_BYTES = 256 * 1024


@dataclass
class ArmProgress:
    """One pipeline arm's training, read from its job state and its trainer's progress file."""

    arm: str
    step: str
    state: str
    alive: bool
    completed_steps: int | None
    target_steps: int | None
    steps_per_second: float | None
    eta_seconds: float | None
    gaussian_count: int | None
    last_progress_age_seconds: float | None
    resumed_from: dict[str, Any] | None
    checkpoint_step: int | None = None

    @property
    def fraction(self) -> float | None:
        if self.completed_steps is None or not self.target_steps:
            return None
        return min(1.0, self.completed_steps / self.target_steps)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def progress_tail(path: Path, *, max_bytes: int = PROGRESS_TAIL_BYTES) -> list[dict[str, Any]]:
    """The trainer's most recent progress records (complete lines only)."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            chunk = handle.read()
    except OSError:
        return []
    lines = chunk.split(b"\n")
    if len(chunk) == max_bytes and size > max_bytes:
        lines = lines[1:]  # the first line was cut by the seek
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get("completed_steps"), int):
            records.append(record)
    return records


def _rate(records: list[dict[str, Any]], *, since: float | None) -> float | None:
    """Steps per second over the records of the current attempt."""
    window = [
        record
        for record in records
        if isinstance(record.get("timestamp_unix"), (int, float))
        and (since is None or float(record["timestamp_unix"]) >= since)
    ]
    if len(window) < 2:
        return None
    first, last = window[0], window[-1]
    seconds = float(last["timestamp_unix"]) - float(first["timestamp_unix"])
    steps = int(last["completed_steps"]) - int(first["completed_steps"])
    if seconds <= 0 or steps <= 0:
        return None
    return steps / seconds


def arm_progress(project: Any, step: PlannedStep, *, now: float) -> ArmProgress:
    from tools.pipeline import JOB_STATE_NAME, _pid_alive, declared_target_steps

    arm = step.pipeline_arm
    layout = project.layout
    job = _read_json(layout.runs / f"{arm}.pipeline" / JOB_STATE_NAME) or {}
    state = str(job.get("state") or "NOT_STARTED")
    pid = job.get("pid")
    alive = state == "RUNNING" and isinstance(pid, int) and _pid_alive(pid)
    target = None
    if step.config is not None and isinstance(step.config.get("controlled_stop_after_steps"), int):
        target = int(step.config["controlled_stop_after_steps"])
    else:
        # The coarse prior's step carries no config; its arm config is written at prepare.
        target = declared_target_steps(Path(step.config_path) if step.config_path else layout.arm_config(arm))
    records = progress_tail(layout.arm_dir(arm) / "monitor" / "progress.jsonl")
    started = job.get("started_at")
    since = float(started) if isinstance(started, (int, float)) else None
    last = records[-1] if records else None
    completed = int(last["completed_steps"]) if last else None
    rate = _rate(records, since=since) if alive else None
    eta = None
    if rate and completed is not None and target:
        eta = max(0.0, (target - completed) / rate)
    age = None
    if last and isinstance(last.get("timestamp_unix"), (int, float)):
        age = max(0.0, now - float(last["timestamp_unix"]))
    training = job.get("training") or {}
    checkpoint_step = (training.get("checkpoint") or {}).get("step") if isinstance(training, dict) else None
    return ArmProgress(
        arm=arm,
        step=step.name,
        state=state,
        alive=alive,
        completed_steps=completed,
        target_steps=target,
        steps_per_second=rate,
        eta_seconds=eta,
        gaussian_count=int(last["gaussian_count"]) if last and isinstance(last.get("gaussian_count"), int) else None,
        last_progress_age_seconds=age,
        resumed_from=job.get("resumed_from") if isinstance(job.get("resumed_from"), dict) else None,
        checkpoint_step=checkpoint_step if isinstance(checkpoint_step, int) else None,
    )


def newest_log(root: Path, *, now: float) -> dict[str, Any] | None:
    """The most recently written log under <work>/logs or <work>/runs, with its last line."""
    candidates: list[Path] = []
    for folder, pattern in ((root / "logs", "*.log"), (root / "runs", "*.log"), (root / "runs", "*.log.err")):
        if folder.is_dir():
            candidates.extend(path for path in folder.glob(pattern) if path.is_file())
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    last_line = ""
    try:
        with newest.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 4096))
            tail = handle.read().decode("utf-8", errors="replace").splitlines()
        last_line = next((line.strip() for line in reversed(tail) if line.strip()), "")
    except OSError:
        pass
    return {
        "path": str(newest),
        "age_seconds": max(0.0, now - newest.stat().st_mtime),
        "last_line": last_line[-240:],
    }


def gpu_lease(root: Path) -> dict[str, Any] | None:
    from tools.pipeline import _pid_alive

    lease = _read_json(root / "runs" / "gpu.lock")
    if not lease:
        return None
    pid = lease.get("pid")
    return {
        "owner": lease.get("owner", ""),
        "pid": pid,
        "alive": isinstance(pid, int) and _pid_alive(pid),
        "since": lease.get("started_at_text", ""),
    }


def _free_bytes(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return None


def collect_status(project: Any, *, now: float | None = None, invocation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything ``render_status`` prints, as plain data (also the ``--json`` output)."""
    from cloudstudio3dgs_sdk.discover import DiscoveryError
    from cloudstudio3dgs_sdk.ingest.errors import IngestError
    from cloudstudio3dgs_sdk.project import StageRefused

    now = time.time() if now is None else now
    root = project.layout.root
    status: dict[str, Any] = {
        "work_root": str(root),
        "profile": project.profile.name,
        "invocation": invocation,
        "stages": [],
        "plan_available": False,
        "plan_note": "",
        "arms": [],
        "remaining_seconds": None,
        "remaining_disk_bytes": None,
        "free_disk_bytes": _free_bytes(root),
        "gpu_lease": gpu_lease(root),
        "newest_log": newest_log(root, now=now),
    }
    plan: Plan | None = None
    try:
        plan = project.plan()
    except (StageRefused, DiscoveryError, IngestError, OSError, ValueError, KeyError) as error:
        status["plan_note"] = (
            f"no plan yet: {error}".splitlines()[0]
            if project.layout.prepare_manifest.is_file()
            else "no plan until prepare has ingested the capture (the plan reads the measured tiles)"
        )
    adopted = project.adopted_paths() if plan is not None else set()
    remaining = 0.0
    for stage in STAGES:
        state = project.stage_state(stage)
        entry: dict[str, Any] = {
            "stage": stage,
            "state": state.state or "NOT_STARTED",
            "reason": state.reason,
            "updated_at": state.get("updated_at", ""),
        }
        if plan is not None:
            steps = plan.stage_steps(stage)
            pending: list[str] = []
            stage_seconds = 0.0
            for step in steps:
                done, why = project.step_done(step, adopted=adopted)
                if not done and step.refresh and step.outputs and all(Path(o).exists() for o in step.outputs):
                    # A refresh step rewrites derived files in seconds on every run; once it has
                    # written them it is not work left.
                    done = True
                if step.pipeline_arm:
                    progress = arm_progress(project, step, now=now)
                    if not done and (progress.state != "NOT_STARTED" or progress.completed_steps is not None):
                        status["arms"].append(asdict(progress) | {"fraction": progress.fraction, "why": why})
                    if not done and progress.alive and progress.eta_seconds is not None:
                        stage_seconds += progress.eta_seconds
                        pending.append(step.name)
                        continue
                if not done:
                    pending.append(step.name)
                    stage_seconds += float(step.estimate.seconds)
            entry.update(
                steps_total=len(steps),
                steps_done=len(steps) - len(pending),
                next_step=pending[0] if pending else "",
                remaining_seconds=stage_seconds,
            )
            remaining += stage_seconds
        status["stages"].append(entry)
    if plan is not None:
        status["plan_available"] = True
        status["remaining_seconds"] = remaining
        status["remaining_disk_bytes"] = int(plan.pending().disk_bytes)
    return status


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _gb(size: int | None) -> str:
    return "-" if size is None else f"{size / 1e9:.1f} GB"


def render_status(status: dict[str, Any]) -> str:
    lines = [f"work    {status['work_root']}  (profile {status['profile']})"]
    invocation = status.get("invocation") or {}
    if invocation.get("dataset"):
        lines.append(
            f"dataset {invocation['dataset']}  (pose route {invocation.get('pose_route') or '-'}, "
            f"recorded {invocation.get('recorded_at', '?')})"
        )
    lines.append("")
    lines.append(f"{'stage':<8} {'state':<12} {'steps':>7}  {'left':>8}  next / reason")
    for entry in status["stages"]:
        steps = f"{entry['steps_done']}/{entry['steps_total']}" if "steps_total" in entry else "-"
        left = _hms(entry.get("remaining_seconds")) if "steps_total" in entry else "-"
        detail = entry.get("next_step") or entry.get("reason") or ""
        lines.append(f"{entry['stage']:<8} {entry['state']:<12} {steps:>7}  {left:>8}  {detail}")
    if not status["plan_available"] and status.get("plan_note"):
        lines.append(f"  ({status['plan_note']})")
    for arm in status["arms"]:
        lines.append("")
        done = arm.get("completed_steps")
        target = arm.get("target_steps")
        if done is None:
            position = "no progress recorded"
        else:
            position = f"step {done:,}/{target:,}" if target else f"step {done:,}"
        if arm.get("fraction") is not None:
            position += f" ({arm['fraction'] * 100:.1f}%)"
        if arm["alive"]:
            rate = arm.get("steps_per_second")
            parts = [f"training {arm['step']}: {position}"]
            if rate:
                parts.append(f"{rate:.2f} steps/s")
            if arm.get("eta_seconds") is not None:
                parts.append(f"ETA {_hms(arm['eta_seconds'])}")
            if arm.get("gaussian_count"):
                parts.append(f"{arm['gaussian_count'] / 1e6:.2f}M gaussians")
            lines.append("  ".join(parts))
            if arm.get("last_progress_age_seconds") is not None and arm["last_progress_age_seconds"] > 600:
                lines.append(f"  warning: no progress record for {_hms(arm['last_progress_age_seconds'])}")
        else:
            lines.append(f"not running {arm['step']}: job {arm['state']}, last progress {position}")
            if arm.get("why"):
                lines.append(f"  next run: {arm['why']}")
        if arm.get("resumed_from"):
            resumed = arm["resumed_from"]
            lines.append(f"  resumed from step {resumed.get('step')} after a {resumed.get('previous_state')} attempt")
    lease = status.get("gpu_lease")
    if lease:
        holder = "alive" if lease["alive"] else "holder gone (stale; the next acquire takes it over)"
        lines.append("")
        lines.append(f"gpu lease: {lease['owner']} (pid {lease['pid']}, since {lease['since']}, {holder})")
    log = status.get("newest_log")
    if log:
        lines.append(f"newest log: {log['path']} ({_hms(log['age_seconds'])} ago)")
        if log.get("last_line"):
            lines.append(f"  {log['last_line']}")
    if status["plan_available"]:
        lines.append("")
        lines.append(
            f"left: about {_hms(status['remaining_seconds'])} of work; "
            f"{_gb(status['free_disk_bytes'])} free, the steps still to run need about "
            f"{_gb(status['remaining_disk_bytes'])} (before the preflight's x1.25 margin)"
        )
    return "\n".join(lines)
