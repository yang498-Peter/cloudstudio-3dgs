#!/usr/bin/env python3
"""Config-driven, resumable orchestration of research arms and deliveries.

The research loop used to live in per-session cmd scripts (``run_arm.cmd``,
``deliver_tag.cmd``, ``queue_runner.cmd``). Every machine path was baked into
them and a crash half-way meant re-running everything by hand. This tool is
the same behaviour with three properties the scripts lacked:

* every machine path comes from one JSON pipeline config
  (``--pipeline-config``, default ``pipeline.json`` in the current directory;
  see ``run_configs/pipeline.house0305.example.json``);
* every external step is a subprocess with its own log file, and a failure
  stops the arm with one clear status line;
* re-running resumes from the first missing artifact instead of redoing work
  that already finished.

    python tools/pipeline.py arm tile0_R1_20k
    python tools/pipeline.py deliver r1d --tile0 tile0_R1_20k
    python tools/pipeline.py queue tile0_R2_20k tile0_R3_20k
    python tools/pipeline.py score RUN/tile0_R1_20k RUN/tile0_R2_20k

Four audit gaps (P0-1..P0-4 of the 2026-09-11 quality-recovery brief) are
closed here rather than in the cmd scripts:

* **Completion is a verdict, not a file.** ``latest.pt`` existing used to
  mean "trained"; a 5k leftover from an earlier job scored as a 20k arm. Now
  every arm and delivery keeps ``<run>/job_state.json`` (RUNNING ->
  CHECKPOINTED -> TRAINING_COMPLETE -> EVALUATED -> QUALITY_ACCEPTED ->
  PUBLISHED, plus FAILED and CONTROLLED_PAUSE). TRAINING_COMPLETE needs a
  loadable checkpoint, completed steps >= the declared target, an allowed
  exit reason in the trainer log and a checkpoint newer than the job start.
  Downstream steps refuse to run on anything else.
* **Configs are frozen.** ``RUN/<arm>.json`` is snapshotted to
  ``config_frozen.json`` before training; a later edit with the same arm
  name is refused (exit 2) instead of silently re-scoring the old run.
* **The delivered PLY is what gets scored.** The exported body PLY is
  re-imported and the battery / three-way / off-trajectory strips run on
  that checkpoint, bound to the PLY's sha256 in ``delivery_report.json``;
  the merged.pt scores stay as a separate pre-export record. Publishing
  lands in ``exports/candidate_<TAG>/`` unless ``--publish`` is passed.
* **One GPU holder at a time, atomically.** Every GPU step takes the
  ``<run_root>/gpu.lock`` lease (O_EXCL create, pid liveness for staleness)
  before starting; the live-process scan from the 2026-09-07 double start
  stays as a secondary check for the train step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import pickle
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = 1
DEFAULT_CONFIG_NAME = "pipeline.json"
EXAMPLE_CONFIG = "run_configs/pipeline.house0305.example.json"

# Paths that must be present; they have no sensible default on any machine.
REQUIRED_PATH_KEYS = (
    "run_root",
    "python",
    "reference_ply",
    "reference_alignment",
    "tile_inputs_manifest",
    "tile_inputs_root",
    "exports_dir",
    "delivery_eval_config",
    "sky_ply",
)
# Optional paths: repo_root falls back to this checkout, env_script to the
# inherited environment.
OPTIONAL_PATH_KEYS = ("repo_root", "env_script")

DEFAULTS: dict[str, Any] = {
    "identity_dir": "research/quality_recovery_v1/identity",
    "scene_tag": "house0305",
    "compare_frames": 6,
    "battery_views": 48,
    "export_min_opacity": 0.05,
    "merge_policy": "core_owner_only",
    "harmonize_exposure": True,
    "delivery_tiles": [1, 2, 3],
    "delivery_tile_arm_pattern": "tile{tile}_{tag}_20k",
    "delivery_baselines": {"compare": [], "offtraj": {}},
    "trainer_process_pattern": "train_gsplat.py",
    "gpu_device": "cuda:0",
    "env": {"PYTHONIOENCODING": "utf-8"},
}

KNOWN_KEYS = frozenset(
    ("schema_version",) + REQUIRED_PATH_KEYS + OPTIONAL_PATH_KEYS + tuple(DEFAULTS)
)

# Opacity thresholds of the export-threshold control step (P0-3): the count
# removed at each one is recorded so a delivery cannot hide a transparency
# problem behind the default 0.05 cut.
EXPORT_THRESHOLD_CONTROL = (0.0, 0.01, 0.05)


class PipelineError(RuntimeError):
    """A configuration or environment problem that makes the run pointless."""


class PipelineConfigError(PipelineError):
    """The pipeline config is missing or malformed; the message names the key."""


class ConfigFrozenError(PipelineError):
    """RUN/<arm>.json changed after the arm was frozen; a new arm name is needed."""


class StepFailed(RuntimeError):
    """One step failed; the arm stops here and the message says why."""


class GpuLeaseBusy(RuntimeError):
    """Another live process holds the GPU lease."""


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_text_atomic(path: Path, text: str) -> None:
    """Write via a sibling temp file so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_atomic(source: Path, target: Path) -> None:
    """Copy through a temp file: a reader never sees a truncated PLY/config."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _same_content(left: Path, right: Path) -> bool:
    if not (left.exists() and right.exists()):
        return False
    return left.read_bytes() == right.read_bytes()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stamp(path: Path) -> dict[str, Any]:
    """Size and mtime: enough to notice a rewrite without hashing gigabytes."""
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime": stat.st_mtime}


def read_ply_vertex_count(path: Path) -> int:
    """Vertex count from a PLY header, without numpy or torch.

    The exporter writes ``element vertex N``; reading it back is how the
    threshold-control step records how many gaussians each opacity cut removed.
    """
    with Path(path).open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"unterminated PLY header: {path}")
            tokens = line.decode("ascii", errors="replace").split()
            if tokens[:2] == ["element", "vertex"]:
                return int(tokens[2])
            if tokens[:1] == ["end_header"]:
                raise ValueError(f"PLY header has no vertex element: {path}")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Resolved machine paths and knobs for one scene/host pair."""

    source: Path | None
    run_root: Path
    python: Path
    repo_root: Path
    env_script: Path | None
    reference_ply: Path
    reference_alignment: Path
    tile_inputs_manifest: Path
    tile_inputs_root: Path
    exports_dir: Path
    delivery_eval_config: Path
    sky_ply: Path
    identity_dir: Path
    scene_tag: str
    compare_frames: int
    battery_views: int
    export_min_opacity: float
    merge_policy: str
    harmonize_exposure: bool
    delivery_tiles: list[int]
    delivery_tile_arm_pattern: str
    delivery_baselines: dict[str, Any]
    trainer_process_pattern: str
    gpu_device: str
    env: dict[str, str]

    # Derived locations -----------------------------------------------------

    def arm_config(self, arm: str) -> Path:
        return self.run_root / f"{arm}.json"

    def arm_dir(self, arm: str) -> Path:
        return self.run_root / arm

    def arm_checkpoint(self, arm: str) -> Path:
        return self.arm_dir(arm) / "checkpoints" / "latest.pt"

    def arm_train_logs(self, arm: str) -> tuple[Path, Path]:
        """stdout and stderr logs of the trainer subprocess."""
        return self.run_root / f"{arm}.log", self.run_root / f"{arm}.log.err"

    def arm_scores_file(self) -> Path:
        return self.run_root / "arm_scores.txt"

    def queue_status_file(self) -> Path:
        return self.run_root / "queue_status.txt"

    def gpu_lock_file(self) -> Path:
        return self.run_root / "gpu.lock"

    def delivery_dir(self, tag: str) -> Path:
        return self.run_root / f"delivery_{tag}"

    def candidate_exports_dir(self, tag: str) -> Path:
        return self.exports_dir / f"candidate_{tag}"

    def delivery_tile_arm(self, tag: str, tile: int) -> str:
        return self.delivery_tile_arm_pattern.format(tile=tile, tag=tag)

    def delivery_ply_name(self, tag: str) -> str:
        return f"{self.scene_tag}_{tag}_merged.ply"

    def delivery_sky_name(self, tag: str) -> str:
        return f"{self.scene_tag}_{tag}_sky.ply"

    def tool(self, name: str) -> Path:
        return self.repo_root / "tools" / name

    def missing_paths(self) -> list[tuple[str, Path]]:
        """Inputs that should already exist on this host (not run outputs)."""
        checks = {
            "run_root": self.run_root,
            "python": self.python,
            "repo_root": self.repo_root,
            "reference_ply": self.reference_ply,
            "reference_alignment": self.reference_alignment,
            "tile_inputs_manifest": self.tile_inputs_manifest,
            "tile_inputs_root": self.tile_inputs_root,
            "delivery_eval_config": self.delivery_eval_config,
            "sky_ply": self.sky_ply,
        }
        if self.env_script is not None:
            checks["env_script"] = self.env_script
        return [(key, path) for key, path in checks.items() if not path.exists()]


def _resolve(value: Any, key: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigError(f"pipeline config key '{key}' must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def parse_pipeline_config(raw: dict[str, Any], *, source: Path | None = None) -> PipelineConfig:
    """Validate a config dict; keys starting with '_' are documentation."""
    if not isinstance(raw, dict):
        raise PipelineConfigError("pipeline config must be a JSON object")
    base = source.parent if source is not None else Path.cwd()
    data = {key: value for key, value in raw.items() if not key.startswith("_")}
    unknown = sorted(set(data) - KNOWN_KEYS)
    if unknown:
        raise PipelineConfigError(f"unknown pipeline config key(s): {', '.join(unknown)}")
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise PipelineConfigError(
            f"pipeline config schema_version {version!r} is not supported (expected {SCHEMA_VERSION})"
        )
    missing = [key for key in REQUIRED_PATH_KEYS if key not in data]
    if missing:
        raise PipelineConfigError(f"pipeline config is missing required key(s): {', '.join(missing)}")

    paths = {key: _resolve(data[key], key, base) for key in REQUIRED_PATH_KEYS}
    repo_root = _resolve(data["repo_root"], "repo_root", base) if "repo_root" in data else ROOT
    env_script = _resolve(data["env_script"], "env_script", base) if data.get("env_script") else None

    merged = {**DEFAULTS, **{key: data[key] for key in DEFAULTS if key in data}}
    for key in ("compare_frames", "battery_views"):
        if not isinstance(merged[key], int) or isinstance(merged[key], bool) or merged[key] <= 0:
            raise PipelineConfigError(f"pipeline config key '{key}' must be a positive integer")
    if not isinstance(merged["export_min_opacity"], (int, float)) or isinstance(merged["export_min_opacity"], bool):
        raise PipelineConfigError("pipeline config key 'export_min_opacity' must be a number")
    if not isinstance(merged["harmonize_exposure"], bool):
        raise PipelineConfigError("pipeline config key 'harmonize_exposure' must be true or false")
    tiles = merged["delivery_tiles"]
    if not isinstance(tiles, list) or not tiles or not all(isinstance(t, int) and not isinstance(t, bool) for t in tiles):
        raise PipelineConfigError("pipeline config key 'delivery_tiles' must be a non-empty list of integers")
    pattern = merged["delivery_tile_arm_pattern"]
    if not isinstance(pattern, str) or "{tile}" not in pattern or "{tag}" not in pattern:
        raise PipelineConfigError("pipeline config key 'delivery_tile_arm_pattern' must contain {tile} and {tag}")
    for key in ("scene_tag", "merge_policy", "trainer_process_pattern", "gpu_device"):
        if not isinstance(merged[key], str) or not merged[key]:
            raise PipelineConfigError(f"pipeline config key '{key}' must be a non-empty string")
    env = merged["env"]
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise PipelineConfigError("pipeline config key 'env' must map strings to strings")

    baselines_raw = merged["delivery_baselines"]
    if not isinstance(baselines_raw, dict):
        raise PipelineConfigError("pipeline config key 'delivery_baselines' must be an object")
    compare_raw = baselines_raw.get("compare", [])
    offtraj_raw = baselines_raw.get("offtraj", {})
    if not isinstance(compare_raw, list) or not isinstance(offtraj_raw, dict):
        raise PipelineConfigError(
            "pipeline config 'delivery_baselines' needs 'compare' (list of dirs) and 'offtraj' (name -> dir)"
        )
    run_root = paths["run_root"]
    baselines = {
        "compare": [_resolve(item, "delivery_baselines.compare", run_root) for item in compare_raw],
        "offtraj": {
            str(name): _resolve(item, f"delivery_baselines.offtraj.{name}", run_root)
            for name, item in offtraj_raw.items()
        },
    }
    identity_dir = _resolve(merged["identity_dir"], "identity_dir", repo_root)

    return PipelineConfig(
        source=source,
        run_root=run_root,
        python=paths["python"],
        repo_root=repo_root,
        env_script=env_script,
        reference_ply=paths["reference_ply"],
        reference_alignment=paths["reference_alignment"],
        tile_inputs_manifest=paths["tile_inputs_manifest"],
        tile_inputs_root=paths["tile_inputs_root"],
        exports_dir=paths["exports_dir"],
        delivery_eval_config=paths["delivery_eval_config"],
        sky_ply=paths["sky_ply"],
        identity_dir=identity_dir,
        scene_tag=merged["scene_tag"],
        compare_frames=int(merged["compare_frames"]),
        battery_views=int(merged["battery_views"]),
        export_min_opacity=float(merged["export_min_opacity"]),
        merge_policy=merged["merge_policy"],
        harmonize_exposure=bool(merged["harmonize_exposure"]),
        delivery_tiles=[int(t) for t in tiles],
        delivery_tile_arm_pattern=pattern,
        delivery_baselines=baselines,
        trainer_process_pattern=merged["trainer_process_pattern"],
        gpu_device=merged["gpu_device"],
        env=dict(env),
    )


def load_pipeline_config(path: Path) -> PipelineConfig:
    if not path.exists():
        raise PipelineConfigError(
            f"pipeline config not found: {path}. Copy {EXAMPLE_CONFIG} to {DEFAULT_CONFIG_NAME} "
            "and edit the machine paths, or pass --pipeline-config."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PipelineConfigError(f"pipeline config {path} is not valid JSON: {error}") from error
    return parse_pipeline_config(raw, source=path.resolve())


# --------------------------------------------------------------------------
# Environment and process helpers
# --------------------------------------------------------------------------


def load_env_script(script: Path) -> dict[str, str]:
    """Capture the variables a cmd env script exports.

    The training env script `call`s vcvars and pins CUDA/JIT variables; the
    cmd scripts inherited them from the shell. A subprocess cannot inherit
    from a batch file, so run it once and read back `set`.
    """
    if os.name != "nt":
        raise PipelineError(f"env_script is a Windows cmd file and cannot be sourced here: {script}")
    if not script.exists():
        raise PipelineError(f"env_script not found: {script}")
    # Passing `call "<path>" && set` as one argv element makes Python quote it
    # for CreateProcess, so cmd receives \"<path>\" and reports "not recognized"
    # (exit 1). A throwaway wrapper batch file sidesteps argv quoting entirely.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="ascii", newline="\r\n") as handle:
        handle.write("@echo off\r\n")
        handle.write(f'call "{script}" >nul 2>&1\r\n')
        handle.write("if errorlevel 1 exit /b 1\r\n")
        handle.write("set\r\n")
        wrapper = pathlib.Path(handle.name)
    try:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(wrapper)], capture_output=True, text=True, check=False, encoding="utf-8", errors="replace"
        )
    finally:
        wrapper.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise PipelineError(f"env_script {script} exited {completed.returncode}: {completed.stderr.strip()[:400]}")
    env: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            env[key] = value
    if not env:
        raise PipelineError(f"env_script {script} produced no environment")
    return env


def _parse_process_listing(text: str) -> list[tuple[int, str]]:
    """Lines of '<pid>|<command line>' (Windows) or '<pid> <args>' (POSIX)."""
    processes: list[tuple[int, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            pid_text, _, command = line.partition("|")
        else:
            pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text.strip())
        except ValueError:
            continue
        processes.append((pid, command.strip()))
    return processes


def list_processes() -> list[tuple[int, str]]:
    """(pid, command line) for every visible process on this host."""
    if os.name == "nt":
        script = (
            "Get-CimInstance Win32_Process | ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"
        )
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    else:
        argv = ["ps", "-eo", "pid=,args="]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise PipelineError(f"process listing failed ({argv[0]} exit {completed.returncode})")
    return _parse_process_listing(completed.stdout)


def find_trainer_processes(pattern: str, processes: Sequence[tuple[int, str]] | None = None) -> list[tuple[int, str]]:
    """Live processes whose command line mentions the trainer script.

    The pipeline itself and its parent shell are excluded so the guard does
    not trip on the process doing the scanning.
    """
    own = {os.getpid(), os.getppid()}
    listing = list_processes() if processes is None else processes
    return [(pid, command) for pid, command in listing if pattern in command and pid not in own]


# --------------------------------------------------------------------------
# GPU lease (P0-4)
# --------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """Is a process with this pid still running?

    ``os.kill(pid, 0)`` is the POSIX idiom; on Windows ``os.kill`` with any
    signal other than the console events *terminates* the target, so the
    liveness probe goes through OpenProcess/GetExitCodeProcess instead.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_query_limited_information = 0x1000
        still_active = 259
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            # ERROR_ACCESS_DENIED means the process exists but belongs to
            # another session; ERROR_INVALID_PARAMETER means no such pid.
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == still_active
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _command_sha256(command: Sequence[str] | str) -> str:
    text = command if isinstance(command, str) else "\x00".join(str(item) for item in command)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_gpu_lease(path: Path) -> dict[str, Any] | None:
    """The holder record, or None when the lock file is absent/unreadable."""
    try:
        payload = _read_json(path)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


@dataclass
class GpuLease:
    """A held ``gpu.lock``; ``release`` removes it only if we still own it."""

    path: Path
    holder: dict[str, Any]

    def release(self) -> None:
        current = read_gpu_lease(self.path)
        if current is not None and current.get("pid") == self.holder.get("pid") and current.get("token") == self.holder.get("token"):
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def acquire_gpu_lease(
    path: Path,
    *,
    command: Sequence[str] | str,
    device: str,
    owner: str = "",
    pid: int | None = None,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> GpuLease:
    """Take the process-level GPU lease or raise :class:`GpuLeaseBusy`.

    The lock is created with ``O_EXCL`` so two processes racing for it cannot
    both succeed; the holder record (pid, start time, command hash, device)
    is written into the file *after* the exclusive create, and a reader that
    sees an empty or half-written file simply treats it as busy. A lease whose
    pid is dead is stale (the holder crashed without releasing) and is
    reclaimed by the next caller.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    holder = {
        "pid": int(os.getpid() if pid is None else pid),
        "started_at": time.time(),
        "started_at_text": _timestamp(),
        "host": socket.gethostname(),
        "device": device,
        "owner": owner,
        "command": command if isinstance(command, str) else [str(item) for item in command],
        "command_sha256": _command_sha256(command),
        "token": f"{os.getpid()}-{time.time_ns()}",
    }
    for attempt in range(2):
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = read_gpu_lease(path)
            if current is None:
                # Unreadable: either being written right now or garbage. Only
                # reclaim garbage that has sat there for a while.
                try:
                    age = time.time() - path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > 30.0 and attempt == 0:
                    path.unlink(missing_ok=True)
                    continue
                raise GpuLeaseBusy(f"GPU lease {path} is being written by another process")
            holder_pid = int(current.get("pid", 0) or 0)
            if holder_pid != holder["pid"] and not pid_alive(holder_pid) and attempt == 0:
                path.unlink(missing_ok=True)
                continue
            raise GpuLeaseBusy(
                f"GPU lease {path} held by pid {holder_pid} since {current.get('started_at_text', '?')} "
                f"({current.get('owner', '')}: {str(current.get('command', ''))[:120]})"
            )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(holder, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        return GpuLease(path, holder)
    raise GpuLeaseBusy(f"GPU lease {path} could not be acquired")


# --------------------------------------------------------------------------
# Checkpoint inspection (P0-1)
# --------------------------------------------------------------------------

# torch.save archives are zip files with <name>/data.pkl inside; a real
# checkpoint is megabytes, so anything under this is a stub or a truncation.
MIN_CHECKPOINT_BYTES = 1024


@dataclass
class CheckpointInfo:
    """What could be established about a checkpoint file."""

    path: Path
    exists: bool
    loadable: bool
    step: int | None
    method: str  # "torch" (payload validated) or "header" (zip layout only)
    reason: str = ""
    size_bytes: int = 0
    mtime: float = 0.0

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "loadable": self.loadable,
            "step": self.step,
            "method": self.method,
            "reason": self.reason,
            "bytes": self.size_bytes,
            "mtime": self.mtime,
        }


CheckpointLoader = Callable[[Path], dict[str, Any]]


def _torch_checkpoint_loader() -> CheckpointLoader | None:
    """The torch path: returns None when torch is not importable here."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None

    def load(path: Path) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:  # older torch without mmap
            payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload is not a dict")
        return payload

    return load


class _StubObject:
    """Stands in for every torch/numpy object while peeking at a pickle."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "_StubObject":
        return object.__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> "_StubObject":
        return _StubObject()

    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def append(self, item: Any) -> None:
        pass

    def extend(self, items: Any) -> None:
        pass


_PEEK_SAFE_CLASSES = frozenset(
    {
        ("collections", "OrderedDict"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "bytearray"),
        ("builtins", "complex"),
        ("builtins", "range"),
        ("builtins", "slice"),
    }
)


class _PeekUnpickler(pickle.Unpickler):
    """Rebuilds the checkpoint dict skeleton without importing torch.

    Tensor storages arrive as persistent ids and become None; every class
    outside a tiny safe list becomes :class:`_StubObject`. Only the scalar
    ``step`` survives intact, which is all the verdict needs, and no foreign
    code can run because nothing outside the safe list is ever imported.
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _PEEK_SAFE_CLASSES:
            return super().find_class(module, name)
        return _StubObject

    def persistent_load(self, pid: Any) -> Any:
        return None


def peek_checkpoint_step(path: Path) -> int | None:
    """``step`` from a torch zip checkpoint without torch; None if unreadable."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith("/data.pkl") or name == "data.pkl"]
            if not names:
                return None
            with archive.open(names[0]) as handle:
                payload = _PeekUnpickler(handle).load()
    except Exception:  # noqa: BLE001 - any failure just means "unknown"
        return None
    if not isinstance(payload, dict):
        return None
    step = payload.get("step")
    return int(step) if isinstance(step, int) and not isinstance(step, bool) else None


def inspect_checkpoint(path: Path, *, loader: CheckpointLoader | None = None) -> CheckpointInfo:
    """Is this checkpoint loadable, and at which step?

    With torch (the training host) the payload is loaded and validated:
    a dict with ``params`` and an integer ``step``. Without torch the zip
    layout and size are checked and ``step`` is peeked from ``data.pkl``;
    that is weaker, so the verdict also cross-checks the trainer log.
    """
    path = Path(path)
    if not path.is_file():
        return CheckpointInfo(path, False, False, None, "none", "checkpoint file missing")
    stat = path.stat()
    info = CheckpointInfo(path, True, False, None, "header", size_bytes=stat.st_size, mtime=stat.st_mtime)
    if stat.st_size < MIN_CHECKPOINT_BYTES:
        info.reason = f"checkpoint is {stat.st_size} bytes (< {MIN_CHECKPOINT_BYTES}); truncated or a stub"
        return info
    loader = _torch_checkpoint_loader() if loader is None else loader
    if loader is not None:
        info.method = "torch"
        try:
            payload = loader(path)
        except Exception as error:  # noqa: BLE001 - the reason is the point
            info.reason = f"torch.load failed: {type(error).__name__}: {str(error)[:200]}"
            return info
        step = payload.get("step")
        if "params" not in payload or not isinstance(step, int) or isinstance(step, bool):
            info.reason = "checkpoint payload lacks 'params' or an integer 'step'"
            return info
        info.loadable = True
        info.step = int(step)
        return info
    if not zipfile.is_zipfile(path):
        info.reason = "not a torch zip archive"
        return info
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if archive.testzip() is not None:
                info.reason = "zip archive has a corrupt member"
                return info
    except zipfile.BadZipFile as error:
        info.reason = f"bad zip archive: {error}"
        return info
    if not any(name.endswith("data.pkl") for name in names):
        info.reason = "zip archive has no data.pkl (not a torch checkpoint)"
        return info
    info.loadable = True
    info.step = peek_checkpoint_step(path)
    return info


# --------------------------------------------------------------------------
# Trainer exit classification (P0-1)
# --------------------------------------------------------------------------

EXIT_CONTROLLED_STOP = "controlled_stop"
EXIT_COMPLETED = "completed"
EXIT_OOM = "oom"
EXIT_CRASH = "crash"
EXIT_UNKNOWN = "unknown"
ALLOWED_EXIT_KINDS = (EXIT_CONTROLLED_STOP, EXIT_COMPLETED)

_CONTROLLED_STOP_RE = re.compile(r"ControlledTrainingInterruption: controlled interruption after (\d+) steps")
_TRAINING_COMPLETE_RE = re.compile(r"training complete: run=.*?steps=(\d+)")
_OOM_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "cudaErrorMemoryAllocation",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
)
LOG_TAIL_BYTES = 64 * 1024


@dataclass
class TrainerExit:
    kind: str
    steps: int | None
    detail: str

    def record(self) -> dict[str, Any]:
        return {"kind": self.kind, "steps": self.steps, "detail": self.detail}


def read_log_tail(paths: Sequence[Path], *, max_bytes: int = LOG_TAIL_BYTES) -> str:
    """Last ``max_bytes`` of each log, concatenated; missing logs are skipped."""
    pieces: list[str] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            pieces.append(handle.read().decode("utf-8", errors="replace"))
    return "\n".join(pieces)


def classify_trainer_exit(exit_code: int | None, tail: str) -> TrainerExit:
    """Why did the trainer stop, according to its log tail?

    A controlled stop is raised as an exception, so its traceback *is* the
    marker; it is checked before the generic Traceback rule. OOM beats a
    plain crash because it is the case a retry can never fix by itself.
    """
    controlled = _CONTROLLED_STOP_RE.findall(tail)
    if controlled:
        return TrainerExit(EXIT_CONTROLLED_STOP, int(controlled[-1]), "ControlledTrainingInterruption marker in log")
    for marker in _OOM_MARKERS:
        if marker in tail:
            return TrainerExit(EXIT_OOM, None, f"{marker} in log (exit {exit_code})")
    if "Traceback (most recent call last)" in tail:
        return TrainerExit(EXIT_CRASH, None, f"traceback in log without controlled-stop marker (exit {exit_code})")
    complete = _TRAINING_COMPLETE_RE.findall(tail)
    if complete and exit_code == 0:
        return TrainerExit(EXIT_COMPLETED, int(complete[-1]), "training complete marker in log")
    if exit_code == 0:
        return TrainerExit(EXIT_UNKNOWN, None, "exit 0 without a completion marker in log")
    return TrainerExit(EXIT_UNKNOWN, None, f"exit {exit_code} without a recognised marker in log")


# --------------------------------------------------------------------------
# Job state (P0-1)
# --------------------------------------------------------------------------

JOB_STATE_VERSION = 1
JOB_STATE_NAME = "job_state.json"

STATE_RUNNING = "RUNNING"
STATE_CHECKPOINTED = "CHECKPOINTED"
STATE_TRAINING_COMPLETE = "TRAINING_COMPLETE"
STATE_EVALUATED = "EVALUATED"
STATE_QUALITY_ACCEPTED = "QUALITY_ACCEPTED"
STATE_PUBLISHED = "PUBLISHED"
STATE_FAILED = "FAILED"
STATE_CONTROLLED_PAUSE = "CONTROLLED_PAUSE"
JOB_STATES = (
    STATE_RUNNING,
    STATE_CHECKPOINTED,
    STATE_TRAINING_COMPLETE,
    STATE_EVALUATED,
    STATE_QUALITY_ACCEPTED,
    STATE_PUBLISHED,
    STATE_FAILED,
    STATE_CONTROLLED_PAUSE,
)
# States that imply verified training; the ordering is the milestone ladder.
TRAINED_STATES = (STATE_TRAINING_COMPLETE, STATE_EVALUATED, STATE_QUALITY_ACCEPTED, STATE_PUBLISHED)


class JobState:
    """``<run>/job_state.json``: the persisted state of one arm or delivery.

    ``state`` is the pipeline's position; ``training`` is the verdict on the
    checkpoint and is kept separately so a failed *scoring* step marks the
    job FAILED without un-verifying a training that genuinely completed
    (otherwise the resume would retrain). Every write is atomic and every
    transition is appended to ``history``.
    """

    def __init__(self, path: Path, *, job: str, name: str) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "schema_version": JOB_STATE_VERSION,
            "job": job,
            "name": name,
            "state": None,
            "reason": "",
            "history": [],
        }
        if self.path.is_file():
            try:
                loaded = _read_json(self.path)
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict) and loaded.get("schema_version") == JOB_STATE_VERSION:
                self.data.update(loaded)

    @property
    def state(self) -> str | None:
        return self.data.get("state")

    @property
    def reason(self) -> str:
        return str(self.data.get("reason", ""))

    def exists(self) -> bool:
        return self.path.is_file()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def training_verified(self) -> bool:
        training = self.data.get("training")
        return isinstance(training, dict) and bool(training.get("verified"))

    def save(self) -> None:
        self.data["updated_at"] = _timestamp()
        _write_json_atomic(self.path, self.data)

    def set(self, state: str, reason: str = "", **fields: Any) -> None:
        if state not in JOB_STATES:
            raise ValueError(f"unknown job state {state!r}")
        if state in TRAINED_STATES and not (self.training_verified() or (fields.get("training") or {}).get("verified")):
            raise ValueError(f"cannot enter {state} without a verified training record")
        self.data.update(fields)
        self.data["state"] = state
        self.data["reason"] = reason
        self.data.setdefault("history", []).append({"state": state, "at": _timestamp(), "reason": reason})
        self.save()

    def update(self, **fields: Any) -> None:
        self.data.update(fields)
        self.save()

    def fail(self, reason: str, **fields: Any) -> None:
        self.set(STATE_FAILED, reason, **fields)


# --------------------------------------------------------------------------
# Training verification (P0-1)
# --------------------------------------------------------------------------

# Slack for filesystem timestamp granularity when comparing the checkpoint
# mtime with the job start recorded by time.time().
MTIME_TOLERANCE_SECONDS = 2.0


def declared_target_steps(arm_config: Path) -> int | None:
    """Steps the arm is declared to run: controlled stop first, else max_steps."""
    try:
        payload = _read_json(arm_config)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("controlled_stop_after_steps", "max_steps"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


@dataclass
class TrainingVerdict:
    """Outcome of :func:`verify_training`; ``state`` is one of three."""

    state: str  # TRAINING_COMPLETE | CONTROLLED_PAUSE | FAILED
    reason: str
    completed_steps: int | None
    target_steps: int | None
    checkpoint: CheckpointInfo
    exit: TrainerExit
    exit_code: int | None
    job_started_at: float | None
    checks: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.state == STATE_TRAINING_COMPLETE

    def record(self) -> dict[str, Any]:
        return {
            "verified": self.complete,
            "verdict": self.state,
            "reason": self.reason,
            "completed_steps": self.completed_steps,
            "target_steps": self.target_steps,
            "checkpoint": self.checkpoint.record(),
            "exit": self.exit.record(),
            "exit_code": self.exit_code,
            "job_started_at": self.job_started_at,
            "checks": dict(self.checks),
            "verified_at": _timestamp(),
        }


def verify_training(
    *,
    checkpoint: Path,
    arm_config: Path,
    log_tail: str,
    exit_code: int | None,
    job_started_at: float | None,
    inspector: Callable[[Path], CheckpointInfo] = inspect_checkpoint,
) -> TrainingVerdict:
    """Decide whether a training is complete from all four kinds of evidence.

    Every check runs so the reason names everything wrong at once; the
    first failing check decides the state. ``job_started_at`` None means a
    run adopted from before job states existed, where the leftover check
    has nothing to compare against and is recorded as not applicable.
    """
    info = inspector(Path(checkpoint))
    exit = classify_trainer_exit(exit_code, log_tail)
    target = declared_target_steps(Path(arm_config))
    checks: dict[str, str] = {}
    failures: list[str] = []

    if not info.exists:
        failures.append(f"checkpoint missing: {info.path}")
        checks["checkpoint"] = "missing"
    elif not info.loadable:
        failures.append(f"checkpoint not loadable: {info.reason}")
        checks["checkpoint"] = "not loadable"
    else:
        checks["checkpoint"] = f"loadable ({info.method})"

    if target is None:
        failures.append(f"arm config declares neither controlled_stop_after_steps nor max_steps: {arm_config}")
        checks["target"] = "undeclared"
    else:
        checks["target"] = str(target)

    if exit.kind in ALLOWED_EXIT_KINDS:
        checks["exit"] = f"{exit.kind}: {exit.detail}"
    else:
        failures.append(f"trainer exit is {exit.kind} ({exit.detail}); checkpoint kept for diagnosis")
        checks["exit"] = f"{exit.kind}: {exit.detail}"

    if job_started_at is None:
        checks["mtime"] = "not applicable (adopted run without a recorded job start)"
    elif info.exists and info.mtime + MTIME_TOLERANCE_SECONDS < job_started_at:
        failures.append(
            f"checkpoint mtime {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.mtime))} predates job start "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job_started_at))}; leftover of a previous job"
        )
        checks["mtime"] = "older than job start"
    elif info.exists:
        checks["mtime"] = "newer than job start"

    completed: int | None = None
    if info.step is not None and exit.steps is not None and info.step != exit.steps:
        failures.append(f"checkpoint step {info.step} disagrees with the trainer log ({exit.steps})")
        checks["steps"] = "inconsistent"
    else:
        completed = info.step if info.step is not None else exit.steps
        if completed is None:
            failures.append("completed steps unknown: no step in checkpoint and no marker in the trainer log")
            checks["steps"] = "unknown"
        else:
            checks["steps"] = f"{completed} from {'checkpoint' if info.step is not None else 'log'}"

    if failures:
        state, reason = STATE_FAILED, "; ".join(failures)
    elif completed is not None and target is not None and completed >= target:
        state, reason = STATE_TRAINING_COMPLETE, f"{completed} >= {target} steps, {exit.kind}"
    elif exit.kind == EXIT_CONTROLLED_STOP:
        state = STATE_CONTROLLED_PAUSE
        reason = f"controlled stop at {completed} steps, short of the declared {target}"
    else:
        state = STATE_FAILED
        reason = f"trainer reported {exit.kind} at {completed} steps, short of the declared {target}"
    return TrainingVerdict(
        state=state,
        reason=reason,
        completed_steps=completed,
        target_steps=target,
        checkpoint=info,
        exit=exit,
        exit_code=exit_code,
        job_started_at=job_started_at,
        checks=checks,
    )


# --------------------------------------------------------------------------
# Config immutability (P0-2)
# --------------------------------------------------------------------------

CONFIG_FROZEN_NAME = "config_frozen.json"
CONFIG_AS_RUN_NAME = "config_as_run.json"


def freeze_arm_config(arm_config: Path, frozen: Path, *, legacy_record: Path | None = None) -> str:
    """Snapshot ``RUN/<arm>.json`` once; refuse if it changed since the snapshot.

    Returns the sha256 of the frozen config. A different sha with the same
    arm name is refused with :class:`ConfigFrozenError` because every
    artifact under the arm directory would otherwise be attributed to a
    config that never produced it. ``legacy_record`` is the pre-P0-2
    ``config_as_run.json``: for a run that predates freezing it is the only
    record of what trained, so it is trusted as the snapshot source and an
    edited ``RUN/<arm>.json`` is refused against it.
    """
    if not arm_config.is_file():
        raise ConfigFrozenError(f"arm config missing: {arm_config}")
    current = file_sha256(arm_config)
    if not frozen.is_file() and legacy_record is not None and legacy_record.is_file():
        recorded = file_sha256(legacy_record)
        if recorded != current:
            raise ConfigFrozenError(
                f"{arm_config.name} differs from the {legacy_record.name} this run trained with "
                f"(recorded sha256 {recorded[:12]}, current {current[:12]}). "
                f"Create a new arm name for the new config instead of editing {arm_config}"
            )
        _copy_atomic(legacy_record, frozen)
        return recorded
    if frozen.is_file():
        recorded = file_sha256(frozen)
        if recorded != current:
            raise ConfigFrozenError(
                f"{arm_config.name} changed since it was frozen for this run "
                f"(frozen sha256 {recorded[:12]}, current {current[:12]}). "
                f"Create a new arm name for the new config instead of editing {arm_config}"
            )
        return recorded
    _copy_atomic(arm_config, frozen)
    return current


# --------------------------------------------------------------------------
# Steps and their executor
# --------------------------------------------------------------------------

RunCommand = Callable[..., int]


@dataclass
class Step:
    """One unit of resumable work.

    ``anchor`` steps define the resume point: the first anchor whose ``done``
    predicate is false, and every step after it, runs again. Non-anchor steps
    are cheap idempotent bookkeeping (prune, copy) that run only when not done
    and never force downstream work to repeat. ``independent`` steps (the
    per-tile trainings of a delivery) still anchor - a missing tile must
    invalidate the merge - but are never repeated once done, because they do
    not consume anything an earlier step produced.
    """

    name: str
    run: Callable[[], None]
    artifacts: tuple[Path, ...] = ()
    done: Callable[[], bool] | None = None
    anchor: bool = True
    independent: bool = False

    def is_done(self) -> bool:
        if self.done is not None:
            return bool(self.done())
        return bool(self.artifacts) and all(path.exists() for path in self.artifacts)


@dataclass
class StepReport:
    name: str
    action: str  # "skip", "done", "failed"
    detail: str = ""


def plan_steps(steps: Sequence[Step], *, force: bool = False) -> list[tuple[Step, bool]]:
    """Which steps would run: (step, will_run) in execution order."""
    resume_from = 0 if force else len(steps)
    if not force:
        for index, step in enumerate(steps):
            if step.anchor and not step.is_done():
                resume_from = index
                break
    plan: list[tuple[Step, bool]] = []
    for index, step in enumerate(steps):
        if force:
            will_run = True
        elif index >= resume_from:
            will_run = not (step.independent and step.is_done())
        else:
            will_run = not step.anchor and not step.is_done()
        plan.append((step, will_run))
    return plan


def run_steps(
    steps: Sequence[Step],
    *,
    force: bool = False,
    status: Callable[[str], None],
) -> list[StepReport]:
    """Execute the plan; stop at the first failure and say which step it was."""
    reports: list[StepReport] = []
    for step, will_run in plan_steps(steps, force=force):
        if not will_run:
            status(f"skip {step.name} (already done)")
            reports.append(StepReport(step.name, "skip"))
            continue
        status(f"start {step.name}")
        try:
            step.run()
        except StepFailed as error:
            status(f"FAILED {step.name}: {error}")
            reports.append(StepReport(step.name, "failed", str(error)))
            break
        missing = [path for path in step.artifacts if not path.exists()]
        if missing:
            detail = f"expected artifact missing after step: {missing[0]}"
            status(f"FAILED {step.name}: {detail}")
            reports.append(StepReport(step.name, "failed", detail))
            break
        status(f"done {step.name}")
        reports.append(StepReport(step.name, "done"))
    return reports


def _subprocess_run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: Path,
    stderr: Path | None,
    append: bool,
) -> int:
    """Run one external step with its output captured to file(s)."""
    stdout.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if append else "wb"
    with stdout.open(mode) as out_handle:
        if stderr is None:
            return subprocess.run(list(argv), cwd=str(cwd), env=env, stdout=out_handle, stderr=subprocess.STDOUT, check=False).returncode
        stderr.parent.mkdir(parents=True, exist_ok=True)
        with stderr.open(mode) as err_handle:
            return subprocess.run(list(argv), cwd=str(cwd), env=env, stdout=out_handle, stderr=err_handle, check=False).returncode


class PipelineContext:
    """Shared services for one invocation; the two callables are injectable.

    ``run_command`` runs an external step and returns its exit code;
    ``trainer_processes`` returns the live trainer processes;
    ``checkpoint_inspector`` judges a checkpoint file. Tests replace them so
    the resume, guard and verdict logic can be exercised without a GPU.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        run_command: RunCommand | None = None,
        trainer_processes: Callable[[], list[tuple[int, str]]] | None = None,
        checkpoint_inspector: Callable[[Path], CheckpointInfo] | None = None,
        stream=None,
    ) -> None:
        self.config = config
        self._run_command = run_command or _subprocess_run_command
        self._trainer_processes = trainer_processes
        self.checkpoint_inspector = checkpoint_inspector or inspect_checkpoint
        self._env: dict[str, str] | None = None
        self.stream = stream or sys.stdout
        self.tee_files: list[Path] = []
        self.gpu_leases_taken: list[str] = []  # owner names, for tests and status

    # Job state ------------------------------------------------------------

    def arm_job(self, arm: str) -> JobState:
        return JobState(self.config.arm_dir(arm) / JOB_STATE_NAME, job="arm", name=arm)

    def delivery_job(self, tag: str) -> JobState:
        return JobState(self.config.delivery_dir(tag) / JOB_STATE_NAME, job="delivery", name=tag)

    def verify_arm_training(self, arm: str, *, exit_code: int | None, job_started_at: float | None) -> TrainingVerdict:
        cfg = self.config
        return verify_training(
            checkpoint=cfg.arm_checkpoint(arm),
            arm_config=cfg.arm_config(arm),
            log_tail=read_log_tail(cfg.arm_train_logs(arm)),
            exit_code=exit_code,
            job_started_at=job_started_at,
            inspector=self.checkpoint_inspector,
        )

    def arm_training_complete(self, arm: str) -> bool:
        """Verified TRAINING_COMPLETE on disk, with the checkpoint still there.

        An arm trained before job states existed has a checkpoint but no
        record; it is verified from its logs once and the verdict persisted
        (``adopted``), so old tile arms stay usable without retraining.
        """
        job = self.arm_job(arm)
        checkpoint = self.config.arm_checkpoint(arm)
        if job.training_verified():
            recorded = job.get("training", {}).get("checkpoint", {})
            return checkpoint.is_file() and checkpoint.stat().st_size == recorded.get("bytes")
        if job.exists() or not checkpoint.is_file():
            return False
        verdict = self.verify_arm_training(arm, exit_code=None, job_started_at=None)
        job.data["adopted"] = True
        if verdict.complete:
            job.set(STATE_TRAINING_COMPLETE, f"adopted: {verdict.reason}", training=verdict.record())
            return True
        job.set(verdict.state, f"adopted: {verdict.reason}", training=verdict.record())
        return False

    def require_arm_training_complete(self, arm: str) -> None:
        if not self.arm_training_complete(arm):
            job = self.arm_job(arm)
            raise StepFailed(f"arm {arm} training is {job.state or 'unrecorded'} ({job.reason or 'no verdict'}); refusing to run on it")

    # GPU lease ------------------------------------------------------------

    @contextmanager
    def gpu_lease(self, owner: str, command: Sequence[str] | str, *, scan_trainers: bool = False) -> Iterator[GpuLease]:
        """Hold ``gpu.lock`` for one GPU step; optionally also scan processes.

        The lease is the primary, atomic guard; the process scan (the
        pre-P0-4 guard) stays as a secondary check for the train step, where
        a foreign trainer started outside this pipeline would not hold a lease.
        """
        try:
            lease = acquire_gpu_lease(
                self.config.gpu_lock_file(), command=command, device=self.config.gpu_device, owner=owner
            )
        except GpuLeaseBusy as error:
            raise StepFailed(str(error)) from error
        self.gpu_leases_taken.append(owner)
        try:
            if scan_trainers:
                self.ensure_gpu_free()
            yield lease
        finally:
            lease.release()

    # Environment ----------------------------------------------------------

    def environment(self) -> dict[str, str]:
        if self._env is None:
            base = load_env_script(self.config.env_script) if self.config.env_script else dict(os.environ)
            env = dict(base)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(self.config.repo_root) + (os.pathsep + existing if existing else "")
            env.update(self.config.env)
            self._env = env
        return self._env

    def trainer_processes(self) -> list[tuple[int, str]]:
        if self._trainer_processes is not None:
            return list(self._trainer_processes())
        return find_trainer_processes(self.config.trainer_process_pattern)

    def ensure_gpu_free(self) -> None:
        running = self.trainer_processes()
        if running:
            pid, command = running[0]
            raise StepFailed(f"another trainer is running (pid {pid}: {command[:120]}); refusing to start a second one")

    # Status ---------------------------------------------------------------

    def status(self, line: str, *files: Path) -> None:
        text = f"{_timestamp()} {line}"
        print(text, file=self.stream, flush=True)
        for path in (*files, *self.tee_files):
            _append_text(path, text + "\n")

    # Commands -------------------------------------------------------------

    def python_tool(self, tool: str, *args: object) -> list[str]:
        return [str(self.config.python), str(self.config.tool(tool)), *(str(item) for item in args)]

    def run(
        self,
        argv: Sequence[str],
        *,
        log: Path,
        stderr_log: Path | None = None,
        append: bool = False,
    ) -> int:
        return self._run_command(
            argv,
            cwd=self.config.repo_root,
            env=self.environment(),
            stdout=log,
            stderr=stderr_log,
            append=append,
        )

    def run_or_fail(self, argv: Sequence[str], *, log: Path, append: bool = False) -> None:
        code = self.run(argv, log=log, append=append)
        if code != 0:
            raise StepFailed(f"exit {code}; see {log}")

    def run_capture_or_fail(self, argv: Sequence[str], *, capture: Path, log: Path, append: bool = False) -> None:
        """stdout is the artifact; a failure removes it so resume retries."""
        code = self.run(argv, log=capture, stderr_log=log, append=append)
        if code != 0:
            if not append and capture.exists():
                capture.unlink()
            raise StepFailed(f"exit {code}; see {log}")


# --------------------------------------------------------------------------
# Arm pipeline
# --------------------------------------------------------------------------


def arm_steps(ctx: PipelineContext, arm: str) -> list[Step]:
    cfg = ctx.config
    out = cfg.arm_dir(arm)
    arm_config = cfg.arm_config(arm)
    checkpoint = cfg.arm_checkpoint(arm)
    train_log, train_err = cfg.arm_train_logs(arm)
    scores_file = cfg.arm_scores_file()
    identity = cfg.identity_dir / f"{arm}.json"
    morph = out / "morph.txt"
    offtraj_dir = out / "offtraj"
    compare_dir = out / "compare"
    scores = out / "scores.txt"
    frozen = out / CONFIG_FROZEN_NAME
    config_as_run = out / CONFIG_AS_RUN_NAME

    def gate() -> None:
        # Every consumer of latest.pt refuses anything but a verified
        # TRAINING_COMPLETE; a 5k leftover must never be scored as a 20k arm.
        ctx.require_arm_training_complete(arm)

    def ensure_frozen() -> None:
        # run_arm freezes first; a direct run_steps caller gets the same guarantee.
        if not arm_config.exists():
            raise StepFailed(f"arm config missing: {arm_config}")
        try:
            freeze_arm_config(arm_config, frozen, legacy_record=config_as_run)
        except ConfigFrozenError as error:
            raise StepFailed(str(error)) from error

    def train() -> None:
        ensure_frozen()
        job = ctx.arm_job(arm)
        holder = job.get("pid")
        if job.state == STATE_RUNNING and isinstance(holder, int) and holder != os.getpid() and _pid_alive(holder):
            raise StepFailed(f"arm {arm} is already being trained by pid {holder}")
        argv = ctx.python_tool("train_gsplat.py", "--config", arm_config)
        with ctx.gpu_lease(f"train {arm}", argv, scan_trainers=True):
            started_at = time.time()
            job.set(
                STATE_RUNNING,
                "trainer launched",
                started_at=started_at,
                started_at_text=_timestamp(),
                pid=os.getpid(),
                config_sha256=file_sha256(frozen),
                config_frozen=str(frozen),
                training=None,
            )
            _append_text(scores_file, f"[{arm}] train start {_timestamp()}\n")
            code = ctx.run(argv, log=train_log, stderr_log=train_err)
        _append_text(train_log, f"EXIT {code}\n")
        verdict = ctx.verify_arm_training(arm, exit_code=code, job_started_at=started_at)
        if verdict.checkpoint.loadable:
            job.set(STATE_CHECKPOINTED, f"trainer exit {code}; checkpoint loadable at step {verdict.checkpoint.step}")
        job.set(verdict.state, verdict.reason, training=verdict.record(), exit_code=code)
        if not verdict.complete:
            marker = "TRAIN_PAUSED" if verdict.state == STATE_CONTROLLED_PAUSE else "TRAIN_FAILED"
            _append_text(scores_file, f"[{arm}] {marker} exit {code} {_timestamp()}: {verdict.reason}\n")
            raise StepFailed(f"{verdict.state}: {verdict.reason}")
        _append_text(scores_file, f"[{arm}] train exit {code} done {_timestamp()} ({verdict.reason})\n")

    def prune_done() -> bool:
        return not any(checkpoint.parent.glob("step_*.pt"))

    def prune() -> None:
        for path in checkpoint.parent.glob("step_*.pt"):
            path.unlink()

    def copy_config() -> None:
        # config_as_run.json is the historical record of what trained; it is
        # written once from the frozen snapshot and never overwritten.
        ensure_frozen()
        if config_as_run.exists() and not _same_content(frozen, config_as_run):
            raise StepFailed(f"{config_as_run} differs from the frozen config; refusing to overwrite the run record")
        _copy_atomic(frozen, config_as_run)

    def morph_run() -> None:
        gate()
        argv = ctx.python_tool("checkpoint_morphology.py", checkpoint, "--label", arm)
        with ctx.gpu_lease(f"morph {arm}", argv):
            ctx.run_capture_or_fail(argv, capture=morph, log=out / "morph.log")

    def offtraj() -> None:
        gate()
        argv = ctx.python_tool(
            "build_offtrajectory_compare.py", arm_config, checkpoint, offtraj_dir, cfg.compare_frames,
            "--reference-ply", cfg.reference_ply, "--reference-alignment", cfg.reference_alignment,
        )
        with ctx.gpu_lease(f"offtraj {arm}", argv):
            ctx.run_or_fail(argv, log=out / "offtraj.log")

    def compare() -> None:
        gate()
        argv = ctx.python_tool(
            "build_three_way_compare.py", "--config", arm_config, "--checkpoint", checkpoint,
            "--reference-ply", cfg.reference_ply, "--reference-alignment", cfg.reference_alignment,
            "--output", compare_dir, "--frames", cfg.compare_frames,
        )
        with ctx.gpu_lease(f"compare {arm}", argv):
            ctx.run_or_fail(argv, log=out / "compare.log")

    def freeze() -> None:
        gate()
        ctx.run_or_fail(
            ctx.python_tool("freeze_run_identity.py", "--run", out, "--output", identity),
            log=out / "identity.log",
        )

    def score() -> None:
        gate()
        log = out / "scores.log"
        ctx.run_capture_or_fail(ctx.python_tool("score_compare_sharpness.py", compare_dir), capture=scores, log=log)
        ctx.run_capture_or_fail(
            ctx.python_tool("score_offtrajectory_strips.py", f"{arm}={offtraj_dir}"), capture=scores, log=log, append=True
        )
        _append_text(scores, morph.read_text(encoding="utf-8"))
        _append_text(scores_file, f"[{arm}] scores {_timestamp()}\n" + scores.read_text(encoding="utf-8"))
        ctx.arm_job(arm).set(STATE_EVALUATED, "strips scored")

    return [
        Step("train", train, done=lambda: ctx.arm_training_complete(arm)),
        Step("prune_step_checkpoints", prune, done=prune_done, anchor=False),
        Step(
            "config_as_run",
            copy_config,
            artifacts=(config_as_run,),
            done=lambda: _same_content(frozen, config_as_run),
            anchor=False,
        ),
        Step("morph", morph_run, artifacts=(morph,)),
        Step("offtraj", offtraj, artifacts=(offtraj_dir / "offtraj_summary.json",)),
        Step("compare", compare, artifacts=(compare_dir / "compare_summary.json",)),
        Step("identity", freeze, artifacts=(identity,)),
        Step("scores", score, artifacts=(scores,)),
    ]


def run_arm(ctx: PipelineContext, arm: str, *, force: bool = False) -> int:
    """Train and score one arm; 0 when every step is done, 2 when refused."""
    cfg = ctx.config
    out = cfg.arm_dir(arm)
    status_file = cfg.run_root / f"{arm}.pipeline_status.txt"

    def status(line: str) -> None:
        ctx.status(f"[{arm}] {line}", status_file)

    status(f"arm start (resume={'off' if force else 'on'})")
    if cfg.arm_config(arm).exists():
        try:
            freeze_arm_config(cfg.arm_config(arm), out / CONFIG_FROZEN_NAME, legacy_record=out / CONFIG_AS_RUN_NAME)
        except ConfigFrozenError as error:
            status(f"ARM_REFUSED: {error}")
            return 2
    reports = run_steps(arm_steps(ctx, arm), force=force, status=status)
    failed = [report for report in reports if report.action == "failed"]
    job = ctx.arm_job(arm)
    if failed:
        # The train step writes its own verdict (FAILED / CONTROLLED_PAUSE);
        # any other failure is recorded without touching the training record.
        if out.is_dir() and not (failed[0].name == "train" and job.state in (STATE_FAILED, STATE_CONTROLLED_PAUSE)):
            job.fail(f"{failed[0].name}: {failed[0].detail}")
        status(f"ARM_FAILED at {failed[0].name}")
        return 1
    if job.training_verified() and job.state not in (STATE_EVALUATED, STATE_QUALITY_ACCEPTED, STATE_PUBLISHED):
        job.set(STATE_EVALUATED, "every arm step done")
    status("ARM_DONE")
    return 0


# --------------------------------------------------------------------------
# Delivery pipeline
# --------------------------------------------------------------------------


def _stamp_matches(record: dict[str, Any] | None, path: Path) -> bool:
    """Does a recorded (bytes, mtime, sha256) still describe ``path``?

    Size and mtime are checked first; only when they moved is the sha256
    recomputed, so a resume check does not hash gigabytes every time.
    """
    if not isinstance(record, dict) or not path.is_file():
        return False
    stamp = _file_stamp(path)
    if record.get("bytes") == stamp["bytes"] and record.get("mtime") == stamp["mtime"]:
        return True
    return bool(record.get("sha256")) and file_sha256(path) == record.get("sha256")


def _ply_record(path: Path) -> dict[str, Any]:
    stamp = _file_stamp(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": stamp["bytes"],
        "mtime": stamp["mtime"],
        "vertex_count": read_ply_vertex_count(path),
    }


def deliver_steps(
    ctx: PipelineContext,
    tag: str,
    tile0_arm: str,
    *,
    publish: bool = False,
    score_threshold_variants: bool = False,
) -> list[Step]:
    cfg = ctx.config
    out = cfg.delivery_dir(tag)
    log = out / "deliver_status.txt"
    merged = out / "merged.pt"
    report = out / "merge_report.json"
    body_ply = out / cfg.delivery_ply_name(tag)
    # Pre-export scores (merged.pt) keep the historical names so earlier
    # deliveries remain comparable as baselines; final scores are suffixed.
    morph = out / "morph.txt"
    battery = out / "battery.json"
    compare_dir = out / "compare_matched"
    offtraj_dir = out / "offtraj_matched"
    scores_pre = out / "scores_pre_export.txt"
    reimported = out / "reimported.pt"
    reimport_record = out / "reimported.json"
    threshold_dir = out / "threshold_control"
    threshold_record = threshold_dir / "threshold_control.json"
    final_morph = out / "morph_final.txt"
    final_battery = out / "battery_final.json"
    final_compare_dir = out / "compare_final"
    final_offtraj_dir = out / "offtraj_final"
    identity = cfg.identity_dir / f"delivery_{tag}_merged.json"
    scores = out / "scores.txt"
    delivery_report = out / "delivery_report.json"
    publish_dir = cfg.exports_dir if publish else cfg.candidate_exports_dir(tag)
    export_ply = publish_dir / cfg.delivery_ply_name(tag)
    export_sky = publish_dir / cfg.delivery_sky_name(tag)
    tile_arms = {tile: cfg.delivery_tile_arm(tag, tile) for tile in cfg.delivery_tiles}
    steps: list[Step] = []

    def job() -> JobState:
        return ctx.delivery_job(tag)

    def read_report() -> dict[str, Any]:
        if delivery_report.is_file():
            try:
                payload = _read_json(delivery_report)
                if isinstance(payload, dict):
                    return payload
            except ValueError:
                pass
        return {"schema_version": 1, "tag": tag, "tile0_arm": tile0_arm, "tile_arms": {str(t): a for t, a in tile_arms.items()}}

    def make_tile_step(tile: int, arm: str) -> Step:
        def train_tile() -> None:
            _append_text(log, f"[train] tile{tile} {_timestamp()}\n")
            code = run_arm(ctx, arm)
            if code != 0 or not ctx.arm_training_complete(arm):
                _append_text(log, f"[FAIL] tile{tile} training\n")
                verdict = ctx.arm_job(arm)
                raise StepFailed(f"tile{tile} arm {arm} exit {code}; training is {verdict.state} ({verdict.reason})")

        return Step(f"train_tile{tile}", train_tile, done=lambda: ctx.arm_training_complete(arm), independent=True)

    for tile, arm in tile_arms.items():
        steps.append(make_tile_step(tile, arm))

    def require_merged() -> None:
        if not job().training_verified():
            raise StepFailed(f"delivery {tag} has no verified merge; refusing to score")

    def merge() -> None:
        _append_text(log, f"[merge] {_timestamp()}\n")
        for arm in (tile0_arm, *tile_arms.values()):
            ctx.require_arm_training_complete(arm)
        argv = ctx.python_tool(
            "merge_v28_tile_checkpoints.py",
            "--tile-inputs", cfg.tile_inputs_manifest,
            "--tile-inputs-root", cfg.tile_inputs_root,
            "--tile-checkpoint", f"0={cfg.arm_checkpoint(tile0_arm)}",
        )
        for tile, arm in tile_arms.items():
            argv += ["--tile-checkpoint", f"{tile}={cfg.arm_checkpoint(arm)}"]
        argv += [
            "--output-checkpoint", str(merged),
            "--output-report", str(report),
            "--merge-policy", cfg.merge_policy,
        ]
        if cfg.harmonize_exposure:
            argv.append("--harmonize-exposure")
        try:
            with ctx.gpu_lease(f"merge {tag}", argv):
                ctx.run_or_fail(argv, log=out / "merge.log")
        except StepFailed:
            _append_text(log, "[FAIL] merge\n")
            raise
        arms = {"tile0": tile0_arm, **{f"tile{tile}": arm for tile, arm in tile_arms.items()}}
        job().set(
            STATE_TRAINING_COMPLETE,
            "every tile verified TRAINING_COMPLETE and merged",
            training={
                "verified": True,
                "arms": {
                    name: {
                        "arm": arm,
                        "completed_steps": ctx.arm_job(arm).get("training", {}).get("completed_steps"),
                        "checkpoint": ctx.arm_job(arm).get("training", {}).get("checkpoint", {}).get("path"),
                    }
                    for name, arm in arms.items()
                },
                "merged": {"path": str(merged), **_file_stamp(merged)},
            },
        )

    def render_steps(label: str, checkpoint: Path, *, morph_out: Path, battery_out: Path, compare_out: Path, offtraj_out: Path, gate: Callable[[], None]) -> list[Step]:
        """morph / battery / three-way / off-trajectory for one checkpoint."""

        def morph_run() -> None:
            gate()
            argv = ctx.python_tool("checkpoint_morphology.py", checkpoint, "--label", f"{label}_{tag}")
            with ctx.gpu_lease(f"{label} morph {tag}", argv):
                ctx.run_capture_or_fail(argv, capture=morph_out, log=out / f"{morph_out.stem}.log")

        def battery_run() -> None:
            gate()
            argv = ctx.python_tool(
                "evaluate_probe_views.py", "--config", cfg.delivery_eval_config, "--checkpoint", checkpoint,
                "--views", cfg.battery_views, "--output", battery_out,
            )
            with ctx.gpu_lease(f"{label} battery {tag}", argv):
                ctx.run_or_fail(argv, log=out / f"{battery_out.stem}.log")

        def compare() -> None:
            gate()
            argv = ctx.python_tool(
                "build_three_way_compare.py", "--config", cfg.delivery_eval_config, "--checkpoint", checkpoint,
                "--reference-ply", cfg.reference_ply, "--reference-alignment", cfg.reference_alignment,
                "--output", compare_out, "--frames", cfg.compare_frames,
            )
            with ctx.gpu_lease(f"{label} compare {tag}", argv):
                ctx.run_or_fail(argv, log=out / f"{compare_out.name}.log")

        def offtraj() -> None:
            gate()
            argv = ctx.python_tool(
                "build_offtrajectory_compare.py", cfg.delivery_eval_config, checkpoint, offtraj_out, cfg.compare_frames,
                "--reference-ply", cfg.reference_ply, "--reference-alignment", cfg.reference_alignment,
            )
            with ctx.gpu_lease(f"{label} offtraj {tag}", argv):
                ctx.run_or_fail(argv, log=out / f"{offtraj_out.name}.log")

        prefix = "" if label == "pre_export" else "final_"
        return [
            Step(f"{prefix}morph", morph_run, artifacts=(morph_out,)),
            Step(f"{prefix}battery", battery_run, artifacts=(battery_out,)),
            Step(f"{prefix}compare_matched", compare, artifacts=(compare_out / "compare_summary.json",)),
            Step(f"{prefix}offtraj_matched", offtraj, artifacts=(offtraj_out / "offtraj_summary.json",)),
        ]

    def score_strips(capture: Path, compare_out: Path, offtraj_out: Path, morph_out: Path, score_log: Path) -> None:
        compare_dirs = [*cfg.delivery_baselines["compare"], compare_out]
        ctx.run_capture_or_fail(ctx.python_tool("score_compare_sharpness.py", *compare_dirs), capture=capture, log=score_log)
        pairs = [f"{name}={path}" for name, path in cfg.delivery_baselines["offtraj"].items()]
        pairs.append(f"{tag}={offtraj_out}")
        ctx.run_capture_or_fail(
            ctx.python_tool("score_offtrajectory_strips.py", *pairs), capture=capture, log=score_log, append=True
        )
        _append_text(capture, morph_out.read_text(encoding="utf-8"))

    def pre_export_scores() -> None:
        require_merged()
        _append_text(log, "[scores pre-export]\n")
        score_strips(scores_pre, compare_dir, offtraj_dir, morph, out / "scores_pre_export.log")
        payload = read_report()
        payload["pre_export"] = {
            "checkpoint": {"path": str(merged), **_file_stamp(merged)},
            "morph": str(morph),
            "battery": str(battery),
            "compare": str(compare_dir),
            "offtraj": str(offtraj_dir),
            "scores": str(scores_pre),
            "scored_at": _timestamp(),
        }
        _write_json_atomic(delivery_report, payload)
        job().set(STATE_EVALUATED, "merged.pt scored (pre-export record)")

    def export() -> None:
        require_merged()
        ctx.run_or_fail(
            ctx.python_tool(
                "export_gaussian_ply.py", "--checkpoint", merged, "--output", body_ply,
                "--min-opacity", cfg.export_min_opacity,
            ),
            log=out / "export.log",
        )

    def threshold_control() -> None:
        require_merged()
        threshold_dir.mkdir(parents=True, exist_ok=True)
        variants: list[dict[str, Any]] = []
        for threshold in EXPORT_THRESHOLD_CONTROL:
            variant = threshold_dir / f"body_min_opacity_{threshold:g}.ply"
            ctx.run_or_fail(
                ctx.python_tool(
                    "export_gaussian_ply.py", "--checkpoint", merged, "--output", variant, "--min-opacity", threshold
                ),
                log=threshold_dir / f"export_{threshold:g}.log",
            )
            record: dict[str, Any] = {"min_opacity": threshold, "path": str(variant), "vertex_count": read_ply_vertex_count(variant)}
            if score_threshold_variants:
                variant_checkpoint = threshold_dir / f"reimported_{threshold:g}.pt"
                variant_battery = threshold_dir / f"battery_{threshold:g}.json"
                ctx.run_or_fail(
                    ctx.python_tool("import_gaussian_ply.py", "--ply", variant, "--output", variant_checkpoint),
                    log=threshold_dir / f"import_{threshold:g}.log",
                )
                argv = ctx.python_tool(
                    "evaluate_probe_views.py", "--config", cfg.delivery_eval_config, "--checkpoint", variant_checkpoint,
                    "--views", cfg.battery_views, "--output", variant_battery,
                )
                with ctx.gpu_lease(f"threshold battery {tag} {threshold:g}", argv):
                    ctx.run_or_fail(argv, log=threshold_dir / f"battery_{threshold:g}.log")
                record["battery"] = str(variant_battery)
            variants.append(record)
        baseline = variants[0]["vertex_count"]
        for record in variants:
            record["removed_vs_zero"] = baseline - record["vertex_count"]
        _write_json_atomic(
            threshold_record,
            {
                "checkpoint": {"path": str(merged), **_file_stamp(merged)},
                "delivery_min_opacity": cfg.export_min_opacity,
                "variants_scored": score_threshold_variants,
                "variants": variants,
                "recorded_at": _timestamp(),
            },
        )
        _append_text(log, "[threshold-control] " + ", ".join(f"{v['min_opacity']:g}: -{v['removed_vs_zero']}" for v in variants) + "\n")

    def reimport_done() -> bool:
        if not (reimported.is_file() and reimport_record.is_file()):
            return False
        try:
            record = _read_json(reimport_record)
        except ValueError:
            return False
        return _stamp_matches(record.get("source"), body_ply)

    def reimport() -> None:
        require_merged()
        # The exported PLY is what the customer opens; scoring merged.pt
        # would grade a model the export threshold never touched.
        ctx.run_or_fail(
            ctx.python_tool("import_gaussian_ply.py", "--ply", body_ply, "--output", reimported),
            log=out / "reimport.log",
        )
        _write_json_atomic(reimport_record, {"source": _ply_record(body_ply), "checkpoint": str(reimported), "imported_at": _timestamp()})

    def require_reimported() -> None:
        require_merged()
        if not reimport_done():
            raise StepFailed(f"{reimported} does not match the current {body_ply.name}; re-import first")

    def final_scores_done() -> bool:
        if not (scores.is_file() and delivery_report.is_file()):
            return False
        final = read_report().get("final")
        return isinstance(final, dict) and _stamp_matches(final.get("ply"), body_ply)

    def final_scores() -> None:
        require_reimported()
        _append_text(log, "[scores]\n")
        score_strips(scores, final_compare_dir, final_offtraj_dir, final_morph, out / "scores.log")
        ply = _ply_record(body_ply)
        source = _read_json(reimport_record).get("source", {})
        if source.get("sha256") != ply["sha256"]:
            raise StepFailed(f"{body_ply.name} changed after re-import (sha {source.get('sha256', '?')[:12]} vs {ply['sha256'][:12]})")
        payload = read_report()
        payload["final"] = {
            "bound_to_ply_sha256": ply["sha256"],
            "ply": ply,
            "scored_checkpoint": str(reimported),
            "morph": str(final_morph),
            "battery": str(final_battery),
            "compare": str(final_compare_dir),
            "offtraj": str(final_offtraj_dir),
            "scores": str(scores),
            "scores_sha256": file_sha256(scores),
            "scored_at": _timestamp(),
        }
        payload["export"] = {"min_opacity": cfg.export_min_opacity, "threshold_control": str(threshold_record)}
        _write_json_atomic(delivery_report, payload)
        _append_text(log, scores.read_text(encoding="utf-8"))
        _append_text(log, f"[complete] {_timestamp()} ply sha256 {ply['sha256']}\n")
        job().set(STATE_QUALITY_ACCEPTED, f"final scores bound to PLY sha256 {ply['sha256'][:12]}")

    def publish_run() -> None:
        if not final_scores_done():
            raise StepFailed("final scores are not bound to the current PLY; refusing to publish")
        if not cfg.sky_ply.exists():
            raise StepFailed(f"sky PLY missing: {cfg.sky_ply}")
        _copy_atomic(body_ply, export_ply)
        _copy_atomic(cfg.sky_ply, export_sky)
        payload = read_report()
        payload["publish"] = {
            "mode": "published" if publish else "candidate",
            "dir": str(publish_dir),
            "body": str(export_ply),
            "sky": str(export_sky),
            "ply_sha256": payload["final"]["bound_to_ply_sha256"],
            "at": _timestamp(),
        }
        _write_json_atomic(delivery_report, payload)
        _append_text(log, f"[export] done -> {publish_dir}\n")
        if publish:
            job().set(STATE_PUBLISHED, f"published to {publish_dir}")
        else:
            job().update(candidate_dir=str(publish_dir))

    steps.append(Step("merge", merge, artifacts=(merged, report), done=lambda: merged.is_file() and report.is_file() and job().training_verified()))
    steps += render_steps("pre_export", merged, morph_out=morph, battery_out=battery, compare_out=compare_dir, offtraj_out=offtraj_dir, gate=require_merged)
    steps += [
        Step("pre_export_scores", pre_export_scores, artifacts=(scores_pre,)),
        Step("export", export, artifacts=(body_ply,)),
        Step("threshold_control", threshold_control, artifacts=(threshold_record,)),
        Step("reimport", reimport, artifacts=(reimported, reimport_record), done=reimport_done),
    ]
    steps += render_steps("final", reimported, morph_out=final_morph, battery_out=final_battery, compare_out=final_compare_dir, offtraj_out=final_offtraj_dir, gate=require_reimported)

    def freeze() -> None:
        require_reimported()
        ctx.run_or_fail(
            ctx.python_tool(
                "freeze_run_identity.py", "--checkpoint", merged, "--extra-file", body_ply, "--output", identity
            ),
            log=out / "identity.log",
        )

    steps += [
        Step("identity", freeze, artifacts=(identity,)),
        Step("scores", final_scores, artifacts=(scores, delivery_report), done=final_scores_done),
        Step("publish", publish_run, artifacts=(export_ply, export_sky)),
    ]
    return steps


def run_deliver(
    ctx: PipelineContext,
    tag: str,
    tile0_arm: str,
    *,
    force: bool = False,
    publish: bool = False,
    score_threshold_variants: bool = False,
) -> int:
    cfg = ctx.config
    if not ctx.arm_training_complete(tile0_arm):
        job = ctx.arm_job(tile0_arm)
        ctx.status(f"[delivery {tag}] FAILED: tile0 arm {tile0_arm} training is {job.state or 'unrecorded'} ({job.reason or 'no checkpoint'})")
        return 1
    out = cfg.delivery_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "deliver_status.txt"
    _append_text(log, f"[start] {tag} delivery {_timestamp()}\n")
    job = ctx.delivery_job(tag)
    if job.state is None:
        job.set(STATE_RUNNING, "delivery started", tile0_arm=tile0_arm, publish=publish)

    def status(line: str) -> None:
        ctx.status(f"[delivery {tag}] {line}", out / "pipeline_status.txt")

    steps = deliver_steps(ctx, tag, tile0_arm, publish=publish, score_threshold_variants=score_threshold_variants)
    reports = run_steps(steps, force=force, status=status)
    failed = [report for report in reports if report.action == "failed"]
    if failed:
        ctx.delivery_job(tag).fail(f"{failed[0].name}: {failed[0].detail}")
        status(f"DELIVERY_FAILED at {failed[0].name}")
        return 1
    job = ctx.delivery_job(tag)
    if publish and job.state != STATE_PUBLISHED:
        job.set(STATE_PUBLISHED, f"published to {cfg.exports_dir}")
    elif not publish and job.state not in (STATE_QUALITY_ACCEPTED, STATE_PUBLISHED):
        job.set(STATE_QUALITY_ACCEPTED, "candidate ready; pass --publish to copy into exports")
    status("DELIVERY_DONE" + ("" if publish else f" (candidate in {cfg.candidate_exports_dir(tag)}; --publish to release)"))
    return 0


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------


def _count_success_lines(queue_file: Path, arm: str) -> int:
    if not queue_file.exists():
        return 0
    needle = f"arm {arm} exit 0"
    return sum(1 for line in queue_file.read_text(encoding="utf-8").splitlines() if needle in line)


def wait_for_arm(
    ctx: PipelineContext,
    arm: str,
    *,
    poll_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Block until ``arm`` records a NEW successful exit in queue_status.txt.

    Counting the lines present at start is what stops an older attempt of the
    same arm from releasing the queue early.
    """
    queue_file = ctx.config.queue_status_file()
    baseline = _count_success_lines(queue_file, arm)
    ctx.status(f"[queue] waiting for arm {arm} (successful exits so far: {baseline})")
    while _count_success_lines(queue_file, arm) <= baseline:
        sleep(poll_seconds)
    while ctx.trainer_processes():
        ctx.status("[queue] trainer still running; waiting for the GPU")
        sleep(poll_seconds)


def run_queue(
    ctx: PipelineContext,
    arms: Sequence[str],
    *,
    force: bool = False,
    deliver: tuple[str, str] | None = None,
) -> int:
    """Run arms in order, one trainer at a time, recording each exit."""
    cfg = ctx.config
    queue_file = cfg.queue_status_file()

    def status(line: str, *extra: Path) -> None:
        ctx.status(f"[queue] {line}", queue_file, *extra)

    running = ctx.trainer_processes()
    if running:
        pid, command = running[0]
        status(f"REFUSED: trainer already running (pid {pid}: {command[:120]})")
        return 2
    status(f"start pid {os.getpid()}")
    exit_code = 0
    if deliver is not None:
        tag, tile0_arm = deliver
        status(f"delivery {tag} start")
        ctx.tee_files.append(cfg.run_root / f"delivery_{tag}.queue.log")
        try:
            code = run_deliver(ctx, tag, tile0_arm, force=force)
        finally:
            ctx.tee_files.pop()
        status(f"delivery {tag} exit {code}")
        exit_code = exit_code or code
    for arm in arms:
        if not force and ctx.arm_training_complete(arm):
            status(f"arm {arm} skip (training verified complete; use --force to re-run)")
            continue
        running = ctx.trainer_processes()
        if running:
            pid, command = running[0]
            status(f"arm {arm} REFUSED: trainer already running (pid {pid}: {command[:120]})")
            return 2
        status(f"arm {arm} start")
        ctx.tee_files.append(cfg.run_root / f"{arm}.queue.log")
        try:
            code = run_arm(ctx, arm, force=force)
        finally:
            ctx.tee_files.pop()
        status(f"arm {arm} exit {code}")
        exit_code = exit_code or code
    status("complete")
    return exit_code


# --------------------------------------------------------------------------
# Scoring existing runs
# --------------------------------------------------------------------------


def score_commands(ctx: PipelineContext, run_dirs: Sequence[Path]) -> list[list[str]]:
    """Commands that re-score finished run directories (arm or delivery)."""
    compare_dirs: list[Path] = []
    pairs: list[str] = []
    for run_dir in run_dirs:
        compare = next((run_dir / name for name in ("compare", "compare_matched") if (run_dir / name).is_dir()), None)
        offtraj = next((run_dir / name for name in ("offtraj", "offtraj_matched") if (run_dir / name).is_dir()), None)
        if compare is None and offtraj is None:
            raise PipelineError(f"{run_dir} has neither compare/ nor offtraj/ strips to score")
        if compare is not None:
            compare_dirs.append(compare)
        if offtraj is not None:
            pairs.append(f"{run_dir.name}={offtraj}")
    commands: list[list[str]] = []
    if compare_dirs:
        commands.append(ctx.python_tool("score_compare_sharpness.py", *compare_dirs))
    if pairs:
        commands.append(ctx.python_tool("score_offtrajectory_strips.py", *pairs))
    return commands


def run_score(ctx: PipelineContext, run_dirs: Sequence[Path], *, output: Path | None) -> int:
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise PipelineError(f"run directory not found: {run_dir}")
    capture = output or (ctx.config.run_root / "score_report.txt")
    first = True
    for argv in score_commands(ctx, run_dirs):
        code = ctx.run(argv, log=capture, append=not first)
        first = False
        if code != 0:
            ctx.status(f"[score] FAILED exit {code}: {' '.join(argv[1:3])}")
            return 1
    for run_dir in run_dirs:
        morph = run_dir / "morph.txt"
        if morph.exists():
            _append_text(capture, morph.read_text(encoding="utf-8"))
    print(capture.read_text(encoding="utf-8"), file=ctx.stream, end="")
    ctx.status(f"[score] wrote {capture}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path(DEFAULT_CONFIG_NAME),
        help=f"machine paths JSON (default ./{DEFAULT_CONFIG_NAME}; see {EXAMPLE_CONFIG})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    arm = sub.add_parser("arm", help="train one arm and produce its strips, identity and scores")
    arm.add_argument("name", help="arm name; RUN/<name>.json must exist")
    arm.add_argument("--force", action="store_true", help="redo every step even if its artifacts exist")
    arm.add_argument("--dry-run", action="store_true", help="print which steps would run and exit")

    deliver = sub.add_parser("deliver", help="four-tile delivery for a winning Tile_0 arm")
    deliver.add_argument("tag", help="delivery tag; outputs go to RUN/delivery_<tag>")
    deliver.add_argument("--tile0", required=True, metavar="ARM", help="Tile_0 arm whose latest.pt is merged")
    deliver.add_argument("--force", action="store_true", help="redo every step even if its artifacts exist")
    deliver.add_argument("--dry-run", action="store_true", help="print which steps would run and exit")
    deliver.add_argument(
        "--publish", action="store_true",
        help="copy the scored PLY into exports_dir; default lands in exports_dir/candidate_<TAG>/",
    )
    deliver.add_argument(
        "--score-threshold-variants", action="store_true",
        help="also re-import and run the battery on each export-threshold control PLY (GPU time)",
    )

    queue = sub.add_parser("queue", help="run arms one after another, never two trainers at once")
    queue.add_argument("arms", nargs="*", help="arm names in execution order")
    queue.add_argument("--force", action="store_true", help="run arms even if their training is verified complete")
    queue.add_argument("--after", metavar="ARM", help="wait until ARM records a new successful exit first")
    queue.add_argument("--poll-seconds", type=float, default=120.0, help="polling interval for --after")
    queue.add_argument(
        "--deliver", metavar="TAG=TILE0_ARM", help="run this delivery before the arms (deliver_then_arms)"
    )

    score = sub.add_parser("score", help="re-score finished run directories")
    score.add_argument("run_dirs", nargs="+", type=Path)
    score.add_argument("--output", type=Path, help="report file (default RUN/score_report.txt)")

    state = sub.add_parser("state", help="print the persisted job state of arms or deliveries")
    state.add_argument("names", nargs="+", help="arm names or delivery_<tag>")
    return parser


def run_state(ctx: PipelineContext, names: Sequence[str]) -> int:
    """Print state, reason and training verdict of each job; 1 if any is not trained."""
    worst = 0
    for name in names:
        if name.startswith("delivery_"):
            job = ctx.delivery_job(name[len("delivery_"):])
        else:
            job = ctx.arm_job(name)
        training = job.get("training") or {}
        print(
            f"{name}: {job.state or 'UNRECORDED'}"
            + (f" - {job.reason}" if job.reason else "")
            + (
                f" [steps {training.get('completed_steps')}/{training.get('target_steps')}, exit {training.get('exit', {}).get('kind')}]"
                if training.get("verdict")
                else ""
            ),
            file=ctx.stream,
        )
        if not job.training_verified():
            worst = 1
    return worst


def _print_plan(ctx: PipelineContext, steps: Sequence[Step], *, force: bool) -> None:
    for step, will_run in plan_steps(steps, force=force):
        print(f"  {'RUN ' if will_run else 'skip'} {step.name}", file=ctx.stream)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_pipeline_config(args.pipeline_config)
        ctx = PipelineContext(config)
        if args.command in ("arm", "deliver", "queue") and not getattr(args, "dry_run", False):
            # Fail before any training starts: a missing reference PLY would
            # otherwise surface hours later, after the GPU time is spent.
            missing = config.missing_paths()
            if missing:
                listing = "; ".join(f"{key}={path}" for key, path in missing)
                raise PipelineError(f"pipeline config paths do not exist on this host: {listing}")
        if args.command == "arm":
            if args.dry_run:
                _print_plan(ctx, arm_steps(ctx, args.name), force=args.force)
                return 0
            return run_arm(ctx, args.name, force=args.force)
        if args.command == "deliver":
            if args.dry_run:
                _print_plan(
                    ctx,
                    deliver_steps(ctx, args.tag, args.tile0, publish=args.publish, score_threshold_variants=args.score_threshold_variants),
                    force=args.force,
                )
                return 0
            return run_deliver(
                ctx, args.tag, args.tile0, force=args.force, publish=args.publish,
                score_threshold_variants=args.score_threshold_variants,
            )
        if args.command == "queue":
            deliver = None
            if args.deliver:
                if "=" not in args.deliver:
                    parser.error("--deliver expects TAG=TILE0_ARM")
                tag, _, tile0_arm = args.deliver.partition("=")
                deliver = (tag, tile0_arm)
            if not args.arms and deliver is None:
                parser.error("queue needs at least one arm or --deliver")
            if args.after:
                wait_for_arm(ctx, args.after, poll_seconds=args.poll_seconds)
            return run_queue(ctx, args.arms, force=args.force, deliver=deliver)
        if args.command == "score":
            return run_score(ctx, args.run_dirs, output=args.output)
        if args.command == "state":
            return run_state(ctx, args.names)
        parser.error(f"unknown command {args.command}")
    except PipelineError as error:
        print(f"pipeline: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
