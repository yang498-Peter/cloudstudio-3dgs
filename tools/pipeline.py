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

Training is never started while another trainer process holds the GPU: the
2026-09-07 double start came from two cmd queues matching a stale status
line, so the guard here scans live processes, not log files.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

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
    "env": {"PYTHONIOENCODING": "utf-8"},
}

KNOWN_KEYS = frozenset(
    ("schema_version",) + REQUIRED_PATH_KEYS + OPTIONAL_PATH_KEYS + tuple(DEFAULTS)
)


class PipelineError(RuntimeError):
    """A configuration or environment problem that makes the run pointless."""


class PipelineConfigError(PipelineError):
    """The pipeline config is missing or malformed; the message names the key."""


class StepFailed(RuntimeError):
    """One step failed; the arm stops here and the message says why."""


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_text_atomic(path: Path, text: str) -> None:
    """Write via a sibling temp file so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


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
    env: dict[str, str]

    # Derived locations -----------------------------------------------------

    def arm_config(self, arm: str) -> Path:
        return self.run_root / f"{arm}.json"

    def arm_dir(self, arm: str) -> Path:
        return self.run_root / arm

    def arm_checkpoint(self, arm: str) -> Path:
        return self.arm_dir(arm) / "checkpoints" / "latest.pt"

    def arm_scores_file(self) -> Path:
        return self.run_root / "arm_scores.txt"

    def queue_status_file(self) -> Path:
        return self.run_root / "queue_status.txt"

    def delivery_dir(self, tag: str) -> Path:
        return self.run_root / f"delivery_{tag}"

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
    for key in ("scene_tag", "merge_policy", "trainer_process_pattern"):
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
    ``trainer_processes`` returns the live trainer processes. Tests replace
    both so the resume and guard logic can be exercised without a GPU.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        run_command: RunCommand | None = None,
        trainer_processes: Callable[[], list[tuple[int, str]]] | None = None,
        stream=None,
    ) -> None:
        self.config = config
        self._run_command = run_command or _subprocess_run_command
        self._trainer_processes = trainer_processes
        self._env: dict[str, str] | None = None
        self.stream = stream or sys.stdout
        self.tee_files: list[Path] = []

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
    run_root = cfg.run_root
    out = cfg.arm_dir(arm)
    arm_config = cfg.arm_config(arm)
    checkpoint = cfg.arm_checkpoint(arm)
    train_log = run_root / f"{arm}.log"
    scores_file = cfg.arm_scores_file()
    identity = cfg.identity_dir / f"{arm}.json"
    morph = out / "morph.txt"
    offtraj_dir = out / "offtraj"
    compare_dir = out / "compare"
    scores = out / "scores.txt"

    def train() -> None:
        if not arm_config.exists():
            raise StepFailed(f"arm config missing: {arm_config}")
        ctx.ensure_gpu_free()
        _append_text(scores_file, f"[{arm}] train start {_timestamp()}\n")
        code = ctx.run(
            ctx.python_tool("train_gsplat.py", "--config", arm_config),
            log=train_log,
            stderr_log=run_root / f"{arm}.log.err",
        )
        _append_text(train_log, f"EXIT {code}\n")
        # A controlled stop exits non-zero yet leaves latest.pt; the checkpoint
        # is the success criterion, exactly as in run_arm.cmd.
        if not checkpoint.exists():
            _append_text(scores_file, f"[{arm}] TRAIN_FAILED exit {code} {_timestamp()}\n")
            raise StepFailed(f"trainer exit {code} and no {checkpoint}")
        _append_text(scores_file, f"[{arm}] train exit {code} done {_timestamp()}\n")

    def prune_done() -> bool:
        return not any(checkpoint.parent.glob("step_*.pt"))

    def prune() -> None:
        for path in checkpoint.parent.glob("step_*.pt"):
            path.unlink()

    def copy_config() -> None:
        _copy_atomic(arm_config, out / "config_as_run.json")

    def morph_run() -> None:
        ctx.run_capture_or_fail(
            ctx.python_tool("checkpoint_morphology.py", checkpoint, "--label", arm),
            capture=morph,
            log=out / "morph.log",
        )

    def offtraj() -> None:
        ctx.run_or_fail(
            ctx.python_tool(
                "build_offtrajectory_compare.py", arm_config, checkpoint, offtraj_dir, cfg.compare_frames,
                "--reference-ply", cfg.reference_ply, "--reference-alignment", cfg.reference_alignment,
            ),
            log=out / "offtraj.log",
        )

    def compare() -> None:
        ctx.run_or_fail(
            ctx.python_tool(
                "build_three_way_compare.py", "--config", arm_config, "--checkpoint", checkpoint,
                "--reference-ply", cfg.reference_ply, "--reference-alignment", cfg.reference_alignment,
                "--output", compare_dir, "--frames", cfg.compare_frames,
            ),
            log=out / "compare.log",
        )

    def freeze() -> None:
        ctx.run_or_fail(
            ctx.python_tool("freeze_run_identity.py", "--run", out, "--output", identity),
            log=out / "identity.log",
        )

    def score() -> None:
        log = out / "scores.log"
        ctx.run_capture_or_fail(ctx.python_tool("score_compare_sharpness.py", compare_dir), capture=scores, log=log)
        ctx.run_capture_or_fail(
            ctx.python_tool("score_offtrajectory_strips.py", f"{arm}={offtraj_dir}"), capture=scores, log=log, append=True
        )
        _append_text(scores, morph.read_text(encoding="utf-8"))
        _append_text(scores_file, f"[{arm}] scores {_timestamp()}\n" + scores.read_text(encoding="utf-8"))

    return [
        Step("train", train, artifacts=(checkpoint,)),
        Step("prune_step_checkpoints", prune, done=prune_done, anchor=False),
        Step(
            "config_as_run",
            copy_config,
            artifacts=(out / "config_as_run.json",),
            done=lambda: _same_content(arm_config, out / "config_as_run.json"),
            anchor=False,
        ),
        Step("morph", morph_run, artifacts=(morph,)),
        Step("offtraj", offtraj, artifacts=(offtraj_dir / "offtraj_summary.json",)),
        Step("compare", compare, artifacts=(compare_dir / "compare_summary.json",)),
        Step("identity", freeze, artifacts=(identity,)),
        Step("scores", score, artifacts=(scores,)),
    ]


def run_arm(ctx: PipelineContext, arm: str, *, force: bool = False) -> int:
    """Train and score one arm; returns 0 when every step is done."""
    status_file = ctx.config.run_root / f"{arm}.pipeline_status.txt"

    def status(line: str) -> None:
        ctx.status(f"[{arm}] {line}", status_file)

    status(f"arm start (resume={'off' if force else 'on'})")
    reports = run_steps(arm_steps(ctx, arm), force=force, status=status)
    failed = [report for report in reports if report.action == "failed"]
    if failed:
        status(f"ARM_FAILED at {failed[0].name}")
        return 1
    status("ARM_DONE")
    return 0


# --------------------------------------------------------------------------
# Delivery pipeline
# --------------------------------------------------------------------------


def deliver_steps(ctx: PipelineContext, tag: str, tile0_arm: str) -> list[Step]:
    cfg = ctx.config
    out = cfg.delivery_dir(tag)
    log = out / "deliver_status.txt"
    merged = out / "merged.pt"
    report = out / "merge_report.json"
    body_ply = out / cfg.delivery_ply_name(tag)
    export_ply = cfg.exports_dir / cfg.delivery_ply_name(tag)
    export_sky = cfg.exports_dir / cfg.delivery_sky_name(tag)
    morph = out / "morph.txt"
    battery = out / "battery.json"
    compare_dir = out / "compare_matched"
    offtraj_dir = out / "offtraj_matched"
    identity = cfg.identity_dir / f"delivery_{tag}_merged.json"
    scores = out / "scores.txt"
    tile_arms = {tile: cfg.delivery_tile_arm(tag, tile) for tile in cfg.delivery_tiles}
    steps: list[Step] = []

    def make_tile_step(tile: int, arm: str) -> Step:
        checkpoint = cfg.arm_checkpoint(arm)

        def train_tile() -> None:
            _append_text(log, f"[train] tile{tile} {_timestamp()}\n")
            if run_arm(ctx, arm) != 0 or not checkpoint.exists():
                _append_text(log, f"[FAIL] tile{tile} training\n")
                raise StepFailed(f"tile{tile} arm {arm} did not produce {checkpoint}")

        return Step(f"train_tile{tile}", train_tile, artifacts=(checkpoint,), independent=True)

    for tile, arm in tile_arms.items():
        steps.append(make_tile_step(tile, arm))

    def merge() -> None:
        _append_text(log, f"[merge] {_timestamp()}\n")
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
            ctx.run_or_fail(argv, log=out / "merge.log")
        except StepFailed:
            _append_text(log, "[FAIL] merge\n")
            raise

    def export() -> None:
        ctx.run_or_fail(
            ctx.python_tool(
                "export_gaussian_ply.py", "--checkpoint", merged, "--output", body_ply,
                "--min-opacity", cfg.export_min_opacity,
            ),
            log=out / "export.log",
        )

    def publish() -> None:
        if not cfg.sky_ply.exists():
            raise StepFailed(f"sky PLY missing: {cfg.sky_ply}")
        _copy_atomic(body_ply, export_ply)
        _copy_atomic(cfg.sky_ply, export_sky)
        _append_text(log, "[export] done\n")

    def morph_run() -> None:
        ctx.run_capture_or_fail(
            ctx.python_tool("checkpoint_morphology.py", merged, "--label", f"merged_{tag}"),
            capture=morph,
            log=out / "morph.log",
        )

    def battery_run() -> None:
        ctx.run_or_fail(
            ctx.python_tool(
                "evaluate_probe_views.py", "--config", cfg.delivery_eval_config, "--checkpoint", merged,
                "--views", cfg.battery_views, "--output", battery,
            ),
            log=out / "battery.log",
        )

    def compare() -> None:
        ctx.run_or_fail(
            ctx.python_tool(
                "build_three_way_compare.py", "--config", cfg.delivery_eval_config, "--checkpoint", merged,
                "--reference-ply", cfg.reference_ply, "--reference-alignment", cfg.reference_alignment,
                "--output", compare_dir, "--frames", cfg.compare_frames,
            ),
            log=out / "compare_matched.log",
        )

    def offtraj() -> None:
        ctx.run_or_fail(
            ctx.python_tool(
                "build_offtrajectory_compare.py", cfg.delivery_eval_config, merged, offtraj_dir, cfg.compare_frames,
                "--reference-ply", cfg.reference_ply, "--reference-alignment", cfg.reference_alignment,
            ),
            log=out / "offtraj_matched.log",
        )

    def freeze() -> None:
        ctx.run_or_fail(
            ctx.python_tool(
                "freeze_run_identity.py", "--checkpoint", merged, "--extra-file", export_ply, "--output", identity
            ),
            log=out / "identity.log",
        )

    def score() -> None:
        _append_text(log, "[scores]\n")
        score_log = out / "scores.log"
        compare_dirs = [*cfg.delivery_baselines["compare"], compare_dir]
        ctx.run_capture_or_fail(
            ctx.python_tool("score_compare_sharpness.py", *compare_dirs), capture=scores, log=score_log
        )
        pairs = [f"{name}={path}" for name, path in cfg.delivery_baselines["offtraj"].items()]
        pairs.append(f"{tag}={offtraj_dir}")
        ctx.run_capture_or_fail(
            ctx.python_tool("score_offtrajectory_strips.py", *pairs), capture=scores, log=score_log, append=True
        )
        _append_text(scores, morph.read_text(encoding="utf-8"))
        _append_text(log, scores.read_text(encoding="utf-8"))
        _append_text(log, f"[complete] {_timestamp()}\n")

    steps += [
        Step("merge", merge, artifacts=(merged, report)),
        Step("export", export, artifacts=(body_ply,)),
        Step("publish", publish, artifacts=(export_ply, export_sky)),
        Step("morph", morph_run, artifacts=(morph,)),
        Step("battery", battery_run, artifacts=(battery,)),
        Step("compare_matched", compare, artifacts=(compare_dir / "compare_summary.json",)),
        Step("offtraj_matched", offtraj, artifacts=(offtraj_dir / "offtraj_summary.json",)),
        Step("identity", freeze, artifacts=(identity,)),
        Step("scores", score, artifacts=(scores,)),
    ]
    return steps


def run_deliver(ctx: PipelineContext, tag: str, tile0_arm: str, *, force: bool = False) -> int:
    cfg = ctx.config
    if not cfg.arm_checkpoint(tile0_arm).exists():
        ctx.status(f"[delivery {tag}] FAILED: tile0 arm checkpoint missing: {cfg.arm_checkpoint(tile0_arm)}")
        return 1
    out = cfg.delivery_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "deliver_status.txt"
    _append_text(log, f"[start] {tag} delivery {_timestamp()}\n")

    def status(line: str) -> None:
        ctx.status(f"[delivery {tag}] {line}", out / "pipeline_status.txt")

    reports = run_steps(deliver_steps(ctx, tag, tile0_arm), force=force, status=status)
    failed = [report for report in reports if report.action == "failed"]
    if failed:
        status(f"DELIVERY_FAILED at {failed[0].name}")
        return 1
    status("DELIVERY_DONE")
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
        if not force and cfg.arm_checkpoint(arm).exists():
            status(f"arm {arm} skip (latest.pt exists; use --force to re-run)")
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

    queue = sub.add_parser("queue", help="run arms one after another, never two trainers at once")
    queue.add_argument("arms", nargs="*", help="arm names in execution order")
    queue.add_argument("--force", action="store_true", help="run arms even if their latest.pt exists")
    queue.add_argument("--after", metavar="ARM", help="wait until ARM records a new successful exit first")
    queue.add_argument("--poll-seconds", type=float, default=120.0, help="polling interval for --after")
    queue.add_argument(
        "--deliver", metavar="TAG=TILE0_ARM", help="run this delivery before the arms (deliver_then_arms)"
    )

    score = sub.add_parser("score", help="re-score finished run directories")
    score.add_argument("run_dirs", nargs="+", type=Path)
    score.add_argument("--output", type=Path, help="report file (default RUN/score_report.txt)")
    return parser


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
                _print_plan(ctx, deliver_steps(ctx, args.tag, args.tile0), force=args.force)
                return 0
            return run_deliver(ctx, args.tag, args.tile0, force=args.force)
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
        parser.error(f"unknown command {args.command}")
    except PipelineError as error:
        print(f"pipeline: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
