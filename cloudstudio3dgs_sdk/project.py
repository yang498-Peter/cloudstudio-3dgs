"""The four-stage project: prepare, train, deliver, report.

Each stage is idempotent, records a state sidecar next to the work root and
can be resumed. The sidecars mirror ``tools/pipeline.py``'s job-state
conventions (schema version, state, reason, append-only history, atomic
write) because the same operator reads both.

Fail-closed is the house rule and it is implemented as a chain: every stage
records a digest of each input it consumed and each output it produced; the
next stage recomputes the digests of the upstream outputs it depends on and
refuses when one moved. A stage also refuses when the profile or the plan it
resumes into is not the one it started under. Refusing costs a message;
continuing costs a delivery scored against a model nobody can reconstruct.

Digests: files at or under :data:`SHA_MAX_BYTES` get a real sha256. Bigger
ones (checkpoints, merged PLYs) get size+mtime, exactly as
``tools/pipeline.py`` does, and the record says which kind was used so nobody
reads a stamp as a hash.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.pipeline import _timestamp, _write_json_atomic, file_sha256

from cloudstudio3dgs_sdk.bundle import PreparedScene, load_dataset_bundle
from cloudstudio3dgs_sdk.plan import (
    STAGES,
    DatasetSummary,
    Plan,
    PlannedStep,
    WorkLayout,
    build_plan,
    coarse_config,
)
from cloudstudio3dgs_sdk.profile import Profile
from cloudstudio3dgs_sdk.requirements import PreflightReport, Probes, preflight

SHA_MAX_BYTES = 256 * 1024 * 1024

STAGE_STATE_VERSION = 1
PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
STAGE_STATES = (PENDING, RUNNING, COMPLETE, FAILED)

# Which stage's outputs each stage consumes. prepare consumes the dataset.
STAGE_DEPENDS_ON: Mapping[str, str | None] = {
    "prepare": None,
    "train": "prepare",
    "deliver": "train",
    "report": "deliver",
}

# Stages that cost GPU time and therefore run the preflight first.
GPU_STAGES = ("train", "deliver")


class StageRefused(RuntimeError):
    """A stage declined to run. Nothing was started."""


class StageFailed(RuntimeError):
    """A step inside a stage failed."""


def digest(path: Path, *, max_bytes: int = SHA_MAX_BYTES) -> dict[str, Any]:
    """A comparable record of one file's current content."""
    target = Path(path)
    stat = target.stat()
    record: dict[str, Any] = {"path": str(target), "bytes": stat.st_size, "mtime": stat.st_mtime}
    if stat.st_size <= max_bytes:
        record["sha256"] = file_sha256(target)
        record["digest_kind"] = "sha256"
    else:
        record["digest_kind"] = "size_mtime"
    return record


def digest_matches(recorded: Mapping[str, Any] | None, *, max_bytes: int = SHA_MAX_BYTES) -> tuple[bool, str]:
    """Does the file the record names still look the way the record says?"""
    if not isinstance(recorded, Mapping) or "path" not in recorded:
        return False, "no digest recorded"
    target = Path(str(recorded["path"]))
    if not target.is_file():
        return False, f"missing {target}"
    current = digest(target, max_bytes=max_bytes)
    if recorded.get("digest_kind") == "sha256":
        if current.get("sha256") != recorded.get("sha256"):
            return False, (
                f"{target.name} sha256 {str(current.get('sha256'))[:12]} != recorded "
                f"{str(recorded.get('sha256'))[:12]}"
            )
        return True, ""
    if int(current["bytes"]) != int(recorded.get("bytes", -1)):
        return False, f"{target.name} is {current['bytes']} bytes, recorded {recorded.get('bytes')}"
    if abs(float(current["mtime"]) - float(recorded.get("mtime", 0.0))) > 2.0:
        return False, f"{target.name} was rewritten (mtime moved)"
    return True, ""


class StageState:
    """``<work>/sdk_state/stage_<name>.json``. Same shape as a pipeline job."""

    def __init__(self, path: Path, *, stage: str) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "schema_version": STAGE_STATE_VERSION,
            "job": "sdk_stage",
            "name": stage,
            "state": None,
            "reason": "",
            "history": [],
            "inputs": {},
            "outputs": {},
        }
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict) and loaded.get("schema_version") == STAGE_STATE_VERSION:
                self.data.update(loaded)

    @property
    def state(self) -> str | None:
        return self.data.get("state")

    @property
    def reason(self) -> str:
        return str(self.data.get("reason", ""))

    @property
    def outputs(self) -> Mapping[str, Any]:
        return self.data.get("outputs") or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def save(self) -> None:
        self.data["updated_at"] = _timestamp()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.path, self.data)

    def set(self, state: str, reason: str = "", **fields: Any) -> None:
        if state not in STAGE_STATES:
            raise ValueError(f"unknown stage state {state!r}")
        self.data.update(fields)
        self.data["state"] = state
        self.data["reason"] = reason
        self.data.setdefault("history", []).append({"state": state, "at": _timestamp(), "reason": reason})
        self.save()


@dataclass
class StageResult:
    stage: str
    action: str  # "ran" | "skipped" | "planned" | "refused" | "failed"
    reason: str = ""
    steps_run: tuple[str, ...] = ()
    steps_skipped: tuple[str, ...] = ()
    preflight: PreflightReport | None = None

    @property
    def ok(self) -> bool:
        return self.action in ("ran", "skipped", "planned")

    def as_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "steps_run": list(self.steps_run),
            "steps_skipped": list(self.steps_skipped),
        }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class SubprocessRunner:
    """Runs a planned step's argv with its output teed to a log file."""

    def __init__(self, *, repo_root: Path, env: Mapping[str, str] | None = None) -> None:
        self.repo_root = Path(repo_root)
        self.env = dict(env or {})

    def __call__(self, step: PlannedStep, *, log: Path) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self.repo_root), environment.get("PYTHONPATH", "")) if part
        )
        environment.update(self.env)
        with log.open("ab") as handle:
            handle.write(f"\n$ {' '.join(step.command)}\n".encode("utf-8"))
            handle.flush()
            return subprocess.run(
                list(step.command),
                cwd=str(self.repo_root),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode


# --------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------


class Project:
    """One scene, one profile, one work root.

    >>> project = Project(dataset_root, work_root, profile)     # doctest: +SKIP
    >>> project.run_all(dry_run=True)                           # doctest: +SKIP
    """

    def __init__(
        self,
        dataset_root: Path | str,
        work_root: Path | str,
        profile: Profile,
        *,
        repo_root: Path | str | None = None,
        python: Path | str | None = None,
        scene_tag: str | None = None,
        delivery_tag: str | None = None,
        runner: Callable[..., int] | None = None,
        probes: Probes | None = None,
        dataset: DatasetSummary | None = None,
        prior_tile_checkpoints: Mapping[int, str] | None = None,
        vram_gib: float | None = None,
        stream: Any = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.work_root = Path(work_root)
        self.profile = profile
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
        self.python = Path(python) if python else Path(sys.executable)
        self.layout = WorkLayout(self.work_root)
        self.scene_tag = scene_tag
        self.delivery_tag = delivery_tag
        self.runner = runner or SubprocessRunner(repo_root=self.repo_root)
        self.probes = probes
        self.vram_gib = vram_gib
        self._prior_override = dict(prior_tile_checkpoints or {})
        self.stream = stream or sys.stdout
        self._dataset = dataset
        self._plan: Plan | None = None

    # -- plumbing --------------------------------------------------------

    def say(self, line: str) -> None:
        print(line, file=self.stream)

    def stage_state(self, stage: str) -> StageState:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}")
        return StageState(self.layout.state / f"stage_{stage}.json", stage=stage)

    def dataset_summary(self) -> DatasetSummary:
        """The prepared scene. Read from the prepare manifest unless injected."""
        if self._dataset is not None:
            return self._dataset
        manifest = self.layout.prepare_manifest
        if not manifest.is_file():
            raise StageRefused(
                f"no prepared dataset: {manifest} does not exist. Run prepare() first, or pass "
                "dataset=DatasetSummary(...) to plan a scene before ingestion."
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self._dataset = DatasetSummary.from_json(payload["dataset"])
        return self._dataset

    def bundle_paths(self) -> dict[str, str]:
        manifest = self.layout.prepare_manifest
        if not manifest.is_file():
            return {}
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return dict(payload.get("trainer_paths") or {})

    def prior_tile_checkpoints(self) -> dict[int, str]:
        """Previous-generation checkpoints the stand-in backdrops may render.

        The caller's override wins: a re-delivery names them on the command
        line before any prepare manifest exists.
        """
        if self._prior_override:
            return dict(self._prior_override)
        manifest = self.layout.prepare_manifest
        if not manifest.is_file():
            return {}
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return {int(key): str(value) for key, value in (payload.get("prior_tile_checkpoints") or {}).items()}

    def plan(self, *, refresh: bool = False, stages: Sequence[str] = STAGES) -> Plan:
        """The plan for this project. ``stages`` narrows it without running."""
        if tuple(stages) != tuple(STAGES):
            # A narrowed plan is a different object; never cache it as *the* plan.
            return self._build_plan(stages)
        if self._plan is None or refresh:
            self._plan = self._build_plan(STAGES)
        return self._plan

    def _build_plan(self, stages: Sequence[str]) -> Plan:
        return build_plan(
            self.profile,
            self.dataset_summary(),
            dataset_root=self.dataset_root,
            work_root=self.work_root,
            repo_root=self.repo_root,
            python=self.python,
            bundle_paths=self.bundle_paths(),
            prior_tile_checkpoints=self.prior_tile_checkpoints(),
            vram_gib=self.vram_gib,
            delivery_tag=self.delivery_tag,
            stages=stages,
        )

    def preflight(self, *, require_gpu: bool = True) -> PreflightReport:
        return preflight(
            self.plan(),
            self.profile,
            repo_root=self.repo_root,
            probes=self.probes,
            require_gpu=require_gpu,
        )

    # -- the fail-closed chain -------------------------------------------

    def _verify_upstream(self, stage: str) -> None:
        """Refuse when the stage this one consumes moved under it."""
        upstream = STAGE_DEPENDS_ON[stage]
        if upstream is None:
            return
        state = self.stage_state(upstream)
        if state.state != COMPLETE:
            raise StageRefused(
                f"{stage} needs {upstream} COMPLETE, found {state.state or 'nothing'}"
                + (f" ({state.reason})" if state.reason else "")
            )
        for name, record in sorted(state.outputs.items()):
            ok, detail = digest_matches(record)
            if not ok:
                raise StageRefused(
                    f"{stage} refuses to run: output '{name}' of {upstream} changed since it was "
                    f"recorded - {detail}. Re-run {upstream} (--force) or restore the artefact."
                )
        recorded_profile = state.get("profile_sha256")
        if recorded_profile and recorded_profile != self.profile.profile_sha256:
            raise StageRefused(
                f"{stage} refuses to run: {upstream} ran under profile sha256 "
                f"{recorded_profile[:12]}, this process carries {self.profile.profile_sha256[:12]}. "
                "A changed profile is a different recipe; start a new work root."
            )

    def _stage_is_current(self, stage: str) -> tuple[bool, str]:
        """COMPLETE and every recorded output still matches? Then skip it."""
        state = self.stage_state(stage)
        if state.state != COMPLETE:
            return False, state.reason or "not complete"
        if state.get("profile_sha256") != self.profile.profile_sha256:
            return False, "recorded under a different profile sha256"
        for name, record in sorted(state.outputs.items()):
            ok, detail = digest_matches(record)
            if not ok:
                return False, f"output '{name}' no longer matches: {detail}"
        return True, "every recorded output still matches"

    def _record(self, stage: str, plan: Plan, steps: Sequence[PlannedStep], *, ran: Sequence[str], skipped: Sequence[str]) -> None:
        outputs: dict[str, dict[str, Any]] = {}
        for step in steps:
            for output in step.outputs:
                path = Path(output)
                if path.is_file():
                    outputs[f"{step.name}:{path.name}"] = digest(path)
        state = self.stage_state(stage)
        state.set(
            COMPLETE,
            f"{len(ran)} step(s) run, {len(skipped)} already done",
            outputs=outputs,
            profile_sha256=self.profile.profile_sha256,
            profile=self.profile.name,
            profile_version=self.profile.version,
            plan_sha256=plan.plan_sha256,
            steps_run=list(ran),
            steps_skipped=list(skipped),
        )

    # -- stage driver -----------------------------------------------------

    def _run_stage(self, stage: str, *, force: bool, dry_run: bool) -> StageResult:
        if dry_run:
            plan = self.plan()
            self.say(plan.render())
            return StageResult(stage, "planned", "dry run")
        if not force:
            current, why = self._stage_is_current(stage)
            if current:
                self.say(f"[{stage}] skip ({why})")
                return StageResult(stage, "skipped", why)
        self._verify_upstream(stage)
        plan = self.plan(refresh=True)
        report: PreflightReport | None = None
        if stage in GPU_STAGES:
            report = self.preflight(require_gpu=True)
            report.raise_for_status()
        steps = plan.stage_steps(stage)
        state = self.stage_state(stage)
        state.set(RUNNING, f"{len(steps)} step(s) planned", plan_sha256=plan.plan_sha256,
                  profile_sha256=self.profile.profile_sha256)
        ran: list[str] = []
        skipped: list[str] = []
        for step in steps:
            if step.blocking:
                state.set(FAILED, f"{step.name}: {step.blocking}")
                raise StageRefused(f"[{stage}] {step.name} cannot run: {step.blocking}")
            if not force and step.outputs and all(Path(output).exists() for output in step.outputs):
                self.say(f"[{stage}] skip {step.name} (outputs present)")
                skipped.append(step.name)
                continue
            self.say(f"[{stage}] run {step.name}")
            try:
                self._execute(step, plan)
            except Exception as error:
                state.set(FAILED, f"{step.name}: {error}")
                raise
            ran.append(step.name)
        self._record(stage, plan, steps, ran=ran, skipped=skipped)
        return StageResult(stage, "ran", "", tuple(ran), tuple(skipped), report)

    def _execute(self, step: PlannedStep, plan: Plan) -> None:
        native = getattr(self, f"_native_{step.name}", None)
        if native is not None:
            native(step, plan)
            return
        if not step.command:
            raise StageFailed(f"step {step.name} has neither a command nor a native handler")
        log = self.layout.root / "logs" / f"{step.stage}_{step.name}.log"
        code = self.runner(step, log=log)
        if code != 0:
            raise StageFailed(f"{step.name} exited {code}; see {log}")
        missing = [output for output in step.outputs if not Path(output).exists()]
        if missing:
            raise StageFailed(f"{step.name} produced no {missing[0]}")

    # -- native steps ------------------------------------------------------

    def _native_ingest_dataset(self, step: PlannedStep, plan: Plan) -> None:
        manifest = self.layout.prepare_manifest
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if "dataset" not in payload or "trainer_paths" not in payload:
                raise StageFailed(
                    f"{manifest} is not a prepare manifest (needs 'dataset' and 'trainer_paths')"
                )
            self.say(f"[prepare] adopting existing {manifest}")
            return
        scene = load_dataset_bundle(self.dataset_root, self.profile, self.work_root)
        self.write_prepare_manifest(scene, self.dataset_summary())

    def _native_write_arm_configs(self, step: PlannedStep, plan: Plan) -> None:
        self.write_arm_configs(plan)

    def _native_threshold_control(self, step: PlannedStep, plan: Plan) -> None:
        """Export the same merge at each control opacity and record the counts."""
        from tools.pipeline import read_ply_vertex_count

        delivery = self.layout.delivery_dir(plan.delivery_tag)
        out = delivery / "threshold_control"
        out.mkdir(parents=True, exist_ok=True)
        variants: list[dict[str, Any]] = []
        for threshold in self.profile.export["threshold_control"]:
            variant = out / f"body_min_opacity_{threshold:g}.ply"
            command = (
                str(self.python),
                str(self.repo_root / "tools" / "export_gaussian_ply.py"),
                "--checkpoint", str(delivery / "merged.pt"),
                "--output", str(variant),
                "--min-opacity", str(threshold),
            )
            code = self.runner(
                PlannedStep(f"threshold_{threshold:g}", "deliver", "cpu", step.estimate, command=command),
                log=self.layout.root / "logs" / "deliver_threshold_control.log",
            )
            if code != 0:
                raise StageFailed(f"threshold export at {threshold:g} exited {code}")
            variants.append(
                {"min_opacity": threshold, "path": str(variant), "vertex_count": read_ply_vertex_count(variant)}
            )
        baseline = variants[0]["vertex_count"]
        for record in variants:
            record["removed_vs_zero"] = baseline - record["vertex_count"]
        _write_json_atomic(
            out / "threshold_control.json",
            {
                "schema_version": 1,
                "delivery_min_opacity": self.profile.export["min_opacity"],
                "variants": variants,
                "recorded_at": _timestamp(),
            },
        )

    def _native_acceptance_report(self, step: PlannedStep, plan: Plan) -> None:
        self.write_report(plan)

    # -- artefact writers --------------------------------------------------

    def write_prepare_manifest(
        self,
        scene: PreparedScene,
        dataset: DatasetSummary,
        *,
        prior_tile_checkpoints: Mapping[int, str] | None = None,
    ) -> Path:
        """Record what prepare produced. This is the SDK's dataset contract."""
        payload = {
            "schema_version": 1,
            "kind": "cloudstudio3dgs_sdk_prepare_manifest",
            "scene_tag": dataset.scene_tag,
            "profile": self.profile.name,
            "profile_sha256": self.profile.profile_sha256,
            "dataset": dataset.as_json(),
            "scene": scene.as_json(),
            "trainer_paths": scene.trainer_paths(),
            "prior_tile_checkpoints": {str(k): str(v) for k, v in (prior_tile_checkpoints or {}).items()},
            "written_at": _timestamp(),
        }
        _write_json_atomic(self.layout.prepare_manifest, payload)
        self._dataset = dataset
        self._plan = None
        return self.layout.prepare_manifest

    def write_arm_configs(self, plan: Plan) -> list[Path]:
        """Materialise every trainer config the plan named, plus pipeline.json."""
        written: list[Path] = []
        self.layout.runs.mkdir(parents=True, exist_ok=True)
        for step in plan.steps:
            if step.config is None or not step.config_path:
                continue
            target = Path(step.config_path)
            _write_json_atomic(target, dict(step.config))
            written.append(target)
        coarse = coarse_config(
            self.profile, plan.dataset, layout=self.layout, bundle_paths=self.bundle_paths()
        )
        coarse_path = self.layout.arm_config(
            str(self.profile.coarse_prior["arm_name"]).format(profile=self.profile.name)
        )
        _write_json_atomic(coarse_path, coarse)
        written.append(coarse_path)
        written.append(self.write_pipeline_config(plan))
        return written

    def write_pipeline_config(self, plan: Plan) -> Path:
        """The machine config ``tools/pipeline.py`` reads for this work root.

        ``reference_ply`` / ``reference_alignment`` / ``sky_ply`` are required
        by that config schema even though a first delivery of a new scene has
        no competitor model to compare against; they are pointed at this work
        root's own artefacts and the plan carries the warning.
        """
        paths = self.bundle_paths()
        payload = {
            "schema_version": 1,
            "run_root": str(self.layout.runs),
            "repo_root": str(self.repo_root),
            "python": str(self.python),
            "reference_ply": paths.get("reference_ply", str(self.layout.exports / "reference.ply")),
            "reference_alignment": paths.get(
                "reference_alignment", str(self.layout.exports / "reference_alignment.json")
            ),
            "tile_inputs_manifest": paths.get("tile_inputs_manifest", "<prepare:tile_inputs_manifest>"),
            "tile_inputs_root": paths.get("tile_inputs_root", "<prepare:tile_inputs_root>"),
            "exports_dir": str(self.layout.exports),
            "delivery_eval_config": str(self.layout.root / "delivery_eval.json"),
            "sky_ply": str(self.layout.caches / "sky_dome.ply"),
            "identity_dir": str(self.layout.report),
            "scene_tag": plan.scene_tag,
            "compare_frames": int(self.profile.battery["compare_frames"]),
            "battery_views": int(self.profile.battery["views"]),
            "export_min_opacity": float(self.profile.export["min_opacity"]),
            "merge_policy": str(self.profile.merge["policy"]),
            "harmonize_exposure": bool(self.profile.merge["harmonize_exposure"]),
            "delivery_tiles": [tile.tile_id for tile in plan.dataset.tiles if tile.tile_id != 0],
            "delivery_tile_arm_pattern": "tile{tile}_" + f"{self.profile.name}_delivery",
            "gpu_device": "cuda:0",
            "env": {"PYTHONIOENCODING": "utf-8"},
        }
        _write_json_atomic(self.layout.pipeline_config, payload)
        return self.layout.pipeline_config

    def write_report(self, plan: Plan) -> tuple[Path, Path]:
        """Acceptance report: profile gates against what the delivery measured."""
        delivery = self.layout.delivery_dir(plan.delivery_tag)
        battery = _read_json_or_none(delivery / "battery_final.json")
        merge_report = _read_json_or_none(delivery / "merge_report.json")
        gates = self.profile.acceptance
        measured: dict[str, Any] = {}
        if isinstance(battery, Mapping):
            summary = battery.get("summary") if isinstance(battery.get("summary"), Mapping) else battery
            for key in ("psnr_mean", "psnr_p10", "alpha_mean", "alpha_p05"):
                if key in summary:
                    measured[key] = summary[key]
        if isinstance(merge_report, Mapping):
            for key in ("merged_gaussian_count", "tile_gaussian_count", "fill_gaussian_count"):
                if key in merge_report:
                    measured[key] = merge_report[key]
        verdicts: list[dict[str, Any]] = []
        for gate, value, comparison in (
            ("battery_alpha_mean_min", measured.get("alpha_mean"), "ge"),
            ("battery_psnr_p10_min", measured.get("psnr_p10"), "ge"),
        ):
            threshold = gates.get(gate)
            if threshold is None or value is None:
                verdicts.append({"gate": gate, "status": "UNVERIFIED", "threshold": threshold, "measured": value})
                continue
            passed = float(value) >= float(threshold) if comparison == "ge" else float(value) <= float(threshold)
            verdicts.append(
                {"gate": gate, "status": "PASS" if passed else "FAIL", "threshold": threshold, "measured": value}
            )
        payload = {
            "schema_version": 1,
            "scene_tag": plan.scene_tag,
            "delivery_tag": plan.delivery_tag,
            "profile": self.profile.identity(),
            "plan_sha256": plan.plan_sha256,
            "measured": measured,
            "gates": verdicts,
            "reference": dict(gates.get("reference", {})),
            "unmeasured_knobs": list(self.profile.unmeasured_knobs()),
            "open_questions": [dict(item) for item in self.profile.open_questions],
            "plan_warnings": list(plan.warnings),
            "written_at": _timestamp(),
        }
        json_path = self.layout.report / f"{plan.delivery_tag}_report.json"
        _write_json_atomic(json_path, payload)

        lines = [
            f"# {plan.scene_tag} / {plan.delivery_tag}",
            "",
            f"profile `{self.profile.name}@{self.profile.version}` sha256 `{self.profile.profile_sha256}`",
            f"plan sha256 `{plan.plan_sha256}`",
            "",
            "## Gates",
            "",
            "| gate | threshold | measured | verdict |",
            "| --- | --- | --- | --- |",
        ]
        for verdict in verdicts:
            lines.append(
                f"| {verdict['gate']} | {verdict['threshold']} | {verdict['measured']} | {verdict['status']} |"
            )
        lines += ["", "## Not measured on this scene", ""]
        for knob in self.profile.unmeasured_knobs():
            why = self.profile.why(knob)
            lines.append(f"- `{knob}` ({why.confidence}): {why.claim}")
        lines += ["", "## Open questions carried by this profile", ""]
        for item in self.profile.open_questions:
            lines.append(f"- **{item['id']}** - {item['what']} (status: {item['status']})")
        if plan.warnings:
            lines += ["", "## Plan warnings", ""] + [f"- {warning}" for warning in plan.warnings]
        md_path = self.layout.report / f"{plan.delivery_tag}_report.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, md_path

    # -- the four stages ---------------------------------------------------

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> StageResult:
        """Ingest the dataset and build every derived cache. CPU only."""
        return self._run_stage("prepare", force=force, dry_run=dry_run)

    def train(self, *, force: bool = False, dry_run: bool = False) -> StageResult:
        """Coarse prior, stand-in backdrops and the per-tile arms. GPU."""
        return self._run_stage("train", force=force, dry_run=dry_run)

    def deliver(self, *, force: bool = False, dry_run: bool = False) -> StageResult:
        """Merge (with the fill layer), export, re-import, battery, identity."""
        return self._run_stage("deliver", force=force, dry_run=dry_run)

    def report(self, *, force: bool = False, dry_run: bool = False) -> StageResult:
        """Acceptance report: profile gates against what was measured."""
        return self._run_stage("report", force=force, dry_run=dry_run)

    def run_all(
        self,
        *,
        stages: Sequence[str] = STAGES,
        force: bool = False,
        dry_run: bool = False,
    ) -> list[StageResult]:
        for stage in stages:
            if stage not in STAGES:
                raise KeyError(f"unknown stage {stage!r}; known: {', '.join(STAGES)}")
        if dry_run:
            self.say(self.plan(stages=stages).render())
            return [StageResult(stage, "planned", "dry run") for stage in stages]
        results: list[StageResult] = []
        for stage in stages:
            result = getattr(self, stage)(force=force)
            results.append(result)
            if not result.ok:
                break
        return results


def _read_json_or_none(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
