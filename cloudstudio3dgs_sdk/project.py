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

from tools.pipeline import _timestamp, _write_json_atomic, file_sha256, read_ply_vertex_count

from cloudstudio3dgs_sdk.bundle import PreparedScene, load_dataset_bundle
from cloudstudio3dgs_sdk.discover import DatasetEstimate, estimate_dataset_summary
from cloudstudio3dgs_sdk.ingest.errors import GpuStepRequired
from cloudstudio3dgs_sdk.plan import (
    STAGES,
    DatasetSummary,
    Plan,
    PlannedStep,
    WorkLayout,
    build_plan,
    coarse_config,
    resolve_cache_paths,
    tile_key,
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
        adapter: str | None = None,
        run_dir: Path | str | None = None,
        pipeline_gate: Path | str | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.work_root = Path(work_root)
        self.profile = profile
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
        self.python = Path(python) if python else Path(sys.executable)
        self.layout = WorkLayout(self.work_root)
        self.scene_tag = scene_tag
        self.delivery_tag = delivery_tag
        # Fresh-dataset inputs only; an adopted scene never reads them.
        self.adapter = adapter
        self.run_dir = Path(run_dir) if run_dir else None
        self.pipeline_gate = Path(pipeline_gate) if pipeline_gate else None
        self.runner = runner or SubprocessRunner(repo_root=self.repo_root)
        self.probes = probes
        self.vram_gib = vram_gib
        self._prior_override = dict(prior_tile_checkpoints or {})
        self.stream = stream or sys.stdout
        self._dataset = dataset
        self._plan: Plan | None = None
        # An estimated summary never lands in ``_dataset`` or ``_plan``: those
        # feed the run path, and a run must refuse an estimate even if a
        # dry-run built one earlier in the same process.
        self._estimate: DatasetEstimate | None = None

    # -- plumbing --------------------------------------------------------

    def say(self, line: str) -> None:
        print(line, file=self.stream)

    def stage_state(self, stage: str) -> StageState:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}")
        return StageState(self.layout.state / f"stage_{stage}.json", stage=stage)

    def dataset_summary(self, *, allow_estimate: bool = False) -> DatasetSummary:
        """The prepared scene. Read from the prepare manifest unless injected.

        ``allow_estimate`` is the planning-only escape hatch: with no prepare
        manifest it derives a summary from the capture
        (:func:`cloudstudio3dgs_sdk.discover.estimate_dataset_summary`) instead
        of refusing, so ``--dry-run`` and ``preflight`` can cost a dataset
        nobody has ingested yet. It is off by default, which is what keeps a
        real run failing closed.
        """
        if self._dataset is not None:
            return self._dataset
        manifest = self.layout.prepare_manifest
        if not manifest.is_file():
            if allow_estimate:
                return self.dataset_estimate().summary
            raise StageRefused(
                f"no prepared dataset: {manifest} does not exist. Run prepare() first, or pass "
                "dataset=DatasetSummary(...) to plan a scene before ingestion."
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self._dataset = DatasetSummary.from_json(payload["dataset"])
        return self._dataset

    def dataset_estimate(self) -> DatasetEstimate:
        """Derive the summary from the capture, with the notes behind it.

        Cached for the life of the project because the derivation streams the
        whole point cloud twice; the cache is separate from ``_dataset`` on
        purpose, so nothing on the run path can pick it up.
        """
        if self._estimate is None:
            self._estimate = estimate_dataset_summary(
                self.dataset_root, self.profile, scene_tag=self.scene_tag
            )
        return self._estimate

    def bundle_paths(self) -> dict[str, str]:
        """Every path the prepare manifest binds: trainer paths plus derived caches.

        ``trainer_paths`` are the scene-level keys each arm config carries
        verbatim; ``derived_paths`` are the tile inputs, the coarse
        initialisation, the lock and every cache (see
        :data:`cloudstudio3dgs_sdk.plan.DERIVED_SCENE_KEYS`). The plan reads
        one merged mapping so there is exactly one place a path can come from.
        """
        manifest = self.layout.prepare_manifest
        if not manifest.is_file():
            return {}
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths = dict(payload.get("trainer_paths") or {})
        paths.update(payload.get("derived_paths") or {})
        return paths

    def adopted_paths(self) -> set[str]:
        """Artefacts the prepare manifest adopted from outside this work root.

        They were verified against their signed manifests when adopted and
        are somebody else's files (house0305's caches live beside the as-run
        arms, not under ``<work>``). No stage may rebuild one in place, not
        even under ``--force``: a rebuilt cache would silently replace the
        input every recorded delivery was scored against.
        """
        manifest = self.layout.prepare_manifest
        if not manifest.is_file():
            return set()
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        digests = payload.get("digests") or {}
        return {
            os.path.normcase(str(record["path"]))
            for record in digests.values()
            if isinstance(record, Mapping) and record.get("path")
        }

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

    def plan(
        self,
        *,
        refresh: bool = False,
        stages: Sequence[str] = STAGES,
        allow_estimate: bool = False,
    ) -> Plan:
        """The plan for this project. ``stages`` narrows it without running.

        ``allow_estimate`` falls back to a capture-derived dataset summary when
        nothing has been prepared. The resulting plan is marked and cannot be
        executed; see :meth:`dataset_summary`.
        """
        if allow_estimate:
            # Never cached: a cached estimated plan would be handed to the next
            # caller, who may be a run.
            return self._build_plan(stages, allow_estimate=True)
        if tuple(stages) != tuple(STAGES):
            # A narrowed plan is a different object; never cache it as *the* plan.
            return self._build_plan(stages)
        if self._plan is None or refresh:
            self._plan = self._build_plan(STAGES)
        return self._plan

    def _build_plan(self, stages: Sequence[str], *, allow_estimate: bool = False) -> Plan:
        return build_plan(
            self.profile,
            self.dataset_summary(allow_estimate=allow_estimate),
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

    def preflight(self, *, require_gpu: bool = True, allow_estimate: bool = False) -> PreflightReport:
        return preflight(
            self.plan(allow_estimate=allow_estimate),
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
        if plan.dataset.estimated:
            # plan(allow_estimate=False) already refuses to build one, so this
            # only fires for a summary handed in through dataset=/--summary
            # that carries the flag. Estimated numbers cost a delivery nobody
            # can reconstruct; they are never a training input.
            raise StageRefused(
                f"[{stage}] refuses an estimated dataset summary: its tile boxes, view counts "
                "and initialisation counts were derived from the capture, not measured by "
                "prepare(). Estimates are for --dry-run and preflight. Run prepare() first."
            )
        report: PreflightReport | None = None
        if stage in GPU_STAGES:
            report = self.preflight(require_gpu=True)
            report.raise_for_status()
        steps = plan.stage_steps(stage)
        state = self.stage_state(stage)
        state.set(RUNNING, f"{len(steps)} step(s) planned", plan_sha256=plan.plan_sha256,
                  profile_sha256=self.profile.profile_sha256)
        if stage == "prepare" and self.layout.prepare_manifest.is_file():
            # The manifest is ingest_dataset's output, so the driver would skip
            # that step and never look inside. Adopting is verifying: every
            # path it names must be there and every digest must still match
            # before an arm config is written against it.
            try:
                self.adopt_prepare_manifest()
            except StageRefused as error:
                state.set(FAILED, f"prepare manifest: {error}")
                raise
        ran: list[str] = []
        skipped: list[str] = []
        adopted = self.adopted_paths()
        for step in steps:
            if step.blocking:
                state.set(FAILED, f"{step.name}: {step.blocking}")
                raise StageRefused(f"[{stage}] {step.name} cannot run: {step.blocking}")
            if step.outputs and all(os.path.normcase(output) in adopted for output in step.outputs):
                # Adopted artefacts are inputs, never outputs: --force re-runs
                # what this work root built, not what it was handed.
                self.say(f"[{stage}] skip {step.name} (adopted artefact; never rebuilt in place)")
                skipped.append(step.name)
                continue
            if (
                not force
                and not step.refresh
                and step.outputs
                and all(Path(output).exists() for output in step.outputs)
            ):
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

    def adopt_prepare_manifest(self) -> list[str]:
        """Verify the existing prepare manifest; refuse if anything it names moved."""
        from cloudstudio3dgs_sdk.adopt import verify_prepare_manifest

        manifest = self.layout.prepare_manifest
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise StageRefused(f"{manifest} is not readable JSON ({error})") from error
        checked = verify_prepare_manifest(payload, manifest_path=manifest)
        self.say(f"[prepare] adopting existing {manifest} ({len(checked)} paths/digests verified)")
        return checked

    def _native_ingest_dataset(self, step: PlannedStep, plan: Plan) -> None:
        if self.layout.prepare_manifest.is_file():
            # Reached only under --force (the driver skips the step otherwise):
            # a manifest that exists is adopted, never rebuilt over.
            self.adopt_prepare_manifest()
            return
        try:
            scene = load_dataset_bundle(
                self.dataset_root,
                self.profile,
                self.work_root,
                python=self.python,
                repo_root=self.repo_root,
                adapter=self.adapter,
                run_dir=self.run_dir,
                pipeline_gate=self.pipeline_gate,
                log=self.say,
            )
        except GpuStepRequired as error:
            # prepare is the CPU stage. A cache that needs CUDA is not built
            # here; the operator gets the exact command, never a silent skip.
            command = getattr(error, "command", None)
            detail = f"[prepare] a cache in this dataset's plan needs a GPU: {error}"
            if command:
                argv = command if isinstance(command, str) else " ".join(str(part) for part in command)
                detail += f"\n  run on a CUDA host, then re-run prepare: {argv}"
            raise StageRefused(detail) from error
        self.write_prepare_manifest(scene, self.dataset_summary())

    def _native_write_arm_configs(self, step: PlannedStep, plan: Plan) -> None:
        self.write_arm_configs(plan)

    def _native_delivery_eval_config(self, step: PlannedStep, plan: Plan) -> None:
        self.write_delivery_eval_config(plan)

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
        derived_paths: Mapping[str, str] | None = None,
        digests: Mapping[str, Mapping[str, Any]] | None = None,
        adopted: Mapping[str, Any] | None = None,
    ) -> Path:
        """Record what prepare produced. This is the SDK's dataset contract.

        ``derived_paths`` defaults to what the scene itself declares (see
        :func:`derived_paths_from_scene`); ``adopt`` passes the verified map
        and the digests it established, so a later prepare can re-check them.
        """
        derived = dict(derived_paths) if derived_paths is not None else derived_paths_from_scene(
            scene, repo_root=self.repo_root
        )
        payload = {
            "schema_version": 2,
            "kind": "cloudstudio3dgs_sdk_prepare_manifest",
            "scene_tag": dataset.scene_tag,
            "profile": self.profile.name,
            "profile_sha256": self.profile.profile_sha256,
            "dataset": dataset.as_json(),
            "scene": scene.as_json(),
            "trainer_paths": scene.trainer_paths(),
            "derived_paths": {key: str(value) for key, value in sorted(derived.items())},
            "digests": {key: dict(value) for key, value in sorted((digests or {}).items())},
            "prior_tile_checkpoints": {str(k): str(v) for k, v in (prior_tile_checkpoints or {}).items()},
            "written_at": _timestamp(),
        }
        if adopted is not None:
            payload["adopted"] = dict(adopted)
        _write_json_atomic(self.layout.prepare_manifest, payload)
        self._dataset = dataset
        self._plan = None
        return self.layout.prepare_manifest

    def write_delivery_eval_config(self, plan: Plan) -> Path:
        """``<work>/delivery_eval.json``: the evaluator's view of this scene.

        ``tools/evaluate_probe_views.py`` reads dataset paths straight out of
        a trainer config (face cache, backgrounds, tile inputs, ownership
        knobs, device, tile id) and never builds a TrainerConfig, so the
        tile-0 delivery config with its run identity changed is exactly the
        config it needs. Its sh_degree stays the trainer's: a lower value
        here clamps the render and scores a model nobody trained.

        The evaluator reads the *validation* caches derived by name from the
        config's training caches; those are checked here, at prepare time,
        because the first SDK delivery trained for seven hours and then died
        at the battery on a validation cache that did not exist. A scene that
        adopted an evaluator config (``delivery_eval_source``) gets it verbatim.
        """
        from cloudstudio_3dgs.training.validation_paths import derive_validation_paths

        bundle = self.bundle_paths()
        target = Path(resolve_cache_paths(self.layout, plan.dataset, bundle)["delivery_eval_config"])
        adopted = bundle.get("delivery_eval_source")
        if adopted:
            source_path = Path(adopted)
            if not source_path.is_file():
                raise StageRefused(f"adopted evaluator config is missing: {source_path}")
            config = json.loads(source_path.read_text(encoding="utf-8"))
            config["lineage"] = {
                "adopted_from": str(source_path),
                "purpose": "evaluate_probe_views.py config adopted verbatim; dataset paths only, never trained",
                "profile_sha256": self.profile.profile_sha256,
            }
        else:
            first = plan.dataset.tiles[0]
            arm = str(self.profile.tile_rules["arm_name_pattern"]).format(
                tile=first.tile_id, profile=self.profile.name, generation="delivery"
            )
            source = next(
                (step for step in plan.steps if step.config is not None and step.name == f"train_{arm}"),
                None,
            )
            if source is None or source.config is None:
                raise StageFailed(f"no delivery config for {first.name} in the plan; cannot derive delivery_eval.json")
            config = json.loads(json.dumps(dict(source.config)))
            config["run_id"] = f"{plan.scene_tag}-delivery-eval"
            config["output_dir"] = str(self.layout.runs / "delivery_eval")
            config["lineage"] = {
                "derived_from": arm,
                "purpose": "evaluate_probe_views.py config; dataset paths only, never trained",
                "profile_sha256": self.profile.profile_sha256,
            }
        # Only paths the derivation actually moved are validation-side artefacts nobody
        # else produces; one it left alone (a backdrop library without "_train" in its
        # name) is the training artefact itself, and the plan builds that later.
        missing = sorted(
            f"{key} -> {path}"
            for key, path in derive_validation_paths(config).items()
            if str(path) != str(config.get(key)) and not Path(path).exists()
        )
        if missing:
            raise StageRefused(
                "the battery's validation caches, derived by name from the evaluator config, do not "
                "exist:\n  " + "\n  ".join(missing)
                + "\n  build them next to the training caches, or adopt an evaluator config that names "
                "existing ones (adopt --eval-config)"
            )
        _write_json_atomic(target, config)
        return target

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
        cache = resolve_cache_paths(self.layout, plan.dataset, paths)
        payload = {
            "schema_version": 1,
            "run_root": str(self.layout.runs),
            "repo_root": str(self.repo_root),
            "python": str(self.python),
            "reference_ply": paths.get("reference_ply", str(self.layout.exports / "reference.ply")),
            "reference_alignment": paths.get(
                "reference_alignment", str(self.layout.exports / "reference_alignment.json")
            ),
            "tile_inputs_manifest": cache["tile_inputs_manifest"],
            "tile_inputs_root": cache["tile_inputs_root"],
            "exports_dir": str(self.layout.exports),
            "delivery_eval_config": cache["delivery_eval_config"],
            "sky_ply": cache["sky_dome_ply"],
            "identity_dir": str(self.layout.report),
            "scene_tag": plan.scene_tag,
            "compare_frames": int(self.profile.battery["compare_frames"]),
            "battery_views": int(self.profile.battery["views"]),
            "export_min_opacity": float(self.profile.export["min_opacity"]),
            "merge_policy": str(self.profile.merge["policy"]),
            "harmonize_exposure": bool(self.profile.merge["harmonize_exposure"]),
            "delivery_tiles": [tile.tile_id for tile in plan.dataset.tiles if tile.tile_id != 0],
            # tools/pipeline.py validates that both {tile} and {tag} appear. The SDK's arm names
            # are tile{N}_{profile}_{generation} and its delivery tag defaults to the profile
            # name, so with {tag} = profile the pattern expands to exactly the arms the plan
            # trained. The first real SDK run died here: the pattern carried the profile name
            # literally and no {tag}, and the pipeline refused it before training anything.
            "delivery_tile_arm_pattern": "tile{tile}_{tag}_delivery",
            "gpu_device": "cuda:0",
            "env": {"PYTHONIOENCODING": "utf-8"},
        }
        _write_json_atomic(self.layout.pipeline_config, payload)
        return self.layout.pipeline_config

    def write_report(self, plan: Plan) -> tuple[Path, Path]:
        """Acceptance report: profile gates against what the delivery measured.

        Two batteries are read and both are kept, labelled ``body_only`` and
        ``delivered_pair``. The gates read the pair - the customer opens body
        plus sky, and the body's sky is transparent by design, so a coverage
        number on the body alone reads that transparency as a hole. Morphology
        and the export count stay on the body: they are shape and size
        comparisons against a competitor's body, which the dome would distort.
        Same shape as ``tools/pipeline.py``'s ``final.coverage`` / ``final.gate``.
        """
        delivery = self.layout.delivery_dir(plan.delivery_tag)
        body_only = _battery_reading(delivery / "battery_final.json", "body")
        delivered_pair = _battery_reading(delivery / "battery_final_pair.json", "body+sky")
        merge_report = _read_json_or_none(delivery / "merge_report.json")
        morphology = _read_json_or_none(delivery / "morph_final.json")
        gates = self.profile.acceptance
        measured: dict[str, Any] = {}
        for key in ("psnr_mean", "psnr_p10", "alpha_mean", "alpha_p05"):
            measured[f"battery_{key}"] = delivered_pair.get(key)
            measured[f"body_only_{key}"] = body_only.get(key)
        if isinstance(merge_report, Mapping):
            for key in ("merged_gaussian_count", "tile_gaussian_count", "fill_gaussian_count"):
                if key in merge_report:
                    measured[key] = merge_report[key]
        body_ply = delivery / f"{plan.scene_tag}_{plan.delivery_tag}_merged.ply"
        try:
            measured["export_gaussian_count"] = read_ply_vertex_count(body_ply)
        except (OSError, ValueError):
            measured["export_gaussian_count"] = None
        stats = morphology.get("stats") if isinstance(morphology, Mapping) else None
        measured["morphology_short_axis_p50_mm"] = (
            stats.get("short_p50_mm") if isinstance(stats, Mapping) else None
        )
        offtraj = _read_json_or_none(delivery / "offtrajectory_summary.json")
        measured["offtrajectory_sharpness_ours_over_ref"] = (
            offtraj.get("sharpness_ours_over_ref") if isinstance(offtraj, Mapping) else None
        )
        # gate key -> (measurement, comparison, which layers it was read from)
        table: list[tuple[str, str, str, str]] = [
            ("battery_alpha_p05_min", "battery_alpha_p05", "ge", "delivered_pair"),
            ("battery_psnr_p10_min", "battery_psnr_p10", "ge", "delivered_pair"),
            ("offtrajectory_sharpness_min", "offtrajectory_sharpness_ours_over_ref", "ge", "body_only"),
            ("export_gaussian_count_max", "export_gaussian_count", "le", "body_only"),
            ("morphology_short_axis_p50_mm_max", "morphology_short_axis_p50_mm", "le", "body_only"),
        ]
        if "battery_alpha_mean_min" in gates:
            # Older profile revisions gate on the mean; read it the same way.
            table.insert(0, ("battery_alpha_mean_min", "battery_alpha_mean", "ge", "delivered_pair"))
        verdicts: list[dict[str, Any]] = []
        for gate, measurement, comparison, layers in table:
            threshold = gates.get(gate)
            value = measured.get(measurement)
            record = {"gate": gate, "threshold": threshold, "measured": value, "reads": layers}
            if threshold is None or value is None:
                record["status"] = "UNVERIFIED"
            else:
                passed = float(value) >= float(threshold) if comparison == "ge" else float(value) <= float(threshold)
                record["status"] = "PASS" if passed else "FAIL"
            verdicts.append(record)
        payload = {
            "schema_version": 2,
            "scene_tag": plan.scene_tag,
            "delivery_tag": plan.delivery_tag,
            "profile": self.profile.identity(),
            "plan_sha256": plan.plan_sha256,
            "measured": measured,
            "coverage": {"body_only": body_only, "delivered_pair": delivered_pair},
            "gate": {
                "reads": "delivered_pair",
                "alpha_p05": delivered_pair.get("alpha_p05"),
                "body_only_alpha_p05": body_only.get("alpha_p05"),
                "psnr_p10": delivered_pair.get("psnr_p10"),
                "body_only_psnr_p10": body_only.get("psnr_p10"),
                "why": "the delivery ships body + sky; the body's sky is transparent by design",
                "morphology_reads": "body_only",
            },
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
            "| gate | threshold | measured | reads | verdict |",
            "| --- | --- | --- | --- | --- |",
        ]
        for verdict in verdicts:
            lines.append(
                f"| {verdict['gate']} | {verdict['threshold']} | {verdict['measured']} | {verdict['reads']} "
                f"| {verdict['status']} |"
            )
        lines += [
            "",
            "## Coverage",
            "",
            "| layers | alpha_p05 | alpha_mean | psnr_p10 | psnr_mean |",
            "| --- | --- | --- | --- | --- |",
        ]
        for label, reading in (("body_only", body_only), ("delivered_pair", delivered_pair)):
            lines.append(
                f"| {label} ({reading['layers']}) | {reading.get('alpha_p05')} | {reading.get('alpha_mean')} "
                f"| {reading.get('psnr_p10')} | {reading.get('psnr_mean')} |"
            )
        lines.append("")
        lines.append("gates read delivered_pair for alpha/PSNR; morphology and the export count read the body alone")
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


def _battery_reading(path: Path, layers: str) -> dict[str, Any]:
    """One battery's numbers, labelled with the layers it scored.

    Missing or malformed batteries read as ``None`` everywhere, which the
    gates turn into UNVERIFIED; nothing here invents a number.
    """
    payload = _read_json_or_none(path)
    summary: Mapping[str, Any] = {}
    if isinstance(payload, Mapping):
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else payload
    return {
        "layers": layers,
        "battery": str(path),
        "present": bool(summary) and summary.get("alpha_p05") is not None,
        "views": summary.get("views"),
        "alpha_p05": summary.get("alpha_p05"),
        "alpha_mean": summary.get("alpha_mean"),
        "psnr_mean": summary.get("psnr_mean"),
        "psnr_p10": summary.get("psnr_p10"),
        "gaussian_count": summary.get("gaussian_count"),
    }


def derived_paths_from_scene(scene: PreparedScene, *, repo_root: Path) -> dict[str, str]:
    """The ``derived_paths`` block a freshly prepared scene declares.

    Scene-level inputs come from the :class:`PreparedScene` fields; caches
    from its ``caches`` record where set; the gsplat lock is this checkout's.
    Per-tile initialisation paths are read from the tile inputs and tile
    geometry manifests when they are readable, because those manifests are
    the only record of the per-tile file names. Anything not resolvable is
    left out and the plan's layout defaults (or placeholders) apply.
    """
    paths: dict[str, str] = {}
    for key in ("tile_inputs_manifest", "tile_inputs_root", "tile_geometry_manifest",
                "global_init_ply", "global_init_geometry", "lidar_cloud"):
        value = getattr(scene, key, None)
        if value is not None:
            paths[key] = str(value)
    paths["gsplat_lock"] = str(Path(repo_root) / "upstream" / "gsplat.lock.json")
    caches = getattr(scene, "caches", None)
    for attribute, key in (
        ("sky_mask_manifest", "sky_mask_manifest"),
        ("sky_mask_root", "sky_mask_root"),
        ("sky_dome_checkpoint", "sky_dome_checkpoint"),
        ("sky_dome_ply", "sky_dome_ply"),
        ("global_background_manifest", "global_view_backgrounds_manifest"),
        ("global_background_root", "global_view_backgrounds_root"),
    ):
        value = getattr(caches, attribute, None)
        if value is not None:
            paths[key] = str(value)
    for attribute, suffixes in (
        ("tile_ownership", ("ownership_manifest", "ownership_root")),
        ("tile_backdrops", ("backdrop_manifest", "backdrop_root")),
    ):
        mapping = getattr(caches, attribute, None) or {}
        for tile_id, pair in mapping.items():
            for suffix, value in zip(suffixes, tuple(pair)):
                paths[tile_key(int(tile_id), suffix)] = str(value)
    tile_inputs = _read_json_or_none(Path(paths["tile_inputs_manifest"])) if "tile_inputs_manifest" in paths else None
    if isinstance(tile_inputs, Mapping) and "tile_inputs_root" in paths:
        root = Path(paths["tile_inputs_root"])
        for entry in tile_inputs.get("tiles", []):
            init = entry.get("initialization") if isinstance(entry, Mapping) else None
            if isinstance(init, Mapping) and init.get("path"):
                paths[tile_key(int(entry["tile_id"]), "initialization_ply")] = str(root / str(init["path"]))
    geometry = _read_json_or_none(Path(paths["tile_geometry_manifest"])) if "tile_geometry_manifest" in paths else None
    if isinstance(geometry, Mapping):
        base = Path(paths["tile_geometry_manifest"]).parent
        for entry in geometry.get("tiles", []):
            block = entry.get("geometry") if isinstance(entry, Mapping) else None
            if isinstance(block, Mapping) and block.get("path"):
                paths[tile_key(int(entry["tile_id"]), "initialization_geometry")] = str(base / str(block["path"]))
    return paths
