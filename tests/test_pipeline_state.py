"""P0-1..P0-4 of tools/pipeline.py: job state, frozen configs, PLY-bound
delivery scores and the atomic GPU lease.

Fault injection follows the 2026-09-11 audit brief: a 5k checkpoint under a
20k target, a corrupted checkpoint, an OOM exit beside an older checkpoint,
an edited config under an existing arm name, two CLIs racing for the GPU,
and a delivery report that must name the exported PLY's sha256.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
import zipfile
from pathlib import Path

from tests.test_pipeline_resume import (
    TARGET_STEPS,
    PipelineFixture,
    controlled_stop_traceback,
    fixture_inspector,
    write_fake_checkpoint,
    write_ply,
)
from tools.pipeline import (
    CONFIG_AS_RUN_NAME,
    CONFIG_FROZEN_NAME,
    EXIT_COMPLETED,
    EXIT_CONTROLLED_STOP,
    EXIT_CRASH,
    EXIT_OOM,
    EXIT_UNKNOWN,
    EXPORT_THRESHOLD_CONTROL,
    MIN_CHECKPOINT_BYTES,
    STATE_CHECKPOINTED,
    STATE_CONTROLLED_PAUSE,
    STATE_EVALUATED,
    STATE_FAILED,
    STATE_PUBLISHED,
    STATE_QUALITY_ACCEPTED,
    STATE_RUNNING,
    STATE_TRAINING_COMPLETE,
    ConfigFrozenError,
    GpuLeaseBusy,
    JobState,
    StepFailed,
    _pid_alive,
    acquire_gpu_lease,
    arm_steps,
    classify_trainer_exit,
    file_sha256,
    freeze_arm_config,
    inspect_checkpoint,
    main,
    peek_checkpoint_step,
    read_gpu_lease,
    read_ply_vertex_count,
    run_arm,
    run_deliver,
    run_queue,
    run_steps,
    verify_training,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Checkpoint inspection
# --------------------------------------------------------------------------


class _FakeStorage:
    pass


class _FakeTensor:
    """Reduces like a torch tensor: a global rebuild function over a persistent storage."""

    def __init__(self, storage: _FakeStorage) -> None:
        self.storage = storage

    def __reduce__(self):
        return (_FAKE_TORCH_UTILS._rebuild_tensor_v2, (self.storage, 0, (3,), (1,), False, {}))


def _rebuild_tensor_v2(*args):  # pragma: no cover - never called by the peeker
    raise AssertionError("real rebuild must not run")


_FAKE_TORCH_UTILS = types.ModuleType("faketorch._utils")
_rebuild_tensor_v2.__module__ = "faketorch._utils"
_FAKE_TORCH_UTILS._rebuild_tensor_v2 = _rebuild_tensor_v2


class _TorchLikePickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _FakeStorage):
            return ("storage", "faketorch.FloatStorage", "0", "cpu", 3)
        return None


def write_torch_like_checkpoint(path: Path, step: int) -> None:
    """A zip whose data.pkl uses persistent storages and foreign globals."""
    payload = {
        "schema_version": 1,
        "step": step,
        "params": {"means": _FakeTensor(_FakeStorage()), "opacities": _FakeTensor(_FakeStorage())},
        "training_state": {"last_metrics": {"loss": 0.1}},
    }
    sys.modules["faketorch"] = types.ModuleType("faketorch")
    sys.modules["faketorch._utils"] = _FAKE_TORCH_UTILS
    try:
        buffer = io.BytesIO()
        _TorchLikePickler(buffer, protocol=2).dump(payload)
    finally:
        sys.modules.pop("faketorch._utils", None)
        sys.modules.pop("faketorch", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("ckpt/data.pkl", buffer.getvalue())
        archive.writestr("ckpt/data/0", b"\0" * MIN_CHECKPOINT_BYTES)
        archive.writestr("ckpt/version", b"3\n")
    path.write_bytes(archive_bytes.getvalue())


class CheckpointInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.dir = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_peek_reads_step_without_importing_torch(self) -> None:
        path = self.dir / "latest.pt"
        write_torch_like_checkpoint(path, 7350)
        self.assertNotIn("faketorch", sys.modules)
        self.assertEqual(peek_checkpoint_step(path), 7350)
        self.assertNotIn("faketorch", sys.modules, "peeking must not import the checkpoint's globals")

    def test_header_path_accepts_torch_like_zip_and_reads_step(self) -> None:
        path = self.dir / "latest.pt"
        write_torch_like_checkpoint(path, 20000)
        info = inspect_checkpoint(path, loader=None) if not _has_torch() else None
        if info is None:
            self.skipTest("torch is importable here; the header path is not the default")
        self.assertTrue(info.loadable, info.reason)
        self.assertEqual(info.method, "header")
        self.assertEqual(info.step, 20000)

    def test_loader_path_validates_payload_and_reads_step(self) -> None:
        path = self.dir / "latest.pt"
        write_fake_checkpoint(path, 12345)
        info = fixture_inspector(path)
        self.assertTrue(info.loadable)
        self.assertEqual(info.method, "torch")
        self.assertEqual(info.step, 12345)
        self.assertGreater(info.size_bytes, MIN_CHECKPOINT_BYTES)

    def test_loader_rejects_payload_without_step_or_params(self) -> None:
        path = self.dir / "latest.pt"
        path.write_bytes(b"\0" * (MIN_CHECKPOINT_BYTES * 2))
        info = inspect_checkpoint(path, loader=lambda _: {"params": {}})
        self.assertFalse(info.loadable)
        self.assertIn("step", info.reason)
        info = inspect_checkpoint(path, loader=lambda _: {"step": 5})
        self.assertFalse(info.loadable)
        self.assertIn("params", info.reason)

    def test_missing_truncated_and_garbage_files_are_not_loadable(self) -> None:
        missing = inspect_checkpoint(self.dir / "none.pt", loader=None)
        self.assertFalse(missing.exists)
        self.assertFalse(missing.loadable)
        truncated = self.dir / "short.pt"
        truncated.write_bytes(b"PK\x03\x04" + b"\0" * 10)
        info = inspect_checkpoint(truncated, loader=None)
        self.assertFalse(info.loadable)
        self.assertIn("truncated", info.reason)
        garbage = self.dir / "garbage.pt"
        garbage.write_bytes(b"\xff" * (MIN_CHECKPOINT_BYTES * 3))
        info = inspect_checkpoint(garbage, loader=None)
        self.assertFalse(info.loadable)
        self.assertIn("not a torch zip", info.reason)
        info = fixture_inspector(garbage)
        self.assertFalse(info.loadable, "the loader path also rejects garbage")
        self.assertIn("failed", info.reason)

    def test_zip_without_data_pkl_is_not_a_checkpoint(self) -> None:
        path = self.dir / "other.zip"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", b"x" * (MIN_CHECKPOINT_BYTES * 2))
        path.write_bytes(buffer.getvalue())
        info = inspect_checkpoint(path, loader=None)
        self.assertFalse(info.loadable)
        self.assertIn("data.pkl", info.reason)


# --------------------------------------------------------------------------
# Trainer exit classification and the verdict function
# --------------------------------------------------------------------------


class TrainerExitTests(unittest.TestCase):
    def test_controlled_stop_marker_wins_over_its_own_traceback(self) -> None:
        exit = classify_trainer_exit(1, controlled_stop_traceback(3500, Path("x/latest.pt")))
        self.assertEqual((exit.kind, exit.steps), (EXIT_CONTROLLED_STOP, 3500))

    def test_oom_crash_completed_and_unknown(self) -> None:
        self.assertEqual(classify_trainer_exit(1, "Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory").kind, EXIT_OOM)
        self.assertEqual(classify_trainer_exit(1, "Traceback (most recent call last):\nKeyError: 'x'").kind, EXIT_CRASH)
        exit = classify_trainer_exit(0, "training complete: run=r, steps=49560, peak_vram=1 bytes, sha256=abc")
        self.assertEqual((exit.kind, exit.steps), (EXIT_COMPLETED, 49560))
        self.assertEqual(classify_trainer_exit(0, "nothing here").kind, EXIT_UNKNOWN)
        self.assertEqual(classify_trainer_exit(137, "").kind, EXIT_UNKNOWN)
        self.assertEqual(classify_trainer_exit(1, "training complete: run=r, steps=10").kind, EXIT_UNKNOWN, "a completion marker with a non-zero exit is not trusted")


class VerifyTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.dir = Path(self._temporary.name)
        self.checkpoint = self.dir / "arm" / "checkpoints" / "latest.pt"
        self.config = self.dir / "arm.json"
        self.config.write_text(json.dumps({"controlled_stop_after_steps": 20000, "max_steps": 49560}), encoding="utf-8")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def verdict(self, *, step: int | None = 20000, tail: str | None = None, exit_code: int = 1, started: float | None = None):
        if step is not None:
            write_fake_checkpoint(self.checkpoint, step)
        if tail is None:
            tail = controlled_stop_traceback(step or 0, self.checkpoint)
        return verify_training(
            checkpoint=self.checkpoint, arm_config=self.config, log_tail=tail, exit_code=exit_code,
            job_started_at=started, inspector=fixture_inspector,
        )

    def test_complete_when_every_check_passes(self) -> None:
        verdict = self.verdict(started=time.time() - 60)
        self.assertTrue(verdict.complete, verdict.reason)
        self.assertEqual(verdict.completed_steps, 20000)
        self.assertEqual(verdict.checks["mtime"], "newer than job start")
        record = verdict.record()
        self.assertTrue(record["verified"])
        self.assertEqual(record["exit"]["kind"], EXIT_CONTROLLED_STOP)

    def test_short_controlled_stop_is_a_pause_not_completion(self) -> None:
        verdict = self.verdict(step=5000)
        self.assertEqual(verdict.state, STATE_CONTROLLED_PAUSE)
        self.assertIn("short of the declared 20000", verdict.reason)

    def test_natural_completion_below_target_is_failed(self) -> None:
        verdict = self.verdict(step=5000, tail="training complete: run=r, steps=5000, peak_vram=1 bytes, sha256=a", exit_code=0)
        self.assertEqual(verdict.state, STATE_FAILED)

    def test_oom_exit_with_older_checkpoint_fails_and_names_both(self) -> None:
        write_fake_checkpoint(self.checkpoint, 20000)
        old = time.time() - 3600
        os.utime(self.checkpoint, (old, old))
        verdict = self.verdict(step=None, tail="Traceback (most recent call last):\nCUDA out of memory\n", started=time.time() - 10)
        self.assertEqual(verdict.state, STATE_FAILED)
        self.assertIn("oom", verdict.reason)
        self.assertIn("predates job start", verdict.reason)
        self.assertTrue(self.checkpoint.exists(), "the verdict never deletes evidence")

    def test_checkpoint_and_log_disagreeing_on_steps_is_failed(self) -> None:
        verdict = self.verdict(step=20000, tail=controlled_stop_traceback(5000, self.checkpoint))
        self.assertEqual(verdict.state, STATE_FAILED)
        self.assertIn("disagrees", verdict.reason)

    def test_step_falls_back_to_the_log_when_the_checkpoint_has_none(self) -> None:
        write_fake_checkpoint(self.checkpoint, 20000)
        verdict = verify_training(
            checkpoint=self.checkpoint, arm_config=self.config, log_tail=controlled_stop_traceback(20000, self.checkpoint),
            exit_code=1, job_started_at=None,
            inspector=_loadable_without_step,
        )
        self.assertTrue(verdict.complete, verdict.reason)
        self.assertEqual(verdict.checks["steps"], "20000 from log")
        self.assertIn("not applicable", verdict.checks["mtime"])

    def test_undeclared_target_and_unknown_steps_fail(self) -> None:
        self.config.write_text("{}", encoding="utf-8")
        verdict = self.verdict()
        self.assertEqual(verdict.state, STATE_FAILED)
        self.assertIn("neither", verdict.reason)
        self.config.write_text(json.dumps({"max_steps": 20000}), encoding="utf-8")
        verdict = verify_training(
            checkpoint=self.checkpoint, arm_config=self.config, log_tail="", exit_code=1, job_started_at=None,
            inspector=_loadable_without_step,
        )
        self.assertEqual(verdict.state, STATE_FAILED)
        self.assertIn("completed steps unknown", verdict.reason)


def _loadable_without_step(path: Path):
    info = fixture_inspector(path)
    info.step = None
    return info


# --------------------------------------------------------------------------
# Job state file
# --------------------------------------------------------------------------


class JobStateTests(unittest.TestCase):
    def test_transitions_persist_atomically_with_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "arm" / "job_state.json"
            job = JobState(path, job="arm", name="armA")
            self.assertIsNone(job.state)
            job.set(STATE_RUNNING, "launched", pid=1)
            with self.assertRaises(ValueError):
                job.set(STATE_TRAINING_COMPLETE, "no verdict yet")
            job.set(STATE_TRAINING_COMPLETE, "ok", training={"verified": True})
            job.set(STATE_EVALUATED, "scored")
            reloaded = JobState(path, job="arm", name="armA")
            self.assertEqual(reloaded.state, STATE_EVALUATED)
            self.assertEqual([h["state"] for h in reloaded.get("history")], [STATE_RUNNING, STATE_TRAINING_COMPLETE, STATE_EVALUATED])
            self.assertTrue(reloaded.training_verified())
            self.assertFalse(list(path.parent.glob("*.tmp")), "atomic write leaves no temp file")
            with self.assertRaises(ValueError):
                job.set("DONE")


# --------------------------------------------------------------------------
# Arm fault injection
# --------------------------------------------------------------------------


class ArmCompletionTests(PipelineFixture):
    def ledger(self) -> str:
        return self.config.arm_scores_file().read_text(encoding="utf-8")

    def test_successful_arm_walks_the_state_ladder(self) -> None:
        self.write_arm_config("armA")
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        job = self.ctx.arm_job("armA")
        self.assertEqual([h["state"] for h in job.get("history")], [STATE_RUNNING, STATE_CHECKPOINTED, STATE_TRAINING_COMPLETE, STATE_EVALUATED])
        training = job.get("training")
        self.assertEqual((training["completed_steps"], training["target_steps"]), (TARGET_STEPS, TARGET_STEPS))
        self.assertEqual(training["checkpoint"]["step"], TARGET_STEPS)
        self.assertEqual(job.get("config_sha256"), file_sha256(self.config.arm_config("armA")))
        self.assertEqual(self.ctx.gpu_leases_taken, ["train armA", "morph armA", "offtraj armA", "compare armA"])
        self.assertFalse(self.config.gpu_lock_file().exists())

    def test_fault1_target_20k_but_checkpoint_at_5k_is_not_complete_and_evaluation_is_refused(self) -> None:
        self.write_arm_config("armA")
        self.runner.train_steps = 5000
        self.assertEqual(run_arm(self.ctx, "armA"), 1)
        job = self.ctx.arm_job("armA")
        self.assertEqual(job.state, STATE_CONTROLLED_PAUSE)
        self.assertFalse(job.training_verified())
        self.assertFalse(self.ctx.arm_training_complete("armA"))
        self.assertIn("TRAIN_PAUSED", self.ledger())
        self.assertNotIn("checkpoint_morphology.py", self.runner.tools_called())
        self.assertTrue(self.config.arm_checkpoint("armA").exists(), "the 5k checkpoint is kept")
        # Every downstream step refuses the paused training outright.
        for step in arm_steps(self.ctx, "armA")[3:]:
            with self.subTest(step=step.name), self.assertRaises(StepFailed) as caught:
                step.run()
            self.assertIn(STATE_CONTROLLED_PAUSE, str(caught.exception))
        # A delivery on it is refused before any tool runs.
        self.runner.calls.clear()
        (self.run_root / "delivery_eval.json").write_text("{}", encoding="utf-8")
        self.assertEqual(run_deliver(self.ctx, "r1d", "armA"), 1)
        self.assertEqual(self.runner.calls, [])

    def test_fault2_corrupted_checkpoint_is_failed(self) -> None:
        self.write_arm_config("armA")
        self.runner.train_outcome = "garbage"
        self.assertEqual(run_arm(self.ctx, "armA"), 1)
        job = self.ctx.arm_job("armA")
        self.assertEqual(job.state, STATE_FAILED)
        self.assertIn("not loadable", job.reason)
        self.assertNotIn(STATE_CHECKPOINTED, [h["state"] for h in job.get("history")])
        self.assertIn("TRAIN_FAILED", self.ledger())
        self.assertNotIn("checkpoint_morphology.py", self.runner.tools_called())

    def test_fault3_oom_exit_with_older_checkpoint_is_failed_and_checkpoint_kept(self) -> None:
        self.write_arm_config("armA")
        checkpoint = self.plant_checkpoint("armA")
        old = time.time() - 3600
        os.utime(checkpoint, (old, old))
        before = checkpoint.read_bytes()
        # A previous pipeline attempt died mid-training: RUNNING with a dead pid.
        self.ctx.arm_job("armA").set(STATE_RUNNING, "previous attempt", pid=0, started_at=old)
        self.runner.train_outcome = "oom"
        self.assertEqual(run_arm(self.ctx, "armA"), 1)
        job = self.ctx.arm_job("armA")
        self.assertEqual(job.state, STATE_FAILED)
        self.assertIn("oom", job.reason)
        self.assertIn("predates job start", job.reason)
        self.assertEqual(checkpoint.read_bytes(), before, "the older checkpoint is kept for diagnosis")
        self.assertNotIn("checkpoint_morphology.py", self.runner.tools_called())
        self.assertFalse(self.ctx.arm_training_complete("armA"))

    def test_silent_exit_zero_without_marker_is_failed(self) -> None:
        self.write_arm_config("armA")
        self.runner.train_outcome = "silent"
        self.assertEqual(run_arm(self.ctx, "armA"), 1)
        job = self.ctx.arm_job("armA")
        self.assertEqual(job.state, STATE_FAILED)
        self.assertIn("unknown", job.reason)

    def test_scoring_failure_marks_failed_without_unverifying_training(self) -> None:
        self.write_arm_config("armA")
        self.runner.fail_on = {"build_three_way_compare.py"}
        self.assertEqual(run_arm(self.ctx, "armA"), 1)
        job = self.ctx.arm_job("armA")
        self.assertEqual(job.state, STATE_FAILED)
        self.assertTrue(job.training_verified(), "a scoring failure must not cause a retrain")
        self.runner.fail_on = set()
        self.runner.calls.clear()
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        self.assertNotIn("train_gsplat.py", self.runner.tools_called())
        self.assertEqual(self.ctx.arm_job("armA").state, STATE_EVALUATED)

    def test_running_arm_with_live_pid_is_not_started_twice(self) -> None:
        self.write_arm_config("armA")
        self.ctx.arm_job("armA").set(STATE_RUNNING, "elsewhere", pid=os.getppid(), started_at=time.time())
        self.assertEqual(run_arm(self.ctx, "armA"), 1)
        self.assertEqual(self.runner.calls, [])
        status = (self.run_root / "armA.pipeline_status.txt").read_text(encoding="utf-8")
        self.assertIn("already being trained", status)

    def test_queue_skips_verified_arms_but_not_paused_ones(self) -> None:
        self.write_arm_config("armA")
        self.write_arm_config("armB")
        self.plant_checkpoint("armA")
        self.plant_checkpoint("armB", step=5000)
        self.assertEqual(run_queue(self.ctx, ["armA", "armB"]), 0)
        log = self.config.queue_status_file().read_text(encoding="utf-8")
        self.assertIn("arm armA skip", log)
        self.assertIn("arm armB start", log)
        trained = [rest[1] for tool, rest in self.runner.calls if tool == "train_gsplat.py"]
        self.assertEqual(trained, [str(self.config.arm_config("armB"))], "a paused arm is re-run, not silently accepted")
        self.assertEqual(self.ctx.arm_job("armB").get("training")["completed_steps"], TARGET_STEPS)


class ConfigImmutabilityTests(PipelineFixture):
    def test_fault4_changed_config_under_same_arm_name_is_refused_with_artifacts_untouched(self) -> None:
        self.write_arm_config("armA")
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        out = self.config.arm_dir("armA")
        original = self.config.arm_config("armA").read_bytes()
        snapshot = {path.name: path.read_bytes() for path in out.rglob("*") if path.is_file()}
        self.write_arm_config("armA", {"arm": "armA", "controlled_stop_after_steps": TARGET_STEPS, "lr": 0.001})
        self.runner.calls.clear()
        self.assertEqual(run_arm(self.ctx, "armA"), 2)
        self.assertEqual(self.runner.calls, [])
        self.assertEqual((out / CONFIG_AS_RUN_NAME).read_bytes(), original)
        self.assertEqual((out / CONFIG_FROZEN_NAME).read_bytes(), original)
        after = {path.name: path.read_bytes() for path in out.rglob("*") if path.is_file()}
        changed = {name for name in snapshot if snapshot[name] != after.get(name)}
        self.assertEqual(changed - {"armA.pipeline_status.txt"}, set(), "old artifacts are not rewritten")
        self.assertEqual(self.ctx.arm_job("armA").state, STATE_EVALUATED, "the refusal is not a job failure")

    def test_refusal_exits_2_from_the_cli(self) -> None:
        self.write_arm_config("armA")
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        self.write_arm_config("armA", {"arm": "armA", "controlled_stop_after_steps": TARGET_STEPS, "lr": 0.001})
        for key in ("python", "reference_ply", "reference_alignment", "tile_inputs_manifest"):
            path = Path(self.raw_config[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
        (self.run_root / "delivery_eval.json").write_text("{}", encoding="utf-8")
        pipeline_json = self.base / "pipeline.json"
        pipeline_json.write_text(json.dumps(self.raw_config), encoding="utf-8")
        with open(os.devnull, "w", encoding="utf-8") as sink:
            saved = sys.stdout
            sys.stdout = sink
            try:
                code = main(["--pipeline-config", str(pipeline_json), "arm", "armA"])
            finally:
                sys.stdout = saved
        self.assertEqual(code, 2)
        self.assertEqual((self.config.arm_dir("armA") / CONFIG_AS_RUN_NAME).read_bytes(), json.dumps({"arm": "armA", "controlled_stop_after_steps": TARGET_STEPS, "max_steps": 49560}).encode())

    def test_legacy_run_is_frozen_from_its_config_as_run_record(self) -> None:
        self.write_arm_config("armA")
        self.plant_checkpoint("armA")
        out = self.config.arm_dir("armA")
        legacy = out / CONFIG_AS_RUN_NAME
        legacy.write_bytes(self.config.arm_config("armA").read_bytes())
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        self.assertEqual((out / CONFIG_FROZEN_NAME).read_bytes(), legacy.read_bytes())
        # An edited RUN/<arm>.json against a legacy record is refused too.
        self.write_arm_config("armB")
        self.plant_checkpoint("armB")
        (self.config.arm_dir("armB") / CONFIG_AS_RUN_NAME).write_text('{"arm": "armB", "old": true}', encoding="utf-8")
        with self.assertRaises(ConfigFrozenError):
            freeze_arm_config(self.config.arm_config("armB"), self.config.arm_dir("armB") / CONFIG_FROZEN_NAME, legacy_record=self.config.arm_dir("armB") / CONFIG_AS_RUN_NAME)
        self.assertEqual(run_arm(self.ctx, "armB"), 2)

    def test_same_config_rerun_is_not_refused(self) -> None:
        self.write_arm_config("armA")
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        self.assertEqual(run_arm(self.ctx, "armA"), 0)


# --------------------------------------------------------------------------
# GPU lease
# --------------------------------------------------------------------------

RACE_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    sys.path.insert(0, sys.argv[4])
    from tools.pipeline import acquire_gpu_lease, GpuLeaseBusy
    lock, go, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    print("READY", flush=True)
    while not go.exists():
        time.sleep(0.002)
    try:
        lease = acquire_gpu_lease(lock, command=["train", name], device="cuda:0", owner=name)
    except GpuLeaseBusy as error:
        print("BUSY", flush=True)
    else:
        print("ACQUIRED", flush=True)
        time.sleep(1.5)
        lease.release()
    """
)


class GpuLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.dir = Path(self._temporary.name)
        self.lock = self.dir / "gpu.lock"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_pid_liveness_probe(self) -> None:
        self.assertTrue(_pid_alive(os.getpid()))
        self.assertFalse(_pid_alive(_exited_pid()))
        self.assertFalse(_pid_alive(0))
        self.assertFalse(_pid_alive(-1))

    def test_fault5_two_processes_racing_for_the_lease_only_one_wins(self) -> None:
        go = self.dir / "go"
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", RACE_SCRIPT, str(self.lock), str(go), name, str(REPO_ROOT)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for name in ("cli_a", "cli_b")
        ]
        try:
            for proc in procs:
                self.assertEqual(proc.stdout.readline().strip(), "READY")
            go.write_text("go", encoding="utf-8")
            outcomes = []
            for proc in procs:
                out, err = proc.communicate(timeout=60)
                self.assertEqual(proc.returncode, 0, err)
                outcomes.append(out.strip().splitlines()[-1])
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
        self.assertEqual(sorted(outcomes), ["ACQUIRED", "BUSY"], outcomes)
        self.assertFalse(self.lock.exists(), "the winner released the lease on exit")

    def test_held_lease_is_busy_and_records_the_holder(self) -> None:
        lease = acquire_gpu_lease(self.lock, command=["python", "train_gsplat.py"], device="cuda:1", owner="train armA")
        holder = read_gpu_lease(self.lock)
        self.assertEqual(holder["pid"], os.getpid())
        self.assertEqual(holder["device"], "cuda:1")
        self.assertEqual(holder["owner"], "train armA")
        self.assertEqual(len(holder["command_sha256"]), 64)
        with self.assertRaises(GpuLeaseBusy) as caught:
            acquire_gpu_lease(self.lock, command=["other"], device="cuda:1", owner="second")
        self.assertIn(str(os.getpid()), str(caught.exception))
        lease.release()
        self.assertFalse(self.lock.exists())
        # Releasing twice, or releasing someone else's lease, is a no-op.
        lease.release()
        other = acquire_gpu_lease(self.lock, command=["x"], device="cuda:0")
        lease.release()
        self.assertTrue(self.lock.exists(), "a stale handle cannot release the new holder's lease")
        other.release()

    def test_stale_lease_from_a_dead_process_is_reclaimed(self) -> None:
        self.lock.write_text(json.dumps({"pid": _exited_pid(), "started_at_text": "2026-09-10 01:00:00", "owner": "crashed"}), encoding="utf-8")
        lease = acquire_gpu_lease(self.lock, command=["x"], device="cuda:0", owner="fresh")
        self.assertEqual(read_gpu_lease(self.lock)["owner"], "fresh")
        lease.release()

    def test_live_foreign_pid_is_not_reclaimed(self) -> None:
        self.lock.write_text(json.dumps({"pid": os.getppid(), "started_at_text": "now", "owner": "parent"}), encoding="utf-8")
        with self.assertRaises(GpuLeaseBusy):
            acquire_gpu_lease(self.lock, command=["x"], device="cuda:0", pid_alive=lambda pid: True)
        self.assertEqual(read_gpu_lease(self.lock)["owner"], "parent")

    def test_unreadable_fresh_lock_is_busy_old_garbage_is_reclaimed(self) -> None:
        self.lock.write_text("{half", encoding="utf-8")
        with self.assertRaises(GpuLeaseBusy):
            acquire_gpu_lease(self.lock, command=["x"], device="cuda:0")
        old = time.time() - 120
        os.utime(self.lock, (old, old))
        lease = acquire_gpu_lease(self.lock, command=["x"], device="cuda:0")
        lease.release()


def _exited_pid() -> int:
    """A pid that certainly belonged to a process which has already exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class GpuLeaseInPipelineTests(PipelineFixture):
    def test_train_step_refuses_while_the_lease_is_held(self) -> None:
        self.write_arm_config("armA")
        lease = acquire_gpu_lease(self.config.gpu_lock_file(), command=["foreign"], device="cuda:0", owner="other cli")
        try:
            reports = run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        finally:
            lease.release()
        self.assertEqual(reports[0].action, "failed")
        self.assertIn("GPU lease", reports[0].detail)
        self.assertEqual(self.runner.calls, [])
        self.assertIsNone(self.ctx.arm_job("armA").state, "no job state is written when the lease is refused")

    def test_delivery_gpu_steps_take_the_lease(self) -> None:
        self.write_arm_config("tile0_R1")
        self.plant_checkpoint("tile0_R1")
        for tile in (1, 2, 3):
            self.write_arm_config(self.config.delivery_tile_arm("r1d", tile))
            self.plant_checkpoint(self.config.delivery_tile_arm("r1d", tile))
        (self.run_root / "delivery_eval.json").write_text("{}", encoding="utf-8")
        self.assertEqual(run_deliver(self.ctx, "r1d", "tile0_R1"), 0)
        owners = self.ctx.gpu_leases_taken
        self.assertEqual(owners[0], "merge r1d")
        self.assertIn("pre_export battery r1d", owners)
        self.assertIn("final compare r1d", owners)
        self.assertIn("final offtraj r1d", owners)
        self.assertFalse(self.config.gpu_lock_file().exists())


# --------------------------------------------------------------------------
# Delivery: scores bound to the exported PLY
# --------------------------------------------------------------------------


class DeliveryScoringTests(PipelineFixture):
    TAG = "r1d"

    def setUp(self) -> None:
        super().setUp()
        self.write_arm_config("tile0_R1")
        self.plant_checkpoint("tile0_R1")
        for tile in (1, 2, 3):
            arm = self.config.delivery_tile_arm(self.TAG, tile)
            self.write_arm_config(arm)
            self.plant_checkpoint(arm)
        (self.run_root / "delivery_eval.json").write_text("{}", encoding="utf-8")
        self.out = self.config.delivery_dir(self.TAG)
        self.body = self.out / "house0305_r1d_merged.ply"

    def report(self) -> dict:
        return json.loads((self.out / "delivery_report.json").read_text(encoding="utf-8"))

    def calls_with(self, tool: str, needle: str) -> list[list[str]]:
        return [rest for name, rest in self.runner.calls if name == tool and any(needle in a for a in rest)]

    def test_fault6_final_scores_bind_to_the_exported_ply_sha256(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        report = self.report()
        sha = file_sha256(self.body)
        self.assertEqual(report["final"]["bound_to_ply_sha256"], sha)
        self.assertEqual(report["final"]["ply"]["sha256"], sha)
        self.assertEqual(report["final"]["ply"]["vertex_count"], 500, "1000 gaussians minus 0.05 * 10000")
        self.assertEqual(report["final"]["scored_checkpoint"], str(self.out / "reimported.pt"))
        self.assertEqual(report["pre_export"]["checkpoint"]["path"], str(self.out / "merged.pt"))
        self.assertEqual(report["publish"]["ply_sha256"], sha)
        self.assertEqual(report["publish"]["mode"], "candidate")
        # Final strips ran on the re-imported checkpoint; pre-export ones on merged.pt.
        reimport = self.calls_with("import_gaussian_ply.py", "house0305_r1d_merged.ply")
        self.assertEqual(reimport[0], ["--ply", str(self.body), "--output", str(self.out / "reimported.pt")])
        batteries = self.calls_with("evaluate_probe_views.py", "delivery_r1d")
        self.assertEqual([b[b.index("--checkpoint") + 1] for b in batteries], [str(self.out / "merged.pt"), str(self.out / "reimported.pt")])
        compares = self.calls_with("build_three_way_compare.py", "delivery_r1d")
        self.assertEqual([c[c.index("--checkpoint") + 1] for c in compares], [str(self.out / "merged.pt"), str(self.out / "reimported.pt")])
        offtrajs = self.calls_with("build_offtrajectory_compare.py", "delivery_r1d")
        self.assertEqual([o[1] for o in offtrajs], [str(self.out / "merged.pt"), str(self.out / "reimported.pt")])
        self.assertTrue((self.out / "scores_pre_export.txt").exists())
        self.assertTrue((self.out / "scores.txt").exists())
        self.assertEqual(self.ctx.delivery_job(self.TAG).state, STATE_QUALITY_ACCEPTED)

    def test_threshold_control_records_removed_counts_without_scoring_by_default(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        control = json.loads((self.out / "threshold_control" / "threshold_control.json").read_text(encoding="utf-8"))
        self.assertEqual([v["min_opacity"] for v in control["variants"]], list(EXPORT_THRESHOLD_CONTROL))
        self.assertEqual([v["vertex_count"] for v in control["variants"]], [1000, 900, 500])
        self.assertEqual([v["removed_vs_zero"] for v in control["variants"]], [0, 100, 500])
        self.assertFalse(control["variants_scored"])
        self.assertNotIn("battery", control["variants"][1])
        self.assertEqual(read_ply_vertex_count(Path(control["variants"][2]["path"])), 500)
        self.assertEqual(len(self.calls_with("evaluate_probe_views.py", "threshold_control")), 0)

    def test_threshold_variants_are_scored_only_on_request(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1", score_threshold_variants=True), 0)
        control = json.loads((self.out / "threshold_control" / "threshold_control.json").read_text(encoding="utf-8"))
        self.assertTrue(control["variants_scored"])
        self.assertEqual(len(self.calls_with("evaluate_probe_views.py", "threshold_control")), 3)
        self.assertTrue(all("battery" in v for v in control["variants"]))

    def test_rewritten_ply_invalidates_reimport_and_final_scores(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        old_sha = self.report()["final"]["bound_to_ply_sha256"]
        time.sleep(0.05)
        write_ply(self.body, 777)
        self.runner.calls.clear()
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        tools = self.runner.tools_called()
        self.assertIn("import_gaussian_ply.py", tools)
        self.assertNotIn("merge_v28_tile_checkpoints.py", tools)
        new_sha = self.report()["final"]["bound_to_ply_sha256"]
        self.assertNotEqual(new_sha, old_sha)
        self.assertEqual(new_sha, file_sha256(self.body))
        self.assertEqual(self.report()["final"]["ply"]["vertex_count"], 777)
        candidate = self.config.candidate_exports_dir(self.TAG) / "house0305_r1d_merged.ply"
        self.assertEqual(file_sha256(candidate), new_sha, "the candidate copy follows the re-scored PLY")

    def test_publish_flag_releases_into_exports_and_marks_published(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        self.assertFalse((self.exports / "house0305_r1d_merged.ply").exists())
        self.runner.calls.clear()
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1", publish=True), 0)
        self.assertEqual(self.runner.calls, [], "publishing a scored candidate re-runs nothing")
        self.assertEqual(file_sha256(self.exports / "house0305_r1d_merged.ply"), self.report()["final"]["bound_to_ply_sha256"])
        self.assertEqual((self.exports / "house0305_r1d_sky.ply").read_bytes(), b"sky")
        self.assertEqual(self.report()["publish"]["mode"], "published")
        self.assertEqual(self.ctx.delivery_job(self.TAG).state, STATE_PUBLISHED)

    def test_delivery_failure_is_recorded_on_its_job_state(self) -> None:
        self.runner.fail_on = {"import_gaussian_ply.py"}
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 1)
        job = self.ctx.delivery_job(self.TAG)
        self.assertEqual(job.state, STATE_FAILED)
        self.assertTrue(job.reason.startswith("reimport:"), job.reason)
        self.assertTrue(job.training_verified(), "the merge stays verified across a scoring failure")
        self.assertFalse(self.config.candidate_exports_dir(self.TAG).exists(), "nothing is published without bound scores")


if __name__ == "__main__":
    unittest.main()
