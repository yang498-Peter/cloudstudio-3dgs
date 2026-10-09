"""An interrupted training continues from its own checkpoint when the same command runs again.

The 2026-09-25 TDR bugcheck rebooted the host with Tile_3 at step 5000: the job stayed RUNNING,
the fresh relaunch met the leftover checkpoint and the trainer refused the non-empty output
directory, so the arm could only continue by hand as a new arm. These pin the rule that
replaces that: same frozen config, a dead previous attempt, a loadable checkpoint short of the
target and a failure a rerun can fix -> the pipeline relaunches with ``--resume-checkpoint`` and
records it; anything else trains as before.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.test_pipeline_resume import PipelineFixture, fixture_inspector, write_fake_checkpoint
from tools.pipeline import (
    CONFIG_FROZEN_NAME,
    STATE_EVALUATED,
    STATE_FAILED,
    STATE_RUNNING,
    file_sha256,
    freeze_arm_config,
    run_arm,
    training_resume_point,
)

DEAD_PID = 0  # _pid_alive(0) is False on every platform (tests/test_pipeline_state.py)


class PipelineResumeTests(PipelineFixture):
    def interrupted(self, arm: str = "armA", *, step: int = 5000, **job_fields) -> Path:
        """An arm whose previous attempt died at ``step``: frozen config, job, checkpoint, logs."""
        config = self.write_arm_config(arm)
        frozen = self.config.arm_meta_dir(arm) / CONFIG_FROZEN_NAME
        freeze_arm_config(config, frozen)
        checkpoint = self.config.arm_checkpoint(arm)
        write_fake_checkpoint(checkpoint, step)
        log, err = self.config.arm_train_logs(arm)
        log.write_text("first attempt\n", encoding="utf-8")
        err.write_text("first attempt stderr\n", encoding="utf-8")
        fields = {"pid": DEAD_PID, "started_at": 1.0, "config_sha256": file_sha256(frozen), "training": None}
        fields.update(job_fields)
        state = fields.pop("state", STATE_RUNNING)
        self.ctx.arm_job(arm).set(state, "previous attempt", **fields)
        return checkpoint

    def train_argv(self) -> list[str]:
        return next(rest for tool, rest in self.runner.calls if tool == "train_gsplat.py")

    def test_a_host_crash_resumes_from_the_arms_own_checkpoint(self) -> None:
        checkpoint = self.interrupted(step=5000)
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        argv = self.train_argv()
        self.assertIn("--resume-checkpoint", argv)
        self.assertEqual(Path(argv[argv.index("--resume-checkpoint") + 1]), checkpoint)
        job = self.ctx.arm_job("armA")
        self.assertEqual(job.state, STATE_EVALUATED, "the resumed training is verified like any other")
        self.assertEqual(job.get("resumed_from")["step"], 5000)
        self.assertEqual(job.get("resumed_from")["previous_state"], STATE_RUNNING)
        self.assertTrue(any("resumed at step 5000" in entry["reason"] for entry in job.get("history")))
        ledger = self.config.arm_scores_file().read_text(encoding="utf-8")
        self.assertIn("[armA] train resume", ledger)
        # The first attempt's logs are kept, and their tail no longer judges the new attempt.
        log, err = self.config.arm_train_logs("armA")
        kept = sorted(path.name for path in log.parent.glob("armA.log*before_resume_*"))
        self.assertEqual(len(kept), 2, kept)
        self.assertNotIn("first attempt", log.read_text(encoding="utf-8"))

    def test_a_trainer_crash_with_a_checkpoint_resumes(self) -> None:
        self.interrupted(step=10000, state=STATE_FAILED, training={"verified": False, "exit": {"kind": "crash"}})
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        self.assertIn("--resume-checkpoint", self.train_argv())

    def test_an_oom_is_retrained_not_resumed(self) -> None:
        # The same population hits the same wall; resuming would only replay the failure.
        self.interrupted(step=10000, state=STATE_FAILED, training={"verified": False, "exit": {"kind": "oom"}})
        run_arm(self.ctx, "armA")
        self.assertNotIn("--resume-checkpoint", self.train_argv())

    def test_a_fresh_arm_launches_without_a_resume(self) -> None:
        self.write_arm_config("armA")
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        self.assertNotIn("--resume-checkpoint", self.train_argv())
        self.assertIsNone(self.ctx.arm_job("armA").get("resumed_from"))


class OrphanedCompletionTests(PipelineFixture):
    """The pipeline died but its trainer ran to the end: verify what it left, do not retrain."""

    def orphan(self, *, step: int = 20000) -> Path:
        config = self.write_arm_config("armO")
        frozen = self.config.arm_meta_dir("armO") / CONFIG_FROZEN_NAME
        freeze_arm_config(config, frozen)
        self.ctx.arm_job("armO").set(
            STATE_RUNNING, "trainer launched", pid=DEAD_PID, started_at=time.time() - 3600,
            config_sha256=file_sha256(frozen), training=None,
        )
        return self.plant_checkpoint("armO", step)  # written after that start, with the trainer's log

    def test_a_finished_orphan_is_verified_not_retrained(self) -> None:
        self.orphan()
        self.assertEqual(run_arm(self.ctx, "armO"), 0)
        self.assertNotIn("train_gsplat.py", self.runner.tools_called())
        job = self.ctx.arm_job("armO")
        self.assertTrue(job.training_verified())
        self.assertTrue(any("after its pipeline exited" in entry["reason"] for entry in job.get("history")))

    def test_an_unfinished_orphan_resumes(self) -> None:
        self.orphan(step=15000)
        self.assertEqual(run_arm(self.ctx, "armO"), 0)
        argv = next(rest for tool, rest in self.runner.calls if tool == "train_gsplat.py")
        self.assertIn("--resume-checkpoint", argv)

    def test_a_live_trainer_is_never_read_under(self) -> None:
        # The orphan may still be writing latest.pt; reading it would block its os.replace.
        self.orphan(step=15000)
        reads: list[Path] = []

        def inspector(path: Path):
            reads.append(path)
            return fixture_inspector(path)

        ctx = self.make_ctx(trainer_processes=lambda: [(4242, "python train_gsplat.py --config armO.json")],
                            checkpoint_inspector=inspector)
        self.assertEqual(run_arm(ctx, "armO"), 1)
        self.assertEqual(reads, [])
        status = (self.config.run_root / "armO.pipeline_status.txt").read_text(encoding="utf-8")
        self.assertIn("another trainer is running", status)
        self.assertNotIn("train_gsplat.py", self.runner.tools_called())


class ResumePointRuleTests(PipelineFixture):
    def setUp(self) -> None:
        super().setUp()
        self.config_path = self.write_arm_config("armR")
        self.frozen = self.config.arm_meta_dir("armR") / CONFIG_FROZEN_NAME
        freeze_arm_config(self.config_path, self.frozen)
        self.checkpoint = self.config.arm_checkpoint("armR")
        self.job = self.ctx.arm_job("armR")

    def point(self, *, step: int = 5000, state: str = STATE_RUNNING, **fields):
        write_fake_checkpoint(self.checkpoint, step)
        record = {"pid": DEAD_PID, "config_sha256": file_sha256(self.frozen), "training": None}
        record.update(fields)
        self.job.set(state, "previous attempt", **record)
        return training_resume_point(
            self.job, checkpoint=self.checkpoint, arm_config=self.config_path, frozen=self.frozen,
            inspector=fixture_inspector,
        )

    def test_short_of_target_is_resumable(self) -> None:
        point = self.point(step=5000)
        self.assertIsNotNone(point)
        self.assertEqual((point.step, point.target), (5000, 20000))

    def test_at_or_past_the_target_is_not(self) -> None:
        self.assertIsNone(self.point(step=20000))

    def test_a_different_config_is_not(self) -> None:
        self.assertIsNone(self.point(config_sha256="0" * 64))

    def test_an_unloadable_checkpoint_is_not(self) -> None:
        self.job.set(STATE_RUNNING, "previous attempt", pid=DEAD_PID, config_sha256=file_sha256(self.frozen))
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_bytes(b"\xff" * 4096)
        self.assertIsNone(
            training_resume_point(
                self.job, checkpoint=self.checkpoint, arm_config=self.config_path, frozen=self.frozen,
                inspector=fixture_inspector,
            )
        )

    def test_a_live_holder_is_not(self) -> None:
        with mock.patch("tools.pipeline._pid_alive", return_value=True):
            self.assertIsNone(self.point(pid=4242))

    def test_no_job_record_is_not(self) -> None:
        write_fake_checkpoint(self.checkpoint, 5000)
        self.assertIsNone(
            training_resume_point(
                self.job, checkpoint=self.checkpoint, arm_config=self.config_path, frozen=self.frozen,
                inspector=fixture_inspector,
            )
        )


try:
    from cloudstudio_3dgs.training import trainer as trainer_module
except ImportError:  # torch is an optional training dependency
    trainer_module = None


@unittest.skipUnless(trainer_module is not None, "torch is an optional training dependency")
class TrainFromJsonResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_schedule_contract import _trainer_dict

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        payload = _trainer_dict()
        payload.pop("schedule_contract", None)
        payload.pop("schedule_contract_fields", None)
        self.payload = payload
        self.config_path = self.root / "arm.json"
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")

    def captured(self, **kwargs):
        seen = []
        with mock.patch.object(trainer_module, "train", side_effect=lambda config: seen.append(config) or {}):
            trainer_module.train_from_json(self.config_path, **kwargs)
        return seen[0]

    def test_the_launch_argument_sets_the_resume_without_touching_the_file(self) -> None:
        before = self.config_path.read_bytes()
        config = self.captured(resume_checkpoint=self.root / "latest.pt")
        self.assertEqual(config.resume_checkpoint, self.root / "latest.pt")
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertIsNone(self.captured().resume_checkpoint)

    def test_a_config_that_names_another_checkpoint_is_refused(self) -> None:
        self.config_path.write_text(json.dumps(dict(self.payload, resume_checkpoint="elsewhere.pt")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "second resume source"):
            self.captured(resume_checkpoint=self.root / "latest.pt")

    def test_the_cli_flag_reaches_train_from_json(self) -> None:
        from tools import train_gsplat

        calls = []

        def fake(path, *, resume_checkpoint=None):
            calls.append((path, resume_checkpoint))
            return {"run_id": "r", "training": {"completed_steps": 1, "peak_vram_bytes": 0}, "run_manifest_sha256": "x"}

        argv = ["train_gsplat.py", "--config", str(self.config_path), "--resume-checkpoint", "ck.pt"]
        with mock.patch.object(train_gsplat, "train_from_json", fake), mock.patch.object(sys, "argv", argv), \
                mock.patch("builtins.print"):
            self.assertEqual(train_gsplat.main(), 0)
        self.assertEqual(calls, [(self.config_path, Path("ck.pt"))])


if __name__ == "__main__":
    unittest.main()
