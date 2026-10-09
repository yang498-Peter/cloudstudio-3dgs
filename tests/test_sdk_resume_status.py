"""Resuming an interrupted delivery, and reading where a work root stands.

* A train step is done only when tools/pipeline.py verified its training: the trainer rewrites
  ``latest.pt`` every ``checkpoint_every`` steps, so a run killed at step 17k left a file the
  stage driver used to read as "outputs present" - and the merge would have taken it.
* ``status`` judges steps as the driver does and reads the trainer's own progress file.
* ``run`` records its invocation so ``run --work W`` alone continues it.
* The threshold control counts rows instead of writing three full PLYs.
* Fresh-ingest commands log per cache and run from the checkout.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path

from cloudstudio3dgs_sdk import __main__ as cli
from cloudstudio3dgs_sdk.bundle import CommandLogRunner, labelled
from cloudstudio3dgs_sdk.status import collect_status, render_status
from tests import test_sdk_project as _project_tests
from tools.pipeline import JOB_STATE_NAME, STATE_RUNNING, STATE_TRAINING_COMPLETE, JobState

ProjectFixture = _project_tests.ProjectFixture


def train_steps(project):
    return [step for step in project.plan().stage_steps("train") if step.pipeline_arm]


class TrainingCompletionTests(ProjectFixture):
    def prepared(self, **overrides):
        project = self.project(**overrides)
        self.seed_prepare_manifest(project)
        project.prepare()
        return project

    def test_every_pipeline_training_step_names_its_arm(self) -> None:
        project = self.prepared()
        steps = train_steps(project)
        self.assertTrue(steps)
        for step in steps:
            self.assertEqual(step.command[-1], step.pipeline_arm)
            self.assertNotIn("pipeline_arm", step.as_json(), "the arm is already in the hashed command")

    def test_a_leftover_checkpoint_is_not_a_finished_training(self) -> None:
        project = self.prepared(training_verifier=lambda arm: False)
        for step in train_steps(project):
            Path(step.outputs[0]).parent.mkdir(parents=True, exist_ok=True)
            Path(step.outputs[0]).write_bytes(b"step 17000 of 20000")
        self.runner.calls.clear()
        project.train()
        ran = set(self.runner.calls)
        for step in train_steps(project):
            self.assertIn(step.name, ran, "an unverified checkpoint must go back to the pipeline")

    def test_a_verified_training_is_skipped(self) -> None:
        project = self.prepared()
        project.train()
        self.runner.calls.clear()
        done, why = project.step_done(train_steps(project)[0])
        self.assertTrue(done)
        self.assertEqual(why, "training verified complete")

    def test_the_default_verifier_asks_the_pipeline(self) -> None:
        project = self.prepared(training_verifier=None)
        self.assertTrue(project.layout.pipeline_config.is_file())
        step = train_steps(project)[0]
        checkpoint = Path(step.outputs[0])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"not a checkpoint")
        done, why = project.step_done(step)
        self.assertFalse(done)
        self.assertIn("not verified", why)
        job = JobState(project.layout.runs / f"{step.pipeline_arm}.pipeline" / JOB_STATE_NAME, job="arm", name=step.pipeline_arm)
        job.set(
            STATE_TRAINING_COMPLETE,
            "verified",
            training={"verified": True, "checkpoint": {"bytes": checkpoint.stat().st_size}},
        )
        self.assertEqual(project.step_done(step), (True, "training verified complete"))


class ThresholdControlTests(ProjectFixture):
    def test_counts_are_recorded_without_control_plys(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.run_all()
        out = self.work / "runs" / "delivery_b5fill2" / "threshold_control"
        record = json.loads((out / "threshold_control.json").read_text(encoding="utf-8"))
        self.assertTrue(all(variant["counted_without_export"] for variant in record["variants"]))
        self.assertEqual(list(out.glob("*.ply")), [])
        step = next(s for s in project.plan().stage_steps("deliver") if s.name == "threshold_control")
        self.assertEqual(step.estimate.disk_bytes, 0)


class StatusTests(ProjectFixture):
    def test_a_finished_work_root_has_nothing_left(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.run_all()
        status = collect_status(self.project())
        self.assertTrue(status["plan_available"])
        for stage in status["stages"]:
            self.assertEqual(stage["steps_done"], stage["steps_total"], stage)
        self.assertEqual(status["remaining_seconds"], 0.0)
        self.assertEqual(status["arms"], [])

    def test_a_running_training_reports_its_step_rate_and_eta(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.prepare()
        step = train_steps(project)[0]
        target = int(step.config["controlled_stop_after_steps"]) if step.config else None
        now = time.time()
        job = JobState(project.layout.runs / f"{step.pipeline_arm}.pipeline" / JOB_STATE_NAME, job="arm", name=step.pipeline_arm)
        job.set(STATE_RUNNING, "trainer launched", pid=os.getpid(), started_at=now - 100)
        progress = project.layout.arm_dir(step.pipeline_arm) / "monitor" / "progress.jsonl"
        progress.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {"timestamp_unix": now - 500, "completed_steps": 100, "gaussian_count": 10},  # an earlier attempt
            {"timestamp_unix": now - 100, "completed_steps": 1000, "gaussian_count": 2_000_000},
            {"timestamp_unix": now - 10, "completed_steps": 1900, "gaussian_count": 2_500_000},
        ]
        progress.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        status = collect_status(project, now=now)
        arm = next(entry for entry in status["arms"] if entry["arm"] == step.pipeline_arm)
        self.assertTrue(arm["alive"])
        self.assertEqual(arm["completed_steps"], 1900)
        self.assertAlmostEqual(arm["steps_per_second"], 10.0)
        if target:
            self.assertAlmostEqual(arm["eta_seconds"], (target - 1900) / 10.0)
        text = render_status(status)
        self.assertIn(f"training {step.name}", text)
        self.assertIn("ETA", text)
        self.assertIn("2.50M gaussians", text)

    def test_status_before_prepare_says_why_there_is_no_plan(self) -> None:
        project = _project_tests.Project(self.dataset_root, self.work, _project_tests.PROFILE_B5FILL2, repo_root=self.repo)
        status = collect_status(project)
        self.assertFalse(status["plan_available"])
        self.assertIn("prepare", status["plan_note"])
        self.assertIn("NOT_STARTED", render_status(status))


class InvocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name) / "work"

    def test_a_recorded_run_fills_the_flags_a_later_one_omits(self) -> None:
        parser = cli.build_parser()
        first = parser.parse_args(
            ["run", "--dataset", "D:/capture", "--work", str(self.work), "--profile", "b12op05d3",
             "--pose-route", "raw_capture_poses", "--vram-gib", "15.9", "--da2-source", "D:/da2"]
        )
        cli.apply_invocation(first, io.StringIO())
        cli.record_invocation(first)
        again = parser.parse_args(["run", "--work", str(self.work)])
        out = io.StringIO()
        filled = cli.apply_invocation(again, out)
        self.assertEqual(again.profile, "b12op05d3")
        self.assertEqual(again.pose_route, "raw_capture_poses")
        self.assertEqual(again.vram_gib, 15.9)
        self.assertEqual(Path(again.dataset), Path("D:/capture").resolve())
        self.assertEqual(Path(again.da2_source), Path("D:/da2").resolve())
        self.assertIn("dataset", filled)
        self.assertIn("using the invocation recorded", out.getvalue())

    def test_an_explicit_flag_beats_the_record(self) -> None:
        parser = cli.build_parser()
        first = parser.parse_args(["run", "--dataset", "D:/capture", "--work", str(self.work), "--profile", "b12op05d3"])
        cli.apply_invocation(first, io.StringIO())
        cli.record_invocation(first)
        again = parser.parse_args(["run", "--work", str(self.work), "--profile", "b13op05d0"])
        cli.apply_invocation(again, io.StringIO())
        self.assertEqual(again.profile, "b13op05d0")

    def test_run_without_a_dataset_or_a_record_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            cli.main(["run", "--work", str(self.work)], stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)


class CommandLogRunnerTests(unittest.TestCase):
    def test_each_cache_gets_its_log_and_runs_from_the_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            runner = CommandLogRunner(root / "logs", repo_root=repo)
            code = labelled(runner, "val_face_cache")(
                [sys.executable, "-c", "import os, sys; print('cwd=' + os.getcwd()); sys.exit(3)"]
            )
            self.assertEqual(code, 3)
            log = (root / "logs" / "prepare_val_face_cache.log").read_text(encoding="utf-8")
            self.assertIn("$ ", log)
            self.assertIn(f"cwd={repo}", log)

    def test_a_plain_runner_is_left_alone(self) -> None:
        plain = lambda command: 0  # noqa: E731
        self.assertIs(labelled(plain, "anything"), plain)


if __name__ == "__main__":
    unittest.main()
