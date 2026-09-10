"""Resume logic of tools/pipeline.py with a fake step runner.

No torch, no GPU: the runner records which tool was invoked and plants the
artifact that tool would have written, so the tests can assert which steps
are skipped for a given set of existing artifacts.

The fake trainer behaves like the real one under a controlled stop: it
writes a zip checkpoint carrying ``step``, prints the
ControlledTrainingInterruption traceback to stderr and exits 1. Its
checkpoints are judged by ``inspect_checkpoint`` with a plain-pickle loader
so the tests do not depend on whether torch is installed on this host.
"""

from __future__ import annotations

import io
import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.pipeline import (
    CONFIG_AS_RUN_NAME,
    CONFIG_FROZEN_NAME,
    MIN_CHECKPOINT_BYTES,
    STATE_EVALUATED,
    STATE_QUALITY_ACCEPTED,
    STATE_TRAINING_COMPLETE,
    PipelineContext,
    Step,
    StepFailed,
    arm_steps,
    deliver_steps,
    inspect_checkpoint,
    parse_pipeline_config,
    plan_steps,
    run_arm,
    run_deliver,
    run_steps,
)

TARGET_STEPS = 20000


def _arg_after(argv: list[str], flag: str) -> Path:
    return Path(argv[argv.index(flag) + 1])


def write_fake_checkpoint(path: Path, step: int) -> None:
    """A torch-like zip: ``archive/data.pkl`` holding the payload dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "step": step, "params": {}, "identity": {"arm": path.parent.parent.name}}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(payload, protocol=2))
        archive.writestr("archive/data/0", b"\0" * MIN_CHECKPOINT_BYTES)
        archive.writestr("archive/version", b"3\n")
    path.write_bytes(buffer.getvalue())


def fake_payload_loader(path: Path) -> dict:
    """Stand-in for torch.load: unpickle data.pkl from the fake zip."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("archive/data.pkl") as handle:
            return pickle.load(handle)


def fixture_inspector(path: Path):
    return inspect_checkpoint(path, loader=fake_payload_loader)


def controlled_stop_traceback(step: int, checkpoint: Path) -> str:
    return (
        "Traceback (most recent call last):\n"
        '  File "tools/train_gsplat.py", line 21, in <module>\n'
        "    manifest = train_from_json(args.config)\n"
        f"cloudstudio_3dgs.training.trainer.ControlledTrainingInterruption: controlled interruption after {step} steps: {checkpoint}\n"
    )


def write_ply(path: Path, vertex_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {vertex_count}\nproperty float x\nend_header\n"
    path.write_bytes(header.encode("ascii") + b"\0" * (4 * vertex_count))


class FakeRunner:
    """Stands in for subprocess: records calls and fabricates artifacts.

    ``train_outcome`` selects the fault injected into the next training:
    ``controlled`` (default), ``oom``, ``crash``, ``garbage`` (unloadable
    checkpoint) or ``silent`` (exit 0, no marker). ``train_steps`` overrides
    the step the fake checkpoint stops at (default: the config's target).
    """

    def __init__(self, run_root: Path, *, fail_on: set[str] | None = None) -> None:
        self.run_root = run_root
        self.fail_on = fail_on or set()
        self.calls: list[tuple[str, list[str]]] = []
        self.train_outcome = "controlled"
        self.train_steps: int | None = None

    def tools_called(self) -> list[str]:
        return [tool for tool, _ in self.calls]

    def __call__(self, argv, *, cwd, env, stdout: Path, stderr, append: bool) -> int:
        tool = Path(argv[1]).name
        rest = [str(item) for item in argv[2:]]
        self.calls.append((tool, rest))
        stdout.parent.mkdir(parents=True, exist_ok=True)
        with stdout.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(f"{tool} ran\n")
        if tool in self.fail_on:
            return 1
        if tool == "train_gsplat.py":
            return self._train(_arg_after(rest, "--config"), stderr)
        if tool == "build_offtrajectory_compare.py":
            out = Path(rest[2])
            out.mkdir(parents=True, exist_ok=True)
            (out / "offtraj_summary.json").write_text("[]", encoding="utf-8")
        elif tool == "build_three_way_compare.py":
            out = _arg_after(rest, "--output")
            out.mkdir(parents=True, exist_ok=True)
            (out / "compare_summary.json").write_text("{}", encoding="utf-8")
        elif tool == "freeze_run_identity.py":
            out = _arg_after(rest, "--output")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}", encoding="utf-8")
        elif tool == "merge_v28_tile_checkpoints.py":
            _arg_after(rest, "--output-checkpoint").write_bytes(b"merged")
            _arg_after(rest, "--output-report").write_text("{}", encoding="utf-8")
        elif tool == "export_gaussian_ply.py":
            threshold = float(rest[rest.index("--min-opacity") + 1])
            # 1000 gaussians at threshold 0; 100 removed per 0.01 of opacity.
            write_ply(_arg_after(rest, "--output"), 1000 - int(round(threshold * 10000)))
        elif tool == "import_gaussian_ply.py":
            _arg_after(rest, "--output").write_bytes(b"reimported:" + _arg_after(rest, "--ply").read_bytes()[:64])
        elif tool == "evaluate_probe_views.py":
            _arg_after(rest, "--output").write_text("{}", encoding="utf-8")
        return 0

    def _train(self, arm_config: Path, stderr: Path | None) -> int:
        arm = arm_config.stem
        config = json.loads(arm_config.read_text(encoding="utf-8"))
        step = self.train_steps if self.train_steps is not None else int(config.get("controlled_stop_after_steps", TARGET_STEPS))
        checkpoints = self.run_root / arm / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoints / "latest.pt"
        outcome = self.train_outcome
        err_text = ""
        code = 1
        if outcome == "controlled":
            write_fake_checkpoint(checkpoint, step)
            (checkpoints / "step_000100.pt").write_bytes(b"stale")
            err_text = controlled_stop_traceback(step, checkpoint)
        elif outcome == "garbage":
            checkpoint.write_bytes(b"\xff" * (MIN_CHECKPOINT_BYTES * 2))
            err_text = controlled_stop_traceback(step, checkpoint)
        elif outcome == "oom":
            err_text = (
                "Traceback (most recent call last):\n  File \"trainer.py\", line 4000, in train\n"
                "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
            )
        elif outcome == "crash":
            err_text = "Traceback (most recent call last):\n  File \"trainer.py\", line 10\nKeyError: 'means'\n"
        elif outcome == "silent":
            write_fake_checkpoint(checkpoint, step)
            code = 0
        if stderr is not None and err_text:
            stderr.parent.mkdir(parents=True, exist_ok=True)
            stderr.write_text(err_text, encoding="utf-8")
        return code


class PipelineFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name).resolve()
        self.run_root = self.base / "runs"
        self.run_root.mkdir()
        self.repo_root = self.base / "repo"
        (self.repo_root / "tools").mkdir(parents=True)
        self.exports = self.base / "exports"
        self.exports.mkdir()
        self.sky = self.exports / "sky.ply"
        self.sky.write_bytes(b"sky")
        raw = {
            "run_root": str(self.run_root),
            "repo_root": str(self.repo_root),
            "python": str(self.base / "python.exe"),
            "reference_ply": str(self.base / "ref.ply"),
            "reference_alignment": str(self.base / "align.json"),
            "tile_inputs_manifest": str(self.base / "tiles" / "manifest.json"),
            "tile_inputs_root": str(self.base / "tiles"),
            "exports_dir": str(self.exports),
            "delivery_eval_config": str(self.run_root / "delivery_eval.json"),
            "sky_ply": str(self.sky),
            "delivery_baselines": {
                "compare": ["delivery_f6/compare_matched"],
                "offtraj": {"F6": "delivery_f6/offtraj_matched"},
            },
        }
        self.raw_config = raw
        self.config = parse_pipeline_config(raw)
        self.runner = FakeRunner(self.run_root)
        self.ctx = self.make_ctx()

    def make_ctx(self, **overrides) -> PipelineContext:
        kwargs = dict(run_command=self.runner, trainer_processes=lambda: [], checkpoint_inspector=fixture_inspector, stream=_Sink())
        kwargs.update(overrides)
        return PipelineContext(self.config, **kwargs)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def write_arm_config(self, arm: str, payload: dict | None = None) -> Path:
        path = self.config.arm_config(arm)
        payload = payload or {"arm": arm, "controlled_stop_after_steps": TARGET_STEPS, "max_steps": 49560}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def plant_checkpoint(self, arm: str, step: int = TARGET_STEPS) -> Path:
        """A trained arm from before job states: checkpoint plus trainer logs."""
        checkpoint = self.config.arm_checkpoint(arm)
        write_fake_checkpoint(checkpoint, step)
        log, err = self.config.arm_train_logs(arm)
        log.write_text("train_gsplat.py ran\nEXIT 1\n", encoding="utf-8")
        err.write_text(controlled_stop_traceback(step, checkpoint), encoding="utf-8")
        return checkpoint


class _Sink:
    def write(self, text: str) -> None:  # status lines are also on disk
        pass

    def flush(self) -> None:
        pass


ARM_STEPS = ["train", "prune_step_checkpoints", "config_as_run", "morph", "offtraj", "compare", "identity", "scores"]
DELIVERY_STEPS_AFTER_TILES = [
    "merge",
    "morph",
    "battery",
    "compare_matched",
    "offtraj_matched",
    "pre_export_scores",
    "export",
    "threshold_control",
    "reimport",
    "final_morph",
    "final_battery",
    "final_compare_matched",
    "final_offtraj_matched",
    "identity",
    "scores",
    "publish",
]


class ArmResumeTests(PipelineFixture):
    def executed(self, reports) -> list[str]:
        return [report.name for report in reports if report.action == "done"]

    def test_fresh_arm_runs_every_step_in_order(self) -> None:
        self.write_arm_config("armA")
        reports = run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        self.assertEqual(self.executed(reports), ARM_STEPS)
        self.assertEqual(
            self.runner.tools_called(),
            [
                "train_gsplat.py",
                "checkpoint_morphology.py",
                "build_offtrajectory_compare.py",
                "build_three_way_compare.py",
                "freeze_run_identity.py",
                "score_compare_sharpness.py",
                "score_offtrajectory_strips.py",
            ],
        )
        out = self.config.arm_dir("armA")
        self.assertFalse(list((out / "checkpoints").glob("step_*.pt")), "stale step checkpoints pruned")
        self.assertEqual((out / CONFIG_AS_RUN_NAME).read_bytes(), self.config.arm_config("armA").read_bytes())
        self.assertEqual((self.config.arm_meta_dir("armA") / CONFIG_FROZEN_NAME).read_bytes(), self.config.arm_config("armA").read_bytes())
        self.assertTrue((self.config.identity_dir / "armA.json").exists())
        scores = (out / "scores.txt").read_text(encoding="utf-8")
        self.assertIn("score_compare_sharpness.py ran", scores)
        self.assertIn("score_offtrajectory_strips.py ran", scores)
        self.assertIn("checkpoint_morphology.py ran", scores, "morph.txt is appended to the scores")
        ledger = self.config.arm_scores_file().read_text(encoding="utf-8")
        self.assertIn("[armA] train start", ledger)
        self.assertIn("[armA] train exit 1 done", ledger, "a controlled stop exits non-zero and still counts")
        self.assertIn("[armA] scores", ledger)
        job = self.ctx.arm_job("armA")
        self.assertEqual(job.state, STATE_EVALUATED)
        self.assertTrue(job.training_verified())

    def test_arm_commands_carry_config_paths(self) -> None:
        self.write_arm_config("armA")
        run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        calls = dict(self.runner.calls)
        self.assertEqual(calls["train_gsplat.py"], ["--config", str(self.config.arm_config("armA"))])
        offtraj = calls["build_offtrajectory_compare.py"]
        self.assertEqual(offtraj[:4], [str(self.config.arm_config("armA")), str(self.config.arm_checkpoint("armA")), str(self.config.arm_dir("armA") / "offtraj"), "6"])
        self.assertEqual(offtraj[4:], ["--reference-ply", str(self.config.reference_ply), "--reference-alignment", str(self.config.reference_alignment)])
        compare = calls["build_three_way_compare.py"]
        self.assertEqual(compare[compare.index("--frames") + 1], "6")
        self.assertEqual(compare[compare.index("--reference-ply") + 1], str(self.config.reference_ply))
        self.assertEqual(calls["score_offtrajectory_strips.py"], [f"armA={self.config.arm_dir('armA') / 'offtraj'}"])
        self.assertEqual(calls["checkpoint_morphology.py"], [str(self.config.arm_checkpoint("armA")), "--label", "armA"])
        self.assertEqual(self.runner.calls[0][0], "train_gsplat.py")
        self.assertTrue((self.run_root / "armA.log").exists())

    def test_second_run_skips_everything(self) -> None:
        self.write_arm_config("armA")
        run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        self.runner.calls.clear()
        reports = run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        self.assertEqual(self.executed(reports), [])
        self.assertEqual({report.action for report in reports}, {"skip"})
        self.assertEqual(self.runner.calls, [])

    def test_resumes_from_first_missing_artifact(self) -> None:
        self.write_arm_config("armA")
        run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        (self.config.arm_dir("armA") / "compare" / "compare_summary.json").unlink()
        self.runner.calls.clear()
        reports = run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        self.assertEqual(self.executed(reports), ["compare", "identity", "scores"])
        self.assertEqual(
            self.runner.tools_called(),
            ["build_three_way_compare.py", "freeze_run_identity.py", "score_compare_sharpness.py", "score_offtrajectory_strips.py"],
        )

    def test_trained_but_unscored_arm_does_not_retrain(self) -> None:
        self.write_arm_config("armA")
        self.plant_checkpoint("armA")
        reports = run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        self.assertEqual(self.executed(reports), ARM_STEPS[2:])
        self.assertNotIn("train_gsplat.py", self.runner.tools_called())
        job = self.ctx.arm_job("armA")
        self.assertTrue(job.get("adopted"), "a pre-job-state run is adopted after verification")
        self.assertTrue(job.training_verified())

    def test_plan_reports_without_running(self) -> None:
        self.write_arm_config("armA")
        self.plant_checkpoint("armA")
        plan = plan_steps(arm_steps(self.ctx, "armA"))
        self.assertEqual([(step.name, will_run) for step, will_run in plan][:3], [("train", False), ("prune_step_checkpoints", False), ("config_as_run", True)])
        self.assertTrue(all(will_run for step, will_run in plan[3:]))
        self.assertEqual(self.runner.calls, [])

    def test_edited_arm_config_is_refused_and_run_record_untouched(self) -> None:
        self.write_arm_config("armA")
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        original = self.config.arm_config("armA").read_bytes()
        self.write_arm_config("armA", {"arm": "armA", "controlled_stop_after_steps": TARGET_STEPS, "edited": True})
        self.runner.calls.clear()
        self.assertEqual(run_arm(self.ctx, "armA"), 2)
        self.assertEqual(self.runner.calls, [])
        out = self.config.arm_dir("armA")
        self.assertEqual((out / CONFIG_AS_RUN_NAME).read_bytes(), original)
        self.assertEqual((self.config.arm_meta_dir("armA") / CONFIG_FROZEN_NAME).read_bytes(), original)
        status = (self.run_root / "armA.pipeline_status.txt").read_text(encoding="utf-8")
        self.assertIn("ARM_REFUSED", status)
        self.assertIn("new arm name", status)

    def test_force_redoes_every_step(self) -> None:
        self.write_arm_config("armA")
        run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        self.runner.calls.clear()
        reports = run_steps(arm_steps(self.ctx, "armA"), force=True, status=lambda _: None)
        self.assertEqual(self.executed(reports), ARM_STEPS)
        self.assertEqual(self.runner.tools_called()[0], "train_gsplat.py")

    def test_step_failure_stops_the_arm_with_a_status_line(self) -> None:
        self.write_arm_config("armA")
        self.runner.fail_on = {"build_offtrajectory_compare.py"}
        lines: list[str] = []
        reports = run_steps(arm_steps(self.ctx, "armA"), status=lines.append)
        self.assertEqual(self.executed(reports), ARM_STEPS[:4])
        self.assertEqual(reports[-1].name, "offtraj")
        self.assertEqual(reports[-1].action, "failed")
        self.assertTrue(any(line.startswith("FAILED offtraj") for line in lines), lines)
        self.assertNotIn("build_three_way_compare.py", self.runner.tools_called())
        # Resume after the failure picks up at the failed step.
        self.runner.fail_on = set()
        self.runner.calls.clear()
        reports = run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        self.assertEqual(self.executed(reports), ["offtraj", "compare", "identity", "scores"])

    def test_failed_morph_capture_is_removed_so_resume_retries(self) -> None:
        self.write_arm_config("armA")
        self.runner.fail_on = {"checkpoint_morphology.py"}
        reports = run_steps(arm_steps(self.ctx, "armA"), status=lambda _: None)
        self.assertEqual(reports[-1].name, "morph")
        self.assertFalse((self.config.arm_dir("armA") / "morph.txt").exists())

    def test_training_failure_without_checkpoint(self) -> None:
        self.write_arm_config("armA")
        self.runner.fail_on = {"train_gsplat.py"}
        self.assertEqual(run_arm(self.ctx, "armA"), 1)
        ledger = self.config.arm_scores_file().read_text(encoding="utf-8")
        self.assertIn("[armA] TRAIN_FAILED exit 1", ledger)
        status = (self.run_root / "armA.pipeline_status.txt").read_text(encoding="utf-8")
        self.assertIn("ARM_FAILED at train", status)

    def test_missing_arm_config_fails_before_training(self) -> None:
        self.assertEqual(run_arm(self.ctx, "ghost"), 1)
        self.assertEqual(self.runner.calls, [])
        self.assertFalse(self.config.arm_dir("ghost").exists())

    def test_train_step_refuses_when_trainer_running(self) -> None:
        self.write_arm_config("armA")
        ctx = self.make_ctx(trainer_processes=lambda: [(4242, "python train_gsplat.py --config other.json")])
        lines: list[str] = []
        reports = run_steps(arm_steps(ctx, "armA"), status=lines.append)
        self.assertEqual(reports[0].action, "failed")
        self.assertIn("4242", reports[0].detail)
        self.assertEqual(self.runner.calls, [])
        self.assertFalse(self.config.gpu_lock_file().exists(), "the lease is released when the secondary scan refuses")

    def test_run_arm_returns_zero_and_writes_done_marker(self) -> None:
        self.write_arm_config("armA")
        self.assertEqual(run_arm(self.ctx, "armA"), 0)
        status = (self.run_root / "armA.pipeline_status.txt").read_text(encoding="utf-8")
        self.assertIn("ARM_DONE", status)


class DeliverResumeTests(PipelineFixture):
    TAG = "r1d"

    def setUp(self) -> None:
        super().setUp()
        self.write_arm_config("tile0_R1")
        self.plant_checkpoint("tile0_R1")
        for tile in (1, 2, 3):
            self.write_arm_config(self.config.delivery_tile_arm(self.TAG, tile))
        (self.run_root / "delivery_eval.json").write_text("{}", encoding="utf-8")

    def executed(self, reports) -> list[str]:
        return [report.name for report in reports if report.action == "done"]

    def test_fresh_delivery_trains_missing_tiles_then_merges(self) -> None:
        self.plant_checkpoint("tile2_r1d_20k")
        reports = run_steps(deliver_steps(self.ctx, self.TAG, "tile0_R1"), status=lambda _: None)
        self.assertEqual(self.executed(reports), ["train_tile1", "train_tile3", *DELIVERY_STEPS_AFTER_TILES])
        trained = [rest[1] for tool, rest in self.runner.calls if tool == "train_gsplat.py"]
        self.assertEqual(trained, [str(self.config.arm_config("tile1_r1d_20k")), str(self.config.arm_config("tile3_r1d_20k"))])
        out = self.config.delivery_dir(self.TAG)
        merge = dict(self.runner.calls)["merge_v28_tile_checkpoints.py"]
        self.assertIn(f"0={self.config.arm_checkpoint('tile0_R1')}", merge)
        self.assertIn(f"2={self.config.arm_checkpoint('tile2_r1d_20k')}", merge)
        self.assertIn("--harmonize-exposure", merge)
        self.assertEqual(merge[merge.index("--merge-policy") + 1], "core_owner_only")
        exports = [rest for tool, rest in self.runner.calls if tool == "export_gaussian_ply.py"]
        self.assertEqual(exports[0][exports[0].index("--min-opacity") + 1], "0.05")
        candidate = self.config.candidate_exports_dir(self.TAG)
        self.assertEqual((candidate / "house0305_r1d_merged.ply").read_bytes(), (out / "house0305_r1d_merged.ply").read_bytes())
        self.assertEqual((candidate / "house0305_r1d_sky.ply").read_bytes(), b"sky")
        self.assertFalse((self.exports / "house0305_r1d_merged.ply").exists(), "not published without --publish")
        batteries = [rest for tool, rest in self.runner.calls if tool == "evaluate_probe_views.py"]
        self.assertEqual(batteries[0][batteries[0].index("--views") + 1], "48")
        self.assertEqual(batteries[0][batteries[0].index("--config") + 1], str(self.run_root / "delivery_eval.json"))
        identity = dict(self.runner.calls)["freeze_run_identity.py"]
        self.assertEqual(identity[identity.index("--extra-file") + 1], str(out / "house0305_r1d_merged.ply"))
        self.assertTrue((self.config.identity_dir / "delivery_r1d_merged.json").exists())
        # The tile arms score themselves first; only the delivery-level calls matter here.
        sharps = [rest for tool, rest in self.runner.calls if tool == "score_compare_sharpness.py" and any("delivery_r1d" in a for a in rest)]
        self.assertEqual(sharps[0], [str(self.run_root / "delivery_f6" / "compare_matched"), str(out / "compare_matched")])
        self.assertEqual(sharps[1], [str(self.run_root / "delivery_f6" / "compare_matched"), str(out / "compare_final")])
        strips = [rest for tool, rest in self.runner.calls if tool == "score_offtrajectory_strips.py" and any("delivery_r1d" in a for a in rest)]
        self.assertEqual(strips[0], [f"F6={self.run_root / 'delivery_f6' / 'offtraj_matched'}", f"r1d={out / 'offtraj_matched'}"])
        self.assertEqual(strips[1], [f"F6={self.run_root / 'delivery_f6' / 'offtraj_matched'}", f"r1d={out / 'offtraj_final'}"])
        self.assertIn("checkpoint_morphology.py ran", (out / "morph.txt").read_text(encoding="utf-8"))
        # The tile arms were fully post-processed, not just trained.
        self.assertTrue((self.config.arm_dir("tile1_r1d_20k") / "scores.txt").exists())
        self.assertEqual(self.ctx.delivery_job(self.TAG).state, STATE_QUALITY_ACCEPTED)

    def test_delivery_status_log_matches_cmd_markers(self) -> None:
        for tile in (1, 2, 3):
            self.plant_checkpoint(self.config.delivery_tile_arm(self.TAG, tile))
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        log = (self.config.delivery_dir(self.TAG) / "deliver_status.txt").read_text(encoding="utf-8")
        for marker in ("[start] r1d delivery", "[merge]", "[scores pre-export]", "[threshold-control]", "[scores]", "[complete]", "[export] done"):
            self.assertIn(marker, log)
        self.assertNotIn("[train]", log)

    def test_delivery_resumes_after_battery(self) -> None:
        for tile in (1, 2, 3):
            self.plant_checkpoint(self.config.delivery_tile_arm(self.TAG, tile))
        run_steps(deliver_steps(self.ctx, self.TAG, "tile0_R1"), status=lambda _: None)
        (self.config.delivery_dir(self.TAG) / "battery.json").unlink()
        self.runner.calls.clear()
        reports = run_steps(deliver_steps(self.ctx, self.TAG, "tile0_R1"), status=lambda _: None)
        self.assertEqual(self.executed(reports), DELIVERY_STEPS_AFTER_TILES[2:])

    def test_delivery_second_run_skips_everything(self) -> None:
        for tile in (1, 2, 3):
            self.plant_checkpoint(self.config.delivery_tile_arm(self.TAG, tile))
        run_steps(deliver_steps(self.ctx, self.TAG, "tile0_R1"), status=lambda _: None)
        self.runner.calls.clear()
        reports = run_steps(deliver_steps(self.ctx, self.TAG, "tile0_R1"), status=lambda _: None)
        self.assertEqual(self.executed(reports), [])
        self.assertEqual(self.runner.calls, [])

    def test_tile_training_failure_stops_delivery(self) -> None:
        self.runner.fail_on = {"train_gsplat.py"}
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 1)
        log = (self.config.delivery_dir(self.TAG) / "deliver_status.txt").read_text(encoding="utf-8")
        self.assertIn("[train] tile1", log)
        self.assertIn("[FAIL] tile1 training", log)
        self.assertNotIn("merge_v28_tile_checkpoints.py", self.runner.tools_called())

    def test_missing_tile0_checkpoint_refuses(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_missing"), 1)
        self.assertEqual(self.runner.calls, [])

    def test_tile_arm_state_recorded_in_delivery(self) -> None:
        for tile in (1, 2, 3):
            self.plant_checkpoint(self.config.delivery_tile_arm(self.TAG, tile))
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        job = self.ctx.delivery_job(self.TAG)
        history = [entry["state"] for entry in job.get("history")]
        self.assertEqual(history[0], "RUNNING")
        self.assertIn(STATE_TRAINING_COMPLETE, history)
        self.assertEqual(job.get("training")["arms"]["tile0"]["arm"], "tile0_R1")
        self.assertEqual(job.get("training")["arms"]["tile2"]["completed_steps"], TARGET_STEPS)


class StepExecutorTests(unittest.TestCase):
    def test_non_anchor_steps_do_not_move_the_resume_point(self) -> None:
        done_a = [True]
        ran: list[str] = []
        steps = [
            Step("a", lambda: ran.append("a"), done=lambda: done_a[0]),
            Step("b", lambda: ran.append("b"), done=lambda: False, anchor=False),
            Step("c", lambda: ran.append("c"), done=lambda: True),
        ]
        reports = run_steps(steps, status=lambda _: None)
        self.assertEqual(ran, ["b"])
        self.assertEqual([(r.name, r.action) for r in reports], [("a", "skip"), ("b", "done"), ("c", "skip")])

    def test_independent_steps_anchor_but_never_repeat_when_done(self) -> None:
        ran: list[str] = []
        steps = [
            Step("tile1", lambda: ran.append("tile1"), done=lambda: False, independent=True),
            Step("tile2", lambda: ran.append("tile2"), done=lambda: True, independent=True),
            Step("merge", lambda: ran.append("merge"), done=lambda: True),
        ]
        run_steps(steps, status=lambda _: None)
        self.assertEqual(ran, ["tile1", "merge"], "a missing tile invalidates the merge; a done tile is left alone")
        ran.clear()
        run_steps(steps, force=True, status=lambda _: None)
        self.assertEqual(ran, ["tile1", "tile2", "merge"])

    def test_missing_artifact_after_run_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "never.txt"
            reports = run_steps([Step("x", lambda: None, artifacts=(artifact,))], status=lambda _: None)
        self.assertEqual(reports[0].action, "failed")
        self.assertIn("never.txt", reports[0].detail)

    def test_step_failed_message_is_reported(self) -> None:
        def boom() -> None:
            raise StepFailed("exit 3; see log")

        reports = run_steps([Step("x", boom, done=lambda: False)], status=lambda _: None)
        self.assertEqual(reports[0].detail, "exit 3; see log")


if __name__ == "__main__":
    unittest.main()
