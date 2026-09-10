"""Queue behaviour of tools/pipeline.py: one trainer at a time, status log.

The process check is injected so the "trainer already running" guard can be
tested without starting anything.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from tests.test_pipeline_resume import PipelineFixture, _Sink
from tools.pipeline import (
    PipelineContext,
    _parse_process_listing,
    find_trainer_processes,
    run_queue,
    wait_for_arm,
)


class TrainerGuardTests(unittest.TestCase):
    def test_pattern_match_excludes_own_and_parent_process(self) -> None:
        listing = [
            (os.getpid(), "python tools/pipeline.py queue armA"),
            (os.getppid(), "cmd.exe /c python tools/train_gsplat.py --config x.json"),
            (4242, "C:\\venv\\python.exe tools\\train_gsplat.py --config RUN\\armB.json"),
            (17, "explorer.exe"),
        ]
        self.assertEqual(find_trainer_processes("train_gsplat.py", listing), [listing[2]])
        self.assertEqual(find_trainer_processes("nonexistent.py", listing), [])

    def test_parse_windows_and_posix_listings(self) -> None:
        windows = "1234|C:\\python.exe tools\\train_gsplat.py\n\n5|\nnot a pid|x\n"
        self.assertEqual(_parse_process_listing(windows), [(1234, "C:\\python.exe tools\\train_gsplat.py"), (5, "")])
        posix = "  77 python3 tools/train_gsplat.py --config a.json\n  1 /sbin/init\n"
        self.assertEqual(_parse_process_listing(posix), [(77, "python3 tools/train_gsplat.py --config a.json"), (1, "/sbin/init")])


class QueueTests(PipelineFixture):
    def queue_log(self) -> str:
        return self.config.queue_status_file().read_text(encoding="utf-8")

    def test_refuses_to_start_when_a_trainer_is_running(self) -> None:
        self.write_arm_config("armA")
        ctx = PipelineContext(
            self.config,
            run_command=self.runner,
            trainer_processes=lambda: [(4242, "python train_gsplat.py --config RUN/other.json")],
            stream=_Sink(),
        )
        self.assertEqual(run_queue(ctx, ["armA"]), 2)
        self.assertEqual(self.runner.calls, [], "no arm may start while another trainer holds the GPU")
        self.assertIn("REFUSED", self.queue_log())
        self.assertIn("4242", self.queue_log())

    def test_guard_is_rechecked_before_each_arm(self) -> None:
        self.write_arm_config("armA")
        self.write_arm_config("armB")
        seen: list[int] = []

        def processes() -> list[tuple[int, str]]:
            seen.append(1)
            # Free at the queue's initial check, at the check before armA and
            # inside armA's own train step; a foreign trainer appears before armB.
            return [] if len(seen) <= 3 else [(9, "python train_gsplat.py --config foreign.json")]

        ctx = PipelineContext(self.config, run_command=self.runner, trainer_processes=processes, stream=_Sink())
        self.assertEqual(run_queue(ctx, ["armA", "armB"]), 2)
        trained = [rest[1] for tool, rest in self.runner.calls if tool == "train_gsplat.py"]
        self.assertEqual(trained, [str(self.config.arm_config("armA"))])
        self.assertIn("arm armB REFUSED", self.queue_log())

    def test_runs_arms_sequentially_and_records_exits(self) -> None:
        self.write_arm_config("armA")
        self.write_arm_config("armB")
        self.assertEqual(run_queue(self.ctx, ["armA", "armB"]), 0)
        trained = [rest[1] for tool, rest in self.runner.calls if tool == "train_gsplat.py"]
        self.assertEqual(trained, [str(self.config.arm_config("armA")), str(self.config.arm_config("armB"))])
        log = self.queue_log()
        self.assertIn("[queue] start pid", log)
        self.assertLess(log.index("arm armA start"), log.index("arm armA exit 0"))
        self.assertLess(log.index("arm armA exit 0"), log.index("arm armB start"))
        self.assertIn("arm armB exit 0", log)
        self.assertTrue(log.rstrip().endswith("complete"))
        self.assertTrue((self.run_root / "armA.queue.log").exists())
        self.assertIn("ARM_DONE", (self.run_root / "armB.queue.log").read_text(encoding="utf-8"))

    def test_skips_arms_whose_checkpoint_exists_unless_forced(self) -> None:
        self.write_arm_config("armA")
        self.write_arm_config("armB")
        self.plant_checkpoint("armA")
        self.assertEqual(run_queue(self.ctx, ["armA", "armB"]), 0)
        trained = [rest[1] for tool, rest in self.runner.calls if tool == "train_gsplat.py"]
        self.assertEqual(trained, [str(self.config.arm_config("armB"))])
        self.assertIn("arm armA skip", self.queue_log())
        self.runner.calls.clear()
        self.assertEqual(run_queue(self.ctx, ["armA"], force=True), 0)
        trained = [rest[1] for tool, rest in self.runner.calls if tool == "train_gsplat.py"]
        self.assertEqual(trained, [str(self.config.arm_config("armA"))])

    def test_failed_arm_is_recorded_and_queue_continues(self) -> None:
        self.write_arm_config("armA")
        self.write_arm_config("armB")
        self.runner.fail_on = {"train_gsplat.py"}
        self.assertEqual(run_queue(self.ctx, ["armA", "armB"]), 1)
        log = self.queue_log()
        self.assertIn("arm armA exit 1", log)
        self.assertIn("arm armB exit 1", log)
        self.assertIn("complete", log)

    def test_delivery_runs_before_arms(self) -> None:
        self.write_arm_config("tile0_R1")
        self.plant_checkpoint("tile0_R1")
        for tile in (1, 2, 3):
            self.write_arm_config(self.config.delivery_tile_arm("r1d", tile))
        (self.run_root / "delivery_eval.json").write_text("{}", encoding="utf-8")
        self.write_arm_config("armZ")
        self.assertEqual(run_queue(self.ctx, ["armZ"], deliver=("r1d", "tile0_R1")), 0)
        log = self.queue_log()
        self.assertLess(log.index("delivery r1d start"), log.index("delivery r1d exit 0"))
        self.assertLess(log.index("delivery r1d exit 0"), log.index("arm armZ start"))
        self.assertTrue((self.run_root / "delivery_r1d.queue.log").exists())

    def test_wait_for_arm_ignores_old_success_lines(self) -> None:
        queue_file = self.config.queue_status_file()
        queue_file.write_text("x arm armA exit 0 old\nx arm armA exit 1 failed\n", encoding="utf-8")
        polls: list[float] = []

        def sleep(seconds: float) -> None:
            polls.append(seconds)
            if len(polls) == 2:
                with queue_file.open("a", encoding="utf-8") as handle:
                    handle.write("y arm armA exit 0 new\n")

        wait_for_arm(self.ctx, "armA", poll_seconds=7, sleep=sleep)
        self.assertEqual(polls, [7, 7])

    def test_wait_for_arm_then_waits_for_gpu(self) -> None:
        queue_file = self.config.queue_status_file()
        queue_file.write_text("", encoding="utf-8")
        gpu_checks: list[int] = []

        def processes() -> list[tuple[int, str]]:
            gpu_checks.append(1)
            return [(1, "train_gsplat.py")] if len(gpu_checks) < 3 else []

        ctx = PipelineContext(self.config, run_command=self.runner, trainer_processes=processes, stream=_Sink())
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) == 1:
                queue_file.write_text("z arm armA exit 0\n", encoding="utf-8")

        wait_for_arm(ctx, "armA", poll_seconds=1, sleep=sleep)
        self.assertEqual(len(gpu_checks), 3)
        self.assertEqual(len(sleeps), 3)


if __name__ == "__main__":
    unittest.main()
