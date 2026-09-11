"""Natural completion is proven by the trainer's run_manifest.json.

Three diagnostic arms completed 3000 steps, then a later launch against
their non-empty output directories overwrote their logs with a traceback;
verification must still read COMPLETE from the manifest, and an adopted run
(no exit code) with a completion marker must classify as completed.
"""

from __future__ import annotations

import json
import unittest

from tests.test_pipeline_resume import PipelineFixture, fixture_inspector, write_fake_checkpoint
from tools.pipeline import EXIT_COMPLETED, classify_trainer_exit, verify_training


class RunManifestVerificationTests(PipelineFixture):
    def _arm_with_manifest(self, arm: str, steps: int, status: str = "COMPLETE"):
        self.write_arm_config(arm, {"arm": arm, "max_steps": 3000})
        run = self.config.arm_dir(arm)
        checkpoint = run / "checkpoints" / "latest.pt"
        write_fake_checkpoint(checkpoint, steps)
        (run / "run_manifest.json").write_text(
            json.dumps({"training": {"status": status, "completed_steps": steps}}), encoding="utf-8"
        )
        return checkpoint

    def test_manifest_complete_outranks_a_traceback_in_the_log(self):
        checkpoint = self._arm_with_manifest("armM", 3000)
        verdict = verify_training(
            checkpoint=checkpoint, arm_config=self.config.arm_config("armM"),
            log_tail="Traceback (most recent call last)\nFileExistsError: training output is not empty",
            exit_code=1, job_started_at=None, inspector=fixture_inspector,
        )
        self.assertTrue(verdict.complete, verdict.reason)

    def test_manifest_short_of_target_is_not_complete(self):
        checkpoint = self._arm_with_manifest("armS", 1500)
        verdict = verify_training(
            checkpoint=checkpoint, arm_config=self.config.arm_config("armS"),
            log_tail="", exit_code=None, job_started_at=None, inspector=fixture_inspector,
        )
        self.assertFalse(verdict.complete)

    def test_manifest_without_complete_status_is_ignored(self):
        checkpoint = self._arm_with_manifest("armI", 3000, status="RUNNING")
        verdict = verify_training(
            checkpoint=checkpoint, arm_config=self.config.arm_config("armI"),
            log_tail="Traceback (most recent call last)", exit_code=1, job_started_at=None, inspector=fixture_inspector,
        )
        self.assertFalse(verdict.complete)

    def test_completion_marker_counts_without_an_exit_code(self):
        exit = classify_trainer_exit(None, "training complete: run=x, steps=3000, peak_vram=1 bytes, sha256=abc")
        self.assertEqual(exit.kind, EXIT_COMPLETED)
        self.assertEqual(exit.steps, 3000)


if __name__ == "__main__":
    unittest.main()
