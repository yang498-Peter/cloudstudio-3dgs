"""The queue must not skip an arm whose training is done but evaluation is not.

Three diagnostic arms finished training under a mis-tracking queue; a later
queue run skipped them as "training verified complete" and their strips and
scores were never produced.
"""

from __future__ import annotations

import json
import unittest

from tests.test_pipeline_resume import PipelineFixture
from tools.pipeline import run_queue


class QueueResumeTests(PipelineFixture):
    def test_verified_training_without_evaluation_is_not_skipped(self):
        self.write_arm_config("armQ", {"arm": "armQ", "max_steps": 3000})
        checkpoint = self.plant_checkpoint("armQ", 3000)
        run = checkpoint.parent.parent
        (run / "run_manifest.json").write_text(
            json.dumps({"training": {"status": "COMPLETE", "completed_steps": 3000}}), encoding="utf-8"
        )
        self.runner.calls.clear()
        run_queue(self.ctx, ["armQ"])
        status = self.config.queue_status_file().read_text(encoding="utf-8")
        self.assertNotIn("skip (", status, status)
        self.assertTrue(self.runner.calls, "evaluation steps must run for a trained arm")
        self.assertFalse(
            any("train_gsplat.py" in " ".join(map(str, call)) for call in self.runner.calls),
            "a verified training must not be retrained",
        )


if __name__ == "__main__":
    unittest.main()
