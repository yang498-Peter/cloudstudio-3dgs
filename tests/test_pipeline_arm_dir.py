"""arm_dir follows the arm config's output_dir.

The diagnostic arms write under RUN/diag_v2/<region>/runs/<arm>; the first
queue run assumed RUN/<arm>, declared every finished training "checkpoint
missing" and skipped its evaluation.
"""

from __future__ import annotations

import json
import unittest

from tests.test_pipeline_resume import PipelineFixture


class ArmDirTests(PipelineFixture):
    def test_output_dir_from_config_wins(self):
        target = self.run_root / "elsewhere" / "runs" / "armX"
        (self.run_root / "armX.json").write_text(json.dumps({"output_dir": str(target)}), encoding="utf-8")
        self.assertEqual(self.config.arm_dir("armX"), target)
        self.assertEqual(self.config.arm_checkpoint("armX"), target / "checkpoints" / "latest.pt")

    def test_fallback_without_config_or_output_dir(self):
        self.assertEqual(self.config.arm_dir("armY"), self.run_root / "armY")
        (self.run_root / "armZ.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
        self.assertEqual(self.config.arm_dir("armZ"), self.run_root / "armZ")


if __name__ == "__main__":
    unittest.main()
