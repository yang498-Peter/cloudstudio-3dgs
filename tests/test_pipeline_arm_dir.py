"""arm_dir follows the arm config's output_dir.

The diagnostic arms write under RUN/diag_v2/<region>/runs/<arm>; the first
queue run assumed RUN/<arm>, declared every finished training "checkpoint
missing" and skipped its evaluation.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_pipeline_resume import make_config  # shared fixture builder


class ArmDirTests(unittest.TestCase):
    def test_output_dir_from_config_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            target = Path(tmp) / "elsewhere" / "runs" / "armX"
            (cfg.run_root / "armX.json").write_text(json.dumps({"output_dir": str(target)}), encoding="utf-8")
            self.assertEqual(cfg.arm_dir("armX"), target)
            self.assertEqual(cfg.arm_checkpoint("armX"), target / "checkpoints" / "latest.pt")

    def test_fallback_without_config_or_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            self.assertEqual(cfg.arm_dir("armY"), cfg.run_root / "armY")
            (cfg.run_root / "armZ.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
            self.assertEqual(cfg.arm_dir("armZ"), cfg.run_root / "armZ")


if __name__ == "__main__":
    unittest.main()
