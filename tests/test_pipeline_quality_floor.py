"""The delivery gate can fail (tools/pipeline.py), CPU only.

Before this, QUALITY_ACCEPTED was reached by completing steps: the coverage number was recorded
and then accepted whatever it said, so a bad delivery passed as readily as a good one. The floor
is what turns the recorded number into a decision.

What is pinned:

* with no floor configured the behaviour is exactly what it was - the number is recorded, the
  delivery is accepted, and the report says the gate was recording only;
* with a floor, a delivered pair below it fails the step, does not reach QUALITY_ACCEPTED, and
  says both the measured value and the floor;
* a delivered pair at or above the floor is accepted and the report records which floor it
  cleared;
* the floor is validated as a number in [0, 1], because a nonsense floor would either block
  every delivery or none.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from pipeline import PipelineConfigError, parse_pipeline_config  # noqa: E402


class QualityFloorConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self._tmp.name).resolve()
        (self.base / "runs").mkdir()
        (self.base / "repo" / "tools").mkdir(parents=True)
        (self.base / "exports").mkdir()
        (self.base / "sky.ply").write_bytes(b"sky")
        self.addCleanup(self._tmp.cleanup)
        self.raw = {
            "run_root": str(self.base / "runs"),
            "repo_root": str(self.base / "repo"),
            "python": str(self.base / "python.exe"),
            "reference_ply": str(self.base / "ref.ply"),
            "reference_alignment": str(self.base / "align.json"),
            "tile_inputs_manifest": str(self.base / "tiles" / "manifest.json"),
            "tile_inputs_root": str(self.base / "tiles"),
            "exports_dir": str(self.base / "exports"),
            "delivery_eval_config": str(self.base / "runs" / "delivery_eval.json"),
            "sky_ply": str(self.base / "sky.ply"),
        }

    def test_no_floor_is_the_default(self):
        config = parse_pipeline_config(self.raw)
        self.assertIsNone(config.min_delivered_alpha_p05)

    def test_a_floor_is_accepted(self):
        config = parse_pipeline_config({**self.raw, "min_delivered_alpha_p05": 0.70})
        self.assertAlmostEqual(config.min_delivered_alpha_p05, 0.70)

    def test_a_floor_outside_zero_to_one_is_refused(self):
        for value in (-0.1, 1.5, "0.7", True):
            with self.subTest(value=value):
                with self.assertRaises(PipelineConfigError):
                    parse_pipeline_config({**self.raw, "min_delivered_alpha_p05": value})


try:
    from test_pipeline_resume import PipelineFixture
    from pipeline import STATE_QUALITY_ACCEPTED, deliver_steps, run_steps
except ImportError:  # pragma: no cover
    PipelineFixture = None


@unittest.skipUnless(PipelineFixture is not None, "needs the pipeline delivery fixture")
class QualityFloorGateTest(PipelineFixture):
    TAG = "floor"

    def setUp(self) -> None:
        super().setUp()
        self.write_arm_config("tile0_B5")
        self.plant_checkpoint("tile0_B5")
        for tile in (1, 2, 3):
            arm = self.config.delivery_tile_arm(self.TAG, tile)
            self.write_arm_config(arm)
            self.plant_checkpoint(arm)
        (self.run_root / "delivery_eval.json").write_text("{}", encoding="utf-8")

    def _deliver(self, floor=None):
        if floor is not None:
            self.config = parse_pipeline_config({**self.raw_config, "min_delivered_alpha_p05": floor})
            self.ctx = self.make_ctx()
        return run_steps(deliver_steps(self.ctx, self.TAG, "tile0_B5"), status=lambda _: None)

    def _report(self):
        import json
        path = self.config.delivery_dir(self.TAG) / "delivery_report.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_without_a_floor_the_delivery_is_accepted_and_marked_recorded_only(self):
        reports = self._deliver()
        self.assertFalse([r for r in reports if r.action == "failed"], reports)
        gate = self._report()["final"]["gate"]
        self.assertEqual(gate["verdict"], "recorded_only")
        self.assertIsNone(gate["min_delivered_alpha_p05"])
        self.assertEqual(self.ctx.delivery_job(self.TAG).state, STATE_QUALITY_ACCEPTED)

    def test_a_pair_below_the_floor_fails_and_is_not_accepted(self):
        # the fixture's pair battery reports the measured house0305 contrast, alpha p05 0.898
        reports = self._deliver(floor=0.95)
        failed = [r for r in reports if r.action == "failed"]
        self.assertTrue(failed, "a pair under the floor must fail the step")
        self.assertIn("0.95", failed[0].detail)
        self.assertNotEqual(self.ctx.delivery_job(self.TAG).state, STATE_QUALITY_ACCEPTED)

    def test_a_pair_above_the_floor_clears_it_and_says_so(self):
        reports = self._deliver(floor=0.70)
        self.assertFalse([r for r in reports if r.action == "failed"], reports)
        gate = self._report()["final"]["gate"]
        self.assertEqual(gate["min_delivered_alpha_p05"], 0.70)
        self.assertEqual(self.ctx.delivery_job(self.TAG).state, STATE_QUALITY_ACCEPTED)
        self.assertIn("floor 0.7", self.ctx.delivery_job(self.TAG).reason)


if __name__ == "__main__":
    unittest.main()
