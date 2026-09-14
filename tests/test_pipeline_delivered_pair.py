"""The delivery gate scores the delivered pair, not half of it (tools/pipeline.py), CPU only.

A delivery ships TWO files: the body PLY and a frozen sky PLY that publish copies beside it.
Nothing ever composited them, so ``evaluate_probe_views.py`` scored the body ALONE against
photographs that contain sky - and the body's sky regions are *correctly* transparent, because
the sky layer supplies them. The coverage metric read that correct transparency as a failure.

Measured on house0305's 48 battery views, body alone -> body plus sky:

    no-fill candidate   alpha p05  0.189 -> 0.898   (sharpness 0.454 -> 0.453, unchanged)
    sharpc1             alpha p05  0.722 -> 0.915
    sharpc0             alpha p05  0.825 -> 0.929

PSNR moved by at most 0.018, because the evaluator already composites a per-view backdrop
behind the render. Only alpha was ever penalised, and alpha is what the gate reads.

What is pinned:

* the ``pair`` step joins exactly two files - the re-imported body PLY and the configured
  ``sky_ply`` (itself re-imported), not merged.pt and not some other sky;
* BOTH batteries are recorded in ``delivery_report.json``, each labelled with the layers it
  scored, so a reader can tell the new number from the body-only one every historical report
  carries, and the gate's own reading names the pair;
* a missing or unreadable ``sky_ply`` FAILS the pair step instead of quietly falling back to
  a body-only score, which is the defect itself;
* morphology still runs on the body alone, because it is a shape comparison against a
  competitor's model that sky dome gaussians would distort.
"""

from __future__ import annotations

import json
import unittest

from tools.pipeline import (
    STATE_QUALITY_ACCEPTED,
    deliver_steps,
    run_deliver,
    run_steps,
)

from test_pipeline_resume import PipelineFixture


class DeliveredPairFixture(PipelineFixture):
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

    def report(self) -> dict:
        return json.loads((self.out / "delivery_report.json").read_text(encoding="utf-8"))

    def calls(self, tool: str) -> list[list[str]]:
        return [rest for name, rest in self.runner.calls if name == tool]


class PairIsBuiltFromTheDeliveredFilesTests(DeliveredPairFixture):
    def test_the_pair_joins_the_reimported_body_and_the_configured_sky(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        joins = self.calls("concat_delivery_layers.py")
        self.assertEqual(len(joins), 1, "one delivered pair per delivery")
        join = joins[0]
        self.assertEqual(join[join.index("--body") + 1], str(self.out / "reimported.pt"))
        self.assertEqual(join[join.index("--sky") + 1], str(self.out / "reimported_sky.pt"))
        self.assertEqual(join[join.index("--output") + 1], str(self.out / "delivery_pair.pt"))
        # The sky arm of the join is the configured sky PLY, re-imported the same way the
        # body is - the delivered pair is the two files the customer opens, nothing else.
        imports = {rest[rest.index("--output") + 1]: rest[rest.index("--ply") + 1] for rest in self.calls("import_gaussian_ply.py")}
        self.assertEqual(imports[str(self.out / "reimported_sky.pt")], str(self.sky))
        self.assertEqual(imports[str(self.out / "reimported.pt")], str(self.out / "house0305_r1d_merged.ply"))
        self.assertNotIn(str(self.out / "merged.pt"), join, "the pre-export merge is not the delivered body")

    def test_the_pair_is_rebuilt_when_the_body_is_reimported_again(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        (self.out / "reimported.pt").write_bytes(b"reimported:something else")
        self.runner.calls.clear()
        reports = run_steps(deliver_steps(self.ctx, self.TAG, "tile0_R1"), status=lambda _: None)
        done = [report.name for report in reports if report.action == "done"]
        self.assertIn("pair", done, "a pair bound to a stale body must not be reused")
        self.assertEqual(len(self.calls("concat_delivery_layers.py")), 1)


class BothBatteriesAreRecordedTests(DeliveredPairFixture):
    def test_the_report_keeps_both_readings_and_says_which_layers_each_scored(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        coverage = self.report()["final"]["coverage"]
        body, pair = coverage["body_only"], coverage["delivered_pair"]
        self.assertEqual(body["layers"], "body")
        self.assertEqual(pair["layers"], "body+sky")
        self.assertEqual(body["battery"], str(self.out / "battery_final.json"))
        self.assertEqual(pair["battery"], str(self.out / "battery_final_pair.json"))
        self.assertNotEqual(body["battery"], pair["battery"], "two batteries, two files")
        # The historical body-only number survives unchanged beside the new one.
        self.assertAlmostEqual(body["alpha_p05"], 0.189)
        self.assertAlmostEqual(pair["alpha_p05"], 0.898)
        self.assertEqual(json.loads((self.out / "battery_final.json").read_text(encoding="utf-8"))["alpha_p05"], body["alpha_p05"])
        self.assertEqual(json.loads((self.out / "battery_final_pair.json").read_text(encoding="utf-8"))["alpha_p05"], pair["alpha_p05"])

    def test_the_gate_reads_the_pair_and_names_the_body_only_number_beside_it(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        gate = self.report()["final"]["gate"]
        self.assertEqual(gate["reads"], "delivered_pair")
        self.assertAlmostEqual(gate["alpha_p05"], 0.898)
        self.assertAlmostEqual(gate["body_only_alpha_p05"], 0.189)
        job = self.ctx.delivery_job(self.TAG)
        self.assertEqual(job.state, STATE_QUALITY_ACCEPTED)
        self.assertIn("0.898", job.reason)
        self.assertIn("0.189", job.reason, "the verdict does not silently replace the historical number")
        log = (self.out / "deliver_status.txt").read_text(encoding="utf-8")
        self.assertIn("[coverage] delivered pair (body+sky) alpha p05 0.898; body alone 0.189", log)

    def test_the_two_batteries_score_two_different_checkpoints(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        final = [rest for rest in self.calls("evaluate_probe_views.py") if str(self.out) in rest[rest.index("--checkpoint") + 1]]
        scored = [rest[rest.index("--checkpoint") + 1] for rest in final]
        self.assertEqual(scored, [str(self.out / "merged.pt"), str(self.out / "reimported.pt"), str(self.out / "delivery_pair.pt")])
        pair_call = final[-1]
        self.assertEqual(pair_call[pair_call.index("--output") + 1], str(self.out / "battery_final_pair.json"))
        self.assertEqual(pair_call[pair_call.index("--views") + 1], "48", "the pair is scored on the same battery")
        self.assertEqual(pair_call[pair_call.index("--config") + 1], str(self.run_root / "delivery_eval.json"))

    def test_a_report_without_the_pair_reading_is_not_accepted_as_scored(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        payload = self.report()
        # A delivery_report.json from before this fix: bound to the right PLY, but its only
        # coverage reading is the body's.
        payload["final"].pop("coverage")
        (self.out / "delivery_report.json").write_text(json.dumps(payload), encoding="utf-8")
        self.runner.calls.clear()
        reports = run_steps(deliver_steps(self.ctx, self.TAG, "tile0_R1"), status=lambda _: None)
        self.assertIn("scores", [report.name for report in reports if report.action == "done"])
        self.assertEqual(self.report()["final"]["coverage"]["delivered_pair"]["layers"], "body+sky")


class MissingSkyFailsClosedTests(DeliveredPairFixture):
    def test_a_missing_sky_ply_fails_the_pair_step(self) -> None:
        self.sky.unlink()
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 1)
        job = self.ctx.delivery_job(self.TAG)
        self.assertIn("pair:", job.reason)
        self.assertIn("sky PLY missing", job.reason)
        self.assertEqual(self.calls("concat_delivery_layers.py"), [])
        self.assertFalse((self.out / "delivery_pair.pt").exists())
        # No body-only fallback: the gate never ran, so nothing was accepted.
        self.assertNotEqual(job.state, STATE_QUALITY_ACCEPTED)
        self.assertFalse((self.out / "battery_final_pair.json").exists())
        status = (self.out / "pipeline_status.txt").read_text(encoding="utf-8")
        self.assertIn("DELIVERY_FAILED at pair", status)

    def test_a_sky_path_that_is_not_a_readable_file_fails_the_pair_step(self) -> None:
        self.sky.unlink()
        self.sky.mkdir()  # something is at the configured path, but it is not the layer
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 1)
        job = self.ctx.delivery_job(self.TAG)
        self.assertIn("pair:", job.reason)
        self.assertEqual(self.calls("concat_delivery_layers.py"), [])

    def test_the_pair_battery_refuses_a_pair_that_no_longer_matches_the_sky(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        (self.out / "battery_final_pair.json").unlink()
        (self.out / "delivery_pair.pt").unlink()
        self.sky.unlink()
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 1)
        self.assertIn("sky PLY missing", self.ctx.delivery_job(self.TAG).reason)


class MorphologyStaysOnTheBodyTests(DeliveredPairFixture):
    def test_morphology_never_sees_the_sky_layer(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        morphs = [rest[0] for rest in self.calls("checkpoint_morphology.py") if str(self.out) in rest[0]]
        self.assertEqual(morphs, [str(self.out / "merged.pt"), str(self.out / "reimported.pt")])
        self.assertNotIn(str(self.out / "delivery_pair.pt"), morphs, "sky dome gaussians would distort the shape comparison")
        self.assertEqual(self.report()["final"]["morph"], str(self.out / "morph_final.txt"))

    def test_the_matched_strips_also_stay_on_the_body(self) -> None:
        self.assertEqual(run_deliver(self.ctx, self.TAG, "tile0_R1"), 0)
        compares = [rest[rest.index("--checkpoint") + 1] for rest in self.calls("build_three_way_compare.py") if str(self.out) in rest[rest.index("--checkpoint") + 1]]
        offtrajs = [rest[1] for rest in self.calls("build_offtrajectory_compare.py") if str(self.out) in rest[1]]
        for scored in (compares, offtrajs):
            self.assertEqual(scored, [str(self.out / "merged.pt"), str(self.out / "reimported.pt")])


if __name__ == "__main__":
    unittest.main()
