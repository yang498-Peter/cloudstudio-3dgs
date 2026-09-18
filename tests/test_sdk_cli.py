"""CLI argument parsing and the dry-run transcript.

``--dry-run`` is what a production team reads before committing a machine for
a day, so what it prints is part of the contract: every stage, every step,
the time and disk it will cost, the confidence behind each number, and any
step this checkout cannot run.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path

from cloudstudio3dgs_sdk.__main__ import (
    EXIT_OK,
    EXIT_REFUSED,
    build_parser,
    main,
    parse_prior_checkpoints,
    parse_stages,
)
from cloudstudio3dgs_sdk.profile import PROFILE_B5FILL2
from tests.test_sdk_plan import make_repo, two_tile_dataset


class StageParsingTests(unittest.TestCase):
    def test_canonical_order_is_restored(self) -> None:
        self.assertEqual(parse_stages("report,prepare"), ("prepare", "report"))

    def test_duplicates_collapse(self) -> None:
        self.assertEqual(parse_stages("train,train,prepare"), ("prepare", "train"))

    def test_whitespace_is_tolerated(self) -> None:
        self.assertEqual(parse_stages(" prepare , train "), ("prepare", "train"))

    def test_unknown_stage_is_rejected_with_the_known_list(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError) as caught:
            parse_stages("prepare,polish")
        self.assertIn("polish", str(caught.exception))
        self.assertIn("deliver", str(caught.exception))

    def test_empty_is_rejected(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_stages(" , ")


class PriorCheckpointParsingTests(unittest.TestCase):
    def test_pairs_become_a_tile_map(self) -> None:
        self.assertEqual(
            parse_prior_checkpoints(["0=a.pt", "3=D:/runs/t3/latest.pt"]),
            {0: "a.pt", 3: "D:/runs/t3/latest.pt"},
        )

    def test_empty_is_empty(self) -> None:
        self.assertEqual(parse_prior_checkpoints([]), {})

    def test_a_malformed_pair_is_rejected(self) -> None:
        for bad in ("nope", "0=", "x=a.pt"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_prior_checkpoints([bad])

    def test_a_repeated_tile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_prior_checkpoints(["1=a.pt", "1=b.pt"])


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_run_requires_dataset_and_work(self) -> None:
        args = self.parser.parse_args(["run", "--dataset", "d", "--work", "w"])
        self.assertEqual(args.command, "run")
        # The default is the recommended recipe (profile.DEFAULT_PROFILE): b5sky since the
        # 2026-09-15 finding that the fill layer answered a measurement defect. b5fill2 stays
        # registered for reproducing the campaign's own deliveries, but is no longer the default.
        self.assertEqual(args.profile, "b5sky")
        self.assertEqual(args.stages, ("prepare", "train", "deliver", "report"))
        self.assertFalse(args.dry_run)

    def test_missing_required_argument_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.parser.parse_args(["run", "--dataset", "d"])
        self.assertEqual(caught.exception.code, 2)

    def test_unknown_profile_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.parser.parse_args(["run", "--dataset", "d", "--work", "w", "--profile", "nope"])
        self.assertEqual(caught.exception.code, 2)

    def test_stage_subset_and_flags(self) -> None:
        args = self.parser.parse_args(
            ["run", "--dataset", "d", "--work", "w", "--dry-run", "--stages", "deliver,prepare", "--vram-gib", "16"]
        )
        self.assertTrue(args.dry_run)
        self.assertEqual(args.stages, ("prepare", "deliver"))
        self.assertEqual(args.vram_gib, 16.0)

    def test_no_subcommand_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.parser.parse_args([])
        self.assertEqual(caught.exception.code, 2)


class CliFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = make_repo(self.root / "repo")
        self.summary = self.root / "summary.json"
        self.summary.write_text(json.dumps(two_tile_dataset().as_json()), encoding="utf-8")

    def run_cli(self, *argv: str) -> tuple[int, str]:
        stream = io.StringIO()
        code = main(list(argv), stream=stream)
        return code, stream.getvalue()

    def dry_run(self, *extra: str) -> tuple[int, str]:
        return self.run_cli(
            "run",
            "--dataset", str(self.root / "dataset"),
            "--work", str(self.root / "work"),
            "--profile", "b5fill2",
            "--dry-run",
            "--summary", str(self.summary),
            "--repo-root", str(self.repo),
            "--python", "python.exe",
            *extra,
        )


class DryRunTests(CliFixture):
    def test_dry_run_prints_every_stage_and_a_total(self) -> None:
        code, text = self.dry_run()
        self.assertEqual(code, EXIT_OK)
        for fragment in ("[prepare]", "[train]", "[deliver]", "[report]", "total ", "peak disk"):
            self.assertIn(fragment, text)

    def test_dry_run_shows_the_profile_and_plan_identity(self) -> None:
        _, text = self.dry_run()
        self.assertIn(PROFILE_B5FILL2.profile_sha256, text)
        self.assertIn("plan_sha256", text)
        self.assertIn("profile=b5fill2@", text)

    def test_dry_run_shows_caps_and_confidence(self) -> None:
        _, text = self.dry_run()
        self.assertIn("cap 1.80M", text)   # 1.756 x 1.0M initialisation
        self.assertIn("cap 3.50M", text)   # 1.756 x 2.0M initialisation
        self.assertIn("[measured]", text)
        self.assertIn("[unmeasured]", text)

    def test_dry_run_prints_the_commands_it_would_run(self) -> None:
        _, text = self.dry_run()
        self.assertIn("build_sky_masks.py", text)
        self.assertIn("build_standin_backgrounds.py", text)
        self.assertIn("merge_v28_tile_checkpoints.py", text)
        self.assertIn("--fill-occupancy-voxel-m 0.2", text)

    def test_dry_run_names_the_seed_generation_when_there_are_no_priors(self) -> None:
        _, text = self.dry_run()
        self.assertIn("generations    seed, delivery", text)
        self.assertIn("train_tile0_b5fill2_seed", text)
        self.assertIn("WARNING", text)

    def test_prior_checkpoints_collapse_the_plan_to_one_generation(self) -> None:
        _, text = self.dry_run("--prior-checkpoint", "0=a.pt", "--prior-checkpoint", "1=b.pt")
        self.assertIn("generations    delivery", text)
        self.assertNotIn("_seed", text)

    def test_a_malformed_prior_checkpoint_is_a_usage_error(self) -> None:
        code, _ = self.dry_run("--prior-checkpoint", "nonsense")
        self.assertEqual(code, 2)

    def test_the_vram_ceiling_only_warns_when_it_actually_clamps(self) -> None:
        _, text = self.dry_run("--vram-gib", "24")
        self.assertNotIn("clamped", text)

    def test_dry_run_creates_nothing(self) -> None:
        self.dry_run()
        self.assertFalse((self.root / "work").exists())

    def test_dry_run_surfaces_a_checkout_that_cannot_run_the_recipe(self) -> None:
        crippled = make_repo(self.root / "crippled", fill_support=False)
        # Only the fill-enabled profile touches --fill-checkpoint; the default profile has the
        # fill layer off, so the block it is checking for would never be planned under it.
        code, text = self.run_cli(
            "run",
            "--profile", "b5fill2",
            "--dataset", str(self.root / "dataset"),
            "--work", str(self.root / "work"),
            "--dry-run",
            "--summary", str(self.summary),
            "--repo-root", str(crippled),
        )
        self.assertEqual(code, EXIT_OK)
        self.assertIn("BLOCKED merge_tiles", text)
        self.assertIn("--fill-checkpoint", text)

    def test_plan_json_is_written_when_asked(self) -> None:
        target = self.root / "out" / "plan.json"
        code, text = self.dry_run("--plan-json", str(target))
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(payload["profile_sha256"], PROFILE_B5FILL2.profile_sha256)
        self.assertTrue(payload["steps"])
        self.assertIn(str(target), text)

    def test_stage_subset_limits_the_transcript(self) -> None:
        _, text = self.dry_run("--stages", "prepare,train")
        self.assertIn("[prepare]", text)
        self.assertIn("[train]", text)
        self.assertNotIn("[deliver]", text)
        self.assertNotIn("[report]", text)

    def test_a_missing_summary_and_no_prepare_is_an_error(self) -> None:
        stream = io.StringIO()
        code = main(
            ["run", "--dataset", str(self.root / "dataset"), "--work", str(self.root / "empty"), "--dry-run"],
            stream=stream,
        )
        self.assertEqual(code, EXIT_REFUSED)


class PreflightCommandTests(CliFixture):
    def test_preflight_returns_one_on_a_host_that_cannot_run(self) -> None:
        code, text = self.run_cli(
            "preflight",
            "--dataset", str(self.root / "dataset"),
            "--work", str(self.root / "work"),
            "--summary", str(self.summary),
            "--repo-root", str(self.repo),
        )
        # This host has no CUDA device inside the test process.
        self.assertEqual(code, EXIT_REFUSED)
        self.assertIn("preflight", text)
        self.assertIn("gpu", text)


class ProfileCommandTests(CliFixture):
    def test_profile_prints_sha_provenance_and_open_questions(self) -> None:
        code, text = self.run_cli("profile", "b5fill2")
        self.assertEqual(code, EXIT_OK)
        self.assertIn(PROFILE_B5FILL2.profile_sha256, text)
        self.assertIn("[measured", text)
        self.assertIn("[unmeasured", text)
        self.assertIn("backdrop-bootstrap", text)
        self.assertIn("pipeline-fill-passthrough", text)

    def test_profile_json_is_canonical_and_parseable(self) -> None:
        code, text = self.run_cli("profile", "b5fill2", "--json")
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(text)
        self.assertEqual(payload["name"], "b5fill2")
        self.assertIn("trainer_base", payload)

    def test_profile_without_a_name_lists_every_profile(self) -> None:
        code, text = self.run_cli("profile")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("b5fill2@", text)


if __name__ == "__main__":
    unittest.main()
