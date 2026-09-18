"""tools/research_ladder.py: a ladder spec becomes single-change arm configs and paired verdicts.

CPU only, no trainer: ``write`` is tested against a small base config, ``score`` against a fake
runner that writes the JSON each tool would write. What is pinned:

* an arm's applied change must be exactly the keys it declares (a typo cannot become a second
  variable), dotted keys must exist in the base unless the arm allows new ones, lineage is
  explicit, writes are idempotent, and a changed arm under an existing name is refused;
* the paired statistic is per-item % change with a sign test, and the verdict reads it
  against the declared rerun band - never the panel median;
* scoring builds only what is missing, skips untrained arms, and the summary carries one
  verdict per arm.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from tools.research_ladder import (
    LadderError,
    call_paired,
    derive_arm_config,
    load_spec,
    paired_stats,
    score_ladder,
    verdict,
    write_arms,
)

BASE = OrderedDict(
    [
        ("run_id", "house0305-t1-base"),
        ("output_dir", "C:/runs/base"),
        ("sh_degree", 1),
        ("da2_depth_weight", 0.15),
        ("default_strategy", OrderedDict([("grow_grad2d", 5e-05), ("absgrad", False), ("refine_stop_iter", 14000)])),
        ("lineage", {"base": "older"}),
    ]
)


def spec_with(arms):
    return {"name": "LT", "base": {"name": "tile1_base", "dir": "C:/runs/base", "config": "C:/runs/base/config_as_run.json"},
            "run_id_prefix": "house0305-t1", "arms": arms}


class WriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.base_path = self.root / "base_config.json"
        self.base_path.write_text(json.dumps(BASE), encoding="utf-8")

    def test_a_single_dotted_change_yields_lineage_identity_and_nothing_else(self) -> None:
        spec = spec_with([{"arm": "tile1_LA_grow35_20k", "change": "grow 5e-5 -> 3.5e-5", "set": {"default_strategy.grow_grad2d": 3.5e-05}}])
        cfg = derive_arm_config(BASE, spec["arms"][0], spec=spec, run_root=self.root / "runs")
        self.assertEqual(cfg["default_strategy"]["grow_grad2d"], 3.5e-05)
        self.assertEqual(cfg["default_strategy"]["absgrad"], False)
        self.assertEqual(cfg["run_id"], "house0305-t1-tile1_LA_grow35_20k")
        self.assertEqual(cfg["output_dir"], str(self.root / "runs" / "tile1_LA_grow35_20k"))
        self.assertEqual(cfg["lineage"]["base"], "tile1_base")
        self.assertEqual(cfg["lineage"]["single_change"], "grow 5e-5 -> 3.5e-5")
        self.assertEqual(cfg["lineage"]["ladder"], "LT")

    def test_an_unknown_key_is_refused_unless_the_arm_allows_new_keys(self) -> None:
        arm = {"arm": "x", "change": "typo", "set": {"default_strategy.grow_grad2D": 1e-05}}
        with self.assertRaises(LadderError) as caught:
            derive_arm_config(BASE, arm, spec=spec_with([arm]), run_root=self.root)
        self.assertIn("does not exist", str(caught.exception))
        arm = {"arm": "x", "change": "new block", "allow_new": True, "set": {"rig_pose_refinement.enabled": True}}
        cfg = derive_arm_config(BASE, arm, spec=spec_with([arm]), run_root=self.root)
        self.assertTrue(cfg["rig_pose_refinement"]["enabled"])

    def test_the_diff_must_equal_the_declaration(self) -> None:
        # a spec cannot smuggle a change through the base: declare one key, the diff has one key
        arm = {"arm": "x", "change": "two knobs, one mechanism", "set": {"sh_degree": 0, "da2_depth_weight": 0.0}}
        cfg = derive_arm_config(BASE, arm, spec=spec_with([arm]), run_root=self.root)
        self.assertEqual((cfg["sh_degree"], cfg["da2_depth_weight"]), (0, 0.0))
        # setting a key to its base value is not a change; the discipline check catches it
        arm = {"arm": "y", "change": "no-op", "set": {"sh_degree": 1}}
        with self.assertRaises(LadderError) as caught:
            derive_arm_config(BASE, arm, spec=spec_with([arm]), run_root=self.root)
        self.assertIn("declared ['sh_degree']", str(caught.exception))

    def test_writes_are_idempotent_and_a_changed_arm_needs_a_new_name(self) -> None:
        spec = spec_with([{"arm": "tile1_LA_sh0_20k", "change": "sh0", "set": {"sh_degree": 0}}])
        runs = self.root / "runs"
        first = write_arms(spec, run_root=runs, base_config=self.base_path)
        self.assertEqual(first, [("tile1_LA_sh0_20k", "written", ["sh_degree"])])
        again = write_arms(spec, run_root=runs, base_config=self.base_path)
        self.assertEqual(again[0][1], "unchanged")
        spec["arms"][0]["set"] = {"sh_degree": 2}
        with self.assertRaises(LadderError):
            write_arms(spec, run_root=runs, base_config=self.base_path)
        planned = write_arms(spec, run_root=runs, base_config=self.base_path, dry_run=True)
        self.assertEqual(planned[0][1], "differs (would refuse)")
        written = json.loads((runs / "tile1_LA_sh0_20k.json").read_text(encoding="utf-8"))
        self.assertEqual(written["sh_degree"], 0, "the refused rewrite left the file alone")

    def test_spec_validation(self) -> None:
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"name": "x", "base": {}, "arms": [{"arm": "a", "set": {"k": 1}}]}), encoding="utf-8")
        with self.assertRaises(LadderError):
            load_spec(bad)  # no change sentence
        bad.write_text(json.dumps({"name": "x", "base": {}, "arms": [{"arm": "a", "change": "c", "set": {"k": 1}}, {"arm": "a", "change": "c", "set": {"k": 2}}]}), encoding="utf-8")
        with self.assertRaises(LadderError):
            load_spec(bad)  # duplicate arm


class PairedTests(unittest.TestCase):
    BAND = {"median_pct": 5.0, "sign_lo": 0.3, "sign_hi": 0.7}

    def rows(self, values):
        return {f"strip_{i}": {"sharp_ratio": v} for i, v in enumerate(values)}

    def test_paired_statistic_matches_items_by_name(self) -> None:
        a = self.rows([0.40, 0.50, 0.60, 0.20])
        b = self.rows([0.44, 0.55, 0.66, 0.22])
        b["extra"] = {"sharp_ratio": 9.0}  # unmatched, ignored
        stat = paired_stats(a, b, "sharp_ratio")
        self.assertEqual(stat["n"], 4)
        self.assertAlmostEqual(stat["median_pct"], 10.0, places=6)
        self.assertEqual(stat["b_wins"], 4)
        self.assertEqual(stat["sign_test_fraction"], 1.0)
        self.assertEqual(paired_stats({}, b, "sharp_ratio"), {"n": 0})

    def test_calls_read_the_band_not_the_panel(self) -> None:
        self.assertEqual(call_paired({"n": 18, "median_pct": 12.0, "sign_test_fraction": 0.83}, self.BAND), "better")
        self.assertEqual(call_paired({"n": 18, "median_pct": -9.0, "sign_test_fraction": 0.11}, self.BAND), "worse")
        # a big median with a split sign test is noise, exactly the rerun-band lesson
        self.assertEqual(call_paired({"n": 49, "median_pct": 13.0, "sign_test_fraction": 0.5}, self.BAND), "within band")
        self.assertEqual(call_paired({"n": 0}, self.BAND), "n/a")

    def test_verdict_composition(self) -> None:
        self.assertEqual(verdict({"roi": "better", "offtraj": "within band"}), "SHARPER (roi)")
        self.assertEqual(verdict({"roi": "worse", "offtraj": "n/a"}), "blurrier (roi)")
        self.assertTrue(verdict({"roi": "better", "offtraj": "worse"}).startswith("mixed"))
        self.assertEqual(verdict({"roi": "n/a", "offtraj": "n/a"}), "unscored")
        self.assertEqual(verdict({"roi": "within band", "offtraj": "within band"}, battery_delta_p10=-0.8), "within rerun band; battery p10 -0.80 dB")


class ScoreTests(unittest.TestCase):
    """The fake runner writes what each tool would; scoring must only ask for what is missing."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.runs = self.root / "runs"
        self.base = self.runs / "tile1_base"
        self.calls: list[str] = []
        for name, sharp in (("tile1_base", 0.40), ("tile1_LA_20k", 0.48), ("tile1_LB_20k", 0.40)):
            self.plant_arm(self.runs / name, sharp)
        (self.runs / "tile1_LC_20k").mkdir(parents=True)  # never trained: no checkpoint

    def plant_arm(self, arm_dir: Path, sharp: float) -> None:
        (arm_dir / "checkpoints").mkdir(parents=True)
        (arm_dir / "checkpoints" / "latest.pt").write_bytes(b"ckpt")
        (arm_dir / "config_as_run.json").write_text("{}", encoding="utf-8")
        (arm_dir / "offtraj").mkdir()
        (arm_dir / "offtraj" / "offtraj_summary.json").write_text("{}", encoding="utf-8")
        (arm_dir / "morph.txt").write_text(f"== {arm_dir.name}  step 20000  N=4,650,000\n  short p50 0.34mm [0.43]\n", encoding="utf-8")
        self.sharp = getattr(self, "sharp", {})
        self.sharp[arm_dir.name] = sharp

    def runner(self, argv, log):
        tool = Path(argv[1]).name
        self.calls.append(tool)
        argv = [str(a) for a in argv]
        if tool == "score_offtrajectory_strips.py":
            tag, directory = argv[2].split("=", 1)
            arm = Path(directory).parent.name
            rows = [{"file": f"offtraj_{i}.png", "psnr_q": 17.0, "sharp_ratio": self.sharp[arm] * (1 + 0.01 * i)} for i in range(18)]
            Path(argv[argv.index("--json") + 1]).write_text(json.dumps({tag: rows}), encoding="utf-8")
        elif tool == "evaluate_probe_views.py":
            out = Path(argv[argv.index("--output") + 1])
            arm = out.parent.name
            psnr = 16.0 + (0.3 if arm == "tile1_LA_20k" else 0.0)
            out.write_text(json.dumps({"psnr_mean": psnr, "psnr_p10": psnr - 1.5, "alpha_p05": 0.2}), encoding="utf-8")
            self.assertIn("--tile-owned", argv)
        return 0

    def spec(self):
        return {"name": "LT", "base": {"name": "tile1_base", "dir": str(self.base)},
                "arms": [{"arm": "tile1_LA_20k", "change": "sharper", "set": {"k": 1}},
                         {"arm": "tile1_LB_20k", "change": "same", "set": {"k": 2}},
                         {"arm": "tile1_LC_20k", "change": "never trained", "set": {"k": 3}}],
                "scoring": {"battery": {"views": 48, "tile_owned": True}, "offtraj": True,
                            "band": {"median_pct": 5.0, "sign_lo": 0.3, "sign_hi": 0.7}}}

    def test_scores_are_built_once_and_verdicts_are_paired(self) -> None:
        summary = score_ladder(self.spec(), run_root=self.runs, out_dir=self.root / "out", python="py", repo_root=self.root / "repo", run=self.runner)
        arms = summary["arms"]
        self.assertTrue(arms["tile1_LA_20k"]["verdict"].startswith("SHARPER (offtraj)"))
        self.assertAlmostEqual(arms["tile1_LA_20k"]["offtraj_sharp"]["median_pct"], 20.0, places=6)
        self.assertEqual(arms["tile1_LA_20k"]["offtraj_sharp"]["sign_test_fraction"], 1.0)
        self.assertAlmostEqual(arms["tile1_LA_20k"]["battery_delta"]["psnr_mean"], 0.3, places=6)
        self.assertEqual(arms["tile1_LB_20k"]["verdict"], "within rerun band")
        self.assertEqual(arms["tile1_LC_20k"]["verdict"], "not trained")
        self.assertTrue((self.root / "out" / "ladder_LT_summary.md").is_file())
        self.assertIn("| tile1_LA_20k |", summary["markdown"])
        calls_first = list(self.calls)
        self.assertEqual(calls_first.count("score_offtrajectory_strips.py"), 3, "base + two trained arms")
        self.assertEqual(calls_first.count("evaluate_probe_views.py"), 3)
        self.calls.clear()
        score_ladder(self.spec(), run_root=self.runs, out_dir=self.root / "out", python="py", repo_root=self.root / "repo", run=self.runner)
        self.assertEqual(self.calls, [], "everything existed: nothing is rebuilt")


if __name__ == "__main__":
    unittest.main()
