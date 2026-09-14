"""Paired rerun band (tools/audit_rerun_band.py), CPU only.

Synthetic ROI reports and morph.txt files. What is pinned:

* the panel median and the paired per-view median are reported separately, because a panel
  median can move far while the paired median says the two runs are indistinguishable - the
  case that made this tool necessary;
* an even sign test is reported as ``within_rerun_noise`` and a one-sided one as
  ``systematic_difference``;
* the negative control fails when the photo or reference panel differs between the runs, so a
  tooling change can never be read as a model change;
* ``max/mid p50`` is not mistaken for ``mid p50`` when morph.txt is parsed.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "audit_rerun_band.py"

MORPH = """== {arm}  step 20000  N={count}
  short p50 {short}mm [0.43]   mid p50 2.632mm [1.31]   long p50 5.099mm [4.41]  long p95 20.7mm [55.8]
  max/min p50 {maxmin} [10.2]   max/mid p50 1.87 [3.06]
  opacity p50 0.104 [0.197]  p95 0.965   frac<0.1 0.493 [0.18]   frac>0.9 0.099
  long>10mm 0.175  long>20mm 0.053  long>50mm 0.008
"""


def _frame(sample_id, ours, ref=40.0, photo=80.0, box=(1, 2, 3, 4)):
    return {
        "sample_id": sample_id,
        "file": sample_id + ".png",
        "box": list(box),
        "samples_in_crop": 400,
        "lv_photo": photo,
        "lv_ours": ours,
        "lv_ref": ref,
        "ours_over_photo": ours / photo,
        "ref_over_photo": ref / photo,
        "ours_over_ref": ours / ref,
    }


def _write_arm(root, arm, ours_values, count, short="0.342", maxmin="16.75",
               ref=40.0, photo=80.0, box=(1, 2, 3, 4)):
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    frames = [_frame("img_%02d" % i, v, ref=ref, photo=photo, box=box)
              for i, v in enumerate(ours_values)]
    ratios = sorted(f["ours_over_ref"] for f in frames)
    doc = {
        "arm": arm,
        "brightness_matched": True,
        "compare_dir": str(d),
        "n": len(frames),
        "n_skipped": 0,
        "skipped": [],
        "ours_over_ref_median": ratios[len(ratios) // 2],
        "ours_over_photo_median": sorted(f["ours_over_photo"] for f in frames)[len(frames) // 2],
        "ref_over_photo_median": sorted(f["ref_over_photo"] for f in frames)[len(frames) // 2],
        "frames": frames,
    }
    (d / "compare_roi_compare_ids.roi_bm.json").write_text(
        json.dumps([doc], ensure_ascii=False), encoding="utf-8")
    (d / "morph.txt").write_text(
        MORPH.format(arm=arm, count="{:,}".format(count), short=short, maxmin=maxmin),
        encoding="utf-8")
    return d


def _run(a, b, out=None):
    cmd = [sys.executable, str(TOOL), str(a), str(b)]
    if out is not None:
        cmd += ["--json", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError("tool failed: %s\n%s" % (proc.stdout, proc.stderr))
    return proc.stdout


class RerunBandTest(unittest.TestCase):
    def test_even_sign_test_is_noise_even_when_the_panel_median_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            # Half the views go up, half go down, so the paired median is near zero, but the two
            # halves are shaped so the panel median still shifts a long way.
            a_vals = [10.0] * 5 + [20.0] * 5
            b_vals = [18.0] * 5 + [11.0] * 5
            ra = _write_arm(root, "a", a_vals, 4654937)
            rb = _write_arm(root, "b", b_vals, 4658981, short="0.345", maxmin="16.55")
            out = root / "band.json"
            text = _run(ra, rb, out)
            rep = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(rep["negative_control"]["passed"])
            self.assertEqual(rep["paired_statistic"]["n"], 10)
            self.assertEqual(rep["paired_statistic"]["b_wins"], 5)
            self.assertEqual(rep["verdict"], "within_rerun_noise")
            # the panel statistic is reported but flagged, and it really did move
            self.assertNotAlmostEqual(rep["panel_statistic"]["panel_change_pct"], 0.0, places=3)
            self.assertIn("do not rank on this", text)

    def test_one_sided_change_is_called_systematic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            a_vals = [10.0 + i for i in range(10)]
            b_vals = [v * 0.8 for v in a_vals]
            ra = _write_arm(root, "a", a_vals, 3336404)
            rb = _write_arm(root, "b", b_vals, 3311214, short="0.312", maxmin="25.27")
            out = root / "band.json"
            _run(ra, rb, out)
            rep = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(rep["paired_statistic"]["b_wins"], 0)
            self.assertAlmostEqual(rep["paired_statistic"]["median_pct"], -20.0, places=6)
            self.assertEqual(rep["verdict"], "systematic_difference")

    def test_negative_control_fails_when_the_reference_panel_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            vals = [10.0] * 6
            ra = _write_arm(root, "a", vals, 1000, ref=40.0)
            rb = _write_arm(root, "b", vals, 1000, ref=41.0)
            out = root / "band.json"
            _run(ra, rb, out)
            rep = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(rep["negative_control"]["passed"])
            self.assertGreater(rep["negative_control"]["max_abs_delta_lv_ref"], 0.0)

    def test_morphology_parse_does_not_confuse_max_over_mid_with_mid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            vals = [10.0] * 6
            ra = _write_arm(root, "a", vals, 4654937)
            rb = _write_arm(root, "b", vals, 4658981)
            out = root / "band.json"
            _run(ra, rb, out)
            rep = json.loads(out.read_text(encoding="utf-8"))
            morph = rep["morphology"]
            self.assertAlmostEqual(morph["max_over_mid_p50"]["a"], 1.87, places=6)
            self.assertAlmostEqual(morph["max_over_min_p50"]["a"], 16.75, places=6)
            self.assertAlmostEqual(morph["short_p50_mm"]["a"], 0.342, places=6)
            self.assertAlmostEqual(morph["gaussian_count"]["a"], 4654937.0, places=6)
            self.assertAlmostEqual(morph["gaussian_count"]["change_pct"], 0.0868755, places=6)


if __name__ == "__main__":
    unittest.main()
