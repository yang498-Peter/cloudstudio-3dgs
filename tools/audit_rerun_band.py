# -*- coding: utf-8 -*-
"""Paired rerun-noise band between two runs of the same configuration.

Panel medians are a fragile statistic: a 49-view distribution can shift its shape and move the
median by 17% while the paired per-view change is 2.5% and the sign test says 50/50. Arm verdicts
and delivery gates must therefore read the paired per-view median and the sign test, not the
panel median. This tool computes both, plus the morphology band, plus a negative control that the
photo and reference panels really are identical between the two runs (otherwise the comparison is
measuring the tooling, not the model).

usage:
  python tools/audit_rerun_band.py RUN_A RUN_B [--roi-name compare_roi_compare_ids]
                                   [--json OUT.json]

RUN_A / RUN_B are arm output directories holding <roi-name>.roi_bm.json and morph.txt.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

MORPH_TOKENS = (
    ("short_p50_mm", "short p50"),
    ("long_p50_mm", "long p50"),
    ("long_p95_mm", "long p95"),
    ("max_over_min_p50", "max/min p50"),
    ("max_over_mid_p50", "max/mid p50"),
    ("opacity_p50", "opacity p50"),
    ("opacity_frac_below_0p1", "frac<0.1"),
    ("long_gt_20mm", "long>20mm"),
)


def _read_roi(run: pathlib.Path, roi_name: str):
    path = run / (roi_name + ".roi_bm.json")
    if not path.is_file():
        raise SystemExit("missing ROI report: %s" % path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        doc = doc[0]
    frames = {f["sample_id"]: f for f in doc["frames"]}
    return doc, frames


def _read_morph(run: pathlib.Path):
    path = run / "morph.txt"
    if not path.is_file():
        return {}
    vals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("== ") and "N=" in line:
            vals["gaussian_count"] = float(line.split("N=")[1].split()[0].replace(",", ""))
        for key, token in MORPH_TOKENS:
            if key in vals:
                continue
            # "max/mid p50" also contains "mid p50", so anchor on the longest token that matches
            idx = line.find(token)
            if idx < 0:
                continue
            if token == "long p50" and "long p95" in line[:idx]:
                continue
            if token in ("short p50", "long p50") and line[max(0, idx - 4):idx].endswith("/"):
                continue
            tail = line[idx + len(token):].strip()
            tok = tail.split()[0].replace("mm", "") if tail else ""
            try:
                vals[key] = float(tok)
            except ValueError:
                pass
    return vals


def _rel(a, b):
    return (b - a) / a * 100.0 if a else float("nan")


def _pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=pathlib.Path)
    ap.add_argument("run_b", type=pathlib.Path)
    ap.add_argument("--roi-name", default="compare_roi_compare_ids")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    doc_a, fa = _read_roi(args.run_a, args.roi_name)
    doc_b, fb = _read_roi(args.run_b, args.roi_name)
    shared = sorted(set(fa) & set(fb))
    if not shared:
        raise SystemExit("the two runs share no ROI sample ids")

    # Negative control: the photo and reference panels must be bit-identical between the runs.
    control = {"shared_views": len(shared), "identical_boxes": 0,
               "max_abs_delta_lv_photo": 0.0, "max_abs_delta_lv_ref": 0.0}
    for k in shared:
        if fa[k]["box"] == fb[k]["box"]:
            control["identical_boxes"] += 1
        control["max_abs_delta_lv_photo"] = max(
            control["max_abs_delta_lv_photo"], abs(fa[k]["lv_photo"] - fb[k]["lv_photo"]))
        control["max_abs_delta_lv_ref"] = max(
            control["max_abs_delta_lv_ref"], abs(fa[k]["lv_ref"] - fb[k]["lv_ref"]))
    control["passed"] = (control["identical_boxes"] == len(shared)
                         and control["max_abs_delta_lv_photo"] == 0.0
                         and control["max_abs_delta_lv_ref"] == 0.0)

    deltas = sorted(_rel(fa[k]["ours_over_ref"], fb[k]["ours_over_ref"])
                    for k in shared if fa[k]["ours_over_ref"])
    wins = sum(1 for d in deltas if d > 0)
    paired = {
        "n": len(deltas),
        "median_pct": _pct(deltas, 0.5),
        "p10_pct": _pct(deltas, 0.10),
        "p90_pct": _pct(deltas, 0.90),
        "max_abs_pct": max(abs(d) for d in deltas),
        "mean_abs_pct": statistics.fmean(abs(d) for d in deltas),
        "stdev_pct": statistics.pstdev(deltas),
        "b_wins": wins,
        "sign_test_fraction": wins / len(deltas),
    }
    panel = {
        "a_ours_over_ref_median": doc_a["ours_over_ref_median"],
        "b_ours_over_ref_median": doc_b["ours_over_ref_median"],
        "panel_change_pct": _rel(doc_a["ours_over_ref_median"], doc_b["ours_over_ref_median"]),
    }
    ma, mb = _read_morph(args.run_a), _read_morph(args.run_b)
    morph = {k: {"a": ma[k], "b": mb[k], "change_pct": _rel(ma[k], mb[k])}
             for k in sorted(set(ma) & set(mb))}

    # A paired median inside the per-view spread with a near-even sign test is noise; a one-sided
    # sign test means the two runs really landed on different solutions.
    systematic = (paired["sign_test_fraction"] <= 0.25 or paired["sign_test_fraction"] >= 0.75)

    report = {
        "schema_version": 1,
        "kind": "cloudstudio_rerun_band",
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "roi_name": args.roi_name,
        "negative_control": control,
        "panel_statistic": panel,
        "paired_statistic": paired,
        "morphology": morph,
        "verdict": "systematic_difference" if systematic else "within_rerun_noise",
    }

    print("negative control: %s  (%d/%d identical boxes, max |dlv_photo| %.3g, max |dlv_ref| %.3g)"
          % ("PASS" if control["passed"] else "FAIL", control["identical_boxes"],
             len(shared), control["max_abs_delta_lv_photo"], control["max_abs_delta_lv_ref"]))
    if not control["passed"]:
        print("  the photo/reference panels differ between the runs; the comparison below is not"
              " a pure model comparison")
    print("panel median ours/ref : %.4f -> %.4f  (%+.1f%%)   <- fragile, do not rank on this"
          % (panel["a_ours_over_ref_median"], panel["b_ours_over_ref_median"],
             panel["panel_change_pct"]))
    print("paired per-view       : median %+.1f%%  p10 %+.1f%%  p90 %+.1f%%  |max| %.1f%%  stdev %.1f%%"
          % (paired["median_pct"], paired["p10_pct"], paired["p90_pct"],
             paired["max_abs_pct"], paired["stdev_pct"]))
    print("sign test             : %d/%d views favour run B (%.0f%%)"
          % (wins, paired["n"], 100 * paired["sign_test_fraction"]))
    if morph:
        print("morphology band:")
        for k, v in morph.items():
            print("  %-24s %14.4f -> %14.4f  (%+.2f%%)" % (k, v["a"], v["b"], v["change_pct"]))
    print("verdict: %s" % report["verdict"])

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
