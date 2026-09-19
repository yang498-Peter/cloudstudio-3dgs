#!/usr/bin/env python3
"""A research ladder: single-change arms from one base, queued, then judged in pairs.

The campaign's way of learning is a ladder - each arm is the base config plus exactly one
declared change (ladder_explicit_lineage: cumulative mutation once let a rejected arm leak
into every later arm) - and arm verdicts read paired per-view statistics against the base,
never a panel median (audit_rerun_band: a 49-view median can move 17% while the paired
change is 2.5%). Until now every ladder was a hand-written scratch script. This tool makes
the loop a spec file:

    python tools/research_ladder.py write  --spec L28.json --run-root RUNS
    python tools/research_ladder.py queue  --spec L28.json --pipeline-config P.json   (prints or runs)
    python tools/research_ladder.py score  --spec L28.json --run-root RUNS --out-dir OUT

``write`` refuses an arm whose applied change touches any key other than the ones it declares
(so a spec typo cannot silently become a second variable), and refuses dotted keys that do not
exist in the base unless the arm says ``"allow_new": true``. ``score`` builds what is missing
for the base and every trained arm (ROI three-way compare + brightness-matched ROI sharpness,
tile-owned battery, off-trajectory strip scores), then writes a summary with one verdict per arm
from the paired statistics and the rerun band the spec declares.

Spec (JSON)::

    {
      "name": "L28",
      "base": {"name": "tile1_b5sky_delivery", "dir": "C:/.../tile1_b5sky_delivery"},
      "run_id_prefix": "house0305-t1",
      "arms": [
        {"arm": "tile1_L1_grow35_20k", "change": "grow_grad2d 5e-5 -> 3.5e-5",
         "set": {"default_strategy.grow_grad2d": 0.000035}}
      ],
      "scoring": {
        "roi": {"name": "compare_roi_compare_ids", "config": "...diag.json",
                "sample_ids": "...roi_compare_ids_f3.json", "selection": "...selection.json",
                "min_samples": 100},
        "battery": {"views": 48, "tile_owned": true},
        "reference": {"ply": "...USAgs.ply", "alignment": "...alignment.json"},
        "band": {"median_pct": 5.0, "sign_lo": 0.3, "sign_hi": 0.7}
      }
    }
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# spec / configs
# --------------------------------------------------------------------------


class LadderError(RuntimeError):
    pass


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "base", "arms"):
        if key not in spec:
            raise LadderError(f"{path}: spec needs '{key}'")
    if not isinstance(spec["arms"], list) or not spec["arms"]:
        raise LadderError(f"{path}: 'arms' must be a non-empty list")
    names = [arm.get("arm") for arm in spec["arms"]]
    if len(set(names)) != len(names) or not all(names):
        raise LadderError(f"{path}: every arm needs a unique 'arm' name")
    for arm in spec["arms"]:
        if not isinstance(arm.get("set"), Mapping) or not arm["set"]:
            raise LadderError(f"{path}: arm {arm.get('arm')} needs a non-empty 'set' mapping")
        if not arm.get("change"):
            raise LadderError(f"{path}: arm {arm.get('arm')} needs a 'change' sentence")
    return spec


def flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            out.update(flatten(value, f"{prefix}{key}."))
        return out
    out[prefix[:-1]] = payload
    return out


def set_dotted(config: dict[str, Any], dotted: str, value: Any, *, allow_new: bool) -> None:
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node:
            if not allow_new:
                raise LadderError(f"key '{dotted}' does not exist in the base (set allow_new to create it)")
            node[part] = OrderedDict()
        node = node[part]
        if not isinstance(node, dict):
            raise LadderError(f"key '{dotted}': '{part}' is not an object in the base")
    if parts[-1] not in node and not allow_new:
        raise LadderError(f"key '{dotted}' does not exist in the base (set allow_new to create it)")
    node[parts[-1]] = value


def derive_arm_config(base: Mapping[str, Any], arm: Mapping[str, Any], *, spec: Mapping[str, Any], run_root: Path) -> OrderedDict:
    cfg = json.loads(json.dumps(base), object_pairs_hook=OrderedDict)
    allow_new = bool(arm.get("allow_new", False))
    for dotted, value in arm["set"].items():
        set_dotted(cfg, dotted, value, allow_new=allow_new)
    name = str(arm["arm"])
    prefix = str(spec.get("run_id_prefix") or spec["name"])
    cfg["run_id"] = f"{prefix}-{name}"
    cfg["output_dir"] = str(Path(run_root) / name)
    cfg["lineage"] = OrderedDict(
        [
            ("base", str(spec["base"].get("name", ""))),
            ("base_config", str(spec["base"].get("config", ""))),
            ("single_change", str(arm["change"])),
            ("ladder", str(spec["name"])),
        ]
    )
    # Fail closed on the discipline itself: the diff must be exactly the declared keys.
    before, after = flatten(base), flatten(cfg)
    touched = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    identity = {"run_id", "output_dir"}
    touched = [k for k in touched if k not in identity and not k.startswith("lineage.")]
    declared = sorted(arm["set"])
    if touched != declared:
        raise LadderError(f"arm {name}: applied change touched {touched}, declared {declared}")
    return cfg


def write_arms(spec: Mapping[str, Any], *, run_root: Path, base_config: Path, dry_run: bool = False) -> list[tuple[str, str, list[str]]]:
    base = json.loads(Path(base_config).read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    results = []
    for arm in spec["arms"]:
        cfg = derive_arm_config(base, arm, spec=spec, run_root=run_root)
        out = Path(run_root) / f"{arm['arm']}.json"
        text = json.dumps(cfg, indent=1, ensure_ascii=False) + "\n"
        if out.is_file() and out.read_text(encoding="utf-8") == text:
            action = "unchanged"
        elif out.is_file():
            # A config that already exists with different content belongs to an arm that may
            # have trained: the pipeline freezes configs, so a changed arm is a new arm name.
            if not dry_run:
                raise LadderError(f"{out} exists with different content; a changed arm needs a new name")
            action = "differs (would refuse)"
        else:
            action = "planned" if dry_run else "written"
            if not dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8", newline="\n")
        results.append((str(arm["arm"]), action, sorted(arm["set"])))
    return results


# --------------------------------------------------------------------------
# paired statistics and verdicts
# --------------------------------------------------------------------------


def paired_stats(rows_a: Mapping[str, Mapping[str, Any]], rows_b: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, Any]:
    """Paired per-item % change b vs a, sign test, and both medians. Items are matched by name."""
    common = sorted(set(rows_a) & set(rows_b))
    pct: list[float] = []
    wins = 0
    for name in common:
        a, b = float(rows_a[name][key]), float(rows_b[name][key])
        if a > 0:
            pct.append((b - a) / a * 100.0)
        if b > a:
            wins += 1
    if not common or not pct:
        return {"n": 0}
    pct.sort()
    return {
        "n": len(common),
        "median_pct": statistics.median(pct),
        "p10_pct": pct[int(0.1 * (len(pct) - 1))],
        "p90_pct": pct[int(0.9 * (len(pct) - 1))],
        "b_wins": wins,
        "sign_test_fraction": wins / len(common),
        "a_median": statistics.median(float(rows_a[n][key]) for n in common),
        "b_median": statistics.median(float(rows_b[n][key]) for n in common),
    }


def call_paired(stat: Mapping[str, Any], band: Mapping[str, float]) -> str:
    """'better' / 'worse' / 'within band' / 'n/a' from one paired statistic and the band."""
    if not stat or stat.get("n", 0) == 0:
        return "n/a"
    median, sign = float(stat["median_pct"]), float(stat["sign_test_fraction"])
    if median > float(band["median_pct"]) and sign >= float(band["sign_hi"]):
        return "better"
    if median < -float(band["median_pct"]) and sign <= float(band["sign_lo"]):
        return "worse"
    return "within band"


def verdict(calls: Mapping[str, str], *, battery_delta_p10: float | None = None, p10_floor_db: float = 0.5) -> str:
    values = [c for c in calls.values() if c != "n/a"]
    if not values:
        return "unscored"
    better = [k for k, c in calls.items() if c == "better"]
    worse = [k for k, c in calls.items() if c == "worse"]
    if better and not worse:
        text = "SHARPER (" + ", ".join(better) + ")"
    elif worse and not better:
        text = "blurrier (" + ", ".join(worse) + ")"
    elif better and worse:
        text = "mixed (" + ", ".join(f"{k}:{c}" for k, c in calls.items() if c != "n/a") + ")"
    else:
        text = "within rerun band"
    if battery_delta_p10 is not None and battery_delta_p10 < -p10_floor_db:
        text += f"; battery p10 {battery_delta_p10:+.2f} dB"
    return text


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

Runner = Callable[[Sequence[str], Path], int]


def _subprocess_runner(argv: Sequence[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(a) for a in argv) + "\n")
        handle.flush()
        return subprocess.run([str(a) for a in argv], cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT).returncode


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def morph_line(arm_dir: Path) -> str:
    path = Path(arm_dir) / "morph.txt"
    if not path.is_file():
        return ""
    keep = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if "N=" in l or "short p50" in l or "max/min" in l or "opacity p50" in l]
    return " | ".join(keep)


def ensure_arm_scores(arm_dir: Path, tag: str, scoring: Mapping[str, Any], *, python: str, repo_root: Path, run: Runner) -> dict[str, Any]:
    """Build what is missing for one arm directory; idempotent. Returns what exists."""
    arm_dir = Path(arm_dir)
    ckpt = arm_dir / "checkpoints" / "latest.pt"
    cfg = arm_dir / "config_as_run.json"
    out: dict[str, Any] = {"dir": str(arm_dir), "checkpoint": ckpt.is_file()}
    if not ckpt.is_file():
        return out
    tool = lambda name, *args: [python, str(repo_root / "tools" / name), *[str(a) for a in args]]
    roi = scoring.get("roi")
    reference = scoring.get("reference") or {}
    if roi and reference:
        roi_dir = arm_dir / roi["name"]
        roi_json = arm_dir / f"{roi['name']}.roi_bm.json"
        if not (roi_dir / "compare_summary.json").is_file():
            run(tool("build_three_way_compare.py", "--config", roi["config"], "--checkpoint", ckpt,
                     "--reference-ply", reference["ply"], "--reference-alignment", reference["alignment"],
                     "--output", roi_dir, "--sample-ids", roi["sample_ids"]), arm_dir / f"{roi['name']}.log")
        if (roi_dir / "compare_summary.json").is_file() and not roi_json.is_file():
            run(tool("score_compare_roi.py", "--selection", roi["selection"], "--min-samples", str(roi.get("min_samples", 100)),
                     "--match-brightness", "--json", roi_json, roi_dir), arm_dir / f"{roi['name']}.roi_bm.log")
        out["roi"] = roi_json.is_file()
    battery = scoring.get("battery")
    if battery:
        path = arm_dir / "battery_tile_owned.json"
        if not path.is_file():
            argv = tool("evaluate_probe_views.py", "--config", cfg, "--checkpoint", ckpt, "--views", str(battery.get("views", 48)), "--output", path)
            if battery.get("tile_owned", True):
                argv += ["--tile-views", "--tile-owned"]
            run(argv, arm_dir / "battery_tile_owned.log")
        out["battery"] = path.is_file()
    if scoring.get("offtraj", True):
        scores = arm_dir / "offtraj_scores.json"
        if (arm_dir / "offtraj" / "offtraj_summary.json").is_file() and not scores.is_file():
            run(tool("score_offtrajectory_strips.py", f"{tag}={arm_dir / 'offtraj'}", "--baseline", tag, "--json", scores),
                arm_dir / "offtraj_scores.log")
        out["offtraj"] = scores.is_file()
    return out


def score_ladder(spec: Mapping[str, Any], *, run_root: Path, out_dir: Path, python: str = sys.executable,
                 repo_root: Path = REPO_ROOT, run: Runner = _subprocess_runner) -> dict[str, Any]:
    scoring = spec.get("scoring") or {}
    band = scoring.get("band") or {"median_pct": 5.0, "sign_lo": 0.3, "sign_hi": 0.7}
    base_dir = Path(spec["base"]["dir"])
    base_tag = "base"
    status = {"base": ensure_arm_scores(base_dir, base_tag, scoring, python=python, repo_root=repo_root, run=run)}
    base_off = _read_json(base_dir / "offtraj_scores.json") or {}
    # One strip set per file; accept whatever tag wrote it (a peek script, an older scorer).
    base_rows = {r["file"]: r for r in (base_off.get(base_tag) or next(iter(base_off.values()), []))}
    base_batt = _read_json(base_dir / "battery_tile_owned.json") or {}
    summary: dict[str, Any] = {
        "ladder": spec["name"],
        "base": {"dir": str(base_dir), "battery": {k: base_batt.get(k) for k in ("psnr_mean", "psnr_p10", "alpha_p05")}, "morph": morph_line(base_dir)},
        "band": band,
        "arms": OrderedDict(),
    }
    header = ("| arm | ROI paired med % | ROI sign | ROI a->b | offtraj sharp med % | sharp sign | sharp a->b | "
              "offtraj psnr1/4 med % | battery dPSNR mean / p10 | N | verdict |")
    lines = [f"# Ladder {spec['name']} - single-change arms vs base {spec['base'].get('name', base_dir.name)}", "",
             f"base tile-owned battery: mean {base_batt.get('psnr_mean')} p10 {base_batt.get('psnr_p10')}", "",
             f"base morphology: {morph_line(base_dir)}", "", header, "| " + " | ".join(["---"] * 11) + " |"]
    roi_name = (scoring.get("roi") or {}).get("name")
    for arm in spec["arms"]:
        name = str(arm["arm"])
        arm_dir = Path(run_root) / name
        tag = name
        status[name] = ensure_arm_scores(arm_dir, tag, scoring, python=python, repo_root=repo_root, run=run)
        entry: dict[str, Any] = {"change": arm["change"], "status": status[name]}
        if not status[name]["checkpoint"]:
            entry["verdict"] = "not trained"
            summary["arms"][name] = entry
            lines.append(f"| {name} | - | - | - | - | - | - | - | - | - | not trained |")
            continue
        roi_stat: dict[str, Any] = {}
        panel: dict[str, Any] = {}
        if roi_name and (arm_dir / f"{roi_name}.roi_bm.json").is_file() and (base_dir / f"{roi_name}.roi_bm.json").is_file():
            band_json = arm_dir / "rerun_band_vs_base.json"
            run([python, str(repo_root / "tools" / "audit_rerun_band.py"), str(base_dir), str(arm_dir), "--roi-name", roi_name, "--json", str(band_json)],
                arm_dir / "rerun_band_vs_base.log")
            band_doc = _read_json(band_json) or {}
            roi_stat = band_doc.get("paired_statistic") or {}
            panel = band_doc.get("panel_statistic") or {}
        off = _read_json(arm_dir / "offtraj_scores.json") or {}
        rows = {r["file"]: r for r in (off.get(tag) or next(iter(off.values()), []))}
        sharp = paired_stats(base_rows, rows, "sharp_ratio") if rows and base_rows else {"n": 0}
        psnrq = paired_stats(base_rows, rows, "psnr_q") if rows and base_rows else {"n": 0}
        batt = _read_json(arm_dir / "battery_tile_owned.json") or {}
        d_mean = d_p10 = None
        if batt.get("psnr_mean") is not None and base_batt.get("psnr_mean") is not None:
            d_mean = float(batt["psnr_mean"]) - float(base_batt["psnr_mean"])
            d_p10 = float(batt["psnr_p10"]) - float(base_batt["psnr_p10"])
        calls = {"roi": call_paired(roi_stat, band), "offtraj": call_paired(sharp, band)}
        text = verdict(calls, battery_delta_p10=d_p10)
        morph = morph_line(arm_dir)
        n_gauss = morph.split("N=")[1].split()[0] if "N=" in morph else "?"
        entry.update(roi=roi_stat, roi_panel=panel, offtraj_sharp=sharp, offtraj_psnr_q=psnrq,
                     battery={k: batt.get(k) for k in ("psnr_mean", "psnr_p10", "alpha_p05")},
                     battery_delta={"psnr_mean": d_mean, "psnr_p10": d_p10}, calls=calls, morph=morph, verdict=text)
        summary["arms"][name] = entry
        f = lambda s, k: ("%.1f" % s[k]) if s.get("n") else "-"
        g = lambda s: ("%.2f" % s["sign_test_fraction"]) if s.get("n") else "-"
        lines.append(
            f"| {name} | {f(roi_stat, 'median_pct')} | {g(roi_stat)} | "
            f"{('%.3f->%.3f' % (panel.get('a_ours_over_ref_median', 0), panel.get('b_ours_over_ref_median', 0))) if panel else '-'} | "
            f"{f(sharp, 'median_pct')} | {g(sharp)} | {('%.3f->%.3f' % (sharp['a_median'], sharp['b_median'])) if sharp.get('n') else '-'} | "
            f"{f(psnrq, 'median_pct')} | {('%+.2f / %+.2f' % (d_mean, d_p10)) if d_mean is not None else '-'} | {n_gauss} | {text} |"
        )
    lines += ["", f"verdict rule: paired median beyond +-{band['median_pct']}% with sign test <= {band['sign_lo']} or >= {band['sign_hi']} "
              "on the ROI or the off-trajectory strips; otherwise within the rerun band.", ""]
    for name, entry in summary["arms"].items():
        lines.append(f"- {name}: {entry['change']}")
        if entry.get("morph"):
            lines.append(f"  - morph: {entry['morph']}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ladder_{spec['name']}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / f"ladder_{spec['name']}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["markdown"] = "\n".join(lines)
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="derive one config per arm from the base, with explicit lineage")
    w.add_argument("--spec", type=Path, required=True)
    w.add_argument("--run-root", type=Path, required=True)
    w.add_argument("--base-config", type=Path, default=None, help="default: spec.base.config, else <base.dir>/config_as_run.json")
    w.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("queue", help="print (or run) the pipeline queue command for the arms")
    q.add_argument("--spec", type=Path, required=True)
    q.add_argument("--pipeline-config", type=Path, required=True)
    q.add_argument("--python", default=sys.executable)
    q.add_argument("--run", action="store_true", help="execute instead of printing")
    s = sub.add_parser("score", help="build missing scores for base + arms and write the paired summary")
    s.add_argument("--spec", type=Path, required=True)
    s.add_argument("--run-root", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec(args.spec)
    if args.command == "write":
        base_config = args.base_config or Path(spec["base"].get("config") or (Path(spec["base"]["dir"]) / "config_as_run.json"))
        spec["base"].setdefault("config", str(base_config))
        for name, action, keys in write_arms(spec, run_root=args.run_root, base_config=base_config, dry_run=args.dry_run):
            print(f"{name}: {action}  ({', '.join(keys)})")
        return 0
    if args.command == "queue":
        argv_out = [args.python, str(REPO_ROOT / "tools" / "pipeline.py"), "--pipeline-config", str(args.pipeline_config), "queue",
                    *[str(a["arm"]) for a in spec["arms"]]]
        print(" ".join(argv_out))
        if args.run:
            return subprocess.run(argv_out, cwd=str(REPO_ROOT)).returncode
        return 0
    if args.command == "score":
        summary = score_ladder(spec, run_root=args.run_root, out_dir=args.out_dir, python=args.python)
        print(summary["markdown"])
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LadderError as error:
        print(f"research_ladder: {error}", file=sys.stderr)
        raise SystemExit(2)
