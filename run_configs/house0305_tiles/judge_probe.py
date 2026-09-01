"""Judge one lifecycle probe against the two criteria fixed in advance.

A probe passes only if BOTH hold across reset-recovery points:
  1. dead mass does not accumulate  - frac<0.1 at recovery stays near 0.10
  2. active mass does not bleed     - active count stops falling cycle to cycle

The clean-vendor baseline passed (1) and failed (2): dead mass returned to
8-11% every cycle but active fell ~18% per cycle. The first contribution probe
(floor 0.05) failed (1): cull collapsed to ~2.7k and dead mass climbed
0.527 -> 0.570. Recovery points are the events NOT immediately after a reset,
i.e. the low-cull ones; the post-reset flush events are the spikes.
"""
import json
import sys
from pathlib import Path


def events(run: str):
    path = Path(r"C:\Peter\3dgs-runs\house0305_sop") / run / "monitor" / "progress.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = r["metrics"]
        if m.get("lifecycle_active_count") is None:
            continue
        out.append(
            {
                "step": r["completed_steps"],
                "raw": r["gaussian_count"],
                "active": m["lifecycle_active_count"],
                "d005": m.get("lifecycle_frac_op_below_005") or 0.0,
                "d010": m.get("lifecycle_frac_op_below_010") or 0.0,
                "clone": m.get("lifecycle_clone_parents") or 0,
                "cull": m.get("lifecycle_cull_total") or 0,
            }
        )
    return out


def judge(run: str):
    ev = events(run)
    if len(ev) < 4:
        return {"run": run, "verdict": "INSUFFICIENT", "events": len(ev)}
    # recovery points: events whose cull is below the run's median cull
    culls = sorted(e["cull"] for e in ev)
    median_cull = culls[len(culls) // 2]
    rec = [e for e in ev if e["cull"] <= median_cull]
    if len(rec) < 2:
        rec = ev[1:]
    dead = [e["d010"] for e in rec]
    act = [e["active"] for e in rec]
    dead_ok = max(dead[-3:]) <= 0.20          # stays drained, not hoarding
    # active must not fall materially across the last recovery points
    bleed = (act[0] - act[-1]) / max(act[0], 1)
    act_ok = bleed <= 0.10
    return {
        "run": run,
        "events": len(ev),
        "recovery_dead_first": round(dead[0], 3),
        "recovery_dead_last": round(dead[-1], 3),
        "recovery_active_first": act[0],
        "recovery_active_last": act[-1],
        "active_bleed_frac": round(bleed, 3),
        "dead_ok": dead_ok,
        "active_ok": act_ok,
        "verdict": "PASS" if (dead_ok and act_ok) else "FAIL",
    }


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(json.dumps(judge(run), ensure_ascii=False))
