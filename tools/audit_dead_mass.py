#!/usr/bin/env python3
"""Who are the low-opacity gaussians? Lineage, age, size and placement.

Half of a capped population sitting below opacity 0.1 is budget spent on
nothing, and the fix depends on where they come from: newborn clones that
never recovered from the next reset, old survivors decaying after refinement
stopped, or oversized blobs the cull thresholds do not reach. The checkpoint
carries birth step and kind per gaussian, so this is a cross-tab, not a guess.

    python tools/audit_dead_mass.py --checkpoint RUN/checkpoints/latest.pt \
        [--dead 0.1] [--refine-stop 14000] [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _quantiles(t, qs=(0.5, 0.9, 0.99)):
    import torch

    if t.numel() == 0:
        return {str(q): None for q in qs}
    sample = t
    if sample.numel() > 2_000_000:
        generator = torch.Generator().manual_seed(0)
        sample = sample[torch.randperm(sample.numel(), generator=generator)[:2_000_000]]
    return {str(q): float(torch.quantile(sample.float(), q)) for q in qs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dead", type=float, default=0.1)
    parser.add_argument("--refine-stop", type=int, default=14000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = payload["params"]
    state = payload.get("strategy_state") or {}
    step = int(payload.get("step", 0))
    opacity = torch.sigmoid(params["opacities"].detach().float().flatten())
    scales = params["scales"].detach().float().exp()
    long_axis = scales.max(dim=1).values
    short_axis = scales.min(dim=1).values
    means = params["means"].detach().float()
    n = opacity.numel()
    dead = opacity < args.dead

    birth_step = state.get("_cloudstudio_birth_step")
    birth_kind = state.get("_cloudstudio_birth_kind")
    count = state.get("count")
    visible_alpha = state.get("_visible_alpha_sum")

    report = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "gaussian_count": n,
        "dead_threshold": args.dead,
        "dead_fraction": float(dead.float().mean()),
        "opacity_quantiles": _quantiles(opacity),
    }

    def bucket_table(labels, mask_of_label):
        rows = {}
        for label in labels:
            mask = mask_of_label(label)
            total = int(mask.sum())
            if total == 0:
                continue
            rows[label] = {
                "count": total,
                "share_of_population": total / n,
                "dead_fraction": float(dead[mask].float().mean()),
                "share_of_dead": float(dead[mask].sum()) / max(1, int(dead.sum())),
                "opacity_p50": float(opacity[mask].median()),
                "long_axis_mm_p50": float(long_axis[mask].median() * 1000.0),
                "short_axis_mm_p50": float(short_axis[mask].median() * 1000.0),
            }
        return rows

    if birth_kind is not None:
        kind = birth_kind.flatten().long()
        names = {0: "init", 1: "clone", 2: "split"}
        report["by_birth_kind"] = bucket_table(
            [names[k] for k in sorted(set(kind.tolist()))],
            lambda label: kind == {v: k for k, v in names.items()}[label],
        )
    if birth_step is not None:
        bstep = birth_step.flatten().long()
        edges = [0, 1, 2000, 5000, 10000, args.refine_stop, step + 1]
        labels = [f"born_{edges[i]}_{edges[i + 1] - 1}" for i in range(len(edges) - 1)]

        def mask_for(label):
            i = labels.index(label)
            return (bstep >= edges[i]) & (bstep < edges[i + 1])

        report["by_birth_step"] = bucket_table(labels, mask_for)
        age = step - bstep
        report["dead_age_steps"] = _quantiles(age[dead].float())
        report["alive_age_steps"] = _quantiles(age[~dead].float())
    if count is not None:
        obs = count.flatten().float()
        edges = [0, 1, 5, 20, 100, 1e9]
        labels = [f"seen_{int(edges[i])}_{int(min(edges[i + 1], 1e6)) - 1}" for i in range(len(edges) - 1)]

        def mask_obs(label):
            i = labels.index(label)
            return (obs >= edges[i]) & (obs < edges[i + 1])

        report["by_observation_count_since_last_refine"] = bucket_table(labels, mask_obs)
    if visible_alpha is not None:
        va = visible_alpha.flatten().float()
        report["visible_alpha_sum_dead"] = _quantiles(va[dead])
        report["visible_alpha_sum_alive"] = _quantiles(va[~dead])

    # Size: are the dead ones the big blobs the scale cull should catch?
    edges_mm = [0, 1, 5, 20, 50, 1e9]
    labels = [f"long_{edges_mm[i]}_{int(min(edges_mm[i + 1], 1e6))}mm" for i in range(len(edges_mm) - 1)]
    lmm = long_axis * 1000.0

    def mask_size(label):
        i = labels.index(label)
        return (lmm >= edges_mm[i]) & (lmm < edges_mm[i + 1])

    report["by_long_axis"] = bucket_table(labels, mask_size)

    # Height: below the 1st percentile of alive gaussians counts as underground.
    z = means[:, 2]
    floor = float(torch.quantile(z[~dead][: 2_000_000], 0.01)) if (~dead).sum() > 0 else float("nan")
    report["height_floor_alive_p01_m"] = floor
    report["dead_below_floor_fraction"] = float((z[dead] < floor - 0.05).float().mean())
    report["alive_below_floor_fraction"] = float((z[~dead] < floor - 0.05).float().mean())

    text = json.dumps(report, indent=1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, args.output)
    print(f"{args.checkpoint.parent.parent.name}: N={n:,} step={step} dead(<{args.dead})={report['dead_fraction']:.3f}")
    for section in ("by_birth_kind", "by_birth_step", "by_observation_count_since_last_refine", "by_long_axis"):
        rows = report.get(section)
        if not rows:
            continue
        print(f"  {section}")
        for label, row in rows.items():
            print(
                f"    {label:24s} n={row['count']:>10,} share={row['share_of_population']:.3f} "
                f"dead={row['dead_fraction']:.3f} of_dead={row['share_of_dead']:.3f} "
                f"op50={row['opacity_p50']:.3f} long50={row['long_axis_mm_p50']:.1f}mm"
            )
    print(f"  dead age p50 {report.get('dead_age_steps', {}).get('0.5')}  alive age p50 {report.get('alive_age_steps', {}).get('0.5')}")
    print(f"  underground: dead {report['dead_below_floor_fraction']:.3f}  alive {report['alive_below_floor_fraction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
