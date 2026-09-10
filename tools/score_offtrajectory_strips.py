"""Re-score the saved off-trajectory strips (ours | reference) with metrics that
tolerate the 3.5 cm alignment residual: PSNR at 1/4 and 1/8 resolution, and a
sharpness ratio (Laplacian variance ours / reference, alignment-free).

    python tools/score_offtrajectory_strips.py F6=RUN/delivery_f6/offtraj_matched R1=RUN/tile0_R1/offtraj

Every strip set is a NAME=DIR pair; the historical machine-specific arm
table is gone, so at least one pair is required. Paired wins are counted
against --baseline (default F6) when that name is among the pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np


def psnr(a, b): return 10*math.log10(1/max(float(np.mean((a.astype(np.float32)/255-b.astype(np.float32)/255)**2)),1e-10))


def parse_arm_pair(text: str) -> tuple[str, Path]:
    name, separator, directory = text.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError(f"expected NAME=DIR, got {text!r}")
    return name, Path(directory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("arms", nargs="+", type=parse_arm_pair, metavar="NAME=DIR", help="strip set to score")
    parser.add_argument("--baseline", default="F6", help="pair name the paired wins are counted against")
    parser.add_argument("--json", type=Path, help="write the per-strip rows here")
    return parser


def score_arms(arms: dict[str, Path], *, baseline: str) -> dict[str, list[dict]]:
    import cv2

    def lapvar(g): return float(cv2.Laplacian(g, cv2.CV_64F).var())
    res = {}
    for arm, d in arms.items():
        rows = []
        for f in sorted(d.glob("offtraj_*.png")):
            im = cv2.imread(str(f)); w = (im.shape[1]-8)//2; ours, ref = im[:, :w], im[:, w+8:w+8+w]
            o4, r4 = cv2.resize(ours, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA), cv2.resize(ref, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
            o8, r8 = cv2.resize(ours, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA), cv2.resize(ref, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA)
            go, gr = cv2.cvtColor(ours, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            rows.append(dict(file=f.name, psnr_full=round(psnr(ours, ref),2), psnr_q=round(psnr(o4, r4),2), psnr_e=round(psnr(o8, r8),2), sharp_ratio=round(lapvar(go)/max(lapvar(gr),1e-6),3)))
        res[arm] = rows
        print(arm, "n", len(rows), "median psnr full %.2f  1/4 %.2f  1/8 %.2f  sharpness ours/ref %.3f" % tuple(statistics.median([r[k] for r in rows]) for k in ("psnr_full","psnr_q","psnr_e","sharp_ratio")))
    # paired wins vs the baseline at 1/4 res
    for arm in [a for a in arms if a != baseline and baseline in res]:
        f6 = {r["file"]: r for r in res[baseline]}
        wins = sum(1 for r in res[arm] if r["file"] in f6 and r["psnr_q"] > f6[r["file"]]["psnr_q"]); n = sum(1 for r in res[arm] if r["file"] in f6)
        print(arm, f"wins vs {baseline} at 1/4 res", wins, "/", n)
    return res


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms = dict(args.arms)
    for name, directory in arms.items():
        if not directory.is_dir():
            print(f"score_offtrajectory_strips: {name}: directory not found: {directory}", file=sys.stderr)
            return 2
    res = score_arms(arms, baseline=args.baseline)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(res, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
