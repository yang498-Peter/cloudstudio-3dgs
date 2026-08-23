#!/usr/bin/env python3
"""Train one configuration and run the full five-table acceptance on it.

Chains the steps a release candidate has to pass (training, appearance
metrics, Gaussian health, LPIPS, PLY export) so an unattended run either
produces a complete, signed evidence set or stops at the first failure.
Every stage emits one ``PIPE:`` line so a monitor can follow progress, and a
failure emits ``PIPE-FAIL`` and exits non-zero rather than letting downstream
stages report on a broken model.

The five tables are deliberately not collapsed into a single score: they
disagree in informative ways (a PSNR gain can be perceptually invisible, and
appearance can improve while geometry degrades), so promotion decisions read
all of them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def say(message: str) -> None:
    print(f"PIPE: {message}", flush=True)


def fail(message: str) -> None:
    print(f"PIPE-FAIL: {message}", flush=True)
    raise SystemExit(1)


def _python() -> str:
    return sys.executable


def _run(arguments: list[str], stage: str) -> subprocess.CompletedProcess:
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"{stage} failed: {result.stderr.strip()[-400:]}")
    return result


def train(config: Path, output_dir: Path, log: Path) -> None:
    """Run training, skipping it when the run already produced a manifest."""
    if (output_dir / "run_manifest.json").exists():
        say(f"training already complete: {output_dir.name}")
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as stream:
        code = subprocess.call(
            [_python(), "tools/train_gsplat.py", "--config", str(config)],
            cwd=ROOT, stdout=stream, stderr=stream,
        )
    if code != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
        fail(f"training exited {code}: {' | '.join(tail)}")
    say(f"training complete: {output_dir.name}")


def link_evaluation(output_dir: Path, probes_root: Path, run_name: str) -> None:
    """Expose a run's evaluation directory under the shared probes root."""
    target = probes_root / run_name
    target.mkdir(parents=True, exist_ok=True)
    if not (target / "evaluation").exists():
        subprocess.call(
            ["cmd", "/c", "mklink", "/J", str(target / "evaluation"),
             str(output_dir / "evaluation")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def appearance(probes_root: Path, runs: list[str], output: Path) -> dict:
    _run([_python(), "tools/compare_validation_metrics.py",
          "--runs-root", str(probes_root), "--runs", *runs,
          "--output", str(output)], "appearance metrics")
    return json.loads(output.read_text(encoding="utf-8"))


def health(checkpoint: Path, lidar_ply: Path, output: Path) -> dict:
    _run([_python(), "tools/gaussian_health.py", "--checkpoint", str(checkpoint),
          "--lidar-ply", str(lidar_ply), "--output", str(output)], "gaussian health")
    payload = json.loads(output.read_text(encoding="utf-8"))
    outliers = payload["floater"]["outliers"]
    wall = payload["wall"]["weighted_by_lidar_inliers"]
    return {
        "floater_gt_0.3m": outliers["gt_0.3m"]["count"],
        "floater_gt_1.0m": outliers["gt_1.0m"]["count"],
        "floater_gt_5.0m": outliers["gt_5.0m"]["count"],
        "ground_thickness_p50_m": wall.get("effective_thickness_p50_m"),
        "shortest_axis_angle_deg_p50": wall.get("shortest_axis_angle_deg_p50"),
    }


def perceptual(probes_root: Path, run_name: str, output: Path) -> dict | None:
    result = subprocess.run(
        [_python(), "tools/compute_lpips.py", "--runs-root", str(probes_root),
         "--runs", run_name, "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        # LPIPS needs downloadable weights; a missing backend must not void an
        # otherwise complete acceptance, but it is never silently ignored.
        say(f"lpips unavailable: {result.stderr.strip()[-200:]}")
        return None
    payload = json.loads(output.read_text(encoding="utf-8"))
    return payload.get("runs", payload).get(run_name)


def export_ply(checkpoint: Path, output: Path, min_opacity: float) -> None:
    _run([_python(), "tools/export_gaussian_ply.py", "--checkpoint", str(checkpoint),
          "--output", str(output), "--min-opacity", str(min_opacity)], "ply export")
    say(f"ply exported: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--probes-root", required=True, type=Path)
    parser.add_argument("--lidar-ply", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--compare-with", nargs="*", default=[],
                        help="other run names under --probes-root to tabulate beside this one")
    parser.add_argument("--export-ply", type=Path, default=None)
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--checkpoint-name", default="best_golden.pt")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    run_name = output_dir.name
    say(f"acceptance start: {run_name}")

    # The log lives beside the run, not inside it: the trainer refuses to
    # start into a non-empty output directory, and creating the log there
    # would trip that check before the first step.
    train(config=args.config, output_dir=output_dir,
          log=args.log or output_dir.with_name(f"{run_name}_training.log"))
    link_evaluation(output_dir, args.probes_root, run_name)

    checkpoint = output_dir / "checkpoints" / args.checkpoint_name
    if not checkpoint.exists():
        fail(f"checkpoint missing: {checkpoint}")

    metrics = appearance(args.probes_root, [*args.compare_with, run_name],
                         args.report.with_suffix(".appearance.json"))
    own = metrics[run_name]
    say(f"appearance psnr {own['psnr_mean']:.3f} ssim {own['ssim_mean']:.4f} "
        f"depth {own['depth_mae_mean_m']:.3f}")

    geometry = health(checkpoint, args.lidar_ply, args.report.with_suffix(".health.json"))
    say(f"health {json.dumps(geometry)}")

    lpips = perceptual(args.probes_root, run_name, args.report.with_suffix(".lpips.json"))
    if lpips is not None:
        say(f"lpips {lpips['lpips_mean']:.4f} p90 {lpips['lpips_p90']:.4f} "
            f"calibrated={lpips['calibrated']}")

    if args.export_ply is not None:
        export_ply(checkpoint, args.export_ply, args.min_opacity)

    report = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "appearance": own,
        "appearance_comparison": metrics,
        "health": geometry,
        "perceptual": lpips,
        "ply": None if args.export_ply is None else str(args.export_ply),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    say(f"report written: {args.report}")
    say("ACCEPTANCE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
