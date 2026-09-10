#!/usr/bin/env python3
"""Derive the S0 / S1 schedule arms of WP02 from one base arm config.

S0 is the base arm relabelled (run_id / output_dir only): the historical
control that keeps the 20-epoch ``max_steps`` truncated by a controlled stop.

S1 moves every schedule field that is linked to the horizon together onto
H = ``--horizon`` under ``schedule_contract: research_rescaled_horizon_v1``:

* ``max_steps`` becomes H and the truncating controlled stop is dropped (or,
  with ``--keep-controlled-stop``, restated as H), so the means LR really
  decays to ``learning_rates.means * means_lr_final_factor`` by the last step;
* the growth window keeps the fraction of the *executed* base horizon it
  actually ran with (``refine_stop / stop_step``, 14000/20000 = 0.7 for the
  tile0 R1 arm) unless ``--refine-stop-fraction`` overrides it, snapped to a
  ``refine_every`` multiple, mirrored into ``mcmc_refine_stop_iter``,
  ``default_strategy.refine_stop_iter`` and ``refine_scale2d_stop_iter``;
* ``prune_switch_step`` becomes H // 2 (the trainer's exact-lifecycle rule),
  which under the 0.7 H window is inside the growth window, so
  ``prune_opa_late`` is reachable for the first time;
* ``metric_scale_calibration.means_step_fraction`` is set to an explicit
  null so ``learning_rates.means`` is the optimizer base; with
  ``--reference-scale-m`` the base is frozen to the value the base arm
  effectively ran with (``means_step_fraction * reference_scale_m``).

Pure CPU, no torch.  The S1 candidate is validated through
``validate_research_schedule_contract`` and audited with ``resolved_schedule``
before anything is written; the JSON lists every moved field under
``schedule_contract_fields``.  Nothing is written under the run root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.schedule_audit import (  # noqa: E402
    RESEARCH_CONTRACT_DECLARABLE_FIELDS,
    RESEARCH_REFINE_STOP_MAX_FRACTION,
    RESEARCH_SCHEDULE_CONTRACT_V1,
    TRAINER_DEFAULTS,
    resolved_schedule,
    validate_research_schedule_contract,
)

DEFAULT_BASE = ROOT / "run_configs" / "house0305_tiles" / "v9" / "tile0_R1_range0_20k.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: dict[str, Any]) -> None:
    # Match the hand-written arm configs (indent=1) so diffs stay readable.
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _sibling_output_dir(base_output_dir: str, stem: str) -> str:
    # Keep the base run root, swap only the leaf; Windows paths in the arm
    # configs are backslash strings, so avoid pathlib normalisation.
    separator = "\\" if "\\" in base_output_dir else "/"
    head, _, _ = base_output_dir.rpartition(separator)
    return f"{head}{separator}{stem}" if head else stem


def _snap(value: float, multiple: int) -> int:
    return int(round(value / multiple)) * multiple


def relabel_control(base: dict[str, Any], *, run_id: str, output_stem: str) -> dict[str, Any]:
    control = copy.deepcopy(base)
    control["run_id"] = run_id
    control["output_dir"] = _sibling_output_dir(str(base["output_dir"]), output_stem)
    return control


def rescale_to_horizon(
    base: dict[str, Any],
    *,
    horizon: int,
    run_id: str,
    output_stem: str,
    reference_scale_m: float | None,
    refine_stop_fraction: float | None,
    keep_controlled_stop: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    base_schedule = resolved_schedule(base, None)
    base_lifecycle = base_schedule["lifecycle"]
    refine_every = int(base_lifecycle["refine_every"])
    base_stop_step = int(base_schedule["steps"]["stop_step"])
    if refine_stop_fraction is None:
        refine_stop_fraction = int(base_lifecycle["refine_stop_iter"]) / base_stop_step
    refine_stop = _snap(refine_stop_fraction * horizon, refine_every)
    if refine_stop > RESEARCH_REFINE_STOP_MAX_FRACTION * horizon:
        raise ValueError(
            f"refine_stop {refine_stop} exceeds {RESEARCH_REFINE_STOP_MAX_FRACTION} * H; "
            "pass a smaller --refine-stop-fraction"
        )
    prune_switch = horizon // 2

    arm = copy.deepcopy(base)
    # Every field the contract governs is recorded, changed or not: a linked
    # field that happens to land on its base value is still part of the
    # intervention and must not read as "untouched".
    linked: dict[str, dict[str, Any]] = {}

    def move(path: str, container: dict[str, Any], key: str, value: Any, *, remove: bool = False) -> None:
        before = container.get(key, "<absent>")
        if remove:
            container.pop(key, None)
            after: Any = "<absent>"
        else:
            container[key] = value
            after = value
        linked[path] = {"base": before, "s1": after, "changed": before != after}

    move("run_id", arm, "run_id", run_id)
    move("output_dir", arm, "output_dir", _sibling_output_dir(str(base["output_dir"]), output_stem))
    move("max_steps", arm, "max_steps", int(horizon))
    if keep_controlled_stop:
        move("controlled_stop_after_steps", arm, "controlled_stop_after_steps", int(horizon))
    else:
        move("controlled_stop_after_steps", arm, "controlled_stop_after_steps", None, remove=True)
    move("mcmc_refine_stop_iter", arm, "mcmc_refine_stop_iter", refine_stop)
    strategy = arm.setdefault("default_strategy", {})
    move("default_strategy.refine_stop_iter", strategy, "refine_stop_iter", refine_stop)
    move("default_strategy.refine_scale2d_stop_iter", strategy, "refine_scale2d_stop_iter", refine_stop)
    move("default_strategy.prune_switch_step", strategy, "prune_switch_step", prune_switch)
    calibration = arm.setdefault("metric_scale_calibration", {})
    base_fraction = calibration.get("means_step_fraction", TRAINER_DEFAULTS["means_step_fraction"])
    move("metric_scale_calibration.means_step_fraction", calibration, "means_step_fraction", None)
    learning_rates = arm.setdefault("learning_rates", {})
    means_lr_note: str
    if reference_scale_m is not None and base_fraction is not None:
        frozen = float(reference_scale_m) * float(base_fraction)
        move("learning_rates.means", learning_rates, "means", frozen)
        means_lr_note = (
            f"frozen to the base arm's effective optimizer base: means_step_fraction {base_fraction} "
            f"x reference_scale_m {reference_scale_m} (runtime fact supplied via --reference-scale-m)"
        )
    else:
        move("learning_rates.means", learning_rates, "means", learning_rates.get("means"))
        means_lr_note = (
            "learning_rates.means kept nominal; NOTE the base arm ran on means_step_fraction x "
            "reference_scale_m instead, so S1 starts from a different base LR unless "
            "--reference-scale-m is supplied"
        )
    # Cadence fields the contract expresses against H but that the trainer's
    # exact-lifecycle rules pin (start=500, every=100, reset profile): linked,
    # deliberately unchanged.
    move("mcmc_refine_start_iter", arm, "mcmc_refine_start_iter", arm.get("mcmc_refine_start_iter", TRAINER_DEFAULTS["mcmc_refine_start_iter"]))
    move("mcmc_refine_every", arm, "mcmc_refine_every", arm.get("mcmc_refine_every", TRAINER_DEFAULTS["mcmc_refine_every"]))
    move("default_strategy.refine_start_iter", strategy, "refine_start_iter", strategy.get("refine_start_iter", arm["mcmc_refine_start_iter"]))
    move("default_strategy.refine_every", strategy, "refine_every", strategy.get("refine_every", arm["mcmc_refine_every"]))
    move("default_strategy.reset_every", strategy, "reset_every", strategy.get("reset_every", 300))
    move("sh_degree_interval", arm, "sh_degree_interval", arm.get("sh_degree_interval", TRAINER_DEFAULTS["sh_degree_interval"]))
    move("means_lr_final_factor", arm, "means_lr_final_factor", arm.get("means_lr_final_factor", TRAINER_DEFAULTS["means_lr_final_factor"]))

    arm["schedule_contract"] = RESEARCH_SCHEDULE_CONTRACT_V1
    # Validate on a copy without the declaration first, then declare the
    # resolved fields so the trainer verifies the declaration on every load.
    record = validate_research_schedule_contract(arm)
    resolved = {key: record["resolved_fields"][key] for key in RESEARCH_CONTRACT_DECLARABLE_FIELDS}
    arm["schedule_contract_fields"] = {
        "arm": "S1",
        "role": "fully_recalibrated_horizon",
        "horizon_steps": int(horizon),
        "refine_stop_fraction_max": RESEARCH_REFINE_STOP_MAX_FRACTION,
        "resolved": resolved,
        "horizon_fractions": record["horizon_fractions"],
        "linked_fields": sorted(linked),
        "changed_fields": sorted(path for path, entry in linked.items() if entry["changed"]),
        "moved_from_base": linked,
        "policy": {
            "refine_stop": (
                f"base refine_stop / base executed stop_step = {refine_stop_fraction:.6f}, "
                f"x H snapped to refine_every={refine_every}"
            ),
            "prune_switch_step": "H // 2 (trainer exact-lifecycle rule); reachable because refine_stop > H / 2",
            "reset_every": "unchanged cadence (multiple of refine_every, vendor profile exact_every300)",
            "refine_start_iter": "unchanged (trainer exact-lifecycle rule start=500)",
            "sh_degree_interval": "unchanged (0 = full degree from step 0)",
            "controlled_stop_after_steps": (
                "restated as H (run ends in ControlledTrainingInterruption at H)"
                if keep_controlled_stop
                else "removed: the run completes at H and writes run_manifest.json"
            ),
            "means_lr": means_lr_note,
        },
        "provenance": provenance,
    }
    validate_research_schedule_contract(arm)
    return arm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--horizon", type=int, default=20000)
    parser.add_argument("--reference-scale-m", type=float, default=None,
                        help="runtime median Gaussian scale of the base arm (surface_initialization_report tangent_scale_median); freezes learning_rates.means to what the base effectively ran")
    parser.add_argument("--refine-stop-fraction", type=float, default=None,
                        help="growth window end as a fraction of H; default keeps the base arm's executed fraction")
    parser.add_argument("--keep-controlled-stop", action="store_true",
                        help="restate controlled_stop_after_steps as H instead of dropping it")
    parser.add_argument("--s1-out", type=Path, default=DEFAULT_BASE.parent / "tile0_S1_rescaled20k.json")
    parser.add_argument("--s0-out", type=Path, default=DEFAULT_BASE.parent / "tile0_S0_control.json")
    parser.add_argument("--s1-run-id", default="house0305-t0-S1-rescaled20k")
    parser.add_argument("--s0-run-id", default="house0305-t0-S0-control")
    parser.add_argument("--view-count", type=int, default=None, help="tile view count, for the audit summary only")
    args = parser.parse_args(argv)

    base = _load(args.base)
    base_sha = hashlib.sha256(args.base.read_bytes()).hexdigest()
    provenance = {
        "base_config": str(args.base.relative_to(ROOT)) if args.base.is_relative_to(ROOT) else str(args.base),
        "base_config_sha256": base_sha,
        "base_run_id": base.get("run_id"),
        "control_config": str(args.s0_out.relative_to(ROOT)) if args.s0_out.is_relative_to(ROOT) else str(args.s0_out),
        "generator": "tools/make_rescaled_schedule_config.py",
        "reference_scale_m": args.reference_scale_m,
    }
    control = relabel_control(base, run_id=args.s0_run_id, output_stem=args.s0_out.stem)
    arm = rescale_to_horizon(
        base,
        horizon=args.horizon,
        run_id=args.s1_run_id,
        output_stem=args.s1_out.stem,
        reference_scale_m=args.reference_scale_m,
        refine_stop_fraction=args.refine_stop_fraction,
        keep_controlled_stop=args.keep_controlled_stop,
        provenance=provenance,
    )
    _dump(args.s0_out, control)
    _dump(args.s1_out, arm)

    for label, config in (("S0", control), ("S1", arm)):
        schedule = resolved_schedule(config, args.view_count, reference_scale_m=args.reference_scale_m)
        print(f"{label}: wrote {args.s0_out if label == 'S0' else args.s1_out}")
        print(f"  stop_step={schedule['steps']['stop_step']} max_steps={schedule['steps']['max_steps']} "
              f"refine_stop={schedule['lifecycle']['refine_stop_iter']} prune_switch={schedule['lifecycle']['prune_switch_step']}")
        print(f"  means LR nominal last/declared_final = {schedule['means_lr']['nominal_last_executed_over_declared_final']:.3f}")
        print(f"  late threshold first step = {schedule['event_summary']['late_threshold']['first_step']}")
        print(f"  mismatches = {[m['name'] for m in schedule['mismatches']]}")
    print(f"S1 linked fields: {arm['schedule_contract_fields']['linked_fields']}")
    print(f"S1 changed fields: {arm['schedule_contract_fields']['changed_fields']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
