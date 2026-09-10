#!/usr/bin/env python3
"""Read-only static audit of CloudStudio 3DGS JSON training configurations.

Targets the scheduling semantics inspected at research commit 1ca1cd453e79.
Does not import torch, start training, alter configs, or inspect live GPU jobs.
Warnings identify audit obligations, not experimentally proven image defects.
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def first_multiple_at_or_after(start: int, period: int) -> int:
    return ((start + period - 1) // period) * period


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Top-level JSON must be an object")
    warnings: list[dict[str, str]] = []
    def warn(code: str, message: str) -> None:
        warnings.append({"code": code, "message": message})
    total = positive_int(config.get("max_steps"), "max_steps")
    controlled = config.get("controlled_stop_after_steps")
    if controlled is not None:
        controlled = positive_int(controlled, "controlled_stop_after_steps")
    horizon = min(total, controlled) if controlled is not None else total
    ds = config.get("default_strategy", {})
    if not isinstance(ds, dict):
        raise ValueError("default_strategy must be an object")
    exact = bool(ds.get("exact_mipmap_lifecycle", False))
    rs = ds.get("refine_start_iter", config.get("mcmc_refine_start_iter", 500))
    re = ds.get("refine_stop_iter", config.get("mcmc_refine_stop_iter", 15000))
    every = positive_int(ds.get("refine_every", config.get("mcmc_refine_every", 100)), "refine_every")
    if isinstance(rs, bool) or not isinstance(rs, int) or rs < 0:
        raise ValueError("refine_start_iter must be a nonnegative integer")
    re = positive_int(re, "refine_stop_iter")
    if rs >= re:
        warn("S000", "refine_start_iter >= refine_stop_iter: no ordinary refinement window.")
    for name in ("refine_start_iter", "refine_stop_iter", "refine_every"):
        top = "mcmc_" + name
        if name in ds and top in config and ds[name] != config[top]:
            warn("S003", f"Nested {name}={ds[name]} differs from {top}={config[top]}; inspect config resolution/validation.")
    first_regular = first_multiple_at_or_after(rs if exact else rs + 1, every)
    regular_end_exclusive = min(horizon, re)
    regular_count = max(0, (regular_end_exclusive - 1 - first_regular) // every + 1)
    last_regular = first_regular + (regular_count - 1) * every if regular_count else None
    switch = ds.get("prune_switch_step")
    late_reachable = None
    post_every = ds.get("post_refine_cull_every")
    post_until = ds.get("post_refine_cull_until")
    if post_every is not None:
        post_every = positive_int(post_every, "post_refine_cull_every")
    if switch is not None and exact:
        if isinstance(switch, bool) or not isinstance(switch, int) or switch < 0:
            raise ValueError("prune_switch_step must be a nonnegative integer")
        first_late_regular = first_multiple_at_or_after(max(rs, switch), every)
        late_regular = first_late_regular < regular_end_exclusive
        late_post = False
        if post_every:
            post_end = horizon if post_until is None else min(horizon, int(post_until) + 1)
            first_late_post = first_multiple_at_or_after(max(re, switch), post_every)
            late_post = first_late_post < post_end
        late_reachable = late_regular or late_post
        if not late_reachable:
            warn("S002", f"prune_switch_step={switch} is unreachable by active cull events; ordinary refinement ends before {re}, horizon={horizon}, post-refine cull={post_every!r}.")
    if horizon < total:
        warn("S001", f"Controlled stop at {horizon} truncates a max_steps={total} schedule; this is not a fully rescaled {horizon}-step schedule.")
    lrs = config.get("learning_rates", {})
    base = lrs.get("means") if isinstance(lrs, dict) else None
    factor = float(config.get("means_lr_final_factor", 1.0))
    if not 0 < factor <= 1 or not math.isfinite(factor):
        raise ValueError("means_lr_final_factor must be finite and in (0, 1]")
    nominal_lr = terminal_lr = ratio = None
    if base is not None:
        base = float(base)
        if base < 0 or not math.isfinite(base):
            raise ValueError("learning_rates.means must be finite and nonnegative")
        nominal_lr = base * factor ** ((horizon - 1) / total)
        terminal_lr = base * factor
        ratio = nominal_lr / terminal_lr if terminal_lr else None
    if config.get("densification_gradient_source") == "total_loss":
        warn("G001", "Birth scores use total-loss rendered-position gradients. Test the existing rgb_only path while retaining all optimizer losses; direct parameter regularizers need not contribute to means2d.")
    exposure = config.get("exposure_compensation", {})
    if isinstance(exposure, dict) and exposure.get("enabled"):
        if not exposure.get("zero_mean_projection", False) and float(exposure.get("mean_anchor_weight", 0.0)) == 0:
            warn("E001", "No explicit zero-mean/mean-anchor exposure gauge enabled. The default per-image L2 prior still exists; inspect canonical brightness, gains, saturation and cross-tile consistency before changing the model.")
    if int(config.get("sh_degree", 0)) > 0 and int(config.get("sh_degree_interval", 0)) == 0:
        warn("A001", "SH is fully active from the beginning. This is not intrinsically invalid; retain as a control against staged appearance activation.")
    reset = ds.get("reset_every")
    if reset is not None:
        reset = positive_int(reset, "reset_every")
    return {
        "run_id": config.get("run_id"),
        "reference_semantics_commit": "1ca1cd453e791bcb2d4ae189078487bf8fa92e7b",
        "max_steps": total,
        "effective_horizon_steps": horizon,
        "last_zero_based_step": horizon - 1,
        "exact_mipmap_lifecycle": exact,
        "regular_refine_event_count": regular_count,
        "last_regular_refine_step": last_regular,
        "late_prune_threshold_reachable": late_reachable,
        "nominal_means_lr_at_last_step": nominal_lr,
        "nominal_means_lr_terminal_target": terminal_lr,
        "nominal_lr_multiple_of_terminal_target": ratio,
        "reset_every_steps": reset,
        "warnings": warnings,
        "limitations": [
            "Static JSON audit only. Resolve presets and as-run overrides on the actual machine.",
            "Learning-rate numbers exclude effective initialization scaling and phase/post-refine geometry multipliers.",
            "View counts, cache contents, image quality, memory demand and live delivery status are not inferred.",
            "Cull reachability models the inspected exact lifecycle; other strategy implementations need separate checks."
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, help="JSON file or glob; repeatable")
    parser.add_argument("--output", type=Path, help="Write report JSON; configs remain unchanged")
    args = parser.parse_args()
    files: list[Path] = []
    for pattern in args.config:
        matches = [Path(p) for p in glob.glob(pattern)]
        if not matches:
            parser.error(f"No config matched {pattern!r}")
        files.extend(matches)
    reports: list[dict[str, Any]] = []
    failed = False
    for path in sorted(set(p.resolve() for p in files)):
        try:
            payload = path.read_bytes()
            config = json.loads(payload.decode("utf-8-sig"))
            result = audit_config(config)
            result.update({"config_path": str(path), "config_sha256": hashlib.sha256(payload).hexdigest()})
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            failed = True
            result = {"config_path": str(path), "error": str(exc)}
        reports.append(result)
    output = json.dumps({"schema_version": 1, "audit_kind": "static_read_only", "configs": reports}, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 2 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
