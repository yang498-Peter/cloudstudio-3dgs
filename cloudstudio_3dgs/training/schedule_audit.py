"""Pure resolution of a trainer config into the schedule it actually executes.

The trainer spreads one schedule across several places: ``max_steps`` is the
means-LR denominator (``trainer.means_lr_for_step``), ``controlled_stop_after_steps``
is where execution really ends, the lifecycle window lives both at the top level
(``mcmc_refine_*``) and inside ``default_strategy`` (``refine_*``), and the late
opacity threshold is keyed on ``prune_switch_step`` which the trainer expects to
equal ``max_steps // 2``.  When those move independently the run silently ends
on a learning rate an order of magnitude above the declared final value and a
threshold that never fires.  This module reproduces the arithmetic without
torch so a config can be audited before (or long after) any GPU touches it.

Everything here is a pure function of the config dict plus optional facts the
trainer only learns at runtime (view count, reference scale).  Nothing is read
from disk; ``tools/audit_training_schedule.py`` does the I/O.
"""

from __future__ import annotations

from typing import Any

# Trainer defaults duplicated on purpose: ``trainer.py`` imports torch
# transitively, so the audit cannot import ``TrainerConfig`` to read them.  The
# unit tests nail these against the trainer source text.
TRAINER_DEFAULTS: dict[str, Any] = {
    "mcmc_refine_start_iter": 500,
    "mcmc_refine_stop_iter": 25_000,
    "mcmc_refine_every": 100,
    "sh_degree": 2,
    "sh_degree_interval": 1000,
    "color_model": "rgb_sigmoid",
    "means_lr_final_factor": 1.0,
    "post_refine_geometry_lr_scale": 1.0,
    "view_sampling_mode": "with_replacement",
    "rgb_l1_weight": 0.8,
    "rgb_ssim_weight": 0.2,
    "lidar_range_weight": 0.05,
    "lidar_alpha_weight": 0.0,
    "lidar_alpha_target": 0.95,
    "da2_depth_weight": 0.0,
    "mesh_depth_weight": 0.0,
    "mesh_normal_weight": 0.0,
    "competitor_loss_schedule_enabled": False,
    "means_step_fraction": 0.0032,
}
ADAPTER_DEFAULTS: dict[str, Any] = {
    "reset_every": 3000,
    "prune_opa": 0.005,
    "prune_opa_late": None,
    "prune_switch_step": None,
    "post_refine_cull_every": None,
    "post_refine_cull_until": None,
}
NORMAL_ALIGNMENT_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "weight_align": 0.01,
    "weight_flatten": 0.01,
    "weight_tangent_isotropy": 0.0,
    "weight_point_to_plane": 0.0,
}
GEOMETRY_REGULARIZATION_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "opacity_sparsity_weight": 1e-4,
    "scale_upper_weight": 1e-4,
    "anisotropy_weight": 1e-4,
}
VENDOR_RESET_INTERVALS = {
    "exact_every300": 300,
    "deferred_every3000_compatibility": 3000,
}
VENDOR_CULL_THRESHOLDS = {
    "exact_0p10_to_0p05": (0.1, 0.05),
    "compatibility_uniform_0p05": (0.05, 0.05),
    "calibrated_uniform_0p04": (0.04, 0.04),
    "calibrated_geometry_only_0p00": (0.0, 0.0),
    "audit_uniform_0p005": (0.005, 0.005),
}
# MipMap epoch-permutation sampling with adaptive growth requires exactly this
# many complete view epochs (trainer.py, train_gsplat pre-flight).
MIPMAP_EPOCHS_PER_RUN = 20

# Research schedule contracts.  A config that names one opts out of the
# competitor-parity horizon (``max_steps == 20 * views`` with a controlled stop
# truncating it) and instead declares ``max_steps`` itself as the horizon H
# that every linked schedule field is expressed against.  The trainer, the
# audit CLI and the config generator all resolve the contract through
# ``research_schedule_contract`` so they cannot disagree.  Absent key = the
# existing parity behaviour, untouched.
RESEARCH_SCHEDULE_CONTRACT_V1 = "research_rescaled_horizon_v1"
RESEARCH_SCHEDULE_CONTRACTS = (RESEARCH_SCHEDULE_CONTRACT_V1,)
# Growth must end early enough to leave a settling phase: by default the last
# 30% of H sees no births, culls or resets.  A config may declare a different
# ceiling under ``schedule_contract_fields.refine_stop_fraction_max``.
RESEARCH_REFINE_STOP_MAX_FRACTION = 0.7
# Fields a contract config may pre-declare under
# ``schedule_contract_fields.resolved``; each declared value must equal the
# value resolved from the config, so the declaration is verified, not trusted.
RESEARCH_CONTRACT_DECLARABLE_FIELDS = (
    "horizon_steps",
    "refine_start_iter",
    "refine_stop_iter",
    "refine_every",
    "refine_scale2d_stop_iter",
    "prune_switch_step",
    "reset_every",
    "sh_degree",
    "sh_degree_interval",
    "means_lr_base",
    "means_lr_final_factor",
)
# resolved_schedule checks the contract replaces with its own horizon rule.
_RESEARCH_CONTRACT_REPLACED_CHECKS = frozenset({"max_steps_is_20_view_epochs"})


def means_lr_for_step(
    base_learning_rate: float,
    final_factor: float,
    *,
    step: int,
    max_steps: int,
) -> float:
    """Transcription of ``trainer.means_lr_for_step`` (exponential to final)."""
    if base_learning_rate < 0.0:
        raise ValueError("base means learning rate must be non-negative")
    if not 0.0 < final_factor <= 1.0:
        raise ValueError("means LR final factor must be in (0, 1]")
    if step < 0 or max_steps <= 0:
        raise ValueError("means LR schedule requires non-negative step and max_steps")
    if base_learning_rate == 0.0:
        return 0.0
    return float(base_learning_rate * (final_factor ** (step / max(1, max_steps))))


def active_sh_degree_for_step(
    *, color_model: str, sh_degree: int, sh_degree_interval: int, step: int
) -> int | None:
    """Transcription of ``trainer.active_sh_degree_for_step``."""
    if step < 0:
        raise ValueError("SH schedule step must be non-negative")
    if color_model != "sh" or sh_degree_interval == 0:
        return None
    return min(int(sh_degree), step // int(sh_degree_interval))


def is_refine_step(
    step: int,
    *,
    refine_start_iter: int,
    refine_stop_iter: int,
    refine_every: int,
    exact_mipmap_lifecycle: bool,
) -> bool:
    """Transcription of ``DefaultStrategyAdapter.is_refine_step``."""
    if exact_mipmap_lifecycle:
        return (
            step < refine_stop_iter
            and step >= refine_start_iter
            and step % refine_every == 0
        )
    return (
        step < refine_stop_iter
        and step > refine_start_iter
        and step % refine_every == 0
    )


def _get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key)
    return default if value is None else value


def _resolve_lifecycle_fields(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve the lifecycle window from its duplicated sources and flag drift.

    ``backend.py`` builds the adapter from ``default_strategy`` and only falls
    back to the top-level ``mcmc_refine_*`` values via ``setdefault``; the
    trainer's own LR/refine-boundary logic reads ``mcmc_refine_*`` directly.
    Both therefore have to agree or the adapter and the trainer disagree on
    where the window ends.
    """
    strategy = dict(config.get("default_strategy") or {})
    max_steps = int(config["max_steps"])
    checks: list[dict[str, Any]] = []

    def duplicate(name: str, top_key: str, nested_key: str, top_default: Any) -> Any:
        top = config.get(top_key)
        nested = strategy.get(nested_key)
        resolved = nested if nested is not None else (top if top is not None else top_default)
        both_present = top is not None and nested is not None
        checks.append(
            {
                "name": name,
                "kind": "duplicate_field",
                "fields": {top_key: top, f"default_strategy.{nested_key}": nested},
                "resolved": resolved,
                "ok": (not both_present) or top == nested,
                "note": (
                    "adapter reads default_strategy value, trainer LR/boundary logic reads the top-level value"
                    if both_present and top != nested
                    else None
                ),
            }
        )
        return resolved

    refine_start = int(duplicate("refine_start_iter", "mcmc_refine_start_iter", "refine_start_iter", TRAINER_DEFAULTS["mcmc_refine_start_iter"]))
    refine_stop = int(duplicate("refine_stop_iter", "mcmc_refine_stop_iter", "refine_stop_iter", TRAINER_DEFAULTS["mcmc_refine_stop_iter"]))
    refine_every = int(duplicate("refine_every", "mcmc_refine_every", "refine_every", TRAINER_DEFAULTS["mcmc_refine_every"]))

    scale2d_stop = strategy.get("refine_scale2d_stop_iter")
    checks.append(
        {
            "name": "refine_scale2d_stop_iter_matches_refine_stop",
            "kind": "duplicate_field",
            "fields": {"default_strategy.refine_scale2d_stop_iter": scale2d_stop, "mcmc_refine_stop_iter": config.get("mcmc_refine_stop_iter")},
            "resolved": scale2d_stop,
            "ok": scale2d_stop is None or scale2d_stop == config.get("mcmc_refine_stop_iter"),
            "note": None,
        }
    )

    prune_switch_step = strategy.get("prune_switch_step", ADAPTER_DEFAULTS["prune_switch_step"])
    checks.append(
        {
            "name": "prune_switch_step_is_half_max_steps",
            "kind": "duplicate_field",
            "fields": {"default_strategy.prune_switch_step": prune_switch_step, "max_steps // 2": max_steps // 2},
            "resolved": prune_switch_step,
            "ok": prune_switch_step is None or int(prune_switch_step) == max_steps // 2,
            "note": "trainer enforces prune_switch_step == max_steps // 2 for exact_mipmap_lifecycle",
        }
    )

    reset_every = int(_get(strategy, "reset_every", ADAPTER_DEFAULTS["reset_every"]))
    reset_profile = strategy.get("vendor_opacity_reset_profile", "exact_every300")
    expected_reset = VENDOR_RESET_INTERVALS.get(reset_profile)
    checks.append(
        {
            "name": "reset_every_matches_vendor_opacity_reset_profile",
            "kind": "duplicate_field",
            "fields": {"default_strategy.reset_every": reset_every, f"profile[{reset_profile}]": expected_reset},
            "resolved": reset_every,
            "ok": expected_reset is None or reset_every == expected_reset,
            "note": None,
        }
    )

    prune_opa = float(_get(strategy, "prune_opa", ADAPTER_DEFAULTS["prune_opa"]))
    prune_opa_late = strategy.get("prune_opa_late", ADAPTER_DEFAULTS["prune_opa_late"])
    cull_profile = strategy.get("vendor_cull_warmup_profile", "exact_0p10_to_0p05")
    expected_cull = VENDOR_CULL_THRESHOLDS.get(cull_profile)
    execution_order = strategy.get("lifecycle_execution_order", "post_optimizer_gsplat")
    checks.append(
        {
            "name": "opacity_thresholds_match_vendor_cull_warmup_profile",
            "kind": "duplicate_field",
            "fields": {
                "default_strategy.prune_opa": prune_opa,
                "default_strategy.prune_opa_late": prune_opa_late,
                f"profile[{cull_profile}]": list(expected_cull) if expected_cull else None,
            },
            "resolved": [prune_opa, prune_opa_late],
            "ok": (
                execution_order != "pre_optimizer_vendor"
                or expected_cull is None
                or (prune_opa, None if prune_opa_late is None else float(prune_opa_late)) == expected_cull
            ),
            "note": "profile only enforced under lifecycle_execution_order=pre_optimizer_vendor",
        }
    )

    fields = {
        "exact_mipmap_lifecycle": bool(strategy.get("exact_mipmap_lifecycle", False)),
        "lifecycle_execution_order": execution_order,
        "refine_start_iter": refine_start,
        "refine_stop_iter": refine_stop,
        "refine_every": refine_every,
        "refine_scale2d_stop_iter": scale2d_stop,
        "reset_every": reset_every,
        "reset_opacity_cap": strategy.get("reset_opacity_cap"),
        "reset_before_cull": strategy.get("reset_before_cull"),
        "prune_opa": prune_opa,
        "prune_opa_late": None if prune_opa_late is None else float(prune_opa_late),
        "prune_switch_step": None if prune_switch_step is None else int(prune_switch_step),
        "post_refine_cull_every": strategy.get("post_refine_cull_every", ADAPTER_DEFAULTS["post_refine_cull_every"]),
        "post_refine_cull_until": strategy.get("post_refine_cull_until", ADAPTER_DEFAULTS["post_refine_cull_until"]),
        "relaxed_cull_when_no_growth": strategy.get("relaxed_cull_when_no_growth"),
        "opacity_cull_policy": strategy.get("opacity_cull_policy", "immediate"),
        "cap_max": config.get("cap_max"),
        "grow_grad2d": strategy.get("grow_grad2d"),
        "absgrad": strategy.get("absgrad"),
    }
    return fields, checks


def lifecycle_events(
    *,
    stop_step: int,
    refine_start_iter: int,
    refine_stop_iter: int,
    refine_every: int,
    reset_every: int,
    exact_mipmap_lifecycle: bool,
    prune_opa: float,
    prune_opa_late: float | None,
    prune_switch_step: int | None,
    post_refine_cull_every: int | None,
    post_refine_cull_until: int | None,
) -> list[dict[str, Any]]:
    """Enumerate every lifecycle event a run executing steps ``0..stop_step-1`` performs.

    Mirrors ``DefaultStrategyAdapter._step_post_backward_mipmap``: past
    ``refine_stop_iter`` only the optional post-refine cull runs (early return,
    no growth, no reset); inside the window every refine step grows, resets on
    the ``reset_every`` cadence and culls at the early or late threshold keyed
    on ``prune_switch_step``.  ``densify_allowed`` ignores the capacity cap
    because the population is a runtime fact.
    """
    events: list[dict[str, Any]] = []

    def threshold(step: int) -> tuple[float, str]:
        if prune_switch_step is not None and prune_opa_late is not None and step >= int(prune_switch_step):
            return float(prune_opa_late), "late"
        return float(prune_opa), "early"

    for step in range(0, int(stop_step)):
        if step >= refine_stop_iter:
            every = post_refine_cull_every
            if not every or step % int(every) != 0:
                continue
            if post_refine_cull_until is not None and step > int(post_refine_cull_until):
                continue
            value, phase = threshold(step)
            events.append(
                {
                    "step": step,
                    "kind": "post_refine_cull",
                    "grow": False,
                    "cull": True,
                    "reset": False,
                    "cull_opacity_threshold": value,
                    "threshold_phase": phase,
                }
            )
            continue
        if not is_refine_step(
            step,
            refine_start_iter=refine_start_iter,
            refine_stop_iter=refine_stop_iter,
            refine_every=refine_every,
            exact_mipmap_lifecycle=exact_mipmap_lifecycle,
        ):
            continue
        value, phase = threshold(step)
        events.append(
            {
                "step": step,
                "kind": "refine",
                "grow": True,
                "cull": True,
                "reset": step % int(reset_every) == 0,
                "cull_opacity_threshold": value,
                "threshold_phase": phase,
            }
        )
    return events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(flag: str) -> dict[str, Any]:
        steps = [event["step"] for event in events if event.get(flag)]
        return {
            "count": len(steps),
            "first_step": steps[0] if steps else None,
            "last_step": steps[-1] if steps else None,
        }

    late_steps = [event["step"] for event in events if event["threshold_phase"] == "late"]
    return {
        "grow": stats("grow"),
        "cull": stats("cull"),
        "reset": stats("reset"),
        "post_refine_cull": {
            "count": sum(1 for event in events if event["kind"] == "post_refine_cull"),
        },
        "late_threshold": {
            "ever_applied": bool(late_steps),
            "first_step": late_steps[0] if late_steps else None,
            "event_count": len(late_steps),
        },
    }


def _sh_schedule(config: dict[str, Any], stop_step: int) -> dict[str, Any]:
    color_model = _get(config, "color_model", TRAINER_DEFAULTS["color_model"])
    sh_degree = int(_get(config, "sh_degree", TRAINER_DEFAULTS["sh_degree"]))
    interval = int(_get(config, "sh_degree_interval", TRAINER_DEFAULTS["sh_degree_interval"]))
    transitions: list[dict[str, int]] = []
    if color_model == "sh" and interval > 0:
        for degree in range(0, sh_degree + 1):
            first_step = degree * interval
            if first_step < stop_step:
                transitions.append({"step": first_step, "active_degree": degree})
    final_active = active_sh_degree_for_step(
        color_model=color_model, sh_degree=sh_degree, sh_degree_interval=interval, step=max(0, stop_step - 1)
    )
    return {
        "color_model": color_model,
        "sh_degree": sh_degree,
        "sh_degree_interval": interval,
        # interval 0 disables the progressive schedule: the renderer receives
        # active_sh_degree=None and uses the full configured degree from step 0.
        "progressive": color_model == "sh" and interval > 0,
        "transitions": transitions,
        "active_degree_at_last_executed_step": sh_degree if final_active is None and color_model == "sh" else final_active,
        "full_degree_reached_before_stop": (
            color_model != "sh" or interval == 0 or sh_degree * interval < stop_step
        ),
    }


def _supervision_weights(config: dict[str, Any]) -> dict[str, Any]:
    normal = {**NORMAL_ALIGNMENT_DEFAULTS, **(config.get("lidar_normal_alignment") or {})}
    geometry = {**GEOMETRY_REGULARIZATION_DEFAULTS, **(config.get("geometry_regularization") or {})}
    return {
        "rgb_l1_weight": float(_get(config, "rgb_l1_weight", TRAINER_DEFAULTS["rgb_l1_weight"])),
        "rgb_ssim_weight": float(_get(config, "rgb_ssim_weight", TRAINER_DEFAULTS["rgb_ssim_weight"])),
        "rgb_ssim_mode": config.get("rgb_ssim_mode"),
        "lidar_range_weight": float(_get(config, "lidar_range_weight", TRAINER_DEFAULTS["lidar_range_weight"])),
        "lidar_range_loss_mode": config.get("lidar_range_loss_mode"),
        "lidar_alpha_weight": float(_get(config, "lidar_alpha_weight", TRAINER_DEFAULTS["lidar_alpha_weight"])),
        "lidar_alpha_target": float(_get(config, "lidar_alpha_target", TRAINER_DEFAULTS["lidar_alpha_target"])),
        "surface_alpha_floor_profile": config.get("surface_alpha_floor_profile"),
        "da2_depth_weight": float(_get(config, "da2_depth_weight", TRAINER_DEFAULTS["da2_depth_weight"])),
        "da2_depth_space": config.get("da2_depth_space"),
        "mesh_depth_weight": float(_get(config, "mesh_depth_weight", TRAINER_DEFAULTS["mesh_depth_weight"])),
        "mesh_normal_weight": float(_get(config, "mesh_normal_weight", TRAINER_DEFAULTS["mesh_normal_weight"])),
        "normal_alignment": {
            "enabled": bool(normal["enabled"]),
            "weight_align": float(normal["weight_align"]),
            "weight_flatten": float(normal["weight_flatten"]),
            "weight_tangent_isotropy": float(normal["weight_tangent_isotropy"]),
            "weight_point_to_plane": float(normal["weight_point_to_plane"]),
        },
        "geometry_regularization": {
            "enabled": bool(geometry["enabled"]),
            "opacity_sparsity_weight": float(geometry["opacity_sparsity_weight"]),
            "scale_upper_weight": float(geometry["scale_upper_weight"]),
            "anisotropy_weight": float(geometry["anisotropy_weight"]),
        },
        # Time-varying weight envelope only exists under the competitor schedule;
        # otherwise every weight above is constant for the whole run.
        "competitor_loss_schedule_enabled": bool(
            _get(config, "competitor_loss_schedule_enabled", TRAINER_DEFAULTS["competitor_loss_schedule_enabled"])
        ),
    }


def resolved_schedule(
    config: dict[str, Any],
    view_count: int | None = None,
    *,
    reference_scale_m: float | None = None,
    training_view_count: int | None = None,
) -> dict[str, Any]:
    """Resolve one trainer config dict into the schedule the trainer executes.

    ``view_count`` is the Tile view count from the tile inputs manifest (the
    epoch basis of the signed step budget); ``training_view_count`` is the
    count after any spatial hold-out (the actual epoch length of the sampler)
    and defaults to ``view_count``.  ``reference_scale_m`` is the runtime
    median Gaussian scale; when given, the effective means LR the optimizer
    actually receives (``reference_scale_m * means_step_fraction``) is
    reported next to the nominal one built from ``learning_rates.means``.
    """
    max_steps = int(config["max_steps"])
    controlled_stop = config.get("controlled_stop_after_steps")
    stop_step = int(controlled_stop) if controlled_stop is not None else max_steps
    last_executed_step = stop_step - 1
    lifecycle, checks = _resolve_lifecycle_fields(config)

    # --- views / sampling -------------------------------------------------
    sampling_mode = _get(config, "view_sampling_mode", TRAINER_DEFAULTS["view_sampling_mode"])
    epoch_views = training_view_count if training_view_count is not None else view_count
    views = {
        "tile_view_count": view_count,
        "training_view_count": epoch_views,
        "holdout_configured": config.get("holdout_spatial_cell_m") is not None,
        "view_sampling_mode": sampling_mode,
        "epoch_length": epoch_views if sampling_mode == "fisher_yates_without_replacement_per_epoch" else None,
        "configured_epochs": None if not view_count else max_steps / view_count,
        "executed_epochs": None if not epoch_views else stop_step / epoch_views,
        "average_visits_per_image": None if not epoch_views else stop_step / epoch_views,
    }
    if view_count and config.get("schedule_contract") is not None:
        # A research contract replaces the 20-epoch rule with H = max_steps;
        # the epoch count becomes a recorded fact instead of a requirement.
        checks.append(
            {
                "name": "max_steps_is_contract_horizon",
                "kind": "consistency",
                "fields": {
                    "max_steps": max_steps,
                    "schedule_contract": config.get("schedule_contract"),
                    "epochs_over_tile_views": max_steps / view_count,
                },
                "resolved": max_steps,
                "ok": True,
                "note": "schedule_contract declares max_steps as the horizon; the 20-epoch pre-flight rule does not apply",
            }
        )
    elif view_count:
        checks.append(
            {
                "name": "max_steps_is_20_view_epochs",
                "kind": "consistency",
                "fields": {"max_steps": max_steps, f"{MIPMAP_EPOCHS_PER_RUN} * tile_view_count": MIPMAP_EPOCHS_PER_RUN * view_count},
                "resolved": max_steps,
                "ok": max_steps == MIPMAP_EPOCHS_PER_RUN * view_count,
                "note": "trainer pre-flight requirement for fisher_yates sampling with adaptive_growth",
            }
        )

    # --- means LR --------------------------------------------------------
    learning_rates = dict(config.get("learning_rates") or {})
    nominal_base = float(learning_rates.get("means", 0.0))
    final_factor = float(_get(config, "means_lr_final_factor", TRAINER_DEFAULTS["means_lr_final_factor"]))
    calibration = dict(config.get("metric_scale_calibration") or {})
    means_step_fraction = calibration.get("means_step_fraction", TRAINER_DEFAULTS["means_step_fraction"])
    effective_base: float | None
    if means_step_fraction is None:
        effective_base = nominal_base
        effective_note = "means_step_fraction is null: optimizer uses learning_rates.means"
    elif reference_scale_m is not None:
        effective_base = float(reference_scale_m) * float(means_step_fraction)
        effective_note = "optimizer uses reference_scale_m * means_step_fraction (scale_calibration.py), not learning_rates.means"
    else:
        effective_base = None
        effective_note = (
            "means_step_fraction is set, so the optimizer base LR is reference_scale_m * means_step_fraction; "
            "reference_scale_m is a runtime fact that was not supplied"
        )
    post_refine_scale = float(_get(config, "post_refine_geometry_lr_scale", TRAINER_DEFAULTS["post_refine_geometry_lr_scale"]))
    key_steps = {
        "start": 0,
        "refine_stop": lifecycle["refine_stop_iter"],
        "last_executed": last_executed_step,
        "declared_final": max_steps,
    }

    def lr_table(base: float) -> dict[str, float]:
        table: dict[str, float] = {}
        for name, step in key_steps.items():
            value = means_lr_for_step(base, final_factor, step=min(step, max_steps), max_steps=max_steps)
            # Past refine stop the trainer multiplies the geometry LR again.
            if step >= lifecycle["refine_stop_iter"]:
                value *= post_refine_scale
            table[name] = value
        return table

    nominal_table = lr_table(nominal_base)
    means_lr = {
        "formula": "base * final_factor ** (step / max_steps); geometry x post_refine_geometry_lr_scale for step >= refine_stop_iter",
        "denominator_max_steps": max_steps,
        "final_factor": final_factor,
        "post_refine_geometry_lr_scale": post_refine_scale,
        "key_steps": key_steps,
        "nominal": {"base": nominal_base, **nominal_table},
        "nominal_last_executed_over_declared_final": (
            None if nominal_table["declared_final"] == 0 else nominal_table["last_executed"] / nominal_table["declared_final"]
        ),
        "decay_fraction_completed": last_executed_step / max_steps,
        "effective": None,
        "effective_note": effective_note,
        "means_step_fraction": means_step_fraction,
        "reference_scale_m": reference_scale_m,
    }
    if effective_base is not None:
        effective_table = lr_table(effective_base)
        means_lr["effective"] = {"base": effective_base, **effective_table}
    checks.append(
        {
            "name": "controlled_stop_reaches_declared_final_lr",
            "kind": "consistency",
            "fields": {"controlled_stop_after_steps": controlled_stop, "max_steps": max_steps},
            "resolved": stop_step,
            "ok": controlled_stop is None or int(controlled_stop) >= max_steps,
            "note": (
                None
                if controlled_stop is None
                else f"run stops with {means_lr['nominal_last_executed_over_declared_final']:.2f}x the declared final means LR"
            ),
        }
    )
    checks.append(
        {
            "name": "learning_rates_means_is_the_optimizer_base",
            "kind": "consistency",
            "fields": {"learning_rates.means": nominal_base, "metric_scale_calibration.means_step_fraction": means_step_fraction},
            "resolved": effective_base,
            "ok": means_step_fraction is None,
            "note": None if means_step_fraction is None else effective_note,
        }
    )

    # --- lifecycle events -------------------------------------------------
    events = lifecycle_events(
        stop_step=stop_step,
        refine_start_iter=lifecycle["refine_start_iter"],
        refine_stop_iter=lifecycle["refine_stop_iter"],
        refine_every=lifecycle["refine_every"],
        reset_every=lifecycle["reset_every"],
        exact_mipmap_lifecycle=lifecycle["exact_mipmap_lifecycle"],
        prune_opa=lifecycle["prune_opa"],
        prune_opa_late=lifecycle["prune_opa_late"],
        prune_switch_step=lifecycle["prune_switch_step"],
        post_refine_cull_every=lifecycle["post_refine_cull_every"],
        post_refine_cull_until=lifecycle["post_refine_cull_until"],
    )
    summary = summarize_events(events)
    switch = lifecycle["prune_switch_step"]
    checks.append(
        {
            "name": "late_opacity_threshold_reachable",
            "kind": "consistency",
            "fields": {
                "default_strategy.prune_switch_step": switch,
                "refine_stop_iter": lifecycle["refine_stop_iter"],
                "stop_step": stop_step,
                "post_refine_cull_every": lifecycle["post_refine_cull_every"],
            },
            "resolved": summary["late_threshold"]["first_step"],
            "ok": switch is None or summary["late_threshold"]["ever_applied"],
            "note": (
                None
                if switch is None or summary["late_threshold"]["ever_applied"]
                else "lifecycle returns early for step >= refine_stop_iter, so prune_opa_late never applies"
            ),
        }
    )
    checks.append(
        {
            "name": "refine_window_ends_before_stop",
            "kind": "consistency",
            "fields": {"refine_stop_iter": lifecycle["refine_stop_iter"], "stop_step": stop_step},
            "resolved": lifecycle["refine_stop_iter"],
            "ok": lifecycle["refine_stop_iter"] <= stop_step,
            "note": None,
        }
    )
    checks.append(
        {
            "name": "reset_cadence_aligns_with_refine_cadence",
            "kind": "consistency",
            "fields": {"reset_every": lifecycle["reset_every"], "refine_every": lifecycle["refine_every"]},
            "resolved": lifecycle["reset_every"],
            "ok": lifecycle["reset_every"] % lifecycle["refine_every"] == 0,
            "note": "resets only fire on refine steps; a cadence that is not a multiple of refine_every silently drops resets",
        }
    )

    last_growth = summary["grow"]["last_step"]
    last_reset = summary["reset"]["last_step"]
    opportunity = {
        "last_growth_step": last_growth,
        "last_reset_step": last_reset,
        # Growth runs pre-optimizer, so the birth step itself already
        # optimizes the new Gaussians: steps last_growth .. stop_step-1.
        "optimizer_steps_after_last_growth": None if last_growth is None else stop_step - last_growth,
        "average_visits_per_image_after_last_growth": (
            None if last_growth is None or not epoch_views else (stop_step - last_growth) / epoch_views
        ),
        "optimizer_steps_after_last_reset": None if last_reset is None else stop_step - last_reset,
        "average_visits_per_image_after_last_reset": (
            None if last_reset is None or not epoch_views else (stop_step - last_reset) / epoch_views
        ),
        "means_lr_at_last_growth_nominal": (
            None if last_growth is None else means_lr_for_step(nominal_base, final_factor, step=last_growth, max_steps=max_steps)
        ),
    }

    return {
        "schema_version": 1,
        "kind": "cloudstudio_training_schedule_audit",
        "run_id": config.get("run_id"),
        "steps": {
            "max_steps": max_steps,
            "controlled_stop_after_steps": controlled_stop,
            "stop_step": stop_step,
            "last_executed_step": last_executed_step,
            "executed_fraction_of_max_steps": stop_step / max_steps,
            "checkpoint_every": config.get("checkpoint_every"),
        },
        "views": views,
        "means_lr": means_lr,
        "sh_degree": _sh_schedule(config, stop_step),
        "lifecycle": lifecycle,
        "event_summary": summary,
        "events": events,
        "opportunity_after_last_birth": opportunity,
        "supervision_weights": _supervision_weights(config),
        "consistency_checks": checks,
        "mismatches": [check for check in checks if not check["ok"]],
    }


def research_schedule_contract(
    config: dict[str, Any],
    *,
    training_view_count: int | None = None,
    holdout_view_count: int | None = None,
) -> dict[str, Any]:
    """Resolve a config's research schedule contract and list every violation.

    Pure: the returned record carries the horizon H = ``max_steps``, every
    linked field both in steps and as a fraction of H (or a multiple of
    ``refine_every``), the lifecycle event summary, the checks inherited from
    ``resolved_schedule`` and the contract's own checks.  ``violations`` is the
    subset of checks that failed; ``validate_research_schedule_contract``
    raises on any.  View counts are runtime facts the trainer supplies at
    pre-flight; without them the epoch figures are ``None``.
    """
    name = config.get("schedule_contract")
    if name is None:
        raise ValueError("config declares no schedule_contract")
    if name not in RESEARCH_SCHEDULE_CONTRACTS:
        raise ValueError(
            f"unknown schedule_contract {name!r}; expected one of {RESEARCH_SCHEDULE_CONTRACTS}"
        )
    declared = config.get("schedule_contract_fields")
    if declared is None:
        declared = {}
    if not isinstance(declared, dict):
        raise ValueError("schedule_contract_fields must be an object")
    refine_stop_fraction_max = float(
        _get(declared, "refine_stop_fraction_max", RESEARCH_REFINE_STOP_MAX_FRACTION)
    )
    if not 0.0 < refine_stop_fraction_max <= 1.0:
        raise ValueError("schedule_contract_fields.refine_stop_fraction_max must be in (0, 1]")

    horizon = int(config["max_steps"])
    if horizon <= 0:
        raise ValueError("schedule contract horizon (max_steps) must be positive")
    schedule = resolved_schedule(
        config,
        None,
        training_view_count=training_view_count,
    )
    lifecycle = schedule["lifecycle"]
    strategy = dict(config.get("default_strategy") or {})
    controlled_stop = config.get("controlled_stop_after_steps")
    stop_step = schedule["steps"]["stop_step"]
    refine_start = int(lifecycle["refine_start_iter"])
    refine_stop = int(lifecycle["refine_stop_iter"])
    refine_every = int(lifecycle["refine_every"])
    scale2d_stop = lifecycle["refine_scale2d_stop_iter"]
    prune_switch = lifecycle["prune_switch_step"]
    reset_every = int(lifecycle["reset_every"])
    sh = schedule["sh_degree"]
    calibration = config.get("metric_scale_calibration")
    means_lr = schedule["means_lr"]
    summary = schedule["event_summary"]

    checks: list[dict[str, Any]] = [
        check
        for check in schedule["consistency_checks"]
        if check["name"] not in _RESEARCH_CONTRACT_REPLACED_CHECKS
    ]

    def contract_check(check_name: str, ok: bool, fields: dict[str, Any], note: str) -> None:
        checks.append(
            {
                "name": check_name,
                "kind": "research_schedule_contract",
                "fields": fields,
                "resolved": None,
                "ok": bool(ok),
                "note": note,
            }
        )

    contract_check(
        "stop_step_is_horizon",
        controlled_stop is None or int(controlled_stop) == horizon,
        {"max_steps": horizon, "controlled_stop_after_steps": controlled_stop},
        "H = max_steps; a controlled stop may only restate H, never truncate it",
    )
    contract_check(
        "refine_window_inside_horizon",
        0 < refine_start < refine_stop <= refine_stop_fraction_max * horizon,
        {
            "refine_start_iter": refine_start,
            "refine_stop_iter": refine_stop,
            "refine_stop_fraction": refine_stop / horizon,
            "refine_stop_fraction_max": refine_stop_fraction_max,
        },
        "growth window must sit inside (0, refine_stop_fraction_max * H] so a settling phase remains",
    )
    contract_check(
        "refine_window_is_refine_every_aligned",
        refine_every > 0 and refine_start % refine_every == 0 and refine_stop % refine_every == 0,
        {"refine_start_iter": refine_start, "refine_stop_iter": refine_stop, "refine_every": refine_every},
        "window bounds are multiples of refine_every so the event table has no partial interval",
    )
    contract_check(
        "refine_scale2d_stop_declared_and_equal",
        scale2d_stop is not None and int(scale2d_stop) == refine_stop,
        {"default_strategy.refine_scale2d_stop_iter": scale2d_stop, "refine_stop_iter": refine_stop},
        "screen-size split gate stops with the growth window (trainer expects equality)",
    )
    contract_check(
        "late_threshold_declared",
        prune_switch is not None and lifecycle["prune_opa_late"] is not None and 0 < int(prune_switch) < horizon,
        {
            "default_strategy.prune_switch_step": prune_switch,
            "default_strategy.prune_opa_late": lifecycle["prune_opa_late"],
            "prune_switch_fraction": None if prune_switch is None else int(prune_switch) / horizon,
        },
        "the late opacity threshold and its switch step must be declared as a fraction of H",
    )
    contract_check(
        "reset_fires_inside_refine_window",
        summary["reset"]["count"] >= 1,
        {"reset_every": reset_every, "refine_window": [refine_start, refine_stop], "reset_count": summary["reset"]["count"]},
        "reset_every is a multiple of refine_every (inherited check) and short enough to fire at least once",
    )
    contract_check(
        "sh_full_degree_reached_before_horizon",
        bool(sh["full_degree_reached_before_stop"]),
        {
            "sh_degree": sh["sh_degree"],
            "sh_degree_interval": sh["sh_degree_interval"],
            "sh_full_degree_step": sh["sh_degree"] * sh["sh_degree_interval"],
            "sh_full_degree_fraction": sh["sh_degree"] * sh["sh_degree_interval"] / horizon,
        },
        "interval 0 disables the progressive schedule; otherwise the full degree must activate before H",
    )
    means_step_fraction_explicit_null = (
        isinstance(calibration, dict)
        and "means_step_fraction" in calibration
        and calibration["means_step_fraction"] is None
    )
    contract_check(
        "means_step_fraction_explicitly_null",
        means_step_fraction_explicit_null,
        {
            "metric_scale_calibration.means_step_fraction": (
                calibration.get("means_step_fraction", "<absent: trainer default>")
                if isinstance(calibration, dict)
                else "<absent: trainer default>"
            ),
            "learning_rates.means": means_lr["nominal"]["base"],
        },
        "learning_rates.means is the optimizer base only when the scale-calibration override is explicitly null",
    )

    resolved_fields = {
        "horizon_steps": horizon,
        "refine_start_iter": refine_start,
        "refine_stop_iter": refine_stop,
        "refine_every": refine_every,
        "refine_scale2d_stop_iter": None if scale2d_stop is None else int(scale2d_stop),
        "prune_switch_step": None if prune_switch is None else int(prune_switch),
        "reset_every": reset_every,
        "sh_degree": sh["sh_degree"],
        "sh_degree_interval": sh["sh_degree_interval"],
        "means_lr_base": means_lr["nominal"]["base"],
        "means_lr_final_factor": means_lr["final_factor"],
    }
    declared_resolved = declared.get("resolved")
    if declared_resolved is not None:
        if not isinstance(declared_resolved, dict):
            raise ValueError("schedule_contract_fields.resolved must be an object")
        unknown = sorted(set(declared_resolved) - set(RESEARCH_CONTRACT_DECLARABLE_FIELDS))
        if unknown:
            raise ValueError(f"schedule_contract_fields.resolved has undeclarable keys: {unknown}")
        disagreements = {
            key: {"declared": declared_resolved[key], "resolved": resolved_fields[key]}
            for key in declared_resolved
            if declared_resolved[key] != resolved_fields[key]
        }
        contract_check(
            "declared_fields_match_resolved",
            not disagreements,
            {"disagreements": disagreements},
            "every pre-declared field must equal the value the config actually resolves to",
        )

    views: dict[str, Any] = {
        "training_view_count": training_view_count,
        "holdout_view_count": holdout_view_count,
        "epochs_over_training_views": (
            None if not training_view_count else horizon / training_view_count
        ),
        "epochs_over_tile_views": (
            None
            if not training_view_count
            else horizon / (training_view_count + int(holdout_view_count or 0))
        ),
        "parity_20_epoch_max_steps": (
            None
            if not training_view_count
            else MIPMAP_EPOCHS_PER_RUN * (training_view_count + int(holdout_view_count or 0))
        ),
    }
    violations = [check for check in checks if not check["ok"]]
    return {
        "schema_version": 1,
        "kind": "cloudstudio_research_schedule_contract",
        "name": name,
        "competitor_parity": False,
        "horizon_steps": horizon,
        "stop_step": stop_step,
        "controlled_stop_after_steps": controlled_stop,
        "resolved_fields": resolved_fields,
        "horizon_fractions": {
            "refine_start": refine_start / horizon,
            "refine_stop": refine_stop / horizon,
            "refine_stop_max": refine_stop_fraction_max,
            "refine_scale2d_stop": None if scale2d_stop is None else int(scale2d_stop) / horizon,
            "prune_switch": None if prune_switch is None else int(prune_switch) / horizon,
            "reset_every": reset_every / horizon,
            "sh_full_degree_step": sh["sh_degree"] * sh["sh_degree_interval"] / horizon,
        },
        "multiples_of_refine_every": {
            "refine_start": refine_start / refine_every if refine_every else None,
            "refine_stop": refine_stop / refine_every if refine_every else None,
            "reset_every": reset_every / refine_every if refine_every else None,
        },
        "means_lr": {
            "authoritative_field": "learning_rates.means",
            "base": means_lr["nominal"]["base"],
            "final_factor": means_lr["final_factor"],
            "declared_final": means_lr["nominal"]["declared_final"],
            "at_last_executed_step": means_lr["nominal"]["last_executed"],
            "at_refine_stop": means_lr["nominal"]["refine_stop"],
        },
        "late_threshold": summary["late_threshold"],
        "event_summary": summary,
        "opportunity_after_last_birth": schedule["opportunity_after_last_birth"],
        "views": views,
        "checks": checks,
        "violations": violations,
    }


def validate_research_schedule_contract(
    config: dict[str, Any],
    *,
    training_view_count: int | None = None,
    holdout_view_count: int | None = None,
) -> dict[str, Any]:
    """``research_schedule_contract`` that raises ``ValueError`` on any violation."""
    record = research_schedule_contract(
        config,
        training_view_count=training_view_count,
        holdout_view_count=holdout_view_count,
    )
    if record["violations"]:
        summary = "; ".join(
            f"{check['name']}: {check['fields']}" for check in record["violations"]
        )
        raise ValueError(
            f"research schedule contract {record['name']!r} violated: {summary}"
        )
    return record


def diff_configs(repo: Any, as_run: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Flat list of leaf differences between two config trees."""
    differences: list[dict[str, Any]] = []
    if isinstance(repo, dict) and isinstance(as_run, dict):
        for key in sorted(set(repo) | set(as_run)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in repo:
                differences.append({"path": path, "repo": None, "as_run": as_run[key], "state": "only_in_as_run"})
            elif key not in as_run:
                differences.append({"path": path, "repo": repo[key], "as_run": None, "state": "only_in_repo"})
            else:
                differences.extend(diff_configs(repo[key], as_run[key], path))
        return differences
    if repo != as_run:
        differences.append({"path": prefix, "repo": repo, "as_run": as_run, "state": "differs"})
    return differences
