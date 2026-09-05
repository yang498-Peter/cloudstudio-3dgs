"""Expose gsplat's reference 3DGS densification through the MCMC call surface.

This trainer replaced classic densification with a homegrown error-weighted MCMC
sampler, and measurement on 2026-08-25 showed that substitution is what degrades
the pictures. Enabling births and changing nothing else costs structural
agreement with the photograph (0.339 -> 0.291) while PSNR rises, and the reason
is where the births land: the image-error map this trainer samples from
overlaps the regions that actually hold missing detail by only 22%, and the
pixels it does prefer carry roughly a quarter of the photo texture of the ones
it skips. Image error marks where the model is wrong, and occlusion, specularity
and exposure mismatch are all wrong without any structure behind them.

Kerbl et al.'s criterion does not share that failure mode. It scores each
Gaussian by the magnitude of the loss gradient with respect to its PROJECTED
position, so a flat region with an exposure offset produces error but almost no
signal - nothing pulls a Gaussian sideways there - while genuine structural
misfit does. gsplat ships that criterion as ``DefaultStrategy``, and the honest
move is to run the reference implementation rather than invent a third scheme.

Two published refinements are exposed as flags rather than baked in:
``absgrad`` (AbsGS) sums per-pixel gradient magnitudes instead of letting
opposing gradients cancel, which is what otherwise leaves large Gaussians
unsplit, and ``revised_opacity`` applies the corrected opacity for split
children.

The adapter exists because the backend and trainer read MCMC-specific
attributes in eight places. Rather than edit each one and risk the existing path,
this presents the same surface:

    min_opacity                 -> DefaultStrategy.prune_opa
    noise_injection_stop_iter   -> 0, since classic densification injects none
    step_post_backward(lr=...)  -> lr dropped; only MCMC's noise term uses it

One difference is load-bearing rather than cosmetic: ``step_pre_backward`` is a
no-op under MCMC and is what retains the means2d gradient under DefaultStrategy.
It is never called today, so the backend must start calling it or the criterion
silently sees no gradient at all.
"""

from __future__ import annotations

import math
from typing import Any

DENSIFICATION_STRATEGIES = ("error_weighted_mcmc", "default_3dgs")


_QUANTILE_INPUT_LIMIT = 1 << 24
_QUANTILE_SAMPLE_SIZE = 1 << 20


def _sampled_quantile(values: Any, levels: Any) -> Any:
    """torch.quantile, but on a bounded sample when the input is large.

    torch.quantile refuses inputs beyond 2**24 elements, which a healthy
    population crosses mid-run - and these are telemetry percentiles, so
    losing the run to a diagnostic is the worst possible trade. A million
    element sample carries a percentile far more precisely than it is ever
    read. The sample uses its own generator so training RNG is untouched and
    the reported numbers stay reproducible.
    """
    import torch

    if values.numel() > _QUANTILE_INPUT_LIMIT:
        generator = torch.Generator(device=values.device)
        generator.manual_seed(0)
        index = torch.randint(
            values.numel(),
            (_QUANTILE_SAMPLE_SIZE,),
            device=values.device,
            generator=generator,
        )
        values = values[index]
    return torch.quantile(values, levels)


# Detail split thresholds the validators accept. 0.02 m is the value the
# adaptive-detail split was first tuned with; the smaller values exist because
# split children of 30-45 mm parents land at 18-28 mm and never split again
# under a 20 mm gate, so the population cannot get smaller than ~12 mm.
DETAIL_SPLIT_SCALES_M = (0.02, 0.01, 0.005)


def _restart_adam_step(optimizer, parameter) -> None:
    """Zero Adam's step counter for ``parameter`` so bias correction restarts."""
    if optimizer is None:
        return
    state = optimizer.state.get(parameter)
    if not state or "step" not in state:
        return
    step = state["step"]
    if hasattr(step, "zero_"):
        step.zero_()
    else:
        state["step"] = 0


class DefaultStrategyAdapter:
    """gsplat ``DefaultStrategy`` behind the attribute surface MCMC exposes."""

    def __init__(
        self,
        *,
        scene_scale: float = 1.0,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 15000,
        refine_every: int = 100,
        reset_every: int = 3000,
        pause_refine_after_reset: int = 0,
        grow_grad2d: float = 0.0002,
        prune_opa: float = 0.005,
        # Metric thresholds are the product-facing form. Upstream normalises by
        # scene_scale, and passing the wrong quantity there is silent and severe:
        # an early run here passed the median initial Gaussian size (0.058 m)
        # instead of the scene extent, which set the split threshold to 0.58 mm
        # and the prune threshold to 5.8 mm against 1-8 cm Gaussians. Every
        # high-gradient Gaussian was therefore split and never cloned, the model
        # grew to 694k, and the arm looked like evidence against the method.
        split_scale_m: float | None = None,
        prune_scale_m: float | None = None,
        grow_scale3d: float = 0.01,
        prune_scale3d: float = 0.1,
        # Screen-space growth and pruning. Upstream gates both behind
        # refine_scale2d_stop_iter, which is 0 by default and disables them -
        # so a footprint problem cannot be addressed without setting it.
        grow_scale2d: float = 0.05,
        prune_scale2d: float = 0.15,
        refine_scale2d_stop_iter: int = 0,
        absgrad: bool = False,
        revised_opacity: bool = False,
        verbose: bool = False,
        exact_mipmap_lifecycle: bool = False,
        growth_min_opacity: float | None = None,
        prune_opa_late: float | None = None,
        prune_switch_step: int | None = None,
        reset_opacity_cap: float | None = None,
        reset_adam_step: bool = False,
        reset_optimizer_state: str = "zero_moments",
        reset_before_cull: bool = False,
        capacity_cap: int | None = None,
        surface_birth_proposal: Any | None = None,
        opacity_cull_policy: str = "immediate",
        opacity_cull_min_observations: int = 0,
        opacity_cull_consecutive_events: int = 1,
        opacity_cull_grace_after_reset_steps: int = 0,
        opacity_cull_max_fraction: float = 1.0,
        opacity_cull_priority: str = "lowest_opacity",
        opacity_cull_local_voxel_m: float = 0.02,
        opacity_cull_local_protection: str = "opacity",
        opacity_cull_local_min_accumulated_alpha: float = 0.0,
        # Fraction of this window's median visible alpha (among seen gaussians)
        # below which a low-opacity row counts as non-contributing and may die.
        contribution_cull_relative_floor: float = 0.05,
        # Kept fraction of accumulated visible alpha across a lifecycle event.
        # Must be <1 so a corpse decays out, and >0 so a contributor survives
        # the post-reset flush that a zeroed window cannot protect it through.
        contribution_history_decay: float = 0.5,
        detail_split_policy: str = "vendor_0_2m",
        detail_split_scale_m: float = 0.02,
        detail_split_screen_radius: float = 0.0035,
        capacity_conserving_clone_opacity: bool = False,
        lifecycle_execution_order: str = "post_optimizer_gsplat",
        vendor_cull_warmup_profile: str = "exact_0p10_to_0p05",
        vendor_opacity_reset_profile: str = "exact_every300",
        lifecycle_dry_run: bool = False,
        relaxed_cull_when_no_growth: bool = False,
        relaxed_cull_at_capacity: bool = True,
        growth_metric: str = "count_mean",
    ) -> None:
        from gsplat.strategy import DefaultStrategy

        self.scene_scale = float(scene_scale)
        if self.scene_scale <= 0.0:
            raise ValueError("scene_scale must be positive")
        # Metric thresholds win when given, so a config can state what it means
        # in metres instead of in units of a scale factor it cannot see.
        self.split_scale_m = split_scale_m
        self.prune_scale_m = prune_scale_m
        self.exact_mipmap_lifecycle = bool(exact_mipmap_lifecycle)
        self.growth_min_opacity = growth_min_opacity
        self.prune_opa_late = prune_opa_late
        self.prune_switch_step = prune_switch_step
        self.reset_opacity_cap = reset_opacity_cap
        # The library reset zeroes Adam's moments but keeps its step counter,
        # so bias correction stays saturated and the first post-reset updates
        # are (1-b1^k)/sqrt((1-b2^k)/(1-b2^t)) ~ 2-4x the learning rate for
        # ~100 steps. Restarting the counter makes them lr-sized again. Kept
        # off by default: an integration audit knob, not a lifecycle fact.
        self.reset_adam_step = bool(reset_adam_step)
        # What the opacity reset does to the opacities' Adam state:
        #   "zero_moments" - the library behaviour (moments zeroed, step kept);
        #   "keep"         - nothing touched: moments and step carry over, so the
        #                    momentum a row had before the clamp keeps acting.
        if reset_optimizer_state not in {"zero_moments", "keep"}:
            raise ValueError("reset_optimizer_state must be zero_moments or keep")
        self.reset_optimizer_state = str(reset_optimizer_state)
        # Execution order inside one refine step: grow -> cull -> reset (library)
        # or grow -> reset -> cull.
        self.reset_before_cull = bool(reset_before_cull)
        self.capacity_cap = None if capacity_cap is None else int(capacity_cap)
        if self.capacity_cap is not None and self.capacity_cap <= 4:
            raise ValueError("capacity_cap must be greater than four")
        self.surface_birth_proposal = surface_birth_proposal
        self.opacity_cull_policy = str(opacity_cull_policy)
        self.opacity_cull_min_observations = int(opacity_cull_min_observations)
        self.opacity_cull_consecutive_events = int(
            opacity_cull_consecutive_events
        )
        self.opacity_cull_grace_after_reset_steps = int(
            opacity_cull_grace_after_reset_steps
        )
        self.opacity_cull_max_fraction = float(opacity_cull_max_fraction)
        self.opacity_cull_priority = str(opacity_cull_priority)
        self.opacity_cull_local_voxel_m = float(opacity_cull_local_voxel_m)
        self.opacity_cull_local_protection = str(
            opacity_cull_local_protection
        )
        self.opacity_cull_local_min_accumulated_alpha = float(
            opacity_cull_local_min_accumulated_alpha
        )
        self.contribution_cull_relative_floor = float(
            contribution_cull_relative_floor
        )
        self.contribution_history_decay = float(contribution_history_decay)
        self.detail_split_policy = str(detail_split_policy)
        self.detail_split_scale_m = float(detail_split_scale_m)
        self.detail_split_screen_radius = float(detail_split_screen_radius)
        self.capacity_conserving_clone_opacity = bool(
            capacity_conserving_clone_opacity
        )
        self.lifecycle_execution_order = str(lifecycle_execution_order)
        # Audit mode: compute every selection, quantile and would-be count at
        # each refine boundary, publish them, and apply NOTHING - no clone, no
        # split, no cull, no opacity reset. One 500-step probe under this flag
        # replaces a thousand steps of guessing which side of the birth/death
        # contract is broken.
        self.lifecycle_dry_run = bool(lifecycle_dry_run)
        self.relaxed_cull_when_no_growth = bool(relaxed_cull_when_no_growth)
        # Whether sitting at the capacity cap counts as "cannot densify" for the
        # anti-starvation branch. It does by the recovered contract, but a run
        # that saturates its cap for tens of thousands of steps is a regime the
        # reference never enters, and there relaxation stops removing dead mass:
        # measured here, near-transparent share climbed 18.6% -> 28.8% after the
        # ceiling was reached, which is exactly the axis where the reference
        # delivery leads us (alpha p05 0.425 vs 0.238). Separating the two
        # triggers lets that be tested rather than assumed.
        self.relaxed_cull_at_capacity = bool(relaxed_cull_at_capacity)
        # Which statistic the growth threshold is compared against.
        # "count_mean": per-axis screen scaling, plain per-observation mean
        #   (the published DefaultStrategy).
        # "footprint_weighted": norm first, isotropic 0.5*max(1600,W,H) scale,
        #   mean weighted by each view's projected radius (recovered contract).
        # The two differ by roughly an order of magnitude on this scene, so a
        # threshold carried across without the matching statistic selects a
        # completely different population.
        if growth_metric not in {"count_mean", "footprint_weighted"}:
            raise ValueError(
                "growth_metric must be count_mean or footprint_weighted"
            )
        self.growth_metric = str(growth_metric)
        self.vendor_cull_warmup_profile = str(vendor_cull_warmup_profile)
        self.vendor_opacity_reset_profile = str(vendor_opacity_reset_profile)
        self.last_lifecycle_event: dict[str, Any] | None = None
        self._last_surface_birth_event: dict[str, Any] | None = None
        self._last_growth_event: dict[str, Any] | None = None
        self._last_cull_event: dict[str, Any] | None = None
        if self.opacity_cull_policy not in {
            "immediate",
            "observation_aware",
            "local_coverage_competition",
            "contribution_aware",
        }:
            raise ValueError(
                "opacity_cull_policy must be immediate, observation_aware, "
                "local_coverage_competition, or contribution_aware"
            )
        if not 0.0 < self.contribution_history_decay < 1.0:
            raise ValueError(
                'contribution_history_decay must be within (0, 1)'
            )
        if not 0.0 <= self.contribution_cull_relative_floor <= 1.0:
            raise ValueError(
                "contribution_cull_relative_floor must be within [0, 1]"
            )
        if self.opacity_cull_min_observations < 0:
            raise ValueError("opacity_cull_min_observations must be non-negative")
        if self.opacity_cull_consecutive_events <= 0:
            raise ValueError("opacity_cull_consecutive_events must be positive")
        if self.opacity_cull_grace_after_reset_steps < 0:
            raise ValueError(
                "opacity_cull_grace_after_reset_steps must be non-negative"
            )
        if not 0.0 < self.opacity_cull_max_fraction <= 1.0:
            raise ValueError("opacity_cull_max_fraction must be within (0, 1]")
        if self.opacity_cull_priority not in {
            "lowest_opacity",
            "lowest_opacity_per_footprint",
        }:
            raise ValueError("opacity_cull_priority is invalid")
        if self.opacity_cull_local_voxel_m <= 0.0:
            raise ValueError("opacity_cull_local_voxel_m must be positive")
        if self.opacity_cull_local_protection not in {
            "opacity",
            "opacity_tangent_area",
        }:
            raise ValueError("opacity_cull_local_protection is invalid")
        if not 0.0 <= self.opacity_cull_local_min_accumulated_alpha < 1.0:
            raise ValueError(
                "opacity_cull_local_min_accumulated_alpha must be within [0, 1)"
            )
        if self.detail_split_policy not in {
            "vendor_0_2m",
            "lidar_surface_screen_detail",
        }:
            raise ValueError("detail_split_policy is invalid")
        if self.lifecycle_execution_order not in {
            "post_optimizer_gsplat",
            "pre_optimizer_vendor",
        }:
            raise ValueError("lifecycle_execution_order is invalid")
        if (
            self.lifecycle_execution_order == "pre_optimizer_vendor"
            and not self.exact_mipmap_lifecycle
        ):
            raise ValueError(
                "pre_optimizer_vendor requires exact_mipmap_lifecycle"
            )
        if self.vendor_cull_warmup_profile not in {
            "exact_0p10_to_0p05",
            "compatibility_uniform_0p05",
            "calibrated_uniform_0p04",
            "calibrated_geometry_only_0p00",
            "audit_uniform_0p005",
        }:
            raise ValueError("vendor_cull_warmup_profile is invalid")
        vendor_reset_intervals = {
            "exact_every300": 300,
            "deferred_every3000_compatibility": 3000,
        }
        expected_reset_every = vendor_reset_intervals.get(
            self.vendor_opacity_reset_profile
        )
        if expected_reset_every is None:
            raise ValueError("vendor_opacity_reset_profile is invalid")
        if self.lifecycle_execution_order == "pre_optimizer_vendor":
            if prune_opa_late is None:
                raise ValueError(
                    "pre_optimizer_vendor requires prune_opa_late"
                )
            expected_cull_thresholds = {
                "exact_0p10_to_0p05": (0.1, 0.05),
                "compatibility_uniform_0p05": (0.05, 0.05),
                "calibrated_uniform_0p04": (0.04, 0.04),
                "calibrated_geometry_only_0p00": (0.0, 0.0),
                # Audit reference only: the published Kerbl et al. cull threshold,
                # to ask whether the reset dynamics collapse under it. Not a vendor fact.
                "audit_uniform_0p005": (0.005, 0.005),
            }[self.vendor_cull_warmup_profile]
            actual_cull_thresholds = (float(prune_opa), float(prune_opa_late))
            if actual_cull_thresholds != expected_cull_thresholds:
                raise ValueError(
                    "vendor cull warm-up profile does not match opacity "
                    f"thresholds: expected={expected_cull_thresholds}, "
                    f"actual={actual_cull_thresholds}"
                )
            if int(reset_every) != expected_reset_every:
                raise ValueError(
                    "vendor opacity reset profile does not match reset interval: "
                    f"expected={expected_reset_every}, actual={int(reset_every)}"
                )
        if self.detail_split_scale_m <= 0.0:
            raise ValueError("detail_split_scale_m must be positive")
        if not 0.0 < self.detail_split_screen_radius < 1.0:
            raise ValueError("detail_split_screen_radius must be within (0, 1)")
        if self.exact_mipmap_lifecycle:
            if growth_min_opacity is not None and not (
                0.0 < growth_min_opacity < 1.0
            ):
                raise ValueError("growth_min_opacity must lie within (0, 1)")
            if prune_opa_late is None or not 0.0 <= prune_opa_late < 1.0:
                raise ValueError("exact MipMap lifecycle requires prune_opa_late")
            if prune_switch_step is None or prune_switch_step <= 0:
                raise ValueError("exact MipMap lifecycle requires prune_switch_step")
            if reset_opacity_cap is None or not 0.0 < reset_opacity_cap < 1.0:
                raise ValueError("exact MipMap lifecycle requires reset_opacity_cap")
            if split_scale_m is None or prune_scale_m is None:
                raise ValueError(
                    "exact MipMap lifecycle requires metric split/prune scales"
                )
        if split_scale_m is not None:
            grow_scale3d = float(split_scale_m) / self.scene_scale
        if prune_scale_m is not None:
            prune_scale3d = float(prune_scale_m) / self.scene_scale

        self.inner = DefaultStrategy(
            prune_opa=float(prune_opa),
            grow_grad2d=float(grow_grad2d),
            grow_scale3d=float(grow_scale3d),
            prune_scale3d=float(prune_scale3d),
            grow_scale2d=float(grow_scale2d),
            prune_scale2d=float(prune_scale2d),
            refine_scale2d_stop_iter=int(refine_scale2d_stop_iter),
            refine_start_iter=int(refine_start_iter),
            refine_stop_iter=int(refine_stop_iter),
            reset_every=int(reset_every),
            refine_every=int(refine_every),
            pause_refine_after_reset=int(pause_refine_after_reset),
            absgrad=bool(absgrad),
            revised_opacity=bool(revised_opacity),
            verbose=bool(verbose),
        )

    # -- the MCMC surface the backend and trainer already read ---------------

    @property
    def min_opacity(self) -> float:
        """MCMC's name for the opacity floor; DefaultStrategy calls it prune_opa."""
        return float(self.inner.prune_opa)

    @property
    def noise_injection_stop_iter(self) -> int:
        """Classic densification injects no positional noise, so it stops at zero."""
        return 0

    @property
    def refine_start_iter(self) -> int:
        return int(self.inner.refine_start_iter)

    @property
    def refine_stop_iter(self) -> int:
        return int(self.inner.refine_stop_iter)

    @property
    def refine_every(self) -> int:
        return int(self.inner.refine_every)

    @property
    def cap_max(self) -> int | None:
        """Classic densification is threshold-driven and has no count cap."""
        return None

    # -- lifecycle -----------------------------------------------------------

    def initialize_state(self) -> dict[str, Any]:
        state = self.inner.initialize_state(scene_scale=self.scene_scale)
        if self.opacity_cull_policy != "immediate":
            # These are per-Gaussian tensors on purpose. gsplat's duplicate,
            # split and remove operations apply the same topology transform to
            # every tensor in strategy state, so the protection survives
            # growth, pruning and checkpoint resume without a parallel index.
            state["_cloudstudio_cull_low_streak"] = None
            state["_cloudstudio_cull_observations"] = None
            state["_cloudstudio_last_opacity_reset_step"] = None
        return state

    def _ensure_cull_tracking(
        self, params: Any, state: dict[str, Any]
    ) -> None:
        if self.opacity_cull_policy == "immediate":
            return
        import torch

        count = len(params["means"])
        device = params["means"].device
        for key, dtype in (
            ("_cloudstudio_cull_low_streak", torch.int16),
            ("_cloudstudio_cull_observations", torch.float32),
        ):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or len(value) != count:
                state[key] = torch.zeros(count, dtype=dtype, device=device)

    @staticmethod
    def _opacity_summary(values: Any) -> dict[str, float] | None:
        if values.numel() == 0:
            return None
        import torch

        sample = values.detach().float().flatten()
        if sample.numel() > 1_000_000:
            # Private generator: this read happens inside _grow_mipmap before
            # the split draws its offsets, so touching the global stream would
            # let a diagnostic change topology (house0305 G9 crossed this
            # threshold at 42 refine events on Tile_0).
            generator = torch.Generator(device=sample.device).manual_seed(0)
            order = torch.randperm(
                sample.numel(), device=sample.device, generator=generator
            )
            sample = sample[order[:1_000_000]]
        q = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.95], device=sample.device))
        return {
            "p50": float(q[0]),
            "p90": float(q[1]),
            "p95": float(q[2]),
            "frac_above_0p9": float((sample > 0.9).float().mean()),
            "frac_saturated": float((sample >= 1.0).float().mean()),
        }

    def _ensure_lineage(self, params: Any, state: dict[str, Any]) -> None:
        """Per-Gaussian birth step and birth kind (0 init, 1 clone, 2 split).

        Stored in strategy state so the library's duplicate/split/remove keep
        them aligned with the population and checkpoints carry them; an
        offline probe can then read child opacity, size and survival by
        birth kind and age, which a population-level histogram cannot.
        """
        import torch

        count = len(params["means"])
        device = params["means"].device
        step_tensor = state.get("_cloudstudio_birth_step")
        if not isinstance(step_tensor, torch.Tensor) or len(step_tensor) != count:
            state["_cloudstudio_birth_step"] = torch.full(
                (count,), -1, dtype=torch.int32, device=device
            )
        kind_tensor = state.get("_cloudstudio_birth_kind")
        if not isinstance(kind_tensor, torch.Tensor) or len(kind_tensor) != count:
            state["_cloudstudio_birth_kind"] = torch.zeros(
                count, dtype=torch.int8, device=device
            )

    def check_sanity(self, params: Any, optimizers: dict[str, Any]) -> None:
        self.inner.check_sanity(params, optimizers)

    def step_pre_backward(
        self,
        params: Any,
        optimizers: dict[str, Any],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
    ) -> None:
        """Retains the means2d gradient. Skipping this yields a silent no-op."""
        self.inner.step_pre_backward(
            params=params, optimizers=optimizers, state=state, step=step, info=info
        )

    def step_post_backward(
        self,
        *,
        params: Any,
        optimizers: dict[str, Any],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
        lr: float | None = None,
        **kwargs: Any,
    ) -> None:
        """``lr`` is accepted and dropped: only MCMC's noise term consumes it."""
        if self.exact_mipmap_lifecycle:
            preserved_gradients = None
            if (
                self.lifecycle_execution_order == "pre_optimizer_vendor"
                and self.is_refine_step(step)
            ):
                preserved_gradients = self._preserve_current_step_gradients(
                    params, state
                )
            self._step_post_backward_mipmap(
                params=params,
                optimizers=optimizers,
                state=state,
                step=step,
                info=info,
            )
            if preserved_gradients is not None:
                self._restore_current_step_gradients(
                    params, state, preserved_gradients
                )
            return
        self.inner.step_post_backward(
            params=params, optimizers=optimizers, state=state, step=step, info=info
        )

    def is_refine_step(self, step: int) -> bool:
        if self.exact_mipmap_lifecycle:
            return (
                step < self.refine_stop_iter
                and step >= self.refine_start_iter
                and step % self.refine_every == 0
            )
        return (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        )

    def _preserve_current_step_gradients(
        self, params: Any, state: dict[str, Any]
    ) -> dict[str, Any | None]:
        """Keep the backward result across pre-Adam topology replacement.

        gsplat's topology ops replace every ``Parameter`` and correctly reshape
        Adam's moment tensors, but the replacement parameters deliberately have
        no ``.grad``.  That is correct for gsplat's normal post-optimizer order;
        in the recovered vendor order it would silently drop the current step.
        A per-row source index is carried through the same duplicate/split/remove
        transforms, then used to restore each parameter's gradient.
        """

        import torch

        key = "_cloudstudio_current_step_gradient_source"
        if key in state:
            raise RuntimeError("stale pre-optimizer gradient provenance")
        count = len(params["means"])
        state[key] = torch.arange(
            count, dtype=torch.int64, device=params["means"].device
        )
        return {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in params.items()
        }

    def _restore_current_step_gradients(
        self,
        params: Any,
        state: dict[str, Any],
        gradients: dict[str, Any | None],
    ) -> None:
        key = "_cloudstudio_current_step_gradient_source"
        provenance = state.pop(key, None)
        if provenance is None:
            raise RuntimeError("pre-optimizer topology lost gradient provenance")
        if len(provenance) != len(params["means"]):
            raise RuntimeError("pre-optimizer gradient provenance count mismatch")
        if provenance.numel() == 0:
            raise RuntimeError("pre-optimizer lifecycle removed every Gaussian")
        for name, parameter in params.items():
            source = gradients.get(name)
            if source is None:
                parameter.grad = None
                continue
            if source.ndim == 0 or source.shape[0] <= int(provenance.max().item()):
                raise RuntimeError(
                    f"pre-optimizer gradient provenance is invalid for {name}"
                )
            parameter.grad = source[provenance].clone()
        if self.last_lifecycle_event is not None:
            self.last_lifecycle_event["execution_order"] = (
                self.lifecycle_execution_order
            )
            self.last_lifecycle_event["current_step_gradient_remapped"] = True
            self.last_lifecycle_event["gradient_row_count"] = int(len(provenance))

    def _grow_mipmap(
        self,
        params: Any,
        optimizers: dict[str, Any],
        state: dict[str, Any],
        *,
        scene: Any | None = None,
    ) -> tuple[int, int]:
        """Classic split/clone with the recovered opacity eligibility gate."""

        import torch
        from gsplat.strategy.ops import duplicate, split

        def duplicate_selected(mask: Any) -> None:
            """Duplicate rows without changing their initial alpha budget."""

            selected = torch.where(mask)[0]
            revised_logits = None
            if self.capacity_conserving_clone_opacity and selected.numel():
                parent_opacity = torch.sigmoid(
                    params["opacities"][selected].detach()
                )
                # Two coincident rows with opacity q composite to 1-(1-q)^2.
                # Solving for the original alpha p gives q=1-sqrt(1-p), so a
                # clone cannot instantly thicken a measured surface before
                # the optimizer has separated the pair.
                revised_opacity = 1.0 - torch.sqrt(
                    (1.0 - parent_opacity).clamp_min(0.0)
                )
                revised_logits = torch.logit(
                    revised_opacity.clamp(1e-6, 1.0 - 1e-6)
                )
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=mask,
                scene=scene,
            )
            if revised_logits is not None:
                with torch.no_grad():
                    reshaped = revised_logits.reshape_as(
                        params["opacities"][selected]
                    )
                    params["opacities"].index_copy_(0, selected, reshaped)
                    params["opacities"][-selected.numel() :].copy_(reshaped)

        def split_selected(mask: Any) -> None:
            """Split rows; the revised child opacity is computed here so a
            parent whose sigmoid has saturated to exactly 1.0 (logit beyond
            float32 resolution) cannot turn into an infinite child logit.
            The library formula 1 - sqrt(1 - p) on such a parent gives 1.0 and
            logit(1.0) is +inf, which the next optimizer step spreads as NaN."""

            selected = torch.where(mask)[0]
            revised_logits = None
            if self.inner.revised_opacity and selected.numel():
                parent_opacity = torch.sigmoid(
                    params["opacities"][selected].detach()
                )
                revised_opacity = 1.0 - torch.sqrt(
                    (1.0 - parent_opacity).clamp_min(0.0)
                )
                revised_logits = torch.logit(
                    revised_opacity.clamp(1e-6, 1.0 - 1e-6)
                )
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=mask,
                revised_opacity=False,
                scene=scene,
            )
            if revised_logits is not None:
                with torch.no_grad():
                    count = selected.numel()
                    tail = params["opacities"][-2 * count :]
                    # The library appends the two children of parent i at
                    # tail rows i and count + i.
                    tail.copy_(revised_logits.reshape(count, *tail.shape[1:]).repeat(2, *([1] * (tail.dim() - 1))))

        if self.growth_metric == "footprint_weighted":
            weight_sum = state.get("_footprint_weight_sum")
            if weight_sum is None or len(weight_sum) != len(params["means"]):
                raise RuntimeError(
                    "footprint_weighted growth needs its accumulator; it is "
                    "filled every step by _accumulate_footprint_weighted_gradient"
                )
            gradients = state["_footprint_grad_sum"] / weight_sum.clamp_min(1e-8)
            observed = weight_sum > 0
        else:
            gradients = state["grad2d"] / state["count"].clamp_min(1)
            observed = state["count"] > 0
        opacity = torch.sigmoid(params["opacities"].flatten())
        finite_observed_gradients = gradients[
            observed & torch.isfinite(gradients)
        ].float()
        gradient_quantiles = {}
        if finite_observed_gradients.numel():
            quantile_levels = torch.tensor(
                [0.5, 0.9, 0.95, 0.99, 0.999],
                dtype=finite_observed_gradients.dtype,
                device=finite_observed_gradients.device,
            )
            quantile_values = _sampled_quantile(
                finite_observed_gradients, quantile_levels
            ).tolist()
            gradient_quantiles = {
                label: float(value)
                for label, value in zip(
                    ("p50", "p90", "p95", "p99", "p999"),
                    quantile_values,
                )
            }
            gradient_quantiles["max"] = float(
                finite_observed_gradients.max().item()
            )
        gradient_thresholds = (
            0.00005,
            0.000075,
            0.0001,
            0.000125,
            0.00015,
            0.0002,
            0.0003,
        )
        opacity_floors = (0.05, 0.1, 0.15)
        threshold_sweep = {
            f"grad_gt_{gradient_threshold:.7f}_opacity_gt_{opacity_floor:.2f}": int(
                (
                    (gradients > gradient_threshold)
                    & (opacity > opacity_floor)
                ).sum().item()
            )
            for gradient_threshold in gradient_thresholds
            for opacity_floor in opacity_floors
        }
        eligible = gradients > float(self.inner.grow_grad2d)
        if self.growth_min_opacity is not None:
            # Optional, and off in the recovered contract: the reference
            # computes this comparison and then never reads it.
            eligible &= opacity > float(self.growth_min_opacity)
        growth_candidate_count = int(eligible.sum().item())
        guarded_births = bool(
            self.surface_birth_proposal is not None
            and self.surface_birth_proposal.config.reject_unsupported_births
        )
        if (
            guarded_births
            and self.lifecycle_execution_order == "pre_optimizer_vendor"
        ):
            raise RuntimeError(
                "pre_optimizer_vendor forbids the CloudStudio surface birth guard"
            )
        if guarded_births:
            eligible &= self.surface_birth_proposal.eligible_parent_mask(
                params["means"]
            )
        supported_candidate_count = int(eligible.sum().item())
        capacity_rejected_count = 0
        if self.capacity_cap is not None:
            available = max(0, self.capacity_cap - len(params["means"]))
            if supported_candidate_count > available:
                candidate_indices = torch.where(eligible)[0]
                limited = torch.zeros_like(eligible)
                if available > 0:
                    keep_local = torch.topk(
                        gradients[candidate_indices],
                        k=available,
                        largest=True,
                        sorted=False,
                    ).indices
                    limited[candidate_indices[keep_local]] = True
                capacity_rejected_count = supported_candidate_count - available
                eligible = limited
        maximum_scale = torch.exp(params["scales"]).max(dim=-1).values
        small = maximum_scale <= float(self.split_scale_m)
        detail_split_mask = torch.zeros_like(eligible)
        radii = state.get("radii")
        if (
            self.detail_split_policy == "lidar_surface_screen_detail"
            and radii is not None
        ):
            # Preserve the recovered 0.2 m world-space split rule. The
            # CloudStudio detail extension only redirects an otherwise cloned
            # high-gradient parent into split when it is both physically large
            # enough and visibly broad in the actual training renders. This
            # avoids globally shrinking the larger thin disks that efficiently
            # cover low-texture walls and snow planes.
            detail_split_mask = eligible & small
            detail_split_mask &= maximum_scale > self.detail_split_scale_m
            detail_split_mask &= radii > self.detail_split_screen_radius
        duplicate_mask = eligible & small & ~detail_split_mask
        split_mask = eligible & (~small | detail_split_mask)
        duplicate_count = int(duplicate_mask.sum().item())
        split_count = int(split_mask.sum().item())
        opacity_quantiles = {
            label: float(value)
            for label, value in zip(
                ("p05", "p10", "p50", "p90", "p95"),
                _sampled_quantile(
                    opacity.float(),
                    torch.tensor(
                        [0.05, 0.1, 0.5, 0.9, 0.95], device=opacity.device
                    ),
                ).tolist(),
            )
        }
        # Convention audit: the threshold's unit depends on whether the
        # accumulator is a per-observation mean (ours, classic) or the raw
        # per-window sum. Publishing both quantile families plus the
        # observation counts lets a probe settle that question from one run.
        window_sums = state["grad2d"][observed & torch.isfinite(gradients)].float()
        observation_counts = state["count"][observed].float()
        footprint_audit = {}
        footprint_sum = state.get("_footprint_grad_sum")
        if footprint_sum is not None and len(footprint_sum) == len(gradients):
            footprint_weight = state["_footprint_weight_sum"]
            equivalent = footprint_sum / footprint_weight.clamp_min(1e-8)
            seen = footprint_weight > 0
            equivalent_seen = equivalent[seen & torch.isfinite(equivalent)]
            if equivalent_seen.numel():
                levels = torch.tensor(
                    [0.5, 0.9, 0.95, 0.99, 0.999], device=equivalent_seen.device
                )
                q = _sampled_quantile(equivalent_seen, levels).tolist()
                footprint_audit = {
                    "equivalent_grad_p50": float(q[0]),
                    "equivalent_grad_p90": float(q[1]),
                    "equivalent_grad_p95": float(q[2]),
                    "equivalent_grad_p99": float(q[3]),
                    "equivalent_grad_p999": float(q[4]),
                    "equivalent_selected_at_00015": int(
                        (equivalent_seen > 1.5e-4).sum()
                    ),
                    "equivalent_selected_at_0002": int(
                        (equivalent_seen > 2.0e-4).sum()
                    ),
                    "equivalent_observed_count": int(equivalent_seen.numel()),
                }
        convention_audit = {}
        if window_sums.numel():
            convention_audit = {
                "window_sum_p90": float(_sampled_quantile(window_sums, 0.9)),
                "window_sum_p99": float(_sampled_quantile(window_sums, 0.99)),
                "observations_p50": float(_sampled_quantile(observation_counts, 0.5)),
                "observations_p90": float(_sampled_quantile(observation_counts, 0.9)),
            }
        self._last_growth_event = {
            "gradient_threshold": float(self.inner.grow_grad2d),
            "opacity_floor": (
                None
                if self.growth_min_opacity is None
                else float(self.growth_min_opacity)
            ),
            "observed_gaussian_count": int(observed.sum().item()),
            "gradient_quantiles": gradient_quantiles,
            "threshold_sweep_counts": threshold_sweep,
            "opacity_quantiles": opacity_quantiles,
            "gradient_convention_audit": convention_audit,
            "footprint_weighted_audit": footprint_audit,
            "fraction_opacity_below_0p10": float((opacity < 0.10).float().mean()),
            "fraction_opacity_above_0p15": float((opacity > 0.15).float().mean()),
            # Four graded buckets instead of one catch-all: the distribution is
            # bimodal, and lumping everything under 0.1 into "dead" hides
            # whether the mass sits at 0.003 (never coming back) or 0.08
            # (one good view from recovering). The active count is the number
            # that actually renders; raw population is not a health metric.
            "fraction_opacity_below_0p005": float((opacity < 0.005).float().mean()),
            "fraction_opacity_below_0p01": float((opacity < 0.01).float().mean()),
            "fraction_opacity_below_0p05": float((opacity < 0.05).float().mean()),
            "active_count_ge_0p10": int((opacity >= 0.10).sum().item()),
            "gradient_only_candidate_count": int(
                (gradients > float(self.inner.grow_grad2d)).sum().item()
            ),
            "selected_parent_count": int(eligible.sum().item()),
            # The budget is consumed on this branch (topk over candidates when
            # they exceed the headroom), so record what it did instead of
            # leaving "was the cap live" to be argued from the population curve.
            "capacity_cap": self.capacity_cap,
            "capacity_available_before_growth": (
                None
                if self.capacity_cap is None
                else max(0, self.capacity_cap - int(observed.numel()))
            ),
            "capacity_rejected_count": capacity_rejected_count,
            "clone_parent_count": duplicate_count,
            "split_parent_count": split_count,
            "split_parent_opacity": self._opacity_summary(opacity[split_mask]),
            "clone_parent_opacity": self._opacity_summary(opacity[duplicate_mask]),
            "capacity_conserving_clone_opacity": (
                self.capacity_conserving_clone_opacity
            ),
            "dry_run": self.lifecycle_dry_run,
        }
        if self.lifecycle_dry_run:
            return 0, 0
        parent_means = None
        if guarded_births and (duplicate_count or split_count):
            clone_indices = torch.where(duplicate_mask)[0]
            split_indices = torch.where(split_mask)[0]
            parent_means = torch.cat(
                [
                    params["means"][clone_indices].detach().clone(),
                    params["means"][split_indices].detach().clone().repeat(2, 1),
                ],
                dim=0,
            )
        if self.lifecycle_execution_order == "pre_optimizer_vendor":
            # The recovered native order is Split -> Clone -> Cull.  The masks
            # are disjoint and were computed on the original rows. After Split,
            # non-split rows lead the tensor and split children occupy the tail,
            # so remap the clone mask onto that layout before duplicating.
            if split_count:
                original_split_mask = split_mask
                split_selected(original_split_mask)
                duplicate_mask = torch.cat(
                    [
                        duplicate_mask[~original_split_mask],
                        torch.zeros(
                            2 * split_count,
                            dtype=torch.bool,
                            device=duplicate_mask.device,
                        ),
                    ]
                )
            if duplicate_count:
                duplicate_selected(duplicate_mask)
        else:
            if duplicate_count:
                duplicate_selected(duplicate_mask)
            if split_count:
                split_mask = torch.cat(
                    [
                        split_mask,
                        torch.zeros(
                            duplicate_count,
                            dtype=torch.bool,
                            device=split_mask.device,
                        ),
                    ]
                )
                split_selected(split_mask)
        if parent_means is not None:
            # ``split`` removes its parents, keeps all non-split rows, then
            # appends two children per parent.  Duplicates therefore precede
            # split children in one contiguous newborn tail beginning at
            # ``old_count - split_count``.
            newborn_count = duplicate_count + 2 * split_count
            newborn_start = len(params["means"]) - newborn_count
            proposed = self.surface_birth_proposal.propose(
                parent_means,
                params["quats"][newborn_start:],
                params["scales"][newborn_start:],
            )
            if not bool(torch.all(proposed["applied"]).item()):
                raise RuntimeError(
                    "LiDAR surface birth guard rejected a parent that passed "
                    "the pre-growth gate"
                )
            with torch.no_grad():
                params["means"][newborn_start:].copy_(proposed["means"])
                if "quats" in proposed:
                    params["quats"][newborn_start:].copy_(proposed["quats"])
                if "scales" in proposed:
                    params["scales"][newborn_start:].copy_(proposed["scales"])
        newborn_total = duplicate_count + 2 * split_count
        if newborn_total:
            self._ensure_lineage(params, state)
            with torch.no_grad():
                current_step = int(state.get("_cloudstudio_current_step", -1))
                tail_step = state["_cloudstudio_birth_step"][-newborn_total:]
                tail_kind = state["_cloudstudio_birth_kind"][-newborn_total:]
                tail_step.fill_(current_step)
                if self.lifecycle_execution_order == "pre_optimizer_vendor":
                    # Split children were appended first, clones after.
                    tail_kind[: 2 * split_count].fill_(2)
                    tail_kind[2 * split_count :].fill_(1)
                else:
                    tail_kind[:duplicate_count].fill_(1)
                    tail_kind[duplicate_count:].fill_(2)
        self._last_surface_birth_event = (
            None
            if not guarded_births
            else {
                "growth_candidates": growth_candidate_count,
                "supported_parents": supported_candidate_count,
                "rejected_parents": growth_candidate_count
                - supported_candidate_count,
                "capacity_rejected_parents": capacity_rejected_count,
                "newborns": duplicate_count + 2 * split_count,
                "proposal": dict(self.surface_birth_proposal.last_stats),
            }
        )
        return duplicate_count, split_count

    def _prune_mipmap(
        self,
        params: Any,
        optimizers: dict[str, Any],
        state: dict[str, Any],
        *,
        step: int,
        scene: Any | None = None,
        densify_allowed: bool | None = None,
    ) -> int:
        """Apply recovered early/late opacity, world, and screen cull gates.

        The recovered contract carries an anti-starvation branch keyed on the
        DENSIFICATION GATE, not on how many parents happened to qualify: when
        this cycle was not permitted to densify at all (outside the refine
        window, or the population sits at the absolute cap), culling relaxes
        (opacity threshold x0.25, world and screen limits x5) instead of
        running at full strength. A permitted cycle that merely selected zero
        parents still culls at full strength - that is the recovered
        semantics, and the distinction is load-bearing.
        """

        import torch
        from gsplat.strategy.ops import remove

        opacity_threshold = (
            float(self.prune_opa_late)
            if step >= int(self.prune_switch_step)
            else float(self.inner.prune_opa)
        )
        world_limit = float(self.prune_scale_m)
        screen_limit = float(self.inner.prune_scale2d)
        relaxed_cull_only = (
            self.relaxed_cull_when_no_growth
            and densify_allowed is not None
            and not densify_allowed
        )
        if relaxed_cull_only:
            opacity_threshold *= 0.25
            world_limit *= 5.0
            screen_limit *= 5.0
        opacity = torch.sigmoid(params["opacities"].flatten())
        raw_opacity_mask = opacity < opacity_threshold
        opacity_threshold_sweep_counts = {
            f"opacity_lt_{threshold:.3f}": int(
                (opacity < threshold).sum().item()
            )
            for threshold in (0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1)
        }
        maximum_scale = torch.exp(params["scales"]).max(dim=-1).values
        world_scale_mask = maximum_scale > world_limit
        radii = state.get("radii")
        screen_scale_mask = torch.zeros_like(raw_opacity_mask)
        if radii is not None:
            screen_scale_mask = radii > screen_limit

        opacity_mask = raw_opacity_mask.clone()
        grace_active = False
        local_competition_protected_count = 0
        local_competition_cell_count = 0
        contribution_spared_count = 0
        if self.opacity_cull_policy == "contribution_aware":
            # Cull on evidence of invisibility, not on a momentary opacity read.
            # The clean-vendor probe showed immediate cull bleeds ~18% of the
            # ACTIVE population per reset cycle: the reset clamps everyone to
            # 0.2, and live front surfaces that are not re-observed inside the
            # window get taken before they climb back. The protections that were
            # meant to stop that instead hoarded corpses (dead mass rising
            # monotonically to 75%). Accumulated visible alpha separates the two
            # cases directly - a suppressed but contributing Gaussian has put
            # alpha on screen this window, a corpse has not - so low opacity is
            # necessary but no longer sufficient to die.
            visible_alpha = state.get("_visible_alpha_sum")
            if visible_alpha is None or len(visible_alpha) != len(opacity):
                raise RuntimeError(
                    "contribution_aware cull needs the visible-alpha "
                    "accumulator, filled every step alongside the footprint "
                    "gradient; enable growth_metric='footprint_weighted'"
                )
            # Scale-free threshold: a fraction of this window's median among
            # gaussians that were seen at all, so it tracks exposure and view
            # count instead of hard-coding a screen-space constant.
            seen = visible_alpha > 0.0
            if bool(seen.any()):
                reference = torch.quantile(
                    visible_alpha[seen].float(), 0.5
                ).clamp_min(1e-12)
                contributes = visible_alpha >= (
                    reference * float(self.contribution_cull_relative_floor)
                )
            else:
                contributes = torch.zeros_like(raw_opacity_mask)
            opacity_mask = raw_opacity_mask & ~contributes
            contribution_spared_count = int(
                (raw_opacity_mask & contributes).sum().item()
            )
        elif self.opacity_cull_policy != "immediate":
            self._ensure_cull_tracking(params, state)
            streak = state["_cloudstudio_cull_low_streak"]
            observations = state["_cloudstudio_cull_observations"]
            observations.add_(state["count"].to(observations.dtype))
            streak.copy_(
                torch.where(
                    raw_opacity_mask,
                    torch.clamp(streak + 1, max=32767),
                    torch.zeros_like(streak),
                )
            )
            last_reset = state.get("_cloudstudio_last_opacity_reset_step")
            grace_active = bool(
                last_reset is not None
                and step - int(last_reset)
                <= self.opacity_cull_grace_after_reset_steps
            )
            opacity_mask = raw_opacity_mask.clone()
            opacity_mask &= (
                observations >= float(self.opacity_cull_min_observations)
            )
            opacity_mask &= streak >= self.opacity_cull_consecutive_events
            if grace_active:
                opacity_mask &= False

            if (
                self.opacity_cull_policy == "local_coverage_competition"
                and bool(opacity_mask.any())
            ):
                # Low-texture regions need a real death path, but deleting the
                # only local representative is exactly how the V33 route made
                # walls transparent.  Keep the strongest-opacity Gaussian in
                # every small world-space cell and let the remaining low-
                # opacity rows compete for removal.  This is deliberately a
                # topology guard, not a texture heuristic: projected gradient
                # still decides births, while learned contribution decides
                # which redundant local rows leave.
                grid = torch.floor(
                    params["means"].detach() / self.opacity_cull_local_voxel_m
                ).to(torch.int64)
                offset = 1 << 20
                shifted = grid + offset
                if bool(((shifted < 0) | (shifted >= (1 << 21))).any()):
                    raise RuntimeError(
                        "local cull voxel index exceeds signed 21-bit encoding"
                    )
                keys = (
                    (shifted[:, 0] << 42)
                    | (shifted[:, 1] << 21)
                    | shifted[:, 2]
                )
                _, inverse = torch.unique(keys, return_inverse=True)
                local_competition_cell_count = int(inverse.max().item()) + 1
                protection_score = opacity
                if self.opacity_cull_local_protection == "opacity_tangent_area":
                    # A low-texture wall is efficiently represented by a few
                    # broad, thin disks. Protecting only the highest opacity
                    # row can discard that coverage carrier in favour of a
                    # small bright splat. The two largest physical axes are a
                    # view-independent proxy for tangential surface area.
                    physical_scale = torch.exp(params["scales"].detach())
                    tangent_area = torch.topk(
                        physical_scale, k=2, dim=-1
                    ).values.prod(dim=-1)
                    protection_score = opacity * tangent_area
                if self.opacity_cull_local_min_accumulated_alpha > 0.0:
                    # Sort by cell, then by coverage score within each cell.
                    # Protect the leading rows until their composited alpha
                    # reaches the signed local budget. This turns the guard
                    # from "keep one point" into "keep a surface", while low-
                    # contribution surplus rows remain eligible for removal.
                    score_order = torch.argsort(
                        protection_score, descending=True, stable=True
                    )
                    order = score_order[
                        torch.argsort(inverse[score_order], stable=True)
                    ]
                    ordered_cell = inverse[order]
                    optical_depth = -torch.log1p(
                        -opacity[order].clamp(max=1.0 - 1e-6)
                    )
                    cumulative = torch.cumsum(optical_depth, dim=0)
                    first = torch.ones_like(ordered_cell, dtype=torch.bool)
                    first[1:] = ordered_cell[1:] != ordered_cell[:-1]
                    start = torch.where(first)[0]
                    prefix = torch.zeros_like(start, dtype=cumulative.dtype)
                    nonzero = start > 0
                    prefix[nonzero] = cumulative[start[nonzero] - 1]
                    cell_prefix = torch.empty(
                        local_competition_cell_count,
                        dtype=cumulative.dtype,
                        device=cumulative.device,
                    )
                    cell_prefix[ordered_cell[start]] = prefix
                    before = cumulative - optical_depth - cell_prefix[ordered_cell]
                    target_depth = -math.log1p(
                        -self.opacity_cull_local_min_accumulated_alpha
                    )
                    protected = torch.zeros_like(opacity_mask)
                    protected[order] = before < target_depth
                else:
                    strongest = torch.full(
                        (local_competition_cell_count,),
                        -torch.inf,
                        dtype=protection_score.dtype,
                        device=protection_score.device,
                    )
                    strongest.scatter_reduce_(
                        0,
                        inverse,
                        protection_score,
                        reduce="amax",
                        include_self=True,
                    )
                    protected = protection_score >= strongest[inverse]
                local_competition_protected_count = int(
                    (opacity_mask & protected).sum().item()
                )
                opacity_mask &= ~protected

            maximum_opacity_culls = int(
                len(opacity_mask) * self.opacity_cull_max_fraction
            )
            selected_count = int(opacity_mask.sum().item())
            if selected_count > maximum_opacity_culls:
                candidates = torch.where(opacity_mask)[0]
                priority = opacity[candidates]
                if self.opacity_cull_priority == "lowest_opacity_per_footprint":
                    footprint = maximum_scale
                    if radii is not None and len(radii) == len(maximum_scale):
                        # ``radii`` is the maximum observed raster radius,
                        # normalized by max(image width, image height). It is a
                        # closer measure of visible blur than world scale.
                        footprint = radii
                    priority = priority / footprint[candidates].clamp_min(1e-6)
                keep = torch.topk(
                    priority,
                    k=maximum_opacity_culls,
                    largest=False,
                    sorted=False,
                ).indices
                limited = torch.zeros_like(opacity_mask)
                limited[candidates[keep]] = True
                opacity_mask = limited

        forced_geometry_mask = world_scale_mask | screen_scale_mask
        remove_mask = opacity_mask | forced_geometry_mask
        prune_count = int(remove_mask.sum().item())
        observation_count = state.get("count")
        raw_candidate_observed_count = None
        raw_candidate_zero_observation_count = None
        raw_candidate_observation_p50 = None
        raw_candidate_observation_p95 = None
        if (
            observation_count is not None
            and observation_count.numel() == raw_opacity_mask.numel()
        ):
            observation_count = observation_count.reshape(-1).to(torch.float32)
            candidate_observations = observation_count[raw_opacity_mask]
            if candidate_observations.numel():
                raw_candidate_observed_count = int(
                    (candidate_observations > 0).sum().item()
                )
                raw_candidate_zero_observation_count = int(
                    (candidate_observations <= 0).sum().item()
                )
                raw_candidate_observation_p50 = float(
                    _sampled_quantile(candidate_observations, 0.50).item()
                )
                raw_candidate_observation_p95 = float(
                    _sampled_quantile(candidate_observations, 0.95).item()
                )
        self._last_cull_event = {
            "policy": self.opacity_cull_policy,
            "opacity_threshold": float(opacity_threshold),
            "raw_opacity_candidate_count": int(raw_opacity_mask.sum().item()),
            "opacity_threshold_sweep_counts": opacity_threshold_sweep_counts,
            "raw_candidate_observed_count": raw_candidate_observed_count,
            "raw_candidate_zero_observation_count": (
                raw_candidate_zero_observation_count
            ),
            "raw_candidate_observation_p50": raw_candidate_observation_p50,
            "raw_candidate_observation_p95": raw_candidate_observation_p95,
            "selected_opacity_count": int(opacity_mask.sum().item()),
            "world_scale_count": int(world_scale_mask.sum().item()),
            "screen_scale_count": int(screen_scale_mask.sum().item()),
            "forced_geometry_union_count": int(forced_geometry_mask.sum().item()),
            "opacity_geometry_overlap_count": int(
                (opacity_mask & forced_geometry_mask).sum().item()
            ),
            "grace_active": grace_active,
            "min_observations": self.opacity_cull_min_observations,
            "consecutive_events": self.opacity_cull_consecutive_events,
            "grace_after_reset_steps": self.opacity_cull_grace_after_reset_steps,
            "max_opacity_cull_fraction": self.opacity_cull_max_fraction,
            "opacity_cull_priority": self.opacity_cull_priority,
            "opacity_cull_local_voxel_m": self.opacity_cull_local_voxel_m,
            "opacity_cull_local_protection": (
                self.opacity_cull_local_protection
            ),
            "opacity_cull_local_min_accumulated_alpha": (
                self.opacity_cull_local_min_accumulated_alpha
            ),
            "local_competition_cell_count": local_competition_cell_count,
            "local_competition_protected_count": (
                local_competition_protected_count
            ),
            "total_cull_count": prune_count,
            "relaxed_cull_only": relaxed_cull_only,
        }
        if self.lifecycle_dry_run:
            self._last_cull_event["dry_run"] = True
            return 0
        if prune_count:
            remove(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=remove_mask,
                scene=scene,
            )
        return prune_count

    def _accumulate_footprint_weighted_gradient(
        self, params: Any, state: dict[str, Any], info: dict[str, Any]
    ) -> None:
        """Second, read-only accumulator for the recovered densification score.

        The published DefaultStrategy scales the projected gradient per-axis by
        width/2 and height/2, then takes a plain per-observation mean. The
        recovered contract instead norms first, scales isotropically by
        0.5*max(1600, W, H), and takes a FOOTPRINT-weighted mean (each view's
        weight is the gaussian's projected radius there). This computes that
        score alongside ours - it selects nothing and prunes nothing; a probe
        reads its quantiles to decide whether the recovered 0.00015 threshold
        even lives in the same unit as our 0.00015.
        """
        import torch

        key = self.inner.key_for_gradient
        packed = info.get("gaussian_ids") is not None and info[key].grad.dim() == 2
        raw = info[key].grad.detach()
        n_gaussian = len(next(iter(params.values())))
        if state.get("_footprint_grad_sum") is None:
            state["_footprint_grad_sum"] = torch.zeros(
                n_gaussian, device=raw.device
            )
            state["_footprint_weight_sum"] = torch.zeros(
                n_gaussian, device=raw.device
            )
        # A warm start / topology change resizes the population; grow the
        # diagnostic buffers to match rather than crashing on the stale length.
        if len(state["_footprint_grad_sum"]) != n_gaussian:
            state["_footprint_grad_sum"] = torch.zeros(
                n_gaussian, device=raw.device
            )
            state["_footprint_weight_sum"] = torch.zeros(
                n_gaussian, device=raw.device
            )
        if (
            state.get("_visible_alpha_sum") is None
            or len(state["_visible_alpha_sum"]) != n_gaussian
        ):
            state["_visible_alpha_sum"] = torch.zeros(
                n_gaussian, device=raw.device
            )

        width = float(info["width"])
        height = float(info["height"])
        screen_scale = 0.5 * max(1600.0, width, height)
        # The recovered footprint is the L2 norm of the radius components, and
        # visibility is that norm being positive - not the per-axis maximum and
        # not every component being positive. For an isotropic splat the two
        # differ only by a constant that the weighted mean divides out, but for
        # anisotropic ones they reorder which views dominate a gaussian's score:
        # radii (10,1) and (7,7) rank 10 vs 7 under the maximum and 10.05 vs
        # 9.90 under the norm. That changes which gaussians clear the threshold.
        radii_all = torch.linalg.vector_norm(info["radii"].float(), dim=-1)
        if packed:
            gs_ids = info["gaussian_ids"]
            radii = radii_all
            grad_norm = raw.norm(dim=-1)
        else:
            visible = radii_all > 0.0
            gs_ids = torch.where(visible)[1]
            radii = radii_all[visible]
            grad_norm = raw[visible].norm(dim=-1)
        contribution = radii * screen_scale * grad_norm
        state["_footprint_grad_sum"].index_add_(0, gs_ids, contribution)
        state["_footprint_weight_sum"].index_add_(0, gs_ids, radii)

        # Visible alpha, accumulated over the same window. Opacity alone cannot
        # tell a corpse from a living Gaussian the periodic reset just clamped:
        # both read low right after a reset. This asks a different question -
        # has this Gaussian actually been putting alpha on screen? - so the cull
        # can spare a suppressed-but-contributing surface while still taking the
        # rows that are invisible no matter how often they are seen.
        opacity_now = torch.sigmoid(params["opacities"].detach().flatten())
        state["_visible_alpha_sum"].index_add_(
            0, gs_ids, opacity_now[gs_ids] * radii
        )

    def _reset_opacities(self, params, optimizers, state) -> None:
        """Clamp opacities to the reset cap with the configured optimizer-state policy."""
        import torch
        from gsplat.strategy.ops import _update_param_with_optimizer, reset_opa

        if self.reset_optimizer_state == "keep":
            cap_logit = torch.logit(torch.tensor(float(self.reset_opacity_cap))).item()

            def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
                if name != "opacities":
                    raise ValueError(f"Unexpected parameter name: {name}")
                return torch.nn.Parameter(
                    torch.clamp(p, max=cap_logit), requires_grad=p.requires_grad
                )

            _update_param_with_optimizer(
                param_fn, lambda key, v: v, params, optimizers, names=["opacities"]
            )
        else:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=float(self.reset_opacity_cap),
            )
        if self.reset_adam_step:
            _restart_adam_step(optimizers.get("opacities"), params["opacities"])

    def _step_post_backward_mipmap(
        self,
        *,
        params: Any,
        optimizers: dict[str, Any],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
    ) -> None:
        import torch

        self.last_lifecycle_event = None
        state["_cloudstudio_current_step"] = int(step)
        self._ensure_lineage(params, state)
        self._last_surface_birth_event = None
        self._last_growth_event = None
        self._last_cull_event = None
        if step >= self.refine_stop_iter:
            return
        self.inner._update_state(params, state, info, packed=False)
        self._accumulate_footprint_weighted_gradient(params, state, info)
        if not self.is_refine_step(step):
            return
        self._ensure_cull_tracking(params, state)
        before = len(params["means"])
        # The recovered control flow decides whether this cycle may densify
        # from the count it ENTERS with, then carries that decision into the
        # cull. Re-reading the count after growth relaxes culling a cycle
        # early: a population that arrives under the cap and grows into it
        # spent a legitimate densification cycle, and should still cull at
        # full strength. Getting this wrong leaves a cycle's worth of
        # low-opacity mass behind exactly at the ceiling.
        at_capacity = (
            self.capacity_cap is not None and before >= self.capacity_cap
        )
        densify_allowed = step < self.refine_stop_iter and not (
            at_capacity and self.relaxed_cull_at_capacity
        )
        clone_count, split_count = self._grow_mipmap(
            params, optimizers, state
        )
        reset = step % int(self.inner.reset_every) == 0
        if reset and self.lifecycle_dry_run:
            reset = False
        if reset and self.reset_before_cull:
            self._reset_opacities(params, optimizers, state)
        cull_count = self._prune_mipmap(
            params,
            optimizers,
            state,
            step=step,
            densify_allowed=densify_allowed,
        )
        if reset and not self.reset_before_cull:
            self._reset_opacities(params, optimizers, state)
        if reset:
            if self.opacity_cull_policy != "immediate":
                state["_cloudstudio_cull_low_streak"].zero_()
                state["_cloudstudio_cull_observations"].zero_()
                state["_cloudstudio_last_opacity_reset_step"] = int(step)
        state["grad2d"].zero_()
        state["count"].zero_()
        if state.get("radii") is not None:
            state["radii"].zero_()
        if state.get("_visible_alpha_sum") is not None:
            # Decay, do NOT zero. The measured failure of the zeroing version:
            # population was stable inside each cycle and lost ~20% in the one
            # flush right after each opacity reset. At that moment every row has
            # an empty fresh window, so a contribution rule that only sees the
            # current window is blind exactly when it has to protect a live
            # surface the reset just clamped. Carrying a decayed history means a
            # Gaussian that was contributing before the reset still reads as a
            # contributor during the flush, while a genuine corpse - never
            # contributing in any window - decays to nothing and still dies.
            state["_visible_alpha_sum"].mul_(
                float(self.contribution_history_decay)
            )
        if state.get("_footprint_grad_sum") is not None:
            state["_footprint_grad_sum"].zero_()
            state["_footprint_weight_sum"].zero_()
        self.last_lifecycle_event = {
            "before_count": int(before),
            "clone_count": clone_count,
            "split_parent_count": split_count,
            "split_child_count": 2 * split_count,
            "cull_count": cull_count,
            "opacity_reset": reset,
            "after_count": int(len(params["means"])),
            "cull_opacity_threshold": (
                float(self.prune_opa_late)
                if step >= int(self.prune_switch_step)
                else float(self.inner.prune_opa)
            ),
        }
        if self._last_surface_birth_event is not None:
            self.last_lifecycle_event["surface_birth_guard"] = dict(
                self._last_surface_birth_event
            )
        if self._last_growth_event is not None:
            self.last_lifecycle_event["growth_diagnostics"] = dict(
                self._last_growth_event
            )
        if self._last_cull_event is not None:
            self.last_lifecycle_event["cull_reasons"] = dict(
                self._last_cull_event
            )
        if params["means"].is_cuda:
            torch.cuda.empty_cache()

    def state_dict(self) -> dict[str, Any]:
        """Every knob, plus the metres each normalised threshold resolves to.

        Recorded in full because the normalised form is unreadable on its own -
        "grow_scale3d 0.01" says nothing about whether a 8 cm Gaussian will be
        split, and the run where it silently meant 0.58 mm looked identical in
        the config.
        """
        return {
            "strategy": "default_3dgs",
            "scene_scale": self.scene_scale,
            "grow_grad2d": float(self.inner.grow_grad2d),
            "grow_scale3d": float(self.inner.grow_scale3d),
            "prune_scale3d": float(self.inner.prune_scale3d),
            "grow_scale2d": float(self.inner.grow_scale2d),
            "prune_scale2d": float(self.inner.prune_scale2d),
            "refine_scale2d_stop_iter": int(self.inner.refine_scale2d_stop_iter),
            "prune_opa": float(self.inner.prune_opa),
            "absgrad": bool(self.inner.absgrad),
            "revised_opacity": bool(self.inner.revised_opacity),
            "reset_every": int(self.inner.reset_every),
            "pause_refine_after_reset": int(self.inner.pause_refine_after_reset),
            "refine_start_iter": int(self.inner.refine_start_iter),
            "refine_stop_iter": int(self.inner.refine_stop_iter),
            "refine_every": int(self.inner.refine_every),
            "exact_mipmap_lifecycle": self.exact_mipmap_lifecycle,
            "lifecycle_execution_order": self.lifecycle_execution_order,
            "vendor_cull_warmup_profile": self.vendor_cull_warmup_profile,
            "vendor_opacity_reset_profile": self.vendor_opacity_reset_profile,
            "growth_min_opacity": self.growth_min_opacity,
            "prune_opa_late": self.prune_opa_late,
            "prune_switch_step": self.prune_switch_step,
            "reset_opacity_cap": self.reset_opacity_cap,
            "reset_adam_step": self.reset_adam_step,
            "reset_optimizer_state": self.reset_optimizer_state,
            "reset_before_cull": self.reset_before_cull,
            "capacity_cap": self.capacity_cap,
            "surface_birth_guard": (
                None
                if self.surface_birth_proposal is None
                else self.surface_birth_proposal.config.to_dict()
            ),
            "opacity_cull_policy": self.opacity_cull_policy,
            "opacity_cull_min_observations": self.opacity_cull_min_observations,
            "opacity_cull_consecutive_events": self.opacity_cull_consecutive_events,
            "opacity_cull_grace_after_reset_steps": (
                self.opacity_cull_grace_after_reset_steps
            ),
            "opacity_cull_max_fraction": self.opacity_cull_max_fraction,
            "opacity_cull_priority": self.opacity_cull_priority,
            "opacity_cull_local_voxel_m": self.opacity_cull_local_voxel_m,
            "opacity_cull_local_protection": (
                self.opacity_cull_local_protection
            ),
            "opacity_cull_local_min_accumulated_alpha": (
                self.opacity_cull_local_min_accumulated_alpha
            ),
            "detail_split_policy": self.detail_split_policy,
            "detail_split_scale_m": self.detail_split_scale_m,
            "detail_split_screen_radius": self.detail_split_screen_radius,
            "capacity_conserving_clone_opacity": (
                self.capacity_conserving_clone_opacity
            ),
            # The resolved metric meaning of the two normalised scale gates.
            "effective_split_scale_m": float(
                self.inner.grow_scale3d * self.scene_scale
            ),
            "effective_prune_scale_m": float(
                self.inner.prune_scale3d * self.scene_scale
            ),
        }
