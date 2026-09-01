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


class DefaultStrategyAdapter:
    """gsplat ``DefaultStrategy`` behind the attribute surface MCMC exposes."""

    def __init__(
        self,
        *,
        scene_scale: float = 1.0,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 15000,
        refine_every: int = 100,
        reset_every: int = 300,
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
        detail_split_policy: str = "vendor_0_2m",
        detail_split_scale_m: float = 0.02,
        detail_split_screen_radius: float = 0.0035,
        capacity_conserving_clone_opacity: bool = False,
        lifecycle_execution_order: str = "post_optimizer_gsplat",
        cloudstudio_lifecycle_extension_profile: str = "disabled",
        vendor_cull_warmup_profile: str = "exact_0p10_to_0p05",
        vendor_capacity_cull_profile: str = "disabled",
        vendor_opacity_reset_profile: str = "exact_every300",
        gradient_statistics_profile: str = "legacy_gsplat",
        gradient_tile_core_box: list[list[float]] | None = None,
        gradient_tile_outside_attenuation: float = 0.1,
        discard_accumulated_gradient_steps: list[int] | None = None,
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
        self.detail_split_policy = str(detail_split_policy)
        self.detail_split_scale_m = float(detail_split_scale_m)
        self.detail_split_screen_radius = float(detail_split_screen_radius)
        self.capacity_conserving_clone_opacity = bool(
            capacity_conserving_clone_opacity
        )
        self.lifecycle_execution_order = str(lifecycle_execution_order)
        self.cloudstudio_lifecycle_extension_profile = str(
            cloudstudio_lifecycle_extension_profile
        )
        self.vendor_cull_warmup_profile = str(vendor_cull_warmup_profile)
        self.vendor_capacity_cull_profile = str(vendor_capacity_cull_profile)
        self.vendor_opacity_reset_profile = str(vendor_opacity_reset_profile)
        self.gradient_statistics_profile = str(gradient_statistics_profile)
        self.gradient_tile_core_box = gradient_tile_core_box
        self.gradient_tile_outside_attenuation = float(
            gradient_tile_outside_attenuation
        )
        self.discard_accumulated_gradient_steps = frozenset(
            int(value) for value in (discard_accumulated_gradient_steps or [])
        )
        if any(value < 0 for value in self.discard_accumulated_gradient_steps):
            raise ValueError("discarded accumulated-gradient steps must be non-negative")
        self.last_lifecycle_event: dict[str, Any] | None = None
        self._last_surface_birth_event: dict[str, Any] | None = None
        self._last_growth_event: dict[str, Any] | None = None
        self._last_cull_event: dict[str, Any] | None = None
        if self.opacity_cull_policy not in {
            "immediate",
            "observation_aware",
            "local_coverage_competition",
        }:
            raise ValueError(
                "opacity_cull_policy must be immediate, observation_aware, "
                "or local_coverage_competition"
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
            "lidar_surface_screen_detail_aggressive",
            "lidar_surface_screen_detail_ultrasharp",
        }:
            raise ValueError("detail_split_policy is invalid")
        if self.lifecycle_execution_order not in {
            "post_optimizer_gsplat",
            "pre_optimizer_vendor",
        }:
            raise ValueError("lifecycle_execution_order is invalid")
        if self.cloudstudio_lifecycle_extension_profile not in {
            "disabled",
            "observation_cull_v1",
            "observation_cull_v2_conservative",
        }:
            raise ValueError(
                "cloudstudio_lifecycle_extension_profile is invalid"
            )
        if self.gradient_statistics_profile not in {
            "legacy_gsplat",
            "mipmap_radius_weighted_probe_v1",
            "mipmap_radius_weighted_v1",
        }:
            raise ValueError("gradient_statistics_profile is invalid")
        if not 0.0 < self.gradient_tile_outside_attenuation <= 1.0:
            raise ValueError(
                "gradient_tile_outside_attenuation must be within (0, 1]"
            )
        if self.gradient_statistics_profile != "legacy_gsplat":
            if (
                not isinstance(self.gradient_tile_core_box, list)
                or len(self.gradient_tile_core_box) != 2
                or any(len(corner) != 3 for corner in self.gradient_tile_core_box)
            ):
                raise ValueError(
                    "MipMap gradient statistics require a 2x3 Tile core box"
                )
        if self.cloudstudio_lifecycle_extension_profile in {
            "observation_cull_v1",
            "observation_cull_v2_conservative",
        }:
            expected_cull_fraction = (
                0.02
                if self.cloudstudio_lifecycle_extension_profile
                == "observation_cull_v2_conservative"
                else 0.05
            )
            required_observation_cull = {
                "lifecycle_execution_order": "pre_optimizer_vendor",
                "opacity_cull_policy": "observation_aware",
                "opacity_cull_min_observations": 64,
                "opacity_cull_consecutive_events": 2,
                "opacity_cull_grace_after_reset_steps": 200,
                "opacity_cull_max_fraction": expected_cull_fraction,
                "opacity_cull_priority": "lowest_opacity",
                "opacity_cull_local_min_accumulated_alpha": 0.0,
                "vendor_capacity_cull_profile": "exact_relaxed_at_cap",
            }
            actual_observation_cull = {
                "lifecycle_execution_order": self.lifecycle_execution_order,
                "opacity_cull_policy": self.opacity_cull_policy,
                "opacity_cull_min_observations": self.opacity_cull_min_observations,
                "opacity_cull_consecutive_events": self.opacity_cull_consecutive_events,
                "opacity_cull_grace_after_reset_steps": (
                    self.opacity_cull_grace_after_reset_steps
                ),
                "opacity_cull_max_fraction": self.opacity_cull_max_fraction,
                "opacity_cull_priority": self.opacity_cull_priority,
                "opacity_cull_local_min_accumulated_alpha": (
                    self.opacity_cull_local_min_accumulated_alpha
                ),
                "vendor_capacity_cull_profile": self.vendor_capacity_cull_profile,
            }
            mismatched_observation_cull = {
                key: (expected, actual_observation_cull[key])
                for key, expected in required_observation_cull.items()
                if actual_observation_cull[key] != expected
            }
            if mismatched_observation_cull:
                raise ValueError(
                    f"{self.cloudstudio_lifecycle_extension_profile} parameters differ: "
                    f"{mismatched_observation_cull}"
                )
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
        }:
            raise ValueError("vendor_cull_warmup_profile is invalid")
        if self.vendor_capacity_cull_profile not in {
            "disabled",
            "exact_relaxed_at_cap",
            "cloudstudio_relaxed_near_cap_0p99",
        }:
            raise ValueError("vendor_capacity_cull_profile is invalid")
        if (
            self.vendor_capacity_cull_profile
            in {"exact_relaxed_at_cap", "cloudstudio_relaxed_near_cap_0p99"}
            and (not self.exact_mipmap_lifecycle or self.capacity_cap is None)
        ):
            raise ValueError(
                "exact relaxed capacity cull requires the exact MipMap "
                "lifecycle and an absolute capacity cap"
            )
        vendor_reset_intervals = {
            "exact_every300": 300,
            "deferred_every3000_compatibility": 3000,
            "deferred_every6000_ab_control": 6000,
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
            if self.lifecycle_execution_order == "pre_optimizer_vendor":
                if growth_min_opacity is not None:
                    raise ValueError(
                        "pre_optimizer_vendor uses the recovered dead opacity "
                        "candidate expression and therefore requires no active "
                        "growth_min_opacity gate"
                    )
            elif growth_min_opacity is None or not 0.0 < growth_min_opacity < 1.0:
                raise ValueError("exact MipMap compatibility lifecycle requires growth_min_opacity")
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
        """Return the signed absolute cap used to admit classic births."""
        return self.capacity_cap

    # -- lifecycle -----------------------------------------------------------

    def initialize_state(self) -> dict[str, Any]:
        state = self.inner.initialize_state(scene_scale=self.scene_scale)
        if self.gradient_statistics_profile != "legacy_gsplat":
            for key in (
                "_cloudstudio_raw_grad_sum",
                "_cloudstudio_raw_grad_count",
                "_cloudstudio_screen_grad_sum",
                "_cloudstudio_mipmap_grad_sum",
                "_cloudstudio_mipmap_weight_sum",
                "_cloudstudio_mipmap_max_screen",
            ):
                state[key] = None
        if self.opacity_cull_policy != "immediate":
            # These are per-Gaussian tensors on purpose. gsplat's duplicate,
            # split and remove operations apply the same topology transform to
            # every tensor in strategy state, so the protection survives
            # growth, pruning and checkpoint resume without a parallel index.
            state["_cloudstudio_cull_low_streak"] = None
            state["_cloudstudio_cull_observations"] = None
            state["_cloudstudio_last_opacity_reset_step"] = None
        return state

    def _update_mipmap_equivalent_state(
        self,
        params: Any,
        state: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        """Accumulate the recovered footprint-weighted MipMap gradient."""

        if self.gradient_statistics_profile == "legacy_gsplat":
            return
        import torch

        projected = info[self.inner.key_for_gradient]
        gradients = projected.grad
        if gradients is None:
            raise RuntimeError("MipMap gradient probe requires means2d.grad")
        raw_norm = torch.linalg.vector_norm(gradients.detach(), dim=-1)
        radii = info["radii"].detach().float()
        footprint = (
            radii
            if radii.ndim == 1
            else torch.linalg.vector_norm(radii, dim=-1)
        )
        if raw_norm.ndim == 1:
            raw_norm = raw_norm.unsqueeze(0)
        if footprint.ndim == 1:
            footprint = footprint.unsqueeze(0)
        if raw_norm.shape != footprint.shape:
            raise RuntimeError(
                "MipMap gradient and footprint shapes differ: "
                f"{tuple(raw_norm.shape)} vs {tuple(footprint.shape)}"
            )
        count = len(params["means"])
        if raw_norm.shape[-1] != count:
            raise RuntimeError("MipMap gradient probe Gaussian count mismatch")
        device = params["means"].device
        for key in (
            "_cloudstudio_raw_grad_sum",
            "_cloudstudio_raw_grad_count",
            "_cloudstudio_screen_grad_sum",
            "_cloudstudio_mipmap_grad_sum",
            "_cloudstudio_mipmap_weight_sum",
            "_cloudstudio_mipmap_max_screen",
        ):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or len(value) != count:
                state[key] = torch.zeros(count, dtype=torch.float32, device=device)

        visible = footprint > 0
        visible_f = visible.float()
        image_scale = 0.5 * max(
            1600, int(info["width"]), int(info["height"])
        )
        raw_norm = raw_norm.float()
        screen_scaled = raw_norm * float(image_scale)
        core_box = torch.as_tensor(
            self.gradient_tile_core_box,
            dtype=params["means"].dtype,
            device=device,
        )
        outside = (
            (params["means"].detach() < core_box[0])
            | (params["means"].detach() > core_box[1])
        ).any(dim=-1)
        attenuation = torch.where(
            outside,
            torch.full(
                (count,),
                self.gradient_tile_outside_attenuation,
                dtype=screen_scaled.dtype,
                device=device,
            ),
            torch.ones(count, dtype=screen_scaled.dtype, device=device),
        )
        weighted_gradient = screen_scaled * attenuation.unsqueeze(0)
        weights = footprint.float()
        state["_cloudstudio_raw_grad_sum"].add_(
            (raw_norm * visible_f).sum(dim=0)
        )
        state["_cloudstudio_raw_grad_count"].add_(visible_f.sum(dim=0))
        state["_cloudstudio_screen_grad_sum"].add_(
            (screen_scaled * visible_f).sum(dim=0)
        )
        state["_cloudstudio_mipmap_grad_sum"].add_(
            (weighted_gradient * weights * visible_f).sum(dim=0)
        )
        state["_cloudstudio_mipmap_weight_sum"].add_(
            (weights * visible_f).sum(dim=0)
        )
        normalized_footprint = weights / float(
            max(int(info["width"]), int(info["height"]))
        )
        per_gaussian_max = torch.where(
            visible,
            normalized_footprint,
            torch.zeros_like(normalized_footprint),
        ).amax(dim=0)
        state["_cloudstudio_mipmap_max_screen"].copy_(
            torch.maximum(
                state["_cloudstudio_mipmap_max_screen"], per_gaussian_max
            )
        )

    def _mipmap_gradient_diagnostics(
        self, state: dict[str, Any], opacity_logits: Any
    ) -> dict[str, Any]:
        if self.gradient_statistics_profile == "legacy_gsplat":
            return {}
        import torch

        raw_count = state["_cloudstudio_raw_grad_count"]
        weight_sum = state["_cloudstudio_mipmap_weight_sum"]
        observed = weight_sum > 0
        raw = state["_cloudstudio_raw_grad_sum"] / raw_count.clamp_min(1.0)
        screen = state["_cloudstudio_screen_grad_sum"] / raw_count.clamp_min(1.0)
        final = state["_cloudstudio_mipmap_grad_sum"] / weight_sum.clamp_min(1e-8)

        def summarize(values: Any) -> dict[str, float]:
            finite = values[observed & torch.isfinite(values)].float()
            if not finite.numel():
                return {}
            levels = torch.tensor(
                [0.5, 0.9, 0.95, 0.99, 0.999],
                dtype=finite.dtype,
                device=finite.device,
            )
            quantiles = torch.quantile(finite, levels).tolist()
            result = {
                key: float(value)
                for key, value in zip(
                    ("p50", "p90", "p95", "p99", "p999"), quantiles
                )
            }
            result["max"] = float(finite.max().item())
            return result

        opacity = torch.sigmoid(opacity_logits.detach().flatten())
        return {
            "profile": self.gradient_statistics_profile,
            "image_scale_formula": "0.5*max(1600,width,height)",
            "footprint_weight": "l2_raster_radius_px",
            "tile_outside_attenuation": self.gradient_tile_outside_attenuation,
            "raw_grad_quantiles": summarize(raw),
            "screen_scaled_grad_quantiles": summarize(screen),
            "mipmap_equivalent_grad_quantiles": summarize(final),
            "mipmap_equivalent_candidate_counts": {
                f"grad_gt_{threshold:.7f}_opacity_gt_{floor:.2f}": int(
                    ((final > threshold) & (opacity > floor)).sum().item()
                )
                for threshold in (
                    float(self.inner.grow_grad2d),
                    0.00015,
                    0.0002,
                )
                for floor in (0.05, 0.1, 0.15)
            },
            "observed_gaussian_count": int(observed.sum().item()),
            "max_screen_footprint_quantiles": summarize(
                state["_cloudstudio_mipmap_max_screen"]
            ),
        }

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
        """Classic split/clone with the recovered projected-gradient gate."""

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

        if self.gradient_statistics_profile == "mipmap_radius_weighted_v1":
            gradients = (
                state["_cloudstudio_mipmap_grad_sum"]
                / state["_cloudstudio_mipmap_weight_sum"].clamp_min(1e-8)
            )
            observed = state["_cloudstudio_mipmap_weight_sum"] > 0
            screen_radii = state["_cloudstudio_mipmap_max_screen"]
        else:
            gradients = state["grad2d"] / state["count"].clamp_min(1)
            observed = state["count"] > 0
            screen_radii = state.get("radii")
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
            quantile_values = torch.quantile(
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
            eligible &= opacity > float(self.growth_min_opacity)
        growth_candidate_count = int(eligible.sum().item())
        guarded_births = bool(
            self.surface_birth_proposal is not None
            and self.surface_birth_proposal.config.reject_unsupported_births
        )
        if (
            guarded_births
            and self.lifecycle_execution_order == "pre_optimizer_vendor"
            and not (
                self.vendor_capacity_cull_profile
                in {
                    "exact_relaxed_at_cap",
                    "cloudstudio_relaxed_near_cap_0p99",
                }
                and self.detail_split_policy
                in {
                    "lidar_surface_screen_detail",
                    "lidar_surface_screen_detail_aggressive",
                    "lidar_surface_screen_detail_ultrasharp",
                }
            )
        ):
            raise RuntimeError(
                "pre_optimizer_vendor permits the CloudStudio surface birth "
                "guard only for a signed cap-aware screen-detail profile"
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
        radii = screen_radii
        if (
            self.detail_split_policy
            in {
                "lidar_surface_screen_detail",
                "lidar_surface_screen_detail_aggressive",
                "lidar_surface_screen_detail_ultrasharp",
            }
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
        self._last_growth_event = {
            "gradient_statistics_profile": self.gradient_statistics_profile,
            "gradient_threshold": float(self.inner.grow_grad2d),
            "opacity_floor": (
                None
                if self.growth_min_opacity is None
                else float(self.growth_min_opacity)
            ),
            "observed_gaussian_count": int(observed.sum().item()),
            "gradient_quantiles": gradient_quantiles,
            "threshold_sweep_counts": threshold_sweep,
            "selected_parent_count": int(eligible.sum().item()),
            "clone_parent_count": duplicate_count,
            "split_parent_count": split_count,
            "capacity_conserving_clone_opacity": (
                self.capacity_conserving_clone_opacity
            ),
        }
        parent_means = None
        if guarded_births and (duplicate_count or split_count):
            clone_indices = torch.where(duplicate_mask)[0]
            split_indices = torch.where(split_mask)[0]
            clone_parent_means = params["means"][clone_indices].detach().clone()
            split_parent_means = (
                params["means"][split_indices].detach().clone().repeat(2, 1)
            )
            if self.lifecycle_execution_order == "pre_optimizer_vendor":
                # Native order is Split -> Clone. ``split`` appends its two
                # children first, then ``duplicate`` appends clone children.
                # The proposal rows must follow that exact newborn-tail order;
                # pairing Clone parents first relocates Split children onto an
                # unrelated LiDAR patch and can create large apparent floaters.
                parent_means = torch.cat(
                    [split_parent_means, clone_parent_means], dim=0
                )
            else:
                parent_means = torch.cat(
                    [clone_parent_means, split_parent_means], dim=0
                )
        if self.lifecycle_execution_order == "pre_optimizer_vendor":
            # The recovered native order is Split -> Clone -> Cull.  The masks
            # are disjoint and were computed on the original rows. After Split,
            # non-split rows lead the tensor and split children occupy the tail,
            # so remap the clone mask onto that layout before duplicating.
            if split_count:
                original_split_mask = split_mask
                split(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=original_split_mask,
                    revised_opacity=self.inner.revised_opacity,
                    scene=scene,
                )
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
                split(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=split_mask,
                    revised_opacity=self.inner.revised_opacity,
                    scene=scene,
                )
        if parent_means is not None:
            # ``split`` removes its parents, keeps all non-split rows, then
            # appends two children per parent. The newborn tail begins at
            # ``old_count - split_count``; its internal order depends on the
            # selected lifecycle execution order and is mirrored above.
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
        capacity_limited: bool = False,
        scene: Any | None = None,
    ) -> int:
        """Apply recovered early/late opacity, world, and screen cull gates."""

        import torch
        from gsplat.strategy.ops import remove

        relaxed_capacity_cull = bool(
            capacity_limited
            and self.vendor_capacity_cull_profile
            in {"exact_relaxed_at_cap", "cloudstudio_relaxed_near_cap_0p99"}
        )
        if relaxed_capacity_cull:
            # Recovered vendor behavior for a lifecycle event where the
            # absolute population cap forbids further densification.  This is
            # intentionally not the normal early/late opacity schedule: the
            # native code relaxes opacity by 0.25x and both geometric limits by
            # 5x, allowing a cap-only maintenance pass without deleting the
            # continuous surface that the cap is meant to preserve.
            opacity_threshold = 0.25 * float(self.prune_opa_late)
            world_scale_threshold = 5.0 * float(self.prune_scale_m)
            screen_scale_threshold = 5.0 * float(self.inner.prune_scale2d)
        else:
            opacity_threshold = (
                float(self.prune_opa_late)
                if step >= int(self.prune_switch_step)
                else float(self.inner.prune_opa)
            )
            world_scale_threshold = float(self.prune_scale_m)
            screen_scale_threshold = float(self.inner.prune_scale2d)
        opacity = torch.sigmoid(params["opacities"].flatten())
        raw_opacity_mask = opacity < opacity_threshold
        opacity_threshold_sweep_counts = {
            f"opacity_lt_{threshold:.3f}": int(
                (opacity < threshold).sum().item()
            )
            for threshold in (0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1)
        }
        maximum_scale = torch.exp(params["scales"]).max(dim=-1).values
        world_scale_mask = maximum_scale > world_scale_threshold
        radii = state.get("radii")
        screen_scale_mask = torch.zeros_like(raw_opacity_mask)
        if radii is not None:
            screen_scale_mask = radii > screen_scale_threshold

        opacity_mask = raw_opacity_mask.clone()
        grace_active = False
        local_competition_protected_count = 0
        local_competition_cell_count = 0
        if self.opacity_cull_policy != "immediate":
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
                    torch.quantile(candidate_observations, 0.50).item()
                )
                raw_candidate_observation_p95 = float(
                    torch.quantile(candidate_observations, 0.95).item()
                )
        self._last_cull_event = {
            "policy": self.opacity_cull_policy,
            "capacity_limited": bool(capacity_limited),
            "capacity_profile": self.vendor_capacity_cull_profile,
            "relaxed_capacity_cull": relaxed_capacity_cull,
            "opacity_threshold": float(opacity_threshold),
            "world_scale_threshold_m": float(world_scale_threshold),
            "screen_scale_threshold": float(screen_scale_threshold),
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
        }
        if prune_count:
            remove(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=remove_mask,
                scene=scene,
            )
        return prune_count

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
        from gsplat.strategy.ops import reset_opa

        self.last_lifecycle_event = None
        self._last_surface_birth_event = None
        self._last_growth_event = None
        self._last_cull_event = None
        if step >= self.refine_stop_iter:
            return
        self.inner._update_state(params, state, info, packed=False)
        self._update_mipmap_equivalent_state(params, state, info)
        if step in self.discard_accumulated_gradient_steps:
            state["grad2d"].zero_()
            state["count"].zero_()
            if state.get("radii") is not None:
                state["radii"].zero_()
            for key in (
                "_cloudstudio_raw_grad_sum",
                "_cloudstudio_raw_grad_count",
                "_cloudstudio_screen_grad_sum",
                "_cloudstudio_mipmap_grad_sum",
                "_cloudstudio_mipmap_weight_sum",
                "_cloudstudio_mipmap_max_screen",
            ):
                if state.get(key) is not None:
                    state[key].zero_()
            self.last_lifecycle_event = {
                "discarded_accumulated_gradient_step": int(step),
                "reason": "signed_supervision_mask_transition",
            }
            return
        if not self.is_refine_step(step):
            return
        mipmap_gradient_diagnostics = self._mipmap_gradient_diagnostics(
            state, params["opacities"]
        )
        self._ensure_cull_tracking(params, state)
        before = len(params["means"])
        # Vendor parity: capacity mode is frozen from the population entering
        # this lifecycle.  A cycle that starts below the cap is allowed to
        # grow and still uses the normal Cull thresholds even when those
        # births reach the cap; only the following cycle enters the relaxed
        # at-cap branch.  Recomputing this after growth retains one extra wave
        # of low-opacity mass and was visible as fog in capacity-bound runs.
        capacity_trigger = self.capacity_cap
        if (
            capacity_trigger is not None
            and self.vendor_capacity_cull_profile
            == "cloudstudio_relaxed_near_cap_0p99"
        ):
            capacity_trigger = max(1, int(0.99 * capacity_trigger))
        capacity_limited = bool(
            capacity_trigger is not None and before >= capacity_trigger
        )
        clone_count, split_count = self._grow_mipmap(
            params, optimizers, state
        )
        cull_count = self._prune_mipmap(
            params,
            optimizers,
            state,
            step=step,
            capacity_limited=capacity_limited,
        )
        reset = step % int(self.inner.reset_every) == 0
        if reset:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=float(self.reset_opacity_cap),
            )
            if self.opacity_cull_policy != "immediate":
                state["_cloudstudio_cull_low_streak"].zero_()
                state["_cloudstudio_cull_observations"].zero_()
                state["_cloudstudio_last_opacity_reset_step"] = int(step)
        state["grad2d"].zero_()
        state["count"].zero_()
        if state.get("radii") is not None:
            state["radii"].zero_()
        for key in (
            "_cloudstudio_raw_grad_sum",
            "_cloudstudio_raw_grad_count",
            "_cloudstudio_screen_grad_sum",
            "_cloudstudio_mipmap_grad_sum",
            "_cloudstudio_mipmap_weight_sum",
            "_cloudstudio_mipmap_max_screen",
        ):
            if state.get(key) is not None:
                state[key].zero_()
        self.last_lifecycle_event = {
            "before_count": int(before),
            "clone_count": clone_count,
            "split_parent_count": split_count,
            "split_child_count": 2 * split_count,
            "cull_count": cull_count,
            "opacity_reset": reset,
            "after_count": int(len(params["means"])),
            "capacity_limited_pre_growth": capacity_limited,
            "capacity_trigger": (
                int(capacity_trigger) if capacity_trigger is not None else None
            ),
            "cull_opacity_threshold": (
                float(self._last_cull_event["opacity_threshold"])
                if self._last_cull_event is not None
                else None
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
            self.last_lifecycle_event["growth_diagnostics"].update(
                mipmap_gradient_diagnostics
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
            "cloudstudio_lifecycle_extension_profile": (
                self.cloudstudio_lifecycle_extension_profile
            ),
            "vendor_cull_warmup_profile": self.vendor_cull_warmup_profile,
            "vendor_capacity_cull_profile": (
                self.vendor_capacity_cull_profile
            ),
            "vendor_opacity_reset_profile": self.vendor_opacity_reset_profile,
            "growth_min_opacity": self.growth_min_opacity,
            "prune_opa_late": self.prune_opa_late,
            "prune_switch_step": self.prune_switch_step,
            "reset_opacity_cap": self.reset_opacity_cap,
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
            "gradient_statistics_profile": self.gradient_statistics_profile,
            "gradient_tile_core_box": self.gradient_tile_core_box,
            "gradient_tile_outside_attenuation": (
                self.gradient_tile_outside_attenuation
            ),
            "discard_accumulated_gradient_steps": sorted(
                self.discard_accumulated_gradient_steps
            ),
            # The resolved metric meaning of the two normalised scale gates.
            "effective_split_scale_m": float(
                self.inner.grow_scale3d * self.scene_scale
            ),
            "effective_prune_scale_m": float(
                self.inner.prune_scale3d * self.scene_scale
            ),
        }
