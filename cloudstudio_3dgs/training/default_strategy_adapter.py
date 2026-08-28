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
        capacity_cap: int | None = None,
        surface_birth_proposal: Any | None = None,
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
        self.last_lifecycle_event: dict[str, Any] | None = None
        self._last_surface_birth_event: dict[str, Any] | None = None
        if self.exact_mipmap_lifecycle:
            if growth_min_opacity is None or not 0.0 < growth_min_opacity < 1.0:
                raise ValueError("exact MipMap lifecycle requires growth_min_opacity")
            if prune_opa_late is None or not 0.0 < prune_opa_late < 1.0:
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
        return self.inner.initialize_state(scene_scale=self.scene_scale)

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
            self._step_post_backward_mipmap(
                params=params,
                optimizers=optimizers,
                state=state,
                step=step,
                info=info,
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

        gradients = state["grad2d"] / state["count"].clamp_min(1)
        opacity = torch.sigmoid(params["opacities"].flatten())
        eligible = gradients > float(self.inner.grow_grad2d)
        eligible &= opacity > float(self.growth_min_opacity)
        growth_candidate_count = int(eligible.sum().item())
        guarded_births = bool(
            self.surface_birth_proposal is not None
            and self.surface_birth_proposal.config.reject_unsupported_births
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
        duplicate_mask = eligible & small
        split_mask = eligible & ~small
        duplicate_count = int(duplicate_mask.sum().item())
        split_count = int(split_mask.sum().item())
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
        if duplicate_count:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=duplicate_mask,
                scene=scene,
            )
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
    ) -> int:
        """Apply recovered early/late opacity, world, and screen cull gates."""

        import torch
        from gsplat.strategy.ops import remove

        opacity_threshold = (
            float(self.prune_opa_late)
            if step >= int(self.prune_switch_step)
            else float(self.inner.prune_opa)
        )
        remove_mask = torch.sigmoid(params["opacities"].flatten()) < opacity_threshold
        maximum_scale = torch.exp(params["scales"]).max(dim=-1).values
        remove_mask |= maximum_scale > float(self.prune_scale_m)
        radii = state.get("radii")
        if radii is not None:
            remove_mask |= radii > float(self.inner.prune_scale2d)
        prune_count = int(remove_mask.sum().item())
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
        if step >= self.refine_stop_iter:
            return
        self.inner._update_state(params, state, info, packed=False)
        if not self.is_refine_step(step):
            return
        before = len(params["means"])
        clone_count, split_count = self._grow_mipmap(
            params, optimizers, state
        )
        cull_count = self._prune_mipmap(
            params, optimizers, state, step=step
        )
        reset = step % int(self.inner.reset_every) == 0
        if reset:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=float(self.reset_opacity_cap),
            )
        state["grad2d"].zero_()
        state["count"].zero_()
        if state.get("radii") is not None:
            state["radii"].zero_()
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
            # The resolved metric meaning of the two normalised scale gates.
            "effective_split_scale_m": float(
                self.inner.grow_scale3d * self.scene_scale
            ),
            "effective_prune_scale_m": float(
                self.inner.prune_scale3d * self.scene_scale
            ),
        }
