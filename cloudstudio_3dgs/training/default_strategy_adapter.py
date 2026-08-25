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
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        prune_opa: float = 0.005,
        prune_scale3d: float = 0.1,
        absgrad: bool = False,
        revised_opacity: bool = False,
        verbose: bool = False,
    ) -> None:
        from gsplat.strategy import DefaultStrategy

        self.scene_scale = float(scene_scale)
        self.inner = DefaultStrategy(
            prune_opa=float(prune_opa),
            grow_grad2d=float(grow_grad2d),
            grow_scale3d=float(grow_scale3d),
            prune_scale3d=float(prune_scale3d),
            refine_start_iter=int(refine_start_iter),
            refine_stop_iter=int(refine_stop_iter),
            reset_every=int(reset_every),
            refine_every=int(refine_every),
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
        self.inner.step_post_backward(
            params=params, optimizers=optimizers, state=state, step=step, info=info
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": "default_3dgs",
            "scene_scale": self.scene_scale,
            "grow_grad2d": float(self.inner.grow_grad2d),
            "grow_scale3d": float(self.inner.grow_scale3d),
            "prune_opa": float(self.inner.prune_opa),
            "absgrad": bool(self.inner.absgrad),
            "revised_opacity": bool(self.inner.revised_opacity),
            "reset_every": int(self.inner.reset_every),
        }
