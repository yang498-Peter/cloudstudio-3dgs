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
    ) -> None:
        from gsplat.strategy import DefaultStrategy

        self.scene_scale = float(scene_scale)
        if self.scene_scale <= 0.0:
            raise ValueError("scene_scale must be positive")
        # Metric thresholds win when given, so a config can state what it means
        # in metres instead of in units of a scale factor it cannot see.
        self.split_scale_m = split_scale_m
        self.prune_scale_m = prune_scale_m
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
        self.inner.step_post_backward(
            params=params, optimizers=optimizers, state=state, step=step, info=info
        )

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
            # The resolved metric meaning of the two normalised scale gates.
            "effective_split_scale_m": float(
                self.inner.grow_scale3d * self.scene_scale
            ),
            "effective_prune_scale_m": float(
                self.inner.prune_scale3d * self.scene_scale
            ),
        }
