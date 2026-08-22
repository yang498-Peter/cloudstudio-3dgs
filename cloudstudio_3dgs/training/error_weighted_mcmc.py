# SPDX-License-Identifier: Apache-2.0
#
# relocate_weighted() and sample_add_weighted() are adapted from
# gsplat 1.5.3 (Apache-2.0) gsplat/strategy/ops.py (relocate / sample_add):
# the only behavioural change is that the multinomial sampling distribution
# is a caller-supplied ``probs`` tensor instead of the hard-coded opacities.
"""Experimental error-weighted MCMC relocation and densification.

This is a CloudStudio heuristic motivated by error/edge-aware densification:
when MCMC teleports dead Gaussians or adds new ones, landing spots are sampled
proportionally to ``opacity * error_score**power`` instead of opacity alone.
It is not a reproduction of arXiv:2508.12313: that paper combines Laplacian
edge weights, per-pixel Gaussian alpha contributions, and absolute coordinate
gradients, while this implementation samples the RGB residual only at each
Gaussian's projected center. The distinction is kept explicit until a real A/B
establishes whether this cheaper proxy helps the S1 scene.

The per-Gaussian error score is an EMA of the rendered-vs-reference pixel
error sampled at each Gaussian's projected center (visible Gaussians only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from gsplat.relocation import compute_relocation
from gsplat.strategy.mcmc import MCMCStrategy
from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer

from cloudstudio_3dgs.training.error_weighted_config import ErrorScoreConfig


class ErrorScoreState:
    """Per-Gaussian EMA of image-space error, indexed like the parameter tensors."""

    def __init__(
        self,
        num_gaussians: int,
        config: ErrorScoreConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config if config is not None else ErrorScoreConfig()
        self.config.validate()
        if int(num_gaussians) < 0:
            raise ValueError("num_gaussians must be non-negative")
        self.scores: Tensor = torch.ones(int(num_gaussians), device=device)

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    @torch.no_grad()
    def update(
        self,
        means2d: Tensor,
        radii: Tensor,
        pixel_error: Tensor,
        height: int,
        width: int,
    ) -> None:
        """EMA-update scores of visible Gaussians from the current view's error map.

        Args:
            means2d: [N, 2] projected pixel centers (gsplat ``info["means2d"]``
                with the camera dim already squeezed).
            radii: [N] or [N, 2] projected radii; a Gaussian counts as visible
                only where every radius component is > 0 (gsplat convention).
            pixel_error: [H, W] per-pixel error of the current step, e.g. the
                channel-mean of ``|render - reference|`` (masked if applicable).
            height/width: dimensions of ``pixel_error``.
        """
        means2d = torch.as_tensor(means2d)
        if means2d.dim() != 2 or means2d.shape[-1] != 2:
            raise ValueError("means2d must have shape [N, 2]")
        n = int(means2d.shape[0])
        if n != len(self):
            raise ValueError(
                f"means2d has {n} Gaussians but state holds {len(self)}; "
                "call resize() after refinement changed the count"
            )
        radii = torch.as_tensor(radii)
        if radii.dim() == 2:
            visible = (radii > 0).all(dim=-1)
        elif radii.dim() == 1:
            visible = radii > 0
        else:
            raise ValueError("radii must have shape [N] or [N, 2]")
        if int(visible.shape[0]) != n:
            raise ValueError("radii count does not match means2d")
        pixel_error = torch.as_tensor(pixel_error)
        if pixel_error.dim() != 2 or pixel_error.shape != (int(height), int(width)):
            raise ValueError("pixel_error must have shape [height, width]")

        vis = visible.nonzero(as_tuple=True)[0]
        if vis.numel() == 0:
            return
        # Invisible entries may carry garbage projections; index visible only.
        # nan_to_num guards against stray non-finite centers before .long().
        centers = torch.nan_to_num(
            means2d[vis].detach().float(), nan=0.0, posinf=1e12, neginf=-1e12
        ).round().long()
        xs = centers[:, 0].clamp(0, int(width) - 1)
        ys = centers[:, 1].clamp(0, int(height) - 1)
        err = pixel_error.detach()[ys, xs]

        if self.scores.device != err.device:
            self.scores = self.scores.to(err.device)
        err = err.to(self.scores.dtype)
        vis = vis.to(self.scores.device)
        decay = float(self.config.ema_decay)
        self.scores[vis] = decay * self.scores[vis] + (1.0 - decay) * err

    @torch.no_grad()
    def resize(self, new_count: int) -> None:
        """Reset to all-ones at ``new_count`` after the Gaussian count changed.

        A full reset is deliberately simple and safe: scores rebuild within a
        few EMA windows, and all-ones weighting degrades exactly to gsplat's
        plain opacity sampling in the meantime.
        """
        if int(new_count) < 0:
            raise ValueError("new_count must be non-negative")
        self.scores = torch.ones(
            int(new_count), device=self.scores.device, dtype=self.scores.dtype
        )

    @torch.no_grad()
    def sampling_weights(self, opacities: Tensor) -> Tensor:
        """Return ``opacity * clamped_score**power`` aligned with the Gaussians."""
        opac = torch.as_tensor(opacities).flatten()
        if int(opac.shape[0]) != len(self):
            raise ValueError(
                f"opacities has {int(opac.shape[0])} entries but state holds {len(self)}"
            )
        score = self.scores.to(device=opac.device, dtype=opac.dtype)
        score = score.clamp_min(float(self.config.min_score_floor))
        return opac * score ** float(self.config.score_power)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return the complete state required for deterministic resume."""
        return {
            "schema_version": 1,
            "scores": self.scores.detach().clone(),
        }

    @torch.no_grad()
    def restore_checkpoint_state(
        self,
        payload: Any,
        *,
        expected_count: int,
    ) -> None:
        """Restore EMA scores, rejecting partial or stale checkpoint state."""
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("checkpoint error-weighted sampling state is invalid")
        scores = payload.get("scores")
        if not isinstance(scores, Tensor) or scores.dim() != 1:
            raise ValueError("checkpoint error scores must be a one-dimensional tensor")
        if int(scores.shape[0]) != int(expected_count):
            raise ValueError(
                "checkpoint error score count does not match restored Gaussians"
            )
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("checkpoint error scores contain non-finite values")
        self.scores = scores.detach().to(
            device=self.scores.device,
            dtype=self.scores.dtype,
        ).clone()


# ---------------------------------------------------------------------------
# Weighted variants of gsplat's MCMC ops.
# Adapted from gsplat 1.5.3 (Apache-2.0) gsplat/strategy/ops.py. Except for
# the injected ``probs``, the logic mirrors the originals line-for-line.
# ---------------------------------------------------------------------------


@torch.no_grad()
def relocate_weighted(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    binoms: Tensor,
    probs: Tensor,
    min_opacity: float = 0.005,
    scene: Any | None = None,
):
    """Inplace relocate dead Gaussians onto live ones sampled by ``probs``.

    Adapted from gsplat.strategy.ops.relocate; ``probs`` is a full-length
    [N] weight vector aligned with the Gaussians (e.g. from
    :meth:`ErrorScoreState.sampling_weights`). The alive subset is taken
    internally, exactly where the original takes ``opacities[alive_indices]``.
    """
    # support "opacities" with shape [N,] or [N, 1]
    opacities = torch.sigmoid(params["opacities"])

    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)

    probs = probs.flatten()
    if probs.shape[0] != mask.shape[0]:
        raise ValueError("probs must be a full-length [N] weight vector")

    # Sample for new GSs
    eps = torch.finfo(torch.float32).eps
    probs = probs[alive_indices]  # weights over the alive subset, shape [N_alive]
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    sampled_idxs = alive_indices[sampled_idxs]
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v[sampled_idxs] = 0
        return v

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v[sampled_idxs] = 0
    if scene is not None:
        scene.on_relocate(dead_indices, sampled_idxs)


@torch.no_grad()
def sample_add_weighted(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    n: int,
    binoms: Tensor,
    probs: Tensor,
    min_opacity: float = 0.005,
    scene: Any | None = None,
):
    """Inplace add ``n`` Gaussians cloned from existing ones sampled by ``probs``.

    Adapted from gsplat.strategy.ops.sample_add; ``probs`` is a full-length
    [N] weight vector aligned with the Gaussians.
    """
    opacities = torch.sigmoid(params["opacities"])

    eps = torch.finfo(torch.float32).eps
    probs = probs.flatten()
    if probs.shape[0] != opacities.flatten().shape[0]:
        raise ValueError("probs must be a full-length [N] weight vector")
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p_new = torch.cat([p, p[sampled_idxs]])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        return torch.cat([v, v_new])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v_new))
    if scene is not None:
        scene.on_sample_add(sampled_idxs)


@dataclass
class ErrorWeightedMCMCStrategy(MCMCStrategy):
    """MCMCStrategy whose relocation/densification samples by opacity*error.

    When ``error_config.enabled`` is False, or ``score_state`` is missing or
    out of sync with the Gaussian count, behaviour falls back to the parent's
    pure-opacity sampling (a fresh all-ones score gives the identical
    distribution anyway). ``initialize_state``/``check_sanity`` are inherited
    unchanged, so this is a drop-in replacement in GsplatBackend.
    """

    score_state: Optional[ErrorScoreState] = None
    error_config: ErrorScoreConfig = field(default_factory=ErrorScoreConfig)

    def __post_init__(self) -> None:
        self.error_config.validate()

    def _weighting_active(self, num_gaussians: int) -> bool:
        return (
            self.error_config.enabled
            and self.score_state is not None
            and len(self.score_state) == int(num_gaussians)
        )

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        scene: Any | None = None,
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())
        if not self._weighting_active(opacities.shape[0]):
            return super()._relocate_gs(params, optimizers, binoms, scene=scene)
        dead_mask = opacities <= self.min_opacity
        n_gs = int(dead_mask.sum().item())
        if n_gs > 0:
            assert self.score_state is not None
            probs = self.score_state.sampling_weights(opacities)
            relocate_weighted(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                probs=probs,
                min_opacity=self.min_opacity,
                scene=scene,
            )
            # Relocation keeps N constant; relocated Gaussians inherit their
            # slot's score and re-converge through the EMA.
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        scene: Any | None = None,
    ) -> int:
        current_n_points = len(params["means"])
        if not self._weighting_active(current_n_points):
            n_gs = super()._add_new_gs(params, optimizers, binoms, scene=scene)
            if n_gs > 0 and self.score_state is not None:
                self.score_state.resize(len(params["means"]))
            return n_gs
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            assert self.score_state is not None
            probs = self.score_state.sampling_weights(
                torch.sigmoid(params["opacities"].flatten())
            )
            sample_add_weighted(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                probs=probs,
                min_opacity=self.min_opacity,
                scene=scene,
            )
            self.score_state.resize(len(params["means"]))
        return n_gs
