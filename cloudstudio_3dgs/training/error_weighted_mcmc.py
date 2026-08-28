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
error attributed to each visible Gaussian. Two attribution modes exist:

* ``aggregation="center"`` (v1, default): sample the error map at the
  Gaussian's projected center pixel only.
* ``aggregation="footprint"`` (v2): aggregate the error map over the
  Gaussian's 2D footprint with weights ``opacity * exp(-0.5 * d^T Sigma^-1 d)``
  computed from the rasterizer's ``conics`` — a PyTorch approximation of
  LichtFeld-style alpha/T-weighted error attribution. The v2 path only runs
  when ``conics`` is supplied to :meth:`ErrorScoreState.update`; otherwise
  behaviour is bit-for-bit the v1 path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from torch import Tensor

from gsplat.relocation import compute_relocation
from gsplat.strategy.mcmc import MCMCStrategy
from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer

from cloudstudio_3dgs.training.error_weighted_config import (
    ErrorScoreConfig as _CoreErrorScoreConfig,
)
from cloudstudio_3dgs.training.gaussian_lifecycle import GaussianLifecycleState

if TYPE_CHECKING:  # pragma: no cover - annotation only, no runtime dependency
    from cloudstudio_3dgs.training.lidar_admission import LidarAdmission
    from cloudstudio_3dgs.training.tangent_proposal import TangentProposal


@dataclass(frozen=True)
class ErrorScoreConfig(_CoreErrorScoreConfig):
    """Core :class:`ErrorScoreConfig` extended with v2 footprint aggregation.

    Extended here instead of in ``error_weighted_config`` so the torch-free
    core module (consumed by the Trainer config parser and synthetic
    acceptance) stays untouched; consumers importing ``ErrorScoreConfig``
    from this module transparently get the extended class. ``to_dict`` is
    deliberately inherited unchanged: the 4-field payload is the schema
    existing telemetry/config consumers expect, and the v2 knobs only affect
    in-process score updates.

    Attributes:
        aggregation: ``"center"`` samples the error map at the projected
            center (v1); ``"footprint"`` aggregates over the conic-weighted
            pixel footprint (v2) whenever ``conics`` is provided to
            :meth:`ErrorScoreState.update`.
        footprint_radius_px: clamp for the footprint window half-size R; the
            v2 gather window is a fixed ``(2R+1)**2`` patch per Gaussian.
    """

    aggregation: str = "center"
    footprint_radius_px: int = 4

    def validate(self) -> None:
        super().validate()
        if self.aggregation not in ("center", "footprint", "contribution"):
            raise ValueError(
                'aggregation must be "center", "footprint" or "contribution"'
            )
        if int(self.footprint_radius_px) < 1:
            raise ValueError("footprint_radius_px must be a positive integer")


class ErrorScoreState:
    """Per-Gaussian EMA of image-space error, indexed like the parameter tensors.

    The scores are no longer owned by this class: they are the ``error_ema``
    column of a :class:`GaussianLifecycleState`, which carries the rest of the
    per-Gaussian bookkeeping (anchors, generation, parent, age) through the
    same relocate/grow/prune index mutations. ``scores`` stays a plain
    read/write attribute so every existing caller and test is unaffected.
    """

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
        self.lifecycle = GaussianLifecycleState(int(num_gaussians), device=device)

    @property
    def scores(self) -> Tensor:
        """The lifecycle's ``error_ema`` column (live tensor, not a copy)."""
        return self.lifecycle.error_ema

    @scores.setter
    def scores(self, value: Any) -> None:
        tensor = torch.as_tensor(value)
        if tensor.dim() != 1:
            raise ValueError("scores must be a one-dimensional tensor")
        if int(tensor.shape[0]) != len(self.lifecycle):
            # Keep the sibling lifecycle columns aligned with the new length
            # instead of silently desynchronizing the index space.
            self.lifecycle.resize(int(tensor.shape[0]))
        if tensor.device != self.lifecycle.device:
            self.lifecycle.to(tensor.device)
        self.lifecycle.error_ema = tensor

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    @torch.no_grad()
    def on_step(self, step: int) -> None:
        """Advance the lifecycle clock (ages) by one training step."""
        self.lifecycle.on_step(int(step))

    @torch.no_grad()
    def reset(self, num_gaussians: int) -> None:
        """Discard all history and start over with ``num_gaussians`` fresh rows.

        For a brand-new cloud only (backend initialization). Refinements must
        use :meth:`resize` / the lifecycle operations, which keep history.
        """
        self.lifecycle = GaussianLifecycleState(
            int(num_gaussians), device=self.lifecycle.device
        )

    @torch.no_grad()
    def update(
        self,
        means2d: Tensor,
        radii: Tensor,
        pixel_error: Tensor,
        height: int,
        width: int,
        conics: Optional[Tensor] = None,
        opacities: Optional[Tensor] = None,
        contribution: Optional[Tensor] = None,
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
            conics: optional [N, 3] (or [1, N, 3]) inverse-2D-covariance upper
                triangles ``(a, b, c)`` from gsplat ``info["conics"]``. Only
                used when ``config.aggregation == "footprint"``; supplying it
                enables the v2 footprint-weighted aggregation. Without it the
                v1 center-sampling path runs bit-for-bit unchanged.
            opacities: optional [N] (or [1, N] / [N, 1]) per-Gaussian
                opacities used as the footprint weight prefactor. Ignored on
                the v1 path.
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
        aggregation = str(getattr(self.config, "aggregation", "center"))
        if aggregation == "contribution" and contribution is not None:
            # The caller already rendered the alpha/transmittance-weighted
            # error per Gaussian (contribution_attribution); EMA it directly.
            score = torch.as_tensor(contribution).reshape(-1)
            if int(score.shape[0]) != n:
                raise ValueError("contribution count does not match means2d")
            if self.scores.device != score.device:
                self.scores = self.scores.to(score.device)
            score = score.to(self.scores.dtype)
            decay = float(self.config.ema_decay)
            finite = torch.isfinite(score[vis])
            rows = vis[finite]
            if rows.numel():
                self.scores[rows] = (
                    decay * self.scores[rows] + (1.0 - decay) * score[rows]
                )
            return
        if aggregation == "footprint" and conics is not None:
            self._update_footprint(
                means2d,
                radii,
                pixel_error,
                int(height),
                int(width),
                conics,
                opacities,
                vis,
            )
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
    def _update_footprint(
        self,
        means2d: Tensor,
        radii: Tensor,
        pixel_error: Tensor,
        height: int,
        width: int,
        conics: Tensor,
        opacities: Optional[Tensor],
        vis: Tensor,
    ) -> None:
        """v2: EMA-update from a conic-weighted aggregate over each footprint.

        Fully vectorized over the visible set V: a fixed ``(2R+1)**2`` offset
        window (R = ``footprint_radius_px``) is gathered per Gaussian; offsets
        beyond the per-Gaussian effective radius ``min(max(radii), R)`` or
        outside the image get zero weight (out-of-bounds gathers are clamped
        for indexing only). Per pixel offset ``d`` the weight is
        ``opacity * exp(-0.5 * d^T Sigma^-1 d)`` via the conic ``(a, b, c)``;
        the aggregate is ``sum(w * err) / sum(w)``. Note the per-Gaussian
        opacity therefore cancels inside the normalized ratio — it only gates
        the ``sum(w) < 1e-8`` skip. Gaussians with a non-finite or
        non-positive-definite conic fall back to the v1 center sample;
        Gaussians whose weights sum below 1e-8 are skipped (no EMA update).
        """
        conics = torch.as_tensor(conics)
        if conics.dim() == 3 and int(conics.shape[0]) == 1:
            conics = conics.squeeze(0)
        if conics.dim() != 2 or int(conics.shape[-1]) != 3:
            raise ValueError("conics must have shape [N, 3] (or [1, N, 3])")
        n = int(means2d.shape[0])
        if int(conics.shape[0]) != n:
            raise ValueError("conics count does not match means2d")
        if opacities is not None:
            opacities = torch.as_tensor(opacities).reshape(-1)
            if int(opacities.shape[0]) != n:
                raise ValueError("opacities count does not match means2d")

        err_map = pixel_error.detach().float()
        device = err_map.device
        radius_cap = int(getattr(self.config, "footprint_radius_px", 4))

        # Sub-pixel centers for the quadratic form; rounded centers anchor the
        # gather window and the degenerate-conic fallback (v1 semantics).
        centers = torch.nan_to_num(
            means2d[vis].detach().float(), nan=0.0, posinf=1e12, neginf=-1e12
        ).to(device)
        cx = centers[:, 0].round()
        cy = centers[:, 1].round()

        radii_vis = radii[vis].detach().float().to(device)
        if radii_vis.dim() == 2:
            radii_vis = radii_vis.max(dim=-1).values
        r_eff = radii_vis.clamp(min=0.0, max=float(radius_cap))  # [V]

        offsets = torch.arange(
            -radius_cap, radius_cap + 1, device=device, dtype=torch.float32
        )
        dy_grid, dx_grid = torch.meshgrid(offsets, offsets, indexing="ij")
        dx = dx_grid.reshape(-1)  # [K], K = (2R+1)**2
        dy = dy_grid.reshape(-1)

        px = cx[:, None] + dx[None, :]  # [V, K]
        py = cy[:, None] + dy[None, :]
        in_bounds = (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
        in_window = (dx.abs()[None, :] <= r_eff[:, None]) & (
            dy.abs()[None, :] <= r_eff[:, None]
        )
        px_idx = px.clamp(0, width - 1).long()
        py_idx = py.clamp(0, height - 1).long()
        err_win = err_map[py_idx, px_idx]  # [V, K]

        con = conics[vis].detach().float().to(device)
        a, b, c = con[:, 0], con[:, 1], con[:, 2]
        conic_ok = (
            torch.isfinite(con).all(dim=-1)
            & (a > 0)
            & (c > 0)
            & (a * c - b * b > 0)
        )

        ddx = px - centers[:, 0][:, None]
        ddy = py - centers[:, 1][:, None]
        quad = (
            a[:, None] * ddx * ddx
            + 2.0 * b[:, None] * ddx * ddy
            + c[:, None] * ddy * ddy
        )
        if opacities is None:
            opac = torch.ones(int(vis.numel()), device=device)
        else:
            opac = torch.nan_to_num(
                opacities[vis].detach().float().to(device), nan=0.0
            ).clamp(min=0.0)
        weights = opac[:, None] * torch.exp(-0.5 * quad)
        weights = torch.where(
            conic_ok[:, None] & in_bounds & in_window,
            weights,
            torch.zeros((), device=device),
        )
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weight_sum = weights.sum(dim=-1)
        aggregated = (weights * err_win).sum(dim=-1) / weight_sum.clamp_min(1e-12)

        center_err = err_map[
            cy.long().clamp(0, height - 1), cx.long().clamp(0, width - 1)
        ]
        err = torch.where(conic_ok, aggregated, center_err)
        # Degenerate conics always update via the center fallback; healthy
        # conics with an (effectively) empty footprint are skipped entirely.
        keep = (~conic_ok) | (weight_sum >= 1e-8)
        if not bool(keep.any()):
            return
        vis_upd = vis.to(device)[keep]
        err = err[keep]

        if self.scores.device != err.device:
            self.scores = self.scores.to(err.device)
        err = err.to(self.scores.dtype)
        vis_upd = vis_upd.to(self.scores.device)
        decay = float(self.config.ema_decay)
        self.scores[vis_upd] = decay * self.scores[vis_upd] + (1.0 - decay) * err

    @torch.no_grad()
    def resize(self, new_count: int) -> None:
        """Align the length to ``new_count``, preserving the surviving scores.

        Growth appends fresh entries (score 1.0), shrinking truncates the tail;
        every entry that stays keeps its accumulated EMA bit-for-bit.

        This used to reset *every* score to 1.0, which was a defect, not a
        simplification: MCMC densification runs every ``refine_every`` steps,
        so a multi-thousand-step multi-view error EMA was discarded on a fixed
        cadence and the sampler kept collapsing back to plain opacity MCMC.
        ``resize`` is now only the fallback for refinements whose parent/child
        mapping is not observable; when it is,
        :meth:`GaussianLifecycleState.on_grow` inherits the parent's score.
        """
        if int(new_count) < 0:
            raise ValueError("new_count must be non-negative")
        self.lifecycle.resize(int(new_count))

    @torch.no_grad()
    def sampling_weights(
        self, opacities: Tensor, admission: Optional[Tensor] = None
    ) -> Tensor:
        """Return ``opacity * clamped_score**power [* admission]`` per Gaussian.

        ``admission`` is the optional soft LiDAR surface-support factor from
        :meth:`~cloudstudio_3dgs.training.lidar_admission.LidarAdmission.admission_weights`,
        a ``[N]`` vector in ``[weight_floor, 1]``. It biases *where* newly
        densified Gaussians are born toward measured surfaces; because its floor
        is strictly positive it can never zero a candidate out, so this stays a
        preference and not a rejection.

        When ``admission`` is ``None`` the returned tensor is bit-for-bit the
        pre-WP-4 result — that is the regression guarantee every disabled run
        and every existing caller relies on.
        """
        opac = torch.as_tensor(opacities).flatten()
        if int(opac.shape[0]) != len(self):
            raise ValueError(
                f"opacities has {int(opac.shape[0])} entries but state holds {len(self)}"
            )
        score = self.scores.to(device=opac.device, dtype=opac.dtype)
        score = score.clamp_min(float(self.config.min_score_floor))
        weights = opac * score ** float(self.config.score_power)
        if admission is None:
            return weights
        factor = torch.as_tensor(admission).flatten()
        if int(factor.shape[0]) != len(self):
            raise ValueError(
                f"admission has {int(factor.shape[0])} entries but state holds {len(self)}"
            )
        return weights * factor.to(device=opac.device, dtype=opac.dtype)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return the complete state required for deterministic resume.

        ``schema_version`` stays 1 and ``scores`` stays the authoritative EMA
        column so checkpoints remain readable in both directions; ``lifecycle``
        is an additive payload that older readers ignore and that older
        checkpoints simply do not carry.
        """
        return {
            "schema_version": 1,
            "scores": self.scores.detach().clone(),
            "lifecycle": self.lifecycle.state_dict(),
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
        lifecycle_payload = payload.get("lifecycle")
        if lifecycle_payload is None:
            # Pre-lifecycle checkpoint: the scores are all that was ever
            # persisted, so the sibling columns start from their defaults at
            # the restored length.
            self.lifecycle = GaussianLifecycleState(
                int(expected_count), device=self.lifecycle.device
            )
        else:
            self.lifecycle.load_state_dict(
                lifecycle_payload, expected_count=int(expected_count)
            )
        target_dtype = self.lifecycle.error_ema.dtype
        self.scores = scores.detach().to(
            device=self.lifecycle.device,
            dtype=target_dtype,
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
) -> tuple[Tensor, Tensor]:
    """Inplace relocate dead Gaussians onto live ones sampled by ``probs``.

    Adapted from gsplat.strategy.ops.relocate; ``probs`` is a full-length
    [N] weight vector aligned with the Gaussians (e.g. from
    :meth:`ErrorScoreState.sampling_weights`). The alive subset is taken
    internally, exactly where the original takes ``opacities[alive_indices]``.

    Unlike the upstream op this returns ``(dead_indices, sampled_idxs)``: the
    dead/source pairing is computed here and is the only place it exists, and
    per-Gaussian lifecycle state cannot be kept consistent without it. Nothing
    else about the adapted logic changes; callers may ignore the return value.
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
    return dead_indices, sampled_idxs


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
    proposal: Optional["TangentProposal"] = None,
    proposal_out: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Inplace add ``n`` Gaussians cloned from existing ones sampled by ``probs``.

    Adapted from gsplat.strategy.ops.sample_add; ``probs`` is a full-length
    [N] weight vector aligned with the Gaussians.

    Unlike the upstream op this returns ``sampled_idxs``, the parent index of
    each appended Gaussian in pre-growth index space and in append order.
    Without it a caller cannot tell which existing Gaussian each new row was
    cloned from, and per-Gaussian state can only be reset. Nothing else about
    the adapted logic changes; callers may ignore the return value.

    Tangent-plane proposal (WP-5)
    -----------------------------
    With ``proposal`` supplied and active, the appended rows' ``means``,
    ``quats`` and ``scales`` are replaced by
    :meth:`~cloudstudio_3dgs.training.tangent_proposal.TangentProposal.propose`
    instead of being bit-exact copies of the parent. Two index-space facts make
    this delicate, and both are pinned by tests:

    * the child's *base* values are the ones upstream would have appended, which
      for ``scales`` and ``opacities`` are the **post-**``compute_relocation``
      values, not the parent's originals. Worse, ``p[sampled_idxs] = ...``
      followed by ``p[sampled_idxs]`` is a scatter-then-gather: when a parent is
      sampled twice, both children read back the *last* value written, not the
      two distinct entries of ``new_scales``. The child base scales are
      therefore reconstructed here with the same scatter-then-gather rather than
      assumed equal to ``new_scales``.
    * the override is computed *before* ``_update_param_with_optimizer``, from a
      snapshot. That function walks ``params`` in dict order, so anything
      computed inside one ``param_fn`` cannot be relied on by another.

    ``proposal_out``, when given, is filled with ``applied`` / ``anchor_index``
    / ``anchor_confidence`` for the appended rows so the caller can write the
    lifecycle anchor columns without recomputing the query.
    """
    opacities = torch.sigmoid(params["opacities"])

    eps = torch.finfo(torch.float32).eps
    probs = probs.flatten()
    if probs.shape[0] != opacities.flatten().shape[0]:
        raise ValueError("probs must be a full-length [N] weight vector")
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    additive_births = bool(
        proposal is not None
        and proposal.active
        and proposal.config.additive_births
    )
    if additive_births:
        # A tangent proposal moves children away from their sampled parents.
        # Upstream compute_relocation assumes the split lobes stay co-located;
        # shrinking the parent before moving the child removes coverage at the
        # original position.  Additive births therefore leave every existing
        # row bit-exact and initialize only the appended opacity.
        new_opacities = torch.full_like(
            opacities[sampled_idxs], float(proposal.config.birth_opacity)
        )
        new_scales = torch.exp(params["scales"])[sampled_idxs]
    else:
        new_opacities, new_scales = compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
            binoms=binoms,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    override: Dict[str, Tensor] = {}
    if proposal is not None and proposal.active and int(sampled_idxs.numel()) > 0:
        # Reproduce the child's base log-scales exactly as param_fn will derive
        # them, without touching the live parameter.
        child_scales = params["scales"].detach().clone()
        if not additive_births:
            child_scales[sampled_idxs] = torch.log(new_scales)
        payload = proposal.propose(
            params["means"].detach()[sampled_idxs],
            params["quats"].detach()[sampled_idxs],
            child_scales[sampled_idxs],
        )
        applied = payload["applied"]
        for name in ("means", "quats", "scales"):
            if name in payload:
                override[name] = payload[name]
        if proposal_out is not None:
            proposal_out.update(
                {
                    "applied": applied,
                    "anchor_index": payload["anchor_index"],
                    "anchor_confidence": payload["anchor_confidence"],
                    "applied_count": int(applied.sum().item()),
                    "stats": dict(proposal.last_stats),
                }
            )
    else:
        applied = None

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities" and not additive_births:
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales" and not additive_births:
            p[sampled_idxs] = torch.log(new_scales)
        children = p[sampled_idxs]
        if name == "opacities" and additive_births:
            children = torch.logit(new_opacities)
        replacement = override.get(name)
        if replacement is not None and applied is not None:
            # Broadcast the [M] mask over the trailing feature dims. Non-applied
            # rows keep the clone verbatim; the proposal already passes those
            # through, so this is a second, structural guarantee.
            mask = applied.reshape(-1, *([1] * (children.dim() - 1)))
            children = torch.where(
                mask, replacement.to(dtype=children.dtype, device=children.device), children
            )
        p_new = torch.cat([p, children])
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
    if proposal_out is not None:
        proposal_out["additive_births"] = additive_births
        proposal_out["birth_opacity"] = (
            float(proposal.config.birth_opacity) if additive_births else None
        )
    return sampled_idxs


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
    # Bookkeeping only: the refine hooks are called without the step number,
    # but lifecycle rows record the step at which they were (re)born.
    current_step: int = 0
    # Optional soft LiDAR surface-support weighting of the birth sites
    # (WP-4). Left as None the strategy is unchanged in every respect.
    admission_state: Optional["LidarAdmission"] = None
    # Optional tangent-plane birth-site proposal (WP-5). Decides *where the
    # newborn lands*, where admission decides *which parent it comes from*.
    proposal_state: Optional["TangentProposal"] = None

    def __post_init__(self) -> None:
        self.error_config.validate()

    def _weighting_active(self, num_gaussians: int) -> bool:
        return (
            self.error_config.enabled
            and self.score_state is not None
            and len(self.score_state) == int(num_gaussians)
        )

    def _admission_weights(self, opacities: Tensor) -> Optional[Tensor]:
        """Admission factor for the current cloud, or ``None`` to skip it.

        Any doubt — no admission state, disabled, stale after a refinement, or
        a length that does not match the live Gaussian count — degrades to pure
        error-weighted sampling. Falling back is always safe here because the
        admission factor is a preference, not a gate.
        """
        if self.admission_state is None:
            return None
        return self.admission_state.admission_weights(
            int(opacities.shape[0]),
            device=opacities.device,
            dtype=opacities.dtype,
        )

    def step_post_backward(self, *args: Any, **kwargs: Any):
        """Record the step, then run the unmodified upstream refinement."""
        step = kwargs.get("step")
        if step is None and len(args) >= 4:
            step = args[3]
        if step is not None:
            self.current_step = int(step)
        return super().step_post_backward(*args, **kwargs)

    @torch.no_grad()
    def _sync_lifecycle_length(self, num_gaussians: int) -> None:
        """Fallback alignment when the parent/child mapping is not observable."""
        if self.score_state is not None and len(self.score_state) != int(num_gaussians):
            self.score_state.resize(int(num_gaussians))

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
            probs = self.score_state.sampling_weights(
                opacities, admission=self._admission_weights(opacities)
            )
            outcome = relocate_weighted(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                probs=probs,
                min_opacity=self.min_opacity,
                scene=scene,
            )
            # Relocation keeps N constant, but each dead slot now holds a copy
            # of its source Gaussian: its lifecycle row must follow the source
            # instead of keeping the corpse's history.
            if isinstance(outcome, tuple) and len(outcome) == 2:
                dead_indices, sampled_idxs = outcome
                self.score_state.lifecycle.on_relocate(
                    dead_indices, sampled_idxs, step=self.current_step
                )
                if self.admission_state is not None:
                    # Event-driven maintenance: relocation keeps N constant, so
                    # only the dead slots' rows moved. Copying the source rows
                    # keeps the cache usable at the *next* refine instead of
                    # blanking it (WP-5; see LidarAdmission.on_relocate).
                    self.admission_state.on_relocate(
                        dead_indices,
                        sampled_idxs,
                        lifecycle=self.score_state.lifecycle,
                    )
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
            # Upstream sample_add keeps its parent indices private, so only a
            # length alignment is possible here; surviving rows are preserved.
            self._sync_lifecycle_length(len(params["means"]))
            return n_gs
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            assert self.score_state is not None
            opacities = torch.sigmoid(params["opacities"].flatten())
            probs = self.score_state.sampling_weights(
                opacities, admission=self._admission_weights(opacities)
            )
            proposal_out: Dict[str, Any] = {}
            sampled_idxs = sample_add_weighted(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                probs=probs,
                min_opacity=self.min_opacity,
                scene=scene,
                proposal=self.proposal_state,
                proposal_out=proposal_out,
            )
            # Children inherit their parent's accumulated error EMA and anchor;
            # nothing that already existed is touched.
            if isinstance(sampled_idxs, Tensor):
                new_rows = self.score_state.lifecycle.on_grow(
                    sampled_idxs, step=self.current_step
                )
                # Order matters: extending admission rewrites the whole anchor
                # columns from its own cache, so the per-birth anchors have to
                # land after it to stay authoritative.
                self._extend_admission(
                    current_n_points, params["means"], sampled_idxs
                )
                self._record_birth_anchors(new_rows, proposal_out)
            self._sync_lifecycle_length(len(params["means"]))
        return n_gs

    @torch.no_grad()
    def _record_birth_anchors(
        self, new_rows: Tensor, proposal_out: Dict[str, Any]
    ) -> None:
        """Write the proposal's surface anchors onto the freshly grown rows.

        ``on_grow`` inherits the parent's ``anchor_index``/``anchor_confidence``,
        which is right for a bit-exact clone and wrong the moment the child was
        moved: the child's nearest surface point is its own, measured at birth.
        These are the first real writers of those two lifecycle columns, which
        WP-1 reserved and nothing had populated per-birth until now.
        """
        anchor_index = proposal_out.get("anchor_index")
        if anchor_index is None or new_rows.numel() == 0:
            return
        assert self.score_state is not None
        lifecycle = self.score_state.lifecycle
        applied = proposal_out["applied"].to(lifecycle.device)
        rows = new_rows[applied]
        if rows.numel() == 0:
            return
        lifecycle.anchor_index[rows] = anchor_index.to(
            device=lifecycle.device, dtype=lifecycle.anchor_index.dtype
        )[applied]
        lifecycle.anchor_confidence[rows] = proposal_out["anchor_confidence"].to(
            device=lifecycle.device, dtype=lifecycle.anchor_confidence.dtype
        )[applied]

    @torch.no_grad()
    def _extend_admission(
        self, old_count: int, means: Tensor, sampled_idxs: Tensor
    ) -> None:
        """Append admission rows for the newborns instead of blanking the cache.

        Falls silently back to the conservative stale path whenever the cache
        was not already aligned with the pre-growth count: extending a vector
        that is out of sync would misalign every row after the join, which is
        strictly worse than having no weights at all.
        """
        admission = self.admission_state
        if admission is None or not admission.in_sync(old_count):
            return
        assert self.score_state is not None
        if self.proposal_state is not None and self.proposal_state.active:
            # Children were moved, so their support has to be measured at their
            # own positions rather than inherited from the parents.
            admission.extend(
                new_means=means[old_count:],
                lifecycle=self.score_state.lifecycle,
            )
        else:
            # A plain clone sits exactly on its parent: copying the parent row
            # is the same query, for free.
            admission.extend(
                parent_indices=sampled_idxs,
                lifecycle=self.score_state.lifecycle,
            )
