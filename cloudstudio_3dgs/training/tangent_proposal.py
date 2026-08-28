# SPDX-License-Identifier: Apache-2.0
"""Tangent-plane proposal: give a newborn Gaussian a *place* and a *pose*.

The gap this closes
-------------------
MCMC densification decides **which** Gaussian to clone from image error, and
:mod:`cloudstudio_3dgs.training.lidar_admission` (WP-4) biases **which parent**
is eligible. Neither of them has any opinion on **where the child lands**:
``sample_add`` concatenates ``p[sampled_idxs]``, so the newborn is a bit-exact
copy of its parent — same mean, same rotation, same shape. Everything after
that is left to MCMC noise, which random-walks the child away from the parent
with no knowledge of the measured surface. Two failure modes follow directly:

* a parent glued to a wall spawns a child at the same point, and the noise is
  isotropic, so half the exploration budget is spent pushing the child *out of*
  the wall along the normal — the one direction where the geometry is already
  known to be correct;
* a parent that is itself a view-dependent floater spawns floaters. Cloning is
  a hereditary process and nothing in the loop ever questions the inheritance.

This module proposes a child position *inside the measured surface's tangent
plane* and, optionally, initializes the child's rotation so its **shortest axis
is the surface normal**. The empirical justification for doing both at once is
the WP-4 A/B series: constraining Gaussian *orientation* was free (the normal
prior improved SSIM over baseline), while constraining Gaussian *position* cost
perceptual quality (-0.022 LPIPS on the attribution arm, -0.024 L1 on the
admission arm). A position constraint that simultaneously hands the child the
correct pose does not need a later regularizer to rotate it back, so the two
effects are expected to partially cancel rather than add.

Fallback is the design, not an escape hatch
-------------------------------------------
A candidate is only projected where the surface is actually trustworthy
(``planarity >= planarity_gate`` **and** surface support above
``support_gate``). Everywhere else the proposal returns the parent's own values
untouched and reports ``mask_applied=False``. Forcing a projection onto a
low-planarity neighborhood would snap Gaussians onto a normal that is an
artifact of vegetation, clutter or a corner — worse than not moving them, and
exactly the mistake a hard rule makes that a soft one does not. The same
reasoning applies to the LiDAR cloud's blind spots (glass, occlusion shadows,
out-of-range structure): those regions have no surface to propose onto, so the
child must stay where cloning put it and let image error do its job.

Pure functions, numpy only
--------------------------
:func:`propose_positions` takes arrays and a config and returns arrays. It
holds no training state, touches no torch and no CUDA, which is what makes the
geometry testable on CPU. :class:`TangentProposal` is the thin torch bridge the
MCMC ops actually hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from cloudstudio_3dgs.geometry.lidar_surface_field import (
    DEFAULT_MIN_SIGMA_M,
    LidarSurfaceField,
    support_weight,
)

_EPS = 1e-12

PROPOSAL_MODES = ("off", "tangent")


# ---------------------------------------------------------------------------
# Config (torch-free, like error_weighted_config / lidar_admission)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalConfig:
    """Tangent-plane birth-site proposal settings.

    Disabled by default: with ``enabled=False`` (or ``mode="off"``)
    ``sample_add_weighted`` never calls into this module and densification is
    bit-for-bit the existing clone-the-parent behaviour.

    Attributes:
        enabled: master switch.
        mode: ``"tangent"`` projects the candidate into the local tangent
            plane; ``"off"`` is a no-op kept so an A/B arm can be expressed
            without touching ``enabled``.
        planarity_gate: minimum neighborhood planarity for the proposal to fire.
            Below it the local normal is not a surface normal but a numerical
            accident, so the candidate falls back to plain cloning.
        support_gate: minimum
            :func:`~cloudstudio_3dgs.geometry.lidar_surface_field.support_weight`
            for the proposal to fire. Planarity alone is not enough: a parent
            floating ten metres in front of a perfectly planar wall would
            otherwise be teleported onto it, which is a *relocation*, not a
            proposal, and destroys whatever the parent was representing.
        support_tangent_factor: maximum tangential distance from the nearest
            measured LiDAR sample, expressed in local-spacing units.  This
            prevents a point on the unbounded extension of a fitted plane from
            being mistaken for support inside the actually scanned patch.
        sigma_perp_factor: forwarded to ``support_weight`` when evaluating
            ``support_gate``. Matches :class:`AdmissionConfig` so both gates are
            calibrated on the same number.
        tangent_sigma_factor: tangential scatter, in units of
            ``local_spacing``. The child is drawn from an isotropic 2-D
            Gaussian in the tangent plane with
            ``sigma = tangent_sigma_factor * local_spacing``. Scaling by the
            *local* sampling scale is the point: it spreads children exactly as
            far as the scanner can resolve geometry there, and no further.
        normal_offset_factor: cap on the retained perpendicular offset, in units
            of ``local_spacing``. The child keeps the *sign* of its parent's
            offset (which side of the surface it was on) but only a bounded
            amount of its magnitude. A hard snap to ``d_perp = 0`` would be a
            stronger position constraint than the A/B evidence supports.
        init_shortest_axis: also initialize the child's quaternion so its
            shortest axis is the surface normal, and its shortest log-scale is
            the surface thickness.
        thickness_factor: shortest-axis scale, in units of the local surface
            thickness ``min(local_spacing, max(roughness, min_thickness_m))``.
        min_thickness_m: absolute floor (metres) on the shortest axis. A
            synthetic plane has ``roughness == 0`` exactly, and ``log(0)`` is
            not a scale; more practically a sub-millimetre Gaussian is a needle
            the rasterizer cannot integrate stably. Where the surface has no
            measurable roughness the floor therefore wins outright and
            ``thickness_factor`` has nothing to scale.
        additive_births: preserve every parent Gaussian when a proposed child
            is moved to a different surface position.  The upstream MCMC
            split conserves volume only while parent and child remain
            co-located; applying that split before a tangent-plane move removes
            opacity from the old position and creates visible holes.
        birth_opacity: initial opacity of an additive child.  It must remain
            above MCMC's dead-Gaussian threshold and low enough that the new
            surface sample can be fitted without abruptly over-covering the
            image.
        reject_unsupported_births: fail-closed LiDAR mode for classic
            split/clone and MCMC relocation/add. Candidate sources must pass
            the same planarity/support gates before they may create or replace
            geometry, and every proposal is placed from that measured source
            on the local tangent
            surface.  This is intentionally stronger than the visual-only MCMC
            fallback and must be enabled only when LiDAR is the authority
            geometry.
    """

    enabled: bool = False
    mode: str = "tangent"
    planarity_gate: float = 0.6
    support_gate: float = 0.1
    support_tangent_factor: float = 3.0
    sigma_perp_factor: float = 1.0
    tangent_sigma_factor: float = 0.5
    normal_offset_factor: float = 0.1
    init_shortest_axis: bool = True
    thickness_factor: float = 0.5
    min_thickness_m: float = 1e-3
    additive_births: bool = False
    birth_opacity: float = 0.05
    reject_unsupported_births: bool = False

    def validate(self) -> None:
        if self.mode not in PROPOSAL_MODES:
            raise ValueError(f"mode must be one of {list(PROPOSAL_MODES)}")
        if not 0.0 <= float(self.planarity_gate) <= 1.0:
            raise ValueError("planarity_gate must be within [0, 1]")
        # A zero support gate would let the proposal teleport an arbitrarily
        # distant floater onto the nearest plane; see the attribute docstring.
        if not 0.0 < float(self.support_gate) <= 1.0:
            raise ValueError("support_gate must be within (0, 1]")
        if not float(self.support_tangent_factor) > 0.0:
            raise ValueError("support_tangent_factor must be positive")
        if not float(self.sigma_perp_factor) > 0.0:
            raise ValueError("sigma_perp_factor must be positive")
        if float(self.tangent_sigma_factor) < 0.0:
            raise ValueError("tangent_sigma_factor must be non-negative")
        if float(self.normal_offset_factor) < 0.0:
            raise ValueError("normal_offset_factor must be non-negative")
        if not float(self.thickness_factor) > 0.0:
            raise ValueError("thickness_factor must be positive")
        if not float(self.min_thickness_m) > 0.0:
            raise ValueError("min_thickness_m must be positive")
        if not 0.005 < float(self.birth_opacity) < 1.0:
            raise ValueError("birth_opacity must be within (0.005, 1)")

    @property
    def active(self) -> bool:
        """True when this config is allowed to move a newborn at all."""
        return bool(self.enabled) and self.mode == "tangent"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "planarity_gate": self.planarity_gate,
            "support_gate": self.support_gate,
            "support_tangent_factor": self.support_tangent_factor,
            "sigma_perp_factor": self.sigma_perp_factor,
            "tangent_sigma_factor": self.tangent_sigma_factor,
            "normal_offset_factor": self.normal_offset_factor,
            "init_shortest_axis": self.init_shortest_axis,
            "thickness_factor": self.thickness_factor,
            "min_thickness_m": self.min_thickness_m,
            "additive_births": self.additive_births,
            "birth_opacity": self.birth_opacity,
            "reject_unsupported_births": self.reject_unsupported_births,
        }


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def rotmat_to_quat_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Batched rotation matrix ``[M, 3, 3]`` -> unit quaternion ``[M, 4]`` (wxyz).

    ``wxyz`` is gsplat's convention (``gsplat.utils.normalized_quat_to_rotmat``),
    and that function reads the matrix in the column-vector convention: column
    ``j`` of ``R`` is the world direction of the Gaussian's local axis ``j``,
    whose extent is ``scales[j]``. Aligning "the shortest axis" therefore means
    putting the surface normal in **column 2** and the thickness in
    ``scales[2]``.

    Shepperd's method: pick whichever of the four branches has the largest
    denominator so the division never amplifies rounding. The naive
    trace-only formula degenerates at 180-degree rotations, which are not rare
    here — half of all surface normals point "away" under the field's arbitrary
    sign convention.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("matrices must have shape [M, 3, 3]")
    m = matrices
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    trace = m00 + m11 + m22

    # Four candidate parameterizations; each is exact where its own scalar
    # (trace, or one diagonal dominance) is the largest.
    def _branch(scalar: np.ndarray, builder) -> tuple[np.ndarray, np.ndarray]:
        s = np.sqrt(np.maximum(scalar, _EPS)) * 2.0
        return s, np.stack(builder(s), axis=1)

    s_w, q_w = _branch(
        trace + 1.0,
        lambda s: (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s),
    )
    s_x, q_x = _branch(
        1.0 + m00 - m11 - m22,
        lambda s: ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s),
    )
    s_y, q_y = _branch(
        1.0 + m11 - m00 - m22,
        lambda s: ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s),
    )
    s_z, q_z = _branch(
        1.0 + m22 - m00 - m11,
        lambda s: ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s),
    )

    denominators = np.stack([s_w, s_x, s_y, s_z], axis=1)
    candidates = np.stack([q_w, q_x, q_y, q_z], axis=1)  # [M, 4 branches, 4]
    best = np.argmax(denominators, axis=1)
    quats = candidates[np.arange(len(m)), best]
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(norm, _EPS)
    # Canonical hemisphere (w >= 0): q and -q are the same rotation, and a
    # stable sign makes the values comparable across runs and in fixtures.
    flip = quats[:, 0] < 0.0
    quats[flip] *= -1.0
    return quats


def surface_frame_matrices(
    tangent_basis: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Proper rotations ``[M, 3, 3]`` with columns ``[t0, t1, n]``.

    The third column is recomputed as ``cross(t0, t1)`` rather than taken from
    ``normal``. Two reasons, and both matter:

    * ``det`` must be ``+1``. The surface field sign-normalizes its normal by a
      deterministic but physically meaningless convention, so ``[t0, t1, n]``
      is a reflection for roughly half the points, and a reflection has no
      quaternion at all.
    * ``cross(t0, t1) == +/- normal`` exactly (the three come from the same
      orthonormal eigenbasis), so choosing the cross product changes nothing
      about the *axis* — only about the handedness. Consumers of a Gaussian
      pose are symmetric under ``n -> -n`` because the covariance is quadratic
      in the axis, which is precisely why this substitution is free.

    ``normal`` is accepted only to validate the alignment assumption.
    """
    tangent_basis = np.asarray(tangent_basis, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if tangent_basis.ndim != 3 or tangent_basis.shape[1:] != (3, 2):
        raise ValueError("tangent_basis must have shape [M, 3, 2]")
    if normal.shape != (len(tangent_basis), 3):
        raise ValueError("normal must have shape [M, 3]")
    t0 = _normalize(tangent_basis[:, :, 0])
    t1 = _normalize(tangent_basis[:, :, 1])
    # Re-orthogonalize t1 against t0 before the cross product: the eigenbasis is
    # orthonormal in exact arithmetic, but float32 storage is not.
    t1 = _normalize(t1 - t0 * np.einsum("mi,mi->m", t0, t1)[:, None])
    third = _normalize(np.cross(t0, t1))
    return np.stack([t0, t1, third], axis=2)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, _EPS)


# ---------------------------------------------------------------------------
# The proposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalResult:
    """Per-candidate proposal outputs; see :func:`propose_positions`."""

    means: np.ndarray
    quats: Optional[np.ndarray]
    log_scales: Optional[np.ndarray]
    applied: np.ndarray
    anchor_index: np.ndarray
    anchor_confidence: np.ndarray
    support: np.ndarray
    child_support: np.ndarray
    fallback_to_parent: np.ndarray


def propose_positions(
    parent_means: np.ndarray,
    surface_field: LidarSurfaceField,
    config: ProposalConfig,
    generator: Optional[np.random.Generator] = None,
    *,
    parent_quats: Optional[np.ndarray] = None,
    parent_log_scales: Optional[np.ndarray] = None,
) -> ProposalResult:
    """Propose a birth site (and pose) for each cloned candidate.

    Position
    --------
    For a candidate whose gates pass::

        sigma  = tangent_sigma_factor * local_spacing
        cap    = normal_offset_factor * local_spacing
        offset = clip(signed_d_perp, -cap, +cap)
        mean   = surface_point + tangent_basis @ (sigma * randn(2)) + normal * offset

    ``signed_d_perp`` rather than fresh noise along the normal: the parent's own
    side of the surface is information (inside vs outside a wall, above vs below
    a floor) and throwing it away would put children on the wrong face of thin
    structure. It is clipped because the *magnitude* of that offset is exactly
    what we do not trust. Note that ``surface_point + normal * signed_d_perp``
    reconstructs the parent's projection onto the normal line, so the whole
    expression is invariant to the field's arbitrary normal sign.

    Pose
    ----
    When ``init_shortest_axis`` the child's rotation is
    :func:`surface_frame_matrices` converted by :func:`rotmat_to_quat_wxyz`, so
    local axes ``(0, 1, 2)`` map to ``(t0, t1, n)``. The two tangential
    log-scales are the parent's **two largest** log-scales in descending order
    and the third is the surface thickness::

        thickness = clip(thickness_factor * min(local_spacing, max(roughness, min_thickness_m)),
                         min_thickness_m,
                         exp(smaller tangential log-scale))

    Keeping the parent's two largest extents (rather than deriving both from
    ``local_spacing``) preserves the spatial footprint the parent earned from
    image error — the proposal is meant to fix *orientation and place*, not to
    reset the capacity the error signal already allocated. Re-sorting is
    necessary because the child's frame is redefined: the parent's axis order
    refers to the parent's own rotation and carries no meaning in the surface
    frame. The final clamp is the invariant that makes the name true: axis 2
    is never larger than the axes it is supposed to be shorter than.

    Fallback
    --------
    Rows failing ``planarity_gate`` or ``support_gate`` pass the parent's values
    through unchanged and report ``applied=False``. Random numbers are drawn for
    **every** row regardless of the gates, so the RNG stream — and therefore
    every accepted row — is independent of how many candidates were rejected.

    Args:
        parent_means: ``[M, 3]`` positions of the parents being cloned.
        surface_field: the shared :class:`LidarSurfaceField`.
        config: :class:`ProposalConfig`; ``mode="off"``/``enabled=False`` still
            evaluates to a full pass-through with ``applied`` all False.
        generator: ``numpy`` generator for the tangential scatter. Defaults to a
            fresh unseeded generator; pass a seeded one for reproducibility.
        parent_quats: ``[M, 4]`` wxyz quaternions, used only to fill the
            pass-through rows of the returned ``quats``.
        parent_log_scales: ``[M, 3]`` log-scales. Required to return
            ``log_scales`` at all; without it the shortest axis cannot be set
            without also inventing the two tangential ones.

    Returns:
        :class:`ProposalResult`. ``quats``/``log_scales`` are ``None`` when
        ``init_shortest_axis`` is False (or the corresponding parent array was
        not supplied); otherwise every row is populated, but only rows with
        ``applied`` were actually derived from the surface.
    """
    config.validate()
    means = np.ascontiguousarray(np.asarray(parent_means, dtype=np.float64))
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("parent_means must have shape [M, 3]")
    count = len(means)
    if generator is None:
        generator = np.random.default_rng()

    if count == 0:
        pose = bool(config.init_shortest_axis)
        return ProposalResult(
            means=means,
            quats=(
                np.asarray(parent_quats, np.float64)
                if pose and parent_quats is not None
                else None
            ),
            log_scales=(
                np.asarray(parent_log_scales, np.float64)
                if pose and parent_log_scales is not None
                else None
            ),
            applied=np.zeros(0, dtype=bool),
            anchor_index=np.zeros(0, dtype=np.int64),
            anchor_confidence=np.zeros(0, dtype=np.float64),
            support=np.zeros(0, dtype=np.float64),
            child_support=np.zeros(0, dtype=np.float64),
            fallback_to_parent=np.zeros(0, dtype=bool),
        )

    query = surface_field.query(means, k=1)
    support = support_weight(
        query, sigma_perp_factor=float(config.sigma_perp_factor)
    )
    applied = _trusted_surface_mask(query, support, config)
    if not config.active:
        applied = np.zeros(count, dtype=bool)

    spacing = np.asarray(query.local_spacing, dtype=np.float64)
    # Draw for every row so the stream does not depend on the gates.
    scatter = generator.standard_normal((count, 2))
    sigma = float(config.tangent_sigma_factor) * spacing
    tangential = np.einsum(
        "mij,mj->mi", np.asarray(query.tangent_basis, dtype=np.float64), scatter * sigma[:, None]
    )
    cap = float(config.normal_offset_factor) * spacing
    offset = np.clip(np.asarray(query.signed_d_perp, dtype=np.float64), -cap, cap)
    proposed = (
        np.asarray(query.surface_point, dtype=np.float64)
        + tangential
        + np.asarray(query.normal, dtype=np.float64) * offset[:, None]
    )
    child_support = np.ones(count, dtype=np.float64)
    fallback_to_parent = np.zeros(count, dtype=bool)
    if config.reject_unsupported_births:
        # A supported parent is necessary but not sufficient: Gaussian scatter
        # is unbounded, so even a small sigma has a non-zero chance of landing
        # beyond the measured patch.  Re-query the actual proposed position and
        # fail closed to the already accepted parent when the child leaves the
        # trusted surface.  This keeps classic split/clone additive capacity on
        # LiDAR without inventing geometry in a scan hole.
        child_query = surface_field.query(proposed, k=1)
        child_support = support_weight(
            child_query, sigma_perp_factor=float(config.sigma_perp_factor)
        )
        child_ok = _trusted_surface_mask(child_query, child_support, config)
        fallback_to_parent = applied & ~child_ok
        proposed = np.where(fallback_to_parent[:, None], means, proposed)
    new_means = np.where(applied[:, None], proposed, means)

    new_quats = None
    new_log_scales = None
    if config.init_shortest_axis:
        if parent_quats is not None:
            frames = surface_frame_matrices(query.tangent_basis, query.normal)
            surface_quats = rotmat_to_quat_wxyz(frames)
            parent_q = np.asarray(parent_quats, dtype=np.float64)
            if parent_q.shape != (count, 4):
                raise ValueError("parent_quats must have shape [M, 4]")
            new_quats = np.where(applied[:, None], surface_quats, parent_q)
        if parent_log_scales is not None:
            parent_s = np.asarray(parent_log_scales, dtype=np.float64)
            if parent_s.shape != (count, 3):
                raise ValueError("parent_log_scales must have shape [M, 3]")
            new_log_scales = np.where(
                applied[:, None],
                _surface_log_scales(parent_s, query, config),
                parent_s,
            )

    return ProposalResult(
        means=new_means,
        quats=new_quats,
        log_scales=new_log_scales,
        applied=applied,
        anchor_index=np.asarray(query.index, dtype=np.int64),
        anchor_confidence=np.asarray(query.confidence, dtype=np.float64),
        support=support,
        child_support=np.asarray(child_support, dtype=np.float64),
        fallback_to_parent=fallback_to_parent,
    )


def _surface_log_scales(
    parent_log_scales: np.ndarray, query: Any, config: ProposalConfig
) -> np.ndarray:
    """``[M, 3]`` log-scales in the surface frame: two tangential, one thickness."""
    descending = np.sort(parent_log_scales, axis=1)[:, ::-1]
    tangential = descending[:, :2]
    spacing = np.asarray(query.local_spacing, dtype=np.float64)
    roughness = np.asarray(query.roughness, dtype=np.float64)
    floor = float(config.min_thickness_m)
    # The surface's own thickness is its roughness, floored so a perfectly flat
    # synthetic plane (roughness == 0 exactly) still has a scale, and capped by
    # ``local_spacing`` because no neighborhood resolves structure finer than
    # its own sampling. On such a degenerate plane the factor has nothing left
    # to scale and the result sits at the floor - which is correct, not a bug.
    thickness = float(config.thickness_factor) * np.minimum(
        spacing, np.maximum(roughness, floor)
    )
    # max() first, then min(): the shortest-axis invariant must survive the
    # floor, so a tangential axis smaller than ``min_thickness_m`` still wins.
    thickness = np.minimum(np.maximum(thickness, floor), np.exp(tangential[:, 1]))
    return np.concatenate(
        [tangential, np.log(np.maximum(thickness, _EPS))[:, None]], axis=1
    )


def _trusted_surface_mask(
    query: Any, support: np.ndarray, config: ProposalConfig
) -> np.ndarray:
    """Measured-patch support, not merely distance to an infinite local plane."""

    spacing = np.maximum(
        np.asarray(query.local_spacing, dtype=np.float64), DEFAULT_MIN_SIGMA_M
    )
    tangent_limit = float(config.support_tangent_factor) * spacing
    return (
        np.asarray(query.planarity, dtype=np.float64)
        >= float(config.planarity_gate)
    ) & (np.asarray(support, dtype=np.float64) >= float(config.support_gate)) & (
        np.asarray(query.d_tangent, dtype=np.float64) <= tangent_limit
    )


# ---------------------------------------------------------------------------
# Torch bridge held by the MCMC ops
# ---------------------------------------------------------------------------


class TangentProposal:
    """Stateless-per-call torch wrapper around :func:`propose_positions`.

    The MCMC ops work in torch on whatever device the parameters live on; the
    surface field is a CPU cKDTree. This class owns exactly that boundary — the
    round trip through numpy, and the one seeded generator that makes a run
    reproducible — and nothing else. All geometry lives in the pure function.
    """

    def __init__(
        self,
        field: LidarSurfaceField,
        config: ProposalConfig,
        seed: int | None = None,
    ) -> None:
        config.validate()
        self.field = field
        self.config = config
        self.generator = np.random.default_rng(seed)
        self.propose_count = 0
        self.applied_total = 0
        self.candidate_total = 0
        self.last_stats: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return self.config.active

    def propose(
        self,
        parent_means: Any,
        parent_quats: Any | None = None,
        parent_log_scales: Any | None = None,
    ) -> dict[str, Any]:
        """Propose for ``[M, ...]`` torch parent tensors; returns torch tensors.

        The returned dict carries ``means``/``quats``/``scales`` already cast to
        the source tensors' dtype and device, an ``applied`` bool mask, and the
        surface anchors for the lifecycle columns. Keys whose parent tensor was
        not supplied are simply absent, so a caller can iterate the dict.
        """
        torch = __import__("torch")
        device = parent_means.device
        dtype = parent_means.dtype
        result = propose_positions(
            parent_means.detach().cpu().to(torch.float64).numpy(),
            self.field,
            self.config,
            self.generator,
            parent_quats=(
                None
                if parent_quats is None
                else parent_quats.detach().cpu().to(torch.float64).numpy()
            ),
            parent_log_scales=(
                None
                if parent_log_scales is None
                else parent_log_scales.detach().cpu().to(torch.float64).numpy()
            ),
        )

        def _to(array: np.ndarray, target_dtype: Any) -> Any:
            return torch.from_numpy(np.ascontiguousarray(array)).to(
                device=device, dtype=target_dtype
            )

        payload: dict[str, Any] = {
            "means": _to(result.means, dtype),
            "applied": _to(result.applied, torch.bool),
            "anchor_index": _to(result.anchor_index, torch.int64),
            "anchor_confidence": _to(result.anchor_confidence, torch.float32),
        }
        if result.quats is not None:
            payload["quats"] = _to(result.quats, parent_quats.dtype)
        if result.log_scales is not None:
            payload["scales"] = _to(result.log_scales, parent_log_scales.dtype)

        self.propose_count += 1
        self.candidate_total += int(result.applied.size)
        self.applied_total += int(np.count_nonzero(result.applied))
        self.last_stats = _summarize(result)
        return payload

    def eligible_parent_mask(self, means: Any) -> Any:
        """Return the hard LiDAR growth gate for classic split/clone parents.

        The MCMC path deliberately keeps a visual-only fallback.  A LiDAR-first
        classic lifecycle has a different invariant: unsupported Gaussians may
        still render, but they may not become parents of additional geometry.
        Querying the exact parent positions here also makes the later newborn
        proposal fail-closed without depending on the random split offset.
        """
        torch = __import__("torch")
        points = means.detach().cpu().to(torch.float64).numpy()
        query = self.field.query(points, k=1)
        support = support_weight(
            query, sigma_perp_factor=float(self.config.sigma_perp_factor)
        )
        eligible = _trusted_surface_mask(query, support, self.config)
        return torch.from_numpy(np.ascontiguousarray(eligible)).to(
            device=means.device, dtype=torch.bool
        )


def _summarize(result: ProposalResult) -> dict[str, Any]:
    """Auditable summary of one proposal batch.

    ``applied_fraction`` is the number to watch: near 0 means the gates never
    fire (the field and the cloud are probably not in the same frame, or the
    scene is genuinely non-planar) and the arm is a no-op dressed as a change;
    near 1 on a cluttered scene means the gates are too loose.
    """
    total = int(result.applied.size)
    if total == 0:
        return {"count": 0, "applied": 0, "applied_fraction": 0.0}
    applied = int(np.count_nonzero(result.applied))
    fallback = int(np.count_nonzero(result.fallback_to_parent))
    return {
        "count": total,
        "applied": applied,
        "applied_fraction": applied / total,
        "fallback_to_parent": fallback,
        "fallback_fraction": fallback / total,
        "support_mean": float(np.mean(result.support)),
        "child_support_mean": float(np.mean(result.child_support)),
    }


def update_proposal_telemetry(
    telemetry: dict[str, Any], stats: dict[str, Any]
) -> None:
    """Fold one proposal batch's stats into the MCMC telemetry payload."""
    if not stats:
        return
    bucket = telemetry.get("tangent_proposal")
    if bucket is None:
        bucket = {"batches": 0, "candidates": 0, "applied": 0}
        telemetry["tangent_proposal"] = bucket
    bucket["batches"] = int(bucket["batches"]) + 1
    bucket["candidates"] = int(bucket["candidates"]) + int(stats["count"])
    bucket["applied"] = int(bucket["applied"]) + int(stats["applied"])
    total = max(int(bucket["candidates"]), 1)
    bucket["applied_fraction"] = bucket["applied"] / total
    bucket["last"] = dict(stats)
