# SPDX-License-Identifier: Apache-2.0
"""Soft LiDAR admission: bias *where new Gaussians are born* toward measured surfaces.

The problem
-----------
MCMC densification answers "which Gaussian should be cloned?" from image error,
and it answers it well. It has no opinion at all on *where the clone should
live*: ``sample_add``/``relocate`` copy the parent's position, so the newborn
inherits whatever place the parent had drifted to. Over a long run that is a
one-way ratchet away from the measured geometry — the delivered UK model ended
with 934 Gaussians more than 0.3 m off the LiDAR surface along its own normal
and a further 3599 with no trustworthy surface support at all.

Regularization (``lidar_normals``) and checkpoint selection both act *after* a
bad Gaussian exists. This module acts at the only place where the decision is
actually made: the multinomial sampling weight that picks birth sites. Each
weight is multiplied by a per-Gaussian *admission* factor in
``[weight_floor, 1]`` derived from the LiDAR surface field, so densification
concentrates where there is real measured support and merely thins out — never
stops — where there is not.

Scope (WP-4)
------------
**Soft weighting only.** No candidate is ever rejected and no Gaussian is ever
moved: ``mode`` accepts ``"off"`` and ``"soft"``, and ``"hard"`` is explicitly
refused. Hard admission and projection-onto-surface belong to WP-5, where the
staleness window documented on :meth:`LidarAdmission.on_count_changed` has to be
closed first — a hard reject computed against stale anchors would delete real
candidates, whereas a soft weight computed against stale anchors merely
mis-prioritizes them.

Why not a ``d_perp`` threshold
------------------------------
``d_perp`` is the distance to the *unbounded* plane fitted at the nearest LiDAR
sample. A Gaussian floating several metres past the end of a wall projects back
onto that plane's extension and scores ``d_perp ~ 0``. The admission weight
therefore combines two independent quantities:

* :func:`~cloudstudio_3dgs.geometry.lidar_surface_field.support_weight`, which
  folds ``d_perp`` against a sampling-adaptive sigma *and* multiplies by the
  neighborhood ``confidence`` (planarity x neighbor support); and
* a **tangential patch gate** on ``d_tangent``, which is what tells us whether
  the query even projects inside the region LiDAR actually measured.

See :meth:`LidarAdmission.refresh` for the exact formula.
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

ADMISSION_MODES = ("off", "soft")


# ---------------------------------------------------------------------------
# Config (torch-free, like error_weighted_config)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionConfig:
    """Soft LiDAR admission settings for MCMC densification.

    Disabled by default: with ``enabled=False`` (or ``mode="off"``) the sampler
    receives no admission tensor at all and densification is bit-for-bit the
    existing error-weighted behaviour.

    Attributes:
        enabled: master switch.
        mode: ``"soft"`` multiplies the sampling weight; ``"off"`` is a
            no-op kept so an A/B arm can be expressed without touching
            ``enabled``. ``"hard"`` is reserved for WP-5 and rejected here.
        sigma_perp_factor: forwarded to
            :func:`~cloudstudio_3dgs.geometry.lidar_surface_field.support_weight`;
            scales the perpendicular tolerance
            ``sigma = factor * max(local_spacing, roughness)``. Larger is more
            permissive.
        weight_floor: lower clamp of the admission factor, in ``(0, 1]``. It
            must be **strictly positive** — see the class docstring of
            :class:`LidarAdmission` and :meth:`validate`.
        refresh_every: step cadence for re-querying the surface field when the
            Gaussian count has *not* changed (the means still drift every step).
            A count change marks the cache stale immediately, independently of
            this cadence.
        gate_tangent_factor: how many ``local_spacing`` units of tangential
            drift are treated as "still inside the scanned patch" before the
            extrapolation penalty starts. Must be positive.
        share_normal_field: when ``lidar_normal_alignment`` is also enabled,
            derive its :class:`NormalField` from this module's surface field
            instead of running a second KNN-PCA pass.

            **This is not behaviour-neutral.** ``build_surface_field`` defaults
            to ``knn=24`` while ``build_normal_field`` defaults to ``knn=16``,
            so sharing changes the normals and planarity the alignment loss
            sees (larger KNN = smoother normals, more smoothing across edges).
            It saves one 1.2-2.2 s pass per 390k points, which is negligible
            against a multi-hour run, so set this to ``False`` for any A/B in
            which the normal prior must stay bit-identical to its solo arm.
    """

    enabled: bool = False
    mode: str = "soft"
    sigma_perp_factor: float = 1.0
    weight_floor: float = 0.05
    refresh_every: int = 500
    gate_tangent_factor: float = 3.0
    share_normal_field: bool = True

    def validate(self) -> None:
        if self.mode == "hard":
            raise ValueError(
                'admission mode "hard" is reserved for WP-5; WP-4 implements '
                "soft weighting only"
            )
        if self.mode not in ADMISSION_MODES:
            raise ValueError(f"mode must be one of {list(ADMISSION_MODES)}")
        if not float(self.sigma_perp_factor) > 0.0:
            raise ValueError("sigma_perp_factor must be positive")
        # Zero would turn the soft weight into a hard reject: an admission of
        # exactly 0 makes a location unreachable by densification forever, and
        # an all-zero probs vector is not a valid multinomial distribution.
        if not 0.0 < float(self.weight_floor) <= 1.0:
            raise ValueError("weight_floor must be within (0, 1]")
        if int(self.refresh_every) < 1:
            raise ValueError("refresh_every must be at least one")
        if not float(self.gate_tangent_factor) > 0.0:
            raise ValueError("gate_tangent_factor must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "sigma_perp_factor": self.sigma_perp_factor,
            "weight_floor": self.weight_floor,
            "refresh_every": self.refresh_every,
            "gate_tangent_factor": self.gate_tangent_factor,
            "share_normal_field": self.share_normal_field,
        }


# ---------------------------------------------------------------------------
# Admission state
# ---------------------------------------------------------------------------


class LidarAdmission:
    """Per-Gaussian LiDAR admission weights for the densification sampler.

    ``refresh`` queries the surface field once for every Gaussian mean (CPU;
    CUDA means are detached and copied) and caches a ``[N]`` weight vector.
    ``admission_weights`` hands that vector to
    :meth:`~cloudstudio_3dgs.training.error_weighted_mcmc.ErrorScoreState.sampling_weights`,
    or returns ``None`` whenever the cache cannot be trusted — in which case the
    caller falls back to pure error-weighted sampling rather than to a guess.

    The floor is the safety property
    --------------------------------
    The weight is clamped into ``[weight_floor, 1]`` and ``weight_floor`` is
    required to be strictly positive. This is not a numerical nicety, it is the
    whole reason this work package is "soft": **the LiDAR cloud is not a
    complete model of the scene.** Glass and polished surfaces return nothing,
    the scanner has occlusion shadows behind every object, and anything past its
    range (facades across the street, sky-adjacent structure) is simply absent.
    Those regions contain real geometry that only the cameras can see. A floor
    of zero would make them permanently unreachable by densification, so the
    only way to represent them would be for existing Gaussians to drift there —
    exactly the failure this module exists to stop. A positive floor keeps a
    visual-only birth channel open at ``weight_floor`` of the nominal rate
    (0.05 = a 20:1 preference for measured surfaces) and lets image error still
    populate what LiDAR could not see.
    """

    def __init__(self, field: LidarSurfaceField, config: AdmissionConfig) -> None:
        config.validate()
        self.field = field
        self.config = config
        self.weights: Any = None  # torch [N] float32 on CPU
        self.anchor_index: Any = None  # torch [N] int64 on CPU
        self.anchor_confidence: Any = None  # torch [N] float32 on CPU
        self.stale = True
        self.refresh_count = 0
        self.last_stats: dict[str, Any] = {}
        self._device_cache: dict[Any, Any] = {}

    def __len__(self) -> int:
        return 0 if self.weights is None else int(self.weights.shape[0])

    @property
    def active(self) -> bool:
        """True when this instance is allowed to influence sampling at all."""
        return bool(self.config.enabled) and self.config.mode == "soft"

    # -- refresh ------------------------------------------------------------

    def refresh(self, means: Any, lifecycle: Any | None = None) -> dict[str, Any]:
        """Re-query the surface field for every Gaussian and cache the weights.

        Formula
        -------
        ::

            base        = support_weight(query, sigma_perp_factor)
                        = exp(-(d_perp / sigma)^2) * confidence
            patch_limit = gate_tangent_factor * max(local_spacing, min_sigma)
            excess      = max(d_tangent / patch_limit - 1, 0)
            gate        = exp(-excess^2)
            admission   = clip(base * gate, weight_floor, 1)

        ``base`` is deliberately blind to tangential drift: that is precisely
        what makes a Gaussian glued to a wall but landing halfway between two
        scan lines score ~1 instead of being punished for the scanner's sampling
        pattern. The price of that blindness is that ``d_perp`` is measured
        against an *unbounded* plane, which keeps extending through open space
        past the edge of whatever the scanner actually hit.

        ``gate`` buys the blindness back only where it is dangerous. Inside the
        measured patch ``d_tangent`` is at most about one ``local_spacing`` (that
        is the definition of ``local_spacing``), so with the default factor of 3
        the scan-gap regime sits far inside the gate and ``excess`` is exactly
        zero — the gap case is untouched, bit for bit. Beyond a few spacings the
        query has projected outside the scanned patch and the plane there is
        pure extrapolation, so support decays as a Gaussian in the *excess*
        ratio. Gaussian rather than a hard cut because densification is a
        sampler: a discontinuity in the weight field would be carved into the
        cloud as a visible geometric boundary.

        Anchors
        -------
        ``anchor_index`` / ``anchor_confidence`` are cached and, when
        ``lifecycle`` is supplied and its length matches, written into the
        matching :class:`~cloudstudio_3dgs.training.gaussian_lifecycle.GaussianLifecycleState`
        columns. Those columns already survive grow/relocate/prune, so WP-5 can
        follow an anchor through a refinement instead of invalidating the cache;
        nothing in WP-4 reads them back.

        Returns the statistics dict also stored on ``last_stats``.
        """
        torch = __import__("torch")
        points = np.ascontiguousarray(
            means.detach().cpu().to(torch.float64).numpy()
        )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("means must have shape [N, 3]")

        count = len(points)
        if count == 0:
            self.weights = torch.zeros(0, dtype=torch.float32)
            self.anchor_index = torch.zeros(0, dtype=torch.int64)
            self.anchor_confidence = torch.zeros(0, dtype=torch.float32)
            self.stale = False
            self.refresh_count += 1
            self._device_cache.clear()
            self.last_stats = _empty_stats()
            return self.last_stats

        query = self.field.query(points, k=1)
        base = support_weight(
            query, sigma_perp_factor=float(self.config.sigma_perp_factor)
        )
        gate = self._tangent_gate(query)
        floor = float(self.config.weight_floor)
        admission = np.clip(base * gate, floor, 1.0)

        self.weights = torch.from_numpy(
            np.ascontiguousarray(admission, dtype=np.float32)
        )
        self.anchor_index = torch.from_numpy(
            np.ascontiguousarray(query.index, dtype=np.int64)
        )
        self.anchor_confidence = torch.from_numpy(
            np.ascontiguousarray(query.confidence, dtype=np.float32)
        )
        self.stale = False
        self.refresh_count += 1
        self._device_cache.clear()

        if lifecycle is not None and int(len(lifecycle)) == count:
            device = lifecycle.anchor_index.device
            lifecycle.anchor_index = self.anchor_index.to(device)
            lifecycle.anchor_confidence = self.anchor_confidence.to(device)

        self.last_stats = _summarize(admission, base, gate, floor)
        return self.last_stats

    def _tangent_gate(self, query: Any) -> np.ndarray:
        """Extrapolation penalty for queries projecting outside the scanned patch."""
        spacing = np.asarray(query.local_spacing, dtype=np.float64)
        patch_limit = float(self.config.gate_tangent_factor) * np.maximum(
            spacing, DEFAULT_MIN_SIGMA_M
        )
        excess = np.maximum(
            np.asarray(query.d_tangent, dtype=np.float64) / patch_limit - 1.0, 0.0
        )
        return np.exp(-np.square(excess))

    # -- consumption --------------------------------------------------------

    def admission_weights(
        self,
        count: int | None = None,
        *,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> Optional[Any]:
        """Cached ``[N]`` weights, or ``None`` when they must not be used.

        ``None`` is returned when the module is disabled or ``mode="off"``, when
        the cache is stale, when no refresh has happened yet, or when ``count``
        is given and does not match the cached length. Every one of those is a
        "fall back to pure error-weighted sampling" signal, never a reason to
        substitute a default: silently sampling against a mis-aligned weight
        vector would attribute one Gaussian's surface support to another.
        """
        if not self.active or self.stale or self.weights is None:
            return None
        if count is not None and int(count) != int(self.weights.shape[0]):
            return None
        if device is None and dtype is None:
            return self.weights
        key = (device, dtype)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = self.weights.to(
                **({} if device is None else {"device": device}),
                **({} if dtype is None else {"dtype": dtype}),
            )
            self._device_cache[key] = cached
        return cached

    def on_count_changed(self, new_count: int | None = None) -> None:
        """Mark the cache stale because the Gaussian index space moved.

        Known limitation (WP-4 accepts it, WP-5 must close it)
        ------------------------------------------------------
        This is the same conservative policy
        :class:`~cloudstudio_3dgs.training.lidar_normals.LidarNormalAnchors`
        uses, and it has the same window: from the moment a refinement changes
        the count until the next :meth:`refresh`, admission is simply switched
        off and densification falls back to pure error weighting. With the usual
        cadence that window is the single step following each refine, so the
        refine *at* which the count changed already sampled against valid
        weights and only a subsequent refine inside the window would be
        unweighted — but the guarantee is cadence-dependent, not structural.

        It is also conservative in the wrong direction for a *hard* rule: a
        stale hard reject would delete real candidates, while a stale soft
        weight only mis-prioritizes them. That asymmetry is why WP-5 has to
        replace this with event-driven maintenance (``on_grow`` /
        ``on_relocate`` / ``on_prune`` carrying the anchors, which the lifecycle
        columns already support) before hard admission is safe to enable.
        """
        self.stale = True
        self._device_cache.clear()
        if new_count is not None and self.weights is not None:
            if int(new_count) != int(self.weights.shape[0]):
                # Drop the cache outright rather than keep a mis-sized tensor
                # around where a later length check might not run.
                self.weights = None
                self.anchor_index = None
                self.anchor_confidence = None


# ---------------------------------------------------------------------------
# Statistics / telemetry
# ---------------------------------------------------------------------------


def _empty_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "mean": 0.0,
        "min": 0.0,
        "max": 0.0,
        "at_floor_fraction": 0.0,
        "gated_fraction": 0.0,
    }


def _summarize(
    admission: np.ndarray, base: np.ndarray, gate: np.ndarray, floor: float
) -> dict[str, Any]:
    """Auditable summary of one refresh.

    ``at_floor_fraction`` is the share of Gaussians whose birth-site preference
    has collapsed to the floor — i.e. the share currently sustained only by the
    visual-only channel. It is the number to watch across a run: it should fall
    if admission is doing its job, and a value near 1 means the field and the
    cloud are not in the same frame.
    """
    total = int(admission.size)
    # Compare against the floor with a small tolerance: the clip produces
    # exact equality, but base*gate can also land just above it.
    at_floor = float(np.count_nonzero(admission <= floor * (1.0 + 1e-6))) / total
    return {
        "count": total,
        "mean": float(np.mean(admission)),
        "min": float(np.min(admission)),
        "max": float(np.max(admission)),
        "at_floor_fraction": at_floor,
        "gated_fraction": float(np.count_nonzero(gate < 1.0)) / total,
        "base_mean": float(np.mean(base)),
    }


def update_admission_telemetry(
    telemetry: dict[str, Any], stats: dict[str, Any]
) -> None:
    """Fold one refresh's stats into the MCMC telemetry payload.

    Only running aggregates plus the latest snapshot are kept: the per-refine
    event list is already the detailed record, and a full history here would
    grow with the run for no audit value.
    """
    if not stats:
        return
    bucket = telemetry.get("admission")
    if bucket is None:
        bucket = {"refresh_count": 0, "mean_sum": 0.0}
        telemetry["admission"] = bucket
    bucket["refresh_count"] = int(bucket["refresh_count"]) + 1
    bucket["mean_sum"] = float(bucket["mean_sum"]) + float(stats["mean"])
    bucket["mean_admission"] = bucket["mean_sum"] / bucket["refresh_count"]
    bucket["last"] = dict(stats)


# ---------------------------------------------------------------------------
# Field sharing
# ---------------------------------------------------------------------------


def normal_field_from_surface_field(field: LidarSurfaceField) -> Any:
    """Adapt a :class:`LidarSurfaceField` to the ``NormalField`` query interface.

    ``build_surface_field`` is a strict superset of ``build_normal_field``: it
    computes the same KNN-PCA with the same normal sign convention and the
    byte-for-byte same planarity definition, plus the tangent basis, roughness
    and spacing this module needs. Reusing it avoids a second KNN-PCA pass and,
    more importantly, a second cKDTree over the same cloud.

    ``NormalField`` is instantiated through ``__new__`` (the same pattern
    ``build_normal_field`` itself uses) so the shared arrays and tree are
    adopted rather than copied.

    Caveat: ``field.knn`` carries over. A surface field built at the default
    ``knn=24`` therefore gives the alignment loss smoother normals than the
    ``knn=16`` it would have computed alone — see
    :attr:`AdmissionConfig.share_normal_field`.
    """
    from cloudstudio_3dgs.training.lidar_normals import NormalField

    adapted = NormalField.__new__(NormalField)
    adapted.xyz = field.xyz
    adapted.normals = field.normals
    adapted.planarity = field.planarity
    adapted.knn = int(field.knn)
    adapted.tree = field.tree
    return adapted
