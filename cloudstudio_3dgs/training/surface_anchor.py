# SPDX-License-Identifier: Apache-2.0
"""Surface-anchor prune: geometry the trusted LiDAR surface does not support dies.

What was measured (2026-09-11, house0305 Tile_1, ``tile1_R1d_20k`` at 20k
steps, initialization cloud of 3.42M LiDAR points): 34% of all gaussians lie
more than 0.2 m from the nearest initialization point, 27% more than 0.5 m and
20% more than 1 m, and those far gaussians sit high (z p50 2.44 m against
0.31 m for the near ones). They are the floaters and eave/sky protrusions the
viewer shows. 35% of the population is outside the Tile's training box. The
per-view backdrop is a sky dome only, so the Tile grows gaussians to paint
sky and trees it has no geometry for, and nothing in the lifecycle ever asks
whether a gaussian is anywhere near a measured surface.

This module adds exactly that question, in three places the classic
lifecycle already has:

* **cull events** - every gaussian farther than ``max_distance_m`` from the
  nearest initialization point (and, with ``outside_box: prune``, every
  gaussian outside the Tile's ``training_and_export_box`` grown by the same
  margin) is removed together with the opacity/size culls, in the same
  ``remove`` call, so optimizer state and lineage stay aligned exactly as
  they do for the existing culls;
* an optional **extra cadence** (``every``) for steps that are not cull
  events, e.g. after ``refine_stop_iter`` where the recovered lifecycle
  otherwise does nothing;
* **births** - with ``reject_unsupported_parents`` a growth candidate that is
  itself unsupported may not clone or split. This is a pure parent mask (no
  newborn repositioning), so unlike the tangent-plane birth guard it has no
  dependency on the lifecycle execution order.

Newborns younger than ``min_age_steps`` are exempt: a clone lands on its
parent, and the split offset is a fraction of the parent's scale, so a child
born this event is exactly as far from the surface as its parent - it needs
the optimizer's next window before its own distance says anything.

Distance semantics
------------------
"Distance" is the exact Euclidean distance to the nearest point of the
initialization cloud as signed in the Tile inputs (before any subsampling
stride), in the metric S1 frame. It is computed with a ``cKDTree`` on the
CPU with ``distance_upper_bound`` - the same mechanism
:class:`~cloudstudio_3dgs.training.lidar_normals.LidarNormalAnchors.refresh`
already uses once per refine event on the whole population, so the cost
class is already paid by every run with normal alignment enabled. The
numpy path *is* the implementation; the torch bridge only moves tensors.
Measured on this machine (16 threads): ``query(15M points, k=1,
distance_upper_bound=0.3)`` against 3.42M anchors is seconds, not minutes
(see ``research/quality_recovery_v2/10_surface_anchor_prune.md`` for the
numbers). A GPU voxel-hash twin was considered and rejected for now:
without padding it needs a variable-length gather kernel, and with padding
a 0.3 m cell over a 1 cm LiDAR surface holds ~900 points.

What it deliberately does not do
--------------------------------
It never touches opacity, never moves a gaussian, and never adds one. A
gaussian that is near the surface but wrong (transparent, oversized) is left
to the existing culls; one that is far but genuinely needed (thin structure
without LiDAR returns, glass, a moving object the scanner caught mid-motion)
is a known risk of this rule and is why the knob ships disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SURFACE_ANCHOR_OUTSIDE_BOX_MODES = ("keep", "prune")

# Thresholds the audit tool tabulates; 0.2/0.5/1.0 are the measurement above.
AUDIT_DISTANCE_THRESHOLDS_M = (0.05, 0.1, 0.2, 0.5, 1.0)

# Birth step the lineage tracker assigns to initialization rows.
_INIT_BIRTH_STEP = -1


@dataclass(frozen=True)
class SurfaceAnchorPruneConfig:
    """Knobs of the surface-anchor prune. Disabled by default: byte-identical.

    Attributes:
        enabled: master switch. Off, nothing below is read and no contract
            key is emitted.
        max_distance_m: a gaussian whose nearest initialization point is
            farther than this (metres, S1 frame) is unsupported.
        start_step: first training step at which the prune may act. Meant to
            sit after the first opacity reset so the rule never competes with
            the warm-up growth.
        every: extra cadence in steps (``None`` = only at cull events). When
            set, steps that are multiples of it also prune, whether or not
            they are cull events; this is how the rule keeps acting after
            ``refine_stop_iter``.
        min_age_steps: rows born fewer than this many steps ago are exempt.
        outside_box: ``"prune"`` also removes rows outside the Tile's
            ``training_and_export_box`` expanded by ``max_distance_m``;
            ``"keep"`` ignores the box.
        reject_unsupported_parents: growth candidates farther than
            ``max_distance_m`` may not clone or split.
    """

    enabled: bool = False
    max_distance_m: float = 0.3
    start_step: int = 0
    every: int | None = None
    min_age_steps: int = 0
    outside_box: str = "keep"
    reject_unsupported_parents: bool = False
    # ``cull: False`` keeps only the growth gate (reject_unsupported_parents)
    # and never removes rows: the growth-only arm the survey ranks above hard
    # pruning (research/quality_recovery_v2/11_floater_handling_survey.zh-CN.md 7.3).
    cull: bool = True

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("surface_anchor_prune.enabled must be a boolean")
        if not isinstance(self.cull, bool):
            raise ValueError("surface_anchor_prune.cull must be a boolean")
        if self.enabled and not self.cull and not self.reject_unsupported_parents:
            raise ValueError(
                "surface_anchor_prune with cull=false needs reject_unsupported_parents=true, "
                "otherwise the knob does nothing"
            )
        if not isinstance(self.reject_unsupported_parents, bool):
            raise ValueError(
                "surface_anchor_prune.reject_unsupported_parents must be a boolean"
            )
        if not (float(self.max_distance_m) > 0.0) or not np.isfinite(
            float(self.max_distance_m)
        ):
            raise ValueError("surface_anchor_prune.max_distance_m must be positive")
        if isinstance(self.start_step, bool) or int(self.start_step) < 0:
            raise ValueError("surface_anchor_prune.start_step must be non-negative")
        if self.every is not None and (
            isinstance(self.every, bool) or int(self.every) <= 0
        ):
            raise ValueError("surface_anchor_prune.every must be a positive step count")
        if isinstance(self.min_age_steps, bool) or int(self.min_age_steps) < 0:
            raise ValueError("surface_anchor_prune.min_age_steps must be non-negative")
        if self.outside_box not in SURFACE_ANCHOR_OUTSIDE_BOX_MODES:
            raise ValueError(
                "surface_anchor_prune.outside_box must be one of "
                + ", ".join(SURFACE_ANCHOR_OUTSIDE_BOX_MODES)
            )

    def to_dict(self) -> dict[str, Any]:
        """Contract form: emitted by the trainer only when enabled."""
        return {
            "enabled": bool(self.enabled),
            "max_distance_m": float(self.max_distance_m),
            "start_step": int(self.start_step),
            "every": None if self.every is None else int(self.every),
            "min_age_steps": int(self.min_age_steps),
            "outside_box": str(self.outside_box),
            "reject_unsupported_parents": bool(self.reject_unsupported_parents),
            "cull": bool(self.cull),
            "distance": "exact_nearest_initialization_point_euclidean_m",
            "box_margin_m": float(self.max_distance_m),
        }


# ---------------------------------------------------------------------------
# Pure numpy core (the implementation; the audit tool and tests call this)
# ---------------------------------------------------------------------------


def build_anchor_tree(anchors: np.ndarray) -> Any:
    """cKDTree over the initialization cloud (float64, [M, 3])."""
    from scipy.spatial import cKDTree

    points = np.ascontiguousarray(np.asarray(anchors, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("anchors must have shape [M, 3] with M > 0")
    return cKDTree(points)


def nearest_surface_distance(
    points: np.ndarray,
    tree: Any,
    *,
    max_distance_m: float | None = None,
    workers: int = -1,
) -> np.ndarray:
    """Exact distance from every point to its nearest anchor, float64 [N].

    With ``max_distance_m`` the search stops at that bound and every point
    beyond it reads ``inf`` - which is all the prune needs to know and is
    what keeps the query cheap on a 15M population. Without it the full
    distance is returned (audit tables).
    """
    query = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if query.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    bound = np.inf if max_distance_m is None else float(max_distance_m)
    distance, _ = tree.query(query, k=1, distance_upper_bound=bound, workers=workers)
    return np.asarray(distance, dtype=np.float64)


def brute_force_nearest_distance(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """O(N*M) oracle for tests; never call it on a real population."""
    query = np.asarray(points, dtype=np.float64)
    anchor = np.asarray(anchors, dtype=np.float64)
    if query.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    delta = query[:, None, :] - anchor[None, :, :]
    return np.sqrt((delta * delta).sum(axis=-1)).min(axis=1)


def outside_box_mask(points: np.ndarray, box: Any, margin_m: float = 0.0) -> np.ndarray:
    """True where a point lies outside ``box`` grown by ``margin_m`` per face.

    ``box`` is the Tile-inputs form ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``.
    """
    query = np.asarray(points, dtype=np.float64)
    lower = np.asarray(box[0], dtype=np.float64) - float(margin_m)
    upper = np.asarray(box[1], dtype=np.float64) + float(margin_m)
    if lower.shape != (3,) or upper.shape != (3,) or bool(np.any(upper < lower)):
        raise ValueError("box must be [[xmin, ymin, zmin], [xmax, ymax, zmax]]")
    if query.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    return np.any((query < lower[None, :]) | (query > upper[None, :]), axis=1)


def _quantiles(values: np.ndarray, levels=(0.05, 0.5, 0.95)) -> dict[str, float | None]:
    if values.size == 0:
        return {f"p{int(round(level * 100)):02d}": None for level in levels}
    q = np.quantile(values.astype(np.float64), levels)
    return {
        f"p{int(round(level * 100)):02d}": float(value)
        for level, value in zip(levels, q)
    }


def audit_far_fraction(
    means: np.ndarray,
    tree: Any,
    *,
    opacity: np.ndarray | None = None,
    opacity_floor: float = 0.05,
    thresholds_m=AUDIT_DISTANCE_THRESHOLDS_M,
    box: Any | None = None,
    z_split_m: float = 0.2,
    workers: int = -1,
) -> dict[str, Any]:
    """The measurement table: far fraction per threshold, z profile, box share.

    Same numbers whether the caller is the audit tool on a checkpoint or a
    test on a synthetic one; the prune reads the same distance.
    """
    points = np.asarray(means, dtype=np.float64)
    count = int(points.shape[0])
    distance = nearest_surface_distance(points, tree, workers=workers)
    visible = None
    if opacity is not None:
        visible = np.asarray(opacity, dtype=np.float64) >= float(opacity_floor)
    rows = []
    for threshold in thresholds_m:
        far = distance > float(threshold)
        row = {
            "threshold_m": float(threshold),
            "far_count": int(far.sum()),
            "far_fraction": float(far.mean()) if count else None,
        }
        if visible is not None:
            visible_count = int(visible.sum())
            row["far_fraction_visible"] = (
                float((far & visible).sum() / visible_count) if visible_count else None
            )
        rows.append(row)
    near = distance <= float(z_split_m)
    z = points[:, 2]
    report: dict[str, Any] = {
        "gaussian_count": count,
        "anchor_count": int(tree.n),
        "opacity_floor": float(opacity_floor),
        "visible_count": None if visible is None else int(visible.sum()),
        "distance_quantiles_m": _quantiles(distance, (0.5, 0.9, 0.95, 0.99)),
        "far_fraction_table": rows,
        "z_split_m": float(z_split_m),
        "z_quantiles_near_m": _quantiles(z[near]),
        "z_quantiles_far_m": _quantiles(z[~near]),
    }
    if box is not None:
        outside = outside_box_mask(points, box)
        report["outside_box_count"] = int(outside.sum())
        report["outside_box_fraction"] = float(outside.mean()) if count else None
        report["outside_box_and_far_fraction"] = (
            float((outside & ~near).mean()) if count else None
        )
    return report


# ---------------------------------------------------------------------------
# Torch bridge held by the classic lifecycle adapter
# ---------------------------------------------------------------------------


class SurfaceAnchorPrune:
    """Config + anchor tree + Tile box; the adapter asks it three questions.

    ``due``/``extra_due`` say whether a step acts, ``masks`` says which rows
    are unsupported, ``unsupported_parent_mask`` gates births. All geometry
    goes through the numpy core above; this class only owns the device
    round trip and the running telemetry.
    """

    def __init__(
        self,
        config: SurfaceAnchorPruneConfig,
        anchors: np.ndarray,
        *,
        box: Any | None = None,
        workers: int = -1,
    ) -> None:
        config.validate()
        if not config.enabled:
            raise ValueError("SurfaceAnchorPrune needs an enabled config")
        if config.outside_box == "prune" and box is None:
            raise ValueError(
                "surface_anchor_prune.outside_box='prune' needs the Tile training box"
            )
        self.config = config
        self.tree = build_anchor_tree(anchors)
        self.box = None if box is None else [list(map(float, box[0])), list(map(float, box[1]))]
        self.workers = int(workers)
        self.event_count = 0
        self.pruned_total = 0
        self.last_stats: dict[str, Any] = {}

    # -- scheduling ---------------------------------------------------------

    def due(self, step: int) -> bool:
        """May the prune act at a cull event on this step?"""
        if not self.config.cull:
            return False
        return int(step) >= int(self.config.start_step)

    def extra_due(self, step: int) -> bool:
        """Does the extra cadence fire on this step (independent of culls)?"""
        every = self.config.every
        return (
            self.config.cull
            and every is not None
            and int(step) >= int(self.config.start_step)
            and int(step) % int(every) == 0
        )

    # -- geometry -------------------------------------------------------------

    def _to_numpy(self, means: Any) -> np.ndarray:
        torch = __import__("torch")
        return means.detach().cpu().to(torch.float64).numpy()

    def _to_torch(self, mask: np.ndarray, like: Any) -> Any:
        torch = __import__("torch")
        return torch.from_numpy(np.ascontiguousarray(mask)).to(
            device=like.device, dtype=torch.bool
        )

    def masks(self, means: Any) -> tuple[Any, Any]:
        """``(far, outside)`` bool tensors on the means' device."""
        points = self._to_numpy(means)
        distance = nearest_surface_distance(
            points,
            self.tree,
            max_distance_m=float(self.config.max_distance_m),
            workers=self.workers,
        )
        far = ~np.isfinite(distance)
        if self.box is not None and self.config.outside_box == "prune":
            outside = outside_box_mask(
                points, self.box, margin_m=float(self.config.max_distance_m)
            )
        else:
            outside = np.zeros(points.shape[0], dtype=bool)
        return self._to_torch(far, means), self._to_torch(outside, means)

    def unsupported_parent_mask(self, means: Any) -> Any:
        """True for rows that may not become parents (``far`` only, no box)."""
        points = self._to_numpy(means)
        distance = nearest_surface_distance(
            points,
            self.tree,
            max_distance_m=float(self.config.max_distance_m),
            workers=self.workers,
        )
        return self._to_torch(~np.isfinite(distance), means)

    # -- one event ----------------------------------------------------------

    def removal_mask(
        self,
        means: Any,
        birth_step: Any | None,
        *,
        step: int,
        already_removed: Any | None = None,
    ) -> Any:
        """Rows this event removes, with telemetry in ``last_stats``.

        ``already_removed`` is the opacity/size cull mask of the same event,
        so the counts attribute each row to one reason only and the caller
        can OR the result into a single ``remove``.
        """
        torch = __import__("torch")
        far, outside = self.masks(means)
        count = int(far.numel())
        candidates = far | outside
        if birth_step is not None and int(birth_step.numel()) != count:
            # Lineage not yet aligned with this population (a direct call
            # before _ensure_lineage): no age is known, so no row is young.
            birth_step = None
        if birth_step is not None and int(self.config.min_age_steps) > 0:
            age = int(step) - birth_step.to(torch.int64)
            young = age < int(self.config.min_age_steps)
            eligible = candidates & ~young
            protected_young = int((candidates & young).sum().item())
        else:
            eligible = candidates
            protected_young = 0
        if already_removed is not None:
            new_removal = eligible & ~already_removed
        else:
            new_removal = eligible
        pruned_far = int((new_removal & far).sum().item())
        pruned_outside = int((new_removal & outside & ~far).sum().item())
        pruned = int(new_removal.sum().item())
        self.event_count += 1
        self.pruned_total += pruned
        self.last_stats = {
            "step": int(step),
            "max_distance_m": float(self.config.max_distance_m),
            "population": count,
            "candidates": int(candidates.sum().item()),
            "far_count": int(far.sum().item()),
            "far_fraction": float(far.sum().item() / count) if count else 0.0,
            "outside_count": int(outside.sum().item()),
            "protected_young": protected_young,
            "already_removed_overlap": (
                0
                if already_removed is None
                else int((eligible & already_removed).sum().item())
            ),
            "pruned_far": pruned_far,
            "pruned_outside": pruned_outside,
            "pruned": pruned,
            "remaining": count - pruned - (
                0 if already_removed is None else int(already_removed.sum().item())
            ),
            "pruned_total": int(self.pruned_total),
        }
        return new_removal

    def state_dict(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "anchor_count": int(self.tree.n),
            "box": self.box,
        }
