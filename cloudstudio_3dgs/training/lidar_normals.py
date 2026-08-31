"""LiDAR normal / planarity priors constraining Gaussian pose and thickness.

The LiDAR initialization cloud (metric, same frame as the Gaussian means) is a
strong geometric prior: on walls, floors and ceilings it tells us both the
local surface normal and how planar the neighborhood is. This module turns
that prior into two differentiable penalties on the Gaussian parameters:

* ``align``  — the shortest principal axis of each anchored Gaussian should be
  parallel to the LiDAR surface normal (fixes flakes lying "edge-on" in the
  wall, needle splats poking out of surfaces, and floating shingles).
* ``flatten`` — on highly planar LiDAR regions the shortest axis should not
  exceed a target metric thickness (fixes thick/bumpy walls). Thinner than the
  target is never rewarded, only excess thickness is penalized.

LiDAR normals are *unoriented* (no reliable viewpoint orientation is
available), so every loss here is symmetric under ``n -> -n``.

Anchoring is refreshed periodically (nearest LiDAR point per Gaussian) rather
than every step; between refreshes the cached anchors are reused. If the
Gaussian count changes between refreshes (MCMC relocation / densification)
the loss degrades to zero and flags itself stale until the next refresh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Normal field (offline, CPU / numpy)
# ---------------------------------------------------------------------------


class NormalField:
    """Nearest-neighbor queryable field of LiDAR normals and planarity.

    Holds per-point unit normals (unoriented; sign is a deterministic but
    physically meaningless convention), a planarity confidence in ``[0, 1]``
    and a cKDTree over the LiDAR positions for nearest-point queries.
    """

    def __init__(
        self,
        xyz: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        *,
        knn: int = 16,
    ) -> None:
        xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float64))
        normals = np.ascontiguousarray(np.asarray(normals, dtype=np.float32))
        planarity = np.ascontiguousarray(np.asarray(planarity, dtype=np.float32))
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N, 3]")
        if normals.shape != xyz.shape:
            raise ValueError("normals must match xyz shape")
        if planarity.shape != (len(xyz),):
            raise ValueError("planarity must have shape [N]")
        self.xyz = xyz
        self.normals = normals
        self.planarity = planarity
        self.knn = int(knn)
        from scipy.spatial import cKDTree

        self.tree = cKDTree(self.xyz)

    def __len__(self) -> int:
        return len(self.xyz)

    def query(
        self, points: np.ndarray, k: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (distance, normal, planarity, index) of nearest LiDAR points.

        ``points`` is ``[P, 3]``. For ``k == 1`` the outputs are ``[P]`` /
        ``[P, 3]`` / ``[P]`` / ``[P]``; for ``k > 1`` a k axis is inserted
        after the point axis.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [P, 3]")
        if k < 1:
            raise ValueError("k must be at least one")
        distance, index = self.tree.query(points, k=k, workers=-1)
        index = np.atleast_1d(index)
        return (
            np.asarray(distance, dtype=np.float64),
            self.normals[index],
            self.planarity[index],
            np.asarray(index, dtype=np.int64),
        )

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the field as compressed npz (tree is rebuilt on load)."""
        np.savez_compressed(
            Path(path),
            xyz=self.xyz,
            normals=self.normals,
            planarity=self.planarity,
            knn=np.int64(self.knn),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NormalField":
        with np.load(Path(path)) as data:
            return cls(
                data["xyz"],
                data["normals"],
                data["planarity"],
                knn=int(data["knn"]),
            )


def build_normal_field(
    lidar_xyz: np.ndarray,
    *,
    knn: int = 16,
    batch_size: int = 50_000,
) -> NormalField:
    """KNN-PCA normals and planarity for every LiDAR point (CPU, one-off).

    For each point the covariance of its ``knn`` nearest neighbors is
    eigen-decomposed (eigenvalues ascending, ``l0 <= l1 <= l2``):

    * normal   = eigenvector of the smallest eigenvalue ``l0`` (unoriented;
      the stored sign is a deterministic convention only — consumers must be
      symmetric under ``n -> -n``).
    * planarity = surface variation complement::

          planarity = 1 - 3 * l0 / (l0 + l1 + l2)

      Range ``[0, 1]``: an exact plane has ``l0 = 0`` so planarity ``= 1``;
      a fully isotropic (volumetric) neighborhood has ``l0 = l1 = l2`` so
      planarity ``= 0``. Degenerate neighborhoods (all eigenvalues ~ 0,
      e.g. duplicated points) get planarity ``0`` — no constraint.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(lidar_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("lidar_xyz must have shape [N, 3]")
    if len(points) < 3:
        raise ValueError("at least three points are required for KNN-PCA")
    if knn < 3:
        raise ValueError("knn must be at least three")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    k = min(int(knn), len(points))
    tree = cKDTree(points)
    normals = np.empty((len(points), 3), dtype=np.float32)
    planarity = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        _, indexes = tree.query(points[start:stop], k=k, workers=-1)
        neighborhoods = points[indexes]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
        values, vectors = np.linalg.eigh(cov)  # ascending eigenvalues
        normal = vectors[:, :, 0]
        # Deterministic (but physically meaningless) sign convention.
        major_axis = np.argmax(np.abs(normal), axis=1)
        signs = np.sign(normal[np.arange(len(normal)), major_axis])
        signs[signs == 0] = 1
        normal *= signs[:, None]
        trace = values.sum(axis=1)
        plan = 1.0 - 3.0 * values[:, 0] / np.maximum(trace, _EPS)
        plan[trace <= _EPS] = 0.0
        normals[start:stop] = normal.astype(np.float32)
        planarity[start:stop] = np.clip(plan, 0.0, 1.0).astype(np.float32)
    field = NormalField.__new__(NormalField)
    field.xyz = np.ascontiguousarray(points)
    field.normals = normals
    field.planarity = planarity
    field.knn = k
    field.tree = tree
    return field


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalAlignmentConfig:
    """LiDAR normal alignment / flattening regularization settings.

    Disabled by default; when enabled it only touches Gaussians that sit on
    highly planar LiDAR neighborhoods (``planarity >= planarity_gate``) and
    close to the LiDAR surface (``distance <= max_anchor_distance_m``).
    """

    enabled: bool = False
    weight_align: float = 0.01
    weight_flatten: float = 0.01
    weight_tangent_isotropy: float = 0.0
    weight_point_to_plane: float = 0.0
    planarity_gate: float = 0.6
    max_anchor_distance_m: float = 0.10
    refresh_every: int = 500
    flatten_mode: str = "absolute_m"
    flatten_target_m: float = 0.02
    flatten_ratio_target: float = 0.15
    point_to_plane_huber_delta_m: float = 0.02
    # The needle guard hinges here: only max/mid ABOVE this ratio is penalized.
    # A round disk has max/mid = 1, but a good surface blade is elongated, and
    # the reference delivery's own max/mid runs to 3 at the median and ~57 at
    # p99. Penalising toward 1 would force us rounder than the reference; the
    # hinge lets the blade shapes through and catches only pathological needles.
    tangent_isotropy_max_ratio: float = 1.0

    def validate(self) -> None:
        if (
            self.weight_align < 0.0
            or self.weight_flatten < 0.0
            or self.weight_tangent_isotropy < 0.0
            or self.weight_point_to_plane < 0.0
        ):
            raise ValueError("normal alignment weights must be non-negative")
        if not 0.0 <= self.planarity_gate <= 1.0:
            raise ValueError("planarity_gate must be within [0, 1]")
        if self.max_anchor_distance_m <= 0.0:
            raise ValueError("max_anchor_distance_m must be positive")
        if self.refresh_every < 1:
            raise ValueError("refresh_every must be at least one")
        if self.flatten_mode not in {"absolute_m", "tangent_ratio"}:
            raise ValueError("flatten_mode must be absolute_m or tangent_ratio")
        if self.flatten_target_m <= 0.0:
            raise ValueError("flatten_target_m must be positive")
        if not 0.0 < self.flatten_ratio_target < 1.0:
            raise ValueError("flatten_ratio_target must be within (0, 1)")
        if self.tangent_isotropy_max_ratio < 1.0:
            raise ValueError("tangent_isotropy_max_ratio must be at least 1")
        if self.point_to_plane_huber_delta_m <= 0.0:
            raise ValueError("point_to_plane_huber_delta_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": self.enabled,
            "weight_align": self.weight_align,
            "weight_flatten": self.weight_flatten,
            "weight_tangent_isotropy": self.weight_tangent_isotropy,
            "weight_point_to_plane": self.weight_point_to_plane,
            "planarity_gate": self.planarity_gate,
            "max_anchor_distance_m": self.max_anchor_distance_m,
            "refresh_every": self.refresh_every,
            "flatten_mode": self.flatten_mode,
            "flatten_target_m": self.flatten_target_m,
            "flatten_ratio_target": self.flatten_ratio_target,
            "tangent_isotropy_max_ratio": self.tangent_isotropy_max_ratio,
            "point_to_plane_huber_delta_m": self.point_to_plane_huber_delta_m,
        }


# ---------------------------------------------------------------------------
# Training-time anchoring and losses
# ---------------------------------------------------------------------------


def _quats_to_rotation_columns(quats: Any) -> Any:
    """Rotation matrices ``[N, 3, 3]`` from unnormalized wxyz quaternions.

    Column ``j`` of each matrix is the j-th principal axis of the Gaussian
    (the image of the j-th basis vector). Fully differentiable through the
    normalization.
    """
    torch = __import__("torch")
    norm = quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = (quats / norm).unbind(dim=-1)
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                dim=-1,
            ),
            torch.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                dim=-1,
            ),
            torch.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                dim=-1,
            ),
        ],
        dim=-2,
    )


class LidarNormalAnchors:
    """Per-Gaussian LiDAR anchors plus the differentiable penalties.

    ``refresh`` snaps every Gaussian to its nearest LiDAR point on the CPU
    (means may live on CUDA; they are detached and copied). ``loss`` then
    reads the cached anchors and returns the align / flatten penalties. If
    the Gaussian count changed since the last refresh (MCMC add/remove) the
    loss is zero and ``stale`` is set until the next ``refresh``.
    """

    def __init__(self, field: NormalField, config: NormalAlignmentConfig) -> None:
        config.validate()
        self.field = field
        self.config = config
        self.anchor_normal: Any = None  # torch [N, 3] on CPU
        self.anchor_position: Any = None  # torch [N, 3] on CPU
        self.planarity: Any = None  # torch [N] on CPU
        self.distance: Any = None  # torch [N] on CPU
        self.valid: Any = None  # torch bool [N] on CPU
        self.anchored_count = 0
        self.stale = True
        self._device_cache: dict[Any, tuple[Any, Any, Any, Any]] = {}

    def refresh(self, means: Any) -> int:
        """Re-anchor every Gaussian to its nearest LiDAR point.

        Accepts a ``[N, 3]`` torch tensor on any device (CUDA tolerated via
        ``detach().cpu()``). Returns the number of valid anchors.
        """
        torch = __import__("torch")
        points = means.detach().cpu().to(torch.float64).numpy()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("means must have shape [N, 3]")
        distance, normal, planarity, index = self.field.query(points, k=1)
        self.anchor_normal = torch.from_numpy(
            np.ascontiguousarray(normal, dtype=np.float32)
        )
        self.anchor_position = torch.from_numpy(
            np.ascontiguousarray(self.field.xyz[index], dtype=np.float32)
        )
        self.planarity = torch.from_numpy(
            np.ascontiguousarray(planarity, dtype=np.float32)
        )
        self.distance = torch.from_numpy(
            np.ascontiguousarray(distance, dtype=np.float32)
        )
        self.valid = (self.distance <= float(self.config.max_anchor_distance_m)) & (
            self.planarity >= float(self.config.planarity_gate)
        )
        self.anchored_count = int(self.valid.sum())
        self.stale = False
        self._device_cache.clear()
        return self.anchored_count

    def _anchors_on(self, device: Any) -> tuple[Any, Any, Any, Any]:
        cached = self._device_cache.get(device)
        if cached is None:
            cached = (
                self.anchor_normal.to(device),
                self.anchor_position.to(device),
                self.planarity.to(device),
                self.valid.to(device),
            )
            self._device_cache[device] = cached
        return cached

    def loss(self, params: Any) -> dict[str, Any]:
        """Weighted align / flatten penalties for the current parameters.

        Returns ``align`` and ``flatten`` already multiplied by their weights
        (``total = align + flatten``); the unweighted planarity-weighted
        means are exposed as ``align_raw`` / ``flatten_raw`` for logging.
        """
        torch = __import__("torch")
        required = {"means", "scales", "quats"}
        if not required <= set(params):
            raise ValueError("normal alignment requires scales and quats")
        scales_log = params["scales"]
        quats = params["quats"]
        if scales_log.ndim != 2 or scales_log.shape[1] != 3:
            raise ValueError("normal alignment expects [N, 3] log scales")
        if quats.ndim != 2 or quats.shape[1] != 4:
            raise ValueError("normal alignment expects [N, 4] wxyz quaternions")
        if quats.shape[0] != scales_log.shape[0]:
            raise ValueError("quaternion and scale counts differ")
        zero = scales_log.new_zeros(())
        result = {
            "align": zero,
            "flatten": zero,
            "tangent_isotropy": zero,
            "point_to_plane": zero,
            "align_raw": zero,
            "flatten_raw": zero,
            "tangent_isotropy_raw": zero,
            "point_to_plane_raw": zero,
            "total": zero,
            "anchored_count": 0,
            "stale": False,
        }
        if not self.config.enabled:
            return result
        if self.anchor_normal is None or self.stale:
            self.stale = True
            result["stale"] = True
            return result
        if scales_log.shape[0] != self.anchor_normal.shape[0]:
            # Gaussian count changed (MCMC add/remove): anchors no longer line
            # up index-wise. Degrade to zero until the next refresh.
            self.stale = True
            result["stale"] = True
            return result
        device = scales_log.device
        normal, anchor_position, planarity, valid = self._anchors_on(device)
        count = int(valid.sum())
        result["anchored_count"] = count
        if count == 0:
            return result

        scales_valid = scales_log[valid]
        quats_valid = quats[valid]
        weight = planarity[valid].to(scales_valid.dtype)
        weight_sum = weight.sum().clamp_min(_EPS)

        # Shortest-axis selection: the argmin index is detached (a hard,
        # non-differentiable choice) but the selected axis vector and scale
        # stay differentiable through quats / scales.
        min_index = scales_valid.argmin(dim=1).detach()
        rotation = _quats_to_rotation_columns(quats_valid)
        axes = rotation.transpose(1, 2)  # [M, axis, xyz]
        u_min = axes.gather(
            1, min_index.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)

        # sin^2 of the angle between shortest axis and (unoriented) normal:
        # symmetric under n -> -n by construction.
        cos = (u_min * normal[valid].to(scales_valid.dtype)).sum(dim=1)
        align_raw = (weight * (1.0 - cos.square())).sum() / weight_sum

        # Only excess thickness beyond the selected target is penalized. The
        # original absolute-metre ceiling is retained for compatibility. The
        # ratio mode is scale invariant and therefore distinguishes a true
        # surface disk from a merely small but still round Gaussian.
        scales_m = torch.exp(scales_valid)
        min_scale = scales_m.gather(
            1, min_index.view(-1, 1)
        ).squeeze(1)
        if self.config.flatten_mode == "tangent_ratio":
            sorted_scale = torch.sort(scales_m, dim=1).values
            tangent_geometric_mean = torch.sqrt(
                (sorted_scale[:, 1] * sorted_scale[:, 2]).clamp_min(_EPS)
            )
            thickness = min_scale / tangent_geometric_mean
            target = float(self.config.flatten_ratio_target)
        else:
            thickness = min_scale
            target = float(self.config.flatten_target_m)
        excess = torch.relu(thickness - target)
        flatten_raw = (weight * excess.square()).sum() / weight_sum

        # Thinning alone does not decide what the remaining two axes do. A
        # needle and a disk are both thin, and only the disk actually covers a
        # surface, so the ratio of the two TANGENTIAL axes is penalized away
        # from one. Sorting makes this the larger over the middle axis, which
        # is >= 1 by construction. This deliberately does NOT touch the shortest
        # axis: a thin disk [10, 10, 0.2] is a good surface primitive and must
        # be free to get thinner, so the guard is on max/mid, never max/min.
        #
        # The penalty is on the LOG ratio, not the raw ratio: log keeps it
        # bounded (a ratio of 1000 costs ~48, not ~10^6, so it cannot dominate
        # the loss the way the raw square did), while still rising without limit
        # in the ratio itself. It is also hinged at tangent_isotropy_max_ratio,
        # so blade shapes up to the reference's own tail pay nothing and only
        # the pathological needles beyond it are pushed back.
        sorted_tangent = torch.sort(scales_m, dim=1).values
        tangent_ratio = sorted_tangent[:, 2] / sorted_tangent[:, 1].clamp_min(_EPS)
        log_excess = torch.relu(
            torch.log(tangent_ratio.clamp_min(_EPS))
            - math.log(float(self.config.tangent_isotropy_max_ratio))
        )
        tangent_isotropy_raw = (weight * log_excess.square()).sum() / weight_sum

        # A soft surface tether: only displacement along the LiDAR normal is
        # penalized.  Tangential motion remains free, which is the essential
        # difference from nearest-point locking and lets projected RGB
        # gradients redistribute centers along a real wall or snow surface.
        means_valid = params["means"][valid]
        signed_distance = (
            (means_valid - anchor_position[valid].to(means_valid.dtype))
            * normal[valid].to(means_valid.dtype)
        ).sum(dim=1)
        absolute_distance = signed_distance.abs()
        delta = float(self.config.point_to_plane_huber_delta_m)
        point_to_plane_per_row = torch.where(
            absolute_distance <= delta,
            0.5 * signed_distance.square() / delta,
            absolute_distance - 0.5 * delta,
        )
        point_to_plane_raw = (
            weight * point_to_plane_per_row
        ).sum() / weight_sum

        align = float(self.config.weight_align) * align_raw
        flatten = float(self.config.weight_flatten) * flatten_raw
        tangent_isotropy = (
            float(self.config.weight_tangent_isotropy) * tangent_isotropy_raw
        )
        point_to_plane = (
            float(self.config.weight_point_to_plane) * point_to_plane_raw
        )
        result.update(
            {
                "align": align,
                "flatten": flatten,
                "tangent_isotropy": tangent_isotropy,
                "point_to_plane": point_to_plane,
                "align_raw": align_raw,
                "flatten_raw": flatten_raw,
                "tangent_isotropy_raw": tangent_isotropy_raw,
                "point_to_plane_raw": point_to_plane_raw,
                "total": align + flatten + tangent_isotropy + point_to_plane,
            }
        )
        return result
