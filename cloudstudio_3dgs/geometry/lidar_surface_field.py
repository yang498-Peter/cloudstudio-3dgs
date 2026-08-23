"""Continuous LiDAR surface field: normal distance instead of nearest-point distance.

Every LiDAR-derived constraint in this repository currently reduces the cloud to
a set of discrete samples and asks "how far is this Gaussian from the closest
LiDAR point?". That question is answered by a quantity that is contaminated by
everything except the geometry we care about: sampling density, scan-line
spacing, voxelization, incidence angle, vegetation and surface edges. A Gaussian
sitting perfectly on a wall but landing halfway between two scan lines can score
a *larger* nearest-point distance than a Gaussian genuinely floating in mid-air.

The quantity that actually encodes "is this Gaussian on the surface?" is the
distance along the local surface normal::

    d_perp = |n . (mu - p)|

where ``p`` is the nearest LiDAR sample, ``n`` its (unoriented) local surface
normal and ``mu`` the Gaussian mean. The complementary tangential distance
``d_tangent`` measures how far along the surface the query drifted from the
sample and is exactly the part that sampling density pollutes. Reporting both
separates "off the surface" from "between two samples".

This module builds a per-point surface description (normal, tangent basis,
planarity, roughness, local spacing, confidence) and exposes a nearest-surface
query returning all of the above plus ``d_perp`` / ``d_tangent`` / ``euclidean``.
It is deliberately a *superset* of
:mod:`cloudstudio_3dgs.training.lidar_normals`; that module is untouched and
still owns the training-time anchoring path.

Related work: LI-GS maintains plane-constrained multimodal GMMs as a continuous
planar prior, and Structured-Li-GS denoises, estimates normals and runs Poisson
reconstruction so that Gaussians can be filtered by point-to-mesh distance. Both
share the same premise implemented here: a LiDAR prior must represent a
*continuous surface*, not a bag of discrete nearest points.

CPU / numpy only. No torch, no CUDA, no training wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EPS = 1e-12

DEFAULT_KNN = 24
DEFAULT_BATCH_SIZE = 50_000
DEFAULT_NEIGHBOR_RADIUS_FACTOR = 4.0
DEFAULT_SPACING_SAMPLE = 20_000
DEFAULT_MIN_SIGMA_M = 1e-4


# ---------------------------------------------------------------------------
# Query result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceQuery:
    """Result of a nearest-surface query.

    Every field has a leading query axis ``[P, ...]``. For ``k == 1`` there is
    no k axis (scalars are ``[P]``, vectors ``[P, 3]``, bases ``[P, 3, 2]``);
    for ``k > 1`` a k axis is inserted right after the query axis.

    Distances
    ---------
    ``euclidean``
        Plain nearest-point distance ``||mu - p||`` — the *old* criterion,
        retained purely for side-by-side comparison.
    ``signed_d_perp``
        ``n . (mu - p)``. LiDAR normals are unoriented, so the sign is only
        meaningful *relative* to other queries hitting the same surface point
        (e.g. deciding whether two Gaussians sit on the same side of a wall).
    ``d_perp``
        ``|signed_d_perp|`` — the core quantity. Distance to the continuous
        local surface, immune to sampling density along the surface.
    ``d_tangent``
        ``sqrt(euclidean^2 - d_perp^2)`` — how far the query is from the sample
        *along* the surface. This is precisely the component that scan-line
        spacing and voxelization inflate. ``euclidean^2 = d_perp^2 +
        d_tangent^2`` holds exactly by construction (clamped at zero against
        floating point noise).
    """

    surface_point: np.ndarray
    normal: np.ndarray
    tangent_basis: np.ndarray
    signed_d_perp: np.ndarray
    d_perp: np.ndarray
    d_tangent: np.ndarray
    euclidean: np.ndarray
    planarity: np.ndarray
    roughness: np.ndarray
    local_spacing: np.ndarray
    confidence: np.ndarray
    index: np.ndarray


def support_weight(
    query: SurfaceQuery,
    *,
    sigma_perp_factor: float = 1.0,
    min_sigma_m: float = DEFAULT_MIN_SIGMA_M,
) -> np.ndarray:
    """Surface support in ``[0, 1]``: 1 = certainly on the surface, 0 = floating.

    ::

        sigma  = sigma_perp_factor * max(local_spacing, roughness, min_sigma_m)
        weight = exp(-(d_perp / sigma)^2) * confidence

    Rationale for each factor:

    * ``d_perp`` (not ``euclidean``) is the numerator, so a Gaussian that is
      glued to a wall but sits between two scan lines scores ~1 even though its
      nearest-point distance is half the scan-line gap.
    * ``sigma`` is *adaptive to the local sampling*: ``local_spacing`` is the
      scale at which this neighborhood can resolve geometry at all, so in a
      sparse region we must not claim sub-spacing accuracy. Where the surface is
      genuinely rough (``roughness > local_spacing``) the surface itself has no
      well-defined position at finer scales, so roughness takes over as the
      tolerance. ``min_sigma_m`` only guards against a division by zero on
      degenerate (duplicated-point) neighborhoods.
    * multiplying by ``confidence`` demotes neighborhoods that are not planar or
      not well sampled (vegetation, isolated outliers, surface edges): being
      close to a normal we do not trust is not evidence of surface support.

    A Gaussian exactly on the surface gets ``confidence``; one a full local
    spacing away gets ``e^-1 * confidence ~ 0.37 * confidence``; two spacings
    away ``e^-4 ~ 0.018``. Raise ``sigma_perp_factor`` to be more permissive.
    """
    if sigma_perp_factor <= 0.0:
        raise ValueError("sigma_perp_factor must be positive")
    if min_sigma_m <= 0.0:
        raise ValueError("min_sigma_m must be positive")
    spacing = np.asarray(query.local_spacing, dtype=np.float64)
    roughness = np.asarray(query.roughness, dtype=np.float64)
    sigma = float(sigma_perp_factor) * np.maximum(
        np.maximum(spacing, roughness), float(min_sigma_m)
    )
    ratio = np.asarray(query.d_perp, dtype=np.float64) / sigma
    return np.exp(-np.square(ratio)) * np.asarray(query.confidence, dtype=np.float64)


# ---------------------------------------------------------------------------
# Surface field
# ---------------------------------------------------------------------------


class LidarSurfaceField:
    """Nearest-surface queryable description of a LiDAR cloud.

    Per LiDAR point it stores

    ``normal``        unit normal of the local KNN-PCA plane, **unoriented**
                      (the stored sign is a deterministic convention only, and
                      matches :func:`cloudstudio_3dgs.training.lidar_normals.build_normal_field`);
    ``tangent_basis`` ``[3, 2]`` orthonormal complement of ``normal`` (the two
                      dominant PCA axes), columns ordered by descending
                      eigenvalue;
    ``planarity``     ``1 - 3 * l0 / (l0 + l1 + l2)`` over the ascending KNN
                      covariance eigenvalues — **identical definition** to
                      ``lidar_normals.build_normal_field`` so the two modules
                      gate on the same number;
    ``roughness``     RMS distance (metres) of the KNN neighbors to the fitted
                      local plane;
    ``local_spacing`` median of the KNN distances (metres, self excluded). Note
                      this is a *neighborhood* scale, not the nearest-neighbor
                      spacing: on a uniform square grid of pitch ``s`` with
                      ``knn = 24`` it evaluates to exactly ``2 * s``. That is the
                      intended behaviour — on an anisotropic scan-line cloud it
                      reports the large cross-line gap rather than the fine
                      along-line pitch, which is exactly the tolerance a
                      surface-support test needs;
    ``confidence``    ``planarity * neighbor_support``, see
                      :func:`build_surface_field`.
    """

    def __init__(
        self,
        xyz: np.ndarray,
        normals: np.ndarray,
        tangent_basis: np.ndarray,
        planarity: np.ndarray,
        roughness: np.ndarray,
        local_spacing: np.ndarray,
        confidence: np.ndarray,
        *,
        knn: int = DEFAULT_KNN,
        neighbor_radius_m: float = 0.0,
    ) -> None:
        xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float64))
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N, 3]")
        count = len(xyz)

        def _vector(name: str, value: np.ndarray) -> np.ndarray:
            array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
            if array.shape != (count, 3):
                raise ValueError(f"{name} must have shape [N, 3]")
            return array

        def _scalar(name: str, value: np.ndarray) -> np.ndarray:
            array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
            if array.shape != (count,):
                raise ValueError(f"{name} must have shape [N]")
            return array

        basis = np.ascontiguousarray(np.asarray(tangent_basis, dtype=np.float32))
        if basis.shape != (count, 3, 2):
            raise ValueError("tangent_basis must have shape [N, 3, 2]")

        self.xyz = xyz
        self.normals = _vector("normals", normals)
        self.tangent_basis = basis
        self.planarity = _scalar("planarity", planarity)
        self.roughness = _scalar("roughness", roughness)
        self.local_spacing = _scalar("local_spacing", local_spacing)
        self.confidence = _scalar("confidence", confidence)
        self.knn = int(knn)
        self.neighbor_radius_m = float(neighbor_radius_m)

        from scipy.spatial import cKDTree

        self.tree = cKDTree(self.xyz)

    def __len__(self) -> int:
        return len(self.xyz)

    # -- query --------------------------------------------------------------

    def query(self, points: np.ndarray, k: int = 1) -> SurfaceQuery:
        """Nearest-surface query for ``[P, 3]`` points; see :class:`SurfaceQuery`."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [P, 3]")
        if k < 1:
            raise ValueError("k must be at least one")
        distance, index = self.tree.query(points, k=k, workers=-1)
        distance = np.asarray(distance, dtype=np.float64)
        index = np.asarray(index, dtype=np.int64)
        surface_point = self.xyz[index]
        normal = self.normals[index].astype(np.float64)
        query_points = points if k == 1 else points[:, None, :]
        delta = query_points - surface_point
        signed = np.einsum("...i,...i->...", normal, delta)
        d_perp = np.abs(signed)
        # euclidean^2 = d_perp^2 + d_tangent^2; clamp the subtraction so float
        # noise on an exactly-on-normal query cannot produce a NaN.
        d_tangent = np.sqrt(np.maximum(distance**2 - d_perp**2, 0.0))
        return SurfaceQuery(
            surface_point=surface_point,
            normal=normal,
            tangent_basis=self.tangent_basis[index].astype(np.float64),
            signed_d_perp=signed,
            d_perp=d_perp,
            d_tangent=d_tangent,
            euclidean=distance,
            planarity=self.planarity[index].astype(np.float64),
            roughness=self.roughness[index].astype(np.float64),
            local_spacing=self.local_spacing[index].astype(np.float64),
            confidence=self.confidence[index].astype(np.float64),
            index=index,
        )

    def support_weight(
        self,
        query: SurfaceQuery,
        *,
        sigma_perp_factor: float = 1.0,
        min_sigma_m: float = DEFAULT_MIN_SIGMA_M,
    ) -> np.ndarray:
        """Method form of the module-level :func:`support_weight`."""
        return support_weight(
            query, sigma_perp_factor=sigma_perp_factor, min_sigma_m=min_sigma_m
        )

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist as compressed npz (the KD-tree is rebuilt on load)."""
        np.savez_compressed(
            Path(path),
            xyz=self.xyz,
            normals=self.normals,
            tangent_basis=self.tangent_basis,
            planarity=self.planarity,
            roughness=self.roughness,
            local_spacing=self.local_spacing,
            confidence=self.confidence,
            knn=np.int64(self.knn),
            neighbor_radius_m=np.float64(self.neighbor_radius_m),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LidarSurfaceField":
        with np.load(Path(path)) as data:
            return cls(
                data["xyz"],
                data["normals"],
                data["tangent_basis"],
                data["planarity"],
                data["roughness"],
                data["local_spacing"],
                data["confidence"],
                knn=int(data["knn"]),
                neighbor_radius_m=float(data["neighbor_radius_m"]),
            )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _global_spacing(tree, points: np.ndarray, k: int, sample: int, seed: int) -> float:
    """Median KNN spacing over a random subsample (cheap pre-pass)."""
    count = len(points)
    if sample >= count:
        subset = points
    else:
        rng = np.random.default_rng(seed)
        subset = points[rng.choice(count, sample, replace=False)]
    distances, _ = tree.query(subset, k=k, workers=-1)
    spacing = np.median(np.atleast_2d(distances)[:, 1:], axis=1)
    value = float(np.median(spacing))
    return value if value > 0.0 else 0.0


def build_surface_field(
    xyz: np.ndarray,
    *,
    knn: int = DEFAULT_KNN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    neighbor_radius_factor: float = DEFAULT_NEIGHBOR_RADIUS_FACTOR,
    neighbor_radius_m: float | None = None,
    spacing_sample: int = DEFAULT_SPACING_SAMPLE,
    seed: int = 0,
) -> LidarSurfaceField:
    """KNN-PCA surface description for every LiDAR point (CPU, one-off, batched).

    For each point the covariance of its ``knn`` nearest neighbors (self
    included, matching ``lidar_normals.build_normal_field``) is eigen-decomposed
    with ascending eigenvalues ``l0 <= l1 <= l2``:

    * ``normal`` = eigenvector of ``l0``, sign-normalized by a deterministic but
      physically meaningless convention (consumers must be symmetric under
      ``n -> -n``);
    * ``tangent_basis`` = eigenvectors of ``l2`` then ``l1`` — an orthonormal
      basis of the plane perpendicular to ``normal``;
    * ``planarity`` = ``1 - 3 * l0 / (l0 + l1 + l2)`` in ``[0, 1]``; an exact
      plane gives 1, an isotropic blob 0, a degenerate (zero-trace) neighborhood
      is forced to 0. This is byte-for-byte the definition used by
      ``lidar_normals.build_normal_field``, so gates tuned on one module carry
      over to the other;
    * ``roughness`` = ``sqrt(mean_j (n . (q_j - centroid))^2)`` over the ``knn``
      neighbors — the RMS residual of the local plane fit, in metres;
    * ``local_spacing`` = median of the KNN distances with the self-distance
      column dropped, in metres.

    Confidence
    ----------
    ::

        neighbor_support = #{neighbors with distance <= neighbor_radius} / (knn - 1)
        confidence       = planarity * neighbor_support

    ``neighbor_radius`` defaults to ``neighbor_radius_factor`` (4) times the
    *global* median KNN spacing, estimated once from a random subsample of
    ``spacing_sample`` points; pass ``neighbor_radius_m`` to pin it explicitly.
    The two factors answer two different failure modes: ``planarity`` rejects
    neighborhoods with no well-defined normal (vegetation, clutter, corners),
    while ``neighbor_support`` rejects neighborhoods that only *look* planar
    because they are starved of points — an isolated outlier or a thin fringe
    has to reach far outside the global sampling scale to collect ``knn``
    neighbors, so most of them fall outside the radius and its support collapses
    toward 0. A dense point on a clean wall scores ``1 * 1 = 1``.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if len(points) < 3:
        raise ValueError("at least three points are required for KNN-PCA")
    if knn < 3:
        raise ValueError("knn must be at least three")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if neighbor_radius_factor <= 0.0:
        raise ValueError("neighbor_radius_factor must be positive")
    if neighbor_radius_m is not None and neighbor_radius_m <= 0.0:
        raise ValueError("neighbor_radius_m must be positive when given")
    if spacing_sample < 1:
        raise ValueError("spacing_sample must be at least one")

    count = len(points)
    k = min(int(knn), count)
    points = np.ascontiguousarray(points)
    tree = cKDTree(points)

    if neighbor_radius_m is None:
        spacing = _global_spacing(tree, points, k, int(spacing_sample), int(seed))
        # A cloud of coincident points has zero spacing; fall back to a radius
        # that accepts everything so confidence degrades to pure planarity.
        radius = (
            float(neighbor_radius_factor) * spacing if spacing > 0.0 else float("inf")
        )
    else:
        radius = float(neighbor_radius_m)

    normals = np.empty((count, 3), dtype=np.float32)
    tangents = np.empty((count, 3, 2), dtype=np.float32)
    planarity = np.empty(count, dtype=np.float32)
    roughness = np.empty(count, dtype=np.float32)
    local_spacing = np.empty(count, dtype=np.float32)
    confidence = np.empty(count, dtype=np.float32)
    denominator = float(max(k - 1, 1))

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        distances, indexes = tree.query(points[start:stop], k=k, workers=-1)
        distances = np.atleast_2d(np.asarray(distances, dtype=np.float64))
        indexes = np.atleast_2d(np.asarray(indexes, dtype=np.int64))
        neighborhoods = points[indexes]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
        values, vectors = np.linalg.eigh(cov)  # ascending eigenvalues

        normal = vectors[:, :, 0]
        major_axis = np.argmax(np.abs(normal), axis=1)
        signs = np.sign(normal[np.arange(len(normal)), major_axis])
        signs[signs == 0] = 1.0
        normal = normal * signs[:, None]

        trace = values.sum(axis=1)
        plan = 1.0 - 3.0 * values[:, 0] / np.maximum(trace, _EPS)
        plan = np.clip(plan, 0.0, 1.0)
        plan[trace <= _EPS] = 0.0

        residual = np.einsum("nki,ni->nk", centered, normal)
        rough = np.sqrt(np.mean(np.square(residual), axis=1))

        neighbor_distances = distances[:, 1:] if k > 1 else distances[:, :1]
        spacing = np.median(neighbor_distances, axis=1)
        support = np.clip(
            (neighbor_distances <= radius).sum(axis=1) / denominator, 0.0, 1.0
        )

        normals[start:stop] = normal.astype(np.float32)
        # Descending-eigenvalue order: dominant tangent first.
        tangents[start:stop] = vectors[:, :, [2, 1]].astype(np.float32)
        planarity[start:stop] = plan.astype(np.float32)
        roughness[start:stop] = rough.astype(np.float32)
        local_spacing[start:stop] = spacing.astype(np.float32)
        confidence[start:stop] = (plan * support).astype(np.float32)

    field = LidarSurfaceField.__new__(LidarSurfaceField)
    field.xyz = points
    field.normals = normals
    field.tangent_basis = tangents
    field.planarity = planarity
    field.roughness = roughness
    field.local_spacing = local_spacing
    field.confidence = confidence
    field.knn = k
    field.neighbor_radius_m = radius
    field.tree = tree
    return field
