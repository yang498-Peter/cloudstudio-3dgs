"""CPU tests for the continuous LiDAR surface field (normal-distance queries).

The headline case is :meth:`SparseScanLineTest.test_flush_gaussian_between_scan_lines`:
a Gaussian glued to a wall but landing halfway between two scan lines. The old
nearest-point criterion calls it 0.25 m off the surface; ``d_perp`` correctly
reports 0 and the support weight stays at 1.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from cloudstudio_3dgs.geometry.lidar_surface_field import (
    LidarSurfaceField,
    SurfaceQuery,
    build_surface_field,
    support_weight,
)

REAL_LIDAR_PLY = Path(r"C:\Peter\3dgs-datasets\ukgs_lidar_init\sparse_pc.ply")


# ---------------------------------------------------------------------------
# Synthetic clouds
# ---------------------------------------------------------------------------


def _grid_plane(pitch: float = 0.1, extent: float = 1.0, noise: float = 0.0) -> np.ndarray:
    """Isotropic z = 0 grid of the given pitch over ``[-extent, extent]^2``."""
    axis = np.arange(-extent, extent + pitch * 0.5, pitch)
    xx, yy = np.meshgrid(axis, axis)
    cloud = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    if noise > 0.0:
        cloud[:, 2] += np.random.default_rng(5).normal(scale=noise, size=len(cloud))
    return cloud


def _scan_line_plane(
    along_pitch: float = 0.1, across_pitch: float = 0.5, extent: float = 2.0
) -> np.ndarray:
    """z = 0 plane sampled densely along x and sparsely across y (scan lines)."""
    x = np.arange(-extent, extent + along_pitch * 0.5, along_pitch)
    y = np.arange(-extent, extent + across_pitch * 0.5, across_pitch)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)


def _tilted_plane(pitch: float = 0.05, extent: float = 1.0) -> np.ndarray:
    """z = x, i.e. a plane tilted 45 degrees; unit normal (1, 0, -1)/sqrt(2)."""
    axis = np.arange(-extent, extent + pitch * 0.5, pitch)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx.ravel(), yy.ravel(), xx.ravel()], axis=1)


_TILT_NORMAL = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Field construction
# ---------------------------------------------------------------------------


class BuildSurfaceFieldTest(unittest.TestCase):
    def test_flat_plane_geometry(self) -> None:
        field = build_surface_field(_grid_plane(), knn=24)
        self.assertEqual(len(field), 441)
        self.assertGreater(float(np.abs(field.normals[:, 2]).min()), 0.999)
        np.testing.assert_allclose(
            np.linalg.norm(field.normals, axis=1), 1.0, atol=1e-5
        )
        self.assertGreater(float(np.median(field.planarity)), 0.99)
        self.assertLess(float(np.max(field.roughness)), 1e-9)
        self.assertGreater(float(np.median(field.confidence)), 0.99)

    def test_tangent_basis_is_orthonormal_complement_of_normal(self) -> None:
        field = build_surface_field(_tilted_plane(), knn=24)
        basis = field.tangent_basis.astype(np.float64)  # [N, 3, 2]
        normals = field.normals.astype(np.float64)
        # Columns are unit length, mutually orthogonal and orthogonal to n.
        gram = np.einsum("nij,nik->njk", basis, basis)
        np.testing.assert_allclose(gram, np.broadcast_to(np.eye(2), gram.shape), atol=1e-5)
        np.testing.assert_allclose(
            np.einsum("nij,ni->nj", basis, normals), 0.0, atol=1e-5
        )

    def test_roughness_tracks_surface_noise(self) -> None:
        flat = build_surface_field(_grid_plane(noise=0.0), knn=24)
        rough = build_surface_field(_grid_plane(noise=0.02), knn=24)
        flat_median = float(np.median(flat.roughness))
        rough_median = float(np.median(rough.roughness))
        self.assertLess(flat_median, 1e-9)
        self.assertGreater(rough_median, 10.0 * max(flat_median, 1e-12))
        # The local plane fit absorbs part of the noise, so roughness lands a
        # little under the injected sigma but on the same order.
        self.assertGreater(rough_median, 0.005)
        self.assertLess(rough_median, 0.02)

    def test_local_spacing_scales_with_construction_pitch(self) -> None:
        # On a uniform square grid with knn=24 the median of the 23 non-self
        # neighbor distances is exactly 2 * pitch (the 12th sorted distance).
        for pitch in (0.1, 0.2, 0.4):
            # 41x41 nodes: the boundary ring is a small enough minority that the
            # median is still the interior value.
            field = build_surface_field(
                _grid_plane(pitch=pitch, extent=20.0 * pitch), knn=24
            )
            self.assertAlmostEqual(
                float(np.median(field.local_spacing)), 2.0 * pitch, places=5
            )

    def test_isolated_outlier_loses_confidence_despite_high_planarity(self) -> None:
        # A stray point 5 m off the wall still sees a *planar* neighborhood (its
        # 24 nearest points all lie on the wall), so planarity alone would trust
        # it. The neighbor-support factor is what kills its confidence.
        cloud = np.vstack([_grid_plane(), [[0.0, 0.0, 5.0]]])
        field = build_surface_field(cloud, knn=24)
        self.assertGreater(float(field.planarity[-1]), 0.9)
        self.assertLess(float(field.confidence[-1]), 0.05)
        self.assertGreater(float(np.median(field.confidence[:-1])), 0.99)

    def test_degenerate_duplicate_cloud_gets_zero_confidence(self) -> None:
        field = build_surface_field(np.zeros((32, 3)), knn=8)
        np.testing.assert_allclose(field.planarity, 0.0)
        np.testing.assert_allclose(field.confidence, 0.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        cloud = _grid_plane()
        with self.assertRaises(ValueError):
            build_surface_field(cloud[:, :2])
        with self.assertRaises(ValueError):
            build_surface_field(cloud[:2])
        with self.assertRaises(ValueError):
            build_surface_field(cloud, knn=2)
        with self.assertRaises(ValueError):
            build_surface_field(cloud, batch_size=0)
        with self.assertRaises(ValueError):
            build_surface_field(cloud, neighbor_radius_m=0.0)
        field = build_surface_field(cloud, knn=8)
        with self.assertRaises(ValueError):
            field.query(cloud[:, :2])
        with self.assertRaises(ValueError):
            field.query(cloud[:4], k=0)
        with self.assertRaises(ValueError):
            support_weight(field.query(cloud[:4]), sigma_perp_factor=0.0)

    def test_batching_does_not_change_results(self) -> None:
        cloud = _grid_plane(noise=0.01, extent=2.0)
        whole = build_surface_field(cloud, knn=16, batch_size=10_000)
        chunked = build_surface_field(cloud, knn=16, batch_size=97)
        np.testing.assert_allclose(whole.planarity, chunked.planarity, atol=1e-6)
        np.testing.assert_allclose(whole.roughness, chunked.roughness, atol=1e-6)
        np.testing.assert_allclose(
            whole.local_spacing, chunked.local_spacing, atol=1e-6
        )
        np.testing.assert_allclose(whole.confidence, chunked.confidence, atol=1e-6)


# ---------------------------------------------------------------------------
# Queries on a dense plane
# ---------------------------------------------------------------------------


class DensePlaneQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.field = build_surface_field(_grid_plane(), knn=24)

    def test_on_surface_query_is_exact(self) -> None:
        query = self.field.query(np.array([[0.1, -0.2, 0.0]]))  # exactly a node
        self.assertAlmostEqual(float(query.d_perp[0]), 0.0, places=9)
        self.assertAlmostEqual(float(query.d_tangent[0]), 0.0, places=6)
        self.assertAlmostEqual(float(query.euclidean[0]), 0.0, places=6)
        self.assertAlmostEqual(float(support_weight(query)[0]), 1.0, places=3)

    def test_normal_offset_gives_exact_perpendicular_distance(self) -> None:
        heights = np.array([0.01, 0.05, 0.2, 1.0])
        points = np.stack(
            [np.zeros_like(heights), np.zeros_like(heights), heights], axis=1
        )
        query = self.field.query(points)
        np.testing.assert_allclose(query.d_perp, heights, atol=1e-9)
        np.testing.assert_allclose(query.d_tangent, 0.0, atol=1e-6)
        np.testing.assert_allclose(query.euclidean, heights, atol=1e-9)

    def test_airborne_gaussian_has_no_surface_support(self) -> None:
        query = self.field.query(np.array([[0.0, 0.0, 1.0]]))
        self.assertAlmostEqual(float(query.d_perp[0]), 1.0, places=9)
        self.assertLess(float(support_weight(query)[0]), 1e-6)

    def test_signed_perp_flips_across_the_surface(self) -> None:
        query = self.field.query(np.array([[0.0, 0.0, 0.3], [0.0, 0.0, -0.3]]))
        self.assertLess(float(query.signed_d_perp[0] * query.signed_d_perp[1]), 0.0)
        np.testing.assert_allclose(query.d_perp, 0.3, atol=1e-9)

    def test_pythagoras_and_dperp_bound_hold_on_random_queries(self) -> None:
        rng = np.random.default_rng(17)
        points = rng.uniform(-1.0, 1.0, size=(500, 3))
        query = self.field.query(points)
        self.assertTrue(bool(np.all(query.d_perp <= query.euclidean + 1e-9)))
        np.testing.assert_allclose(
            query.d_perp**2 + query.d_tangent**2, query.euclidean**2, atol=1e-9
        )
        # On a dense z = 0 plane d_perp reduces to |z|.
        np.testing.assert_allclose(query.d_perp, np.abs(points[:, 2]), atol=1e-9)

    def test_support_weight_is_monotonic_in_dperp(self) -> None:
        heights = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
        points = np.stack(
            [np.zeros_like(heights), np.zeros_like(heights), heights], axis=1
        )
        weights = support_weight(self.field.query(points))
        self.assertTrue(bool(np.all(np.diff(weights) < 0.0)))
        self.assertTrue(bool(np.all((weights >= 0.0) & (weights <= 1.0))))
        # A larger sigma factor is strictly more permissive off the surface.
        loose = support_weight(self.field.query(points), sigma_perp_factor=2.0)
        self.assertTrue(bool(np.all(loose[1:] > weights[1:])))

    def test_query_k_greater_than_one_shapes(self) -> None:
        points = np.array([[0.0, 0.0, 0.1], [0.5, 0.5, -0.2]])
        query = self.field.query(points, k=3)
        self.assertEqual(query.d_perp.shape, (2, 3))
        self.assertEqual(query.normal.shape, (2, 3, 3))
        self.assertEqual(query.tangent_basis.shape, (2, 3, 3, 2))
        self.assertEqual(query.index.shape, (2, 3))
        np.testing.assert_allclose(query.d_perp[:, 0], [0.1, 0.2], atol=1e-9)
        self.assertEqual(support_weight(query).shape, (2, 3))
        # Neighbors are ordered by euclidean distance.
        self.assertTrue(bool(np.all(np.diff(query.euclidean, axis=1) >= -1e-12)))


# ---------------------------------------------------------------------------
# The case WP-3 exists for: sparse scan lines
# ---------------------------------------------------------------------------


class SparseScanLineTest(unittest.TestCase):
    """Dense along the scan line (0.1 m), sparse across it (0.5 m)."""

    def setUp(self) -> None:
        self.field = build_surface_field(_scan_line_plane(), knn=24)

    def test_field_still_recovers_the_plane(self) -> None:
        self.assertGreater(float(np.abs(self.field.normals[:, 2]).min()), 0.999)
        self.assertGreater(float(np.median(self.field.planarity)), 0.99)
        # local_spacing reports the *coarse* cross-line gap, not the fine
        # along-line pitch: that is the tolerance a surface test must use.
        self.assertAlmostEqual(float(np.median(self.field.local_spacing)), 0.5, places=6)

    def test_flush_gaussian_between_scan_lines(self) -> None:
        """A Gaussian exactly on the wall, halfway between two scan lines.

        Nearest-point distance says 0.25 m off the surface (a floater under the
        old 0.3 m gate is only a hair away); d_perp says 0 and the support
        weight says fully supported. This is the entire point of WP-3.
        """
        flush = np.array([[0.0, 0.25, 0.0]])
        query = self.field.query(flush)
        self.assertAlmostEqual(float(query.euclidean[0]), 0.25, places=9)
        self.assertAlmostEqual(float(query.d_perp[0]), 0.0, places=9)
        self.assertAlmostEqual(float(query.d_tangent[0]), 0.25, places=9)
        self.assertAlmostEqual(float(support_weight(query)[0]), 1.0, places=6)

    def test_true_floater_is_still_rejected(self) -> None:
        query = self.field.query(np.array([[0.0, 0.25, 1.0]]))
        self.assertAlmostEqual(float(query.d_perp[0]), 1.0, places=9)
        self.assertLess(float(support_weight(query)[0]), 0.05)

    def test_nearest_point_criterion_inverts_the_ranking(self) -> None:
        """Old metric ranks the flush Gaussian as *worse* than a real floater.

        The flush Gaussian is 0.25 m from its nearest sample but 0 m from the
        surface; the floater is 0.20 m from its nearest sample and 0.20 m from
        the surface. Sorting by ``euclidean`` puts the innocent Gaussian first
        in the floater list; sorting by ``d_perp`` gets the order right.
        """
        flush = self.field.query(np.array([[0.0, 0.25, 0.0]]))
        floater = self.field.query(np.array([[0.0, 0.0, 0.2]]))
        self.assertGreater(float(flush.euclidean[0]), float(floater.euclidean[0]))
        self.assertLess(float(flush.d_perp[0]), float(floater.d_perp[0]))
        self.assertGreater(
            float(support_weight(flush)[0]), float(support_weight(floater)[0])
        )

    def test_support_tolerance_follows_the_scan_line_gap(self) -> None:
        """Tolerance is the cross-line gap, so only real outliers collapse.

        A 0.2 m offset is genuinely unresolvable at 0.5 m sampling (weight stays
        high — refusing to over-claim is the point of the adaptive sigma), while
        a 1.5 m offset collapses.
        """
        near = support_weight(self.field.query(np.array([[0.0, 0.0, 0.2]])))[0]
        far = support_weight(self.field.query(np.array([[0.0, 0.0, 1.5]])))[0]
        self.assertGreater(float(near), 0.8)
        self.assertLess(float(far), 1e-3)


# ---------------------------------------------------------------------------
# Tilted surface
# ---------------------------------------------------------------------------


class TiltedPlaneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.field = build_surface_field(_tilted_plane(), knn=24)

    def test_normal_is_the_45_degree_plane_normal(self) -> None:
        dots = np.abs(self.field.normals.astype(np.float64) @ _TILT_NORMAL)
        self.assertGreater(float(np.median(dots)), 0.999)

    def test_dperp_matches_the_analytic_plane_distance(self) -> None:
        rng = np.random.default_rng(23)
        points = rng.uniform(-0.5, 0.5, size=(300, 3))
        query = self.field.query(points)
        analytic = np.abs(points @ _TILT_NORMAL)
        np.testing.assert_allclose(query.d_perp, analytic, atol=1e-6)
        # Euclidean overstates the surface distance for every off-normal query.
        self.assertTrue(bool(np.all(query.euclidean >= query.d_perp - 1e-9)))

    def test_vertical_offset_is_shortened_by_the_tilt(self) -> None:
        # Offsetting straight up by h on a 45 deg wall is only h/sqrt(2) of
        # actual surface separation; the euclidean metric on a dense cloud
        # happens to agree here, so the two criteria must match.
        height = 0.2
        query = self.field.query(np.array([[0.0, 0.0, height]]))
        self.assertAlmostEqual(
            float(query.d_perp[0]), height / np.sqrt(2.0), places=6
        )
        self.assertAlmostEqual(
            float(query.euclidean[0]), height / np.sqrt(2.0), delta=0.01
        )

    def test_normal_offset_puts_everything_in_dperp(self) -> None:
        for offset in (0.05, 0.2, 0.5):
            query = self.field.query((offset * _TILT_NORMAL)[None, :])
            self.assertAlmostEqual(float(query.d_perp[0]), offset, places=6)
            self.assertAlmostEqual(float(query.d_tangent[0]), 0.0, places=3)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class PersistenceTest(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        cloud = _grid_plane(noise=0.005, extent=1.5)
        field = build_surface_field(cloud, knn=16)
        probes = np.array([[0.0, 0.0, 0.3], [0.31, -0.22, 0.0], [1.0, 1.0, -0.7]])
        before = field.query(probes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface_field.npz"
            field.save(path)
            restored = LidarSurfaceField.load(path)
        self.assertEqual(len(restored), len(field))
        self.assertEqual(restored.knn, field.knn)
        self.assertAlmostEqual(restored.neighbor_radius_m, field.neighbor_radius_m)
        np.testing.assert_allclose(restored.xyz, field.xyz)
        np.testing.assert_array_equal(restored.normals, field.normals)
        np.testing.assert_array_equal(restored.tangent_basis, field.tangent_basis)
        np.testing.assert_array_equal(restored.planarity, field.planarity)
        np.testing.assert_array_equal(restored.roughness, field.roughness)
        np.testing.assert_array_equal(restored.local_spacing, field.local_spacing)
        np.testing.assert_array_equal(restored.confidence, field.confidence)
        after = restored.query(probes)
        np.testing.assert_allclose(after.d_perp, before.d_perp)
        np.testing.assert_allclose(after.d_tangent, before.d_tangent)
        np.testing.assert_allclose(support_weight(after), support_weight(before))

    def test_constructor_validates_shapes(self) -> None:
        field = build_surface_field(_grid_plane(), knn=16)
        with self.assertRaises(ValueError):
            LidarSurfaceField(
                field.xyz,
                field.normals[:-1],
                field.tangent_basis,
                field.planarity,
                field.roughness,
                field.local_spacing,
                field.confidence,
            )
        with self.assertRaises(ValueError):
            LidarSurfaceField(
                field.xyz,
                field.normals,
                field.tangent_basis[:, :, :1],
                field.planarity,
                field.roughness,
                field.local_spacing,
                field.confidence,
            )


# ---------------------------------------------------------------------------
# Real data smoke test
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    REAL_LIDAR_PLY.exists(), f"real LiDAR cloud not available: {REAL_LIDAR_PLY}"
)
class RealCloudSmokeTest(unittest.TestCase):
    def test_build_and_query_the_ukgs_init_cloud(self) -> None:
        from tools.gaussian_health import read_ply_xyz

        cloud = read_ply_xyz(REAL_LIDAR_PLY)
        self.assertGreater(len(cloud), 100_000)
        started = time.perf_counter()
        field = build_surface_field(cloud, knn=24)
        elapsed = time.perf_counter() - started
        print(
            f"\n[real] {len(cloud)} points, knn=24 built in {elapsed:.2f}s; "
            f"neighbor_radius={field.neighbor_radius_m:.4f} m"
        )
        for name, values in (
            ("planarity", field.planarity),
            ("roughness_m", field.roughness),
            ("local_spacing_m", field.local_spacing),
            ("confidence", field.confidence),
        ):
            percentiles = np.percentile(values, [5, 50, 95])
            print(
                f"[real] {name:>16}: p5={percentiles[0]:.5f} "
                f"p50={percentiles[1]:.5f} p95={percentiles[2]:.5f}"
            )

        # Sanity, not tuning: an indoor scan is dominated by planar structure.
        self.assertGreater(float(np.median(field.planarity)), 0.5)
        self.assertTrue(bool(np.all(field.planarity >= 0.0)))
        self.assertTrue(bool(np.all(field.planarity <= 1.0)))
        self.assertTrue(bool(np.all(field.confidence >= 0.0)))
        self.assertTrue(bool(np.all(field.confidence <= 1.0)))
        # Metric plausibility: sub-metre sampling, sub-decimetre roughness.
        self.assertLess(float(np.median(field.local_spacing)), 1.0)
        self.assertGreater(float(np.median(field.local_spacing)), 0.0)
        self.assertLess(float(np.median(field.roughness)), 0.1)

        # Querying the cloud with itself must land exactly on the surface.
        rng = np.random.default_rng(3)
        sample = cloud[rng.choice(len(cloud), 5_000, replace=False)]
        query = field.query(sample)
        self.assertIsInstance(query, SurfaceQuery)
        np.testing.assert_allclose(query.euclidean, 0.0, atol=1e-9)
        np.testing.assert_allclose(query.d_perp, 0.0, atol=1e-9)

        # Points lifted 2 m along their own surface normal must be rejected.
        # A minority genuinely lands on a facing surface (2 m inside a room hits
        # the opposite wall / floor), so this is a distribution claim, not a
        # per-point one.
        lifted_query = field.query(sample + 2.0 * query.normal)
        weights = support_weight(lifted_query)
        print(
            f"[real] lifted 2 m: d_perp p50={np.median(lifted_query.d_perp):.4f} m  "
            f"support p50={np.median(weights):.3e}  "
            f"fraction(support>0.5)={float((weights > 0.5).mean()):.4f}"
        )
        self.assertGreater(float(np.median(lifted_query.d_perp)), 0.5)
        self.assertLess(float(np.median(weights)), 1e-3)
        self.assertLess(float((weights > 0.5).mean()), 0.15)


if __name__ == "__main__":
    unittest.main()
