"""CPU tests for LiDAR normal / planarity regularization."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cloudstudio_3dgs.training.lidar_normals import (
    LidarNormalAnchors,
    NormalAlignmentConfig,
    NormalField,
    build_normal_field,
)


def _plane_cloud(n_side: int = 41, extent: float = 2.0, noise: float = 0.0) -> np.ndarray:
    """Regular 0.1 m grid on z = 0 (41 nodes hit integer coordinates)."""
    axis = np.linspace(-extent, extent, n_side)
    xx, yy = np.meshgrid(axis, axis)
    cloud = np.stack([xx.ravel(), yy.ravel(), np.zeros(n_side * n_side)], axis=1)
    if noise > 0.0:
        rng = np.random.default_rng(7)
        cloud[:, 2] += rng.normal(scale=noise, size=len(cloud))
    return cloud


def _volumetric_cloud(count: int = 4000, extent: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.uniform(-extent, extent, size=(count, 3))


def _quat_about_x(angle_rad: float) -> list[float]:
    """wxyz quaternion for a rotation about the +x axis."""
    return [math.cos(angle_rad / 2.0), math.sin(angle_rad / 2.0), 0.0, 0.0]


def _params(
    means: list[list[float]],
    scales_log: list[list[float]],
    quats: list[list[float]],
    *,
    quats_grad: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor(means, dtype=torch.float32),
        "scales": torch.tensor(scales_log, dtype=torch.float32),
        "quats": torch.tensor(quats, dtype=torch.float32, requires_grad=quats_grad),
    }


class BuildNormalFieldTest(unittest.TestCase):
    def test_plane_normals_are_z_and_planarity_high(self) -> None:
        field = build_normal_field(_plane_cloud(noise=1e-4), knn=16)
        alignment = np.abs(field.normals[:, 2])
        self.assertGreater(float(alignment.min()), 0.99)
        self.assertGreater(float(np.median(field.planarity)), 0.95)
        self.assertTrue(np.all(field.planarity >= 0.0))
        self.assertTrue(np.all(field.planarity <= 1.0))
        # Unit normals.
        norms = np.linalg.norm(field.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_volumetric_cloud_planarity_low(self) -> None:
        # Finite-k sampling noise keeps uniform clouds well below plane-level
        # planarity (empirically ~0.48 at k=16) but never near 1.
        field = build_normal_field(_volumetric_cloud(), knn=16)
        self.assertLess(float(np.median(field.planarity)), 0.6)
        plane = build_normal_field(_plane_cloud(noise=1e-4), knn=16)
        self.assertLess(
            float(np.median(field.planarity)),
            float(np.median(plane.planarity)) - 0.3,
        )

    def test_degenerate_duplicated_points_get_zero_planarity(self) -> None:
        cloud = np.zeros((10, 3))
        field = build_normal_field(cloud, knn=4)
        np.testing.assert_allclose(field.planarity, 0.0)

    def test_query_nearest_distance_and_normal(self) -> None:
        field = build_normal_field(_plane_cloud(), knn=16)
        distance, normal, planarity, index = field.query(
            np.array([[0.0, 0.0, 0.5], [1.0, -1.0, -0.25]])
        )
        # Query points sit directly above/below grid nodes of z = 0.
        np.testing.assert_allclose(distance, [0.5, 0.25], atol=1e-9)
        np.testing.assert_allclose(np.abs(normal[:, 2]), 1.0, atol=1e-4)
        self.assertEqual(planarity.shape, (2,))
        self.assertEqual(index.shape, (2,))
        np.testing.assert_allclose(
            field.xyz[index][:, :2], [[0.0, 0.0], [1.0, -1.0]], atol=1e-9
        )

    def test_batching_matches_single_batch(self) -> None:
        cloud = _volumetric_cloud(count=500)
        small = build_normal_field(cloud, knn=8, batch_size=64)
        full = build_normal_field(cloud, knn=8, batch_size=10_000)
        np.testing.assert_allclose(small.normals, full.normals, atol=1e-6)
        np.testing.assert_allclose(small.planarity, full.planarity, atol=1e-6)

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_normal_field(np.zeros((5, 2)))
        with self.assertRaises(ValueError):
            build_normal_field(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            build_normal_field(np.zeros((10, 3)), knn=2)

    def test_save_load_round_trip(self) -> None:
        field = build_normal_field(_plane_cloud(n_side=15), knn=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            field.save(path)
            loaded = NormalField.load(path)
            np.testing.assert_array_equal(loaded.xyz, field.xyz)
            np.testing.assert_array_equal(loaded.normals, field.normals)
            np.testing.assert_array_equal(loaded.planarity, field.planarity)
            self.assertEqual(loaded.knn, field.knn)
            probe = np.array([[0.3, -0.2, 0.4]])
            for original, reloaded in zip(
                field.query(probe), loaded.query(probe)
            ):
                np.testing.assert_allclose(original, reloaded)


class NormalAlignmentConfigTest(unittest.TestCase):
    def test_defaults_valid_and_round_trip(self) -> None:
        config = NormalAlignmentConfig()
        config.validate()
        self.assertFalse(config.enabled)
        payload = config.to_dict()
        self.assertEqual(payload["weight_align"], 0.01)
        self.assertEqual(payload["planarity_gate"], 0.6)
        self.assertEqual(payload["refresh_every"], 500)
        self.assertEqual(payload["flatten_mode"], "absolute_m")
        self.assertEqual(payload["flatten_target_m"], 0.02)
        self.assertEqual(payload["flatten_ratio_target"], 0.15)
        self.assertEqual(payload["weight_point_to_plane"], 0.0)
        self.assertEqual(payload["point_to_plane_huber_delta_m"], 0.02)

    def test_rejects_invalid_values(self) -> None:
        bad = [
            NormalAlignmentConfig(weight_align=-1.0),
            NormalAlignmentConfig(weight_flatten=-0.1),
            NormalAlignmentConfig(weight_point_to_plane=-0.1),
            NormalAlignmentConfig(planarity_gate=1.5),
            NormalAlignmentConfig(max_anchor_distance_m=0.0),
            NormalAlignmentConfig(refresh_every=0),
            NormalAlignmentConfig(flatten_mode="unknown"),
            NormalAlignmentConfig(flatten_target_m=-0.01),
            NormalAlignmentConfig(flatten_ratio_target=0.0),
            NormalAlignmentConfig(flatten_ratio_target=1.0),
            NormalAlignmentConfig(point_to_plane_huber_delta_m=0.0),
        ]
        for config in bad:
            with self.assertRaises(ValueError):
                config.validate()


class LidarNormalAnchorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.field = build_normal_field(_plane_cloud(), knn=16)
        self.config = NormalAlignmentConfig(enabled=True)

    def _anchors(self, config: NormalAlignmentConfig | None = None) -> LidarNormalAnchors:
        return LidarNormalAnchors(self.field, config or self.config)

    def test_flat_gaussian_on_plane_has_near_zero_align(self) -> None:
        # Shortest axis (z column, smallest log scale) parallel to plane normal.
        params = _params(
            [[0.0, 0.0, 0.01]], [[-3.0, -3.0, -5.0]], [[1.0, 0.0, 0.0, 0.0]]
        )
        anchors = self._anchors()
        anchors.refresh(params["means"])
        result = anchors.loss(params)
        self.assertEqual(result["anchored_count"], 1)
        self.assertFalse(result["stale"])
        self.assertLess(float(result["align_raw"]), 1e-6)
        self.assertEqual(float(result["flatten"]), 0.0)

    def test_sign_symmetry_min_axis_antiparallel_to_normal(self) -> None:
        # 180 degrees about x maps +z to -z: still aligned for unoriented n.
        params = _params(
            [[0.0, 0.0, 0.01]], [[-3.0, -3.0, -5.0]], [_quat_about_x(math.pi)]
        )
        anchors = self._anchors()
        anchors.refresh(params["means"])
        result = anchors.loss(params)
        self.assertEqual(result["anchored_count"], 1)
        self.assertLess(float(result["align_raw"]), 1e-6)

    def test_tilted_gaussian_has_positive_align_and_quat_gradient(self) -> None:
        params = _params(
            [[0.0, 0.0, 0.01]],
            [[-3.0, -3.0, -5.0]],
            [_quat_about_x(math.pi / 4.0)],
            quats_grad=True,
        )
        anchors = self._anchors()
        anchors.refresh(params["means"])
        result = anchors.loss(params)
        # sin^2(45 deg) = 0.5 up to planarity weighting of the mean.
        self.assertAlmostEqual(float(result["align_raw"].detach()), 0.5, places=3)
        self.assertGreater(float(result["total"].detach()), 0.0)
        result["total"].backward()
        gradient = params["quats"].grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_flatten_penalizes_thick_only(self) -> None:
        # Thick: exp(-2.3) ~ 0.100 m > 0.02 m target. Thin: exp(-5) ~ 0.0067 m.
        params = _params(
            [[0.0, 0.0, 0.01], [1.0, 1.0, 0.01]],
            [[-1.0, -1.0, -2.3], [-3.0, -3.0, -5.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        )
        anchors = self._anchors()
        anchors.refresh(params["means"])
        result = anchors.loss(params)
        self.assertEqual(result["anchored_count"], 2)
        self.assertGreater(float(result["flatten_raw"]), 0.0)
        # Thin-only cloud: flatten must be exactly zero (no reward for thinner).
        thin = _params(
            [[0.0, 0.0, 0.01]], [[-3.0, -3.0, -5.0]], [[1.0, 0.0, 0.0, 0.0]]
        )
        anchors.refresh(thin["means"])
        self.assertEqual(float(anchors.loss(thin)["flatten_raw"]), 0.0)

    def test_flatten_gradient_reaches_scales(self) -> None:
        scales = torch.tensor([[-1.0, -1.0, -2.3]], requires_grad=True)
        params = {
            "means": torch.tensor([[0.0, 0.0, 0.01]]),
            "scales": scales,
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
        anchors = self._anchors()
        anchors.refresh(params["means"])
        anchors.loss(params)["total"].backward()
        self.assertIsNotNone(scales.grad)
        # Only the shortest axis is pushed, and pushed to shrink (positive
        # gradient on the log scale under minimization).
        self.assertGreater(float(scales.grad[0, 2]), 0.0)
        self.assertEqual(float(scales.grad[0, 0]), 0.0)

    def test_ratio_flatten_penalizes_round_shapes_at_any_metric_scale(self) -> None:
        config = NormalAlignmentConfig(
            enabled=True,
            weight_align=0.0,
            weight_flatten=1.0,
            flatten_mode="tangent_ratio",
            flatten_ratio_target=0.15,
        )
        anchors = self._anchors(config)
        params = _params(
            [[0.0, 0.0, 0.01], [1.0, 1.0, 0.01]],
            [
                [math.log(0.01), math.log(0.01), math.log(0.005)],
                [math.log(0.10), math.log(0.10), math.log(0.01)],
            ],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        )
        anchors.refresh(params["means"])
        result = anchors.loss(params)
        # The 0.5-thickness round row is penalized; the geometrically ten
        # times larger but ratio-0.1 disk is accepted.
        self.assertGreater(float(result["flatten_raw"]), 0.0)
        disk = _params(
            [[0.0, 0.0, 0.01]],
            [[math.log(0.10), math.log(0.10), math.log(0.01)]],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        anchors.refresh(disk["means"])
        self.assertEqual(float(anchors.loss(disk)["flatten_raw"]), 0.0)

    def test_point_to_plane_tether_allows_tangent_motion_but_pushes_normal_motion(self) -> None:
        config = NormalAlignmentConfig(
            enabled=True,
            weight_align=0.0,
            weight_flatten=0.0,
            weight_point_to_plane=1.0,
            point_to_plane_huber_delta_m=0.02,
        )
        anchors = self._anchors(config)
        anchors.refresh(torch.tensor([[0.0, 0.0, 0.0]]))

        tangent_means = torch.tensor([[0.05, 0.0, 0.0]], requires_grad=True)
        tangent = {
            "means": tangent_means,
            "scales": torch.tensor([[-3.0, -3.0, -5.0]]),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
        tangent_loss = anchors.loss(tangent)
        self.assertLess(
            float(tangent_loss["point_to_plane_raw"].detach()), 1e-7
        )

        normal_means = torch.tensor([[0.05, 0.0, 0.01]], requires_grad=True)
        normal = {**tangent, "means": normal_means}
        normal_loss = anchors.loss(normal)
        self.assertGreater(
            float(normal_loss["point_to_plane_raw"].detach()), 0.0
        )
        normal_loss["total"].backward()
        self.assertAlmostEqual(float(normal_means.grad[0, 0]), 0.0, places=6)
        self.assertGreater(abs(float(normal_means.grad[0, 2])), 0.0)

    def test_max_anchor_distance_filters_far_gaussians(self) -> None:
        params = _params(
            [[0.0, 0.0, 1.0]], [[-3.0, -3.0, -5.0]], [_quat_about_x(math.pi / 4.0)]
        )
        anchors = self._anchors()
        anchors.refresh(params["means"])
        result = anchors.loss(params)
        self.assertEqual(result["anchored_count"], 0)
        self.assertEqual(float(result["total"]), 0.0)

    def test_planarity_gate_filters_non_planar_regions(self) -> None:
        field = build_normal_field(_volumetric_cloud(), knn=16)
        anchors = LidarNormalAnchors(
            field, NormalAlignmentConfig(enabled=True, planarity_gate=0.95)
        )
        params = _params(
            [[0.0, 0.0, 0.0]], [[-3.0, -3.0, -5.0]], [_quat_about_x(math.pi / 4.0)]
        )
        anchors.refresh(params["means"])
        result = anchors.loss(params)
        self.assertEqual(result["anchored_count"], 0)
        self.assertEqual(float(result["total"]), 0.0)

    def test_disabled_config_returns_zeros(self) -> None:
        anchors = self._anchors(NormalAlignmentConfig(enabled=False))
        params = _params(
            [[0.0, 0.0, 0.01]], [[-3.0, -3.0, -5.0]], [_quat_about_x(math.pi / 4.0)]
        )
        anchors.refresh(params["means"])
        result = anchors.loss(params)
        self.assertEqual(float(result["total"]), 0.0)
        self.assertEqual(result["anchored_count"], 0)

    def test_count_mismatch_returns_zero_and_stale(self) -> None:
        anchors = self._anchors()
        four = _params(
            [[0.0, 0.0, 0.01]] * 4,
            [[-3.0, -3.0, -5.0]] * 4,
            [_quat_about_x(math.pi / 4.0)] * 4,
        )
        anchors.refresh(four["means"])
        self.assertFalse(anchors.stale)
        five = _params(
            [[0.0, 0.0, 0.01]] * 5,
            [[-3.0, -3.0, -5.0]] * 5,
            [_quat_about_x(math.pi / 4.0)] * 5,
        )
        result = anchors.loss(five)
        self.assertTrue(result["stale"])
        self.assertTrue(anchors.stale)
        self.assertEqual(float(result["total"]), 0.0)
        self.assertEqual(result["anchored_count"], 0)
        # Re-refresh with the new population restores the constraint.
        anchors.refresh(five["means"])
        restored = anchors.loss(five)
        self.assertFalse(restored["stale"])
        self.assertEqual(restored["anchored_count"], 5)
        self.assertGreater(float(restored["total"]), 0.0)

    def test_loss_before_first_refresh_is_zero_and_stale(self) -> None:
        anchors = self._anchors()
        params = _params(
            [[0.0, 0.0, 0.01]], [[-3.0, -3.0, -5.0]], [[1.0, 0.0, 0.0, 0.0]]
        )
        result = anchors.loss(params)
        self.assertTrue(result["stale"])
        self.assertEqual(float(result["total"]), 0.0)

    def test_refresh_accepts_grad_bearing_means(self) -> None:
        means = torch.tensor([[0.0, 0.0, 0.01]], requires_grad=True)
        anchors = self._anchors()
        self.assertEqual(anchors.refresh(means), 1)

    def test_unnormalized_quaternions_are_handled(self) -> None:
        scaled = [value * 3.7 for value in _quat_about_x(math.pi / 4.0)]
        params = _params([[0.0, 0.0, 0.01]], [[-3.0, -3.0, -5.0]], [scaled])
        anchors = self._anchors()
        anchors.refresh(params["means"])
        self.assertAlmostEqual(
            float(anchors.loss(params)["align_raw"]), 0.5, places=3
        )


    def test_visible_scope_evaluates_only_the_rows_the_view_rasterized(self) -> None:
        # Row 0 is a clean surfel; row 1 is tilted 45 degrees and thick.
        params = _params(
            [[0.0, 0.0, 0.01], [0.2, 0.1, 0.01]],
            [[-3.0, -3.0, -5.0], [-3.0, -3.0, -3.2]],
            [[1.0, 0.0, 0.0, 0.0], _quat_about_x(math.pi / 4.0)],
        )
        anchors = self._anchors(NormalAlignmentConfig(enabled=True, scope="visible_current_view"))
        anchors.refresh(params["means"])
        everything = anchors.loss(params)
        only_clean = anchors.loss(params, visibility_mask=torch.tensor([True, False]))
        only_tilted = anchors.loss(params, visibility_mask=torch.tensor([False, True]))
        self.assertEqual(everything["anchored_count"], 2)
        self.assertEqual(only_clean["anchored_count"], 1)
        self.assertLess(float(only_clean["align_raw"]), 1e-6)
        self.assertAlmostEqual(float(only_tilted["align_raw"]), 0.5, places=3)
        # The weighted mean is over the rows evaluated: the tilted row alone weighs twice
        # what it weighs averaged with the clean one.
        self.assertAlmostEqual(
            float(only_tilted["align_raw"]), 2.0 * float(everything["align_raw"]), places=3
        )

    def test_visible_scope_sends_no_gradient_to_hidden_rows(self) -> None:
        params = _params(
            [[0.0, 0.0, 0.01], [0.2, 0.1, 0.01]],
            [[-3.0, -3.0, -3.2], [-3.0, -3.0, -3.2]],
            [_quat_about_x(math.pi / 4.0), _quat_about_x(math.pi / 4.0)],
            quats_grad=True,
        )
        params["scales"].requires_grad_(True)
        anchors = self._anchors(NormalAlignmentConfig(enabled=True, scope="visible_current_view"))
        anchors.refresh(params["means"])
        anchors.loss(params, visibility_mask=torch.tensor([True, False]))["total"].backward()
        self.assertGreater(float(params["quats"].grad[0].abs().sum()), 0.0)
        self.assertEqual(float(params["quats"].grad[1].abs().sum()), 0.0)
        self.assertEqual(float(params["scales"].grad[1].abs().sum()), 0.0)

    def test_no_visible_anchor_is_a_plain_zero(self) -> None:
        params = _params([[0.0, 0.0, 0.01]], [[-3.0, -3.0, -5.0]], [[1.0, 0.0, 0.0, 0.0]])
        anchors = self._anchors()
        anchors.refresh(params["means"])
        result = anchors.loss(params, visibility_mask=torch.tensor([False]))
        self.assertEqual(result["anchored_count"], 0)
        self.assertEqual(float(result["total"]), 0.0)
        self.assertFalse(result["stale"])

    def test_a_mask_of_the_wrong_length_is_refused(self) -> None:
        params = _params([[0.0, 0.0, 0.01]], [[-3.0, -3.0, -5.0]], [[1.0, 0.0, 0.0, 0.0]])
        anchors = self._anchors()
        anchors.refresh(params["means"])
        with self.assertRaises(ValueError):
            anchors.loss(params, visibility_mask=torch.tensor([True, False]))

    def test_scope_is_validated_and_absent_from_the_contract_by_default(self) -> None:
        self.assertNotIn("scope", NormalAlignmentConfig(enabled=True).to_dict())
        self.assertEqual(
            NormalAlignmentConfig(enabled=True, scope="visible_current_view").to_dict()["scope"],
            "visible_current_view",
        )
        with self.assertRaises(ValueError):
            NormalAlignmentConfig(scope="some").validate()


class BoundedAnchorQueryTest(unittest.TestCase):
    """The re-anchor search stops at the distance beyond which a row is invalid anyway."""

    def setUp(self) -> None:
        self.field = build_normal_field(_plane_cloud(), knn=16)
        rng = np.random.default_rng(5)
        near = np.column_stack([rng.uniform(-0.9, 0.9, 400), rng.uniform(-0.9, 0.9, 400), rng.uniform(-0.08, 0.08, 400)])
        far = np.column_stack([rng.uniform(-0.9, 0.9, 200), rng.uniform(-0.9, 0.9, 200), rng.uniform(0.15, 3.0, 200)])
        self.points = np.vstack([near, far])

    def test_valid_rows_get_the_same_anchor_as_the_unbounded_search(self) -> None:
        distance, normal, planarity, index = self.field.query(self.points, k=1)
        b_distance, b_normal, b_planarity, b_index = self.field.query(self.points, k=1, max_distance=0.10)
        inside = distance <= 0.10
        self.assertGreater(int(inside.sum()), 100)
        self.assertGreater(int((~inside).sum()), 100)
        np.testing.assert_array_equal(b_index[inside], index[inside])
        np.testing.assert_array_equal(b_distance[inside], distance[inside])
        np.testing.assert_array_equal(b_normal[inside], normal[inside])
        np.testing.assert_array_equal(b_planarity[inside], planarity[inside])
        self.assertTrue(np.all(np.isinf(b_distance[~inside])))
        self.assertTrue(np.all(b_index[~inside] == 0))

    def test_refresh_marks_exactly_the_rows_the_unbounded_search_would(self) -> None:
        config = NormalAlignmentConfig(enabled=True, max_anchor_distance_m=0.10)
        anchors = LidarNormalAnchors(self.field, config)
        means = torch.tensor(self.points, dtype=torch.float32)
        count = anchors.refresh(means)
        distance, _, planarity, index = self.field.query(means.to(torch.float64).numpy(), k=1)
        expected = (distance <= 0.10) & (planarity >= config.planarity_gate)
        self.assertEqual(anchors.valid.tolist(), expected.tolist())
        self.assertEqual(count, int(expected.sum()))
        valid = torch.from_numpy(expected)
        self.assertTrue(
            torch.equal(
                anchors.anchor_position[valid],
                torch.from_numpy(self.field.xyz[index].astype(np.float32))[valid],
            )
        )

    def test_a_non_positive_bound_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.field.query(self.points, k=1, max_distance=0.0)


class AnchorsThroughRefineTest(unittest.TestCase):
    """The anchors ride a refine event in the strategy state instead of being re-queried."""

    def setUp(self) -> None:
        self.field = build_normal_field(_plane_cloud(), knn=16)
        self.params = _params(
            [[0.0, 0.0, 0.01], [0.2, 0.1, 0.01], [-0.3, 0.2, 0.9]],
            [[-3.0, -3.0, -5.0]] * 3,
            [[1.0, 0.0, 0.0, 0.0]] * 3,
        )

    def _anchors(self, carry: bool = True) -> LidarNormalAnchors:
        anchors = LidarNormalAnchors(
            self.field, NormalAlignmentConfig(enabled=True, carry_through_refine=carry)
        )
        anchors.refresh(self.params["means"])
        return anchors

    def test_duplicate_split_and_remove_keep_every_row_on_its_own_anchor(self) -> None:
        anchors = self._anchors()
        valid_before = anchors.valid.clone()
        self.assertEqual(valid_before.tolist(), [True, True, False])  # row 2 is 0.9 m off the plane
        state: dict = {"grad2d": torch.zeros(3)}
        self.assertTrue(anchors.attach(state, 3, "cpu"))
        # What gsplat's ops do to every per-gaussian tensor in the state:
        for key, value in list(state.items()):
            duplicated = torch.cat((value, value[torch.tensor([1])]))  # clone row 1 -> row 3
            rest = torch.tensor([0, 2, 3])  # split row 1 away: survivors, then its two children
            split = torch.cat((duplicated[rest], duplicated[torch.tensor([1])].repeat([2] + [1] * (value.dim() - 1))))
            state[key] = split[torch.tensor([True, False, True, True, True])]  # remove the far row
        self.assertTrue(anchors.detach(state, 4))
        self.assertFalse(any(key.startswith("_cloudstudio_normal_anchor_") for key in state))
        self.assertFalse(anchors.stale)
        self.assertEqual(anchors.anchored_count, 4)
        self.assertEqual(anchors.valid.tolist(), [True, True, True, True])
        # Rows 1-3 all descend from the original row 1 and carry its anchor.
        fresh = self._anchors(carry=False)
        for row in (1, 2, 3):
            self.assertTrue(torch.equal(anchors.anchor_normal[row], fresh.anchor_normal[1]))
            self.assertTrue(torch.equal(anchors.anchor_position[row], fresh.anchor_position[1]))
        # The loss runs on the carried anchors without a refresh.
        grown = _params(
            [[0.0, 0.0, 0.01], [0.2, 0.1, 0.01], [0.2, 0.1, 0.01], [0.2, 0.1, 0.01]],
            [[-3.0, -3.0, -5.0]] * 4,
            [[1.0, 0.0, 0.0, 0.0]] * 4,
        )
        result = anchors.loss(grown)
        self.assertFalse(result["stale"])
        self.assertEqual(result["anchored_count"], 4)

    def test_an_event_that_does_not_transform_the_anchors_leaves_them_stale(self) -> None:
        anchors = self._anchors()
        state: dict = {}
        self.assertTrue(anchors.attach(state, 3, "cpu"))
        self.assertFalse(anchors.detach(state, 5))  # population changed, tensors did not
        self.assertTrue(anchors.stale)
        self.assertEqual(state, {})

    def test_nothing_is_attached_when_off_stale_or_mismatched(self) -> None:
        self.assertFalse(self._anchors(carry=False).attach({}, 3, "cpu"))
        anchors = self._anchors()
        self.assertFalse(anchors.attach({}, 4, "cpu"))
        anchors.stale = True
        self.assertFalse(anchors.attach({}, 3, "cpu"))

    def test_a_cadence_refresh_replaces_carried_anchors(self) -> None:
        anchors = self._anchors()
        state: dict = {}
        anchors.attach(state, 3, "cpu")
        anchors.detach(state, 3)
        self.assertIsNone(anchors.distance)
        anchors.refresh(self.params["means"])
        self.assertIsNotNone(anchors.distance)
        self.assertEqual(anchors.anchored_count, 2)

    def test_the_contract_records_it_only_when_on(self) -> None:
        self.assertNotIn("carry_through_refine", NormalAlignmentConfig(enabled=True).to_dict())
        self.assertTrue(
            NormalAlignmentConfig(enabled=True, carry_through_refine=True).to_dict()["carry_through_refine"]
        )


if __name__ == "__main__":
    unittest.main()
