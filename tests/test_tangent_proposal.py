"""CPU-only tests for the WP-5 tangent-plane birth-site proposal.

No GPU is ever touched and the CUDA-heavy relocation op is mocked where it is
not needed. Do NOT mutate CUDA_VISIBLE_DEVICES at module level - unittest
discovery imports this file into the shared process and the hidden device would
break the GPU contract tests that run after it.

The cases that justify the design are :class:`FallbackTests` (a low-planarity
neighborhood must NOT be projected onto, because its normal is an artifact) and
:class:`IndexSpaceTests` (the override has to land on the appended rows, in the
same order gsplat concatenates them, with the optimizer state and every
lifecycle column still aligned).
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from cloudstudio_3dgs.geometry.lidar_surface_field import build_surface_field
from cloudstudio_3dgs.training.tangent_proposal import (
    ProposalConfig,
    TangentProposal,
    propose_positions,
    rotmat_to_quat_wxyz,
    surface_frame_matrices,
    update_proposal_telemetry,
)
from cloudstudio_3dgs.training.lidar_admission import AdmissionConfig, LidarAdmission

try:
    import torch
except ImportError:  # pragma: no cover - optional training dependency
    torch = None

if torch is not None:
    try:
        from cloudstudio_3dgs.training.error_weighted_mcmc import (
            ErrorScoreConfig,
            ErrorScoreState,
            ErrorWeightedMCMCStrategy,
            sample_add_weighted,
        )
        from cloudstudio_3dgs.training.gaussian_lifecycle import GaussianLifecycleState
        from gsplat.utils import normalized_quat_to_rotmat

        _IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - e.g. CUDA-only gsplat build
        ErrorScoreConfig = ErrorScoreState = ErrorWeightedMCMCStrategy = None
        GaussianLifecycleState = sample_add_weighted = None
        normalized_quat_to_rotmat = None
        _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = ImportError("torch is not installed")


def _requires_torch(test_case: unittest.TestCase) -> None:
    if torch is None:
        test_case.skipTest("torch is not installed")


def _requires_module(test_case: unittest.TestCase) -> None:
    if _IMPORT_ERROR is not None:
        test_case.skipTest(f"error_weighted_mcmc unavailable on this host: {_IMPORT_ERROR}")


# ---------------------------------------------------------------------------
# Synthetic clouds (same construction as tests/test_lidar_admission.py)
# ---------------------------------------------------------------------------


def _grid_plane(pitch: float = 0.05, extent: float = 1.0) -> np.ndarray:
    """Isotropic z = 0 grid of the given pitch over ``[-extent, extent]^2``."""
    axis = np.arange(-extent, extent + pitch * 0.5, pitch)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)


def _tilted_plane(pitch: float = 0.05, extent: float = 1.0) -> np.ndarray:
    """The same grid rotated so its normal is not an axis of the world frame.

    An axis-aligned plane hides sign/permutation bugs in the frame construction:
    every candidate rotation matrix is a signed permutation and several wrong
    ones still pass an "is the shortest axis vertical?" assertion.
    """
    plane = _grid_plane(pitch, extent)
    angle = 0.4
    rot = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    return plane @ rot.T


def _blob(count: int = 900, seed: int = 7) -> np.ndarray:
    """Isotropic noise ball: median planarity ~ 0.39, so most normals are noise.

    Measured on this fixture: only 3.2% of points reach planarity 0.6. It is a
    *statistical* fixture, not a uniformly bad one, which is why the fallback
    tests select the sub-planar rows explicitly instead of asserting that the
    whole batch was rejected.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.25, size=(count, 3))


def _noisy_plane(
    pitch: float = 0.05, extent: float = 1.0, scale: float = 0.005, seed: int = 3
) -> np.ndarray:
    """The grid plus 5 mm of noise: measured roughness ~ 0.0046 m.

    A perfectly flat synthetic plane has ``roughness == 0`` exactly, where the
    thickness floor wins outright and ``thickness_factor`` has nothing to
    scale. Anything asserting on the thickness *factor* needs a surface with
    real roughness, like real LiDAR always has.
    """
    plane = _grid_plane(pitch, extent)
    return plane + np.random.default_rng(seed).normal(scale=scale, size=plane.shape)


def _field(cloud: np.ndarray):
    return build_surface_field(cloud)


def _config(**kwargs) -> ProposalConfig:
    return ProposalConfig(**{"enabled": True, **kwargs})


def _cpu_compute_relocation(*, opacities, scales, ratios, binoms):
    """CPU stand-in for gsplat's CUDA-only ``compute_relocation``.

    Only two properties matter to these tests, and the real kernel has both:
    the returned scales are (a) per *sampled position*, not per parent, and
    (b) different from the parent's. The position-dependent factor makes (a)
    observable - two entries sampling the same parent get different values,
    which is exactly what turns ``p[idx] = v; p[idx]`` into a lossy
    scatter-then-gather.
    """
    factor = 1.0 + 0.1 * torch.arange(
        len(scales), dtype=scales.dtype, device=scales.device
    ).unsqueeze(1)
    return opacities * 0.9, scales * factor


def _patch_relocation():
    return mock.patch(
        "cloudstudio_3dgs.training.error_weighted_mcmc.compute_relocation",
        side_effect=_cpu_compute_relocation,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ProposalConfigTests(unittest.TestCase):
    def test_defaults_are_valid_and_disabled(self) -> None:
        config = ProposalConfig()
        config.validate()
        self.assertFalse(config.enabled)
        self.assertFalse(config.active)
        self.assertEqual(config.mode, "tangent")

    def test_off_mode_is_inactive_even_when_enabled(self) -> None:
        self.assertFalse(ProposalConfig(enabled=True, mode="off").active)
        self.assertTrue(ProposalConfig(enabled=True, mode="tangent").active)

    def test_invalid_values_are_rejected(self) -> None:
        cases = [
            {"mode": "hard"},
            {"planarity_gate": 1.5},
            {"planarity_gate": -0.1},
            {"support_gate": 0.0},
            {"support_gate": 1.5},
            {"support_tangent_factor": 0.0},
            {"sigma_perp_factor": 0.0},
            {"tangent_sigma_factor": -1.0},
            {"normal_offset_factor": -0.5},
            {"thickness_factor": 0.0},
            {"min_thickness_m": 0.0},
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    ProposalConfig(**kwargs).validate()

    def test_to_dict_roundtrips_all_fields(self) -> None:
        config = ProposalConfig(
            enabled=True,
            mode="tangent",
            planarity_gate=0.7,
            support_gate=0.2,
            sigma_perp_factor=1.5,
            tangent_sigma_factor=0.25,
            normal_offset_factor=0.05,
            init_shortest_axis=False,
            thickness_factor=0.3,
            min_thickness_m=2e-3,
        )
        payload = config.to_dict()
        self.assertEqual(ProposalConfig(**payload), config)


# ---------------------------------------------------------------------------
# Position: projection onto the tangent plane
# ---------------------------------------------------------------------------


class ProjectionTests(unittest.TestCase):
    """Candidates land on the plane, scattered tangentially by sigma."""

    def setUp(self) -> None:
        self.field = _field(_grid_plane())

    def test_candidates_are_projected_onto_the_plane(self) -> None:
        rng = np.random.default_rng(0)
        parents = np.column_stack(
            [
                rng.uniform(-0.5, 0.5, 400),
                rng.uniform(-0.5, 0.5, 400),
                rng.uniform(-0.02, 0.02, 400),  # near, but not on, the surface
            ]
        )
        result = propose_positions(
            parents, self.field, _config(normal_offset_factor=0.0), np.random.default_rng(1)
        )
        self.assertTrue(bool(result.applied.all()))
        # normal_offset_factor = 0 means "snap flush"; the only residual is the
        # plane's own numerical thickness.
        self.assertLess(float(np.abs(result.means[:, 2]).max()), 1e-9)

    def test_tangential_scatter_matches_the_configured_sigma(self) -> None:
        parents = np.zeros((4000, 3))
        factor = 0.5
        result = propose_positions(
            parents,
            self.field,
            _config(tangent_sigma_factor=factor, normal_offset_factor=0.0),
            np.random.default_rng(3),
        )
        spacing = float(self.field.local_spacing[self.field.tree.query([[0, 0, 0]])[1][0]])
        expected = factor * spacing
        # In-plane radial spread of an isotropic 2-D normal: per-axis std is sigma.
        for axis in (0, 1):
            self.assertAlmostEqual(
                float(np.std(result.means[:, axis])), expected, delta=expected * 0.08
            )

    def test_zero_tangent_sigma_lands_exactly_on_the_surface_point(self) -> None:
        parents = np.array([[0.021, -0.013, 0.004]])
        result = propose_positions(
            parents,
            self.field,
            _config(tangent_sigma_factor=0.0, normal_offset_factor=0.0),
            np.random.default_rng(0),
        )
        nearest = self.field.xyz[self.field.tree.query(parents)[1][0]]
        np.testing.assert_allclose(result.means[0], nearest, atol=1e-9)

    def test_strict_lidar_birth_falls_back_when_scatter_leaves_measured_patch(self) -> None:
        parents = np.zeros((64, 3))
        result = propose_positions(
            parents,
            self.field,
            _config(
                tangent_sigma_factor=100.0,
                normal_offset_factor=0.0,
                reject_unsupported_births=True,
            ),
            np.random.default_rng(9),
        )
        self.assertTrue(bool(result.applied.all()))
        self.assertGreater(int(result.fallback_to_parent.sum()), 0)
        np.testing.assert_array_equal(
            result.means[result.fallback_to_parent],
            parents[result.fallback_to_parent],
        )

    def test_parent_gate_rejects_unbounded_plane_extension(self) -> None:
        proposal = TangentProposal(
            self.field,
            _config(reject_unsupported_births=True),
            seed=1,
        )
        import torch

        eligible = proposal.eligible_parent_mask(
            torch.tensor([[0.0, 0.0, 0.0], [20.0, 20.0, 0.0]])
        )
        self.assertEqual(eligible.tolist(), [True, False])

    def test_proposal_is_a_pass_through_when_inactive(self) -> None:
        parents = np.array([[0.02, 0.02, 0.01], [0.1, -0.1, -0.03]])
        for config in (
            ProposalConfig(enabled=False),
            ProposalConfig(enabled=True, mode="off"),
        ):
            with self.subTest(mode=config.mode, enabled=config.enabled):
                result = propose_positions(
                    parents, self.field, config, np.random.default_rng(0)
                )
                np.testing.assert_array_equal(result.means, parents)
                self.assertFalse(bool(result.applied.any()))


# ---------------------------------------------------------------------------
# Normal offset clamping
# ---------------------------------------------------------------------------


class NormalOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.field = _field(_grid_plane())

    def test_offset_is_clamped_to_the_configured_fraction_of_spacing(self) -> None:
        factor = 0.1
        # Measured on this fixture: local_spacing = 0.10 m, so the cap is 0.01 m
        # and support at |z| = 0.2 is 1.8e-2. Well inside the patch tangentially,
        # twenty caps off along the normal.
        parents = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2], [0.1, 0.1, 0.2]])
        result = propose_positions(
            parents,
            self.field,
            _config(
                normal_offset_factor=factor,
                tangent_sigma_factor=0.0,
                support_gate=1e-3,  # isolate the clamp from the support gate
            ),
            np.random.default_rng(0),
        )
        self.assertTrue(bool(result.applied.all()))
        spacing = self.field.local_spacing[result.anchor_index].astype(np.float64)
        cap = factor * spacing
        residual = np.abs(result.means[:, 2])
        self.assertTrue(bool(np.all(residual <= cap + 1e-9)))
        # And it saturates: these parents were far enough to hit the cap.
        np.testing.assert_allclose(residual, cap, atol=1e-9)

    def test_the_parents_side_of_the_surface_is_preserved(self) -> None:
        parents = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]])
        result = propose_positions(
            parents,
            self.field,
            _config(tangent_sigma_factor=0.0, support_gate=1e-3),
            np.random.default_rng(0),
        )
        self.assertGreater(float(result.means[0, 2]), 0.0)
        self.assertLess(float(result.means[1, 2]), 0.0)

    def test_a_small_offset_is_kept_verbatim_rather_than_snapped(self) -> None:
        # 1 mm off a 0.1 m local spacing: well inside a cap of 0.1 * spacing.
        parents = np.array([[0.0, 0.0, 0.001]])
        result = propose_positions(
            parents,
            self.field,
            _config(tangent_sigma_factor=0.0),
            np.random.default_rng(0),
        )
        self.assertAlmostEqual(float(result.means[0, 2]), 0.001, places=9)


# ---------------------------------------------------------------------------
# Pose: shortest axis aligned to the surface normal
# ---------------------------------------------------------------------------


class PoseInitializationTests(unittest.TestCase):
    """The generated quaternion must put the shortest axis on the normal."""

    def setUp(self) -> None:
        _requires_module(self)

    def _shortest_axis(self, quats: np.ndarray) -> np.ndarray:
        matrices = normalized_quat_to_rotmat(
            torch.as_tensor(quats, dtype=torch.float64)
        ).numpy()
        return matrices[:, :, 2]  # column 2 = local axis 2 = the shortest one

    def _run(self, cloud: np.ndarray, parents: np.ndarray, **kwargs):
        field = _field(cloud)
        return field, propose_positions(
            parents,
            field,
            _config(**kwargs),
            np.random.default_rng(11),
            parent_quats=np.tile([1.0, 0.0, 0.0, 0.0], (len(parents), 1)),
            parent_log_scales=np.log(np.tile([0.05, 0.03, 0.02], (len(parents), 1))),
        )

    def test_shortest_axis_aligns_with_the_surface_normal(self) -> None:
        parents = np.column_stack(
            [
                np.linspace(-0.4, 0.4, 60),
                np.linspace(0.4, -0.4, 60),
                np.full(60, 0.01),
            ]
        )
        field, result = self._run(_grid_plane(), parents)
        self.assertTrue(bool(result.applied.all()))
        axis = self._shortest_axis(result.quats)
        normals = field.normals[result.anchor_index].astype(np.float64)
        # Symmetric under n -> -n: a Gaussian axis is an unoriented direction.
        alignment = np.abs(np.einsum("mi,mi->m", axis, normals))
        self.assertGreater(float(alignment.min()), 1.0 - 1e-9)

    def test_alignment_holds_on_a_plane_that_is_not_axis_aligned(self) -> None:
        cloud = _tilted_plane()
        rng = np.random.default_rng(5)
        parents = cloud[rng.choice(len(cloud), 80, replace=False)] + rng.normal(
            scale=0.005, size=(80, 3)
        )
        field, result = self._run(cloud, parents)
        self.assertTrue(bool(result.applied.all()))
        axis = self._shortest_axis(result.quats)
        normals = field.normals[result.anchor_index].astype(np.float64)
        alignment = np.abs(np.einsum("mi,mi->m", axis, normals))
        # 1e-6 rather than 1e-9: the tangent basis is stored as float32.
        self.assertGreater(float(alignment.min()), 1.0 - 1e-6)

    def test_generated_quaternions_are_unit_and_proper_rotations(self) -> None:
        cloud = _tilted_plane()
        parents = cloud[:64] + 0.002
        _field_, result = self._run(cloud, parents)
        norms = np.linalg.norm(result.quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)
        matrices = normalized_quat_to_rotmat(
            torch.as_tensor(result.quats, dtype=torch.float64)
        ).numpy()
        np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=1e-9)

    def test_shortest_log_scale_is_the_smallest_of_the_three(self) -> None:
        parents = np.zeros((32, 3))
        _field_, result = self._run(_grid_plane(), parents)
        self.assertTrue(bool(result.applied.all()))
        smallest = result.log_scales.argmin(axis=1)
        np.testing.assert_array_equal(smallest, np.full(32, 2))

    def test_tangential_scales_are_the_parents_two_largest(self) -> None:
        parents = np.zeros((16, 3))
        _field_, result = self._run(_grid_plane(), parents)
        # The parent's scales were (0.05, 0.03, 0.02): the 0.02 axis is dropped
        # and replaced by the surface thickness.
        expected = np.tile(np.log([0.05, 0.03]), (16, 1))
        np.testing.assert_allclose(result.log_scales[:, :2], expected, atol=1e-12)

    def test_thinner_thickness_factor_gives_a_thinner_shortest_axis(self) -> None:
        # Needs a surface with measurable roughness: on a perfectly flat plane
        # every factor lands on the min_thickness_m floor by construction.
        cloud = _noisy_plane()
        parents = cloud[:8]
        _f1, thick = self._run(cloud, parents, thickness_factor=0.5)
        _f2, thin = self._run(cloud, parents, thickness_factor=0.25)
        self.assertTrue(bool(thick.applied.all()))
        self.assertLess(
            float(thin.log_scales[:, 2].max()), float(thick.log_scales[:, 2].min())
        )

    def test_pose_is_not_produced_when_the_switch_is_off(self) -> None:
        parents = np.zeros((4, 3))
        _field_, result = self._run(_grid_plane(), parents, init_shortest_axis=False)
        self.assertIsNone(result.quats)
        self.assertIsNone(result.log_scales)

    def test_missing_parent_arrays_disable_only_their_own_output(self) -> None:
        field = _field(_grid_plane())
        result = propose_positions(
            np.zeros((4, 3)),
            field,
            _config(),
            np.random.default_rng(0),
            parent_quats=np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        )
        self.assertIsNotNone(result.quats)
        self.assertIsNone(result.log_scales)


class RotationHelperTests(unittest.TestCase):
    """The quaternion conversion has to survive the 180-degree branch."""

    def setUp(self) -> None:
        _requires_module(self)

    def test_roundtrip_through_gsplats_own_converter(self) -> None:
        rng = np.random.default_rng(19)
        raw = rng.normal(size=(200, 3, 3))
        matrices = np.stack([np.linalg.qr(m)[0] for m in raw])
        matrices[np.linalg.det(matrices) < 0, :, 0] *= -1.0  # force det = +1
        quats = rotmat_to_quat_wxyz(matrices)
        recovered = normalized_quat_to_rotmat(
            torch.as_tensor(quats, dtype=torch.float64)
        ).numpy()
        np.testing.assert_allclose(recovered, matrices, atol=1e-9)

    def test_half_turns_are_exact(self) -> None:
        # trace = -1 here, which is exactly where the naive trace-only formula
        # divides by zero. Half of all surface normals produce such a frame.
        matrices = np.stack(
            [
                np.diag([1.0, -1.0, -1.0]),
                np.diag([-1.0, 1.0, -1.0]),
                np.diag([-1.0, -1.0, 1.0]),
            ]
        )
        quats = rotmat_to_quat_wxyz(matrices)
        self.assertTrue(np.isfinite(quats).all())
        recovered = normalized_quat_to_rotmat(
            torch.as_tensor(quats, dtype=torch.float64)
        ).numpy()
        np.testing.assert_allclose(recovered, matrices, atol=1e-12)

    def test_surface_frames_are_right_handed_whatever_the_normal_sign(self) -> None:
        field = _field(_tilted_plane())
        frames = surface_frame_matrices(field.tangent_basis, field.normals)
        np.testing.assert_allclose(np.linalg.det(frames), 1.0, atol=1e-6)
        # And the third column is still the normal up to sign.
        dots = np.abs(
            np.einsum("mi,mi->m", frames[:, :, 2], field.normals.astype(np.float64))
        )
        self.assertGreater(float(dots.min()), 1.0 - 1e-5)


# ---------------------------------------------------------------------------
# Fallback: do not project onto a surface we do not trust
# ---------------------------------------------------------------------------


class FallbackTests(unittest.TestCase):
    """Untrustworthy neighborhoods must clone in place, not be snapped."""

    def test_low_planarity_region_falls_back_to_the_parent_position(self) -> None:
        field = _field(_blob())
        rng = np.random.default_rng(2)
        parents = rng.normal(scale=0.2, size=(150, 3))
        result = propose_positions(parents, field, _config(), np.random.default_rng(0))
        planarity = field.planarity[result.anchor_index]
        sub_planar = planarity < ProposalConfig().planarity_gate
        # Sanity: the fixture really is mostly non-planar.
        self.assertGreater(float(sub_planar.mean()), 0.8)
        # Every sub-planar row cloned in place, verbatim. Snapping these onto a
        # normal that is a numerical accident is worse than not moving them.
        self.assertFalse(bool(result.applied[sub_planar].any()))
        np.testing.assert_array_equal(
            result.means[sub_planar], parents[sub_planar]
        )

    def test_planarity_gate_is_the_mechanism(self) -> None:
        field = _field(_blob())
        rng = np.random.default_rng(2)
        parents = rng.normal(scale=0.2, size=(150, 3))
        strict = propose_positions(
            parents, field, _config(planarity_gate=0.9), np.random.default_rng(0)
        )
        permissive = propose_positions(
            parents,
            field,
            _config(planarity_gate=0.0, support_gate=1e-9),
            np.random.default_rng(0),
        )
        self.assertFalse(bool(strict.applied.any()))
        self.assertGreater(float(permissive.applied.mean()), 0.8)

    def test_distant_floater_is_not_teleported_onto_the_plane(self) -> None:
        field = _field(_grid_plane())
        parents = np.array([[0.0, 0.0, 5.0]])  # planarity is perfect, support is not
        result = propose_positions(parents, field, _config(), np.random.default_rng(0))
        self.assertFalse(bool(result.applied[0]))
        np.testing.assert_array_equal(result.means, parents)

    def test_support_gate_is_the_mechanism_for_the_floater(self) -> None:
        field = _field(_grid_plane())
        # Measured support here is 1.2e-4: rejected by the default gate of 0.1,
        # accepted by a permissive one. (At z = 5 the support underflows to
        # exactly 0, which no positive gate can accept - hence the milder
        # floater for the mechanism test.)
        parents = np.array([[0.0, 0.0, 0.3]])
        strict = propose_positions(parents, field, _config(), np.random.default_rng(0))
        permissive = propose_positions(
            parents, field, _config(support_gate=1e-6), np.random.default_rng(0)
        )
        self.assertFalse(bool(strict.applied[0]))
        self.assertTrue(bool(permissive.applied[0]))

    def test_fallback_rows_keep_their_parent_pose_verbatim(self) -> None:
        _requires_module(self)
        field = _field(_grid_plane())
        # Far enough off the plane that support underflows: guaranteed rejects.
        parents = np.full((6, 3), 5.0)
        quats = np.tile([0.5, 0.5, 0.5, 0.5], (6, 1))
        log_scales = np.log(np.tile([0.02, 0.05, 0.03], (6, 1)))
        result = propose_positions(
            parents,
            field,
            _config(),
            np.random.default_rng(0),
            parent_quats=quats,
            parent_log_scales=log_scales,
        )
        self.assertFalse(bool(result.applied.any()))
        np.testing.assert_array_equal(result.quats, quats)
        # Crucially NOT re-sorted: a rejected row must be byte-identical.
        np.testing.assert_array_equal(result.log_scales, log_scales)

    def test_mixed_batch_applies_only_the_trustworthy_rows(self) -> None:
        field = _field(_grid_plane())
        parents = np.array(
            [
                [0.0, 0.0, 0.005],  # on the surface
                [0.0, 0.0, 6.0],  # floater, no support
                [0.2, -0.2, 0.002],  # on the surface
            ]
        )
        result = propose_positions(parents, field, _config(), np.random.default_rng(0))
        np.testing.assert_array_equal(result.applied, [True, False, True])
        np.testing.assert_array_equal(result.means[1], parents[1])
        self.assertFalse(np.allclose(result.means[0], parents[0]))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.field = _field(_grid_plane())
        self.parents = np.column_stack(
            [
                np.linspace(-0.5, 0.5, 200),
                np.linspace(0.5, -0.5, 200),
                np.zeros(200),
            ]
        )

    def test_same_seed_reproduces_the_batch(self) -> None:
        first = propose_positions(
            self.parents, self.field, _config(), np.random.default_rng(1234)
        )
        second = propose_positions(
            self.parents, self.field, _config(), np.random.default_rng(1234)
        )
        np.testing.assert_array_equal(first.means, second.means)

    def test_different_seeds_differ(self) -> None:
        first = propose_positions(
            self.parents, self.field, _config(), np.random.default_rng(1)
        )
        second = propose_positions(
            self.parents, self.field, _config(), np.random.default_rng(2)
        )
        self.assertFalse(np.array_equal(first.means, second.means))

    def test_rng_stream_does_not_depend_on_how_many_rows_were_rejected(self) -> None:
        """Rejected rows still consume their draws, so accepted rows are stable.

        Without this, turning one parent of a batch into a reject would shift
        every *later* candidate's scatter - making an A/B arm's accepted
        geometry depend on its rejects, which is untraceable in a diff.
        """
        good = self.parents[:5]
        spoiled = good.copy()
        spoiled[2] = [0.0, 0.0, 9.0]  # row 2 becomes a guaranteed reject
        clean = propose_positions(
            good, self.field, _config(), np.random.default_rng(77)
        )
        mixed = propose_positions(
            spoiled, self.field, _config(), np.random.default_rng(77)
        )
        np.testing.assert_array_equal(clean.applied, [True] * 5)
        np.testing.assert_array_equal(
            mixed.applied, [True, True, False, True, True]
        )
        # Rows after the reject are bit-identical: the reject consumed its draw.
        keep = np.array([0, 1, 3, 4])
        np.testing.assert_array_equal(clean.means[keep], mixed.means[keep])

    def test_empty_batch_is_not_an_error(self) -> None:
        result = propose_positions(
            np.zeros((0, 3)), self.field, _config(), np.random.default_rng(0)
        )
        self.assertEqual(int(result.applied.size), 0)
        self.assertEqual(result.means.shape, (0, 3))


# ---------------------------------------------------------------------------
# Event-driven admission maintenance (the WP-4 gap this WP closes)
# ---------------------------------------------------------------------------


class IncrementalAdmissionTests(unittest.TestCase):
    """extend()/on_relocate() keep the cache usable without a full refresh."""

    def setUp(self) -> None:
        _requires_torch(self)
        self.field = _field(_grid_plane())

    def _admission(self, means: np.ndarray) -> LidarAdmission:
        admission = LidarAdmission(self.field, AdmissionConfig(enabled=True))
        admission.refresh(torch.as_tensor(means, dtype=torch.float32))
        return admission

    def test_extend_appends_rows_and_leaves_the_old_ones_bit_identical(self) -> None:
        original = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, 0.0, 4.0]])
        admission = self._admission(original)
        before = admission.admission_weights().clone()
        new_means = torch.as_tensor(
            np.array([[0.2, 0.2, 0.0], [0.0, 0.0, 7.0]]), dtype=torch.float32
        )
        admission.extend(new_means=new_means)
        after = admission.admission_weights()
        self.assertIsNotNone(after)
        self.assertEqual(int(after.shape[0]), 5)
        self.assertTrue(torch.equal(after[:3], before))
        # The appended rows are real, distinct measurements, not a default.
        self.assertGreater(float(after[3]), float(after[4]))

    def test_extend_keeps_the_cache_usable_immediately(self) -> None:
        admission = self._admission(np.zeros((3, 3)))
        admission.extend(new_means=torch.zeros(2, 3))
        self.assertFalse(admission.stale)
        self.assertTrue(admission.in_sync(5))
        self.assertIsNotNone(admission.admission_weights(5))

    def test_extend_by_parent_indices_copies_the_parent_rows(self) -> None:
        original = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        admission = self._admission(original)
        before = admission.admission_weights().clone()
        admission.extend(parent_indices=torch.tensor([1, 0, 1]))
        after = admission.admission_weights()
        self.assertTrue(torch.equal(after[:2], before))
        self.assertTrue(torch.equal(after[2:], before[torch.tensor([1, 0, 1])]))

    def test_extend_refuses_a_stale_cache(self) -> None:
        admission = self._admission(np.zeros((3, 3)))
        admission.on_count_changed(5)
        with self.assertRaises(RuntimeError):
            admission.extend(new_means=torch.zeros(2, 3))

    def test_extend_requires_exactly_one_source(self) -> None:
        admission = self._admission(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            admission.extend()
        with self.assertRaises(ValueError):
            admission.extend(new_means=torch.zeros(1, 3), parent_indices=torch.tensor([0]))

    def test_extend_rejects_out_of_range_parents(self) -> None:
        admission = self._admission(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            admission.extend(parent_indices=torch.tensor([9]))

    def test_on_relocate_copies_the_source_rows(self) -> None:
        means = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.1, 0.1, 0.0]])
        admission = self._admission(means)
        before = admission.admission_weights().clone()
        admission.on_relocate(torch.tensor([1]), torch.tensor([2]))
        after = admission.admission_weights()
        self.assertAlmostEqual(float(after[1]), float(before[2]), places=7)
        self.assertAlmostEqual(float(after[0]), float(before[0]), places=7)
        self.assertTrue(admission.in_sync(3))

    def test_on_relocate_can_requery_supplied_positions(self) -> None:
        means = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        admission = self._admission(means)
        admission.on_relocate(
            torch.tensor([1]),
            torch.tensor([0]),
            means=torch.as_tensor(np.array([[0.0, 0.0, 8.0]]), dtype=torch.float32),
        )
        # Re-queried at the *given* position, so it is a floater, not a copy of
        # the on-surface source.
        self.assertAlmostEqual(
            float(admission.admission_weights()[1]),
            AdmissionConfig().weight_floor,
            places=6,
        )

    def test_on_relocate_gathers_before_it_scatters(self) -> None:
        # Slot 0 is both a destination and another slot's source; a naive
        # in-place loop would propagate the freshly written value.
        means = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 9.0]])
        admission = self._admission(means)
        before = admission.admission_weights().clone()
        admission.on_relocate(torch.tensor([0, 1]), torch.tensor([2, 0]))
        after = admission.admission_weights()
        self.assertAlmostEqual(float(after[1]), float(before[0]), places=7)

    def test_extend_updates_the_lifecycle_anchor_columns(self) -> None:
        _requires_module(self)
        admission = self._admission(np.zeros((3, 3)))
        lifecycle = GaussianLifecycleState(3)
        lifecycle.on_grow(torch.tensor([0, 1]), step=5)
        admission.extend(new_means=torch.zeros(2, 3), lifecycle=lifecycle)
        self.assertEqual(int(lifecycle.anchor_index.shape[0]), 5)
        self.assertTrue(bool((lifecycle.anchor_index >= 0).all()))

    def test_in_sync_reports_the_fallback_condition(self) -> None:
        admission = self._admission(np.zeros((3, 3)))
        self.assertTrue(admission.in_sync(3))
        self.assertFalse(admission.in_sync(4))
        admission.on_count_changed(3)
        self.assertFalse(admission.in_sync(3))


# ---------------------------------------------------------------------------
# Index space: the override must land on the appended rows
# ---------------------------------------------------------------------------


class IndexSpaceTests(unittest.TestCase):
    """sample_add_weighted keeps params, optimizer state and lifecycle aligned."""

    def setUp(self) -> None:
        _requires_module(self)

    def _params(self, means: np.ndarray) -> dict:
        n = len(means)
        return {
            "means": torch.nn.Parameter(torch.as_tensor(means, dtype=torch.float32)),
            "scales": torch.nn.Parameter(torch.log(torch.full((n, 3), 0.04))),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
            ),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5))),
        }

    def _optimizers(self, params: dict) -> dict:
        optimizers = {}
        for name, param in params.items():
            optimizer = torch.optim.Adam([param], lr=1e-3)
            # Materialize exp_avg / exp_avg_sq so the state actually has rows.
            param.grad = torch.ones_like(param)
            optimizer.step()
            optimizer.zero_grad()
            optimizers[name] = optimizer
        return optimizers

    def _proposal(self, cloud: np.ndarray | None = None) -> TangentProposal:
        return TangentProposal(
            _field(_grid_plane() if cloud is None else cloud), _config(), seed=99
        )

    def _run_sample_add(self, params, optimizers, n, probs, proposal, out):
        strategy = ErrorWeightedMCMCStrategy(cap_max=1000)
        binoms = strategy.initialize_state()["binoms"]
        with _patch_relocation():
            return sample_add_weighted(
                params=params,
                optimizers=optimizers,
                state={},
                n=n,
                binoms=binoms,
                probs=probs,
                proposal=proposal,
                proposal_out=out,
            )

    def test_only_the_appended_rows_are_overridden(self) -> None:
        means = np.array([[0.0, 0.0, 0.3], [0.2, 0.2, 0.004], [-0.3, 0.1, 0.002]])
        params = self._params(means)
        optimizers = self._optimizers(params)
        before = params["means"].detach().clone()
        out: dict = {}
        sampled = self._run_sample_add(
            params, optimizers, 4, torch.ones(3), self._proposal(), out
        )
        self.assertEqual(int(params["means"].shape[0]), 7)
        # The pre-existing rows are untouched by the proposal.
        torch.testing.assert_close(params["means"][:3], before)
        self.assertEqual(int(sampled.numel()), 4)

    def test_additive_births_preserve_parent_opacity_and_scale(self) -> None:
        params = self._params(np.zeros((3, 3)))
        optimizers = self._optimizers(params)
        before_opacity = params["opacities"].detach().clone()
        before_scale = params["scales"].detach().clone()
        proposal = TangentProposal(
            _field(_grid_plane()),
            _config(additive_births=True, birth_opacity=0.03),
            seed=7,
        )
        out: dict = {}
        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc._multinomial_sample",
            return_value=torch.tensor([1, 1, 0]),
        ):
            self._run_sample_add(
                params, optimizers, 3, torch.ones(3), proposal, out
            )
        torch.testing.assert_close(params["opacities"][:3], before_opacity)
        torch.testing.assert_close(params["scales"][:3], before_scale)
        torch.testing.assert_close(
            torch.sigmoid(params["opacities"][3:]), torch.full((3,), 0.03)
        )
        self.assertTrue(out["additive_births"])
        self.assertEqual(out["birth_opacity"], 0.03)

    def test_appended_rows_follow_the_sampled_parent_order(self) -> None:
        # Row 0 is a hopeless floater (falls back), rows 1-2 sit on the plane.
        means = np.array([[0.0, 0.0, 9.0], [0.2, 0.2, 0.003], [-0.3, 0.1, 0.002]])
        params = self._params(means)
        optimizers = self._optimizers(params)
        out: dict = {}
        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc._multinomial_sample",
            return_value=torch.tensor([0, 2, 1, 0]),
        ):
            self._run_sample_add(
                params, optimizers, 4, torch.ones(3), self._proposal(), out
            )
        applied = out["applied"]
        # Position j of the appended block corresponds to sampled_idxs[j].
        np.testing.assert_array_equal(
            applied.numpy(), np.array([False, True, True, False])
        )
        appended = params["means"].detach()[3:]
        # The two fallback children are exact clones of their floating parent.
        torch.testing.assert_close(appended[0], params["means"].detach()[0])
        torch.testing.assert_close(appended[3], params["means"].detach()[0])
        # The two projected children landed on the plane instead.
        self.assertLess(float(appended[1, 2].abs()), 0.01)
        self.assertLess(float(appended[2, 2].abs()), 0.01)

    def test_optimizer_state_stays_aligned_with_the_parameters(self) -> None:
        params = self._params(np.zeros((5, 3)))
        optimizers = self._optimizers(params)
        out: dict = {}
        self._run_sample_add(
            params, optimizers, 3, torch.ones(5), self._proposal(), out
        )
        for name, optimizer in optimizers.items():
            param = params[name]
            state = optimizer.state[param]
            self.assertEqual(int(param.shape[0]), 8, name)
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(int(state[key].shape[0]), 8, f"{name}.{key}")
                # New rows start at zero, exactly as the upstream op does.
                self.assertTrue(bool((state[key][5:] == 0).all()), f"{name}.{key}")

    def test_child_base_scales_survive_a_duplicated_parent(self) -> None:
        """A parent sampled twice: both children read the last scatter, not new_scales.

        This is the trap the wiring reconstructs deliberately - if the override
        had been built from ``new_scales`` directly, the two children of a
        duplicated parent would carry different tangential scales than the
        clone-only path produces, and the A/B arms would not be comparable.
        """
        params = self._params(np.zeros((3, 3)))
        optimizers = self._optimizers(params)
        # Fall back everywhere so the appended scales are pure clone semantics.
        proposal = TangentProposal(_field(_blob()), _config(), seed=1)
        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc._multinomial_sample",
            return_value=torch.tensor([1, 1, 0]),
        ):
            self._run_sample_add(params, optimizers, 3, torch.ones(3), proposal, {})
        appended = params["scales"].detach()[3:]
        parents = params["scales"].detach()[torch.tensor([1, 1, 0])]
        torch.testing.assert_close(appended, parents)

    def test_disabled_proposal_is_bit_for_bit_the_clone_path(self) -> None:
        means = np.array([[0.0, 0.0, 0.002], [0.2, 0.2, 0.001], [-0.1, 0.1, 0.0]])
        results = []
        for proposal in (None, TangentProposal(_field(_grid_plane()), ProposalConfig())):
            params = self._params(means)
            optimizers = self._optimizers(params)
            with mock.patch(
                "cloudstudio_3dgs.training.error_weighted_mcmc._multinomial_sample",
                return_value=torch.tensor([2, 0, 1]),
            ):
                self._run_sample_add(
                    params, optimizers, 3, torch.ones(3), proposal, {}
                )
            results.append(
                {name: params[name].detach().clone() for name in params}
            )
        for name in results[0]:
            torch.testing.assert_close(results[0][name], results[1][name], msg=name)
            # And the appended block really is a verbatim clone.
            torch.testing.assert_close(
                results[0][name][3:], results[0][name][torch.tensor([2, 0, 1])]
            )

    def test_proposal_out_reports_the_anchors_for_every_child(self) -> None:
        params = self._params(np.zeros((4, 3)))
        optimizers = self._optimizers(params)
        out: dict = {}
        self._run_sample_add(
            params, optimizers, 5, torch.ones(4), self._proposal(), out
        )
        self.assertEqual(int(out["applied"].shape[0]), 5)
        self.assertEqual(int(out["anchor_index"].shape[0]), 5)
        self.assertEqual(int(out["anchor_confidence"].shape[0]), 5)
        self.assertEqual(out["applied_count"], 5)


class StrategyProposalWiringTests(unittest.TestCase):
    """_add_new_gs writes the anchors and extends admission incrementally."""

    def setUp(self) -> None:
        _requires_module(self)

    def _params(self, means: np.ndarray) -> dict:
        n = len(means)
        return {
            "means": torch.nn.Parameter(torch.as_tensor(means, dtype=torch.float32)),
            "scales": torch.nn.Parameter(torch.log(torch.full((n, 3), 0.04))),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
            ),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5))),
        }

    def _optimizers(self, params: dict) -> dict:
        optimizers = {}
        for name, param in params.items():
            optimizer = torch.optim.Adam([param], lr=1e-3)
            param.grad = torch.ones_like(param)
            optimizer.step()
            optimizer.zero_grad()
            optimizers[name] = optimizer
        return optimizers

    def _strategy(self, means: np.ndarray, *, with_admission: bool = True):
        field = _field(_grid_plane())
        state = ErrorScoreState(len(means), ErrorScoreConfig(enabled=True))
        admission = None
        if with_admission:
            admission = LidarAdmission(field, AdmissionConfig(enabled=True))
            admission.refresh(
                torch.as_tensor(means, dtype=torch.float32), lifecycle=state.lifecycle
            )
        strategy = ErrorWeightedMCMCStrategy(
            cap_max=10_000,
            score_state=state,
            error_config=state.config,
            admission_state=admission,
            proposal_state=TangentProposal(field, _config(), seed=3),
        )
        return strategy, state, admission

    def test_lifecycle_anchors_are_written_for_the_projected_children(self) -> None:
        means = np.column_stack(
            [
                np.linspace(-0.4, 0.4, 40),
                np.linspace(0.4, -0.4, 40),
                np.full(40, 0.002),
            ]
        )
        params = self._params(means)
        optimizers = self._optimizers(params)
        strategy, state, _admission = self._strategy(means)
        binoms = strategy.initialize_state()["binoms"]
        with _patch_relocation():
            added = strategy._add_new_gs(params, optimizers, binoms)
        self.assertGreater(added, 0)
        lifecycle = state.lifecycle
        self.assertEqual(len(lifecycle), 40 + added)
        # WP-1 reserved these columns and nothing populated them per birth
        # until now; every newborn must carry a real anchor.
        self.assertTrue(bool((lifecycle.anchor_index[40:] >= 0).all()))
        self.assertTrue(bool((lifecycle.anchor_confidence[40:] > 0).all()))

    def test_admission_is_extended_rather_than_blanked(self) -> None:
        means = np.column_stack(
            [
                np.linspace(-0.4, 0.4, 40),
                np.linspace(0.4, -0.4, 40),
                np.full(40, 0.002),
            ]
        )
        params = self._params(means)
        optimizers = self._optimizers(params)
        strategy, _state, admission = self._strategy(means)
        before = admission.admission_weights().clone()
        binoms = strategy.initialize_state()["binoms"]
        with _patch_relocation():
            added = strategy._add_new_gs(params, optimizers, binoms)
        # The WP-4 behaviour was: stale, weights unusable until a full refresh.
        self.assertFalse(admission.stale)
        self.assertTrue(admission.in_sync(40 + added))
        weights = admission.admission_weights(40 + added)
        self.assertIsNotNone(weights)
        self.assertTrue(torch.equal(weights[:40], before))

    def test_relocation_keeps_the_admission_cache_in_sync(self) -> None:
        means = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.2, 0.0, 0.0]])
        params = self._params(means)
        params["opacities"] = torch.nn.Parameter(
            torch.logit(torch.tensor([0.5, 0.001, 0.7]))
        )
        optimizers = self._optimizers(params)
        strategy, _state, admission = self._strategy(means)
        binoms = strategy.initialize_state()["binoms"]
        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.relocate_weighted",
            return_value=(torch.tensor([1]), torch.tensor([2])),
        ):
            strategy._relocate_gs(params, optimizers, binoms)
        self.assertTrue(admission.in_sync(3))

    def test_stale_admission_is_not_extended(self) -> None:
        means = np.zeros((20, 3))
        params = self._params(means)
        optimizers = self._optimizers(params)
        strategy, _state, admission = self._strategy(means)
        admission.on_count_changed(20)
        binoms = strategy.initialize_state()["binoms"]
        with _patch_relocation():
            strategy._add_new_gs(params, optimizers, binoms)
        # Left stale: growing an out-of-sync vector would misalign every row.
        self.assertTrue(admission.stale)

    def test_proposal_without_admission_still_records_anchors(self) -> None:
        means = np.column_stack(
            [np.linspace(-0.3, 0.3, 30), np.zeros(30), np.full(30, 0.001)]
        )
        params = self._params(means)
        optimizers = self._optimizers(params)
        strategy, state, _none = self._strategy(means, with_admission=False)
        binoms = strategy.initialize_state()["binoms"]
        with _patch_relocation():
            added = strategy._add_new_gs(params, optimizers, binoms)
        self.assertTrue(bool((state.lifecycle.anchor_index[30:] >= 0).all()))
        self.assertEqual(len(state.lifecycle), 30 + added)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_torch(self)

    def test_batches_accumulate_into_the_payload(self) -> None:
        proposal = TangentProposal(_field(_grid_plane()), _config(), seed=0)
        telemetry: dict = {}
        proposal.propose(torch.zeros(4, 3))
        update_proposal_telemetry(telemetry, proposal.last_stats)
        proposal.propose(torch.full((6, 3), 9.0))
        update_proposal_telemetry(telemetry, proposal.last_stats)
        bucket = telemetry["tangent_proposal"]
        self.assertEqual(bucket["batches"], 2)
        self.assertEqual(bucket["candidates"], 10)
        self.assertEqual(bucket["applied"], 4)
        self.assertAlmostEqual(bucket["applied_fraction"], 0.4)

    def test_empty_stats_are_ignored(self) -> None:
        telemetry: dict = {}
        update_proposal_telemetry(telemetry, {})
        self.assertEqual(telemetry, {})

    def test_bridge_preserves_dtype_and_reports_counts(self) -> None:
        proposal = TangentProposal(_field(_grid_plane()), _config(), seed=0)
        parents = torch.zeros(3, 3)
        payload = proposal.propose(
            parents,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
            torch.log(torch.full((3, 3), 0.05)),
        )
        self.assertEqual(payload["means"].dtype, parents.dtype)
        self.assertEqual(payload["applied"].dtype, torch.bool)
        self.assertEqual(set(payload) >= {"means", "quats", "scales"}, True)
        self.assertEqual(proposal.propose_count, 1)
        self.assertEqual(proposal.applied_total, 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
