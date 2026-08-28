"""CPU-only tests for soft LiDAR admission weighting of MCMC densification.

No GPU is ever touched and the CUDA-heavy relocation ops are mocked. Do NOT
mutate CUDA_VISIBLE_DEVICES at module level - unittest discovery imports this
file into the shared process and the hidden device would break the GPU contract
tests that run after it.

The two cases that justify the whole design are
:class:`ScanLineGapTests` (a Gaussian glued to a wall but landing between two
scan lines must still be admitted) and :class:`ExtrapolationTests` (a Gaussian
metres past the edge of the scanned patch must NOT be, even though its
perpendicular distance to the fitted plane is exactly zero). Together they show
why the admission weight can be neither a euclidean-distance test nor a bare
``d_perp`` threshold.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from cloudstudio_3dgs.geometry.lidar_surface_field import (
    build_surface_field,
    support_weight,
)
from cloudstudio_3dgs.training.lidar_admission import (
    AdmissionConfig,
    LidarAdmission,
    normal_field_from_surface_field,
    update_admission_telemetry,
)

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
        )
        from cloudstudio_3dgs.training.gaussian_lifecycle import GaussianLifecycleState
        from gsplat.strategy.ops import _multinomial_sample

        _IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - e.g. CUDA-only gsplat build
        ErrorScoreConfig = ErrorScoreState = ErrorWeightedMCMCStrategy = None
        GaussianLifecycleState = None
        _multinomial_sample = None
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
# Synthetic clouds (identical construction to tests/test_lidar_surface_field.py)
# ---------------------------------------------------------------------------


def _grid_plane(pitch: float = 0.05, extent: float = 1.0) -> np.ndarray:
    """Isotropic z = 0 grid of the given pitch over ``[-extent, extent]^2``.

    Bounded on purpose: everything outside ``|x|, |y| > extent`` is unmeasured
    space that the fitted plane nevertheless extends into.
    """
    axis = np.arange(-extent, extent + pitch * 0.5, pitch)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)


def _scan_line_plane(
    along_pitch: float = 0.05, across_pitch: float = 0.5, extent: float = 2.0
) -> np.ndarray:
    """z = 0 plane sampled densely along x and sparsely across y (scan lines)."""
    x = np.arange(-extent, extent + along_pitch * 0.5, along_pitch)
    y = np.arange(-extent, extent + across_pitch * 0.5, across_pitch)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)


def _means(points: np.ndarray):
    return torch.as_tensor(np.asarray(points, dtype=np.float64), dtype=torch.float32)


def _admission_for(cloud: np.ndarray, candidates: np.ndarray, **config_kwargs):
    """Enabled admission over ``cloud``, refreshed at ``candidates``."""
    config = AdmissionConfig(**{"enabled": True, **config_kwargs})
    admission = LidarAdmission(build_surface_field(cloud), config)
    admission.refresh(_means(candidates))
    weights = admission.admission_weights()
    assert weights is not None
    return admission, weights


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class AdmissionConfigTests(unittest.TestCase):
    def test_defaults_are_valid_and_disabled(self) -> None:
        config = AdmissionConfig()
        config.validate()
        self.assertFalse(config.enabled)
        self.assertEqual(config.mode, "soft")
        self.assertEqual(config.sigma_perp_factor, 1.0)
        self.assertEqual(config.weight_floor, 0.05)
        self.assertEqual(config.refresh_every, 500)
        self.assertEqual(config.gate_tangent_factor, 3.0)

    def test_hard_mode_is_reserved_for_wp5(self) -> None:
        with self.assertRaises(ValueError) as caught:
            AdmissionConfig(mode="hard").validate()
        self.assertIn("WP-5", str(caught.exception))

    def test_zero_weight_floor_is_rejected(self) -> None:
        # A floor of zero is a hard reject in disguise: it makes a region
        # permanently unreachable by densification and can produce an all-zero
        # multinomial distribution.
        with self.assertRaises(ValueError):
            AdmissionConfig(weight_floor=0.0).validate()

    def test_invalid_values_are_rejected(self) -> None:
        for kwargs in (
            {"mode": "nonsense"},
            {"weight_floor": -0.1},
            {"weight_floor": 1.5},
            {"sigma_perp_factor": 0.0},
            {"sigma_perp_factor": -1.0},
            {"refresh_every": 0},
            {"gate_tangent_factor": 0.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    AdmissionConfig(**kwargs).validate()

    def test_to_dict_roundtrips_all_fields(self) -> None:
        payload = AdmissionConfig(enabled=True).to_dict()
        self.assertEqual(
            set(payload),
            {
                "enabled",
                "mode",
                "sigma_perp_factor",
                "weight_floor",
                "refresh_every",
                "gate_tangent_factor",
                "share_normal_field",
            },
        )
        self.assertEqual(AdmissionConfig(**payload), AdmissionConfig(enabled=True))


# ---------------------------------------------------------------------------
# Planar behaviour
# ---------------------------------------------------------------------------


class PlanarAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_torch(self)

    def test_flush_candidate_is_admitted_and_far_candidate_falls_to_floor(self) -> None:
        cloud = _grid_plane()
        candidates = np.array(
            [
                [0.0, 0.0, 0.0],  # exactly on the surface
                [0.012, 0.011, 0.0],  # on the surface, between samples
                [0.0, 0.0, 1.0],  # one metre off along the normal
            ]
        )
        _, weights = _admission_for(cloud, candidates)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=5)
        self.assertAlmostEqual(float(weights[1]), 1.0, places=5)
        self.assertAlmostEqual(float(weights[2]), 0.05, places=6)  # the floor

    def test_admission_decays_smoothly_with_perpendicular_distance(self) -> None:
        # local_spacing is 0.1 on this grid, so sigma = 0.1 and a candidate one
        # spacing off the surface must score exp(-1) = 0.3679.
        cloud = _grid_plane()
        candidates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
        _, weights = _admission_for(cloud, candidates)
        self.assertAlmostEqual(float(weights[1]), float(np.exp(-1.0)), places=4)
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[2]))

    def test_larger_sigma_factor_is_more_permissive(self) -> None:
        cloud = _grid_plane()
        candidates = np.array([[0.0, 0.0, 0.2]])
        _, tight = _admission_for(cloud, candidates, sigma_perp_factor=1.0)
        _, loose = _admission_for(cloud, candidates, sigma_perp_factor=3.0)
        self.assertGreater(float(loose[0]), float(tight[0]))


class ScanLineGapTests(unittest.TestCase):
    """The case that forces support_weight instead of euclidean distance."""

    def setUp(self) -> None:
        _requires_torch(self)

    def test_candidate_between_scan_lines_keeps_full_admission(self) -> None:
        cloud = _scan_line_plane(along_pitch=0.05, across_pitch=0.5)
        candidate = np.array([[0.0, 0.25, 0.0]])

        # The old euclidean criterion sees this Gaussian as 0.25 m off the
        # surface, purely because of where the scanner's lines happened to fall.
        query = build_surface_field(cloud).query(candidate, k=1)
        self.assertAlmostEqual(float(query.euclidean[0]), 0.25, places=6)
        self.assertAlmostEqual(float(query.d_perp[0]), 0.0, places=9)

        _, weights = _admission_for(cloud, candidate)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=5)

    def test_scan_gap_survives_while_a_genuine_floater_does_not(self) -> None:
        # Same euclidean order of magnitude, opposite verdicts: the gap
        # candidate is 0.25 m from its nearest sample and admitted, the floater
        # is 1.03 m away *along the normal* and floored.
        cloud = _scan_line_plane(along_pitch=0.05, across_pitch=0.5)
        candidates = np.array([[0.0, 0.25, 0.0], [0.0, 0.25, 1.0]])
        _, weights = _admission_for(cloud, candidates)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=5)
        self.assertAlmostEqual(float(weights[1]), 0.05, places=6)


class ExtrapolationTests(unittest.TestCase):
    """The case that forces the tangential gate: d_perp alone is not enough."""

    def setUp(self) -> None:
        _requires_torch(self)

    def test_bare_support_weight_cannot_see_the_patch_edge(self) -> None:
        # Documents the WP-3 warning as an executable fact. The patch ends at
        # x = 1.0; a candidate at x = 5.0 is four metres out in open space, yet
        # its perpendicular distance to the *unbounded* fitted plane is zero and
        # support_weight rates it a perfect 1.0.
        field = build_surface_field(_grid_plane())
        query = field.query(np.array([[5.0, 0.0, 0.0]]), k=1)
        self.assertAlmostEqual(float(query.d_perp[0]), 0.0, places=9)
        self.assertAlmostEqual(float(query.d_tangent[0]), 4.0, places=6)
        self.assertAlmostEqual(
            float(support_weight(query, sigma_perp_factor=1.0)[0]), 1.0, places=6
        )

    def test_tangential_gate_floors_the_extrapolated_candidate(self) -> None:
        cloud = _grid_plane()
        candidates = np.array(
            [
                [1.2, 0.0, 0.0],  # just past the edge, within 3 * local_spacing
                [2.0, 0.0, 0.0],  # a metre out: ~3 spacings of excess
                [5.0, 0.0, 0.0],  # four metres out
            ]
        )
        _, weights = _admission_for(cloud, candidates)
        # Inside the tolerance the gate is exactly 1: a Gaussian sitting on the
        # boundary of the scan is legitimate and must not be penalized.
        self.assertAlmostEqual(float(weights[0]), 1.0, places=5)
        self.assertAlmostEqual(float(weights[1]), 0.05, places=6)
        self.assertAlmostEqual(float(weights[2]), 0.05, places=6)

    def test_gate_factor_is_the_mechanism(self) -> None:
        # Control: with the tangential tolerance widened the very same
        # candidate is admitted again, proving the demotion came from the gate
        # and not from the perpendicular term.
        cloud = _grid_plane()
        candidates = np.array([[2.0, 0.0, 0.0]])
        _, gated = _admission_for(cloud, candidates, gate_tangent_factor=3.0)
        _, ungated = _admission_for(cloud, candidates, gate_tangent_factor=1000.0)
        self.assertAlmostEqual(float(gated[0]), 0.05, places=6)
        self.assertAlmostEqual(float(ungated[0]), 1.0, places=5)

    def test_gate_is_monotone_in_tangential_distance(self) -> None:
        cloud = _grid_plane()
        candidates = np.array(
            [[1.05, 0.0, 0.0], [1.4, 0.0, 0.0], [1.7, 0.0, 0.0], [2.5, 0.0, 0.0]]
        )
        _, weights = _admission_for(cloud, candidates, weight_floor=1e-6)
        values = [float(value) for value in weights]
        self.assertEqual(values, sorted(values, reverse=True))


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


class WeightFloorTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_torch(self)

    def test_floor_is_never_zero_however_hopeless_the_candidate(self) -> None:
        # A LiDAR-invisible region (glass, occlusion shadow, out of range) still
        # needs a visual-only birth channel; an admission of 0 would make the
        # region permanently unreachable by densification.
        cloud = _grid_plane()
        candidates = np.array([[0.0, 0.0, 100.0], [500.0, 500.0, 500.0]])
        _, weights = _admission_for(cloud, candidates)
        self.assertTrue(bool((weights > 0).all()))
        self.assertAlmostEqual(float(weights[0]), 0.05, places=6)
        self.assertAlmostEqual(float(weights[1]), 0.05, places=6)

    def test_floor_value_is_configurable(self) -> None:
        cloud = _grid_plane()
        candidates = np.array([[0.0, 0.0, 100.0]])
        _, loose = _admission_for(cloud, candidates, weight_floor=0.2)
        self.assertAlmostEqual(float(loose[0]), 0.2, places=6)

    def test_weights_stay_within_the_floor_and_one(self) -> None:
        cloud = _grid_plane()
        rng = np.random.default_rng(7)
        candidates = rng.uniform(-3.0, 3.0, size=(400, 3))
        _, weights = _admission_for(cloud, candidates)
        self.assertGreaterEqual(float(weights.min()), 0.05 - 1e-6)
        self.assertLessEqual(float(weights.max()), 1.0 + 1e-6)

    def test_refresh_statistics_report_the_floored_share(self) -> None:
        cloud = _grid_plane()
        candidates = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 100.0], [0.0, 0.0, 100.0], [0.0, 0.0, 100.0]]
        )
        admission, _ = _admission_for(cloud, candidates)
        stats = admission.last_stats
        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["at_floor_fraction"], 0.75, places=6)
        self.assertAlmostEqual(stats["max"], 1.0, places=5)
        self.assertAlmostEqual(stats["min"], 0.05, places=6)


# ---------------------------------------------------------------------------
# sampling_weights integration
# ---------------------------------------------------------------------------


class SamplingWeightsAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_none_admission_is_bit_for_bit_the_old_formula(self) -> None:
        state = ErrorScoreState(4, ErrorScoreConfig(min_score_floor=1e-3))
        state.scores = torch.tensor([0.0, 0.2, 0.7, 1.0])
        opacities = torch.tensor([0.3, 0.6, 0.9, 0.05])
        legacy = opacities * state.scores.clamp_min(1e-3) ** 0.4
        self.assertTrue(torch.equal(state.sampling_weights(opacities), legacy))
        self.assertTrue(
            torch.equal(state.sampling_weights(opacities, admission=None), legacy)
        )

    def test_admission_multiplies_the_weights(self) -> None:
        state = ErrorScoreState(3)
        state.scores = torch.tensor([0.4, 0.4, 0.4])
        opacities = torch.full((3,), 0.5)
        admission = torch.tensor([1.0, 0.5, 0.05])
        base = state.sampling_weights(opacities)
        weighted = state.sampling_weights(opacities, admission=admission)
        self.assertTrue(torch.allclose(weighted, base * admission))

    def test_admission_length_mismatch_is_rejected(self) -> None:
        state = ErrorScoreState(3)
        with self.assertRaises(ValueError):
            state.sampling_weights(torch.ones(3), admission=torch.ones(2))

    def test_admission_shifts_the_multinomial_toward_the_supported_site(self) -> None:
        # Two candidates identical in opacity and error score; only surface
        # support separates them. At the default floor that is a 20:1 preference
        # and the sampler must actually realize it.
        torch.manual_seed(42)
        state = ErrorScoreState(2, ErrorScoreConfig(score_power=1.0))
        state.scores = torch.tensor([1.0, 1.0])
        opacities = torch.tensor([0.7, 0.7])
        admission = torch.tensor([1.0, 0.05])
        weights = state.sampling_weights(opacities, admission=admission)
        self.assertAlmostEqual(float(weights[0] / weights[1]), 20.0, places=4)
        draws = 20000
        idx = _multinomial_sample(weights, draws, replacement=True)
        share = float(torch.bincount(idx, minlength=2)[0]) / draws
        expected = 20.0 / 21.0
        self.assertLess(abs(share - expected), 0.02)
        # The unsupported site is still reachable - this is soft, not a reject.
        self.assertGreater(int(torch.bincount(idx, minlength=2)[1]), 0)


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


class StaleSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_torch(self)

    def _admission(self, **kwargs) -> LidarAdmission:
        config = AdmissionConfig(**{"enabled": True, **kwargs})
        return LidarAdmission(build_surface_field(_grid_plane()), config)

    def test_starts_stale_and_returns_none_before_the_first_refresh(self) -> None:
        admission = self._admission()
        self.assertTrue(admission.stale)
        self.assertIsNone(admission.admission_weights())

    def test_refresh_clears_staleness(self) -> None:
        admission = self._admission()
        admission.refresh(_means(np.zeros((5, 3))))
        self.assertFalse(admission.stale)
        weights = admission.admission_weights()
        self.assertIsNotNone(weights)
        self.assertEqual(int(weights.shape[0]), 5)

    def test_count_change_makes_weights_unavailable_until_refresh(self) -> None:
        admission = self._admission()
        admission.refresh(_means(np.zeros((5, 3))))
        admission.on_count_changed(7)
        self.assertTrue(admission.stale)
        self.assertIsNone(admission.admission_weights())
        admission.refresh(_means(np.zeros((7, 3))))
        self.assertIsNotNone(admission.admission_weights())
        self.assertEqual(int(admission.admission_weights().shape[0]), 7)

    def test_length_mismatch_returns_none_even_when_fresh(self) -> None:
        admission = self._admission()
        admission.refresh(_means(np.zeros((5, 3))))
        self.assertIsNone(admission.admission_weights(6))
        self.assertIsNotNone(admission.admission_weights(5))

    def test_disabled_and_off_mode_never_produce_weights(self) -> None:
        for kwargs in ({"enabled": False}, {"mode": "off"}):
            with self.subTest(**kwargs):
                config = AdmissionConfig(**{"enabled": True, **kwargs})
                admission = LidarAdmission(
                    build_surface_field(_grid_plane()), config
                )
                admission.refresh(_means(np.zeros((5, 3))))
                self.assertFalse(admission.stale)
                self.assertIsNone(admission.admission_weights())

    def test_empty_cloud_refresh_is_not_an_error(self) -> None:
        admission = self._admission()
        stats = admission.refresh(_means(np.zeros((0, 3))))
        self.assertEqual(stats["count"], 0)
        self.assertFalse(admission.stale)
        self.assertEqual(int(admission.admission_weights().shape[0]), 0)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class LifecycleIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_anchor_columns_are_written_back_at_matching_length(self) -> None:
        candidates = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 5.0]])
        admission = LidarAdmission(
            build_surface_field(_grid_plane()), AdmissionConfig(enabled=True)
        )
        lifecycle = GaussianLifecycleState(3)
        self.assertTrue(bool((lifecycle.anchor_index == -1).all()))

        admission.refresh(_means(candidates), lifecycle=lifecycle)

        self.assertEqual(int(lifecycle.anchor_index.shape[0]), 3)
        self.assertEqual(int(lifecycle.anchor_confidence.shape[0]), 3)
        self.assertTrue(bool((lifecycle.anchor_index >= 0).all()))
        self.assertTrue(torch.equal(lifecycle.anchor_index, admission.anchor_index))
        self.assertTrue(
            bool(
                ((lifecycle.anchor_confidence >= 0.0)
                 & (lifecycle.anchor_confidence <= 1.0)).all()
            )
        )

    def test_mismatched_lifecycle_length_is_left_untouched(self) -> None:
        admission = LidarAdmission(
            build_surface_field(_grid_plane()), AdmissionConfig(enabled=True)
        )
        lifecycle = GaussianLifecycleState(5)
        admission.refresh(_means(np.zeros((3, 3))), lifecycle=lifecycle)
        self.assertEqual(int(lifecycle.anchor_index.shape[0]), 5)
        self.assertTrue(bool((lifecycle.anchor_index == -1).all()))

    def test_cached_weights_stay_aligned_with_the_lifecycle_length(self) -> None:
        admission = LidarAdmission(
            build_surface_field(_grid_plane()), AdmissionConfig(enabled=True)
        )
        state = ErrorScoreState(6, ErrorScoreConfig(enabled=True))
        admission.refresh(_means(np.zeros((6, 3))), lifecycle=state.lifecycle)
        weights = admission.admission_weights(len(state))
        self.assertIsNotNone(weights)
        self.assertEqual(int(weights.shape[0]), len(state))
        # Growing the cloud must invalidate rather than silently misalign.
        state.resize(8)
        admission.on_count_changed(len(state))
        self.assertIsNone(admission.admission_weights(len(state)))


# ---------------------------------------------------------------------------
# Strategy wiring
# ---------------------------------------------------------------------------


class StrategyAdmissionWiringTests(unittest.TestCase):
    """Verify which probs reach the weighted MCMC ops, with the heavy ops mocked."""

    def setUp(self) -> None:
        _requires_module(self)

    def _params(self, opacities) -> dict:
        n = opacities.shape[0]
        return {
            "means": torch.nn.Parameter(torch.zeros(n, 3)),
            "scales": torch.nn.Parameter(torch.zeros(n, 3)),
            "quats": torch.nn.Parameter(torch.zeros(n, 4)),
            "opacities": torch.nn.Parameter(torch.logit(opacities)),
        }

    def _admission(self, candidates: np.ndarray) -> LidarAdmission:
        admission = LidarAdmission(
            build_surface_field(_grid_plane()), AdmissionConfig(enabled=True)
        )
        admission.refresh(_means(candidates))
        return admission

    def test_relocate_multiplies_probs_by_admission(self) -> None:
        opacities = torch.tensor([0.5, 0.2, 0.8, 0.002])
        params = self._params(opacities)
        state = ErrorScoreState(4, ErrorScoreConfig(enabled=True))
        state.scores = torch.tensor([0.2, 0.4, 0.9, 0.1])
        admission = self._admission(
            np.array(
                [
                    [0.0, 0.0, 0.0],  # on surface
                    [0.0, 0.0, 5.0],  # floater
                    [0.5, 0.5, 0.0],  # on surface
                    [9.0, 9.0, 0.0],  # far outside the patch
                ]
            )
        )
        strategy = ErrorWeightedMCMCStrategy(
            score_state=state, error_config=state.config, admission_state=admission
        )
        binoms = strategy.initialize_state()["binoms"]
        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.relocate_weighted"
        ) as relocate_mock:
            strategy._relocate_gs(params, {}, binoms)
        probs = relocate_mock.call_args.kwargs["probs"]
        real_opac = torch.sigmoid(params["opacities"].flatten())
        expected = state.sampling_weights(
            real_opac, admission=admission.admission_weights()
        )
        expected = expected.clone()
        expected[real_opac <= strategy.min_opacity] = 0.0
        self.assertTrue(torch.allclose(probs, expected))
        # And it is genuinely different from the un-admitted distribution.
        unadmitted = state.sampling_weights(real_opac).clone()
        unadmitted[real_opac <= strategy.min_opacity] = 0.0
        self.assertFalse(
            torch.allclose(probs, unadmitted)
        )

    def test_add_new_gs_multiplies_probs_by_admission(self) -> None:
        opacities = torch.full((40,), 0.5)
        params = self._params(opacities)
        state = ErrorScoreState(40, ErrorScoreConfig(enabled=True))
        candidates = np.zeros((40, 3))
        candidates[20:, 2] = 5.0  # half the cloud floats off the surface
        admission = self._admission(candidates)
        strategy = ErrorWeightedMCMCStrategy(
            cap_max=1000,
            score_state=state,
            error_config=state.config,
            admission_state=admission,
        )
        binoms = strategy.initialize_state()["binoms"]

        def fake_sample_add(*, params, n, **_kwargs):
            params["means"] = torch.nn.Parameter(
                torch.cat([params["means"], torch.zeros(n, 3)])
            )

        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.sample_add_weighted",
            side_effect=fake_sample_add,
        ) as add_mock:
            strategy._add_new_gs(params, {}, binoms)
        probs = add_mock.call_args.kwargs["probs"]
        self.assertEqual(int(probs.shape[0]), 40)
        # Supported half outweighs the floating half by the floor ratio.
        self.assertAlmostEqual(
            float(probs[:20].mean() / probs[20:].mean()), 20.0, places=3
        )
        self.assertTrue(bool((probs > 0).all()))

    def test_stale_admission_degrades_to_pure_error_weighting(self) -> None:
        opacities = torch.tensor([0.5, 0.001, 0.8, 0.002])
        params = self._params(opacities)
        state = ErrorScoreState(4, ErrorScoreConfig(enabled=True))
        state.scores = torch.tensor([0.2, 0.4, 0.9, 0.1])
        admission = self._admission(np.zeros((4, 3)))
        admission.on_count_changed(9)
        strategy = ErrorWeightedMCMCStrategy(
            score_state=state, error_config=state.config, admission_state=admission
        )
        binoms = strategy.initialize_state()["binoms"]
        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.relocate_weighted"
        ) as relocate_mock:
            strategy._relocate_gs(params, {}, binoms)
        real_opac = torch.sigmoid(params["opacities"].flatten())
        expected = state.sampling_weights(real_opac).clone()
        expected[real_opac <= strategy.min_opacity] = 0.0
        self.assertTrue(
            torch.equal(
                relocate_mock.call_args.kwargs["probs"],
                expected,
            )
        )

    def test_absent_admission_uses_error_weights_for_eligible_sources(self) -> None:
        opacities = torch.tensor([0.5, 0.001, 0.8])
        params = self._params(opacities)
        state = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        strategy = ErrorWeightedMCMCStrategy(
            score_state=state, error_config=state.config
        )
        self.assertIsNone(strategy.admission_state)
        binoms = strategy.initialize_state()["binoms"]
        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.relocate_weighted"
        ) as relocate_mock:
            strategy._relocate_gs(params, {}, binoms)
        real_opac = torch.sigmoid(params["opacities"].flatten())
        expected = state.sampling_weights(real_opac).clone()
        expected[real_opac <= strategy.min_opacity] = 0.0
        self.assertTrue(
            torch.equal(
                relocate_mock.call_args.kwargs["probs"],
                expected,
            )
        )


# ---------------------------------------------------------------------------
# Telemetry and field sharing
# ---------------------------------------------------------------------------


class TelemetryTests(unittest.TestCase):
    def test_aggregates_refresh_statistics(self) -> None:
        telemetry: dict = {}
        update_admission_telemetry(telemetry, {"mean": 0.4, "count": 10})
        update_admission_telemetry(telemetry, {"mean": 0.8, "count": 10})
        bucket = telemetry["admission"]
        self.assertEqual(bucket["refresh_count"], 2)
        self.assertAlmostEqual(bucket["mean_admission"], 0.6, places=6)
        self.assertEqual(bucket["last"]["mean"], 0.8)

    def test_empty_statistics_are_ignored(self) -> None:
        telemetry: dict = {}
        update_admission_telemetry(telemetry, {})
        self.assertNotIn("admission", telemetry)


class NormalFieldSharingTests(unittest.TestCase):
    def test_adapter_exposes_the_normal_field_query_contract(self) -> None:
        cloud = _grid_plane()
        surface = build_surface_field(cloud)
        adapted = normal_field_from_surface_field(surface)
        self.assertEqual(len(adapted), len(cloud))
        self.assertIs(adapted.tree, surface.tree)  # the KD-tree is reused
        distance, normal, planarity, index = adapted.query(
            np.array([[0.0, 0.0, 0.3]]), k=1
        )
        self.assertAlmostEqual(float(distance[0]), 0.3, places=6)
        self.assertEqual(normal.shape, (1, 3))
        self.assertAlmostEqual(abs(float(normal[0][2])), 1.0, places=5)
        self.assertAlmostEqual(float(planarity[0]), 1.0, places=5)
        self.assertEqual(int(index[0]), int(surface.query(
            np.array([[0.0, 0.0, 0.3]]), k=1
        ).index[0]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
