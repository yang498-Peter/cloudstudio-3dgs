"""CPU-only tests for error-weighted MCMC sampling (no GPU is ever touched).

All tensors live on CPU and the CUDA-heavy relocation ops are mocked. Do NOT
mutate CUDA_VISIBLE_DEVICES at module level - unittest discovery imports this
file into the shared process and the hidden device would break the GPU
contract tests that run after it.
"""

from __future__ import annotations

import unittest
from unittest import mock

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
        from gsplat.strategy.ops import _multinomial_sample

        _IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - e.g. CUDA-only gsplat build
        ErrorScoreConfig = ErrorScoreState = ErrorWeightedMCMCStrategy = None
        _multinomial_sample = None
        _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = ImportError("torch is not installed")


def _requires_module(test_case: unittest.TestCase) -> None:
    if _IMPORT_ERROR is not None:
        test_case.skipTest(f"error_weighted_mcmc unavailable on this host: {_IMPORT_ERROR}")


class ErrorScoreConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_defaults_are_valid_and_disabled(self) -> None:
        config = ErrorScoreConfig()
        config.validate()
        self.assertFalse(config.enabled)
        self.assertEqual(config.ema_decay, 0.9)
        self.assertEqual(config.score_power, 0.4)
        self.assertEqual(config.min_score_floor, 1e-3)

    def test_to_dict_roundtrips_all_fields(self) -> None:
        payload = ErrorScoreConfig(enabled=True).to_dict()
        self.assertEqual(
            set(payload),
            {"enabled", "ema_decay", "score_power", "min_score_floor"},
        )
        self.assertTrue(payload["enabled"])

    def test_validate_rejects_illegal_values(self) -> None:
        for bad in (
            ErrorScoreConfig(ema_decay=1.0),
            ErrorScoreConfig(ema_decay=-0.1),
            ErrorScoreConfig(score_power=-0.4),
            ErrorScoreConfig(min_score_floor=0.0),
            ErrorScoreConfig(min_score_floor=-1e-3),
        ):
            with self.assertRaises(ValueError):
                bad.validate()
            with self.assertRaises(ValueError):
                bad.to_dict()


class ErrorScoreStateUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_visible_gaussians_sample_their_pixel_error(self) -> None:
        state = ErrorScoreState(3, ErrorScoreConfig(ema_decay=0.5))
        pixel_error = torch.zeros(4, 5)
        pixel_error[1, 2] = 0.8
        pixel_error[3, 4] = 0.2
        means2d = torch.tensor([[2.0, 1.0], [4.0, 3.0], [0.0, 0.0]])
        radii = torch.tensor([3, 2, 5])
        state.update(means2d, radii, pixel_error, height=4, width=5)
        # score = 0.5 * 1.0 + 0.5 * err
        self.assertTrue(
            torch.allclose(state.scores, torch.tensor([0.9, 0.6, 0.5]))
        )

    def test_invisible_gaussians_keep_their_score(self) -> None:
        state = ErrorScoreState(3, ErrorScoreConfig(ema_decay=0.5))
        pixel_error = torch.full((4, 4), 1.0)
        means2d = torch.tensor([[1.0, 1.0], [float("nan"), float("nan")], [2.0, 2.0]])
        radii = torch.tensor([2, 0, 0])  # only the first is visible
        state.update(means2d, radii, pixel_error, height=4, width=4)
        self.assertTrue(
            torch.allclose(state.scores, torch.tensor([1.0, 1.0, 1.0]))
        )
        # And a NaN projection on an invisible Gaussian must not crash or leak.
        state2 = ErrorScoreState(3, ErrorScoreConfig(ema_decay=0.0))
        pixel_error2 = torch.zeros(4, 4)
        pixel_error2[1, 1] = 0.7
        state2.update(means2d, radii, pixel_error2, height=4, width=4)
        self.assertTrue(
            torch.allclose(state2.scores, torch.tensor([0.7, 1.0, 1.0]))
        )

    def test_radii_accepts_n_by_2_visibility(self) -> None:
        state = ErrorScoreState(2, ErrorScoreConfig(ema_decay=0.0))
        pixel_error = torch.zeros(3, 3)
        pixel_error[0, 0] = 0.4
        means2d = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        radii = torch.tensor([[2, 3], [0, 4]])  # second has a zero component
        state.update(means2d, radii, pixel_error, height=3, width=3)
        self.assertTrue(torch.allclose(state.scores, torch.tensor([0.4, 1.0])))

    def test_ema_accumulates_over_multiple_updates(self) -> None:
        decay = 0.9
        state = ErrorScoreState(1, ErrorScoreConfig(ema_decay=decay))
        means2d = torch.tensor([[0.0, 0.0]])
        radii = torch.tensor([1])
        expected = 1.0
        for err in (0.5, 0.25, 1.0):
            pixel_error = torch.full((2, 2), err)
            state.update(means2d, radii, pixel_error, height=2, width=2)
            expected = decay * expected + (1.0 - decay) * err
            self.assertAlmostEqual(float(state.scores[0]), expected, places=6)

    def test_out_of_bounds_projection_is_clamped_into_image(self) -> None:
        state = ErrorScoreState(2, ErrorScoreConfig(ema_decay=0.0))
        h, w = 4, 6
        pixel_error = torch.zeros(h, w)
        pixel_error[0, 0] = 0.3  # clamp target of (-7.2, -1.5)
        pixel_error[h - 1, w - 1] = 0.9  # clamp target of (100.0, 50.0)
        means2d = torch.tensor([[-7.2, -1.5], [100.0, 50.0]])
        radii = torch.tensor([1, 1])
        state.update(means2d, radii, pixel_error, height=h, width=w)
        self.assertTrue(torch.allclose(state.scores, torch.tensor([0.3, 0.9])))

    def test_count_mismatch_is_rejected(self) -> None:
        state = ErrorScoreState(2)
        with self.assertRaisesRegex(ValueError, "resize"):
            state.update(
                torch.zeros(3, 2),
                torch.ones(3),
                torch.zeros(2, 2),
                height=2,
                width=2,
            )

    def test_resize_resets_to_ones(self) -> None:
        state = ErrorScoreState(2, ErrorScoreConfig(ema_decay=0.0))
        pixel_error = torch.full((2, 2), 0.5)
        state.update(
            torch.zeros(2, 2), torch.ones(2), pixel_error, height=2, width=2
        )
        self.assertTrue(torch.allclose(state.scores, torch.tensor([0.5, 0.5])))
        state.resize(5)
        self.assertEqual(len(state), 5)
        self.assertTrue(torch.allclose(state.scores, torch.ones(5)))


class SamplingWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_higher_error_yields_higher_weight_at_equal_opacity(self) -> None:
        state = ErrorScoreState(3)
        state.scores = torch.tensor([0.1, 0.5, 0.9])
        opacities = torch.full((3,), 0.6)
        weights = state.sampling_weights(opacities)
        self.assertTrue(bool(weights[0] < weights[1] < weights[2]))
        # Explicit formula check: opacity * score**0.4.
        expected = 0.6 * torch.tensor([0.1, 0.5, 0.9]) ** 0.4
        self.assertTrue(torch.allclose(weights, expected))

    def test_floor_prevents_zero_weights(self) -> None:
        config = ErrorScoreConfig(min_score_floor=1e-3)
        state = ErrorScoreState(2, config)
        state.scores = torch.tensor([0.0, 1e-9])
        weights = state.sampling_weights(torch.tensor([0.5, 0.5]))
        floor_weight = 0.5 * (1e-3**0.4)
        self.assertTrue(bool((weights > 0).all()))
        self.assertTrue(
            torch.allclose(weights, torch.full((2,), floor_weight), rtol=1e-5)
        )

    def test_count_mismatch_is_rejected(self) -> None:
        state = ErrorScoreState(2)
        with self.assertRaises(ValueError):
            state.sampling_weights(torch.ones(3))

    def test_multinomial_sampling_prefers_high_error_gaussian(self) -> None:
        # Two Gaussians with identical opacity, error scores at a 10:1
        # weight ratio; frequency of the high-error one must track its share.
        torch.manual_seed(42)
        state = ErrorScoreState(2, ErrorScoreConfig(score_power=1.0))
        state.scores = torch.tensor([1.0, 0.1])
        weights = state.sampling_weights(torch.tensor([0.7, 0.7]))
        self.assertAlmostEqual(float(weights[0] / weights[1]), 10.0, places=4)
        draws = 20000
        counts = torch.zeros(2)
        for _ in range(10):
            idx = _multinomial_sample(weights, draws // 10, replacement=True)
            counts += torch.bincount(idx, minlength=2).float()
        share = float(counts[0] / draws)
        expected = 10.0 / 11.0
        # Binomial std at n=20000 is ~0.002; a 0.02 band is > 9 sigma.
        self.assertLess(abs(share - expected), 0.02)
        self.assertGreater(share, 0.5)


class ErrorScoreCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_checkpoint_roundtrip_restores_exact_scores(self) -> None:
        source = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        source.scores = torch.tensor([0.15, 0.5, 0.95])
        payload = source.checkpoint_state()
        restored = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        restored.restore_checkpoint_state(payload, expected_count=3)
        self.assertTrue(torch.equal(restored.scores, source.scores))
        self.assertIsNot(restored.scores, payload["scores"])

    def test_restore_rejects_missing_stale_and_non_finite_state(self) -> None:
        state = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        bad_payloads = (
            None,
            {"schema_version": 2, "scores": torch.ones(3)},
            {"schema_version": 1, "scores": torch.ones(2)},
            {"schema_version": 1, "scores": torch.tensor([1.0, float("nan"), 1.0])},
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    state.restore_checkpoint_state(payload, expected_count=3)


class StrategyWiringTests(unittest.TestCase):
    """Exercise the overridden refine hooks with the heavy ops mocked out.

    compute_relocation needs the CUDA extension, so these tests verify the
    strategy-level contract on CPU: which probs are passed, mask semantics,
    fallback behaviour, and score_state resizing after growth.
    """

    def setUp(self) -> None:
        _requires_module(self)

    def _params(self, opacities: torch.Tensor) -> dict:
        n = opacities.shape[0]
        return {
            "means": torch.nn.Parameter(torch.zeros(n, 3)),
            "scales": torch.nn.Parameter(torch.zeros(n, 3)),
            "quats": torch.nn.Parameter(torch.zeros(n, 4)),
            "opacities": torch.nn.Parameter(torch.logit(opacities)),
        }

    def test_initialize_state_and_config_validation(self) -> None:
        strategy = ErrorWeightedMCMCStrategy(
            score_state=ErrorScoreState(4),
            error_config=ErrorScoreConfig(enabled=True),
        )
        state = strategy.initialize_state()
        self.assertIn("binoms", state)
        with self.assertRaises(ValueError):
            ErrorWeightedMCMCStrategy(error_config=ErrorScoreConfig(ema_decay=2.0))

    def test_relocate_passes_error_weights_and_dead_mask(self) -> None:
        opacities = torch.tensor([0.5, 0.001, 0.8, 0.002])
        params = self._params(opacities)
        state = ErrorScoreState(4, ErrorScoreConfig(enabled=True))
        state.scores = torch.tensor([0.2, 0.4, 0.9, 0.1])
        strategy = ErrorWeightedMCMCStrategy(
            score_state=state, error_config=state.config
        )
        binoms = strategy.initialize_state()["binoms"]
        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.relocate_weighted"
        ) as relocate_mock:
            n_gs = strategy._relocate_gs(params, {}, binoms)
        self.assertEqual(n_gs, 2)
        kwargs = relocate_mock.call_args.kwargs
        real_opac = torch.sigmoid(params["opacities"].flatten())
        self.assertTrue(
            torch.equal(kwargs["mask"], real_opac <= strategy.min_opacity)
        )
        expected_probs = state.sampling_weights(real_opac)
        self.assertTrue(torch.allclose(kwargs["probs"], expected_probs))
        self.assertEqual(kwargs["probs"].shape[0], 4)  # full-length [N]

    def test_add_new_gs_uses_weights_and_resizes_state(self) -> None:
        opacities = torch.full((40,), 0.5)
        params = self._params(opacities)
        state = ErrorScoreState(40, ErrorScoreConfig(enabled=True))
        strategy = ErrorWeightedMCMCStrategy(
            cap_max=1000, score_state=state, error_config=state.config
        )
        binoms = strategy.initialize_state()["binoms"]

        def fake_sample_add(*, params, n, **_kwargs):
            grown = torch.nn.Parameter(
                torch.cat([params["means"], torch.zeros(n, 3)])
            )
            params["means"] = grown

        with mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.sample_add_weighted",
            side_effect=fake_sample_add,
        ) as add_mock:
            n_gs = strategy._add_new_gs(params, {}, binoms)
        self.assertEqual(n_gs, 2)  # 5% growth of 40
        self.assertEqual(add_mock.call_args.kwargs["n"], 2)
        self.assertEqual(add_mock.call_args.kwargs["probs"].shape[0], 40)
        self.assertEqual(len(state), 42)
        self.assertTrue(torch.allclose(state.scores, torch.ones(42)))

    def test_disabled_config_falls_back_to_parent_sampling(self) -> None:
        opacities = torch.tensor([0.5, 0.001, 0.8])
        params = self._params(opacities)
        strategy = ErrorWeightedMCMCStrategy(
            score_state=ErrorScoreState(3),
            error_config=ErrorScoreConfig(enabled=False),
        )
        binoms = strategy.initialize_state()["binoms"]
        with mock.patch(
            "gsplat.strategy.mcmc.relocate"
        ) as parent_relocate, mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.relocate_weighted"
        ) as weighted_relocate:
            n_gs = strategy._relocate_gs(params, {}, binoms)
        self.assertEqual(n_gs, 1)
        self.assertEqual(parent_relocate.call_count, 1)
        self.assertEqual(weighted_relocate.call_count, 0)

    def test_stale_score_count_falls_back_to_parent_sampling(self) -> None:
        opacities = torch.tensor([0.5, 0.001, 0.8])
        params = self._params(opacities)
        strategy = ErrorWeightedMCMCStrategy(
            score_state=ErrorScoreState(7),  # out of sync with N=3
            error_config=ErrorScoreConfig(enabled=True),
        )
        binoms = strategy.initialize_state()["binoms"]
        with mock.patch(
            "gsplat.strategy.mcmc.relocate"
        ) as parent_relocate, mock.patch(
            "cloudstudio_3dgs.training.error_weighted_mcmc.relocate_weighted"
        ) as weighted_relocate:
            strategy._relocate_gs(params, {}, binoms)
        self.assertEqual(parent_relocate.call_count, 1)
        self.assertEqual(weighted_relocate.call_count, 0)


def _footprint_oracle(mean, conic, opacity, err_map, radius_cap, r_eff):
    """Naive per-pixel reference implementation of the v2 aggregation.

    Loops the fixed (2R+1)^2 window, drops offsets beyond ``r_eff`` and
    pixels outside the image, weights by ``opacity * exp(-0.5 * d^T C d)``
    around the true sub-pixel center, and normalizes. Returns None when the
    weight mass is below the 1e-8 skip threshold.
    """
    import math

    height, width = err_map.shape
    cx, cy = int(round(mean[0])), int(round(mean[1]))
    a, b, c = conic
    numerator = 0.0
    denominator = 0.0
    for dy in range(-radius_cap, radius_cap + 1):
        for dx in range(-radius_cap, radius_cap + 1):
            if abs(dx) > r_eff or abs(dy) > r_eff:
                continue
            x, y = cx + dx, cy + dy
            if not (0 <= x < width and 0 <= y < height):
                continue
            ddx, ddy = x - mean[0], y - mean[1]
            quad = a * ddx * ddx + 2.0 * b * ddx * ddy + c * ddy * ddy
            weight = opacity * math.exp(-0.5 * quad)
            numerator += weight * float(err_map[y, x])
            denominator += weight
    if denominator < 1e-8:
        return None
    return numerator / denominator


class FootprintConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_defaults_extend_center_sampling(self) -> None:
        config = ErrorScoreConfig()
        config.validate()
        self.assertEqual(config.aggregation, "center")
        self.assertEqual(config.footprint_radius_px, 4)
        # to_dict schema is intentionally unchanged (4 core fields).
        self.assertEqual(
            set(config.to_dict()),
            {"enabled", "ema_decay", "score_power", "min_score_floor"},
        )

    def test_validate_accepts_footprint_and_rejects_bad_values(self) -> None:
        ErrorScoreConfig(aggregation="footprint", footprint_radius_px=2).validate()
        for bad in (
            ErrorScoreConfig(aggregation="alpha_t"),
            ErrorScoreConfig(aggregation=""),
            ErrorScoreConfig(footprint_radius_px=0),
            ErrorScoreConfig(footprint_radius_px=-3),
        ):
            with self.assertRaises(ValueError):
                bad.validate()


class FootprintAggregationTests(unittest.TestCase):
    """v2 conic-weighted footprint aggregation (all CPU, no rasterizer)."""

    def setUp(self) -> None:
        _requires_module(self)

    def _v2_config(self, **overrides) -> "ErrorScoreConfig":
        kwargs = {"aggregation": "footprint"}
        kwargs.update(overrides)
        return ErrorScoreConfig(**kwargs)

    def test_matches_center_on_isotropic_footprint_constant_error(self) -> None:
        # A constant error map makes any normalized weighted average equal the
        # center sample, so v1 and v2 must agree to float tolerance.
        err = torch.full((12, 12), 0.37)
        means2d = torch.tensor([[5.3, 6.1], [2.0, 9.0]])
        radii = torch.tensor([1, 1])  # small isotropic footprint
        conics = torch.tensor([[2.0, 0.0, 2.0], [2.0, 0.0, 2.0]])
        opacities = torch.tensor([0.8, 0.4])
        state_v1 = ErrorScoreState(2, ErrorScoreConfig(ema_decay=0.5))
        state_v1.update(means2d, radii, err, height=12, width=12)
        state_v2 = ErrorScoreState(2, self._v2_config(ema_decay=0.5))
        state_v2.update(
            means2d, radii, err, height=12, width=12,
            conics=conics, opacities=opacities,
        )
        self.assertTrue(
            torch.allclose(state_v1.scores, state_v2.scores, atol=1e-5)
        )

    def test_footprint_sees_offset_high_error_block(self) -> None:
        # Center sits on zero error; the right side of the footprint covers a
        # high-error block. v1 attributes nothing, v2 attributes a large share
        # -- the whole point of the upgrade.
        h = w = 16
        err = torch.zeros(h, w)
        err[:, 7:12] = 1.0  # block right of the center column 5
        means2d = torch.tensor([[5.0, 8.0]])
        radii = torch.tensor([10])
        conics = torch.tensor([[0.08, 0.0, 0.08]])  # sigma ~ 3.5 px isotropic
        opacities = torch.tensor([0.9])
        state_v1 = ErrorScoreState(1, ErrorScoreConfig(ema_decay=0.0))
        state_v1.update(means2d, radii, err, height=h, width=w)
        state_v2 = ErrorScoreState(1, self._v2_config(ema_decay=0.0))
        state_v2.update(
            means2d, radii, err, height=h, width=w,
            conics=conics, opacities=opacities,
        )
        self.assertAlmostEqual(float(state_v1.scores[0]), 0.0, places=6)
        self.assertGreater(float(state_v2.scores[0]), 0.15)
        self.assertGreater(
            float(state_v2.scores[0]) - float(state_v1.scores[0]), 0.1
        )
        # And the aggregate matches the naive oracle.
        expected = _footprint_oracle(
            (5.0, 8.0), (0.08, 0.0, 0.08), 0.9, err, radius_cap=4, r_eff=4
        )
        self.assertAlmostEqual(float(state_v2.scores[0]), expected, places=5)

    def test_anisotropic_conic_responds_more_along_major_axis(self) -> None:
        # Conic elongated along x (small a => large sigma_x): an error spike
        # 3 px away along x must contribute far more than the same spike 3 px
        # away along y.
        h = w = 21
        means2d = torch.tensor([[10.0, 10.0]])
        radii = torch.tensor([10])
        conics = torch.tensor([[0.05, 0.0, 1.0]])
        opacities = torch.tensor([1.0])
        err_x = torch.zeros(h, w)
        err_x[10, 13] = 1.0  # dx = +3
        err_y = torch.zeros(h, w)
        err_y[13, 10] = 1.0  # dy = +3
        scores = []
        for err in (err_x, err_y):
            state = ErrorScoreState(1, self._v2_config(ema_decay=0.0))
            state.update(
                means2d, radii, err, height=h, width=w,
                conics=conics, opacities=opacities,
            )
            scores.append(float(state.scores[0]))
        score_x, score_y = scores
        self.assertGreater(score_x, 0.0)
        self.assertGreater(score_x, 5.0 * score_y)

    def test_out_of_bounds_pixels_get_zero_weight(self) -> None:
        # Near-edge Gaussian: clamped out-of-image offsets must contribute
        # nothing, so the result matches an oracle that simply drops them.
        torch.manual_seed(7)
        err = torch.rand(6, 9)
        means2d = torch.tensor([[0.4, 2.6]])
        radii = torch.tensor([5])  # capped to r_eff = 2 below
        conics = torch.tensor([[0.3, 0.05, 0.5]])
        opacities = torch.tensor([0.6])
        state = ErrorScoreState(
            1, self._v2_config(ema_decay=0.0, footprint_radius_px=2)
        )
        state.update(
            means2d, radii, err, height=6, width=9,
            conics=conics, opacities=opacities,
        )
        expected = _footprint_oracle(
            (0.4, 2.6), (0.3, 0.05, 0.5), 0.6, err, radius_cap=2, r_eff=2
        )
        self.assertAlmostEqual(float(state.scores[0]), expected, places=5)

        # Fully off-screen center: every window pixel is out of bounds, the
        # weight mass is zero, and the Gaussian is skipped (score unchanged).
        state_off = ErrorScoreState(1, self._v2_config(ema_decay=0.5))
        state_off.update(
            torch.tensor([[-40.0, -40.0]]),
            torch.tensor([3]),
            err,
            height=6,
            width=9,
            conics=torch.tensor([[0.5, 0.0, 0.5]]),
            opacities=torch.tensor([0.9]),
        )
        self.assertTrue(torch.allclose(state_off.scores, torch.ones(1)))

    def test_degenerate_conic_falls_back_to_center_sample(self) -> None:
        # Non-finite, non-positive, and non-PD conics must all degrade to the
        # v1 center sample for that Gaussian (and still update the EMA).
        err = (torch.arange(30, dtype=torch.float32) / 100.0).reshape(5, 6)
        means2d = torch.tensor([[2.0, 1.0], [4.0, 3.0], [1.0, 4.0]])
        radii = torch.tensor([3, 3, 3])
        conics = torch.tensor(
            [
                [float("nan"), 0.0, 1.0],  # non-finite
                [-1.0, 0.0, 1.0],  # a <= 0
                [1.0, 2.0, 1.0],  # a*c - b^2 < 0
            ]
        )
        opacities = torch.tensor([0.5, 0.5, 0.5])
        state = ErrorScoreState(3, self._v2_config(ema_decay=0.0))
        state.update(
            means2d, radii, err, height=5, width=6,
            conics=conics, opacities=opacities,
        )
        expected = torch.tensor([err[1, 2], err[3, 4], err[4, 1]])
        self.assertTrue(torch.allclose(state.scores, expected, atol=1e-6))

    def test_center_aggregation_is_bitwise_identical_to_v1(self) -> None:
        # Regression guard: aggregation="center" must run the exact v1 code
        # path whether or not conics/opacities are passed, and a footprint
        # config without conics must also fall back to v1 bit-for-bit.
        torch.manual_seed(3)
        err = torch.rand(10, 10)
        means2d = torch.rand(6, 2) * 12.0 - 1.0  # includes out-of-bounds
        radii = torch.tensor([2, 0, 1, 3, 0, 5])
        conics = torch.rand(6, 3) + torch.tensor([1.0, 0.0, 1.0])
        opacities = torch.rand(6)
        reference = ErrorScoreState(6, ErrorScoreConfig(ema_decay=0.7))
        reference.update(means2d, radii, err, height=10, width=10)

        center_with_extras = ErrorScoreState(6, ErrorScoreConfig(ema_decay=0.7))
        center_with_extras.update(
            means2d, radii, err, height=10, width=10,
            conics=conics, opacities=opacities,
        )
        self.assertTrue(
            torch.equal(reference.scores, center_with_extras.scores)
        )

        footprint_no_conics = ErrorScoreState(6, self._v2_config(ema_decay=0.7))
        footprint_no_conics.update(means2d, radii, err, height=10, width=10)
        self.assertTrue(
            torch.equal(reference.scores, footprint_no_conics.scores)
        )

    def test_conics_shapes_and_count_mismatch(self) -> None:
        err = torch.full((8, 8), 0.25)
        means2d = torch.tensor([[3.0, 3.0], [5.0, 5.0]])
        radii = torch.tensor([2, 2])
        conics_flat = torch.tensor([[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]])
        opacities = torch.tensor([0.7, 0.7])
        state_flat = ErrorScoreState(2, self._v2_config(ema_decay=0.0))
        state_flat.update(
            means2d, radii, err, height=8, width=8,
            conics=conics_flat, opacities=opacities,
        )
        # The gsplat packed=False layout [1, N, 3] / [1, N] is accepted too.
        state_batched = ErrorScoreState(2, self._v2_config(ema_decay=0.0))
        state_batched.update(
            means2d, radii, err, height=8, width=8,
            conics=conics_flat.unsqueeze(0), opacities=opacities.unsqueeze(0),
        )
        self.assertTrue(torch.equal(state_flat.scores, state_batched.scores))
        with self.assertRaisesRegex(ValueError, "conics"):
            ErrorScoreState(2, self._v2_config()).update(
                means2d, radii, err, height=8, width=8,
                conics=torch.zeros(3, 3), opacities=opacities,
            )
        with self.assertRaisesRegex(ValueError, "opacities"):
            ErrorScoreState(2, self._v2_config()).update(
                means2d, radii, err, height=8, width=8,
                conics=conics_flat, opacities=torch.ones(5),
            )

    def test_zero_opacity_gaussian_is_skipped(self) -> None:
        err = torch.full((6, 6), 0.9)
        means2d = torch.tensor([[3.0, 3.0], [2.0, 2.0]])
        radii = torch.tensor([2, 2])
        conics = torch.tensor([[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]])
        opacities = torch.tensor([0.0, 0.8])  # first has zero weight mass
        state = ErrorScoreState(2, self._v2_config(ema_decay=0.0))
        state.update(
            means2d, radii, err, height=6, width=6,
            conics=conics, opacities=opacities,
        )
        self.assertTrue(
            torch.allclose(state.scores, torch.tensor([1.0, 0.9]), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
