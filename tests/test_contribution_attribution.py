"""Tests for dummy-channel contribution attribution (WP-2 prototype).

The CPU lane covers config validation, error-map construction and the full
gradient path via a linear stub rasterizer whose alpha/T weights are known in
closed form, so ``dL/d(dummy)`` has an exact expected value without a GPU.

The CUDA lane exercises the real gsplat rasterizer on both CloudStudio render
paths. Do NOT mutate CUDA_VISIBLE_DEVICES at module level - unittest discovery
imports this file into the shared process and a hidden device would break the
GPU contract tests that run after it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - optional training dependency
    torch = None

if torch is not None:
    try:
        from cloudstudio_3dgs.training.contribution_attribution import (
            ContributionConfig,
            build_error_map,
            compute_contribution_scores,
            ssim_dissimilarity_map,
        )

        _IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover
        ContributionConfig = None
        build_error_map = compute_contribution_scores = None
        ssim_dissimilarity_map = None
        _IMPORT_ERROR = exc
else:  # pragma: no cover
    ContributionConfig = None
    build_error_map = compute_contribution_scores = None
    ssim_dissimilarity_map = None
    _IMPORT_ERROR = "torch missing"

HAS_MODULE = torch is not None and _IMPORT_ERROR is None
HAS_CUDA = HAS_MODULE and torch.cuda.is_available()

H, W, N = 6, 8, 5


def _sample(camera_model="pinhole"):
    K = [[10.0, 0.0, W / 2], [0.0, 10.0, H / 2], [0.0, 0.0, 1.0]]
    fields = dict(
        c2w=torch.eye(4).tolist(),
        K=K,
        width=W,
        height=H,
        camera_model=camera_model,
    )
    if camera_model == "fisheye":
        fields["radial_coeffs"] = [0.0, 0.0, 0.0, 0.0]
    return SimpleNamespace(**fields)


class _StubBackend:
    """Linear stand-in for gsplat: ``render[y,x,d] = sum_i weights[i,y,x]*colors[i,d]``.

    That is exactly the algebraic form of alpha-blending a per-Gaussian scalar
    channel, so the gradient the module reads back has a closed-form value.
    """

    device = "cpu"
    pinhole_rasterize_mode = "antialiased"

    def __init__(self, weights):
        self.weights = weights  # [N, H, W]
        self.calls = []

    def rasterization(self, **kwargs):
        self.calls.append(kwargs)
        colors = kwargs["colors"]
        # [N,H,W] x [N,D] -> [1,H,W,D]
        render = torch.einsum("nhw,nd->hwd", self.weights, colors)[None]
        alpha = self.weights.sum(dim=0)[None, ..., None]
        return render, alpha, {}


def _weights():
    torch.manual_seed(7)
    return torch.rand(N, H, W)


def _params(requires_grad=True):
    torch.manual_seed(11)
    return {
        "means": torch.randn(N, 3, requires_grad=requires_grad),
        "quats": torch.randn(N, 4, requires_grad=requires_grad),
        "scales": torch.randn(N, 3, requires_grad=requires_grad),
        "opacities": torch.randn(N, requires_grad=requires_grad),
    }


@unittest.skipUnless(HAS_MODULE, f"module unavailable: {_IMPORT_ERROR}")
class ContributionConfigTest(unittest.TestCase):
    def test_defaults_are_inert(self):
        config = ContributionConfig()
        self.assertFalse(config.enabled)
        config.validate()

    def test_to_dict_round_trips_every_field(self):
        config = ContributionConfig(
            enabled=True, error_map_mode="ssim", ssim_weight=0.25, normalize=False
        )
        payload = config.to_dict()
        self.assertEqual(payload["error_map_mode"], "ssim")
        self.assertEqual(payload["ssim_weight"], 0.25)
        self.assertFalse(payload["normalize"])
        self.assertEqual(ContributionConfig(**payload), config)

    def test_rejects_unknown_error_map_mode(self):
        with self.assertRaises(ValueError):
            ContributionConfig(error_map_mode="lpips").validate()

    def test_rejects_out_of_range_ssim_weight(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                ContributionConfig(ssim_weight=bad).validate()

    def test_rejects_even_or_tiny_ssim_window(self):
        for bad in (10, 1, 2):
            with self.assertRaises(ValueError):
                ContributionConfig(ssim_window=bad).validate()

    def test_rejects_non_positive_sigma_and_eps(self):
        with self.assertRaises(ValueError):
            ContributionConfig(ssim_sigma=0.0).validate()
        with self.assertRaises(ValueError):
            ContributionConfig(eps=0.0).validate()


@unittest.skipUnless(HAS_MODULE, f"module unavailable: {_IMPORT_ERROR}")
class ErrorMapTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.pred = torch.rand(H, W, 3)
        self.ref = torch.rand(H, W, 3)

    def test_l1_matches_channel_mean_absolute_residual(self):
        got = build_error_map(
            self.pred, self.ref, config=ContributionConfig(error_map_mode="l1")
        )
        expected = (self.pred - self.ref).abs().mean(dim=-1)
        self.assertEqual(got.shape, (H, W))
        self.assertTrue(torch.allclose(got, expected))

    def test_ssim_of_identical_images_is_zero(self):
        config = ContributionConfig(error_map_mode="ssim")
        got = build_error_map(self.pred, self.pred, config=config)
        self.assertLess(float(got.max()), 1e-5)

    def test_ssim_of_different_images_is_positive(self):
        config = ContributionConfig(error_map_mode="ssim")
        got = build_error_map(self.pred, self.ref, config=config)
        self.assertGreater(float(got.max()), 1e-3)
        self.assertGreaterEqual(float(got.min()), 0.0)

    def test_l1_ssim_is_the_documented_blend(self):
        config = ContributionConfig(error_map_mode="l1_ssim", ssim_weight=0.3)
        blended = build_error_map(self.pred, self.ref, config=config)
        l1 = build_error_map(
            self.pred, self.ref, config=ContributionConfig(error_map_mode="l1")
        )
        dssim = build_error_map(
            self.pred, self.ref, config=ContributionConfig(error_map_mode="ssim")
        )
        self.assertTrue(torch.allclose(blended, 0.7 * l1 + 0.3 * dssim, atol=1e-6))

    def test_mask_zeroes_invalid_pixels(self):
        mask = torch.ones(H, W)
        mask[:, : W // 2] = 0.0
        got = build_error_map(
            self.pred, self.ref, config=ContributionConfig(), mask=mask
        )
        self.assertEqual(float(got[:, : W // 2].abs().max()), 0.0)
        self.assertGreater(float(got[:, W // 2 :].abs().max()), 0.0)

    def test_output_is_detached_from_the_training_graph(self):
        pred = self.pred.clone().requires_grad_(True)
        got = build_error_map(pred, self.ref, config=ContributionConfig())
        self.assertFalse(got.requires_grad)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            build_error_map(self.pred, torch.rand(H, W, 1), config=ContributionConfig())
        with self.assertRaises(ValueError):
            build_error_map(
                self.pred, self.ref, config=ContributionConfig(), mask=torch.ones(2, 2)
            )

    def test_ssim_map_rejects_non_image_input(self):
        with self.assertRaises(ValueError):
            ssim_dissimilarity_map(
                torch.rand(H, W), torch.rand(H, W), config=ContributionConfig()
            )


@unittest.skipUnless(HAS_MODULE, f"module unavailable: {_IMPORT_ERROR}")
class ContributionScoreTest(unittest.TestCase):
    def setUp(self):
        self.weights = _weights()
        self.backend = _StubBackend(self.weights)
        self.params = _params()
        torch.manual_seed(5)
        self.error_map = torch.rand(H, W)

    def _scores(self, **overrides):
        config = ContributionConfig(**{"normalize": False, **overrides})
        return compute_contribution_scores(
            self.backend,
            self.params,
            _sample(overrides.pop("camera_model", "pinhole")),
            self.error_map,
            config=config,
        )

    def test_scores_equal_the_error_weighted_alpha_T_integral(self):
        scores = self._scores()
        expected = (self.weights * self.error_map[None]).sum(dim=(1, 2))
        self.assertEqual(scores.shape, (N,))
        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))

    def test_occluded_gaussian_scores_below_an_identical_visible_one(self):
        # Two Gaussians with the same footprint but the second carrying a tenth
        # of the transmittance-weighted alpha must rank strictly lower.
        weights = torch.zeros(2, H, W)
        weights[0] = 0.5
        weights[1] = 0.05
        backend = _StubBackend(weights)
        params = {k: v[:2] for k, v in _params().items()}
        scores = compute_contribution_scores(
            backend,
            params,
            _sample(),
            torch.ones(H, W),
            config=ContributionConfig(normalize=False),
        )
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertAlmostEqual(float(scores[0] / scores[1]), 10.0, places=4)

    def test_normalization_maps_the_peak_to_one(self):
        scores = compute_contribution_scores(
            self.backend,
            self.params,
            _sample(),
            self.error_map,
            config=ContributionConfig(normalize=True),
        )
        self.assertAlmostEqual(float(scores.max()), 1.0, places=5)
        self.assertGreaterEqual(float(scores.min()), 0.0)

    def test_dummy_channel_is_single_channel_and_bypasses_sh(self):
        self._scores()
        call = self.backend.calls[-1]
        self.assertIsNone(call["sh_degree"])
        self.assertEqual(tuple(call["colors"].shape), (N, 1))
        self.assertEqual(call["render_mode"], "RGB")

    def test_real_colour_params_are_never_read_or_polluted(self):
        params = _params()
        params["sh0"] = torch.randn(N, 1, 3, requires_grad=True)
        params["shN"] = torch.randn(N, 8, 3, requires_grad=True)
        params["colors"] = torch.randn(N, 3, requires_grad=True)
        compute_contribution_scores(
            self.backend,
            params,
            _sample(),
            self.error_map,
            config=ContributionConfig(),
        )
        colour_call = self.backend.calls[-1]["colors"]
        for name in ("sh0", "shN", "colors"):
            self.assertIsNone(params[name].grad, f"{name}.grad was written")
            self.assertIsNot(colour_call, params[name])

    def test_no_gradient_is_accumulated_on_geometry_params(self):
        self._scores()
        for name in ("means", "quats", "scales", "opacities"):
            self.assertIsNone(self.params[name].grad, f"{name}.grad was written")
            self.assertFalse(self.backend.calls[-1][name].requires_grad)

    def test_stored_activations_are_applied_before_rasterizing(self):
        self._scores()
        call = self.backend.calls[-1]
        self.assertTrue(
            torch.allclose(call["scales"], torch.exp(self.params["scales"].detach()))
        )
        self.assertTrue(
            torch.allclose(
                call["opacities"], torch.sigmoid(self.params["opacities"].detach())
            )
        )

    def test_pinhole_path_disables_ut_and_honours_backend_rasterize_mode(self):
        self._scores()
        call = self.backend.calls[-1]
        self.assertFalse(call["with_ut"])
        self.assertFalse(call["with_eval3d"])
        self.assertTrue(call["global_z_order"])
        self.assertNotIn("radial_coeffs", call)
        self.assertEqual(call["rasterize_mode"], "antialiased")

    def test_fisheye_path_enables_ut_and_passes_distortion(self):
        compute_contribution_scores(
            self.backend,
            self.params,
            _sample("fisheye"),
            self.error_map,
            config=ContributionConfig(),
        )
        call = self.backend.calls[-1]
        self.assertTrue(call["with_ut"])
        self.assertTrue(call["with_eval3d"])
        self.assertFalse(call["global_z_order"])
        self.assertEqual(call["rasterize_mode"], "classic")
        self.assertEqual(tuple(call["radial_coeffs"].shape), (1, 4))

    def test_rejects_unsupported_camera_model(self):
        with self.assertRaises(ValueError):
            compute_contribution_scores(
                self.backend,
                self.params,
                _sample("equirect"),
                self.error_map,
                config=ContributionConfig(),
            )

    def test_rejects_error_map_of_the_wrong_shape(self):
        with self.assertRaises(ValueError):
            compute_contribution_scores(
                self.backend,
                self.params,
                _sample(),
                torch.rand(H + 1, W),
                config=ContributionConfig(),
            )

    def test_empty_cloud_returns_an_empty_tensor(self):
        params = {k: v[:0] for k, v in _params().items()}
        scores = compute_contribution_scores(
            _StubBackend(torch.zeros(0, H, W)),
            params,
            _sample(),
            self.error_map,
            config=ContributionConfig(),
        )
        self.assertEqual(tuple(scores.shape), (0,))

    def test_scores_land_on_the_parameter_device_and_are_detached(self):
        scores = self._scores()
        self.assertEqual(scores.device, self.params["means"].device)
        self.assertFalse(scores.requires_grad)


@unittest.skipUnless(HAS_CUDA, "CUDA + gsplat required for the real rasterizer")
class RealRasterizerContributionTest(unittest.TestCase):
    """Exercise the true gsplat kernels on both CloudStudio render paths.

    The rendered channel is exactly linear in the dummy values, so an all-ones
    channel must reproduce the rasterizer's own alpha output - an end-to-end
    check that the gradient really is the alpha/transmittance weight.
    """

    COUNT = 24
    SIZE = 32

    def _backend(self):
        from gsplat.rendering import rasterization

        return SimpleNamespace(
            device="cuda", rasterization=rasterization, pinhole_rasterize_mode="classic"
        )

    def _cloud(self):
        torch.manual_seed(0)
        dev = "cuda"
        means = torch.randn(self.COUNT, 3, device=dev) * 0.35
        means[:, 2] += torch.linspace(2.0, 5.0, self.COUNT, device=dev)
        quats = torch.randn(self.COUNT, 4, device=dev)
        return {
            "means": means,
            "quats": quats / quats.norm(dim=-1, keepdim=True),
            "scales": torch.log(torch.rand(self.COUNT, 3, device=dev) * 0.15 + 0.08),
            "opacities": torch.logit(
                torch.rand(self.COUNT, device=dev) * 0.6 + 0.2
            ),
        }

    def _sample(self, camera_model):
        size = self.SIZE
        K = [[30.0, 0.0, size / 2], [0.0, 30.0, size / 2], [0.0, 0.0, 1.0]]
        fields = dict(
            c2w=torch.eye(4).tolist(),
            K=K,
            width=size,
            height=size,
            camera_model=camera_model,
        )
        if camera_model == "fisheye":
            fields["radial_coeffs"] = [0.0, 0.0, 0.0, 0.0]
        return SimpleNamespace(**fields)

    def _check(self, camera_model):
        backend, params = self._backend(), self._cloud()
        sample = self._sample(camera_model)
        torch.manual_seed(1)
        error_map = torch.rand(self.SIZE, self.SIZE, device="cuda")
        scores = compute_contribution_scores(
            backend,
            params,
            sample,
            error_map,
            config=ContributionConfig(normalize=False),
        )
        self.assertEqual(tuple(scores.shape), (self.COUNT,))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertGreater(float(scores.max()), 0.0)
        self.assertGreaterEqual(float(scores.min()), 0.0)

        # Linearity: an all-ones channel renders sum_i alpha_i*T_i == alpha.
        from cloudstudio_3dgs.training.contribution_attribution import _camera_tensors

        render, alpha, _ = backend.rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.ones(self.COUNT, 1, device="cuda"),
            sh_degree=None,
            **_camera_tensors(backend, sample),
        )
        self.assertLess(float((render[..., 0] - alpha[..., 0]).abs().max()), 1e-4)

    def test_pinhole_classic_path(self):
        self._check("pinhole")

    def test_fisheye_3dgut_path(self):
        self._check("fisheye")

    def test_occluded_gaussian_ranks_below_the_occluder(self):
        # Eight identical gaussians stacked along the optical axis: contribution
        # must decay monotonically with depth as transmittance is consumed.
        backend = self._backend()
        dev, k = "cuda", 8
        zs = torch.arange(k, dtype=torch.float32, device=dev) + 2.0
        means = torch.zeros(k, 3, device=dev)
        means[:, 2] = zs
        quats = torch.zeros(k, 4, device=dev)
        quats[:, 0] = 1.0
        params = {
            "means": means,
            "quats": quats,
            "scales": torch.log((0.25 * zs / zs[0])[:, None].repeat(1, 3)),
            "opacities": torch.logit(torch.full((k,), 0.5, device=dev)),
        }
        scores = compute_contribution_scores(
            backend,
            params,
            self._sample("pinhole"),
            torch.ones(self.SIZE, self.SIZE, device=dev),
            config=ContributionConfig(normalize=False),
        )
        diffs = scores[:-1] - scores[1:]
        self.assertTrue(bool((diffs > 0).all()), f"not monotone: {scores.tolist()}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
