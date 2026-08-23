# SPDX-License-Identifier: Apache-2.0
#
# The dummy-channel attribution technique implemented here is adapted from
# CAFe-GS (https://github.com/InterDigitalInc/CAFe-GS, commit 89b63b9,
# BSD-3-Clause, Copyright (c) 2010-2024 InterDigital), specifically
# ``compute_per_gaussian_score`` in ``cafe_densification_strategy.py``:
# render a non-learnable scalar channel through the rasterizer and read
# ``dL/d(channel)`` as the alpha/transmittance-weighted per-Gaussian error.
# No CAFe-GS source is copied verbatim; the camera plumbing, config surface
# and error-map construction are CloudStudio's. Note that the upstream
# licence explicitly withholds patent rights.
"""Real contribution attribution for per-Gaussian error scores (WP-2 prototype).

Motivation
----------
``ErrorScoreState._update_footprint`` (v2) weights the RGB residual inside each
Gaussian's projected 2D footprint by ``opacity * exp(-0.5 d^T Sigma^-1 d)`` and
then divides by the weight sum. Two consequences make it a *coverage* measure
rather than a *contribution* measure:

* the per-Gaussian opacity cancels in the normalized ratio, so a nearly
  transparent Gaussian scores the same as an opaque one over the same pixels;
* the forward pass's transmittance ``T`` never enters, so a Gaussian fully
  hidden behind an opaque foreground still collects the foreground's error.

This module computes the quantity the rasterizer itself already knows. Because
the rendered image is *exactly linear* in a per-Gaussian scalar channel ``e``::

    contrib(x) = sum_i alpha_i(x) * T_i(x) * e_i

back-propagating ``L = sum_x E(x) * contrib(x)`` into ``e`` yields::

    dL/de_i = sum_x E(x) * alpha_i(x) * T_i(x)

i.e. the error mass genuinely blended out of Gaussian ``i``, occlusion and
opacity included, for the cost of one extra forward+backward over a single
channel. Nothing needs to be stored per Gaussian-pixel pair.

Verified on this machine against gsplat 1.5.3 for both CloudStudio render
paths (pinhole classic and fisheye 3DGUT ``with_ut/with_eval3d``): the returned
gradient matches per-Gaussian weight maps rendered explicitly with one-hot
channels to ~1e-8 relative error, and an all-ones channel reproduces the
rasterizer's own alpha output to ~2e-7 absolute.

Scope
-----
Prototype only: nothing here is wired into ``error_weighted_mcmc`` yet. The
attribution runs as a *separate* rasterization pass so it cannot perturb the
training graph. gsplat also accepts an ``extra_signals`` channel on the main
render, which the probe showed to be bit-identical and saves the extra forward;
that route requires touching ``backend.render`` and is deferred to WP-2 step 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

ERROR_MAP_MODES = ("l1", "ssim", "l1_ssim")


@dataclass(frozen=True)
class ContributionConfig:
    """Configuration for dummy-channel contribution attribution.

    Attributes:
        enabled: master switch; off by default so importing this module cannot
            change training behaviour.
        error_map_mode: how the per-pixel error map is built from the rendered
            and reference RGB. ``"l1"`` is the channel-mean absolute residual
            (what the existing v1/v2 score paths consume), ``"ssim"`` is the
            local dissimilarity map ``1 - SSIM``, and ``"l1_ssim"`` blends them
            with ``ssim_weight``.
        ssim_weight: blend factor for ``"l1_ssim"``; ``0`` is pure L1 and ``1``
            is pure SSIM.
        ssim_window: side length of the Gaussian SSIM window in pixels; must be
            odd so the window stays centered.
        ssim_sigma: standard deviation of that window.
        normalize: divide the returned scores by their maximum, making them
            comparable across views with different error magnitudes and image
            sizes. Recommended when the scores feed an EMA shared across views.
        eps: numerical floor for the normalization divisor.
    """

    enabled: bool = False
    error_map_mode: str = "l1"
    ssim_weight: float = 0.5
    ssim_window: int = 11
    ssim_sigma: float = 1.5
    normalize: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        if self.error_map_mode not in ERROR_MAP_MODES:
            raise ValueError(
                "error_map_mode must be one of " + ", ".join(ERROR_MAP_MODES)
            )
        if not 0.0 <= float(self.ssim_weight) <= 1.0:
            raise ValueError("ssim_weight must be within [0, 1]")
        window = int(self.ssim_window)
        if window < 3 or window % 2 == 0:
            raise ValueError("ssim_window must be an odd integer >= 3")
        if not float(self.ssim_sigma) > 0.0:
            raise ValueError("ssim_sigma must be positive")
        if not float(self.eps) > 0.0:
            raise ValueError("eps must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": self.enabled,
            "error_map_mode": self.error_map_mode,
            "ssim_weight": self.ssim_weight,
            "ssim_window": self.ssim_window,
            "ssim_sigma": self.ssim_sigma,
            "normalize": self.normalize,
            "eps": self.eps,
        }


def _gaussian_window(size: int, sigma: float, device: Any, dtype: Any) -> Tensor:
    """Separable 2D Gaussian kernel normalized to unit sum, shape [1, 1, k, k]."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    line = torch.exp(-(coords**2) / (2.0 * float(sigma) ** 2))
    line = line / line.sum()
    return (line[:, None] * line[None, :])[None, None]


def ssim_dissimilarity_map(
    prediction: Tensor, reference: Tensor, *, config: ContributionConfig
) -> Tensor:
    """Per-pixel ``1 - SSIM`` for two [H, W, 3] images, returned as [H, W].

    Pure torch (no ``fused_ssim`` dependency) so the CPU test lane can exercise
    it. Windows are applied with ``'same'`` padding in replicate mode; SSIM is
    averaged over the colour channels.
    """
    if prediction.shape != reference.shape or prediction.dim() != 3:
        raise ValueError("prediction and reference must both be [H, W, C]")
    channels = int(prediction.shape[-1])
    # [H, W, C] -> [1, C, H, W] for conv2d.
    pred = prediction.permute(2, 0, 1)[None].float()
    ref = reference.permute(2, 0, 1)[None].float()

    window = int(config.ssim_window)
    kernel = _gaussian_window(window, config.ssim_sigma, pred.device, pred.dtype)
    kernel = kernel.expand(channels, 1, window, window).contiguous()
    pad = window // 2

    def blur(value: Tensor) -> Tensor:
        padded = torch.nn.functional.pad(
            value, (pad, pad, pad, pad), mode="replicate"
        )
        return torch.nn.functional.conv2d(padded, kernel, groups=channels)

    mu_p, mu_r = blur(pred), blur(ref)
    mu_p2, mu_r2, mu_pr = mu_p * mu_p, mu_r * mu_r, mu_p * mu_r
    # Local (co)variances; clamping keeps the numerically-negative variances
    # that a finite window can produce from flipping the SSIM sign.
    sigma_p2 = (blur(pred * pred) - mu_p2).clamp_min(0.0)
    sigma_r2 = (blur(ref * ref) - mu_r2).clamp_min(0.0)
    sigma_pr = blur(pred * ref) - mu_pr

    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2.0 * mu_pr + c1) * (2.0 * sigma_pr + c2)) / (
        (mu_p2 + mu_r2 + c1) * (sigma_p2 + sigma_r2 + c2)
    )
    return (1.0 - ssim).mean(dim=1)[0].clamp_min(0.0)


def build_error_map(
    prediction: Tensor,
    reference: Tensor,
    *,
    config: ContributionConfig,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Build the [H, W] per-pixel error map the attribution pass integrates.

    Args:
        prediction: [H, W, 3] rendered RGB.
        reference: [H, W, 3] ground-truth RGB.
        config: mode selector; see :class:`ContributionConfig`.
        mask: optional [H, W] validity mask (fisheye borders, person masks).
            Applied multiplicatively so invalid pixels contribute no error.

    Returns:
        [H, W] non-negative error map, detached from the training graph.
    """
    config.validate()
    if prediction.shape != reference.shape or prediction.dim() != 3:
        raise ValueError("prediction and reference must both be [H, W, C]")
    # Detached throughout: this map is a *weight* on the attribution loss, and
    # letting it carry gradient would feed image-space error back into the real
    # parameters a second time.
    prediction = prediction.detach()
    reference = reference.detach().to(prediction.device)

    mode = config.error_map_mode
    if mode in ("l1", "l1_ssim"):
        l1 = (prediction - reference).abs().mean(dim=-1)
    if mode in ("ssim", "l1_ssim"):
        dssim = ssim_dissimilarity_map(prediction, reference, config=config)

    if mode == "l1":
        error = l1
    elif mode == "ssim":
        error = dssim
    else:
        weight = float(config.ssim_weight)
        error = (1.0 - weight) * l1 + weight * dssim

    if mask is not None:
        mask = mask.detach().to(error.device)
        if mask.shape != error.shape:
            raise ValueError("mask must have shape [H, W] matching the images")
        error = error * mask.to(error.dtype)
    return error


def _camera_tensors(backend: Any, sample: Any, c2w_override: Any = None) -> dict:
    """Mirror ``backend.render``'s camera plumbing for the attribution pass.

    Deliberately duplicated rather than refactored out of ``backend.render``:
    this prototype must not modify existing files. WP-2 step 2 should hoist the
    shared setup once the wiring lands.
    """
    device = backend.device
    c2w = torch.as_tensor(
        sample.c2w if c2w_override is None else c2w_override,
        dtype=torch.float32,
        device=device,
    )[None]
    camera_model = getattr(sample, "camera_model", "fisheye")
    if camera_model not in ("fisheye", "pinhole"):
        raise ValueError(f"unsupported sample camera_model {camera_model!r}")
    fisheye = camera_model == "fisheye"
    kwargs = {
        "viewmats": torch.linalg.inv(c2w),
        "Ks": torch.as_tensor(sample.K, device=device)[None],
        "width": int(sample.width),
        "height": int(sample.height),
        "packed": False,
        # Colour only: the attribution channel carries no depth semantics, and
        # dropping the depth channel avoids the eval3d hit-distance machinery.
        "render_mode": "RGB",
        "camera_model": camera_model,
        "with_ut": fisheye,
        "with_eval3d": fisheye,
        "global_z_order": not fisheye,
        "rasterize_mode": "classic"
        if fisheye
        else getattr(backend, "pinhole_rasterize_mode", "classic"),
    }
    if fisheye:
        kwargs["radial_coeffs"] = torch.as_tensor(
            sample.radial_coeffs, device=device
        )[None]
    return kwargs


def compute_contribution_scores(
    backend: Any,
    params: Any,
    sample: Any,
    error_map: Tensor,
    *,
    config: ContributionConfig,
    c2w_override: Any = None,
) -> Tensor:
    """Per-Gaussian ``sum_x E(x) * alpha_i(x) * T_i(x)``, shape [N].

    Renders a single non-learnable scalar channel through the same rasterizer
    configuration ``backend.render`` uses for this sample, integrates it against
    ``error_map``, and reads the channel's gradient.

    The real colour parameters are never touched: the pass overrides ``colors``
    with the dummy channel and passes ``sh_degree=None``, so the SH vs
    ``rgb_sigmoid`` colour model is irrelevant here. Geometry parameters are
    detached and the gradient is taken with :func:`torch.autograd.grad` against
    the dummy channel alone, so no ``.grad`` on any optimized leaf is written.

    Args:
        backend: object exposing ``rasterization``, ``device``, and optionally
            ``pinhole_rasterize_mode`` (the training backend).
        params: mapping with ``means``/``quats``/``scales``/``opacities``;
            ``scales`` are log-scales and ``opacities`` are logits, matching the
            storage convention ``backend.render`` activates.
        sample: view descriptor with ``c2w``, ``K``, ``width``, ``height``,
            ``camera_model`` and (for fisheye) ``radial_coeffs``.
        error_map: [H, W] per-pixel error, e.g. from :func:`build_error_map`.
        config: see :class:`ContributionConfig`.
        c2w_override: optional pose override, matching ``backend.render``.

    Returns:
        [N] detached, non-negative float tensor. Optionally max-normalized.
        A cloud with zero Gaussians returns an empty tensor.
    """
    config.validate()
    means = params["means"]
    count = int(means.shape[0])
    device = means.device
    if count == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    error_map = torch.as_tensor(error_map).detach()
    expected = (int(sample.height), int(sample.width))
    if error_map.dim() != 2 or tuple(error_map.shape) != expected:
        raise ValueError(
            f"error_map must have shape {expected}, got {tuple(error_map.shape)}"
        )
    error_map = error_map.to(device=device, dtype=torch.float32)

    # The only leaf we differentiate. Value is irrelevant (the render is linear
    # in it); zeros keep the forward output at exactly 0 so a stray consumer of
    # the contribution image cannot mistake it for radiance.
    dummy = torch.zeros(count, 1, dtype=torch.float32, device=device)
    dummy.requires_grad_(True)

    kwargs = _camera_tensors(backend, sample, c2w_override)
    render, _alpha, _info = backend.rasterization(
        means=means.detach(),
        quats=params["quats"].detach(),
        # Same activations backend.render applies: stored log-scales and
        # opacity logits must be exponentiated / squashed before rasterizing.
        scales=torch.exp(params["scales"].detach()),
        opacities=torch.sigmoid(params["opacities"].detach()),
        colors=dummy,
        sh_degree=None,
        **kwargs,
    )
    contribution = render[0, ..., 0]
    loss = (contribution * error_map).sum()
    (grad,) = torch.autograd.grad(loss, dummy)
    scores = grad[:, 0].detach()

    # alpha, T and a non-negative error map are all non-negative, so the exact
    # result is too; clamp only absorbs float32 round-off near zero.
    scores = scores.clamp_min(0.0)
    if config.normalize:
        scores = scores / scores.max().clamp_min(float(config.eps))
    return scores
