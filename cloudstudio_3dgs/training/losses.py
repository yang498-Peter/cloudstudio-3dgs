"""Mask-aware RGB and Euclidean ray-range losses."""

from __future__ import annotations

from typing import Any


def _validate_masked_pair(prediction: Any, target: Any, mask: Any, label: str) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"{label} prediction and target shapes differ")
    if mask.shape != prediction.shape[:-1]:
        raise ValueError(f"{label} mask shape does not match the image")
    if not bool(mask.any().item()):
        raise ValueError(f"{label} mask contains no valid pixels")


def masked_rgb_l1(prediction: Any, target: Any, mask: Any) -> Any:
    _validate_masked_pair(prediction, target, mask, "RGB")
    return (prediction - target).abs()[mask].mean()


def global_masked_rgb_ssim_loss(prediction: Any, target: Any, mask: Any) -> Any:
    """Global masked SSIM loss with channel-wise moments.

    This deliberately computes moments only inside the valid region; black
    fisheye borders therefore cannot improve or degrade the score.
    """
    _validate_masked_pair(prediction, target, mask, "RGB")
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    mean_prediction = selected_prediction.mean(dim=0)
    mean_target = selected_target.mean(dim=0)
    centered_prediction = selected_prediction - mean_prediction
    centered_target = selected_target - mean_target
    variance_prediction = (centered_prediction * centered_prediction).mean(dim=0)
    variance_target = (centered_target * centered_target).mean(dim=0)
    covariance = (centered_prediction * centered_target).mean(dim=0)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (
        (2.0 * mean_prediction * mean_target + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (variance_prediction + variance_target + c2)
        )
    )
    return 1.0 - ssim.mean()


def masked_rgb_ssim_loss(
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    min_valid_fraction: float = 0.8,
) -> Any:
    """Mask-aware local Gaussian-window SSIM.

    Invalid fisheye/person pixels are zeroed before convolution and local
    moments are divided by the valid kernel support. A window contributes only
    when its center is valid and its weighted valid coverage reaches the
    configured threshold.
    """

    _validate_masked_pair(prediction, target, mask, "RGB")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("SSIM window_size must be a positive odd integer")
    if sigma <= 0.0:
        raise ValueError("SSIM sigma must be positive")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("SSIM min_valid_fraction must be in (0, 1]")

    import torch
    import torch.nn.functional as functional

    radius = window_size // 2
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
    channels = prediction.shape[-1]
    channel_kernel = kernel_2d.expand(channels, 1, window_size, window_size)
    mask_4d = mask.to(dtype=prediction.dtype)[None, None]
    valid_prediction = torch.where(mask[..., None], prediction, torch.zeros_like(prediction))
    valid_target = torch.where(mask[..., None], target, torch.zeros_like(target))
    prediction_4d = valid_prediction.permute(2, 0, 1)[None]
    target_4d = valid_target.permute(2, 0, 1)[None]
    support = functional.conv2d(mask_4d, kernel_2d, padding=radius)
    denominator = support.clamp_min(torch.finfo(prediction.dtype).eps)

    def local_mean(value: Any) -> Any:
        return functional.conv2d(value, channel_kernel, padding=radius, groups=channels) / denominator

    mean_prediction = local_mean(prediction_4d)
    mean_target = local_mean(target_4d)
    second_prediction = local_mean(prediction_4d.square())
    second_target = local_mean(target_4d.square())
    cross = local_mean(prediction_4d * target_4d)
    variance_prediction = (second_prediction - mean_prediction.square()).clamp_min(0.0)
    variance_target = (second_target - mean_target.square()).clamp_min(0.0)
    covariance = cross - mean_prediction * mean_target
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (
        (2.0 * mean_prediction * mean_target + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (variance_prediction + variance_target + c2)
        )
    ).mean(dim=1)[0]
    valid_windows = mask & (support[0, 0] >= min_valid_fraction)
    if not bool(valid_windows.any().item()):
        raise ValueError("SSIM mask has no valid local windows at the configured coverage")
    return 1.0 - ssim[valid_windows].mean()


def confidence_weighted_range_l1(
    prediction_range_m: Any,
    target_range_m: Any,
    confidence: Any,
    mask: Any,
) -> Any:
    if prediction_range_m.shape != target_range_m.shape:
        raise ValueError("range prediction and target shapes differ")
    if confidence.shape != target_range_m.shape or mask.shape != target_range_m.shape:
        raise ValueError("range confidence/mask shapes do not match the target")
    valid = (
        mask
        & prediction_range_m.isfinite()
        & target_range_m.isfinite()
        & confidence.isfinite()
        & (prediction_range_m > 0.0)
        & (target_range_m > 0.0)
        & (confidence > 0.0)
    )
    if not bool(valid.any().item()):
        raise ValueError("range mask contains no finite positive supervision")
    weights = confidence[valid]
    return ((prediction_range_m[valid] - target_range_m[valid]).abs() * weights).sum() / weights.sum()


def confidence_weighted_log_range_huber(
    prediction_range_m: Any,
    target_range_m: Any,
    confidence: Any,
    mask: Any,
    *,
    delta: float = 0.05,
) -> Any:
    """Confidence-weighted smooth-L1 over log Euclidean ray range."""

    if delta <= 0.0:
        raise ValueError("log-range Huber delta must be positive")
    import torch

    if prediction_range_m.shape != target_range_m.shape:
        raise ValueError("range prediction and target shapes differ")
    if confidence.shape != target_range_m.shape or mask.shape != target_range_m.shape:
        raise ValueError("range confidence/mask shapes do not match the target")
    valid = (
        mask
        & prediction_range_m.isfinite()
        & target_range_m.isfinite()
        & confidence.isfinite()
        & (prediction_range_m > 0.0)
        & (target_range_m > 0.0)
        & (confidence > 0.0)
    )
    if not bool(valid.any().item()):
        raise ValueError("range mask contains no finite positive supervision")
    residual = (
        prediction_range_m[valid].log() - target_range_m[valid].log()
    ).abs()
    quadratic = 0.5 * residual.square() / delta
    linear = residual - 0.5 * delta
    per_pixel = torch.where(residual <= delta, quadratic, linear)
    weights = confidence[valid]
    return (per_pixel * weights).sum() / weights.sum()
