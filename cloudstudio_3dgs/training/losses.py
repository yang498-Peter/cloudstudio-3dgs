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


def masked_rgb_psnr_db(
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    data_range: float = 1.0,
) -> Any:
    """Return mask-aware RGB PSNR in dB using evaluation-compatible clipping."""

    _validate_masked_pair(prediction, target, mask, "RGB PSNR")
    if data_range <= 0.0:
        raise ValueError("RGB PSNR data_range must be positive")
    import torch

    clipped = prediction.clamp(0.0, data_range)
    mse = (clipped - target).square()[mask].mean()
    mse = mse.clamp_min(torch.finfo(mse.dtype).tiny)
    scale = mse.new_tensor(float(data_range * data_range))
    return 10.0 * torch.log10(scale / mse)


def masked_rgb_gradient_l1(prediction: Any, target: Any, mask: Any) -> Any:
    """Match horizontal and vertical RGB finite differences inside the mask."""

    _validate_masked_pair(prediction, target, mask, "RGB gradient")
    horizontal_mask = mask[:, 1:] & mask[:, :-1]
    vertical_mask = mask[1:, :] & mask[:-1, :]
    valid_pairs = int(horizontal_mask.sum().item() + vertical_mask.sum().item())
    if valid_pairs == 0:
        raise ValueError("RGB gradient mask contains no valid neighboring pixels")
    horizontal_error = (
        (prediction[:, 1:] - prediction[:, :-1])
        - (target[:, 1:] - target[:, :-1])
    ).abs()
    vertical_error = (
        (prediction[1:, :] - prediction[:-1, :])
        - (target[1:, :] - target[:-1, :])
    ).abs()
    channels = prediction.shape[-1]
    return (
        horizontal_error[horizontal_mask].sum()
        + vertical_error[vertical_mask].sum()
    ) / (valid_pairs * channels)


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
    luminance_gain: Any | None = None,
) -> Any:
    """Mask-aware local Gaussian-window SSIM.

    Invalid fisheye/person pixels are zeroed before convolution and local
    moments are divided by the valid kernel support. A window contributes only
    when its center is valid and its weighted valid coverage reaches the
    configured threshold.

    With ``luminance_gain`` the SSIM is decoupled for exposure compensation:
    only the luminance term sees the gain-corrected prediction while the
    contrast/structure term compares the raw prediction against the target, so
    the exposure parameters can move brightness but can never mask structural
    error. For a scalar gain g the corrected local moments are exactly
    mean*g / var*g^2 / cov*g, so no second convolution pass is needed.
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
    if luminance_gain is None:
        luminance_mean = mean_prediction
    else:
        luminance_mean = mean_prediction * luminance_gain
    luminance = (2.0 * luminance_mean * mean_target + c1) / (
        luminance_mean.square() + mean_target.square() + c1
    )
    contrast_structure = (2.0 * covariance + c2) / (
        variance_prediction + variance_target + c2
    )
    ssim = (luminance * contrast_structure).mean(dim=1)[0]
    valid_windows = mask & (support[0, 0] >= min_valid_fraction)
    if not bool(valid_windows.any().item()):
        raise ValueError("SSIM mask has no valid local windows at the configured coverage")
    return 1.0 - ssim[valid_windows].mean()


def _range_supervision_masks(prediction_range_m, target_range_m, confidence, mask):
    """Split "no supervision" from "no prediction" - they mean different things.

    An empty mask, target or confidence means the dataset or config is broken
    and there is genuinely nothing to learn from; that must stay fatal. A
    degenerate PREDICTION alongside valid supervision is a transient training
    state - classic 3DGS resets every opacity periodically, so the render is
    legitimately empty for one step - and raising there rejects a published
    mechanism rather than catching a bug. Conflating the two made the reference
    densification path die at exactly its first opacity reset.
    """
    supervised = (
        mask
        & target_range_m.isfinite()
        & confidence.isfinite()
        & (target_range_m > 0.0)
        & (confidence > 0.0)
    )
    predicted = prediction_range_m.isfinite() & (prediction_range_m > 0.0)
    return supervised, supervised & predicted


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
    supervised, valid = _range_supervision_masks(
        prediction_range_m, target_range_m, confidence, mask
    )
    if not bool(supervised.any().item()):
        raise ValueError("range mask contains no finite positive supervision")
    if not bool(valid.any().item()):
        return (prediction_range_m * 0.0).sum()
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
    supervised, valid = _range_supervision_masks(
        prediction_range_m, target_range_m, confidence, mask
    )
    if not bool(supervised.any().item()):
        raise ValueError("range mask contains no finite positive supervision")
    if not bool(valid.any().item()):
        # Zero, but still attached to the prediction so the graph and the
        # downstream finite-checks behave exactly as on a normal step.
        return (prediction_range_m * 0.0).sum()
    residual = (
        prediction_range_m[valid].log() - target_range_m[valid].log()
    ).abs()
    quadratic = 0.5 * residual.square() / delta
    linear = residual - 0.5 * delta
    per_pixel = torch.where(residual <= delta, quadratic, linear)
    weights = confidence[valid]
    return (per_pixel * weights).sum() / weights.sum()
