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


def masked_rgb_ssim_loss(prediction: Any, target: Any, mask: Any) -> Any:
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
