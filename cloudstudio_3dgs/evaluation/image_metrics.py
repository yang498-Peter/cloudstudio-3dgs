"""Mask-aware image metrics with explicit empty-mask failures."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage


def masked_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != expected.shape:
        raise ValueError("prediction and target shapes differ")
    if predicted.ndim not in (2, 3):
        raise ValueError("images must have shape [H, W] or [H, W, C]")
    if valid.shape != predicted.shape[:2]:
        raise ValueError("mask shape does not match image spatial shape")
    if not np.any(valid):
        raise ValueError("masked metric has no valid pixels")
    difference = predicted[valid] - expected[valid]
    return float(np.mean(difference * difference))


def masked_psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    data_range: float = 1.0,
) -> float:
    if data_range <= 0.0:
        raise ValueError("data_range must be positive")
    mse = masked_mse(prediction, target, mask)
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def masked_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    data_range: float = 1.0,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Mask-normalized Gaussian SSIM; invalid pixels cannot affect windows."""
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != expected.shape:
        raise ValueError("prediction and target shapes differ")
    if predicted.ndim == 2:
        predicted = predicted[:, :, None]
        expected = expected[:, :, None]
    if predicted.ndim != 3 or valid.shape != predicted.shape[:2]:
        raise ValueError("mask shape does not match image spatial shape")
    if not np.any(valid):
        raise ValueError("masked metric has no valid pixels")
    if data_range <= 0.0 or sigma <= 0.0:
        raise ValueError("data_range and sigma must be positive")

    weights = ndimage.gaussian_filter(valid.astype(np.float64), sigma=sigma, mode="reflect")
    safe_weights = np.maximum(weights, np.finfo(np.float64).eps)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    values: list[np.ndarray] = []
    for channel in range(predicted.shape[2]):
        x = predicted[:, :, channel]
        y = expected[:, :, channel]
        mu_x = ndimage.gaussian_filter(x * valid, sigma=sigma, mode="reflect") / safe_weights
        mu_y = ndimage.gaussian_filter(y * valid, sigma=sigma, mode="reflect") / safe_weights
        second_x = ndimage.gaussian_filter(x * x * valid, sigma=sigma, mode="reflect") / safe_weights
        second_y = ndimage.gaussian_filter(y * y * valid, sigma=sigma, mode="reflect") / safe_weights
        cross = ndimage.gaussian_filter(x * y * valid, sigma=sigma, mode="reflect") / safe_weights
        variance_x = np.maximum(0.0, second_x - mu_x * mu_x)
        variance_y = np.maximum(0.0, second_y - mu_y * mu_y)
        covariance = cross - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * covariance + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (
            variance_x + variance_y + c2
        )
        values.append(np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator != 0))
    ssim_map = np.mean(np.stack(values, axis=2), axis=2)
    return float(np.mean(ssim_map[valid]))


def masked_lpips_from_distance_map(
    distance_map: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Aggregate a spatial LPIPS map without allowing invalid-border pixels."""
    distances = np.asarray(distance_map, dtype=np.float64).squeeze()
    if distances.ndim != 2 or not np.all(np.isfinite(distances)):
        raise ValueError("LPIPS distance map must be a finite 2D array")
    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("LPIPS mask must be 2D")
    if valid.shape != distances.shape:
        valid = (
            np.asarray(
                Image.fromarray(valid.astype(np.uint8) * 255).resize(
                    (distances.shape[1], distances.shape[0]), Image.Resampling.NEAREST
                )
            )
            > 0
        )
    if not np.any(valid):
        raise ValueError("masked metric has no valid pixels")
    return float(np.mean(distances[valid]))


def masked_lpips(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    net: str = "alex",
    device: str = "cpu",
    model: object | None = None,
) -> float:
    """Run optional BSD-2-Clause LPIPS with an explicit spatial mask."""
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "masked LPIPS requires the optional lpips and torch packages"
        ) from exc
    predicted = np.asarray(prediction, dtype=np.float32)
    expected = np.asarray(target, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != expected.shape or predicted.ndim != 3 or predicted.shape[2] != 3:
        raise ValueError("LPIPS images must both have shape [H, W, 3]")
    if valid.shape != predicted.shape[:2] or not np.any(valid):
        raise ValueError("LPIPS mask does not match the images or is empty")
    # Equal neutral pixels outside the valid region prevent black borders from
    # creating a perceptual difference through the network receptive field.
    predicted = predicted.copy()
    expected = expected.copy()
    predicted[~valid] = 0.5
    expected[~valid] = 0.5
    predicted_tensor = torch.from_numpy(predicted).permute(2, 0, 1).unsqueeze(0).to(device)
    expected_tensor = torch.from_numpy(expected).permute(2, 0, 1).unsqueeze(0).to(device)
    if model is None:
        model = lpips.LPIPS(net=net, spatial=True).to(device).eval()
    with torch.no_grad():
        distance = model(predicted_tensor, expected_tensor, normalize=True)
    return masked_lpips_from_distance_map(distance.detach().cpu().numpy(), valid)


def create_lpips_model(*, net: str = "alex", device: str = "cpu") -> object:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requires the optional lpips package") from exc
    return lpips.LPIPS(net=net, spatial=True).to(device).eval()


def masked_depth_metrics(
    prediction_range_m: np.ndarray,
    target_range_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    minimum_prediction_coverage: float = 1.0,
) -> dict[str, float | int]:
    predicted = np.asarray(prediction_range_m, dtype=np.float64)
    expected = np.asarray(target_range_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted.shape != expected.shape or valid.shape != expected.shape:
        raise ValueError("depth prediction, target, and mask shapes must match")
    if not 0.0 < minimum_prediction_coverage <= 1.0:
        raise ValueError("minimum depth prediction coverage must be in (0, 1]")
    target_valid = valid & np.isfinite(expected) & (expected > 0.0)
    if not np.any(target_valid):
        raise ValueError("depth metric has no valid target pixels")
    prediction_valid = target_valid & np.isfinite(predicted) & (predicted > 0.0)
    target_count = int(np.count_nonzero(target_valid))
    predicted_count = int(np.count_nonzero(prediction_valid))
    coverage = predicted_count / target_count
    if coverage < minimum_prediction_coverage:
        raise ValueError(
            "rendered depth coverage at supervised pixels is below the gate: "
            f"{coverage:.6f} < {minimum_prediction_coverage:.6f}"
        )
    error = predicted[prediction_valid] - expected[prediction_valid]
    absolute = np.abs(error)
    if confidence is None:
        weights = np.ones(len(error), dtype=np.float64)
    else:
        confidence_array = np.asarray(confidence, dtype=np.float64)
        if confidence_array.shape != expected.shape:
            raise ValueError("depth confidence shape does not match target")
        weights = confidence_array[prediction_valid]
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("depth confidence must be finite and positive at valid pixels")
    weight_sum = float(np.sum(weights))
    mae = float(np.sum(weights * absolute) / weight_sum)
    rmse = float(np.sqrt(np.sum(weights * error * error) / weight_sum))
    return {
        "valid_pixels": predicted_count,
        "target_valid_pixels": target_count,
        "missing_prediction_pixels": target_count - predicted_count,
        "prediction_coverage_fraction": float(coverage),
        "mae_m": mae,
        "rmse_m": rmse,
        "absolute_error_p95_m": float(np.percentile(absolute, 95)),
    }
