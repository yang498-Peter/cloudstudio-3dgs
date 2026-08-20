"""Mask-aware image metrics with explicit empty-mask failures."""

from __future__ import annotations

import math

import numpy as np


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
