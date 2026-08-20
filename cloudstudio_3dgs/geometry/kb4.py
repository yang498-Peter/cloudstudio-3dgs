from __future__ import annotations

from typing import Mapping

import numpy as np


def project_kb4(
    points_camera: np.ndarray,
    intrinsic: Mapping[str, float],
    distortion: Mapping[str, float],
    *,
    min_range_m: float = 0.0,
    max_range_m: float = float("inf"),
    max_theta_rad: float = np.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape [N, 3]")
    x, y, z = points.T
    radial = np.hypot(x, y)
    theta = np.arctan2(radial, z)
    theta2 = theta * theta
    theta_distorted = theta * (
        1.0
        + float(distortion["k1"]) * theta2
        + float(distortion["k2"]) * theta2**2
        + float(distortion["k3"]) * theta2**3
        + float(distortion["k4"]) * theta2**4
    )
    scale = np.divide(
        theta_distorted,
        radial,
        out=np.zeros_like(theta_distorted),
        where=radial > 1e-12,
    )
    uv = np.column_stack(
        [
            float(intrinsic["fl_x"]) * x * scale + float(intrinsic["cx"]),
            float(intrinsic["fl_y"]) * y * scale + float(intrinsic["cy"]),
        ]
    )
    ranges = np.linalg.norm(points, axis=1)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= min_range_m)
        & (ranges <= max_range_m)
        & (theta <= max_theta_rad)
    )
    return uv, ranges, valid


def unproject_kb4(
    pixels: np.ndarray,
    intrinsic: Mapping[str, float],
    distortion: Mapping[str, float],
    *,
    iterations: int = 12,
) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape [N, 2]")
    xd = (pixels[:, 0] - float(intrinsic["cx"])) / float(intrinsic["fl_x"])
    yd = (pixels[:, 1] - float(intrinsic["cy"])) / float(intrinsic["fl_y"])
    theta_distorted = np.hypot(xd, yd)
    theta = theta_distorted.copy()
    k1, k2, k3, k4 = (float(distortion[key]) for key in ("k1", "k2", "k3", "k4"))
    for _ in range(iterations):
        theta2 = theta * theta
        value = theta * (1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
        derivative = 1 + 3 * k1 * theta2 + 5 * k2 * theta2**2 + 7 * k3 * theta2**3 + 9 * k4 * theta2**4
        theta -= np.divide(
            value - theta_distorted,
            derivative,
            out=np.zeros_like(theta),
            where=np.abs(derivative) > 1e-12,
        )
    azimuth_scale = np.divide(
        np.sin(theta),
        theta_distorted,
        out=np.ones_like(theta),
        where=theta_distorted > 1e-12,
    )
    rays = np.column_stack([xd * azimuth_scale, yd * azimuth_scale, np.cos(theta)])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)
