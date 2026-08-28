"""Constrained shared single-focal KB4 refinement for independent-camera AT."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares


def _selected_image_ids(reconstruction: Any, image_ids: set[int] | None) -> list[int]:
    selected = sorted(
        reconstruction.reg_image_ids() if image_ids is None else image_ids
    )
    if not selected:
        raise ValueError("single-focal KB4 refinement has no selected images")
    unknown = set(selected) - {int(value) for value in reconstruction.reg_image_ids()}
    if unknown:
        raise ValueError(f"single-focal KB4 refinement has unknown images: {sorted(unknown)[:4]}")
    return selected


def enforce_single_focal_kb4(
    reconstruction: Any,
    *,
    image_ids: set[int] | None = None,
    focal_source: str = "fx",
) -> dict[int, dict[str, Any]]:
    """Set every selected physical OPENCV_FISHEYE camera to ``fx == fy``."""
    if focal_source not in {"fx", "mean"}:
        raise ValueError("focal_source must be 'fx' or 'mean'")
    selected = _selected_image_ids(reconstruction, image_ids)
    camera_ids = sorted({int(reconstruction.image(value).camera_id) for value in selected})
    output: dict[int, dict[str, Any]] = {}
    for camera_id in camera_ids:
        camera = reconstruction.camera(camera_id)
        if str(camera.model_name) != "OPENCV_FISHEYE":
            raise ValueError(f"camera {camera_id} is not OPENCV_FISHEYE")
        before = np.asarray(camera.params, dtype=np.float64).copy()
        if before.shape != (8,) or not np.all(np.isfinite(before)):
            raise ValueError(f"camera {camera_id} has invalid KB4 parameters")
        focal = float(before[0] if focal_source == "fx" else np.mean(before[:2]))
        if focal <= 0.0:
            raise ValueError(f"camera {camera_id} has non-positive focal length")
        after = before.copy()
        after[0] = focal
        after[1] = focal
        camera.params = after
        output[camera_id] = {
            "before": before.tolist(),
            "after": after.tolist(),
            "focal_source": focal_source,
        }
    return output


def _camera_observations(
    reconstruction: Any,
    camera_id: int,
    selected_image_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    camera_points: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    for image_id in selected_image_ids:
        image = reconstruction.image(image_id)
        if int(image.camera_id) != camera_id:
            continue
        matrix = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        image_points: list[np.ndarray] = []
        image_pixels: list[np.ndarray] = []
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            point3d = reconstruction.point3D(int(point2d.point3D_id))
            image_points.append(np.asarray(point3d.xyz, dtype=np.float64))
            image_pixels.append(np.asarray(point2d.xy, dtype=np.float64))
        if not image_points:
            continue
        world = np.asarray(image_points, dtype=np.float64)
        camera_points.append(world @ matrix[:, :3].T + matrix[:, 3])
        pixels.append(np.asarray(image_pixels, dtype=np.float64))
    if not camera_points:
        raise ValueError(f"camera {camera_id} has no registered observations")
    points = np.concatenate(camera_points, axis=0)
    observed = np.concatenate(pixels, axis=0)
    radial = np.hypot(points[:, 0], points[:, 1])
    theta = np.arctan2(radial, points[:, 2])
    usable = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(observed), axis=1)
        & np.isfinite(theta)
        & (theta < np.pi)
    )
    points = points[usable]
    observed = observed[usable]
    if len(points) < 32:
        raise ValueError(f"camera {camera_id} has too few usable observations")
    return points, observed


def _project_and_jacobian(
    parameters: np.ndarray,
    camera_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    focal, cx, cy, k1, k2, k3, k4 = parameters
    x = camera_points[:, 0]
    y = camera_points[:, 1]
    z = camera_points[:, 2]
    radial = np.hypot(x, y)
    theta = np.arctan2(radial, z)
    theta2 = theta * theta
    powers = np.column_stack((theta**3, theta**5, theta**7, theta**9))
    theta_distorted = theta + powers @ np.asarray([k1, k2, k3, k4])
    direction_x = np.divide(x, radial, out=np.zeros_like(x), where=radial > 1e-12)
    direction_y = np.divide(y, radial, out=np.zeros_like(y), where=radial > 1e-12)
    distorted_x = direction_x * theta_distorted
    distorted_y = direction_y * theta_distorted
    projected = np.column_stack(
        (focal * distorted_x + cx, focal * distorted_y + cy)
    )

    jacobian = np.zeros((len(camera_points) * 2, 7), dtype=np.float64)
    jacobian[0::2, 0] = distorted_x
    jacobian[1::2, 0] = distorted_y
    jacobian[0::2, 1] = 1.0
    jacobian[1::2, 2] = 1.0
    jacobian[0::2, 3:] = focal * direction_x[:, None] * powers
    jacobian[1::2, 3:] = focal * direction_y[:, None] * powers
    return projected, jacobian


def refine_shared_single_focal_kb4_intrinsics(
    reconstruction: Any,
    *,
    image_ids: set[int] | None = None,
    max_nfev: int = 50,
    huber_scale_px: float = 2.0,
) -> dict[str, Any]:
    """Refine one ``f,cx,cy,k1..k4`` vector per shared physical camera.

    Poses and points are held fixed in this block. Calling it between pose/point
    BA blocks gives an explicitly constrained block-coordinate joint solve.
    """
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    if not np.isfinite(huber_scale_px) or huber_scale_px <= 0.0:
        raise ValueError("huber_scale_px must be finite and positive")
    selected = _selected_image_ids(reconstruction, image_ids)
    camera_ids = sorted({int(reconstruction.image(value).camera_id) for value in selected})
    summaries: dict[str, Any] = {}
    for camera_id in camera_ids:
        camera = reconstruction.camera(camera_id)
        if str(camera.model_name) != "OPENCV_FISHEYE":
            raise ValueError(f"camera {camera_id} is not OPENCV_FISHEYE")
        raw = np.asarray(camera.params, dtype=np.float64)
        if raw.shape != (8,) or not np.all(np.isfinite(raw)):
            raise ValueError(f"camera {camera_id} has invalid KB4 parameters")
        if not np.isclose(raw[0], raw[1], atol=1e-12):
            raise ValueError(f"camera {camera_id} does not satisfy fx == fy")
        points, observed = _camera_observations(
            reconstruction, camera_id, selected
        )
        initial = np.asarray(
            [raw[0], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]],
            dtype=np.float64,
        )

        def residuals(parameters: np.ndarray) -> np.ndarray:
            projected, _ = _project_and_jacobian(parameters, points)
            return (projected - observed).reshape(-1)

        def jacobian(parameters: np.ndarray) -> np.ndarray:
            _, value = _project_and_jacobian(parameters, points)
            return value

        initial_residuals = residuals(initial)
        width = float(camera.width)
        height = float(camera.height)
        lower = np.asarray(
            [0.5 * initial[0], -0.25 * width, -0.25 * height, -1.0, -1.0, -1.0, -1.0]
        )
        upper = np.asarray(
            [1.5 * initial[0], 1.25 * width, 1.25 * height, 1.0, 1.0, 1.0, 1.0]
        )
        result = least_squares(
            residuals,
            initial,
            jac=jacobian,
            bounds=(lower, upper),
            loss="huber",
            f_scale=float(huber_scale_px),
            x_scale="jac",
            max_nfev=int(max_nfev),
        )
        final_residuals = residuals(result.x)
        initial_rmse = float(np.sqrt(np.mean(initial_residuals**2)))
        final_rmse = float(np.sqrt(np.mean(final_residuals**2)))
        if (
            not result.success
            or not np.all(np.isfinite(result.x))
            or final_rmse > initial_rmse + 1e-9
        ):
            raise RuntimeError(
                f"camera {camera_id} single-focal KB4 solve failed: {result.message}; "
                f"RMSE {initial_rmse:.6f} -> {final_rmse:.6f} px"
            )
        focal, cx, cy, k1, k2, k3, k4 = (float(value) for value in result.x)
        camera.params = np.asarray(
            [focal, focal, cx, cy, k1, k2, k3, k4], dtype=np.float64
        )
        parameter_scale = np.asarray([width, width, height, 1.0, 1.0, 1.0, 1.0])
        summaries[str(camera_id)] = {
            "success": True,
            "termination": str(result.message),
            "observations": int(len(observed)),
            "nfev": int(result.nfev),
            "before": initial.tolist(),
            "after": result.x.tolist(),
            "delta_after_minus_before": (result.x - initial).tolist(),
            "max_scaled_parameter_step": float(
                np.max(np.abs(result.x - initial) / parameter_scale)
            ),
            "component_rmse_px_before": initial_rmse,
            "component_rmse_px_after": final_rmse,
        }
    return {
        "parameter_order": ["focal_px", "cx", "cy", "k1", "k2", "k3", "k4"],
        "huber_scale_px": float(huber_scale_px),
        "cameras": summaries,
        "max_scaled_parameter_step": max(
            float(value["max_scaled_parameter_step"]) for value in summaries.values()
        ),
    }
