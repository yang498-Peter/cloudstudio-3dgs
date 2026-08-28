"""Bounded gradient, update, and point-to-plane drift telemetry."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def shortest_axis_normals(scales_m: np.ndarray, quaternions_wxyz: np.ndarray) -> np.ndarray:
    scales = np.asarray(scales_m, dtype=np.float64)
    quats = np.asarray(quaternions_wxyz, dtype=np.float64)
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales_m must have shape [N, 3]")
    if quats.shape != (len(scales), 4):
        raise ValueError("quaternions_wxyz must have shape [N, 4]")
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norm <= 0.0) or not np.all(np.isfinite(norm)):
        raise ValueError("quaternions must be finite and nonzero")
    w, x, y, z = (quats / norm).T
    rotation = np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=1),
            np.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), axis=1),
            np.stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), axis=1),
        ),
        axis=2,
    )
    shortest = np.argmin(scales, axis=1)
    normals = rotation[np.arange(len(rotation)), :, shortest]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    return np.ascontiguousarray(normals, dtype=np.float32)


def point_to_plane_drift_summary(
    current_means: np.ndarray,
    initial_means: np.ndarray,
    initial_normals: np.ndarray,
) -> dict[str, Any]:
    current = np.asarray(current_means, dtype=np.float64)
    initial = np.asarray(initial_means, dtype=np.float64)
    normals = np.asarray(initial_normals, dtype=np.float64)
    if current.shape != initial.shape or current.shape != normals.shape:
        raise ValueError("means and normals must share shape [N, 3]")
    absolute = np.abs(np.sum((current - initial) * normals, axis=1))
    return {
        "count": int(len(absolute)),
        "p50_m": float(np.percentile(absolute, 50)),
        "p95_m": float(np.percentile(absolute, 95)),
        "p99_m": float(np.percentile(absolute, 99)),
        "max_m": float(np.max(absolute, initial=0.0)),
        "over_5cm_count": int(np.count_nonzero(absolute > 0.05)),
        "over_10cm_count": int(np.count_nonzero(absolute > 0.10)),
    }


def gradient_norms(params: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameter in params.items():
        gradient = parameter.grad
        if gradient is None:
            result[name] = None
            continue
        detached = gradient.detach()
        result[name] = {
            "l2": float(detached.double().norm().cpu()),
            "max_abs": 0.0
            if detached.numel() == 0
            else float(detached.abs().max().cpu()),
            "finite": bool(detached.isfinite().all().cpu()),
        }
    return result


def parameter_update_norms(
    before: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, previous in before.items():
        delta = params[name].detach() - previous
        result[name] = {
            "l2": float(delta.double().norm().cpu()),
            "max_abs": 0.0 if delta.numel() == 0 else float(delta.abs().max().cpu()),
            "changed_count": int(np.count_nonzero(delta.detach().cpu().numpy())),
        }
    return result


def component_gradient_audit(
    params: Mapping[str, Any], components: Mapping[str, Any | None]
) -> dict[str, Any]:
    """Measure per-loss geometry gradients and their pairwise cosine."""

    names = tuple(name for name in ("means", "scales", "quats", "opacities") if name in params)
    parameters = [params[name] for name in names]
    gradients: dict[str, dict[str, Any | None]] = {}
    raw: dict[str, tuple[Any | None, ...] | None] = {}
    for component, value in components.items():
        if value is None or not getattr(value, "requires_grad", False):
            gradients[component] = {name: None for name in names}
            raw[component] = None
            continue
        values = __import__("torch").autograd.grad(
            value,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        raw[component] = values
        gradients[component] = {}
        for name, gradient in zip(names, values):
            gradients[component][name] = (
                None
                if gradient is None
                else {
                    "l2": float(gradient.detach().double().norm().cpu()),
                    "max_abs": 0.0
                    if gradient.numel() == 0
                    else float(gradient.detach().abs().max().cpu()),
                }
            )

    angles: dict[str, dict[str, float | None]] = {}
    component_names = list(components)
    for left_index, left in enumerate(component_names):
        for right in component_names[left_index + 1 :]:
            pair = f"{left}__{right}"
            angles[pair] = {}
            left_values = raw[left]
            right_values = raw[right]
            for index, name in enumerate(names):
                if left_values is None or right_values is None:
                    angles[pair][name] = None
                    continue
                left_gradient = left_values[index]
                right_gradient = right_values[index]
                if left_gradient is None or right_gradient is None:
                    angles[pair][name] = None
                    continue
                left_norm = left_gradient.detach().double().norm()
                right_norm = right_gradient.detach().double().norm()
                if float(left_norm.cpu()) == 0.0 or float(right_norm.cpu()) == 0.0:
                    angles[pair][name] = None
                    continue
                cosine = (
                    (left_gradient.detach().double() * right_gradient.detach().double()).sum()
                    / (left_norm * right_norm)
                )
                angles[pair][name] = float(cosine.clamp(-1.0, 1.0).cpu())
    return {"gradient_norms": gradients, "pairwise_cosine": angles}
