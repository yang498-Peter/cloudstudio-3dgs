"""Bounded gradient, update, and point-to-plane drift telemetry."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class AuditedLossTerm:
    """One loss term as the optimizer sees it: raw value, weight, stage scale.

    ``raw`` is the unweighted differentiable scalar (None when the term is
    absent this step). ``weight`` is the configured nominal weight and
    ``stage_multiplier`` the schedule/phase envelope applied on top of it; the
    gradient is measured on ``raw * weight * stage_multiplier``, i.e. on the
    tensor that actually reaches the parameters.
    """

    raw: Any | None
    weight: float = 1.0
    stage_multiplier: float = 1.0

    @property
    def effective_weight(self) -> float:
        return float(self.weight) * float(self.stage_multiplier)

    @property
    def present(self) -> bool:
        return self.raw is not None and bool(getattr(self.raw, "requires_grad", False))

    def weighted(self) -> Any | None:
        if not self.present:
            return None
        return self.raw * self.effective_weight


# Parameter-group order for the report: the geometry groups first (they are
# what densification and the drift audit read), then whatever else the model
# carries (colors / sh0 / shN / ...), in dictionary order.
_PRIMARY_GROUPS = ("means", "scales", "quats", "opacities")
MEANS2D_GROUP = "means2d"


def _tensor_norms(gradient: Any) -> dict[str, float]:
    detached = gradient.detach()
    return {
        "l2": float(detached.double().norm().cpu()),
        "max_abs": 0.0 if detached.numel() == 0 else float(detached.abs().max().cpu()),
    }


def component_gradient_audit(
    params: Mapping[str, Any],
    components: Mapping[str, Any | None],
    *,
    means2d: Any | None = None,
) -> dict[str, Any]:
    """Measure per-loss gradients per parameter group and their pairwise cosine.

    ``components`` maps a term name to either a differentiable scalar (legacy
    form: already weighted), ``None`` (absent this step), or an
    :class:`AuditedLossTerm`. When ``means2d`` (the projected-position tensor
    the densification criterion reads) is given, every component also reports
    its gradient norm there. A term with no autograd path to ``means2d`` - a
    direct parameter regulariser, for instance - is reported as
    ``not_applicable`` rather than as zero or as an error: zero would claim a
    measurement that was never made.

    A negative pairwise cosine proves the two objectives compete on that
    group; it does not, on its own, say which supervision is wrong.
    """

    torch = __import__("torch")
    names = tuple(name for name in _PRIMARY_GROUPS if name in params) + tuple(
        name for name in params if name not in _PRIMARY_GROUPS
    )
    parameters = [params[name] for name in names]
    probe_means2d = means2d is not None and bool(
        getattr(means2d, "requires_grad", False)
    )
    inputs = parameters + ([means2d] if probe_means2d else [])
    report_groups = names + ((MEANS2D_GROUP,) if probe_means2d else ())

    gradients: dict[str, dict[str, Any | None]] = {}
    terms: dict[str, dict[str, Any]] = {}
    raw: dict[str, tuple[Any | None, ...] | None] = {}
    for component, value in components.items():
        term = (
            value
            if isinstance(value, AuditedLossTerm)
            else AuditedLossTerm(raw=value)
        )
        weighted = term.weighted()
        record: dict[str, Any] = {
            "present": term.present,
            "raw_loss": None if term.raw is None else float(term.raw.detach().cpu()),
            "weight": float(term.weight),
            "stage_multiplier": float(term.stage_multiplier),
            "effective_weight": term.effective_weight,
            "weighted_loss": None if weighted is None else float(weighted.detach().cpu()),
            "means2d_gradient": "absent" if not probe_means2d else "not_applicable",
            "means2d_gradient_l2": None,
        }
        if weighted is None:
            gradients[component] = {name: None for name in report_groups}
            raw[component] = None
            terms[component] = record
            continue
        values = torch.autograd.grad(
            weighted,
            inputs,
            retain_graph=True,
            allow_unused=True,
        )
        raw[component] = values
        gradients[component] = {
            name: None if gradient is None else _tensor_norms(gradient)
            for name, gradient in zip(report_groups, values)
        }
        if probe_means2d:
            means2d_gradient = values[-1]
            if means2d_gradient is not None:
                record["means2d_gradient"] = "measured"
                record["means2d_gradient_l2"] = gradients[component][MEANS2D_GROUP]["l2"]
        terms[component] = record

    angles: dict[str, dict[str, float | None]] = {}
    component_names = list(components)
    for left_index, left in enumerate(component_names):
        for right in component_names[left_index + 1 :]:
            pair = f"{left}__{right}"
            angles[pair] = {}
            left_values = raw[left]
            right_values = raw[right]
            for index, name in enumerate(report_groups):
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
    return {
        "gradient_norms": gradients,
        "pairwise_cosine": angles,
        "terms": terms,
        "means2d_probed": probe_means2d,
    }
