"""Rig-aware train-time pose refinement with fail-closed publication gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from cloudstudio_3dgs.geometry.rig import distribution


@dataclass(frozen=True)
class RigPoseRefinementConfig:
    """Configuration for one shared six-dimensional correction per Rig Frame."""

    enabled: bool = False
    learning_rate: float = 1e-4
    translation_prior_weight: float = 1e-3
    rotation_prior_weight: float = 1e-3
    maximum_translation_m: float = 0.25
    maximum_rotation_deg: float = 2.0
    minimum_loss_improvement_fraction: float = 0.01
    evaluation_rig_frames: int = 32

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        positive = {
            "learning_rate": self.learning_rate,
            "maximum_translation_m": self.maximum_translation_m,
            "maximum_rotation_deg": self.maximum_rotation_deg,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        non_negative = {
            "translation_prior_weight": self.translation_prior_weight,
            "rotation_prior_weight": self.rotation_prior_weight,
            "minimum_loss_improvement_fraction": self.minimum_loss_improvement_fraction,
        }
        for name, value in non_negative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_loss_improvement_fraction >= 1.0:
            raise ValueError("minimum_loss_improvement_fraction must be smaller than one")
        if self.evaluation_rig_frames <= 0:
            raise ValueError("evaluation_rig_frames must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "parameterization": "world_left_multiply_translation_axis_angle_v1",
            "learning_rate": self.learning_rate,
            "translation_prior_weight": self.translation_prior_weight,
            "rotation_prior_weight": self.rotation_prior_weight,
            "maximum_translation_m": self.maximum_translation_m,
            "maximum_rotation_deg": self.maximum_rotation_deg,
            "minimum_loss_improvement_fraction": self.minimum_loss_improvement_fraction,
            "evaluation_rig_frames": self.evaluation_rig_frames,
            "validation_pose_policy": "never_optimized",
            "publication_policy": "original_on_no_improvement_or_bounds_failure",
        }


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        if not value:
            raise ValueError("Rig Frame IDs must be non-empty")
        if value not in seen:
            result.append(value)
            seen.add(value)
    if not result:
        raise ValueError("pose refinement requires at least one Rig Frame")
    return tuple(result)


class RigPoseRefiner:
    """Own one differentiable correction shared by both images of each Rig Frame."""

    def __init__(
        self,
        rig_frame_ids: Iterable[str],
        *,
        config: RigPoseRefinementConfig,
        device: str,
        rig_frame_centers: dict[str, Any] | None = None,
    ) -> None:
        import torch

        config.validate()
        if not config.enabled:
            raise ValueError("RigPoseRefiner cannot be constructed when refinement is disabled")
        self.torch = torch
        self.config = config
        self.rig_frame_ids = _ordered_unique(rig_frame_ids)
        self._indices = {value: index for index, value in enumerate(self.rig_frame_ids)}
        centers = (
            {rig_frame_id: np.zeros(3, dtype=np.float32) for rig_frame_id in self.rig_frame_ids}
            if rig_frame_centers is None
            else rig_frame_centers
        )
        if set(centers) != set(self.rig_frame_ids):
            raise ValueError("Rig Frame centers must exactly cover the refinement IDs")
        center_values = np.stack(
            [np.asarray(centers[rig_frame_id], dtype=np.float32) for rig_frame_id in self.rig_frame_ids]
        )
        if center_values.shape != (len(self.rig_frame_ids), 3) or not np.all(
            np.isfinite(center_values)
        ):
            raise ValueError("Rig Frame centers must be finite three-vectors")
        self.centers = torch.as_tensor(center_values, dtype=torch.float32, device=device)
        self.deltas = torch.nn.Parameter(
            torch.zeros((len(self.rig_frame_ids), 6), dtype=torch.float32, device=device)
        )

    def make_optimizer(self) -> Any:
        return self.torch.optim.Adam([self.deltas], lr=self.config.learning_rate)

    def delta_to_matrix(self, delta: Any, pivot: Any | None = None) -> Any:
        """Convert [tx, ty, tz, rx, ry, rz] to a differentiable rigid matrix."""
        torch = self.torch
        if delta.shape[-1] != 6:
            raise ValueError("Rig pose deltas must have six components")
        translation = delta[..., :3]
        omega = delta[..., 3:]
        wx, wy, wz = omega.unbind(dim=-1)
        zero = torch.zeros_like(wx)
        skew = torch.stack(
            (
                zero,
                -wz,
                wy,
                wz,
                zero,
                -wx,
                -wy,
                wx,
                zero,
            ),
            dim=-1,
        ).reshape(delta.shape[:-1] + (3, 3))
        theta = torch.linalg.vector_norm(omega, dim=-1)
        a = torch.sinc(theta / torch.pi)
        b = 0.5 * torch.sinc(theta / (2.0 * torch.pi)) ** 2
        identity = torch.eye(3, dtype=delta.dtype, device=delta.device).expand(
            delta.shape[:-1] + (3, 3)
        )
        rotation = identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)
        translation_column = translation
        if pivot is not None:
            pivot_tensor = torch.as_tensor(pivot, dtype=delta.dtype, device=delta.device)
            if pivot_tensor.shape != delta.shape[:-1] + (3,):
                raise ValueError("Rig pose pivot shape does not match its delta")
            translation_column = (
                translation + pivot_tensor - (rotation @ pivot_tensor[..., None])[..., 0]
            )
        upper = torch.cat((rotation, translation_column[..., None]), dim=-1)
        bottom = torch.zeros(delta.shape[:-1] + (1, 4), dtype=delta.dtype, device=delta.device)
        bottom[..., 0, 3] = 1.0
        return torch.cat((upper, bottom), dim=-2)

    def apply(self, rig_frame_id: str, c2w: Any) -> Any:
        index = self._indices.get(str(rig_frame_id))
        if index is None:
            raise ValueError(f"unknown Rig Frame for pose refinement: {rig_frame_id}")
        matrix = self.torch.as_tensor(c2w, dtype=self.deltas.dtype, device=self.deltas.device)
        if matrix.shape != (4, 4):
            raise ValueError("camera c2w must be a 4x4 matrix")
        return self.delta_to_matrix(self.deltas[index], self.centers[index]) @ matrix

    def prior_loss(self) -> Any:
        translation = self.deltas[:, :3]
        rotation = self.deltas[:, 3:]
        return (
            self.config.translation_prior_weight * translation.square().sum(dim=-1).mean()
            + self.config.rotation_prior_weight * rotation.square().sum(dim=-1).mean()
        )

    def snapshot(self) -> np.ndarray:
        return self.deltas.detach().cpu().numpy().astype(np.float64, copy=True)

    def zero_(self) -> None:
        with self.torch.no_grad():
            self.deltas.zero_()


def build_pose_refinement_report(
    rig_frame_ids: Iterable[str],
    deltas: np.ndarray,
    *,
    loss_before: float,
    loss_after: float,
    config: RigPoseRefinementConfig,
) -> dict[str, Any]:
    """Select refined poses only when they improve fit and remain within policy."""
    config.validate()
    ids = _ordered_unique(rig_frame_ids)
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (len(ids), 6) or not np.all(np.isfinite(values)):
        raise ValueError("pose refinement deltas must be a finite Rig-count by six array")
    before = float(loss_before)
    after = float(loss_after)
    if not np.isfinite(before) or not np.isfinite(after) or before <= 0.0 or after < 0.0:
        raise ValueError("pose comparison losses must be finite with before > 0 and after >= 0")

    translations = np.linalg.norm(values[:, :3], axis=1)
    rotations_deg = np.degrees(np.linalg.norm(values[:, 3:], axis=1))
    maximum_translation = float(np.max(translations))
    maximum_rotation = float(np.max(rotations_deg))
    improvement = float((before - after) / before)
    bounds_pass = (
        maximum_translation <= config.maximum_translation_m
        and maximum_rotation <= config.maximum_rotation_deg
    )
    improvement_pass = (
        improvement > 0.0
        and improvement >= config.minimum_loss_improvement_fraction
    )
    accepted = bool(bounds_pass and improvement_pass)
    return {
        "schema_version": 1,
        "algorithm_version": "rig_pose_refinement_v1",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "candidate_accepted": accepted,
        "published_pose_set": "refined" if accepted else "original",
        "rig_frame_count": len(ids),
        "rig_frame_ids": list(ids),
        "corrections": [
            {
                "rig_frame_id": rig_frame_id,
                "translation_m": values[index, :3].tolist(),
                "rotation_axis_angle_rad": values[index, 3:].tolist(),
            }
            for index, rig_frame_id in enumerate(ids)
        ],
        "correction_translation_m": distribution(translations.tolist()),
        "correction_rotation_deg": distribution(rotations_deg.tolist()),
        "comparison": {
            "loss_before": before,
            "loss_after": after,
            "improvement_fraction": improvement,
        },
        "gates": {
            "correction_bounds": {
                "status": "PASS" if bounds_pass else "FAIL",
                "maximum_translation_m": maximum_translation,
                "translation_limit_m": config.maximum_translation_m,
                "maximum_rotation_deg": maximum_rotation,
                "rotation_limit_deg": config.maximum_rotation_deg,
            },
            "loss_improvement": {
                "status": "PASS" if improvement_pass else "FAIL",
                "actual_fraction": improvement,
                "minimum_fraction": config.minimum_loss_improvement_fraction,
            },
            "rig_baseline_invariant": {
                "status": "PASS",
                "maximum_translation_drift_m": 0.0,
                "maximum_rotation_drift_deg": 0.0,
                "reason": "one world-left-multiplied correction is shared by both cameras",
            },
        },
        "config": config.to_dict(),
    }


def disabled_pose_refinement_report(config: RigPoseRefinementConfig) -> dict[str, Any]:
    config.validate()
    return {
        "schema_version": 1,
        "algorithm_version": "rig_pose_refinement_v1",
        "status": "DISABLED",
        "candidate_accepted": False,
        "published_pose_set": "original",
        "gates": {
            "correction_bounds": {"status": "NOT_RUN"},
            "loss_improvement": {"status": "NOT_RUN"},
            "rig_baseline_invariant": {"status": "NOT_RUN"},
        },
        "config": config.to_dict(),
    }
