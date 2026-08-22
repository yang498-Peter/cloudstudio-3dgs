"""Appearance representation and deterministic spherical-harmonic schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppearanceConfig:
    mode: str = "sh"
    maximum_degree: int = 3
    degree_interval: int = 1_000
    rest_lr_scale: float = 0.05

    def validate(self) -> None:
        if self.mode not in {"rgb", "sh"}:
            raise ValueError("appearance mode must be 'rgb' or 'sh'")
        if not 0 <= self.maximum_degree <= 4:
            raise ValueError("maximum SH degree must be in [0, 4]")
        if self.degree_interval <= 0:
            raise ValueError("SH degree interval must be positive")
        if self.rest_lr_scale <= 0.0:
            raise ValueError("SH rest learning-rate scale must be positive")
        if self.mode == "rgb" and self.maximum_degree != 0:
            raise ValueError("RGB appearance must use maximum_degree=0")

    def degree_for_step(self, step: int) -> int:
        self.validate()
        if step < 0:
            raise ValueError("SH schedule step must be non-negative")
        if self.mode == "rgb":
            return 0
        return min(self.maximum_degree, int(step) // self.degree_interval)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mode": self.mode,
            "maximum_degree": self.maximum_degree,
            "degree_interval": self.degree_interval,
            "rest_lr_scale": self.rest_lr_scale,
        }

    def state_for_step(self, step: int) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "maximum_degree": self.maximum_degree,
            "active_degree": self.degree_for_step(step),
        }


def verify_appearance_resume_state(
    config: AppearanceConfig,
    *,
    completed_steps: int,
    restored: Any,
) -> dict[str, Any]:
    expected = config.state_for_step(completed_steps)
    if restored != expected:
        raise ValueError(
            "checkpoint appearance stage does not match the deterministic SH schedule"
        )
    return expected
