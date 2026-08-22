"""CUDA-independent configuration for error-weighted MCMC sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ErrorScoreConfig:
    """Configuration for error-weighted MCMC sampling."""

    enabled: bool = False
    ema_decay: float = 0.9
    # Experimental tempering exponent; it is a CloudStudio probe parameter,
    # not a value reproduced from the Improved-GS paper.
    score_power: float = 0.4
    # Floor keeps never-seen / zero-error Gaussians samplable (no zero weights).
    min_score_floor: float = 1e-3

    def validate(self) -> None:
        if not 0.0 <= float(self.ema_decay) < 1.0:
            raise ValueError("ema_decay must be within [0, 1)")
        if float(self.score_power) < 0.0:
            raise ValueError("score_power must be non-negative")
        if not float(self.min_score_floor) > 0.0:
            raise ValueError("min_score_floor must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": self.enabled,
            "ema_decay": self.ema_decay,
            "score_power": self.score_power,
            "min_score_floor": self.min_score_floor,
        }
