"""Per-view colour affine for the photometric losses.

The scalar exposure gain (exposure.py) is one number per parent image, shared by its faces.
Measured on a trained house0305 tile, on its own training views, that leaves 14% of the L1 on
the table: the best per-view, per-channel affine is ``photo = a * render + b`` with ``a``
spread over 0.67-1.15 and ``b`` up to 0.19 (veiling glare lifts a photo's blacks, which no
gain can express), and the same photo wants different gains in different tiles. What the gain
cannot absorb, the gaussians do - as colour compromises between views.

``ViewColourAffine`` gives every training view (``base_image::face``) a log-gain and an offset
per channel, applied to the render after the scalar gain and only inside the photometric
losses. Validation, export and every evaluation tool keep rendering the model as it is.

The correction cannot drift into the model: after every optimizer step both tensors are
projected to zero mean over the views, so the canonical colours stay the average the photos
agree on, and both are clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ViewColourConfig:
    enabled: bool = False
    learning_rate: float = 1e-2
    prior_weight: float = 1e-2
    max_abs_log_gain: float = 0.35
    max_abs_offset: float = 0.15

    def validate(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("view colour learning_rate must be non-negative")
        if self.prior_weight < 0.0:
            raise ValueError("view colour prior_weight must be non-negative")
        if not 0.0 < self.max_abs_log_gain <= math.log(4.0):
            raise ValueError("view colour max_abs_log_gain must be within (0, ln 4]")
        if not 0.0 < self.max_abs_offset <= 0.5:
            raise ValueError("view colour max_abs_offset must be within (0, 0.5]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": self.enabled,
            "learning_rate": self.learning_rate,
            "prior_weight": self.prior_weight,
            "max_abs_log_gain": self.max_abs_log_gain,
            "max_abs_offset": self.max_abs_offset,
            "parameters": "per_view_per_channel_log_gain_and_offset",
            "gauge": "zero_mean_over_views_after_every_step",
        }


class ViewColourAffine:
    """Log-gain and offset per (view, channel); identity at initialization."""

    def __init__(self, view_ids: Iterable[str], *, config: ViewColourConfig, device: str) -> None:
        import torch

        config.validate()
        ids = list(view_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("view colour ids must be unique")
        if not ids:
            raise ValueError("view colour needs at least one view")
        self.torch = torch
        self.config = config
        self._index = {view_id: position for position, view_id in enumerate(ids)}
        self.log_gains = torch.nn.Parameter(torch.zeros(len(ids), 3, device=device))
        self.offsets = torch.nn.Parameter(torch.zeros(len(ids), 3, device=device))

    def __len__(self) -> int:
        return len(self._index)

    def _row(self, view_id: str) -> int:
        try:
            return self._index[view_id]
        except KeyError:
            raise KeyError(f"view {view_id!r} has no colour parameters") from None

    def apply(self, rendered: Any, view_id: str) -> Any:
        """``rendered`` ([H, W, 3]) under this view's affine."""
        row = self._row(view_id)
        gain = self.torch.exp(
            self.log_gains[row].clamp(-self.config.max_abs_log_gain, self.config.max_abs_log_gain)
        )
        offset = self.offsets[row].clamp(-self.config.max_abs_offset, self.config.max_abs_offset)
        return rendered * gain + offset

    def prior_loss(self) -> Any:
        scale = self.config.max_abs_offset / self.config.max_abs_log_gain
        return self.config.prior_weight * (
            self.log_gains.square().mean() + (self.offsets / scale).square().mean()
        )

    def make_optimizer(self) -> Any:
        return self.torch.optim.Adam(
            [self.log_gains, self.offsets], lr=self.config.learning_rate
        )

    def project(self) -> None:
        """Zero mean over the views, per channel, and inside the clamps."""
        with self.torch.no_grad():
            self.log_gains.sub_(self.log_gains.mean(dim=0, keepdim=True))
            self.offsets.sub_(self.offsets.mean(dim=0, keepdim=True))
            self.log_gains.clamp_(-self.config.max_abs_log_gain, self.config.max_abs_log_gain)
            self.offsets.clamp_(-self.config.max_abs_offset, self.config.max_abs_offset)

    def summary(self) -> dict[str, float]:
        with self.torch.no_grad():
            return {
                "view_colour_log_gain_std": float(self.log_gains.std().cpu()),
                "view_colour_offset_std": float(self.offsets.std().cpu()),
                "view_colour_log_gain_abs_max": float(self.log_gains.abs().max().cpu()),
                "view_colour_offset_abs_max": float(self.offsets.abs().max().cpu()),
            }
