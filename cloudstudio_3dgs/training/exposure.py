"""Per-frame scalar exposure compensation for auto-exposure fisheye captures.

The S1 rig runs both fisheye cameras on independent auto exposure, so the same
static surface is observed at different brightness from frame to frame. Without
compensation the trainer averages the contradiction into washed-out texture and
absorbs shading differences into fake geometry. This module owns one
differentiable log-gain per TRAINING image, applied to the rendered RGB before
the photometric losses only; validation always renders at gain 1.0 so metrics
stay honest, and a strong zero-pull prior keeps the gains from re-encoding real
scene appearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ExposureCompensationConfig:
    enabled: bool = False
    learning_rate: float = 5e-3
    regularization_weight: float = 1e-2
    max_abs_log_gain: float = 0.6931471805599453  # ln(2): gain clamped to [0.5, 2]
    zero_mean_projection: bool = False

    def validate(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("exposure learning_rate must be positive")
        if self.regularization_weight < 0.0:
            raise ValueError("exposure regularization_weight must be non-negative")
        if self.max_abs_log_gain <= 0.0:
            raise ValueError("exposure max_abs_log_gain must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "learning_rate": self.learning_rate,
            "regularization_weight": self.regularization_weight,
            "max_abs_log_gain": self.max_abs_log_gain,
            "zero_mean_projection": self.zero_mean_projection,
        }


class ExposureCompensator:
    """Own one clamped scalar log-gain per training image."""

    def __init__(
        self,
        image_ids: Iterable[str],
        *,
        config: ExposureCompensationConfig,
        device: str,
        group_by_image: dict[str, str] | None = None,
    ) -> None:
        import torch

        config.validate()
        if not config.enabled:
            raise ValueError(
                "ExposureCompensator cannot be constructed when compensation is disabled"
            )
        ordered = sorted(set(str(image_id) for image_id in image_ids))
        if not ordered:
            raise ValueError("exposure compensation requires at least one training image")
        self.config = config
        self.device = device
        self.index = {image_id: position for position, image_id in enumerate(ordered)}
        self.log_gains = torch.nn.Parameter(
            torch.zeros(len(ordered), dtype=torch.float32, device=device)
        )
        # Anchor groups: with independent auto-exposure per physical camera, a
        # single global anchor lets the two cameras drift in opposite
        # directions and still sum to zero, so each camera group is projected
        # to zero mean on its own.
        self.group_members: dict[str, list[int]] = {}
        if group_by_image is not None:
            missing = [image_id for image_id in ordered if image_id not in group_by_image]
            if missing:
                raise ValueError(
                    f"exposure groups missing for {len(missing)} images, e.g. {missing[0]!r}"
                )
            for image_id in ordered:
                key = str(group_by_image[image_id])
                self.group_members.setdefault(key, []).append(self.index[image_id])
        else:
            self.group_members["all"] = list(range(len(ordered)))

    def make_optimizer(self) -> Any:
        import torch

        return torch.optim.Adam(
            [{"params": [self.log_gains], "lr": self.config.learning_rate, "name": "exposure"}],
            eps=1e-15,
        )

    def gain(self, image_id: str) -> Any:
        import torch

        position = self.index.get(str(image_id))
        if position is None:
            raise KeyError(f"exposure compensation has no gain for image {image_id!r}")
        bound = self.config.max_abs_log_gain
        return torch.exp(torch.clamp(self.log_gains[position], -bound, bound))

    def prior_loss(self) -> Any:
        return self.config.regularization_weight * (self.log_gains**2).mean()

    def project_zero_mean(self) -> None:
        """Remove the dataset-mean log gain after an optimizer step.

        Per-image gains and global model brightness are jointly unobservable
        from the photometric loss alone; over a long run the gains drift bright
        while the model itself darkens, and validation (always gain 1.0) pays
        the bill. Projecting the gains onto the zero-mean subspace pins the
        global-brightness degree of freedom inside the model where validation
        can see it, while per-image differences remain free.
        """
        import torch

        if not self.config.zero_mean_projection:
            return
        with torch.no_grad():
            for members in self.group_members.values():
                subset = self.log_gains[members]
                self.log_gains[members] = subset - subset.mean()

    def report(self) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            bound = self.config.max_abs_log_gain
            clamped = torch.clamp(self.log_gains.detach(), -bound, bound)
            gains = torch.exp(clamped)
            absolute = torch.abs(clamped)
            quantiles = torch.quantile(
                absolute, torch.tensor([0.5, 0.95], device=absolute.device)
            )
        return {
            "image_count": int(self.log_gains.shape[0]),
            "mean_log_gain": float(self.log_gains.detach().mean()),
            "mean_log_gain_by_group": {
                key: float(self.log_gains.detach()[members].mean())
                for key, members in sorted(self.group_members.items())
            },
            "abs_log_gain_p50": float(quantiles[0]),
            "abs_log_gain_p95": float(quantiles[1]),
            "abs_log_gain_max": float(absolute.max()),
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "saturated_fraction": float(
                (absolute >= bound - 1e-6).float().mean()
            ),
        }
