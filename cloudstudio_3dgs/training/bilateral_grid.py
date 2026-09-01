"""Per-camera 3D bilateral RGB-affine grid for photometric nuisance removal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class BilateralGridConfig:
    enabled: bool = False
    learning_rate: float = 0.002
    grid_width: int = 16
    grid_height: int = 16
    grid_depth: int = 8
    tv_weight: float = 5.0
    warmup_fraction: float = 1.0 / 30.0
    warmup_start_multiplier: float = 0.01
    final_lr_multiplier: float = 0.01

    def validate(self) -> None:
        if self.learning_rate < 0.0 or self.tv_weight < 0.0:
            raise ValueError("bilateral grid LR and TV weight must be non-negative")
        if min(self.grid_width, self.grid_height, self.grid_depth) < 2:
            raise ValueError("bilateral grid dimensions must be at least two")
        if not 0.0 < self.warmup_fraction < 1.0:
            raise ValueError("bilateral grid warmup_fraction must be within (0, 1)")
        if not 0.0 < self.warmup_start_multiplier <= 1.0:
            raise ValueError("bilateral grid warmup multiplier must be within (0, 1]")
        if not 0.0 < self.final_lr_multiplier <= 1.0:
            raise ValueError("bilateral grid final LR multiplier must be within (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "learning_rate": self.learning_rate,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "grid_depth": self.grid_depth,
            "channels": 12,
            "tv_weight": self.tv_weight,
            "warmup_fraction": self.warmup_fraction,
            "warmup_start_multiplier": self.warmup_start_multiplier,
            "final_lr_multiplier": self.final_lr_multiplier,
        }

    def learning_rate_for_step(self, *, step: int, total_steps: int) -> float:
        if total_steps <= 0 or step < 0 or step >= total_steps:
            raise ValueError("invalid bilateral grid schedule position")
        warmup_steps = max(1, int(math.floor(total_steps * self.warmup_fraction)))
        if step < warmup_steps:
            progress = float(step) / float(max(1, warmup_steps - 1))
            multiplier = self.warmup_start_multiplier + progress * (
                1.0 - self.warmup_start_multiplier
            )
        else:
            remaining = max(1, total_steps - warmup_steps - 1)
            progress = min(max(float(step - warmup_steps) / remaining, 0.0), 1.0)
            multiplier = math.exp(math.log(self.final_lr_multiplier) * progress)
        return float(self.learning_rate * multiplier)


class BilateralGridCorrector:
    """Train one identity-initialized 3x4 affine grid per physical camera."""

    def __init__(
        self,
        image_ids: Iterable[str],
        *,
        camera_by_image: Mapping[str, str],
        config: BilateralGridConfig,
        device: str,
    ) -> None:
        import torch

        config.validate()
        if not config.enabled:
            raise ValueError("cannot construct a disabled bilateral grid")
        images = sorted(set(str(value) for value in image_ids))
        missing = [value for value in images if value not in camera_by_image]
        if not images or missing:
            raise ValueError("bilateral grid requires a complete image-camera mapping")
        cameras = sorted({str(camera_by_image[value]) for value in images})
        self.camera_index = {value: index for index, value in enumerate(cameras)}
        self.camera_by_image = {
            value: str(camera_by_image[value]) for value in images
        }
        self.config = config
        grid = torch.zeros(
            len(cameras),
            12,
            config.grid_depth,
            config.grid_height,
            config.grid_width,
            dtype=torch.float32,
            device=device,
        )
        grid[:, 0] = 1.0
        grid[:, 5] = 1.0
        grid[:, 10] = 1.0
        self.grid = torch.nn.Parameter(grid)

    def make_optimizer(self) -> Any:
        import torch

        return torch.optim.Adam(
            [{"params": [self.grid], "lr": self.config.learning_rate, "name": "bilateral_grid"}],
            betas=(0.9, 0.999),
            eps=1e-15,
        )

    def apply(
        self,
        rgb: Any,
        image_id: str,
        *,
        alpha: Any | None = None,
        background_rgb: tuple[float, float, float] | None = None,
    ) -> Any:
        import torch

        camera = self.camera_by_image.get(str(image_id))
        if camera is None:
            raise KeyError(f"bilateral grid has no image {image_id!r}")
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, dtype=rgb.dtype, device=rgb.device),
            torch.linspace(-1.0, 1.0, width, dtype=rgb.dtype, device=rgb.device),
            indexing="ij",
        )
        guide = (
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        ).clamp(0.0, 1.0)
        coordinates = torch.stack((xx, yy, 2.0 * guide - 1.0), dim=-1)[
            None, None
        ]
        sampled = torch.nn.functional.grid_sample(
            self.grid[self.camera_index[camera] : self.camera_index[camera] + 1],
            coordinates,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, :, 0].permute(1, 2, 0)
        affine = sampled.reshape(height, width, 3, 4)
        if alpha is None:
            augmented = torch.cat((rgb, torch.ones_like(rgb[..., :1])), dim=-1)
            return torch.einsum("hwij,hwj->hwi", affine, augmented)

        alpha_channel = alpha[..., None].to(dtype=rgb.dtype, device=rgb.device)
        background = torch.zeros(3, dtype=rgb.dtype, device=rgb.device)
        if background_rgb is not None:
            background = torch.as_tensor(
                background_rgb, dtype=rgb.dtype, device=rgb.device
            )
        foreground_premultiplied = rgb - (1.0 - alpha_channel) * background
        linear = torch.einsum(
            "hwij,hwj->hwi", affine[..., :3], foreground_premultiplied
        )
        bias = affine[..., 3] * alpha_channel
        return linear + bias + (1.0 - alpha_channel) * background

    def tv_loss(self) -> Any:
        grid = self.grid
        terms = (
            (grid[..., 1:] - grid[..., :-1]).abs().mean(),
            (grid[..., 1:, :] - grid[..., :-1, :]).abs().mean(),
            (grid[..., 1:, :, :] - grid[..., :-1, :, :]).abs().mean(),
        )
        return self.config.tv_weight * sum(terms)

    def report(self) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            identity = torch.zeros_like(self.grid)
            identity[:, 0] = identity[:, 5] = identity[:, 10] = 1.0
            delta = torch.abs(self.grid - identity)
        return {
            "camera_count": len(self.camera_index),
            "shape": list(self.grid.shape),
            "delta_abs_p50": float(torch.quantile(delta, 0.5)),
            "delta_abs_p95": float(torch.quantile(delta, 0.95)),
            "delta_abs_max": float(delta.max()),
        }
