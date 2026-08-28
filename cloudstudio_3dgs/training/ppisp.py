# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from nv-tlabs/ppisp (Apache-2.0),
# commit df33809f7b3b20ac06de088dfc871b144b8fb54d.
# The upstream production path is a CUDA extension; this file is a pure
# PyTorch equivalent derived from the upstream math (ppisp/src/ppisp_math.cuh
# via the upstream torch reference tests/torch_reference.py and the
# regularization reference in tests/test_regularization_loss.py). The
# upstream CNN controller and LR scheduler are intentionally not ported.

"""Per-camera physically-plausible ISP correction (PPISP) for 3DGS training.

This generalizes the per-frame scalar exposure compensation in
``cloudstudio_3dgs.training.exposure``:

- exposure: one log2 gain per parameter group, ``rgb * exp2(e)``;
- vignetting: per camera, per channel radial polynomial
  ``1 + a0*r^2 + a1*r^4 + a2*r^6`` around a learned optical center
  ``(cx, cy)``, clamped to [0, 1];
- color: a spatially uniform chromaticity homography in RGI space
  (Red, Green, Intensity), parameterized by 4 ZCA-whitened latent
  control-point offsets (8 scalars);
- CRF (optional, ``param_type="crf"``): per camera, per channel parametric
  toe/shoulder tone curve with gamma (4 raw params per channel).

Default ``param_type="no_crf"`` in ``mode="per_camera"`` costs
1 (exposure) + 3*5 (vignetting) + 8 (color) = 24 parameters per physical
camera; ``"crf"`` adds 3*4 = 12 more.

Relationship to ``ExposureCompensator``: our scalar compensator hard-projects
each camera group's log gains to zero mean after every optimizer step; PPISP
keeps the same anchoring semantics as a soft prior instead (SmoothL1 pull of
the exposure mean and of the mean color offsets toward zero, upstream default
weights 1.0). Both pin the global-brightness degree of freedom inside the
model where validation can see it. When PPISP is enabled it subsumes the
scalar exposure compensator - do not run both on the same images.

Trainer application order: render -> background compositing -> PPISP ->
photometric losses. Validation renders must bypass PPISP entirely (identity
correction) so metrics stay honest, mirroring the exposure module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_VALID_PARAM_TYPES = ("no_crf", "crf")
_VALID_MODES = ("per_camera", "per_image")
_VALID_LR_SCHEDULES = ("constant", "linear_warmup_exponential_decay")

# Upstream layout constants (match ppisp/src/ppisp_constants.h).
COLOR_PARAMS = 8
CRF_PARAMS_PER_CHANNEL = 4
NUM_VIGNETTING_ALPHA_TERMS = 3
VIGNETTING_PARAMS_PER_CHANNEL = 2 + NUM_VIGNETTING_ALPHA_TERMS  # cx, cy, a0..a2

# ZCA pinv blocks for color correction [Blue, Red, Green, Neutral], stored as
# an 8x8 block-diagonal matrix. Values copied verbatim from upstream
# ppisp/__init__.py (_COLOR_PINV_BLOCK_DIAG); each 2x2 block maps a whitened
# latent (dr, dg) to real chromaticity offsets of one control point.
_COLOR_PINV_BLOCKS = (
    ((0.0480542, -0.0043631), (-0.0043631, 0.0481283)),  # Blue
    ((0.0580570, -0.0179872), (-0.0179872, 0.0431061)),  # Red
    ((0.0433336, -0.0180537), (-0.0180537, 0.0580500)),  # Green
    ((0.0128369, -0.0034654), (-0.0034654, 0.0128158)),  # Neutral
)


@dataclass(frozen=True)
class PpispConfig:
    """Configuration mirroring upstream PPISPConfig defaults where applicable."""

    enabled: bool = False
    param_type: str = "no_crf"
    mode: str = "per_camera"
    learning_rate: float = 2e-3  # upstream ppisp_lr
    lr_schedule: str = "constant"
    warmup_fraction: float = 1.0 / 30.0
    warmup_start_multiplier: float = 0.01
    final_lr_multiplier: float = 0.01
    # Regularization weights (upstream defaults).
    exposure_mean_weight: float = 1.0  # SmoothL1(beta=0.1) on exposure mean
    vig_center_weight: float = 0.02  # squared distance of optical center from 0
    vig_channel_weight: float = 0.1  # variance across RGB channels
    vig_non_pos_weight: float = 0.01  # relu penalty on positive alphas
    color_mean_weight: float = 1.0  # SmoothL1(beta=0.005) on mean color offsets
    crf_channel_weight: float = 0.1  # variance across RGB channels (crf only)

    def validate(self) -> None:
        if self.param_type not in _VALID_PARAM_TYPES:
            raise ValueError(
                f"ppisp param_type must be one of {_VALID_PARAM_TYPES}, got {self.param_type!r}"
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"ppisp mode must be one of {_VALID_MODES}, got {self.mode!r}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError("ppisp learning_rate must be positive")
        if self.lr_schedule not in _VALID_LR_SCHEDULES:
            raise ValueError(
                "ppisp lr_schedule must be one of "
                f"{_VALID_LR_SCHEDULES}, got {self.lr_schedule!r}"
            )
        if not 0.0 < self.warmup_fraction < 1.0:
            raise ValueError("ppisp warmup_fraction must be within (0, 1)")
        if not 0.0 < self.warmup_start_multiplier <= 1.0:
            raise ValueError(
                "ppisp warmup_start_multiplier must be within (0, 1]"
            )
        if not 0.0 < self.final_lr_multiplier <= 1.0:
            raise ValueError("ppisp final_lr_multiplier must be within (0, 1]")
        for name in (
            "exposure_mean_weight",
            "vig_center_weight",
            "vig_channel_weight",
            "vig_non_pos_weight",
            "color_mean_weight",
            "crf_channel_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"ppisp {name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "param_type": self.param_type,
            "mode": self.mode,
            "learning_rate": self.learning_rate,
            "lr_schedule": self.lr_schedule,
            "warmup_fraction": self.warmup_fraction,
            "warmup_start_multiplier": self.warmup_start_multiplier,
            "final_lr_multiplier": self.final_lr_multiplier,
            "exposure_mean_weight": self.exposure_mean_weight,
            "vig_center_weight": self.vig_center_weight,
            "vig_channel_weight": self.vig_channel_weight,
            "vig_non_pos_weight": self.vig_non_pos_weight,
            "color_mean_weight": self.color_mean_weight,
            "crf_channel_weight": self.crf_channel_weight,
        }

    def learning_rate_for_step(self, *, step: int, total_steps: int) -> float:
        """Return the deterministic PPISP LR for one zero-based training step.

        ``linear_warmup_exponential_decay`` mirrors the recovered MipMap
        BilateralGrid schedule: warm up from 1% to the base LR during the first
        ``floor(total/30)`` steps, then decay exponentially to 1% at the final
        step.  The formula is stateless so interrupted resumes are exact.
        """

        import math

        if total_steps <= 0:
            raise ValueError("ppisp total_steps must be positive")
        if step < 0 or step >= total_steps:
            raise ValueError("ppisp step must be within the training schedule")
        if self.lr_schedule == "constant":
            return float(self.learning_rate)
        warmup_steps = max(1, int(math.floor(total_steps * self.warmup_fraction)))
        if step < warmup_steps:
            progress = float(step) / float(max(1, warmup_steps - 1))
            multiplier = self.warmup_start_multiplier + progress * (
                1.0 - self.warmup_start_multiplier
            )
        else:
            remaining_steps = total_steps - warmup_steps
            if remaining_steps <= 1:
                progress = 1.0
            else:
                decay_steps = remaining_steps - 1
                progress = float(step - warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)
            multiplier = math.exp(math.log(self.final_lr_multiplier) * progress)
        return float(self.learning_rate * multiplier)


def _softplus_inverse(x: float, min_value: float = 0.0, epsilon: float = 1e-5) -> float:
    """Inverse of ``min_value + softplus(raw)`` at value ``x`` (upstream init)."""
    import math

    clamped = max(epsilon, x - min_value)
    return math.log(math.expm1(clamped))


class PpispCorrector:
    """Own the differentiable PPISP parameters and apply them to renders.

    Parameter layout (upstream shapes, group counts adapted to ``mode``):

    - ``exposure_params``: [F] log2 gains (F = frame slots);
    - ``vignetting_params``: [C, 3, 5] per camera, per channel
      ``[cx, cy, a0, a1, a2]``;
    - ``color_params``: [F, 8] latent homography control-point offsets;
    - ``crf_params``: [C, 3, 4] raw toe/shoulder/gamma/center (crf only).

    ``mode="per_camera"``: frame slots == camera slots (one full parameter set
    per physical camera, the product default for the two-fisheye S1 rig).
    ``mode="per_image"``: exposure and color are per training image (the
    upstream per-frame semantics); vignetting and CRF stay per camera.
    """

    def __init__(
        self,
        image_ids: Iterable[str],
        *,
        config: PpispConfig,
        device: str,
        camera_by_image: Mapping[str, str],
    ) -> None:
        import torch

        config.validate()
        if not config.enabled:
            raise ValueError(
                "PpispCorrector cannot be constructed when ppisp is disabled"
            )
        ordered_images = sorted(set(str(image_id) for image_id in image_ids))
        if not ordered_images:
            raise ValueError("ppisp requires at least one training image")
        missing = [i for i in ordered_images if i not in camera_by_image]
        if missing:
            raise ValueError(
                f"ppisp camera mapping missing for {len(missing)} images, e.g. {missing[0]!r}"
            )
        self.config = config
        self.device = device

        cameras = sorted(set(str(camera_by_image[i]) for i in ordered_images))
        self.camera_index = {camera: pos for pos, camera in enumerate(cameras)}
        self.camera_by_image = {
            image_id: str(camera_by_image[image_id]) for image_id in ordered_images
        }
        if config.mode == "per_camera":
            self.frame_index = dict(self.camera_index)
        else:
            self.frame_index = {
                image_id: pos for pos, image_id in enumerate(ordered_images)
            }
        num_cameras = len(self.camera_index)
        num_frames = len(self.frame_index)

        self.exposure_params = torch.nn.Parameter(
            torch.zeros(num_frames, dtype=torch.float32, device=device)
        )
        self.vignetting_params = torch.nn.Parameter(
            torch.zeros(
                num_cameras,
                3,
                VIGNETTING_PARAMS_PER_CHANNEL,
                dtype=torch.float32,
                device=device,
            )
        )
        self.color_params = torch.nn.Parameter(
            torch.zeros(num_frames, COLOR_PARAMS, dtype=torch.float32, device=device)
        )
        if config.param_type == "crf":
            # Upstream identity-like CRF init: toe=shoulder=1 (min 0.3),
            # gamma=1 (min 0.1), center=sigmoid(0)=0.5.
            crf_raw = torch.tensor(
                [
                    _softplus_inverse(1.0, min_value=0.3),
                    _softplus_inverse(1.0, min_value=0.3),
                    _softplus_inverse(1.0, min_value=0.1),
                    0.0,
                ],
                dtype=torch.float32,
                device=device,
            )
            self.crf_params: Any = torch.nn.Parameter(
                crf_raw.view(1, 1, CRF_PARAMS_PER_CHANNEL)
                .repeat(num_cameras, 3, 1)
                .contiguous()
            )
        else:
            self.crf_params = None

        self._color_pinv = torch.tensor(
            _COLOR_PINV_BLOCKS, dtype=torch.float32, device=device
        )  # [4, 2, 2] -> block-diagonalized lazily in _color_offsets
        self._uv_cache: dict[tuple[int, int], Any] = {}

    # ------------------------------------------------------------------
    # Slot resolution
    # ------------------------------------------------------------------

    def _resolve(self, image_id: str) -> tuple[int, int]:
        """Return (camera_slot, frame_slot) for an image id.

        In per_camera mode a bare camera id is also accepted, since every
        image of one camera shares the same parameter set.
        """
        key = str(image_id)
        camera = self.camera_by_image.get(key)
        if camera is None and self.config.mode == "per_camera" and key in self.camera_index:
            camera = key
        if camera is None:
            raise KeyError(f"ppisp has no parameters for image {key!r}")
        camera_slot = self.camera_index[camera]
        if self.config.mode == "per_camera":
            frame_slot = camera_slot
        else:
            frame_slot = self.frame_index[key]
        return camera_slot, frame_slot

    def parameters(self) -> list[Any]:
        params = [self.exposure_params, self.vignetting_params, self.color_params]
        if self.crf_params is not None:
            params.append(self.crf_params)
        return params

    def make_optimizer(self) -> Any:
        """Adam over all PPISP parameters, matching our exposure module style.

        Upstream nv-tlabs/ppisp uses Adam(lr=2e-3, eps=1e-15).  The trainer
        applies the optional deterministic warmup/decay configured above by
        rewriting this optimizer's param-group LR before each step.
        """
        import torch

        return torch.optim.Adam(
            [
                {
                    "params": self.parameters(),
                    "lr": self.config.learning_rate,
                    "name": "ppisp",
                }
            ],
            eps=1e-15,
        )

    # ------------------------------------------------------------------
    # Forward pieces (pure PyTorch equivalents of upstream ppisp_math.cuh)
    # ------------------------------------------------------------------

    def _pixel_uv(self, height: int, width: int) -> Any:
        """Normalized pixel-center coordinates [H, W, 2] in [-0.5, 0.5].

        Matches the upstream convention: pixel centers (x+0.5, y+0.5),
        ``uv = (coords - resolution/2) / max(resolution)``.
        """
        import torch

        cached = self._uv_cache.get((height, width))
        if cached is not None:
            return cached
        ys = torch.arange(height, dtype=torch.float32, device=self.device) + 0.5
        xs = torch.arange(width, dtype=torch.float32, device=self.device) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2] (x, y)
        resolution = torch.tensor(
            [float(width), float(height)], dtype=torch.float32, device=self.device
        )
        uv = (coords - resolution * 0.5) / resolution.max()
        self._uv_cache[(height, width)] = uv
        return uv

    def _vignetting_falloff(self, camera_slot: int, uv: Any) -> Any:
        """Per-channel radial falloff [H, W, 3], clamped to [0, 1]."""
        params = self.vignetting_params[camera_slot]  # [3, 5]
        centers = params[:, :2]  # [3, 2]
        alphas = params[:, 2:]  # [3, 3]
        delta = uv.unsqueeze(2) - centers.view(1, 1, 3, 2)  # [H, W, 3, 2]
        r2 = (delta * delta).sum(dim=-1)  # [H, W, 3]
        falloff = 1.0 + alphas[:, 0] * r2 + alphas[:, 1] * r2**2 + alphas[:, 2] * r2**3
        return falloff.clamp(0.0, 1.0)

    def _color_homography(self, frame_slot: int) -> Any:
        """Chromaticity homography [3, 3] from 8 latent params (upstream math)."""
        import torch

        cp = self.color_params[frame_slot]  # [8]
        block_diag = torch.block_diag(*self._color_pinv)  # [8, 8]
        offsets = cp @ block_diag  # real (dr, dg) per control point

        device = cp.device
        # Fixed source chromaticities (r, g, 1): Blue, Red, Green, Neutral.
        sources = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        targets = torch.cat(
            [sources[:, :2] + offsets.view(4, 2), sources[:, 2:]], dim=-1
        )  # [4, 3]

        T = targets[:3].transpose(0, 1)  # [3, 3], columns t_b, t_r, t_g
        t_gray = targets[3]
        zero = torch.zeros((), dtype=torch.float32, device=device)
        skew = torch.stack(
            [
                torch.stack([zero, -t_gray[2], t_gray[1]]),
                torch.stack([t_gray[2], zero, -t_gray[0]]),
                torch.stack([-t_gray[1], t_gray[0], zero]),
            ]
        )
        M = skew @ T

        # Nullspace vector via the largest of the three row cross products
        # (branch-free selection, mirrors the upstream torch reference).
        r0, r1, r2 = M[0], M[1], M[2]
        lam01 = torch.linalg.cross(r0, r1)
        lam02 = torch.linalg.cross(r0, r2)
        lam12 = torch.linalg.cross(r1, r2)
        n01 = (lam01 * lam01).sum()
        n02 = (lam02 * lam02).sum()
        n12 = (lam12 * lam12).sum()
        lam = torch.where(
            n01 >= n02,
            torch.where(n01 >= n12, lam01, lam12),
            torch.where(n02 >= n12, lam02, lam12),
        )

        # Precomputed inverse of S = [s_b s_r s_g] (columns).
        S_inv = torch.tensor(
            [[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        H = T @ torch.diag(lam) @ S_inv
        return H / (H[2, 2] + 1e-10)

    def _apply_color(self, rgb: Any, frame_slot: int) -> Any:
        import torch

        H = self._color_homography(frame_slot)
        intensity = rgb.sum(dim=-1, keepdim=True)  # [..., 1]
        rgi = torch.cat([rgb[..., 0:1], rgb[..., 1:2], intensity], dim=-1)
        rgi = rgi @ H.transpose(0, 1)
        rgi = rgi * (intensity / (rgi[..., 2:3] + 1e-5))
        r_out = rgi[..., 0]
        g_out = rgi[..., 1]
        b_out = rgi[..., 2] - r_out - g_out
        return torch.stack([r_out, g_out, b_out], dim=-1)

    def _apply_crf(self, rgb: Any, camera_slot: int) -> Any:
        import torch
        import torch.nn.functional as F

        crf = self.crf_params[camera_slot]  # [3, 4]
        toe = 0.3 + F.softplus(crf[:, 0])  # [3]
        shoulder = 0.3 + F.softplus(crf[:, 1])
        gamma = 0.1 + F.softplus(crf[:, 2])
        center = torch.sigmoid(crf[:, 3])

        lerp_val = toe + center * (shoulder - toe)
        a = (shoulder * center) / lerp_val
        b = 1.0 - a

        x = rgb.clamp(0.0, 1.0)
        eps = 1e-6  # avoid NaN gradients from pow(0, fractional)
        y_low = a * torch.pow((x / center).clamp(min=eps), toe)
        y_high = 1.0 - b * torch.pow(((1.0 - x) / (1.0 - center)).clamp(min=eps), shoulder)
        y = torch.where(x <= center, y_low, y_high)
        return torch.pow(y.clamp(min=eps), gamma)

    def apply(
        self,
        rgb: Any,
        image_id: str,
        pixel_coords: Any = None,
        resolution: tuple[int, int] | None = None,
    ) -> Any:
        """Apply the full pipeline to a rendered image, differentiably.

        Order (upstream): exposure -> vignetting -> color -> CRF.

        Args:
            rgb: Rendered RGB [H, W, 3], after background compositing,
                linear-ish [0, 1] range.
            image_id: Training image id (or camera id in per_camera mode).
            pixel_coords: Optional absolute pixel coordinates [H, W, 2] (x, y)
                for crops rendered from a larger sensor; defaults to the full
                image's pixel centers.
            resolution: Optional (width, height) of the full sensor when
                pixel_coords is given; defaults to rgb's own shape.

        Returns:
            Corrected RGB, same shape as ``rgb``. Only call this on training
            renders; validation must stay at identity correction.
        """
        if rgb.dim() != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"ppisp expects rgb of shape [H, W, 3], got {tuple(rgb.shape)}")
        camera_slot, frame_slot = self._resolve(image_id)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])

        # Exposure: rgb * 2^e (upstream exp2 convention; our scalar exposure
        # module uses natural exp, hence its distinct log-gain scale).
        rgb = rgb * (2.0 ** self.exposure_params[frame_slot])

        # Vignetting.
        if pixel_coords is None:
            uv = self._pixel_uv(height, width)
        else:
            if pixel_coords.shape != rgb.shape[:2] + (2,) and tuple(
                pixel_coords.shape
            ) != (height, width, 2):
                raise ValueError(
                    f"pixel_coords must be [H, W, 2] matching rgb, got {tuple(pixel_coords.shape)}"
                )
            if resolution is None:
                res_w, res_h = width, height
            else:
                res_w, res_h = int(resolution[0]), int(resolution[1])
            res = pixel_coords.new_tensor([float(res_w), float(res_h)])
            uv = (pixel_coords - res * 0.5) / res.max()
        rgb = rgb * self._vignetting_falloff(camera_slot, uv)

        # Color homography in RGI space.
        rgb = self._apply_color(rgb, frame_slot)

        # Optional CRF tone curve.
        if self.crf_params is not None:
            rgb = self._apply_crf(rgb, camera_slot)
        return rgb

    # ------------------------------------------------------------------
    # Regularization (upstream tests/test_regularization_loss.py reference)
    # ------------------------------------------------------------------

    def regularization_loss(self) -> Any:
        import torch
        import torch.nn.functional as F

        cfg = self.config
        total = torch.zeros((), dtype=torch.float32, device=self.exposure_params.device)

        if cfg.exposure_mean_weight > 0:
            residual = self.exposure_params.mean()
            total = total + cfg.exposure_mean_weight * F.smooth_l1_loss(
                residual, torch.zeros_like(residual), beta=0.1
            )

        if cfg.vig_center_weight > 0:
            centers = self.vignetting_params[:, :, :2]
            total = total + cfg.vig_center_weight * (centers**2).sum(dim=-1).mean()

        if cfg.vig_non_pos_weight > 0:
            alphas = self.vignetting_params[:, :, 2:]
            total = total + cfg.vig_non_pos_weight * F.relu(alphas).mean()

        if cfg.vig_channel_weight > 0:
            total = total + cfg.vig_channel_weight * self.vignetting_params.var(
                dim=1, unbiased=False
            ).mean()

        if cfg.color_mean_weight > 0:
            block_diag = torch.block_diag(*self._color_pinv)
            offsets = self.color_params @ block_diag
            residual = offsets.mean(dim=0)
            total = total + cfg.color_mean_weight * F.smooth_l1_loss(
                residual, torch.zeros_like(residual), beta=0.005, reduction="mean"
            )

        if self.crf_params is not None and cfg.crf_channel_weight > 0:
            total = total + cfg.crf_channel_weight * self.crf_params.var(
                dim=1, unbiased=False
            ).mean()

        return total

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            exposure = self.exposure_params.detach()
            centers = self.vignetting_params.detach()[:, :, :2]
            alphas = self.vignetting_params.detach()[:, :, 2:]
            block_diag = torch.block_diag(*self._color_pinv)
            color_offsets = self.color_params.detach() @ block_diag
            result: dict[str, Any] = {
                "mode": self.config.mode,
                "param_type": self.config.param_type,
                "camera_count": len(self.camera_index),
                "frame_slot_count": int(self.exposure_params.shape[0]),
                "parameter_count": int(sum(p.numel() for p in self.parameters())),
                "exposure_log2_mean": float(exposure.mean()),
                "exposure_log2_abs_max": float(exposure.abs().max()),
                "exposure_gain_min": float((2.0**exposure).min()),
                "exposure_gain_max": float((2.0**exposure).max()),
                "vig_center_offset_max": float(centers.norm(dim=-1).max()),
                "vig_alpha_min": float(alphas.min()),
                "vig_alpha_max": float(alphas.max()),
                "vig_positive_alpha_fraction": float((alphas > 0.0).float().mean()),
                "color_offset_abs_mean": float(color_offsets.abs().mean()),
                "color_offset_abs_max": float(color_offsets.abs().max()),
            }
            if self.crf_params is not None:
                crf = self.crf_params.detach()
                result["crf_raw_abs_max"] = float(crf.abs().max())
                result["crf_channel_var_mean"] = float(
                    crf.var(dim=1, unbiased=False).mean()
                )
        return result
