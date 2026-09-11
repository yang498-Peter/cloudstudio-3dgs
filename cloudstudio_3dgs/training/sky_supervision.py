# SPDX-License-Identifier: Apache-2.0
"""Sky supervision: sky pixels belong to the backdrop, the Tile must not paint them.

What was measured (research/quality_recovery_v2, 2026-09-11): the per-view
backdrop of the Tile route is rendered from a sky dome only. Where the photo
shows sky (and the trees in front of it) the dome is a blotchy blue, so under
full-frame photometric supervision the Tile grows gaussians to paint the sky
and those gaussians float in front of walls and eaves - 34% of the 20k Tile_1
population sits more than 0.2 m off the LiDAR surface, concentrated in the
eave/sky band. The user's direction: gaussians may grow freely where there is
no LiDAR; the sky is the special case. On sky pixels the Tile must NOT paint.

This module gives the trainer a signed per-face sky mask (produced offline by
``tools/build_sky_masks.py`` from a semantic segmentation) and three uses of
it, every one of them off by default and byte-identical when off:

* **alpha term** - on the effective sky pixels the accumulated alpha the
  rasterizer produced BEFORE the backdrop composite (``backend.render``
  returns it; the composite is ``rgb + (1 - alpha) * backdrop``) is pulled to
  ``alpha_target`` (0.0) with a plain L1, normalised by the number of
  effective sky pixels;
* **exclusion** - the photo-derived terms (RGB L1, SSIM, gradient L1,
  LiDAR-supported RGB L1, train-view PSNR) and the DA2 depth term are
  computed on non-sky pixels only, through the same mask plumbing
  ``rgb_supervision_mask`` already uses (``exclude=``);
* **growth block** - a gaussian whose projected centre fell inside the
  effective sky mask in at least ``SKY_GROWTH_BLOCK_MIN_SKY_FRACTION`` of the
  views it was rendered in since the last refine event may not clone or
  split (``SkyGrowthBlock``; a pure parent mask in the classic lifecycle,
  like the surface-anchor parent guard).

Effective sky mask
------------------
Segmentation errors are the risk: glass, overexposed white walls and bright
metal roofs get labelled sky, and the alpha term would force them
transparent. Two guards, both config knobs, turn the raw label into the
*effective* mask the three uses read:

1. the raw sky mask is eroded by ``mask_erosion_px`` (a separable square
   min-filter; pixels outside the image do not erode the border, so sky at
   the top edge of a crop stays sky);
2. every pixel within ``require_no_lidar_within_px`` (square window) of a
   signed LiDAR return of the view is removed - a real surface the scanner
   hit is never sky, whatever the label says. The depth mask the dataset
   already carries per sample (after the tile crop and ownership masking)
   is the return set;
3. the result is intersected with the renderer mask (``rgb_mask``).

Two twins on purpose, as for the alpha support and the ownership mask: a
torch version for the trainer (``max_pool2d`` on the device - the ownership
mask showed that full-resolution dilation on the CPU every step costs 3x
wall clock) and a numpy version for CPU audits; ``tests/test_sky_supervision.py``
pins them against a plain-loop oracle.

Manifest contract (``sky_mask_train.json``, produced by the other task)::

    {schema_version, kind: "face4_sky_mask_cache", split, model{...},
     label_ids, rule, source_face_manifest_sha256,
     masks: [{image_id, camera_id, face_id, width, height, mask_path,
              mask_sha256, sky_pixels, sky_fraction}],
     summary, manifest_sha256}

``mask_path`` is relative to ``mask_root`` (uint8 PNG, sky = 255). When
``cloudstudio_3dgs.data.sky_masks`` exists its verifier is used; otherwise
the minimal verifier below (same canonical-JSON signature rule as every
other manifest of this package) applies. This module never creates that
file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


try:
    # The data-side module (written by the mask build task) owns the manifest
    # contract; when it is importable its constants, signer and verifier are
    # authoritative and the local definitions below are never used.
    from cloudstudio_3dgs.data.sky_masks import (  # type: ignore[import-not-found]
        SKY_MASK_KIND as _DATA_SKY_MASK_KIND,
        SKY_MASK_MANIFEST_SHA_KEY as _DATA_SKY_MASK_SIGNATURE_KEY,
        SKY_MASK_SCHEMA_VERSION as _DATA_SKY_MASK_SCHEMA_VERSION,
        sign_sky_mask_manifest as _data_sign_sky_mask_manifest,
        verify_sky_mask_manifest as _data_verify_sky_mask_manifest,
    )

    HAS_DATA_SKY_MASKS = True
except ImportError:  # pragma: no cover - exercised only before that file exists
    HAS_DATA_SKY_MASKS = False
    _DATA_SKY_MASK_KIND = "face4_sky_mask_cache"
    _DATA_SKY_MASK_SIGNATURE_KEY = "sky_mask_manifest_sha256"
    _DATA_SKY_MASK_SCHEMA_VERSION = 1
    _data_sign_sky_mask_manifest = None
    _data_verify_sky_mask_manifest = None

SKY_MASK_SCHEMA_VERSION = int(_DATA_SKY_MASK_SCHEMA_VERSION)
SKY_MASK_KIND = str(_DATA_SKY_MASK_KIND)
SKY_MASK_SIGNATURE_KEY = str(_DATA_SKY_MASK_SIGNATURE_KEY)

# Share of a gaussian's observations (since the last refine event) whose
# projected centre lay in the effective sky mask, at or above which the row
# may not become a parent.
SKY_GROWTH_BLOCK_MIN_SKY_FRACTION = 0.5

# Loss terms the exclusion silences; recorded in the contract so a reader of a
# run manifest does not need this source to know what was masked.
SKY_EXCLUDED_PHOTOMETRIC_TERMS: tuple[str, ...] = (
    "rgb_l1",
    "rgb_ssim",
    "rgb_gradient_l1",
    "lidar_rgb_l1",
    "train_view_psnr",
)
SKY_EXCLUDED_DEPTH_TERMS: tuple[str, ...] = ("da2_depth",)

# Strategy-state keys of the growth block accumulators. Tensors of population
# length ride gsplat's duplicate/split/remove like grad2d/count; the total is
# a plain int the topology ops leave alone (survives a checkpoint resume).
SKY_SEEN_KEY = "_cloudstudio_sky_seen"
SKY_HIT_KEY = "_cloudstudio_sky_hits"
SKY_BLOCKED_TOTAL_KEY = "_cloudstudio_sky_growth_blocked_total"

# ``info`` key under which the loss hands the step's effective sky mask to
# the strategy (the same dict reaches strategy_post_step).
SKY_MASK_INFO_KEY = "cloudstudio_sky_mask"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkySupervisionConfig:
    """Knobs of the sky supervision. Disabled by default: byte-identical.

    Attributes:
        enabled: master switch. Off, nothing below is read, the dataset loads
            no sky mask and no contract key is emitted.
        mask_manifest: signed ``face4_sky_mask_cache`` manifest bound to the
            training Face4 cache.
        mask_root: directory the manifest's ``mask_path`` entries are
            relative to.
        alpha_weight: weight of the sky alpha term.
        alpha_target: alpha the effective sky pixels are pulled to.
        exclude_photometric: silence the photo-derived terms on sky pixels.
        exclude_mono_depth: silence the DA2 depth term on sky pixels.
        growth_block: refuse growth from parents seen predominantly in sky.
        mask_erosion_px: erosion radius of the raw sky label (guard 1).
        require_no_lidar_within_px: half-width of the square window around a
            signed LiDAR return inside which no pixel is sky (guard 2); 0
            disables the guard.
    """

    enabled: bool = False
    mask_manifest: Path | None = None
    mask_root: Path | None = None
    alpha_weight: float = 0.5
    alpha_target: float = 0.0
    exclude_photometric: bool = True
    exclude_mono_depth: bool = True
    growth_block: bool = True
    mask_erosion_px: int = 8
    require_no_lidar_within_px: int = 24

    @classmethod
    def from_value(cls, value: Any) -> "SkySupervisionConfig":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("sky_supervision must be an object")
        payload = dict(value)
        for key in ("mask_manifest", "mask_root"):
            if payload.get(key) is not None:
                payload[key] = Path(str(payload[key]))
        try:
            return cls(**payload)
        except TypeError as error:
            raise ValueError(f"sky_supervision has an unknown field: {error}") from None

    def validate(self) -> None:
        for name in ("enabled", "exclude_photometric", "exclude_mono_depth", "growth_block"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"sky_supervision.{name} must be a boolean")
        for name in ("alpha_weight", "alpha_target"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(float(value)):
                raise ValueError(f"sky_supervision.{name} must be a finite number")
        if float(self.alpha_weight) < 0.0:
            raise ValueError("sky_supervision.alpha_weight must be non-negative")
        if not 0.0 <= float(self.alpha_target) <= 1.0:
            raise ValueError("sky_supervision.alpha_target must lie within [0, 1]")
        for name in ("mask_erosion_px", "require_no_lidar_within_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 256:
                raise ValueError(
                    f"sky_supervision.{name} must be an integer within [0, 256]"
                )
        if self.enabled:
            if self.mask_manifest is None or self.mask_root is None:
                raise ValueError(
                    "sky_supervision.enabled requires mask_manifest and mask_root"
                )
            if (
                float(self.alpha_weight) <= 0.0
                and not self.exclude_photometric
                and not self.exclude_mono_depth
                and not self.growth_block
            ):
                raise ValueError(
                    "sky_supervision.enabled with alpha_weight 0 and every "
                    "exclusion off would do nothing"
                )
        elif self.mask_manifest is not None or self.mask_root is not None:
            raise ValueError(
                "sky_supervision mask paths are set but enabled is false"
            )

    def to_dict(self) -> dict[str, Any]:
        """Contract form: emitted by the trainer only when enabled.

        The manifest is bound by its SHA256 (added by the trainer), not by
        its path, so two machines with the same cache sign the same contract.
        """
        return {
            "enabled": bool(self.enabled),
            "alpha_weight": float(self.alpha_weight),
            "alpha_target": float(self.alpha_target),
            "exclude_photometric": bool(self.exclude_photometric),
            "exclude_mono_depth": bool(self.exclude_mono_depth),
            "growth_block": bool(self.growth_block),
            "mask_erosion_px": int(self.mask_erosion_px),
            "require_no_lidar_within_px": int(self.require_no_lidar_within_px),
            "alpha_source": "accumulated_alpha_before_backdrop_composite",
            "alpha_loss": "mean_abs_alpha_minus_target_over_effective_sky_pixels",
            "effective_sky": (
                "erode(sky_mask, mask_erosion_px) & ~dilate(depth_mask, "
                "require_no_lidar_within_px) & rgb_mask"
            ),
            "excluded_photometric_terms": (
                list(SKY_EXCLUDED_PHOTOMETRIC_TERMS) if self.exclude_photometric else []
            ),
            "excluded_depth_terms": (
                list(SKY_EXCLUDED_DEPTH_TERMS) if self.exclude_mono_depth else []
            ),
            "exclusion_normalisation": "supervised_pixels",
            "growth_block_rule": (
                {
                    "min_sky_fraction": SKY_GROWTH_BLOCK_MIN_SKY_FRACTION,
                    "observation": "projected_centre_inside_effective_sky_mask",
                    "window": "since_last_refine_event",
                }
                if self.growth_block
                else None
            ),
        }


# ---------------------------------------------------------------------------
# manifest verification (imports the data-side verifier when it exists)
# ---------------------------------------------------------------------------


def sign_sky_mask_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with its canonical-JSON SHA256 appended."""
    if _data_sign_sky_mask_manifest is not None:
        return _data_sign_sky_mask_manifest(payload)
    unsigned = dict(payload)
    unsigned.pop(SKY_MASK_SIGNATURE_KEY, None)
    signed = dict(unsigned)
    signed[SKY_MASK_SIGNATURE_KEY] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def _verify_sky_mask_manifest_local(manifest: dict[str, Any]) -> str:
    """Minimal fail-closed structural + signature check; returns the SHA256."""
    expected = str(manifest.get(SKY_MASK_SIGNATURE_KEY, ""))
    if len(expected) != 64:
        raise ValueError("sky mask manifest is unsigned")
    unsigned = dict(manifest)
    unsigned.pop(SKY_MASK_SIGNATURE_KEY, None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("sky mask manifest signature mismatch")
    if int(manifest.get("schema_version", -1)) != SKY_MASK_SCHEMA_VERSION:
        raise ValueError("unsupported sky mask manifest schema_version")
    if manifest.get("kind") != SKY_MASK_KIND:
        raise ValueError("manifest is not a Face4 sky mask cache")
    if len(str(manifest.get("source_face_manifest_sha256", ""))) != 64:
        raise ValueError("sky mask manifest is not bound to a Face4 cache")
    records = manifest.get("masks", [])
    keys = [
        (str(record.get("image_id", "")), str(record.get("face_id", "")))
        for record in records
    ]
    if not records or len(keys) != len(set(keys)) or any(not all(key) for key in keys):
        raise ValueError("sky mask manifest has invalid or duplicate records")
    for record in records:
        if "\\" in str(record.get("mask_path", "")) or not record.get("mask_path"):
            raise ValueError("sky mask manifest records need forward-slash mask paths")
        if len(str(record.get("mask_sha256", ""))) != 64:
            raise ValueError("sky mask manifest records need a mask_sha256")
    return expected


def verify_sky_mask_manifest(manifest: dict[str, Any]) -> str:
    """Verify a sky mask manifest, preferring the data-side verifier.

    ``cloudstudio_3dgs.data.sky_masks`` is written by the mask build task;
    when it is importable its verifier is authoritative (it also checks the
    model identity, the decision rule, the per-record pixel counts and the
    summary). Until then the local rule above applies. Either way a tampered
    or unsigned manifest raises.
    """
    if _data_verify_sky_mask_manifest is not None:
        return str(_data_verify_sky_mask_manifest(manifest))
    return _verify_sky_mask_manifest_local(manifest)


# ---------------------------------------------------------------------------
# effective sky mask (torch + numpy twins)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkySupervision:
    """``mask`` is the effective sky mask (within rgb_mask).

    ``raw_sky_pixels`` counts ``sky_mask & rgb_mask`` before the guards so
    telemetry can show how much the erosion/LiDAR guard removed.
    """

    mask: Any
    sky_pixels: int
    raw_sky_pixels: int
    rgb_mask_pixels: int

    @property
    def fraction(self) -> float:
        """Effective sky pixels as a share of rgb_mask."""
        if self.rgb_mask_pixels <= 0:
            return 0.0
        return float(self.sky_pixels) / float(self.rgb_mask_pixels)


def _max_pool_bool(torch: Any, mask: Any, radius: int) -> Any:
    """Square max-filter of a bool [H, W] tensor, zero-padded (radius 0: same)."""
    if radius <= 0:
        return mask
    pooled = torch.nn.functional.max_pool2d(
        mask[None, None].to(dtype=torch.float32),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[0, 0]
    return pooled > 0.5


def dilate_bool_numpy(mask: np.ndarray, radius: int) -> np.ndarray:
    """Separable square max-filter with zero padding (numpy twin)."""
    if radius <= 0:
        return mask
    out = mask
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="constant", constant_values=False)
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for shift in range(2 * radius + 1):
            acc |= np.take(padded, range(shift, shift + n), axis=axis)
        out = acc
    return out


def sky_supervision_mask(
    torch: Any,
    *,
    sky_mask: Any,
    rgb_mask: Any,
    depth_mask: Any | None,
    erosion_px: int,
    lidar_window_px: int,
) -> SkySupervision:
    """Torch construction of the effective sky mask for one view.

    Erosion is the complement of a dilation of the complement, zero-padded,
    so pixels beyond the image never erode the border. A view without any
    signed LiDAR return (``depth_mask is None``) skips guard 2.
    """
    if tuple(sky_mask.shape) != tuple(rgb_mask.shape):
        raise ValueError(
            f"sky mask shape {tuple(sky_mask.shape)} differs from rgb_mask "
            f"{tuple(rgb_mask.shape)}"
        )
    raw = sky_mask & rgb_mask
    effective = sky_mask
    if int(erosion_px) > 0:
        effective = ~_max_pool_bool(torch, ~sky_mask, int(erosion_px))
    if int(lidar_window_px) > 0 and depth_mask is not None:
        effective = effective & ~_max_pool_bool(torch, depth_mask, int(lidar_window_px))
    mask = effective & rgb_mask
    return SkySupervision(
        mask=mask,
        sky_pixels=int(mask.sum().item()),
        raw_sky_pixels=int(raw.sum().item()),
        rgb_mask_pixels=int(rgb_mask.sum().item()),
    )


def sky_supervision_mask_numpy(
    *,
    sky_mask: np.ndarray,
    rgb_mask: np.ndarray,
    depth_mask: np.ndarray | None,
    erosion_px: int,
    lidar_window_px: int,
) -> SkySupervision:
    """numpy twin of :func:`sky_supervision_mask` (same mask, same counts)."""
    sky_mask = np.asarray(sky_mask, dtype=bool)
    rgb_mask = np.asarray(rgb_mask, dtype=bool)
    if sky_mask.shape != rgb_mask.shape:
        raise ValueError(
            f"sky mask shape {sky_mask.shape} differs from rgb_mask {rgb_mask.shape}"
        )
    raw = sky_mask & rgb_mask
    effective = sky_mask
    if int(erosion_px) > 0:
        effective = ~dilate_bool_numpy(~sky_mask, int(erosion_px))
    if int(lidar_window_px) > 0 and depth_mask is not None:
        effective = effective & ~dilate_bool_numpy(
            np.asarray(depth_mask, dtype=bool), int(lidar_window_px)
        )
    mask = effective & rgb_mask
    return SkySupervision(
        mask=mask,
        sky_pixels=int(mask.sum()),
        raw_sky_pixels=int(raw.sum()),
        rgb_mask_pixels=int(rgb_mask.sum()),
    )


# ---------------------------------------------------------------------------
# growth block (held by the classic lifecycle adapter)
# ---------------------------------------------------------------------------


class SkyGrowthBlock:
    """Per-gaussian sky-observation counters and the parent mask they imply.

    ``accumulate`` runs every pre-refine-stop step next to the strategy's
    own ``_update_state`` and reads the same ``info`` (projected centres
    ``means2d`` and the visibility ``radii``), plus the step's effective sky
    mask the loss stashed under ``info[SKY_MASK_INFO_KEY]``. A visible
    gaussian whose centre pixel is sky counts one hit; every visible gaussian
    counts one observation. ``blocked_parent_mask`` marks rows whose hit
    share is at least ``min_sky_fraction``; ``reset`` zeroes both counters
    at the refine event, exactly when ``grad2d``/``count`` are zeroed.

    This is the documented cheap proxy for "growth gradient came
    predominantly from sky pixels": the gradient itself is one scalar per
    gaussian per view and cannot be split by pixel after the fact, while
    the centre pixel is free. A sky-painting gaussian's centre is in the sky;
    a wall gaussian whose footprint merely touches the sky band is not.
    """

    def __init__(self, min_sky_fraction: float = SKY_GROWTH_BLOCK_MIN_SKY_FRACTION) -> None:
        if not 0.0 < float(min_sky_fraction) <= 1.0:
            raise ValueError("min_sky_fraction must lie within (0, 1]")
        self.min_sky_fraction = float(min_sky_fraction)
        self.last_stats: dict[str, Any] = {}

    def _ensure_buffers(self, params: Any, state: dict[str, Any]) -> tuple[Any, Any]:
        torch = __import__("torch")
        reference = params["means"]
        count = len(reference)
        for key in (SKY_SEEN_KEY, SKY_HIT_KEY):
            value = state.get(key)
            # A warm start or topology change resizes the population; a
            # stale length is rebuilt rather than crashing (footprint
            # accumulator precedent).
            if not isinstance(value, torch.Tensor) or len(value) != count:
                state[key] = torch.zeros(count, dtype=torch.float32, device=reference.device)
        return state[SKY_SEEN_KEY], state[SKY_HIT_KEY]

    def accumulate(self, params: Any, state: dict[str, Any], info: dict[str, Any]) -> None:
        torch = __import__("torch")
        sky = info.get(SKY_MASK_INFO_KEY)
        if sky is None:
            raise RuntimeError(
                "sky growth block needs the step's effective sky mask "
                f"(info[{SKY_MASK_INFO_KEY!r}]); the loss did not provide one"
            )
        seen, hits = self._ensure_buffers(params, state)
        means2d = info["means2d"].detach()
        radii = info["radii"]
        if info.get("gaussian_ids") is not None and means2d.dim() == 2:
            gs_ids = info["gaussian_ids"]
            centres = means2d
        else:
            visible = (radii > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(visible)[1]
            centres = means2d[visible]
        if gs_ids.numel() == 0:
            self.last_stats = {"observed": 0, "sky_hits": 0}
            return
        height, width = int(sky.shape[0]), int(sky.shape[1])
        # gsplat projects to pixel coordinates with pixel (i, j) covering
        # [i, i+1) x [j, j+1) (its rasterizer samples centres at +0.5), so the
        # floor is the pixel the centre lands in.
        px = torch.floor(centres[:, 0]).to(torch.int64)
        py = torch.floor(centres[:, 1]).to(torch.int64)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        hit = torch.zeros(gs_ids.shape[0], dtype=torch.bool, device=gs_ids.device)
        if bool(inside.any()):
            hit[inside] = sky.to(device=gs_ids.device)[py[inside], px[inside]]
        seen.index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        hits.index_add_(0, gs_ids, hit.to(torch.float32))
        self.last_stats = {
            "observed": int(gs_ids.numel()),
            "sky_hits": int(hit.sum().item()),
        }

    def blocked_parent_mask(self, state: dict[str, Any], eligible: Any) -> Any:
        """True for eligible rows seen predominantly in sky (no counters: none)."""
        torch = __import__("torch")
        seen = state.get(SKY_SEEN_KEY)
        hits = state.get(SKY_HIT_KEY)
        if (
            not isinstance(seen, torch.Tensor)
            or not isinstance(hits, torch.Tensor)
            or len(seen) != len(eligible)
            or len(hits) != len(eligible)
        ):
            return torch.zeros_like(eligible)
        fraction = hits / seen.clamp_min(1.0)
        return eligible & (seen > 0) & (fraction >= self.min_sky_fraction)

    def record_blocked(self, state: dict[str, Any], blocked_count: int) -> int:
        """Accumulate the running total in strategy state; returns it."""
        state[SKY_BLOCKED_TOTAL_KEY] = int(state.get(SKY_BLOCKED_TOTAL_KEY, 0)) + int(
            blocked_count
        )
        return int(state[SKY_BLOCKED_TOTAL_KEY])

    def reset(self, state: dict[str, Any]) -> None:
        torch = __import__("torch")
        for key in (SKY_SEEN_KEY, SKY_HIT_KEY):
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                value.zero_()

    def state_dict(self) -> dict[str, Any]:
        return {
            "min_sky_fraction": self.min_sky_fraction,
            "observation": "projected_centre_inside_effective_sky_mask",
            "window": "since_last_refine_event",
        }
