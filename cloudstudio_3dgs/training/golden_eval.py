"""Deterministic in-training evaluation on signed golden validation views."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.image_metrics import (
    masked_depth_metrics,
    masked_psnr,
    masked_ssim,
)


@dataclass(frozen=True)
class GoldenEvaluationConfig:
    """Small, repeatable validation set used to choose a training checkpoint."""

    enabled: bool = True
    every: int = 1_000
    min_psnr_improvement_db: float = 0.001

    def validate(self) -> None:
        if self.every <= 0:
            raise ValueError("golden evaluation interval must be positive")
        if self.min_psnr_improvement_db < 0.0:
            raise ValueError("golden PSNR improvement threshold must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "every": self.every,
            "selection_metric": "masked_rgb_psnr_db_mean",
            "min_psnr_improvement_db": self.min_psnr_improvement_db,
        }


def golden_image_ids(split_manifest: dict[str, Any]) -> tuple[str, ...]:
    """Read golden views in their signed Rig Frame and camera ordering."""
    validation = {str(value) for value in split_manifest["splits"]["val"]}
    result: list[str] = []
    for frame in split_manifest.get("golden_views", []):
        for image_id in frame.get("image_ids", []):
            image_id = str(image_id)
            if image_id not in validation:
                raise ValueError(f"golden image {image_id} is not in the validation split")
            if image_id in result:
                raise ValueError(f"golden image {image_id} appears more than once")
            result.append(image_id)
    if not result:
        raise ValueError("split manifest has no golden validation images")
    return tuple(result)


def evaluate_golden_views(
    *,
    backend: Any,
    params: Any,
    dataset: Any,
    split_manifest: dict[str, Any],
    completed_steps: int,
    background_rgb: Any | None = None,
) -> dict[str, Any]:
    """Measure PSNR/SSIM and optional range error without writing image assets.

    The selection score is intentionally RGB-only: all runs have an RGB mask,
    whereas sparse LiDAR availability varies by capture.  The depth result is
    nevertheless recorded so a visually sharper but geometrically broken
    checkpoint is visible before formal full-validation acceptance.
    """
    if completed_steps <= 0:
        raise ValueError("golden evaluation requires at least one completed step")
    ordered_ids = golden_image_ids(split_manifest)
    index_by_id = {str(image_id): index for index, image_id in enumerate(dataset.image_ids)}
    missing = [image_id for image_id in ordered_ids if image_id not in index_by_id]
    if missing:
        raise ValueError(f"golden validation images are absent from dataset: {missing[:4]}")

    torch = backend.torch
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    depth_mae_values: list[float] = []
    frames: list[dict[str, Any]] = []
    with torch.no_grad():
        for image_id in ordered_ids:
            sample = dataset[index_by_id[image_id]]
            render_options: dict[str, Any] = {
                "with_range": sample.depth_range_m is not None,
            }
            if background_rgb is not None:
                render_options["background_rgb"] = background_rgb
            rendered, rendered_range, _, _ = backend.render(params, sample, **render_options)
            prediction = rendered.detach().clamp(0.0, 1.0).cpu().numpy()
            reference = np.asarray(sample.image, dtype=np.float32) / 255.0
            psnr = masked_psnr(prediction, reference, sample.rgb_mask)
            ssim = masked_ssim(prediction, reference, sample.rgb_mask)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            frame: dict[str, Any] = {
                "image_id": image_id,
                "valid_rgb_pixels": int(np.asarray(sample.rgb_mask, dtype=bool).sum()),
                "psnr_db": None if np.isinf(psnr) else float(psnr),
                "perfect_psnr": bool(np.isinf(psnr)),
                "ssim": float(ssim),
                "depth": {"status": "NOT_RUN"},
            }
            if sample.depth_range_m is not None:
                assert sample.depth_mask is not None and sample.depth_confidence is not None
                if rendered_range is None:
                    raise ValueError(f"golden render has no range output for {image_id}")
                depth = masked_depth_metrics(
                    rendered_range.detach().cpu().numpy(),
                    sample.depth_range_m,
                    sample.depth_mask,
                    confidence=sample.depth_confidence,
                )
                depth_mae_values.append(float(depth["mae_m"]))
                frame["depth"] = {"status": "MEASURED", **depth}
            frames.append(frame)

    finite_psnr = [value for value in psnr_values if np.isfinite(value)]
    result: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "golden_validation_v1",
        "completed_steps": int(completed_steps),
        "split_manifest_sha256": str(split_manifest["split_manifest_sha256"]),
        "image_ids": list(ordered_ids),
        "frames": frames,
        "summary": {
            "selection_metric": "masked_rgb_psnr_db_mean",
            "psnr_db_mean": None if not finite_psnr else float(np.mean(finite_psnr)),
            "perfect_psnr_frame_count": int(len(psnr_values) - len(finite_psnr)),
            "ssim_mean": float(np.mean(ssim_values)),
            "depth_mae_m_mean": None
            if not depth_mae_values
            else float(np.mean(depth_mae_values)),
        },
    }
    result["golden_evaluation_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def is_golden_improvement(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    min_psnr_improvement_db: float,
) -> bool:
    """Return whether a finite candidate clears the configured promotion bar."""
    candidate_score = candidate["summary"]["psnr_db_mean"]
    if candidate_score is None:
        return False
    if best is None or best["summary"]["psnr_db_mean"] is None:
        return True
    return float(candidate_score) >= float(best["summary"]["psnr_db_mean"]) + min_psnr_improvement_db
