"""Deterministic in-training evaluation on signed golden validation views."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.backend import rendered_range_to_euclidean
from cloudstudio_3dgs.evaluation.image_metrics import (
    masked_depth_metrics,
    masked_psnr,
    masked_ssim,
)


MINIMUM_DEPTH_PREDICTION_COVERAGE = 0.9


@dataclass(frozen=True)
class GoldenEvaluationConfig:
    """Small, repeatable validation set used to choose a training checkpoint."""

    enabled: bool = True
    every: int = 1_000
    full_every: int = 4_000
    artifact_every: int = 1_000
    min_psnr_improvement_db: float = 0.001
    max_depth_regression_m: float | None = None
    max_floater_growth_ratio: float | None = None
    max_floater_count: int | None = None

    def validate(self) -> None:
        if self.every <= 0 or self.full_every <= 0 or self.artifact_every <= 0:
            raise ValueError("golden/full/artifact evaluation intervals must be positive")
        if self.min_psnr_improvement_db < 0.0:
            raise ValueError("golden PSNR improvement threshold must be non-negative")
        if self.max_depth_regression_m is not None and (
            not math.isfinite(self.max_depth_regression_m)
            or self.max_depth_regression_m < 0.0
        ):
            raise ValueError("golden depth regression guard must be finite and non-negative")
        if self.max_floater_growth_ratio is not None and (
            not math.isfinite(self.max_floater_growth_ratio)
            or self.max_floater_growth_ratio < 1.0
        ):
            raise ValueError("golden floater guard ratio must be finite and at least one")
        if self.max_floater_count is not None and self.max_floater_count < 0:
            raise ValueError("golden floater budget must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "every": self.every,
            "full_every": self.full_every,
            "artifact_every": self.artifact_every,
            "selection_metric": "masked_rgb_psnr_db_mean",
            "min_psnr_improvement_db": self.min_psnr_improvement_db,
        }
        # Preserve the existing PSNR-only contract unless the geometry guard is
        # explicitly requested.
        if self.max_depth_regression_m is not None:
            result["max_depth_regression_m"] = self.max_depth_regression_m
        if self.max_floater_growth_ratio is not None:
            result["max_floater_growth_ratio"] = self.max_floater_growth_ratio
        if self.max_floater_count is not None:
            result["max_floater_count"] = self.max_floater_count
        return result


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_views(
    *,
    backend: Any,
    params: Any,
    dataset: Any,
    split_manifest: dict[str, Any],
    ordered_ids: tuple[str, ...],
    completed_steps: int,
    evaluation_kind: str,
    background_rgb: Any | None = None,
    artifact_output_dir: Path | None = None,
) -> dict[str, Any]:
    if evaluation_kind not in {"golden", "full"}:
        raise ValueError("evaluation_kind must be golden or full")
    if completed_steps <= 0:
        raise ValueError("periodic evaluation requires at least one completed step")
    index_by_id = {str(image_id): index for index, image_id in enumerate(dataset.image_ids)}
    missing = [image_id for image_id in ordered_ids if image_id not in index_by_id]
    if missing:
        raise ValueError(f"golden validation images are absent from dataset: {missing[:4]}")

    torch = backend.torch
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    depth_mae_values: list[float] = []
    depth_coverage_values: list[float] = []
    frames: list[dict[str, Any]] = []
    with torch.no_grad():
        for image_id in ordered_ids:
            sample = dataset[index_by_id[image_id]]
            render_options: dict[str, Any] = {
                "with_range": sample.depth_range_m is not None,
            }
            if background_rgb is not None:
                render_options["background_rgb"] = background_rgb
            rendered, rendered_range, _, render_info = backend.render(
                params, sample, **render_options
            )
            rendered_range = rendered_range_to_euclidean(
                backend.torch, rendered_range, sample, render_info
            )
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
            if artifact_output_dir is not None:
                if any(token in image_id for token in ("/", "\\", "..")):
                    raise ValueError(f"unsafe golden artifact image ID: {image_id!r}")
                artifact_dir = (
                    Path(artifact_output_dir)
                    / "evaluation"
                    / "periodic_golden"
                    / f"step_{completed_steps:08d}"
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                render_path = artifact_dir / f"{image_id}_rendered.png"
                reference_path = artifact_dir / f"{image_id}_reference.png"
                mask_path = artifact_dir / f"{image_id}_mask.png"
                rendered_u8 = (
                    rendered.detach()
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                Image.fromarray(rendered_u8).save(render_path, format="PNG", optimize=False)
                Image.fromarray(np.asarray(sample.image, dtype=np.uint8)).save(
                    reference_path, format="PNG", optimize=False
                )
                Image.fromarray(np.asarray(sample.rgb_mask, dtype=np.uint8) * 255).save(
                    mask_path, format="PNG", optimize=False
                )
                frame["artifacts"] = {
                    "rendered_path": render_path.relative_to(artifact_output_dir).as_posix(),
                    "rendered_sha256": _sha256_file(render_path),
                    "reference_path": reference_path.relative_to(artifact_output_dir).as_posix(),
                    "reference_sha256": _sha256_file(reference_path),
                    "mask_path": mask_path.relative_to(artifact_output_dir).as_posix(),
                    "mask_sha256": _sha256_file(mask_path),
                }
            if sample.depth_range_m is not None:
                assert sample.depth_mask is not None and sample.depth_confidence is not None
                if rendered_range is None:
                    raise ValueError(f"golden render has no range output for {image_id}")
                try:
                    depth = masked_depth_metrics(
                        rendered_range.detach().cpu().numpy(),
                        sample.depth_range_m,
                        sample.depth_mask,
                        confidence=sample.depth_confidence,
                        minimum_prediction_coverage=(
                            MINIMUM_DEPTH_PREDICTION_COVERAGE
                        ),
                    )
                except ValueError as error:
                    # Early/mid-training renders legitimately miss coverage at
                    # some supervised pixels; selection is RGB-only, so record
                    # the gap instead of killing the training run. Formal
                    # full-validation acceptance keeps its fail-closed check.
                    frame["depth"] = {"status": "UNMEASURABLE", "reason": str(error)}
                else:
                    depth_mae_values.append(float(depth["mae_m"]))
                    depth_coverage_values.append(
                        float(depth["prediction_coverage_fraction"])
                    )
                    frame["depth"] = {"status": "MEASURED", **depth}
            frames.append(frame)

    finite_psnr = [value for value in psnr_values if np.isfinite(value)]
    result: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": f"{evaluation_kind}_validation_v3",
        "evaluation_kind": evaluation_kind,
        "completed_steps": int(completed_steps),
        "split_manifest_sha256": str(split_manifest["split_manifest_sha256"]),
        "image_ids": list(ordered_ids),
        "frames": frames,
        "summary": {
            "selection_metric": "masked_rgb_psnr_db_mean",
            "psnr_db_mean": None if not finite_psnr else float(np.mean(finite_psnr)),
            "psnr_db_p10": None
            if not finite_psnr
            else float(np.percentile(finite_psnr, 10)),
            "perfect_psnr_frame_count": int(len(psnr_values) - len(finite_psnr)),
            "ssim_mean": float(np.mean(ssim_values)),
            "depth_mae_m_mean": None
            if not depth_mae_values
            else float(np.mean(depth_mae_values)),
            "depth_prediction_coverage_fraction_min": None
            if not depth_coverage_values
            else float(np.min(depth_coverage_values)),
            "depth_prediction_coverage_fraction_mean": None
            if not depth_coverage_values
            else float(np.mean(depth_coverage_values)),
            "minimum_depth_prediction_coverage_gate": (
                MINIMUM_DEPTH_PREDICTION_COVERAGE
            ),
        },
    }
    signature_key = (
        "golden_evaluation_sha256"
        if evaluation_kind == "golden"
        else "full_evaluation_sha256"
    )
    result[signature_key] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def evaluate_golden_views(
    *,
    backend: Any,
    params: Any,
    dataset: Any,
    split_manifest: dict[str, Any],
    completed_steps: int,
    background_rgb: Any | None = None,
    artifact_output_dir: Path | None = None,
    geometry_tree: Any | None = None,
) -> dict[str, Any]:
    """Evaluate signed golden views and optionally persist visual evidence."""
    report = _evaluate_views(
        backend=backend,
        params=params,
        dataset=dataset,
        split_manifest=split_manifest,
        ordered_ids=golden_image_ids(split_manifest),
        completed_steps=completed_steps,
        evaluation_kind="golden",
        background_rgb=background_rgb,
        artifact_output_dir=artifact_output_dir,
    )
    if geometry_tree is not None:
        report["summary"]["floater_count"] = count_floaters(params, geometry_tree)
        report["summary"]["floater_threshold_m"] = FLOATER_DISTANCE_THRESHOLD_M
        report.pop("golden_evaluation_sha256", None)
        report["golden_evaluation_sha256"] = hashlib.sha256(
            canonical_json_bytes(report)
        ).hexdigest()
    return report


def evaluate_full_validation(
    *,
    backend: Any,
    params: Any,
    dataset: Any,
    split_manifest: dict[str, Any],
    completed_steps: int,
    background_rgb: Any | None = None,
) -> dict[str, Any]:
    """Evaluate every validation image at configured intermediate steps."""
    expected = tuple(str(value) for value in split_manifest["splits"]["val"])
    if tuple(str(value) for value in dataset.image_ids) != expected:
        raise ValueError("full validation dataset order differs from split manifest")
    return _evaluate_views(
        backend=backend,
        params=params,
        dataset=dataset,
        split_manifest=split_manifest,
        ordered_ids=expected,
        completed_steps=completed_steps,
        evaluation_kind="full",
        background_rgb=background_rgb,
    )


FLOATER_DISTANCE_THRESHOLD_M = 0.3
FLOATER_MINIMUM_OPACITY = 0.1


def count_floaters(params: Any, geometry_tree: Any) -> int:
    """Count visible gaussians that sit far from the measured LiDAR geometry.

    A "floater" is a gaussian the renderer can actually see (opacity above
    ``FLOATER_MINIMUM_OPACITY``) whose center is further than
    ``FLOATER_DISTANCE_THRESHOLD_M`` from any measured LiDAR point. Long
    training accumulates these faster than the appearance metrics notice: on
    this scene the count grew 326 -> 2376 between 8k and 30k steps at
    unchanged PSNR-only selection, which is what the guard exists to catch.
    """
    import torch

    with torch.no_grad():
        opacity = torch.sigmoid(params["opacities"].detach().reshape(-1))
        visible = opacity > FLOATER_MINIMUM_OPACITY
        if not bool(visible.any()):
            return 0
        centers = params["means"].detach()[visible].cpu().numpy()
    distances, _ = geometry_tree.query(centers, k=1, workers=-1)
    return int((distances > FLOATER_DISTANCE_THRESHOLD_M).sum())


def is_golden_improvement(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    min_psnr_improvement_db: float,
    max_depth_regression_m: float | None = None,
    max_floater_growth_ratio: float | None = None,
    max_floater_count: int | None = None,
) -> bool:
    """Return whether a candidate clears appearance and optional geometry bars.

    ``max_floater_count`` is an absolute geometry budget: "give me the best
    appearance among models with at most N floaters". Prefer it over
    ``max_floater_growth_ratio`` for long runs - the ratio form anchors on
    whatever the first evaluation happened to measure, and early checkpoints
    have almost no floaters simply because densification has not run yet,
    which locks selection onto a barely-trained model.
    """
    candidate_score = candidate["summary"]["psnr_db_mean"]
    if candidate_score is None:
        return False
    if max_floater_count is not None:
        candidate_floaters = candidate["summary"].get("floater_count")
        if candidate_floaters is None:
            return False
        if int(candidate_floaters) > int(max_floater_count):
            return False
    candidate_depth = None
    if max_depth_regression_m is not None:
        depth_frames = [
            frame.get("depth", {})
            for frame in candidate.get("frames", [])
            if frame.get("depth", {}).get("status") != "NOT_RUN"
        ]
        if not depth_frames or any(
            frame.get("status") != "MEASURED" for frame in depth_frames
        ):
            return False
        value = candidate["summary"].get("depth_mae_m_mean")
        if value is None or not math.isfinite(float(value)):
            return False
        candidate_depth = float(value)
    if best is None or best["summary"]["psnr_db_mean"] is None:
        return True
    if float(candidate_score) < (
        float(best["summary"]["psnr_db_mean"]) + min_psnr_improvement_db
    ):
        return False
    if max_floater_growth_ratio is not None:
        candidate_floaters = candidate["summary"].get("floater_count")
        best_floaters = best["summary"].get("floater_count")
        if candidate_floaters is None or best_floaters is None:
            return False
        # A candidate may not buy its PSNR with geometry: the floater count is
        # allowed to grow only within the configured ratio of the incumbent.
        allowance = max(1.0, float(best_floaters) * float(max_floater_growth_ratio))
        if float(candidate_floaters) > allowance:
            return False
    if max_depth_regression_m is None:
        return True
    best_depth_frames = [
        frame.get("depth", {})
        for frame in best.get("frames", [])
        if frame.get("depth", {}).get("status") != "NOT_RUN"
    ]
    if not best_depth_frames or any(
        frame.get("status") != "MEASURED" for frame in best_depth_frames
    ):
        return False
    best_depth_value = best["summary"].get("depth_mae_m_mean")
    if best_depth_value is None or not math.isfinite(float(best_depth_value)):
        return False
    assert candidate_depth is not None
    return candidate_depth <= float(best_depth_value) + max_depth_regression_m


def verify_golden_history(payload: dict[str, Any]) -> str:
    """Verify the persisted checkpoint-selection evidence before quality use."""
    expected = str(payload.get("golden_history_sha256", ""))
    if not expected:
        raise ValueError("golden history has no SHA256")
    unsigned = dict(payload)
    unsigned.pop("golden_history_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"golden history SHA256 mismatch: expected {expected}, computed {actual}"
        )
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported golden history schema")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("golden history has no configuration")
    config = GoldenEvaluationConfig(
        enabled=bool(configuration.get("enabled")),
        every=int(configuration.get("every", 0)),
        full_every=int(configuration.get("full_every", 0)),
        artifact_every=int(configuration.get("artifact_every", 0)),
        min_psnr_improvement_db=float(
            configuration.get("min_psnr_improvement_db", -1.0)
        ),
        max_depth_regression_m=(
            None
            if configuration.get("max_depth_regression_m") is None
            else float(configuration["max_depth_regression_m"])
        ),
        max_floater_growth_ratio=(
            None
            if configuration.get("max_floater_growth_ratio") is None
            else float(configuration["max_floater_growth_ratio"])
        ),
        max_floater_count=(
            None
            if configuration.get("max_floater_count") is None
            else int(configuration["max_floater_count"])
        ),
    )
    config.validate()
    if configuration != config.to_dict():
        raise ValueError("golden history configuration is not canonical")
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("golden history records are invalid")
    best: dict[str, Any] | None = None
    previous_step = 0
    for record in history:
        if not isinstance(record, dict):
            raise ValueError("golden history contains a non-object record")
        record_sha = str(record.get("golden_evaluation_sha256", ""))
        unsigned_record = dict(record)
        unsigned_record.pop("golden_evaluation_sha256", None)
        if not record_sha or hashlib.sha256(canonical_json_bytes(unsigned_record)).hexdigest() != record_sha:
            raise ValueError("golden history record SHA256 mismatch")
        step = int(record.get("completed_steps", 0))
        if step <= previous_step:
            raise ValueError("golden history steps are not strictly increasing")
        previous_step = step
        if is_golden_improvement(
            record,
            best,
            min_psnr_improvement_db=config.min_psnr_improvement_db,
            max_depth_regression_m=config.max_depth_regression_m,
            max_floater_growth_ratio=config.max_floater_growth_ratio,
            max_floater_count=config.max_floater_count,
        ):
            best = record
    if not config.enabled and history:
        raise ValueError("disabled golden evaluation has history records")
    recorded_best = payload.get("best")
    if recorded_best != best:
        raise ValueError("golden history best record does not match promotion rule")
    return actual


def verify_full_evaluation_history(payload: dict[str, Any]) -> str:
    """Verify periodic full-validation records and their strict step order."""
    expected = str(payload.get("full_evaluation_history_sha256", ""))
    if not expected:
        raise ValueError("full evaluation history has no SHA256")
    unsigned = dict(payload)
    unsigned.pop("full_evaluation_history_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            "full evaluation history SHA256 mismatch: "
            f"expected {expected}, computed {actual}"
        )
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported full evaluation history schema")
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("full evaluation history records are invalid")
    previous_step = 0
    for record in history:
        if not isinstance(record, dict) or record.get("evaluation_kind") != "full":
            raise ValueError("full evaluation history contains an invalid record")
        record_sha = str(record.get("full_evaluation_sha256", ""))
        unsigned_record = dict(record)
        unsigned_record.pop("full_evaluation_sha256", None)
        if not record_sha or hashlib.sha256(canonical_json_bytes(unsigned_record)).hexdigest() != record_sha:
            raise ValueError("full evaluation record SHA256 mismatch")
        step = int(record.get("completed_steps", 0))
        if step <= previous_step:
            raise ValueError("full evaluation history steps are not strictly increasing")
        previous_step = step
    return actual
