"""Deterministic mask-aware run evaluation and self-contained quality report."""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.image_metrics import (
    create_lpips_model,
    masked_depth_metrics,
    masked_lpips,
    masked_psnr,
    masked_ssim,
)
from cloudstudio_3dgs.evaluation.splits import verify_split_manifest
from cloudstudio_3dgs.training.golden_eval import (
    MINIMUM_DEPTH_PREDICTION_COVERAGE,
    verify_full_evaluation_history,
    verify_golden_history,
)


def _safe_path(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"run artifact paths must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe run artifact path: {value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"run artifact path escapes its root: {value!r}")
    return resolved


def sign_run_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if "run_manifest_sha256" in manifest:
        raise ValueError("run manifest is already signed")
    output = dict(manifest)
    output["run_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(output)
    ).hexdigest()
    return output


def verify_run_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("run_manifest_sha256", ""))
    if not expected:
        raise ValueError("run manifest has no run_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("run_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"run manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    frames = manifest.get("frames", [])
    image_ids = [str(frame.get("image_id", "")) for frame in frames]
    if not frames or not all(image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError("run manifest contains invalid or duplicate image IDs")
    return actual


def verify_quality_report(report: dict[str, Any]) -> str:
    expected = str(report.get("quality_report_sha256", ""))
    if not expected:
        raise ValueError("quality report has no quality_report_sha256")
    unsigned = dict(report)
    unsigned.pop("quality_report_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"quality report SHA256 mismatch: expected {expected}, computed {actual}"
        )
    return actual


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing RGB artifact: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing mask artifact: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _load_dense_depth(path: Path, key: str = "depth") -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing rendered depth artifact: {path}")
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"{path} does not contain depth array {key!r}")
            value = archive[key]
    else:
        raise ValueError("rendered depth must be .npy or .npz")
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("rendered depth must be a dense 2D array")
    return value


def _finite_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return {
        "count": len(values),
        "finite_count": int(len(finite)),
        "infinite_count": len(values) - int(len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "p50": float(np.percentile(finite, 50)) if len(finite) else None,
        "p95": float(np.percentile(finite, 95)) if len(finite) else None,
        "min": float(np.min(finite)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None,
    }


def _golden_checkpoint_selection(
    run: dict[str, Any], run_root: Path
) -> tuple[dict[str, Any], list[str]]:
    """Bind formal quality output to the trainer's persisted model selection."""
    declaration = run.get("golden_evaluation")
    if declaration is None:
        return {"status": "NOT_RUN"}, ["golden_evaluation:NOT_RUN"]
    if not isinstance(declaration, dict):
        raise ValueError("run manifest golden evaluation declaration is invalid")
    history_value = declaration.get("history_path")
    if not isinstance(history_value, str):
        raise ValueError("run manifest golden evaluation has no history path")
    history_path = _safe_path(run_root, history_value)
    if not history_path.is_file():
        raise FileNotFoundError(f"missing golden evaluation history: {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, dict):
        raise ValueError("golden evaluation history must be an object")
    history_sha = verify_golden_history(history)
    if history_sha != declaration.get("history_sha256"):
        raise ValueError("golden evaluation history does not match run manifest")
    if declaration.get("configuration") != history.get("configuration"):
        raise ValueError("golden evaluation configuration does not match history")
    records = history["history"]
    if int(declaration.get("evaluation_count", -1)) != len(records):
        raise ValueError("golden evaluation count does not match history")
    if declaration.get("best") != history.get("best"):
        raise ValueError("golden evaluation best record does not match history")
    completed_steps = int(run.get("training", {}).get("completed_steps", 0))
    if completed_steps <= 0:
        raise ValueError("run manifest has no completed step count for golden evaluation")
    configuration = history["configuration"]
    expected_golden_steps = sorted(
        set(range(int(configuration["every"]), completed_steps + 1, int(configuration["every"])))
        | set(
            range(
                int(configuration["artifact_every"]),
                completed_steps + 1,
                int(configuration["artifact_every"]),
            )
        )
        | {completed_steps}
    )
    actual_golden_steps = [int(record["completed_steps"]) for record in records]
    if configuration["enabled"] and actual_golden_steps != expected_golden_steps:
        raise ValueError("golden evaluation steps do not match configured schedule")
    artifact_steps: list[int] = []
    for record in records:
        record_has_artifacts = False
        for frame in record.get("frames", []):
            artifacts = frame.get("artifacts")
            if artifacts is None:
                continue
            if not isinstance(artifacts, dict):
                raise ValueError("golden evaluation artifact declaration is invalid")
            record_has_artifacts = True
            for name in ("rendered", "reference", "mask"):
                path_value = artifacts.get(f"{name}_path")
                expected_sha = artifacts.get(f"{name}_sha256")
                if not isinstance(path_value, str) or not isinstance(expected_sha, str):
                    raise ValueError("golden evaluation artifact identity is incomplete")
                artifact_path = _safe_path(run_root, path_value)
                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"missing golden evaluation artifact: {artifact_path}"
                    )
                if _sha256_file(artifact_path) != expected_sha:
                    raise ValueError("golden evaluation artifact SHA256 mismatch")
        if record_has_artifacts:
            artifact_steps.append(int(record["completed_steps"]))
    if records and not artifact_steps:
        raise ValueError("golden evaluation history has no periodic render artifacts")
    expected_artifact_steps = sorted(
        set(
            range(
                int(configuration["artifact_every"]),
                completed_steps + 1,
                int(configuration["artifact_every"]),
            )
        )
        | {completed_steps}
    )
    if configuration["enabled"] and artifact_steps != expected_artifact_steps:
        raise ValueError("golden artifact steps do not match configured schedule")
    best = history.get("best")
    checkpoint_value = declaration.get("best_checkpoint_path")
    if best is None:
        if checkpoint_value is not None or declaration.get("best_checkpoint_sha256") is not None:
            raise ValueError("golden evaluation has a checkpoint without a best record")
    else:
        if not isinstance(checkpoint_value, str):
            raise ValueError("golden evaluation best checkpoint path is missing")
        checkpoint = _safe_path(run_root, checkpoint_value)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing golden best checkpoint: {checkpoint}")
        if _sha256_file(checkpoint) != declaration.get("best_checkpoint_sha256"):
            raise ValueError("golden best checkpoint SHA256 mismatch")
    return {
        "status": "VERIFIED",
        "history_sha256": history_sha,
        "evaluation_count": len(records),
        "best_completed_steps": None if best is None else best["completed_steps"],
        "best_psnr_db_mean": None if best is None else best["summary"]["psnr_db_mean"],
        "artifact_steps": artifact_steps,
    }, []


def _periodic_full_evaluation(
    run: dict[str, Any], run_root: Path
) -> tuple[dict[str, Any], list[str]]:
    declaration = run.get("periodic_full_evaluation")
    if declaration is None:
        return {"status": "NOT_RUN"}, ["periodic_full_evaluation:NOT_RUN"]
    if not isinstance(declaration, dict):
        raise ValueError("run manifest periodic full evaluation declaration is invalid")
    history_value = declaration.get("history_path")
    if not isinstance(history_value, str):
        raise ValueError("periodic full evaluation has no history path")
    history_path = _safe_path(run_root, history_value)
    if not history_path.is_file():
        raise FileNotFoundError(f"missing periodic full evaluation history: {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, dict):
        raise ValueError("periodic full evaluation history must be an object")
    history_sha = verify_full_evaluation_history(history)
    if history_sha != declaration.get("history_sha256"):
        raise ValueError("periodic full evaluation history does not match run manifest")
    records = history["history"]
    if int(declaration.get("evaluation_count", -1)) != len(records):
        raise ValueError("periodic full evaluation count does not match history")
    expected_latest = None if not records else records[-1]
    if declaration.get("latest") != expected_latest:
        raise ValueError("periodic full evaluation latest record does not match history")
    configuration = run.get("golden_evaluation", {}).get("configuration")
    completed_steps = int(run.get("training", {}).get("completed_steps", 0))
    if not isinstance(configuration, dict) or completed_steps <= 0:
        raise ValueError("periodic full evaluation has no signed schedule")
    expected_steps = (
        sorted(
            set(
                range(
                    int(configuration["full_every"]),
                    completed_steps + 1,
                    int(configuration["full_every"]),
                )
            )
            | {completed_steps}
        )
        if configuration.get("enabled")
        else []
    )
    if [int(record["completed_steps"]) for record in records] != expected_steps:
        raise ValueError("periodic full evaluation steps do not match configured schedule")
    return {
        "status": "VERIFIED" if records else "NOT_RUN",
        "history_sha256": history_sha,
        "evaluation_count": len(records),
        "latest_completed_steps": None
        if expected_latest is None
        else expected_latest["completed_steps"],
        "latest_psnr_db_mean": None
        if expected_latest is None
        else expected_latest["summary"]["psnr_db_mean"],
    }, ([] if records else ["periodic_full_evaluation:NOT_RUN"])


def _resource_metrics(run: dict[str, Any], run_root: Path) -> tuple[dict[str, Any], list[str]]:
    source = dict(run.get("training", {}))
    warnings: list[str] = []
    output: dict[str, Any] = {}
    for key in ("duration_seconds", "peak_vram_bytes", "gaussian_count"):
        value = source.get(key)
        if value is None:
            output[key] = {"status": "NOT_RUN", "value": None}
            warnings.append(f"training.{key}:NOT_RUN")
        else:
            number = float(value) if key == "duration_seconds" else int(value)
            if number < 0:
                raise ValueError(f"training.{key} must be non-negative")
            output[key] = {"status": "MEASURED", "value": number}
    model_path_value = source.get("model_path")
    if model_path_value is None:
        output["model_size_bytes"] = {"status": "NOT_RUN", "value": None}
        warnings.append("training.model_size_bytes:NOT_RUN")
    else:
        model_path = _safe_path(run_root, str(model_path_value))
        if not model_path.is_file():
            raise FileNotFoundError(f"missing model artifact: {model_path}")
        model_sha = _sha256_file(model_path)
        expected_model_sha = source.get("model_sha256")
        if expected_model_sha is not None and model_sha != expected_model_sha:
            raise ValueError("selected model SHA256 does not match run manifest")
        output["model_size_bytes"] = {
            "status": "MEASURED",
            "value": model_path.stat().st_size,
            "path": str(model_path_value),
            "sha256": model_sha,
        }
    return output, warnings


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_comparison(
    output_path: Path,
    reference_path: Path,
    rendered_path: Path,
    mask_path: Path,
) -> None:
    with Image.open(reference_path) as source:
        reference = source.convert("RGB")
    with Image.open(rendered_path) as source:
        rendered = source.convert("RGB")
    with Image.open(mask_path) as source:
        mask = source.convert("L")
    if reference.size != rendered.size or reference.size != mask.size:
        raise ValueError("golden-view reference, render, and mask sizes differ")
    maximum = 480
    reference = ImageOps.contain(reference, (maximum, maximum), Image.Resampling.LANCZOS)
    rendered = ImageOps.contain(rendered, reference.size, Image.Resampling.LANCZOS)
    mask = ImageOps.contain(mask, reference.size, Image.Resampling.NEAREST)
    masked_render = Image.composite(rendered, Image.new("RGB", rendered.size, (32, 32, 32)), mask)
    canvas = Image.new("RGB", (reference.width * 3, reference.height), (24, 24, 24))
    canvas.paste(reference, (0, 0))
    canvas.paste(rendered, (reference.width, 0))
    canvas.paste(masked_render, (reference.width * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        canvas.save(temporary, format="JPEG", quality=90, optimize=False)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _html_report(report: dict[str, Any]) -> str:
    image_metrics = report["summary"]["image_metrics"]
    rows = []
    for frame in report["frames"]:
        lpips_value = frame["image_metrics"]["lpips"]
        lpips_text = (
            f"{lpips_value['value']:.6f}"
            if lpips_value["status"] == "MEASURED"
            else lpips_value["status"]
        )
        psnr = frame["image_metrics"]["psnr_db"]
        psnr_text = "∞" if psnr is None else f"{psnr:.4f}"
        depth = frame["depth_metrics"]
        depth_text = (
            f"MAE {depth['mae_m']:.4f} m / RMSE {depth['rmse_m']:.4f} m"
            if depth["status"] == "MEASURED"
            else depth["status"]
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(frame['image_id'])}</td>"
            f"<td>{html.escape(frame['split'])}</td>"
            f"<td>{psnr_text}</td>"
            f"<td>{frame['image_metrics']['ssim']:.6f}</td>"
            f"<td>{lpips_text}</td>"
            f"<td>{depth_text}</td>"
            "</tr>"
        )
    warning_items = "".join(f"<li>{html.escape(value)}</li>" for value in report["warnings"])
    golden = "".join(
        f"<figure><img src=\"{html.escape(item['asset'])}\" alt=\"{html.escape(item['image_id'])}\">"
        f"<figcaption>{html.escape(item['image_id'])}: reference / render / masked render</figcaption></figure>"
        for item in report["golden_views"]
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>3DGS Quality Report</title>
<style>body{{font:14px system-ui;margin:2rem;color:#18212b}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4dc;padding:.45rem;text-align:left}}th{{background:#eef3f7}}.status{{font-size:1.3rem;font-weight:700}}figure{{margin:1rem 0}}img{{max-width:100%;border:1px solid #bbb}}code{{word-break:break-all}}</style></head>
<body><h1>3DGS Quality Report</h1><p class="status">状态：{html.escape(report['status'])}</p>
<p>Run：{html.escape(report['run_id'])}<br>Report SHA256：<code>{report['quality_report_sha256']}</code></p>
<h2>汇总</h2><ul><li>Masked PSNR mean：{image_metrics['psnr_db']['mean']}</li>
<li>Masked SSIM mean：{image_metrics['ssim']['mean']}</li><li>Masked LPIPS：{html.escape(str(image_metrics['lpips']))}</li></ul>
<h2>开放项/告警</h2><ul>{warning_items or '<li>无</li>'}</ul>
<h2>逐图指标</h2><table><thead><tr><th>Image</th><th>Split</th><th>PSNR dB</th><th>SSIM</th><th>LPIPS</th><th>Depth</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Golden views</h2>{golden or '<p>没有可用的 golden view 渲染。</p>'}</body></html>"""


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_quality_report(
    run: dict[str, Any],
    split_manifest: dict[str, Any],
    run_root: Path,
    output_dir: Path,
    *,
    run_lpips_metric: bool = False,
    lpips_net: str = "alex",
    lpips_device: str = "cpu",
    force: bool = False,
) -> dict[str, Any]:
    run_sha = verify_run_manifest(run)
    split_sha = verify_split_manifest(split_manifest)
    if run.get("split_manifest_sha256") != split_sha:
        raise ValueError("run manifest is bound to a different split manifest")
    if run.get("dataset_manifest_sha256") != split_manifest.get("dataset_manifest_sha256"):
        raise ValueError("run and split manifests are bound to different datasets")
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"quality output is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"quality output is not empty: {output_dir}; pass --force")
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "quality_assets"
    assets_dir.mkdir(exist_ok=True)

    split_by_image = {
        image_id: split
        for split, image_ids in split_manifest["splits"].items()
        for image_id in image_ids
    }
    validation_images = {str(value) for value in split_manifest["splits"]["val"]}
    evaluated_images = {str(frame["image_id"]) for frame in run["frames"]}
    if evaluated_images != validation_images:
        missing = sorted(validation_images - evaluated_images)
        non_validation = sorted(evaluated_images - validation_images)
        raise ValueError(
            "formal quality report must cover exactly the validation images; "
            f"missing={missing[:4]}, non_validation={non_validation[:4]}"
        )
    golden_image_ids = {
        image_id
        for item in split_manifest.get("golden_views", [])
        for image_id in item["image_ids"]
    }
    frame_reports: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    depth_mae: list[float] = []
    depth_rmse: list[float] = []
    depth_coverage: list[float] = []
    golden_reports: list[dict[str, str]] = []
    warnings: list[str] = []
    lpips_model = (
        create_lpips_model(net=lpips_net, device=lpips_device)
        if run_lpips_metric
        else None
    )

    for frame in sorted(run["frames"], key=lambda item: str(item["image_id"])):
        image_id = str(frame["image_id"])
        if image_id not in split_by_image:
            raise ValueError(f"run frame {image_id} is not present in the split manifest")
        split = str(frame.get("split", split_by_image[image_id]))
        if split != split_by_image[image_id]:
            raise ValueError(f"run frame {image_id} has a split assignment mismatch")
        reference_path = _safe_path(run_root, str(frame["reference_rgb_path"]))
        rendered_path = _safe_path(run_root, str(frame["rendered_rgb_path"]))
        mask_path = _safe_path(run_root, str(frame["combined_mask_path"]))
        reference = _load_rgb(reference_path)
        rendered = _load_rgb(rendered_path)
        mask = _load_mask(mask_path)
        if reference.shape != rendered.shape or mask.shape != reference.shape[:2]:
            raise ValueError(f"RGB/mask shape mismatch for {image_id}")
        psnr = masked_psnr(rendered, reference, mask)
        ssim = masked_ssim(rendered, reference, mask)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_report: dict[str, Any]
        if run_lpips_metric:
            lpips_value = masked_lpips(
                rendered,
                reference,
                mask,
                net=lpips_net,
                device=lpips_device,
                model=lpips_model,
            )
            lpips_values.append(lpips_value)
            lpips_report = {"status": "MEASURED", "value": lpips_value}
        else:
            lpips_report = {"status": "NOT_RUN", "value": None}

        depth_report: dict[str, Any] = {"status": "NOT_RUN"}
        rendered_depth_value = frame.get("rendered_depth_path")
        lidar_depth_value = frame.get("lidar_depth_cache_path")
        if (rendered_depth_value is None) != (lidar_depth_value is None):
            raise ValueError(f"frame {image_id} must provide both rendered and LiDAR depth")
        if rendered_depth_value is not None:
            semantics = frame.get("rendered_depth_semantics")
            if semantics != "euclidean_ray_range_m":
                raise ValueError(f"frame {image_id} rendered depth is not Euclidean ray range")
            rendered_depth = _load_dense_depth(
                _safe_path(run_root, str(rendered_depth_value)),
                str(frame.get("rendered_depth_key", "depth")),
            )
            sparse_target = load_sparse_depth(_safe_path(run_root, str(lidar_depth_value)))
            target_depth, confidence, target_valid = sparse_target.to_dense()
            if rendered_depth.shape != target_depth.shape or mask.shape != target_depth.shape:
                raise ValueError(f"depth/RGB spatial shape mismatch for {image_id}")
            metrics = masked_depth_metrics(
                rendered_depth,
                target_depth,
                mask & target_valid,
                confidence=confidence,
                minimum_prediction_coverage=(
                    MINIMUM_DEPTH_PREDICTION_COVERAGE
                ),
            )
            depth_report = {"status": "MEASURED", **metrics}
            depth_mae.append(float(metrics["mae_m"]))
            depth_rmse.append(float(metrics["rmse_m"]))
            depth_coverage.append(float(metrics["prediction_coverage_fraction"]))
        frame_report = {
            "image_id": image_id,
            "split": split,
            "valid_pixels": int(mask.sum()),
            "image_metrics": {
                "psnr_db": None if np.isinf(psnr) else psnr,
                "perfect_psnr": bool(np.isinf(psnr)),
                "ssim": ssim,
                "lpips": lpips_report,
            },
            "depth_metrics": depth_report,
        }
        frame_reports.append(frame_report)
        if image_id in golden_image_ids:
            asset_relative = f"quality_assets/{image_id}.jpg"
            _write_comparison(
                output_dir / asset_relative,
                reference_path,
                rendered_path,
                mask_path,
            )
            golden_reports.append({"image_id": image_id, "asset": asset_relative})

    golden_selection, golden_warnings = _golden_checkpoint_selection(
        run, Path(run_root)
    )
    warnings.extend(golden_warnings)
    periodic_full, periodic_full_warnings = _periodic_full_evaluation(
        run, Path(run_root)
    )
    warnings.extend(periodic_full_warnings)
    resources, resource_warnings = _resource_metrics(run, Path(run_root))
    warnings.extend(resource_warnings)
    if not run_lpips_metric:
        warnings.append("image_metrics.lpips:NOT_RUN")
    if not depth_mae:
        warnings.append("depth_metrics:NOT_RUN")
    missing_golden = sorted(golden_image_ids - {item["image_id"] for item in golden_reports})
    if missing_golden:
        warnings.append(f"golden_views_missing:{len(missing_golden)}")
    status = "COMPLETE" if not warnings else "PARTIAL"
    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "masked_quality_report_v1",
        "run_id": str(run["run_id"]),
        "run_manifest_sha256": run_sha,
        "split_manifest_sha256": split_sha,
        "dataset_manifest_sha256": run["dataset_manifest_sha256"],
        "status": status,
        "warnings": warnings,
        "resources": resources,
        "golden_checkpoint_selection": golden_selection,
        "periodic_full_evaluation": periodic_full,
        "frames": frame_reports,
        "golden_views": golden_reports,
        "summary": {
            "frame_count": len(frame_reports),
            "image_metrics": {
                "psnr_db": _finite_summary(psnr_values),
                "ssim": _finite_summary(ssim_values),
                "lpips": _finite_summary(lpips_values) if lpips_values else {"status": "NOT_RUN"},
            },
            "depth_metrics": {
                "mae_m": _finite_summary(depth_mae),
                "rmse_m": _finite_summary(depth_rmse),
                "prediction_coverage_fraction": _finite_summary(depth_coverage),
                "minimum_prediction_coverage_gate": (
                    MINIMUM_DEPTH_PREDICTION_COVERAGE
                ),
            }
            if depth_mae
            else {"status": "NOT_RUN"},
        },
    }
    report["quality_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    json_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    html_payload = _html_report(report).encode("utf-8")
    _atomic_write(output_dir / "quality_report.json", json_payload)
    _atomic_write(output_dir / "quality_report.html", html_payload)
    return report
