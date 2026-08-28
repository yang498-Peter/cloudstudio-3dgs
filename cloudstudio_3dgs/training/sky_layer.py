"""Deterministic far-field hemisphere augmentation for a warm-start checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest


SH_C0 = 0.28209479177387814


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SkyLayerConfig:
    count: int = 100_000
    radius_m: float = 100.0
    scale_m: float = 0.85
    opacity: float = 0.02
    rgb: tuple[float, float, float] = (0.45, 0.58, 0.78)
    min_world_z_direction: float = 0.0

    def validate(self) -> None:
        if self.count < 100:
            raise ValueError("sky layer count must be at least 100")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("sky layer radius_m must be finite and positive")
        if not math.isfinite(self.scale_m) or self.scale_m <= 0.0:
            raise ValueError("sky layer scale_m must be finite and positive")
        if not 0.0 < self.opacity < 1.0:
            raise ValueError("sky layer opacity must be within (0, 1)")
        if len(self.rgb) != 3 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.rgb
        ):
            raise ValueError("sky layer rgb must contain three finite values in [0, 1]")
        if not -1.0 <= self.min_world_z_direction < 1.0:
            raise ValueError("min_world_z_direction must be within [-1, 1)")


def camera_centre_from_manifest(manifest: dict[str, Any]) -> np.ndarray:
    verify_dataset_manifest(manifest)
    centres: list[np.ndarray] = []
    for image in manifest.get("images", []):
        matrix = np.asarray(image.get("c2w"), dtype=np.float64)
        if matrix.shape == (4, 4) and np.all(np.isfinite(matrix)):
            centres.append(matrix[:3, 3])
    if not centres:
        raise ValueError("dataset manifest has no finite c2w camera centres")
    return np.mean(np.stack(centres), axis=0)


def build_sky_layer(
    centre: np.ndarray,
    config: SkyLayerConfig,
) -> dict[str, np.ndarray]:
    """Build a uniform world-up spherical cap with deterministic Fibonacci samples."""

    config.validate()
    centre = np.asarray(centre, dtype=np.float64)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("sky layer centre must be a finite XYZ vector")

    indices = np.arange(config.count, dtype=np.float64)
    z = config.min_world_z_direction + (
        1.0 - config.min_world_z_direction
    ) * (indices + 0.5) / config.count
    azimuth = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), z], axis=1
    )
    means = centre[None, :] + config.radius_m * directions

    sh0_rgb = (np.asarray(config.rgb, dtype=np.float64) - 0.5) / SH_C0
    opacity_logit = math.log(config.opacity / (1.0 - config.opacity))
    quaternions = np.zeros((config.count, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    return {
        "means": means.astype(np.float32),
        "scales": np.full(
            (config.count, 3), math.log(config.scale_m), dtype=np.float32
        ),
        "quats": quaternions,
        "opacities": np.full((config.count,), opacity_logit, dtype=np.float32),
        "sh0": np.broadcast_to(
            sh0_rgb.astype(np.float32), (config.count, 1, 3)
        ).copy(),
    }


def augment_checkpoint_with_sky(
    source_checkpoint: Path,
    dataset_manifest_path: Path,
    output_checkpoint: Path,
    report_path: Path,
    config: SkyLayerConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Append a far-field cap and emit a model-only warm-start checkpoint."""

    import torch

    source_checkpoint = Path(source_checkpoint).resolve()
    dataset_manifest_path = Path(dataset_manifest_path).resolve()
    output_checkpoint = Path(output_checkpoint).resolve()
    report_path = Path(report_path).resolve()
    if output_checkpoint == source_checkpoint:
        raise ValueError("sky augmentation output must differ from the source checkpoint")
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {source_checkpoint}")
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is missing: {dataset_manifest_path}")
    for path in (output_checkpoint, report_path):
        if path.exists() and not force:
            raise FileExistsError(f"refusing to replace existing output: {path}")

    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = verify_dataset_manifest(manifest)
    centre = camera_centre_from_manifest(manifest)
    sky = build_sky_layer(centre, config)

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("schema_version") != 1 or not isinstance(source.get("identity"), dict):
        raise ValueError("source is not an identity-bound schema-1 checkpoint")
    source_identity = source["identity"]
    nested_identity = source_identity.get("source_identity", {})
    source_dataset_sha256 = source_identity.get(
        "dataset_manifest_sha256",
        nested_identity.get("dataset_manifest_sha256")
        if isinstance(nested_identity, dict)
        else None,
    )
    if source_dataset_sha256 != dataset_sha256:
        raise ValueError("source checkpoint is bound to a different dataset manifest")
    params = source.get("params")
    required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    if not isinstance(params, dict) or set(params) != required:
        raise ValueError("source checkpoint parameters do not match the SH trainer schema")
    source_count = int(len(params["means"]))
    if any(len(value) != source_count for value in params.values()):
        raise ValueError("source checkpoint parameter row counts differ")
    if tuple(params["sh0"].shape[1:]) != (1, 3) or params["shN"].ndim != 3:
        raise ValueError("source checkpoint spherical harmonics have invalid shapes")

    derived_params: dict[str, Any] = {}
    for name, value in params.items():
        if name == "shN":
            addition = torch.zeros(
                (config.count, value.shape[1], value.shape[2]), dtype=value.dtype
            )
        else:
            addition = torch.from_numpy(sky[name]).to(dtype=value.dtype)
        derived_params[name] = torch.cat([value.detach().cpu(), addition], dim=0)

    layer = {
        "schema_version": 1,
        "kind": "world_up_fibonacci_spherical_cap",
        "source_gaussian_count": source_count,
        "sky_gaussian_start": source_count,
        "sky_gaussian_count": config.count,
        "total_gaussian_count": source_count + config.count,
        "centre_m": centre.tolist(),
        "radius_m": float(config.radius_m),
        "scale_m": float(config.scale_m),
        "opacity": float(config.opacity),
        "rgb": [float(value) for value in config.rgb],
        "min_world_z_direction": float(config.min_world_z_direction),
    }
    payload = {
        "schema_version": 1,
        "step": int(source.get("step", 0)),
        "identity": source["identity"],
        "params": derived_params,
        "auxiliary_params": {
            name: value.detach().cpu()
            for name, value in source.get("auxiliary_params", {}).items()
        },
        "derived_warm_start_only": True,
        "source_checkpoint_sha256": _sha256_file(source_checkpoint),
        "sky_layer": layer,
    }

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_name(output_checkpoint.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_checkpoint)
    finally:
        temporary.unlink(missing_ok=True)

    report = {
        **layer,
        "dataset_manifest_sha256": dataset_sha256,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": _sha256_file(output_checkpoint),
        "output_checkpoint_bytes": output_checkpoint.stat().st_size,
        "resume_supported": False,
        "warm_start_supported": True,
    }
    report["sky_layer_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    try:
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_report, report_path)
    finally:
        temporary_report.unlink(missing_ok=True)
    return report
