"""Auditable independent sky/background evidence and initialization.

The vendor loss uses invalid monocular depth as a background proxy.  Official
Depth Anything V2 does not preserve that proprietary invalid-depth behaviour,
so CloudStudio uses an explicit compatibility policy: accepted Face4 pixels,
valid LiDAR-aligned DA2, a far aligned range, and a world-up ray.  The policy
is recorded in the signed manifest and never silently labels an entire view.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mono_depth import verify_mono_depth_manifest
from cloudstudio_3dgs.training.face_dataset import verify_face_manifest


SKY_EVIDENCE_SCHEMA_VERSION = 1
SKY_EVIDENCE_KIND = "independent_far_background_evidence_v1"
SKY_INITIALIZATION_KIND = "independent_sky_gaussian_initialization_v1"
SH_C0 = 0.28209479177387814


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if "\\" in value or pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"artifact path escapes its root: {value!r}")
    return resolved


@dataclass(frozen=True)
class SkyEvidenceConfig:
    far_aligned_range_m: float = 30.0
    minimum_world_z_direction: float = 0.0
    minimum_candidate_fraction_per_view: float = 0.01
    azimuth_bins: int = 72
    elevation_bins: int = 36
    maximum_color_samples_per_view: int = 100_000

    def validate(self) -> None:
        if not math.isfinite(self.far_aligned_range_m) or self.far_aligned_range_m <= 0.0:
            raise ValueError("far aligned range must be finite and positive")
        if not -1.0 <= self.minimum_world_z_direction < 1.0:
            raise ValueError("minimum world-Z direction must be within [-1, 1)")
        if not 0.0 <= self.minimum_candidate_fraction_per_view < 1.0:
            raise ValueError("minimum candidate fraction must be within [0, 1)")
        if self.azimuth_bins < 4 or self.elevation_bins < 2:
            raise ValueError("directional color grid is too small")
        if self.maximum_color_samples_per_view <= 0:
            raise ValueError("maximum color samples per view must be positive")


@dataclass(frozen=True)
class IndependentSkyConfig:
    count: int = 100_000
    radius_m: float = 100.0
    scale_m: float = 0.85
    opacity: float = 0.02
    minimum_world_z_direction: float = 0.0
    sh_degree: int = 1

    def validate(self) -> None:
        if self.count < 100:
            raise ValueError("independent sky requires at least 100 Gaussians")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("sky radius must be finite and positive")
        if not math.isfinite(self.scale_m) or self.scale_m <= 0.0:
            raise ValueError("sky scale must be finite and positive")
        if not 0.0 < self.opacity < 1.0:
            raise ValueError("sky opacity must be within (0, 1)")
        if not -1.0 <= self.minimum_world_z_direction < 1.0:
            raise ValueError("minimum world-Z direction must be within [-1, 1)")
        if self.sh_degree != 1:
            raise ValueError("independent sky compatibility model currently requires SH degree 1")


def _face_lookup(face_manifest: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    images: dict[tuple[str, str], dict[str, Any]] = {}
    faces: dict[tuple[str, str], dict[str, Any]] = {}
    specs = {
        (str(camera_id), str(face["face_id"])): face
        for camera_id, camera in face_manifest["cameras"].items()
        for face in camera["faces"]
    }
    for image in face_manifest["images"]:
        for face in image["faces"]:
            key = (str(image["image_id"]), str(face["face_id"]))
            if key in images:
                raise ValueError(f"duplicate Face4 sample: {key}")
            images[key] = image
            faces[key] = face
    return images, faces


def _parent_rays(
    spec: dict[str, Any],
    output_shape: tuple[int, int],
) -> np.ndarray:
    height, width = output_shape
    source_width = int(spec["width"])
    source_height = int(spec["height"])
    K = np.asarray(spec["K_face"], dtype=np.float32)
    R_face = np.asarray(spec["R_face"], dtype=np.float32)
    x = (np.arange(width, dtype=np.float32) + 0.5) * source_width / width - 0.5
    y = (np.arange(height, dtype=np.float32) + 0.5) * source_height / height - 0.5
    xx, yy = np.meshgrid(x, y)
    rays = np.stack(
        [
            (xx - K[0, 2]) / K[0, 0],
            (yy - K[1, 2]) / K[1, 1],
            np.ones_like(xx),
        ],
        axis=-1,
    )
    rays = rays @ R_face.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.astype(np.float32)


def _direction_bins(
    directions: np.ndarray,
    config: SkyEvidenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    elevation = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    azimuth_index = np.floor(
        (azimuth + math.pi) / (2.0 * math.pi) * config.azimuth_bins
    ).astype(np.int64) % config.azimuth_bins
    elevation_index = np.floor(
        (elevation + math.pi / 2.0) / math.pi * config.elevation_bins
    ).astype(np.int64)
    elevation_index = np.clip(elevation_index, 0, config.elevation_bins - 1)
    return azimuth_index, elevation_index


def build_sky_evidence_cache(
    face_manifest_path: Path,
    face_cache_root: Path,
    mono_manifest_path: Path,
    mono_cache_root: Path,
    output_root: Path,
    *,
    config: SkyEvidenceConfig = SkyEvidenceConfig(),
    force: bool = False,
) -> dict[str, Any]:
    """Build per-Face4 sky/background masks plus a signed color summary."""

    config.validate()
    face_manifest_path = Path(face_manifest_path).resolve()
    mono_manifest_path = Path(mono_manifest_path).resolve()
    face_cache_root = Path(face_cache_root).resolve()
    mono_cache_root = Path(mono_cache_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "sky_evidence_manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"refusing to replace existing sky evidence: {manifest_path}")
    face_manifest = json.loads(face_manifest_path.read_text(encoding="utf-8"))
    mono_manifest = json.loads(mono_manifest_path.read_text(encoding="utf-8"))
    face_sha = verify_face_manifest(face_manifest)
    mono_sha = verify_mono_depth_manifest(mono_manifest)
    if mono_manifest.get("source_face_manifest_sha256") != face_sha:
        raise ValueError("mono depth and Face4 manifests have different identities")
    image_lookup, face_lookup = _face_lookup(face_manifest)
    records = mono_manifest.get("records", [])
    if not records or len(records) != int(mono_manifest.get("expected_face_count", -1)):
        raise ValueError("mono depth manifest is incomplete")
    mask_directory = output_root / "masks"
    mask_directory.mkdir(parents=True, exist_ok=True)

    color_sum = np.zeros((config.elevation_bins, config.azimuth_bins, 3), dtype=np.float64)
    color_count = np.zeros((config.elevation_bins, config.azimuth_bins), dtype=np.int64)
    ray_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
    output_records: list[dict[str, Any]] = []
    total_pixels = 0
    total_candidates = 0
    accepted_views = 0
    invalid_alignment_views = 0
    below_fraction_views = 0

    for record in records:
        key = (str(record["image_id"]), str(record["face_id"]))
        if key not in image_lookup:
            raise ValueError(f"mono record is absent from Face4 manifest: {key}")
        image = image_lookup[key]
        face = face_lookup[key]
        camera_id = str(image["camera_id"])
        spec = next(
            value
            for value in face_manifest["cameras"][camera_id]["faces"]
            if str(value["face_id"]) == key[1]
        )
        depth_path = _safe_artifact(mono_cache_root, str(record["path"]))
        with np.load(depth_path, allow_pickle=False) as payload:
            relative_depth = np.asarray(payload["relative_depth"], dtype=np.float32)
        height, width = relative_depth.shape
        face_mask_path = _safe_artifact(face_cache_root, str(face["mask_path"]))
        with Image.open(face_mask_path) as source_mask:
            face_mask = np.asarray(
                source_mask.resize((width, height), Image.Resampling.NEAREST),
                dtype=np.uint8,
            ) > 0
        cache_key = (camera_id, key[1], height, width)
        parent_rays = ray_cache.get(cache_key)
        if parent_rays is None:
            parent_rays = _parent_rays(spec, (height, width))
            ray_cache[cache_key] = parent_rays
        rotation = np.asarray(image["c2w"], dtype=np.float32)[:3, :3]
        world_directions = parent_rays @ rotation.T
        world_z = world_directions[:, :, 2]
        alignment = record.get("alignment", {})
        alignment_valid = bool(alignment.get("valid", False))
        if alignment_valid:
            metric_depth = (
                float(alignment["scale"]) * relative_depth
                + float(alignment["shift"])
            )
            candidate = (
                face_mask
                & np.isfinite(metric_depth)
                & (metric_depth >= config.far_aligned_range_m)
                & (world_z >= config.minimum_world_z_direction)
            )
        else:
            invalid_alignment_views += 1
            candidate = np.zeros((height, width), dtype=bool)
        valid_pixels = int(np.count_nonzero(face_mask))
        raw_candidate_pixels = int(np.count_nonzero(candidate))
        candidate_fraction = raw_candidate_pixels / valid_pixels if valid_pixels else 0.0
        accepted = alignment_valid and candidate_fraction >= config.minimum_candidate_fraction_per_view
        if not accepted:
            if alignment_valid:
                below_fraction_views += 1
            candidate.fill(False)
        else:
            accepted_views += 1
        candidate_pixels = int(np.count_nonzero(candidate))
        total_pixels += valid_pixels
        total_candidates += candidate_pixels

        relative_path = f"masks/{record['sample_id']}.npz"
        destination = _safe_artifact(output_root, relative_path)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, sky_mask=candidate.astype(np.uint8))
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        if accepted and candidate_pixels:
            flat_indices = np.flatnonzero(candidate.reshape(-1))
            stride = max(1, math.ceil(len(flat_indices) / config.maximum_color_samples_per_view))
            flat_indices = flat_indices[::stride]
            directions = world_directions.reshape(-1, 3)[flat_indices]
            rgb_path = _safe_artifact(face_cache_root, str(face["rgb_path"]))
            with Image.open(rgb_path) as source_rgb:
                rgb = np.asarray(
                    source_rgb.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
            colors = rgb.reshape(-1, 3)[flat_indices]
            azimuth_index, elevation_index = _direction_bins(directions, config)
            np.add.at(color_sum, (elevation_index, azimuth_index), colors)
            np.add.at(color_count, (elevation_index, azimuth_index), 1)

        output_records.append(
            {
                "sample_id": str(record["sample_id"]),
                "image_id": key[0],
                "camera_id": camera_id,
                "face_id": key[1],
                "path": relative_path,
                "sha256": _sha256_file(destination),
                "shape": [height, width],
                "face_valid_pixels": valid_pixels,
                "raw_candidate_pixels": raw_candidate_pixels,
                "candidate_pixels": candidate_pixels,
                "candidate_fraction": candidate_fraction,
                "accepted": accepted,
                "alignment_valid": alignment_valid,
            }
        )

    nonempty = color_count > 0
    directional_color = np.zeros_like(color_sum)
    directional_color[nonempty] = color_sum[nonempty] / color_count[nonempty, None]
    global_color = (
        color_sum.sum(axis=(0, 1)) / max(1, int(color_count.sum()))
    )
    directional_color[~nonempty] = global_color
    payload: dict[str, Any] = {
        "schema_version": SKY_EVIDENCE_SCHEMA_VERSION,
        "kind": SKY_EVIDENCE_KIND,
        "split": str(face_manifest.get("split", "")),
        "complete_face_cache": len(output_records) == len(records),
        "source_face_manifest_sha256": face_sha,
        "source_mono_depth_manifest_sha256": mono_sha,
        "dataset_manifest_sha256": mono_manifest.get("dataset_manifest_sha256"),
        "lidar_depth_manifest_sha256": mono_manifest.get("lidar_depth_manifest_sha256"),
        "policy": {
            "name": "aligned_da2_far_world_up_v1",
            "keep_expression": (
                "face4_mask & alignment_valid & aligned_metric_depth>=far_range "
                "& world_direction_z>=minimum & per_view_fraction>=minimum"
            ),
            "far_aligned_range_m": config.far_aligned_range_m,
            "minimum_world_z_direction": config.minimum_world_z_direction,
            "minimum_candidate_fraction_per_view": config.minimum_candidate_fraction_per_view,
            "invalid_alignment_action": "disable_sky_supervision_for_view",
            "vendor_difference": (
                "MipMap uses proprietary mono_depth<=0 as a background proxy; official DA2 "
                "is positive over sky, so this explicit far-range compatibility proxy is used."
            ),
        },
        "directional_color_grid": {
            "azimuth_bins": config.azimuth_bins,
            "elevation_bins": config.elevation_bins,
            "global_rgb": global_color.tolist(),
            "sample_count": int(color_count.sum()),
            "nonempty_bin_count": int(np.count_nonzero(nonempty)),
            "rgb": directional_color.tolist(),
            "counts": color_count.tolist(),
        },
        "summary": {
            "record_count": len(output_records),
            "accepted_view_count": accepted_views,
            "invalid_alignment_view_count": invalid_alignment_views,
            "below_minimum_fraction_view_count": below_fraction_views,
            "face_valid_pixels": total_pixels,
            "candidate_pixels": total_candidates,
            "candidate_fraction": total_candidates / total_pixels if total_pixels else 0.0,
        },
        "records": output_records,
    }
    payload["sky_evidence_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return payload


def verify_sky_evidence_manifest(
    manifest: dict[str, Any],
    *,
    root: Path | None = None,
    verify_artifacts: bool = False,
) -> str:
    expected = str(manifest.get("sky_evidence_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("sky evidence manifest is unsigned")
    unsigned = copy.deepcopy(manifest)
    unsigned.pop("sky_evidence_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("sky evidence manifest signature mismatch")
    if manifest.get("schema_version") != SKY_EVIDENCE_SCHEMA_VERSION or manifest.get("kind") != SKY_EVIDENCE_KIND:
        raise ValueError("unsupported sky evidence schema")
    records = manifest.get("records", [])
    if not records or not manifest.get("complete_face_cache"):
        raise ValueError("sky evidence cache is incomplete")
    if int(manifest.get("summary", {}).get("record_count", -1)) != len(records):
        raise ValueError("sky evidence record count is inconsistent")
    if verify_artifacts:
        if root is None:
            raise ValueError("artifact verification requires a sky evidence root")
        for record in records:
            path = _safe_artifact(Path(root), str(record["path"]))
            if not path.is_file() or _sha256_file(path) != record.get("sha256"):
                raise ValueError(f"sky evidence artifact mismatch: {path}")
    return expected


def _directional_colors(directions: np.ndarray, evidence: dict[str, Any]) -> np.ndarray:
    grid = evidence["directional_color_grid"]
    proxy = SkyEvidenceConfig(
        azimuth_bins=int(grid["azimuth_bins"]),
        elevation_bins=int(grid["elevation_bins"]),
    )
    azimuth_index, elevation_index = _direction_bins(directions, proxy)
    colors = np.asarray(grid["rgb"], dtype=np.float32)
    return colors[elevation_index, azimuth_index]


def build_independent_sky_initialization(
    evidence_manifest: dict[str, Any],
    camera_centre: np.ndarray,
    output_path: Path,
    *,
    config: IndependentSkyConfig = IndependentSkyConfig(),
    force: bool = False,
) -> dict[str, Any]:
    """Create a standalone SH1 sky model; never append it to a surface model."""

    evidence_sha = verify_sky_evidence_manifest(evidence_manifest)
    config.validate()
    centre = np.asarray(camera_centre, dtype=np.float64)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("camera centre must be finite XYZ")
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for path in (output_path, manifest_path):
        if path.exists() and not force:
            raise FileExistsError(f"refusing to replace independent sky output: {path}")
    indices = np.arange(config.count, dtype=np.float64)
    z = config.minimum_world_z_direction + (
        1.0 - config.minimum_world_z_direction
    ) * (indices + 0.5) / config.count
    azimuth = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), z], axis=1
    )
    colors = _directional_colors(directions, evidence_manifest)
    means = centre[None, :] + config.radius_m * directions
    quaternions = np.zeros((config.count, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    opacity_logit = math.log(config.opacity / (1.0 - config.opacity))
    sh0 = ((colors - 0.5) / SH_C0)[:, None, :]
    coefficient_count = (config.sh_degree + 1) ** 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                means=means.astype(np.float32),
                scales=np.full((config.count, 3), math.log(config.scale_m), dtype=np.float32),
                quats=quaternions,
                opacities=np.full((config.count,), opacity_logit, dtype=np.float32),
                sh0=sh0.astype(np.float32),
                shN=np.zeros((config.count, coefficient_count - 1, 3), dtype=np.float32),
                directions=directions.astype(np.float32),
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": SKY_INITIALIZATION_KIND,
        "source_sky_evidence_manifest_sha256": evidence_sha,
        "independent_from_surface_geometry": True,
        "surface_checkpoint_appended": False,
        "training_contract": {
            "surface_geometry_frozen": True,
            "sky_model_trained_separately": True,
            "fallback": "retain_surface_only_model_if_sky_validation_regresses",
        },
        "gaussian_count": config.count,
        "camera_centre_m": centre.tolist(),
        "radius_m": config.radius_m,
        "scale_m": config.scale_m,
        "opacity": config.opacity,
        "minimum_world_z_direction": config.minimum_world_z_direction,
        "color_model": {"mode": "sh", "degree": config.sh_degree},
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "bytes": output_path.stat().st_size,
    }
    payload["sky_initialization_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return payload


def verify_sky_initialization_manifest(manifest: dict[str, Any]) -> str:
    expected = str(manifest.get("sky_initialization_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("sky initialization manifest is unsigned")
    unsigned = copy.deepcopy(manifest)
    unsigned.pop("sky_initialization_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("sky initialization manifest signature mismatch")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != SKY_INITIALIZATION_KIND:
        raise ValueError("unsupported sky initialization schema")
    if manifest.get("independent_from_surface_geometry") is not True:
        raise ValueError("sky initialization is not independent from the surface")
    if manifest.get("surface_checkpoint_appended") is not False:
        raise ValueError("sky initialization unexpectedly appends a surface checkpoint")
    if int(manifest.get("gaussian_count", 0)) <= 0:
        raise ValueError("sky initialization has no Gaussians")
    if manifest.get("color_model") != {"mode": "sh", "degree": 1}:
        raise ValueError("independent sky initialization must reserve SH degree 1")
    return expected
