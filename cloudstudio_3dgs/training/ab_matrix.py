"""Signed, one-variable Gate 2 Trainer A/B configuration matrices."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.trainer import TrainerConfig


AB_ARMS: tuple[tuple[str, str, str], ...] = (
    ("reference", "legacy_minimal_v1", "reference"),
    ("knn_only", "gate2_knn_only_v1", "single_variable"),
    ("sh_only", "gate2_sh_only_v1", "single_variable"),
    ("local_ssim_only", "gate2_local_ssim_only_v1", "single_variable"),
    ("quality_candidate", "gate2_quality_australian_p5_v1", "combined_candidate"),
)

_FILE_INPUTS = (
    "dataset_manifest",
    "mask_manifest",
    "split_manifest",
    "initialization_ply",
    "gsplat_lock",
    "person_mask_manifest",
    "depth_manifest",
)

_DIRECTORY_INPUTS = (
    "recording_root",
    "mask_root",
    "person_mask_root",
    "depth_root",
)

_PRESET_FIELDS = {
    "metric_scale_calibration",
    "color_model",
    "sh_degree",
    "sh_degree_interval",
    "rgb_ssim_mode",
    "lidar_range_loss_mode",
    "means_lr_final_factor",
    "background_color",
    "exposure_compensation",
    "geometry_regularization",
}

_ALLOWED_SINGLE_VARIABLE_PREFIXES = {
    "knn_only": (
        "initialization",
        "optimizer.means_step_fraction",
        "strategy.noise_std_fraction",
    ),
    "sh_only": ("color_model",),
    "local_ssim_only": ("loss_contract.rgb_ssim",),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
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


def _resolve_input_paths(value: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    output = copy.deepcopy(value)
    for key in (*_FILE_INPUTS, *_DIRECTORY_INPUTS):
        if output.get(key) is None:
            continue
        path = Path(str(output[key]))
        if not path.is_absolute():
            path = base_dir / path
        output[key] = str(path.resolve())
    return output


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child))
        return result
    return {prefix: value}


def _contract_differences(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    left = _flatten(reference)
    right = _flatten(candidate)
    return sorted(
        key
        for key in set(left) | set(right)
        if key != "trainer_preset" and left.get(key) != right.get(key)
    )


def _safe_relative(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError("A/B artifact paths must use forward slashes")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError("unsafe A/B artifact path")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError("A/B artifact path escapes matrix root")
    return resolved


def build_trainer_ab_matrix(
    base: dict[str, Any],
    *,
    base_config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    """Create five immutable Trainer configs with shared data and execution knobs."""
    if not experiment_id or any(character in experiment_id for character in "\\/\0"):
        raise ValueError("experiment_id must be a non-empty path-safe name")
    forbidden = sorted(
        ({"run_id", "output_dir", "resume_checkpoint", "trainer_preset"} | _PRESET_FIELDS)
        & set(base)
    )
    if forbidden:
        raise ValueError(
            "A/B base configuration contains arm-specific fields: "
            + ", ".join(forbidden)
        )
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"A/B output is not empty: {output_dir}")
    resolved_base = _resolve_input_paths(base, Path(base_config_path).resolve().parent)
    for key in _FILE_INPUTS:
        value = resolved_base.get(key)
        if value is not None and not Path(value).is_file():
            raise FileNotFoundError(f"A/B file input is missing: {key}")
    for key in _DIRECTORY_INPUTS:
        value = resolved_base.get(key)
        if value is not None and not Path(value).is_dir():
            raise NotADirectoryError(f"A/B directory input is missing: {key}")

    configs: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    arm_records: list[dict[str, Any]] = []
    for arm, preset, role in AB_ARMS:
        value = copy.deepcopy(resolved_base)
        value.update(
            {
                "run_id": f"{experiment_id}-{arm}",
                "trainer_preset": preset,
                "output_dir": str((output_dir / "runs" / arm).resolve()),
            }
        )
        config = TrainerConfig.from_dict(value)
        config.validate()
        contract = config.contract_dict()
        configs[arm] = value
        contracts[arm] = contract
        arm_records.append(
            {
                "arm": arm,
                "role": role,
                "trainer_preset": preset,
                "config_path": f"configs/{arm}.json",
                "trainer_config_sha256": hashlib.sha256(
                    canonical_json_bytes(contract)
                ).hexdigest(),
            }
        )

    reference = contracts["reference"]
    for record in arm_records:
        arm = str(record["arm"])
        differences = _contract_differences(reference, contracts[arm])
        record["contract_differences_from_reference"] = differences
        if record["role"] == "single_variable":
            allowed = _ALLOWED_SINGLE_VARIABLE_PREFIXES[arm]
            unexpected = [
                path
                for path in differences
                if not any(path == prefix or path.startswith(prefix + ".") for prefix in allowed)
            ]
            if not differences or unexpected:
                raise ValueError(
                    f"A/B arm {arm} is not single-variable; unexpected={unexpected}"
                )

    input_hashes = {
        key: _sha256_file(Path(resolved_base[key]))
        for key in _FILE_INPUTS
        if resolved_base.get(key) is not None
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "gate2_trainer_ab_matrix_v1",
        "experiment_id": experiment_id,
        "base_config_sha256": hashlib.sha256(canonical_json_bytes(base)).hexdigest(),
        "shared_input_file_sha256": input_hashes,
        "arms": arm_records,
        "invariants": {
            "same_dataset_masks_split_pose_initialization": True,
            "same_seed_steps_factor_loss_weights_and_mcmc": True,
            "single_variable_arms": ["knn_only", "sh_only", "local_ssim_only"],
            "combined_candidate": "quality_candidate",
            "formal_training_not_started_by_builder": True,
        },
    }
    manifest["ab_matrix_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm, value in configs.items():
        _atomic_json(output_dir / "configs" / f"{arm}.json", value)
    for record in manifest["arms"]:
        path = output_dir / str(record["config_path"])
        record["config_file_sha256"] = _sha256_file(path)
    # Config file identities are part of the final signed matrix.
    manifest.pop("ab_matrix_sha256")
    manifest["ab_matrix_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _atomic_json(output_dir / "ab_matrix_manifest.json", manifest)
    return manifest


def verify_trainer_ab_matrix(manifest: dict[str, Any], root: Path) -> str:
    expected = str(manifest.get("ab_matrix_sha256", ""))
    if not expected:
        raise ValueError("A/B matrix has no SHA256")
    unsigned = dict(manifest)
    unsigned.pop("ab_matrix_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(f"A/B matrix SHA256 mismatch: expected {expected}, computed {actual}")
    arms = manifest.get("arms")
    if not isinstance(arms, list) or [record.get("arm") for record in arms] != [
        arm for arm, _, _ in AB_ARMS
    ]:
        raise ValueError("A/B matrix arm order or coverage is invalid")
    root = Path(root)
    contracts: dict[str, dict[str, Any]] = {}
    for record in arms:
        path = _safe_relative(root, str(record["config_path"]))
        if not path.is_file() or _sha256_file(path) != record.get("config_file_sha256"):
            raise ValueError("A/B config file identity mismatch")
        value = json.loads(path.read_text(encoding="utf-8"))
        config = TrainerConfig.from_dict(value)
        config.validate()
        contract_sha = hashlib.sha256(canonical_json_bytes(config.contract_dict())).hexdigest()
        if contract_sha != record.get("trainer_config_sha256"):
            raise ValueError("A/B Trainer contract identity mismatch")
        if config.trainer_preset != record.get("trainer_preset"):
            raise ValueError("A/B Trainer preset identity mismatch")
        contracts[str(record["arm"])] = config.contract_dict()
        for key, input_sha in manifest.get("shared_input_file_sha256", {}).items():
            input_path = getattr(config, str(key), None)
            if input_path is None or not Path(input_path).is_file():
                raise FileNotFoundError(f"A/B shared input is missing: {key}")
            if _sha256_file(Path(input_path)) != input_sha:
                raise ValueError(f"A/B shared input identity mismatch: {key}")
    reference = contracts["reference"]
    for record in arms:
        arm = str(record["arm"])
        differences = _contract_differences(reference, contracts[arm])
        if differences != record.get("contract_differences_from_reference"):
            raise ValueError(f"A/B contract difference record mismatch: {arm}")
        if record.get("role") == "single_variable":
            allowed = _ALLOWED_SINGLE_VARIABLE_PREFIXES[arm]
            if any(
                not any(path == prefix or path.startswith(prefix + ".") for prefix in allowed)
                for path in differences
            ):
                raise ValueError(f"A/B arm is not single-variable: {arm}")
    return actual
