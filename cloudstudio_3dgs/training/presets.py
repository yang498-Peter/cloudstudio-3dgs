"""Named, immutable Trainer presets for controlled Gate 2 comparisons."""

from __future__ import annotations

import copy
from typing import Any


_EXPOSURE_DISABLED = {
    "enabled": False,
    "learning_rate": 5e-3,
    "regularization_weight": 1e-2,
    "max_abs_log_gain": 0.6931471805599453,
    "zero_mean_projection": False,
}

_EXPOSURE_ENABLED = {**_EXPOSURE_DISABLED, "enabled": True}

_REGULARIZATION_DISABLED = {
    "enabled": False,
    "opacity_sparsity_weight": 1e-4,
    "scale_upper_weight": 1e-4,
    "anisotropy_weight": 1e-4,
    "max_scale_ratio_to_reference": 8.0,
    "max_anisotropy": 10.0,
    "screen_clip_enabled": False,
    "max_screen_fraction": 0.15,
    "screen_clip_hardness": 1.5,
    "screen_clip_opacity_bump": 3.0,
    "max_world_size_m": None,
}

_REGULARIZATION_ENABLED = {**_REGULARIZATION_DISABLED, "enabled": True}

_FIXED_SCALE = {
    "mode": "fixed",
    "knn_neighbors": 3,
    "scale_multiplier": 1.0,
    "clamp_min_ratio": 0.25,
    "clamp_max_ratio": 4.0,
    "means_step_fraction": None,
    "noise_std_fraction": None,
}

_KNN_SCALE = {
    "mode": "knn",
    "knn_neighbors": 3,
    "scale_multiplier": 1.0,
    "clamp_min_ratio": 0.25,
    "clamp_max_ratio": 4.0,
    "means_step_fraction": 0.0032,
    "noise_std_fraction": 0.25,
}


TRAINER_PRESETS: dict[str, dict[str, Any]] = {
    # Replays the pre-Gate-2 Trainer semantics for an honest reference arm.
    "legacy_minimal_v1": {
        "metric_scale_calibration": _FIXED_SCALE,
        "color_model": "rgb_sigmoid",
        "sh_degree": 2,
        "sh_degree_interval": 0,
        "rgb_ssim_mode": "global_moments",
        "decoupled_ssim": False,
        "sh_regularization_weight": 0.0,
        "lidar_range_loss_mode": "linear_l1",
        "means_lr_final_factor": 1.0,
        "background_color": None,
        "exposure_compensation": _EXPOSURE_DISABLED,
        "geometry_regularization": _REGULARIZATION_DISABLED,
    },
    # Three one-variable arms used against legacy_minimal_v1.
    "gate2_knn_only_v1": {
        "metric_scale_calibration": _KNN_SCALE,
        "color_model": "rgb_sigmoid",
        "sh_degree": 2,
        "sh_degree_interval": 0,
        "rgb_ssim_mode": "global_moments",
        "decoupled_ssim": False,
        "sh_regularization_weight": 0.0,
        "lidar_range_loss_mode": "linear_l1",
        "means_lr_final_factor": 1.0,
        "background_color": None,
        "exposure_compensation": _EXPOSURE_DISABLED,
        "geometry_regularization": _REGULARIZATION_DISABLED,
    },
    "gate2_sh_only_v1": {
        "metric_scale_calibration": _FIXED_SCALE,
        "color_model": "sh",
        "sh_degree": 3,
        "sh_degree_interval": 1_000,
        "rgb_ssim_mode": "global_moments",
        "decoupled_ssim": False,
        "sh_regularization_weight": 0.0,
        "lidar_range_loss_mode": "linear_l1",
        "means_lr_final_factor": 1.0,
        "background_color": None,
        "exposure_compensation": _EXPOSURE_DISABLED,
        "geometry_regularization": _REGULARIZATION_DISABLED,
    },
    "gate2_local_ssim_only_v1": {
        "metric_scale_calibration": _FIXED_SCALE,
        "color_model": "rgb_sigmoid",
        "sh_degree": 2,
        "sh_degree_interval": 0,
        "rgb_ssim_mode": "local_gaussian",
        "decoupled_ssim": False,
        "sh_regularization_weight": 0.0,
        "lidar_range_loss_mode": "linear_l1",
        "means_lr_final_factor": 1.0,
        "background_color": None,
        "exposure_compensation": _EXPOSURE_DISABLED,
        "geometry_regularization": _REGULARIZATION_DISABLED,
    },
    # Australian P5 is the prioritized appearance foundation. Gate 2 adds the
    # metric/data terms around it without changing its proven appearance knobs.
    "gate2_quality_australian_p5_v1": {
        "metric_scale_calibration": _KNN_SCALE,
        "color_model": "sh",
        "sh_degree": 3,
        "sh_degree_interval": 1_000,
        "rgb_ssim_mode": "local_gaussian",
        "decoupled_ssim": False,
        "sh_regularization_weight": 0.0,
        "lidar_range_loss_mode": "robust_log_huber",
        "means_lr_final_factor": 0.01,
        "background_color": [1.0, 1.0, 1.0],
        "exposure_compensation": _EXPOSURE_ENABLED,
        "geometry_regularization": _REGULARIZATION_ENABLED,
    },
}


def available_trainer_presets() -> tuple[str, ...]:
    return tuple(sorted(TRAINER_PRESETS))


def expand_trainer_preset(value: dict[str, Any]) -> dict[str, Any]:
    """Apply one named preset while refusing contradictory algorithm fields."""
    if not isinstance(value, dict):
        raise ValueError("trainer configuration must be an object")
    output = copy.deepcopy(value)
    name = str(output.get("trainer_preset", "custom"))
    if name == "custom":
        output["trainer_preset"] = name
        return output
    if name not in TRAINER_PRESETS:
        raise ValueError(
            f"unknown trainer_preset {name!r}; expected one of {available_trainer_presets()}"
        )
    for key, preset_value in TRAINER_PRESETS[name].items():
        if key in output and output[key] != preset_value:
            raise ValueError(
                f"trainer_preset {name!r} fixes {key}; use trainer_preset='custom' "
                "for an override"
            )
        output[key] = copy.deepcopy(preset_value)
    output["trainer_preset"] = name
    return output


def assert_trainer_preset_matches(name: str, values: dict[str, Any]) -> None:
    """Prevent direct dataclass construction from mislabelling a preset."""
    if name == "custom":
        return
    expected = TRAINER_PRESETS.get(name)
    if expected is None:
        raise ValueError(f"unknown trainer_preset {name!r}")
    mismatched = [key for key, expected_value in expected.items() if values.get(key) != expected_value]
    if mismatched:
        raise ValueError(
            f"trainer_preset {name!r} does not match fields: {', '.join(sorted(mismatched))}"
        )
