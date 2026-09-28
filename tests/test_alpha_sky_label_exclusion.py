"""lidar_alpha_exclude_sky_label: the alpha-coverage floor stops at the sky label.

house0305 (research/quality_recovery_v2/20_deep_analysis, 2026-09-28): even under
strict_visibility support, 40-58% of the alpha-0.95 demand at 8-40 m lands on pixels
that are sky in the photo - a branch against the sky has no second surface in its
window, so the window "agrees" and every branch return still grows a 6 px disc of
"be opaque" over the surrounding sky gaps. The sky label arrives at the loss before
the sky term's LiDAR guard and covers ~80% of that sky demand, but only 1.5% of the
demand at 0-8 m, so excluding it frees the canopy gaps without touching walls.

Pins:
* off by default, and off keeps the contract and trainer_config_sha256 unchanged;
* enabling it without sky_supervision (or without the alpha term) is refused;
* the contract gains exactly one key when enabled;
* the alpha term then ignores sky-label pixels: its value is the mean over the
  remaining supported pixels and its gradient on sky pixels is zero.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cloudstudio_3dgs.data.manifest import canonical_json_bytes

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is an optional training dependency")

HEIGHT, WIDTH = 24, 24
SKY_ROWS = 10


def _base_config(**overrides):
    base = {
        "run_id": "alpha-sky-exclusion",
        "trainer_preset": "custom",
        "dataset_manifest": "dataset.json",
        "recording_root": "recording",
        "mask_manifest": "masks.json",
        "mask_root": "masks",
        "split_manifest": "split.json",
        "initialization_ply": "sparse_pc.ply",
        "output_dir": "run",
        "gsplat_lock": "upstream/gsplat.lock.json",
        "require_person_masks": False,
        "rgb_l1_weight": 0.6,
        "rgb_ssim_weight": 0.4,
        "lidar_range_weight": 0.0,
        "depth_manifest": "depth.json",
        "depth_root": "depth",
        "lidar_alpha_weight": 0.1,
        "lidar_alpha_target": 0.95,
    }
    base.update(overrides)
    return base


def _sky_settings(**overrides):
    settings = {
        "enabled": True,
        "mask_manifest": "sky.json",
        "mask_root": "sky",
        "alpha_weight": 0.5,
        "alpha_target": 0.0,
        "mask_erosion_px": 0,
        "require_no_lidar_within_px": 0,
        "exclude_photometric": False,
    }
    settings.update(overrides)
    return settings


def _contract_sha(config) -> str:
    return hashlib.sha256(canonical_json_bytes(config.contract_dict())).hexdigest()


def _enabled_pair(flag_values):
    """Contracts need a real signed sky manifest bound to a face cache; reuse the
    sky-supervision suite's minimal fixtures."""
    import json
    import tempfile

    import test_sky_supervision as sky_suite
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    out = []
    with tempfile.TemporaryDirectory(prefix="alpha-sky-contract-") as temporary:
        root = Path(temporary)
        face_path = sky_suite._minimal_face_cache(root)
        face_sha = json.loads(face_path.read_text(encoding="utf-8"))["face_manifest_sha256"]
        sky_path = sky_suite._minimal_sky_manifest(root, face_sha)
        for flag in flag_values:
            overrides = dict(
                face_cache_manifest=str(face_path),
                face_cache_root=str(root),
                sky_supervision=sky_suite._sky_settings(mask_manifest=str(sky_path), mask_root=str(root)),
                lidar_alpha_weight=0.1,
            )
            if flag is not None:
                overrides["lidar_alpha_exclude_sky_label"] = flag
            config = TrainerConfig.from_dict(sky_suite._classic_config(**overrides))
            out.append((config, config.contract_dict()))
    return out


def _find_flag(node):
    if isinstance(node, dict):
        if "sky_label_excluded" in node:
            return node
        for value in node.values():
            hit = _find_flag(value)
            if hit is not None:
                return hit
    return None


def test_off_by_default_and_off_is_signature_neutral() -> None:
    (default, default_contract), (explicit, explicit_contract) = _enabled_pair([None, False])
    assert default.lidar_alpha_exclude_sky_label is False
    assert default_contract == explicit_contract
    assert _find_flag(default_contract) is None


def test_contract_gains_one_key_when_enabled() -> None:
    (_, off), (_, on) = _enabled_pair([False, True])
    assert _find_flag(on)["sky_label_excluded"] is True
    stripped = json_copy(on)
    _find_flag(stripped).pop("sky_label_excluded")
    assert stripped == off


def json_copy(value):
    import json

    return json.loads(json.dumps(value))


def test_validation_refusals() -> None:
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    with pytest.raises(ValueError, match="lidar_alpha_exclude_sky_label"):
        TrainerConfig.from_dict(_base_config(lidar_alpha_exclude_sky_label=True)).validate()
    with pytest.raises(ValueError, match="lidar_alpha_exclude_sky_label"):
        TrainerConfig.from_dict(
            _base_config(
                sky_supervision=_sky_settings(),
                lidar_alpha_exclude_sky_label=True,
                lidar_alpha_weight=0.0,
            )
        ).validate()
    with pytest.raises(ValueError, match="boolean"):
        TrainerConfig.from_dict(
            _base_config(sky_supervision=_sky_settings(), lidar_alpha_exclude_sky_label="yes")
        ).validate()


def _run_alpha(config, *, torch, alpha):
    from cloudstudio_3dgs.training.trainer import _render_supervision_loss

    rng = np.random.default_rng(7)
    rgb = rng.random((HEIGHT, WIDTH, 3), dtype=np.float32)
    rendered = torch.tensor(rng.random((HEIGHT, WIDTH, 3), dtype=np.float32))
    sky = np.zeros((HEIGHT, WIDTH), dtype=bool)
    sky[:SKY_ROWS] = True
    # a return on every pixel, so the support covers the sky band too - the case
    # a branch against the sky produces under both support modes
    depth_mask = np.ones((HEIGHT, WIDTH), dtype=bool)
    confidence = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)

    class Backend:
        @staticmethod
        def render(*args, **kwargs):
            return rendered, None, alpha, {}

    Backend.torch = torch
    tensors = {
        "rgb": torch.tensor(rgb),
        "rgb_mask": torch.ones((HEIGHT, WIDTH), dtype=torch.bool),
        "depth_mask": torch.tensor(depth_mask),
        "confidence": torch.tensor(confidence),
        "sky_mask": torch.tensor(sky),
    }
    _, _, _, _, info = _render_supervision_loss(
        backend=Backend(),
        params={},
        sample=SimpleNamespace(camera_model="fisheye", K=np.eye(3, dtype=np.float32)),
        tensors=tensors,
        config=config,
    )
    return info, sky


@requires_torch
def test_alpha_term_skips_sky_label_pixels() -> None:
    import torch
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    alpha_values = np.linspace(0.1, 0.9, HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
    off = TrainerConfig.from_dict(_base_config(sky_supervision=_sky_settings()))
    on = TrainerConfig.from_dict(
        _base_config(sky_supervision=_sky_settings(), lidar_alpha_exclude_sky_label=True)
    )

    leaf_off = torch.tensor(alpha_values, requires_grad=True)
    info_off, sky = _run_alpha(off, torch=torch, alpha=leaf_off * 1.0)
    leaf_on = torch.tensor(alpha_values, requires_grad=True)
    info_on, _ = _run_alpha(on, torch=torch, alpha=leaf_on * 1.0)

    deficit = np.maximum(0.95 - alpha_values, 0.0) ** 2
    assert float(info_off["cloudstudio_lidar_alpha_loss"]) == pytest.approx(float(deficit.mean()), rel=1e-5)
    assert float(info_on["cloudstudio_lidar_alpha_loss"]) == pytest.approx(float(deficit[~sky].mean()), rel=1e-5)

    info_on["cloudstudio_lidar_alpha_loss"].backward()
    grad = leaf_on.grad.numpy()
    assert np.abs(grad[sky]).max() == 0.0  # no push toward opaque on sky-label pixels
    assert (grad[~sky] < 0.0).all()  # the floor still pushes every other supported pixel up
