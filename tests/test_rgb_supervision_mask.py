"""rgb_supervision_mask: photometric ownership mask (research/quality_recovery_v2/09).

Pins:
* the default ("all") path is byte-identical: the loss equals the inline
  oracle on rgb_mask, and lidar_support with a support that covers rgb_mask
  reproduces it exactly;
* in lidar_support mode pixels outside the dilated LiDAR support get zero
  gradient (RGB L1, SSIM and the DA2 term), and the terms are normalised by
  the supervised pixel count;
* a view with no supervised pixel yields a graph-connected zero loss;
* the contract gains keys only off the default, the config validation
  rejects the mode without LiDAR inputs, and the adaptive-growth gate refuses
  to sign it (research departure, like the schedule contracts);
* the numpy and torch twins agree.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    ORDERED_STAGES,
    UPSTREAM_DATA_READY_STATUS,
    advance_adaptive_growth_gate,
    sign_gate,
)
from cloudstudio_3dgs.training.rgb_supervision import (
    RGB_SUPERVISION_MASK_MODES,
    RGB_SUPERVISION_MASKED_TERMS,
    rgb_supervision_mask_contract,
    rgb_supervision_mask_numpy,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is an optional training dependency")

HEIGHT, WIDTH = 32, 32
RADIUS = 3
RETURN_COLUMNS = 8  # sparse returns in columns [0, 8); dilated support reaches column 10


def _base_config(**overrides):
    base = {
        "run_id": "rgb-supervision",
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
    }
    base.update(overrides)
    return base


def _synthetic_view(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rgb_mask = np.ones((HEIGHT, WIDTH), dtype=bool)
    rgb_mask[:, -2:] = False  # renderer mask excludes a border strip
    depth_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    depth_mask[::2, :RETURN_COLUMNS:2] = True
    confidence = np.where(depth_mask, np.float32(0.5), np.float32(0.0)).astype(np.float32)
    return {
        "rgb": rng.random((HEIGHT, WIDTH, 3), dtype=np.float32),
        "rendered": rng.random((HEIGHT, WIDTH, 3), dtype=np.float32),
        "rgb_mask": rgb_mask,
        "depth_mask": depth_mask,
        "confidence": confidence,
    }


def _expected_support(view: dict[str, np.ndarray]) -> np.ndarray:
    """Plain-loop oracle of rgb_mask & max_pool(valid, radius)."""
    valid = view["depth_mask"] & (view["confidence"] > 0.0)
    out = np.zeros_like(valid)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            ys, ye = max(0, y - RADIUS), min(HEIGHT, y + RADIUS + 1)
            xs, xe = max(0, x - RADIUS), min(WIDTH, x + RADIUS + 1)
            out[y, x] = bool(valid[ys:ye, xs:xe].any())
    return out & view["rgb_mask"]


def _run_loss(config, view, *, torch, extra_tensors=None, requires_grad=True):
    from cloudstudio_3dgs.training.trainer import _render_supervision_loss

    rendered_leaf = torch.tensor(view["rendered"], requires_grad=requires_grad)
    rendered = rendered_leaf * 1.0

    class Backend:
        @staticmethod
        def render(*args, **kwargs):
            return rendered, None, None, {}

    Backend.torch = torch
    tensors = {
        "rgb": torch.tensor(view["rgb"]),
        "rgb_mask": torch.tensor(view["rgb_mask"]),
        "depth_mask": torch.tensor(view["depth_mask"]),
        "confidence": torch.tensor(view["confidence"]),
        "range_m": torch.where(
            torch.tensor(view["depth_mask"]), torch.tensor(3.0), torch.tensor(0.0)
        ),
    }
    if extra_tensors:
        tensors.update(extra_tensors)
    loss, l1, ssim, range_loss, info = _render_supervision_loss(
        backend=Backend(),
        params={},
        sample=SimpleNamespace(camera_model="fisheye"),
        tensors=tensors,
        config=config,
    )
    return loss, l1.detach(), ssim.detach(), info, rendered_leaf


# ----------------------------------------------------------------------------
# loss behaviour
# ----------------------------------------------------------------------------


@requires_torch
def test_default_path_is_unchanged_and_full_support_reproduces_it() -> None:
    import torch
    from cloudstudio_3dgs.training.losses import masked_rgb_l1, masked_rgb_ssim_loss
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    view = _synthetic_view()
    default = TrainerConfig.from_dict(_base_config())
    assert default.rgb_supervision_mask == "all"
    loss, l1, ssim, info, _ = _run_loss(default, view, torch=torch)
    rendered = torch.tensor(view["rendered"])
    target = torch.tensor(view["rgb"])
    mask = torch.tensor(view["rgb_mask"])
    oracle_l1 = masked_rgb_l1(rendered, target, mask)
    oracle_ssim = masked_rgb_ssim_loss(rendered, target, mask)
    assert float(l1) == pytest.approx(float(oracle_l1), abs=0.0)
    assert float(ssim) == pytest.approx(float(oracle_ssim), abs=0.0)
    assert float(loss.detach()) == pytest.approx(0.6 * float(oracle_l1) + 0.4 * float(oracle_ssim), abs=1e-7)
    assert info["cloudstudio_rgb_supervised_fraction"] == 1.0
    assert info["cloudstudio_rgb_supervised_pixels"] == int(view["rgb_mask"].sum())

    # lidar_support whose dilated support covers every rgb_mask pixel is the
    # same computation on the same pixels: bit-identical loss.
    full = dict(view)
    full["depth_mask"] = view["rgb_mask"].copy()
    full["confidence"] = np.where(full["depth_mask"], np.float32(1.0), np.float32(0.0)).astype(np.float32)
    masked = TrainerConfig.from_dict(
        _base_config(rgb_supervision_mask="lidar_support", rgb_supervision_dilation_radius_px=RADIUS)
    )
    loss_m, l1_m, ssim_m, info_m, _ = _run_loss(masked, full, torch=torch)
    assert float(l1_m) == float(l1)
    assert float(ssim_m) == float(ssim)
    assert float(loss_m) == float(loss)
    assert info_m["cloudstudio_rgb_supervised_fraction"] == 1.0


@requires_torch
def test_masked_path_gives_zero_gradient_outside_support_and_normalises_by_support() -> None:
    import torch
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    view = _synthetic_view()
    expected = _expected_support(view)
    assert 0 < expected.sum() < view["rgb_mask"].sum()
    config = TrainerConfig.from_dict(
        _base_config(rgb_supervision_mask="lidar_support", rgb_supervision_dilation_radius_px=RADIUS)
    )
    loss, l1, ssim, info, leaf = _run_loss(config, view, torch=torch)
    loss.backward()
    grad = leaf.grad.detach().numpy()
    outside = ~expected
    assert np.abs(grad[outside]).max() == 0.0
    assert np.abs(grad[expected]).sum() > 0.0
    # L1 is the mean over the supervised pixels, not the full image.
    diff = np.abs(view["rendered"] - view["rgb"])
    assert float(l1) == pytest.approx(float(diff[expected].mean()), rel=1e-6)
    assert float(l1) != pytest.approx(float(diff[view["rgb_mask"]].mean()), rel=1e-3)
    assert info["cloudstudio_rgb_supervised_pixels"] == int(expected.sum())
    assert info["cloudstudio_rgb_supervised_fraction"] == pytest.approx(
        expected.sum() / view["rgb_mask"].sum()
    )
    assert float(ssim) > 0.0  # the local SSIM found covered windows inside the support


@requires_torch
def test_da2_term_is_masked_with_the_photometric_terms() -> None:
    import torch
    from cloudstudio_3dgs.training.trainer import _render_supervision_loss, TrainerConfig

    view = _synthetic_view()
    expected = _expected_support(view)
    rendered_range = torch.full((HEIGHT, WIDTH), 3.0)
    # DA2 target agrees inside the support and is wildly off outside it.
    da2 = np.where(expected, np.float32(3.0), np.float32(30.0)).astype(np.float32)
    da2_tensors = {
        "da2_range_m": torch.tensor(da2),
        "da2_mask": torch.tensor(view["rgb_mask"]),
    }

    class Backend:
        @staticmethod
        def render(*args, **kwargs):
            return torch.tensor(view["rendered"]), rendered_range, None, {}

    Backend.torch = torch

    def run(mode: str):
        config = TrainerConfig.from_dict(
            _base_config(
                rgb_supervision_mask=mode,
                rgb_supervision_dilation_radius_px=RADIUS,
                da2_depth_weight=0.15,
                mono_depth_manifest="da2.json",
                mono_depth_root="da2",
            )
        )
        tensors = {
            "rgb": torch.tensor(view["rgb"]),
            "rgb_mask": torch.tensor(view["rgb_mask"]),
            "depth_mask": torch.tensor(view["depth_mask"]),
            "confidence": torch.tensor(view["confidence"]),
            **da2_tensors,
        }
        _, _, _, _, info = _render_supervision_loss(
            backend=Backend(),
            params={},
            sample=SimpleNamespace(camera_model="fisheye", K=np.eye(3, dtype=np.float32)),
            tensors=tensors,
            config=config,
        )
        return info["cloudstudio_da2_depth_loss"]

    assert float(run("all")) > 0.0
    assert float(run("lidar_support")) == 0.0


@requires_torch
def test_view_without_support_contributes_a_graph_connected_zero() -> None:
    import torch
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    view = _synthetic_view()
    view["depth_mask"] = np.zeros_like(view["depth_mask"])
    view["confidence"] = np.zeros_like(view["confidence"])
    config = TrainerConfig.from_dict(
        _base_config(rgb_supervision_mask="lidar_support", rgb_supervision_dilation_radius_px=RADIUS)
    )
    loss, l1, ssim, info, leaf = _run_loss(config, view, torch=torch)
    assert float(loss) == 0.0 and float(l1) == 0.0 and float(ssim) == 0.0
    loss.backward()  # must not raise: the zero is connected to the graph
    assert float(leaf.grad.abs().sum()) == 0.0
    assert info["cloudstudio_rgb_supervised_fraction"] == 0.0
    assert info["cloudstudio_rgb_psnr"] is None

    # The same view under "all" keeps the loss functions' fail-closed guard
    # when the renderer mask itself is empty.
    empty = dict(view)
    empty["rgb_mask"] = np.zeros_like(view["rgb_mask"])
    with pytest.raises(ValueError, match="no valid pixels"):
        _run_loss(TrainerConfig.from_dict(_base_config()), empty, torch=torch)


@requires_torch
def test_numpy_and_torch_twins_agree_with_the_oracle() -> None:
    import torch
    from cloudstudio_3dgs.training.rgb_supervision import rgb_supervision_mask

    view = _synthetic_view(seed=3)
    expected = _expected_support(view)
    numpy_mask = rgb_supervision_mask_numpy(
        rgb_mask=view["rgb_mask"],
        depth_mask=view["depth_mask"],
        confidence=view["confidence"],
        mode="lidar_support",
        dilation_radius_px=RADIUS,
    )
    torch_mask = rgb_supervision_mask(
        torch,
        rgb_mask=torch.tensor(view["rgb_mask"]),
        depth_mask=torch.tensor(view["depth_mask"]),
        confidence=torch.tensor(view["confidence"]),
        mode="lidar_support",
        dilation_radius_px=RADIUS,
    )
    assert np.array_equal(numpy_mask.mask, expected)
    assert np.array_equal(torch_mask.mask.numpy(), expected)
    assert numpy_mask.supervised_pixels == torch_mask.supervised_pixels == int(expected.sum())
    assert numpy_mask.fraction == pytest.approx(torch_mask.fraction)
    # "all" hands back the renderer mask itself.
    same = rgb_supervision_mask_numpy(
        rgb_mask=view["rgb_mask"], depth_mask=None, confidence=None, mode="all", dilation_radius_px=0
    )
    assert same.mask is view["rgb_mask"] or np.array_equal(same.mask, view["rgb_mask"])
    assert same.fraction == 1.0


# ----------------------------------------------------------------------------
# config, contract, gate
# ----------------------------------------------------------------------------


@requires_torch
def test_contract_gains_keys_only_off_the_default() -> None:
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    default = TrainerConfig.from_dict(_base_config())
    explicit = TrainerConfig.from_dict(
        _base_config(rgb_supervision_mask="all", rgb_supervision_dilation_radius_px=24)
    )
    assert "rgb_supervision_mask" not in default.contract_dict()["loss_contract"]
    assert default.contract_dict() == explicit.contract_dict()
    # A different radius under "all" must not leak into the identity either.
    other_radius = TrainerConfig.from_dict(
        _base_config(rgb_supervision_mask="all", rgb_supervision_dilation_radius_px=7)
    )
    assert other_radius.contract_dict() == default.contract_dict()

    masked = TrainerConfig.from_dict(
        _base_config(rgb_supervision_mask="lidar_support", rgb_supervision_dilation_radius_px=24)
    )
    masked.validate()
    contract = masked.contract_dict()
    assert contract["loss_contract"]["rgb_supervision_mask"] == rgb_supervision_mask_contract(
        mode="lidar_support", dilation_radius_px=24
    )
    assert contract["loss_contract"]["rgb_supervision_mask"]["masked_terms"] == list(
        RGB_SUPERVISION_MASKED_TERMS
    )
    assert "da2_depth" in RGB_SUPERVISION_MASKED_TERMS
    without = copy.deepcopy(contract)
    without["loss_contract"].pop("rgb_supervision_mask")
    assert without == default.contract_dict()


@requires_torch
def test_validation_requires_lidar_inputs_and_a_positive_radius() -> None:
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    no_depth = _base_config(rgb_supervision_mask="lidar_support")
    no_depth.pop("depth_manifest")
    no_depth.pop("depth_root")
    with pytest.raises(ValueError, match="requires LiDAR depth inputs"):
        TrainerConfig.from_dict(no_depth).validate()
    with pytest.raises(ValueError, match="positive rgb_supervision_dilation_radius_px"):
        TrainerConfig.from_dict(
            _base_config(rgb_supervision_mask="lidar_support", rgb_supervision_dilation_radius_px=0)
        ).validate()
    with pytest.raises(ValueError, match="rgb_supervision_mask must be one of"):
        TrainerConfig.from_dict(_base_config(rgb_supervision_mask="everything")).validate()
    with pytest.raises(ValueError, match="integer within"):
        TrainerConfig.from_dict(_base_config(rgb_supervision_dilation_radius_px=65)).validate()
    assert RGB_SUPERVISION_MASK_MODES == ("all", "lidar_support")


def _upstream_data_gate() -> dict:
    sha = {name: value * 64 for name, value in (("dataset", "1"), ("split", "2"), ("face", "3"), ("mask", "4"), ("da2", "5"), ("tile", "6"))}
    return sign_gate(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "profile": GATE_PROFILE,
            "status": UPSTREAM_DATA_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": ["training implementation is not aligned"],
            "bindings": {
                "training_dataset_manifest_sha256": sha["dataset"],
                "split_manifest_sha256": sha["split"],
                "face4_train_manifest_sha256": sha["face"],
                "face4_val_manifest_sha256": sha["face"],
                "training_circle_mask_manifest_sha256": sha["mask"],
                "da2_train_manifest_sha256": sha["da2"],
                "spatial_tile_plan_manifest_sha256": sha["tile"],
            },
        }
    )


def _signed(config: dict) -> dict:
    signed = copy.deepcopy(config)
    signed.pop("config_manifest_sha256", None)
    signed["config_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(signed)).hexdigest()
    return signed


# The parity arm the schedule-contract test signs (tests/test_schedule_contract.py).
PARITY_ARM = {
    "run_id": "v26-boundary",
    "mipmap_tile_id": 1,
    "topology_policy": {"mode": "adaptive_growth"},
    "densification_strategy": "default_3dgs",
    "densification_gradient_source": "rgb_only",
    "mcmc_refine_start_iter": 500,
    "mcmc_refine_every": 100,
    "mcmc_refine_stop_iter": 5610,
    "mcmc_noise_injection_stop_iter": 0,
    "mcmc_noise_lr": 0.0,
    "max_steps": 7480,
    "controlled_stop_after_steps": 502,
    "factor": 1,
    "cap_max": 2200000,
    "sh_degree": 0,
    "da2_depth_weight": 0.0,
    "error_weighted_sampling": {"enabled": False},
    "default_strategy": {
        "exact_mipmap_lifecycle": True,
        "grow_grad2d": 0.00015,
        "growth_min_opacity": 0.15,
        "split_scale_m": 0.2,
        "prune_scale_m": 0.2,
        "prune_opa": 0.1,
        "prune_opa_late": 0.05,
        "prune_switch_step": 3740,
        "prune_scale2d": 0.15,
        "reset_every": 300,
        "reset_opacity_cap": 0.2,
        "absgrad": True,
        "revised_opacity": True,
    },
    "tangent_proposal": {"enabled": True, "reject_unsupported_births": True},
    "geometry_regularization": {
        "enabled": True,
        "opacity_sparsity_weight": 1e-4,
        "scale_upper_weight": 1e-4,
        "anisotropy_weight": 1e-4,
        "max_scale_ratio_to_reference": 8.0,
        "max_anisotropy": 10.0,
    },
}


def test_adaptive_growth_gate_refuses_a_masked_supervision_arm() -> None:
    gate = advance_adaptive_growth_gate(_upstream_data_gate(), _signed(PARITY_ARM), stage="boundary")
    assert gate["training_allowed"]
    explicit_all = dict(PARITY_ARM, rgb_supervision_mask="all", rgb_supervision_dilation_radius_px=24)
    assert advance_adaptive_growth_gate(_upstream_data_gate(), _signed(explicit_all), stage="boundary")["training_allowed"]
    research = dict(PARITY_ARM, rgb_supervision_mask="lidar_support", rgb_supervision_dilation_radius_px=24)
    with pytest.raises(ValueError, match="full-frame photometric supervision only"):
        advance_adaptive_growth_gate(_upstream_data_gate(), _signed(research), stage="boundary")
    # Signature is still verified first.
    tampered = _signed(research)
    tampered["max_steps"] = 7481
    with pytest.raises(ValueError, match="signature mismatch"):
        advance_adaptive_growth_gate(_upstream_data_gate(), tampered, stage="boundary")
