"""LiDAR alpha-support construction: dilated (legacy) vs strict_visibility.

Pins three things:
* ``dilated`` is bit-identical to the inline construction the trainer used
  before the mode knob existed (the legacy code is inlined here as oracle);
* ``strict_visibility`` on a wall with a foreground pole keeps the wall's
  interior (including pixels with no return of their own) and the pole's
  interior supported, and rejects the band around the pole silhouette that
  the dilated mode pushes across the depth discontinuity;
* the numpy and torch twins agree, and the knob is plumbed through
  ``TrainerConfig`` (contract shape unchanged on the default, identity
  changed on the strict mode) and into the loss.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from cloudstudio_3dgs.training.alpha_support import (
    ALPHA_SUPPORT_MODES,
    STRICT_VISIBILITY_EDGE_EROSION_PX,
    STRICT_VISIBILITY_MARGIN_M,
    STRICT_VISIBILITY_TOLERANCE,
    lidar_alpha_support,
    lidar_alpha_support_numpy,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig, _render_supervision_loss

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is an optional training dependency")

RADIUS = 6
WALL_M = 5.0
POLE_M = 2.0
POLE_X0, POLE_X1 = 40, 70  # pole occupies columns [40, 70)
HEIGHT, WIDTH = 48, 112


def wall_with_pole(*, stride: int = 3) -> dict[str, np.ndarray]:
    """Sparse wall returns on a ``stride`` grid; dense pole returns in front."""
    valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
    valid[::stride, ::stride] = True
    valid[:, POLE_X0:POLE_X1] = False
    ranges = np.where(valid, np.float32(WALL_M), np.float32(0.0))
    pole = np.zeros_like(valid)
    pole[:, POLE_X0:POLE_X1] = True
    valid |= pole
    ranges = np.where(pole, np.float32(POLE_M), ranges).astype(np.float32)
    confidence = np.where(valid, np.float32(0.8), np.float32(0.0)).astype(np.float32)
    confidence[pole] = 1.0
    return {
        "depth_mask": valid,
        "confidence": confidence,
        "range_m": ranges,
        "pole": pole,
    }


def _reference_strict(depth_mask, confidence, range_m, radius):
    """Plain-loop oracle of the strict rule for small maps."""
    valid = depth_mask & np.isfinite(confidence) & (confidence > 0.0)
    valid &= np.isfinite(range_m) & (range_m > 0.0)
    h, w = valid.shape
    agrees = np.zeros((h, w), dtype=bool)
    has = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            ys, ye = max(0, y - radius), min(h, y + radius + 1)
            xs, xe = max(0, x - radius), min(w, x + radius + 1)
            block_valid = valid[ys:ye, xs:xe]
            if not block_valid.any():
                continue
            block = range_m[ys:ye, xs:xe][block_valid].astype(np.float32)
            has[y, x] = True
            nearest = np.float32(block.min())
            farthest = np.float32(block.max())
            agrees[y, x] = bool(
                farthest
                <= nearest * np.float32(1.0 + STRICT_VISIBILITY_TOLERANCE)
                + np.float32(STRICT_VISIBILITY_MARGIN_M)
            )
    discontinuity = has & ~agrees
    e = STRICT_VISIBILITY_EDGE_EROSION_PX
    near_edge = np.zeros_like(discontinuity)
    for y in range(h):
        for x in range(w):
            ys, ye = max(0, y - e), min(h, y + e + 1)
            xs, xe = max(0, x - e), min(w, x + e + 1)
            near_edge[y, x] = discontinuity[ys:ye, xs:xe].any()
    return agrees & ~near_edge


def test_strict_mode_keeps_interiors_and_rejects_the_pole_silhouette():
    scene = wall_with_pole()
    dilated = lidar_alpha_support_numpy(
        depth_mask=scene["depth_mask"],
        confidence=scene["confidence"],
        range_m=scene["range_m"],
        mode="dilated",
        dilation_radius_px=RADIUS,
    )
    strict = lidar_alpha_support_numpy(
        depth_mask=scene["depth_mask"],
        confidence=scene["confidence"],
        range_m=scene["range_m"],
        mode="strict_visibility",
        dilation_radius_px=RADIUS,
    )
    # The grid stride is far below the window, so the dilated mode supports
    # the whole face, silhouette band included: that is the over-extension.
    assert dilated.support.all()
    assert np.array_equal(strict.weights, dilated.weights)
    assert not (strict.support & ~dilated.support).any()

    # The wall's last return column left of the pole is 39 and the first one
    # right of it is 72 (pole spans 40..69). Windows of radius 6 that reach
    # both surfaces are x in [34, 45] and [66, 75]; the erosion radius grows
    # the rejected bands to [31, 48] and [63, 78].
    assert strict.discontinuity is not None
    assert strict.discontinuity[:, 34:46].all() and strict.discontinuity[:, 66:76].all()
    assert not strict.discontinuity[:, :34].any() and not strict.discontinuity[:, 46:66].any()
    assert not strict.discontinuity[:, 76:].any()
    band_left = slice(34 - STRICT_VISIBILITY_EDGE_EROSION_PX, 45 + STRICT_VISIBILITY_EDGE_EROSION_PX + 1)
    band_right = slice(66 - STRICT_VISIBILITY_EDGE_EROSION_PX, 75 + STRICT_VISIBILITY_EDGE_EROSION_PX + 1)
    assert not strict.support[:, band_left].any()
    assert not strict.support[:, band_right].any()
    # Wall interior stays supported, including pixels without a return of
    # their own (the sparse grid gaps), and so does the pole interior.
    assert strict.support[:, : band_left.start].all()
    assert strict.support[:, band_right.stop :].all()
    assert strict.support[:, band_left.stop : band_right.start].all()
    assert (~scene["depth_mask"][:, : band_left.start]).any()


def test_numpy_strict_matches_loop_reference_on_random_maps():
    rng = np.random.default_rng(20260911)
    for radius in (0, 2, 4):
        h, w = 24, 30
        valid = rng.random((h, w)) < 0.15
        ranges = np.where(rng.random((h, w)) < 0.5, 3.0, 3.0 + rng.random((h, w)) * 2.0)
        ranges = ranges.astype(np.float32)
        confidence = (rng.random((h, w)) * 0.9 + 0.1).astype(np.float32)
        confidence[rng.random((h, w)) < 0.05] = 0.0
        got = lidar_alpha_support_numpy(
            depth_mask=valid,
            confidence=confidence,
            range_m=ranges,
            mode="strict_visibility",
            dilation_radius_px=radius,
        )
        expected = _reference_strict(valid, confidence, ranges, radius)
        assert np.array_equal(got.support, expected), radius


def test_radius_zero_strict_equals_dilated():
    scene = wall_with_pole()
    kwargs = dict(
        depth_mask=scene["depth_mask"],
        confidence=scene["confidence"],
        range_m=scene["range_m"],
        dilation_radius_px=0,
    )
    dilated = lidar_alpha_support_numpy(mode="dilated", **kwargs)
    strict = lidar_alpha_support_numpy(mode="strict_visibility", **kwargs)
    # No window means no disagreement; only the returns themselves count.
    assert np.array_equal(strict.support, dilated.support)
    assert np.array_equal(dilated.support, scene["depth_mask"])


def test_unknown_mode_and_missing_ranges_are_rejected():
    scene = wall_with_pole()
    with pytest.raises(ValueError, match="lidar_alpha_support_mode"):
        lidar_alpha_support_numpy(
            depth_mask=scene["depth_mask"],
            confidence=scene["confidence"],
            range_m=scene["range_m"],
            mode="loose",
            dilation_radius_px=RADIUS,
        )
    with pytest.raises(ValueError, match="requires LiDAR ranges"):
        lidar_alpha_support_numpy(
            depth_mask=scene["depth_mask"],
            confidence=scene["confidence"],
            range_m=None,
            mode="strict_visibility",
            dilation_radius_px=RADIUS,
        )


def _legacy_dilated(torch, tensors, radius):
    """The inline trainer construction before the mode knob (oracle)."""
    valid = (
        tensors["depth_mask"]
        & torch.isfinite(tensors["confidence"])
        & (tensors["confidence"] > 0.0)
    )
    confidence = torch.where(
        valid, tensors["confidence"], torch.zeros_like(tensors["confidence"])
    )
    if radius > 0:
        confidence = torch.nn.functional.max_pool2d(
            confidence[None, None], kernel_size=2 * radius + 1, stride=1, padding=radius
        )[0, 0]
        valid = confidence > 0.0
    return valid, confidence


@requires_torch
def test_dilated_mode_is_bit_identical_to_legacy_inline_construction():
    import torch

    rng = np.random.default_rng(7)
    h, w = 40, 52
    valid = torch.from_numpy(rng.random((h, w)) < 0.1)
    confidence = torch.from_numpy((rng.random((h, w))).astype(np.float32))
    confidence[torch.from_numpy(rng.random((h, w)) < 0.1)] = float("nan")
    tensors = {"depth_mask": valid, "confidence": confidence}
    for radius in (0, 1, 6):
        legacy_mask, legacy_weights = _legacy_dilated(torch, tensors, radius)
        got = lidar_alpha_support(
            torch,
            depth_mask=valid,
            confidence=confidence,
            range_m=None,
            mode="dilated",
            dilation_radius_px=radius,
        )
        assert torch.equal(got.support, legacy_mask)
        assert torch.equal(got.weights, legacy_weights)


@requires_torch
def test_numpy_and_torch_twins_agree():
    import torch

    rng = np.random.default_rng(11)
    h, w = 37, 61
    valid = rng.random((h, w)) < 0.12
    ranges = np.where(rng.random((h, w)) < 0.5, 4.0, 4.0 + rng.random((h, w)) * 3.0).astype(np.float32)
    confidence = (rng.random((h, w))).astype(np.float32)
    pole_scene = wall_with_pole()
    scenes = [
        (valid, confidence, ranges),
        (pole_scene["depth_mask"], pole_scene["confidence"], pole_scene["range_m"]),
    ]
    for depth_mask, conf, rng_m in scenes:
        for mode in ALPHA_SUPPORT_MODES:
            for radius in (0, 3, RADIUS):
                np_result = lidar_alpha_support_numpy(
                    depth_mask=depth_mask,
                    confidence=conf,
                    range_m=rng_m,
                    mode=mode,
                    dilation_radius_px=radius,
                )
                torch_result = lidar_alpha_support(
                    torch,
                    depth_mask=torch.from_numpy(np.asarray(depth_mask)),
                    confidence=torch.from_numpy(np.asarray(conf)),
                    range_m=torch.from_numpy(np.asarray(rng_m)),
                    mode=mode,
                    dilation_radius_px=radius,
                )
                assert np.array_equal(torch_result.support.numpy(), np_result.support), (mode, radius)
                assert np.array_equal(torch_result.weights.numpy(), np_result.weights), (mode, radius)


_BASE_CONFIG = {
    "run_id": "alpha-support-mode",
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
    "rgb_l1_weight": 1.0,
    "rgb_ssim_weight": 0.0,
    "lidar_range_weight": 0.0,
    "lidar_alpha_weight": 0.1,
    "lidar_alpha_target": 0.95,
    "lidar_alpha_dilation_radius_px": 6,
}


def test_default_mode_keeps_the_contract_shape_and_strict_changes_identity():
    default = TrainerConfig.from_dict(dict(_BASE_CONFIG))
    assert default.lidar_alpha_support_mode == "dilated"
    coverage = default.contract_dict()["loss_contract"]["lidar_alpha_coverage"]
    assert set(coverage) == {"enabled", "source", "target", "dilation_radius_px", "loss"}
    assert coverage["source"] == "signed_lidar_depth_mask_confidence_max_dilated"

    strict = TrainerConfig.from_dict(
        {**_BASE_CONFIG, "lidar_alpha_support_mode": "strict_visibility"}
    )
    strict_coverage = strict.contract_dict()["loss_contract"]["lidar_alpha_coverage"]
    assert strict_coverage["support_mode"] == "strict_visibility"
    assert strict_coverage["source"] == "signed_lidar_depth_mask_strict_visibility"
    assert strict_coverage["strict_visibility"] == {
        "search_radius_px": 6,
        "tolerance": STRICT_VISIBILITY_TOLERANCE,
        "margin_m": STRICT_VISIBILITY_MARGIN_M,
        "edge_erosion_px": STRICT_VISIBILITY_EDGE_EROSION_PX,
        "rule": strict_coverage["strict_visibility"]["rule"],
    }
    assert default.contract_dict() != strict.contract_dict()
    # Only the coverage block differs between the two contracts.
    left = default.contract_dict()
    right = strict.contract_dict()
    left["loss_contract"].pop("lidar_alpha_coverage")
    right["loss_contract"].pop("lidar_alpha_coverage")
    assert left == right


def test_invalid_modes_fail_validation():
    bad = TrainerConfig.from_dict({**_BASE_CONFIG, "lidar_alpha_support_mode": "loose"})
    with pytest.raises(ValueError, match="lidar_alpha_support_mode must be one of"):
        bad.validate()
    without_weight = TrainerConfig.from_dict(
        {
            **_BASE_CONFIG,
            "lidar_alpha_weight": 0.0,
            "lidar_alpha_dilation_radius_px": 0,
            "lidar_alpha_support_mode": "strict_visibility",
        }
    )
    with pytest.raises(ValueError, match="requires positive lidar_alpha_weight"):
        without_weight.validate()


@requires_torch
def test_loss_supports_only_the_strict_mask_under_strict_mode():
    import torch

    scene = wall_with_pole()
    tensors = {
        "rgb": torch.zeros((HEIGHT, WIDTH, 3)),
        "rgb_mask": torch.ones((HEIGHT, WIDTH), dtype=torch.bool),
        "depth_mask": torch.from_numpy(scene["depth_mask"]),
        "confidence": torch.from_numpy(scene["confidence"]),
        "range_m": torch.from_numpy(scene["range_m"]),
    }
    rendered = torch.zeros((HEIGHT, WIDTH, 3), dtype=torch.float32)
    alpha = torch.zeros((HEIGHT, WIDTH), dtype=torch.float32, requires_grad=True)

    class Backend:
        @staticmethod
        def render(*args, **kwargs):
            return rendered, None, alpha, {}

    Backend.torch = torch
    results = {}
    for mode in ALPHA_SUPPORT_MODES:
        config = TrainerConfig.from_dict({**_BASE_CONFIG, "lidar_alpha_support_mode": mode})
        _, _, _, _, info = _render_supervision_loss(
            backend=Backend(),
            params={},
            sample=SimpleNamespace(camera_model="pinhole"),
            tensors=tensors,
            config=config,
        )
        results[mode] = float(info["cloudstudio_lidar_alpha_support_fraction"])
    expected_strict = lidar_alpha_support_numpy(
        depth_mask=scene["depth_mask"],
        confidence=scene["confidence"],
        range_m=scene["range_m"],
        mode="strict_visibility",
        dilation_radius_px=RADIUS,
    ).support.mean()
    assert results["dilated"] == pytest.approx(1.0)
    assert results["strict_visibility"] == pytest.approx(float(expected_strict), abs=1e-6)
    assert results["strict_visibility"] < results["dilated"]
