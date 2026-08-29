"""Shape-preserving world-size shrink versus the per-axis clamp."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cloudstudio_3dgs.training.regularization import (  # noqa: E402
    GeometryRegularizationConfig,
    clip_oversized_gaussians,
)


def _params(scales_m):
    log_scales = torch.log(torch.tensor(scales_m, dtype=torch.float32))
    return {
        "scales": log_scales.clone(),
        "opacities": torch.zeros(log_scales.shape[0], dtype=torch.float32),
    }


def _config(**overrides):
    base = {
        "enabled": True,
        "screen_clip_enabled": False,
        "max_world_size_m": 0.2,
    }
    base.update(overrides)
    return GeometryRegularizationConfig(**base)


def test_clamp_flattens_the_two_long_axes():
    """The existing clamp collapses distinct long axes onto the bound."""
    params = _params([[1.0, 0.5, 0.001]])
    clip_oversized_gaussians(
        params, radii_px=None, image_size_px=1024, config=_config()
    )
    scales = torch.exp(params["scales"])[0]
    assert scales[0] == pytest.approx(0.2, rel=1e-5)
    assert scales[1] == pytest.approx(0.2, rel=1e-5)
    # Two axes that differed by 2x now sit on top of each other.
    assert scales[0] == pytest.approx(scales[1], rel=1e-5)


def test_shrink_preserves_the_aspect_ratio():
    params = _params([[1.0, 0.5, 0.001]])
    report = clip_oversized_gaussians(
        params,
        radii_px=None,
        image_size_px=1024,
        config=_config(world_shrink_factor=0.8),
    )
    scales = torch.exp(params["scales"])[0]
    assert report["world_shrunk_count"] == 1
    assert scales[0] == pytest.approx(0.8, rel=1e-5)
    assert scales[1] == pytest.approx(0.4, rel=1e-5)
    assert scales[2] == pytest.approx(0.0008, rel=1e-5)
    assert (scales[0] / scales[1]).item() == pytest.approx(2.0, rel=1e-5)


def test_shrink_converges_under_the_bound():
    params = _params([[1.0, 0.5, 0.001]])
    config = _config(world_shrink_factor=0.8)
    steps = 0
    while torch.exp(params["scales"]).max() > 0.2 and steps < 100:
        clip_oversized_gaussians(
            params, radii_px=None, image_size_px=1024, config=config
        )
        steps += 1
    scales = torch.exp(params["scales"])[0]
    assert scales.max() <= 0.2
    # log(0.2)/log(0.8) over the 1.0 m axis is just under 8 steps.
    assert steps == math.ceil(math.log(0.2) / math.log(0.8))
    assert (scales[0] / scales[1]).item() == pytest.approx(2.0, rel=1e-4)


def test_compliant_gaussians_are_untouched():
    params = _params([[0.1, 0.05, 0.001]])
    before = params["scales"].clone()
    report = clip_oversized_gaussians(
        params,
        radii_px=None,
        image_size_px=1024,
        config=_config(world_shrink_factor=0.8),
    )
    assert report["world_shrunk_count"] == 0
    assert torch.equal(params["scales"], before)


def test_shrink_factor_requires_a_trigger_bound():
    with pytest.raises(ValueError, match="requires max_world_size_m"):
        GeometryRegularizationConfig(
            max_world_size_m=None, world_shrink_factor=0.8
        ).validate()


@pytest.mark.parametrize("factor", [0.0, 1.0, -0.5, 1.5])
def test_shrink_factor_must_be_a_proper_fraction(factor):
    with pytest.raises(ValueError, match="world_shrink_factor"):
        GeometryRegularizationConfig(
            max_world_size_m=0.2, world_shrink_factor=factor
        ).validate()
