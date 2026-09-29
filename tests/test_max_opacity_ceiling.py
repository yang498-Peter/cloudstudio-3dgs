"""geometry_regularization.max_opacity: a hard per-gaussian opacity ceiling.

house0305 (research/quality_recovery_v2/20_deep_analysis, 2026-09-29): 34% of the
delivered gaussians sit above opacity 0.9 against 2% in the competitor, whose solid
surfaces are built from stacked 0.1-0.4 layers. Opaque blades show their outlines,
which the user reads as "gaussian texture". The ceiling forces surfaces to be
layered.

Pins:
* None (the default) leaves to_dict - and so every contract - unchanged;
* values outside (0, 1) are refused;
* clip_oversized_gaussians clamps opacity logits to logit(max_opacity), counts the
  clamped rows, and leaves every row already below the ceiling bit-identical.
"""
from __future__ import annotations

import importlib.util
import math

import pytest

from cloudstudio_3dgs.training.regularization import GeometryRegularizationConfig, clip_oversized_gaussians

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is an optional training dependency")


def test_default_is_contract_neutral() -> None:
    default = GeometryRegularizationConfig()
    explicit = GeometryRegularizationConfig(max_opacity=None)
    assert default.to_dict() == explicit.to_dict()
    assert "max_opacity" not in default.to_dict()
    assert GeometryRegularizationConfig(max_opacity=0.8).to_dict()["max_opacity"] == 0.8


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_out_of_range_is_refused(value) -> None:
    with pytest.raises(ValueError, match="max_opacity"):
        GeometryRegularizationConfig(max_opacity=value).validate()


@requires_torch
def test_clamp_caps_only_rows_above_the_ceiling() -> None:
    import torch

    logits = torch.tensor([-4.0, 0.0, 1.0, 2.0, 6.0])  # sigmoid: 0.018, 0.5, 0.73, 0.88, 0.998
    params = {"opacities": torch.nn.Parameter(logits.clone()), "scales": torch.nn.Parameter(torch.zeros(5, 3))}
    config = GeometryRegularizationConfig(max_opacity=0.8)
    report = clip_oversized_gaussians(params, radii_px=None, image_size_px=100, config=config)
    bound = math.log(0.8 / 0.2)
    after = params["opacities"].detach()
    assert report["opacity_capped_count"] == 2
    assert torch.equal(after[:3], logits[:3])  # untouched below the ceiling
    assert torch.allclose(after[3:], torch.full((2,), bound))
    assert float(torch.sigmoid(after).max()) == pytest.approx(0.8, abs=1e-6)


@requires_torch
def test_off_by_default_does_nothing() -> None:
    import torch

    logits = torch.tensor([0.0, 6.0])
    params = {"opacities": torch.nn.Parameter(logits.clone()), "scales": torch.nn.Parameter(torch.zeros(2, 3))}
    report = clip_oversized_gaussians(params, radii_px=None, image_size_px=100, config=GeometryRegularizationConfig())
    assert "opacity_capped_count" not in report
    assert torch.equal(params["opacities"].detach(), logits)
