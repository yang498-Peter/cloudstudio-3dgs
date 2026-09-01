"""Tests for the scale-invariant tangential disk-shape penalty."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cloudstudio_3dgs.training.lidar_normals import NormalAlignmentConfig  # noqa: E402


def _config(**overrides):
    base = {"enabled": True, "weight_align": 0.0, "weight_flatten": 0.0}
    base.update(overrides)
    return NormalAlignmentConfig(**base)


def _raw_term(scales_m):
    scales = torch.tensor(scales_m, dtype=torch.float32)
    sorted_tangent = torch.sort(scales, dim=1).values
    ratio = sorted_tangent[:, 2] / sorted_tangent[:, 1].clamp_min(1e-12)
    return (ratio - 1.0).square().mean().item()


def test_disk_is_not_penalized():
    assert _raw_term([[0.010, 0.010, 0.001]]) == pytest.approx(0.0, abs=1e-9)


def test_needle_is_penalized():
    assert _raw_term([[0.100, 0.010, 0.001]]) == pytest.approx(81.0, rel=1e-5)


def test_term_is_scale_invariant():
    assert _raw_term([[0.001, 0.001, 0.0001]]) == pytest.approx(
        _raw_term([[1.0, 1.0, 0.1]]), abs=1e-9
    )


def test_weight_defaults_to_disabled_and_serializes_when_enabled():
    assert NormalAlignmentConfig().weight_tangent_isotropy == 0.0
    payload = _config(weight_tangent_isotropy=0.05).to_dict()
    assert payload["weight_tangent_isotropy"] == 0.05


def test_negative_weight_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        _config(weight_tangent_isotropy=-1.0).validate()
