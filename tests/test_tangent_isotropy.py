"""The tangential isotropy term separates surface disks from needles.

Thinning constrains only the shortest axis, so a needle and a disk are equally
"flat" under it. This term penalizes the ratio of the two tangential axes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cloudstudio_3dgs.training.lidar_normals import (  # noqa: E402
    NormalAlignmentConfig,
)


def _config(**overrides):
    base = {"enabled": True, "weight_align": 0.0, "weight_flatten": 0.0}
    base.update(overrides)
    return NormalAlignmentConfig(**base)


def _raw_term(scales_m):
    """Reproduce the term the regularizer computes, on unit anchor weights."""
    scales = torch.tensor(scales_m, dtype=torch.float32)
    weight = torch.ones(scales.shape[0], dtype=torch.float32)
    sorted_tangent = torch.sort(scales, dim=1).values
    ratio = sorted_tangent[:, 2] / sorted_tangent[:, 1].clamp_min(1e-12)
    return ((weight * (ratio - 1.0).square()).sum() / weight.sum()).item()


def test_disk_is_not_penalized():
    """Equal tangential axes over a thin short axis is the target shape."""
    assert _raw_term([[0.010, 0.010, 0.001]]) == pytest.approx(0.0, abs=1e-9)


def test_needle_is_penalized():
    """One tangential axis ten times the other is a needle, not a surface."""
    assert _raw_term([[0.100, 0.010, 0.001]]) == pytest.approx(81.0, rel=1e-5)


def test_penalty_grows_with_tangential_imbalance():
    mild = _raw_term([[0.020, 0.010, 0.001]])
    severe = _raw_term([[0.100, 0.010, 0.001]])
    assert 0.0 < mild < severe


def test_term_ignores_overall_size():
    """Scale invariance: a disk stays unpenalized at any size."""
    small = _raw_term([[0.001, 0.001, 0.0001]])
    large = _raw_term([[1.000, 1.000, 0.1000]])
    assert small == pytest.approx(large, abs=1e-9)


def test_thinness_alone_cannot_tell_them_apart():
    """Both shapes have the same shortest axis, which is what flatten sees."""
    disk = torch.tensor([0.010, 0.010, 0.001])
    needle = torch.tensor([0.100, 0.010, 0.001])
    assert disk.min() == needle.min()
    assert _raw_term([disk.tolist()]) != _raw_term([needle.tolist()])


def test_weight_defaults_to_disabled():
    assert NormalAlignmentConfig().weight_tangent_isotropy == 0.0


def test_negative_weight_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        _config(weight_tangent_isotropy=-1.0).validate()


def test_weight_is_serialized():
    payload = _config(weight_tangent_isotropy=0.05).to_dict()
    assert payload["weight_tangent_isotropy"] == 0.05
