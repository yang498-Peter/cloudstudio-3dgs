"""The two shape knobs must survive the JSON config round trip.

Both regularizer configs are built by splatting a dict from the trainer config,
so a new field is reachable only if it also round-trips through to_dict and
survives validation. These pin that, so a future field addition that forgets
the plumbing fails here rather than silently doing nothing during a long run.
"""

from __future__ import annotations

import pytest

from cloudstudio_3dgs.training.lidar_normals import NormalAlignmentConfig
from cloudstudio_3dgs.training.regularization import GeometryRegularizationConfig


def test_world_shrink_factor_round_trips():
    source = {
        "enabled": True,
        "max_world_size_m": 0.2,
        "world_shrink_factor": 0.8,
    }
    config = GeometryRegularizationConfig(**source)
    config.validate()
    payload = config.to_dict()
    assert payload["world_shrink_factor"] == 0.8
    assert payload["max_world_size_m"] == 0.2
    # A dict produced by to_dict must rebuild the same config.
    rebuilt = GeometryRegularizationConfig(**payload)
    assert rebuilt == config


def test_tangent_isotropy_round_trips():
    source = {
        "enabled": True,
        "weight_flatten": 0.01,
        "weight_tangent_isotropy": 0.05,
    }
    config = NormalAlignmentConfig(**source)
    config.validate()
    payload = config.to_dict()
    assert payload["weight_tangent_isotropy"] == 0.05
    rebuilt = NormalAlignmentConfig(**payload)
    assert rebuilt == config


def test_defaults_keep_previous_behaviour():
    """Neither knob may change anything unless a config asks for it."""
    assert GeometryRegularizationConfig().world_shrink_factor is None
    assert NormalAlignmentConfig().weight_tangent_isotropy == 0.0


def test_unknown_field_is_rejected():
    """Splatting means a typo would otherwise pass silently as a no-op."""
    with pytest.raises(TypeError):
        GeometryRegularizationConfig(world_shrink_facter=0.8)
    with pytest.raises(TypeError):
        NormalAlignmentConfig(weight_tangent_isotrpy=0.05)
