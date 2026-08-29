"""Tiles are cut on the population they hold, not the pixels they name.

Views stream from disk one sample at a time, so the sum of a tile's view
pixels never occupies memory at once. Cutting on that number splits a scene
far past what the card requires, and every extra cut widens the halo, so views
land in more than one tile and get trained more than once.
"""

from __future__ import annotations

import pytest

from cloudstudio_3dgs.pipeline.adaptive_tiling import (
    GaussianResidencyModel,
    startup_budget_gib,
)


def test_defaults_match_the_largest_measured_run():
    """18.758M gaussians peaked at 10.011 GiB, frozen topology, factor 1."""
    steady = GaussianResidencyModel(lifecycle_multiplier=1.0, growth_ratio=1.0)
    assert steady.peak_gib(18_757_869) == pytest.approx(10.011, abs=0.02)


def test_small_runs_would_have_understated_the_slope():
    """Why the fit is anchored at the large point rather than across all three.

    The two small runs sit mostly in the fixed workspace, so their apparent
    slope is shallow; carrying it to 18.76M understated the real peak by
    roughly half.
    """
    optimistic = GaussianResidencyModel(
        gib_per_million=0.332, lifecycle_multiplier=1.0, growth_ratio=1.0
    )
    assert optimistic.peak_gib(18_757_869) < 7.0
    steady = GaussianResidencyModel(lifecycle_multiplier=1.0, growth_ratio=1.0)
    assert steady.peak_gib(18_757_869) / optimistic.peak_gib(18_757_869) > 1.4


def test_peak_scales_with_the_refinement_transient():
    """Clone and split briefly hold old and new tensors together."""
    steady = GaussianResidencyModel(lifecycle_multiplier=1.0, growth_ratio=1.0)
    peaky = GaussianResidencyModel(lifecycle_multiplier=3.0, growth_ratio=1.0)
    n = 10_000_000
    gaussian_part_steady = steady.peak_gib(n) - steady.base_gib
    gaussian_part_peaky = peaky.peak_gib(n) - peaky.base_gib
    assert gaussian_part_peaky == pytest.approx(3.0 * gaussian_part_steady)


def test_growth_ratio_projects_anchors_forward():
    """A tile is sized by what it grows into, not by what it seeds with."""
    model = GaussianResidencyModel(lifecycle_multiplier=1.0, growth_ratio=4.0)
    seeded = GaussianResidencyModel(lifecycle_multiplier=1.0, growth_ratio=1.0)
    assert model.peak_gib(1_000_000) == pytest.approx(seeded.peak_gib(4_000_000))


def test_capacity_and_cap_agree_with_each_other():
    model = GaussianResidencyModel(lifecycle_multiplier=3.0, growth_ratio=5.0)
    budget = 12.92
    anchors = model.anchor_capacity(budget)
    # Growing that many anchors by the ratio must land on the gaussian cap.
    assert anchors * model.growth_ratio == pytest.approx(
        model.gaussian_cap(budget), rel=1e-3
    )
    # And the projected peak of a tile at capacity must fit.
    assert model.peak_gib(anchors) <= budget + 1e-6


def test_capacity_is_zero_when_the_budget_cannot_hold_the_workspace():
    model = GaussianResidencyModel(base_gib=4.0)
    assert model.anchor_capacity(2.0) == 0
    assert model.gaussian_cap(2.0) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gib_per_million": 0.0},
        {"lifecycle_multiplier": 0.5},
        {"growth_ratio": 0.5},
        {"base_gib": -1.0},
    ],
)
def test_invalid_models_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GaussianResidencyModel(**kwargs).validate()


def test_budget_reports_its_own_formula():
    """The recorded formula must describe the ceiling actually applied."""
    budget = startup_budget_gib(
        gpu0_available_gib=12.9, system_available_gib=19.5, ceiling_gib=64.0
    )
    assert budget["budget_gib"] == pytest.approx(12.9)
    assert budget["ceiling_gib"] == 64.0
    assert "12_GiB" not in budget["formula"]
    assert "ceiling_gib" in budget["formula"]


def test_budget_still_honours_a_tighter_ceiling():
    budget = startup_budget_gib(
        gpu0_available_gib=40.0, system_available_gib=60.0, ceiling_gib=12.0
    )
    assert budget["budget_gib"] == pytest.approx(12.0)
