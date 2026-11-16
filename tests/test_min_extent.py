"""Sampling-rate lower bound on gaussian extent (regularization.py), CPU only.

What is pinned:

* a gaussian's rate is the largest focal / depth among the views that rasterized it; views
  that did not see it, and depths inside the near guard, do not change it;
* the bound raises exactly the axes below min_extent_px / rate and leaves everything else,
  including gaussians no view has seen, untouched;
* the rate tensor lives in the strategy state at the population's length, so the lifecycle
  ops that transform that state carry it;
* the knob is absent from the contract when off.
"""

from __future__ import annotations

import math
import unittest

import torch

from cloudstudio_3dgs.training.regularization import (
    SAMPLING_RATE_KEY,
    GeometryRegularizationConfig,
    apply_min_extent,
    observe_sampling_rate,
)


def _observe(state, radii, depths, focal=800.0):
    return observe_sampling_rate(
        state,
        radii=torch.tensor(radii, dtype=torch.float32),
        depths=torch.tensor(depths, dtype=torch.float32),
        focal_px=focal,
        gaussian_count=len(depths),
    )


class SamplingRateTest(unittest.TestCase):
    def test_the_rate_is_the_closest_view_that_saw_the_gaussian(self) -> None:
        state: dict = {}
        _observe(state, [3.0, 2.0, 0.0], [4.0, 2.0, 1.0])
        self.assertEqual(state[SAMPLING_RATE_KEY].tolist(), [200.0, 400.0, 0.0])
        _observe(state, [1.0, 0.0, 5.0], [2.0, 0.5, 8.0])  # row 1 is not visible in this view
        self.assertEqual(state[SAMPLING_RATE_KEY].tolist(), [400.0, 400.0, 100.0])
        _observe(state, [1.0, 1.0, 1.0], [10.0, 10.0, 10.0])  # a farther view lowers nothing
        self.assertEqual(state[SAMPLING_RATE_KEY].tolist(), [400.0, 400.0, 100.0])

    def test_two_component_radii_and_the_camera_axis_are_accepted(self) -> None:
        state: dict = {}
        observe_sampling_rate(
            state,
            radii=torch.tensor([[[2, 0], [0, 0]]], dtype=torch.int32),
            depths=torch.tensor([[2.0, 2.0]]),
            focal_px=800.0,
            gaussian_count=2,
        )
        self.assertEqual(state[SAMPLING_RATE_KEY].tolist(), [400.0, 0.0])

    def test_a_gaussian_grazing_the_camera_earns_no_rate(self) -> None:
        state: dict = {}
        _observe(state, [4.0, 4.0], [0.01, -0.3])
        self.assertEqual(state[SAMPLING_RATE_KEY].tolist(), [0.0, 0.0])

    def test_a_population_of_another_length_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            observe_sampling_rate(
                {}, radii=torch.ones(3), depths=torch.ones(2), focal_px=800.0, gaussian_count=3
            )


class MinExtentTest(unittest.TestCase):
    def test_only_axes_below_the_bound_are_raised(self) -> None:
        state: dict = {}
        _observe(state, [1.0, 1.0, 0.0], [1.0, 4.0, 1.0], focal=800.0)  # rates 800, 200, unseen
        # 0.447 px: 0.559 mm at 1 m, 2.235 mm at 4 m.
        scales_m = torch.tensor(
            [[0.004, 0.002, 0.0002], [0.004, 0.002, 0.0002], [0.004, 0.002, 0.0002]]
        )
        params = {"scales": torch.nn.Parameter(scales_m.log())}
        before = params["scales"].detach().clone()
        report = apply_min_extent(params, state, min_extent_px=0.447)
        out = params["scales"].detach().exp()
        self.assertAlmostEqual(float(out[0, 2]), 0.447 / 800.0, places=7)
        # Axes already above the bound keep their exact parameter bits.
        self.assertTrue(torch.equal(params["scales"].detach()[0, :2], before[0, :2]))
        self.assertTrue(torch.equal(params["scales"].detach()[1, 0], before[1, 0]))
        self.assertAlmostEqual(float(out[1, 2]), 0.447 / 200.0, places=7)
        self.assertAlmostEqual(float(out[1, 1]), 0.447 / 200.0, places=7)  # 2 mm < 2.235 mm
        self.assertAlmostEqual(float(out[1, 0]), 0.004, places=7)
        self.assertTrue(torch.allclose(out[2], scales_m[2]))  # never rasterized: untouched
        self.assertEqual(report["min_extent_raised_count"], 2)
        self.assertEqual(report["min_extent_observed_count"], 2)

    def test_the_anisotropy_of_a_thin_disk_is_bounded_by_its_distance(self) -> None:
        state: dict = {}
        _observe(state, [1.0], [1.0], focal=787.6)
        params = {"scales": torch.nn.Parameter(torch.tensor([[0.0045, 0.0018, 0.00026]]).log())}
        apply_min_extent(params, state, min_extent_px=0.447)
        out = params["scales"].detach().exp()[0]
        self.assertAlmostEqual(float(out.min()) * 1000, 0.5675, places=3)  # mm at 1 m
        self.assertLess(float(out.max() / out.min()), 8.0)  # was 17.3

    def test_off_or_without_rates_nothing_moves(self) -> None:
        params = {"scales": torch.nn.Parameter(torch.full((2, 3), math.log(1e-5)))}
        before = params["scales"].detach().clone()
        self.assertEqual(apply_min_extent(params, {}, min_extent_px=0.447)["min_extent_raised_count"], 0)
        state: dict = {}
        _observe(state, [1.0, 1.0], [1.0, 1.0])
        self.assertEqual(apply_min_extent(params, state, min_extent_px=0.0)["min_extent_raised_count"], 0)
        self.assertTrue(torch.equal(params["scales"].detach(), before))

    def test_the_rate_follows_the_population_like_any_strategy_tensor(self) -> None:
        state: dict = {}
        _observe(state, [1.0, 1.0, 1.0], [1.0, 2.0, 4.0])
        rate = state[SAMPLING_RATE_KEY]
        # duplicate row 0, then remove row 1 - what gsplat's ops do to every state tensor
        rate = torch.cat((rate, rate[torch.tensor([0])]))[torch.tensor([True, False, True, True])]
        state[SAMPLING_RATE_KEY] = rate
        self.assertEqual(rate.tolist(), [800.0, 200.0, 800.0])
        params = {"scales": torch.nn.Parameter(torch.full((3, 3), math.log(1e-5)))}
        apply_min_extent(params, state, min_extent_px=0.4)
        self.assertTrue(
            torch.allclose(params["scales"].detach().exp()[:, 0], torch.tensor([0.0005, 0.002, 0.0005]))
        )


class ContractTest(unittest.TestCase):
    def test_the_knob_is_absent_when_off_and_bounded(self) -> None:
        self.assertNotIn("min_extent_px", GeometryRegularizationConfig().to_dict())
        self.assertEqual(GeometryRegularizationConfig(min_extent_px=0.447).to_dict()["min_extent_px"], 0.447)
        with self.assertRaises(ValueError):
            GeometryRegularizationConfig(min_extent_px=-0.1).validate()
        with self.assertRaises(ValueError):
            GeometryRegularizationConfig(min_extent_px=9.0).validate()


if __name__ == "__main__":
    unittest.main()
