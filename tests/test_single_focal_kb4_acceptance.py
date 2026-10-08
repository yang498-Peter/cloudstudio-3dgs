"""When a shared single-focal KB4 intrinsic solve counts as failed.

The solver minimises a Huber cost. The old acceptance test required the plain RMSE not to rise by
more than 1e-9 px, which a converged solve violates by float noise near a fixed point: the UK
capture's independent AT died after 112 minutes of outer iterations on "Both `ftol` and `xtol`
termination conditions are satisfied.; RMSE 0.913865 -> 0.913865 px". What is pinned: a solve is
refused when it failed, went non-finite, raised the Huber cost, or raised the RMSE by anything
pixel-relevant - and not for float noise on a converged step.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.optimize import least_squares

from cloudstudio_3dgs.ba.single_focal_kb4 import (
    RMSE_RISE_TOLERANCE_PX,
    _huber_cost,
    intrinsic_solve_accepted,
)

SCALE = 2.0


def _residuals(inlier: float, outlier: float) -> np.ndarray:
    return np.asarray([inlier] * 100 + [outlier], dtype=np.float64)


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


class IntrinsicSolveAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.initial = _residuals(0.5, 10.0)
        self.params = np.ones(7)

    def _outlier_for_rmse_rise(self, rise: float) -> float:
        # inliers shrink 0.5 -> 0.49; grow the outlier until the RMSE has risen by `rise`
        target = (_rmse(self.initial) + rise) ** 2 * 101 - 100 * 0.49**2
        return math.sqrt(target)

    def test_float_noise_on_a_converged_step_is_accepted(self) -> None:
        final = _residuals(0.49, self._outlier_for_rmse_rise(5e-7))
        self.assertLess(_huber_cost(final, SCALE), _huber_cost(self.initial, SCALE))
        self.assertGreater(_rmse(final), _rmse(self.initial) + 1e-9)  # the old test refused this
        self.assertTrue(intrinsic_solve_accepted(True, self.params, self.initial, final, SCALE))

    def test_a_pixel_relevant_rmse_rise_is_refused_even_at_lower_huber_cost(self) -> None:
        final = _residuals(0.49, self._outlier_for_rmse_rise(10 * RMSE_RISE_TOLERANCE_PX))
        self.assertLess(_huber_cost(final, SCALE), _huber_cost(self.initial, SCALE))
        self.assertFalse(intrinsic_solve_accepted(True, self.params, self.initial, final, SCALE))

    def test_a_higher_huber_cost_is_refused(self) -> None:
        final = self.initial * 1.001
        self.assertFalse(intrinsic_solve_accepted(True, self.params, self.initial, final, SCALE))

    def test_an_unsuccessful_or_non_finite_solve_is_refused(self) -> None:
        better = self.initial * 0.5
        self.assertFalse(intrinsic_solve_accepted(False, self.params, self.initial, better, SCALE))
        bad = self.params.copy()
        bad[0] = np.nan
        self.assertFalse(intrinsic_solve_accepted(True, bad, self.initial, better, SCALE))

    def test_an_unchanged_solve_is_accepted(self) -> None:
        self.assertTrue(intrinsic_solve_accepted(True, self.params, self.initial, self.initial.copy(), SCALE))

    def test_the_cost_is_the_one_scipy_minimises(self) -> None:
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 1.0, 50)
        y = 3.0 * x + 1.0 + rng.normal(0.0, 0.1, size=x.shape)
        y[::10] += 5.0  # outliers in the Huber tail

        def residuals(p: np.ndarray) -> np.ndarray:
            return p[0] * x + p[1] - y

        result = least_squares(residuals, np.zeros(2), loss="huber", f_scale=SCALE)
        self.assertAlmostEqual(_huber_cost(result.fun, SCALE), float(result.cost), places=9)


if __name__ == "__main__":
    unittest.main()
