from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.training.optimization_audit import (
    component_gradient_audit,
    gradient_norms,
    parameter_update_norms,
    point_to_plane_drift_summary,
    shortest_axis_normals,
)


class OptimizationAuditTests(unittest.TestCase):
    def test_component_gradient_audit_reports_alignment_and_conflict(self) -> None:
        import torch

        means = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        report = component_gradient_audit(
            {"means": means},
            {
                "rgb": means.square().sum(),
                "range": -means.square().sum(),
                "normal": None,
            },
        )
        self.assertGreater(report["gradient_norms"]["rgb"]["means"]["l2"], 0.0)
        self.assertAlmostEqual(
            report["pairwise_cosine"]["rgb__range"]["means"], -1.0
        )
        self.assertIsNone(report["pairwise_cosine"]["rgb__normal"]["means"])

    def test_empty_shn_gradient_and_update_are_zero(self) -> None:
        import torch

        parameter = torch.nn.Parameter(torch.empty((2, 0, 3)))
        parameter.grad = torch.empty_like(parameter)
        gradients = gradient_norms({"shN": parameter})
        updates = parameter_update_norms({"shN": parameter.detach().clone()}, {"shN": parameter})
        self.assertEqual(gradients["shN"]["max_abs"], 0.0)
        self.assertEqual(updates["shN"]["changed_count"], 0)

    def test_identity_quaternion_uses_shortest_scale_axis(self) -> None:
        normals = shortest_axis_normals(
            np.asarray([[2.0, 1.0, 0.5], [0.25, 1.0, 2.0]]),
            np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        )
        np.testing.assert_allclose(normals[0], [0, 0, 1])
        np.testing.assert_allclose(normals[1], [1, 0, 0])

    def test_point_to_plane_ignores_tangent_motion(self) -> None:
        initial = np.zeros((3, 3), dtype=np.float64)
        normals = np.asarray([[0, 0, 1]] * 3, dtype=np.float64)
        current = np.asarray([[1, 0, 0], [0, 2, 0.01], [0, 0, 0.20]])
        report = point_to_plane_drift_summary(current, initial, normals)
        self.assertEqual(report["over_5cm_count"], 1)
        self.assertEqual(report["over_10cm_count"], 1)
        self.assertAlmostEqual(report["max_m"], 0.20)


if __name__ == "__main__":
    unittest.main()
