import unittest

import torch

from cloudstudio_3dgs.training.backend import gaussian_shortest_axis_normals


class GaussianShortestAxisNormalTests(unittest.TestCase):
    def test_selects_current_shortest_rotation_column(self) -> None:
        scales = torch.tensor([[0.0, -2.0, -1.0]], dtype=torch.float32)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        normal = gaussian_shortest_axis_normals(torch, scales, quat)
        torch.testing.assert_close(normal, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_normal_loss_backpropagates_to_quaternion(self) -> None:
        scales = torch.tensor([[-2.0, 0.0, -1.0]], dtype=torch.float32)
        quat = torch.tensor([[0.98, 0.0, 0.2, 0.0]], requires_grad=True)
        normal = gaussian_shortest_axis_normals(torch, scales, quat)
        loss = torch.abs(normal - torch.tensor([[0.0, 0.0, 1.0]])).sum()
        loss.backward()
        self.assertIsNotNone(quat.grad)
        self.assertGreater(float(torch.linalg.vector_norm(quat.grad)), 0.0)


if __name__ == "__main__":
    unittest.main()
