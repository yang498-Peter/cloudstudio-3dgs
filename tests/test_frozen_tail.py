from __future__ import annotations

import unittest

import torch

from cloudstudio_3dgs.training.trainer import _register_frozen_tail_hooks


class FrozenTailTests(unittest.TestCase):
    def test_tail_gradients_are_zeroed_and_head_gradients_survive(self) -> None:
        params = {
            "means": torch.nn.Parameter(torch.randn(6, 3)),
            "opacities": torch.nn.Parameter(torch.randn(6)),
        }
        registered = _register_frozen_tail_hooks(params, 4)
        self.assertEqual(registered, 2)

        loss = (params["means"] * 2.0).sum() + (params["opacities"] * 3.0).sum()
        loss.backward()

        self.assertTrue(torch.all(params["means"].grad[:4] == 2.0))
        self.assertTrue(torch.all(params["means"].grad[4:] == 0.0))
        self.assertTrue(torch.all(params["opacities"].grad[:4] == 3.0))
        self.assertTrue(torch.all(params["opacities"].grad[4:] == 0.0))

    def test_optimizer_never_moves_the_frozen_tail(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(5, 2))
        _register_frozen_tail_hooks({"p": parameter}, 3)
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        for _ in range(20):
            optimizer.zero_grad()
            (parameter - 1.0).square().sum().backward()
            optimizer.step()
        self.assertTrue(torch.all(parameter.detach()[3:] == 0.0))
        self.assertGreater(float(parameter.detach()[:3].min()), 0.5)

    def test_short_parameter_passes_through_whole(self) -> None:
        parameter = torch.nn.Parameter(torch.randn(2))
        _register_frozen_tail_hooks({"p": parameter}, 10)
        parameter.sum().backward()
        self.assertTrue(torch.all(parameter.grad == 1.0))


if __name__ == "__main__":
    unittest.main()
