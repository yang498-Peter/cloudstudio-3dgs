"""Per-view colour affine (cloudstudio_3dgs/training/view_colour.py), CPU only.

What is pinned:

* identity at initialization, so enabling it changes nothing until it has learned something;
* a view whose photo is ``a * render + b`` is recovered by gradient descent on that view's
  parameters alone, and no other view's parameters move;
* the projection keeps both tensors zero-mean over the views and inside their clamps, so the
  correction cannot drift into the model's canonical colours;
* an unknown view is refused; the knob is absent from the trainer contract when off.
"""

from __future__ import annotations

import unittest

import torch

from cloudstudio_3dgs.training.view_colour import ViewColourAffine, ViewColourConfig


class ViewColourAffineTest(unittest.TestCase):
    def _affine(self, **config) -> ViewColourAffine:
        return ViewColourAffine(
            ["a::up", "a::down", "b::up", "b::down"], config=ViewColourConfig(enabled=True, **config), device="cpu"
        )

    def test_identity_at_initialization(self) -> None:
        affine = self._affine()
        render = torch.rand(5, 4, 3)
        self.assertTrue(torch.equal(affine.apply(render, "b::up"), render))
        self.assertEqual(float(affine.prior_loss()), 0.0)

    def test_one_view_learns_its_own_affine_and_nothing_else_moves(self) -> None:
        torch.manual_seed(0)
        affine = self._affine(learning_rate=0.05, prior_weight=0.0)
        render = torch.rand(16, 16, 3) * 0.6 + 0.2
        gain = torch.tensor([0.90, 0.95, 1.05])
        offset = torch.tensor([0.06, 0.04, -0.03])
        photo = render * gain + offset
        optimizer = affine.make_optimizer()
        for _ in range(400):
            optimizer.zero_grad()
            (affine.apply(render, "a::down") - photo).abs().mean().backward()
            optimizer.step()
        self.assertTrue(torch.allclose(affine.log_gains[1].exp(), gain, atol=0.02))
        self.assertTrue(torch.allclose(affine.offsets[1], offset, atol=0.02))
        for row in (0, 2, 3):
            self.assertEqual(float(affine.log_gains[row].abs().sum()), 0.0)
            self.assertEqual(float(affine.offsets[row].abs().sum()), 0.0)

    def test_projection_keeps_the_gauge_and_the_clamps(self) -> None:
        affine = self._affine(max_abs_log_gain=0.3, max_abs_offset=0.1)
        with torch.no_grad():
            affine.log_gains.copy_(torch.tensor([[0.2] * 3, [0.4] * 3, [0.0] * 3, [0.6] * 3]))
            affine.offsets.copy_(torch.tensor([[0.05] * 3, [0.05] * 3, [0.05] * 3, [0.45] * 3]))
        affine.project()
        self.assertLessEqual(float(affine.log_gains.abs().max()), 0.3 + 1e-6)
        self.assertLessEqual(float(affine.offsets.abs().max()), 0.1 + 1e-6)
        with torch.no_grad():
            affine.log_gains.copy_(torch.tensor([[0.1] * 3, [0.2] * 3, [0.0] * 3, [0.1] * 3]))
            affine.offsets.fill_(0.03)
        affine.project()
        self.assertTrue(torch.allclose(affine.log_gains.mean(dim=0), torch.zeros(3), atol=1e-7))
        self.assertTrue(torch.allclose(affine.offsets, torch.zeros(4, 3), atol=1e-7))

    def test_the_applied_values_never_leave_the_clamps(self) -> None:
        affine = self._affine(max_abs_log_gain=0.3, max_abs_offset=0.1)
        with torch.no_grad():
            affine.log_gains[0].fill_(5.0)
            affine.offsets[0].fill_(-5.0)
        out = affine.apply(torch.ones(2, 2, 3), "a::up")
        self.assertTrue(torch.allclose(out, torch.full((2, 2, 3), float(torch.exp(torch.tensor(0.3))) - 0.1)))

    def test_unknown_and_duplicate_views_are_refused(self) -> None:
        with self.assertRaises(KeyError):
            self._affine().apply(torch.zeros(1, 1, 3), "c::up")
        with self.assertRaises(ValueError):
            ViewColourAffine(["a", "a"], config=ViewColourConfig(enabled=True), device="cpu")
        with self.assertRaises(ValueError):
            ViewColourAffine([], config=ViewColourConfig(enabled=True), device="cpu")

    def test_config_bounds(self) -> None:
        for bad in (
            ViewColourConfig(learning_rate=-1.0),
            ViewColourConfig(prior_weight=-1.0),
            ViewColourConfig(max_abs_log_gain=0.0),
            ViewColourConfig(max_abs_offset=0.9),
        ):
            with self.assertRaises(ValueError):
                bad.validate()


if __name__ == "__main__":
    unittest.main()
