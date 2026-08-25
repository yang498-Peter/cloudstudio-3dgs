"""Missing supervision must stay fatal; a momentarily empty render must not.

The range losses originally treated both as the same condition. That made the
reference 3DGS densification path die at exactly its first opacity reset - the
mechanism resets every opacity, the render is empty for one step, and the loss
raised on a training state that is both deliberate and published.

The protective intent is worth keeping: if the mask, target or confidence
carries nothing, the dataset or config is broken and training on it would be
meaningless. So the two cases are now separated, and these tests pin both
directions so a later simplification cannot quietly merge them again.
"""

from __future__ import annotations

import unittest

import torch

from cloudstudio_3dgs.training.losses import (
    confidence_weighted_log_range_huber,
    confidence_weighted_range_l1,
)

LOSSES = (confidence_weighted_log_range_huber, confidence_weighted_range_l1)


def _fields(size=8):
    prediction = torch.full((size,), 3.0, requires_grad=True)
    target = torch.full((size,), 3.1)
    confidence = torch.ones(size)
    mask = torch.ones(size, dtype=torch.bool)
    return prediction, target, confidence, mask


class MissingSupervisionStaysFatalTests(unittest.TestCase):
    def test_an_empty_mask_still_raises(self):
        prediction, target, confidence, mask = _fields()
        for loss in LOSSES:
            with self.subTest(loss=loss.__name__):
                with self.assertRaises(ValueError):
                    loss(prediction, target, confidence, torch.zeros_like(mask))

    def test_a_non_positive_target_still_raises(self):
        prediction, target, confidence, mask = _fields()
        for loss in LOSSES:
            with self.subTest(loss=loss.__name__):
                with self.assertRaises(ValueError):
                    loss(prediction, torch.zeros_like(target), confidence, mask)

    def test_zero_confidence_still_raises(self):
        prediction, target, confidence, mask = _fields()
        for loss in LOSSES:
            with self.subTest(loss=loss.__name__):
                with self.assertRaises(ValueError):
                    loss(prediction, target, torch.zeros_like(confidence), mask)


class DegeneratePredictionIsToleratedTests(unittest.TestCase):
    def test_an_empty_render_yields_zero_rather_than_raising(self):
        # What an opacity reset produces: supervision intact, prediction gone.
        _, target, confidence, mask = _fields()
        empty = torch.zeros(8, requires_grad=True)
        for loss in LOSSES:
            with self.subTest(loss=loss.__name__):
                value = loss(empty, target, confidence, mask)
                self.assertEqual(float(value), 0.0)

    def test_the_zero_stays_attached_to_the_graph(self):
        # A detached constant would break backward on that step; the trainer
        # calls .backward() unconditionally.
        _, target, confidence, mask = _fields()
        empty = torch.zeros(8, requires_grad=True)
        for loss in LOSSES:
            with self.subTest(loss=loss.__name__):
                value = loss(empty, target, confidence, mask)
                self.assertTrue(value.requires_grad)
                value.backward()
                self.assertIsNotNone(empty.grad)
                empty.grad = None

    def test_a_partially_empty_render_uses_the_pixels_that_survive(self):
        prediction, target, confidence, mask = _fields()
        partial = prediction.detach().clone()
        partial[:6] = 0.0                      # only two pixels still render
        partial.requires_grad_(True)
        for loss in LOSSES:
            with self.subTest(loss=loss.__name__):
                value = loss(partial, target, confidence, mask)
                self.assertGreater(float(value), 0.0)

    def test_normal_steps_are_unchanged(self):
        prediction, target, confidence, mask = _fields()
        for loss in LOSSES:
            with self.subTest(loss=loss.__name__):
                value = loss(prediction, target, confidence, mask)
                self.assertGreater(float(value), 0.0)
                self.assertTrue(torch.isfinite(value))


if __name__ == "__main__":
    unittest.main()
