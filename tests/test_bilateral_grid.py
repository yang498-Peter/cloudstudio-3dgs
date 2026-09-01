import unittest

import torch

from cloudstudio_3dgs.training.bilateral_grid import (
    BilateralGridConfig,
    BilateralGridCorrector,
)


class BilateralGridTests(unittest.TestCase):
    def test_identity_initialization_and_gradients(self) -> None:
        corrector = BilateralGridCorrector(
            ["a", "b"],
            camera_by_image={"a": "left", "b": "right"},
            config=BilateralGridConfig(enabled=True),
            device="cpu",
        )
        rgb = torch.rand(9, 11, 3, requires_grad=True)
        output = corrector.apply(rgb, "a")
        torch.testing.assert_close(output, rgb, atol=2e-6, rtol=2e-6)
        output.square().mean().backward()
        self.assertGreater(float(corrector.grid.grad.abs().sum()), 0.0)

    def test_schedule_warms_up_then_decays(self) -> None:
        config = BilateralGridConfig(enabled=True)
        self.assertAlmostEqual(config.learning_rate_for_step(step=0, total_steps=3000), 0.00002)
        self.assertAlmostEqual(config.learning_rate_for_step(step=99, total_steps=3000), 0.002)
        self.assertAlmostEqual(config.learning_rate_for_step(step=2999, total_steps=3000), 0.00002)

    def test_affine_bias_cannot_paint_fully_transparent_background(self) -> None:
        corrector = BilateralGridCorrector(
            ["a"],
            camera_by_image={"a": "left"},
            config=BilateralGridConfig(enabled=True),
            device="cpu",
        )
        with torch.no_grad():
            corrector.grid[:, 3] = 0.4
            corrector.grid[:, 7] = -0.2
            corrector.grid[:, 11] = 0.3
        alpha = torch.zeros(5, 7)
        black = torch.zeros(5, 7, 3)
        output_black = corrector.apply(
            black, "a", alpha=alpha, background_rgb=(0.0, 0.0, 0.0)
        )
        torch.testing.assert_close(output_black, black)
        white = torch.ones(5, 7, 3)
        output_white = corrector.apply(
            white, "a", alpha=alpha, background_rgb=(1.0, 1.0, 1.0)
        )
        torch.testing.assert_close(output_white, white)


if __name__ == "__main__":
    unittest.main()
