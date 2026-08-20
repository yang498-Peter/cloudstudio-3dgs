from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cloudstudio_3dgs.training.rig_pose import (
    RigPoseRefinementConfig,
    RigPoseRefiner,
    build_pose_refinement_report,
)
from cloudstudio_3dgs.training.checkpoint import load_checkpoint, save_checkpoint


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class RigPoseRefinementTests(unittest.TestCase):
    def test_one_shared_delta_per_rig_preserves_stereo_baseline(self) -> None:
        import torch

        config = RigPoseRefinementConfig(enabled=True)
        refiner = RigPoseRefiner(
            ["rig_001", "rig_001", "rig_002"],
            config=config,
            device="cpu",
            rig_frame_centers={
                "rig_001": np.asarray([100.0, 200.0, 3.0]),
                "rig_002": np.asarray([101.0, 200.0, 3.0]),
            },
        )
        self.assertEqual(refiner.rig_frame_ids, ("rig_001", "rig_002"))

        left = torch.eye(4, dtype=torch.float32)
        right = torch.eye(4, dtype=torch.float32)
        left[:3, 3] = torch.tensor([100.0, 199.95, 3.0])
        right[:3, 3] = torch.tensor([100.0, 200.05, 3.0])
        with torch.no_grad():
            refiner.deltas[0] = torch.tensor(
                [0.08, -0.03, 0.02, 0.04, -0.02, 0.01], dtype=refiner.deltas.dtype
            )
        corrected_left = refiner.apply("rig_001", left)
        corrected_right = refiner.apply("rig_001", right)
        original_baseline = torch.linalg.inv(left) @ right
        corrected_baseline = torch.linalg.inv(corrected_left) @ corrected_right

        torch.testing.assert_close(corrected_baseline, original_baseline, atol=2e-5, rtol=1e-6)
        corrected_center = (corrected_left[:3, 3] + corrected_right[:3, 3]) / 2.0
        torch.testing.assert_close(
            corrected_center,
            torch.tensor([100.08, 199.97, 3.02]),
            atol=2e-5,
            rtol=1e-6,
        )
        corrected_left[:3, 3].sum().backward()
        self.assertGreater(float(refiner.deltas.grad.abs().sum()), 0.0)

    def test_synthetic_rigid_correction_converges(self) -> None:
        import torch

        config = RigPoseRefinementConfig(
            enabled=True,
            learning_rate=5e-2,
            translation_prior_weight=0.0,
            rotation_prior_weight=0.0,
        )
        refiner = RigPoseRefiner(["rig_000"], config=config, device="cpu")
        optimizer = refiner.make_optimizer()
        original = torch.eye(4)
        target_delta = torch.tensor([0.12, -0.04, 0.03, 0.05, -0.03, 0.02])
        target = refiner.delta_to_matrix(target_delta).detach()
        initial = None
        final = None
        for _ in range(250):
            prediction = refiner.apply("rig_000", original)
            loss = torch.mean((prediction - target) ** 2)
            if initial is None:
                initial = float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final = float(loss.detach())

        assert initial is not None and final is not None
        self.assertLess(final, initial * 1e-4)
        np.testing.assert_allclose(
            refiner.deltas.detach().numpy()[0], target_delta.numpy(), atol=2e-3
        )

    def test_no_improvement_or_excessive_correction_fails_closed(self) -> None:
        config = RigPoseRefinementConfig(
            enabled=True,
            minimum_loss_improvement_fraction=0.0,
            maximum_translation_m=0.2,
            maximum_rotation_deg=2.0,
        )
        deltas = np.asarray([[0.05, 0.0, 0.0, 0.0, 0.0, np.deg2rad(0.5)]])
        accepted = build_pose_refinement_report(
            ["rig_000"], deltas, loss_before=1.0, loss_after=0.8, config=config
        )
        self.assertTrue(accepted["candidate_accepted"])
        self.assertEqual(accepted["published_pose_set"], "refined")

        no_improvement = build_pose_refinement_report(
            ["rig_000"], deltas, loss_before=1.0, loss_after=1.0, config=config
        )
        self.assertFalse(no_improvement["candidate_accepted"])
        self.assertEqual(no_improvement["published_pose_set"], "original")
        self.assertEqual(no_improvement["gates"]["loss_improvement"]["status"], "FAIL")

        excessive = deltas.copy()
        excessive[0, 0] = 0.25
        out_of_bounds = build_pose_refinement_report(
            ["rig_000"], excessive, loss_before=1.0, loss_after=0.5, config=config
        )
        self.assertFalse(out_of_bounds["candidate_accepted"])
        self.assertEqual(out_of_bounds["gates"]["correction_bounds"]["status"], "FAIL")

    def test_unknown_rig_fails_closed(self) -> None:
        import torch

        refiner = RigPoseRefiner(
            ["rig_000"], config=RigPoseRefinementConfig(enabled=True), device="cpu"
        )
        with self.assertRaisesRegex(ValueError, "unknown Rig Frame"):
            refiner.apply("rig_missing", torch.eye(4))

    def test_checkpoint_restores_pose_delta_and_optimizer(self) -> None:
        import torch

        gaussian = torch.nn.ParameterDict(
            {"value": torch.nn.Parameter(torch.tensor([1.0]))}
        )
        gaussian_optimizers = {"value": torch.optim.Adam([gaussian["value"]], lr=0.1)}
        refiner = RigPoseRefiner(
            ["rig_000"], config=RigPoseRefinementConfig(enabled=True), device="cpu"
        )
        pose_optimizer = refiner.make_optimizer()
        with torch.no_grad():
            refiner.deltas[0] = torch.tensor([0.1, -0.2, 0.3, 0.01, -0.02, 0.03])
        generator = torch.Generator().manual_seed(19)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "pose.pt"
            save_checkpoint(
                checkpoint,
                step=7,
                identity={"dataset": "synthetic"},
                params=gaussian,
                optimizers=gaussian_optimizers,
                strategy_state={},
                sampler_state=generator.get_state(),
                training_state={"last_metrics": {}, "initial_loss": 1.0, "best_loss": 0.5},
                auxiliary_params={"rig_pose_deltas": refiner.deltas},
                auxiliary_optimizers={"rig_pose_deltas": pose_optimizer},
            )
            with torch.no_grad():
                refiner.deltas.zero_()
            load_checkpoint(
                checkpoint,
                expected_identity={"dataset": "synthetic"},
                params=gaussian,
                optimizers=gaussian_optimizers,
                map_location="cpu",
                auxiliary_params={"rig_pose_deltas": refiner.deltas},
                auxiliary_optimizers={"rig_pose_deltas": pose_optimizer},
            )

        torch.testing.assert_close(
            refiner.deltas.detach(),
            torch.tensor([[0.1, -0.2, 0.3, 0.01, -0.02, 0.03]]),
        )


if __name__ == "__main__":
    unittest.main()
