import importlib.util
import unittest

import numpy as np

from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.losses import global_masked_rgb_ssim_loss
from cloudstudio_3dgs.training.presets import (
    available_trainer_presets,
    expand_trainer_preset,
)
from cloudstudio_3dgs.training.trainer import (
    TrainerConfig,
    _render_supervision_loss,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _base_config(preset: str) -> dict:
    return {
        "run_id": f"preset-{preset}",
        "trainer_preset": preset,
        "dataset_manifest": "dataset.json",
        "recording_root": "recording",
        "mask_manifest": "masks.json",
        "mask_root": "masks",
        "split_manifest": "split.json",
        "initialization_ply": "sparse_pc.ply",
        "output_dir": "run",
        "gsplat_lock": "upstream/cloudstudio_trainer.lock.json",
        "require_person_masks": False,
        "lidar_range_weight": 0.0,
    }


class TrainerPresetTests(unittest.TestCase):
    def test_named_presets_are_complete_and_fail_on_contradiction(self) -> None:
        self.assertEqual(
            available_trainer_presets(),
            (
                "gate2_knn_only_v1",
                "gate2_local_ssim_only_v1",
                "gate2_quality_australian_p5_v1",
                "gate2_sh_only_v1",
                "legacy_minimal_v1",
            ),
        )
        for name in available_trainer_presets():
            config = TrainerConfig.from_dict(_base_config(name))
            config.validate()
            self.assertEqual(config.contract_dict()["trainer_preset"], name)

        quality = TrainerConfig.from_dict(
            _base_config("gate2_quality_australian_p5_v1")
        )
        self.assertEqual(quality.color_model, "sh")
        self.assertEqual(quality.sh_degree, 3)
        self.assertEqual(quality.background_color, [1.0, 1.0, 1.0])
        self.assertTrue(quality.exposure_compensation.enabled)
        self.assertEqual(quality.rgb_ssim_mode, "local_gaussian")
        self.assertEqual(quality.metric_scale_calibration.mode, "knn")

        contradictory = _base_config("gate2_quality_australian_p5_v1")
        contradictory["background_color"] = None
        with self.assertRaisesRegex(ValueError, "fixes background_color"):
            expand_trainer_preset(contradictory)

    def test_direct_dataclass_cannot_mislabel_a_preset(self) -> None:
        config = TrainerConfig(
            run_id="mislabelled",
            trainer_preset="gate2_quality_australian_p5_v1",
            dataset_manifest="dataset.json",
            recording_root="recording",
            mask_manifest="masks.json",
            mask_root="masks",
            split_manifest="split.json",
            initialization_ply="sparse_pc.ply",
            output_dir="run",
            gsplat_lock="lock.json",
            require_person_masks=False,
            lidar_range_weight=0.0,
        )
        with self.assertRaisesRegex(ValueError, "does not match fields"):
            config.validate()


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class LegacyPresetLossTests(unittest.TestCase):
    def test_legacy_preset_executes_global_masked_moment_loss(self) -> None:
        import torch

        config = TrainerConfig.from_dict(_base_config("legacy_minimal_v1"))
        prediction = torch.linspace(0.0, 1.0, 16 * 16 * 3).reshape(16, 16, 3)
        target = torch.flip(prediction, dims=(1,))
        mask = torch.ones((16, 16), dtype=torch.bool)
        sample = TrainingSample(
            image_id="legacy",
            rig_frame_id="rig",
            camera_id="left",
            image=(target.numpy() * 255.0).round().astype(np.uint8),
            rgb_mask=mask.numpy(),
            depth_range_m=None,
            depth_confidence=None,
            depth_mask=None,
            depth_cache_path=None,
            c2w=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            width=16,
            height=16,
        )

        class Backend:
            @staticmethod
            def render(params, current, **kwargs):
                return prediction, None, None, {}

        tensors = {
            "rgb": target,
            "rgb_mask": mask,
        }
        _, _, actual, _, _ = _render_supervision_loss(
            backend=Backend(),
            params=None,
            sample=sample,
            tensors=tensors,
            config=config,
        )
        expected = global_masked_rgb_ssim_loss(prediction, target, mask)
        self.assertAlmostEqual(float(actual), float(expected), places=7)
