import importlib.util
import unittest

import numpy as np

from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.golden_eval import (
    GoldenEvaluationConfig,
    evaluate_golden_views,
    golden_image_ids,
    is_golden_improvement,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _split_manifest() -> dict:
    return {
        "split_manifest_sha256": "split-sha",
        "splits": {
            "train": ["train-left", "train-right"],
            "val": ["golden-left", "golden-right", "other-val"],
        },
        "golden_views": [{"image_ids": ["golden-left", "golden-right"]}],
    }


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class GoldenEvaluationTests(unittest.TestCase):
    def test_golden_evaluation_uses_signed_order_and_renderer_background(self) -> None:
        import torch

        def sample(image_id: str, pixel: int) -> TrainingSample:
            return TrainingSample(
                image_id=image_id,
                rig_frame_id=image_id,
                camera_id="left",
                image=np.full((16, 16, 3), pixel, dtype=np.uint8),
                rgb_mask=np.ones((16, 16), dtype=bool),
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

        class Dataset:
            image_ids = ["other-val", "golden-right", "golden-left"]

            def __init__(self) -> None:
                self.samples = [sample("other-val", 90), sample("golden-right", 60), sample("golden-left", 30)]

            def __getitem__(self, index: int) -> TrainingSample:
                return self.samples[index]

        class Backend:
            def __init__(self) -> None:
                self.torch = torch
                self.backgrounds: list[tuple[float, float, float] | None] = []

            def render(self, params, current, *, with_range, background_rgb=None):
                self.backgrounds.append(background_rgb)
                self.assert_no_range(with_range)
                value = float(current.image[0, 0, 0]) / 255.0 + 0.05
                return torch.full((16, 16, 3), value), None, None, None

            @staticmethod
            def assert_no_range(with_range: bool) -> None:
                if with_range:
                    raise AssertionError("fixture does not contain depth")

        backend = Backend()
        report = evaluate_golden_views(
            backend=backend,
            params=None,
            dataset=Dataset(),
            split_manifest=_split_manifest(),
            completed_steps=10,
            background_rgb=(1.0, 1.0, 1.0),
        )
        self.assertEqual(report["image_ids"], ["golden-left", "golden-right"])
        self.assertEqual([frame["image_id"] for frame in report["frames"]], report["image_ids"])
        self.assertGreater(float(report["summary"]["psnr_db_mean"]), 20.0)
        self.assertEqual(backend.backgrounds, [(1.0, 1.0, 1.0)] * 2)
        self.assertTrue(is_golden_improvement(report, None, min_psnr_improvement_db=0.001))
        self.assertFalse(is_golden_improvement(report, report, min_psnr_improvement_db=0.001))

    def test_golden_contract_rejects_empty_or_non_validation_views(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in the validation"):
            golden_image_ids(
                {
                    "splits": {"val": ["val"]},
                    "golden_views": [{"image_ids": ["train"]}],
                }
            )
        with self.assertRaisesRegex(ValueError, "interval"):
            GoldenEvaluationConfig(every=0).validate()
