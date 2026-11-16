"""The device copy of a training sample holds the values the old CPU-widening path produced."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch

    from cloudstudio_3dgs.training import trainer as trainer_module
except ImportError:  # torch is an optional training dependency
    torch = None


class _Sample:
    def __init__(self) -> None:
        rng = np.random.default_rng(3)
        self.image = rng.integers(0, 256, (9, 7, 3), dtype=np.uint8)
        self.image.setflags(write=False)  # PIL hands back read-only arrays
        self.rgb_mask = rng.random((9, 7)) > 0.3
        self.depth_range_m = rng.random((9, 7)).astype(np.float32) * 30
        self.depth_confidence = rng.random((9, 7)).astype(np.float32)
        self.depth_mask = self.rgb_mask & (rng.random((9, 7)) > 0.5)
        self.sky_mask = (rng.integers(0, 2, (9, 7)) * 255).astype(np.uint8)  # non-bool mask
        self.mono_depth_range_m = np.asfortranarray(rng.random((9, 7)).astype(np.float32))
        self.mono_depth_mask = rng.random((9, 7)) > 0.2
        self.mesh_depth_range_m = None


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class TensorSampleUploadTests(unittest.TestCase):
    def test_values_and_dtypes_match_the_cpu_widening_path(self) -> None:
        sample = _Sample()
        tensors = trainer_module._tensor_sample(sample, torch, "cpu")
        expected = {
            "rgb": torch.as_tensor(np.array(sample.image, copy=True), dtype=torch.float32) / 255.0,
            "rgb_mask": torch.as_tensor(np.array(sample.rgb_mask, copy=True), dtype=torch.bool),
            "range_m": torch.as_tensor(np.array(sample.depth_range_m, copy=True), dtype=torch.float32),
            "confidence": torch.as_tensor(np.array(sample.depth_confidence, copy=True), dtype=torch.float32),
            "depth_mask": torch.as_tensor(np.array(sample.depth_mask, copy=True), dtype=torch.bool),
            "sky_mask": torch.as_tensor(np.array(sample.sky_mask, copy=True), dtype=torch.bool),
            "da2_range_m": torch.as_tensor(np.array(sample.mono_depth_range_m, copy=True), dtype=torch.float32),
            "da2_mask": torch.as_tensor(np.array(sample.mono_depth_mask, copy=True), dtype=torch.bool),
        }
        self.assertEqual(set(tensors), set(expected))
        for key, value in expected.items():
            self.assertEqual(tensors[key].dtype, value.dtype, key)
            self.assertTrue(torch.equal(tensors[key], value), key)


if __name__ == "__main__":
    unittest.main()
