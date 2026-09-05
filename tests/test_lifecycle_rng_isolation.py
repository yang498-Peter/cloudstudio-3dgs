"""Diagnostics must not advance the generator that decides topology.

`_opacity_summary` subsamples large parent sets with `torch.randperm` and is
called inside `_grow_mipmap` *before* the split draws its random offsets. On
house0305 G9 the clone-parent set exceeded the 1M subsample threshold at 42
refine events on Tile_0, so a telemetry read silently changed the stream the
split consumed. Same recipe, same seed, different population is the symptom
this pins down.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised on the CPU channel
    torch = None

from cloudstudio_3dgs.training.default_strategy_adapter import DefaultStrategyAdapter


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class TelemetryRngIsolationTests(unittest.TestCase):
    def _global_stream_signature(self) -> tuple:
        # Read both the CPU and (when present) CUDA global generators; the
        # split offsets live on whichever device the parameters live on.
        signature = [torch.get_rng_state().clone()]
        if torch.cuda.is_available():
            signature.extend(state.clone() for state in torch.cuda.get_rng_state_all())
        return tuple(signature)

    def _assert_same_stream(self, before: tuple, after: tuple) -> None:
        self.assertEqual(len(before), len(after))
        for index, (lhs, rhs) in enumerate(zip(before, after)):
            self.assertTrue(torch.equal(lhs, rhs), f"generator {index} advanced")

    def test_telemetry_does_not_advance_training_rng(self):
        torch.manual_seed(7)
        values = torch.rand(1_000_001)
        before = self._global_stream_signature()
        summary = DefaultStrategyAdapter._opacity_summary(values)
        self._assert_same_stream(before, self._global_stream_signature())
        self.assertIsNotNone(summary)
        self.assertIn("p50", summary)

    def test_telemetry_does_not_advance_training_rng_on_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA + gsplat required for the real rasterizer")
        torch.manual_seed(7)
        values = torch.rand(1_000_001, device="cuda")
        before = self._global_stream_signature()
        DefaultStrategyAdapter._opacity_summary(values)
        self._assert_same_stream(before, self._global_stream_signature())

    def test_subsampled_summary_is_repeatable_across_calls(self):
        # The summary has its own generator, so two reads of the same tensor
        # agree bit-for-bit instead of drifting with whatever the trainer
        # happened to draw in between.
        values = torch.rand(1_000_001)
        first = DefaultStrategyAdapter._opacity_summary(values)
        torch.rand(12345)  # perturb the global stream between reads
        second = DefaultStrategyAdapter._opacity_summary(values)
        self.assertEqual(first, second)

    def test_small_summary_reads_the_full_tensor(self):
        values = torch.tensor([0.1, 0.5, 0.95, 1.0])
        summary = DefaultStrategyAdapter._opacity_summary(values)
        self.assertAlmostEqual(summary["frac_saturated"], 0.25)
        self.assertAlmostEqual(summary["frac_above_0p9"], 0.5)


if __name__ == "__main__":
    unittest.main()
