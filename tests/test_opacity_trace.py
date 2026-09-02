"""The opacity-trajectory trace window: validation and dump cadence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cloudstudio_3dgs.training.trainer import _maybe_dump_opacity_trace


class OpacityTraceDumpTests(unittest.TestCase):
    def _params(self, n: int = 4):
        return {
            "opacities": torch.nn.Parameter(torch.linspace(-2.0, 2.0, n)),
            "means": torch.nn.Parameter(torch.zeros(n, 3)),
            "scales": torch.nn.Parameter(torch.zeros(n, 3)),
        }

    def test_dumps_only_inside_the_window_on_the_cadence(self) -> None:
        trace = {"start_step": 600, "stop_step": 699, "every": 5}
        state = {
            "count": torch.arange(4, dtype=torch.int32),
            "grad2d": torch.zeros(4),
            "_cloudstudio_birth_kind": torch.tensor([0, 1, 2, 0], dtype=torch.int8),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for completed in (599, 600, 601, 605, 610, 698, 699, 700):
                _maybe_dump_opacity_trace(
                    trace,
                    completed=completed,
                    params=self._params(),
                    state=state,
                    output_dir=root,
                )
            names = sorted(p.name for p in (root / "opacity_trace").glob("*.npz"))
            self.assertEqual(
                names,
                ["step_00600.npz", "step_00605.npz", "step_00610.npz", "step_00699.npz"],
            )
            with np.load(root / "opacity_trace" / "step_00600.npz") as first:
                self.assertIn("means", first.files)
                self.assertIn("birth_kind", first.files)
                np.testing.assert_allclose(first["opacities"], np.linspace(-2.0, 2.0, 4))
            with np.load(root / "opacity_trace" / "step_00605.npz") as later:
                self.assertNotIn("means", later.files)
                self.assertEqual(int(later["count"][3]), 3)


if __name__ == "__main__":
    unittest.main()
