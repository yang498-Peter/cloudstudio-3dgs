from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_da2_mesh_aligned_tile_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_da2_mesh_aligned_tile_manifest", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Da2MeshAlignmentBuilderTests(unittest.TestCase):
    def test_native_pairs_use_crop_and_valid_mesh_hits(self) -> None:
        relative = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        metric = np.full((2, 2), 7.0, dtype=np.float32)
        valid = np.asarray([[True, False], [True, True]])
        mono, mesh = MODULE.native_da2_mesh_pairs(
            relative,
            metric,
            valid,
            source_shape=(4, 4),
            crop={"x": 1, "y": 1, "width": 2, "height": 2},
        )
        np.testing.assert_array_equal(mono, np.asarray([6.0, 10.0, 11.0]))
        np.testing.assert_array_equal(mesh, np.asarray([7.0, 7.0, 7.0]))


if __name__ == "__main__":
    unittest.main()
