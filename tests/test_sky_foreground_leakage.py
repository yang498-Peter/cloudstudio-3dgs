from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import SparseDepthMap, sparse_depth_npz_bytes
from tools.audit_sky_foreground_leakage import audit


class SkyForegroundLeakageTests(unittest.TestCase):
    def test_sky_change_on_lidar_foreground_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface = root / "surface"
            composed = root / "composed"
            surface.mkdir()
            composed.mkdir()
            image_id = "frame"
            base = np.full((2, 2, 3), 64, dtype=np.uint8)
            leaked = base.copy()
            leaked[0, 0] = 200
            Image.fromarray(base).save(surface / f"{image_id}_rendered.png")
            Image.fromarray(leaked).save(composed / f"{image_id}_rendered.png")
            np.save(surface / f"{image_id}_alpha.npy", np.full((2, 2), 0.5, np.float32))
            sparse = SparseDepthMap(
                (2, 2),
                np.array([0], dtype=np.int32),
                np.array([2.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([-1], dtype=np.int64),
                np.array([0], dtype=np.int32),
            )
            (surface / f"{image_id}_lidar.npz").write_bytes(
                sparse_depth_npz_bytes(sparse)
            )
            report = audit(
                surface,
                composed,
                maximum_mean_rgb_delta=0.01,
                maximum_fraction_over_0_05=0.02,
                minimum_foreground_alpha=0.9,
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertIn(
                "foreground_alpha_mean_at_least_minimum", report["failed_checks"]
            )
            self.assertGreater(report["summary"]["sky_rgb_delta_mean"], 0.01)


if __name__ == "__main__":
    unittest.main()
