"""tools/checkpoint_morphology.py on synthetic gaussians (no torch needed).

The PLY route goes through the repository's own reader so the tool measures
an exported delivery on a CPU host exactly like a checkpoint on machine B.
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.checkpoint_morphology import (
    REFERENCE_TARGETS,
    compute_morphology,
    format_report,
    load_ply,
    main,
)


def _write_gaussian_ply(path: Path, log_scales: np.ndarray, logit_opacity: np.ndarray) -> None:
    count = log_scales.shape[0]
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {count}\n" + "".join(f"property float {n}\n" for n in names) + "end_header\n"
    rows = np.zeros((count, len(names)), dtype=np.float32)
    rows[:, 6] = logit_opacity
    rows[:, 7:10] = log_scales
    rows[:, 10] = 1.0
    path.write_bytes(header.encode("ascii") + rows.tobytes())


class ComputeMorphologyTests(unittest.TestCase):
    def test_known_axes_and_opacities(self) -> None:
        # Five gaussians, axes given unsorted so the sort per gaussian is exercised.
        scales = np.array(
            [
                [0.004, 0.001, 0.002],
                [0.002, 0.004, 0.001],
                [0.001, 0.002, 0.004],
                [0.004, 0.002, 0.001],
                [0.060, 0.001, 0.002],
            ],
            dtype=np.float32,
        )
        opacities = np.array([0.05, 0.2, 0.5, 0.95, 0.99], dtype=np.float32)
        stats = compute_morphology(scales, opacities)
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["sampled"], 5)
        self.assertAlmostEqual(stats["short_p50_mm"], 1.0, places=4)
        self.assertAlmostEqual(stats["mid_p50_mm"], 2.0, places=4)
        self.assertAlmostEqual(stats["long_p50_mm"], 4.0, places=4)
        self.assertAlmostEqual(stats["max_min_p50"], 4.0, places=4)
        self.assertAlmostEqual(stats["max_mid_p50"], 2.0, places=4)
        self.assertAlmostEqual(stats["opacity_p50"], 0.5, places=6)
        self.assertAlmostEqual(stats["opacity_frac_lt_0_1"], 0.2, places=6)
        self.assertAlmostEqual(stats["opacity_frac_gt_0_9"], 0.4, places=6)
        self.assertAlmostEqual(stats["long_frac_gt_10mm"], 0.2, places=6)
        self.assertAlmostEqual(stats["long_frac_gt_20mm"], 0.2, places=6)
        self.assertAlmostEqual(stats["long_frac_gt_50mm"], 0.2, places=6)
        self.assertGreater(stats["long_p95_mm"], 4.0)

    def test_subsample_is_deterministic_and_bounded(self) -> None:
        rng = np.random.default_rng(1)
        scales = np.exp(rng.normal(-6.0, 0.5, size=(5000, 3))).astype(np.float32)
        opacities = rng.uniform(0.0, 1.0, size=5000).astype(np.float32)
        first = compute_morphology(scales, opacities, sample_limit=1000)
        second = compute_morphology(scales, opacities, sample_limit=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["sampled"], 1000)
        full = compute_morphology(scales, opacities)
        self.assertEqual(full["sampled"], 5000)
        self.assertAlmostEqual(first["long_p50_mm"], full["long_p50_mm"], delta=0.2)

    def test_rejects_mismatched_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_morphology(np.zeros((3, 3)), np.zeros(2))
        with self.assertRaises(ValueError):
            compute_morphology(np.zeros((0, 3)), np.zeros(0))

    def test_report_layout_carries_reference_targets(self) -> None:
        stats = compute_morphology(np.full((4, 3), 0.002, dtype=np.float32), np.full(4, 0.3, dtype=np.float32))
        text = format_report("armX", 20000, stats)
        lines = text.splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], "== armX  step 20000  N=4")
        self.assertIn(f"[{REFERENCE_TARGETS['short_p50_mm']}]", lines[1])
        self.assertIn("max/min p50 1.00 [10.2]", lines[2])
        self.assertIn("opacity p50 0.300 [0.197]", lines[3])
        self.assertTrue(lines[4].startswith("  long>10mm 0.000"))


class PlyRouteTests(unittest.TestCase):
    def test_ply_is_read_with_repo_reader_and_json_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ply = root / "delivery.ply"
            log_scales = np.log(np.array([[0.001, 0.002, 0.004]] * 3, dtype=np.float32))
            logit = np.log(np.array([0.2, 0.5, 0.8]) / (1 - np.array([0.2, 0.5, 0.8]))).astype(np.float32)
            _write_gaussian_ply(ply, log_scales, logit)
            step, scales, opacities = load_ply(ply)
            self.assertEqual(step, -1)
            np.testing.assert_allclose(scales, np.exp(log_scales), rtol=1e-5)
            np.testing.assert_allclose(opacities, [0.2, 0.5, 0.8], atol=1e-5)

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main([str(ply), "--label", "merged_x", "--json", str(root / "morph.json")])
            self.assertEqual(code, 0)
            self.assertTrue(out.getvalue().startswith("== merged_x  step -1  N=3\n"))
            record = json.loads((root / "morph.json").read_text(encoding="utf-8"))
            self.assertEqual(record["label"], "merged_x")
            self.assertEqual(record["stats"]["count"], 3)
            self.assertAlmostEqual(record["stats"]["opacity_p50"], 0.5, places=5)
            self.assertEqual(record["reference_targets"], REFERENCE_TARGETS)
            self.assertFalse((root / "morph.json.tmp").exists())

    def test_missing_path_exits_two(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = main([str(Path(tempfile.gettempdir()) / "definitely_missing.pt")])
        self.assertEqual(code, 2)
        self.assertIn("not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
