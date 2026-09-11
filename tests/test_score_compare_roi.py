"""ROI-only compare scoring: boxes are applied to the native-size panels, frames
without a box (or with too few region samples) are skipped, and a panel that
does not match the recorded Tile crop is refused instead of silently mis-cropped.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.score_compare_roi import roi_boxes, score_compare_dir, split_panels


def _strip(width, height, gap=8, noise=None):
    """photo | ours | reference, each ``width`` x ``height``; ours gets ``noise`` inside the box."""
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    ours = np.full_like(photo, 128)
    ref = photo.copy()
    if noise is not None:
        y0, y1, x0, x1 = noise
        ours[y0:y1, x0:x1] = photo[y0:y1, x0:x1]
    strip = np.full((height, width * 3 + 2 * gap, 3), 24, np.uint8)
    for i, panel in enumerate((photo, ours, ref)):
        strip[:, i * (width + gap): i * (width + gap) + width] = panel
    return strip


class ScoreCompareRoiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.compare = Path(self.tmp.name) / "armA" / "compare"
        self.compare.mkdir(parents=True)
        self.images = {}

    def tearDown(self):
        self.tmp.cleanup()

    def _imread(self, path):
        return self.images.get(str(path))

    def _write(self, name, sample_id, strip):
        self.images[str(self.compare / name)] = strip
        return {"file": name, "image_id": sample_id, "panels": ["photo", "ours", "reference"]}

    def test_split_panels_uses_the_gap_geometry(self):
        strip = _strip(50, 20)
        panels = split_panels(strip)
        self.assertEqual([p.shape[1] for p in panels], [50, 50, 50])
        np.testing.assert_array_equal(panels[0], panels[2])

    def test_box_is_scored_and_unboxed_frames_are_skipped(self):
        frames = [
            self._write("compare_00_a.png", "img_a::yaw_plus_35", _strip(60, 40, noise=(10, 30, 10, 50))),
            self._write("compare_01_b.png", "img_b::pitch_up_56", _strip(60, 40)),
            self._write("compare_02_c.png", "img_c::yaw_plus_35", _strip(60, 40)),
        ]
        (self.compare / "compare_summary.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
        selection = {"roi_in_crops": [
            {"sample_id": "img_a::yaw_plus_35", "roi": {"crop": {"width": 60, "height": 40}, "x0": 10, "y0": 10, "x1": 50, "y1": 30, "samples_in_crop": 40}},
            {"sample_id": "img_b::pitch_up_56", "roi": None},
            {"sample_id": "img_c::yaw_plus_35", "roi": {"crop": {"width": 60, "height": 40}, "x0": 10, "y0": 10, "x1": 50, "y1": 30, "samples_in_crop": 3}},
        ]}
        result = score_compare_dir(self.compare, roi_boxes(selection), 10, imread=self._imread)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["n_skipped"], 2)
        row = result["frames"][0]
        # inside the box "ours" equals the photo, so the ratio is exactly 1 even though the rest of the panel is flat
        self.assertAlmostEqual(row["ours_over_photo"], 1.0, places=6)
        self.assertAlmostEqual(row["ref_over_photo"], 1.0, places=6)
        reasons = [s["reason"] for s in result["skipped"]]
        self.assertTrue(any("no ROI" in r for r in reasons))
        self.assertTrue(any("3 region samples" in r for r in reasons))

    def test_brightness_matching_removes_a_global_gain(self):
        strip = _strip(60, 40, noise=(10, 30, 10, 50))
        # darken "ours" by a global gain of 0.5: raw Laplacian variance drops 4x, matched stays 1
        ours = strip[:, 68:128].astype(np.float64) * 0.5
        strip[:, 68:128] = ours.astype(np.uint8)
        frames = [self._write("compare_00_a.png", "img_a::yaw_plus_35", strip)]
        (self.compare / "compare_summary.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
        selection = {"roi_in_crops": [
            {"sample_id": "img_a::yaw_plus_35", "roi": {"crop": {"width": 60, "height": 40}, "x0": 10, "y0": 10, "x1": 50, "y1": 30, "samples_in_crop": 40}},
        ]}
        raw = score_compare_dir(self.compare, roi_boxes(selection), 10, imread=self._imread)
        matched = score_compare_dir(self.compare, roi_boxes(selection), 10, imread=self._imread, match_brightness=True)
        self.assertLess(raw["frames"][0]["ours_over_photo"], 0.35)
        self.assertGreater(matched["frames"][0]["ours_over_photo"], 0.9)
        self.assertTrue(matched["brightness_matched"])

    def test_panel_not_matching_the_recorded_crop_is_refused(self):
        frames = [self._write("compare_00_a.png", "img_a::yaw_plus_35", _strip(60, 40))]
        (self.compare / "compare_summary.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
        selection = {"roi_in_crops": [
            {"sample_id": "img_a::yaw_plus_35", "roi": {"crop": {"width": 61, "height": 40}, "x0": 10, "y0": 10, "x1": 50, "y1": 30, "samples_in_crop": 40}},
        ]}
        with self.assertRaises(ValueError):
            score_compare_dir(self.compare, roi_boxes(selection), 10, imread=self._imread)


if __name__ == "__main__":
    unittest.main()
