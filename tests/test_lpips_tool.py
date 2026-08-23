"""Synthetic CPU tests for tools/compute_lpips.py.

Everything runs on CPU with generated images -- no GPU, no run artifacts. Tests
that need a pretrained backbone are skipped (not failed) when no weights are
available, so an offline box reports "skipped" rather than a fake green; the
mask-policy unit test is weight-free and always runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.compute_lpips import (
    MASK_POLICY,
    LpipsScorer,
    WeightsUnavailableError,
    apply_mask_policy,
    frame_ids,
    score_run,
    to_lpips_tensor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scorer_or_none():
    try:
        return LpipsScorer(net="alex", device="cpu")
    except (WeightsUnavailableError, Exception):  # noqa: BLE001
        return None


_SCORER = _scorer_or_none()
_HAVE_WEIGHTS = _SCORER is not None
_SKIP_REASON = "no pretrained perceptual backbone available on this machine"


def _rng_image(seed: int, size: int = 64) -> np.ndarray:
    """Structured (not pure-noise) image so perceptual features have content."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = np.stack(
        [
            0.5 + 0.4 * np.sin(6.0 * np.pi * xx),
            0.5 + 0.4 * np.cos(5.0 * np.pi * yy),
            0.5 + 0.3 * np.sin(4.0 * np.pi * (xx + yy)),
        ],
        axis=-1,
    )
    base += 0.05 * rng.standard_normal(base.shape).astype(np.float32)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _add_noise(image: np.ndarray, sigma: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image + sigma * rng.standard_normal(image.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def _write_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(path)


class MaskPolicyTest(unittest.TestCase):
    """Weight-free: the mask policy itself, independent of any backbone."""

    def test_invalid_pixels_set_to_fill_in_place(self):
        image = _rng_image(1, size=16)
        mask = np.ones((16, 16), dtype=bool)
        mask[4:8, 4:8] = False
        out = apply_mask_policy(image, mask, fill=1.0)
        self.assertTrue(np.all(out[4:8, 4:8] == 1.0))
        np.testing.assert_allclose(out[mask], image[mask])
        # input is not mutated
        self.assertFalse(np.all(image[4:8, 4:8] == 1.0))

    def test_invalid_region_content_is_erased_from_both_inputs(self):
        """Whatever was in the invalid region cannot influence the comparison."""
        mask = np.ones((16, 16), dtype=bool)
        mask[2:9, 3:11] = False
        rendered = _rng_image(2, size=16)
        reference = _rng_image(3, size=16)

        # Two different garbage fills of the same invalid region.
        rendered_a, reference_a = rendered.copy(), reference.copy()
        rendered_a[~mask] = 0.0
        reference_a[~mask] = 0.13
        rendered_b, reference_b = rendered.copy(), reference.copy()
        rendered_b[~mask] = 0.97
        reference_b[~mask] = 0.42

        np.testing.assert_array_equal(
            apply_mask_policy(rendered_a, mask), apply_mask_policy(rendered_b, mask)
        )
        np.testing.assert_array_equal(
            apply_mask_policy(reference_a, mask), apply_mask_policy(reference_b, mask)
        )
        # And after the policy the invalid region is identical across the pair.
        np.testing.assert_array_equal(
            apply_mask_policy(rendered_a, mask)[~mask],
            apply_mask_policy(reference_a, mask)[~mask],
        )

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            apply_mask_policy(np.zeros((8, 8), dtype=np.float32), np.ones((8, 8), dtype=bool))
        with self.assertRaises(ValueError):
            apply_mask_policy(np.zeros((8, 8, 3), dtype=np.float32), np.ones((4, 4), dtype=bool))

    def test_tensor_normalisation_to_minus_one_one(self):
        tensor = to_lpips_tensor(np.zeros((4, 5, 3), dtype=np.float32))
        self.assertEqual(tuple(tensor.shape), (1, 3, 4, 5))
        self.assertAlmostEqual(float(tensor.min()), -1.0, places=6)
        tensor = to_lpips_tensor(np.ones((4, 5, 3), dtype=np.float32))
        self.assertAlmostEqual(float(tensor.max()), 1.0, places=6)


@unittest.skipUnless(_HAVE_WEIGHTS, _SKIP_REASON)
class LpipsValueTest(unittest.TestCase):
    def test_identical_images_score_zero(self):
        image = _rng_image(11)
        self.assertLess(_SCORER.distance(image, image), 1e-5)

    def test_noise_increases_lpips_monotonically(self):
        image = _rng_image(12)
        values = [_SCORER.distance(_add_noise(image, s), image) for s in (0.02, 0.06, 0.15, 0.3)]
        self.assertGreater(values[0], 0.0)
        for lower, higher in zip(values, values[1:]):
            self.assertLess(lower, higher, f"not monotonic: {values}")

    def test_masked_region_content_does_not_change_lpips(self):
        """The whole point of the mask policy, verified end to end."""
        mask = np.ones((64, 64), dtype=bool)
        mask[8:40, 10:50] = False
        rendered = _rng_image(13)
        reference = _add_noise(rendered, 0.05, seed=99)

        def scored(fill_rendered: float, fill_reference: float) -> float:
            r, f = rendered.copy(), reference.copy()
            r[~mask] = fill_rendered
            f[~mask] = fill_reference
            return _SCORER.distance(apply_mask_policy(r, mask), apply_mask_policy(f, mask))

        a = scored(0.0, 1.0)
        b = scored(0.83, 0.11)
        self.assertAlmostEqual(a, b, places=6)

    def test_reported_backend_is_labelled(self):
        info = _SCORER.describe()
        self.assertIn(info["backend"], ("lpips_package", "torchvision_uncalibrated"))
        self.assertEqual(info["calibrated"], info["backend"] == "lpips_package")


def _make_fake_eval_dir(root: Path, run: str, n_frames: int = 3, size: int = 48) -> Path:
    eval_dir = root / run / "evaluation"
    eval_dir.mkdir(parents=True)
    for index in range(n_frames):
        image_id = f"img_{index:024x}"
        reference = _rng_image(100 + index, size=size)
        rendered = _add_noise(reference, 0.05, seed=200 + index)
        mask = np.ones((size, size), dtype=np.uint8) * 255
        mask[: size // 4, :] = 0
        _write_png(eval_dir / f"{image_id}_reference.png", reference)
        _write_png(eval_dir / f"{image_id}_rendered.png", rendered)
        Image.fromarray(mask).save(eval_dir / f"{image_id}_mask.png")
    return eval_dir


class FrameDiscoveryTest(unittest.TestCase):
    def test_ids_match_compare_validation_metrics_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = _make_fake_eval_dir(Path(tmp), "run_a", n_frames=3)
            ids = frame_ids(eval_dir)
            self.assertEqual(len(ids), 3)
            self.assertTrue(all(i.startswith("img_") for i in ids))
            self.assertTrue(all((eval_dir / f"{i}_reference.png").exists() for i in ids))


@unittest.skipUnless(_HAVE_WEIGHTS, _SKIP_REASON)
class ScoreRunTest(unittest.TestCase):
    def test_score_run_reports_stats_and_mask_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = _make_fake_eval_dir(Path(tmp), "run_a", n_frames=3)
            result = score_run(eval_dir, _SCORER)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(len(result["lpips_per_frame"]), 3)
        self.assertGreater(result["lpips_mean"], 0.0)
        self.assertEqual(result["mask_policy"], MASK_POLICY)
        self.assertAlmostEqual(result["valid_pixel_fraction_mean"], 0.75, places=6)
        self.assertIn("calibrated", result)

    def test_missing_frames_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "evaluation"
            empty.mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                score_run(empty, _SCORER)


@unittest.skipUnless(_HAVE_WEIGHTS, _SKIP_REASON)
class CliSmokeTest(unittest.TestCase):
    def test_cli_produces_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_eval_dir(root, "run_a", n_frames=3)
            _make_fake_eval_dir(root, "run_b", n_frames=3)
            out = root / "lpips.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "compute_lpips.py"),
                    "--runs-root", str(root),
                    "--runs", "run_a", "run_b",
                    "--output", str(out),
                    "--device", "cpu",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("LPIPS", proc.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"run_a", "run_b"})
        for run in payload.values():
            self.assertEqual(run["frames"], 3)
            self.assertEqual(run["mask_policy"], MASK_POLICY)
            self.assertGreaterEqual(run["lpips_p90"], run["lpips_median"])


if __name__ == "__main__":
    unittest.main()
