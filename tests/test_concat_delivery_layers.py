"""Joining a delivery's body and sky layers (tools/concat_delivery_layers.py), CPU only.

A delivery ships as a body PLY plus a frozen sky PLY that publish copies beside it. Nothing
composited them, so every quality number scored the body alone against photographs containing
sky, and the body's correctly transparent sky read as a coverage failure. Compositing the pair
took the no-fill candidate from alpha p05 0.189 to 0.898 with its sharpness unchanged.

What is pinned:

* the join is a plain concatenation, body rows first, every parameter carried through;
* a DC-only sky layer is zero-padded up to the body's SH bands, which is exact - a spherical
  harmonic sum with zero rest terms equals its DC term - and the padded rows really are zero;
* a sky layer with MORE bands than the body is refused rather than truncated;
* optimizer and strategy state are dropped, so the joined file cannot be mistaken for a
  trainable model;
* the output records where each layer came from and how many gaussians it contributed.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "concat_delivery_layers.py"


def _params(count: int, sh_bands: int, value: float):
    return {
        "means": torch.full((count, 3), value, dtype=torch.float32),
        "scales": torch.full((count, 3), value, dtype=torch.float32),
        "quats": torch.full((count, 4), value, dtype=torch.float32),
        "opacities": torch.full((count,), value, dtype=torch.float32),
        "sh0": torch.full((count, 1, 3), value, dtype=torch.float32),
        "shN": torch.full((count, sh_bands, 3), value, dtype=torch.float32),
    }


def _write(path, count, sh_bands, value, extra=None):
    blob = {"schema_version": 1, "step": 20000, "params": _params(count, sh_bands, value)}
    if extra:
        blob.update(extra)
    torch.save(blob, path)
    return path


@unittest.skipUnless(torch is not None, "torch is required")
class ConcatDeliveryLayersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _run(self, body, sky, out):
        return subprocess.run(
            [sys.executable, str(TOOL), "--body", str(body), "--sky", str(sky), "--output", str(out)],
            capture_output=True, text=True,
        )

    def test_body_rows_come_first_and_every_parameter_is_carried(self):
        body = _write(self.root / "body.pt", 5, 3, 1.0)
        sky = _write(self.root / "sky.pt", 2, 3, 2.0)
        out = self.root / "pair.pt"
        done = self._run(body, sky, out)
        self.assertEqual(done.returncode, 0, done.stderr)
        joined = torch.load(out, map_location="cpu", weights_only=False)["params"]
        for key in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            self.assertEqual(joined[key].shape[0], 7, key)
            self.assertTrue(torch.all(joined[key][:5] == 1.0), key)
            self.assertTrue(torch.all(joined[key][5:] == 2.0), key)

    def test_a_dc_only_sky_layer_is_zero_padded_to_the_bodys_bands(self):
        body = _write(self.root / "body.pt", 4, 3, 1.0)
        sky = _write(self.root / "sky.pt", 3, 0, 2.0)
        out = self.root / "pair.pt"
        done = self._run(body, sky, out)
        self.assertEqual(done.returncode, 0, done.stderr)
        joined = torch.load(out, map_location="cpu", weights_only=False)["params"]
        self.assertEqual(tuple(joined["shN"].shape), (7, 3, 3))
        # the padding really is zero, so those rows render as the degree-0 model they were
        self.assertTrue(torch.all(joined["shN"][4:] == 0.0))
        # and the DC term the sky layer did carry survives untouched
        self.assertTrue(torch.all(joined["sh0"][4:] == 2.0))

    def test_a_sky_layer_with_more_bands_than_the_body_is_refused(self):
        body = _write(self.root / "body.pt", 4, 0, 1.0)
        sky = _write(self.root / "sky.pt", 3, 3, 2.0)
        done = self._run(body, sky, self.root / "pair.pt")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("truncate real coefficients", done.stdout + done.stderr)

    def test_training_state_is_dropped_and_provenance_recorded(self):
        body = _write(self.root / "body.pt", 4, 3, 1.0,
                      extra={"optimizers": {"means": {}}, "strategy_state": {"step": 1}})
        sky = _write(self.root / "sky.pt", 2, 3, 2.0)
        out = self.root / "pair.pt"
        done = self._run(body, sky, out)
        self.assertEqual(done.returncode, 0, done.stderr)
        joined = torch.load(out, map_location="cpu", weights_only=False)
        self.assertNotIn("optimizers", joined)
        self.assertNotIn("strategy_state", joined)
        self.assertEqual(joined["delivery_layers"]["body"]["gaussian_count"], 4)
        self.assertEqual(joined["delivery_layers"]["sky"]["gaussian_count"], 2)

    def test_a_checkpoint_without_params_is_refused_by_name(self):
        bad = self.root / "bad.pt"
        torch.save({"schema_version": 1}, bad)
        sky = _write(self.root / "sky.pt", 2, 3, 2.0)
        done = self._run(bad, sky, self.root / "pair.pt")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("body", done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
