"""tools/fit_exposure_curve.py on synthetic gains with a known curve.

Two cameras, a known piecewise-linear log-gain curve on a 10 s grid, four
tiles with known brightness offsets, each image trained by two or three
tiles, gaussian noise and a few clamp-saturated outliers.  The fit must
recover the curve (up to each camera's mean, which the anchor pins to zero),
the tile offsets (up to the common shift the anchor moves into them), give
the outliers a small Huber weight, and produce a frozen-curve file that
``ExposureCurve`` reproduces gain for gain.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # torch is an optional training dependency
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from fit_exposure_curve import (  # noqa: E402
    CurveDesign,
    fit_exposure_curve,
    huber_weights,
    leave_one_tile_out,
    main,
    read_gain_rows,
    solve_penalised,
)
from cloudstudio_3dgs.training.exposure import (  # noqa: E402
    ExposureCompensationConfig,
    ExposureCurve,
    evaluate_curve,
)

T0 = 1_772_726_380_342_686_976
KNOT_S = 10.0
TRUE_KNOTS = {
    "left": [0.05, 0.10, -0.05, -0.20, -0.15, 0.00, 0.12, 0.08, -0.02, 0.05, 0.10, 0.04],
    "right": [-0.10, 0.05, 0.15, 0.10, -0.05, -0.20, -0.10, 0.00, 0.08, 0.12, 0.02, -0.06],
}
TRUE_OFFSETS = {"Tile_0": -0.12, "Tile_1": -0.07, "Tile_2": -0.15, "Tile_3": -0.05}
SPAN_S = 109.5  # images to 109.5 s: every one of the 12 knots (0..110 s) is observed


def _synthetic(seed: int = 7, *, outliers: int = 12, noise: float = 0.02) -> tuple[list[dict], set[str]]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    tiles = sorted(TRUE_OFFSETS)
    k = 0
    t = 0.0
    while t <= SPAN_S:
        for camera in ("left", "right"):
            image_id = f"img_{k:04d}"
            k += 1
            truth = evaluate_curve(TRUE_KNOTS[camera], [t], knot_seconds=KNOT_S)[0]
            members = rng.choice(4, size=int(rng.integers(2, 4)), replace=False)
            for m in members:
                tile = tiles[int(m)]
                rows.append(
                    {
                        "image_id": image_id,
                        "camera": camera,
                        "tile": tile,
                        "environment": "indoor" if t < 50 else "outdoor",
                        "timestamp_ns": T0 + int(t * 1e9),
                        "log_gain": truth + TRUE_OFFSETS[tile] + float(rng.normal(0.0, noise)),
                        "log_gain_raw": 0.0,
                        "saturated_clamp": 0,
                    }
                )
        t += 0.5
    for row in rows:
        row["log_gain_raw"] = row["log_gain"]
    flagged: set[str] = set()
    for index in rng.choice(len(rows), size=outliers, replace=False):
        row = rows[int(index)]
        sign = 1.0 if rng.random() < 0.5 else -1.0
        row["log_gain_raw"] = sign * 0.9
        row["log_gain"] = sign * math.log(2.0)  # what the trainer applied
        row["saturated_clamp"] = 1
        flagged.add(f"{row['image_id']}|{row['tile']}")
    return rows, flagged


def _centre(camera: str, rows: list[dict]) -> float:
    """Mean of the true curve over the camera's unique images (the anchor target)."""
    seen: dict[str, float] = {}
    for row in rows:
        if row["camera"] == camera:
            seen.setdefault(row["image_id"], (row["timestamp_ns"] - T0) / 1e9)
    values = evaluate_curve(TRUE_KNOTS[camera], list(seen.values()), knot_seconds=KNOT_S)
    return float(np.mean(values))


class FitTests(unittest.TestCase):
    def test_recovers_curve_offsets_and_downweights_outliers(self) -> None:
        rows, flagged = _synthetic()
        payload, records = fit_exposure_curve(
            rows, knot_seconds=KNOT_S, smoothness_weight=0.1, anchor_weight=1e4, huber_delta=0.1,
            sweep=(0.1, 10.0), holdout=True,
        )
        self.assertEqual(payload["kind"], "exposure_camera_curve")
        self.assertEqual(payload["time_origin_ns"], T0)
        self.assertEqual(payload["provenance"]["knot_count_by_camera"], {"left": 12, "right": 12})
        # Curve up to the anchored mean.
        for camera, truth in TRUE_KNOTS.items():
            fitted = np.array(payload["cameras"][camera]["knot_log_gains"])
            centred = np.array(truth) - _centre(camera, rows)
            self.assertLess(float(np.abs(fitted - centred).max()), 0.03, camera)
            self.assertLess(abs(payload["provenance"]["residuals"]["by_camera"][camera]["curve_mean_over_images"]), 1e-3)
        # Tile offsets up to the common shift the anchor moved into them.
        fitted_offsets = payload["provenance"]["tile_offsets"]
        shift = np.mean([fitted_offsets[t] - TRUE_OFFSETS[t] for t in TRUE_OFFSETS])
        for tile, truth in TRUE_OFFSETS.items():
            self.assertAlmostEqual(fitted_offsets[tile] - shift, truth, delta=0.01)
        # Outliers: large residual, small Huber weight; clean rows at weight 1.
        by_key = {f"{r['image_id']}|{r['tile']}": r for r in records}
        for key in flagged:
            self.assertLess(by_key[key]["huber_weight"], 0.5, key)
            self.assertGreater(abs(by_key[key]["residual"]), 0.3, key)
        clean = [r["huber_weight"] for k, r in by_key.items() if k not in flagged]
        self.assertGreater(float(np.mean(np.array(clean) > 0.99)), 0.97)
        res = payload["provenance"]["residuals"]
        self.assertLess(res["unsaturated_rows"]["rms"], 0.03)
        self.assertGreater(res["r2_vs_tile_offsets_only_unsaturated"], 0.9)
        self.assertLess(res["r2_vs_tile_offsets_only"], res["r2_vs_tile_offsets_only_unsaturated"])
        self.assertEqual(res["saturated_rows"]["n"], len(flagged))
        # Effective dof: below the 24 raw knots at lambda 0.1, and monotone in the sweep.
        edf = payload["provenance"]["effective_dof"]
        self.assertLess(edf["curve_total"], 24.0)
        self.assertGreater(edf["curve_total"], 18.0)
        sweep = payload["provenance"]["smoothness_sweep"]
        self.assertGreater(sweep[0]["edf_curve"], sweep[1]["edf_curve"])
        # Transfers to a tile it was not fitted on.
        for tile, entry in payload["provenance"]["leave_one_tile_out"].items():
            self.assertGreater(entry["r2_vs_offset_only_unsaturated"], 0.9, tile)
        self.assertEqual(len(records), len(rows))

    def test_anchor_is_required(self) -> None:
        rows, _ = _synthetic(outliers=0)
        design = CurveDesign(rows, knot_seconds=KNOT_S)
        with self.assertRaisesRegex(ValueError, "anchor_weight must be positive"):
            solve_penalised(design, smoothness_weight=1.0, anchor_weight=0.0, huber_delta=0.1, iterations=2)

    def test_huber_weights(self) -> None:
        w = huber_weights(np.array([0.0, 0.05, 0.2, -0.4]), 0.1)
        np.testing.assert_allclose(w, [1.0, 1.0, 0.5, 0.25])
        np.testing.assert_allclose(huber_weights(np.array([5.0]), 0.0), [1.0])

    def test_leave_one_tile_out_uses_the_shared_grid(self) -> None:
        rows, _ = _synthetic(outliers=0)
        design = CurveDesign(rows, knot_seconds=KNOT_S)
        out = leave_one_tile_out(
            rows, knot_seconds=KNOT_S, smoothness_weight=0.1, anchor_weight=1e4, huber_delta=0.1, iterations=3,
            time_origin_ns=design.time_origin_ns, knot_count_by_camera=design.knot_count_by_camera,
        )
        self.assertEqual(sorted(out), sorted(TRUE_OFFSETS))
        for entry in out.values():
            self.assertLess(entry["residual"]["rms"], 0.03)

    def test_csv_reader_and_cli_roundtrip_into_exposure_curve(self) -> None:
        rows, _ = _synthetic(outliers=4)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "gains.csv"
            with src.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(read_gain_rows(src)), len(rows))
            bad = Path(tmp) / "bad.csv"
            bad.write_text("image_id,camera\nx,left\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks columns"):
                read_gain_rows(bad)
            out_curve = Path(tmp) / "curve.json"
            out_csv = Path(tmp) / "fit.csv"
            code = main([
                "--consistency-csv", str(src), "--knot-seconds", "10", "--smoothness-weight", "0.1",
                "--anchor-weight", "1e4", "--huber-delta", "0.1", "--no-holdout",
                "--out-curve", str(out_curve), "--out-csv", str(out_csv),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out_curve.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["provenance"]["source_csv_sha256"]), 64)
            with out_csv.open("r", encoding="utf-8", newline="") as handle:
                fitted = list(csv.DictReader(handle))
            self.assertEqual(len(fitted), len(rows))
            self.assertIn("curve_log_gain", fitted[0])
            if torch is None:
                return
            # Frozen into the trainer-side model: gain == exp(curve at the image time).
            ids = sorted({r["image_id"] for r in rows})
            cameras = {r["image_id"]: r["camera"] for r in rows}
            times = {r["image_id"]: r["timestamp_ns"] for r in rows}
            config = ExposureCompensationConfig(enabled=True, mode="camera_curve", knot_seconds=10.0, frozen_curve=str(out_curve))
            curve = ExposureCurve(ids, config=config, device="cpu", camera_by_image=cameras, timestamp_ns_by_image=times)
            by_image = {}
            for row in fitted:
                by_image.setdefault(row["image_id"], float(row["curve_log_gain"]))
            for image_id in ids:
                self.assertAlmostEqual(float(curve.gain(image_id)), math.exp(by_image[image_id]), places=5)
            self.assertIsNone(curve.make_optimizer())


if __name__ == "__main__":
    unittest.main()
