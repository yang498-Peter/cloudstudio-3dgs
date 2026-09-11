from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_exposure_gains import (  # noqa: E402
    LN2,
    CameraModel,
    ViewRecord,
    collect_observations,
    colour_dispersion,
    cross_tile_disagreement,
    extract_gains,
    gain_index_from_views,
    join_membership,
    merge_tile_gain,
    oracle_image_gains,
    saturation_fraction,
    summarize_dispersion,
    summarize_gains,
    torch_style_median,
    variance_decomposition,
    window_median_rgb,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402
from cloudstudio_3dgs.training.exposure import ExposureCompensationConfig, ExposureCompensator  # noqa: E402


def _views(*sample_ids: str) -> list[dict]:
    return [{"sample_id": s, "x": 0, "y": 0, "width": 1, "height": 1} for s in sample_ids]


def _membership(image_id: str, camera: str, rig: str, t_ns: int, env: str, tiles: str = "0|1") -> dict:
    return {
        "image_id": image_id, "camera": camera, "rig_frame_id": rig, "timestamp_ns": t_ns,
        "capture_fraction": 0.0, "environment": env, "tile_ids": tiles,
    }


class IndexContractTests(unittest.TestCase):
    def test_order_matches_exposure_compensator(self) -> None:
        views = _views("img_b::yaw_plus_35", "img_a::pitch_up_56", "img_b::pitch_down_56", "img_c::yaw_minus_35")
        ordered = gain_index_from_views(views)
        self.assertEqual(ordered, ["img_a", "img_b", "img_c"])
        # first-seen order of exposure_image_ids differs from sorted order; the
        # compensator sorts, so the checkpoint index must follow sorted order
        first_seen = ["img_b", "img_a", "img_c"]
        comp = ExposureCompensator(
            first_seen, config=ExposureCompensationConfig(enabled=True), device="cpu",
            group_by_image={"img_a": "left", "img_b": "right", "img_c": "left"},
        )
        self.assertEqual([comp.index[i] for i in ordered], [0, 1, 2])

    def test_extract_clamps_and_flags_saturation(self) -> None:
        records = extract_gains([0.1, -0.9, 0.8], ["a", "b", "c"])
        self.assertAlmostEqual(records["a"].gain, math.exp(0.1))
        self.assertAlmostEqual(records["b"].log_gain, -LN2)
        self.assertAlmostEqual(records["b"].log_gain_raw, -0.9)
        self.assertTrue(records["b"].saturated)
        self.assertTrue(records["c"].saturated)
        self.assertFalse(records["a"].saturated)
        self.assertEqual(records["c"].index, 2)

    def test_extract_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            extract_gains([0.0, 0.0], ["a", "b", "c"])

    def test_torch_style_median_matches_torch(self) -> None:
        import torch

        values = [0.3, -0.2, 0.9, 0.1]
        self.assertAlmostEqual(torch_style_median(values), float(torch.tensor(values).median()))
        values = [0.3, -0.2, 0.9]
        self.assertAlmostEqual(torch_style_median(values), float(torch.tensor(values).median()))

    def test_merge_tile_gain_uses_unclamped_exp(self) -> None:
        records = extract_gains([-0.9, -0.9, -0.9, 0.0], list("abcd"))
        # median of exp(raw) with torch lower-middle rule = exp(-0.9), beyond the clamp
        self.assertAlmostEqual(merge_tile_gain(records), math.exp(-0.9))


class JoinAndSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gains = {
            "Tile_0": extract_gains([math.log(0.8), math.log(1.2), 0.0], ["a", "b", "c"]),
            "Tile_1": extract_gains([math.log(0.9), math.log(1.5)], ["b", "c"]),
        }
        self.membership = [
            _membership("a", "left", "r1", 1_000_000_000, "outdoor", "0"),
            _membership("b", "right", "r1", 1_000_000_000, "outdoor", "0|1"),
            _membership("c", "left", "r2", 2_000_000_000, "indoor", "0|1"),
        ]

    def test_join_rows_and_implied_shift(self) -> None:
        rows = join_membership(self.gains, self.membership, saturation={"a": {"saturation_frac": 0.25, "mean_luma": 100.0}})
        self.assertEqual(len(rows), 5)
        row_a = next(r for r in rows if r["image_id"] == "a")
        self.assertEqual(row_a["tile"], "Tile_0")
        self.assertEqual(row_a["environment"], "outdoor")
        self.assertAlmostEqual(row_a["gain"], 0.8)
        # render ~= photo / g => canonical is brighter than the photo by 1/0.8
        self.assertAlmostEqual(row_a["log_canonical_over_photo"], -math.log(0.8))
        tile0_median = merge_tile_gain(self.gains["Tile_0"])  # torch median of {0.8,1.2,1.0} -> 1.0
        self.assertAlmostEqual(tile0_median, 1.0)
        self.assertAlmostEqual(row_a["log_baked_over_photo"], math.log(1.0) - math.log(0.8))
        self.assertAlmostEqual(row_a["photo_saturation_frac"], 0.25)
        row_c1 = next(r for r in rows if r["image_id"] == "c" and r["tile"] == "Tile_1")
        self.assertAlmostEqual(row_c1["gain"], 1.5)

    def test_join_requires_membership(self) -> None:
        with self.assertRaises(KeyError):
            join_membership(self.gains, self.membership[:2])

    def test_summary_groups(self) -> None:
        rows = join_membership(self.gains, self.membership)
        summary = summarize_gains(rows)
        self.assertEqual(summary["by_tile"]["Tile_1"]["gain"]["n"], 2)
        self.assertEqual(summary["by_environment"]["indoor"]["gain"]["n"], 2)
        self.assertEqual(summary["by_tile_environment"]["Tile_0|outdoor"]["gain"]["n"], 2)
        self.assertAlmostEqual(summary["by_camera"]["right"]["gain"]["mean"], (1.2 + 0.9) / 2)

    def test_cross_tile_disagreement(self) -> None:
        rows = join_membership(self.gains, self.membership)
        out = cross_tile_disagreement(rows)
        self.assertEqual(out["images_in_multiple_tiles"], 2)
        # image b: |log1.2 - log0.9|, image c: |log1.0 - log1.5|
        expected = sorted([abs(math.log(1.2) - math.log(0.9)), abs(math.log(1.5))])
        got = out["by_tile_pair"]["Tile_0~Tile_1"]
        self.assertEqual(got["n"], 2)
        self.assertAlmostEqual(got["min"], expected[0])
        self.assertAlmostEqual(got["max"], expected[1])
        self.assertEqual(out["by_environment"]["indoor"]["n"], 1)
        # after removing each tile's median (Tile_0: 1.0, Tile_1: torch median of {0.9,1.5} = 0.9)
        baked_c = abs((math.log(1.0) - 0.0) - (math.log(1.5) - math.log(0.9)))
        self.assertAlmostEqual(out["by_environment_after_tile_bake"]["indoor"]["p50"], baked_c)


class VarianceDecompositionTests(unittest.TestCase):
    def test_camera_explains_everything(self) -> None:
        gains = {"Tile_0": extract_gains([0.2, -0.2, 0.2, -0.2], ["a", "b", "c", "d"])}
        membership = [
            _membership("a", "left", "r1", 0, "outdoor"),
            _membership("b", "right", "r1", 0, "outdoor"),
            _membership("c", "left", "r2", 30_000_000_000, "indoor"),
            _membership("d", "right", "r2", 30_000_000_000, "indoor"),
        ]
        rows = join_membership(gains, membership)
        out = variance_decomposition(rows, time_blocks_s=(10.0,))
        self.assertAlmostEqual(out["r2_camera"], 1.0)
        self.assertAlmostEqual(out["r2_rig_frame"], 0.0)
        self.assertAlmostEqual(out["r2_time_block_10s"], 0.0)
        self.assertAlmostEqual(out["r2_camera_x_time_block_10s"], 1.0)
        self.assertEqual(out["groups_time_block_10s"], 2)
        self.assertEqual(out["left_right_same_frame_pairs"], 2)
        self.assertAlmostEqual(out["left_minus_right_mean"], 0.4)

    def test_time_explains_everything(self) -> None:
        gains = {"Tile_0": extract_gains([0.3, 0.3, -0.3, -0.3], ["a", "b", "c", "d"])}
        membership = [
            _membership("a", "left", "r1", 0, "outdoor"),
            _membership("b", "right", "r1", 0, "outdoor"),
            _membership("c", "left", "r2", 30_000_000_000, "outdoor"),
            _membership("d", "right", "r2", 30_000_000_000, "outdoor"),
        ]
        rows = join_membership(gains, membership)
        out = variance_decomposition(rows, time_blocks_s=(10.0,))
        self.assertAlmostEqual(out["r2_camera"], 0.0)
        self.assertAlmostEqual(out["r2_rig_frame"], 1.0)
        self.assertAlmostEqual(out["r2_time_block_10s"], 1.0)
        self.assertAlmostEqual(out["left_right_same_frame_corr"], 1.0)

    def test_rejects_mixed_tiles(self) -> None:
        gains = {"Tile_0": extract_gains([0.0], ["a"]), "Tile_1": extract_gains([0.0], ["a"])}
        rows = join_membership(gains, [_membership("a", "left", "r1", 0, "outdoor")])
        with self.assertRaises(ValueError):
            variance_decomposition(rows)


class SaturationTests(unittest.TestCase):
    def test_fraction(self) -> None:
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[:2, :] = 255
        gray[2, :5] = 250
        gray[3, :5] = 249
        self.assertAlmostEqual(saturation_fraction(gray), 25.0 / 100.0)

    def test_window_median(self) -> None:
        photo = np.zeros((9, 9, 3), dtype=np.uint8)
        photo[4, 4] = (255, 255, 255)  # single hot pixel is rejected by the median
        photo[:, :, 1] = 40
        med = window_median_rgb(photo, np.array([4, 0]), np.array([4, 0]), radius=1)
        np.testing.assert_allclose(med[0], [0.0, 40.0, 0.0])
        np.testing.assert_allclose(med[1], [0.0, 40.0, 0.0])


def _look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_down = np.array([0.0, 0.0, -1.0])
    right = np.cross(world_down, forward)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


class ColourDispersionTests(unittest.TestCase):
    """Three views of one grey surface point, each photo exposed with a known
    gain: raw dispersion is large, dividing by the true gains removes it."""

    def setUp(self) -> None:
        self.camera = CameraModel(
            camera_id="cam",
            intrinsic={"fl_x": 500.0, "fl_y": 500.0, "cx": 500.0, "cy": 500.0},
            distortion={"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
            width=1000, height=1000, max_theta_rad=math.radians(95.0),
        )
        self.face = FaceSpec(
            face_id="front", R_face=np.eye(3),
            K_face=np.array([[400.0, 0.0, 300.0], [0.0, 400.0, 300.0], [0.0, 0.0, 1.0]]),
            width=600, height=600, half_fov_deg=36.87,
        )
        self.target = np.array([0.0, 0.0, 5.0])
        self.true_gain = {"A": 0.8, "B": 1.0, "C": 1.25}
        self.views = [
            ViewRecord("A", "cam", "rA", 0, _look_at(np.array([0.0, 0.0, 0.0]), self.target)),
            ViewRecord("B", "cam", "rB", 1_000_000_000, _look_at(np.array([0.5, 0.0, 0.0]), self.target)),
            ViewRecord("C", "cam", "rC", 2_000_000_000, _look_at(np.array([-0.5, 0.0, 0.0]), self.target)),
            ViewRecord("D", "cam", "rD", 3_000_000_000, _look_at(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -5.0]))),
        ]
        self.geometry = {}
        for view in self.views[:3]:
            dense = np.zeros((600, 600), dtype=np.float32)
            pc = (self.target - view.c2w[:3, 3]) @ view.c2w[:3, :3]
            pix, _ = self.face.directions_to_pixels(pc[None, :])
            px, py = int(round(pix[0, 0] - 0.5)), int(round(pix[0, 1] - 0.5))
            dense[py, px] = float(np.linalg.norm(pc))
            self.geometry[(view.image_id, "front")] = dense
        self.photos = {
            i: np.full((1000, 1000, 3), 120 * g, dtype=np.float64).round().astype(np.uint8)
            for i, g in self.true_gain.items()
        }

    def _obs(self):
        return collect_observations(
            self.target[None, :], self.views, {"cam": self.camera}, {"cam": [self.face]},
            face_geometry=lambda i, f: self.geometry.get((i, f)),
            face_mask=lambda i, f: np.ones((600, 600), dtype=bool),
            photo_rgb=lambda i: self.photos.get(i),
        )

    def test_collect_observations_strict_views_only(self) -> None:
        obs = self._obs()
        self.assertEqual(sorted(i for i, _, _ in obs[0]), ["A", "B", "C"])
        for image_id, rgb, rng in obs[0]:
            self.assertAlmostEqual(rgb[0], round(120 * self.true_gain[image_id]))
            self.assertAlmostEqual(rng, float(np.linalg.norm(self.target - next(v.c2w[:3, 3] for v in self.views if v.image_id == image_id))), places=6)

    def test_gains_remove_dispersion(self) -> None:
        obs = self._obs()
        rows = colour_dispersion(obs, {"learned": self.true_gain, "partial": {"A": 0.8, "B": 1.0}})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertGreater(row["luma_cv_raw"], 0.15)
        self.assertLess(row["luma_cv_learned"], 0.01)
        self.assertAlmostEqual(row["luma_cv_raw_paired_learned"], row["luma_cv_raw"])
        self.assertEqual(row["n_views_partial"], 2)
        self.assertTrue(math.isnan(row["luma_cv_partial"]))  # below min_views
        summary = summarize_dispersion(rows, ["learned", "partial"])
        self.assertEqual(summary["learned"]["points_with_gain"], 1)
        self.assertEqual(summary["partial"]["points_with_gain"], 0)
        self.assertAlmostEqual(summary["learned"]["frac_points_improved"], 1.0)

    def test_oracle_recovers_gains_up_to_constant(self) -> None:
        obs = self._obs()
        oracle = oracle_image_gains(obs)
        ratio = {i: oracle[i] / self.true_gain[i] for i in oracle}
        values = np.asarray(list(ratio.values()))
        np.testing.assert_allclose(values, values.mean(), rtol=0.02)


if __name__ == "__main__":
    unittest.main()
