from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_observation_coverage import (  # noqa: E402
    STATUS_BEHIND,
    STATUS_NO_LIDAR,
    STATUS_OCCLUDED,
    STATUS_SUPPORTED,
    CameraModel,
    ViewRecord,
    WindowStats,
    audit_samples,
    classify_support,
    face_footprint_px_per_m,
    fisheye_footprint_px_per_m,
    laplacian_4,
    max_pairwise_angle_deg,
    summarize,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402
from cloudstudio_3dgs.geometry.lidar_projection import DepthProjectionConfig  # noqa: E402


def _look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """c2w with OpenCV axes (+z forward, +y down) looking from position to target."""
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


class SyntheticSetup:
    """Equidistant fisheye (KB4 with zero distortion), one forward pinhole face."""

    def __init__(self) -> None:
        self.camera = CameraModel(
            camera_id="cam",
            intrinsic={"fl_x": 500.0, "fl_y": 500.0, "cx": 500.0, "cy": 500.0},
            distortion={"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
            width=1000,
            height=1000,
            max_theta_rad=math.radians(95.0),
        )
        self.face = FaceSpec(
            face_id="front",
            R_face=np.eye(3),
            K_face=np.array([[400.0, 0.0, 300.0], [0.0, 400.0, 300.0], [0.0, 0.0, 1.0]]),
            width=600,
            height=600,
            half_fov_deg=36.87,
        )
        self.geometry: dict[tuple[str, str], np.ndarray] = {}
        self.masks: dict[tuple[str, str], np.ndarray] = {}
        self.photos: dict[str, np.ndarray] = {}

    def face_geometry(self, image_id, face_id):
        return self.geometry.get((image_id, face_id))

    def face_mask(self, image_id, face_id):
        return self.masks.get((image_id, face_id))

    def photo(self, image_id):
        return self.photos.get(image_id)


class ClassifySupportTests(unittest.TestCase):
    def test_statuses(self) -> None:
        dense = np.zeros((20, 20), dtype=np.float32)
        dense[5, 5] = 2.0  # near return
        dense[10, 10] = 5.0  # consistent return
        dense[15, 15] = 9.0  # far return
        px = np.array([5, 10, 15, 2])
        py = np.array([5, 10, 15, 18])
        ranges = np.array([5.0, 5.0, 5.0, 5.0])
        status = classify_support(dense, px, py, ranges, radius_px=1, tolerance=0.2, margin_m=0.1)
        self.assertEqual(status.tolist(), [STATUS_OCCLUDED, STATUS_SUPPORTED, STATUS_BEHIND, STATUS_NO_LIDAR])

    def test_window_radius_reaches_neighbours(self) -> None:
        dense = np.zeros((20, 20), dtype=np.float32)
        dense[7, 7] = 1.0
        status = classify_support(dense, np.array([10]), np.array([10]), np.array([5.0]), radius_px=3)
        self.assertEqual(status.tolist(), [STATUS_OCCLUDED])
        status = classify_support(dense, np.array([10]), np.array([10]), np.array([5.0]), radius_px=2)
        self.assertEqual(status.tolist(), [STATUS_NO_LIDAR])


class FootprintAndSharpnessTests(unittest.TestCase):
    def test_on_axis_footprint_matches_focal_over_depth(self) -> None:
        setup = SyntheticSetup()
        pts = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 10.0]])
        fish = fisheye_footprint_px_per_m(setup.camera, pts)
        face = face_footprint_px_per_m(setup.face, pts)
        np.testing.assert_allclose(fish, [100.0, 50.0], rtol=2e-3)
        np.testing.assert_allclose(face, [80.0, 40.0], rtol=2e-3)

    def test_laplacian_variance_ranks_sharp_over_blurred(self) -> None:
        rng = np.random.default_rng(3)
        img = np.zeros((128, 128), dtype=np.float32)
        img[:, 64:] = 200.0  # sharp step
        blurred = img.copy()
        for _ in range(6):  # box blur
            blurred = (np.roll(blurred, 1, 1) + blurred + np.roll(blurred, -1, 1)) / 3.0
        flat = np.full((128, 128), 50.0, dtype=np.float32) + rng.normal(0, 0.0, (128, 128)).astype(np.float32)
        stats_sharp = WindowStats(laplacian_4(img))
        stats_blur = WindowStats(laplacian_4(blurred))
        stats_flat = WindowStats(laplacian_4(flat))
        c = np.array([64]), np.array([64])
        _, v_sharp = stats_sharp.mean_var(*c, 64)
        _, v_blur = stats_blur.mean_var(*c, 64)
        _, v_flat = stats_flat.mean_var(*c, 64)
        self.assertGreater(v_sharp[0], v_blur[0])
        self.assertGreater(v_blur[0], 0.0)
        self.assertEqual(v_flat[0], 0.0)

    def test_window_mean_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        img = rng.uniform(0, 255, (100, 120)).astype(np.float32)
        stats = WindowStats(img)
        m, v = stats.mean_var(np.array([60]), np.array([50]), 64)
        patch = img[18:82, 28:92]
        self.assertAlmostEqual(m[0], float(patch.mean()), places=4)
        self.assertAlmostEqual(v[0], float(patch.var()), places=2)

    def test_pairwise_angle(self) -> None:
        dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(max_pairwise_angle_deg(dirs), 90.0, places=6)
        self.assertTrue(math.isnan(max_pairwise_angle_deg(dirs[:1])))


class AuditSamplesTests(unittest.TestCase):
    def test_synthetic_cameras_and_occluder(self) -> None:
        setup = SyntheticSetup()
        target = np.array([0.0, 0.0, 5.0])
        # view A: on-axis, sharp photo; view B: 2 m to the side, blurred photo;
        # view C: same as A but a LiDAR return at 2 m in front -> occluded;
        # view D: looking away -> never sees the point.
        views = [
            ViewRecord("A", "cam", "rigA", 1_000_000_000, _look_at(np.array([0.0, 0.0, 0.0]), target)),
            ViewRecord("B", "cam", "rigB", 4_000_000_000, _look_at(np.array([2.0, 0.0, 0.0]), target)),
            ViewRecord("C", "cam", "rigC", 7_000_000_000, _look_at(np.array([0.0, 0.0, 0.0]), target)),
            ViewRecord("D", "cam", "rigD", 9_000_000_000, _look_at(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -5.0]))),
        ]
        h = w = 600
        supported = np.zeros((h, w), dtype=np.float32)
        supported[300, 300] = 5.0
        occluding = np.zeros((h, w), dtype=np.float32)
        occluding[300, 300] = 2.0
        setup.geometry[("A", "front")] = supported
        supported_b = np.zeros((h, w), dtype=np.float32)
        supported_b[300, 300] = float(math.sqrt(29.0))  # B's own range: strictly supported
        setup.geometry[("B", "front")] = supported_b
        setup.geometry[("C", "front")] = occluding
        ones = np.ones((h, w), dtype=bool)
        setup.masks[("A", "front")] = ones
        setup.masks[("B", "front")] = ones
        setup.masks[("C", "front")] = ones
        photo = np.zeros((1000, 1000), dtype=np.float32)
        photo[:, 500:] = 200.0
        rng = np.random.default_rng(0)
        setup.photos["A"] = photo
        setup.photos["B"] = np.full((1000, 1000), 90.0, dtype=np.float32) + rng.normal(0, 1.0, (1000, 1000)).astype(np.float32)
        setup.photos["C"] = photo

        rows = audit_samples(
            target[None, :], views, {"cam": setup.camera}, {"cam": [setup.face]},
            face_geometry=setup.face_geometry, face_mask=setup.face_mask, photo=setup.photo,
            projection=DepthProjectionConfig(visibility_cell_px=6),
        )
        row = rows[0]
        self.assertEqual(row["n_images_fisheye"], 3)
        self.assertEqual(row["n_images_face"], 3)
        self.assertEqual(row["n_occluded"], 1)
        self.assertEqual(row["n_depth_supported"], 2)
        self.assertEqual(row["n_no_lidar"], 0)
        self.assertEqual(row["n_strict_supported"], 2)
        self.assertEqual(row["n_strict_occluded"], 1)
        self.assertEqual(row["n_views_loose_ok"], 2)
        self.assertEqual(row["n_effective_views"], 2)  # A and B
        self.assertEqual(row["n_effective_views_le5m"], 1)  # only A is within 5 m (B is at sqrt(29))
        self.assertEqual(row["n_rig_frames"], 2)
        self.assertEqual(row["n_cameras"], 1)
        self.assertAlmostEqual(row["time_span_s"], 3.0)
        self.assertAlmostEqual(row["angle_span_deg"], math.degrees(math.atan2(2.0, 5.0)), places=3)
        self.assertAlmostEqual(row["frac_occluded"], 1.0 / 3.0)
        self.assertAlmostEqual(row["strict_frac_occluded"], 1.0 / 3.0)
        self.assertAlmostEqual(row["rgb_valid_frac"], 1.0)
        self.assertAlmostEqual(row["depth_support_frac"], 2.0 / 3.0)
        self.assertAlmostEqual(row["range_min_m"], 5.0, places=6)
        self.assertGreater(row["sharpness_lapvar_best"], 100.0)
        self.assertEqual(row["sharpness_best_image"], "A")
        # median of view A (400/5 = 80 px/m) and view B (400/sqrt(29) px/m)
        self.assertAlmostEqual(row["footprint_px_per_m_face"], (80.0 + 400.0 / math.sqrt(29.0)) / 2.0, delta=0.5)

        summary = summarize(rows)
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["n_effective_views"]["median"], 2.0)

    def test_loose_support_is_not_effective_without_strict_evidence(self) -> None:
        setup = SyntheticSetup()
        target = np.array([0.0, 0.0, 5.0])
        views = [ViewRecord("A", "cam", "rigA", 0, _look_at(np.zeros(3), target))]
        setup.masks[("A", "front")] = np.ones((600, 600), dtype=bool)
        dense = np.zeros((600, 600), dtype=np.float32)
        dense[300, 300] = 4.3  # 0.7 m in front: inside the loose 20 % band, outside the strict band
        setup.geometry[("A", "front")] = dense
        rows = audit_samples(
            target[None, :], views, {"cam": setup.camera}, {"cam": [setup.face]},
            face_geometry=setup.face_geometry, face_mask=setup.face_mask, photo=setup.photo,
        )
        self.assertEqual(rows[0]["n_depth_supported"], 1)
        self.assertEqual(rows[0]["n_views_loose_ok"], 1)
        self.assertEqual(rows[0]["n_strict_occluded"], 1)
        self.assertEqual(rows[0]["n_effective_views"], 0)

    def test_mask_false_removes_effective_view(self) -> None:
        setup = SyntheticSetup()
        target = np.array([0.0, 0.0, 5.0])
        views = [ViewRecord("A", "cam", "rigA", 0, _look_at(np.zeros(3), target))]
        setup.masks[("A", "front")] = np.zeros((600, 600), dtype=bool)
        dense = np.zeros((600, 600), dtype=np.float32)
        dense[300, 300] = 5.0
        setup.geometry[("A", "front")] = dense
        rows = audit_samples(
            target[None, :], views, {"cam": setup.camera}, {"cam": [setup.face]},
            face_geometry=setup.face_geometry, face_mask=setup.face_mask, photo=setup.photo,
        )
        self.assertEqual(rows[0]["n_images_face"], 1)
        self.assertEqual(rows[0]["n_effective_views"], 0)
        self.assertAlmostEqual(rows[0]["rgb_valid_frac"], 0.0)


if __name__ == "__main__":
    unittest.main()
