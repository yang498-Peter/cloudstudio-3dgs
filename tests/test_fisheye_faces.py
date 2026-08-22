from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from cloudstudio_3dgs.geometry.fisheye_faces import (
    EDGE_FADE_FRAC,
    MAX_FACES_PER_RING,
    RING_HALF_FOV_MAX_DEG,
    RING_HALF_FOV_MIN_DEG,
    FaceSpec,
    face_coverage_check,
    face_weight,
    kb4_project,
    kb4_unproject,
    plan_fisheye_faces,
)

# Realistic S1 left-lens parameters (values mirror the ukgs manifest calibration).
S1_K = np.array(
    [
        [777.4788779034258, 0.0, 1453.9369883854906],
        [0.0, 777.7524641168766, 1450.786805386424],
        [0.0, 0.0, 1.0],
    ]
)
S1_COEFFS = np.array(
    [0.08179302700881765, -0.011926722405662877, -0.002896186185757192, -0.00012363296834601707]
)
S1_IMAGE_SIZE = (2912, 2912)
S1_FOV_DEG = 190.0

MANIFEST_PATH = Path(r"C:\Peter\3dgs-datasets\ukgs_manifest\dataset_manifest.json")


def _sample_directions(rng: np.random.Generator, count: int, theta_min_deg: float, theta_max_deg: float) -> np.ndarray:
    theta = np.radians(rng.uniform(theta_min_deg, theta_max_deg, count))
    azimuth = rng.uniform(0.0, 2.0 * math.pi, count)
    sin_t = np.sin(theta)
    return np.column_stack([sin_t * np.cos(azimuth), sin_t * np.sin(azimuth), np.cos(theta)])


class Kb4RoundTripTests(unittest.TestCase):
    def test_project_unproject_roundtrip_random_directions(self) -> None:
        rng = np.random.default_rng(20260823)
        dirs = _sample_directions(rng, 4000, 0.0, 94.0)
        pixels = kb4_project(dirs, S1_K, S1_COEFFS)
        rays = kb4_unproject(pixels, S1_K, S1_COEFFS)
        angle_err = np.arccos(np.clip(np.sum(rays * dirs, axis=1), -1.0, 1.0))
        # arccos(dot) has a float64 measurement floor around sqrt(eps) ~ 1.5e-8 rad;
        # the load-bearing accuracy contract is the pixel round-trip below.
        self.assertLess(float(angle_err.max()), 1e-7)
        reprojected = kb4_project(rays, S1_K, S1_COEFFS)
        pixel_err = np.linalg.norm(reprojected - pixels, axis=1)
        self.assertLess(float(pixel_err.max()), 1e-6)

    def test_roundtrip_holds_at_large_angles_beyond_85_deg(self) -> None:
        rng = np.random.default_rng(7)
        dirs = _sample_directions(rng, 2000, 85.0, 94.5)
        pixels = kb4_project(dirs, S1_K, S1_COEFFS)
        rays = kb4_unproject(pixels, S1_K, S1_COEFFS)
        reprojected = kb4_project(rays, S1_K, S1_COEFFS)
        self.assertLess(float(np.linalg.norm(reprojected - pixels, axis=1).max()), 1e-6)

    def test_principal_point_unprojects_to_optical_axis(self) -> None:
        center = np.array([[S1_K[0, 2], S1_K[1, 2]]])
        ray = kb4_unproject(center, S1_K, S1_COEFFS)
        np.testing.assert_allclose(ray, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_on_axis_point_projects_to_principal_point(self) -> None:
        uv = kb4_project(np.array([[0.0, 0.0, 5.0]]), S1_K, S1_COEFFS)
        np.testing.assert_allclose(uv, [[S1_K[0, 2], S1_K[1, 2]]], atol=1e-9)


class FacePlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.faces = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE)

    def test_190_deg_plan_has_front_face_plus_one_ring(self) -> None:
        ids = [face.face_id for face in self.faces]
        self.assertEqual(ids[0], "front")
        ring_ids = [i for i in ids if i.startswith("ring0_")]
        self.assertEqual(len(ring_ids), len(ids) - 1, "190 deg should need exactly one ring")
        self.assertGreaterEqual(len(ring_ids), 4)
        self.assertLessEqual(len(ring_ids), MAX_FACES_PER_RING)
        self.assertLessEqual(len(self.faces), 1 + MAX_FACES_PER_RING)

    def test_ring_half_fov_is_within_recipe_clamp(self) -> None:
        for face in self.faces:
            if face.face_id.startswith("ring"):
                self.assertGreaterEqual(face.half_fov_deg, RING_HALF_FOV_MIN_DEG - 1e-9)
                self.assertLessEqual(face.half_fov_deg, RING_HALF_FOV_MAX_DEG + 1e-9)

    def test_full_fov_coverage_is_complete(self) -> None:
        report = face_coverage_check(self.faces, S1_FOV_DEG, samples=20000)
        self.assertEqual(report["uncovered"], 0)
        self.assertEqual(report["uncovered_fraction"], 0.0)
        self.assertGreaterEqual(report["min_faces_per_direction"], 1)

    def test_coverage_check_detects_missing_ring(self) -> None:
        report = face_coverage_check(self.faces[:1], S1_FOV_DEG, samples=5000)
        self.assertGreater(report["uncovered_fraction"], 0.1)

    def test_face_geometry_is_square_and_consistent(self) -> None:
        for face in self.faces:
            self.assertEqual(face.width, face.height)
            f = face.K_face[0, 0]
            self.assertAlmostEqual(face.K_face[1, 1], f)
            np.testing.assert_allclose(
                [face.K_face[0, 2], face.K_face[1, 2]], [face.width / 2.0, face.height / 2.0]
            )
            derived_half_fov = math.degrees(math.atan((face.width / 2.0) / f))
            self.assertAlmostEqual(derived_half_fov, face.half_fov_deg, places=9)

    def test_plan_is_deterministic(self) -> None:
        again = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE)
        self.assertEqual(len(again), len(self.faces))
        for a, b in zip(again, self.faces):
            self.assertEqual(a.face_id, b.face_id)
            np.testing.assert_array_equal(a.R_face, b.R_face)
            np.testing.assert_array_equal(a.K_face, b.K_face)
            self.assertEqual(a.width, b.width)


class FaceResolutionTests(unittest.TestCase):
    def test_resolution_never_exceeds_cap(self) -> None:
        faces = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE, max_face_px=4096)
        for face in faces:
            self.assertLessEqual(face.width, 4096)
            self.assertLessEqual(face.height, 4096)

    def test_resolution_is_monotone_in_source_density(self) -> None:
        base = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE)
        doubled_K = S1_K.copy()
        doubled_K[0, 0] *= 2.0
        doubled_K[1, 1] *= 2.0
        doubled = plan_fisheye_faces(S1_FOV_DEG, doubled_K, S1_COEFFS, S1_IMAGE_SIZE)
        by_id = {face.face_id: face for face in doubled}
        self.assertEqual(set(by_id), {face.face_id for face in base})
        for face in base:
            other = by_id[face.face_id]
            self.assertGreater(other.width, face.width)
            self.assertLessEqual(other.width, 4096)

    def test_huge_source_density_clamps_at_cap(self) -> None:
        big_K = S1_K.copy()
        big_K[0, 0] *= 10.0
        big_K[1, 1] *= 10.0
        faces = plan_fisheye_faces(S1_FOV_DEG, big_K, S1_COEFFS, S1_IMAGE_SIZE)
        self.assertTrue(all(face.width <= 4096 for face in faces))
        self.assertTrue(any(face.width == 4096 for face in faces))


class FaceWeightTests(unittest.TestCase):
    @staticmethod
    def _unit_face() -> FaceSpec:
        return FaceSpec(
            face_id="probe",
            R_face=np.eye(3),
            K_face=np.array([[500.0, 0.0, 500.0], [0.0, 500.0, 500.0], [0.0, 0.0, 1.0]]),
            width=1000,
            height=1000,
            half_fov_deg=45.0,
        )

    def test_center_is_one_and_edge_is_zero(self) -> None:
        face = self._unit_face()
        w = face_weight(face, np.array([[500.0, 500.0], [0.0, 500.0], [1000.0, 500.0], [500.0, 1000.0]]))
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 0.0)
        self.assertAlmostEqual(w[2], 0.0)
        self.assertAlmostEqual(w[3], 0.0)

    def test_fade_starts_at_eighty_percent_from_center(self) -> None:
        face = self._unit_face()
        inner_limit = 500.0 + 500.0 * (1.0 - EDGE_FADE_FRAC)  # u=900: last full-weight point
        w = face_weight(
            face,
            np.array([[750.0, 500.0], [inner_limit, 500.0], [inner_limit + 10.0, 500.0], [975.0, 500.0]]),
        )
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 1.0)
        self.assertLess(w[2], 1.0)
        self.assertGreater(w[2], 0.0)
        self.assertLess(w[3], w[2])

    def test_weight_is_monotone_toward_the_edge(self) -> None:
        face = self._unit_face()
        us = np.linspace(500.0, 1000.0, 101)
        w = face_weight(face, np.column_stack([us, np.full_like(us, 500.0)]))
        self.assertTrue(np.all(np.diff(w) <= 1e-12))

    def test_corner_weight_is_product_of_axes(self) -> None:
        face = self._unit_face()
        w = face_weight(face, np.array([[950.0, 950.0], [950.0, 500.0]]))
        self.assertAlmostEqual(w[0], w[1] ** 2)


class GravityAlignmentTests(unittest.TestCase):
    def test_default_up_hint_keeps_horizontal_axis_level(self) -> None:
        up = np.array([0.0, -1.0, 0.0])
        faces = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE)
        for face in faces:
            x_axis = face.R_face[:, 0]
            y_axis = face.R_face[:, 1]
            self.assertAlmostEqual(float(x_axis @ up), 0.0, places=9, msg=face.face_id)
            self.assertGreaterEqual(float(y_axis @ -up), -1e-9, msg=face.face_id)

    def test_front_face_frame_is_identity_for_default_up(self) -> None:
        faces = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE)
        np.testing.assert_allclose(faces[0].R_face, np.eye(3), atol=1e-12)

    def test_tilted_up_hint_is_respected(self) -> None:
        up = np.array([0.2, -0.9, 0.1])
        up = up / np.linalg.norm(up)
        faces = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE, up_hint_camera=up)
        for face in faces:
            x_axis = face.R_face[:, 0]
            y_axis = face.R_face[:, 1]
            # Upright frame: horizontal axis perpendicular to gravity, image-down
            # axis in the plane spanned by the face axis and gravity.
            self.assertAlmostEqual(float(x_axis @ up), 0.0, places=9, msg=face.face_id)
            self.assertGreaterEqual(float(y_axis @ -up), -1e-9, msg=face.face_id)
            np.testing.assert_allclose(
                face.R_face @ face.R_face.T, np.eye(3), atol=1e-9, err_msg=face.face_id
            )
            self.assertAlmostEqual(float(np.linalg.det(face.R_face)), 1.0, places=9)
        report = face_coverage_check(faces, S1_FOV_DEG, samples=20000)
        self.assertEqual(report["uncovered"], 0)


class FaceSpecSerializationTests(unittest.TestCase):
    def test_to_dict_from_dict_roundtrip_through_json(self) -> None:
        faces = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE)
        for face in faces:
            payload = json.loads(json.dumps(face.to_dict()))
            restored = FaceSpec.from_dict(payload)
            self.assertEqual(restored.face_id, face.face_id)
            self.assertEqual(restored.width, face.width)
            self.assertEqual(restored.height, face.height)
            self.assertEqual(restored.half_fov_deg, face.half_fov_deg)
            np.testing.assert_array_equal(restored.R_face, face.R_face)
            np.testing.assert_array_equal(restored.K_face, face.K_face)


class FacePixelRayConsistencyTests(unittest.TestCase):
    def test_pixels_to_directions_roundtrip(self) -> None:
        faces = plan_fisheye_faces(S1_FOV_DEG, S1_K, S1_COEFFS, S1_IMAGE_SIZE)
        rng = np.random.default_rng(11)
        for face in faces:
            pixels = np.column_stack(
                [rng.uniform(0.5, face.width - 0.5, 200), rng.uniform(0.5, face.height - 0.5, 200)]
            )
            dirs = face.pixels_to_directions(pixels)
            back, inside = face.directions_to_pixels(dirs)
            self.assertTrue(inside.all(), face.face_id)
            self.assertLess(float(np.abs(back - pixels).max()), 1e-9, face.face_id)


@unittest.skipUnless(MANIFEST_PATH.exists(), "real S1 manifest not available on this machine")
class RealS1CalibrationSmokeTests(unittest.TestCase):
    def test_plan_with_real_manifest_calibration(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for camera in manifest["cameras"]:
            intr = camera["intrinsic"]
            params = camera["distortion"]["params"]
            K = np.array(
                [
                    [intr["fl_x"], 0.0, intr["cx"]],
                    [0.0, intr["fl_y"], intr["cy"]],
                    [0.0, 0.0, 1.0],
                ]
            )
            coeffs = np.array([params["k1"], params["k2"], params["k3"], params["k4"]])
            image_size = (camera["width"], camera["height"])
            faces = plan_fisheye_faces(S1_FOV_DEG, K, coeffs, image_size)
            report = face_coverage_check(faces, S1_FOV_DEG, samples=20000)

            total_px = sum(face.width * face.height for face in faces)
            print(f"\n[real-S1 smoke] camera={camera['camera_id']} faces={len(faces)} "
                  f"coverage_uncovered={report['uncovered_fraction']:.4f} "
                  f"mean_faces_per_dir={report['mean_faces_per_direction']:.2f} "
                  f"total_face_px={total_px/1e6:.1f}MP vs source {image_size[0]*image_size[1]/1e6:.1f}MP")
            for face in faces:
                print(f"    {face.face_id:12s} half_fov={face.half_fov_deg:5.1f} deg "
                      f"size={face.width}x{face.height} f={face.K_face[0,0]:.1f}px")

            self.assertEqual(report["uncovered"], 0)
            self.assertGreaterEqual(len(faces), 7)
            self.assertLessEqual(len(faces), 1 + MAX_FACES_PER_RING)
            for face in faces:
                self.assertLessEqual(face.width, 4096)
                self.assertGreaterEqual(face.width, 256, "face resolution implausibly small")


if __name__ == "__main__":
    unittest.main()
