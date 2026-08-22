"""CPU-only synthetic tests for cloudstudio_3dgs.data.face_warp.

FaceSpec is imported from cloudstudio_3dgs.geometry.fisheye_faces when that
module exists; otherwise a minimal stub with the agreed field names
(face_id, R_face, K_face, width, height) is used so this suite does not
depend on the face-planner landing first.
"""

from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.data.face_warp import (
    build_face_warp_grid,
    kb4_project_dirs,
    kb4_unproject_pixels,
    warp_image_to_face,
    warp_mask_to_face,
    warp_sparse_depth_to_face,
)

try:  # pragma: no cover - depends on parallel agent's module landing
    import math

    from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec as _PlannerFaceSpec

    def FaceSpec(*, face_id, R_face, K_face, width, height):
        # Warp only reads the shared duck-typed fields; derive the planner's
        # extra half_fov_deg from the intrinsics so real-class construction
        # keeps exercising interface compatibility.
        half_fov = math.degrees(math.atan((width / 2.0) / float(K_face[0, 0])))
        return _PlannerFaceSpec(
            face_id=face_id,
            R_face=R_face,
            K_face=K_face,
            width=width,
            height=height,
            half_fov_deg=half_fov,
        )
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FaceSpec:  # minimal stub matching the agreed interface contract
        face_id: str
        R_face: np.ndarray  # (3,3) face -> camera
        K_face: np.ndarray  # (3,3) pinhole intrinsics
        width: int
        height: int


def make_K(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


# Synthetic S1-like fisheye camera (scaled down from 2912x2912 for CPU tests).
FISH_W = FISH_H = 512
FISH_K = make_K(210.0, 210.0, 255.5, 255.5)
COEFFS = (-0.05, 0.003, 0.0, 0.0)
ZERO_COEFFS = (0.0, 0.0, 0.0, 0.0)


def render_fisheye_from_direction_fn(value_fn, coeffs, width=FISH_W, height=FISH_H):
    """Analytically render a fisheye image from a function of ray direction."""
    jj, ii = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    uv = np.stack([jj.ravel(), ii.ravel()], axis=1)
    dirs = kb4_unproject_pixels(uv, FISH_K, coeffs)
    values = np.where(dirs[:, 2] > 1e-6, value_fn(dirs), 0.0)
    return values.reshape(height, width)


class TestKB4RoundTrip(unittest.TestCase):
    def test_project_unproject_roundtrip(self):
        rng = np.random.default_rng(42)
        dirs = rng.normal(size=(500, 3))
        dirs[:, 2] = np.abs(dirs[:, 2]) + 0.5
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        uv, valid = kb4_project_dirs(dirs, FISH_K, COEFFS)
        self.assertTrue(valid.all())
        back = kb4_unproject_pixels(uv, FISH_K, COEFFS)
        self.assertLess(np.max(np.abs(back - dirs)), 1e-8)


class TestIdentityFace(unittest.TestCase):
    def test_center_face_matches_center_crop(self):
        # Small-FoV face on the optical axis with the fisheye focal length:
        # near the center the KB4 mapping is ~identical to pinhole, so the
        # warped face must match a plain center crop of the fisheye image.
        size = 48
        cxf = (size - 1) / 2.0  # aligns face pixel j with fisheye col 232 + j
        face = FaceSpec(
            face_id="identity",
            R_face=np.eye(3),
            K_face=make_K(210.0, 210.0, cxf, cxf),
            width=size,
            height=size,
        )
        jj, ii = np.meshgrid(np.arange(FISH_W, dtype=np.float64),
                             np.arange(FISH_H, dtype=np.float64))
        image = 0.5 + 0.25 * np.sin(2 * np.pi * jj / 97.0) + 0.2 * np.cos(2 * np.pi * ii / 71.0)

        face_image, face_valid = warp_image_to_face(
            image, FISH_K, (-0.01, 0.0, 0.0, 0.0), face)
        self.assertTrue(face_valid.all())

        offset = int(round(255.5 - cxf))  # 232
        crop = image[offset:offset + size, offset:offset + size]
        mse = float(np.mean((face_image - crop) ** 2))
        psnr = 10.0 * np.log10(1.0 / max(mse, 1e-20))
        self.assertGreater(psnr, 40.0, f"PSNR {psnr:.1f} dB too low for identity face")


class TestStraightLines(unittest.TestCase):
    def test_gnomonic_lines_are_straight_on_face(self):
        # Pattern of vertical planes x/z = const in the camera frame renders
        # as curved stripes in the fisheye image; on a (rotated) face they
        # must come out perfectly straight -- the zero-distortion criterion.
        period = 0.08
        image = render_fisheye_from_direction_fn(
            lambda d: np.sin(2 * np.pi * (d[:, 0] / np.maximum(d[:, 2], 1e-9)) / period),
            COEFFS,
        )
        face = FaceSpec(
            face_id="yaw15",
            R_face=rot_y(np.deg2rad(15.0)),
            K_face=make_K(250.0, 250.0, 79.5, 79.5),
            width=160,
            height=160,
        )
        face_image, face_valid = warp_image_to_face(image, FISH_K, COEFFS, face)
        self.assertTrue(face_valid.all())

        # Rotation about y keeps these planes vertical in the face, so track
        # per row the ascending zero crossing nearest the face center column.
        crossings = []
        for row in range(8, 152):
            vals = face_image[row]
            sign_change = (vals[:-1] < 0.0) & (vals[1:] >= 0.0)
            cols = np.nonzero(sign_change)[0]
            self.assertGreater(cols.size, 0)
            j = cols[np.argmin(np.abs(cols - 80))]
            sub = j + vals[j] / (vals[j] - vals[j + 1])
            crossings.append((row, sub))
        rows = np.array([c[0] for c in crossings], dtype=np.float64)
        cols = np.array([c[1] for c in crossings], dtype=np.float64)
        a, b = np.polyfit(rows, cols, 1)
        residual = np.max(np.abs(cols - (a * rows + b)))
        self.assertLess(residual, 0.5, f"line fit residual {residual:.3f} px")

        # Sanity: the same stripe is genuinely curved in the fisheye domain,
        # otherwise this test would not prove the warp removes distortion.
        fish_crossings = []
        prev = 340.0
        for row in range(140, 372):
            vals = image[row]
            sign_change = (vals[:-1] < 0.0) & (vals[1:] >= 0.0)
            cols_f = np.nonzero(sign_change)[0]
            if cols_f.size:
                j = cols_f[np.argmin(np.abs(cols_f - prev))]
                sub = j + vals[j] / (vals[j] - vals[j + 1])
                if abs(sub - prev) < 6.0:  # track one stripe, tolerate drift
                    fish_crossings.append((row, sub))
                    prev = sub
        rows_f = np.array([c[0] for c in fish_crossings])
        cols_f = np.array([c[1] for c in fish_crossings])
        af, bf = np.polyfit(rows_f, cols_f, 1)
        fish_residual = np.max(np.abs(cols_f - (af * rows_f + bf)))
        self.assertGreater(fish_residual, 1.0,
                           "fixture stripe is not curved in the fisheye image")


class TestMaskConservative(unittest.TestCase):
    def test_hole_is_conservative_with_no_bleed(self):
        face = FaceSpec(
            face_id="center64",
            R_face=np.eye(3),
            K_face=make_K(210.0, 210.0, 31.5, 31.5),
            width=64,
            height=64,
        )
        mask = np.ones((FISH_H, FISH_W), dtype=bool)
        y0h, y1h, x0h, x1h = 244, 258, 238, 252
        mask[y0h:y1h, x0h:x1h] = False

        grid = build_face_warp_grid(FISH_K, COEFFS, face)
        face_mask, face_valid = warp_mask_to_face(mask, FISH_K, COEFFS, face, grid=grid)
        self.assertTrue(face_valid.all())
        self.assertGreater(np.count_nonzero(~face_mask), 0)
        self.assertGreater(np.count_nonzero(face_mask), 0)

        # No bleed: any face pixel whose bilinear footprint touches the hole
        # must be False; any pixel whose footprint is fully outside must be True.
        xf = np.floor(grid.u).astype(int)
        yf = np.floor(grid.v).astype(int)
        touches = (xf + 1 >= x0h) & (xf < x1h) & (yf + 1 >= y0h) & (yf < y1h)
        self.assertFalse(np.any(face_mask & touches), "invalid source bled into face mask")
        self.assertTrue(np.array_equal(face_mask, ~touches))


class TestSparseDepthRoundTrip(unittest.TestCase):
    def test_forward_splat_roundtrip_under_2cm(self):
        rng = np.random.default_rng(7)
        face = FaceSpec(
            face_id="pitch10",
            R_face=rot_x(np.deg2rad(10.0)),
            K_face=make_K(300.0, 300.0, 63.5, 63.5),
            width=128,
            height=128,
        )
        # Sparse sources at exact integer fisheye pixels around the face
        # center so the only discretization is the face-side rounding.
        center_dir = face.R_face @ np.array([0.0, 0.0, 1.0])
        center_uv, _ = kb4_project_dirs(center_dir[None, :], FISH_K, COEFFS)
        n = 400
        px = np.unique(np.stack([
            np.clip(np.rint(center_uv[0, 0] + rng.uniform(-38, 38, n)), 0, FISH_W - 1),
            np.clip(np.rint(center_uv[0, 1] + rng.uniform(-38, 38, n)), 0, FISH_H - 1),
        ], axis=1).astype(int), axis=0)
        dirs = kb4_unproject_pixels(px.astype(np.float64), FISH_K, COEFFS)
        ranges = rng.uniform(1.0, 4.0, px.shape[0])
        points_cam = dirs * ranges[:, None]

        depth = np.zeros((FISH_H, FISH_W))
        conf = np.zeros((FISH_H, FISH_W))
        valid = np.zeros((FISH_H, FISH_W), dtype=bool)
        depth[px[:, 1], px[:, 0]] = ranges
        conf[px[:, 1], px[:, 0]] = 0.25 + 0.5 * (ranges / 4.0)
        valid[px[:, 1], px[:, 0]] = True

        face_range, face_conf, face_valid = warp_sparse_depth_to_face(
            depth, conf, valid, FISH_K, COEFFS, face)

        hits = np.count_nonzero(face_valid)
        self.assertGreaterEqual(hits, int(0.8 * px.shape[0]))
        self.assertTrue(np.all(face_conf[face_valid] > 0.0))
        self.assertTrue(np.all(face_range[~face_valid] == 0.0))

        vs, us = np.nonzero(face_valid)
        pix_h = np.stack([us.astype(float), vs.astype(float), np.ones(us.size)], axis=1)
        dirs_face = pix_h @ np.linalg.inv(face.K_face).T
        dirs_face /= np.linalg.norm(dirs_face, axis=1, keepdims=True)
        rec_cam = (dirs_face * face_range[vs, us][:, None]) @ face.R_face.T
        dist = np.linalg.norm(rec_cam[:, None, :] - points_cam[None, :, :], axis=2)
        nearest = dist.min(axis=1)
        self.assertLess(float(nearest.max()), 0.02,
                        f"round-trip error {nearest.max() * 100:.2f} cm")

    def test_zbuffer_keeps_smaller_range_and_its_confidence(self):
        # Coarse face: adjacent fisheye pixels collapse into one face pixel.
        face = FaceSpec(
            face_id="coarse",
            R_face=np.eye(3),
            K_face=make_K(50.0, 50.0, 15.5, 15.5),
            width=32,
            height=32,
        )
        depth = np.zeros((FISH_H, FISH_W))
        conf = np.zeros((FISH_H, FISH_W))
        valid = np.zeros((FISH_H, FISH_W), dtype=bool)
        depth[256, 256], conf[256, 256], valid[256, 256] = 3.0, 0.9, True
        depth[256, 257], conf[256, 257], valid[256, 257] = 2.0, 0.4, True

        face_range, face_conf, face_valid = warp_sparse_depth_to_face(
            depth, conf, valid, FISH_K, COEFFS, face)
        self.assertEqual(np.count_nonzero(face_valid), 1)
        v, u = [int(a[0]) for a in np.nonzero(face_valid)]
        self.assertAlmostEqual(face_range[v, u], 2.0)
        self.assertAlmostEqual(face_conf[v, u], 0.4)


class TestInvalidRegions(unittest.TestCase):
    def test_pixels_outside_image_bounds_are_invalid(self):
        # Face yawed far to the side: part of it projects beyond the fisheye
        # raster edge and must be invalid + zero-filled.
        face = FaceSpec(
            face_id="edge",
            R_face=rot_y(np.deg2rad(60.0)),
            K_face=make_K(150.0, 150.0, 63.5, 63.5),
            width=128,
            height=128,
        )
        image = np.full((FISH_H, FISH_W), 0.7)
        face_image, face_valid = warp_image_to_face(image, FISH_K, ZERO_COEFFS, face)
        self.assertTrue(np.any(face_valid))
        self.assertTrue(np.any(~face_valid))
        grid = build_face_warp_grid(FISH_K, ZERO_COEFFS, face)
        outside = (grid.u < 0) | (grid.u > FISH_W - 1) | (grid.v < 0) | (grid.v > FISH_H - 1)
        self.assertFalse(np.any(face_valid & outside))
        self.assertTrue(np.all(face_image[~face_valid] == 0.0))

    def test_pixels_beyond_fov_cone_are_invalid(self):
        # FoV limit tighter than the face extent: corners invalid, center valid.
        face = FaceSpec(
            face_id="fov",
            R_face=np.eye(3),
            K_face=make_K(100.0, 100.0, 63.5, 63.5),
            width=128,
            height=128,
        )
        image = np.full((FISH_H, FISH_W), 0.3)
        _, face_valid = warp_image_to_face(
            image, FISH_K, ZERO_COEFFS, face, max_theta_rad=0.3)
        self.assertTrue(face_valid[64, 64])
        self.assertFalse(face_valid[0, 0])
        self.assertFalse(face_valid[0, 127])
        self.assertFalse(face_valid[127, 0])
        self.assertFalse(face_valid[127, 127])

    def test_precomputed_grid_matches_direct_call(self):
        face = FaceSpec(
            face_id="cached",
            R_face=rot_y(np.deg2rad(20.0)),
            K_face=make_K(220.0, 220.0, 47.5, 47.5),
            width=96,
            height=96,
        )
        rng = np.random.default_rng(3)
        image = rng.random((FISH_H, FISH_W, 3))
        grid = build_face_warp_grid(FISH_K, COEFFS, face)
        img_a, valid_a = warp_image_to_face(image, FISH_K, COEFFS, face)
        img_b, valid_b = warp_image_to_face(image, FISH_K, COEFFS, face, grid=grid)
        self.assertTrue(np.array_equal(valid_a, valid_b))
        self.assertTrue(np.array_equal(img_a, img_b))


if __name__ == "__main__":
    unittest.main()
