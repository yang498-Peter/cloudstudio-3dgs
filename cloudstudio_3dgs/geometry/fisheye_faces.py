"""Fisheye face-split planner: expand one KB4 fisheye camera into K zero-distortion
pinhole sub-cameras ("faces") so full-resolution training never renders through a
>90-degree distorted projection (Spirula warp_to_pinhole recipe, docs
2026-08-22_Spirula吸收分析与训练核心补强计划 §2.6).

Pure geometry, numpy only. No GPU, no torch.

Conventions (load-bearing for the GT warp module `cloudstudio_3dgs/data/face_warp.py`):

- Camera frame is OpenCV: +x right, +y down, +z forward along the optical axis.
- ``FaceSpec.R_face`` is the face->camera rotation. Its *columns* are the face frame
  axes (x_f, y_f, z_f) expressed in camera coordinates, and it is stored/serialized
  row-major (``R_face[i][j]`` = row i, column j), so ``d_cam = R_face @ d_face``.
- Face pinhole: continuous pixel coordinates, image spans [0, width] x [0, height],
  principal point at (width/2, height/2); pixel-center sampling grids should use
  (i + 0.5, j + 0.5). ``u = fx * x/z + cx``.
- ``half_fov_deg`` measures from the face center to the *midpoint of a square edge*
  (the apothem): ``tan(half_fov) = (width/2) / fx``. Corners reach farther.
- KB4 / OPENCV_FISHEYE: ``theta_d = theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)``,
  ``r = f * theta_d``.

Planner recipe constants come from the Spirula GeometryWarp analysis:
front face +-45 deg x1.1 overlap; 1..3 outer rings of 4..12 upright square faces
(half-FOV clamped 30..40 deg), rings press inward past the front-face edge
(inner reach 40 deg), extend outward to lens FoV + 3 deg, ~15% azimuthal overlap
between neighbours and 10 deg polar overlap between rings; faces are gravity
aligned (upright frames); per-face resolution equals the source lens'
pixels-per-radian density at the face center (measured by numerical
differentiation), capped at 4096; fusion weight fades to 0 with a smoothstep over
the outer 20% of each face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

# --- Recipe constants (GeometryWarp.cpp top-of-file constants, re-derived) -------
FOV_MARGIN_DEG = 3.0  # rings extend to lens FoV + 3 deg so rim droop between faces still covers the FoV
RING_INNER_PRESS_DEG = 40.0  # rings must reach inward to 40 deg polar, under the front-face edge
RING_OVERLAP_DEG = 10.0  # polar overlap between adjacent rings
RING_HALF_FOV_MIN_DEG = 30.0
RING_HALF_FOV_MAX_DEG = 40.0
MIN_FACES_PER_RING = 4
MAX_FACES_PER_RING = 12
MAX_RINGS = 3
FRONT_OVERLAP_FACTOR = 1.1  # front face covers +-front_half_fov * 1.1
EDGE_FADE_FRAC = 0.2  # face_weight smoothsteps 1 -> 0 over the outer 20% of the face
MIN_FACE_PX = 64

_EPS = 1e-12
_COVERAGE_GRID_STEP_DEG = 0.5  # planner-internal dense verification grid


# =============================== KB4 model ======================================


def _split_intrinsics(K: Any) -> tuple[float, float, float, float]:
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _split_coeffs(radial_coeffs: Any) -> tuple[float, float, float, float]:
    coeffs = np.asarray(radial_coeffs, dtype=np.float64).reshape(-1)
    if coeffs.shape[0] != 4:
        raise ValueError("radial_coeffs must contain exactly 4 values (k1..k4)")
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])


def kb4_project(points_cam: np.ndarray, K: Any, radial_coeffs: Any) -> np.ndarray:
    """Project camera-frame points/directions through the KB4 fisheye model.

    ``points_cam``: [N, 3]. Only the direction matters (KB4 is a central model).
    Returns pixels [N, 2] (continuous coordinates). Directions at theta up to
    (but excluding) pi are supported; the exact backward axis maps to NaN-free
    output only in the radial limit and is not used by the planner.
    """
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_cam must have shape [N, 3]")
    fx, fy, cx, cy = _split_intrinsics(K)
    k1, k2, k3, k4 = _split_coeffs(radial_coeffs)

    x, y, z = pts.T
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    theta_d = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = np.divide(theta_d, r, out=np.zeros_like(theta_d), where=r > _EPS)
    return np.column_stack([fx * x * scale + cx, fy * y * scale + cy])


def kb4_unproject(
    pixels: np.ndarray,
    K: Any,
    radial_coeffs: Any,
    *,
    tol: float = 1e-9,
    max_iterations: int = 20,
) -> np.ndarray:
    """Invert KB4: pixels [N, 2] -> unit direction rays [N, 3] in the camera frame.

    Solves ``theta_d = theta * poly(theta^2)`` for theta with Newton iteration
    (initial guess theta = theta_d, step tolerance ``tol`` rad, at most
    ``max_iterations`` iterations).
    """
    px = np.asarray(pixels, dtype=np.float64)
    if px.ndim != 2 or px.shape[1] != 2:
        raise ValueError("pixels must have shape [N, 2]")
    fx, fy, cx, cy = _split_intrinsics(K)
    k1, k2, k3, k4 = _split_coeffs(radial_coeffs)

    xd = (px[:, 0] - cx) / fx
    yd = (px[:, 1] - cy) / fy
    theta_d = np.hypot(xd, yd)
    theta = theta_d.copy()
    for _ in range(max_iterations):
        t2 = theta * theta
        value = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
        derivative = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
        step = np.divide(value, derivative, out=np.zeros_like(value), where=np.abs(derivative) > _EPS)
        theta -= step
        if float(np.abs(step).max(initial=0.0)) < tol:
            break
    sin_scale = np.divide(np.sin(theta), theta_d, out=np.ones_like(theta), where=theta_d > _EPS)
    rays = np.column_stack([xd * sin_scale, yd * sin_scale, np.cos(theta)])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


# =============================== FaceSpec =======================================


@dataclass
class FaceSpec:
    """One zero-distortion pinhole sub-camera of a fisheye face split.

    ``R_face`` maps face-frame vectors into the parent camera frame
    (``d_cam = R_face @ d_face``); its columns are the face axes in camera
    coordinates and it serializes row-major. ``K_face`` is the pinhole intrinsic
    matrix for a ``width`` x ``height`` image with principal point at the image
    center. Adaptive faces are square; fixed product-derived plans may use
    rectangular rasters.
    """

    face_id: str
    R_face: np.ndarray = field(repr=False)
    K_face: np.ndarray = field(repr=False)
    width: int
    height: int
    half_fov_deg: float

    def __post_init__(self) -> None:
        self.R_face = np.asarray(self.R_face, dtype=np.float64)
        self.K_face = np.asarray(self.K_face, dtype=np.float64)
        if self.R_face.shape != (3, 3) or self.K_face.shape != (3, 3):
            raise ValueError("R_face and K_face must be 3x3 matrices")
        if not np.allclose(self.R_face @ self.R_face.T, np.eye(3), atol=1e-8):
            raise ValueError("R_face must be orthonormal")
        if np.linalg.det(self.R_face) < 0.0:
            raise ValueError("R_face must be a proper rotation (det=+1)")
        self.width = int(self.width)
        self.height = int(self.height)
        self.half_fov_deg = float(self.half_fov_deg)

    # -- geometry helpers (used by the planner, tests, and the GT warp module) --

    @property
    def center_direction_camera(self) -> np.ndarray:
        """Face optical axis (z_f) in camera coordinates."""
        return self.R_face[:, 2].copy()

    def directions_to_pixels(self, directions_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame directions [N, 3] -> (face pixels [N, 2], inside mask [N])."""
        d = np.asarray(directions_cam, dtype=np.float64)
        if d.ndim != 2 or d.shape[1] != 3:
            raise ValueError("directions_cam must have shape [N, 3]")
        df = d @ self.R_face  # == (R_face.T @ d.T).T
        z = df[:, 2]
        safe_z = np.where(np.abs(z) > _EPS, z, _EPS)
        u = self.K_face[0, 0] * df[:, 0] / safe_z + self.K_face[0, 2]
        v = self.K_face[1, 1] * df[:, 1] / safe_z + self.K_face[1, 2]
        pixels = np.column_stack([u, v])
        slack = 1e-9 * max(self.width, self.height)
        inside = (
            (z > _EPS)
            & (u >= -slack)
            & (u <= self.width + slack)
            & (v >= -slack)
            & (v <= self.height + slack)
        )
        return pixels, inside

    def contains(self, directions_cam: np.ndarray) -> np.ndarray:
        """Boolean mask: which camera-frame directions fall inside this face frustum."""
        _pixels, inside = self.directions_to_pixels(directions_cam)
        return inside

    def pixels_to_directions(self, pixels: np.ndarray) -> np.ndarray:
        """Face pixels [N, 2] -> unit camera-frame directions [N, 3]."""
        px = np.asarray(pixels, dtype=np.float64)
        if px.ndim != 2 or px.shape[1] != 2:
            raise ValueError("pixels must have shape [N, 2]")
        x = (px[:, 0] - self.K_face[0, 2]) / self.K_face[0, 0]
        y = (px[:, 1] - self.K_face[1, 2]) / self.K_face[1, 1]
        rays_face = np.column_stack([x, y, np.ones_like(x)])
        rays_face /= np.linalg.norm(rays_face, axis=1, keepdims=True)
        return rays_face @ self.R_face.T  # R_face @ ray, batched

    # -- serialization ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_id": self.face_id,
            "R_face": self.R_face.tolist(),  # row-major nested lists
            "K_face": self.K_face.tolist(),
            "width": self.width,
            "height": self.height,
            "half_fov_deg": self.half_fov_deg,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FaceSpec":
        return cls(
            face_id=str(payload["face_id"]),
            R_face=np.asarray(payload["R_face"], dtype=np.float64),
            K_face=np.asarray(payload["K_face"], dtype=np.float64),
            width=int(payload["width"]),
            height=int(payload["height"]),
            half_fov_deg=float(payload["half_fov_deg"]),
        )


# =============================== fusion weight ==================================


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def face_weight(face: FaceSpec, pixels: np.ndarray) -> np.ndarray:
    """Seam-fusion weight for face pixels [N, 2] -> [N] in [0, 1].

    Weight is 1 over the central 80% of each axis and smoothsteps to 0 across the
    outer ``EDGE_FADE_FRAC`` (20%) so blended faces are C1 across seams. Pixels
    outside the face get 0. Per-axis weights multiply, keeping corners C1 too.
    """
    px = np.asarray(pixels, dtype=np.float64)
    if px.ndim != 2 or px.shape[1] != 2:
        raise ValueError("pixels must have shape [N, 2]")
    half = np.array([face.width / 2.0, face.height / 2.0], dtype=np.float64)
    normalized = np.abs((px - half) / half)  # 0 at center, 1 at the edge
    edge_distance = 1.0 - normalized  # fraction of the half-size left to the edge
    t = np.clip(edge_distance / EDGE_FADE_FRAC, 0.0, 1.0)
    w = _smoothstep(t)
    return w[:, 0] * w[:, 1]


# =============================== planner ========================================


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        raise ValueError("zero-length vector")
    return v / n


def _upright_frame(center_dir: np.ndarray, up_camera: np.ndarray) -> np.ndarray:
    """Gravity-aligned face frame: z_f = center, y_f = projected gravity *down*
    (image y points down), x_f = y_f x z_f (horizontal, perpendicular to gravity).

    Falls back to the camera +z axis as the vertical reference when the face
    points along gravity (upright is then undefined).
    """
    z_f = _normalize(center_dir)
    down = -_normalize(up_camera)
    y_f = down - float(down @ z_f) * z_f
    if float(np.linalg.norm(y_f)) < 1e-6:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(float(fallback @ z_f)) > 0.999:
            fallback = np.array([1.0, 0.0, 0.0])
        y_f = fallback - float(fallback @ z_f) * z_f
    y_f = _normalize(y_f)
    x_f = np.cross(y_f, z_f)
    return np.column_stack([x_f, y_f, z_f])


def _source_density_px_per_rad(
    center_dir: np.ndarray, R: np.ndarray, K: Any, radial_coeffs: Any, eps_rad: float = 1e-4
) -> float:
    """Source-lens pixel density (px/rad) at ``center_dir``, measured by central
    differences of the KB4 forward projection along the two face tangent axes.
    Returns the max of the two directional densities (avoid undersampling)."""
    c = _normalize(center_dir)
    densities = []
    for axis in (R[:, 0], R[:, 1]):
        d_plus = math.cos(eps_rad) * c + math.sin(eps_rad) * axis
        d_minus = math.cos(eps_rad) * c - math.sin(eps_rad) * axis
        uv = kb4_project(np.vstack([d_plus, d_minus]), K, radial_coeffs)
        densities.append(float(np.linalg.norm(uv[0] - uv[1])) / (2.0 * eps_rad))
    return max(densities)


def _make_face(
    face_id: str,
    center_dir: np.ndarray,
    up_camera: np.ndarray,
    half_fov_deg: float,
    K: Any,
    radial_coeffs: Any,
    max_face_px: int,
) -> FaceSpec:
    R = _upright_frame(center_dir, up_camera)
    tan_h = math.tan(math.radians(half_fov_deg))
    density = _source_density_px_per_rad(center_dir, R, K, radial_coeffs)
    width = int(math.ceil(2.0 * density * tan_h))
    width += width % 2  # even
    width = int(np.clip(width, MIN_FACE_PX, max_face_px))
    f = width / (2.0 * tan_h)  # re-derive focal so the half-FOV stays exact after rounding/clamping
    K_face = np.array([[f, 0.0, width / 2.0], [0.0, f, width / 2.0], [0.0, 0.0, 1.0]])
    return FaceSpec(
        face_id=face_id,
        R_face=R,
        K_face=K_face,
        width=width,
        height=width,
        half_fov_deg=float(half_fov_deg),
    )


def _polar_dirs(theta_rad: np.ndarray, azimuth_rad: np.ndarray) -> np.ndarray:
    sin_t = np.sin(theta_rad)
    return np.column_stack(
        [sin_t * np.cos(azimuth_rad), sin_t * np.sin(azimuth_rad), np.cos(theta_rad)]
    )


def _radial_face_contains(tilt_rad: float, half_fov_rad: float, dirs: np.ndarray) -> np.ndarray:
    """Membership in a meridian-aligned (non-upright) square face at azimuth 0.
    Used only to seed the per-ring face-count estimate."""
    z_f = np.array([math.sin(tilt_rad), 0.0, math.cos(tilt_rad)])
    x_f = np.array([0.0, 1.0, 0.0])
    y_f = np.array([-math.cos(tilt_rad), 0.0, math.sin(tilt_rad)])
    R = np.column_stack([x_f, y_f, z_f])
    df = dirs @ R
    tan_h = math.tan(half_fov_rad)
    z = df[:, 2]
    return (z > _EPS) & (np.abs(df[:, 0]) <= z * tan_h) & (np.abs(df[:, 1]) <= z * tan_h)


def _ring_face_count_seed(
    tilt_rad: float, half_fov_rad: float, theta_worst_rad: float, overlap_frac: float
) -> int:
    """Initial faces-per-ring estimate: azimuthal half-width of one face at the
    worst polar angle it must cover, with ``overlap_frac`` azimuthal overlap."""
    lo, hi = 0.0, math.pi
    dirs0 = _polar_dirs(np.array([theta_worst_rad]), np.array([0.0]))
    if not bool(_radial_face_contains(tilt_rad, half_fov_rad, dirs0)[0]):
        return MAX_FACES_PER_RING
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d = _polar_dirs(np.array([theta_worst_rad]), np.array([mid]))
        if bool(_radial_face_contains(tilt_rad, half_fov_rad, d)[0]):
            lo = mid
        else:
            hi = mid
    step = 2.0 * lo * (1.0 - overlap_frac)
    if step <= 0.0:
        return MAX_FACES_PER_RING
    count = int(math.ceil(2.0 * math.pi / step))
    return int(np.clip(count, MIN_FACES_PER_RING, MAX_FACES_PER_RING))


def _coverage_grid(fov_half_deg: float) -> np.ndarray:
    n_polar = int(round(fov_half_deg / _COVERAGE_GRID_STEP_DEG)) + 1
    theta = np.radians(np.linspace(0.0, fov_half_deg, n_polar))
    azimuth = np.radians(np.arange(0.0, 360.0, _COVERAGE_GRID_STEP_DEG))
    tt, aa = np.meshgrid(theta, azimuth, indexing="ij")
    return _polar_dirs(tt.reshape(-1), aa.reshape(-1))


def _uncovered_polar_deg(faces: Sequence[FaceSpec], grid_dirs: np.ndarray) -> np.ndarray:
    covered = np.zeros(grid_dirs.shape[0], dtype=bool)
    for face in faces:
        covered |= face.contains(grid_dirs)
        if covered.all():
            return np.empty(0)
    missing = grid_dirs[~covered]
    return np.degrees(np.arccos(np.clip(missing[:, 2], -1.0, 1.0)))


def _build_rings(
    tilts_deg: Sequence[float],
    counts: Sequence[int],
    half_fov_deg: float,
    azimuth0_rad: float,
    up_camera: np.ndarray,
    K: Any,
    radial_coeffs: Any,
    max_face_px: int,
) -> list[FaceSpec]:
    faces: list[FaceSpec] = []
    for ring_idx, (tilt_deg, count) in enumerate(zip(tilts_deg, counts)):
        tilt = math.radians(tilt_deg)
        stagger = (math.pi / count) * (ring_idx % 2)  # offset alternate rings half a step
        for k in range(count):
            azimuth = azimuth0_rad + stagger + 2.0 * math.pi * k / count
            center = np.array(
                [
                    math.sin(tilt) * math.cos(azimuth),
                    math.sin(tilt) * math.sin(azimuth),
                    math.cos(tilt),
                ]
            )
            faces.append(
                _make_face(
                    f"ring{ring_idx}_face{k}",
                    center,
                    up_camera,
                    half_fov_deg,
                    K,
                    radial_coeffs,
                    max_face_px,
                )
            )
    return faces


def plan_fisheye_faces(
    fisheye_fov_deg: float,
    K: Any,
    radial_coeffs: Any,
    image_size: tuple[int, int],
    *,
    up_hint_camera: np.ndarray | None = None,
    max_face_px: int = 4096,
    front_half_fov_deg: float = 45.0,
    overlap_frac: float = 0.15,
) -> list[FaceSpec]:
    """Plan the pinhole face split for one KB4 fisheye camera.

    ``fisheye_fov_deg``: full lens field of view (e.g. 190 for the S1 lenses).
    ``K`` / ``radial_coeffs``: source KB4 intrinsics (3x3 matrix, k1..k4).
    ``image_size``: source (width, height) in pixels; sanity metadata only, the
    per-face resolution comes from the measured pixel density, not from here.
    ``up_hint_camera``: gravity *up* in camera coordinates (defaults to -y, the
    OpenCV upright camera). Faces use gravity-aligned upright frames because the
    downstream mono depth/normal networks carry a gravity prior.

    The planner seeds ring tilts / face counts from the recipe constants, then
    verifies direction coverage on a dense polar grid and bumps face counts (and
    if needed ring half-FOV) until the fisheye FoV is fully covered. Raises
    ``RuntimeError`` if no configuration within the recipe bounds covers the FoV.
    """
    if not 10.0 <= fisheye_fov_deg <= 250.0:
        raise ValueError("fisheye_fov_deg out of supported range [10, 250]")
    width_src, height_src = int(image_size[0]), int(image_size[1])
    if width_src <= 0 or height_src <= 0:
        raise ValueError("image_size must be positive")
    up = _normalize(up_hint_camera) if up_hint_camera is not None else np.array([0.0, -1.0, 0.0])

    fov_half = fisheye_fov_deg / 2.0
    front_half = front_half_fov_deg * FRONT_OVERLAP_FACTOR
    front = _make_face(
        "front", np.array([0.0, 0.0, 1.0]), up, front_half, K, radial_coeffs, max_face_px
    )
    grid = _coverage_grid(fov_half)

    if fov_half <= front_half:
        if _uncovered_polar_deg([front], grid).size:
            raise RuntimeError("front face unexpectedly fails to cover the fisheye FoV")
        return [front]

    theta_max = fov_half + FOV_MARGIN_DEG
    span = theta_max - RING_INNER_PRESS_DEG
    n_rings = None
    for candidate in range(1, MAX_RINGS + 1):
        if (span + (candidate - 1) * RING_OVERLAP_DEG) / (2.0 * candidate) <= RING_HALF_FOV_MAX_DEG:
            n_rings = candidate
            break
    if n_rings is None:
        raise RuntimeError("fisheye FoV too wide for the ring recipe bounds")
    h_needed = (span + (n_rings - 1) * RING_OVERLAP_DEG) / (2.0 * n_rings)
    h0 = float(np.clip(h_needed, RING_HALF_FOV_MIN_DEG, RING_HALF_FOV_MAX_DEG))

    # Azimuth origin: projection of gravity-down onto the sensor plane, so faces
    # aligned with gravity sit at cardinal azimuths (zero upright twist there).
    down = -up
    down_planar = down - down[2] * np.array([0.0, 0.0, 1.0])
    if float(np.linalg.norm(down_planar)) < 1e-6:
        azimuth0 = 0.0
    else:
        azimuth0 = math.atan2(down_planar[1], down_planar[0])

    h_candidates = list(np.arange(h0, RING_HALF_FOV_MAX_DEG, 2.0)) + [RING_HALF_FOV_MAX_DEG]
    for h in h_candidates:
        tilts = [theta_max - h - i * (2.0 * h - RING_OVERLAP_DEG) for i in range(n_rings)]
        counts = []
        for i, tilt in enumerate(tilts):
            if i == 0:
                theta_worst = min(tilt + h, fov_half)
            else:
                theta_worst = tilt + h - RING_OVERLAP_DEG / 2.0
            counts.append(
                _ring_face_count_seed(
                    math.radians(tilt), math.radians(h), math.radians(theta_worst), overlap_frac
                )
            )
        for _ in range((MAX_FACES_PER_RING + 1) * n_rings):
            faces = [front] + _build_rings(
                tilts, counts, h, azimuth0, up, K, radial_coeffs, max_face_px
            )
            missing_polar = _uncovered_polar_deg(faces, grid)
            if missing_polar.size == 0:
                return faces
            worst_polar = float(missing_polar.max())
            ring_idx = int(np.argmin([abs(t - worst_polar) for t in tilts]))
            bumped = False
            for offset in range(n_rings):  # bump the responsible ring, else any ring with slack
                idx = (ring_idx + offset) % n_rings
                if counts[idx] < MAX_FACES_PER_RING:
                    counts[idx] += 1
                    bumped = True
                    break
            if not bumped:
                break  # all rings saturated at this half-FOV; widen the faces
    raise RuntimeError(
        "face planning failed to reach full FoV coverage within recipe bounds "
        f"(fov={fisheye_fov_deg} deg)"
    )


def plan_mipmap_face4(image_size: tuple[int, int]) -> list[FaceSpec]:
    """Return the four fixed pinhole views recovered from MipMap snow output.

    The product first optimizes the physical raw-fisheye cameras, then derives
    this cross-shaped set without introducing additional pose variables. The
    reference geometry was measured from ``mvs_undistort.xml`` for a
    2912-by-2912 source. Resolutions and focal lengths scale together for a
    square source of another resolution, preserving the exact FOVs.
    """
    source_width, source_height = (int(image_size[0]), int(image_size[1]))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("image_size must be positive")
    if source_width != source_height:
        raise ValueError("MipMap Face4 requires a square source image")

    scale = source_width / 2912.0

    def scaled_even(value: int) -> int:
        result = max(2, int(round(value * scale)))
        return result + result % 2

    def rotation_x(angle_deg: float) -> np.ndarray:
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=np.float64,
        )

    def rotation_y(angle_deg: float) -> np.ndarray:
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        return np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float64,
        )

    def fixed_face(
        face_id: str,
        rotation: np.ndarray,
        reference_width: int,
        reference_height: int,
        reference_focal_px: float,
    ) -> FaceSpec:
        width = scaled_even(reference_width)
        height = scaled_even(reference_height)
        focal = float(reference_focal_px) * scale
        K_face = np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        half_fov_deg = math.degrees(
            math.atan(max(width, height) / (2.0 * focal))
        )
        return FaceSpec(
            face_id=face_id,
            R_face=rotation,
            K_face=K_face,
            width=width,
            height=height,
            half_fov_deg=half_fov_deg,
        )

    return [
        fixed_face("yaw_minus_35", rotation_y(-35.0), 1456, 2912, 1039.691749),
        fixed_face("yaw_plus_35", rotation_y(35.0), 1456, 2912, 1039.691749),
        fixed_face("pitch_up_56", rotation_x(56.0), 2912, 1456, 2308.921016),
        fixed_face("pitch_down_56", rotation_x(-56.0), 2912, 1456, 2308.921016),
    ]


# =============================== coverage check =================================


def face_coverage_check(
    faces: Sequence[FaceSpec],
    fisheye_fov_deg: float,
    samples: int = 10000,
    *,
    seed: int = 20260823,
) -> dict[str, Any]:
    """Monte-Carlo verification that the face set covers the whole fisheye FoV.

    Samples ``samples`` directions uniformly (by solid angle) over the spherical
    cap theta <= fov/2 about +z and tests membership against every face.
    ``uncovered_fraction`` must be 0.0 for a valid plan.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    cos_min = math.cos(math.radians(fisheye_fov_deg / 2.0))
    cos_theta = rng.uniform(cos_min, 1.0, samples)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    azimuth = rng.uniform(0.0, 2.0 * math.pi, samples)
    dirs = _polar_dirs(theta, azimuth)

    hits = np.zeros(samples, dtype=np.int64)
    for face in faces:
        hits += face.contains(dirs).astype(np.int64)
    uncovered = int(np.count_nonzero(hits == 0))
    return {
        "samples": int(samples),
        "uncovered": uncovered,
        "uncovered_fraction": uncovered / float(samples),
        "min_faces_per_direction": int(hits.min()),
        "max_faces_per_direction": int(hits.max()),
        "mean_faces_per_direction": float(hits.mean()),
    }
