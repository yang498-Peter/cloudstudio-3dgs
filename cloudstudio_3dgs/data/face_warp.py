"""GT resampling (warp) from the fisheye image domain to planar face domains.

Fisheye camera model: OPENCV_FISHEYE / Kannala-Brandt KB4.
    theta   = angle between ray and optical axis
    theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    u = fx * theta_d * x/r + cx,  v = fy * theta_d * y/r + cy,  r = hypot(x, y)

FaceSpec contract (interface owned by ``cloudstudio_3dgs.geometry.fisheye_faces``,
which may land after this module; this module only duck-types the attributes):
    face_id : hashable identifier
    R_face  : (3, 3) rotation matrix, face -> camera (dir_cam = R_face @ dir_face)
    K_face  : (3, 3) pinhole intrinsics of the face raster
    width   : face raster width in pixels
    height  : face raster height in pixels

Pixel convention (both domains): array index (row i, col j) has its pixel
center at continuous coordinate (u=j, v=i). K and K_face are expected to use
the same convention.

Implementation choice: pure numpy (vectorized gather) instead of
``torch.nn.functional.grid_sample``. Rationale: keeps this module free of any
torch import so it can never touch CUDA state while training is running,
avoids grid_sample's align_corners/normalized-coordinate pitfalls, and CPU
numpy gather is fast enough for a per-(camera, face) cached grid workflow
(the grid is built once via :func:`build_face_warp_grid` and reused per image).

Range semantics: sparse depth carries Euclidean ray range in meters. A pure
rotation between face frame and camera frame does not change the range, so
ranges are carried through the splat unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FaceWarpGrid",
    "kb4_project_dirs",
    "kb4_unproject_pixels",
    "build_face_warp_grid",
    "warp_image_to_face",
    "warp_mask_to_face",
    "warp_sparse_depth_to_face",
]

_EPS = 1e-12


def _k_params(K: np.ndarray) -> Tuple[float, float, float, float]:
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _radial(radial_coeffs: Sequence[float]) -> Tuple[float, float, float, float]:
    coeffs = [float(c) for c in radial_coeffs]
    if len(coeffs) != 4:
        raise ValueError("radial_coeffs must contain exactly 4 KB4 coefficients")
    return coeffs[0], coeffs[1], coeffs[2], coeffs[3]


def kb4_project_dirs(
    dirs: np.ndarray,
    K: np.ndarray,
    radial_coeffs: Sequence[float],
    *,
    max_theta_rad: float = np.pi,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project camera-frame direction vectors through the KB4 fisheye model.

    Returns (uv[N, 2] float64, valid[N] bool). ``valid`` is False for
    non-finite results and for rays whose off-axis angle exceeds
    ``max_theta_rad`` (the fisheye FoV limit). Image-bounds checks are the
    caller's job (they need the image size).
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("dirs must have shape [N, 3]")
    fx, fy, cx, cy = _k_params(K)
    k1, k2, k3, k4 = _radial(radial_coeffs)

    x, y, z = dirs.T
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.divide(theta_d, r, out=np.zeros_like(theta_d), where=r > _EPS)
    uv = np.column_stack([fx * x * scale + cx, fy * y * scale + cy])
    valid = np.isfinite(uv).all(axis=1) & (theta <= max_theta_rad)
    return uv, valid


def kb4_unproject_pixels(
    uv: np.ndarray,
    K: np.ndarray,
    radial_coeffs: Sequence[float],
    *,
    tol: float = 1e-9,
    max_iterations: int = 100,
) -> np.ndarray:
    """Unproject fisheye pixel coordinates to unit direction vectors.

    Solves theta from theta_d with Newton iterations until the largest update
    is below ``tol`` (or ``max_iterations`` is reached).
    """
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape [N, 2]")
    fx, fy, cx, cy = _k_params(K)
    k1, k2, k3, k4 = _radial(radial_coeffs)

    xd = (uv[:, 0] - cx) / fx
    yd = (uv[:, 1] - cy) / fy
    theta_d = np.hypot(xd, yd)
    theta = theta_d.copy()
    for _ in range(max_iterations):
        t2 = theta * theta
        f_val = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4) - theta_d
        f_der = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t2**2 + 7.0 * k3 * t2**3 + 9.0 * k4 * t2**4
        step = np.divide(f_val, f_der, out=np.zeros_like(f_val), where=np.abs(f_der) > _EPS)
        theta -= step
        if np.max(np.abs(step), initial=0.0) < tol:
            break
    sin_scale = np.divide(np.sin(theta), theta_d, out=np.ones_like(theta), where=theta_d > _EPS)
    dirs = np.column_stack([xd * sin_scale, yd * sin_scale, np.cos(theta)])
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


@dataclass(frozen=True)
class FaceWarpGrid:
    """Precomputed fisheye sampling coordinates for one (camera, face) pair.

    ``u``/``v`` are (height, width) float64 fisheye pixel coordinates for every
    face pixel center; ``fov_valid`` marks face pixels whose ray is finite and
    inside the fisheye FoV cone (``max_theta_rad``). Image-bounds validity is
    applied at warp time from the actual source image shape.
    """

    face_id: object
    u: np.ndarray
    v: np.ndarray
    fov_valid: np.ndarray
    max_theta_rad: float


def build_face_warp_grid(
    K: np.ndarray,
    radial_coeffs: Sequence[float],
    face,
    *,
    max_theta_rad: float = np.pi,
) -> FaceWarpGrid:
    """Build the reusable face->fisheye sampling grid for one face.

    The grid depends only on (K, radial_coeffs, face geometry), so it is
    constant for a given camera+face and should be cached and reused across
    all images during training.
    """
    width = int(face.width)
    height = int(face.height)
    K_face = np.asarray(face.K_face, dtype=np.float64)
    R_face = np.asarray(face.R_face, dtype=np.float64)
    if K_face.shape != (3, 3) or R_face.shape != (3, 3):
        raise ValueError("face.K_face and face.R_face must be 3x3 matrices")

    # The face planner and the gsplat rasterizer both place pixel centers at
    # (i + 0.5, j + 0.5): sample the face at those centers and convert the
    # projected source coordinate from the same center convention back to
    # array-index space (u - 0.5) for the bilinear gather, so warped GT stays
    # aligned with what gsplat renders through either camera model.
    jj, ii = np.meshgrid(np.arange(width, dtype=np.float64) + 0.5,
                         np.arange(height, dtype=np.float64) + 0.5)
    pix_h = np.stack([jj.ravel(), ii.ravel(), np.ones(width * height)], axis=1)
    dirs_face = pix_h @ np.linalg.inv(K_face).T
    dirs_cam = dirs_face @ R_face.T  # dir_cam = R_face @ dir_face, row-vector form
    uv, valid = kb4_project_dirs(dirs_cam, K, radial_coeffs, max_theta_rad=max_theta_rad)
    return FaceWarpGrid(
        face_id=getattr(face, "face_id", None),
        u=uv[:, 0].reshape(height, width) - 0.5,
        v=uv[:, 1].reshape(height, width) - 0.5,
        fov_valid=valid.reshape(height, width),
        max_theta_rad=float(max_theta_rad),
    )


def _bilinear_footprint(u, v, src_h, src_w):
    """Integer 4-neighbor footprint and in-bounds mask for bilinear sampling.

    Conservative: a target pixel is in-bounds only when all 4 contributing
    source pixels exist.
    """
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    inb = (x0 >= 0) & (x0 + 1 <= src_w - 1) & (y0 >= 0) & (y0 + 1 <= src_h - 1)
    x0c = np.clip(x0, 0, src_w - 2)
    y0c = np.clip(y0, 0, src_h - 2)
    wx = np.clip(u - x0c, 0.0, 1.0)
    wy = np.clip(v - y0c, 0.0, 1.0)
    return x0c, y0c, wx, wy, inb


def warp_image_to_face(
    fisheye_image: np.ndarray,
    K: np.ndarray,
    radial_coeffs: Sequence[float],
    face,
    *,
    interpolation: str = "bilinear",
    grid: Optional[FaceWarpGrid] = None,
    max_theta_rad: float = np.pi,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a fisheye GT image onto a planar face raster.

    Returns (face_image, face_valid_mask). Invalid face pixels (outside the
    fisheye image bounds or outside the fisheye FoV) are zero-filled and
    marked False. ``grid`` may be a precomputed :func:`build_face_warp_grid`
    result for this (K, radial_coeffs, face).

    RGB/float data should use ``interpolation="bilinear"`` (output float32,
    or float64 when the input is float64); ``"nearest"`` preserves dtype.
    """
    image = np.asarray(fisheye_image)
    if image.ndim not in (2, 3):
        raise ValueError("fisheye_image must be HxW or HxWxC")
    if grid is None:
        grid = build_face_warp_grid(K, radial_coeffs, face, max_theta_rad=max_theta_rad)
    src_h, src_w = image.shape[:2]
    u, v = grid.u, grid.v

    if interpolation == "bilinear":
        x0, y0, wx, wy, inb = _bilinear_footprint(u, v, src_h, src_w)
        valid = grid.fov_valid & inb
        out_dtype = np.float64 if image.dtype == np.float64 else np.float32
        img = image.astype(out_dtype, copy=False)
        if img.ndim == 2:
            img = img[:, :, None]
        wxe = wx[..., None]
        wye = wy[..., None]
        top = img[y0, x0] * (1.0 - wxe) + img[y0, x0 + 1] * wxe
        bot = img[y0 + 1, x0] * (1.0 - wxe) + img[y0 + 1, x0 + 1] * wxe
        face_image = (top * (1.0 - wye) + bot * wye).astype(out_dtype)
        face_image[~valid] = 0
        if image.ndim == 2:
            face_image = face_image[:, :, 0]
        return face_image, valid
    if interpolation == "nearest":
        xi = np.rint(u).astype(np.int64)
        yi = np.rint(v).astype(np.int64)
        inb = (xi >= 0) & (xi <= src_w - 1) & (yi >= 0) & (yi <= src_h - 1)
        valid = grid.fov_valid & inb
        face_image = image[np.clip(yi, 0, src_h - 1), np.clip(xi, 0, src_w - 1)].copy()
        face_image[~valid] = 0
        return face_image, valid
    raise ValueError(f"unsupported interpolation: {interpolation!r}")


def warp_mask_to_face(
    fisheye_mask: np.ndarray,
    K: np.ndarray,
    radial_coeffs: Sequence[float],
    face,
    *,
    grid: Optional[FaceWarpGrid] = None,
    max_theta_rad: float = np.pi,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a boolean validity mask onto a face raster, conservatively.

    A face pixel is True only when its geometric mapping is valid AND every
    source pixel in its bilinear 4-neighbor footprint is inside the fisheye
    image and True. Any contributing invalid source pixel invalidates the
    face pixel (no bleeding of invalid data across the resampling kernel).

    Returns (face_mask, face_valid_mask): ``face_valid_mask`` is the purely
    geometric validity (FoV + image bounds); ``face_mask`` additionally
    requires the source mask to be True over the whole footprint.
    """
    mask = np.asarray(fisheye_mask)
    if mask.ndim != 2:
        raise ValueError("fisheye_mask must be HxW")
    mask = mask.astype(bool, copy=False)
    if grid is None:
        grid = build_face_warp_grid(K, radial_coeffs, face, max_theta_rad=max_theta_rad)
    src_h, src_w = mask.shape
    x0, y0, _, _, inb = _bilinear_footprint(grid.u, grid.v, src_h, src_w)
    valid = grid.fov_valid & inb
    all_true = mask[y0, x0] & mask[y0, x0 + 1] & mask[y0 + 1, x0] & mask[y0 + 1, x0 + 1]
    face_mask = valid & all_true
    return face_mask, valid


def warp_sparse_depth_to_face(
    depth_range_m: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    K: np.ndarray,
    radial_coeffs: Sequence[float],
    face,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-splat a sparse fisheye range map onto a face raster.

    Sparse depth is NOT resampled by inverse interpolation (that would blend
    unrelated depths across holes). Instead every valid fisheye depth pixel is
    unprojected (KB4), rotated into the face frame, projected through K_face,
    and rounded to its face pixel. Conflicts z-buffer to the smallest range;
    the winning pixel's confidence is carried along with it.

    Range is Euclidean ray distance in meters and is invariant under the pure
    rotation between camera and face frames, so values are carried unchanged.

    Returns (face_range, face_conf, face_valid); non-hit pixels are 0/0/False.
    """
    depth = np.asarray(depth_range_m, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    valid = np.asarray(valid).astype(bool, copy=False)
    if depth.shape != valid.shape or conf.shape != valid.shape or depth.ndim != 2:
        raise ValueError("depth_range_m, confidence and valid must share an HxW shape")

    width = int(face.width)
    height = int(face.height)
    K_face = np.asarray(face.K_face, dtype=np.float64)
    R_face = np.asarray(face.R_face, dtype=np.float64)

    face_range = np.zeros((height, width), dtype=np.float64)
    face_conf = np.zeros((height, width), dtype=np.float64)
    face_valid = np.zeros((height, width), dtype=bool)

    ys, xs = np.nonzero(valid & np.isfinite(depth) & (depth > 0.0))
    if ys.size == 0:
        return face_range, face_conf, face_valid

    # Array index (x, y) -> pixel-center coordinate (+0.5) before unprojecting,
    # and the face-plane projection comes back in center coordinates, so the
    # nearest array index is rint(coord - 0.5). Keeps the splat on the same
    # pixel-center convention as the planner, the warp grid, and gsplat.
    uv = np.stack(
        [xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5], axis=1
    )
    dirs_cam = kb4_unproject_pixels(uv, K, radial_coeffs)
    dirs_face = dirs_cam @ R_face  # dir_face = R_face.T @ dir_cam, row-vector form
    z = dirs_face[:, 2]
    front = z > _EPS
    if not np.any(front):
        return face_range, face_conf, face_valid

    fxf, fyf, cxf, cyf = _k_params(K_face)
    uf = np.rint(fxf * dirs_face[front, 0] / z[front] + cxf - 0.5).astype(np.int64)
    vf = np.rint(fyf * dirs_face[front, 1] / z[front] + cyf - 0.5).astype(np.int64)
    ranges = depth[ys[front], xs[front]]
    confs = conf[ys[front], xs[front]]
    inb = (uf >= 0) & (uf < width) & (vf >= 0) & (vf < height)
    uf, vf, ranges, confs = uf[inb], vf[inb], ranges[inb], confs[inb]
    if ranges.size == 0:
        return face_range, face_conf, face_valid

    # Z-buffer: write in descending range order so the smallest range (and its
    # confidence) lands last and wins every collision.
    order = np.argsort(-ranges, kind="stable")
    vf, uf, ranges, confs = vf[order], uf[order], ranges[order], confs[order]
    face_range[vf, uf] = ranges
    face_conf[vf, uf] = confs
    face_valid[vf, uf] = True
    return face_range, face_conf, face_valid
