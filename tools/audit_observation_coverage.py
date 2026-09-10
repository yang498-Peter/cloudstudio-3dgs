#!/usr/bin/env python3
"""Audit how well a set of 3D surface samples is actually observed by the
training photos (WP03 "effective observation" diagnostic).

For every sample point the tool reports, over the training parent images:

* ``n_images_fisheye``   images whose raw KB4 fisheye sees the point (in FoV,
                         inside the sensor, range window)
* ``n_images_face``      images where the point also lands inside at least one
                         Face4 pinhole face (what the trainer actually sees)
* ``n_effective_views``  face views that are RGB-valid (face mask) and not
                         occluded according to the Face4 LiDAR geometry
* ``n_cameras`` / ``n_rig_frames``   physical cameras (left/right) and rig
                         frames among the effective views
* ``time_span_s``        capture-time span of the effective views
* ``angle_span_deg``     largest pairwise angle between viewing directions of
                         the effective views
* ``footprint_px_per_m_fisheye`` / ``..._face``   median pixels-per-metre at
                         the sample (numeric Jacobian of the projection)
* ``frac_occluded``      occluded / (occluded + depth-supported) face views
* ``depth_support_frac`` face views with a consistent LiDAR return next to the
                         projection (vis6 sparse geometry)
* ``rgb_valid_frac``     face views whose face mask is true at the projection
* ``sharpness_lapvar``   median Laplacian variance in a 64 px window of the
                         ORIGINAL photo (not the face cache) over the effective
                         views, plus the best view's value
* ``luma_mean``          median window luminance of the original photo

Reused repository code (cited so the numbers are comparable with training):

* ``cloudstudio_3dgs.geometry.kb4.project_kb4`` for the raw fisheye model,
* ``cloudstudio_3dgs.geometry.fisheye_faces.FaceSpec`` for the Face4 pinhole
  faces (``directions_to_pixels`` is the planner/warp convention, pixel
  centres at ``i + 0.5``),
* ``cloudstudio_3dgs.data.depth_cache.load_sparse_depth`` for the vis6 sparse
  LiDAR range rasters,
* ``cloudstudio_3dgs.geometry.lidar_projection.DepthProjectionConfig`` for the
  occlusion tolerance/margin semantics (same numbers as the vis6 build).

Sample points are drawn from the tile initialization PLY (a box crop of the
recording LAS, see ``cloudstudio_3dgs/training/tile_inputs.py``) inside each
region's world box; the region file documents how each box was derived.

Example::

    python tools/audit_observation_coverage.py \
        --dataset-manifest C:/Peter/3dgs-datasets/house0305_sop_v8/dataset_manifest.json \
        --face-manifest C:/Peter/3dgs-datasets/house0305_sop_v9/face4_train/face_manifest.json \
        --lidar-geometry-manifest C:/Peter/3dgs-datasets/house0305_sop_v9/face4_lidar_train_vis6/face_lidar_geometry_manifest.json \
        --tile-inputs-manifest C:/Peter/3dgs-runs/house0305_sop/tile_inputs_v9/tile_inputs_manifest.json \
        --recording-root "C:/baidunetdiskdownload/house/2026-03-05_10-58-54 - house" \
        --regions research/quality_recovery_v2/03_roi_provisional.json \
        --out-csv research/quality_recovery_v2/03_observation_coverage.csv \
        --out-json research/quality_recovery_v2/03_observation_coverage_summary.json

The core (``audit_samples``) is independent of the manifests so it can be unit
tested with a synthetic camera / point setup (``tests/test_audit_observation_coverage.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth  # noqa: E402
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402
from cloudstudio_3dgs.geometry.kb4 import project_kb4  # noqa: E402
from cloudstudio_3dgs.geometry.lidar_projection import DepthProjectionConfig  # noqa: E402

SHARPNESS_WINDOW_PX = 64
SUPPORT_SEARCH_RADIUS_PX = 6  # 13x13 px window; the vis6 raster has ~2.4 % valid pixels
# Strict LiDAR-support band used to declare a view "effective": the sample must
# have a return within 0.1 m + 3 % of its range in the search window and no
# nearer return. The loose vis6 rule (20 % + 0.1 m) is reported alongside
# because it is what the depth supervision itself uses; at 7 m it accepts a
# wall 1.5 m in front of the sample as "consistent".
STRICT_TOLERANCE = 0.03
STRICT_MARGIN_M = 0.1
NEAR_VIEW_RANGE_M = 5.0
FOOTPRINT_STEP_M = 0.01
MAX_PAIRWISE_VIEWS = 400


# ----------------------------------------------------------------------------
# data classes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraModel:
    """Raw fisheye (KB4 / OPENCV_FISHEYE) camera as stored in the dataset manifest."""

    camera_id: str
    intrinsic: Mapping[str, float]
    distortion: Mapping[str, float]
    width: int
    height: int
    max_theta_rad: float = math.radians(95.0)

    @classmethod
    def from_manifest(cls, camera: Mapping[str, Any], *, max_theta_deg: float = 95.0) -> "CameraModel":
        return cls(
            camera_id=str(camera["camera_id"]),
            intrinsic=dict(camera["intrinsic"]),
            distortion=dict(camera["distortion"]["params"]),
            width=int(camera["width"]),
            height=int(camera["height"]),
            max_theta_rad=math.radians(max_theta_deg),
        )

    def project(self, points_camera: np.ndarray, *, min_range_m: float, max_range_m: float):
        """Pixel-centre coordinates (i + 0.5 convention), ranges, validity."""
        uv, ranges, valid = project_kb4(
            points_camera,
            self.intrinsic,
            self.distortion,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            max_theta_rad=self.max_theta_rad,
        )
        inside = (uv[:, 0] >= 0.0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < self.height)
        return uv, ranges, valid & inside


@dataclass(frozen=True)
class ViewRecord:
    image_id: str
    camera_id: str
    rig_frame_id: str
    timestamp_ns: int
    c2w: np.ndarray  # 4x4, camera->world, OpenCV camera axes


@dataclass
class Region:
    label: str
    tile: str
    world_box: np.ndarray  # (2, 3) min/max
    status: str = "provisional"
    derivation: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------


def world_to_camera(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return (points_world - t) @ R  # R^T (X - t) in row-vector form


def tangent_basis(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to each direction (N,3)."""
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    helper = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    return e1, e2


def fisheye_footprint_px_per_m(camera: CameraModel, points_camera: np.ndarray) -> np.ndarray:
    """sqrt(|det J|) of the KB4 projection w.r.t. a metre of surface
    perpendicular to the viewing ray, evaluated numerically."""
    e1, e2 = tangent_basis(points_camera)
    uv0, _, _ = project_kb4(points_camera, camera.intrinsic, camera.distortion)
    uv1, _, _ = project_kb4(points_camera + FOOTPRINT_STEP_M * e1, camera.intrinsic, camera.distortion)
    uv2, _, _ = project_kb4(points_camera + FOOTPRINT_STEP_M * e2, camera.intrinsic, camera.distortion)
    j1 = (uv1 - uv0) / FOOTPRINT_STEP_M
    j2 = (uv2 - uv0) / FOOTPRINT_STEP_M
    det = j1[:, 0] * j2[:, 1] - j1[:, 1] * j2[:, 0]
    return np.sqrt(np.abs(det))


def face_footprint_px_per_m(face: FaceSpec, points_camera: np.ndarray) -> np.ndarray:
    e1, e2 = tangent_basis(points_camera)
    p0, _ = face.directions_to_pixels(points_camera)
    p1, _ = face.directions_to_pixels(points_camera + FOOTPRINT_STEP_M * e1)
    p2, _ = face.directions_to_pixels(points_camera + FOOTPRINT_STEP_M * e2)
    j1 = (p1 - p0) / FOOTPRINT_STEP_M
    j2 = (p2 - p0) / FOOTPRINT_STEP_M
    det = j1[:, 0] * j2[:, 1] - j1[:, 1] * j2[:, 0]
    return np.sqrt(np.abs(det))


def pixel_centre_to_index(coords: np.ndarray) -> np.ndarray:
    """Pixel-centre coordinate -> array index, same as the face warp/splat (rint(c - 0.5))."""
    return np.rint(np.asarray(coords, dtype=np.float64) - 0.5).astype(np.int64)


def max_pairwise_angle_deg(directions: np.ndarray, rng: np.random.Generator | None = None) -> float:
    if len(directions) < 2:
        return float("nan")
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    if len(d) > MAX_PAIRWISE_VIEWS:
        rng = rng or np.random.default_rng(0)
        d = d[rng.choice(len(d), MAX_PAIRWISE_VIEWS, replace=False)]
    cosines = np.clip(d @ d.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosines.min())))


# ----------------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------------


def laplacian_4(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian (same kernel as cv2.Laplacian ksize=1), zero at the border."""
    g = np.asarray(gray, dtype=np.float32)
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return out


class WindowStats:
    """Integral-image based mean / variance of an image inside square windows."""

    def __init__(self, image: np.ndarray):
        img = np.asarray(image, dtype=np.float64)
        self.h, self.w = img.shape
        self.s1 = np.zeros((self.h + 1, self.w + 1), dtype=np.float64)
        self.s2 = np.zeros((self.h + 1, self.w + 1), dtype=np.float64)
        self.s1[1:, 1:] = img.cumsum(0).cumsum(1)
        self.s2[1:, 1:] = (img * img).cumsum(0).cumsum(1)

    def _box(self, s: np.ndarray, y0, y1, x0, x1) -> np.ndarray:
        return s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]

    def mean_var(self, cx: np.ndarray, cy: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        half = window // 2
        x0 = np.clip(cx - half, 0, self.w)
        x1 = np.clip(cx - half + window, 0, self.w)
        y0 = np.clip(cy - half, 0, self.h)
        y1 = np.clip(cy - half + window, 0, self.h)
        n = np.maximum((x1 - x0) * (y1 - y0), 1).astype(np.float64)
        m1 = self._box(self.s1, y0, y1, x0, x1) / n
        m2 = self._box(self.s2, y0, y1, x0, x1) / n
        return m1, np.maximum(m2 - m1 * m1, 0.0)


def sharpness_maps(gray: np.ndarray) -> tuple[WindowStats, WindowStats]:
    """(luminance stats, Laplacian stats) for windowed queries on one photo."""
    return WindowStats(gray), WindowStats(laplacian_4(gray))


# ----------------------------------------------------------------------------
# occlusion / support classification
# ----------------------------------------------------------------------------

STATUS_NO_LIDAR = 0
STATUS_SUPPORTED = 1
STATUS_OCCLUDED = 2
STATUS_BEHIND = 3  # LiDAR raster only has returns *behind* the sample here
# merge priority across the faces of one photo: occluded > supported > behind > none
STATUS_RANK = np.array([0, 2, 3, 1], dtype=np.int8)


def classify_support(
    dense_range: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    sample_range: np.ndarray,
    *,
    radius_px: int = SUPPORT_SEARCH_RADIUS_PX,
    tolerance: float = 0.2,
    margin_m: float = 0.1,
) -> np.ndarray:
    """Classify each (px, py, range) against a dense LiDAR range raster (0 = no return).

    Uses the vis6 hidden-point rule: a return at range r_l occludes the sample
    when ``sample_range > (1 + tolerance) * r_l + margin``. A return within that
    band supports the sample. Returns farther than the sample mean the raster
    saw *through* where the sample sits (sample absent from the LAS raster or
    sparse) -> BEHIND, which is neither support nor occlusion.
    """
    nearest = nearest_return_in_window(dense_range, radius_px)[np.asarray(py), np.asarray(px)]
    return classify_against_nearest(nearest, sample_range, tolerance=tolerance, margin_m=margin_m)


def classify_against_nearest(nearest: np.ndarray, sample_range: np.ndarray, *, tolerance: float, margin_m: float) -> np.ndarray:
    """Status from the nearest LiDAR return next to each projection (inf = none)."""
    nearest = np.asarray(nearest, dtype=np.float64)
    r = np.asarray(sample_range, dtype=np.float64)
    out = np.full(len(r), STATUS_NO_LIDAR, dtype=np.int8)
    has = np.isfinite(nearest)
    occluded = has & (r > (1.0 + tolerance) * nearest + margin_m)
    supported = has & ~occluded & (nearest <= (1.0 + tolerance) * r + margin_m)
    behind = has & ~occluded & ~supported
    out[occluded] = STATUS_OCCLUDED
    out[supported] = STATUS_SUPPORTED
    out[behind] = STATUS_BEHIND
    return out


def nearest_return_in_window(dense_range: np.ndarray, radius_px: int) -> np.ndarray:
    """Separable (2r+1)^2 min filter over positive returns; inf where the window is empty."""
    work = np.where(dense_range > 0.0, dense_range, np.inf).astype(np.float64)
    if radius_px <= 0:
        return work
    h, w = work.shape
    padded = np.pad(work, radius_px, mode="constant", constant_values=np.inf)
    rows = work.copy()
    for dy in range(2 * radius_px + 1):
        np.minimum(rows, padded[dy : dy + h, radius_px : radius_px + w], out=rows)
    padded = np.pad(rows, radius_px, mode="constant", constant_values=np.inf)
    out = rows.copy()
    for dx in range(2 * radius_px + 1):
        np.minimum(out, padded[radius_px : radius_px + h, dx : dx + w], out=out)
    return out


# ----------------------------------------------------------------------------
# core audit
# ----------------------------------------------------------------------------

FaceGeometryLoader = Callable[[str, str], Optional[np.ndarray]]  # -> dense range HxW (0 = none)
FaceMaskLoader = Callable[[str, str], Optional[np.ndarray]]  # -> bool HxW
PhotoLoader = Callable[[str], Optional[np.ndarray]]  # -> gray float32 HxW


def audit_samples(
    samples_world: np.ndarray,
    views: Sequence[ViewRecord],
    cameras: Mapping[str, CameraModel],
    faces_by_camera: Mapping[str, Sequence[FaceSpec]],
    *,
    face_geometry: FaceGeometryLoader,
    face_mask: FaceMaskLoader,
    photo: PhotoLoader,
    projection: DepthProjectionConfig | None = None,
    sharpness_window: int = SHARPNESS_WINDOW_PX,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Per-sample observation statistics. See the module docstring for fields."""
    cfg = projection or DepthProjectionConfig()
    n = len(samples_world)
    samples_world = np.asarray(samples_world, dtype=np.float64)

    # per-sample accumulators (lists of per-view scalars)
    fisheye_hits = np.zeros(n, dtype=np.int32)
    face_hits = np.zeros(n, dtype=np.int32)
    status_counts = np.zeros((n, 4), dtype=np.int32)  # loose (vis6) rule
    strict_counts = np.zeros((n, 4), dtype=np.int32)  # strict band
    rgb_valid_hits = np.zeros(n, dtype=np.int32)
    loose_ok_hits = np.zeros(n, dtype=np.int32)  # rgb valid & not occluded under the loose rule
    near_eff_hits = np.zeros(n, dtype=np.int32)
    eff_views: list[list[tuple[str, str, str, int]]] = [[] for _ in range(n)]
    eff_dirs: list[list[np.ndarray]] = [[] for _ in range(n)]
    eff_ranges: list[list[float]] = [[] for _ in range(n)]
    eff_foot_fish: list[list[float]] = [[] for _ in range(n)]
    eff_foot_face: list[list[float]] = [[] for _ in range(n)]
    eff_sharp: list[list[float]] = [[] for _ in range(n)]
    eff_luma: list[list[float]] = [[] for _ in range(n)]
    eff_best: list[tuple[float, str, float, float]] = [(-1.0, "", float("nan"), float("nan")) for _ in range(n)]
    fisheye_ranges: list[list[float]] = [[] for _ in range(n)]

    t0 = time.time()
    for vi, view in enumerate(views):
        camera = cameras[view.camera_id]
        pc = world_to_camera(samples_world, view.c2w)
        uv, ranges, valid = camera.project(pc, min_range_m=cfg.min_range_m, max_range_m=cfg.max_range_m)
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        fisheye_hits[idx] += 1
        for i in idx:
            fisheye_ranges[i].append(float(ranges[i]))
        pc_v = pc[idx]
        foot_fish = fisheye_footprint_px_per_m(camera, pc_v)
        # face pass: best (highest-weight) face is not needed; count every face the
        # trainer would sample the point in, and take the per-view effective
        # verdict as "any face valid & not occluded".
        in_any_face = np.zeros(idx.size, dtype=bool)
        any_valid_unoccluded = np.zeros(idx.size, dtype=bool)
        any_valid_strict = np.zeros(idx.size, dtype=bool)
        any_rgb_valid = np.zeros(idx.size, dtype=bool)
        view_status = np.full(idx.size, STATUS_NO_LIDAR, dtype=np.int8)
        view_strict = np.full(idx.size, STATUS_NO_LIDAR, dtype=np.int8)
        best_face_foot = np.full(idx.size, np.nan)
        for face in faces_by_camera[view.camera_id]:
            pix, inside = face.directions_to_pixels(pc_v)
            if not inside.any():
                continue
            j = np.flatnonzero(inside)
            in_any_face[j] = True
            px = np.clip(pixel_centre_to_index(pix[j, 0]), 0, face.width - 1)
            py = np.clip(pixel_centre_to_index(pix[j, 1]), 0, face.height - 1)
            dense = face_geometry(view.image_id, face.face_id)
            if dense is not None:
                nearest = nearest_return_in_window(dense, SUPPORT_SEARCH_RADIUS_PX)[py, px]
                st = classify_against_nearest(nearest, ranges[idx[j]], tolerance=cfg.visibility_tolerance, margin_m=cfg.visibility_margin_m)
                st_strict = classify_against_nearest(nearest, ranges[idx[j]], tolerance=STRICT_TOLERANCE, margin_m=STRICT_MARGIN_M)
            else:
                st = np.full(j.size, STATUS_NO_LIDAR, dtype=np.int8)
                st_strict = st.copy()
            mask = face_mask(view.image_id, face.face_id)
            rgb_ok = mask[py, px] if mask is not None else np.ones(j.size, dtype=bool)
            any_rgb_valid[j] |= rgb_ok
            # status priority: occluded > supported > behind > none (a point occluded
            # in one face of the same photo is occluded for that photo)
            prev = view_status[j]
            view_status[j] = np.where(STATUS_RANK[st] > STATUS_RANK[prev], st, prev)
            prev = view_strict[j]
            view_strict[j] = np.where(STATUS_RANK[st_strict] > STATUS_RANK[prev], st_strict, prev)
            ff = face_footprint_px_per_m(face, pc_v[j])
            best_face_foot[j] = np.where(np.isnan(best_face_foot[j]), ff, np.maximum(best_face_foot[j], ff))
            any_valid_unoccluded[j] |= rgb_ok & (st != STATUS_OCCLUDED)
            any_valid_strict[j] |= rgb_ok & (st_strict == STATUS_SUPPORTED)
        face_idx = np.flatnonzero(in_any_face)
        if face_idx.size == 0:
            continue
        face_hits[idx[face_idx]] += 1
        rgb_valid_hits[idx[face_idx[any_rgb_valid[face_idx]]]] += 1
        for k in face_idx:
            status_counts[idx[k], view_status[k]] += 1
            strict_counts[idx[k], view_strict[k]] += 1
        loose_ok = np.flatnonzero(any_valid_unoccluded & (view_status != STATUS_OCCLUDED))
        loose_ok_hits[idx[loose_ok]] += 1
        # effective = RGB valid and positive strict LiDAR evidence that the sample is
        # the front surface in this photo
        eff = np.flatnonzero(any_valid_strict & (view_strict != STATUS_OCCLUDED))
        if eff.size == 0:
            continue
        near_eff_hits[idx[eff[ranges[idx[eff]] <= NEAR_VIEW_RANGE_M]]] += 1
        gray = photo(view.image_id)
        if gray is not None:
            luma_stats, lap_stats = sharpness_maps(gray)
            cx = pixel_centre_to_index(uv[idx[eff], 0])
            cy = pixel_centre_to_index(uv[idx[eff], 1])
            luma, _ = luma_stats.mean_var(cx, cy, sharpness_window)
            _, lapvar = lap_stats.mean_var(cx, cy, sharpness_window)
        else:
            luma = np.full(eff.size, np.nan)
            lapvar = np.full(eff.size, np.nan)
        cam_centre = view.c2w[:3, 3]
        for e_i, k in enumerate(eff):
            s = idx[k]
            eff_views[s].append((view.image_id, view.camera_id, view.rig_frame_id, view.timestamp_ns))
            eff_dirs[s].append(cam_centre - samples_world[s])
            eff_ranges[s].append(float(ranges[s]))
            eff_foot_fish[s].append(float(foot_fish[k]))
            eff_foot_face[s].append(float(best_face_foot[k]))
            eff_sharp[s].append(float(lapvar[e_i]))
            eff_luma[s].append(float(luma[e_i]))
            if np.isfinite(lapvar[e_i]) and lapvar[e_i] > eff_best[s][0]:
                eff_best[s] = (float(lapvar[e_i]), view.image_id, float(ranges[s]), float(luma[e_i]))
        if log and (vi % 50 == 0 or vi == len(views) - 1):
            log(f"  view {vi + 1}/{len(views)} ({time.time() - t0:.0f}s)")

    rows: list[dict[str, Any]] = []
    for s in range(n):
        occl = int(status_counts[s, STATUS_OCCLUDED])
        supp = int(status_counts[s, STATUS_SUPPORTED])
        behind = int(status_counts[s, STATUS_BEHIND])
        none = int(status_counts[s, STATUS_NO_LIDAR])
        s_occl = int(strict_counts[s, STATUS_OCCLUDED])
        s_supp = int(strict_counts[s, STATUS_SUPPORTED])
        s_behind = int(strict_counts[s, STATUS_BEHIND])
        nf = int(face_hits[s])
        ne = len(eff_views[s])
        ts = [v[3] for v in eff_views[s]]
        rows.append(
            {
                "x": float(samples_world[s, 0]),
                "y": float(samples_world[s, 1]),
                "z": float(samples_world[s, 2]),
                "n_images_fisheye": int(fisheye_hits[s]),
                "n_images_face": nf,
                "n_views_loose_ok": int(loose_ok_hits[s]),
                "n_effective_views": ne,
                "n_effective_views_le5m": int(near_eff_hits[s]),
                "n_strict_occluded": s_occl,
                "n_strict_supported": s_supp,
                "n_strict_behind": s_behind,
                "strict_frac_occluded": s_occl / (s_occl + s_supp) if (s_occl + s_supp) else float("nan"),
                "strict_support_frac": s_supp / nf if nf else float("nan"),
                "n_cameras": len({v[1] for v in eff_views[s]}),
                "n_rig_frames": len({v[2] for v in eff_views[s]}),
                "time_span_s": (max(ts) - min(ts)) / 1e9 if len(ts) >= 2 else float("nan"),
                "angle_span_deg": max_pairwise_angle_deg(np.asarray(eff_dirs[s])) if ne >= 2 else float("nan"),
                "range_min_m": float(min(eff_ranges[s])) if ne else float("nan"),
                "range_median_m": float(np.median(eff_ranges[s])) if ne else (
                    float(np.median(fisheye_ranges[s])) if fisheye_ranges[s] else float("nan")),
                "footprint_px_per_m_fisheye": float(np.median(eff_foot_fish[s])) if ne else float("nan"),
                "footprint_px_per_m_face": float(np.nanmedian(eff_foot_face[s])) if ne else float("nan"),
                "n_occluded": occl,
                "n_depth_supported": supp,
                "n_lidar_behind": behind,
                "n_no_lidar": none,
                "frac_occluded": occl / (occl + supp) if (occl + supp) else float("nan"),
                "depth_support_frac": supp / nf if nf else float("nan"),
                "rgb_valid_frac": int(rgb_valid_hits[s]) / nf if nf else float("nan"),
                "sharpness_lapvar_median": float(np.nanmedian(eff_sharp[s])) if ne else float("nan"),
                "sharpness_lapvar_best": eff_best[s][0] if eff_best[s][0] >= 0 else float("nan"),
                "sharpness_best_image": eff_best[s][1],
                "luma_mean_median": float(np.nanmedian(eff_luma[s])) if ne else float("nan"),
                # photometric consistency proxies across the effective views of the same point
                "luma_min": float(np.nanmin(eff_luma[s])) if ne else float("nan"),
                "luma_max": float(np.nanmax(eff_luma[s])) if ne else float("nan"),
                "luma_cv_across_views": (
                    float(np.nanstd(eff_luma[s]) / max(np.nanmean(eff_luma[s]), 1e-6)) if ne >= 2 else float("nan")
                ),
                "sharpness_lapvar_min": float(np.nanmin(eff_sharp[s])) if ne else float("nan"),
            }
        )
    return rows


# ----------------------------------------------------------------------------
# manifest-backed loaders
# ----------------------------------------------------------------------------


def read_ply_xyz_rgb(path: Path) -> np.ndarray:
    """Binary little-endian PLY with float x/y/z + uchar r/g/b (tile initialization layout)."""
    with open(path, "rb") as handle:
        header = b""
        while b"end_header" not in header:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: PLY header not terminated")
            header += line
        lines = header.decode("ascii", "replace").splitlines()
        if "format binary_little_endian 1.0" not in lines:
            raise ValueError(f"{path}: expected binary_little_endian PLY")
        count = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
        props = [l.split()[1:] for l in lines if l.startswith("property")]
        expected = [["float", "x"], ["float", "y"], ["float", "z"], ["uchar", "red"], ["uchar", "green"], ["uchar", "blue"]]
        if props != expected:
            raise ValueError(f"{path}: unexpected PLY properties {props}")
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        return np.fromfile(handle, dtype=dtype, count=count)


def sample_points_in_box(records: np.ndarray, box: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    xyz = np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float64)
    keep = np.all((xyz >= box[0]) & (xyz <= box[1]), axis=1)
    inside = np.flatnonzero(keep)
    if inside.size == 0:
        raise ValueError("no PLY points inside the region box")
    pick = inside if inside.size <= count else rng.choice(inside, count, replace=False)
    return xyz[np.sort(pick)], int(inside.size)


def load_regions(path: Path) -> list[Region]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["regions"] if isinstance(payload, dict) else payload
    regions = []
    for entry in entries:
        regions.append(
            Region(
                label=str(entry["label"]),
                tile=str(entry["tile"]),
                world_box=np.asarray(entry["world_box"], dtype=np.float64).reshape(2, 3),
                status=str(entry.get("status", "provisional")),
                derivation=str(entry.get("derivation", "")),
                extra={k: v for k, v in entry.items() if k not in {"label", "tile", "world_box", "status", "derivation"}},
            )
        )
    return regions


class ManifestLoaders:
    """File-backed loaders with a tiny per-image cache (the audit walks views sequentially)."""

    def __init__(
        self,
        face_manifest: dict[str, Any],
        face_root: Path,
        geometry_manifest: dict[str, Any] | None,
        geometry_root: Path | None,
        dataset_manifest: dict[str, Any],
        recording_root: Path | None,
    ):
        from PIL import Image

        self._Image = Image
        self.face_root = face_root
        self.geometry_root = geometry_root
        self.recording_root = recording_root
        self.face_records: dict[tuple[str, str], dict[str, Any]] = {}
        for image in face_manifest["images"]:
            for face in image["faces"]:
                self.face_records[(image["image_id"], face["face_id"])] = face
        self.geometry_records: dict[tuple[str, str], dict[str, Any]] = {}
        if geometry_manifest is not None:
            for record in geometry_manifest["records"]:
                self.geometry_records[(record["image_id"], record["face_id"])] = record
        self.photo_paths: dict[str, Path] = {}
        if recording_root is not None:
            for image in dataset_manifest["images"]:
                if image.get("path_root", "recording") == "recording":
                    self.photo_paths[image["image_id"]] = recording_root / image["path"]
        self.loaded_photos = 0
        self.loaded_faces = 0

    def face_geometry(self, image_id: str, face_id: str) -> np.ndarray | None:
        record = self.geometry_records.get((image_id, face_id))
        if record is None or self.geometry_root is None:
            return None
        sparse = load_sparse_depth(self.geometry_root / record["path"])
        depth, _conf, valid = sparse.to_dense()
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        self.loaded_faces += 1
        return depth

    def face_mask(self, image_id: str, face_id: str) -> np.ndarray | None:
        record = self.face_records.get((image_id, face_id))
        if record is None:
            return None
        with self._Image.open(self.face_root / record["mask_path"]) as img:
            return np.asarray(img.convert("L"), dtype=np.uint8) > 0

    def photo(self, image_id: str) -> np.ndarray | None:
        path = self.photo_paths.get(image_id)
        if path is None or not path.exists():
            return None
        with self._Image.open(path) as img:
            # Same decode path as the face cache builder (PIL, convert("RGB"), no
            # ICC / EXIF handling); luminance = ITU-R 601 as PIL "L".
            gray = np.asarray(img.convert("L"), dtype=np.float32)
        self.loaded_photos += 1
        return gray


def build_views(face_manifest: dict[str, Any], dataset_manifest: dict[str, Any], *, warn: Callable[[str], None]) -> list[ViewRecord]:
    timestamps = {img["image_id"]: int(img["timestamp_ns"]) for img in dataset_manifest["images"]}
    dataset_c2w = {img["image_id"]: np.asarray(img["c2w"], dtype=np.float64) for img in dataset_manifest["images"]}
    views = []
    mismatched = 0
    for image in face_manifest["images"]:
        c2w = np.asarray(image["c2w"], dtype=np.float64)
        ref = dataset_c2w.get(image["image_id"])
        if ref is not None and not np.allclose(ref, c2w, atol=1e-4):
            mismatched += 1
        views.append(
            ViewRecord(
                image_id=str(image["image_id"]),
                camera_id=str(image["camera_id"]),
                rig_frame_id=str(image.get("rig_frame_id", "")),
                timestamp_ns=timestamps.get(image["image_id"], 0),
                c2w=c2w,
            )
        )
    if mismatched:
        warn(f"{mismatched} face-manifest poses differ from the dataset manifest by > 1e-4")
    return views


SUMMARY_FIELDS = [
    "n_images_fisheye", "n_images_face", "n_views_loose_ok", "n_effective_views", "n_effective_views_le5m",
    "strict_frac_occluded", "strict_support_frac", "n_cameras", "n_rig_frames",
    "time_span_s", "angle_span_deg", "range_min_m", "range_median_m",
    "footprint_px_per_m_fisheye", "footprint_px_per_m_face",
    "frac_occluded", "depth_support_frac", "rgb_valid_frac",
    "sharpness_lapvar_median", "sharpness_lapvar_best", "sharpness_lapvar_min", "luma_mean_median",
    "luma_min", "luma_max", "luma_cv_across_views",
]


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_count": len(rows)}
    for key in SUMMARY_FIELDS:
        values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[key] = {"median": None, "p10": None, "p90": None, "mean": None, "count": 0}
            continue
        out[key] = {
            "median": float(np.median(finite)),
            "p10": float(np.percentile(finite, 10)),
            "p90": float(np.percentile(finite, 90)),
            "mean": float(finite.mean()),
            "count": int(finite.size),
        }
    eff = np.asarray([r["n_effective_views"] for r in rows])
    out["samples_with_zero_effective_views"] = int((eff == 0).sum())
    out["samples_with_lt3_effective_views"] = int((eff < 3).sum())
    out["samples_with_two_cameras"] = int(sum(1 for r in rows if r["n_cameras"] >= 2))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--lidar-geometry-manifest", type=Path, default=None)
    parser.add_argument("--tile-inputs-manifest", required=True, type=Path)
    parser.add_argument("--recording-root", type=Path, default=None, help="folder holding camera/<side>/<ts>.jpg")
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-views", type=int, default=None, help="debug: only use the first N training views")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    face_manifest = json.loads(args.face_manifest.read_text(encoding="utf-8"))
    tile_manifest = json.loads(args.tile_inputs_manifest.read_text(encoding="utf-8"))
    geometry_manifest = None
    if args.lidar_geometry_manifest is not None:
        geometry_manifest = json.loads(args.lidar_geometry_manifest.read_text(encoding="utf-8"))
        if geometry_manifest.get("source_face_manifest_sha256") != face_manifest.get("face_manifest_sha256"):
            log("WARNING: LiDAR geometry manifest was built from a different face manifest")

    max_theta = 95.0
    if geometry_manifest is not None:
        max_theta = float(geometry_manifest.get("projection_config", {}).get("max_theta_deg", 95.0))
    cameras = {c["camera_id"]: CameraModel.from_manifest(c, max_theta_deg=max_theta) for c in dataset_manifest["cameras"]}
    faces_by_camera = {
        cam: [FaceSpec.from_dict(f) for f in payload["faces"]] for cam, payload in face_manifest["cameras"].items()
    }
    views = build_views(face_manifest, dataset_manifest, warn=lambda m: log("WARNING: " + m))
    if args.max_views:
        views = views[: args.max_views]
    projection = DepthProjectionConfig()
    if geometry_manifest is not None:
        pc = geometry_manifest.get("projection_config", {})
        projection = DepthProjectionConfig(
            min_range_m=float(pc.get("min_range_m", projection.min_range_m)),
            max_range_m=float(pc.get("max_range_m", projection.max_range_m)),
            max_theta_deg=max_theta,
            visibility_cell_px=int(pc.get("visibility_cell_px", 6)),
            visibility_tolerance=float(pc.get("visibility_tolerance", 0.2)),
            visibility_margin_m=float(pc.get("visibility_margin_m", 0.1)),
        )

    loaders = ManifestLoaders(
        face_manifest,
        args.face_manifest.parent,
        geometry_manifest,
        args.lidar_geometry_manifest.parent if args.lidar_geometry_manifest else None,
        dataset_manifest,
        args.recording_root,
    )
    tiles = {t["name"]: t for t in tile_manifest["tiles"]}
    regions = load_regions(args.regions)
    rng = np.random.default_rng(args.seed)

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "inputs": {
            "dataset_manifest_sha256": dataset_manifest.get("manifest_sha256"),
            "face_manifest_sha256": face_manifest.get("face_manifest_sha256"),
            "lidar_geometry_manifest_sha256": geometry_manifest.get("face_lidar_geometry_manifest_sha256") if geometry_manifest else None,
            "tile_inputs_manifest_sha256": tile_manifest.get("tile_inputs_manifest_sha256"),
            "training_views": len(views),
            "projection_config": projection.to_dict(),
            "sharpness_window_px": SHARPNESS_WINDOW_PX,
            "support_search_radius_px": SUPPORT_SEARCH_RADIUS_PX,
            "seed": args.seed,
            "samples_per_region": args.samples,
        },
        "regions": {},
    }
    for region in regions:
        tile = tiles[region.tile]
        ply_path = args.tile_inputs_manifest.parent / tile["initialization"]["path"]
        log(f"[{region.label}] reading {ply_path}")
        records = read_ply_xyz_rgb(ply_path)
        samples, inside_count = sample_points_in_box(records, region.world_box, args.samples, rng)
        log(f"[{region.label}] {inside_count} PLY points in box, sampling {len(samples)}")
        t0 = time.time()
        rows = audit_samples(
            samples, views, cameras, faces_by_camera,
            face_geometry=loaders.face_geometry, face_mask=loaders.face_mask, photo=loaders.photo,
            projection=projection, log=log,
        )
        for row in rows:
            row["region"] = region.label
            row["tile"] = region.tile
            row["region_status"] = region.status
        all_rows.extend(rows)
        region_summary = summarize(rows)
        region_summary.update(
            {
                "tile": region.tile,
                "status": region.status,
                "world_box": region.world_box.tolist(),
                "derivation": region.derivation,
                "ply_path": str(ply_path),
                "ply_sha256": tile["initialization"].get("sha256"),
                "ply_points_in_box": inside_count,
                "elapsed_s": round(time.time() - t0, 1),
                **region.extra,
            }
        )
        summary["regions"][region.label] = region_summary
        log(f"[{region.label}] done in {time.time() - t0:.0f}s; photos loaded so far {loaders.loaded_photos}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["region", "tile", "region_status"] + [k for k in all_rows[0].keys() if k not in {"region", "tile", "region_status"}]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {args.out_csv} ({len(all_rows)} rows) and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
