#!/usr/bin/env python3
"""Build a WP03 diagnostic view set for one region and emit the signed Tile
inputs it trains on.

Given a region box (``03_roi_provisional.json`` entry), its Tile and the WP03
coverage CSV (which carries the region's LiDAR sample points), the tool

1. scores every parent image of the Tile by *strict effective visibility* of
   the region, restricted to the faces that are actually Tile views of that
   image (``per_image_coverage``): the fraction of region samples that land in
   one of the image's Tile faces, are RGB-valid there and have a strict LiDAR
   return (0.1 m + 3 % of range, no nearer return) — the same rule
   ``tools/audit_observation_coverage.py`` uses for ``n_effective_views``;
   plus the effective range, the face pixel footprint and the original photo's
   Laplacian-variance sharpness / luminance at the projections;
2. selects presets from that table (``select_views``):
   ``U0``   1 view   — the sharpest strict-supported near view (<= 5 m),
   ``U1``   5 views  — well-matched views alternating across both physical
                       cameras, best support first, one per rig frame,
   ``DIAG`` 40 views — best support / nearest, half per camera;
3. writes, per preset, a *derived* Tile inputs manifest and Tile geometry
   manifest that reference the Tile's initialization PLY / geometry / backdrops
   verbatim (no copies) but restrict ``views`` to the selected parent images,
   both signed with the repository's own canonical signing so the trainer's
   ``verify_tile_inputs_manifest`` / ``verify_tile_geometry_manifest`` and the
   geometry<->inputs binding check pass unchanged;
4. writes ``selection.json`` with the coverage numbers of every selected image
   and the region's pixel bounding box inside each selected Tile crop (the
   coordinates a compare strip panel uses), for ROI scoring.

Native resolution is kept: crops are the Tile's own ``x/y/width/height``.

Example::

    python tools/build_diagnostic_set.py \
        --region-file research/quality_recovery_v2/03_roi_provisional.json \
        --region indoor_door_leaf_Tile_1 \
        --coverage-csv research/quality_recovery_v2/03_observation_coverage.csv \
        --tile-inputs-manifest C:/Peter/3dgs-runs/house0305_sop/tile_inputs_v9/tile_inputs_manifest.json \
        --tile-geometry-manifest C:/Peter/3dgs-runs/house0305_sop/tile_geometry_v9/tile_geometry_manifest.json \
        --dataset-manifest C:/Peter/3dgs-datasets/house0305_sop_v8/dataset_manifest.json \
        --face-manifest C:/Peter/3dgs-datasets/house0305_sop_v9/face4_train/face_manifest.json \
        --lidar-geometry-manifest C:/Peter/3dgs-datasets/house0305_sop_v9/face4_lidar_train_vis6/face_lidar_geometry_manifest.json \
        --recording-root "C:/baidunetdiskdownload/house/2026-03-05_10-58-54 - house" \
        --out-root C:/Peter/3dgs-runs/house0305_sop/diag_v2 --count 1 --count 5 --count 40

A face-set variant of an existing preset (F3: the same 40 parent images
without their ``pitch_up_56`` faces) reuses the parents verbatim and only
changes the derived views, so the arm differs from the original by the face
set alone::

    python tools/build_diagnostic_set.py ... --count 40 --preset-name DIAG_40_F3 \
        --reuse-selection <out-root>/<region>/DIAG_40/selection.json --exclude-faces pitch_up_56

Without ``--exclude-faces`` the output is byte-identical to before the option
existed; with it, ``selection.json`` records ``excluded_faces`` and the
per-face counts, and the run refuses if a parent image would keep no face.

The pure parts (``select_views``, ``reuse_selection``,
``derive_tile_inputs_manifest``, ``derive_tile_geometry_manifest``,
``roi_bbox_in_crop``) are unit tested in ``tests/test_diagnostic_set.py``
without any dataset.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes  # noqa: E402
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402
from cloudstudio_3dgs.geometry.lidar_projection import DepthProjectionConfig  # noqa: E402
from cloudstudio_3dgs.training.mipmap_tile_geometry import (  # noqa: E402
    sign_tile_geometry_manifest,
    verify_tile_geometry_manifest,
)
from cloudstudio_3dgs.training.tile_inputs import (  # noqa: E402
    TILE_INPUT_KIND,
    TILE_INPUT_SCHEMA_VERSION,
    verify_tile_inputs_manifest,
)
from audit_observation_coverage import (  # noqa: E402
    NEAR_VIEW_RANGE_M,
    SHARPNESS_WINDOW_PX,
    STATUS_OCCLUDED,
    STATUS_RANK,
    STATUS_SUPPORTED,
    STRICT_MARGIN_M,
    STRICT_TOLERANCE,
    SUPPORT_SEARCH_RADIUS_PX,
    CameraModel,
    ManifestLoaders,
    Region,
    ViewRecord,
    build_views,
    classify_against_nearest,
    face_footprint_px_per_m,
    load_regions,
    nearest_return_in_window,
    pixel_centre_to_index,
    sharpness_maps,
    world_to_camera,
)

SAMPLE_ID_SEPARATOR = "::"
PRESETS: dict[int, str] = {1: "U0", 5: "U1", 40: "DIAG"}
# An image counts as a candidate when at least this fraction of the region's
# samples is strictly supported in one of its Tile faces.  Relaxed (and
# recorded) when a preset cannot be filled otherwise.
DEFAULT_MIN_SUPPORT_FRACTION = 0.5
# Relaxation never admits an image that strictly sees less than this share of
# the region: a "coverage" set padded with views that barely see the surface
# would test something else.
MIN_SUPPORT_FLOOR = 0.2
# Ranking treats support fractions within one bin as equal: a view 1.4 m from
# a 1.6 m wide box cannot cover all of it, and a 0.49 vs 0.52 difference says
# nothing that the range does not say better.
SUPPORT_BIN = 0.1
# ROI pixel box = this percentile band of the projected sample pixels, so a few
# samples on the region's rim do not blow the box up.
ROI_PERCENTILE = (2.0, 98.0)
GENERATOR = "tools/build_diagnostic_set.py"


def preset_name(count: int) -> str:
    return PRESETS.get(int(count), f"N{int(count)}")


def preset_dir_name(count: int) -> str:
    return f"{preset_name(count)}_{int(count)}"


# ----------------------------------------------------------------------------
# samples and Tile bookkeeping
# ----------------------------------------------------------------------------


def region_samples_from_coverage(
    csv_path: Path, label: str, *, samples: int | None, seed: int
) -> np.ndarray:
    """The region's audited LiDAR sample points (x, y, z) from the WP03 CSV."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["region"] == label]
    if not rows:
        raise ValueError(f"{csv_path}: no rows for region {label!r}")
    xyz = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float64)
    if samples is not None and samples < len(xyz):
        rng = np.random.default_rng(seed)
        xyz = xyz[np.sort(rng.choice(len(xyz), samples, replace=False))]
    return xyz


def split_sample_id(sample_id: str) -> tuple[str, str]:
    image_id, separator, face_id = str(sample_id).partition(SAMPLE_ID_SEPARATOR)
    if not separator or not image_id or not face_id:
        raise ValueError(f"Tile view sample_id is not image::face: {sample_id!r}")
    return image_id, face_id


def tile_views_by_image(tile: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """image_id -> the Tile's face views (crops) of that parent image, in manifest order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for view in tile["views"]:
        image_id, _face = split_sample_id(view["sample_id"])
        out.setdefault(image_id, []).append(view)
    return out


# ----------------------------------------------------------------------------
# per-image coverage (mirrors audit_observation_coverage.audit_samples, but
# aggregated per parent image and restricted to that image's Tile faces)
# ----------------------------------------------------------------------------


def per_image_coverage(
    samples_world: np.ndarray,
    views: Sequence[ViewRecord],
    cameras: Mapping[str, CameraModel],
    faces_by_camera: Mapping[str, Sequence[FaceSpec]],
    tile_faces: Mapping[str, Sequence[str]],
    *,
    face_geometry: Callable[[str, str], np.ndarray | None],
    face_mask: Callable[[str, str], np.ndarray | None],
    photo: Callable[[str], np.ndarray | None],
    projection: DepthProjectionConfig | None = None,
    sharpness_window: int = SHARPNESS_WINDOW_PX,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """One row per parent image that has Tile faces: how much of the region it
    strictly sees, how near, how sharp.  ``tile_faces`` maps image_id -> face ids
    that are Tile views (other faces of the photo are not trained and ignored)."""
    cfg = projection or DepthProjectionConfig()
    samples_world = np.asarray(samples_world, dtype=np.float64)
    n = len(samples_world)
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for vi, view in enumerate(views):
        face_ids = tile_faces.get(view.image_id)
        if not face_ids:
            continue
        camera = cameras[view.camera_id]
        pc = world_to_camera(samples_world, view.c2w)
        uv, ranges, valid = camera.project(pc, min_range_m=cfg.min_range_m, max_range_m=cfg.max_range_m)
        idx = np.flatnonzero(valid)
        row: dict[str, Any] = {
            "image_id": view.image_id,
            "camera_id": view.camera_id,
            "rig_frame_id": view.rig_frame_id,
            "timestamp_ns": int(view.timestamp_ns),
            "camera_x": float(view.c2w[0, 3]),
            "camera_y": float(view.c2w[1, 3]),
            "camera_z": float(view.c2w[2, 3]),
            "tile_faces": list(face_ids),
            "n_samples": n,
            "n_in_fisheye": int(idx.size),
            "n_in_tile_face": 0,
            "n_rgb_valid": 0,
            "n_strict_supported": 0,
            "n_strict_occluded": 0,
            "n_effective": 0,
            "support_fraction": 0.0,
            "occluded_fraction": float("nan"),
            "range_min_m": float("nan"),
            "range_median_m": float("nan"),
            "footprint_px_per_m_face": float("nan"),
            "sharpness_lapvar_median": float("nan"),
            "sharpness_lapvar_p90": float("nan"),
            "luma_median": float("nan"),
            "effective_by_face": {},
        }
        if idx.size:
            pc_v = pc[idx]
            in_face = np.zeros(idx.size, dtype=bool)
            rgb_valid = np.zeros(idx.size, dtype=bool)
            strict_ok = np.zeros(idx.size, dtype=bool)
            status = np.full(idx.size, 0, dtype=np.int8)
            best_foot = np.full(idx.size, np.nan)
            per_face: dict[str, int] = {}
            for face in faces_by_camera[view.camera_id]:
                if face.face_id not in face_ids:
                    continue
                pix, inside = face.directions_to_pixels(pc_v)
                if not inside.any():
                    continue
                j = np.flatnonzero(inside)
                in_face[j] = True
                px = np.clip(pixel_centre_to_index(pix[j, 0]), 0, face.width - 1)
                py = np.clip(pixel_centre_to_index(pix[j, 1]), 0, face.height - 1)
                dense = face_geometry(view.image_id, face.face_id)
                if dense is not None:
                    nearest = nearest_return_in_window(dense, SUPPORT_SEARCH_RADIUS_PX)[py, px]
                    st = classify_against_nearest(
                        nearest, ranges[idx[j]], tolerance=STRICT_TOLERANCE, margin_m=STRICT_MARGIN_M
                    )
                else:
                    st = np.zeros(j.size, dtype=np.int8)
                mask = face_mask(view.image_id, face.face_id)
                ok = mask[py, px] if mask is not None else np.ones(j.size, dtype=bool)
                rgb_valid[j] |= ok
                prev = status[j]
                status[j] = np.where(STATUS_RANK[st] > STATUS_RANK[prev], st, prev)
                face_ok = ok & (st == STATUS_SUPPORTED)
                strict_ok[j] |= face_ok
                per_face[face.face_id] = int(face_ok.sum())
                ff = face_footprint_px_per_m(face, pc_v[j])
                best_foot[j] = np.where(np.isnan(best_foot[j]), ff, np.maximum(best_foot[j], ff))
            eff = np.flatnonzero(strict_ok & (status != STATUS_OCCLUDED) & in_face)
            row["n_in_tile_face"] = int(in_face.sum())
            row["n_rgb_valid"] = int((rgb_valid & in_face).sum())
            row["n_strict_supported"] = int((status == STATUS_SUPPORTED).sum())
            row["n_strict_occluded"] = int((status == STATUS_OCCLUDED).sum())
            row["n_effective"] = int(eff.size)
            row["support_fraction"] = float(eff.size / n) if n else 0.0
            judged = row["n_strict_supported"] + row["n_strict_occluded"]
            row["occluded_fraction"] = float(row["n_strict_occluded"] / judged) if judged else float("nan")
            row["effective_by_face"] = per_face
            if eff.size:
                eff_ranges = ranges[idx[eff]]
                row["range_min_m"] = float(eff_ranges.min())
                row["range_median_m"] = float(np.median(eff_ranges))
                row["footprint_px_per_m_face"] = float(np.nanmedian(best_foot[eff]))
                gray = photo(view.image_id)
                if gray is not None:
                    luma_stats, lap_stats = sharpness_maps(gray)
                    cx = pixel_centre_to_index(uv[idx[eff], 0])
                    cy = pixel_centre_to_index(uv[idx[eff], 1])
                    luma, _ = luma_stats.mean_var(cx, cy, sharpness_window)
                    _, lapvar = lap_stats.mean_var(cx, cy, sharpness_window)
                    row["sharpness_lapvar_median"] = float(np.median(lapvar))
                    row["sharpness_lapvar_p90"] = float(np.percentile(lapvar, 90))
                    row["luma_median"] = float(np.median(luma))
        rows.append(row)
        if log and (vi % 50 == 0 or vi == len(views) - 1):
            log(f"  view {vi + 1}/{len(views)} ({time.time() - t0:.0f}s)")
    return rows


# ----------------------------------------------------------------------------
# selection (pure)
# ----------------------------------------------------------------------------


def _finite(value: Any, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _support_bin(row: Mapping[str, Any]) -> float:
    """Support fraction quantised to SUPPORT_BIN so hundredths do not outrank range."""
    return math.floor(_finite(row.get("support_fraction"), 0.0) / SUPPORT_BIN + 1e-9) * SUPPORT_BIN


def _support_order(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Best support bin first; nearer first within a bin; sharper as a last tie-break."""
    return sorted(
        rows,
        key=lambda r: (
            -_support_bin(r),
            _finite(r.get("range_median_m"), float("inf")),
            -_finite(r.get("sharpness_lapvar_median"), -1.0),
            str(r.get("image_id")),
        ),
    )


def _candidates(
    rows: Sequence[Mapping[str, Any]], count: int, min_support_fraction: float, floor: float
) -> tuple[list[Mapping[str, Any]], float, bool]:
    """Rows with enough strict support; if fewer than ``count`` exist, the
    threshold is relaxed to the ``count``-th best support fraction, but never
    below ``floor`` (recorded)."""
    ordered = _support_order([r for r in rows if _finite(r.get("n_effective"), 0.0) > 0])
    kept = [r for r in ordered if _finite(r.get("support_fraction"), 0.0) >= min_support_fraction]
    if len(kept) >= count or len(ordered) <= len(kept):
        return kept, float(min_support_fraction), False
    relaxed_to = max(float(floor), _finite(ordered[min(count, len(ordered)) - 1].get("support_fraction"), 0.0))
    kept = [r for r in ordered if _finite(r.get("support_fraction"), 0.0) >= relaxed_to]
    return kept, float(relaxed_to), True


def _round_robin_cameras(ordered: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    """Alternate physical cameras (best first) so both are represented whenever
    both have candidates.  The two cameras of one rig frame are deliberately
    both eligible: they are different physical cameras (rig_frame_id is
    shared by left and right), which is the cross-camera match U1 wants."""
    queues: dict[str, list[Mapping[str, Any]]] = {}
    for row in ordered:
        queues.setdefault(str(row.get("camera_id")), []).append(row)
    camera_order = sorted(queues, key=lambda c: ordered.index(queues[c][0]))
    picked: list[Mapping[str, Any]] = []
    while len(picked) < count and any(queues.values()):
        for camera in camera_order:
            if queues[camera]:
                picked.append(queues[camera].pop(0))
            if len(picked) >= count:
                break
    return picked


def select_views(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    preset: str | None = None,
    min_support_fraction: float = DEFAULT_MIN_SUPPORT_FRACTION,
    support_floor: float = MIN_SUPPORT_FLOOR,
    near_range_m: float = NEAR_VIEW_RANGE_M,
) -> dict[str, Any]:
    """Pick ``count`` parent images from a per-image coverage table.

    Returns ``{"preset", "count", "selected": [rows...], "policy": {...}}``;
    ``selected`` rows carry ``selection_rank`` and ``selection_reason``.
    """
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    preset = preset or preset_name(count)
    candidates, threshold, relaxed = _candidates(rows, count, min_support_fraction, support_floor)
    policy: dict[str, Any] = {
        "preset": preset,
        "min_support_fraction_requested": float(min_support_fraction),
        "min_support_fraction_applied": threshold,
        "min_support_fraction_floor": float(support_floor),
        "threshold_relaxed": relaxed,
        "candidate_count": len(candidates),
        "table_rows": len(rows),
    }
    if not candidates:
        raise ValueError("no parent image strictly sees the region in any of its Tile faces")

    if preset == "U0":
        # Near first: a close view of a 1.6 m box necessarily covers less of
        # it, so the support threshold is relaxed (down to the floor) inside
        # the near pool before a far view is ever considered.
        def is_near(r: Mapping[str, Any]) -> bool:
            return _finite(r.get("range_median_m"), float("inf")) <= near_range_m

        near = [r for r in candidates if is_near(r)]
        stage = "near_and_min_support"
        if not near:
            near = [
                r for r in _support_order(rows)
                if is_near(r) and _finite(r.get("support_fraction"), 0.0) >= support_floor
            ]
            stage = "near_and_support_floor"
        pool = near if near else candidates
        if not near:
            stage = "far_fallback"
        policy["near_range_m"] = float(near_range_m)
        policy["near_candidate_count"] = len(near)
        policy["near_pool_stage"] = stage
        policy["near_filter_relaxed"] = stage != "near_and_min_support"
        policy["rule"] = (
            "max sharpness_lapvar_median among strict-supported views with median range <= near_range_m "
            "(support relaxed to the floor within the near pool before any far view); ties -> support_fraction"
        )
        best = sorted(
            pool,
            key=lambda r: (
                -_finite(r.get("sharpness_lapvar_median"), -1.0),
                -_finite(r.get("support_fraction"), 0.0),
                _finite(r.get("range_median_m"), float("inf")),
                str(r.get("image_id")),
            ),
        )
        picked = best[:count]
        reason = "sharpest near strict-supported view"
    elif preset == "U1":
        policy["rule"] = "support_fraction desc, range asc; alternate physical cameras (a stereo pair of one rig frame is allowed)"
        picked = _round_robin_cameras(candidates, count)
        reason = "well-matched view, cameras alternated"
    else:
        policy["rule"] = "support_fraction desc, range asc; half the budget per physical camera when available, remainder by rank"
        picked = _round_robin_cameras(candidates, count)
        reason = "coverage view, camera-balanced by rank"

    selected = []
    for rank, row in enumerate(picked):
        entry = dict(row)
        entry["selection_rank"] = rank
        entry["selection_reason"] = reason
        selected.append(entry)
    policy["selected_count"] = len(selected)
    policy["short_by"] = max(0, count - len(selected))
    cameras = sorted({str(r.get("camera_id")) for r in selected})
    policy["cameras_represented"] = cameras
    return {"preset": preset, "count": count, "selected": selected, "policy": policy}


# ----------------------------------------------------------------------------
# ROI box in the Tile crop (pure)
# ----------------------------------------------------------------------------


def roi_bbox_in_crop(
    samples_world: np.ndarray,
    c2w: np.ndarray,
    face: FaceSpec,
    crop: Mapping[str, Any],
    *,
    percentiles: tuple[float, float] = ROI_PERCENTILE,
) -> dict[str, Any] | None:
    """Pixel box of the region inside one Tile crop of one face, or None when
    the region does not project into the crop.

    Coordinates are array indices in the crop the trainer (and therefore a
    compare-strip panel) sees: face pixel index minus ``crop.x / crop.y``.
    """
    pc = world_to_camera(np.asarray(samples_world, dtype=np.float64), np.asarray(c2w, dtype=np.float64))
    pix, inside = face.directions_to_pixels(pc)
    if not inside.any():
        return None
    px = pixel_centre_to_index(pix[inside, 0]).astype(np.float64)
    py = pixel_centre_to_index(pix[inside, 1]).astype(np.float64)
    x0 = int(crop["x"]); y0 = int(crop["y"]); w = int(crop["width"]); h = int(crop["height"])
    in_crop = (px >= x0) & (px < x0 + w) & (py >= y0) & (py < y0 + h)
    if not in_crop.any():
        return None
    cx = px[in_crop] - x0
    cy = py[in_crop] - y0
    lo, hi = percentiles
    bx0, bx1 = np.percentile(cx, lo), np.percentile(cx, hi)
    by0, by1 = np.percentile(cy, lo), np.percentile(cy, hi)
    return {
        "face_id": face.face_id,
        "crop": {"x": x0, "y": y0, "width": w, "height": h},
        "x0": int(math.floor(bx0)),
        "y0": int(math.floor(by0)),
        "x1": int(math.ceil(bx1)) + 1,
        "y1": int(math.ceil(by1)) + 1,
        "samples_in_face": int(inside.sum()),
        "samples_in_crop": int(in_crop.sum()),
        "fraction_of_samples_in_crop": float(in_crop.sum() / len(pc)),
        "percentiles": [float(lo), float(hi)],
    }


# ----------------------------------------------------------------------------
# derived manifests (pure; signing reuses the repository's canonical form)
# ----------------------------------------------------------------------------


def sign_tile_inputs_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Same signature rule as ``tile_inputs.materialize_lidar_tile_inputs``."""
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("tile_inputs_manifest_sha256", None)
    signed = copy.deepcopy(unsigned)
    signed["tile_inputs_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return signed


def face_counts(views: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """face_id -> number of Tile views with that face, in first-seen order."""
    counts: dict[str, int] = {}
    for view in views:
        _image, face_id = split_sample_id(view["sample_id"])
        counts[face_id] = counts.get(face_id, 0) + 1
    return counts


def parse_face_list(spec: str | None) -> list[str]:
    """``--exclude-faces`` value: comma-separated face ids; empty -> none."""
    if not spec:
        return []
    faces: list[str] = []
    for item in str(spec).split(","):
        face = item.strip()
        if face and face not in faces:
            faces.append(face)
    return faces


def reuse_selection(payload: Mapping[str, Any], *, region_label: str, count: int) -> dict[str, Any]:
    """The parent-image selection of an existing ``selection.json`` (same ids,
    same order, same coverage rows and policy) in the shape ``select_views``
    returns, so a derived preset differs from the original only by what is
    done to the faces afterwards."""
    if payload.get("kind") != "wp03_diagnostic_view_selection_v1":
        raise ValueError("reused selection is not a wp03_diagnostic_view_selection_v1 file")
    found_label = (payload.get("region") or {}).get("label")
    if str(found_label) != str(region_label):
        raise ValueError(f"reused selection is for region {found_label!r}, not {region_label!r}")
    if int(payload.get("count", -1)) != int(count):
        raise ValueError(f"reused selection has count {payload.get('count')}, not {count}")
    selected = [dict(row) for row in payload.get("selected") or []]
    if not selected:
        raise ValueError("reused selection has no selected images")
    ids = [str(r["image_id"]) for r in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("reused selection contains duplicate image ids")
    policy = dict(payload.get("policy") or {})
    policy["reused_selection"] = True
    return {"preset": str(payload["preset"]), "count": int(count), "selected": selected, "policy": policy}


def derive_tile_inputs_manifest(
    base: Mapping[str, Any],
    *,
    tile_name: str,
    image_ids: Sequence[str],
    provenance: Mapping[str, Any],
    face_ids_by_image: Mapping[str, Sequence[str]] | None = None,
    exclude_face_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """A one-Tile inputs manifest whose ``views`` are only the Tile faces of
    ``image_ids`` (optionally only ``face_ids_by_image[image]``, minus every
    face in ``exclude_face_ids``), everything else (initialization artefact
    path/sha, boxes, plan binding) verbatim.  Without exclusions the output is
    byte-identical to what it was before the option existed: the exclusion
    bookkeeping only enters the signed ``diagnostic`` block when used."""
    if base.get("kind") != TILE_INPUT_KIND or base.get("schema_version") != TILE_INPUT_SCHEMA_VERSION:
        raise ValueError("base Tile inputs manifest has an unsupported schema")
    base_sha = verify_tile_inputs_manifest(dict(base))
    matches = [t for t in base["tiles"] if t.get("name") == tile_name]
    if len(matches) != 1:
        raise ValueError(f"base Tile inputs do not contain a unique Tile named {tile_name!r}")
    tile = copy.deepcopy(matches[0])
    wanted = [str(i) for i in image_ids]
    wanted_set = set(wanted)
    if len(wanted_set) != len(wanted):
        raise ValueError("image_ids contain duplicates")
    excluded = [str(f) for f in (exclude_face_ids or [])]
    excluded_set = set(excluded)
    views = []
    removed: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_before_exclusion: set[str] = set()
    for view in tile["views"]:
        image_id, face_id = split_sample_id(view["sample_id"])
        if image_id not in wanted_set:
            continue
        if face_ids_by_image is not None and face_id not in set(face_ids_by_image.get(image_id, ())):
            continue
        seen_before_exclusion.add(image_id)
        if face_id in excluded_set:
            removed.append(view)
            continue
        views.append(copy.deepcopy(view))
        seen.add(image_id)
    missing = sorted(wanted_set - seen_before_exclusion)
    if missing:
        raise ValueError(f"selected images have no Tile views in {tile_name}: {missing}")
    emptied = sorted(seen_before_exclusion - seen)
    if emptied:
        # Losing a parent image changes the image set, not just the face set;
        # refuse so the derived arm keeps exactly the same parents.
        raise ValueError(
            f"excluding faces {excluded} would leave {len(emptied)} selected image(s) with zero Tile views: {emptied}"
        )
    if not views:
        raise ValueError("derived Tile inputs would have no views")
    tile["views"] = views
    tile["view_count"] = len(views)
    recommended = dict(tile.get("recommended_training") or {})
    if "steps" in recommended:
        # Keep the field's meaning (20 view epochs) for the restricted set; the
        # arm config's schedule contract supplies the real horizon.
        recommended["steps"] = 20 * len(views)
        tile["recommended_training"] = recommended
    payload = {k: copy.deepcopy(v) for k, v in base.items() if k not in {"tiles", "tile_count", "tile_inputs_manifest_sha256"}}
    payload["tile_count"] = 1
    payload["tiles"] = [tile]
    payload["diagnostic"] = {
        "generator": GENERATOR,
        "derived_from_tile_inputs_manifest_sha256": base_sha,
        "tile": tile_name,
        "parent_image_ids": wanted,
        "view_count": len(views),
        "faces_restricted": face_ids_by_image is not None,
        **dict(provenance),
    }
    if excluded:
        payload["diagnostic"]["excluded_face_ids"] = excluded
        payload["diagnostic"]["view_count_before_exclusion"] = len(views) + len(removed)
        payload["diagnostic"]["views_removed_by_exclusion"] = len(removed)
        payload["diagnostic"]["face_counts_before_exclusion"] = face_counts(views + removed)
        payload["diagnostic"]["face_counts"] = face_counts(views)
    return sign_tile_inputs_manifest(payload)


def derive_tile_geometry_manifest(
    base: Mapping[str, Any],
    *,
    tile_id: int,
    geometry_path: str,
    tile_inputs_manifest_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """A one-Tile geometry manifest bound to the derived Tile inputs; the
    geometry artefact keeps its sha256/bytes and is referenced at
    ``geometry_path`` (relative to the derived manifest's directory)."""
    base_sha = verify_tile_geometry_manifest(dict(base))
    matches = [t for t in base["tiles"] if int(t["tile_id"]) == int(tile_id)]
    if len(matches) != 1:
        raise ValueError(f"base Tile geometry does not contain a unique Tile {tile_id}")
    tile = copy.deepcopy(matches[0])
    tile["geometry"]["path"] = str(geometry_path)
    payload = {k: copy.deepcopy(v) for k, v in base.items() if k not in {"tiles", "tile_count", "tile_geometry_manifest_sha256", "tile_inputs_manifest_sha256"}}
    payload["tile_inputs_manifest_sha256"] = str(tile_inputs_manifest_sha256)
    payload["tile_count"] = 1
    payload["tiles"] = [tile]
    payload["diagnostic"] = {
        "generator": GENERATOR,
        "derived_from_tile_geometry_manifest_sha256": base_sha,
        "derived_from_tile_inputs_manifest_sha256": str(base.get("tile_inputs_manifest_sha256")),
        **dict(provenance),
    }
    return sign_tile_geometry_manifest(payload)


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

TABLE_FIELDS = [
    "image_id", "camera_id", "rig_frame_id", "timestamp_ns", "camera_x", "camera_y", "camera_z",
    "n_samples", "n_in_fisheye", "n_in_tile_face", "n_rgb_valid", "n_strict_supported", "n_strict_occluded",
    "n_effective", "support_fraction", "occluded_fraction", "range_min_m", "range_median_m",
    "footprint_px_per_m_face", "sharpness_lapvar_median", "sharpness_lapvar_p90", "luma_median",
    "tile_faces", "effective_by_face",
]


def write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["tile_faces"] = "|".join(row.get("tile_faces") or [])
            out["effective_by_face"] = json.dumps(row.get("effective_by_face") or {}, sort_keys=True)
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in out.items()})


def read_table(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        parsed: dict[str, Any] = dict(row)
        for key in TABLE_FIELDS:
            if key in {"image_id", "camera_id", "rig_frame_id", "tile_faces", "effective_by_face"}:
                continue
            parsed[key] = _finite(row.get(key), float("nan"))
        for key in ("timestamp_ns", "n_samples", "n_in_fisheye", "n_in_tile_face", "n_rgb_valid",
                    "n_strict_supported", "n_strict_occluded", "n_effective"):
            parsed[key] = int(parsed[key]) if math.isfinite(parsed[key]) else 0
        parsed["tile_faces"] = [f for f in str(row.get("tile_faces", "")).split("|") if f]
        parsed["effective_by_face"] = json.loads(row.get("effective_by_face") or "{}")
        out.append(parsed)
    return out


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(target: Path, start: Path, *, max_climb: int = 3) -> str:
    """Relative POSIX path from ``start`` to ``target``; absolute when it would
    climb more than ``max_climb`` levels (a long ``..`` chain is not resolved
    before Windows' MAX_PATH check and ``is_file()`` then reports False)."""
    relative = Path(os.path.relpath(target, start))
    if sum(1 for part in relative.parts if part == "..") > max_climb:
        return target.resolve().as_posix()
    return relative.as_posix()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region-file", required=True, type=Path)
    parser.add_argument("--region", required=True, help="region label in --region-file")
    parser.add_argument("--coverage-csv", required=True, type=Path)
    parser.add_argument("--tile-inputs-manifest", required=True, type=Path)
    parser.add_argument("--tile-geometry-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--lidar-geometry-manifest", required=True, type=Path)
    parser.add_argument("--recording-root", type=Path, default=None)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--count", type=int, action="append", default=None, help="preset size; repeatable (default 1, 5, 40)")
    parser.add_argument("--samples", type=int, default=400, help="region samples drawn from the coverage CSV")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--min-support-fraction", type=float, default=DEFAULT_MIN_SUPPORT_FRACTION)
    parser.add_argument("--support-floor", type=float, default=MIN_SUPPORT_FLOOR, help="relaxation never goes below this support fraction")
    parser.add_argument("--per-image-table", type=Path, default=None, help="reuse a previously written per_image_coverage.csv")
    parser.add_argument("--restrict-faces", action="store_true", help="keep only the Tile faces in which the region is strictly supported")
    parser.add_argument(
        "--exclude-faces", default=None, metavar="FACE[,FACE]",
        help="drop these face ids from every selected image (default none: output byte-identical); refuses if an image would keep no face",
    )
    parser.add_argument(
        "--preset-name", default=None,
        help="output directory name under <out-root>/<region> instead of <PRESET>_<count> (single --count only)",
    )
    parser.add_argument(
        "--reuse-selection", type=Path, default=None,
        help="take the parent images (ids and order) from this selection.json instead of re-selecting; the per-image table is then not needed",
    )
    parser.add_argument("--max-views", type=int, default=None, help="debug: only walk the first N training views")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    counts = args.count or [1, 5, 40]
    excluded_faces = parse_face_list(args.exclude_faces)
    if (args.preset_name is not None or args.reuse_selection is not None) and len(counts) != 1:
        raise SystemExit("--preset-name / --reuse-selection apply to exactly one --count")
    reused_payload = None
    if args.reuse_selection is not None:
        reused_payload = json.loads(args.reuse_selection.read_text(encoding="utf-8"))
        # Validated for real (region/count/ids) once the region is loaded.
    regions = {r.label: r for r in load_regions(args.region_file)}
    if args.region not in regions:
        raise SystemExit(f"region {args.region!r} not in {args.region_file}: {sorted(regions)}")
    region: Region = regions[args.region]
    tile_inputs = json.loads(args.tile_inputs_manifest.read_text(encoding="utf-8"))
    tile_inputs_sha = verify_tile_inputs_manifest(tile_inputs)
    tiles = {t["name"]: t for t in tile_inputs["tiles"]}
    if region.tile not in tiles:
        raise SystemExit(f"Tile {region.tile!r} not in {args.tile_inputs_manifest}")
    tile = tiles[region.tile]
    tile_geometry = json.loads(args.tile_geometry_manifest.read_text(encoding="utf-8"))
    tile_geometry_sha = verify_tile_geometry_manifest(tile_geometry)
    if tile_geometry.get("tile_inputs_manifest_sha256") != tile_inputs_sha:
        raise SystemExit("Tile geometry manifest is bound to different Tile inputs than the one given")
    geometry_tile = next(t for t in tile_geometry["tiles"] if int(t["tile_id"]) == int(tile["tile_id"]))
    geometry_npz = (args.tile_geometry_manifest.parent / geometry_tile["geometry"]["path"]).resolve()
    if not geometry_npz.is_file():
        raise SystemExit(f"Tile geometry artefact missing: {geometry_npz}")

    samples = region_samples_from_coverage(args.coverage_csv, region.label, samples=args.samples, seed=args.seed)
    inside = np.all((samples >= region.world_box[0]) & (samples <= region.world_box[1]), axis=1)
    if not inside.all():
        log(f"WARNING: {int((~inside).sum())} coverage samples fall outside the region box; dropping them")
        samples = samples[inside]
    log(f"[{region.label}] {len(samples)} region samples, Tile {region.tile} ({tile['view_count']} views)")

    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    face_manifest = json.loads(args.face_manifest.read_text(encoding="utf-8"))
    geometry_manifest = json.loads(args.lidar_geometry_manifest.read_text(encoding="utf-8"))
    pc = geometry_manifest.get("projection_config", {})
    max_theta = float(pc.get("max_theta_deg", 95.0))
    projection = DepthProjectionConfig(
        min_range_m=float(pc.get("min_range_m", 0.2)),
        max_range_m=float(pc.get("max_range_m", 80.0)),
        max_theta_deg=max_theta,
        visibility_cell_px=int(pc.get("visibility_cell_px", 6)),
        visibility_tolerance=float(pc.get("visibility_tolerance", 0.2)),
        visibility_margin_m=float(pc.get("visibility_margin_m", 0.1)),
    )
    cameras = {c["camera_id"]: CameraModel.from_manifest(c, max_theta_deg=max_theta) for c in dataset_manifest["cameras"]}
    faces_by_camera = {cam: [FaceSpec.from_dict(f) for f in payload["faces"]] for cam, payload in face_manifest["cameras"].items()}
    views = build_views(face_manifest, dataset_manifest, warn=lambda m: log("WARNING: " + m))
    if args.max_views:
        views = views[: args.max_views]
    views_by_image = {v.image_id: v for v in views}
    by_image = tile_views_by_image(tile)
    tile_faces = {image: [split_sample_id(v["sample_id"])[1] for v in vs] for image, vs in by_image.items()}

    region_dir = args.out_root / region.label
    table_path = region_dir / "per_image_coverage.csv"
    if args.per_image_table is not None:
        table = read_table(args.per_image_table)
        log(f"[{region.label}] reusing per-image table {args.per_image_table} ({len(table)} rows)")
    elif reused_payload is not None:
        # The reused selection carries its coverage rows; the (7 min) table
        # walk is only needed to select, not to derive.
        table = []
        table_path = Path(str((reused_payload.get("inputs") or {}).get("per_image_table") or table_path))
        log(f"[{region.label}] reusing selection {args.reuse_selection}; per-image table not recomputed")
    else:
        loaders = ManifestLoaders(
            face_manifest, args.face_manifest.parent, geometry_manifest, args.lidar_geometry_manifest.parent,
            dataset_manifest, args.recording_root,
        )
        t0 = time.time()
        table = per_image_coverage(
            samples, views, cameras, faces_by_camera, tile_faces,
            face_geometry=loaders.face_geometry, face_mask=loaders.face_mask, photo=loaders.photo,
            projection=projection, log=log,
        )
        write_table(table_path, table)
        log(f"[{region.label}] per-image table: {len(table)} parent images with Tile views in {time.time() - t0:.0f}s -> {table_path}")

    inputs_record = {
        "region_file": str(args.region_file),
        "coverage_csv": str(args.coverage_csv),
        "tile_inputs_manifest": str(args.tile_inputs_manifest),
        "tile_inputs_manifest_sha256": tile_inputs_sha,
        "tile_geometry_manifest": str(args.tile_geometry_manifest),
        "tile_geometry_manifest_sha256": tile_geometry_sha,
        "dataset_manifest_sha256": dataset_manifest.get("manifest_sha256"),
        "face_manifest_sha256": face_manifest.get("face_manifest_sha256"),
        "lidar_geometry_manifest_sha256": geometry_manifest.get("face_lidar_geometry_manifest_sha256"),
        "samples_used": int(len(samples)),
        "seed": int(args.seed),
        "projection_config": projection.to_dict(),
        "strict_band": {"tolerance": STRICT_TOLERANCE, "margin_m": STRICT_MARGIN_M, "search_radius_px": SUPPORT_SEARCH_RADIUS_PX},
        "per_image_table": str(table_path),
    }

    for count in counts:
        if reused_payload is not None:
            selection = reuse_selection(reused_payload, region_label=region.label, count=count)
            if (reused_payload.get("inputs") or {}).get("tile_inputs_manifest_sha256") != tile_inputs_sha:
                raise SystemExit("reused selection was built from different Tile inputs than the manifest given")
            missing_views = [i for i in (str(r["image_id"]) for r in selection["selected"]) if i not in views_by_image]
            if missing_views:
                raise SystemExit(f"reused selection images are not training views: {missing_views}")
        else:
            selection = select_views(table, count, min_support_fraction=args.min_support_fraction, support_floor=args.support_floor)
        preset_dir = region_dir / (args.preset_name or preset_dir_name(count))
        preset_dir.mkdir(parents=True, exist_ok=True)
        image_ids = [str(r["image_id"]) for r in selection["selected"]]
        faces_restriction = None
        if args.restrict_faces:
            faces_restriction = {
                str(r["image_id"]): [f for f, n in (r.get("effective_by_face") or {}).items() if int(n) > 0]
                for r in selection["selected"]
            }
        provenance = {
            "region": region.label,
            "world_box": region.world_box.tolist(),
            "preset": selection["preset"],
            "count": int(count),
            "selection_file": "selection.json",
        }
        if args.preset_name:
            provenance["preset_dir"] = str(args.preset_name)
        diag_inputs = derive_tile_inputs_manifest(
            tile_inputs, tile_name=region.tile, image_ids=image_ids, provenance=provenance,
            face_ids_by_image=faces_restriction, exclude_face_ids=excluded_faces or None,
        )
        inputs_path = preset_dir / "tile_inputs_manifest.json"
        _json_dump(inputs_path, diag_inputs)
        # The trainer resolves the geometry artefact relative to the derived
        # manifest's directory; point back at the original npz (no copy).
        geometry_rel = _relative_posix(geometry_npz, preset_dir)
        diag_geometry = derive_tile_geometry_manifest(
            tile_geometry, tile_id=int(tile["tile_id"]), geometry_path=geometry_rel,
            tile_inputs_manifest_sha256=diag_inputs["tile_inputs_manifest_sha256"], provenance=provenance,
        )
        geometry_path = preset_dir / "tile_geometry_manifest.json"
        _json_dump(geometry_path, diag_geometry)
        # Verify exactly as the trainer will (artefact hashes included).
        verify_tile_inputs_manifest(json.loads(inputs_path.read_text(encoding="utf-8")), root=args.tile_inputs_manifest.parent, verify_artifacts=True)
        verify_tile_geometry_manifest(json.loads(geometry_path.read_text(encoding="utf-8")), root=preset_dir, verify_artifacts=True)

        selected_views = diag_inputs["tiles"][0]["views"]
        roi = []
        for image_id in image_ids:
            view = views_by_image[image_id]
            faces = {f.face_id: f for f in faces_by_camera[view.camera_id]}
            for tile_view in selected_views:
                vid, face_id = split_sample_id(tile_view["sample_id"])
                if vid != image_id:
                    continue
                box = roi_bbox_in_crop(samples, view.c2w, faces[face_id], tile_view)
                roi.append({"sample_id": tile_view["sample_id"], "image_id": image_id, "camera_id": view.camera_id, "roi": box})
        selection_payload = {
            "schema_version": 1,
            "kind": "wp03_diagnostic_view_selection_v1",
            "generator": GENERATOR,
            "region": {"label": region.label, "tile": region.tile, "tile_id": int(tile["tile_id"]), "status": region.status, "world_box": region.world_box.tolist()},
            "preset": selection["preset"],
            "count": int(count),
            "policy": selection["policy"],
            "inputs": inputs_record,
            "tile_inputs_manifest": str(inputs_path),
            "tile_inputs_manifest_sha256": diag_inputs["tile_inputs_manifest_sha256"],
            "tile_inputs_root": str(args.tile_inputs_manifest.parent),
            "tile_geometry_manifest": str(geometry_path),
            "tile_geometry_manifest_sha256": diag_geometry["tile_geometry_manifest_sha256"],
            "initialization_ply": str((args.tile_inputs_manifest.parent / tile["initialization"]["path"]).resolve()),
            "initialization_geometry": str(geometry_npz),
            "view_count": len(selected_views),
            "view_sample_ids": [v["sample_id"] for v in selected_views],
            "selected": selection["selected"],
            "roi_in_crops": roi,
            "roi_note": (
                "roi.x0/y0/x1/y1 are array indices inside the Tile crop (face pixel index minus crop.x/crop.y), "
                "i.e. the coordinates of a compare-strip panel for that sample_id; see make_diagnostic_arm_config.py --eval"
            ),
        }
        if args.preset_name:
            selection_payload["preset_dir"] = str(args.preset_name)
        if reused_payload is not None:
            selection_payload["reused_selection"] = {
                "path": str(args.reuse_selection),
                "sha256": _sha256_file(args.reuse_selection),
                "tile_inputs_manifest_sha256": reused_payload.get("tile_inputs_manifest_sha256"),
                "view_count": reused_payload.get("view_count"),
                "parent_image_ids_identical": [str(r["image_id"]) for r in reused_payload.get("selected", [])] == image_ids,
            }
        if excluded_faces:
            diagnostic = diag_inputs["diagnostic"]
            selection_payload["excluded_faces"] = list(excluded_faces)
            selection_payload["face_exclusion"] = {
                "view_count_before": diagnostic["view_count_before_exclusion"],
                "view_count_after": len(selected_views),
                "views_removed": diagnostic["views_removed_by_exclusion"],
                "face_counts_before": diagnostic["face_counts_before_exclusion"],
                "face_counts_after": diagnostic["face_counts"],
                "parent_images_kept": len(image_ids),
            }
        _json_dump(preset_dir / "selection.json", selection_payload)
        log(f"[{region.label}] {preset_dir.name}: {len(image_ids)} images / {len(selected_views)} Tile views -> {preset_dir}")
        if excluded_faces:
            log(f"    excluded faces {excluded_faces}: {selection_payload['face_exclusion']['view_count_before']} -> {len(selected_views)} views, "
                f"faces {selection_payload['face_exclusion']['face_counts_before']} -> {selection_payload['face_exclusion']['face_counts_after']}")
        for row in selection["selected"]:
            log(
                f"    {row['image_id']} {row['camera_id']:5s} support {row['support_fraction']:.3f} "
                f"range {row['range_median_m']:.2f} m (min {row['range_min_m']:.2f}) lapvar {row['sharpness_lapvar_median']:.0f} luma {row['luma_median']:.0f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
