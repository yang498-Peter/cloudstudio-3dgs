#!/usr/bin/env python3
"""WP06 photometric audit: what the learned per-image exposure gains actually do.

Measurement only (no trainer behaviour is touched). Three questions:

1. **What did each tile learn?** ``auxiliary_params["exposure_log_gains"]`` in
   a tile checkpoint is a bare ``[N]`` tensor; the index is the
   ``ExposureCompensator`` contract ``sorted(set(base image ids))`` over the
   tile's training views (``FaceCacheDataset.exposure_image_ids`` feeds the
   unique base ids, the compensator sorts them, faces ``base::face`` share the
   base gain). The tool rebuilds that index from the tile-inputs manifest,
   joins it with the WP01 view membership (camera, rig frame, timestamp,
   indoor/outdoor) and reports gain mean / P5 / P95 per camera, per tile and
   per environment, the saturation of each source photo, the disagreement
   between the gains two tiles learned for the SAME image (halo overlap), and
   the brightness shift implied between a canonical render (gain 1.0, what the
   evaluators and the trainer's own validation use) and the train-compensated
   frame, before and after the per-tile median bake of the merge.

2. **Is a low-dimensional shared correction enough?** One-way variance
   decompositions of the log gains by physical camera, rig frame, capture-time
   block and environment, plus the left/right correlation on the same rig
   frame, say how much of the per-image freedom is actually shared structure.

3. **Do the gains reduce multi-view colour inconsistency?** For the WP03
   regions, LiDAR samples are projected into every training parent photo that
   strictly sees them (same fisheye / Face4 / vis6 rules as
   ``audit_observation_coverage.py``), the photo colour is read in a small
   window, and the per-point luminance dispersion across views is reported
   raw, after dividing by the learned per-image gain (photo / g, the model
   frame the trainer optimises in: ``g * render ~= photo``), after
   low-dimensional alternatives (per-camera mean, per-time-block mean) and
   after an oracle per-image scalar fitted from the multi-view observations
   themselves (the floor any per-image scalar can reach).

Gains are applied in the encoded (sRGB 8-bit) domain, exactly as the trainer
multiplies the rendered RGB before the losses; nothing is linearised.

Example::

    python tools/audit_exposure_gains.py \
        --checkpoint Tile_0=C:/Peter/3dgs-runs/house0305_sop/tile0_R1_range0_20k/checkpoints/latest.pt \
        --checkpoint Tile_1=C:/Peter/3dgs-runs/house0305_sop/tile1_R1d_20k/checkpoints/latest.pt \
        --checkpoint Tile_2=C:/Peter/3dgs-runs/house0305_sop/tile2_R1d_20k/checkpoints/latest.pt \
        --checkpoint Tile_3=C:/Peter/3dgs-runs/house0305_sop/tile3_R1d_cap13m_20k/checkpoints/latest.pt \
        --merge-report R1d=C:/Peter/3dgs-runs/house0305_sop/delivery_R1d/merge_report.json \
        --tile-inputs-manifest C:/Peter/3dgs-runs/house0305_sop/tile_inputs_v9/tile_inputs_manifest.json \
        --membership-csv research/quality_recovery_v2/01_view_membership.csv \
        --dataset-manifest C:/Peter/3dgs-datasets/house0305_sop_v8/dataset_manifest.json \
        --face-manifest C:/Peter/3dgs-datasets/house0305_sop_v9/face4_train/face_manifest.json \
        --lidar-geometry-manifest C:/Peter/3dgs-datasets/house0305_sop_v9/face4_lidar_train_vis6/face_lidar_geometry_manifest.json \
        --recording-root C:/Peter/testdata/S1/house0305 \
        --regions research/quality_recovery_v2/03_roi_provisional.json \
        --out-csv research/quality_recovery_v2/06_photometric_consistency.csv \
        --out-json research/quality_recovery_v2/06_exposure_summary.json \
        --out-dispersion-csv research/quality_recovery_v2/06_colour_dispersion.csv

The pure functions (``gain_index_from_views``, ``extract_gains``,
``join_membership``, ``summarize_gains``, ``cross_tile_disagreement``,
``variance_decomposition``, ``saturation_fraction``, ``colour_dispersion``)
take plain Python / numpy inputs so ``tests/test_audit_exposure_gains.py``
exercises them on synthetic data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from audit_observation_coverage import (  # noqa: E402
    STATUS_OCCLUDED,
    STATUS_SUPPORTED,
    STATUS_RANK,
    STATUS_NO_LIDAR,
    STRICT_MARGIN_M,
    STRICT_TOLERANCE,
    SUPPORT_SEARCH_RADIUS_PX,
    CameraModel,
    ManifestLoaders,
    ViewRecord,
    build_views,
    classify_against_nearest,
    load_regions,
    nearest_return_in_window,
    pixel_centre_to_index,
    read_ply_xyz_rgb,
    sample_points_in_box,
    world_to_camera,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402
from cloudstudio_3dgs.geometry.lidar_projection import DepthProjectionConfig  # noqa: E402

LN2 = math.log(2.0)  # ExposureCompensationConfig.max_abs_log_gain default
SAMPLE_ID_SEPARATOR = "::"
SATURATION_THRESHOLD = 250
SATURATION_DECODE_SCALE = 8  # JPEG DCT-domain 1/8 decode, 2912 -> 364 px
COLOUR_WINDOW_RADIUS_PX = 2  # 5x5 median on the source photo
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])  # ITU-R 601, same as PIL "L"
MIN_VIEWS_FOR_DISPERSION = 3
DEFAULT_TIME_BLOCKS_S = (10.0, 20.0, 60.0)


# ----------------------------------------------------------------------------
# 1. gain extraction and join
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GainRecord:
    image_id: str
    index: int
    log_gain_raw: float  # stored parameter (may exceed the clamp)
    log_gain: float  # clamped, what gain() applies
    saturated: bool

    @property
    def gain(self) -> float:
        return math.exp(self.log_gain)


def base_image_id(sample_id: str) -> str:
    """``base::face`` -> ``base`` (FaceCacheDataset.exposure_id_for)."""
    return str(sample_id).rsplit(SAMPLE_ID_SEPARATOR, 1)[0]


def gain_index_from_views(views: Iterable[Mapping[str, Any]]) -> list[str]:
    """Position -> base image id, exactly as ``ExposureCompensator.__init__``
    orders ``exposure_image_ids`` (``sorted(set(...))``)."""
    return sorted({base_image_id(view["sample_id"]) for view in views})


def extract_gains(
    log_gains: Sequence[float] | np.ndarray,
    ordered_ids: Sequence[str],
    *,
    max_abs_log_gain: float = LN2,
) -> dict[str, GainRecord]:
    values = np.asarray(log_gains, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(ordered_ids):
        raise ValueError(
            f"checkpoint carries {values.shape[0]} exposure gains but the tile views "
            f"name {len(ordered_ids)} base images; index cannot be trusted"
        )
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("ordered_ids must be unique")
    out: dict[str, GainRecord] = {}
    for position, image_id in enumerate(ordered_ids):
        raw = float(values[position])
        clamped = float(np.clip(raw, -max_abs_log_gain, max_abs_log_gain))
        out[image_id] = GainRecord(
            image_id=str(image_id),
            index=position,
            log_gain_raw=raw,
            log_gain=clamped,
            saturated=bool(abs(raw) >= max_abs_log_gain - 1e-6),
        )
    return out


def load_checkpoint_log_gains(path: Path) -> np.ndarray:
    """CPU-only read of ``auxiliary_params["exposure_log_gains"]`` (mmap, no CUDA)."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    aux = payload.get("auxiliary_params") or {}
    if "exposure_log_gains" not in aux:
        raise KeyError(f"{path}: checkpoint carries no exposure_log_gains")
    return aux["exposure_log_gains"].detach().float().cpu().numpy().astype(np.float64)


def torch_style_median(values: Sequence[float]) -> float:
    """``torch.median`` returns the lower of the two middle values for even N."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        return float("nan")
    return float(ordered[(ordered.size - 1) // 2])


def merge_tile_gain(records: Mapping[str, GainRecord]) -> float:
    """What ``merge_v28_tile_checkpoints.py --harmonize-exposure`` bakes: the
    torch median of ``exp(raw log gain)`` (note: unclamped)."""
    return torch_style_median([math.exp(r.log_gain_raw) for r in records.values()])


def join_membership(
    gains_by_tile: Mapping[str, Mapping[str, GainRecord]],
    membership: Sequence[Mapping[str, Any]],
    *,
    saturation: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """One row per (image, tile that trained it). Membership rows need
    ``image_id``, ``camera``, ``rig_frame_id``, ``timestamp_ns``,
    ``environment``; ``capture_fraction`` and ``tile_ids`` are carried if
    present."""
    by_image = {str(row["image_id"]): row for row in membership}
    tile_median = {tile: merge_tile_gain(records) for tile, records in gains_by_tile.items()}
    rows: list[dict[str, Any]] = []
    for tile in sorted(gains_by_tile):
        records = gains_by_tile[tile]
        log_tile_median = math.log(tile_median[tile])
        for image_id in sorted(records):
            record = records[image_id]
            member = by_image.get(image_id)
            if member is None:
                raise KeyError(f"{tile}: image {image_id} has a gain but no membership row")
            sat = (saturation or {}).get(image_id, {})
            rows.append(
                {
                    "image_id": image_id,
                    "camera": str(member["camera"]),
                    "rig_frame_id": str(member["rig_frame_id"]),
                    "timestamp_ns": int(member["timestamp_ns"]),
                    "capture_fraction": float(member.get("capture_fraction", float("nan"))),
                    "environment": str(member["environment"]),
                    "tile_ids_membership": str(member.get("tile_ids", "")),
                    "tile": tile,
                    "gain_index": record.index,
                    "log_gain_raw": record.log_gain_raw,
                    "log_gain": record.log_gain,
                    "gain": record.gain,
                    "saturated_clamp": int(record.saturated),
                    "tile_median_gain": tile_median[tile],
                    # g * render ~= photo  =>  render ~= photo / g
                    "log_canonical_over_photo": -record.log_gain,
                    # merge multiplies DC colour by the tile median gain
                    "log_baked_over_photo": log_tile_median - record.log_gain,
                    "photo_saturation_frac": float(sat.get("saturation_frac", float("nan"))),
                    "photo_mean_luma": float(sat.get("mean_luma", float("nan"))),
                }
            )
    return rows


def _stats(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _gain_group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gains = [r["gain"] for r in rows]
    logs = [r["log_gain"] for r in rows]
    out = {
        "gain": _stats(gains),
        "log_gain": _stats(logs),
        "saturated_clamp_frac": float(np.mean([r["saturated_clamp"] for r in rows])) if rows else float("nan"),
        "log_canonical_over_photo": _stats([r["log_canonical_over_photo"] for r in rows]),
        "log_baked_over_photo": _stats([r["log_baked_over_photo"] for r in rows]),
    }
    sats = [r["photo_saturation_frac"] for r in rows if np.isfinite(r.get("photo_saturation_frac", float("nan")))]
    if sats:
        out["photo_saturation_frac"] = _stats(sats)
        out["photo_mean_luma"] = _stats([r["photo_mean_luma"] for r in rows])
    return out


def summarize_gains(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def grouped(keys: Sequence[str]) -> dict[str, Any]:
        buckets: dict[str, list] = defaultdict(list)
        for row in rows:
            buckets["|".join(str(row[k]) for k in keys)].append(row)
        return {name: _gain_group_stats(members) for name, members in sorted(buckets.items())}

    return {
        "all": _gain_group_stats(rows),
        "by_tile": grouped(["tile"]),
        "by_camera": grouped(["camera"]),
        "by_environment": grouped(["environment"]),
        "by_tile_environment": grouped(["tile", "environment"]),
        "by_tile_camera": grouped(["tile", "camera"]),
        "by_camera_environment": grouped(["camera", "environment"]),
    }


def cross_tile_disagreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """For images trained by >= 2 tiles: |log g_a - log g_b| raw and after the
    per-tile median bake (what survives into the merged checkpoint)."""
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(row)
    pairs: dict[str, list[float]] = defaultdict(list)
    pairs_baked: dict[str, list[float]] = defaultdict(list)
    per_env: dict[str, list[float]] = defaultdict(list)
    per_env_baked: dict[str, list[float]] = defaultdict(list)
    per_image_spread: list[float] = []
    per_image_spread_baked: list[float] = []
    multi = 0
    for image_id, members in by_image.items():
        if len(members) < 2:
            continue
        multi += 1
        logs = [m["log_gain"] for m in members]
        baked = [m["log_gain"] - math.log(m["tile_median_gain"]) for m in members]
        per_image_spread.append(max(logs) - min(logs))
        per_image_spread_baked.append(max(baked) - min(baked))
        env = str(members[0]["environment"])
        per_env[env].append(max(logs) - min(logs))
        per_env_baked[env].append(max(baked) - min(baked))
        ordered = sorted(members, key=lambda m: str(m["tile"]))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                key = f"{ordered[i]['tile']}~{ordered[j]['tile']}"
                pairs[key].append(abs(ordered[i]["log_gain"] - ordered[j]["log_gain"]))
                pairs_baked[key].append(abs(baked[i] - baked[j]))
    return {
        "images_total": len(by_image),
        "images_in_multiple_tiles": multi,
        "abs_log_gain_spread_per_image": _stats(per_image_spread),
        "abs_log_gain_spread_per_image_after_tile_bake": _stats(per_image_spread_baked),
        "by_environment": {env: _stats(v) for env, v in sorted(per_env.items())},
        "by_environment_after_tile_bake": {env: _stats(v) for env, v in sorted(per_env_baked.items())},
        "by_tile_pair": {key: _stats(v) for key, v in sorted(pairs.items())},
        "by_tile_pair_after_tile_bake": {key: _stats(v) for key, v in sorted(pairs_baked.items())},
    }


def _one_way_r2(values: np.ndarray, labels: Sequence[Any]) -> tuple[float, int, float]:
    """(R^2, group count, adjusted R^2) of the group means.

    R^2 = 1 - SS_within / SS_total is inflated when there are many small
    groups (a singleton group has zero within-variance by construction, e.g.
    rig frames with ~1.6 images each); the adjusted value
    1 - (1 - R^2) (n - 1) / (n - k) charges for the k-1 fitted means.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    total = float(((values - values.mean()) ** 2).sum())
    if total <= 0.0:
        return float("nan"), 0, float("nan")
    groups: dict[Any, list[float]] = defaultdict(list)
    for value, label in zip(values, labels):
        groups[label].append(float(value))
    within = sum(float(((np.asarray(g) - np.mean(g)) ** 2).sum()) for g in groups.values())
    r2 = 1.0 - within / total
    k = len(groups)
    adjusted = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    return r2, k, adjusted


def variance_decomposition(
    rows: Sequence[Mapping[str, Any]],
    *,
    time_blocks_s: Sequence[float] = DEFAULT_TIME_BLOCKS_S,
) -> dict[str, Any]:
    """Low-dimensional structure in the log gains of ONE tile (rows must all
    carry the same ``tile``): R^2 of camera, rig frame, time block, environment
    and camera x time block groupings, plus the left/right correlation on the
    same rig frame."""
    if not rows:
        return {"n": 0}
    tiles = {r["tile"] for r in rows}
    if len(tiles) != 1:
        raise ValueError("variance_decomposition expects the rows of exactly one tile")
    logs = np.asarray([r["log_gain"] for r in rows], dtype=np.float64)
    t_ns = np.asarray([r["timestamp_ns"] for r in rows], dtype=np.int64)
    t_s = (t_ns - t_ns.min()) / 1e9
    out: dict[str, Any] = {
        "n": int(logs.size),
        "log_gain_var": float(logs.var()),
        "log_gain_std": float(logs.std()),
    }
    for name, labels in (
        ("camera", [r["camera"] for r in rows]),
        ("environment", [r["environment"] for r in rows]),
        ("rig_frame", [r["rig_frame_id"] for r in rows]),
        ("camera_environment", [f"{r['camera']}|{r['environment']}" for r in rows]),
    ):
        r2, groups, adj = _one_way_r2(logs, labels)
        out[f"r2_{name}"] = r2
        out[f"r2adj_{name}"] = adj
        out[f"groups_{name}"] = groups
    for block in time_blocks_s:
        labels = np.floor(t_s / float(block)).astype(int).tolist()
        r2, groups, adj = _one_way_r2(logs, labels)
        out[f"r2_time_block_{block:g}s"] = r2
        out[f"r2adj_time_block_{block:g}s"] = adj
        out[f"groups_time_block_{block:g}s"] = groups
        labels_cam = [f"{r['camera']}|{b}" for r, b in zip(rows, labels)]
        r2c, groupsc, adjc = _one_way_r2(logs, labels_cam)
        out[f"r2_camera_x_time_block_{block:g}s"] = r2c
        out[f"r2adj_camera_x_time_block_{block:g}s"] = adjc
        out[f"groups_camera_x_time_block_{block:g}s"] = groupsc
    # left/right of the same rig frame
    by_frame: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_frame[str(r["rig_frame_id"])][str(r["camera"])] = float(r["log_gain"])
    left, right = [], []
    for frame in by_frame.values():
        if "left" in frame and "right" in frame:
            left.append(frame["left"])
            right.append(frame["right"])
    if len(left) >= 2:
        l_arr, r_arr = np.asarray(left), np.asarray(right)
        out["left_right_same_frame_pairs"] = len(left)
        out["left_right_same_frame_corr"] = float(np.corrcoef(l_arr, r_arr)[0, 1])
        out["left_right_same_frame_abs_diff"] = _stats(np.abs(l_arr - r_arr))
        out["left_minus_right_mean"] = float((l_arr - r_arr).mean())
    # lag-1 autocorrelation along capture time within each camera
    for camera in sorted({r["camera"] for r in rows}):
        seq = sorted(((r["timestamp_ns"], r["log_gain"]) for r in rows if r["camera"] == camera))
        vals = np.asarray([v for _, v in seq], dtype=np.float64)
        if vals.size >= 4:
            centred = vals - vals.mean()
            denom = float((centred**2).sum())
            out[f"lag1_autocorr_{camera}"] = float((centred[:-1] * centred[1:]).sum() / denom) if denom > 0 else float("nan")
            out[f"mean_log_gain_{camera}"] = float(vals.mean())
    return out


# ----------------------------------------------------------------------------
# 2. saturation proxy
# ----------------------------------------------------------------------------


def saturation_fraction(gray: np.ndarray, *, threshold: int = SATURATION_THRESHOLD) -> float:
    arr = np.asarray(gray)
    if arr.size == 0:
        return float("nan")
    return float((arr >= threshold).mean())


def photo_saturation(path: Path, *, scale: int = SATURATION_DECODE_SCALE) -> dict[str, float]:
    """Luma saturation fraction and mean luma of a JPEG decoded at 1/scale
    (libjpeg DCT scaling via ``Image.draft``; sRGB 8-bit, no colour management,
    same decode family as the face cache)."""
    from PIL import Image

    with Image.open(path) as img:
        w, h = img.size
        img.draft("L", (max(1, w // scale), max(1, h // scale)))
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
    return {
        "saturation_frac": saturation_fraction(gray),
        "mean_luma": float(gray.mean()),
        "decoded_width": int(gray.shape[1]),
        "decoded_height": int(gray.shape[0]),
    }


# ----------------------------------------------------------------------------
# 3. multi-view colour dispersion
# ----------------------------------------------------------------------------

PhotoRgbLoader = Callable[[str], Optional[np.ndarray]]  # -> uint8 HxWx3


def window_median_rgb(photo: np.ndarray, px: np.ndarray, py: np.ndarray, *, radius: int = COLOUR_WINDOW_RADIUS_PX) -> np.ndarray:
    """Per-channel median in a (2r+1)^2 window around each (px, py); windows are
    clipped at the border."""
    h, w = photo.shape[:2]
    px = np.asarray(px, dtype=np.int64)
    py = np.asarray(py, dtype=np.int64)
    offsets = np.arange(-radius, radius + 1)
    xs = np.clip(px[:, None, None] + offsets[None, None, :], 0, w - 1)
    ys = np.clip(py[:, None, None] + offsets[None, :, None], 0, h - 1)
    patch = photo[ys, xs].reshape(len(px), -1, photo.shape[2]).astype(np.float64)
    return np.median(patch, axis=1)


def luma_of(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ LUMA_WEIGHTS


def collect_observations(
    samples_world: np.ndarray,
    views: Sequence[ViewRecord],
    cameras: Mapping[str, CameraModel],
    faces_by_camera: Mapping[str, Sequence[FaceSpec]],
    *,
    face_geometry: Callable[[str, str], Optional[np.ndarray]],
    face_mask: Callable[[str, str], Optional[np.ndarray]],
    photo_rgb: PhotoRgbLoader,
    projection: DepthProjectionConfig | None = None,
    window_radius: int = COLOUR_WINDOW_RADIUS_PX,
    log: Callable[[str], None] | None = None,
) -> list[list[tuple[str, np.ndarray, float]]]:
    """Per sample: list of (image_id, rgb median, range) over the training
    photos that strictly see the sample (RGB-valid face mask, positive strict
    LiDAR support, not occluded - the WP03 "effective view" rule)."""
    cfg = projection or DepthProjectionConfig()
    samples_world = np.asarray(samples_world, dtype=np.float64)
    n = len(samples_world)
    obs: list[list[tuple[str, np.ndarray, float]]] = [[] for _ in range(n)]
    t0 = time.time()
    for vi, view in enumerate(views):
        camera = cameras[view.camera_id]
        pc = world_to_camera(samples_world, view.c2w)
        uv, ranges, valid = camera.project(pc, min_range_m=cfg.min_range_m, max_range_m=cfg.max_range_m)
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        pc_v = pc[idx]
        any_valid_strict = np.zeros(idx.size, dtype=bool)
        view_strict = np.full(idx.size, STATUS_NO_LIDAR, dtype=np.int8)
        for face in faces_by_camera[view.camera_id]:
            pix, inside = face.directions_to_pixels(pc_v)
            if not inside.any():
                continue
            j = np.flatnonzero(inside)
            px = np.clip(pixel_centre_to_index(pix[j, 0]), 0, face.width - 1)
            py = np.clip(pixel_centre_to_index(pix[j, 1]), 0, face.height - 1)
            dense = face_geometry(view.image_id, face.face_id)
            if dense is None:
                continue
            nearest = nearest_return_in_window(dense, SUPPORT_SEARCH_RADIUS_PX)[py, px]
            st = classify_against_nearest(nearest, ranges[idx[j]], tolerance=STRICT_TOLERANCE, margin_m=STRICT_MARGIN_M)
            mask = face_mask(view.image_id, face.face_id)
            rgb_ok = mask[py, px] if mask is not None else np.ones(j.size, dtype=bool)
            prev = view_strict[j]
            view_strict[j] = np.where(STATUS_RANK[st] > STATUS_RANK[prev], st, prev)
            any_valid_strict[j] |= rgb_ok & (st == STATUS_SUPPORTED)
        eff = np.flatnonzero(any_valid_strict & (view_strict != STATUS_OCCLUDED))
        if eff.size == 0:
            continue
        photo = photo_rgb(view.image_id)
        if photo is None:
            continue
        cx = np.clip(pixel_centre_to_index(uv[idx[eff], 0]), 0, photo.shape[1] - 1)
        cy = np.clip(pixel_centre_to_index(uv[idx[eff], 1]), 0, photo.shape[0] - 1)
        rgb = window_median_rgb(photo, cx, cy, radius=window_radius)
        for e_i, k in enumerate(eff):
            obs[idx[k]].append((view.image_id, rgb[e_i], float(ranges[idx[k]])))
        if log and (vi % 50 == 0 or vi == len(views) - 1):
            log(f"  view {vi + 1}/{len(views)} ({time.time() - t0:.0f}s)")
    return obs


def oracle_image_gains(
    obs: Sequence[Sequence[tuple[str, np.ndarray, float]]],
    *,
    min_views: int = MIN_VIEWS_FOR_DISPERSION,
    iterations: int = 3,
) -> dict[str, float]:
    """Per-image scalar that best explains the multi-view luminance
    observations (alternating medians: point reference = median over views of
    L / g, image gain = median over points of L_obs / L_ref). Gains are
    returned normalised to a geometric mean of 1 so the floor is comparable
    with the learned gains only up to a global constant."""
    gains: dict[str, float] = {}
    usable = [[(i, luma_of(rgb)) for i, rgb, _ in point] for point in obs if len(point) >= min_views]
    if not usable:
        return gains
    for _ in range(iterations):
        refs = []
        for point in usable:
            vals = np.asarray([lum / gains.get(i, 1.0) for i, lum in point])
            refs.append(float(np.median(vals)))
        residual: dict[str, list[float]] = defaultdict(list)
        for point, ref in zip(usable, refs):
            if ref <= 1e-6:
                continue
            for image_id, lum in point:
                if lum > 1e-6:
                    residual[image_id].append(math.log(lum / ref))
        gains = {image_id: math.exp(float(np.median(v))) for image_id, v in residual.items()}
        mean_log = float(np.mean([math.log(g) for g in gains.values()]))
        gains = {k: v / math.exp(mean_log) for k, v in gains.items()}
    return gains


def colour_dispersion(
    obs: Sequence[Sequence[tuple[str, np.ndarray, float]]],
    gain_variants: Mapping[str, Mapping[str, float]],
    *,
    min_views: int = MIN_VIEWS_FOR_DISPERSION,
) -> list[dict[str, Any]]:
    """Per point: luminance dispersion across views, raw and under each gain
    variant (photo / gain). Views without a gain in a variant are dropped for
    that variant AND for the matching raw baseline (``*_paired``) so the
    comparison is on identical view sets."""
    rows: list[dict[str, Any]] = []
    for point_index, point in enumerate(obs):
        if len(point) < min_views:
            continue
        lum = np.asarray([luma_of(rgb) for _, rgb, _ in point], dtype=np.float64)
        rgb = np.asarray([rgb for _, rgb, _ in point], dtype=np.float64)
        ids = [i for i, _, _ in point]
        row: dict[str, Any] = {
            "point_index": point_index,
            "n_views": len(point),
            "luma_mean_raw": float(lum.mean()),
            "luma_cv_raw": float(lum.std() / max(lum.mean(), 1e-6)),
            "luma_logstd_raw": float(np.log(np.maximum(lum, 1.0)).std()),
            "luma_min_raw": float(lum.min()),
            "luma_max_raw": float(lum.max()),
            "rgb_std_mean_raw": float(rgb.std(axis=0).mean()),
            "range_median_m": float(np.median([r for _, _, r in point])),
        }
        for name, gains in gain_variants.items():
            keep = np.asarray([i in gains for i in ids], dtype=bool)
            row[f"n_views_{name}"] = int(keep.sum())
            if keep.sum() < min_views:
                row[f"luma_cv_{name}"] = float("nan")
                row[f"luma_cv_raw_paired_{name}"] = float("nan")
                row[f"luma_logstd_{name}"] = float("nan")
                row[f"luma_logstd_raw_paired_{name}"] = float("nan")
                row[f"rgb_std_mean_{name}"] = float("nan")
                continue
            g = np.asarray([gains[i] for i, k in zip(ids, keep) if k], dtype=np.float64)
            lum_k = lum[keep]
            rgb_k = rgb[keep]
            corrected = lum_k / g
            rgb_corrected = rgb_k / g[:, None]
            row[f"luma_cv_{name}"] = float(corrected.std() / max(corrected.mean(), 1e-6))
            row[f"luma_cv_raw_paired_{name}"] = float(lum_k.std() / max(lum_k.mean(), 1e-6))
            row[f"luma_logstd_{name}"] = float(np.log(np.maximum(corrected, 1.0)).std())
            row[f"luma_logstd_raw_paired_{name}"] = float(np.log(np.maximum(lum_k, 1.0)).std())
            row[f"rgb_std_mean_{name}"] = float(rgb_corrected.std(axis=0).mean())
        rows.append(row)
    return rows


def summarize_dispersion(rows: Sequence[Mapping[str, Any]], variant_names: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"points": len(rows)}
    if not rows:
        return out
    out["n_views"] = _stats([r["n_views"] for r in rows])
    out["luma_cv_raw"] = _stats([r["luma_cv_raw"] for r in rows])
    out["luma_logstd_raw"] = _stats([r["luma_logstd_raw"] for r in rows])
    out["rgb_std_mean_raw"] = _stats([r["rgb_std_mean_raw"] for r in rows])
    for name in variant_names:
        paired_raw = np.asarray([r.get(f"luma_cv_raw_paired_{name}", float("nan")) for r in rows])
        corrected = np.asarray([r.get(f"luma_cv_{name}", float("nan")) for r in rows])
        ok = np.isfinite(paired_raw) & np.isfinite(corrected)
        entry = {
            "points_with_gain": int(ok.sum()),
            "n_views_with_gain": _stats([r.get(f"n_views_{name}", 0) for r in rows]),
            "luma_cv": _stats(corrected[ok]),
            "luma_cv_raw_paired": _stats(paired_raw[ok]),
            "luma_logstd": _stats([r.get(f"luma_logstd_{name}", float("nan")) for r in rows]),
            "luma_logstd_raw_paired": _stats([r.get(f"luma_logstd_raw_paired_{name}", float("nan")) for r in rows]),
            "rgb_std_mean": _stats([r.get(f"rgb_std_mean_{name}", float("nan")) for r in rows]),
        }
        if ok.any():
            ratio = corrected[ok] / np.maximum(paired_raw[ok], 1e-9)
            entry["cv_ratio_corrected_over_raw"] = _stats(ratio)
            entry["frac_points_improved"] = float((corrected[ok] < paired_raw[ok]).mean())
            entry["median_cv_change_frac"] = float(np.median(corrected[ok]) / max(np.median(paired_raw[ok]), 1e-9) - 1.0)
        out[name] = entry
    return out


# ----------------------------------------------------------------------------
# manifest-backed driver
# ----------------------------------------------------------------------------


class RgbPhotoLoader:
    """Full-resolution RGB decode of the source JPEG (PIL convert("RGB"), the
    face cache's decode path), one-entry cache because views are walked
    sequentially."""

    def __init__(self, dataset_manifest: Mapping[str, Any], recording_root: Path):
        from PIL import Image

        self._Image = Image
        self.paths = {
            img["image_id"]: recording_root / img["path"]
            for img in dataset_manifest["images"]
            if img.get("path_root", "recording") == "recording"
        }
        self.loaded = 0

    def __call__(self, image_id: str) -> np.ndarray | None:
        path = self.paths.get(image_id)
        if path is None or not path.exists():
            return None
        with self._Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        self.loaded += 1
        return rgb


def _parse_named(values: Sequence[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def read_membership_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items() if k in fieldnames})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True, help="Tile_N=path/to/latest.pt (repeatable)")
    parser.add_argument("--merge-report", action="append", default=None, help="NAME=merge_report.json to cross-check the baked tile gains (repeatable)")
    parser.add_argument("--tile-inputs-manifest", required=True, type=Path)
    parser.add_argument("--membership-csv", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--face-manifest", type=Path, default=None)
    parser.add_argument("--lidar-geometry-manifest", type=Path, default=None)
    parser.add_argument("--recording-root", type=Path, default=None)
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--time-blocks-s", type=float, nargs="+", default=list(DEFAULT_TIME_BLOCKS_S))
    parser.add_argument("--skip-saturation", action="store_true")
    parser.add_argument("--skip-dispersion", action="store_true")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-dispersion-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    checkpoints = _parse_named(args.checkpoint)
    tile_manifest = json.loads(args.tile_inputs_manifest.read_text(encoding="utf-8"))
    tiles = {t["name"]: t for t in tile_manifest["tiles"]}
    membership = read_membership_csv(args.membership_csv)
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))

    summary: dict[str, Any] = {
        "inputs": {
            "tile_inputs_manifest_sha256": tile_manifest.get("tile_inputs_manifest_sha256"),
            "dataset_manifest_sha256": dataset_manifest.get("manifest_sha256"),
            "membership_csv": str(args.membership_csv),
            "checkpoints": {},
            "max_abs_log_gain": LN2,
            "gain_domain": "encoded sRGB 8-bit, multiplied on the render before the losses (trainer contract); photos divided by the gain here",
        },
    }

    # ---- 1. gains --------------------------------------------------------
    gains_by_tile: dict[str, dict[str, GainRecord]] = {}
    for tile_name, ckpt in sorted(checkpoints.items()):
        if tile_name not in tiles:
            raise KeyError(f"{tile_name} not in tile inputs manifest ({sorted(tiles)})")
        ordered = gain_index_from_views(tiles[tile_name]["views"])
        log(f"[{tile_name}] reading {ckpt} ({len(ordered)} base images in tile views)")
        values = load_checkpoint_log_gains(ckpt)
        gains_by_tile[tile_name] = extract_gains(values, ordered)
        summary["inputs"]["checkpoints"][tile_name] = {
            "path": str(ckpt),
            "gain_count": int(values.size),
            "tile_view_count": int(tiles[tile_name]["view_count"]),
            "tile_base_images": len(ordered),
            "raw_log_gain_min": float(values.min()),
            "raw_log_gain_max": float(values.max()),
            "beyond_clamp_count": int((np.abs(values) > LN2).sum()),
            "merge_style_tile_gain": merge_tile_gain(gains_by_tile[tile_name]),
            "clamped_median_gain": float(np.median([r.gain for r in gains_by_tile[tile_name].values()])),
        }

    # cross-check with merge reports
    if args.merge_report:
        summary["merge_report_crosscheck"] = {}
        for name, path in _parse_named(args.merge_report).items():
            report = json.loads(path.read_text(encoding="utf-8"))
            entry = {"exposure_harmonized": report.get("exposure_harmonized"), "tiles": {}}
            for record in report.get("records", []):
                tile_name = f"Tile_{record['tile_id']}"
                applied = record.get("exposure_gain_applied")
                mine = summary["inputs"]["checkpoints"].get(tile_name, {}).get("merge_style_tile_gain")
                entry["tiles"][tile_name] = {
                    "checkpoint": record.get("checkpoint"),
                    "exposure_gain_applied": applied,
                    "recomputed_from_checkpoint": mine,
                    "abs_diff": (abs(applied - mine) if applied is not None and mine is not None else None),
                    "same_checkpoint_as_audited": (
                        Path(str(record.get("checkpoint", ""))).resolve() == checkpoints[tile_name].resolve()
                        if tile_name in checkpoints else None
                    ),
                }
            summary["merge_report_crosscheck"][name] = entry

    # ---- 2. saturation proxy ---------------------------------------------
    saturation: dict[str, dict[str, float]] = {}
    if not args.skip_saturation and args.recording_root is not None:
        needed = sorted({i for records in gains_by_tile.values() for i in records})
        paths = {img["image_id"]: args.recording_root / img["path"] for img in dataset_manifest["images"]}
        t0 = time.time()
        for k, image_id in enumerate(needed):
            path = paths.get(image_id)
            if path is None or not path.exists():
                continue
            saturation[image_id] = photo_saturation(path)
            if k % 100 == 0:
                log(f"  saturation {k + 1}/{len(needed)} ({time.time() - t0:.0f}s)")
        summary["inputs"]["saturation"] = {
            "photos": len(saturation),
            "threshold_luma": SATURATION_THRESHOLD,
            "decode_scale": SATURATION_DECODE_SCALE,
        }

    rows = join_membership(gains_by_tile, membership, saturation=saturation)
    fieldnames = list(rows[0].keys())
    write_csv(args.out_csv, rows, fieldnames)
    summary["gains"] = summarize_gains(rows)
    summary["cross_tile"] = cross_tile_disagreement(rows)
    summary["variance_decomposition"] = {
        tile: variance_decomposition([r for r in rows if r["tile"] == tile], time_blocks_s=args.time_blocks_s)
        for tile in sorted(gains_by_tile)
    }
    if saturation:
        sat = np.asarray([r["photo_saturation_frac"] for r in rows])
        lg = np.asarray([r["log_gain"] for r in rows])
        ml = np.asarray([r["photo_mean_luma"] for r in rows])
        ok = np.isfinite(sat) & np.isfinite(lg)
        summary["gain_vs_photo"] = {
            "corr_log_gain_vs_saturation_frac": float(np.corrcoef(lg[ok], sat[ok])[0, 1]) if ok.sum() > 3 else None,
            "corr_log_gain_vs_mean_luma": float(np.corrcoef(lg[ok], ml[ok])[0, 1]) if ok.sum() > 3 else None,
        }
    log(f"wrote {args.out_csv} ({len(rows)} rows)")

    # ---- 3. colour dispersion --------------------------------------------
    if not args.skip_dispersion:
        if args.regions is None or args.face_manifest is None or args.recording_root is None:
            raise SystemExit("--regions, --face-manifest and --recording-root are required unless --skip-dispersion")
        face_manifest = json.loads(args.face_manifest.read_text(encoding="utf-8"))
        geometry_manifest = None
        if args.lidar_geometry_manifest is not None:
            geometry_manifest = json.loads(args.lidar_geometry_manifest.read_text(encoding="utf-8"))
        max_theta = float((geometry_manifest or {}).get("projection_config", {}).get("max_theta_deg", 95.0))
        cameras = {c["camera_id"]: CameraModel.from_manifest(c, max_theta_deg=max_theta) for c in dataset_manifest["cameras"]}
        faces_by_camera = {cam: [FaceSpec.from_dict(f) for f in payload["faces"]] for cam, payload in face_manifest["cameras"].items()}
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
            face_manifest, args.face_manifest.parent, geometry_manifest,
            args.lidar_geometry_manifest.parent if args.lidar_geometry_manifest else None,
            dataset_manifest, args.recording_root,
        )
        photo_rgb = RgbPhotoLoader(dataset_manifest, args.recording_root)
        regions = load_regions(args.regions)
        rng = np.random.default_rng(args.seed)
        all_samples: list[np.ndarray] = []
        region_of: list[str] = []
        region_meta: dict[str, Any] = {}
        for region in regions:
            tile = tiles[region.tile]
            ply_path = args.tile_inputs_manifest.parent / tile["initialization"]["path"]
            records = read_ply_xyz_rgb(ply_path)
            samples, inside = sample_points_in_box(records, region.world_box, args.samples, rng)
            all_samples.append(samples)
            region_of.extend([region.label] * len(samples))
            region_meta[region.label] = {"tile": region.tile, "ply_points_in_box": inside, "samples": len(samples), "world_box": region.world_box.tolist()}
            log(f"[{region.label}] {inside} PLY points in box, sampling {len(samples)}")
        samples_world = np.concatenate(all_samples, axis=0)
        t0 = time.time()
        obs = collect_observations(
            samples_world, views, cameras, faces_by_camera,
            face_geometry=loaders.face_geometry, face_mask=loaders.face_mask, photo_rgb=photo_rgb,
            projection=projection, log=log,
        )
        log(f"observations collected in {time.time() - t0:.0f}s; photos decoded {photo_rgb.loaded}")

        env_by_image = {str(r["image_id"]): str(r["environment"]) for r in membership}
        summary["colour_dispersion"] = {
            "inputs": {
                "face_manifest_sha256": face_manifest.get("face_manifest_sha256"),
                "lidar_geometry_manifest_sha256": (geometry_manifest or {}).get("face_lidar_geometry_manifest_sha256"),
                "training_views": len(views),
                "window_radius_px": COLOUR_WINDOW_RADIUS_PX,
                "min_views": MIN_VIEWS_FOR_DISPERSION,
                "seed": args.seed,
                "effective_view_rule": "WP03 strict: face mask valid, vis6 return within 0.1 m + 3 % range in 13x13 px, no nearer return",
            },
            "regions": {},
        }
        dispersion_rows: list[dict[str, Any]] = []
        region_labels = np.asarray(region_of)
        for region in regions:
            idx = np.flatnonzero(region_labels == region.label)
            region_obs = [obs[i] for i in idx]
            tile_records = gains_by_tile.get(region.tile, {})
            learned = {i: r.gain for i, r in tile_records.items()}
            tile_rows = [r for r in rows if r["tile"] == region.tile]
            by_camera_mean: dict[str, float] = {}
            for camera in {r["camera"] for r in tile_rows}:
                by_camera_mean[camera] = math.exp(float(np.mean([r["log_gain"] for r in tile_rows if r["camera"] == camera])))
            camera_of = {r["image_id"]: r["camera"] for r in tile_rows}
            per_camera = {i: by_camera_mean[camera_of[i]] for i in learned}
            t_ns = {r["image_id"]: r["timestamp_ns"] for r in tile_rows}
            t_min = min(t_ns.values()) if t_ns else 0
            variants: dict[str, dict[str, float]] = {"learned": learned, "per_camera_mean": per_camera}
            for block in args.time_blocks_s:
                block_of = {i: f"{camera_of[i]}|{int((t_ns[i] - t_min) / 1e9 // block)}" for i in learned}
                block_mean: dict[str, list[float]] = defaultdict(list)
                for i, key in block_of.items():
                    block_mean[key].append(math.log(learned[i]))
                variants[f"per_camera_time_block_{block:g}s"] = {i: math.exp(float(np.mean(block_mean[key]))) for i, key in block_of.items()}
            oracle = oracle_image_gains(region_obs)
            variants["oracle_per_image"] = oracle
            oracle_in_tile = {i: g for i, g in oracle.items() if i in learned}
            variants["oracle_per_image_tile_views"] = oracle_in_tile
            # views from images this tile did not train carry no gain; report how many
            all_view_ids = {i for point in region_obs for i, _, _ in point}
            point_rows = colour_dispersion(region_obs, variants)
            for row in point_rows:
                row["region"] = region.label
                row["tile"] = region.tile
                dispersion_rows.append(row)
            region_summary = summarize_dispersion(point_rows, list(variants))
            region_summary.update(region_meta[region.label])
            region_summary["views_seen"] = len(all_view_ids)
            region_summary["views_seen_with_tile_gain"] = len(all_view_ids & set(learned))
            region_summary["views_seen_by_environment"] = dict(sorted(
                {env: sum(1 for i in all_view_ids if env_by_image.get(i) == env) for env in set(env_by_image.values())}.items()
            ))
            common = sorted(set(oracle) & set(learned))
            if len(common) >= 3:
                lo = np.asarray([math.log(oracle[i]) for i in common])
                ll = np.asarray([math.log(learned[i]) for i in common])
                region_summary["oracle_vs_learned"] = {
                    "images": len(common),
                    "corr_log": float(np.corrcoef(lo, ll)[0, 1]),
                    "oracle_log_std": float(lo.std()),
                    "learned_log_std": float(ll.std()),
                    "slope_learned_on_oracle": float(np.polyfit(lo, ll, 1)[0]),
                    "residual_log_std_after_fit": float((ll - np.polyval(np.polyfit(lo, ll, 1), lo)).std()),
                }
            summary["colour_dispersion"]["regions"][region.label] = region_summary
            log(f"[{region.label}] {len(point_rows)} points with >= {MIN_VIEWS_FOR_DISPERSION} effective views")
        if args.out_dispersion_csv is not None and dispersion_rows:
            names = ["region", "tile"] + [k for k in dispersion_rows[0] if k not in {"region", "tile"}]
            write_csv(args.out_dispersion_csv, dispersion_rows, names)
            log(f"wrote {args.out_dispersion_csv} ({len(dispersion_rows)} rows)")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    log(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
