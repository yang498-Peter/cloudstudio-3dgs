#!/usr/bin/env python
"""Current vs strict LiDAR alpha-support masks for the WP03 diagnostic views.

For every Face4 tile view of a diagnostic selection (``selection.json`` from
``tools/build_diagnostic_set.py``) this tool rebuilds, on CPU and without any
new cache:

(a) the CURRENT alpha support exactly as the trainer builds it
    (``lidar_alpha_support_mode = "dilated"``: depth_mask & finite confidence
    & confidence > 0, zero-fill, (2r+1)^2 max-pool of the confidence, > 0;
    ``cloudstudio_3dgs/training/alpha_support.py`` mirrors the inline code
    the trainer carried at ``trainer.py`` ``_render_supervision_loss`` before
    the knob existed and is pinned to it by ``tests/test_alpha_support.py``);
(b) the STRICT support (``"strict_visibility"``: a pixel is supported only
    when every return in its window lies within 0.1 m + 3 % of the nearest
    return, and windows that disagree are grown by the edge erosion radius
    before being removed);
(c) their difference, classified per rejected pixel:
    ``discontinuity``  window spread also violates the loose vis6 rule
                       (farthest > 1.2 x nearest + 0.1 m);
    ``interior_band``  spread violates only the strict band (oblique or
                       curved surface, or a small step);
    ``edge_erosion``   window agrees but lies within the erosion radius of a
                       disagreeing window.

Inputs are read the way ``FaceCacheDataset.__getitem__`` reads them
(``cloudstudio_3dgs/training/face_dataset.py``): the renderer mask PNG is the
``rgb_mask``; the Face4 LiDAR geometry npz is densified with
``SparseDepthMap.to_dense`` and ``depth_mask = rgb_mask & depth_valid``; the
tile view crop (x, y, width, height) is applied to every raster before the
window filters run, so the crop border pads with "no return" exactly as the
trainer's max-pool does on the cropped tensors. The trainer's reported
``lidar_alpha_support_fraction`` is ``(rgb_mask & support).mean()`` over all
crop pixels; the CSV reports that and the fraction over RGB-valid pixels.

What cannot be reproduced offline: nothing in the mask itself. Two things
are deliberately outside this tool: tile ownership masking (the diagnostic
arms do not enable it; the tool refuses a config that does) and the
rendered alpha, which is what the loss compares against the mask.

Outputs (``--output-dir``): ``support_fractions.csv`` (one row per view),
``summary.json`` (per-region aggregates, trainer line references, optional
torch cross-check), ``<region>__<n>__<sample>.png`` whole-view panels and
``<region>__<n>__<sample>__roi.png`` full-resolution ROI panels for the
representative views.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth  # noqa: E402
from cloudstudio_3dgs.training.alpha_support import (  # noqa: E402
    STRICT_VISIBILITY_EDGE_EROSION_PX,
    STRICT_VISIBILITY_MARGIN_M,
    STRICT_VISIBILITY_TOLERANCE,
    lidar_alpha_support_numpy,
    window_range_spread_numpy,
)

TRAINER_REFERENCES = {
    "alpha_mask_construction": (
        "cloudstudio_3dgs/training/trainer.py::_render_supervision_loss "
        "(lidar_alpha_loss block) -> cloudstudio_3dgs/training/alpha_support.py::lidar_alpha_support"
    ),
    "legacy_inline_construction": (
        "trainer.py (pre-knob): lidar_alpha_valid = depth_mask & isfinite(confidence) & (confidence > 0); "
        "lidar_alpha_confidence = where(valid, confidence, 0); if radius > 0: max_pool2d(kernel 2r+1, stride 1, pad r); "
        "lidar_alpha_valid = pooled > 0; lidar_alpha_mask = rgb_mask & lidar_alpha_valid; "
        "lidar_alpha_support_fraction = lidar_alpha_mask.mean()"
    ),
    "tensor_sample": "trainer.py::_tensor_sample (range_m / confidence / depth_mask from TrainingSample)",
    "face_dataset_load": (
        "face_dataset.py::FaceCacheDataset.__getitem__: rgb_mask = renderer mask PNG > 0; "
        "depth_range, depth_confidence, depth_valid = load_sparse_depth(npz).to_dense(); "
        "depth_mask = rgb_mask & depth_valid; crop [y:y+h, x:x+w]"
    ),
}

REJECT_KINDS = ("discontinuity", "interior_band", "edge_erosion")


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Region:
    """One diagnostic selection bound to the trainer config that consumes it."""

    def __init__(self, selection_path: Path, config_path: Path) -> None:
        self.selection_path = Path(selection_path)
        self.config_path = Path(config_path)
        self.selection = _read_json(self.selection_path)
        self.config = _read_json(self.config_path)
        self.label = str(self.selection["region"]["label"])
        if self.config.get("tile_ownership_masking"):
            raise SystemExit(
                f"{config_path}: tile_ownership_masking is enabled; this tool does not "
                "reproduce the owned/foreign split and would misreport the mask"
            )
        if int(self.config.get("factor", 1)) != 1:
            raise SystemExit(f"{config_path}: only factor = 1 face views are supported")
        if str(self.config.get("lidar_alpha_support_mode", "dilated")) != "dilated":
            raise SystemExit(
                f"{config_path}: the CURRENT mask is defined as the dilated mode; "
                "the config already selects another mode"
            )
        self.radius = int(self.config.get("lidar_alpha_dilation_radius_px", 0))
        self.alpha_weight = float(self.config.get("lidar_alpha_weight", 0.0))
        self.alpha_target = float(self.config.get("lidar_alpha_target", 0.95))
        self.face_cache_root = Path(self.config["face_cache_root"])
        self.geometry_root = Path(self.config["face_lidar_geometry_root"])
        geometry_manifest = _read_json(Path(self.config["face_lidar_geometry_manifest"]))
        self.projection_config = dict(geometry_manifest.get("projection_config", {}))
        self.geometry_by_sample = {
            str(record["sample_id"]): record for record in geometry_manifest["records"]
        }
        renderer_manifest = _read_json(Path(self.config["renderer_mask_manifest"]))
        self.renderer_by_face = {
            (str(record["image_id"]), str(record["face_id"])): record
            for record in renderer_manifest["masks"]
        }
        face_manifest = _read_json(Path(self.config["face_cache_manifest"]))
        self.face_by_sample: dict[str, dict[str, Any]] = {}
        self.camera_by_image: dict[str, str] = {}
        for image in face_manifest["images"]:
            image_id = str(image["image_id"])
            self.camera_by_image[image_id] = str(image["camera_id"])
            for face in image["faces"]:
                self.face_by_sample[f"{image_id}::{face['face_id']}"] = face
        tile_manifest = _read_json(Path(self.selection["tile_inputs_manifest"]))
        (tile,) = tile_manifest["tiles"]
        self.tile_name = str(tile["name"])
        self.views = {str(view["sample_id"]): view for view in tile["views"]}
        self.roi_by_sample = {
            str(entry["sample_id"]): entry.get("roi")
            for entry in self.selection.get("roi_in_crops", [])
        }
        self.sample_ids = [str(sid) for sid in self.selection["view_sample_ids"]]
        missing = [sid for sid in self.sample_ids if sid not in self.views]
        if missing:
            raise SystemExit(f"{selection_path}: {len(missing)} sample ids missing from the tile views")

    def load_view(self, sample_id: str) -> dict[str, Any]:
        image_id, face_id = sample_id.split("::", 1)
        face = self.face_by_sample[sample_id]
        renderer = self.renderer_by_face.get((image_id, face_id), face)
        with Image.open(self.face_cache_root / str(face["rgb_path"])) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        with Image.open(self.face_cache_root / str(renderer["mask_path"])) as source:
            rgb_mask = np.asarray(source.convert("L"), dtype=np.uint8) > 0
        record = self.geometry_by_sample[sample_id]
        if record.get("path"):
            sparse = load_sparse_depth(self.geometry_root / str(record["path"]))
            depth_range, depth_confidence, depth_valid = sparse.to_dense()
        else:
            depth_range = np.zeros(rgb_mask.shape, dtype=np.float32)
            depth_confidence = np.zeros(rgb_mask.shape, dtype=np.float32)
            depth_valid = np.zeros(rgb_mask.shape, dtype=bool)
        depth_mask = rgb_mask & depth_valid
        view = self.views[sample_id]
        x, y = int(view["x"]), int(view["y"])
        right, bottom = x + int(view["width"]), y + int(view["height"])
        crop = (slice(y, bottom), slice(x, right))
        return {
            "sample_id": sample_id,
            "image_id": image_id,
            "face_id": face_id,
            "camera_id": self.camera_by_image[image_id],
            "crop": {"x": x, "y": y, "width": int(view["width"]), "height": int(view["height"])},
            "rgb": np.ascontiguousarray(rgb[crop]),
            "rgb_mask": np.ascontiguousarray(rgb_mask[crop]),
            "range_m": np.ascontiguousarray(depth_range[crop]),
            "confidence": np.ascontiguousarray(depth_confidence[crop]),
            "depth_mask": np.ascontiguousarray(depth_mask[crop]),
            "roi": self.roi_by_sample.get(sample_id),
        }


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------


def analyse_view(view: dict[str, Any], *, radius: int, loose_tolerance: float, loose_margin_m: float) -> dict[str, Any]:
    current = lidar_alpha_support_numpy(
        depth_mask=view["depth_mask"],
        confidence=view["confidence"],
        range_m=view["range_m"],
        mode="dilated",
        dilation_radius_px=radius,
    )
    strict = lidar_alpha_support_numpy(
        depth_mask=view["depth_mask"],
        confidence=view["confidence"],
        range_m=view["range_m"],
        mode="strict_visibility",
        dilation_radius_px=radius,
    )
    rgb_mask = view["rgb_mask"]
    current_mask = rgb_mask & current.support
    strict_mask = rgb_mask & strict.support
    rejected = current_mask & ~strict_mask
    valid = view["depth_mask"] & np.isfinite(view["confidence"]) & (view["confidence"] > 0.0)
    valid &= np.isfinite(view["range_m"]) & (view["range_m"] > 0.0)
    nearest, farthest = window_range_spread_numpy(valid=valid, range_m=view["range_m"], radius=radius)
    loose_violation = np.isfinite(farthest) & (
        farthest > nearest * np.float32(1.0 + loose_tolerance) + np.float32(loose_margin_m)
    )
    discontinuity = strict.discontinuity
    kinds = {
        "discontinuity": rejected & loose_violation,
        "interior_band": rejected & discontinuity & ~loose_violation,
        "edge_erosion": rejected & ~discontinuity,
    }
    # Raster purity, independent of the support rule: how many of the
    # signed returns sit within the strict band of the nearest return of
    # their own window, and how many 3x3 windows already mix surfaces
    # beyond the loose vis6 rule (the hidden-point filter's own tolerance).
    own = view["range_m"][valid]
    in_band = own <= nearest[valid] * np.float32(1.0 + STRICT_VISIBILITY_TOLERANCE) + np.float32(STRICT_VISIBILITY_MARGIN_M)
    near1, far1 = window_range_spread_numpy(valid=valid, range_m=view["range_m"], radius=1)
    has1 = np.isfinite(far1)
    loose1 = has1 & (far1 > near1 * np.float32(1.0 + loose_tolerance) + np.float32(loose_margin_m))
    return {
        "current": current_mask,
        "strict": strict_mask,
        "rejected": rejected,
        "kinds": kinds,
        "returns": valid,
        "loose_violation": loose_violation & current_mask,
        "return_range_median_m": float(np.median(own)) if valid.any() else float("nan"),
        "returns_in_strict_band_of_window_nearest_frac": float(in_band.mean()) if valid.any() else float("nan"),
        "windows_r1_loose_violation_frac": float(loose1[has1].mean()) if has1.any() else float("nan"),
    }


def _fractions(masks: dict[str, Any], window: tuple[slice, slice] | None = None) -> dict[str, Any]:
    def area(mask: np.ndarray) -> int:
        return int(mask[window].sum()) if window is not None else int(mask.sum())

    def total(mask: np.ndarray) -> int:
        return int(mask[window].size) if window is not None else int(mask.size)

    current_px = area(masks["current"])
    strict_px = area(masks["strict"])
    rejected_px = area(masks["rejected"])
    rgb_px = area(masks["rgb_mask"])
    all_px = total(masks["current"])
    out = {
        "pixels": all_px,
        "rgb_valid_px": rgb_px,
        "lidar_return_px": area(masks["returns"]),
        "current_support_px": current_px,
        "strict_support_px": strict_px,
        "rejected_px": rejected_px,
        "current_support_frac_all": current_px / all_px if all_px else float("nan"),
        "current_support_frac_rgb": current_px / rgb_px if rgb_px else float("nan"),
        "strict_support_frac_all": strict_px / all_px if all_px else float("nan"),
        "strict_support_frac_rgb": strict_px / rgb_px if rgb_px else float("nan"),
        "rejected_frac_of_current": rejected_px / current_px if current_px else float("nan"),
        "loose_violation_frac_of_current": (
            area(masks["loose_violation"]) / current_px if current_px else float("nan")
        ),
    }
    for kind in REJECT_KINDS:
        px = area(masks["kinds"][kind])
        out[f"rejected_{kind}_px"] = px
        out[f"rejected_{kind}_frac_of_current"] = px / current_px if current_px else float("nan")
    return out


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------


def _block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array.astype(np.float32)
    h, w = array.shape[:2]
    h2, w2 = h // factor * factor, w // factor * factor
    trimmed = array[:h2, :w2].astype(np.float32)
    if trimmed.ndim == 2:
        return trimmed.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
    return trimmed.reshape(h2 // factor, factor, w2 // factor, factor, -1).mean(axis=(1, 3))


def _tint(base: np.ndarray, weight: np.ndarray, colour: tuple[int, int, int], strength: float) -> np.ndarray:
    alpha = np.clip(weight * strength, 0.0, 1.0)[..., None]
    return base * (1.0 - alpha) + np.asarray(colour, dtype=np.float32)[None, None, :] * alpha


COLOURS = {
    "current": (70, 150, 255),
    "strict": (60, 220, 90),
    "discontinuity": (255, 50, 40),
    "interior_band": (255, 175, 0),
    "edge_erosion": (225, 70, 225),
    "returns": (255, 240, 80),
}


def render_panels(
    view: dict[str, Any],
    masks: dict[str, Any],
    *,
    factor: int,
    window: tuple[slice, slice] | None,
    title: str,
    stats: dict[str, Any],
) -> Image.Image:
    rgb = view["rgb"] if window is None else view["rgb"][window]

    def sub(mask: np.ndarray) -> np.ndarray:
        return _block_mean(mask if window is None else mask[window], factor)

    base = _block_mean(rgb, factor) * 0.45
    rgb_valid = sub(view["rgb_mask"])
    panels: list[tuple[str, np.ndarray]] = []
    returns = sub(masks["returns"])
    panels.append(("photo + LiDAR returns", _tint(_block_mean(rgb, factor), returns * factor * factor / 2.0, COLOURS["returns"], 1.0)))
    panels.append(("CURRENT support (dilated r=%d)" % stats["radius"], _tint(base, sub(masks["current"]), COLOURS["current"], 0.7)))
    panels.append(("STRICT support", _tint(base, sub(masks["strict"]), COLOURS["strict"], 0.7)))
    diff = _tint(base, sub(masks["strict"]), COLOURS["strict"], 0.25)
    for kind in REJECT_KINDS:
        diff = _tint(diff, sub(masks["kinds"][kind]), COLOURS[kind], 0.95)
    panels.append(("rejected by strict: red=discontinuity orange=band magenta=erosion", diff))
    dim = np.clip(rgb_valid, 0.0, 1.0)[..., None] * 0.6 + 0.4
    header = 34
    ph, pw = panels[0][1].shape[:2]
    canvas = Image.new("RGB", (pw * len(panels) + (len(panels) - 1) * 4, ph + header), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 3), title, fill=(240, 240, 240))
    draw.text(
        (6, 17),
        "current %.3f | strict %.3f | rejected %.1f%% of current (disc %.1f%%, band %.1f%%, erosion %.1f%%)"
        % (
            stats["current_support_frac_rgb"],
            stats["strict_support_frac_rgb"],
            100.0 * stats["rejected_frac_of_current"],
            100.0 * stats["rejected_discontinuity_frac_of_current"],
            100.0 * stats["rejected_interior_band_frac_of_current"],
            100.0 * stats["rejected_edge_erosion_frac_of_current"],
        ),
        fill=(200, 200, 200),
    )
    for index, (label, panel) in enumerate(panels):
        image = Image.fromarray(np.clip(panel * dim, 0, 255).astype(np.uint8))
        x0 = index * (pw + 4)
        canvas.paste(image, (x0, header))
        panel_draw = ImageDraw.Draw(canvas)
        panel_draw.rectangle((x0, header, x0 + min(pw, 6 * len(label) + 8), header + 12), fill=(0, 0, 0))
        panel_draw.text((x0 + 3, header), label, fill=(255, 255, 255))
    return canvas


def _roi_window(roi: dict[str, Any], shape: tuple[int, int], margin: int) -> tuple[slice, slice]:
    h, w = shape
    x0 = max(0, int(roi["x0"]) - margin)
    y0 = max(0, int(roi["y0"]) - margin)
    x1 = min(w, int(roi["x1"]) + margin)
    y1 = min(h, int(roi["y1"]) + margin)
    return (slice(y0, y1), slice(x0, x1))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def _pick_representatives(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Evenly spaced along rejected fraction, ROI views first."""
    with_roi = [row for row in rows if row["has_roi"]]
    pool = with_roi if len(with_roi) >= count else rows
    pool = sorted(pool, key=lambda row: row["rejected_frac_of_current"])
    if len(pool) <= count:
        return pool
    picks = np.unique(np.linspace(0, len(pool) - 1, count).round().astype(int))
    return [pool[i] for i in picks]


def _summarise(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    keys = [
        "current_support_frac_all",
        "current_support_frac_rgb",
        "strict_support_frac_all",
        "strict_support_frac_rgb",
        "rejected_frac_of_current",
        "rejected_discontinuity_frac_of_current",
        "rejected_interior_band_frac_of_current",
        "rejected_edge_erosion_frac_of_current",
        "loose_violation_frac_of_current",
    ]
    if not prefix:
        keys += [
            "returns_in_strict_band_of_window_nearest_frac",
            "windows_r1_loose_violation_frac",
            "return_range_median_m",
        ]
    out: dict[str, Any] = {"views": len(rows)}
    for key in keys:
        values = np.asarray([row[prefix + key] for row in rows if np.isfinite(row.get(prefix + key, np.nan))])
        if values.size:
            out[key] = {
                "median": float(np.median(values)),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
                "mean": float(values.mean()),
            }
    total_current = sum(int(row[prefix + "current_support_px"]) for row in rows)
    total_strict = sum(int(row[prefix + "strict_support_px"]) for row in rows)
    total_rgb = sum(int(row[prefix + "rgb_valid_px"]) for row in rows)
    out["pixel_weighted"] = {
        "current_support_frac_rgb": total_current / total_rgb if total_rgb else float("nan"),
        "strict_support_frac_rgb": total_strict / total_rgb if total_rgb else float("nan"),
        "rejected_frac_of_current": (total_current - total_strict) / total_current if total_current else float("nan"),
    }
    for kind in REJECT_KINDS:
        px = sum(int(row[prefix + f"rejected_{kind}_px"]) for row in rows)
        out["pixel_weighted"][f"rejected_{kind}_frac_of_current"] = px / total_current if total_current else float("nan")
    return out


def _torch_cross_check(view: dict[str, Any], masks: dict[str, Any], radius: int) -> dict[str, Any]:
    import torch

    from cloudstudio_3dgs.training.alpha_support import lidar_alpha_support

    tensors = {
        "depth_mask": torch.from_numpy(view["depth_mask"]),
        "confidence": torch.from_numpy(view["confidence"]),
        "range_m": torch.from_numpy(view["range_m"]),
        "rgb_mask": torch.from_numpy(view["rgb_mask"]),
    }
    result = {}
    for mode, key in (("dilated", "current"), ("strict_visibility", "strict")):
        support = lidar_alpha_support(
            torch,
            depth_mask=tensors["depth_mask"],
            confidence=tensors["confidence"],
            range_m=tensors["range_m"],
            mode=mode,
            dilation_radius_px=radius,
        )
        mask = (tensors["rgb_mask"] & support.support).numpy()
        result[key] = bool(np.array_equal(mask, masks[key]))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", action="append", required=True, type=Path, help="selection.json (repeatable, pairs with --config)")
    parser.add_argument("--config", action="append", required=True, type=Path, help="trainer arm config consuming that selection")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overlay-views", type=int, default=12, help="representative views rendered per region")
    parser.add_argument("--downscale", type=int, default=4, help="whole-view panel downscale factor")
    parser.add_argument("--roi-margin-px", type=int, default=48)
    parser.add_argument("--roi-max-width", type=int, default=720, help="ROI panel width cap (downscaled to fit)")
    parser.add_argument("--limit", type=int, default=None, help="analyse only the first N views per region (smoke)")
    parser.add_argument("--cross-check-torch", type=int, default=0, help="also run the trainer's torch construction on CPU for N views per region")
    args = parser.parse_args(argv)
    if len(args.selection) != len(args.config):
        parser.error("--selection and --config must be given in pairs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "generator": "tools/build_alpha_support_overlays.py",
        "trainer_references": TRAINER_REFERENCES,
        "strict_rule": {
            "tolerance": STRICT_VISIBILITY_TOLERANCE,
            "margin_m": STRICT_VISIBILITY_MARGIN_M,
            "edge_erosion_px": STRICT_VISIBILITY_EDGE_EROSION_PX,
        },
        "regions": {},
        "rendered": [],
    }
    started = time.time()
    for selection_path, config_path in zip(args.selection, args.config):
        region = Region(selection_path, config_path)
        loose_tolerance = float(region.projection_config.get("visibility_tolerance", 0.2))
        loose_margin = float(region.projection_config.get("visibility_margin_m", 0.1))
        region_rows: list[dict[str, Any]] = []
        cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        cross_checks: list[dict[str, Any]] = []
        sample_ids = region.sample_ids[: args.limit] if args.limit else region.sample_ids
        print(f"[{region.label}] {len(sample_ids)} views, radius {region.radius}, alpha weight {region.alpha_weight}", flush=True)
        for index, sample_id in enumerate(sample_ids):
            view = region.load_view(sample_id)
            masks = analyse_view(view, radius=region.radius, loose_tolerance=loose_tolerance, loose_margin_m=loose_margin)
            masks["rgb_mask"] = view["rgb_mask"]
            row: dict[str, Any] = {
                "region": region.label,
                "tile": region.tile_name,
                "sample_id": sample_id,
                "image_id": view["image_id"],
                "camera_id": view["camera_id"],
                "face_id": view["face_id"],
                "crop_x": view["crop"]["x"],
                "crop_y": view["crop"]["y"],
                "crop_w": view["crop"]["width"],
                "crop_h": view["crop"]["height"],
                "radius_px": region.radius,
                "return_range_median_m": masks["return_range_median_m"],
                "returns_in_strict_band_of_window_nearest_frac": masks["returns_in_strict_band_of_window_nearest_frac"],
                "windows_r1_loose_violation_frac": masks["windows_r1_loose_violation_frac"],
                "has_roi": view["roi"] is not None,
            }
            row.update(_fractions(masks))
            if view["roi"] is not None:
                window = _roi_window(view["roi"], view["rgb_mask"].shape, 0)
                row.update({"roi_" + key: value for key, value in _fractions(masks, window).items()})
            region_rows.append(row)
            if index < args.cross_check_torch:
                check = _torch_cross_check(view, masks, region.radius)
                check["sample_id"] = sample_id
                cross_checks.append(check)
            cache[sample_id] = (view, masks)
            if (index + 1) % 10 == 0 or index + 1 == len(sample_ids):
                print(f"  {index + 1}/{len(sample_ids)} ({time.time() - started:.0f}s)", flush=True)
            # Keep memory bounded: only the views that may be rendered stay cached.
            if len(cache) > 400:
                cache.pop(next(iter(cache)))
        representatives = _pick_representatives(region_rows, args.overlay_views)
        for order, row in enumerate(representatives):
            view, masks = cache[row["sample_id"]]
            stats = {**row, "radius": region.radius}
            short = row["sample_id"].replace("::", "__")
            title = f"{region.label} | {row['sample_id']} | {row['camera_id']} | crop {row['crop_w']}x{row['crop_h']} | median return {row['return_range_median_m']:.2f} m"
            panel = render_panels(view, masks, factor=args.downscale, window=None, title=title, stats=stats)
            name = f"{region.label}__{order:02d}__{short}.png"
            panel.save(output_dir / name, optimize=True)
            rendered = {"region": region.label, "sample_id": row["sample_id"], "file": name, "rejected_frac_of_current": row["rejected_frac_of_current"]}
            if view["roi"] is not None:
                window = _roi_window(view["roi"], view["rgb_mask"].shape, args.roi_margin_px)
                roi_w = window[1].stop - window[1].start
                roi_factor = max(1, int(np.ceil(roi_w / args.roi_max_width)))
                roi_stats = {**{k[4:]: v for k, v in row.items() if k.startswith("roi_")}, "radius": region.radius}
                roi_title = f"ROI {region.label} | {row['sample_id']} | roi {view['roi']['x0']},{view['roi']['y0']}-{view['roi']['x1']},{view['roi']['y1']} (+{args.roi_margin_px} px) | 1:{roi_factor}"
                roi_panel = render_panels(view, masks, factor=roi_factor, window=window, title=roi_title, stats=roi_stats)
                roi_name = f"{region.label}__{order:02d}__{short}__roi.png"
                roi_panel.save(output_dir / roi_name, optimize=True)
                rendered["roi_file"] = roi_name
                rendered["roi_rejected_frac_of_current"] = row.get("roi_rejected_frac_of_current")
            summary["rendered"].append(rendered)
        region_summary = {
            "selection": str(region.selection_path),
            "config": str(region.config_path),
            "tile": region.tile_name,
            "radius_px": region.radius,
            "lidar_alpha_weight": region.alpha_weight,
            "lidar_alpha_target": region.alpha_target,
            "loose_rule": {"tolerance": loose_tolerance, "margin_m": loose_margin},
            "all_views": _summarise(region_rows),
            "roi_views": _summarise([row for row in region_rows if row["has_roi"]], prefix="roi_"),
            "by_face": {
                face: _summarise([row for row in region_rows if row["face_id"] == face])["pixel_weighted"]
                for face in sorted({row["face_id"] for row in region_rows})
            },
            "by_camera": {
                camera: _summarise([row for row in region_rows if row["camera_id"] == camera])["pixel_weighted"]
                for camera in sorted({row["camera_id"] for row in region_rows})
            },
        }
        if cross_checks:
            region_summary["torch_cross_check"] = {
                "views": len(cross_checks),
                "all_identical": all(c["current"] and c["strict"] for c in cross_checks),
                "details": cross_checks,
            }
        summary["regions"][region.label] = region_summary
        rows.extend(region_rows)
        cache.clear()

    csv_path = output_dir / "support_fractions.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {csv_path} ({len(rows)} rows) and summary.json in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
