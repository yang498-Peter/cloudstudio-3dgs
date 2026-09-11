"""CPU estimate of the pixels ``rgb_supervision_mask: lidar_support`` keeps.

For every view of a diagnostic selection (``DIAG_40/selection.json``) the tool
rebuilds exactly what the trainer would supervise under the F-line knob:
renderer mask PNG (cropped to the Tile view) AND the signed face-LiDAR
support dilated by ``rgb_supervision_dilation_radius_px`` (the numpy twin
of the trainer's construction, ``rgb_supervision_mask_numpy``). It reports
the supervised share of the renderer mask per view, per face type
(pitch_up / pitch_down / yaw) and inside the region's ROI boxes
(``roi_in_crops``), so we know how much of the door / gravel ROI keeps its
photometric supervision before the GPU arm runs.

No torch, no CUDA; reads the data read-only and writes one JSON + one
Markdown table.

    python tools/estimate_rgb_supervision_fraction.py \
        --config C:/Peter/3dgs-runs/house0305_sop/diag_indoor_door_leaf_Tile_1_40_F1_c134.json \
        --selection C:/Peter/3dgs-runs/house0305_sop/diag_v2/indoor_door_leaf_Tile_1/DIAG_40/selection.json \
        --out research/quality_recovery_v2/09_rgb_supervision_fraction_indoor.json

``--lidar-manifest/--lidar-root`` override the config's face LiDAR cache so
the same views can be measured against another cache (e.g. vis6f).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth  # noqa: E402
from cloudstudio_3dgs.training.face_dataset import (  # noqa: E402
    _dilate_bool,
    tile_ownership_masks,
)
from cloudstudio_3dgs.training.rgb_supervision import (  # noqa: E402
    rgb_supervision_mask_numpy,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _face_type(face_id: str) -> str:
    if face_id.startswith("pitch_up"):
        return "pitch_up"
    if face_id.startswith("pitch_down"):
        return "pitch_down"
    return "yaw"


class Region:
    def __init__(
        self,
        config_path: Path,
        selection_path: Path,
        *,
        lidar_manifest: Path | None,
        lidar_root: Path | None,
    ) -> None:
        self.config = _read_json(config_path)
        self.selection = _read_json(selection_path)
        self.label = str(self.selection["region"]["label"])
        if self.config.get("tile_ownership_masking"):
            raise SystemExit("tile_ownership_masking is enabled; the owned/foreign split is not reproduced here")
        if int(self.config.get("factor", 1)) != 1:
            raise SystemExit("only factor = 1 face views are supported")
        self.mode = str(self.config.get("rgb_supervision_mask", "all"))
        self.radius = int(self.config.get("rgb_supervision_dilation_radius_px", 24))
        self.face_cache_root = Path(self.config["face_cache_root"])
        manifest_path = Path(lidar_manifest or self.config["face_lidar_geometry_manifest"])
        self.lidar_root = Path(lidar_root or self.config["face_lidar_geometry_root"])
        self.lidar_manifest_path = manifest_path
        geometry_manifest = _read_json(manifest_path)
        self.lidar_manifest_sha = geometry_manifest.get("face_lidar_geometry_manifest_sha256")
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
        self.image_by_id: dict[str, dict[str, Any]] = {}
        for image in face_manifest["images"]:
            image_id = str(image["image_id"])
            self.image_by_id[image_id] = image
            for face in image["faces"]:
                self.face_by_sample[f"{image_id}::{face['face_id']}"] = face
        # Face plan (K_face / R_face) per physical camera, as the dataset uses
        # it to build the pinhole sample pose: c2w = c2w_base @ [R_face].
        self.face_plan: dict[tuple[str, str], dict[str, Any]] = {}
        for camera_id, camera in face_manifest["cameras"].items():
            for face in camera["faces"]:
                self.face_plan[(str(camera_id), str(face["face_id"]))] = face
        tile_manifest = _read_json(Path(self.selection["tile_inputs_manifest"]))
        (tile,) = tile_manifest["tiles"]
        self.tile_name = str(tile["name"])
        # The same box the trainer hands the dataset under tile_ownership_masking.
        self.tile_box = np.asarray(tile["training_and_export_box"], dtype=np.float64)
        self.ownership_margin_m = float(self.config.get("tile_ownership_margin_m", 0.5))
        self.ownership_dilation_px = int(self.config.get("tile_ownership_dilation_px", 15))
        self.tile_ownership = False
        self.views = {str(view["sample_id"]): view for view in tile["views"]}
        self.roi_by_sample = {
            str(entry["sample_id"]): entry.get("roi")
            for entry in self.selection.get("roi_in_crops", [])
        }
        self.sample_ids = [str(sid) for sid in self.selection["view_sample_ids"]]
        missing = [sid for sid in self.sample_ids if sid not in self.views]
        if missing:
            raise SystemExit(f"{len(missing)} sample ids missing from the tile views")

    def measure(self, sample_id: str) -> dict[str, Any]:
        image_id, face_id = sample_id.split("::", 1)
        face = self.face_by_sample[sample_id]
        renderer = self.renderer_by_face.get((image_id, face_id), face)
        with Image.open(self.face_cache_root / str(renderer["mask_path"])) as source:
            rgb_mask = np.asarray(source.convert("L"), dtype=np.uint8) > 0
        record = self.geometry_by_sample.get(sample_id)
        view = self.views[sample_id]
        x, y = int(view["x"]), int(view["y"])
        crop = (slice(y, y + int(view["height"])), slice(x, x + int(view["width"])))
        rgb_mask = np.ascontiguousarray(rgb_mask[crop])
        if record is not None and record.get("path"):
            sparse = load_sparse_depth(self.lidar_root / str(record["path"]))
            depth_range, confidence, depth_valid = sparse.to_dense()
            depth_mask = rgb_mask & np.ascontiguousarray(depth_valid[crop])
            confidence = np.ascontiguousarray(confidence[crop])
            depth_range = np.ascontiguousarray(depth_range[crop])
        else:
            depth_mask = None
            confidence = None
            depth_range = None
        # The dataset crops before the trainer pools, so crop-then-dilate is
        # the faithful order (support cannot leak in from outside the crop).
        supervision = rgb_supervision_mask_numpy(
            rgb_mask=rgb_mask,
            depth_mask=depth_mask,
            confidence=confidence,
            mode="lidar_support",
            dilation_radius_px=self.radius,
        )
        returns = int(depth_mask.sum()) if depth_mask is not None else 0
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "image_id": image_id,
            "face_id": face_id,
            "face_type": _face_type(face_id),
            "crop": {"x": x, "y": y, "width": int(view["width"]), "height": int(view["height"])},
            "rgb_mask_pixels": supervision.rgb_mask_pixels,
            "lidar_return_pixels": returns,
            "supervised_pixels": supervision.supervised_pixels,
            "supervised_fraction_of_rgb_mask": supervision.fraction,
            "supervised_fraction_of_crop": float(supervision.supervised_pixels) / float(rgb_mask.size),
            "roi": None,
        }
        ownership_masks: dict[str, np.ndarray] = {}
        if self.tile_ownership and depth_mask is not None and returns > 0:
            # Which of the returns lie inside the Tile's own box: the face
            # LiDAR cache is scene-wide, so "LiDAR support" is not the same as
            # "Tile-owned". Reproduces FaceCacheDataset's tile_ownership_masks
            # call (K shifted by the crop, c2w = c2w_base @ [R_face]).
            plan = self.face_plan[(str(self.image_by_id[image_id]["camera_id"]), face_id)]
            K = np.asarray(plan["K_face"], dtype=np.float64).copy()
            K[0, 2] -= float(x)
            K[1, 2] -= float(y)
            face_to_base = np.eye(4)
            face_to_base[:3, :3] = np.asarray(plan["R_face"], dtype=np.float64)
            c2w = np.asarray(self.image_by_id[image_id]["c2w"], dtype=np.float64) @ face_to_base
            owned, foreign_region = tile_ownership_masks(
                depth_range, depth_mask, K, c2w, self.tile_box,
                self.ownership_margin_m, self.ownership_dilation_px,
            )
            owned_valid = owned & (confidence > 0.0)
            ownership_masks = {
                # F1 knob + the existing tile_ownership_masking knob.
                "lidar_support_minus_foreign": supervision.mask & ~foreign_region,
                # Hypothetical stricter construction: dilate owned returns only.
                "owned_support_only": rgb_mask & _dilate_bool(owned_valid, self.radius),
            }
            row["ownership"] = {
                "owned_return_fraction": float(owned.sum()) / float(returns),
                "foreign_region_fraction_of_rgb_mask": float((foreign_region & rgb_mask).sum()) / float(max(1, supervision.rgb_mask_pixels)),
                **{
                    f"{name}_fraction_of_rgb_mask": float(mask.sum()) / float(max(1, supervision.rgb_mask_pixels))
                    for name, mask in ownership_masks.items()
                },
            }
        roi = self.roi_by_sample.get(sample_id)
        if roi:
            window = (slice(int(roi["y0"]), int(roi["y1"])), slice(int(roi["x0"]), int(roi["x1"])))
            roi_rgb = int(rgb_mask[window].sum())
            roi_sup = int(supervision.mask[window].sum())
            row["roi"] = {
                "box": {key: int(roi[key]) for key in ("x0", "y0", "x1", "y1")},
                "rgb_mask_pixels": roi_rgb,
                "supervised_pixels": roi_sup,
                "supervised_fraction_of_rgb_mask": (roi_sup / roi_rgb) if roi_rgb else None,
            }
            for name, mask in ownership_masks.items():
                row["roi"][f"{name}_fraction_of_rgb_mask"] = (
                    (int(mask[window].sum()) / roi_rgb) if roi_rgb else None
                )
        return row


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "q1": ordered[len(ordered) // 4],
        "q3": ordered[(3 * len(ordered)) // 4],
        "mean": statistics.fmean(ordered),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_face: dict[str, list[float]] = {}
    for row in rows:
        by_face.setdefault(row["face_type"], []).append(row["supervised_fraction_of_rgb_mask"])
    roi_rows = [row for row in rows if row["roi"] and row["roi"]["supervised_fraction_of_rgb_mask"] is not None]
    roi_by_face: dict[str, list[float]] = {}
    for row in roi_rows:
        roi_by_face.setdefault(row["face_type"], []).append(row["roi"]["supervised_fraction_of_rgb_mask"])
    total_rgb = sum(row["rgb_mask_pixels"] for row in rows)
    total_sup = sum(row["supervised_pixels"] for row in rows)
    summary = {
        "views": len(rows),
        "all_views": _stats([row["supervised_fraction_of_rgb_mask"] for row in rows]),
        "pixel_weighted_fraction": (total_sup / total_rgb) if total_rgb else None,
        "by_face_type": {face: _stats(values) for face, values in sorted(by_face.items())},
        "views_without_returns": sum(1 for row in rows if row["lidar_return_pixels"] == 0),
        "roi_views": len(roi_rows),
        "roi_all": _stats([row["roi"]["supervised_fraction_of_rgb_mask"] for row in roi_rows]),
        "roi_by_face_type": {face: _stats(values) for face, values in sorted(roi_by_face.items())},
    }
    owned_rows = [row for row in rows if row.get("ownership")]
    if owned_rows:
        ownership: dict[str, Any] = {"views": len(owned_rows)}
        for key in (
            "owned_return_fraction",
            "foreign_region_fraction_of_rgb_mask",
            "lidar_support_minus_foreign_fraction_of_rgb_mask",
            "owned_support_only_fraction_of_rgb_mask",
        ):
            ownership[key] = _stats([row["ownership"][key] for row in owned_rows])
            ownership[f"{key}_by_face_type"] = {
                face: _stats([row["ownership"][key] for row in owned_rows if row["face_type"] == face])
                for face in sorted({row["face_type"] for row in owned_rows})
            }
        for name in ("lidar_support_minus_foreign", "owned_support_only"):
            key = f"{name}_fraction_of_rgb_mask"
            values = [row["roi"][key] for row in owned_rows if row["roi"] and row["roi"].get(key) is not None]
            ownership[f"roi_{key}"] = _stats(values)
        summary["ownership"] = ownership
    return summary


def markdown(label: str, summary: dict[str, Any], *, radius: int) -> str:
    def fmt(stat: dict[str, Any]) -> str:
        if not stat.get("n"):
            return "| - | - | - | - |"
        return f"| {stat['n']} | {stat['median']:.3f} | {stat['min']:.3f} | {stat['q1']:.3f}-{stat['q3']:.3f} |"

    lines = [
        f"### {label} (radius {radius} px)",
        "",
        "| scope | n | median | min | Q1-Q3 |",
        "|---|---|---|---|---|",
        f"| all views {fmt(summary['all_views'])}",
    ]
    for face, stat in summary["by_face_type"].items():
        lines.append(f"| {face} {fmt(stat)}")
    lines.append(f"| ROI (all) {fmt(summary['roi_all'])}")
    for face, stat in summary["roi_by_face_type"].items():
        lines.append(f"| ROI {face} {fmt(stat)}")
    pw = summary["pixel_weighted_fraction"]
    lines.append("")
    lines.append(
        f"pixel-weighted supervised share {pw:.3f}; views without any LiDAR return: "
        f"{summary['views_without_returns']} / {summary['views']}"
    )
    ownership = summary.get("ownership")
    if ownership:
        lines += [
            "",
            f"Tile ownership (returns inside the Tile box + margin; {ownership['views']} views):",
            "",
            "| quantity | n | median | min | Q1-Q3 |",
            "|---|---|---|---|---|",
            f"| owned share of returns {fmt(ownership['owned_return_fraction'])}",
        ]
        for face, stat in ownership["owned_return_fraction_by_face_type"].items():
            lines.append(f"| owned share of returns, {face} {fmt(stat)}")
        lines.append(f"| foreign region share of rgb_mask {fmt(ownership['foreign_region_fraction_of_rgb_mask'])}")
        lines.append(f"| supervised: lidar_support minus foreign region {fmt(ownership['lidar_support_minus_foreign_fraction_of_rgb_mask'])}")
        for face, stat in ownership["lidar_support_minus_foreign_fraction_of_rgb_mask_by_face_type"].items():
            lines.append(f"| supervised: lidar_support minus foreign, {face} {fmt(stat)}")
        lines.append(f"| supervised: owned returns only, dilated {fmt(ownership['owned_support_only_fraction_of_rgb_mask'])}")
        for face, stat in ownership["owned_support_only_fraction_of_rgb_mask_by_face_type"].items():
            lines.append(f"| supervised: owned only, {face} {fmt(stat)}")
        lines.append(f"| ROI: lidar_support minus foreign {fmt(ownership['roi_lidar_support_minus_foreign_fraction_of_rgb_mask'])}")
        lines.append(f"| ROI: owned returns only {fmt(ownership['roi_owned_support_only_fraction_of_rgb_mask'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="arm config carrying rgb_supervision_* and the cache paths")
    parser.add_argument("--selection", type=Path, required=True, help="DIAG_40/selection.json of the region")
    parser.add_argument("--radius", type=int, default=None, help="override rgb_supervision_dilation_radius_px")
    parser.add_argument("--lidar-manifest", type=Path, default=None)
    parser.add_argument("--lidar-root", type=Path, default=None)
    parser.add_argument(
        "--tile-ownership",
        action="store_true",
        help="also split the returns into Tile-owned / foreign (tile_ownership_masks) and report the combined masks",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON output; a .md sibling is written too")
    args = parser.parse_args(argv)

    region = Region(args.config, args.selection, lidar_manifest=args.lidar_manifest, lidar_root=args.lidar_root)
    if args.radius is not None:
        region.radius = int(args.radius)
    region.tile_ownership = bool(args.tile_ownership)
    started = time.time()
    rows = []
    for index, sample_id in enumerate(region.sample_ids, start=1):
        rows.append(region.measure(sample_id))
        if index % 20 == 0 or index == len(region.sample_ids):
            print(f"[{region.label}] {index}/{len(region.sample_ids)} views, {time.time() - started:.0f}s", flush=True)
    summary = summarise(rows)
    payload = {
        "kind": "rgb_supervision_fraction_estimate_v1",
        "generator": "tools/estimate_rgb_supervision_fraction.py",
        "region": region.label,
        "tile": region.tile_name,
        "config": str(args.config),
        "selection": str(args.selection),
        "rgb_supervision_mask": "lidar_support",
        "rgb_supervision_dilation_radius_px": region.radius,
        "face_lidar_geometry_manifest": str(region.lidar_manifest_path),
        "face_lidar_geometry_manifest_sha256": region.lidar_manifest_sha,
        "tile_ownership": (
            None
            if not region.tile_ownership
            else {
                "training_and_export_box": region.tile_box.tolist(),
                "margin_m": region.ownership_margin_m,
                "dilation_px": region.ownership_dilation_px,
            }
        ),
        "construction": (
            "rgb_mask = renderer mask PNG > 0, cropped to the Tile view; depth_mask = rgb_mask & sparse.valid; "
            "support = max_pool(depth_mask & confidence > 0, 2r+1); supervised = rgb_mask & support "
            "(cloudstudio_3dgs.training.rgb_supervision.rgb_supervision_mask_numpy)"
        ),
        "summary": summary,
        "views": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    md = markdown(region.label, summary, radius=region.radius)
    args.out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
