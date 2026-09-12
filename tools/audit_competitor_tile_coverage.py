#!/usr/bin/env python3
"""How much of a Gaussian model lies outside the union of our Tile boxes?

Our pipeline trains one model per Tile and merges with ``core_owner_only``,
so nothing we deliver can lie outside the union of the four core boxes, and
nothing we *train* lies outside the union of the training/export boxes
except floaters. A competitor model has no such partition: whatever its
photos show - trees, far ground, the neighbours' fences - it owns. The
difference is exactly "content nobody in our pipeline owns", which is what a
Tile's per-view backdrop has to stand in for
(research/quality_recovery_v2/14_standin_backdrop_design.md).

The tool reads any binary/ascii PLY with x/y/z (a foreign splat PLY, one of
our exported PLYs, or a plain point cloud), optionally applies the rigid
alignment stored by tools/align_gaussian_ply.py (``transform`` maps model to
the reference LAS frame - the s1_local frame the Tile boxes are in), and
classifies every point against the Tile boxes of a signed Tile inputs
manifest. Where an ``opacity`` property exists the same counts are repeated
on the sigmoid(opacity) >= floor subset, so dead mass does not inflate the
outside share. A top-down raster (inside = blue, outside = red, boxes drawn)
goes with the JSON so the outside content can be seen, not just counted.

    python tools/audit_competitor_tile_coverage.py \
        --ply C:/baidunetdiskdownload/house/USAgs.ply \
        --transform-json C:/Peter/3dgs-runs/probes/usa_gs_alignment.json \
        --tile-inputs C:/Peter/3dgs-runs/house0305_sop/tile_inputs_v9/tile_inputs_manifest.json \
        --output-json coverage_usa.json --png coverage_usa.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest  # noqa: E402
from tools.gaussian_health import read_ply_records  # noqa: E402

BOX_KINDS = ("training_and_export_box", "core_box")
# Horizontal distance bands (metres beyond the union box) used to describe
# where the outside content sits: hugging the boundary, the yard, far field.
DISTANCE_BANDS_M = (0.0, 1.0, 5.0, 20.0, float("inf"))


def load_transform(path: Path | None) -> np.ndarray | None:
    """The 4x4 ``transform`` of an alignment JSON, or None for identity."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = np.asarray(payload["transform"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"alignment transform must be a finite 4x4 matrix: {path}")
    return matrix


def apply_transform(xyz: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    """Rigid model -> reference, the convention of tools/align_gaussian_ply.py."""
    points = np.asarray(xyz, dtype=np.float64)
    if transform is None:
        return points
    return points @ transform[:3, :3].T + transform[:3, 3]


def tile_boxes(manifest: dict[str, Any], kind: str) -> list[tuple[int, str, np.ndarray]]:
    if kind not in BOX_KINDS:
        raise ValueError(f"box kind must be one of {BOX_KINDS}")
    tiles = sorted(manifest["tiles"], key=lambda tile: int(tile["tile_id"]))
    boxes = []
    for tile in tiles:
        box = np.asarray(tile[kind], dtype=np.float64)
        if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[0] >= box[1]):
            raise ValueError(f"Tile_{tile['tile_id']} has an invalid {kind}")
        boxes.append((int(tile["tile_id"]), str(tile.get("name", f"Tile_{tile['tile_id']}")), box))
    return boxes


def classify_points(
    xyz: np.ndarray, boxes: list[tuple[int, str, np.ndarray]], *, tolerance_m: float = 0.0
) -> np.ndarray:
    """Owner Tile id per point (first box that contains it, ascending id) or -1."""
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N, 3]")
    owner = np.full(len(points), -1, dtype=np.int64)
    for tile_id, _, box in boxes:
        inside = np.all(
            (points >= box[0] - tolerance_m) & (points <= box[1] + tolerance_m), axis=1
        )
        owner[(owner < 0) & inside] = tile_id
    return owner


def horizontal_distance_beyond_union(xyz: np.ndarray, boxes) -> np.ndarray:
    """Horizontal (xy) distance from each point to the union bounding box; 0 inside."""
    lower = np.min([box[0] for _, _, box in boxes], axis=0)
    upper = np.max([box[1] for _, _, box in boxes], axis=0)
    points = np.asarray(xyz, dtype=np.float64)
    dx = np.maximum(np.maximum(lower[0] - points[:, 0], points[:, 0] - upper[0]), 0.0)
    dy = np.maximum(np.maximum(lower[1] - points[:, 1], points[:, 1] - upper[1]), 0.0)
    return np.hypot(dx, dy)


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        key: float(np.percentile(values, q))
        for key, q in (("p05", 5), ("p50", 50), ("p95", 95))
    }


def summarize(
    xyz: np.ndarray,
    owner: np.ndarray,
    boxes,
    *,
    subset_name: str,
) -> dict[str, Any]:
    total = int(len(owner))
    inside = owner >= 0
    outside = ~inside
    beyond = horizontal_distance_beyond_union(xyz, boxes)
    bands = []
    for low, high in zip(DISTANCE_BANDS_M[:-1], DISTANCE_BANDS_M[1:]):
        in_band = outside & (beyond >= low) & (beyond < high)
        bands.append(
            {
                "beyond_union_xy_m": [low, None if np.isinf(high) else high],
                "count": int(np.count_nonzero(in_band)),
            }
        )
    # Outside points that are still within the union's xy footprint are above
    # or below the boxes (sky band, below-ground) rather than beside them.
    over_footprint = outside & (beyond <= 0.0)
    above = over_footprint & (xyz[:, 2] > np.max([b[1][2] for _, _, b in boxes]))
    below = over_footprint & (xyz[:, 2] < np.min([b[0][2] for _, _, b in boxes]))
    return {
        "subset": subset_name,
        "count": total,
        "inside_union_count": int(np.count_nonzero(inside)),
        "inside_union_fraction": float(np.count_nonzero(inside) / max(1, total)),
        "outside_union_count": int(np.count_nonzero(outside)),
        "outside_union_fraction": float(np.count_nonzero(outside) / max(1, total)),
        "per_tile_count": {
            name: int(np.count_nonzero(owner == tile_id)) for tile_id, name, _ in boxes
        },
        "outside_by_horizontal_distance": bands,
        "outside_over_footprint_above_boxes_count": int(np.count_nonzero(above)),
        "outside_over_footprint_below_boxes_count": int(np.count_nonzero(below)),
        "outside_z_m": _percentiles(xyz[outside, 2]),
        "outside_beyond_union_xy_m": _percentiles(beyond[outside]),
        "inside_z_m": _percentiles(xyz[inside, 2]),
    }


def render_topdown(
    xyz: np.ndarray,
    owner: np.ndarray,
    boxes,
    png: Path,
    *,
    cell_m: float,
    title: str,
    highlight: np.ndarray | None = None,
) -> None:
    """Top-down occupancy raster: blue = inside the boxes, red = outside.

    With ``highlight`` (a boolean mask) red marks the highlighted rows
    instead - used for LiDAR-unsupported content, since for this scene the
    boxes contain everything and inside/outside alone draws a blue map.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lower = np.min([box[0] for _, _, box in boxes], axis=0)
    upper = np.max([box[1] for _, _, box in boxes], axis=0)
    pad = 20.0
    x_edges = np.arange(min(lower[0] - pad, xyz[:, 0].min()), max(upper[0] + pad, xyz[:, 0].max()) + cell_m, cell_m)
    y_edges = np.arange(min(lower[1] - pad, xyz[:, 1].min()), max(upper[1] + pad, xyz[:, 1].max()) + cell_m, cell_m)
    inside = (owner >= 0) if highlight is None else ~np.asarray(highlight, dtype=bool)
    h_in, _, _ = np.histogram2d(xyz[inside, 0], xyz[inside, 1], bins=(x_edges, y_edges))
    h_out, _, _ = np.histogram2d(xyz[~inside, 0], xyz[~inside, 1], bins=(x_edges, y_edges))
    rgb = np.ones(h_in.shape + (3,), dtype=np.float64)
    strength_in = np.log1p(h_in) / max(1e-9, np.log1p(h_in.max()))
    strength_out = np.log1p(h_out) / max(1e-9, np.log1p(max(h_out.max(), 1.0)))
    # inside -> blue ramp, outside -> red ramp; both drawn, red on top where present
    rgb[..., 0] -= 0.85 * strength_in
    rgb[..., 1] -= 0.55 * strength_in
    rgb[..., 1] -= 0.85 * strength_out
    rgb[..., 2] -= 0.85 * strength_out
    rgb = np.clip(rgb, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=110)
    ax.imshow(
        np.transpose(rgb, (1, 0, 2)),
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        interpolation="nearest",
    )
    for tile_id, name, box in boxes:
        ax.add_patch(
            plt.Rectangle(
                (box[0][0], box[0][1]),
                box[1][0] - box[0][0],
                box[1][1] - box[0][1],
                fill=False,
                edgecolor="black",
                linewidth=1.2,
            )
        )
        ax.text(box[0][0] + 1.0, box[1][1] - 3.0, name, fontsize=9)
    ax.set_xlabel("x (m, s1_local)")
    ax.set_ylabel("y (m, s1_local)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png)
    plt.close(fig)


def lidar_support_mask(
    xyz: np.ndarray, anchors: np.ndarray, *, max_distance_m: float
) -> np.ndarray:
    """True where a LiDAR point lies within ``max_distance_m`` (bounded kd-tree query)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(anchors, dtype=np.float64))
    distance, _ = tree.query(
        np.asarray(xyz, dtype=np.float64), k=1,
        distance_upper_bound=float(max_distance_m), workers=-1,
    )
    return np.isfinite(distance)


def read_xyz(path: Path) -> np.ndarray:
    records = read_ply_records(Path(path))
    return np.stack(
        [np.asarray(records[axis], dtype=np.float64) for axis in ("x", "y", "z")], axis=1
    )


def audit(
    *,
    ply: Path,
    tile_inputs: Path,
    transform_json: Path | None,
    box_kind: str,
    opacity_floor: float,
    label: str,
    anchor_ply: Path | None = None,
    anchor_distance_m: float = 0.5,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None]:
    """Classify the PLY's points; returns (report, xyz, owner, supported_or_None)."""
    manifest = json.loads(Path(tile_inputs).read_text(encoding="utf-8"))
    manifest_sha = verify_tile_inputs_manifest(manifest)
    boxes = tile_boxes(manifest, box_kind)
    records = read_ply_records(Path(ply))
    xyz = np.stack(
        [np.asarray(records[axis], dtype=np.float64) for axis in ("x", "y", "z")], axis=1
    )
    transform = load_transform(transform_json)
    xyz = apply_transform(xyz, transform)
    finite = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite]
    owner = classify_points(xyz, boxes)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gaussian_model_tile_coverage_v1",
        "label": label,
        "ply": str(Path(ply).resolve()),
        "vertex_count": int(len(records)),
        "non_finite_dropped": int(np.count_nonzero(~finite)),
        "transform_json": None if transform_json is None else str(Path(transform_json).resolve()),
        "transform_applied": transform is not None,
        "tile_inputs_manifest": str(Path(tile_inputs).resolve()),
        "tile_inputs_manifest_sha256": manifest_sha,
        "box_kind": box_kind,
        "boxes": {name: box.tolist() for _, name, box in boxes},
        "bounds_m": {"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()},
        "subsets": [summarize(xyz, owner, boxes, subset_name="all")],
    }
    alive = None
    if "opacity" in records.dtype.names:
        logits = np.asarray(records["opacity"], dtype=np.float64)[finite]
        alive = 1.0 / (1.0 + np.exp(-logits)) >= opacity_floor
        report["opacity_floor"] = float(opacity_floor)
        report["subsets"].append(
            summarize(xyz[alive], owner[alive], boxes, subset_name=f"opacity_ge_{opacity_floor:g}")
        )
    supported = None
    if anchor_ply is not None:
        # The Tile boxes partition a padded LAS bounding box, so "outside the
        # boxes" is empty for any model of this scene. What nobody in a
        # LiDAR-initialised Tile pipeline owns is content with no LiDAR
        # return near it: canopies, far vegetation, whatever lies beyond the
        # scan. Measured as distance to the nearest LiDAR point.
        anchors = read_xyz(anchor_ply)
        supported = lidar_support_mask(xyz, anchors, max_distance_m=anchor_distance_m)
        report["anchor"] = {
            "ply": str(Path(anchor_ply).resolve()),
            "point_count": int(len(anchors)),
            "bounds_m": {"min": anchors.min(axis=0).tolist(), "max": anchors.max(axis=0).tolist()},
            "max_distance_m": float(anchor_distance_m),
        }
        far = ~supported
        report["subsets"].append(
            summarize(xyz[far], owner[far], boxes, subset_name=f"lidar_unsupported_gt_{anchor_distance_m:g}m")
        )
        if alive is not None:
            both = far & alive
            report["subsets"].append(
                summarize(
                    xyz[both], owner[both], boxes,
                    subset_name=f"lidar_unsupported_gt_{anchor_distance_m:g}m_opacity_ge_{opacity_floor:g}",
                )
            )
        # Where the unsupported content sits relative to the LAS extent: past
        # the scanned footprint (far field) or over it (canopy, sky band).
        lower, upper = anchors.min(axis=0), anchors.max(axis=0)
        beyond_las_xy = np.any((xyz[:, :2] < lower[:2]) | (xyz[:, :2] > upper[:2]), axis=1)
        report["lidar_unsupported_location"] = {
            "beyond_las_xy_extent_count": int(np.count_nonzero(far & beyond_las_xy)),
            "over_las_footprint_count": int(np.count_nonzero(far & ~beyond_las_xy)),
            "over_footprint_z_m": _percentiles(xyz[far & ~beyond_las_xy, 2]),
        }
    return report, xyz, owner, supported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--tile-inputs", type=Path, required=True)
    parser.add_argument("--transform-json", type=Path)
    parser.add_argument("--box-kind", choices=BOX_KINDS, default="training_and_export_box")
    parser.add_argument("--opacity-floor", type=float, default=0.05)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--cell-m", type=float, default=0.25)
    parser.add_argument("--anchor-ply", type=Path,
                        help="LiDAR point cloud PLY; adds the lidar_unsupported subset (distance > --anchor-distance-m)")
    parser.add_argument("--anchor-distance-m", type=float, default=0.5)
    parser.add_argument("--png-color", choices=("inside", "lidar"), default="inside",
                        help="what red marks in the PNG: outside the boxes, or LiDAR-unsupported rows (needs --anchor-ply)")
    args = parser.parse_args()
    if args.png_color == "lidar" and args.anchor_ply is None:
        parser.error("--png-color lidar needs --anchor-ply")

    report, xyz, owner, supported = audit(
        ply=args.ply,
        tile_inputs=args.tile_inputs,
        transform_json=args.transform_json,
        box_kind=args.box_kind,
        opacity_floor=args.opacity_floor,
        label=args.label or args.ply.stem,
        anchor_ply=args.anchor_ply,
        anchor_distance_m=args.anchor_distance_m,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.png is not None:
        first = report["subsets"][0]
        if args.png_color == "lidar":
            far = ~supported
            title = (
                f"{report['label']}: {int(np.count_nonzero(far)):,} of {first['count']:,} "
                f"points > {args.anchor_distance_m:g} m from any LiDAR point "
                f"({100.0 * np.count_nonzero(far) / max(1, first['count']):.1f}%) - red = LiDAR-unsupported"
            )
            highlight = far
        else:
            title = (
                f"{report['label']}: {first['outside_union_count']:,} of "
                f"{first['count']:,} points outside the {args.box_kind} union "
                f"({100.0 * first['outside_union_fraction']:.1f}%) - red = outside"
            )
            highlight = None
        render_topdown(
            xyz, owner, tile_boxes(
                json.loads(args.tile_inputs.read_text(encoding="utf-8")), args.box_kind
            ),
            args.png,
            cell_m=args.cell_m,
            title=title,
            highlight=highlight,
        )
        report["png"] = str(args.png.resolve())
        args.output_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for subset in report["subsets"]:
        print(
            f"{report['label']} [{subset['subset']}]: {subset['count']:,} points, "
            f"outside {subset['outside_union_count']:,} "
            f"({100.0 * subset['outside_union_fraction']:.2f}%), "
            f"per tile {subset['per_tile_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
