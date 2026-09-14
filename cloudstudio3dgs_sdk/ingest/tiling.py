"""Deterministic tile boxes for an arbitrary scene.

house0305's four tiles came from the projected-pixel kd planner
(``cloudstudio_3dgs.pipeline.adaptive_tiling``), which needs a full Face4
observation table and a measured VRAM budget before it can cut anything.  That
is the right planner once the caches exist, and the wrong one for ingestion:
tiling has to be decided *before* the per-tile caches are built.

The rule implemented here
-------------------------
1. Take the LiDAR bounding box.  Expand it by ``scene_padding_fraction`` per
   side (0.2, the same padding the production planner uses as its root box) so
   points at the rim keep a halo.
2. Pick the split axis: the longer of the two horizontal axes (X or Y) of the
   **unpadded** point box, or an explicit ``axis``.  Never Z - slabs in Z would
   cut floors from ceilings.
3. Cut that axis into ``tile_count`` slabs of roughly equal *point count*
   (not equal length), at the point-count quantiles.  Quantiles come from a
   fixed-width histogram (``histogram_bins``, default 4096) with linear
   interpolation inside the straddling bin, and the cut is rounded to
   ``cut_decimals``; the same points therefore always give the same cuts,
   independent of chunk order or platform.
4. Grow each core box into ``training_and_export_box`` by
   ``overlap_margin_m`` (absolute metres) when given, else by
   ``halo_fraction_per_side`` of the box extent - 0.002, matching
   ``AdaptiveTilingConfig.spatial_halo_fraction_per_side``.

Core boxes stay a gap-free, non-overlapping partition of the padded root box,
which is what ``cloudstudio_3dgs.training.tile_ownership`` requires of any tile
set it is asked to assign core ownership for.  Only the export boxes overlap.

The emitted plan is the tile-plan schema the trainer already validates
(``adaptive_projected_pixel_kd_xy_v1``), so ``materialize_lidar_tile_inputs``
consumes it unchanged.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.adaptive_tiling import (
    GIB,
    TILE_PLAN_KIND,
    TILE_PLAN_SCHEMA_VERSION,
    AdaptiveTilingConfig,
    AxisAlignedBox,
    ProjectedObservationTable,
    _point_mask,
    _rectangle_summary,
    bytes_per_pixel,
)

from .errors import IngestError

AXIS_NAMES = ("x", "y", "z")
TILING_RULE_VERSION = "slab_equal_point_count_v1"


class TilingError(IngestError):
    """The tiling rule cannot produce a usable partition for this scene."""


@dataclass(frozen=True)
class TilingRule:
    """Every knob of the slab rule, all of them recorded into the plan."""

    tile_count: int = 4
    axis: str = "auto"
    scene_padding_fraction: float = 0.2
    halo_fraction_per_side: float = 0.002
    overlap_margin_m: float | None = None
    histogram_bins: int = 4096
    cut_decimals: int = 6
    minimum_slab_extent_m: float = 1.0
    resolution_level: int = 1

    def validate(self) -> None:
        if self.tile_count < 1:
            raise TilingError("tile_count must be at least 1")
        if self.axis not in ("auto", "x", "y"):
            raise TilingError("axis must be 'auto', 'x' or 'y' (never 'z')")
        if not 0.0 <= self.scene_padding_fraction < 1.0:
            raise TilingError("scene_padding_fraction must be within [0, 1)")
        if not 0.0 <= self.halo_fraction_per_side < 0.5:
            raise TilingError("halo_fraction_per_side must be within [0, 0.5)")
        if self.overlap_margin_m is not None and self.overlap_margin_m < 0.0:
            raise TilingError("overlap_margin_m must be non-negative")
        if self.histogram_bins < 16:
            raise TilingError("histogram_bins must be at least 16")
        if self.cut_decimals < 0:
            raise TilingError("cut_decimals must be non-negative")
        if self.minimum_slab_extent_m <= 0.0:
            raise TilingError("minimum_slab_extent_m must be positive")
        bytes_per_pixel(self.resolution_level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": TILING_RULE_VERSION,
            "tile_count": int(self.tile_count),
            "axis": self.axis,
            "scene_padding_fraction": float(self.scene_padding_fraction),
            "halo_fraction_per_side": float(self.halo_fraction_per_side),
            "overlap_margin_m": (
                None if self.overlap_margin_m is None else float(self.overlap_margin_m)
            ),
            "histogram_bins": int(self.histogram_bins),
            "cut_decimals": int(self.cut_decimals),
            "minimum_slab_extent_m": float(self.minimum_slab_extent_m),
            "resolution_level": int(self.resolution_level),
        }


@dataclass(frozen=True)
class AxisHistogram:
    """Per-axis occupancy of the point cloud; the only statistic the rule uses."""

    minimum: np.ndarray
    maximum: np.ndarray
    counts: np.ndarray  # [3, bins]
    point_count: int

    @property
    def bins(self) -> int:
        return int(self.counts.shape[1])

    @property
    def extent(self) -> np.ndarray:
        return self.maximum - self.minimum

    def edges(self, axis: int) -> np.ndarray:
        return self.minimum[axis] + (
            np.arange(self.bins + 1, dtype=np.float64) / self.bins
        ) * (self.maximum[axis] - self.minimum[axis])


def histogram_from_points(points: np.ndarray, *, bins: int = 4096) -> AxisHistogram:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise TilingError("points must be a non-empty [N, 3] array")
    if not np.all(np.isfinite(points)):
        raise TilingError("points contain non-finite coordinates")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    counts = np.zeros((3, bins), dtype=np.int64)
    _accumulate(counts, points, minimum, maximum, bins)
    return AxisHistogram(minimum, maximum, counts, int(len(points)))


def histogram_from_las(
    path: Path, *, bins: int = 4096, chunk_size: int = 2_000_000
) -> AxisHistogram:
    """Stream a LAS/LAZ once; never loads the whole cloud."""

    import laspy

    counts = np.zeros((3, bins), dtype=np.int64)
    total = 0
    with laspy.open(Path(path)) as reader:
        minimum = np.asarray(reader.header.mins, dtype=np.float64)
        maximum = np.asarray(reader.header.maxs, dtype=np.float64)
        if np.any(maximum <= minimum):
            raise TilingError(f"LAS header reports a degenerate bounding box: {path}")
        for chunk in reader.chunk_iterator(chunk_size):
            points = np.column_stack([chunk.x, chunk.y, chunk.z]).astype(np.float64)
            total += len(points)
            _accumulate(counts, points, minimum, maximum, bins)
    if total == 0:
        raise TilingError(f"point cloud is empty: {path}")
    return AxisHistogram(minimum, maximum, counts, total)


def _accumulate(
    counts: np.ndarray,
    points: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    bins: int,
) -> None:
    for axis in range(3):
        span = max(float(maximum[axis] - minimum[axis]), 1e-9)
        index = np.clip(
            ((points[:, axis] - minimum[axis]) / span * bins).astype(np.int64), 0, bins - 1
        )
        counts[axis] += np.bincount(index, minlength=bins)


def choose_axis(histogram: AxisHistogram, rule: TilingRule) -> int:
    if rule.axis != "auto":
        return AXIS_NAMES.index(rule.axis)
    horizontal = histogram.extent[:2]
    # Ties go to X so the choice never depends on floating-point noise.
    return 0 if horizontal[0] >= horizontal[1] else 1


def quantile_cuts(histogram: AxisHistogram, axis: int, rule: TilingRule) -> list[float]:
    """Point-count quantile positions along ``axis``, deterministic by construction."""

    counts = histogram.counts[axis]
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])
    if total <= 0:
        raise TilingError("the point histogram is empty on the split axis")
    edges = histogram.edges(axis)
    cuts: list[float] = []
    for index in range(1, rule.tile_count):
        target = total * index / rule.tile_count
        bin_index = int(np.searchsorted(cumulative, target, side="left"))
        bin_index = min(bin_index, histogram.bins - 1)
        previous = int(cumulative[bin_index - 1]) if bin_index > 0 else 0
        occupancy = int(counts[bin_index])
        fraction = (target - previous) / occupancy if occupancy > 0 else 0.0
        fraction = min(max(fraction, 0.0), 1.0)
        position = edges[bin_index] + fraction * (edges[bin_index + 1] - edges[bin_index])
        cuts.append(round(float(position), rule.cut_decimals))
    return cuts


@dataclass(frozen=True)
class SlabPlan:
    """Boxes plus the evidence for why they are where they are."""

    rule: TilingRule
    axis: int
    root_box: AxisAlignedBox
    cuts: tuple[float, ...]
    core_boxes: tuple[AxisAlignedBox, ...]
    export_boxes: tuple[AxisAlignedBox, ...]
    histogram_counts: tuple[int, ...]

    @property
    def axis_name(self) -> str:
        return AXIS_NAMES[self.axis]

    def balance(self) -> float:
        """max/min of the per-slab histogram point counts; 1.0 is perfect."""

        low = max(min(self.histogram_counts), 1)
        return max(self.histogram_counts) / low

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.to_dict(),
            "split_axis": self.axis_name,
            "root_box": self.root_box.to_list(),
            "cuts_m": list(self.cuts),
            "histogram_point_counts": list(self.histogram_counts),
            "histogram_balance_max_over_min": self.balance(),
        }


def slab_split(histogram: AxisHistogram, rule: TilingRule = TilingRule()) -> SlabPlan:
    rule.validate()
    axis = choose_axis(histogram, rule)
    extent = histogram.extent
    if np.any(extent <= 0.0):
        raise TilingError("the point cloud does not span a 3D box")
    padding = extent * rule.scene_padding_fraction
    root = AxisAlignedBox(histogram.minimum - padding, histogram.maximum + padding)

    cuts = quantile_cuts(histogram, axis, rule)
    bounds = [float(root.minimum[axis]), *cuts, float(root.maximum[axis])]
    for index in range(len(bounds) - 1):
        if bounds[index + 1] - bounds[index] < rule.minimum_slab_extent_m:
            raise TilingError(
                f"slab {index} would be "
                f"{bounds[index + 1] - bounds[index]:.3f} m along "
                f"{AXIS_NAMES[axis]}, below minimum_slab_extent_m="
                f"{rule.minimum_slab_extent_m}; use fewer tiles"
            )

    core_boxes: list[AxisAlignedBox] = []
    export_boxes: list[AxisAlignedBox] = []
    for index in range(rule.tile_count):
        low = root.minimum.copy()
        high = root.maximum.copy()
        low[axis] = bounds[index]
        high[axis] = bounds[index + 1]
        core = AxisAlignedBox(low, high)
        if rule.overlap_margin_m is not None:
            margin = float(rule.overlap_margin_m)
            export = AxisAlignedBox(core.minimum - margin, core.maximum + margin)
        else:
            export = core.expanded(rule.halo_fraction_per_side)
        core_boxes.append(core)
        export_boxes.append(export)

    counts = _histogram_slab_counts(histogram, axis, bounds)
    return SlabPlan(
        rule=rule,
        axis=axis,
        root_box=root,
        cuts=tuple(cuts),
        core_boxes=tuple(core_boxes),
        export_boxes=tuple(export_boxes),
        histogram_counts=tuple(counts),
    )


def _histogram_slab_counts(
    histogram: AxisHistogram, axis: int, bounds: Sequence[float]
) -> list[int]:
    edges = histogram.edges(axis)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = histogram.counts[axis]
    rows: list[int] = []
    for index in range(len(bounds) - 1):
        inside = (centres >= bounds[index]) & (centres < bounds[index + 1])
        if index == len(bounds) - 2:
            inside |= centres >= bounds[index + 1]
        rows.append(int(counts[inside].sum()))
    return rows


def exact_slab_counts(points: np.ndarray, boxes: Iterable[AxisAlignedBox]) -> list[int]:
    """Exact per-box point counts; the histogram counts are bin-quantized."""

    points = np.asarray(points, dtype=np.float64)
    return [
        int(np.count_nonzero(np.all((points >= box.minimum) & (points <= box.maximum), axis=1)))
        for box in boxes
    ]


def build_slab_tile_plan(
    slab: SlabPlan,
    *,
    observations: ProjectedObservationTable | None = None,
    view_ids: Sequence[str] | None = None,
    source_bindings: Mapping[str, str] | None = None,
    point_cloud_sha256: str | None = None,
    tiling_config: AdaptiveTilingConfig | None = None,
) -> dict[str, Any]:
    """Emit the signed tile plan the trainer's own validator accepts.

    ``observations`` is the projected LiDAR/Face4 table
    (:mod:`cloudstudio_3dgs.pipeline.lidar_face4_observations`).  Without it the
    plan carries boxes but no view rectangles, and is explicitly marked
    ``views_source: "deferred"`` - ``materialize_lidar_tile_inputs`` would write
    tiles no view can train, so the cache planner refuses to run it in that
    state rather than producing an empty-crop manifest.
    """

    config = tiling_config or replace(
        AdaptiveTilingConfig(),
        resolution_level=slab.rule.resolution_level,
        spatial_halo_fraction_per_side=slab.rule.halo_fraction_per_side,
    )
    config.validate()
    bpp = bytes_per_pixel(config.resolution_level)

    tiles: list[dict[str, Any]] = []
    for index, (core, export) in enumerate(zip(slab.core_boxes, slab.export_boxes)):
        anchors = 0
        pixels = 0
        view_count = 0
        rectangles: list[dict[str, Any]] = []
        if observations is not None:
            selected = _point_mask(observations.points, core)
            anchors = int(np.count_nonzero(selected))
            pixels, view_count, rectangles = _rectangle_summary(
                observations, selected, config, include_rectangles=True
            )
            if view_ids is not None:
                for rectangle in rectangles:
                    rectangle["sample_id"] = str(view_ids[int(rectangle["image_index"])])
        tiles.append(
            {
                "tile_id": index,
                "name": f"Tile_{index}",
                "core_box": core.to_list(),
                "training_and_export_box": export.to_list(),
                "anchor_count": anchors,
                "pixel_load": pixels,
                "valid_view_count": view_count,
                "estimated_memory_gib": pixels * bpp / GIB,
                "low_support_discarded": False,
                "histogram_point_count": int(slab.histogram_counts[index]),
                "views": rectangles,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": TILE_PLAN_SCHEMA_VERSION,
        "kind": TILE_PLAN_KIND,
        "evidence_boundary": (
            "Boxes come from the SDK slab rule (equal point count along the "
            "longest horizontal axis), not from the projected-pixel kd planner; "
            "the schema is shared so the trainer consumes both unchanged."
        ),
        "planner": TILING_RULE_VERSION,
        "views_source": "deferred" if observations is None else "projected_observations",
        "source_bindings": dict(sorted((source_bindings or {}).items())),
        "input": {
            "histogram_point_count": int(sum(slab.histogram_counts)),
            "point_cloud_sha256": point_cloud_sha256,
            "root_box": slab.root_box.to_list(),
            "root_source": (
                "LiDAR bounding box expanded by "
                f"{slab.rule.scene_padding_fraction} per side"
            ),
            "split_axis": slab.axis_name,
            "cuts_m": list(slab.cuts),
        },
        "config": {**slab.rule.to_dict(), "bytes_per_pixel": bpp},
        "execution_contract": {
            "strict_serial_tiles": True,
            "multiple_tiles_resident_on_cuda": False,
            "empty_cuda_cache_even_steps": True,
            "empty_cuda_cache_after_each_tile": True,
            "halo_merge": "retain_full_tile_outputs_without_core_deduplication",
        },
        "tree": _slab_tree(slab, config),
        "leaf_count": len(tiles),
        "retained_tile_count": len(tiles),
        "tiles": tiles,
    }
    payload["tile_plan_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _slab_tree(slab: SlabPlan, config: AdaptiveTilingConfig) -> dict[str, Any]:
    """A left-leaning cut tree describing the slab boundaries, in plan shape."""

    def node(depth: int, first: int, last: int, box: AxisAlignedBox) -> dict[str, Any]:
        row: dict[str, Any] = {
            "depth": depth,
            "core_box": box.to_list(),
            "export_box": box.expanded(config.spatial_halo_fraction_per_side).to_list(),
            "anchor_count": int(sum(slab.histogram_counts[first : last + 1])),
            "valid_view_count": 0,
            "pixel_load": 0,
            "estimated_memory_gib": 0.0,
            "split_comparison_memory_gib": 0.0,
            "low_support": False,
        }
        if first == last:
            return row
        cut = slab.cuts[first]
        left_box, right_box = box.split(slab.axis, cut)
        row["split"] = {
            "axis": slab.axis_name,
            "position": cut,
            "criterion": "equal_point_count_quantile",
        }
        row["children"] = [
            node(depth + 1, first, first, left_box),
            node(depth + 1, first + 1, last, right_box),
        ]
        return row

    return node(0, 0, len(slab.core_boxes) - 1, slab.root_box)


def compare_tile_boxes(
    plan_tiles: Sequence[Mapping[str, Any]],
    reference_tiles: Sequence[Mapping[str, Any]],
    *,
    key: str = "training_and_export_box",
) -> dict[str, Any]:
    """Volume/extent comparison used by the house0305 regression report."""

    def volume(box: Sequence[Sequence[float]]) -> float:
        low = np.asarray(box[0], dtype=np.float64)
        high = np.asarray(box[1], dtype=np.float64)
        return float(np.prod(high - low))

    plan_volumes = [volume(tile[key]) for tile in plan_tiles]
    reference_volumes = [volume(tile[key]) for tile in reference_tiles]
    return {
        "plan_tile_count": len(plan_tiles),
        "reference_tile_count": len(reference_tiles),
        "plan_volume_m3": plan_volumes,
        "reference_volume_m3": reference_volumes,
        "plan_total_volume_m3": float(sum(plan_volumes)),
        "reference_total_volume_m3": float(sum(reference_volumes)),
    }


def signed_plan_copy(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Re-sign a plan after a caller mutated it (e.g. added sample ids)."""

    payload = copy.deepcopy(dict(plan))
    payload.pop("tile_plan_manifest_sha256", None)
    payload["tile_plan_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


__all__ = [
    "AXIS_NAMES",
    "AxisHistogram",
    "SlabPlan",
    "TILING_RULE_VERSION",
    "TilingError",
    "TilingRule",
    "build_slab_tile_plan",
    "choose_axis",
    "compare_tile_boxes",
    "exact_slab_counts",
    "histogram_from_las",
    "histogram_from_points",
    "quantile_cuts",
    "signed_plan_copy",
    "slab_split",
]
