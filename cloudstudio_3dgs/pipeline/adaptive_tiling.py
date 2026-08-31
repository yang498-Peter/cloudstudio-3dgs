"""Production, vendor-independent adaptive spatial tiling contracts.

The planner implements the evidence-backed MipMap ``divide_mode=2`` behaviour
without importing the proprietary protobuf decoder.  Callers must supply an
explicit point/observation table, so every generated plan can be bound to the
accepted CloudStudio inputs and verified before tile training starts.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


GIB = 1024**3
TILE_PLAN_SCHEMA_VERSION = 1
TILE_PLAN_KIND = "adaptive_projected_pixel_kd_xy_v1"


@dataclass(frozen=True)
class AdaptiveTilingConfig:
    resolution_level: int = 1
    candidate_positions: int = 64
    minimum_image_rectangle_pixels: int = 256
    image_halo_pixels_per_side: int = 128
    spatial_halo_fraction_per_side: float = 0.002
    maximum_depth: int = 10
    minimum_anchor_count: int = 100
    minimum_pixel_load: int = 100_000
    minimum_eligible_axis_ratio: float = 0.2
    split_estimate_multiplier: float = 0.8
    near_axis_cost_fraction: float = 0.1

    def validate(self) -> None:
        bytes_per_pixel(self.resolution_level)
        if self.candidate_positions < 3:
            raise ValueError("adaptive tiling requires at least three cut candidates")
        if self.minimum_image_rectangle_pixels <= 0:
            raise ValueError("minimum image rectangle must be positive")
        if self.image_halo_pixels_per_side < 0:
            raise ValueError("image halo must be non-negative")
        if not 0.0 <= self.spatial_halo_fraction_per_side < 0.5:
            raise ValueError("spatial halo fraction must be within [0, 0.5)")
        if self.maximum_depth < 0:
            raise ValueError("maximum depth must be non-negative")
        if self.minimum_anchor_count < 0 or self.minimum_pixel_load < 0:
            raise ValueError("minimum support thresholds must be non-negative")
        if not 0.0 < self.minimum_eligible_axis_ratio <= 1.0:
            raise ValueError("minimum eligible axis ratio must be within (0, 1]")
        if not 0.0 < self.split_estimate_multiplier <= 1.0:
            raise ValueError("split estimate multiplier must be within (0, 1]")
        if not 0.0 <= self.near_axis_cost_fraction < 1.0:
            raise ValueError("near-axis cost fraction must be within [0, 1)")


@dataclass(frozen=True)
class AxisAlignedBox:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("tile boxes must contain XYZ minimum and maximum")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("tile boxes must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("tile box maximum must exceed minimum on every axis")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def extent(self) -> np.ndarray:
        return self.maximum - self.minimum

    def split(self, axis: int, value: float) -> tuple["AxisAlignedBox", "AxisAlignedBox"]:
        left_maximum = self.maximum.copy()
        right_minimum = self.minimum.copy()
        left_maximum[axis] = value
        right_minimum[axis] = value
        return (
            AxisAlignedBox(self.minimum.copy(), left_maximum),
            AxisAlignedBox(right_minimum, self.maximum.copy()),
        )

    def expanded(self, fraction: float) -> "AxisAlignedBox":
        padding = self.extent * fraction
        return AxisAlignedBox(self.minimum - padding, self.maximum + padding)

    def to_list(self) -> list[list[float]]:
        return [self.minimum.tolist(), self.maximum.tolist()]


@dataclass(frozen=True)
class ProjectedObservationTable:
    points: np.ndarray
    observation_xy: np.ndarray
    observation_image: np.ndarray
    observation_point: np.ndarray
    image_sizes: np.ndarray

    def validated(self) -> "ProjectedObservationTable":
        points = np.asarray(self.points, dtype=np.float64)
        observation_xy = np.asarray(self.observation_xy, dtype=np.float64)
        observation_image = np.asarray(self.observation_image, dtype=np.int64)
        observation_point = np.asarray(self.observation_point, dtype=np.int64)
        image_sizes = np.asarray(self.image_sizes, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points must have non-empty shape [N, 3]")
        if not np.all(np.isfinite(points)):
            raise ValueError("points contain non-finite coordinates")
        if observation_xy.ndim != 2 or observation_xy.shape[1] != 2:
            raise ValueError("observation_xy must have shape [M, 2]")
        if not np.all(np.isfinite(observation_xy)):
            raise ValueError("observations contain non-finite pixels")
        if len(observation_image) != len(observation_xy) or len(observation_point) != len(observation_xy):
            raise ValueError("observation arrays have different row counts")
        if image_sizes.ndim != 2 or image_sizes.shape[1] != 2 or len(image_sizes) == 0:
            raise ValueError("image_sizes must have non-empty shape [I, 2]")
        if np.any(image_sizes <= 0):
            raise ValueError("image sizes must be positive")
        if len(observation_xy):
            if observation_image.min() < 0 or observation_image.max() >= len(image_sizes):
                raise ValueError("observation image index is out of range")
            if observation_point.min() < 0 or observation_point.max() >= len(points):
                raise ValueError("observation point index is out of range")
        return ProjectedObservationTable(
            points, observation_xy, observation_image, observation_point, image_sizes
        )

    @property
    def support(self) -> np.ndarray:
        return np.bincount(self.observation_point, minlength=len(self.points))

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in (
            ("points", self.points),
            ("observation_xy", self.observation_xy),
            ("observation_image", self.observation_image),
            ("observation_point", self.observation_point),
            ("image_sizes", self.image_sizes),
        ):
            contiguous = np.ascontiguousarray(value)
            digest.update(name.encode("ascii"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


@dataclass
class _Node:
    box: AxisAlignedBox
    depth: int
    pixels: int
    anchors: int
    image_count: int
    low_support: bool
    split: dict[str, Any] | None = None
    children: tuple["_Node", "_Node"] | None = None


def bytes_per_pixel(resolution_level: int) -> float:
    if resolution_level in (0, 1):
        return 5.5
    if resolution_level == 2:
        return 1.375
    if resolution_level == 3:
        return 1.1875
    raise ValueError(f"unsupported resolution level: {resolution_level}")


@dataclass(frozen=True)
class GaussianResidencyModel:
    """Predict a tile's peak VRAM from the gaussians that stay resident.

    Views do not stay resident: the face dataset opens one sample per item and
    releases it, so the sum of a tile's view pixels never occupies memory at
    once. What does occupy memory is the gaussian population - parameters,
    Adam moments and gradients - plus a fixed rasterizer workspace.

    Defaults come from this machine's own full-resolution runs:

        0.971M gaussians ->  0.892 GiB
        7.036M gaussians ->  2.903 GiB
       18.758M gaussians -> 10.011 GiB

    The slope is fitted at the largest point, not across all three. The two
    small runs are dominated by the fixed workspace, and extrapolating their
    apparent slope understated an 18.76M scene by 47 percent. Peak arrives
    during refinement, where clone and split briefly hold old and new tensors
    together, so the multiplier applies on top of the steady-state slope.
    """

    base_gib: float = 0.570
    gib_per_million: float = 0.5033
    # Measured 2026-08-31 on the full-density adaptive run: 18,757,869
    # gaussians peaked at 12.71 GiB, i.e. 0.647 GiB per million including the
    # lifecycle transient, against the 1.51 this multiplier used to predict.
    # The 3.0 was set when split was expected to hold old and new tensors
    # together for a large fraction of the population; under the recovered
    # 0.2 m split boundary essentially nothing splits on metric-scale
    # gaussians, so the transient is a clone-sized fraction, not a doubling.
    # 1.4 leaves ~9% over the measurement. Raise it again for any arm that
    # restores a split boundary small enough to fire.
    lifecycle_multiplier: float = 1.4
    growth_ratio: float = 3.2

    def validate(self) -> None:
        if self.base_gib < 0.0:
            raise ValueError("base_gib must be non-negative")
        if self.gib_per_million <= 0.0:
            raise ValueError("gib_per_million must be positive")
        if self.lifecycle_multiplier < 1.0:
            raise ValueError("lifecycle_multiplier must be at least one")
        if self.growth_ratio < 1.0:
            raise ValueError("growth_ratio must be at least one")

    @property
    def gib_per_million_peak(self) -> float:
        return self.lifecycle_multiplier * self.gib_per_million

    def peak_gib(self, anchor_count: int) -> float:
        """Predicted peak for a tile seeded with ``anchor_count`` anchors."""
        projected_millions = anchor_count * self.growth_ratio / 1e6
        return self.base_gib + self.gib_per_million_peak * projected_millions

    def anchor_capacity(self, budget_gib: float) -> int:
        """Largest anchor count whose projected growth still fits the budget."""
        usable = float(budget_gib) - self.base_gib
        if usable <= 0.0:
            return 0
        millions = usable / self.gib_per_million_peak
        return int(millions * 1e6 / self.growth_ratio)

    def gaussian_cap(self, budget_gib: float) -> int:
        """Absolute gaussian cap to hand the trainer so the budget is binding."""
        usable = float(budget_gib) - self.base_gib
        if usable <= 0.0:
            return 0
        return int(usable / self.gib_per_million_peak * 1e6)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "base_gib": self.base_gib,
            "gib_per_million": self.gib_per_million,
            "lifecycle_multiplier": self.lifecycle_multiplier,
            "growth_ratio": self.growth_ratio,
            "gib_per_million_peak": self.gib_per_million_peak,
            "calibration": "full-resolution runs on this machine, frozen topology",
        }


def startup_budget_gib(
    *,
    gpu0_available_gib: float | None,
    system_available_gib: float,
    ceiling_gib: float = 12.0,
) -> dict[str, Any]:
    """Return the recovered fail-safe startup budget and its provenance."""

    values = [float(system_available_gib), float(ceiling_gib)]
    if gpu0_available_gib is not None:
        values.append(float(gpu0_available_gib))
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("memory budget inputs must be finite and positive")
    budget = min(values)
    return {
        "budget_gib": budget,
        "gpu0_available_gib": gpu0_available_gib,
        "system_available_gib": float(system_available_gib),
        "ceiling_gib": float(ceiling_gib),
        "gpu_query_fallback": gpu0_available_gib is None,
        "formula": "min(gpu0_available_if_known, ceiling_gib, system_available)",
    }


def _point_mask(points: np.ndarray, box: AxisAlignedBox) -> np.ndarray:
    return np.all((points >= box.minimum) & (points <= box.maximum), axis=1)


def _padded_rectangles(
    minimum_xy: np.ndarray,
    maximum_xy: np.ndarray,
    image_sizes: np.ndarray,
    config: AdaptiveTilingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_observation = np.isfinite(minimum_xy[:, 0]) & np.isfinite(maximum_xy[:, 0])
    left_top = np.floor(minimum_xy)
    right_bottom = np.ceil(maximum_xy)
    size = right_bottom - left_top
    valid = valid_observation & np.all(size >= config.minimum_image_rectangle_pixels, axis=1)
    padded_minimum = np.maximum(
        left_top - config.image_halo_pixels_per_side, np.zeros((1, 2))
    )
    padded_maximum = np.minimum(
        right_bottom + config.image_halo_pixels_per_side, image_sizes
    )
    padded_size = np.maximum(padded_maximum - padded_minimum, 0)
    area = np.where(valid, padded_size[:, 0] * padded_size[:, 1], 0).astype(np.int64)
    rectangle_values = np.concatenate([padded_minimum, padded_size], axis=1)
    rectangle_values = np.where(valid[:, None], rectangle_values, 0)
    rectangles = rectangle_values.astype(np.int64)
    return area, valid, rectangles


def _rectangle_summary(
    table: ProjectedObservationTable,
    selected_points: np.ndarray,
    config: AdaptiveTilingConfig,
    *,
    include_rectangles: bool = False,
) -> tuple[int, int, list[dict[str, int]]]:
    selected_observations = selected_points[table.observation_point]
    images = table.observation_image[selected_observations]
    xy = table.observation_xy[selected_observations]
    minimum = np.full((len(table.image_sizes), 2), np.inf)
    maximum = np.full((len(table.image_sizes), 2), -np.inf)
    np.minimum.at(minimum, images, xy)
    np.maximum.at(maximum, images, xy)
    area, valid, rectangles = _padded_rectangles(minimum, maximum, table.image_sizes, config)
    rows: list[dict[str, int]] = []
    if include_rectangles:
        for image_index in np.flatnonzero(valid):
            x, y, width, height = rectangles[image_index].tolist()
            rows.append(
                {
                    "image_index": int(image_index),
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                    "pixel_load": int(area[image_index]),
                }
            )
    return int(area.sum()), int(valid.sum()), rows


def _candidate_cost(
    table: ProjectedObservationTable,
    selected_points: np.ndarray,
    axis: int,
    box: AxisAlignedBox,
    config: AdaptiveTilingConfig,
) -> dict[str, Any]:
    count = config.candidate_positions
    cuts = np.linspace(box.minimum[axis], box.maximum[axis], count)
    bins = np.searchsorted(cuts, table.points[:, axis], side="left")
    bins = np.clip(bins, 0, count - 1)
    selected_observations = selected_points[table.observation_point]
    obs_images = table.observation_image[selected_observations]
    obs_points = table.observation_point[selected_observations]
    obs_bins = bins[obs_points]
    obs_xy = table.observation_xy[selected_observations]
    shape = (len(table.image_sizes), count)
    min_x = np.full(shape, np.inf)
    min_y = np.full(shape, np.inf)
    max_x = np.full(shape, -np.inf)
    max_y = np.full(shape, -np.inf)
    indices = (obs_images, obs_bins)
    np.minimum.at(min_x, indices, obs_xy[:, 0])
    np.minimum.at(min_y, indices, obs_xy[:, 1])
    np.maximum.at(max_x, indices, obs_xy[:, 0])
    np.maximum.at(max_y, indices, obs_xy[:, 1])

    def accumulated(reverse: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arrays = (min_x, min_y, max_x, max_y)
        operators = (np.minimum.accumulate, np.minimum.accumulate, np.maximum.accumulate, np.maximum.accumulate)
        result = []
        for array, operator in zip(arrays, operators):
            value = operator(array[:, ::-1], axis=1)[:, ::-1] if reverse else operator(array, axis=1)
            result.append(value)
        return tuple(result)  # type: ignore[return-value]

    left = accumulated(False)
    right = accumulated(True)

    def areas(extrema: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        minimum = np.stack(extrema[:2], axis=2)
        maximum = np.stack(extrema[2:], axis=2)
        loads = np.zeros(count, dtype=np.int64)
        views = np.zeros(count, dtype=np.int64)
        for index in range(count):
            area, valid, _ = _padded_rectangles(
                minimum[:, index], maximum[:, index], table.image_sizes, config
            )
            loads[index] = area.sum()
            views[index] = valid.sum()
        return loads, views

    left_pixels, left_images = areas(left)
    right_pixels, right_images = areas(right)
    difference = left_pixels - right_pixels
    positive = np.flatnonzero(difference > 0)
    if len(positive) == 0:
        chosen = count // 2
    else:
        first_positive = int(positive[0])
        nonnegative = np.flatnonzero(difference >= 0)
        zeros = np.flatnonzero(difference == 0)
        chosen = (int(nonnegative[0]) + first_positive) // 2
        if len(zeros) and int(zeros[0]) <= first_positive:
            chosen = (int(zeros[0]) + first_positive) // 2
        if chosen in (0, count - 1):
            chosen = count // 2
    return {
        "axis_index": int(axis),
        "axis": "XYZ"[axis],
        "candidate_index": int(chosen),
        "value": float(cuts[chosen]),
        "left_pixel_load": int(left_pixels[chosen]),
        "right_pixel_load": int(right_pixels[chosen]),
        "left_view_count": int(left_images[chosen]),
        "right_view_count": int(right_images[chosen]),
        "maximum_child_pixel_load": int(max(left_pixels[chosen], right_pixels[chosen])),
    }


def _choose_split(
    table: ProjectedObservationTable,
    selected_points: np.ndarray,
    box: AxisAlignedBox,
    config: AdaptiveTilingConfig,
) -> dict[str, Any]:
    extent = box.extent
    longest = max(float(extent[0]), float(extent[1]))
    axes = [axis for axis in sorted((0, 1), key=lambda value: -extent[value]) if extent[axis] >= longest * config.minimum_eligible_axis_ratio]
    candidates = sorted(
        (_candidate_cost(table, selected_points, axis, box, config) for axis in axes),
        key=lambda row: (row["maximum_child_pixel_load"], row["axis_index"]),
    )
    if len(candidates) == 1:
        return candidates[0]
    best, second = candidates[:2]
    denominator = second["maximum_child_pixel_load"]
    improvement = (denominator - best["maximum_child_pixel_load"]) / denominator if denominator else 0.0
    if improvement < config.near_axis_cost_fraction:
        return max((best, second), key=lambda row: (extent[row["axis_index"]], -row["axis_index"]))
    return best


def _build_tree(
    split_table: ProjectedObservationTable,
    cost_table: ProjectedObservationTable,
    box: AxisAlignedBox,
    config: AdaptiveTilingConfig,
    budget_bytes: float | None,
    force_depth: int | None,
    depth: int = 0,
    residency: "GaussianResidencyModel | None" = None,
    budget_gib: float | None = None,
) -> _Node:
    split_selected = _point_mask(split_table.points, box)
    cost_selected = _point_mask(cost_table.points, box)
    pixels, image_count, _ = _rectangle_summary(cost_table, cost_selected, config)
    anchors = int(np.count_nonzero(split_selected))
    low_support = anchors < config.minimum_anchor_count and pixels < config.minimum_pixel_load
    node = _Node(box, depth, pixels, anchors, image_count, low_support)
    split_for_depth = force_depth is not None and depth < force_depth
    if residency is not None and budget_gib is not None:
        # Splitting is not free: every extra cut widens the halo, and a view
        # inside two tiles is trained twice. Cut only while the population a
        # tile would end up holding does not fit.
        split_for_budget = residency.peak_gib(anchors) > float(budget_gib)
    else:
        raw_bytes = pixels * bytes_per_pixel(config.resolution_level)
        split_for_budget = (
            budget_bytes is not None
            and raw_bytes * config.split_estimate_multiplier > budget_bytes
        )
    if low_support or depth >= config.maximum_depth or not (split_for_depth or split_for_budget):
        return node
    split = _choose_split(split_table, split_selected, box, config)
    left_box, right_box = box.split(split["axis_index"], split["value"])
    node.split = split
    node.children = (
        _build_tree(
            split_table, cost_table, left_box, config, budget_bytes, force_depth,
            depth + 1, residency, budget_gib,
        ),
        _build_tree(
            split_table, cost_table, right_box, config, budget_bytes, force_depth,
            depth + 1, residency, budget_gib,
        ),
    )
    return node


def _leaves(node: _Node) -> list[_Node]:
    if node.children is None:
        return [node]
    return _leaves(node.children[0]) + _leaves(node.children[1])


def _node_dict(node: _Node, config: AdaptiveTilingConfig) -> dict[str, Any]:
    bpp = bytes_per_pixel(config.resolution_level)
    row: dict[str, Any] = {
        "depth": node.depth,
        "core_box": node.box.to_list(),
        "export_box": node.box.expanded(config.spatial_halo_fraction_per_side).to_list(),
        "anchor_count": node.anchors,
        "valid_view_count": node.image_count,
        "pixel_load": node.pixels,
        "estimated_memory_gib": node.pixels * bpp / GIB,
        "split_comparison_memory_gib": node.pixels * bpp * config.split_estimate_multiplier / GIB,
        "low_support": node.low_support,
    }
    if node.split is not None:
        row["split"] = {key: value for key, value in node.split.items() if key != "axis_index"}
        row["children"] = [_node_dict(child, config) for child in node.children or ()]
    return row


def _config_dict(config: AdaptiveTilingConfig) -> dict[str, Any]:
    return {
        "resolution_level": config.resolution_level,
        "bytes_per_pixel": bytes_per_pixel(config.resolution_level),
        "candidate_positions": config.candidate_positions,
        "minimum_image_rectangle_pixels": config.minimum_image_rectangle_pixels,
        "image_halo_pixels_per_side": config.image_halo_pixels_per_side,
        "spatial_halo_fraction_per_side": config.spatial_halo_fraction_per_side,
        "maximum_depth": config.maximum_depth,
        "discard_only_if_anchor_count_below": config.minimum_anchor_count,
        "and_pixel_load_below": config.minimum_pixel_load,
        "minimum_eligible_axis_ratio": config.minimum_eligible_axis_ratio,
        "split_estimate_multiplier": config.split_estimate_multiplier,
        "near_axis_cost_fraction": config.near_axis_cost_fraction,
    }


def build_adaptive_tile_plan(
    split_table: ProjectedObservationTable,
    *,
    cost_table: ProjectedObservationTable | None = None,
    root_box: AxisAlignedBox | None = None,
    budget_gib: float | None = None,
    force_depth: int | None = None,
    config: AdaptiveTilingConfig = AdaptiveTilingConfig(),
    source_bindings: Mapping[str, str] | None = None,
    residency: GaussianResidencyModel | None = None,
) -> dict[str, Any]:
    """Build and sign one serial-training tile plan.

    ``force_depth`` is an explicit reproduction/test mode.  Production callers
    normally pass only the measured startup ``budget_gib``.

    With ``residency`` the cut criterion is the population a tile would hold
    rather than the sum of its view pixels, and the plan reports what the extra
    cuts cost in repeated views.
    """

    config.validate()
    split_table = split_table.validated()
    cost_table = split_table if cost_table is None else cost_table.validated()
    if budget_gib is None and force_depth is None:
        raise ValueError("one of budget_gib or force_depth is required")
    if budget_gib is not None and (not math.isfinite(budget_gib) or budget_gib <= 0.0):
        raise ValueError("budget_gib must be finite and positive")
    if force_depth is not None and not 0 <= force_depth <= config.maximum_depth:
        raise ValueError("force_depth is outside the configured depth range")
    if root_box is None:
        supported = split_table.support >= 3
        if not np.any(supported):
            raise ValueError("cannot derive a root box without points supported by three views")
        minimum = split_table.points[supported].min(axis=0)
        maximum = split_table.points[supported].max(axis=0)
        extent = maximum - minimum
        if np.any(extent <= 0.0):
            raise ValueError("supported points do not span a 3D root box")
        root_box = AxisAlignedBox(minimum - 0.2 * extent, maximum + 0.2 * extent)
        root_source = "support>=3 point box expanded by 20 percent"
    else:
        root_source = "caller-supplied accepted scene ROI"
    tree = _build_tree(
        split_table,
        cost_table,
        root_box,
        config,
        None if budget_gib is None else budget_gib * GIB,
        force_depth,
        residency=residency,
        budget_gib=None if residency is None else budget_gib,
    )
    leaf_rows = []
    for index, leaf in enumerate(_leaves(tree)):
        selected = _point_mask(cost_table.points, leaf.box)
        pixels, view_count, rectangles = _rectangle_summary(
            cost_table, selected, config, include_rectangles=True
        )
        row = {
            "tile_id": index,
            "name": f"Tile_{index}",
            "core_box": leaf.box.to_list(),
            "training_and_export_box": leaf.box.expanded(config.spatial_halo_fraction_per_side).to_list(),
            "anchor_count": leaf.anchors,
            "pixel_load": pixels,
            "valid_view_count": view_count,
            "estimated_memory_gib": pixels * bytes_per_pixel(config.resolution_level) / GIB,
            "low_support_discarded": leaf.low_support,
            "views": rectangles,
        }
        if residency is not None:
            # A prediction the trainer cannot exceed: the cap turns the budget
            # into an enforced limit instead of a hope about how growth lands.
            row["predicted_peak_gib"] = residency.peak_gib(leaf.anchors)
            row["gaussian_capacity_cap"] = residency.gaussian_cap(budget_gib)
        leaf_rows.append(row)
    payload: dict[str, Any] = {
        "schema_version": TILE_PLAN_SCHEMA_VERSION,
        "kind": TILE_PLAN_KIND,
        "evidence_boundary": (
            "Recovered constants and control flow are compatibility evidence; "
            "CloudStudio recomputes projections and signs its own observation table."
        ),
        "source_bindings": dict(sorted((source_bindings or {}).items())),
        "input": {
            "split_observation_table_sha256": split_table.sha256(),
            "cost_observation_table_sha256": cost_table.sha256(),
            "point_count": int(len(split_table.points)),
            "observation_count": int(len(split_table.observation_xy)),
            "image_count": int(len(split_table.image_sizes)),
            "root_box": root_box.to_list(),
            "root_source": root_source,
            "budget_gib": budget_gib,
            "force_depth": force_depth,
        },
        "config": _config_dict(config),
        "execution_contract": {
            "strict_serial_tiles": True,
            "multiple_tiles_resident_on_cuda": False,
            "empty_cuda_cache_even_steps": True,
            "empty_cuda_cache_after_each_tile": True,
            "halo_merge": "retain_full_tile_outputs_without_core_deduplication",
        },
        "tree": _node_dict(tree, config),
        "leaf_count": len(leaf_rows),
        "retained_tile_count": sum(not row["low_support_discarded"] for row in leaf_rows),
        "tiles": leaf_rows,
    }
    if residency is not None:
        retained = [row for row in leaf_rows if not row["low_support_discarded"]]
        instances = sum(int(row["valid_view_count"]) for row in retained)
        unique = len(
            {int(view["image_index"]) for row in retained for view in row["views"]
             if "image_index" in view}
        )
        payload["residency_model"] = residency.to_dict()
        payload["split_cost"] = {
            "view_instances": instances,
            "unique_views": unique,
            # Every cut widens the halo, and a view landing in two tiles is
            # trained in both. This is the price of splitting, stated instead
            # of hidden in the tile count.
            "halo_overlap_factor": (instances / unique) if unique else None,
            "predicted_peak_gib_max": max(
                (row["predicted_peak_gib"] for row in retained), default=None
            ),
            "budget_gib": budget_gib,
        }
    payload["tile_plan_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def verify_adaptive_tile_plan(plan: dict[str, Any]) -> str:
    expected = str(plan.get("tile_plan_manifest_sha256", ""))
    if len(expected) != 64:
        raise ValueError("adaptive tile plan is unsigned")
    unsigned = copy.deepcopy(plan)
    unsigned.pop("tile_plan_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("adaptive tile plan signature mismatch")
    if plan.get("schema_version") != TILE_PLAN_SCHEMA_VERSION or plan.get("kind") != TILE_PLAN_KIND:
        raise ValueError("unsupported adaptive tile plan schema")
    tiles = plan.get("tiles")
    if not isinstance(tiles, list) or len(tiles) != int(plan.get("leaf_count", -1)):
        raise ValueError("adaptive tile plan leaf count is inconsistent")
    if plan.get("execution_contract", {}).get("strict_serial_tiles") is not True:
        raise ValueError("adaptive tile plan does not require strict serial execution")
    for index, tile in enumerate(tiles):
        if tile.get("tile_id") != index or tile.get("name") != f"Tile_{index}":
            raise ValueError("adaptive tile identifiers are not contiguous and deterministic")
        AxisAlignedBox(*np.asarray(tile["core_box"], dtype=np.float64))
        AxisAlignedBox(*np.asarray(tile["training_and_export_box"], dtype=np.float64))
    return expected
