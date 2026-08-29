"""Read-only compatibility replay for MipMap's adaptive MVS tiler.

The implementation mirrors the primitives recovered from divide_engine.exe:

* planar (X/Y) recursive KD splitting when pipeline_mode != 0;
* 64 uniformly spaced cut candidates per eligible axis;
* per-side, per-image rectangles from a fresh world-to-camera projection;
* a 256 px minimum valid rectangle and a 128 px image halo;
* the cut where left/right pixel loads cross;
* separate continuous ownership boxes and post-cut anchor-point envelopes;
* the axis minimizing the larger child load, except that a <10 percent cost
  difference is resolved in favor of the longer spatial axis;
* 5.5 estimated bytes per retained undistorted pixel at levels 0/1;
* a 0.8 multiplier on that raw estimate for the split/no-split comparison;
* 0.2 percent spatial halo per leaf side for exported ROIs.

The point/image association comes from MVS observations, but the observation
pixel coordinates do not: divide_engine.exe reprojects each associated point,
applies C ``round()``, and rejects projections outside the image.  The separate
ceil/floor candidate-bin convention is mirrored as well.

This is an evidence-backed compatibility implementation, not vendor source.
It never modifies the supplied MVS/task files.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf import message_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_mipmap_mvs as mvs_decoder  # noqa: E402


GIB = 1024**3
SAMPLES = 64
MIN_RECT = 256
IMAGE_HALO = 128
SPATIAL_HALO_FRACTION = 0.002
MAX_DEPTH = 10
MIN_ANCHORS = 100
MIN_PIXELS = 100_000
MIN_AXIS_RATIO = 0.2
SPLIT_ESTIMATE_MULTIPLIER = 0.8


@dataclass
class Box:
    minimum: np.ndarray
    maximum: np.ndarray

    def copy(self) -> "Box":
        return Box(self.minimum.copy(), self.maximum.copy())

    @property
    def extent(self) -> np.ndarray:
        return self.maximum - self.minimum

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) * 0.5

    def as_list(self) -> list[list[float]]:
        return [self.minimum.tolist(), self.maximum.tolist()]


@dataclass
class Node:
    box: Box
    candidate_box: Box
    depth: int
    pixels: int
    anchors: int
    image_count: int
    split_axis: int | None = None
    split_value: float | None = None
    left_pixels: int | None = None
    right_pixels: int | None = None
    children: tuple["Node", "Node"] | None = None


def load_mvs(engine_path: Path, mvs_path: Path) -> dict[str, Any]:
    pool = mvs_decoder._build_pool(engine_path)
    descriptor = pool.FindMessageTypeByName("mipmap.engine.message.MVSBlock")
    block_class = message_factory.GetMessageClass(descriptor)
    block = block_class()
    block.ParseFromString(mvs_path.read_bytes())

    points = np.asarray(
        [[p.point.x, p.point.y, p.point.z] for p in block.point],
        dtype=np.float32,
    )
    observation_xy = np.asarray(
        [[o.x, o.y] for o in block.observation], dtype=np.float64
    )
    observation_image = np.asarray(
        [o.img_index for o in block.observation], dtype=np.int64
    )
    observation_point = np.asarray(
        [o.pnt_index for o in block.observation], dtype=np.int64
    )
    camera_by_id = {int(camera.camera_id): camera for camera in block.camera}
    projection_matrices = np.asarray(
        [image.projection_matrix for image in block.image], dtype=np.float64
    ).reshape(-1, 3, 4)
    intrinsics = np.asarray(
        [camera_by_id[int(image.camera_id)].camera_params[:4] for image in block.image],
        dtype=np.float64,
    )
    image_sizes = np.asarray(
        [
            [
                camera_by_id[int(image.camera_id)].width,
                camera_by_id[int(image.camera_id)].height,
            ]
            for image in block.image
        ],
        dtype=np.int64,
    )
    homogeneous = np.concatenate(
        [points[observation_point].astype(np.float64), np.ones((len(observation_point), 1))],
        axis=1,
    )
    camera_xyz = np.einsum(
        "nij,nj->ni", projection_matrices[observation_image], homogeneous
    )
    focal_and_centre = intrinsics[observation_image]
    projected_xy = np.empty((len(observation_point), 2), dtype=np.float64)
    projected_xy[:, 0] = (
        focal_and_centre[:, 0] * camera_xyz[:, 0] / camera_xyz[:, 2]
        + focal_and_centre[:, 2]
    )
    projected_xy[:, 1] = (
        focal_and_centre[:, 1] * camera_xyz[:, 1] / camera_xyz[:, 2]
        + focal_and_centre[:, 3]
    )
    # C round(): halfway cases go away from zero.  MVS pixels are finite here,
    # but retain an explicit mask so corrupt inputs fail closed.
    rounded_xy = np.where(
        projected_xy >= 0,
        np.floor(projected_xy + 0.5),
        np.ceil(projected_xy - 0.5),
    ).astype(np.int64)
    projected_valid = (
        np.isfinite(projected_xy).all(axis=1)
        & (camera_xyz[:, 2] != 0)
        & (rounded_xy[:, 0] >= 0)
        & (rounded_xy[:, 1] >= 0)
        & (rounded_xy[:, 0] < image_sizes[observation_image, 0])
        & (rounded_xy[:, 1] < image_sizes[observation_image, 1])
    )
    rotations = projection_matrices[:, :, :3]
    translations = projection_matrices[:, :, 3]
    camera_centres = np.asarray(
        [np.linalg.solve(rotation, -translation) for rotation, translation in zip(rotations, translations)],
        dtype=np.float32,
    )
    support = np.bincount(observation_point, minlength=len(points))
    rounded_stored = np.where(
        observation_xy >= 0,
        np.floor(observation_xy + 0.5),
        np.ceil(observation_xy - 0.5),
    ).astype(np.int64)
    return {
        "points": points,
        "observation_xy": rounded_xy,
        "stored_observation_xy": observation_xy,
        "observation_projection_valid": projected_valid,
        "projection_round_match_fraction": float(
            np.mean(np.all(rounded_xy == rounded_stored, axis=1))
        ),
        "observation_image": observation_image,
        "observation_point": observation_point,
        "image_sizes": image_sizes,
        "camera_centres": camera_centres,
        "support": support,
        "image_count": len(block.image),
    }


def use_stored_observations(data: dict[str, Any]) -> None:
    """Switch a decoded parent-photo block to measured observation pixels.

    The vendor division entry prefers the undistorted MVSBlock.  For snow, a
    parent-view audit proves that its 1368 virtual views form 342 groups of four
    and that the collapsed observation graph is a strict subset of the original
    MVS graph.  The exact compatibility replay therefore uses the original
    stored pixels as a bounded surrogate for the still-unrecovered parent-photo
    load aggregation; this is not evidence that the vendor loads the raw block
    at the recursive split entry.
    """

    stored = data["stored_observation_xy"]
    rounded = np.where(stored >= 0, np.floor(stored + 0.5), np.ceil(stored - 0.5)).astype(
        np.int64
    )
    image_index = data["observation_image"]
    sizes = data["image_sizes"][image_index]
    data["observation_xy"] = rounded
    data["observation_projection_valid"] = (
        np.isfinite(stored).all(axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 0] < sizes[:, 0])
        & (rounded[:, 1] < sizes[:, 1])
    )


def bytes_per_pixel(resolution_level: int) -> float:
    # Direct binary structure is B + 4.5*A. The snow leaf records provide a
    # runtime calibration: max_memory / replayed undistorted area is 5.5 B/px
    # within 0.2% for Tile_1..3. At RVA 0x3C876 the separate 0.8 constant
    # multiplies this raw estimate for the split/no-split comparison; it is not
    # part of the leaf's exported max_memory value.
    if resolution_level in (0, 1):
        return 5.5
    if resolution_level == 2:
        return 1.375
    if resolution_level == 3:
        return 1.1875
    raise ValueError(f"unsupported resolution level: {resolution_level}")


def point_mask(points: np.ndarray, box: Box) -> np.ndarray:
    return np.all((points >= box.minimum) & (points <= box.maximum), axis=1)


def padded_areas(
    min_x: np.ndarray,
    min_y: np.ndarray,
    max_x: np.ndarray,
    max_y: np.ndarray,
    image_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-candidate pixel load and valid-image count.

    Extrema arrays have shape [image, candidate]. The rectangle conversion is
    the integer equivalent of the recovered vendor block at RVA 0x3EC16..0x3ED0C.
    """
    valid_observation = np.isfinite(min_x) & np.isfinite(max_x)
    left = min_x
    top = min_y
    right = max_x
    bottom = max_y

    widths = right - left
    heights = bottom - top
    valid = valid_observation & (widths >= MIN_RECT) & (heights >= MIN_RECT)

    image_width = image_sizes[:, 0, None]
    image_height = image_sizes[:, 1, None]
    padded_left = np.maximum(left - IMAGE_HALO, 0)
    padded_top = np.maximum(top - IMAGE_HALO, 0)
    padded_right = np.minimum(right + IMAGE_HALO, image_width)
    padded_bottom = np.minimum(bottom + IMAGE_HALO, image_height)
    area = np.where(
        valid,
        np.maximum(padded_right - padded_left, 0)
        * np.maximum(padded_bottom - padded_top, 0),
        0,
    )
    return area.sum(axis=0).astype(np.int64), valid.sum(axis=0).astype(np.int64)


def rectangle_cost(data: dict[str, Any], selected_points: np.ndarray) -> tuple[int, int]:
    obs_selected = (
        selected_points[data["observation_point"]]
        & data["observation_projection_valid"]
    )
    images = data["observation_image"][obs_selected]
    xy = data["observation_xy"][obs_selected]
    shape = (data["image_count"], 1)
    min_x = np.full(shape, np.inf)
    min_y = np.full(shape, np.inf)
    max_x = np.full(shape, -np.inf)
    max_y = np.full(shape, -np.inf)
    np.minimum.at(min_x[:, 0], images, xy[:, 0])
    np.minimum.at(min_y[:, 0], images, xy[:, 1])
    np.maximum.at(max_x[:, 0], images, xy[:, 0])
    np.maximum.at(max_y[:, 0], images, xy[:, 1])
    pixels, images_retained = padded_areas(
        min_x, min_y, max_x, max_y, data["image_sizes"]
    )
    return int(pixels[0]), int(images_retained[0])


def split_axis_costs(
    data: dict[str, Any], selected_points: np.ndarray, axis: int, box: Box
) -> dict[str, Any]:
    minimum = np.float32(box.minimum[axis])
    maximum = np.float32(box.maximum[axis])
    step = np.float32((maximum - minimum) / np.float32(SAMPLES - 1))
    cuts = (minimum + np.arange(SAMPLES, dtype=np.float32) * step).astype(np.float32)
    coordinates = data["points"][:, axis].astype(np.float32)
    normalized = ((coordinates - minimum) / step).astype(np.float32)
    left_point_bins = np.maximum(np.ceil(normalized).astype(np.int64), 0)
    right_point_bins = np.minimum(
        np.floor(normalized).astype(np.int64), SAMPLES - 1
    )

    obs_selected = (
        selected_points[data["observation_point"]]
        & data["observation_projection_valid"]
    )
    obs_images = data["observation_image"][obs_selected]
    obs_points = data["observation_point"][obs_selected]
    obs_xy = data["observation_xy"][obs_selected]

    shape = (data["image_count"], SAMPLES)
    left_min_x = np.full(shape, np.inf)
    left_min_y = np.full(shape, np.inf)
    left_max_x = np.full(shape, -np.inf)
    left_max_y = np.full(shape, -np.inf)
    left_bins = left_point_bins[obs_points]
    left_valid = left_bins < SAMPLES
    left_indices = (obs_images[left_valid], left_bins[left_valid])
    np.minimum.at(left_min_x, left_indices, obs_xy[left_valid, 0])
    np.minimum.at(left_min_y, left_indices, obs_xy[left_valid, 1])
    np.maximum.at(left_max_x, left_indices, obs_xy[left_valid, 0])
    np.maximum.at(left_max_y, left_indices, obs_xy[left_valid, 1])
    left_min_x = np.minimum.accumulate(left_min_x, axis=1)
    left_min_y = np.minimum.accumulate(left_min_y, axis=1)
    left_max_x = np.maximum.accumulate(left_max_x, axis=1)
    left_max_y = np.maximum.accumulate(left_max_y, axis=1)

    right_min_x = np.full(shape, np.inf)
    right_min_y = np.full(shape, np.inf)
    right_max_x = np.full(shape, -np.inf)
    right_max_y = np.full(shape, -np.inf)
    right_bins = right_point_bins[obs_points]
    right_valid = right_bins >= 0
    right_indices = (obs_images[right_valid], right_bins[right_valid])
    np.minimum.at(right_min_x, right_indices, obs_xy[right_valid, 0])
    np.minimum.at(right_min_y, right_indices, obs_xy[right_valid, 1])
    np.maximum.at(right_max_x, right_indices, obs_xy[right_valid, 0])
    np.maximum.at(right_max_y, right_indices, obs_xy[right_valid, 1])
    right_min_x = np.minimum.accumulate(right_min_x[:, ::-1], axis=1)[:, ::-1]
    right_min_y = np.minimum.accumulate(right_min_y[:, ::-1], axis=1)[:, ::-1]
    right_max_x = np.maximum.accumulate(right_max_x[:, ::-1], axis=1)[:, ::-1]
    right_max_y = np.maximum.accumulate(right_max_y[:, ::-1], axis=1)[:, ::-1]

    left_pixels, left_images = padded_areas(
        left_min_x,
        left_min_y,
        left_max_x,
        left_max_y,
        data["image_sizes"],
    )
    right_pixels, right_images = padded_areas(
        right_min_x,
        right_min_y,
        right_max_x,
        right_max_y,
        data["image_sizes"],
    )

    difference = left_pixels.astype(np.int64) - right_pixels.astype(np.int64)
    positive = np.flatnonzero(difference > 0)
    if len(positive) == 0:
        chosen = SAMPLES // 2
    else:
        first_positive = int(positive[0])
        zeros = np.flatnonzero(difference == 0)
        first_nonnegative = int(np.flatnonzero(difference >= 0)[0])
        chosen = (first_nonnegative + first_positive) // 2
        if len(zeros) and zeros[0] <= first_positive:
            chosen = (int(zeros[0]) + first_positive) // 2
        if chosen in (0, SAMPLES - 1):
            chosen = SAMPLES // 2

    return {
        "axis": axis,
        "cut": float(cuts[chosen]),
        "index": chosen,
        "left_pixels": int(left_pixels[chosen]),
        "right_pixels": int(right_pixels[chosen]),
        "left_images": int(left_images[chosen]),
        "right_images": int(right_images[chosen]),
        "maximum_pixels": int(max(left_pixels[chosen], right_pixels[chosen])),
    }


def choose_split(data: dict[str, Any], selected_points: np.ndarray, box: Box) -> dict[str, Any]:
    extent = box.extent
    planar_axes = [0, 1]
    longest = max(float(extent[axis]) for axis in planar_axes)
    axes = [
        axis
        for axis in sorted(planar_axes, key=lambda value: -extent[value])
        if extent[axis] >= longest * MIN_AXIS_RATIO
    ]
    candidates = sorted(
        (split_axis_costs(data, selected_points, axis, box) for axis in axes),
        key=lambda row: (row["maximum_pixels"], row["axis"]),
    )
    if len(candidates) == 1:
        return candidates[0]

    best, second = candidates[:2]
    relative_improvement = (
        (second["maximum_pixels"] - best["maximum_pixels"])
        / second["maximum_pixels"]
        if second["maximum_pixels"]
        else 0.0
    )
    # RVA 0x3DF21..0x3DFAF: if the best two costs differ by less than 10%,
    # prefer the candidate on the longer spatial axis. This avoids unstable
    # axis flips for nearly equivalent image-load cuts.
    if relative_improvement < 0.1:
        return max(
            (best, second),
            key=lambda row: (extent[row["axis"]], -row["axis"]),
        )
    return best


def build_tree(
    split_data: dict[str, Any],
    candidate_data: dict[str, Any],
    cost_data: dict[str, Any],
    box: Box,
    resolution_level: int,
    budget_bytes: float | None,
    force_depth: int | None,
    forced_axis_by_depth: list[int] | None,
    child_envelope_min_observations: int,
    depth: int = 0,
) -> Node:
    split_selected = point_mask(split_data["points"], box)
    candidate_selected = point_mask(candidate_data["points"], box)
    cost_selected = point_mask(cost_data["points"], box)
    pixels, image_count = rectangle_cost(cost_data, cost_selected)
    anchors = int(np.count_nonzero(split_selected))
    if depth > 0 and child_envelope_min_observations > 0:
        candidate_selected = candidate_selected & (
            candidate_data["support"] >= child_envelope_min_observations
        )
    if np.any(candidate_selected):
        selected_xyz = candidate_data["points"][candidate_selected]
        candidate_box = Box(
            selected_xyz.min(axis=0).astype(np.float64),
            selected_xyz.max(axis=0).astype(np.float64),
        )
    else:
        candidate_box = box.copy()
    node = Node(box, candidate_box, depth, pixels, anchors, image_count)

    low_support = anchors < MIN_ANCHORS and pixels < MIN_PIXELS
    estimated_bytes = pixels * bytes_per_pixel(resolution_level)
    split_for_depth = force_depth is not None and depth < force_depth
    split_for_budget = (
        budget_bytes is not None
        and estimated_bytes * SPLIT_ESTIMATE_MULTIPLIER > budget_bytes
    )
    if low_support or depth >= MAX_DEPTH or not (split_for_depth or split_for_budget):
        return node

    if forced_axis_by_depth is not None and depth < len(forced_axis_by_depth):
        split = split_axis_costs(
            split_data,
            split_selected,
            forced_axis_by_depth[depth],
            candidate_box,
        )
    else:
        split = choose_split(split_data, split_selected, candidate_box)
    node.split_axis = int(split["axis"])
    node.split_value = float(split["cut"])
    node.left_pixels = int(split["left_pixels"])
    node.right_pixels = int(split["right_pixels"])
    left_box = box.copy()
    right_box = box.copy()
    left_box.maximum[node.split_axis] = node.split_value
    right_box.minimum[node.split_axis] = node.split_value
    # RVA 0x3F02D..0x3F166 creates continuous child half-space boxes in node
    # offsets +0x00..+0x14.  RVA 0x3F31B..0x3F6F0 independently recomputes the
    # selected-point envelope at +0x18..+0x2c.  The next split's 64-bin grid
    # reads that second envelope (RVA 0x3D94F..0x3D961); it must not shrink the
    # ownership/training box.  build_tree recomputes candidate_box on entry.
    node.children = (
        build_tree(
            split_data,
            candidate_data,
            cost_data,
            left_box,
            resolution_level,
            budget_bytes,
            force_depth,
            forced_axis_by_depth,
            child_envelope_min_observations,
            depth + 1,
        ),
        build_tree(
            split_data,
            candidate_data,
            cost_data,
            right_box,
            resolution_level,
            budget_bytes,
            force_depth,
            forced_axis_by_depth,
            child_envelope_min_observations,
            depth + 1,
        ),
    )
    return node


def leaves(node: Node) -> list[Node]:
    if node.children is None:
        return [node]
    return leaves(node.children[0]) + leaves(node.children[1])


def expanded_box(box: Box) -> Box:
    padding = box.extent * SPATIAL_HALO_FRACTION
    return Box(box.minimum - padding, box.maximum + padding)


def runtime_tiles(tiles_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(tiles_path.read_text(encoding="utf-8"))
    rows = []
    for tile in payload["tiles"]:
        boundary = np.asarray(tile["roi"]["boundary"], dtype=np.float64)
        expanded_min = np.asarray(
            [boundary[:, 0].min(), boundary[:, 1].min(), tile["roi"]["min_z"]]
        )
        expanded_max = np.asarray(
            [boundary[:, 0].max(), boundary[:, 1].max(), tile["roi"]["max_z"]]
        )
        expanded_extent = expanded_max - expanded_min
        core_padding = expanded_extent * (
            SPATIAL_HALO_FRACTION / (1 + 2 * SPATIAL_HALO_FRACTION)
        )
        core = Box(expanded_min + core_padding, expanded_max - core_padding)
        rows.append(
            {
                "name": tile["name"],
                "max_memory_gib": float(tile["max_memory"]),
                "expanded": Box(expanded_min, expanded_max),
                "core": core,
            }
        )
    return rows


def runtime_root(rows: list[dict[str, Any]]) -> Box:
    return Box(
        np.min(np.asarray([row["core"].minimum for row in rows]), axis=0),
        np.max(np.asarray([row["core"].maximum for row in rows]), axis=0),
    )


def supported_root(data: dict[str, Any]) -> Box:
    """Mirror divide_engine's camera + support>=3 root and 20% expansion."""
    supported = data["support"] >= 3
    samples = np.concatenate(
        [
            data["points"][supported].astype(np.float32),
            data["camera_centres"].astype(np.float32),
        ],
        axis=0,
    )
    minimum = samples.min(axis=0).astype(np.float32)
    maximum = samples.max(axis=0).astype(np.float32)
    padding = (maximum.astype(np.float64) - minimum.astype(np.float64)) * 0.2
    return Box(
        (minimum.astype(np.float64) - padding).astype(np.float32).astype(np.float64),
        (maximum.astype(np.float64) + padding).astype(np.float32).astype(np.float64),
    )


def best_leaf_matching(predicted: list[Node], runtime: list[dict[str, Any]]) -> list[int] | None:
    if len(predicted) != len(runtime) or len(predicted) > 9:
        return None
    best_score = math.inf
    best: list[int] | None = None
    for permutation in itertools.permutations(range(len(runtime))):
        score = sum(
            float(np.linalg.norm(predicted[index].box.center - runtime[target]["core"].center))
            for index, target in enumerate(permutation)
        )
        if score < best_score:
            best_score = score
            best = list(permutation)
    return best


def node_to_dict(node: Node, resolution_level: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "depth": node.depth,
        "ownership_box": node.box.as_list(),
        "candidate_point_envelope": node.candidate_box.as_list(),
        "core_box": node.box.as_list(),
        "exported_box_with_0_2_percent_halo": expanded_box(node.box).as_list(),
        "anchor_count": node.anchors,
        "valid_image_count": node.image_count,
        "pixel_load": node.pixels,
        "estimated_memory_gib": node.pixels
        * bytes_per_pixel(resolution_level)
        / GIB,
        "split_comparison_memory_gib": node.pixels
        * bytes_per_pixel(resolution_level)
        * SPLIT_ESTIMATE_MULTIPLIER
        / GIB,
    }
    if node.split_axis is not None:
        row["split"] = {
            "axis": "XYZ"[node.split_axis],
            "value": node.split_value,
            "left_pixel_load": node.left_pixels,
            "right_pixel_load": node.right_pixels,
        }
        row["children"] = [
            node_to_dict(child, resolution_level) for child in node.children or ()
        ]
    return row


def empirical_split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = runtime_root(rows)
    x_groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (round(float(row["core"].minimum[0]), 6), round(float(row["core"].maximum[0]), 6))
        x_groups.setdefault(key, []).append(row)
    x_boundaries = sorted({value for key in x_groups for value in key})
    result: dict[str, Any] = {"root_core_box": root.as_list()}
    if len(x_boundaries) == 3:
        result["root_x_split"] = x_boundaries[1]
    result["column_y_splits"] = []
    for key, group in sorted(x_groups.items()):
        y_boundaries = sorted(
            {round(float(value), 6) for row in group for value in (row["core"].minimum[1], row["core"].maximum[1])}
        )
        if len(y_boundaries) == 3:
            result["column_y_splits"].append(
                {"x_range": list(key), "y_split": y_boundaries[1]}
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument(
        "mvs",
        type=Path,
        help=(
            "compatibility MVS used to choose split axes/cuts; the vendor "
            "division entry itself prefers mvs_undistort.pb.bin"
        ),
    )
    parser.add_argument(
        "--memory-mvs",
        type=Path,
        help="optional undistorted MVS used to replay final leaf memory",
    )
    parser.add_argument(
        "--candidate-mvs",
        type=Path,
        help=(
            "optional MVS whose point envelopes define the 64-bin coordinates; "
            "snow closes exactly with mvs_undistort.pb.bin"
        ),
    )
    parser.add_argument(
        "--split-use-stored-observations",
        action="store_true",
        help=(
            "use measured parent-photo observation pixels as the compatibility "
            "surrogate for undistorted-view parent-load aggregation"
        ),
    )
    parser.add_argument(
        "--root-mvs",
        type=Path,
        help=(
            "optional MVS whose camera centres and support>=3 points construct "
            "the root; overrides the legacy runtime-tile root"
        ),
    )
    parser.add_argument("--tiles", type=Path)
    parser.add_argument("--resolution-level", type=int, default=1)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument(
        "--force-depth",
        type=int,
        help="validation mode: split every supported node to this depth",
    )
    parser.add_argument(
        "--force-axis-by-depth",
        help=(
            "comma-separated validation override such as X,Y; cuts are still "
            "selected by the recovered 64-bin pixel-cost evaluator"
        ),
    )
    parser.add_argument(
        "--child-envelope-min-observations",
        type=int,
        default=0,
        help=(
            "compatibility hypothesis for post-root candidate envelopes; snow "
            "matches the vendor right-column cut exactly at 2"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.budget_gib is None and args.force_depth is None:
        parser.error("one of --budget-gib or --force-depth is required")
    forced_axis_by_depth = None
    if args.force_axis_by_depth:
        tokens = [token.strip().upper() for token in args.force_axis_by_depth.split(",")]
        if any(token not in "XYZ" or len(token) != 1 for token in tokens):
            parser.error("--force-axis-by-depth accepts only X,Y,Z tokens")
        forced_axis_by_depth = ["XYZ".index(token) for token in tokens]

    data = load_mvs(args.engine, args.mvs)
    if args.split_use_stored_observations:
        use_stored_observations(data)
    candidate_data = (
        load_mvs(args.engine, args.candidate_mvs) if args.candidate_mvs else data
    )
    memory_data = load_mvs(args.engine, args.memory_mvs) if args.memory_mvs else data
    root_data = load_mvs(args.engine, args.root_mvs) if args.root_mvs else data
    runtime = runtime_tiles(args.tiles) if args.tiles else []
    if args.root_mvs:
        root = supported_root(root_data)
        root_source = (
            "camera centres plus support>=3 points from root MVS, expanded "
            "20 percent per side"
        )
    elif runtime:
        root = runtime_root(runtime)
        root_source = "runtime tiles with the 0.2 percent export halo inverted"
    else:
        root = supported_root(root_data)
        root_source = (
            "camera centres plus support>=3 points, expanded 20 percent per side"
        )

    tree = build_tree(
        data,
        candidate_data,
        memory_data,
        root,
        args.resolution_level,
        args.budget_gib * GIB if args.budget_gib is not None else None,
        args.force_depth,
        forced_axis_by_depth,
        args.child_envelope_min_observations,
    )
    leaf_nodes = leaves(tree)
    result: dict[str, Any] = {
        "method": "MipMap adaptive tiling compatibility replay",
        "evidence_boundary": (
            "Recovered constants/control flow and the vendor preference for the undistorted "
            "MVSBlock are direct binary evidence. The optional original-MVS stored-pixel path "
            "is a parent-photo load surrogate; child support filtering and root-box fallback "
            "remain compatibility interpretations."
        ),
        "input": {
            "engine": str(args.engine),
            "mvs": str(args.mvs),
            "memory_mvs": str(args.memory_mvs) if args.memory_mvs else str(args.mvs),
            "candidate_mvs": (
                str(args.candidate_mvs) if args.candidate_mvs else str(args.mvs)
            ),
            "split_use_stored_observations": args.split_use_stored_observations,
            "root_mvs": str(args.root_mvs) if args.root_mvs else str(args.mvs),
            "point_count": len(data["points"]),
            "observation_count": len(data["observation_point"]),
            "image_count": data["image_count"],
            "resolution_level": args.resolution_level,
            "bytes_per_pixel": bytes_per_pixel(args.resolution_level),
            "budget_gib": args.budget_gib,
            "force_depth": args.force_depth,
            "forced_axis_by_depth": args.force_axis_by_depth,
            "child_envelope_min_observations": args.child_envelope_min_observations,
            "projection_round_match_fraction": data[
                "projection_round_match_fraction"
            ],
        },
        "recovered_constants": {
            "candidate_positions": SAMPLES,
            "minimum_image_rectangle_pixels": MIN_RECT,
            "image_halo_pixels_per_side": IMAGE_HALO,
            "export_spatial_halo_fraction_per_side": SPATIAL_HALO_FRACTION,
            "maximum_depth": MAX_DEPTH,
            "discard_only_if_anchor_count_below": MIN_ANCHORS,
            "and_pixel_load_below": MIN_PIXELS,
            "minimum_eligible_axis_ratio": MIN_AXIS_RATIO,
            "split_estimate_multiplier": SPLIT_ESTIMATE_MULTIPLIER,
        },
        "root_source": root_source,
        "tree": node_to_dict(tree, args.resolution_level),
        "leaf_count": len(leaf_nodes),
    }

    if runtime:
        runtime_core_memory_replay = []
        for row in runtime:
            core_mask = point_mask(memory_data["points"], row["core"])
            memory_pixels, memory_images = rectangle_cost(memory_data, core_mask)
            replay_gib = (
                memory_pixels * bytes_per_pixel(args.resolution_level) / GIB
            )
            runtime_core_memory_replay.append(
                {
                    "name": row["name"],
                    "memory_replay_pixel_load": memory_pixels,
                    "memory_replay_valid_images": memory_images,
                    "memory_replay_gib": replay_gib,
                    "vendor_max_memory_gib": row["max_memory_gib"],
                    "relative_error": (
                        (replay_gib - row["max_memory_gib"])
                        / row["max_memory_gib"]
                        if row["max_memory_gib"]
                        else None
                    ),
                }
            )
        result["runtime"] = {
            "empirical_splits": empirical_split_summary(runtime),
            "actual_core_memory_replay": runtime_core_memory_replay,
            "tiles": [
                {
                    "name": row["name"],
                    "core_box": row["core"].as_list(),
                    "expanded_box": row["expanded"].as_list(),
                    "vendor_max_memory_gib": row["max_memory_gib"],
                    "vendor_implied_pixel_load_at_replayed_Bpp": row["max_memory_gib"]
                    * GIB
                    / bytes_per_pixel(args.resolution_level),
                }
                for row in runtime
            ],
        }
        matching = best_leaf_matching(leaf_nodes, runtime)
        if matching is not None:
            comparisons = []
            for predicted_index, runtime_index in enumerate(matching):
                predicted = leaf_nodes[predicted_index]
                actual = runtime[runtime_index]
                memory_pixels, memory_images = rectangle_cost(
                    memory_data, point_mask(memory_data["points"], predicted.box)
                )
                comparisons.append(
                    {
                        "predicted_leaf": predicted_index,
                        "runtime_tile": actual["name"],
                        "center_error_m": float(
                            np.linalg.norm(predicted.box.center - actual["core"].center)
                        ),
                        "minimum_error_m": (
                            predicted.box.minimum - actual["core"].minimum
                        ).tolist(),
                        "maximum_error_m": (
                            predicted.box.maximum - actual["core"].maximum
                        ).tolist(),
                        "predicted_pixel_load": predicted.pixels,
                        "memory_replay_pixel_load": memory_pixels,
                        "memory_replay_valid_images": memory_images,
                        "memory_replay_gib": memory_pixels
                        * bytes_per_pixel(args.resolution_level)
                        / GIB,
                        "vendor_implied_pixel_load": actual["max_memory_gib"]
                        * GIB
                        / bytes_per_pixel(args.resolution_level),
                    }
                )
            result["runtime"]["matched_leaf_comparison"] = comparisons

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
