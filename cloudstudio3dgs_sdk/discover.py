"""A planning-time :class:`DatasetSummary` derived from the capture itself.

``build_plan`` needs a :class:`~cloudstudio3dgs_sdk.plan.DatasetSummary` before
it can cost anything, and until now the only source of one was
``prepare()``'s ``prepare_manifest.json``. That made the first question anyone
asks about a new capture - *how long, how much disk* - the one question the
CLI could not answer, because answering it required running the whole
ingestion first.

This module answers it from the capture. It reads the dataset through the
ingestion adapters, cuts the LiDAR cloud with the ingestion slab rule, and
counts, per tile, the points inside the tile box and the face samples that can
see it. Nothing here writes a cache, touches a GPU or produces anything the
trainer will bind.

What is measured and what is estimated
--------------------------------------
Exactly two of the numbers are measurements of the file on disk:

``lidar_point_count``
    every point in the LAS/LAZ, counted by streaming it.
``TileSummary.init_point_count``
    every point inside that tile's ``training_and_export_box``, counted the
    same way. Against the real house0305 boxes this reproduces the four as-run
    initialisation point counts exactly (7044777 / 3417320 / 3309574 /
    5651827, 0.00% error), because ``materialize_lidar_tile_inputs`` writes
    precisely that set of points.

Everything else is an estimate, and each one is listed in
:attr:`DatasetEstimate.notes` so the dry-run transcript can print it:

tile boxes
    The production tiles came from the projected-pixel kd planner
    (``cloudstudio_3dgs.pipeline.adaptive_tiling``), which needs the Face4
    caches this fallback exists to avoid building. The boxes here come from
    :func:`cloudstudio3dgs_sdk.ingest.tiling.slab_split` instead: equal *point
    count* slabs along the longest horizontal axis. Same schema, different cut.
    Since ``init_point_count`` is exact *for the box it is given*, a different
    box is the whole error in the per-tile numbers. On house0305 the slab rule
    cuts X four ways; the kd planner cut X once and then Y twice, so the two
    tile sets do not correspond tile for tile and a per-tile comparison of the
    end-to-end output is meaningless. What does compare is the scene total:
    19,471,467 estimated initialisation points against 19,423,498 as-run
    (+0.2%), the difference being halo overlap.

``TileSummary.view_count``
    the rule below. It drives ``max_steps`` (20 epochs of the tile's own
    views), so it is the number that moves the time estimate most.

``train_view_count``
    images times faces per image. A capture that does not declare a split is
    assumed to be all-train, which is an over-count by whatever fraction
    ``prepare()`` later holds out.

``global_init_point_count``
    an upper bound, reported as the undecimated cloud. This field is the
    coarse prior's *decimated* whole-scene cloud, and its size is set by
    ``tools/build_lidar_init.py --target-points`` - a command-line argument,
    not a profile knob - so nothing the fallback can read predicts it.
    house0305 was built at a 2,000,000-point target and landed on 1,863,918
    against a cloud of 18,757,869. ``build_plan`` does not consume the field;
    it is carried for the report.

The view-count rule
-------------------
``capture_summary_v1``, in full:

1. Faces. A square fisheye camera is expanded into the four MipMap Face4
   pinhole views the recipe trains on (``plan_mipmap_face4``); any other
   camera contributes one face, its own pinhole view. The face count per image
   is therefore the trainer's, not the raw image count.
2. Points. The LiDAR cloud is decimated to :data:`DEFAULT_POINT_SAMPLE_BUDGET`
   points by a fixed global stride, and each tile takes the sample points
   inside its ``training_and_export_box``.
3. Projection. For every (image, face) pair, the tile's sample points are
   projected into the face with the face's own intrinsics. Points behind the
   face or outside its image are dropped.
4. Acceptance. The surviving points' bounding rectangle must be at least
   :data:`MINIMUM_FACE_RECTANGLE_PIXELS` pixels on both sides - the same
   ``AdaptiveTilingConfig.minimum_image_rectangle_pixels`` constant the
   production planner applies, so a face that catches only a sliver of a tile
   is not a training view here either.

``view_count`` is the number of accepted (image, face) pairs.

Measured against the real house0305 tile boxes, so that only the rule is under
test, it over-counts by +4.0% to +6.8% (2227/1907/1798/2409 against the as-run
2132/1829/1684/2317; +4.8% on the total). The bias is one-sided and has one
dominant cause: the production observation table is built from *visible* LiDAR
z-buffer samples (``build_lidar_face4_projected_observations``), so a point
behind a wall produces no observation. This rule has no occlusion test and
counts through walls.

The decimation in step 2 pushes the other way - a face that catches only a few
square metres of a tile can miss the sample entirely - so the two biases
partly cancel, and the error is *not* monotone in the sample budget. On the
same boxes: 12,500 points gives +0.9% to +2.4%, 50,000 gives +4.0% to +6.8%,
200,000 gives +5.1% to +9.4%. The denser sample is the more faithful reading
of the rule as written, and about +5% to +9% is the honest size of the missing
occlusion term; the small budget looks better only by accident. The default is
a cost choice, not an accuracy choice - do not tune it to make a number match.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cloudstudio3dgs_sdk.ingest.adapters import adapter_by_name, detect_adapter
from cloudstudio3dgs_sdk.ingest.bundle import CAPABILITY_LIDAR, DatasetBundle
from cloudstudio3dgs_sdk.ingest.tiling import (
    TilingRule,
    build_slab_tile_plan,
    histogram_from_las,
    slab_split,
)
from cloudstudio3dgs_sdk.plan import DatasetSummary, TileSummary, tile_count_for
from cloudstudio3dgs_sdk.profile import Profile

#: Bumped whenever a rule below changes, so a recorded estimate says which one
#: produced it.
DISCOVERY_RULE_VERSION = "capture_summary_v1"

#: Points kept from the cloud for the view test. Not an accuracy knob - see the
#: module docstring.
DEFAULT_POINT_SAMPLE_BUDGET = 50_000

#: ``AdaptiveTilingConfig.minimum_image_rectangle_pixels``. Copied rather than
#: imported so the fallback keeps working if the planner's defaults move under
#: it; a divergence is then visible here instead of silent.
MINIMUM_FACE_RECTANGLE_PIXELS = 256

_LAS_SUFFIXES = ("las", "laz")

_READ_CHUNK_POINTS = 2_000_000


class DiscoveryError(RuntimeError):
    """The capture does not carry what a planning-time summary needs."""


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEstimate:
    """An estimated :class:`DatasetSummary` plus everything behind it."""

    summary: DatasetSummary
    rule_version: str
    tile_plan: Mapping[str, Any]
    measurements: Mapping[str, Any]
    notes: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "cloudstudio3dgs_sdk_dataset_estimate",
            "rule_version": self.rule_version,
            "dataset": self.summary.as_json(),
            "measurements": dict(self.measurements),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines = [f"dataset estimate {self.summary.scene_tag} rule={self.rule_version}"]
        for key in sorted(self.measurements):
            lines.append(f"  {key}: {self.measurements[key]}")
        lines.append("  ESTIMATED - not what prepare() will produce:")
        for note in self.notes:
            lines.append(f"    - {note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Face:
    """One pinhole view of one camera: the unit ``view_count`` counts."""

    camera_id: str
    face_id: str
    rotation: np.ndarray  # face -> camera, d_cam = R @ d_face
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def _faces_for_camera(camera: Any) -> tuple[_Face, ...]:
    """Face4 for a square fisheye, otherwise the camera's own single view.

    Face4 is what the recipe trains on: the trainer never samples the raw
    fisheye, it samples the four rectified faces, and the production view
    count is a count of those. A camera the recipe would not re-face
    contributes one view per image instead, which is the right unit for a
    pinhole capture.
    """
    if camera.is_fisheye and camera.width == camera.height:
        from cloudstudio_3dgs.geometry.fisheye_faces import plan_mipmap_face4

        faces = plan_mipmap_face4((camera.width, camera.height))
        return tuple(
            _Face(
                camera_id=camera.camera_id,
                face_id=str(spec.face_id),
                rotation=np.asarray(spec.R_face, dtype=np.float64),
                fx=float(spec.K_face[0, 0]),
                fy=float(spec.K_face[1, 1]),
                cx=float(spec.K_face[0, 2]),
                cy=float(spec.K_face[1, 2]),
                width=int(spec.width),
                height=int(spec.height),
            )
            for spec in faces
        )
    return (
        _Face(
            camera_id=camera.camera_id,
            face_id="whole_image",
            rotation=np.eye(3, dtype=np.float64),
            fx=float(camera.fx),
            fy=float(camera.fy),
            cx=float(camera.cx),
            cy=float(camera.cy),
            width=int(camera.width),
            height=int(camera.height),
        ),
    )


def face_plan(bundle: DatasetBundle) -> dict[str, tuple[_Face, ...]]:
    """camera_id -> the faces the recipe would train on for that camera."""
    return {camera.camera_id: _faces_for_camera(camera) for camera in bundle.cameras}


# --------------------------------------------------------------------------
# Streaming the cloud
# --------------------------------------------------------------------------


def _point_cloud_path(bundle: DatasetBundle) -> Path:
    if bundle.point_cloud is None or CAPABILITY_LIDAR not in bundle.capabilities:
        raise DiscoveryError(
            f"{bundle.dataset_id}: the capture declares no LiDAR cloud, and the tile boxes, "
            "the per-tile initialisation counts and the per-tile view counts are all derived "
            "from one. Plan this scene with dataset=DatasetSummary(...) instead."
        )
    path = Path(bundle.point_cloud.path)
    if bundle.point_cloud.format.lower() not in _LAS_SUFFIXES:
        raise DiscoveryError(
            f"{path}: the planning-time summary streams LAS/LAZ only, this cloud is "
            f"'{bundle.point_cloud.format}'."
        )
    if not path.is_file():
        raise DiscoveryError(f"the capture's point cloud is missing: {path}")
    return path


def _count_and_sample(
    path: Path, boxes: Sequence[np.ndarray], *, budget: int
) -> tuple[list[int], np.ndarray]:
    """One pass: exact points per box, plus a stride-decimated sample.

    The stride comes from the header's point count, so it is global rather
    than per chunk: the sample does not depend on the reader's chunk size, and
    two runs on the same file take the same points. Only the sample is held in
    memory; the counts accumulate chunk by chunk.
    """
    if budget < 1:
        raise DiscoveryError("point_sample_budget must be at least 1")
    import laspy

    counts = [0] * len(boxes)
    kept: list[np.ndarray] = []
    offset = 0
    with laspy.open(path) as reader:
        total = int(reader.header.point_count)
        stride = max(1, total // budget)
        for chunk in reader.chunk_iterator(_READ_CHUNK_POINTS):
            points = np.column_stack([chunk.x, chunk.y, chunk.z]).astype(np.float64)
            for index, box in enumerate(boxes):
                inside = np.all((points >= box[0]) & (points <= box[1]), axis=1)
                counts[index] += int(np.count_nonzero(inside))
            first = (-offset) % stride
            kept.append(points[first::stride])
            offset += len(points)
    sample = np.concatenate(kept) if kept else np.zeros((0, 3), dtype=np.float64)
    return counts, sample


# --------------------------------------------------------------------------
# The view rule
# --------------------------------------------------------------------------


def _train_images(bundle: DatasetBundle) -> tuple[tuple[Any, ...], bool]:
    """The images the recipe would train on, and whether the capture said so."""
    declared = tuple(image for image in bundle.images if image.split == "train")
    if declared:
        return declared, True
    return tuple(bundle.images), False


def tile_view_counts(
    boxes: Sequence[np.ndarray],
    sample: np.ndarray,
    bundle: DatasetBundle,
    *,
    minimum_rectangle_px: int = MINIMUM_FACE_RECTANGLE_PIXELS,
) -> list[int]:
    """Per-tile count of (image, face) pairs that can see the tile.

    See the module docstring for the rule and its measured error. Nothing here
    tests occlusion, so the result is an upper-biased estimate.
    """
    faces_by_camera = face_plan(bundle)
    images, _ = _train_images(bundle)
    poses = np.asarray([[list(row) for row in image.c2w] for image in images], dtype=np.float64)
    centres = poses[:, :3, 3]
    rotations = poses[:, :3, :3]
    camera_ids = [image.camera_id for image in images]

    counts: list[int] = []
    for box in boxes:
        inside = np.all((sample >= box[0]) & (sample <= box[1]), axis=1)
        tile_points = sample[inside]
        total = 0
        if len(tile_points):
            for index, camera_id in enumerate(camera_ids):
                relative = (tile_points - centres[index]) @ rotations[index]
                for face in faces_by_camera[camera_id]:
                    if _face_sees(relative, face, minimum_rectangle_px):
                        total += 1
        counts.append(total)
    return counts


def _face_sees(relative: np.ndarray, face: _Face, minimum_rectangle_px: int) -> bool:
    """Does this face catch a rectangle of the tile worth training on?"""
    in_face = relative @ face.rotation
    depth = in_face[:, 2]
    front = depth > 1e-6
    if not front.any():
        return False
    u = in_face[front, 0] / depth[front] * face.fx + face.cx
    v = in_face[front, 1] / depth[front] * face.fy + face.cy
    visible = (u >= 0.0) & (u < face.width) & (v >= 0.0) & (v < face.height)
    if not visible.any():
        return False
    width = float(np.ceil(u[visible].max()) - np.floor(u[visible].min()))
    height = float(np.ceil(v[visible].max()) - np.floor(v[visible].min()))
    return width >= minimum_rectangle_px and height >= minimum_rectangle_px


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def load_capture(
    dataset_root: Path | str, *, adapter: str | None = None, run_dir: Path | str | None = None
) -> DatasetBundle:
    """Load the capture for planning: no content hashes, nothing derived.

    Hashing 884 images and a 675 MB cloud is what ``prepare()`` does to bind a
    scene it will train; a cost estimate has nothing to bind, so the hashes are
    skipped wherever the adapter offers to skip them. ``run_dir`` is the processed
    half of a split capture (house0614 keeps poses and cloud apart from the images).
    """
    root = Path(dataset_root)
    module = adapter_by_name(adapter) if adapter else detect_adapter(root)
    parameters = inspect.signature(module.load).parameters
    kwargs: dict[str, object] = {
        name: False
        for name in ("hash_images", "hash_point_cloud")
        if name in parameters
    }
    if run_dir is not None:
        if "run_dir" not in parameters:
            raise DiscoveryError(
                f"adapter {getattr(module, 'NAME', module.__name__)} does not read a separate run "
                "directory; drop --run-dir or name the adapter that does"
            )
        kwargs["run_dir"] = Path(run_dir)
    return module.load(root, **kwargs)


def estimate_dataset_summary(
    dataset_root: Path | str,
    profile: Profile,
    *,
    scene_tag: str | None = None,
    adapter: str | None = None,
    run_dir: Path | str | None = None,
    bundle: DatasetBundle | None = None,
    tiling_rule: TilingRule | None = None,
    point_sample_budget: int = DEFAULT_POINT_SAMPLE_BUDGET,
    vram_gib: float | None = None,
) -> DatasetEstimate:
    """Summarise an un-prepared capture well enough to cost a delivery.

    The returned :class:`DatasetSummary` carries ``estimated=True`` and the
    notes below, and :func:`cloudstudio3dgs_sdk.plan.build_plan` and the
    project refuse to *train* from it: it is for ``--dry-run`` and
    ``preflight`` only.
    """
    capture = bundle if bundle is not None else load_capture(dataset_root, adapter=adapter, run_dir=run_dir)
    cloud = _point_cloud_path(capture)

    rule = tiling_rule or TilingRule(tile_count=int(profile.tiling["reference_tile_count"]))
    histogram = histogram_from_las(cloud, bins=rule.histogram_bins)
    if tiling_rule is None:
        # As many tiles as the cloud needs for the cap rule, never fewer than the reference;
        # past the reference count one axis cannot hold them, so the cut becomes a grid.
        count = tile_count_for(profile, int(histogram.point_count), vram_gib=vram_gib)
        reference = int(profile.tiling["reference_tile_count"])
        rule = dataclasses.replace(rule, tile_count=count, layout="grid" if count > reference else rule.layout)
    slab = slab_split(histogram, rule)
    plan = build_slab_tile_plan(
        slab,
        point_cloud_sha256=(capture.point_cloud.sha256 if capture.point_cloud else None),
        source_bindings={"dataset_id": capture.dataset_id, "adapter": capture.adapter},
    )

    boxes = [
        np.asarray(entry["training_and_export_box"], dtype=np.float64) for entry in plan["tiles"]
    ]
    init_counts, sample = _count_and_sample(cloud, boxes, budget=point_sample_budget)
    views = tile_view_counts(boxes, sample, capture)

    images, split_declared = _train_images(capture)
    faces_by_camera = face_plan(capture)
    train_view_count = sum(len(faces_by_camera[image.camera_id]) for image in images)

    tiles = tuple(
        TileSummary(
            tile_id=int(entry["tile_id"]),
            name=str(entry["name"]),
            view_count=int(view_count),
            init_point_count=int(init_count),
        )
        for entry, view_count, init_count in zip(plan["tiles"], views, init_counts)
    )

    notes = _notes(
        capture=capture,
        rule=rule,
        slab_axis=str(plan["input"]["split_axis"]),
        split_declared=split_declared,
        image_count=len(images),
        sample_size=int(len(sample)),
    )
    summary = DatasetSummary(
        scene_tag=scene_tag or capture.dataset_id,
        tiles=tiles,
        train_view_count=int(train_view_count),
        global_init_point_count=int(histogram.point_count),
        lidar_point_count=int(histogram.point_count),
        has_reference_model=False,
        estimated=True,
        estimate_notes=notes,
    )
    measurements = {
        "adapter": capture.adapter,
        "point_cloud": str(cloud),
        "lidar_point_count": int(histogram.point_count),
        "images_considered_train": len(images),
        "faces_per_image": sorted({len(f) for f in faces_by_camera.values()}),
        "split_axis": str(plan["input"]["split_axis"]),
        "cuts_m": list(plan["input"]["cuts_m"]),
        "point_sample_used": int(len(sample)),
        "tile_count": len(tiles),
    }
    return DatasetEstimate(
        summary=summary,
        rule_version=DISCOVERY_RULE_VERSION,
        tile_plan=plan,
        measurements=measurements,
        notes=notes,
    )


def _notes(
    *,
    capture: DatasetBundle,
    rule: TilingRule,
    slab_axis: str,
    split_declared: bool,
    image_count: int,
    sample_size: int,
) -> tuple[str, ...]:
    """The sentences the dry-run transcript prints under ESTIMATED."""
    split = (
        "the capture declares a train split, so only its train images were counted"
        if split_declared
        else (
            f"the capture declares no split, so all {image_count} images were counted as train; "
            "prepare() will hold some out, which lowers train_view_count and every per-tile "
            "view_count"
        )
    )
    return (
        (
            f"tile boxes: {rule.tile_count} grid cells (equal-point-count slabs along {slab_axis}, "
            "then equal-point-count strips inside each) "
            if rule.layout == "grid"
            else f"tile boxes: {rule.tile_count} equal-point-count slabs along {slab_axis} "
        )
        + f"({rule.__class__.__name__} {rule.to_dict()['rule']}), not the projected-pixel kd "
        "planner prepare() runs; different boxes mean different per-tile numbers",
        "init_point_count: exact count of cloud points inside each estimated box - exact for "
        "that box, estimated only because the box is",
        f"view_count: {DISCOVERY_RULE_VERSION} - a face counts when a {sample_size}-point "
        f"decimated sample of the tile projects into it with a bounding rectangle of at least "
        f"{MINIMUM_FACE_RECTANGLE_PIXELS}px per side. No occlusion test, so it counts through "
        "walls and reads high; measured +4.0% to +6.8% per tile against the house0305 as-run "
        "counts when run on the as-run boxes",
        f"train_view_count: {split}",
        "global_init_point_count: the undecimated cloud, an UPPER BOUND. The coarse prior's "
        "decimation target is a build_lidar_init.py argument, not a profile knob, so nothing "
        "here predicts it (house0305: 1863918 actual against 18757869 reported here)",
        f"nothing above was produced by prepare(); adapter '{capture.adapter}' read the capture "
        "and no cache was built",
    )


__all__ = [
    "DEFAULT_POINT_SAMPLE_BUDGET",
    "DISCOVERY_RULE_VERSION",
    "MINIMUM_FACE_RECTANGLE_PIXELS",
    "DatasetEstimate",
    "DiscoveryError",
    "estimate_dataset_summary",
    "face_plan",
    "load_capture",
    "tile_view_counts",
]
