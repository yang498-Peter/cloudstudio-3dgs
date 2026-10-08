"""``CachePlan``: every cache the training recipe consumes, in dependency order.

The reference recipe (``C:\\Peter\\3dgs-runs\\house0305_sop\\tile1_B5_cap6_20k.json``)
names fourteen signed artifacts.  None of them is optional at training time and
each binds to the SHA256 of the ones above it, so rebuilding one silently
invalidates everything below.  This module states that graph once: what builds
each cache, on CPU or GPU, what it costs, and which SHA bindings prove a cache
still matches its inputs.

Two rules shape the design:

* **Skip what already matches.**  A cache whose signed manifest exists *and*
  whose recorded bindings equal the current upstream SHAs is not rebuilt.  A
  cache whose bindings disagree is reported as ``STALE``, never quietly reused.
* **Never start CUDA from here.**  ``build(dry_run=False)`` runs the CPU
  builders and refuses the GPU ones with an explicit message; the SDK runner
  owns device placement, queueing and the VRAM budget.

Cost figures come from the house0305 v9 rebuild (884 train images, 3536 faces);
each spec records whether its number was measured from that run's artifacts and
logs, or estimated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .bundle import (
    CAPABILITY_FISHEYE,
    CAPABILITY_LIDAR,
    CAPABILITY_TIMESTAMPS,
    DatasetBundle,
)
from .errors import DatasetIncompleteError, GpuStepRequired, IngestError

REPO_ROOT = Path(__file__).resolve().parents[2]

CPU = "cpu"
GPU = "gpu"

STATUS_PRESENT = "PRESENT"
STATUS_STALE = "STALE"
STATUS_MISSING = "MISSING"
STATUS_BLOCKED = "BLOCKED"


class CachePlanError(IngestError):
    """The plan cannot be built or executed as requested."""


@dataclass(frozen=True)
class Binding:
    """``manifest_key`` inside this cache's manifest must equal the dep's own sha."""

    manifest_key: str  # dotted path, e.g. "source_identity.dataset_manifest_sha256"
    depends_on: str  # cache name
    optional: bool = False


@dataclass(frozen=True)
class CacheSpec:
    name: str
    title: str
    device: str
    builder: str
    command: tuple[str, ...]
    manifest: Path
    root: Path | None
    sha_key: str
    depends_on: tuple[str, ...] = ()
    bindings: tuple[Binding, ...] = ()
    requires_capabilities: tuple[str, ...] = ()
    estimated_minutes: float = 0.0
    cost_basis: str = "estimated"
    output_gib: float = 0.0
    tile_id: int | None = None
    note: str = ""
    # (dotted manifest key, value) pairs the manifest must carry to count as present. A signed
    # result can still be a refusal: an AT report that did not converge, a time-sync audit
    # that found an offset. Without this the graph would call them built and never re-run them.
    requires: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.device not in (CPU, GPU):
            raise CachePlanError(f"{self.name}: device must be 'cpu' or 'gpu'")


# Every cost below was measured (or estimated, see each spec's cost_basis) on house0305 v9:
# 886 images, 3536 train faces, 4 tiles. A scene of another size scales from here by the
# basis named next to each cache, so a capture seven times larger is costed seven times
# larger before anything is built instead of at house0305's size.
REFERENCE_IMAGES = 886
REFERENCE_TRAIN_FACES = 3536
REFERENCE_TILES = 4
FACES_PER_IMAGE = {"mipmap_face4": 4}
#: The default rig-frame split holds out about a tenth of the images (house0305 v8: 90 of
#: 886); the validation graph is costed at that share of the faces.
VALIDATION_FRACTION = 0.10

# name: (minutes, GiB, basis). "images": every posed image; "faces": this split's faces;
# "faces_per_tile": one tile's share of the faces (per-tile caches); "fixed": size-free.
REFERENCE_COSTS: Mapping[str, tuple[float, float, str]] = {
    "dataset_manifest": (2.0, 0.002, "images"),
    "mask_manifest": (1.0, 0.02, "images"),
    "person_mask_manifest": (25.0, 0.9, "images"),
    "depth_cache": (95.0, 9.8, "images"),
    "split_manifest": (0.5, 0.001, "fixed"),
    "face_cache": (31.0, 25.0, "faces"),
    "renderer_mask": (6.0, 0.002, "faces"),
    "face_lidar_geometry": (72.0, 7.6, "faces"),
    "mono_depth": (53.0, 1.6, "faces"),
    "sky_masks": (168.0, 0.06, "faces"),
    "sky_masks_refined": (3.75, 0.19, "faces"),
    "tile_plan": (1.0, 0.01, "fixed"),
    "tile_inputs": (3.0, 0.28, "images"),
    "tile_geometry": (40.0, 0.96, "images"),
    "tile_ownership": (6.0, 0.19, "faces_per_tile"),
    "view_backgrounds": (12.0, 2.5, "faces_per_tile"),
    # The independent-AT pose route (house0305_at_v2 re-run of 2026-09-22 unless noted).
    "raw_dataset_manifest": (2.0, 0.002, "images"),
    "raw_mask_manifest": (1.0, 0.02, "images"),
    "raw_person_mask_manifest": (25.0, 0.9, "images"),
    "raw_split_manifest": (0.5, 0.001, "fixed"),
    "at_pairs": (0.5, 0.001, "images"),
    "at_features_raw": (11.5, 1.3, "images"),
    "at_features": (13.0, 1.3, "images"),
    "at_triangulation": (5.0, 0.6, "images"),
    # UK capture 2026-10-08: 112 min for 80 outer iterations at 1044 images (1.4 min each), still
    # short of the 1e-6 / 1e-5 tolerances and closing ~1% per iteration; ~150 iterations expected.
    "at_solve": (180.0, 0.3, "images"),
    # runbook A1/A2 scale (3000 steps at factor 4, then a 6-offset sweep): estimates.
    "timesync_model": (15.0, 0.3, "images"),
    "timesync": (20.0, 0.01, "images"),
}

POSE_ROUTE_RAW = "raw_capture_poses"
POSE_ROUTE_AT = "independent_at"
#: Caches only the independent-AT route builds; the raw route's estimate leaves them out.
AT_ROUTE_CACHES = (
    "raw_dataset_manifest",
    "raw_mask_manifest",
    "raw_person_mask_manifest",
    "raw_split_manifest",
    "at_pairs",
    "at_features_raw",
    "at_features",
    "at_triangulation",
    "at_solve",
    "timesync_model",
    "timesync",
)


@dataclass(frozen=True)
class SceneScale:
    """How much bigger than house0305 one split of a scene is, per cost basis."""

    images: int
    faces: int
    tiles: int

    @staticmethod
    def of(bundle: Any, profile: "CacheProfile") -> "SceneScale":
        images = len(bundle.images)
        faces = images * FACES_PER_IMAGE.get(profile.face_plan, 4)
        if profile.split != "train":
            faces = max(1, int(round(faces * VALIDATION_FRACTION)))
        return SceneScale(images=images, faces=faces, tiles=max(1, len(profile.tile_ids)))

    def factor(self, basis: str) -> float:
        if basis == "images":
            return self.images / REFERENCE_IMAGES
        if basis == "faces":
            return self.faces / REFERENCE_TRAIN_FACES
        if basis == "faces_per_tile":
            return (self.faces / REFERENCE_TRAIN_FACES) * (REFERENCE_TILES / self.tiles)
        if basis == "fixed":
            return 1.0
        raise CachePlanError(f"unknown cost basis {basis!r}")

    def minutes(self, name: str) -> float:
        minutes, _, basis = REFERENCE_COSTS[name]
        return minutes * self.factor(basis)

    def gib(self, name: str) -> float:
        _, gib, basis = REFERENCE_COSTS[name]
        return gib * self.factor(basis)


@dataclass
class CacheProfile:
    """Where caches live and which optional ones this profile wants.

    Deliberately duck-typed: :meth:`from_any` accepts the SDK ``Profile`` object
    owned by the profile/plan layer, a plain mapping, or nothing at all, so the
    ingestion layer never hard-depends on that module's field names.
    """

    dataset_root: Path
    cache_root: Path
    run_root: Path
    recording_root: Path
    source_run_dir: Path
    split: str = "train"
    tile_count: int = 4
    # raw_capture_poses: train on the capture's own poses (no gate owed). independent_at: raw
    # tier -> features -> triangulation -> independent AT -> training manifest, then the
    # signed gate chain (cloudstudio3dgs_sdk.ingest.gates).
    pose_route: str = POSE_ROUTE_RAW
    # "grid" once a scene needs more tiles than one axis can hold (see ingest.tiling)
    tile_layout: str = "slab"
    tile_ids: tuple[int, ...] = ()
    person_masks: bool = True
    mono_depth: bool = True
    sky_masks: bool = True
    # Photometric refinement of the SegFormer sky label (tools/refine_sky_masks.py parameters:
    # dark_ratio, edge_ratio, dilate_px). When set, the trainer reads the refined cache.
    sky_mask_refinement: Mapping[str, Any] | None = None
    tile_ownership: bool = True
    view_backgrounds: bool = True
    fov_deg: float = 190.0
    # The recipe trains on the four MipMap-aligned faces; build_face_cache.py defaults to
    # adaptive_full_fov, a different face layout the trainer's caches do not match.
    face_plan: str = "mipmap_face4"
    visibility_cell_px: int = 6
    workers: int = 10
    threads: int = 6
    device: str = "cuda:0"
    python: str = field(default_factory=lambda: sys.executable)
    repo_root: Path = REPO_ROOT
    da2_model_source: Path | None = None
    da2_checkpoint: Path | None = None
    person_weights: Path | None = None
    trainer_config: Path | None = None
    manual_split: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "dataset_root",
            "cache_root",
            "run_root",
            "recording_root",
            "source_run_dir",
            "repo_root",
        ):
            setattr(self, name, Path(getattr(self, name)))
        if not self.tile_ids:
            self.tile_ids = tuple(range(self.tile_count))
        else:
            self.tile_ids = tuple(int(value) for value in self.tile_ids)
            self.tile_count = max(self.tile_count, len(self.tile_ids))

    @staticmethod
    def from_any(source: Any, **overrides: Any) -> "CacheProfile":
        """Read a profile object or mapping, keeping only fields we know."""

        known = set(CacheProfile.__dataclass_fields__)
        values: dict[str, Any] = {}
        if isinstance(source, Mapping):
            values.update({k: v for k, v in source.items() if k in known})
        elif source is not None:
            for key in known:
                if hasattr(source, key):
                    values[key] = getattr(source, key)
        values.update({k: v for k, v in overrides.items() if k in known})
        missing = [
            key
            for key in (
                "dataset_root",
                "cache_root",
                "run_root",
                "recording_root",
                "source_run_dir",
            )
            if key not in values
        ]
        if missing:
            raise CachePlanError(
                "cache profile is missing required roots: " + ", ".join(missing)
            )
        return CacheProfile(**values)

    def tool(self, name: str) -> str:
        return str(self.repo_root / "tools" / name)


def _dotted(payload: Mapping[str, Any], key: str) -> Any:
    node: Any = payload
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def build_cache_specs(bundle: DatasetBundle, profile: CacheProfile) -> list[CacheSpec]:
    """The full inventory for one split of one dataset."""

    scale = SceneScale.of(bundle, profile)
    dataset = profile.dataset_root
    cache = profile.cache_root
    run = profile.run_root
    split = profile.split
    masks_root = dataset / "masks"
    depth_root = dataset / "depth"
    person_root = dataset / "person_masks"
    face_root = cache / f"face4_{split}"
    face_lidar_root = cache / f"face4_lidar_{split}_vis{profile.visibility_cell_px}"
    da2_root = cache / f"da2_{split}"
    sky_root = cache / f"sky_mask_{split}"
    sky_refined_root = cache / f"sky_mask_{split}_refined"
    tile_plan_root = run / "tile_plan"
    tile_inputs_root = run / "tile_inputs"
    tile_geometry_root = run / "tile_geometry"
    ownership_root = run / "tile_ownership"
    backgrounds_root = run / "tile_backgrounds"

    dataset_manifest = dataset / "dataset_manifest.json"
    mask_manifest = masks_root / "mask_manifest.json"
    # build_person_masks.py / rebind_person_mask_base.py publish it inside --output
    person_manifest = person_root / "person_mask_manifest.json"
    depth_manifest = depth_root / "depth_manifest.json"
    split_manifest = cache / "split_manifest.json"
    face_manifest = face_root / "face_manifest.json"
    renderer_manifest = cache / f"renderer_mask_{split}.json"
    face_lidar_manifest = face_lidar_root / "face_lidar_geometry_manifest.json"
    da2_manifest = da2_root / "mono_depth_manifest.json"
    sky_manifest = sky_root / f"sky_mask_{split}.json"
    sky_refined_manifest = sky_refined_root / f"sky_mask_{split}.json"
    tile_plan_manifest = tile_plan_root / "adaptive_tile_plan.json"
    tile_inputs_manifest = tile_inputs_root / "tile_inputs_manifest.json"
    tile_geometry_manifest = tile_geometry_root / "tile_geometry_manifest.json"

    python = profile.python
    specs: list[CacheSpec] = [
        CacheSpec(
            name="dataset_manifest",
            title="Signed capture manifest (cameras, poses, per-image sha, point cloud)",
            device=CPU,
            builder="cloudstudio_3dgs.data.manifest",
            command=(
                python,
                "-m",
                "cloudstudio_3dgs.data.manifest",
                "--recording",
                str(profile.recording_root),
                "--run",
                str(profile.source_run_dir),
                "--output",
                str(dataset),
            ),
            manifest=dataset_manifest,
            root=dataset,
            sha_key="manifest_sha256",
            estimated_minutes=scale.minutes("dataset_manifest"),
            cost_basis="estimated (SHA256 runs at ~1 GB/s here; 3.4 GB images + 0.7 GB LAS)",
            output_gib=scale.gib("dataset_manifest"),
        ),
        CacheSpec(
            name="mask_manifest",
            title="Per-image fisheye validity masks",
            device=CPU,
            builder="tools/build_per_image_masks.py",
            command=(
                python,
                profile.tool("build_per_image_masks.py"),
                "--manifest",
                str(dataset_manifest),
                "--output",
                str(masks_root),
            ),
            manifest=mask_manifest,
            root=masks_root,
            sha_key="mask_manifest_sha256",
            depends_on=("dataset_manifest",),
            bindings=(Binding("dataset_manifest_sha256", "dataset_manifest"),),
            estimated_minutes=scale.minutes("mask_manifest"),
            cost_basis="measured (v8 artifact span: manifest 18:18 -> masks 18:19)",
            output_gib=scale.gib("mask_manifest"),
        ),
        CacheSpec(
            name="person_mask_manifest",
            title="Mask R-CNN person masks (privacy + moving-subject removal)",
            device=GPU,
            builder="tools/build_person_masks.py",
            command=(
                python,
                profile.tool("build_person_masks.py"),
                "--manifest",
                str(dataset_manifest),
                "--base-mask-manifest",
                str(mask_manifest),
                "--recording-root",
                str(profile.recording_root),
                "--weights",
                str(profile.person_weights or "<person_weights>"),
                "--output",
                str(person_root),
                "--device",
                profile.device,
            ),
            manifest=person_manifest,
            root=person_root,
            sha_key="person_mask_manifest_sha256",
            depends_on=("dataset_manifest", "mask_manifest"),
            bindings=(
                Binding("dataset_manifest_sha256", "dataset_manifest"),
                Binding("base_mask_manifest_sha256", "mask_manifest"),
            ),
            estimated_minutes=scale.minutes("person_mask_manifest"),
            cost_basis="estimated (886 images through maskrcnn_resnet50_fpn_v2 at 800 px)",
            output_gib=scale.gib("person_mask_manifest"),
            note="Manual review pass (tools/finalize_person_mask_review.py) is not in this plan.",
        ),
        CacheSpec(
            name="depth_cache",
            title="Sparse LiDAR range cache on the raw fisheye frames",
            device=CPU,
            builder="tools/build_depth_cache.py",
            command=(
                python,
                profile.tool("build_depth_cache.py"),
                "--manifest",
                str(dataset_manifest),
                "--mask-manifest",
                str(mask_manifest),
                "--point-cloud",
                str(bundle.point_cloud.path if bundle.point_cloud else "<point_cloud>"),
                "--output",
                str(depth_root),
                "--workers",
                str(profile.workers),
            ),
            manifest=depth_manifest,
            root=depth_root,
            sha_key="depth_manifest_sha256",
            depends_on=("dataset_manifest", "mask_manifest"),
            bindings=(
                Binding("dataset_manifest_sha256", "dataset_manifest"),
                Binding("mask_manifest_sha256", "mask_manifest"),
            ),
            requires_capabilities=(CAPABILITY_LIDAR,),
            estimated_minutes=scale.minutes("depth_cache"),
            cost_basis="estimated (v8 artifact span masks 18:19 -> depth 20:16 is an upper bound)",
            output_gib=scale.gib("depth_cache"),
        ),
        CacheSpec(
            name="split_manifest",
            title="Rig-frame train/val/golden split",
            device=CPU,
            builder="tools/build_split_manifest.py",
            command=(
                python,
                profile.tool("build_split_manifest.py"),
                "--manifest",
                str(dataset_manifest),
                "--output",
                str(split_manifest),
            )
            + (
                ("--manual", str(profile.manual_split))
                if profile.manual_split
                else ()
            ),
            manifest=split_manifest,
            root=cache,
            sha_key="split_manifest_sha256",
            depends_on=("dataset_manifest",),
            bindings=(Binding("dataset_manifest_sha256", "dataset_manifest"),),
            requires_capabilities=(CAPABILITY_TIMESTAMPS,),
            estimated_minutes=scale.minutes("split_manifest"),
            cost_basis="measured (v9 artifact span, under a minute)",
            output_gib=scale.gib("split_manifest"),
        ),
        CacheSpec(
            name="face_cache",
            title="Face4 pinhole faces warped out of the fisheye images",
            device=CPU,
            builder="tools/build_face_cache.py",
            command=(
                python,
                profile.tool("build_face_cache.py"),
                "--dataset-manifest",
                str(dataset_manifest),
                "--recording-root",
                str(profile.recording_root),
                "--mask-manifest",
                str(mask_manifest),
                "--mask-root",
                str(masks_root),
                "--split-manifest",
                str(split_manifest),
                "--person-mask-manifest",
                str(person_manifest),
                "--person-mask-root",
                str(person_root),
                "--depth-manifest",
                str(depth_manifest),
                "--depth-root",
                str(depth_root),
                "--fov-deg",
                str(profile.fov_deg),
                "--face-plan",
                profile.face_plan,
                "--split",
                split,
                "--output",
                str(face_root),
            ),
            manifest=face_manifest,
            root=face_root,
            sha_key="face_manifest_sha256",
            depends_on=(
                "dataset_manifest",
                "mask_manifest",
                "person_mask_manifest",
                "depth_cache",
                "split_manifest",
            ),
            bindings=(
                Binding("source_identity.dataset_manifest_sha256", "dataset_manifest"),
                Binding("source_identity.mask_manifest_sha256", "mask_manifest"),
                Binding(
                    "source_identity.person_mask_manifest_sha256",
                    "person_mask_manifest",
                    optional=True,
                ),
                Binding("source_identity.depth_manifest_sha256", "depth_cache"),
                Binding("source_identity.split_manifest_sha256", "split_manifest"),
            ),
            requires_capabilities=(CAPABILITY_FISHEYE,),
            estimated_minutes=scale.minutes("face_cache"),
            cost_basis="measured (v9: 02:14 -> 02:45 for 884 images -> 3536 faces)",
            output_gib=scale.gib("face_cache"),
        ),
        CacheSpec(
            name="renderer_mask",
            title="Renderer forward-visibility mask manifest over the Face4 cache",
            device=CPU,
            builder="tools/build_renderer_mask_manifest.py",
            command=(
                python,
                profile.tool("build_renderer_mask_manifest.py"),
                "--face-manifest",
                str(face_manifest),
                "--face-cache-root",
                str(face_root),
                "--output",
                str(renderer_manifest),
            ),
            manifest=renderer_manifest,
            root=face_root,
            sha_key="renderer_mask_manifest_sha256",
            depends_on=("face_cache",),
            bindings=(Binding("source_face_manifest_sha256", "face_cache"),),
            estimated_minutes=scale.minutes("renderer_mask"),
            cost_basis="measured (v9: 02:46 -> 02:52, train + val, dominated by artifact SHA checks)",
            output_gib=scale.gib("renderer_mask"),
        ),
        CacheSpec(
            name="face_lidar_geometry",
            title="Per-face sparse LiDAR range + hidden-point removal",
            device=CPU,
            builder="tools/build_face4_lidar_geometry.py",
            command=(
                python,
                profile.tool("build_face4_lidar_geometry.py"),
                "--face-manifest",
                str(face_manifest),
                "--face-root",
                str(face_root),
                "--dataset-manifest",
                str(dataset_manifest),
                "--depth-manifest",
                str(depth_manifest),
                "--depth-root",
                str(depth_root),
                "--visibility-cell-px",
                str(profile.visibility_cell_px),
                "--workers",
                str(profile.workers),
                "--output",
                str(face_lidar_root),
            ),
            manifest=face_lidar_manifest,
            root=face_lidar_root,
            sha_key="face_lidar_geometry_manifest_sha256",
            depends_on=("face_cache", "depth_cache", "dataset_manifest"),
            bindings=(
                Binding("source_face_manifest_sha256", "face_cache"),
                Binding("source_depth_manifest_sha256", "depth_cache"),
                Binding("dataset_manifest_sha256", "dataset_manifest"),
            ),
            requires_capabilities=(CAPABILITY_LIDAR,),
            estimated_minutes=scale.minutes("face_lidar_geometry"),
            cost_basis=(
                "measured (v9: 03:11 -> 04:23 with visibility_cell_px=6; the same "
                "builder without hidden-point removal took 19 min)"
            ),
            output_gib=scale.gib("face_lidar_geometry"),
        ),
        CacheSpec(
            name="mono_depth",
            title="Depth Anything V2 relative depth, affine-fitted to the LiDAR range",
            device=GPU,
            builder="tools/build_da2_face_cache.py",
            command=(
                python,
                profile.tool("build_da2_face_cache.py"),
                "--face-manifest",
                str(face_manifest),
                "--face-root",
                str(face_root),
                "--dataset-manifest",
                str(dataset_manifest),
                "--depth-manifest",
                str(depth_manifest),
                "--depth-root",
                str(depth_root),
                "--model-source",
                str(profile.da2_model_source or "<da2_model_source>"),
                "--checkpoint",
                str(profile.da2_checkpoint or "<da2_checkpoint>"),
                "--device",
                profile.device,
                "--output",
                str(da2_root),
            ),
            manifest=da2_manifest,
            root=da2_root,
            sha_key="mono_depth_manifest_sha256",
            depends_on=("face_cache", "depth_cache", "dataset_manifest"),
            bindings=(
                Binding("source_face_manifest_sha256", "face_cache"),
                Binding("lidar_depth_manifest_sha256", "depth_cache"),
                Binding("dataset_manifest_sha256", "dataset_manifest"),
            ),
            requires_capabilities=(CAPABILITY_LIDAR,),
            estimated_minutes=scale.minutes("mono_depth"),
            cost_basis="measured (v9 shard logs: 0.9 s/face; 5 shards finished in ~11 min wall)",
            output_gib=scale.gib("mono_depth"),
            note="Shardable: five concurrent shards then one assembling pass for the manifest.",
        ),
        CacheSpec(
            name="sky_masks",
            title="SegFormer ADE20K sky masks per face",
            device=CPU,
            builder="tools/build_sky_masks.py",
            command=(
                python,
                profile.tool("build_sky_masks.py"),
                "--face-manifest",
                str(face_manifest),
                "--face-cache-root",
                str(face_root),
                "--threads",
                str(profile.threads),
                "--output-root",
                str(sky_root),
            ),
            manifest=sky_manifest,
            root=sky_root,
            sha_key="sky_mask_manifest_sha256",
            depends_on=("face_cache",),
            bindings=(Binding("source_face_manifest_sha256", "face_cache"),),
            estimated_minutes=scale.minutes("sky_masks"),
            cost_basis="measured (v9 log: 21:46:06 -> 00:34:40 on CPU with 6 threads)",
            output_gib=scale.gib("sky_masks"),
            note="--allow-cuda turns this into a GPU step; the default is CPU on purpose.",
        ),
    ]
    refinement = profile.sky_mask_refinement
    if refinement:
        specs.append(
            CacheSpec(
                name="sky_masks_refined",
                title="Sky label refined against the photo (bright, smooth pixels stay sky)",
                device=CPU,
                builder="tools/refine_sky_masks.py",
                command=(
                    python,
                    profile.tool("refine_sky_masks.py"),
                    "--source-manifest",
                    str(sky_manifest),
                    "--source-root",
                    str(sky_root),
                    "--face-cache-manifest",
                    str(face_manifest),
                    "--face-cache-root",
                    str(face_root),
                    "--output-root",
                    str(sky_refined_root),
                    "--dark-ratio",
                    str(refinement["dark_ratio"]),
                    "--edge-ratio",
                    str(refinement["edge_ratio"]),
                    "--dilate-px",
                    str(refinement["dilate_px"]),
                    "--workers",
                    str(profile.threads),
                ),
                manifest=sky_refined_manifest,
                root=sky_refined_root,
                sha_key="sky_mask_manifest_sha256",
                depends_on=("sky_masks", "face_cache"),
                bindings=(
                    Binding("rule.refinement.source_sky_mask_manifest_sha256", "sky_masks"),
                    Binding("source_face_manifest_sha256", "face_cache"),
                ),
                estimated_minutes=scale.minutes("sky_masks_refined"),
                cost_basis="measured (house0305 refine_stats.json: 225 s for 3536 faces, 4 workers)",
                output_gib=scale.gib("sky_masks_refined"),
                note="A subset of the raw label; every trainer guard (erosion, LiDAR proximity) still applies.",
            )
        )
    specs += [
        CacheSpec(
            name="tile_plan",
            title="Tile boxes (SDK slab rule, or the projected-pixel kd planner)",
            device=CPU,
            builder="cloudstudio3dgs_sdk.ingest.cli tile",
            command=(
                python,
                "-m",
                "cloudstudio3dgs_sdk.ingest.cli",
                "tile",
                "--point-cloud",
                str(bundle.point_cloud.path if bundle.point_cloud else "<point_cloud>"),
                "--tile-count",
                str(profile.tile_count),
                *(("--layout", profile.tile_layout) if profile.tile_layout != "slab" else ()),
                "--output",
                str(tile_plan_manifest),
                # Bind the views: without them the plan is boxes only and every tile
                # materializes with no training view.
                "--dataset-manifest",
                str(dataset_manifest),
                "--depth-manifest",
                str(depth_manifest),
                "--depth-root",
                str(depth_root),
                "--face",
                str(face_manifest),
                str(face_root),
            ),
            manifest=tile_plan_manifest,
            root=tile_plan_root,
            sha_key="tile_plan_manifest_sha256",
            depends_on=("face_lidar_geometry",),
            bindings=(),
            requires_capabilities=(CAPABILITY_LIDAR,),
            estimated_minutes=scale.minutes("tile_plan"),
            cost_basis="measured (one streaming LAS pass: 2.7 s for 18.76 M points)",
            output_gib=scale.gib("tile_plan"),
            note=(
                "Boxes only need the cloud; the view rectangles come from the LiDAR depth "
                "projected into the Face4 views, which is why this sits after the face and "
                "depth caches. No readiness gate is read, so raw-pose captures tile too."
            ),
        ),
        CacheSpec(
            name="tile_inputs",
            title="Per-tile initialization PLY with the halo retained",
            device=CPU,
            builder="tools/materialize_lidar_tile_inputs.py",
            command=(
                python,
                profile.tool("materialize_lidar_tile_inputs.py"),
                "--tile-plan",
                str(tile_plan_manifest),
                "--source-las",
                str(bundle.point_cloud.path if bundle.point_cloud else "<point_cloud>"),
                "--expected-las-sha256",
                str(
                    bundle.point_cloud.sha256
                    if bundle.point_cloud and bundle.point_cloud.sha256
                    else "<point_cloud_sha256>"
                ),
                "--output",
                str(tile_inputs_root),
            ),
            manifest=tile_inputs_manifest,
            root=tile_inputs_root,
            sha_key="tile_inputs_manifest_sha256",
            depends_on=("tile_plan",),
            bindings=(Binding("tile_plan_manifest_sha256", "tile_plan"),),
            requires_capabilities=(CAPABILITY_LIDAR,),
            estimated_minutes=scale.minutes("tile_inputs"),
            cost_basis="measured (one LAS pass, 6.2 s for 4 boxes, plus 280 MB of PLY writes and SHA)",
            output_gib=scale.gib("tile_inputs"),
        ),
        CacheSpec(
            name="tile_geometry",
            title="K=7 spacing / K=30 PCA surface geometry per tile",
            device=CPU,
            builder="tools/build_mipmap_tile_geometry.py",
            command=(
                python,
                profile.tool("build_mipmap_tile_geometry.py"),
                "--tile-inputs",
                str(tile_inputs_manifest),
                "--tile-inputs-root",
                str(tile_inputs_root),
                "--output",
                str(tile_geometry_root),
            ),
            manifest=tile_geometry_manifest,
            root=tile_geometry_root,
            sha_key="tile_geometry_manifest_sha256",
            depends_on=("tile_inputs",),
            bindings=(Binding("tile_inputs_manifest_sha256", "tile_inputs"),),
            requires_capabilities=(CAPABILITY_LIDAR,),
            estimated_minutes=scale.minutes("tile_geometry"),
            cost_basis="estimated (K=30 PCA over 19.4 M halo-inclusive points, 964 MB of npz)",
            output_gib=scale.gib("tile_geometry"),
        ),
    ]

    for tile_id in profile.tile_ids:
        tile_name = f"Tile_{tile_id}"
        specs.append(
            CacheSpec(
                name=f"tile_ownership_{tile_id}",
                title=f"{tile_name}: per-view core-ownership masks",
                device=CPU,
                builder="tools/build_tile_ownership_masks.py",
                command=(
                    python,
                    profile.tool("build_tile_ownership_masks.py"),
                    "--config",
                    str(profile.trainer_config or "<trainer_config>"),
                    "--tile-id",
                    str(tile_id),
                    "--workers",
                    str(profile.workers),
                    "--output-root",
                    str(ownership_root / tile_name),
                ),
                manifest=ownership_root / tile_name / "tile_ownership_manifest.json",
                root=ownership_root / tile_name,
                sha_key="tile_ownership_manifest_sha256",
                depends_on=(
                    "face_cache",
                    "renderer_mask",
                    "face_lidar_geometry",
                    "tile_inputs",
                ),
                bindings=(
                    Binding("source_face_manifest_sha256", "face_cache"),
                    Binding("renderer_mask_manifest_sha256", "renderer_mask"),
                    Binding("face_lidar_geometry_manifest_sha256", "face_lidar_geometry"),
                    Binding("tile_inputs_manifest_sha256", "tile_inputs"),
                ),
                requires_capabilities=(CAPABILITY_LIDAR,),
                estimated_minutes=scale.minutes("tile_ownership"),
                cost_basis="measured (v9: 4-8 min per tile with 10 workers)",
                output_gib=scale.gib("tile_ownership"),
                tile_id=tile_id,
            )
        )
        specs.append(
            CacheSpec(
                name=f"view_backgrounds_{tile_id}",
                title=f"{tile_name}: per-view rendered backdrop (sky dome + stand-in)",
                device=GPU,
                builder="tools/build_tile_view_backgrounds.py",
                command=(
                    python,
                    profile.tool("build_tile_view_backgrounds.py"),
                    "--background-manifest",
                    str(backgrounds_root / "dome" / "background_manifest.json"),
                    "--background-root",
                    str(backgrounds_root / "dome"),
                    "--tile-inputs",
                    str(tile_inputs_manifest),
                    "--face-manifest",
                    str(face_manifest),
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--tile-id",
                    str(tile_id),
                    "--output-root",
                    str(backgrounds_root / tile_name),
                ),
                manifest=backgrounds_root / tile_name / "background_manifest.json",
                root=backgrounds_root / tile_name,
                sha_key="manifest_sha256",
                depends_on=("tile_inputs", "face_cache"),
                bindings=(
                    Binding("source_tile_inputs_manifest_sha256", "tile_inputs"),
                ),
                estimated_minutes=scale.minutes("view_backgrounds"),
                cost_basis="measured (v9 log: 1829 views rendered in 1.5 min, plus stand-in assembly)",
                output_gib=scale.gib("view_backgrounds"),
                tile_id=tile_id,
                note=(
                    "Needs trained neighbour-tile checkpoints for the stand-in; on a "
                    "first pass use tools/build_sky_dome.py output only."
                ),
            )
        )

    if profile.pose_route == POSE_ROUTE_AT:
        specs = _independent_at_specs(
            specs,
            bundle=bundle,
            profile=profile,
            scale=scale,
            dataset=dataset,
            cache=cache,
            dataset_manifest=dataset_manifest,
            mask_manifest=mask_manifest,
            masks_root=masks_root,
            person_manifest=person_manifest,
            person_root=person_root,
        )
    elif profile.pose_route != POSE_ROUTE_RAW:
        raise CachePlanError(f"unknown pose route {profile.pose_route!r}")

    disabled = set()
    if not profile.person_masks:
        disabled.add("person_mask_manifest")
    if not profile.mono_depth:
        disabled.add("mono_depth")
    if not profile.sky_masks:
        disabled.update(("sky_masks", "sky_masks_refined"))
    if not profile.tile_ownership:
        disabled.update(s.name for s in specs if s.name.startswith("tile_ownership_"))
    if not profile.view_backgrounds:
        disabled.update(s.name for s in specs if s.name.startswith("view_backgrounds_"))
    return [spec for spec in specs if spec.name not in disabled]


#: Caches the SDK plan budgets as steps of its own (sky masks, tile ownership, backgrounds);
#: the ingest estimate leaves them out so the preflight never counts them twice.
PLAN_BUDGETED_CACHES = ("sky_masks", "sky_masks_refined", "tile_ownership", "view_backgrounds")


def estimate_ingest(
    train_faces: int,
    tile_count: int,
    *,
    validation_caches: Sequence[str],
    face_plan: str = "mipmap_face4",
    pose_route: str = POSE_ROUTE_RAW,
) -> tuple[float, float]:
    """(minutes, GiB) the ingest graph needs for a scene of this size, before it exists.

    The train split is costed at ``train_faces`` and the caches in ``validation_caches`` again
    at the validation share. This is what a dry run and the disk preflight read for a capture
    nobody has prepared: house0614 (6086 images) comes out near 300 GiB, not house0305's 45.
    """
    images = max(1, int(round(train_faces / FACES_PER_IMAGE.get(face_plan, 4))))
    tiles = max(1, int(tile_count))
    train = SceneScale(images=images, faces=max(1, int(train_faces)), tiles=tiles)
    val = SceneScale(images=images, faces=max(1, int(round(train_faces * VALIDATION_FRACTION))), tiles=tiles)
    minutes = gib = 0.0
    for name in REFERENCE_COSTS:
        if name in PLAN_BUDGETED_CACHES:
            continue
        if name in AT_ROUTE_CACHES and pose_route != POSE_ROUTE_AT:
            continue
        minutes += train.minutes(name)
        gib += train.gib(name)
        if name in validation_caches:
            minutes += val.minutes(name)
            gib += val.gib(name)
    return minutes, gib


def _independent_at_specs(
    specs: list[CacheSpec],
    *,
    bundle: DatasetBundle,
    profile: CacheProfile,
    scale: SceneScale,
    dataset: Path,
    cache: Path,
    dataset_manifest: Path,
    mask_manifest: Path,
    masks_root: Path,
    person_manifest: Path,
    person_root: Path,
) -> list[CacheSpec]:
    """The raw-pose route's graph, rewired to train on independent-AT poses.

    The capture's own manifest, circle masks, person masks and split move to a raw tier
    (``<work>/dataset_raw``): the AT reads them, and the frontend gate demands them as raw
    evidence. ``dataset_manifest`` becomes the AT-published training manifest, so every
    downstream cache (masks, depth, split, faces, ...) binds to the corrected poses without
    changing. Person masks of the training tier are a rebind of the raw ones, not a second
    Mask R-CNN pass. Steps follow house0305 (docs/2026-09-11_house0305_v9...): pairs ->
    ALIKED/LightGlue unmasked, then masked -> known-pose triangulation -> independent AT
    (80 outer iterations at 1e-6: the defaults did not converge on house0305) -> training
    manifest; plus the time-sync audit the frontend gate requires.
    """
    python = profile.python
    raw = dataset.parent / f"{dataset.name}_raw"
    work_root = dataset.parent
    at = cache / "at"
    camera_dir = Path(profile.recording_root) / "camera"
    gsplat_lock = Path(profile.repo_root) / "upstream" / "gsplat.lock.json"
    by_name = {spec.name: spec for spec in specs}

    raw_dataset_manifest = raw / "dataset_manifest.json"
    raw_masks_root = raw / "masks"
    raw_mask_manifest = raw_masks_root / "mask_manifest.json"
    raw_person_root = raw / "person_masks"
    raw_person_manifest = raw_person_root / "person_mask_manifest.json"
    raw_split = raw / "split_manifest.json"
    features_raw = at / "features_raw"
    features = at / "features"
    triangulation = at / "triangulation"
    solve = at / "solve"
    pairs = at / "pairs.txt"

    def retarget(part: str) -> str:
        """A path under the training tier, moved to the raw tier."""
        for source, target in (
            (str(dataset_manifest), str(raw_dataset_manifest)),
            (str(mask_manifest), str(raw_mask_manifest)),
            (str(person_root), str(raw_person_root)),
            (str(masks_root), str(raw_masks_root)),
            (str(dataset), str(raw)),
        ):
            if part == source or part.startswith(source + os.sep):
                return target + part[len(source):]
        return part

    def raw_copy(name: str, **changes: object) -> CacheSpec:
        spec = by_name[name]
        fields_: dict[str, object] = {
            "name": f"raw_{name}",
            "title": f"{spec.title} (raw poses, AT input)",
            "command": tuple(retarget(part) for part in spec.command),
            "manifest": Path(retarget(str(spec.manifest))),
            "root": None if spec.root is None else Path(retarget(str(spec.root))),
            "depends_on": tuple(f"raw_{dep}" for dep in spec.depends_on),
            "bindings": tuple(
                replace(binding, depends_on=f"raw_{binding.depends_on}") for binding in spec.bindings
            ),
            "estimated_minutes": scale.minutes(f"raw_{name}"),
            "output_gib": scale.gib(f"raw_{name}"),
        }
        fields_.update(changes)
        return replace(spec, **fields_)

    def fresh(output: Path, *command: str) -> tuple[str, ...]:
        return (
            python, "-m", "cloudstudio3dgs_sdk.ingest.at_steps", "fresh",
            "--output", str(output), "--work-root", str(work_root), "--", *command,
        )

    split_spec = by_name["split_manifest"]
    raw_tier = [
        raw_copy("dataset_manifest"),
        raw_copy("mask_manifest"),
        raw_copy("person_mask_manifest"),
        raw_copy(
            "split_manifest",
            command=tuple(
                str(raw_split) if part == str(split_spec.manifest) else retarget(part) for part in split_spec.command
            ),
            manifest=raw_split,
            root=raw,
        ),
    ]
    at_steps = [
        CacheSpec(
            name="at_pairs",
            title="AT image pairs: stereo, temporal and loop neighbours",
            device=CPU,
            builder="tools/build_independent_at_pairs.py",
            command=(
                python, profile.tool("build_independent_at_pairs.py"),
                "--manifest", str(raw_dataset_manifest),
                "--output", str(at / "pairs_report.json"),
                "--pairs", str(pairs),
            ),
            manifest=at / "pairs_report.json",
            root=at,
            sha_key="manifest_sha256",
            depends_on=("raw_dataset_manifest",),
            bindings=(Binding("dataset_manifest_sha256", "raw_dataset_manifest"),),
            estimated_minutes=scale.minutes("at_pairs"),
            cost_basis="measured (house0305: 443 rig frames -> 4695 pairs in seconds)",
            output_gib=scale.gib("at_pairs"),
        ),
        CacheSpec(
            name="at_features_raw",
            title="ALIKED features + LightGlue matches, unmasked pass",
            device=GPU,
            builder="tools/run_hloc_aliked_lightglue.py",
            command=(
                python, profile.tool("run_hloc_aliked_lightglue.py"),
                "--image-dir", str(camera_dir),
                "--pairs", str(pairs),
                "--output", str(features_raw),
                "--require-cuda", "--overwrite",
            ),
            manifest=features_raw / "feature_runtime_manifest.json",
            root=features_raw,
            sha_key="runtime_manifest_sha256",
            depends_on=("at_pairs",),
            estimated_minutes=scale.minutes("at_features_raw"),
            cost_basis="measured (house0305_at_v2: 886 images extracted in 3 m 18 s, 4695 pairs matched in ~8 m)",
            output_gib=scale.gib("at_features_raw"),
            note="the masked pass reuses this pass's keypoints; it cannot run without them",
        ),
        CacheSpec(
            name="at_features",
            title="ALIKED + LightGlue, keypoints outside circle-valid & ~person dropped, re-matched",
            device=GPU,
            builder="tools/run_hloc_aliked_lightglue.py",
            command=(
                python, profile.tool("run_hloc_aliked_lightglue.py"),
                "--image-dir", str(camera_dir),
                "--pairs", str(pairs),
                "--output", str(features),
                "--require-cuda", "--overwrite",
                "--base-features", str(features_raw / "features-aliked-n16.h5"),
                "--base-feature-runtime-manifest", str(features_raw / "feature_runtime_manifest.json"),
                "--dataset-manifest", str(raw_dataset_manifest),
                "--mask-manifest", str(raw_mask_manifest),
                "--mask-root", str(raw_masks_root),
                "--person-mask-manifest", str(raw_person_manifest),
                "--person-mask-root", str(raw_person_root),
            ),
            manifest=features / "feature_runtime_manifest.json",
            root=features,
            sha_key="runtime_manifest_sha256",
            depends_on=("at_features_raw", "raw_dataset_manifest", "raw_mask_manifest", "raw_person_mask_manifest"),
            bindings=(
                Binding("feature_filter.dataset_manifest_sha256", "raw_dataset_manifest"),
                Binding("feature_filter.mask_manifest_sha256", "raw_mask_manifest"),
                Binding("feature_filter.person_mask_manifest_sha256", "raw_person_mask_manifest"),
            ),
            estimated_minutes=scale.minutes("at_features"),
            cost_basis="measured (house0305_at_v2: ~8 min re-match + ~5 min mask filter)",
            output_gib=scale.gib("at_features"),
        ),
        CacheSpec(
            name="at_triangulation",
            title="Known-pose triangulation (pycolmap)",
            device=CPU,
            builder="tools/run_hloc_triangulation.py",
            command=fresh(
                triangulation,
                python, profile.tool("run_hloc_triangulation.py"),
                "--image-dir", str(camera_dir),
                "--pairs", str(pairs),
                "--features", str(features / "features-aliked-n16.h5"),
                "--matches", str(features / "matches-aliked-lightglue.h5"),
                "--feature-runtime-manifest", str(features / "feature_runtime_manifest.json"),
                "--dataset-manifest", str(raw_dataset_manifest),
                "--output", str(triangulation),
            ),
            manifest=triangulation / "triangulation_runtime_manifest.json",
            root=triangulation,
            sha_key="triangulation_manifest_sha256",
            depends_on=("at_features",),
            bindings=(
                Binding("inputs.feature_runtime_manifest_sha256", "at_features"),
                Binding("inputs.dataset_manifest_sha256", "raw_dataset_manifest"),
            ),
            estimated_minutes=scale.minutes("at_triangulation"),
            cost_basis="measured (house0305_at_v2: 14:33:42 -> 14:38:47)",
            output_gib=scale.gib("at_triangulation"),
        ),
        CacheSpec(
            name="at_solve",
            title="Independent AT: POS-prior BA with one shared KB4 focal per camera",
            device=CPU,
            builder="tools/run_independent_at.py",
            command=fresh(
                solve,
                python, profile.tool("run_independent_at.py"),
                "--model", str(triangulation / "sfm"),
                "--manifest", str(raw_dataset_manifest),
                "--output", str(solve),
                "--triangulation-runtime-manifest", str(triangulation / "triangulation_runtime_manifest.json"),
                # house0305's accepted run used 80 and converged at 55; the UK capture was still
                # 1.85x the step tolerance at 80, closing ~1% per iteration. The tolerances stay.
                "--intrinsic-outer-iterations", "200",
                "--intrinsic-convergence-tol", "1e-6",
            ),
            manifest=solve / "at_report.json",
            root=solve,
            sha_key="report_sha256",
            depends_on=("at_triangulation", "raw_dataset_manifest"),
            bindings=(
                Binding("dataset_manifest_sha256", "raw_dataset_manifest"),
                Binding("triangulation_identity.triangulation_manifest_sha256", "at_triangulation"),
            ),
            estimated_minutes=scale.minutes("at_solve"),
            cost_basis="measured rate (UK: 112 min / 80 outer iterations, 1044 images); iteration count estimated",
            output_gib=scale.gib("at_solve"),
            note="exit 2 when not converged: prepare refuses rather than train on an unconverged AT",
            requires=(("solver_converged", True), ("intrinsic_outer_converged", True)),
        ),
        CacheSpec(
            name="timesync_model",
            title="Time-sync model: short raw-pose whole-scene run (3000 steps, factor 4)",
            device=GPU,
            builder="cloudstudio3dgs_sdk.ingest.at_steps timesync-model",
            command=(
                python, "-m", "cloudstudio3dgs_sdk.ingest.at_steps", "timesync-model",
                "--dataset-manifest", str(raw_dataset_manifest),
                "--split-manifest", str(raw_split),
                "--mask-manifest", str(raw_mask_manifest),
                "--mask-root", str(raw_masks_root),
                "--person-mask-manifest", str(raw_person_manifest),
                "--person-mask-root", str(raw_person_root),
                "--recording-root", str(profile.recording_root),
                "--run-dir", str(profile.source_run_dir),
                "--gsplat-lock", str(gsplat_lock),
                "--output", str(at / "timesync_model"),
            ),
            manifest=at / "timesync_model" / "timesync_model.json",
            root=at / "timesync_model",
            sha_key="sdk_step_manifest_sha256",
            depends_on=("raw_dataset_manifest", "raw_mask_manifest", "raw_person_mask_manifest", "raw_split_manifest"),
            bindings=(Binding("dataset_manifest_sha256", "raw_dataset_manifest"),),
            estimated_minutes=scale.minutes("timesync_model"),
            cost_basis="estimated (house0614 runbook A1: 3000 steps, factor 4, cap 1M)",
            output_gib=scale.gib("timesync_model"),
        ),
        CacheSpec(
            name="timesync",
            title="Camera-time sync audit (render sweep -10..+20 ms)",
            device=GPU,
            builder="cloudstudio3dgs_sdk.ingest.at_steps timesync-audit",
            command=(
                python, "-m", "cloudstudio3dgs_sdk.ingest.at_steps", "timesync-audit",
                "--model-manifest", str(at / "timesync_model" / "timesync_model.json"),
                "--dataset-manifest", str(raw_dataset_manifest),
                "--output", str(at / "timesync"),
            ),
            manifest=at / "timesync" / "time_sync_step.json",
            root=at / "timesync",
            sha_key="sdk_step_manifest_sha256",
            depends_on=("timesync_model",),
            bindings=(Binding("base_dataset_manifest_sha256", "raw_dataset_manifest"),),
            estimated_minutes=scale.minutes("timesync"),
            cost_basis="estimated (house0305 first pass at factor 2, 40 frames: ~19 min)",
            output_gib=scale.gib("timesync"),
            note="a non-zero best offset refuses: the frontend gate only admits 0 ms",
            requires=(("accepted", True),),
        ),
    ]
    training_manifest = CacheSpec(
        name="dataset_manifest",
        title="Training manifest: the capture with AT poses (and refined KB4) and its lineage",
        device=CPU,
        builder="tools/build_ba_training_manifest.py",
        command=(
            python, profile.tool("build_ba_training_manifest.py"),
            "--manifest", str(raw_dataset_manifest),
            "--split-manifest", str(raw_split),
            "--independent-at-report", str(solve / "at_report.json"),
            "--candidate-model", str(solve / "candidate_model"),
            "--output", str(dataset_manifest),
            "--force",
        ),
        manifest=dataset_manifest,
        root=dataset,
        sha_key="manifest_sha256",
        depends_on=("at_solve", "raw_split_manifest", "timesync"),
        bindings=(
            Binding("training_lineage.base_dataset_manifest_sha256", "raw_dataset_manifest"),
            Binding("training_lineage.independent_at_report_sha256", "at_solve"),
            Binding("training_lineage.split_manifest_sha256", "raw_split_manifest"),
        ),
        estimated_minutes=0.5,
        cost_basis="measured (seconds)",
        output_gib=0.002,
    )
    person_rebind = replace(
        by_name["person_mask_manifest"],
        title="Person masks rebound to the training manifest (no second Mask R-CNN pass)",
        device=CPU,
        builder="tools/rebind_person_mask_base.py",
        command=fresh(
            person_root,
            python, profile.tool("rebind_person_mask_base.py"),
            "--person-mask-manifest", str(raw_person_manifest),
            "--base-mask-manifest", str(mask_manifest),
            "--dataset-manifest", str(dataset_manifest),
            "--output", str(person_root),
        ),
        depends_on=("dataset_manifest", "mask_manifest", "raw_person_mask_manifest"),
        estimated_minutes=1.0,
        cost_basis="estimated (re-signs the raw person masks against the training tier)",
    )
    rewired: list[CacheSpec] = [*raw_tier, *at_steps]
    for spec in specs:
        if spec.name == "dataset_manifest":
            rewired.append(training_manifest)
        elif spec.name == "person_mask_manifest":
            rewired.append(person_rebind)
        else:
            rewired.append(spec)
    return rewired


def _topological(specs: Sequence[CacheSpec]) -> list[CacheSpec]:
    by_name = {spec.name: spec for spec in specs}
    ordered: list[CacheSpec] = []
    placed: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in placed or name not in by_name:
            return
        if name in visiting:
            raise CachePlanError(f"cache dependency cycle through '{name}'")
        visiting.add(name)
        for dependency in by_name[name].depends_on:
            visit(dependency)
        visiting.discard(name)
        placed.add(name)
        ordered.append(by_name[name])

    # Declaration order is already dependency order; visit() only fixes the rest,
    # and keeping declaration order makes the printed plan stable.
    for spec in specs:
        visit(spec.name)
    return ordered


@dataclass(frozen=True)
class CacheStatus:
    spec: CacheSpec
    status: str
    reason: str
    sha: str | None = None

    @property
    def must_build(self) -> bool:
        return self.status in (STATUS_MISSING, STATUS_STALE)


class CachePlan:
    """The ordered cache graph for one bundle + profile, with skip logic."""

    def __init__(
        self,
        bundle: DatasetBundle,
        profile: CacheProfile,
        *,
        specs: Sequence[CacheSpec] | None = None,
    ) -> None:
        self.bundle = bundle
        self.profile = profile
        self.specs = _topological(
            list(specs) if specs is not None else build_cache_specs(bundle, profile)
        )
        self._manifest_cache: dict[str, Mapping[str, Any] | None] = {}

    # -- inspection ------------------------------------------------------

    def __iter__(self):
        return iter(self.specs)

    def spec(self, name: str) -> CacheSpec:
        for candidate in self.specs:
            if candidate.name == name:
                return candidate
        raise CachePlanError(f"unknown cache '{name}'")

    def _manifest(self, name: str) -> Mapping[str, Any] | None:
        if name not in self._manifest_cache:
            spec = self.spec(name)
            payload: Mapping[str, Any] | None = None
            if spec.manifest.is_file():
                try:
                    payload = json.loads(spec.manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
            self._manifest_cache[name] = payload
        return self._manifest_cache[name]

    def sha_of(self, name: str) -> str | None:
        payload = self._manifest(name)
        if payload is None:
            return None
        value = payload.get(self.spec(name).sha_key)
        return str(value) if value else None

    def invalidate(self, name: str | None = None) -> None:
        if name is None:
            self._manifest_cache.clear()
        else:
            self._manifest_cache.pop(name, None)

    def status_of(self, spec: CacheSpec) -> CacheStatus:
        missing_capabilities = sorted(
            set(spec.requires_capabilities) - self.bundle.capabilities
        )
        if missing_capabilities:
            return CacheStatus(
                spec,
                STATUS_BLOCKED,
                "bundle lacks " + ", ".join(missing_capabilities),
            )
        payload = self._manifest(spec.name)
        if payload is None:
            return CacheStatus(spec, STATUS_MISSING, f"no manifest at {spec.manifest}")
        own_sha = payload.get(spec.sha_key)
        if not own_sha:
            return CacheStatus(
                spec, STATUS_STALE, f"manifest carries no {spec.sha_key}"
            )
        for binding in spec.bindings:
            recorded = _dotted(payload, binding.manifest_key)
            upstream = self.sha_of(binding.depends_on)
            if recorded is None:
                if binding.optional:
                    continue
                return CacheStatus(
                    spec,
                    STATUS_STALE,
                    f"manifest has no {binding.manifest_key}",
                    str(own_sha),
                )
            if upstream is None:
                return CacheStatus(
                    spec,
                    STATUS_STALE,
                    f"upstream '{binding.depends_on}' has no manifest to bind to",
                    str(own_sha),
                )
            if str(recorded) != upstream:
                return CacheStatus(
                    spec,
                    STATUS_STALE,
                    (
                        f"{binding.manifest_key} is {str(recorded)[:12]}.. but "
                        f"'{binding.depends_on}' is {upstream[:12]}.."
                    ),
                    str(own_sha),
                )
        for key, expected in spec.requires:
            value = _dotted(payload, key)
            if value != expected:
                return CacheStatus(
                    spec, STATUS_STALE, f"{key} is {value!r}, the cache needs {expected!r}", str(own_sha)
                )
        return CacheStatus(spec, STATUS_PRESENT, "bindings match", str(own_sha))

    def statuses(self) -> list[CacheStatus]:
        return [self.status_of(spec) for spec in self.specs]

    def pending(self) -> list[CacheStatus]:
        return [status for status in self.statuses() if status.must_build]

    def total_minutes(self, *, pending_only: bool = True) -> float:
        rows = self.pending() if pending_only else self.statuses()
        return float(sum(row.spec.estimated_minutes for row in rows))

    # -- reporting -------------------------------------------------------

    def describe(self) -> list[str]:
        lines = [
            f"dataset: {self.bundle.dataset_id} (adapter {self.bundle.adapter}, "
            f"{len(self.bundle.images)} images)",
            f"capabilities: {', '.join(sorted(self.bundle.capabilities)) or 'none'}",
            "",
            f"{'#':>2}  {'cache':<24} {'dev':<3} {'status':<8} {'min':>6}  detail",
        ]
        for index, status in enumerate(self.statuses(), start=1):
            spec = status.spec
            lines.append(
                f"{index:>2}  {spec.name:<24} {spec.device:<3} {status.status:<8} "
                f"{spec.estimated_minutes:>6.1f}  {status.reason}"
            )
        lines.append("")
        lines.append(
            f"to build: {len(self.pending())} caches, "
            f"~{self.total_minutes():.0f} min "
            f"({sum(s.spec.estimated_minutes for s in self.pending() if s.spec.device == GPU):.0f} min of it on GPU)"
        )
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.bundle.dataset_id,
            "adapter": self.bundle.adapter,
            "capabilities": sorted(self.bundle.capabilities),
            "caches": [
                {
                    "name": status.spec.name,
                    "title": status.spec.title,
                    "device": status.spec.device,
                    "builder": status.spec.builder,
                    "command": list(status.spec.command),
                    "manifest": str(status.spec.manifest),
                    "sha_key": status.spec.sha_key,
                    "depends_on": list(status.spec.depends_on),
                    "bindings": [
                        {"key": b.manifest_key, "from": b.depends_on, "optional": b.optional}
                        for b in status.spec.bindings
                    ],
                    "estimated_minutes": status.spec.estimated_minutes,
                    "cost_basis": status.spec.cost_basis,
                    "output_gib": status.spec.output_gib,
                    "status": status.status,
                    "reason": status.reason,
                    "sha": status.sha,
                    "note": status.spec.note,
                }
                for status in self.statuses()
            ],
        }

    # -- execution -------------------------------------------------------

    def build(
        self,
        *,
        dry_run: bool = True,
        only: Iterable[str] | None = None,
        runner: Callable[[Sequence[str]], int] | None = None,
    ) -> list[str]:
        """Print the plan (``dry_run=True``) or run the CPU half of it.

        GPU caches always raise :class:`GpuStepRequired`; this process must not
        take a CUDA context.  Blocked caches raise
        :class:`DatasetIncompleteError` naming the missing capability.
        """

        selected = set(only) if only is not None else None
        report: list[str] = []
        if dry_run:
            report.extend(self.describe())
            report.append("")
            for status in self.statuses():
                if selected is not None and status.spec.name not in selected:
                    continue
                marker = "BUILD" if status.must_build else "skip "
                report.append(f"[{marker}] {status.spec.name} ({status.spec.device})")
                report.append("         " + " ".join(status.spec.command))
            return report

        for status in self.statuses():
            spec = status.spec
            if selected is not None and spec.name not in selected:
                continue
            if status.status == STATUS_BLOCKED:
                raise DatasetIncompleteError(
                    f"{spec.name}: {status.reason}; this dataset cannot produce it"
                )
            if not status.must_build:
                report.append(f"skip  {spec.name}: {status.reason}")
                continue
            if spec.device == GPU:
                raise GpuStepRequired(
                    f"{spec.name} is a GPU step, run through the SDK runner "
                    f"(builder: {spec.builder})"
                )
            if any(part.startswith("<") and part.endswith(">") for part in spec.command):
                placeholders = [
                    part for part in spec.command if part.startswith("<") and part.endswith(">")
                ]
                raise CachePlanError(
                    f"{spec.name}: the profile does not supply "
                    + ", ".join(placeholders)
                )
            report.append(f"build {spec.name}: " + " ".join(spec.command))
            code = (
                runner(spec.command)
                if runner is not None
                else subprocess.run(
                    list(spec.command), cwd=str(self.profile.repo_root), check=False
                ).returncode
            )
            if code != 0:
                raise CachePlanError(
                    f"{spec.name} failed with exit code {code}: " + " ".join(spec.command)
                )
            self.invalidate(spec.name)
            refreshed = self.status_of(spec)
            if refreshed.status != STATUS_PRESENT:
                raise CachePlanError(
                    f"{spec.name} finished but its manifest does not verify: "
                    f"{refreshed.reason}"
                )
            report.append(f"  -> {spec.sha_key}={refreshed.sha}")
        return report


def plan_caches(
    bundle: DatasetBundle,
    profile: Any,
    **overrides: Any,
) -> CachePlan:
    """Convenience entry point: accepts an SDK profile object or a mapping."""

    return CachePlan(bundle, CacheProfile.from_any(profile, **overrides))


__all__ = [
    "CPU",
    "GPU",
    "STATUS_BLOCKED",
    "STATUS_MISSING",
    "STATUS_PRESENT",
    "STATUS_STALE",
    "Binding",
    "CachePlan",
    "CachePlanError",
    "CacheProfile",
    "CacheSpec",
    "CacheStatus",
    "AT_ROUTE_CACHES",
    "PLAN_BUDGETED_CACHES",
    "POSE_ROUTE_AT",
    "POSE_ROUTE_RAW",
    "REFERENCE_COSTS",
    "SceneScale",
    "build_cache_specs",
    "estimate_ingest",
    "plan_caches",
]
