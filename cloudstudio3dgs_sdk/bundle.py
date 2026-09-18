"""The interface between dataset ingestion and the delivery engine.

There are two different objects in this package and keeping them apart is the
whole point of this module:

``cloudstudio3dgs_sdk.ingest.bundle.DatasetBundle``
    The *capture*: image list, per-camera intrinsics, poses, rig transforms,
    timestamps, split, LiDAR cloud reference and the capability set derived
    from them. Owned by the ingestion task, produced by its adapters
    (``load_dataset``), and the thing ``plan_caches`` reasons about.

:class:`PreparedScene` (here)
    The *prepared scene*: the signed manifests and cache roots the trainer
    binds by path, plus the tile inventory the plan needs. It is the narrow
    projection the delivery engine consumes; it holds paths and counts, never
    images or poses.

``Project.prepare()`` turns the first into the second and records it in
``<work>/prepare/prepare_manifest.json``. Everything downstream reads that
manifest and never touches the raw dataset again - which is what lets a later
stage refuse when a digest moved.

The ingestion half is not implemented here. :func:`load_dataset_bundle`
raises :class:`NotImplementedError` naming the entry points that owe it.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

# The parallel task's package and the two calls that produce a prepared scene:
# an adapter run, then the signed cache graph built on top of it.
INGEST_PACKAGE = "cloudstudio3dgs_sdk.ingest"
INGEST_LOAD = f"{INGEST_PACKAGE}.load_dataset"
INGEST_PLAN_CACHES = f"{INGEST_PACKAGE}.plan_caches"


@dataclass(frozen=True)
class DerivedCaches:
    """Caches ``prepare()`` builds once and every tile arm then binds.

    All of them are signed manifests: the trainer validates each against the
    Face4 cache sha before it will train, so a stale cache is a refusal rather
    than a silently wrong run. The ingestion task's ``CachePlan`` is what
    builds them; this is where their locations are recorded afterwards.
    """

    sky_mask_manifest: Path | None = None
    sky_mask_root: Path | None = None
    sky_dome_checkpoint: Path | None = None
    global_background_manifest: Path | None = None
    global_background_root: Path | None = None
    # tile_id -> (manifest, root)
    tile_ownership: Mapping[int, tuple[Path, Path]] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            key: (None if value is None else str(value))
            for key, value in (
                ("sky_mask_manifest", self.sky_mask_manifest),
                ("sky_mask_root", self.sky_mask_root),
                ("sky_dome_checkpoint", self.sky_dome_checkpoint),
                ("global_background_manifest", self.global_background_manifest),
                ("global_background_root", self.global_background_root),
            )
        }
        payload["tile_ownership"] = {
            str(tile): [str(manifest), str(root)] for tile, (manifest, root) in sorted(self.tile_ownership.items())
        }
        return payload


@dataclass(frozen=True)
class PreparedScene:
    """Everything one prepared scene offers the recipe, grouped by source.

    images / poses / intrinsics
        ``dataset_manifest`` is the signed image+pose+intrinsics record that
        the ingest ``DatasetBundle`` was written into; ``recording_root`` is
        where the raw frames live; ``face_cache_*`` is the Face4
        rectification of the fisheye rig that training actually samples;
        ``renderer_mask_manifest`` the per-face validity masks.

    split
        ``split_manifest`` - rig-frame split, train/val/golden.

    supervision masks
        ``mask_*`` (dataset masks), ``person_mask_*`` (people removed from
        the photometric target), ``mono_depth_*`` (DA2 relative depth).

    LiDAR
        ``lidar_cloud`` is the customer cloud; ``depth_*`` the per-face
        projected returns; ``face_lidar_geometry_*`` the per-face geometry
        the surface terms read.

    tiling
        ``tile_inputs_manifest`` / ``tile_inputs_root`` carry, per tile, the
        core box, the training-and-export box, the view list, the LiDAR
        initialisation PLY and its point count; ``tile_geometry_manifest``
        the matching kNN geometry. ``global_init_ply`` /
        ``global_init_geometry`` are the decimated whole-scene initialisation
        the coarse prior starts from.

    gate
        ``pipeline_gate`` is the signed readiness gate the trainer refuses to
        start without.
    """

    scene_tag: str
    dataset_root: Path
    dataset_manifest: Path
    split_manifest: Path
    recording_root: Path
    face_cache_manifest: Path
    face_cache_root: Path
    renderer_mask_manifest: Path
    mask_manifest: Path
    mask_root: Path
    person_mask_manifest: Path
    person_mask_root: Path
    depth_manifest: Path
    depth_root: Path
    mono_depth_manifest: Path
    mono_depth_root: Path
    face_lidar_geometry_manifest: Path
    face_lidar_geometry_root: Path
    lidar_cloud: Path
    tile_inputs_manifest: Path
    tile_inputs_root: Path
    tile_geometry_manifest: Path
    global_init_ply: Path
    global_init_geometry: Path
    pipeline_gate: Path
    caches: DerivedCaches = field(default_factory=DerivedCaches)

    def trainer_paths(self) -> dict[str, str]:
        """The scene-level path block every arm config carries verbatim.

        The keys are trainer config keys, and the profile lists them in
        ``dataset_contract.trainer_path_keys``; a key present in one and
        absent from the other is a wiring bug, not a missing file.
        """
        return {
            "dataset_manifest": str(self.dataset_manifest),
            "split_manifest": str(self.split_manifest),
            "mask_manifest": str(self.mask_manifest),
            "mask_root": str(self.mask_root),
            "person_mask_manifest": str(self.person_mask_manifest),
            "person_mask_root": str(self.person_mask_root),
            "recording_root": str(self.recording_root),
            "face_cache_manifest": str(self.face_cache_manifest),
            "face_cache_root": str(self.face_cache_root),
            "renderer_mask_manifest": str(self.renderer_mask_manifest),
            "depth_manifest": str(self.depth_manifest),
            "depth_root": str(self.depth_root),
            "mono_depth_manifest": str(self.mono_depth_manifest),
            "mono_depth_root": str(self.mono_depth_root),
            "face_lidar_geometry_manifest": str(self.face_lidar_geometry_manifest),
            "face_lidar_geometry_root": str(self.face_lidar_geometry_root),
            "mipmap_pipeline_gate": str(self.pipeline_gate),
        }

    def as_json(self) -> dict[str, Any]:
        payload = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(self).items()
            if key != "caches"
        }
        payload["caches"] = self.caches.as_json()
        return payload


#: Which ingest cache feeds which :class:`PreparedScene` field. The ingest layer names caches
#: by what they are; the trainer names paths by the config key that binds them. This is the
#: one place the two vocabularies meet, so a rename on either side fails here, loudly.
_SCENE_FIELD_BY_CACHE: Mapping[str, tuple[str, str | None]] = MappingProxyType(
    {
        # cache name: (manifest field, root field or None)
        "dataset_manifest": ("dataset_manifest", None),
        "split_manifest": ("split_manifest", None),
        "mask_manifest": ("mask_manifest", "mask_root"),
        "person_mask_manifest": ("person_mask_manifest", "person_mask_root"),
        "depth_cache": ("depth_manifest", "depth_root"),
        "face_cache": ("face_cache_manifest", "face_cache_root"),
        "renderer_mask": ("renderer_mask_manifest", None),
        "face_lidar_geometry": ("face_lidar_geometry_manifest", "face_lidar_geometry_root"),
        "mono_depth": ("mono_depth_manifest", "mono_depth_root"),
        "tile_inputs": ("tile_inputs_manifest", "tile_inputs_root"),
        "tile_geometry": ("tile_geometry_manifest", None),
    }
)

GATE_TOOLS = (
    "tools/advance_mipmap_da2_gate.py",
    "tools/advance_mipmap_sky_gate.py",
)


class FreshBuildBlocked(Exception):
    """A fresh build stopped at a step this process must not run itself.

    Carries the exact command so the caller (or a human) can run it where it belongs. Raised
    for GPU caches - this process must never take a CUDA context - and re-raised as the ingest
    layer's own :class:`GpuStepRequired` so callers that already catch that keep working.
    """

    def __init__(self, cache: str, command: tuple[str, ...], reason: str) -> None:
        self.cache = cache
        self.command = command
        self.reason = reason
        super().__init__(f"{cache}: {reason}\n  " + " ".join(command))


def _subprocess_runner(command: Sequence[str]) -> int:
    import subprocess

    return subprocess.run(list(command)).returncode


def load_dataset_bundle(
    dataset_root: Path,
    profile: Any,
    work_root: Path,
    *,
    python: Path | str | None = None,
    repo_root: Path | str | None = None,
    pipeline_gate: Path | str | None = None,
    runner: Callable[[Sequence[str]], int] | None = None,
    log: Callable[[str], None] | None = None,
) -> PreparedScene:
    """Ingest a capture and build every CPU cache it needs into ``work_root``.

    This is the fresh-dataset path: the adapter reads the capture, the ingest layer derives
    the signed cache graph, and this runs the graph's CPU half in dependency order. Two
    things it deliberately does NOT do:

    * take a CUDA context. The first GPU cache whose inputs are ready raises
      :class:`GpuStepRequired` with the exact command; the SDK's stage runner, which holds the
      GPU lease, runs it and calls back in. Caches that do not depend on the GPU one are
      built first, so one call does as much as it can.
    * build the mipmap pipeline gate. The gate is a thirteen-stage signed readiness contract
      the trainer refuses to start without on fisheye data, and its chain lives in the gate
      tools, not in ingestion. Pass ``pipeline_gate`` to a gate that chain produced; without
      one this raises :class:`DatasetIncompleteError` naming the tools.

    The whole-scene initialisation the coarse prior starts from is built here too
    (``tools/build_lidar_init.py`` at the profile's decimation), because nothing in the cache
    graph produces it and every prepared scene needs it.

    A scene that was already prepared by hand - house0305 - does not come through here;
    ``Project.prepare()`` adopts its manifest instead.
    """
    from cloudstudio3dgs_sdk.ingest import load_dataset, plan_caches
    from cloudstudio3dgs_sdk.ingest.errors import DatasetIncompleteError, GpuStepRequired
    from cloudstudio3dgs_sdk.ingest.caches import GPU, STATUS_BLOCKED

    say = log or (lambda line: None)
    run = runner or _subprocess_runner
    dataset_root = Path(dataset_root)
    work = Path(work_root)
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    interpreter = str(python) if python else sys.executable

    bundle = load_dataset(dataset_root)
    say(f"[prepare] adapter {bundle.adapter}: {len(bundle.images)} images, "
        f"cloud {'present' if bundle.point_cloud else 'absent'}")
    if bundle.point_cloud is None:
        raise DatasetIncompleteError(
            f"{dataset_root}: no LiDAR point cloud. The recipe initialises every tile from LiDAR "
            "and reads range/normal supervision from it; a capture without one cannot run it."
        )

    plan = plan_caches(
        bundle,
        profile,
        dataset_root=work / "dataset",
        cache_root=work / "caches",
        run_root=work / "runs",
        recording_root=bundle.source_root,
        source_run_dir=bundle.source_root,
        repo_root=repo,
        python=interpreter,
    )

    # Run the CPU half in dependency order. A GPU cache is not an error until something that
    # is not yet built depends on it; everything else keeps going so the caller gets the
    # longest possible run out of one call.
    built: set[str] = set()
    pending_gpu: tuple[str, tuple[str, ...]] | None = None
    for status in plan.statuses():
        spec = status.spec
        if status.status == STATUS_BLOCKED:
            raise DatasetIncompleteError(f"{spec.name}: {status.reason}; this dataset cannot produce it")
        if not status.must_build:
            say(f"[prepare] {spec.name}: present")
            built.add(spec.name)
            continue
        unmet = [dep for dep in spec.depends_on if dep not in built]
        if unmet:
            say(f"[prepare] {spec.name}: waiting on {', '.join(unmet)}")
            continue
        if spec.device == GPU:
            if pending_gpu is None:
                pending_gpu = (spec.name, tuple(spec.command))
            say(f"[prepare] {spec.name}: needs the GPU")
            continue
        say(f"[prepare] {spec.name}: building")
        plan.build(dry_run=False, only=[spec.name], runner=run)
        built.add(spec.name)

    if pending_gpu is not None:
        name, command = pending_gpu
        raise GpuStepRequired(
            f"{name} needs the GPU; run it under the SDK's GPU lease, then call prepare again:\n  "
            + " ".join(command)
        )

    # The coarse prior's whole-scene initialisation. Not part of the cache graph, needed by
    # every scene; built at the profile's decimation so its density is the profile's, not the
    # capture's.
    global_init_dir = work / "caches" / "global_init"
    global_init_ply = global_init_dir / "sparse_pc.ply"
    global_init_geometry = global_init_dir / "lidar_init_geometry.npz"
    if not (global_init_ply.is_file() and global_init_geometry.is_file()):
        say("[prepare] global_init: building")
        global_init_dir.mkdir(parents=True, exist_ok=True)
        decimation = float(profile.coarse_prior["init_decimation_m"])
        command = (
            interpreter, str(repo / "tools" / "build_lidar_init.py"),
            "--run", str(bundle.source_root),
            "--output", str(global_init_dir),
            "--voxel-size", str(decimation),
            "--with-pca",
            "--seed", "42",
        )
        code = run(command)
        if code != 0 or not (global_init_ply.is_file() and global_init_geometry.is_file()):
            raise FreshBuildBlocked("global_init", command, f"build_lidar_init.py exited {code}")
    else:
        say("[prepare] global_init: present")

    # The readiness gate. Outside ingestion by design; refuse rather than forge one.
    if pipeline_gate is None:
        raise DatasetIncompleteError(
            "no mipmap pipeline gate. The trainer refuses fisheye data without the signed "
            "thirteen-stage readiness gate, and its chain is not part of ingestion. Produce it "
            "with " + " then ".join(GATE_TOOLS) + " against this work root's caches, then pass "
            "--pipeline-gate PATH."
        )
    gate_path = Path(pipeline_gate)
    from cloudstudio_3dgs.pipeline.mipmap_gate import load_and_verify_gate

    load_and_verify_gate(gate_path)  # raises on a bad signature

    # Project the built graph onto the trainer's path contract.
    specs = {status.spec.name: status.spec for status in plan.statuses()}
    fields: dict[str, Any] = {}
    for cache, (manifest_field, root_field) in _SCENE_FIELD_BY_CACHE.items():
        spec = specs[cache]
        fields[manifest_field] = spec.manifest
        if root_field:
            fields[root_field] = spec.root
    tile_ownership = {
        spec.tile_id: (spec.manifest, spec.root)
        for spec in specs.values()
        if spec.name.startswith("tile_ownership_") and spec.tile_id is not None
    }
    sky = specs["sky_masks"]
    caches = DerivedCaches(
        sky_mask_manifest=sky.manifest,
        sky_mask_root=sky.root,
        tile_ownership=tile_ownership,
    )
    return PreparedScene(
        scene_tag=str(bundle.dataset_id),
        dataset_root=dataset_root,
        recording_root=Path(bundle.source_root),
        lidar_cloud=Path(bundle.point_cloud.path),
        global_init_ply=global_init_ply,
        global_init_geometry=global_init_geometry,
        pipeline_gate=gate_path,
        caches=caches,
        **fields,
    )
