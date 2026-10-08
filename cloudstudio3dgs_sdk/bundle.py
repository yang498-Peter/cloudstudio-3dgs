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

# Where the training poses come from. The readiness gate chain starts from an independent
# AT report (tools/build_mipmap_frontend_gate.py), and the trainer requires the gate only for
# data with that lineage (TrainerConfig.validate). A capture trained on its own S1Mapper poses
# therefore has no gate - a weaker route, recorded as such, never a silently skipped check.
POSE_ROUTE_AT = "independent_at"
POSE_ROUTE_RAW = "raw_capture_poses"


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
        start without on independent-AT data. It is ``None`` only on the
        ``raw_capture_poses`` route (``pose_route``), where the trainer does not
        ask for one.
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
    pipeline_gate: Path | None
    caches: DerivedCaches = field(default_factory=DerivedCaches)
    pose_route: str = POSE_ROUTE_AT

    def trainer_paths(self) -> dict[str, str | None]:
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
            # JSON null on the raw-pose route: the trainer reads a missing gate as "none".
            "mipmap_pipeline_gate": str(self.pipeline_gate) if self.pipeline_gate is not None else None,
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

#: The readiness gate chain, in order, as house0305's gates_v9 recorded it. The frontend gate
#: needs the aerotriangulation campaign's artefacts (time-sync report, feature and
#: triangulation runtimes, AT report, candidate model), which is why ingestion cannot
#: produce it: the pose pipeline is a separate SOP that runs before any of this.
GATE_TOOLS = (
    "tools/build_mipmap_frontend_gate.py",
    "tools/advance_mipmap_renderer_mask_gate.py",
    "tools/advance_mipmap_lidar_depth_gate.py",
    "tools/advance_mipmap_da2_gate.py",
    "tools/advance_mipmap_tile_gate.py",
    "tools/promote_surface_frozen_training_gate.py",
    "tools/bind_monocular_depth_gate.py",
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


#: The split-specific caches the held-out battery reads (validation_paths.FACE_KEYS and
#: TRAIN_VAL_KEYS). Masks, depth, person masks and the split itself are shared by both splits.
VALIDATION_CACHES = ("face_cache", "renderer_mask", "face_lidar_geometry")


def _build_cpu_half(
    plan: Any,
    *,
    run: Callable[[Sequence[str]], int],
    say: Callable[[str], None],
    only: Sequence[str] | None = None,
    tag: str = "",
) -> tuple[str, tuple[str, ...]] | None:
    """Build every CPU cache whose inputs are ready, in dependency order.

    Returns the first GPU cache that is due (name, command), or ``None``. With ``only``,
    caches outside it are inputs: present ones count as built, missing ones are not built
    here (their own graph owns them).
    """
    from cloudstudio3dgs_sdk.ingest.caches import GPU, STATUS_BLOCKED
    from cloudstudio3dgs_sdk.ingest.errors import DatasetIncompleteError

    wanted = set(only) if only is not None else None
    built: set[str] = set()
    pending_gpu: tuple[str, tuple[str, ...]] | None = None
    for status in plan.statuses():
        spec = status.spec
        if wanted is not None and spec.name not in wanted:
            if not status.must_build:
                built.add(spec.name)
            continue
        if status.status == STATUS_BLOCKED:
            raise DatasetIncompleteError(f"{spec.name}: {status.reason}; this dataset cannot produce it")
        if not status.must_build:
            say(f"[prepare] {tag}{spec.name}: present")
            built.add(spec.name)
            continue
        unmet = [dep for dep in spec.depends_on if dep not in built]
        if unmet:
            say(f"[prepare] {tag}{spec.name}: waiting on {', '.join(unmet)}")
            continue
        if spec.device == GPU:
            if pending_gpu is None:
                pending_gpu = (spec.name, tuple(spec.command))
            say(f"[prepare] {tag}{spec.name}: needs the GPU")
            continue
        say(f"[prepare] {tag}{spec.name}: building")
        plan.build(dry_run=False, only=[spec.name], runner=run)
        built.add(spec.name)
    return pending_gpu


def _tile_count(profile: Any, cloud: Path, *, vram_gib: float | None) -> int | None:
    """The profile's tile count for this cloud, from the LAS header alone; None without one."""
    if not hasattr(profile, "tiling") or not hasattr(profile, "tile_rules"):
        return None
    try:
        import laspy

        with laspy.open(str(cloud)) as reader:
            points = int(reader.header.point_count)
    except Exception:  # noqa: BLE001 - an unreadable header leaves the reference count in force
        return None
    from cloudstudio3dgs_sdk.plan import tile_count_for

    return tile_count_for(profile, points, vram_gib=vram_gib)


def _smoke_cap(profile: Any, tile_inputs: Path, *, vram_gib: float | None) -> int:
    """Tile_0's cap under the profile's own rule: what the full-resolution smoke measures VRAM at."""
    import json

    manifest = json.loads(Path(tile_inputs).read_text(encoding="utf-8"))
    tile = next(entry for entry in manifest["tiles"] if int(entry["tile_id"]) == 0)
    points = int(tile["initialization"]["point_count"])
    if not hasattr(profile, "tile_rules"):
        return max(2 * points, points + 1)
    from cloudstudio3dgs_sdk.plan import TileSummary, tile_cap

    summary = TileSummary(0, "Tile_0", int(tile.get("view_count", 0) or 0), points)
    return max(tile_cap(profile, summary, vram_gib=vram_gib), points + 1)


def load_dataset_bundle(
    dataset_root: Path,
    profile: Any,
    work_root: Path,
    *,
    python: Path | str | None = None,
    repo_root: Path | str | None = None,
    pipeline_gate: Path | str | None = None,
    adapter: str | None = None,
    run_dir: Path | str | None = None,
    runner: Callable[[Sequence[str]], int] | None = None,
    log: Callable[[str], None] | None = None,
    vram_gib: float | None = None,
    assets: Mapping[str, Path | str | None] | None = None,
    pose_route: str = POSE_ROUTE_RAW,
) -> PreparedScene:
    """Ingest a capture and build every CPU cache it needs into ``work_root``.

    ``adapter`` names the ingest adapter instead of detecting it; ``run_dir`` is for captures
    whose processed half (poses, colourised cloud) lives in a second directory, as the
    S1Mapper "Raw_Data" / "Processed_by_S1Mapper" pair does.

    This is the fresh-dataset path: the adapter reads the capture, the ingest layer derives
    the signed cache graph, and this runs the graph's CPU half in dependency order. Two
    things it deliberately does NOT do:

    * take a CUDA context. The first GPU cache whose inputs are ready raises
      :class:`GpuStepRequired` with the exact command; the SDK's stage runner, which holds the
      GPU lease, runs it and calls back in. Caches that do not depend on the GPU one are
      built first, so one call does as much as it can.
    * forge a readiness gate. On the independent-AT route (``pose_route="independent_at"``)
      ingestion runs the AT chain and then the gate chain (``ingest.gates``) with the repo's
      gate tools, so the gate is evidence, not a stub. A manifest with AT lineage on any other
      route needs ``pipeline_gate``, or this raises :class:`DatasetIncompleteError`.

    The whole-scene initialisation the coarse prior starts from is built here too
    (``tools/build_lidar_init.py`` at the profile's decimation), because nothing in the cache
    graph produces it and every prepared scene needs it.

    A scene that was already prepared by hand - house0305 - does not come through here;
    ``Project.prepare()`` adopts its manifest instead.
    """
    from cloudstudio3dgs_sdk.ingest import load_dataset, plan_caches
    from cloudstudio3dgs_sdk.ingest.errors import DatasetIncompleteError, GpuStepRequired

    say = log or (lambda line: None)
    run = runner or _subprocess_runner
    dataset_root = Path(dataset_root)
    work = Path(work_root)
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    interpreter = str(python) if python else sys.executable

    load_kwargs: dict[str, Any] = {}
    if run_dir is not None:
        load_kwargs["run_dir"] = Path(run_dir)
    bundle = load_dataset(dataset_root, adapter=adapter, **load_kwargs)
    say(f"[prepare] adapter {bundle.adapter}: {len(bundle.images)} images, "
        f"cloud {'present' if bundle.point_cloud else 'absent'}")
    if bundle.point_cloud is None:
        raise DatasetIncompleteError(
            f"{dataset_root}: no LiDAR point cloud. The recipe initialises every tile from LiDAR "
            "and reads range/normal supervision from it; a capture without one cannot run it."
        )

    # A split capture (recording and S1Mapper output in two folders, house0614) keeps the poses
    # and the cloud in the run folder; a single-folder capture keeps them next to the images.
    source_run_dir = Path(run_dir) if run_dir is not None else Path(bundle.source_root)
    roots = dict(
        dataset_root=work / "dataset",
        cache_root=work / "caches",
        run_root=work / "runs",
        recording_root=bundle.source_root,
        source_run_dir=source_run_dir,
        repo_root=repo,
        python=interpreter,
        # The profile's sky-label refinement, if any; the cache layer does not know Profile.
        sky_mask_refinement=(getattr(profile, "dataset_contract", None) or {}).get("sky_mask_refinement"),
        # Ownership masks and per-view backgrounds are the SDK plan's own steps: ownership needs
        # the written arm config (the ingest spec only had a <trainer_config> placeholder) and
        # the backgrounds need trained neighbour tiles. Built here they could only fail.
        tile_ownership=False,
        view_backgrounds=False,
        # independent_at: raw tier -> AT -> training manifest, then the signed gate chain
        pose_route=pose_route,
    )
    # Weights the GPU caches load (person masks, DA2): machine paths, never profile data.
    for key in ("person_weights", "da2_model_source", "da2_checkpoint"):
        value = (assets or {}).get(key)
        if value:
            roots[key] = Path(value)
    # As many tiles as the cloud needs for the profile's cap rule (never fewer than its reference
    # count): a 100M-point capture cut four ways starts every tile above its cap.
    tiles = _tile_count(profile, Path(bundle.point_cloud.path), vram_gib=vram_gib)
    if tiles is not None:
        roots["tile_count"] = tiles
        if tiles > int(profile.tiling["reference_tile_count"]):
            roots["tile_layout"] = "grid"
        say(f"[prepare] tiling: {tiles} tile(s) for this cloud under profile {getattr(profile, 'name', '?')}")
    plan = plan_caches(bundle, profile, **roots)

    # Run the CPU half in dependency order. A GPU cache is not an error until something that
    # is not yet built depends on it; everything else keeps going so the caller gets the
    # longest possible run out of one call.
    pending_gpu = _build_cpu_half(plan, run=run, say=say)
    # The held-out battery reads validation caches derived by name from the training ones
    # (face4_train -> face4_val, renderer_mask_train -> renderer_mask_val, ...; see
    # cloudstudio_3dgs/training/validation_paths.py). The same graph at split="val" names
    # exactly those; everything they depend on is shared and already present.
    val_plan = plan_caches(bundle, profile, split="val", **roots)
    # The training gate binds DA2 for both splits, so the AT route builds the val DA2 too.
    validation = VALIDATION_CACHES + (("mono_depth",) if pose_route == POSE_ROUTE_AT else ())
    val_pending = _build_cpu_half(val_plan, run=run, say=say, only=validation, tag="val ")
    pending_gpu = pending_gpu or val_pending

    if pending_gpu is not None:
        name, command = pending_gpu
        raise GpuStepRequired(
            f"{name} needs the GPU; run it under the SDK's GPU lease, then call prepare again:\n  "
            + " ".join(command),
            cache=name,
            command=tuple(command),
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
            "--run", str(source_run_dir),
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

    specs = {status.spec.name: status.spec for status in plan.statuses()}

    if pose_route == POSE_ROUTE_AT and pipeline_gate is None:
        # The AT route publishes its own readiness gate: the chain house0305 v9 was trained
        # under, run against this work root's caches (cloudstudio3dgs_sdk.ingest.gates).
        from cloudstudio3dgs_sdk.ingest.gates import build_gate_chain, gate_inputs_from_specs

        val_specs = {status.spec.name: status.spec for status in val_plan.statuses()}
        inputs = gate_inputs_from_specs(
            specs, val_specs, recording_root=Path(bundle.source_root),
            gsplat_lock=repo / "upstream" / "gsplat.lock.json",
        )
        pipeline_gate = build_gate_chain(
            inputs,
            work / "runs" / "gates",
            python=interpreter,
            repo_root=repo,
            cap_max=_smoke_cap(profile, inputs.tile_inputs, vram_gib=vram_gib),
            run=run,
            say=say,
        )

    # The readiness gate. Outside ingestion by design; refuse rather than forge one. It is
    # owed exactly when the trainer will ask for it: on independent-AT data. A manifest that
    # cannot be read counts as AT data, so an unknown lineage still needs a gate.
    at_lineage = _has_independent_at_lineage(specs["dataset_manifest"].manifest)
    if pipeline_gate is None:
        if at_lineage is not False:
            raise DatasetIncompleteError(
                "no mipmap pipeline gate. The trainer refuses independent-AT fisheye data without "
                "its signed readiness gate. Re-run with --pose-route independent_at so the SDK "
                "builds the AT and gate chains itself, or produce it with "
                + " then ".join(GATE_TOOLS) + " against this work root's caches and pass "
                "--pipeline-gate PATH."
            )
        gate_path = None
        pose_route = POSE_ROUTE_RAW
        say(
            "[prepare] the dataset manifest carries no independent AT lineage: training on the "
            "capture's own poses with no readiness gate (the trainer asks for one only on AT "
            "data). Quality is bounded by those poses; --pose-route independent_at runs the AT "
            "chain for a delivery."
        )
    else:
        gate_path = Path(pipeline_gate)
        from cloudstudio_3dgs.pipeline.mipmap_gate import load_and_verify_gate

        load_and_verify_gate(gate_path)  # raises on a bad signature
        if pose_route == POSE_ROUTE_AT and at_lineage is False:
            # The AT route's training manifest is published by build_ba_training_manifest.py
            # with that lineage; one without it is not the manifest the gate was built for.
            raise DatasetIncompleteError(
                f"{specs['dataset_manifest'].manifest} carries no independent-AT lineage on the AT route"
            )
        pose_route = POSE_ROUTE_AT if (at_lineage or pose_route == POSE_ROUTE_AT) else POSE_ROUTE_RAW

    # Project the built graph onto the trainer's path contract.
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
    # With refinement on, the trainer supervises against the refined label.
    sky = specs.get("sky_masks_refined") or specs["sky_masks"]
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
        pose_route=pose_route,
        **fields,
    )


def _has_independent_at_lineage(dataset_manifest: Path) -> bool | None:
    """True / False from the built dataset manifest, ``None`` when it cannot be read."""
    import json

    try:
        payload = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    from cloudstudio_3dgs.pipeline.mipmap_gate import INDEPENDENT_AT_ALGORITHM

    lineage = payload.get("training_lineage") or {}
    return lineage.get("independent_at_algorithm_version") == INDEPENDENT_AT_ALGORITHM
