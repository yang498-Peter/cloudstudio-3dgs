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

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

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


def load_dataset_bundle(dataset_root: Path, profile: Any, work_root: Path) -> PreparedScene:
    """Ingest ``dataset_root`` and build its caches into a :class:`PreparedScene`.

    Not implemented here. The adapters, the signed cache graph and the
    automatic tiling rule are the ingestion task's deliverable and already
    live in :mod:`cloudstudio3dgs_sdk.ingest`; what is missing is the glue
    that runs them for a work root and projects the result onto the path
    contract above.

    A caller that already has a prepared scene - house0305, or a dataset an
    earlier SDK run prepared - does not need this at all:
    ``Project.prepare()`` adopts an existing ``prepare_manifest.json`` and
    verifies it instead.
    """
    raise NotImplementedError(
        "dataset ingestion is not implemented in this module. The pieces exist in "
        f"{INGEST_PACKAGE}: call {INGEST_LOAD}(dataset_root) for the capture bundle, then "
        f"{INGEST_PLAN_CACHES}(bundle, profile, cache_root=..., run_root=...) and build the "
        "plan; the remaining work is projecting that onto PreparedScene and writing "
        f"{Path(work_root) / 'prepare' / 'prepare_manifest.json'}. Until that glue lands, place "
        "a verified prepare_manifest.json there and Project.prepare() will adopt it."
    )
