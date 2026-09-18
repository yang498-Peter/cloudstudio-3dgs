"""Adopt a scene that was prepared before the SDK existed.

house0305 was tiled, cached and trained by hand; its as-run trainer configs
are the only complete record of where everything lives. ``adopt_scene``
reads those configs, checks every artefact they name against the shas the
signed manifests carry, and projects the result onto the SDK's prepare
manifest - the single source of truth every later stage binds by path.

Fail closed is the whole design. A path that does not exist, a sha that does
not match, two configs that disagree about the dataset, a tile the manifest
knows and no config covers: each is a :class:`StageRefused` naming the file.
Nothing here is a warning, because a warning would be read once and the arm
configs would then be written against the wrong cache.

What is verified, and against what:

- the tile inputs manifest: its own recorded sha, every tile's
  initialisation PLY (``initialization.sha256``, plus the header's vertex
  count against ``point_count``) and the source LiDAR cloud
  (``source_point_cloud.sha256``);
- the tile geometry manifest: its own sha, its binding to the tile inputs
  manifest, every tile's geometry ``.npz`` and initialisation PLY sha;
- every ownership manifest: its own sha, its binding to the tile inputs
  manifest and to the Face4 cache;
- every backdrop manifest and the coarse prior's background library: their
  own sha and the sky dome checkpoint they rendered (``dome_sha256``);
- the sky mask manifest: its own sha, its binding to the Face4 cache and
  every mask file it lists;
- the Face4 cache manifest: its own sha;
- the coarse initialisation PLY (vertex count), its geometry, the gsplat lock
  and, when given, the frozen sky PLY (row count against the dome).

``verify_prepare_manifest`` is the cheaper re-check ``Project.prepare()``
runs when it adopts a manifest that already exists: every recorded path
must still exist and every recorded digest must still match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.pipeline import file_sha256, read_ply_vertex_count

from cloudstudio3dgs_sdk.bundle import DerivedCaches, PreparedScene
from cloudstudio3dgs_sdk.plan import DERIVED_SCENE_KEYS, DERIVED_TILE_KEYS, DatasetSummary, tile_key
from cloudstudio3dgs_sdk.profile import Profile
from cloudstudio3dgs_sdk.project import StageRefused, digest_matches

# Trainer config key -> PreparedScene field. Everything else shares its name.
SCENE_FIELD_BY_TRAINER_KEY: Mapping[str, str] = {"mipmap_pipeline_gate": "pipeline_gate"}

# The key each signed manifest stores its own sha under. The sha is over the
# canonical (sorted, compact) JSON of the manifest without that key.
SELF_SHA_KEY = {
    "tile_inputs": "tile_inputs_manifest_sha256",
    "tile_geometry": "tile_geometry_manifest_sha256",
    "ownership": "tile_ownership_manifest_sha256",
    "backdrop": "manifest_sha256",
    "view_backgrounds": "manifest_sha256",
    "sky_mask": "sky_mask_manifest_sha256",
    "face_cache": "face_manifest_sha256",
}


def self_sha256(payload: Mapping[str, Any], key: str) -> str:
    """The sha every signed manifest in this repo records under ``key``."""
    body = {name: value for name, value in payload.items() if name != key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass
class AdoptedScene:
    """What ``adopt_scene`` established, ready for ``write_prepare_manifest``."""

    scene: PreparedScene
    dataset: DatasetSummary
    derived_paths: dict[str, str]
    # key -> digest record (same shape as project.digest, always sha256)
    digests: dict[str, dict[str, Any]]
    verified: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sources: dict[str, Any] = field(default_factory=dict)
    # tile_id -> the as-run checkpoint under the config's output_dir, when it
    # exists. With one per tile the plan needs no seed generation: the
    # stand-in backdrops were rendered from exactly these.
    prior_tile_checkpoints: dict[int, str] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "sources": dict(self.sources),
            "verified": list(self.verified),
            "notes": list(self.notes),
        }


class _Ledger:
    """Existence and sha checks that refuse loudly and remember what passed."""

    def __init__(self) -> None:
        self.digests: dict[str, dict[str, Any]] = {}
        self.verified: list[str] = []

    @staticmethod
    def file(path: Path | str, *, what: str) -> Path:
        target = Path(str(path))
        if not target.is_file():
            raise StageRefused(f"{what}: {target} does not exist or is not a file")
        return target

    @staticmethod
    def directory(path: Path | str, *, what: str) -> Path:
        target = Path(str(path))
        if not target.is_dir():
            raise StageRefused(f"{what}: {target} does not exist or is not a directory")
        return target

    def record(self, key: str, path: Path, sha: str) -> None:
        stat = path.stat()
        self.digests[key] = {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": sha,
            "digest_kind": "sha256",
        }

    def sha(self, key: str, path: Path | str, expected: str | None, *, what: str, recorded_in: str) -> str:
        """Hash ``path``; refuse when it is not ``expected``; record the digest."""
        target = self.file(path, what=what)
        actual = file_sha256(target)
        if expected is not None and actual != str(expected):
            raise StageRefused(
                f"{what}: {target} sha256 {actual[:12]} != {str(expected)[:12]} recorded in {recorded_in}"
            )
        self.record(key, target, actual)
        self.verified.append(f"{what}: {target.name} sha256 {actual[:12]}" + (" (recorded)" if expected else ""))
        return actual

    def manifest(self, key: str, path: Path | str, *, kind: str, what: str) -> tuple[dict[str, Any], str | None]:
        """Load a signed manifest, verify its own sha, record its file digest."""
        target = self.file(path, what=what)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise StageRefused(f"{what}: {target} is not readable JSON ({error})") from error
        if not isinstance(payload, dict):
            raise StageRefused(f"{what}: {target} is not a JSON object")
        sha_key = SELF_SHA_KEY[kind]
        recorded = payload.get(sha_key)
        if recorded is not None:
            actual = self_sha256(payload, sha_key)
            if actual != str(recorded):
                raise StageRefused(
                    f"{what}: {target} records {sha_key} {str(recorded)[:12]} but its content hashes to "
                    f"{actual[:12]}; the manifest was edited after it was signed"
                )
            self.verified.append(f"{what}: {target.name} {sha_key} {actual[:12]} verified")
        self.record(key, target, file_sha256(target))
        return payload, (None if recorded is None else str(recorded))


def _load_config(path: Path) -> dict[str, Any]:
    target = _Ledger.file(path, what="trainer config")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StageRefused(f"trainer config {target} is not readable JSON ({error})") from error
    if not isinstance(payload, dict):
        raise StageRefused(f"trainer config {target} is not a JSON object")
    return payload


def _agree(configs: Mapping[str, Mapping[str, Any]], key: str, *, what: str) -> str:
    """One value for ``key`` across every config, or a refusal naming the odd one."""
    values: dict[str, str] = {}
    for label, config in configs.items():
        if key not in config:
            raise StageRefused(f"{what}: {label} carries no '{key}'")
        values[label] = str(config[key])
    distinct = sorted(set(values.values()))
    if len(distinct) != 1:
        detail = "; ".join(f"{label}={value}" for label, value in values.items())
        raise StageRefused(f"{what}: the configs disagree on '{key}': {detail}")
    return distinct[0]


def _same_file(left: Path | str, right: Path | str) -> bool:
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except OSError:
        return False


def _dataclass_kwargs(cls: type, candidates: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the candidates the (possibly evolving) dataclass actually declares."""
    names = set(getattr(cls, "__dataclass_fields__", {}))
    return {key: value for key, value in candidates.items() if key in names}


def adopt_scene(
    tile_configs: Sequence[Path | str],
    coarse_config: Path | str,
    *,
    profile: Profile,
    scene_tag: str | None = None,
    sky_ply: Path | str | None = None,
    sky_dome: Path | str | None = None,
    dataset_root: Path | str | None = None,
    reference_ply: Path | str | None = None,
    reference_alignment: Path | str | None = None,
) -> AdoptedScene:
    """Build the prepared scene from as-run configs, verifying every artefact.

    ``reference_ply`` / ``reference_alignment`` name the competitor model the campaign's
    three-way and off-trajectory comparisons score against. A scene that has one gets those
    comparisons; a first delivery of a new scene has none, and the pipeline's arm steps skip
    them by name rather than fail after a full training run.
    """
    if not tile_configs:
        raise StageRefused("adopt needs at least one tile config")
    ledger = _Ledger()
    notes: list[str] = []

    # -- configs ------------------------------------------------------------
    tiles: dict[int, dict[str, Any]] = {}
    tile_sources: dict[int, str] = {}
    for path in tile_configs:
        config = _load_config(Path(path))
        if "mipmap_tile_id" not in config:
            raise StageRefused(f"{path} is not a tile config: it carries no 'mipmap_tile_id'")
        tile_id = int(config["mipmap_tile_id"])
        if tile_id in tiles:
            raise StageRefused(f"tile {tile_id} is given twice: {tile_sources[tile_id]} and {path}")
        tiles[tile_id] = config
        tile_sources[tile_id] = str(path)
    coarse = _load_config(Path(coarse_config))
    if "mipmap_tile_id" in coarse:
        raise StageRefused(f"{coarse_config} carries 'mipmap_tile_id'; the coarse prior is tile-free")
    labelled_tiles = {f"tile {tile_id} config": config for tile_id, config in sorted(tiles.items())}
    everything = dict(labelled_tiles)
    everything["coarse config"] = coarse

    # -- the scene-level trainer paths ------------------------------------
    trainer_paths: dict[str, str] = {}
    for key in profile.dataset_contract["trainer_path_keys"]:
        value = _agree(everything, key, what="scene path")
        if key.endswith("_root"):
            ledger.directory(value, what=key)
        else:
            ledger.file(value, what=key)
        trainer_paths[key] = value

    # -- Face4 cache: the count every view-based estimate rests on ---------
    face_manifest, face_sha = ledger.manifest(
        "face_cache_manifest", trainer_paths["face_cache_manifest"], kind="face_cache", what="face cache manifest"
    )
    images = face_manifest.get("images")
    if not isinstance(images, list) or not images:
        raise StageRefused(f"face cache manifest {trainer_paths['face_cache_manifest']} lists no images")
    if all(isinstance(image, Mapping) and isinstance(image.get("faces"), list) for image in images):
        train_view_count = sum(len(image["faces"]) for image in images)
    else:
        train_view_count = len(images)
    ledger.verified.append(f"face cache manifest: {train_view_count} training face records")

    # -- tile inputs --------------------------------------------------------
    tile_inputs_path = _agree(labelled_tiles, "tile_inputs_manifest", what="tile inputs")
    tile_inputs_root = ledger.directory(
        _agree(labelled_tiles, "tile_inputs_root", what="tile inputs"), what="tile_inputs_root"
    )
    tile_inputs, tile_inputs_sha = ledger.manifest(
        "tile_inputs_manifest", tile_inputs_path, kind="tile_inputs", what="tile inputs manifest"
    )
    entries = tile_inputs.get("tiles")
    if not isinstance(entries, list) or not entries:
        raise StageRefused(f"tile inputs manifest {tile_inputs_path} lists no tiles")
    manifest_tiles = {int(entry["tile_id"]): entry for entry in entries}
    missing_configs = sorted(set(manifest_tiles) - set(tiles))
    if missing_configs:
        raise StageRefused(
            f"tile inputs manifest {tile_inputs_path} has tiles {missing_configs} but no config was given for them"
        )
    extra_configs = sorted(set(tiles) - set(manifest_tiles))
    if extra_configs:
        raise StageRefused(f"configs name tiles {extra_configs} that {tile_inputs_path} does not know")
    derived: dict[str, str] = {
        "tile_inputs_manifest": str(Path(tile_inputs_path)),
        "tile_inputs_root": str(tile_inputs_root),
    }
    for tile_id, entry in sorted(manifest_tiles.items()):
        name = str(entry.get("name", f"Tile_{tile_id}"))
        init = entry.get("initialization")
        if not isinstance(init, Mapping) or "path" not in init:
            raise StageRefused(f"tile inputs manifest {tile_inputs_path}: {name} has no initialization block")
        ply = tile_inputs_root / str(init["path"])
        configured = tiles[tile_id].get("initialization_ply")
        if configured is None or not _same_file(configured, ply):
            raise StageRefused(
                f"{name}: config initialization_ply {configured} is not the manifest's {ply}"
            )
        key = tile_key(tile_id, "initialization_ply")
        ledger.sha(key, ply, init.get("sha256"), what=f"{name} initialisation PLY", recorded_in=Path(tile_inputs_path).name)
        if "point_count" in init:
            rows = read_ply_vertex_count(ply)
            if rows != int(init["point_count"]):
                raise StageRefused(
                    f"{name}: {ply} has {rows} vertices, the manifest records point_count {init['point_count']}"
                )
        derived[key] = str(ply)
    source_cloud = tile_inputs.get("source_point_cloud")
    if isinstance(source_cloud, Mapping) and source_cloud.get("path"):
        ledger.sha(
            "lidar_cloud",
            source_cloud["path"],
            source_cloud.get("sha256"),
            what="source LiDAR cloud",
            recorded_in=Path(tile_inputs_path).name,
        )
        derived["lidar_cloud"] = str(Path(str(source_cloud["path"])))
    else:
        raise StageRefused(f"tile inputs manifest {tile_inputs_path} records no source_point_cloud")

    # -- tile geometry ------------------------------------------------------
    geometry_path = Path(_agree(labelled_tiles, "initialization_geometry_manifest", what="tile geometry"))
    geometry, _ = ledger.manifest("tile_geometry_manifest", geometry_path, kind="tile_geometry", what="tile geometry manifest")
    bound = geometry.get("tile_inputs_manifest_sha256")
    if bound is not None and tile_inputs_sha is not None and str(bound) != tile_inputs_sha:
        raise StageRefused(
            f"tile geometry manifest {geometry_path} was built for tile inputs {str(bound)[:12]}, "
            f"not the given {tile_inputs_sha[:12]}"
        )
    geometry_tiles = {int(entry["tile_id"]): entry for entry in geometry.get("tiles", [])}
    for tile_id, entry in sorted(manifest_tiles.items()):
        name = str(entry.get("name", f"Tile_{tile_id}"))
        geo = geometry_tiles.get(tile_id)
        if geo is None:
            raise StageRefused(f"tile geometry manifest {geometry_path} has no entry for {name}")
        expected_ply_sha = geo.get("initialization_ply_sha256")
        actual_ply_sha = ledger.digests[tile_key(tile_id, "initialization_ply")]["sha256"]
        if expected_ply_sha is not None and str(expected_ply_sha) != actual_ply_sha:
            raise StageRefused(
                f"{name}: geometry was computed for initialisation PLY {str(expected_ply_sha)[:12]}, "
                f"the PLY on disk is {actual_ply_sha[:12]}"
            )
        block = geo.get("geometry")
        if not isinstance(block, Mapping) or "path" not in block:
            raise StageRefused(f"tile geometry manifest {geometry_path}: {name} has no geometry path")
        npz = geometry_path.parent / str(block["path"])
        configured = tiles[tile_id].get("initialization_geometry")
        if configured is None or not _same_file(configured, npz):
            raise StageRefused(f"{name}: config initialization_geometry {configured} is not the manifest's {npz}")
        key = tile_key(tile_id, "initialization_geometry")
        ledger.sha(key, npz, block.get("sha256"), what=f"{name} initialisation geometry", recorded_in=geometry_path.name)
        derived[key] = str(npz)
    derived["tile_geometry_manifest"] = str(geometry_path)

    # -- gsplat lock, coarse initialisation ---------------------------------
    lock = ledger.file(_agree(everything, "gsplat_lock", what="gsplat lock"), what="gsplat_lock")
    ledger.record("gsplat_lock", lock, file_sha256(lock))
    derived["gsplat_lock"] = str(lock)

    # -- competitor reference model (optional) --------------------------------
    # The campaign scores three-way and off-trajectory strips against a competitor delivery.
    # A scene that has one records it here; the pipeline skips those comparisons by name for
    # a scene that does not, instead of failing after a full training run.
    if (reference_ply is None) != (reference_alignment is None):
        raise StageRefused(
            "give both --reference-ply and --reference-alignment or neither; a reference model "
            "without its rigid alignment (or the reverse) cannot be scored against"
        )
    if reference_ply is not None:
        ref = ledger.file(reference_ply, what="reference PLY")
        align = ledger.file(reference_alignment, what="reference alignment")
        ledger.record("reference_ply", ref, file_sha256(ref))
        ledger.record("reference_alignment", align, file_sha256(align))
        derived["reference_ply"] = str(ref)
        derived["reference_alignment"] = str(align)
        ledger.verified.append(f"reference model: {ref.name} + {align.name}")
    else:
        notes.append(
            "no reference (competitor) model given: the three-way and off-trajectory "
            "comparisons will be skipped, and the report's sharpness gate stays UNVERIFIED"
        )
    global_ply = ledger.file(coarse.get("initialization_ply", ""), what="coarse initialization_ply")
    global_init_point_count = read_ply_vertex_count(global_ply)
    ledger.record("global_init_ply", global_ply, file_sha256(global_ply))
    ledger.verified.append(f"coarse initialisation PLY: {global_ply.name} {global_init_point_count} vertices")
    global_geometry = ledger.file(coarse.get("initialization_geometry", ""), what="coarse initialization_geometry")
    ledger.record("global_init_geometry", global_geometry, file_sha256(global_geometry))
    derived["global_init_ply"] = str(global_ply)
    derived["global_init_geometry"] = str(global_geometry)

    # -- sky masks ----------------------------------------------------------
    sky_blocks = {
        label: config.get("sky_supervision") for label, config in labelled_tiles.items()
    }
    if any(not isinstance(block, Mapping) for block in sky_blocks.values()):
        raise StageRefused("every tile config must carry a sky_supervision block to adopt its masks")
    sky_manifest_path = _agree(sky_blocks, "mask_manifest", what="sky masks")
    sky_root = ledger.directory(_agree(sky_blocks, "mask_root", what="sky masks"), what="sky mask root")
    sky_manifest, _ = ledger.manifest("sky_mask_manifest", sky_manifest_path, kind="sky_mask", what="sky mask manifest")
    _check_face_binding(sky_manifest, face_sha, what=f"sky mask manifest {Path(sky_manifest_path).name}")
    masks = sky_manifest.get("masks")
    if not isinstance(masks, list) or not masks:
        raise StageRefused(f"sky mask manifest {sky_manifest_path} lists no masks")
    for record in masks:
        mask = ledger.file(sky_root / str(record.get("mask_path", "")), what="sky mask file")
        expected = record.get("mask_sha256")
        if expected is not None and file_sha256(mask) != str(expected):
            raise StageRefused(
                f"sky mask {mask} does not match mask_sha256 {str(expected)[:12]} in {Path(sky_manifest_path).name}"
            )
    ledger.verified.append(f"sky masks: {len(masks)} mask files match {Path(sky_manifest_path).name}")
    derived["sky_mask_manifest"] = str(Path(sky_manifest_path))
    derived["sky_mask_root"] = str(sky_root)

    # -- ownership and backdrops, per tile ----------------------------------
    dome_shas: dict[str, str] = {}
    dome_sources: dict[str, str] = {}
    dome_counts: set[int] = set()
    ownership_caches: dict[int, tuple[Path, Path]] = {}
    for tile_id, entry in sorted(manifest_tiles.items()):
        name = str(entry.get("name", f"Tile_{tile_id}"))
        config = tiles[tile_id]
        own_manifest_path = config.get("tile_ownership_cache_manifest")
        own_root = config.get("tile_ownership_cache_root")
        if not own_manifest_path or not own_root:
            raise StageRefused(f"{name}: config carries no tile_ownership_cache_manifest / _root")
        own_root = ledger.directory(own_root, what=f"{name} ownership root")
        own_key = tile_key(tile_id, "ownership_manifest")
        own, _ = ledger.manifest(own_key, own_manifest_path, kind="ownership", what=f"{name} ownership manifest")
        if "tile_id" in own and int(own["tile_id"]) != tile_id:
            raise StageRefused(f"{name}: ownership manifest {own_manifest_path} is for tile {own['tile_id']}")
        bound = own.get("tile_inputs_manifest_sha256")
        if bound is not None and tile_inputs_sha is not None and str(bound) != tile_inputs_sha:
            raise StageRefused(
                f"{name}: ownership manifest {own_manifest_path} was built for tile inputs {str(bound)[:12]}, "
                f"not the given {tile_inputs_sha[:12]}"
            )
        _check_face_binding(own, face_sha, what=f"{name} ownership manifest")
        derived[own_key] = str(Path(str(own_manifest_path)))
        derived[tile_key(tile_id, "ownership_root")] = str(own_root)
        ownership_caches[tile_id] = (Path(str(own_manifest_path)), own_root)

        bd_manifest_path = config.get("background_image_manifest")
        bd_root = config.get("background_image_root")
        if not bd_manifest_path or not bd_root:
            raise StageRefused(f"{name}: config carries no background_image_manifest / _root")
        bd_root = ledger.directory(bd_root, what=f"{name} backdrop root")
        bd_key = tile_key(tile_id, "backdrop_manifest")
        backdrop, _ = ledger.manifest(bd_key, bd_manifest_path, kind="backdrop", what=f"{name} backdrop manifest")
        if "tile_id" in backdrop and int(backdrop["tile_id"]) != tile_id:
            raise StageRefused(f"{name}: backdrop manifest {bd_manifest_path} is for tile {backdrop['tile_id']}")
        bound = backdrop.get("source_tile_inputs_manifest_sha256")
        if bound is not None and tile_inputs_sha is not None and str(bound) != tile_inputs_sha:
            raise StageRefused(
                f"{name}: backdrop manifest {bd_manifest_path} was rendered for tile inputs {str(bound)[:12]}, "
                f"not the given {tile_inputs_sha[:12]}"
            )
        _collect_dome(backdrop, dome_shas, dome_sources, dome_counts, what=f"{name} backdrop manifest")
        derived[bd_key] = str(Path(str(bd_manifest_path)))
        derived[tile_key(tile_id, "backdrop_root")] = str(bd_root)

    # -- the coarse prior's background library ------------------------------
    vb_manifest_path = coarse.get("background_image_manifest")
    vb_root = coarse.get("background_image_root")
    if not vb_manifest_path or not vb_root:
        raise StageRefused("coarse config carries no background_image_manifest / _root")
    vb_root = ledger.directory(vb_root, what="global view backgrounds root")
    view_backgrounds, _ = ledger.manifest(
        "global_view_backgrounds_manifest", vb_manifest_path, kind="view_backgrounds", what="global view backgrounds manifest"
    )
    _collect_dome(view_backgrounds, dome_shas, dome_sources, dome_counts, what="global view backgrounds manifest")
    derived["global_view_backgrounds_manifest"] = str(Path(str(vb_manifest_path)))
    derived["global_view_backgrounds_root"] = str(vb_root)

    # -- the sky dome and its frozen PLY ------------------------------------
    if len(set(dome_shas.values())) > 1:
        detail = "; ".join(f"{label}={sha[:12]}" for label, sha in dome_shas.items())
        raise StageRefused(f"the backdrops were rendered from different sky domes: {detail}")
    expected_dome_sha = next(iter(dome_shas.values()), None)
    dome_path = sky_dome if sky_dome is not None else next(iter(dome_sources.values()), None)
    if dome_path is None:
        raise StageRefused(
            "no sky dome checkpoint: none of the backdrop manifests records dome_source and --sky-dome was not given"
        )
    ledger.sha(
        "sky_dome_checkpoint",
        dome_path,
        expected_dome_sha,
        what="sky dome checkpoint",
        recorded_in="the backdrop manifests (dome_sha256)",
    )
    derived["sky_dome_checkpoint"] = str(Path(str(dome_path)))
    if sky_ply is not None:
        ply = ledger.file(sky_ply, what="sky PLY")
        rows = read_ply_vertex_count(ply)
        if dome_counts and rows not in dome_counts:
            raise StageRefused(
                f"sky PLY {ply} has {rows} rows; the backdrops record a dome of {sorted(dome_counts)} rows, "
                "so this is not that dome's export"
            )
        ledger.record("sky_dome_ply", ply, file_sha256(ply))
        ledger.verified.append(f"sky PLY: {ply.name} {rows} rows")
        derived["sky_dome_ply"] = str(ply)
    else:
        notes.append("no --sky-ply given: the sky_dome_ply prepare step will export the dome checkpoint")

    # -- summary ------------------------------------------------------------
    tag = scene_tag or str(tiles[min(tiles)].get("run_id", "")).split("-")[0]
    if not tag:
        raise StageRefused("no scene tag: give --scene-tag; the tile configs' run_id carries none")
    # The cap floor rule ("not below the previous generation's final population") needs that
    # population. The adopted arms ARE the previous generation, and their trainer telemetry
    # records the final count exactly; reading it costs nothing, where loading a 2 GB checkpoint
    # to count rows would. A tile without telemetry simply gets no floor, and says so.
    previous_population: dict[int, int] = {}
    for tile_id, config in sorted(tiles.items()):
        telemetry = Path(str(config.get("output_dir", ""))) / "monitor" / "progress.jsonl"
        if not config.get("output_dir") or not telemetry.is_file():
            notes.append(f"tile {tile_id}: no trainer telemetry, so the cap floor rule cannot fire for it")
            continue
        last = None
        for line in telemetry.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = line
        try:
            count = int(json.loads(last)["gaussian_count"]) if last else None
        except (ValueError, KeyError, TypeError):
            count = None
        if count is None:
            notes.append(f"tile {tile_id}: telemetry has no final gaussian_count; no cap floor for it")
            continue
        previous_population[tile_id] = count
    if previous_population:
        notes.append(
            "previous-generation final populations read from trainer telemetry: "
            + ", ".join(f"tile {k}={v:,}" for k, v in sorted(previous_population.items()))
            + "; the cap floor rule sees these, not any hand-set cap from an earlier generation"
        )
    dataset = DatasetSummary.from_tile_inputs_manifest(
        tile_inputs,
        scene_tag=tag,
        train_view_count=train_view_count,
        global_init_point_count=global_init_point_count,
        lidar_point_count=0,
        previous_final_population=previous_population or None,
    )
    notes.append(
        "lidar_point_count is 0: the tile inputs manifest records the source cloud's sha, not its point count, "
        "and adopt does not read LAS headers"
    )
    priors: dict[int, str] = {}
    for tile_id, config in sorted(tiles.items()):
        checkpoint = Path(str(config.get("output_dir", ""))) / "checkpoints" / "latest.pt"
        if config.get("output_dir") and checkpoint.is_file():
            priors[tile_id] = str(checkpoint)
    if priors:
        notes.append(
            "prior tile checkpoints recorded from the configs' output_dir (existence only, not hashed): "
            + ", ".join(f"tile {tile_id}" for tile_id in priors)
        )
    root = Path(str(dataset_root)) if dataset_root is not None else Path(trainer_paths["recording_root"])
    scene_fields: dict[str, Any] = {"scene_tag": tag, "dataset_root": root}
    for key, value in trainer_paths.items():
        scene_fields[SCENE_FIELD_BY_TRAINER_KEY.get(key, key)] = Path(value)
    for key in ("lidar_cloud", "tile_inputs_manifest", "tile_inputs_root", "tile_geometry_manifest",
                "global_init_ply", "global_init_geometry"):
        scene_fields[key] = Path(derived[key])
    caches = DerivedCaches(
        **_dataclass_kwargs(
            DerivedCaches,
            {
                "sky_mask_manifest": Path(derived["sky_mask_manifest"]),
                "sky_mask_root": Path(derived["sky_mask_root"]),
                "sky_dome_checkpoint": Path(derived["sky_dome_checkpoint"]),
                "sky_dome_ply": Path(derived["sky_dome_ply"]) if "sky_dome_ply" in derived else None,
                "global_background_manifest": Path(derived["global_view_backgrounds_manifest"]),
                "global_background_root": Path(derived["global_view_backgrounds_root"]),
                "tile_ownership": ownership_caches,
            },
        )
    )
    scene_fields["caches"] = caches
    scene = PreparedScene(**_dataclass_kwargs(PreparedScene, scene_fields))
    return AdoptedScene(
        scene=scene,
        dataset=dataset,
        derived_paths=derived,
        digests=ledger.digests,
        verified=ledger.verified,
        notes=notes,
        sources={
            "tile_configs": {str(tile_id): source for tile_id, source in sorted(tile_sources.items())},
            "coarse_config": str(coarse_config),
            "sky_ply": None if sky_ply is None else str(sky_ply),
        },
        prior_tile_checkpoints=priors,
    )


def _check_face_binding(manifest: Mapping[str, Any], face_sha: str | None, *, what: str) -> None:
    bound = manifest.get("source_face_manifest_sha256")
    if bound is not None and face_sha is not None and str(bound) != face_sha:
        raise StageRefused(
            f"{what} was built on Face4 cache {str(bound)[:12]}, the given face cache manifest is {face_sha[:12]}"
        )


def _collect_dome(
    manifest: Mapping[str, Any],
    shas: dict[str, str],
    sources: dict[str, str],
    counts: set[int],
    *,
    what: str,
) -> None:
    if manifest.get("dome_sha256"):
        shas[what] = str(manifest["dome_sha256"])
    if manifest.get("dome_source"):
        sources[what] = str(manifest["dome_source"])
    standin = manifest.get("standin")
    if isinstance(standin, Mapping) and standin.get("dome_count") is not None:
        counts.add(int(standin["dome_count"]))


# --------------------------------------------------------------------------
# Re-verification of an existing prepare manifest
# --------------------------------------------------------------------------

# Scene inputs that must exist for any manifest, adopted or freshly built:
# without them there is nothing to train from.
REQUIRED_DERIVED = (
    "tile_inputs_manifest",
    "tile_inputs_root",
    "tile_geometry_manifest",
    "global_init_ply",
    "global_init_geometry",
    "gsplat_lock",
)


def verify_prepare_manifest(payload: Mapping[str, Any], *, manifest_path: Path | str = "prepare_manifest.json") -> list[str]:
    """Refuse unless every path the manifest records is still there and unchanged.

    Trainer paths and the required scene inputs must exist. Derived caches a
    fresh build has not produced yet may be absent from ``derived_paths`` or
    carry no digest; every derived path that *does* carry a digest (which is
    every artefact ``adopt_scene`` checked) must exist and still match it.
    """
    where = Path(str(manifest_path))
    for key in ("dataset", "trainer_paths"):
        if key not in payload:
            raise StageRefused(f"{where} is not a prepare manifest (needs 'dataset' and 'trainer_paths')")
    trainer_paths = payload.get("trainer_paths") or {}
    derived = payload.get("derived_paths") or {}
    digests = payload.get("digests") or {}
    if not isinstance(trainer_paths, Mapping) or not isinstance(derived, Mapping) or not isinstance(digests, Mapping):
        raise StageRefused(f"{where}: trainer_paths, derived_paths and digests must be objects")
    checked: list[str] = []
    for key, value in sorted(trainer_paths.items()):
        target = Path(str(value))
        if key.endswith("_root"):
            if not target.is_dir():
                raise StageRefused(f"{where}: trainer path {key} = {target} is not a directory")
        elif not target.is_file():
            raise StageRefused(f"{where}: trainer path {key} = {target} is not a file")
        checked.append(key)
    for key in REQUIRED_DERIVED:
        if key not in derived:
            raise StageRefused(f"{where}: derived_paths carries no '{key}'; prepare has not produced it")
        target = Path(str(derived[key]))
        if not target.exists():
            raise StageRefused(f"{where}: {key} = {target} does not exist")
        checked.append(key)
    for key, value in sorted(derived.items()):
        if key in digests and not Path(str(value)).exists():
            raise StageRefused(f"{where}: {key} = {value} was verified at adopt time and is now missing")
    for key, record in sorted(digests.items()):
        ok, detail = _adopted_digest_matches(record)
        if not ok:
            raise StageRefused(
                f"{where}: '{key}' no longer matches the digest recorded when it was adopted - {detail}"
            )
        checked.append(f"digest:{key}")
    return checked


def _adopted_digest_matches(record: Mapping[str, Any]) -> tuple[bool, str]:
    """Like ``project.digest_matches`` but re-hashes a sha256 record at any size.

    ``adopt_scene`` hashes every artefact it verifies, including the ones
    above the project's size cut-off (the LiDAR cloud, the tile geometry);
    the recorded sha is what has to hold, so it is recomputed rather than
    downgraded to a size/mtime stamp.
    """
    if not isinstance(record, Mapping) or record.get("digest_kind") != "sha256" or "path" not in record:
        return digest_matches(record)
    target = Path(str(record["path"]))
    if not target.is_file():
        return False, f"missing {target}"
    size = target.stat().st_size
    if int(record.get("bytes", -1)) != size:
        return False, f"{target.name} is {size} bytes, recorded {record.get('bytes')}"
    actual = file_sha256(target)
    if actual != str(record.get("sha256")):
        return False, f"{target.name} sha256 {actual[:12]} != recorded {str(record.get('sha256'))[:12]}"
    return True, ""


__all__ = [
    "DERIVED_SCENE_KEYS",
    "DERIVED_TILE_KEYS",
    "AdoptedScene",
    "SELF_SHA_KEY",
    "adopt_scene",
    "self_sha256",
    "verify_prepare_manifest",
]
