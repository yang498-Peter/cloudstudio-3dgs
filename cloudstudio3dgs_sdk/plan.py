"""Turn a profile plus a discovered dataset into the concrete list of steps.

Nothing here runs, opens a checkpoint or touches a GPU. ``build_plan`` is a
pure function of (profile, dataset summary, roots, options), which is what
makes ``run_all(dry_run=True)`` honest: the plan the user reads is the same
object the stages later execute, step for step.

Each step carries its argv, its declared outputs and an :class:`Estimate`.
Estimates are derived from the profile's ``cost_model`` and every one of them
says whether the number behind it was measured, extrapolated or guessed, so
"6 hours" never reads as more certain than it is.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cloudstudio3dgs_sdk.profile import (
    EXTRAPOLATED,
    INFERRED,
    MEASURED,
    UNMEASURED,
    Profile,
    canonical_json,
    thaw,
)

STAGES = ("prepare", "train", "deliver", "report")

GIB = 1024 ** 3


# --------------------------------------------------------------------------
# Discovered dataset
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSummary:
    """One tile as ``prepare()`` recorded it."""

    tile_id: int
    name: str
    view_count: int
    init_point_count: int
    init_sha256: str = ""
    # Only a re-run of a scene that was already delivered can fill this in;
    # it is what lets the cap floor rule fire.
    previous_final_population: int | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "name": self.name,
            "view_count": self.view_count,
            "init_point_count": self.init_point_count,
            "init_sha256": self.init_sha256,
            "previous_final_population": self.previous_final_population,
        }


@dataclass(frozen=True)
class DatasetSummary:
    """The facts a plan needs about a prepared scene. No file access."""

    scene_tag: str
    tiles: tuple[TileSummary, ...]
    train_view_count: int
    global_init_point_count: int
    lidar_point_count: int = 0
    has_reference_model: bool = False
    # True when these numbers were derived from the capture rather than read
    # from a prepare manifest (see cloudstudio3dgs_sdk.discover). A plan built
    # on one is costable but not runnable: Project refuses to train from it.
    estimated: bool = False
    # One sentence per estimated field. render() prints them under ESTIMATED.
    estimate_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tiles:
            raise ValueError("a dataset summary needs at least one tile")
        ids = [tile.tile_id for tile in self.tiles]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate tile ids: {sorted(ids)}")
        if sorted(ids) != ids:
            raise ValueError(f"tiles must be ordered by tile_id, got {ids}")

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    def as_json(self) -> dict[str, Any]:
        return {
            "scene_tag": self.scene_tag,
            "tiles": [tile.as_json() for tile in self.tiles],
            "train_view_count": self.train_view_count,
            "global_init_point_count": self.global_init_point_count,
            "lidar_point_count": self.lidar_point_count,
            "has_reference_model": self.has_reference_model,
            "estimated": self.estimated,
            "estimate_notes": list(self.estimate_notes),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "DatasetSummary":
        tiles = tuple(
            TileSummary(
                tile_id=int(tile["tile_id"]),
                name=str(tile["name"]),
                view_count=int(tile["view_count"]),
                init_point_count=int(tile["init_point_count"]),
                init_sha256=str(tile.get("init_sha256", "")),
                previous_final_population=(
                    None if tile.get("previous_final_population") is None else int(tile["previous_final_population"])
                ),
            )
            for tile in payload["tiles"]
        )
        return cls(
            scene_tag=str(payload["scene_tag"]),
            tiles=tiles,
            train_view_count=int(payload["train_view_count"]),
            global_init_point_count=int(payload["global_init_point_count"]),
            lidar_point_count=int(payload.get("lidar_point_count", 0)),
            has_reference_model=bool(payload.get("has_reference_model", False)),
            # An estimate written out and fed back in through --summary is
            # still an estimate; the flag has to survive the round trip or the
            # refusal in Project can be walked around with a text editor.
            estimated=bool(payload.get("estimated", False)),
            estimate_notes=tuple(str(note) for note in payload.get("estimate_notes", ())),
        )

    @classmethod
    def from_tile_inputs_manifest(
        cls,
        payload: Mapping[str, Any],
        *,
        scene_tag: str,
        train_view_count: int,
        global_init_point_count: int,
        lidar_point_count: int = 0,
        has_reference_model: bool = False,
        previous_final_population: Mapping[int, int] | None = None,
    ) -> "DatasetSummary":
        """Read the existing ``tile_inputs_manifest.json`` shape directly.

        This is the format ``materialize_lidar_tile_inputs.py`` already writes,
        so a scene that was tiled before the SDK existed (house0305) can be
        planned without re-running anything.
        """
        previous = dict(previous_final_population or {})
        tiles: list[TileSummary] = []
        for entry in sorted(payload["tiles"], key=lambda item: int(item["tile_id"])):
            tile_id = int(entry["tile_id"])
            init = entry.get("initialization", {})
            tiles.append(
                TileSummary(
                    tile_id=tile_id,
                    name=str(entry.get("name", f"Tile_{tile_id}")),
                    view_count=int(entry["view_count"]),
                    init_point_count=int(init["point_count"]),
                    init_sha256=str(init.get("sha256", "")),
                    previous_final_population=previous.get(tile_id),
                )
            )
        return cls(
            scene_tag=scene_tag,
            tiles=tuple(tiles),
            train_view_count=train_view_count,
            global_init_point_count=global_init_point_count,
            lidar_point_count=lidar_point_count,
            has_reference_model=has_reference_model,
        )


# --------------------------------------------------------------------------
# Estimates and steps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    seconds: float
    disk_bytes: int
    basis: str
    confidence: str = MEASURED

    def __add__(self, other: "Estimate") -> "Estimate":
        order = (MEASURED, EXTRAPOLATED, INFERRED, UNMEASURED)
        worst = max(self.confidence, other.confidence, key=lambda value: order.index(value) if value in order else 99)
        return Estimate(self.seconds + other.seconds, self.disk_bytes + other.disk_bytes, "sum", worst)

    def as_json(self) -> dict[str, Any]:
        return {
            "seconds": round(self.seconds, 1),
            "disk_bytes": self.disk_bytes,
            "basis": self.basis,
            "confidence": self.confidence,
        }


ZERO_ESTIMATE = Estimate(0.0, 0, "none")


@dataclass(frozen=True)
class PlannedStep:
    """One unit of work, fully addressed before anything runs."""

    name: str
    stage: str
    resource: str  # "cpu" | "gpu" | "external"
    estimate: Estimate
    command: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    config: Mapping[str, Any] | None = None
    config_path: str = ""
    note: str = ""
    blocking: str = ""  # non-empty means this step cannot run yet, and why

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "stage": self.stage,
            "resource": self.resource,
            "estimate": self.estimate.as_json(),
            "command": list(self.command),
            "outputs": list(self.outputs),
            "note": self.note,
            "blocking": self.blocking,
        }
        if self.config is not None:
            payload["config_path"] = self.config_path
            payload["config"] = thaw(self.config)
        return payload


@dataclass(frozen=True)
class Plan:
    profile_name: str
    profile_version: str
    profile_sha256: str
    scene_tag: str
    dataset_root: str
    work_root: str
    generations: tuple[str, ...]
    steps: tuple[PlannedStep, ...]
    dataset: DatasetSummary
    tile_caps: Mapping[int, int]
    delivery_tag: str
    warnings: tuple[str, ...] = ()

    def stage_steps(self, stage: str) -> tuple[PlannedStep, ...]:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}")
        return tuple(step for step in self.steps if step.stage == stage)

    def total(self, stage: str | None = None) -> Estimate:
        steps = self.steps if stage is None else self.stage_steps(stage)
        total = ZERO_ESTIMATE
        for step in steps:
            total = total + step.estimate
        return total

    def blocking_steps(self) -> tuple[PlannedStep, ...]:
        return tuple(step for step in self.steps if step.blocking)

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": self.profile_name,
            "profile_version": self.profile_version,
            "profile_sha256": self.profile_sha256,
            "scene_tag": self.scene_tag,
            "dataset_root": self.dataset_root,
            "work_root": self.work_root,
            "delivery_tag": self.delivery_tag,
            "generations": list(self.generations),
            "dataset": self.dataset.as_json(),
            "tile_caps": {str(tile): cap for tile, cap in sorted(self.tile_caps.items())},
            "warnings": list(self.warnings),
            "steps": [step.as_json() for step in self.steps],
        }

    @property
    def plan_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.as_json()).encode("utf-8")).hexdigest()

    def render(self, *, show_commands: bool = True) -> str:
        """The dry-run transcript."""
        lines: list[str] = []
        lines.append(f"plan {self.scene_tag} profile={self.profile_name}@{self.profile_version}")
        lines.append(f"  profile_sha256 {self.profile_sha256}")
        lines.append(f"  plan_sha256    {self.plan_sha256}")
        lines.append(f"  dataset        {self.dataset_root}")
        lines.append(f"  work           {self.work_root}")
        marker = " [ESTIMATED]" if self.dataset.estimated else ""
        lines.append(
            f"  tiles          {self.dataset.tile_count}{marker}"
            f" ({', '.join(f'{t.name} {t.view_count}v init {t.init_point_count/1e6:.2f}M cap {self.tile_caps[t.tile_id]/1e6:.2f}M' for t in self.dataset.tiles)})"
        )
        lines.append(f"  generations    {', '.join(self.generations)}")
        if self.dataset.estimated:
            # Printed before the step list, not after it: whoever reads only the
            # top of the transcript has to see that the tile numbers under these
            # costs were guessed from the capture.
            lines.append("")
            lines.append(
                "ESTIMATED DATASET - this scene has no prepare manifest, so the tile boxes, "
                "the per-tile view counts and the initialisation counts below were derived "
                "from the capture, not measured by prepare(). Every time and disk figure "
                "downstream inherits their error. A real run refuses this summary."
            )
            for note in self.dataset.estimate_notes:
                lines.append(f"  - {note}")
        for stage in STAGES:
            steps = self.stage_steps(stage)
            if not steps:
                continue
            total = self.total(stage)
            lines.append("")
            lines.append(
                f"[{stage}] {len(steps)} steps, {_hms(total.seconds)}, {_gb(total.disk_bytes)} disk"
            )
            for step in steps:
                flag = "!" if step.blocking else " "
                lines.append(
                    f" {flag} {step.name:<34} {step.resource:<8} {_hms(step.estimate.seconds):>9}"
                    f" {_gb(step.estimate.disk_bytes):>9}  [{step.estimate.confidence}]"
                )
                if step.note:
                    lines.append(f"      note: {step.note}")
                if step.blocking:
                    lines.append(f"      BLOCKING: {step.blocking}")
                if show_commands and step.command:
                    lines.append(f"      $ {' '.join(step.command)}")
                if show_commands and step.config_path:
                    lines.append(f"      writes config {step.config_path}")
        grand = self.total()
        lines.append("")
        lines.append(f"total {_hms(grand.seconds)} wall, {_gb(grand.disk_bytes)} peak disk [{grand.confidence}]")
        for warning in self.warnings:
            lines.append(f"WARNING {warning}")
        for step in self.blocking_steps():
            lines.append(f"BLOCKED {step.name}: {step.blocking}")
        return "\n".join(lines)


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _gb(size: int) -> str:
    if size <= 0:
        return "-"
    return f"{size / GIB:.1f}GB"


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------


def tile_cap(profile: Profile, tile: TileSummary, *, vram_gib: float | None = None) -> int:
    """cap_max for one tile: the measured ratio, clamped by floor and ceiling.

    The ratio is the whole rule on a fresh scene. The floor only exists on a
    re-delivery (it needs a previous population); the ceiling only bites on a
    card too small for the tile, and biting is the point - the campaign lost a
    run to exactly that.
    """
    rules = profile.tile_rules
    runtime = profile.runtime
    raw = tile.init_point_count * float(rules["cap_ratio_of_initialisation"])
    previous = tile.previous_final_population
    # The floor fires only when the ratio would run this tile tighter than it
    # already ran; then it takes that population back plus a little headroom.
    if previous is not None and raw < float(previous):
        raw = float(previous) * float(rules.get("cap_floor_headroom", 1.0))
    step = int(rules["cap_round_to"])
    cap = int(math.floor(raw / step + 0.5)) * step
    if vram_gib is not None:
        ceiling_step = int(runtime["max_gaussians_per_gib_vram"]) * vram_gib * float(runtime["vram_safety_factor"])
        ceiling = int(math.floor(ceiling_step / step)) * step
        cap = min(cap, max(ceiling, step))
    return max(cap, step)


def tile_max_steps(profile: Profile, view_count: int) -> int:
    return int(profile.tiling["epochs_for_max_steps"]) * int(view_count)


def prune_switch_step(profile: Profile, max_steps: int) -> int:
    return int(round(max_steps * float(profile.tiling["prune_switch_fraction_of_max_steps"])))


def tile_seconds_per_step(profile: Profile, cap_max: int) -> float:
    model = profile.cost_model["tile_train_seconds_per_step"]
    return float(model["intercept"]) + float(model["per_million_cap"]) * (cap_max / 1e6)


# --------------------------------------------------------------------------
# Paths inside the work root
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkLayout:
    """Where the SDK puts things. ``runs`` is a tools/pipeline.py run_root."""

    root: Path

    @property
    def prepare(self) -> Path:
        return self.root / "prepare"

    @property
    def prepare_manifest(self) -> Path:
        return self.prepare / "prepare_manifest.json"

    @property
    def caches(self) -> Path:
        return self.root / "caches"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def state(self) -> Path:
        return self.root / "sdk_state"

    @property
    def report(self) -> Path:
        return self.root / "report"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def pipeline_config(self) -> Path:
        return self.root / "pipeline.json"

    def arm_config(self, arm: str) -> Path:
        return self.runs / f"{arm}.json"

    def arm_dir(self, arm: str) -> Path:
        return self.runs / arm

    def arm_checkpoint(self, arm: str) -> Path:
        return self.arm_dir(arm) / "checkpoints" / "latest.pt"

    def backdrop_dir(self, tile_name: str) -> Path:
        return self.caches / "backdrops" / tile_name

    def ownership_dir(self, tile_name: str) -> Path:
        return self.caches / "ownership" / tile_name

    def delivery_dir(self, tag: str) -> Path:
        return self.runs / f"delivery_{tag}"


# --------------------------------------------------------------------------
# build_plan
# --------------------------------------------------------------------------


def _tool(repo_root: Path, name: str) -> str:
    return str(repo_root / "tools" / name)


def _arm_name(profile: Profile, tile_id: int, generation: str) -> str:
    return str(profile.tile_rules["arm_name_pattern"]).format(
        tile=tile_id, profile=profile.name, generation=generation
    )


def _run_id(profile: Profile, scene: str, tile_id: int, generation: str) -> str:
    return str(profile.tile_rules["run_id_pattern"]).format(
        scene=scene, tile=tile_id, profile=profile.name, generation=generation
    )


def _coarse_arm(profile: Profile) -> str:
    return str(profile.coarse_prior["arm_name"]).format(profile=profile.name)


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive override, so a nested block replaces only the keys it names."""
    out = thaw(base)
    for key, value in thaw(override).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def build_plan(
    profile: Profile,
    dataset: DatasetSummary,
    *,
    dataset_root: Path,
    work_root: Path,
    repo_root: Path,
    python: Path,
    bundle_paths: Mapping[str, str] | None = None,
    prior_tile_checkpoints: Mapping[int, str] | None = None,
    vram_gib: float | None = None,
    delivery_tag: str | None = None,
    stages: Sequence[str] = STAGES,
) -> Plan:
    """The concrete step list for one (profile, dataset) pair.

    ``bundle_paths`` is :meth:`DatasetBundle.trainer_paths` when prepare has
    run; on a dry run before ingestion it may be ``None`` and the arm configs
    carry ``<prepare:key>`` placeholders instead, which is enough to read the
    plan and estimate it but not to execute it.

    ``prior_tile_checkpoints`` decides the generation count. A stand-in
    backdrop for tile N renders the other tiles' checkpoints; without them the
    plan grows a seed generation and says so.
    """
    layout = WorkLayout(Path(work_root))
    tag = delivery_tag or profile.name
    stages = tuple(stages)
    steps: list[PlannedStep] = []
    warnings: list[str] = []
    cost = profile.cost_model
    prior = dict(prior_tile_checkpoints or {})
    paths = dict(bundle_paths or {})
    caps = {tile.tile_id: tile_cap(profile, tile, vram_gib=vram_gib) for tile in dataset.tiles}

    ratio = float(profile.tile_rules["cap_ratio_of_initialisation"])
    for tile in dataset.tiles:
        cap = caps[tile.tile_id]
        # Compare against the same rule without the card ceiling, so rounding
        # alone never reads as a clamp.
        unclamped = tile_cap(profile, tile, vram_gib=None)
        if cap < unclamped:
            warnings.append(
                f"{tile.name}: cap clamped {unclamped/1e6:.2f}M -> {cap/1e6:.2f}M by the "
                f"{vram_gib:.0f} GiB VRAM ceiling"
            )
        elif tile.previous_final_population is not None and unclamped > tile.init_point_count * ratio:
            warnings.append(
                f"{tile.name}: the {ratio}x rule gives {tile.init_point_count * ratio/1e6:.2f}M, below its "
                f"previous final population {tile.previous_final_population/1e6:.2f}M; the floor rule "
                f"raised the cap to {cap/1e6:.2f}M"
            )

    if dataset.estimated:
        warnings.append(
            "the dataset summary is ESTIMATED (derived from the capture, not from a prepare "
            "manifest): tile boxes, per-tile view counts and initialisation counts are "
            "derivations, so every cost below is too. Plan only - a run refuses it"
        )

    missing_prior = [tile.tile_id for tile in dataset.tiles if tile.tile_id not in prior]
    generations = ("seed", "delivery") if missing_prior else ("delivery",)
    if missing_prior:
        warnings.append(
            "no previous-generation tile checkpoints for tiles "
            + ", ".join(str(t) for t in missing_prior)
            + ": the plan adds a seed generation (profile open question 'backdrop-bootstrap'); "
              "seed arms carry INFERRED settings"
        )
    if not dataset.has_reference_model:
        warnings.append(
            "no reference (competitor) model declared: tools/pipeline.py deliver runs "
            "build_three_way_compare / build_offtrajectory_compare against reference_ply and "
            "delivery_baselines, which a first delivery of a new scene does not have"
        )

    def path_of(key: str) -> str:
        return paths.get(key, f"<prepare:{key}>")

    # ---------------- prepare ----------------
    # The cache list below is the house0305 graph written out longhand.
    # cloudstudio3dgs_sdk.ingest.plan_caches derives the same graph from the
    # capture bundle with real dependency bindings; see the profile's
    # "prepare-step-source" open question.
    if "prepare" in stages:
        steps.append(
            PlannedStep(
                name="ingest_dataset",
                stage="prepare",
                resource="external",
                estimate=Estimate(0.0, 0, "owned by the ingestion task", UNMEASURED),
                outputs=(str(layout.prepare_manifest),),
                note=(
                    "adapters, the signed cache graph and the automatic tiling rule live in "
                    "cloudstudio3dgs_sdk.ingest; an existing verified prepare_manifest.json "
                    "is adopted instead"
                ),
            )
        )
        steps.append(
            PlannedStep(
                name="write_arm_configs",
                stage="prepare",
                resource="cpu",
                estimate=Estimate(1.0, 0, "file writes", MEASURED),
                outputs=tuple(
                    str(layout.arm_config(_arm_name(profile, tile.tile_id, generation)))
                    for generation in generations
                    for tile in dataset.tiles
                )
                + (str(layout.arm_config(_coarse_arm(profile))), str(layout.pipeline_config)),
                note="arm configs are profile + dataset paths + per-tile derivations; nothing else",
            )
        )
        sky_faces = dataset.train_view_count
        steps.append(
            PlannedStep(
                name="sky_masks",
                stage="prepare",
                resource="cpu",
                estimate=Estimate(
                    sky_faces * float(cost["sky_mask_seconds_per_face"]),
                    int(sky_faces * int(cost["sky_mask_bytes_per_face"])),
                    f"{sky_faces} faces x {cost['sky_mask_seconds_per_face']}s (never timed)",
                    UNMEASURED,
                ),
                command=(
                    str(python),
                    _tool(repo_root, "build_sky_masks.py"),
                    "--face-manifest", path_of("face_cache_manifest"),
                    "--face-cache-root", path_of("face_cache_root"),
                    "--output-root", str(layout.caches / "sky_masks"),
                    "--device", "cpu",
                ),
                outputs=(str(layout.caches / "sky_masks" / "sky_mask_train.json"),),
                note=(
                    "SegFormer b4 ADE20k, NVIDIA non-commercial licence: supervision masks only, "
                    "nothing derived from it ships"
                ),
            )
        )
        steps.append(
            PlannedStep(
                name="sky_dome",
                stage="prepare",
                resource="cpu",
                estimate=Estimate(
                    float(cost["sky_dome_seconds"]),
                    int(cost["sky_dome_bytes"]),
                    "never timed",
                    UNMEASURED,
                ),
                command=(
                    str(python),
                    _tool(repo_root, "build_sky_dome.py"),
                    "--dataset-manifest", path_of("dataset_manifest"),
                    "--recording-root", path_of("recording_root"),
                    "--depth-manifest", path_of("depth_manifest"),
                    "--depth-root", path_of("depth_root"),
                    "--person-mask-manifest", path_of("person_mask_manifest"),
                    "--output", str(layout.caches / "sky_dome.pt"),
                    "--count", str(profile.backdrop["sky_dome"]["count"]),
                    "--radius-m", str(profile.backdrop["sky_dome"]["radius_m"]),
                    "--seed", str(profile.backdrop["sky_dome"]["seed"]),
                ),
                outputs=(str(layout.caches / "sky_dome.pt"),),
            )
        )
        for tile in dataset.tiles:
            steps.append(
                PlannedStep(
                    name=f"ownership_{tile.name}",
                    stage="prepare",
                    resource="cpu",
                    estimate=Estimate(
                        tile.view_count * float(cost["ownership_seconds_per_view"]),
                        int(tile.view_count * int(cost["ownership_bytes_per_view"])),
                        f"{tile.view_count} views x {cost['ownership_seconds_per_view']}s/view (19 min / 6133 views)",
                        MEASURED,
                    ),
                    command=(
                        str(python),
                        _tool(repo_root, "build_tile_ownership_masks.py"),
                        "--config", str(layout.arm_config(_arm_name(profile, tile.tile_id, "delivery"))),
                        "--output-root", str(layout.ownership_dir(tile.name)),
                        "--tile-id", str(tile.tile_id),
                        "--margin-m", str(profile.trainer_base["tile_ownership_margin_m"]),
                        "--dilation-px", str(profile.trainer_base["tile_ownership_dilation_px"]),
                    ),
                    outputs=(str(layout.ownership_dir(tile.name) / "tile_ownership_manifest.json"),),
                )
            )

    # ---------------- train ----------------
    coarse_arm = _coarse_arm(profile)
    if "train" in stages:
        # The coarse prior needs a whole-scene background library of its own.
        steps.append(
            PlannedStep(
                name="global_view_backgrounds",
                stage="train",
                resource="gpu",
                estimate=Estimate(
                    dataset.train_view_count * float(cost["global_backgrounds_seconds_per_view"]),
                    int(dataset.train_view_count * int(cost["global_backgrounds_bytes_per_view"]))
                    // int(profile.coarse_prior.get("background_downsample", 4)) ** 2,
                    f"{dataset.train_view_count} views x {cost['global_backgrounds_seconds_per_view']}s/view",
                    EXTRAPOLATED,
                ),
                command=(
                    str(python),
                    _tool(repo_root, "build_view_backgrounds.py"),
                    "--config", str(layout.arm_config(coarse_arm)),
                    "--dome", str(layout.caches / "sky_dome.pt"),
                    "--split", "train",
                    "--output", str(layout.caches / "view_backgrounds"),
                    "--downsample", str(profile.coarse_prior.get("background_downsample", 4)),
                ),
                outputs=(str(layout.caches / "view_backgrounds" / "view_background_manifest_train.json"),),
            )
        )
        coarse_steps = int(profile.coarse_prior["overrides"]["controlled_stop_after_steps"])
        coarse_cap = int(profile.coarse_prior["overrides"]["cap_max"])
        steps.append(
            PlannedStep(
                name=f"train_{coarse_arm}",
                stage="train",
                resource="gpu",
                estimate=Estimate(
                    coarse_steps * tile_seconds_per_step(profile, coarse_cap),
                    int(cost["coarse_run_bytes"]),
                    f"{coarse_steps} steps at cap {coarse_cap/1e6:.1f}M through the seconds/step model",
                    EXTRAPOLATED,
                ),
                command=(
                    str(python),
                    _tool(repo_root, "pipeline.py"),
                    "--pipeline-config", str(layout.pipeline_config),
                    "queue", coarse_arm,
                ),
                outputs=(str(layout.arm_checkpoint(coarse_arm)),),
                note="tile-free whole-scene prior; feeds both the backdrop and the merge fill layer",
            )
        )

        for generation in generations:
            for tile in dataset.tiles:
                arm = _arm_name(profile, tile.tile_id, generation)
                cap = caps[tile.tile_id]
                if generation == "delivery":
                    sources = [str(layout.caches / "sky_dome.pt")]
                    others = [
                        prior.get(other.tile_id)
                        or str(layout.arm_checkpoint(_arm_name(profile, other.tile_id, "seed")))
                        for other in dataset.tiles
                        if other.tile_id != tile.tile_id
                    ]
                    backdrop_cmd: list[str] = [
                        str(python),
                        _tool(repo_root, "build_standin_backgrounds.py"),
                        "--config", str(layout.arm_config(arm)),
                        "--dome", sources[0],
                    ]
                    for checkpoint in [*others, str(layout.arm_checkpoint(coarse_arm))]:
                        backdrop_cmd += ["--standin-checkpoint", checkpoint]
                    backdrop_cmd += [
                        "--output", str(layout.backdrop_dir(tile.name)),
                        "--exclude-box-kind", str(profile.backdrop["exclude_box_kind"]),
                        "--exclude-margin-m", str(profile.backdrop["exclude_margin_m"]),
                        "--min-opacity", str(profile.backdrop["min_opacity"]),
                        "--target-gain", str(profile.backdrop["target_gain"]),
                        "--downsample", str(profile.backdrop["downsample"]),
                    ]
                    steps.append(
                        PlannedStep(
                            name=f"backdrop_{tile.name}",
                            stage="train",
                            resource="gpu",
                            estimate=Estimate(
                                tile.view_count * float(cost["backdrop_seconds_per_view"]),
                                int(tile.view_count * int(cost["backdrop_bytes_per_view"])),
                                f"{tile.view_count} views x {cost['backdrop_seconds_per_view']}s/view "
                                "(6 min / 6133 views, 1.16-1.37 MB/view)",
                                MEASURED,
                            ),
                            command=tuple(backdrop_cmd),
                            outputs=(str(layout.backdrop_dir(tile.name) / "background_manifest.json"),),
                            note="sky dome + the other tiles' checkpoints + the coarse prior, own box excluded",
                        )
                    )
                config = tile_config(
                    profile,
                    dataset,
                    tile,
                    generation=generation,
                    layout=layout,
                    bundle_paths=paths,
                    cap_max=cap,
                )
                stop = int(config["controlled_stop_after_steps"])
                per_step = tile_seconds_per_step(profile, cap)
                steps.append(
                    PlannedStep(
                        name=f"train_{arm}",
                        stage="train",
                        resource="gpu",
                        estimate=Estimate(
                            stop * per_step,
                            int(cost["tile_run_bytes"]),
                            f"{stop} steps x {per_step:.3f}s at cap {cap/1e6:.2f}M "
                            "(fit over 4 measured 20k runs)",
                            EXTRAPOLATED,
                        ),
                        command=(
                            str(python),
                            _tool(repo_root, "pipeline.py"),
                            "--pipeline-config", str(layout.pipeline_config),
                            "queue", arm,
                        ),
                        outputs=(str(layout.arm_checkpoint(arm)),),
                        config=config,
                        config_path=str(layout.arm_config(arm)),
                        note=(
                            "seed generation: ownership and sky supervision off, no backdrop "
                            "(INFERRED, never A/B'd)"
                            if generation == "seed"
                            else ""
                        ),
                    )
                )

    # ---------------- deliver ----------------
    if "deliver" in stages:
        delivery_dir = layout.delivery_dir(tag)
        tile_arms = {tile.tile_id: _arm_name(profile, tile.tile_id, "delivery") for tile in dataset.tiles}
        merged_gaussians = int(
            sum(caps.values()) * float(cost["merge_retained_fraction"])
        )
        exported = int(merged_gaussians * float(cost["export_retained_fraction"]))
        merge_cmd: list[str] = [
            str(python),
            _tool(repo_root, "merge_v28_tile_checkpoints.py"),
            "--tile-inputs", path_of("tile_inputs_manifest"),
            "--tile-inputs-root", path_of("tile_inputs_root"),
        ]
        for tile_id, arm in sorted(tile_arms.items()):
            merge_cmd += ["--tile-checkpoint", f"{tile_id}={layout.arm_checkpoint(arm)}"]
        merge_cmd += [
            "--output-checkpoint", str(delivery_dir / "merged.pt"),
            "--output-report", str(delivery_dir / "merge_report.json"),
            "--merge-policy", str(profile.merge["policy"]),
            "--tolerance-m", str(profile.merge["tolerance_m"]),
        ]
        fill = profile.merge["fill"]
        fill_blocking = ""
        if fill["enabled"]:
            merge_cmd += [
                "--fill-checkpoint", str(layout.arm_checkpoint(coarse_arm)),
                "--fill-occupancy-voxel-m", str(fill["occupancy_voxel_m"]),
                "--fill-occupancy-clearance-voxels", str(fill["occupancy_clearance_voxels"]),
                "--fill-min-opacity", str(fill["min_opacity"]),
            ]
            fill_blocking = _fill_support_gap(repo_root)
        if profile.merge["harmonize_exposure"]:
            merge_cmd.append("--harmonize-exposure")
        steps.append(
            PlannedStep(
                name="merge_tiles",
                stage="deliver",
                resource="cpu",
                estimate=Estimate(
                    float(cost["merge_seconds"]),
                    merged_gaussians * int(cost["merged_bytes_per_gaussian"]),
                    f"4 min CPU; {merged_gaussians/1e6:.1f}M merged x {cost['merged_bytes_per_gaussian']} B",
                    EXTRAPOLATED,
                ),
                command=tuple(merge_cmd),
                outputs=(str(delivery_dir / "merged.pt"), str(delivery_dir / "merge_report.json")),
                note=(
                    "fill layer: coarse-prior rows kept only where no merged tile gaussian occupies "
                    f"their {fill['occupancy_voxel_m']} m voxel"
                    if fill["enabled"]
                    else "no fill layer"
                ),
                blocking=fill_blocking,
            )
        )
        body_ply = delivery_dir / f"{dataset.scene_tag}_{tag}_merged.ply"
        steps.append(
            PlannedStep(
                name="export_ply",
                stage="deliver",
                resource="cpu",
                estimate=Estimate(
                    float(cost["export_seconds"]),
                    exported * int(cost["ply_bytes_per_gaussian"]),
                    f"{exported/1e6:.1f}M exported x {cost['ply_bytes_per_gaussian']} B",
                    EXTRAPOLATED,
                ),
                command=(
                    str(python),
                    _tool(repo_root, "export_gaussian_ply.py"),
                    "--checkpoint", str(delivery_dir / "merged.pt"),
                    "--output", str(body_ply),
                    "--min-opacity", str(profile.export["min_opacity"]),
                ),
                outputs=(str(body_ply),),
            )
        )
        thresholds = tuple(profile.export["threshold_control"])
        steps.append(
            PlannedStep(
                name="threshold_control",
                stage="deliver",
                resource="cpu",
                estimate=Estimate(
                    float(cost["export_seconds"]) * len(thresholds),
                    exported * int(cost["ply_bytes_per_gaussian"]) * len(thresholds),
                    f"{len(thresholds)} control exports at {', '.join(str(v) for v in thresholds)}",
                    EXTRAPOLATED,
                ),
                outputs=(str(delivery_dir / "threshold_control" / "threshold_control.json"),),
                note="records how many gaussians the delivery threshold removed",
            )
        )
        steps.append(
            PlannedStep(
                name="reimport_ply",
                stage="deliver",
                resource="cpu",
                estimate=Estimate(
                    float(cost["reimport_seconds"]),
                    exported * int(cost["merged_bytes_per_gaussian"]),
                    "the customer opens the PLY, so the PLY is what gets scored",
                    EXTRAPOLATED,
                ),
                command=(
                    str(python),
                    _tool(repo_root, "import_gaussian_ply.py"),
                    "--ply", str(body_ply),
                    "--output", str(delivery_dir / "reimported.pt"),
                ),
                outputs=(str(delivery_dir / "reimported.pt"),),
            )
        )
        steps.append(
            PlannedStep(
                name="battery",
                stage="deliver",
                resource="gpu",
                estimate=Estimate(
                    float(cost["battery_seconds"]),
                    0,
                    f"{profile.battery['views']} probe views, PSNR + alpha coverage",
                    MEASURED,
                ),
                command=(
                    str(python),
                    _tool(repo_root, "evaluate_probe_views.py"),
                    "--config", str(layout.root / "delivery_eval.json"),
                    "--checkpoint", str(delivery_dir / "reimported.pt"),
                    "--views", str(profile.battery["views"]),
                    "--output", str(delivery_dir / "battery_final.json"),
                ),
                outputs=(str(delivery_dir / "battery_final.json"),),
                note="alpha coverage is reported beside PSNR; a coverage gap reads as blur otherwise",
            )
        )
        steps.append(
            PlannedStep(
                name="morphology",
                stage="deliver",
                resource="gpu",
                estimate=Estimate(float(cost["battery_seconds"]), 0, "checkpoint_morphology", EXTRAPOLATED),
                command=(
                    str(python),
                    _tool(repo_root, "checkpoint_morphology.py"),
                    str(delivery_dir / "reimported.pt"),
                    "--label", f"final_{tag}",
                ),
                outputs=(str(delivery_dir / "morph_final.txt"),),
            )
        )
        if dataset.has_reference_model:
            for name, tool_name, seconds in (
                ("compare_matched", "build_three_way_compare.py", float(cost["compare_seconds"])),
                ("offtrajectory", "build_offtrajectory_compare.py", float(cost["offtraj_seconds"])),
            ):
                steps.append(
                    PlannedStep(
                        name=name,
                        stage="deliver",
                        resource="gpu",
                        estimate=Estimate(seconds, 0, "never timed separately", UNMEASURED),
                        command=(str(python), _tool(repo_root, tool_name)),
                        outputs=(str(delivery_dir / f"{name}_summary.json"),),
                        note="needs reference_ply + reference_alignment",
                    )
                )
        steps.append(
            PlannedStep(
                name="freeze_identity",
                stage="deliver",
                resource="cpu",
                estimate=Estimate(float(cost["identity_seconds"]), 0, "sha of the delivered PLY", MEASURED),
                command=(
                    str(python),
                    _tool(repo_root, "freeze_run_identity.py"),
                    "--checkpoint", str(delivery_dir / "merged.pt"),
                    "--extra-file", str(body_ply),
                    "--output", str(layout.report / f"delivery_{tag}_identity.json"),
                ),
                outputs=(str(layout.report / f"delivery_{tag}_identity.json"),),
                note="final scores are bound to the PLY sha256; a changed PLY invalidates them",
            )
        )

    # ---------------- report ----------------
    if "report" in stages:
        steps.append(
            PlannedStep(
                name="acceptance_report",
                stage="report",
                resource="cpu",
                estimate=Estimate(float(cost["report_seconds"]), 2 * 1024 * 1024, "reads recorded JSON", MEASURED),
                outputs=(
                    str(layout.report / f"{tag}_report.json"),
                    str(layout.report / f"{tag}_report.md"),
                ),
                note="profile gates vs measured battery/morphology/merge numbers, plus every UNMEASURED knob",
            )
        )

    return Plan(
        profile_name=profile.name,
        profile_version=profile.version,
        profile_sha256=profile.profile_sha256,
        scene_tag=dataset.scene_tag,
        dataset_root=str(dataset_root),
        work_root=str(work_root),
        generations=generations,
        steps=tuple(steps),
        dataset=dataset,
        tile_caps=caps,
        delivery_tag=tag,
        warnings=tuple(warnings),
    )


def _fill_support_gap(repo_root: Path) -> str:
    """Empty when this checkout can actually run the fill merge."""
    merge_tool = Path(repo_root) / "tools" / "merge_v28_tile_checkpoints.py"
    pipeline_tool = Path(repo_root) / "tools" / "pipeline.py"
    missing: list[str] = []
    try:
        if "--fill-checkpoint" not in merge_tool.read_text(encoding="utf-8"):
            missing.append(f"{merge_tool.name} has no --fill-checkpoint")
    except OSError:
        missing.append(f"{merge_tool} is not readable")
    try:
        if "fill_checkpoint" not in pipeline_tool.read_text(encoding="utf-8"):
            missing.append("tools/pipeline.py deliver does not forward the fill arguments")
    except OSError:
        missing.append(f"{pipeline_tool} is not readable")
    if not missing:
        return ""
    return (
        "the profile asks for a fill layer this checkout cannot produce: "
        + "; ".join(missing)
        + " (the flag exists on the research branch only)"
    )


def _trainer_path_block(profile: Profile, bundle_paths: Mapping[str, str]) -> dict[str, str]:
    """Every dataset path an arm config carries, unresolved ones made visible.

    A config that is merely missing ``split_manifest`` fails deep inside the
    trainer; one carrying ``<prepare:split_manifest>`` fails while the plan is
    still on screen.
    """
    block = {key: f"<prepare:{key}>" for key in profile.dataset_contract["trainer_path_keys"]}
    block.update({key: value for key, value in bundle_paths.items() if key in block})
    return block


def tile_config(
    profile: Profile,
    dataset: DatasetSummary,
    tile: TileSummary,
    *,
    generation: str,
    layout: WorkLayout,
    bundle_paths: Mapping[str, str],
    cap_max: int,
) -> dict[str, Any]:
    """The trainer config for one tile arm: profile + paths + derivations."""
    arm = _arm_name(profile, tile.tile_id, generation)
    max_steps = tile_max_steps(profile, tile.view_count)
    config = thaw(profile.trainer_base)
    if generation == "seed":
        config = _merge_dicts(config, profile.tile_rules["seed_generation_overrides"])
    config.update(_trainer_path_block(profile, bundle_paths))
    config.update(
        {
            "run_id": _run_id(profile, dataset.scene_tag, tile.tile_id, generation),
            "output_dir": str(layout.arm_dir(arm)),
            "device": "cuda:0",
            "mipmap_tile_id": tile.tile_id,
            "max_steps": max_steps,
            "cap_max": cap_max,
            "tile_inputs_manifest": bundle_paths.get("tile_inputs_manifest", "<prepare:tile_inputs_manifest>"),
            "tile_inputs_root": bundle_paths.get("tile_inputs_root", "<prepare:tile_inputs_root>"),
            "initialization_ply": bundle_paths.get(
                f"tile{tile.tile_id}_initialization_ply", f"<prepare:{tile.name}/initialization_full_lidar.ply>"
            ),
            "initialization_geometry": bundle_paths.get(
                f"tile{tile.tile_id}_initialization_geometry", f"<prepare:{tile.name}/initialization_geometry.npz>"
            ),
            "initialization_geometry_manifest": bundle_paths.get(
                "tile_geometry_manifest", "<prepare:tile_geometry_manifest>"
            ),
        }
    )
    config["default_strategy"] = dict(config["default_strategy"])
    config["default_strategy"]["prune_switch_step"] = prune_switch_step(profile, max_steps)
    if config.get("tile_ownership_masking"):
        config["tile_ownership_cache_manifest"] = str(
            layout.ownership_dir(tile.name) / "tile_ownership_manifest.json"
        )
        config["tile_ownership_cache_root"] = str(layout.ownership_dir(tile.name))
    if config.get("sky_supervision", {}).get("enabled"):
        config["sky_supervision"] = dict(config["sky_supervision"])
        config["sky_supervision"]["mask_manifest"] = str(layout.caches / "sky_masks" / "sky_mask_train.json")
        config["sky_supervision"]["mask_root"] = str(layout.caches / "sky_masks")
    if generation == "delivery" and profile.backdrop["enabled"]:
        config["background_image_manifest"] = str(layout.backdrop_dir(tile.name) / "background_manifest.json")
        config["background_image_root"] = str(layout.backdrop_dir(tile.name))
    else:
        config["background_image_manifest"] = str(
            layout.caches / "view_backgrounds" / "view_background_manifest_train.json"
        )
        config["background_image_root"] = str(layout.caches / "view_backgrounds")
    config["gsplat_lock"] = bundle_paths.get("gsplat_lock", "<prepare:gsplat_lock>")
    config["lineage"] = {
        "profile": profile.name,
        "profile_version": profile.version,
        "profile_sha256": profile.profile_sha256,
        "generation": generation,
        "cap_rule": (
            f"{profile.tile_rules['cap_ratio_of_initialisation']}x of {tile.init_point_count} "
            f"initialisation points -> {cap_max}"
        ),
    }
    return config


def coarse_config(
    profile: Profile,
    dataset: DatasetSummary,
    *,
    layout: WorkLayout,
    bundle_paths: Mapping[str, str],
) -> dict[str, Any]:
    """The trainer config for the coarse whole-scene prior."""
    arm = _coarse_arm(profile)
    max_steps = tile_max_steps(profile, dataset.train_view_count)
    config = _merge_dicts(profile.trainer_base, profile.coarse_prior["overrides"])
    for key in profile.coarse_prior["drop_keys"]:
        config.pop(key, None)
    config.update(_trainer_path_block(profile, bundle_paths))
    config.update(
        {
            "run_id": str(profile.coarse_prior["run_id_pattern"]).format(
                scene=dataset.scene_tag, profile=profile.name
            ),
            "output_dir": str(layout.arm_dir(arm)),
            "device": "cuda:0",
            "max_steps": max_steps,
            "initialization_ply": bundle_paths.get("global_init_ply", "<prepare:global_init_ply>"),
            "initialization_geometry": bundle_paths.get("global_init_geometry", "<prepare:global_init_geometry>"),
            "background_image_manifest": str(
                layout.caches / "view_backgrounds" / "view_background_manifest_train.json"
            ),
            "background_image_root": str(layout.caches / "view_backgrounds"),
            "gsplat_lock": bundle_paths.get("gsplat_lock", "<prepare:gsplat_lock>"),
        }
    )
    config["default_strategy"] = dict(config["default_strategy"])
    config["default_strategy"]["prune_switch_step"] = prune_switch_step(profile, max_steps)
    config["lineage"] = {
        "profile": profile.name,
        "profile_version": profile.version,
        "profile_sha256": profile.profile_sha256,
        "generation": "coarse_prior",
        "note": "tile-free whole-scene prior; backdrop source and merge fill source",
    }
    return config
