"""Frozen, versioned delivery recipes.

A :class:`Profile` is *data*: every knob the recipe needs, plus the measured
row that justifies it. The engine (:mod:`cloudstudio3dgs_sdk.plan`,
:mod:`cloudstudio3dgs_sdk.project`) reads profiles and never branches on the
profile name, so a second recipe is a second object in :data:`PROFILES` and
no new code path.

Three rules keep that true:

* Nothing in a profile is a path. Paths belong to a dataset and are resolved
  by :mod:`cloudstudio3dgs_sdk.plan` from what ``prepare()`` recorded.
* Every value carries a :class:`Provenance` naming the measurement that fixed
  it and how much that measurement is worth (``measured`` down to
  ``unmeasured``). A knob nobody measured is still allowed - it just has to
  say so, so the report can print it.
* ``profile_sha256`` covers the whole object. A run records it; a resume
  refuses when it changed. Editing a knob is therefore a new recipe, not a
  quiet mutation of an old one.

``PROFILE_B5FILL2`` is the house0305 delivery candidate of 2026-09-14:
per tile LiDAR-initialised training to 20k steps with a cap near 1.75x the
initialisation, tile-ownership masking off a precomputed cache, SegFormer sky
supervision, and a per-view stand-in backdrop; then a four-tile merge with a
coarse-prior fill layer, export, re-import and battery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

# Confidence ladder for a knob's justification, worst last.
MEASURED = "measured"          # an A/B on this dataset moved a reported metric
EXTRAPOLATED = "extrapolated"  # fitted from measurements at other settings
INHERITED = "inherited"        # measured in an earlier campaign, carried over
INFERRED = "inferred"          # follows from a measured fact but not itself tested
UNMEASURED = "unmeasured"      # a default nobody has challenged
CONFIDENCE_LEVELS = (MEASURED, EXTRAPOLATED, INHERITED, INFERRED, UNMEASURED)


@dataclass(frozen=True)
class Provenance:
    """Why one knob (or one group of knobs) holds the value it holds."""

    claim: str
    source: str
    confidence: str = MEASURED

    def __post_init__(self) -> None:
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"unknown confidence {self.confidence!r}")

    def as_dict(self) -> dict[str, str]:
        return {"claim": self.claim, "source": self.source, "confidence": self.confidence}


def freeze(value: Any) -> Any:
    """Deep-freeze plain JSON data: dict -> MappingProxyType, list -> tuple.

    Shallow ``frozen=True`` only stops rebinding the attribute; without this
    a caller could still edit ``profile.trainer_base["cap_max"]`` in place and
    invalidate the sha every run recorded.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"profiles hold JSON data only, got {type(value).__name__}")


def thaw(value: Any) -> Any:
    """Inverse of :func:`freeze`; the mutable copy callers may edit."""
    if isinstance(value, Mapping):
        return {str(key): thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class Profile:
    """One frozen recipe. Construct through :func:`make_profile`."""

    name: str
    version: str
    summary: str
    # Sections. Each is deep-frozen JSON data.
    runtime: Mapping[str, Any]
    dataset_contract: Mapping[str, Any]
    tiling: Mapping[str, Any]
    trainer_base: Mapping[str, Any]
    tile_rules: Mapping[str, Any]
    coarse_prior: Mapping[str, Any]
    backdrop: Mapping[str, Any]
    merge: Mapping[str, Any]
    export: Mapping[str, Any]
    battery: Mapping[str, Any]
    acceptance: Mapping[str, Any]
    cost_model: Mapping[str, Any]
    external_assets: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Provenance]
    open_questions: tuple[Mapping[str, Any], ...] = ()

    SECTIONS = (
        "runtime",
        "dataset_contract",
        "tiling",
        "trainer_base",
        "tile_rules",
        "coarse_prior",
        "backdrop",
        "merge",
        "export",
        "battery",
        "acceptance",
        "cost_model",
    )

    def section(self, name: str) -> Mapping[str, Any]:
        if name not in self.SECTIONS:
            raise KeyError(f"unknown profile section {name!r}")
        return getattr(self, name)

    def as_dict(self) -> dict[str, Any]:
        """The mutable, JSON-serialisable view. Excludes ``profile_sha256``."""
        payload: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "summary": self.summary,
        }
        for name in self.SECTIONS:
            payload[name] = thaw(getattr(self, name))
        payload["external_assets"] = thaw(self.external_assets)
        payload["provenance"] = {key: item.as_dict() for key, item in sorted(self.provenance.items())}
        payload["open_questions"] = thaw(self.open_questions)
        return payload

    @property
    def profile_sha256(self) -> str:
        """sha256 of the canonical JSON of everything above.

        Provenance is inside the digest on purpose: a knob whose justification
        was rewritten is a different recipe for review purposes even when the
        number did not move.
        """
        return hashlib.sha256(canonical_json(self.as_dict()).encode("utf-8")).hexdigest()

    def identity(self) -> dict[str, str]:
        return {"profile": self.name, "profile_version": self.version, "profile_sha256": self.profile_sha256}

    def why(self, key: str) -> Provenance:
        """Provenance for a dotted knob path, falling back to its prefixes."""
        parts = key.split(".")
        for stop in range(len(parts), 0, -1):
            candidate = ".".join(parts[:stop])
            if candidate in self.provenance:
                return self.provenance[candidate]
        raise KeyError(f"no provenance recorded for {key!r}")

    def unmeasured_knobs(self) -> tuple[str, ...]:
        return tuple(
            sorted(key for key, item in self.provenance.items() if item.confidence in (INFERRED, UNMEASURED))
        )


def make_profile(**fields: Any) -> Profile:
    """Build a :class:`Profile`, deep-freezing every data section."""
    provenance = fields.pop("provenance")
    if not all(isinstance(item, Provenance) for item in provenance.values()):
        raise TypeError("provenance values must be Provenance instances")
    frozen: dict[str, Any] = {
        "name": fields.pop("name"),
        "version": fields.pop("version"),
        "summary": fields.pop("summary"),
        "provenance": MappingProxyType(dict(provenance)),
    }
    for key in ("external_assets", "open_questions"):
        frozen[key] = freeze(fields.pop(key, ()))
    for key in Profile.SECTIONS:
        frozen[key] = freeze(fields.pop(key))
    if fields:
        raise TypeError(f"unexpected profile fields: {sorted(fields)}")
    return Profile(**frozen)


# --------------------------------------------------------------------------
# PROFILE_B5FILL2
# --------------------------------------------------------------------------

# Trainer knobs that are the same for every tile of every scene under this
# recipe. Everything a dataset decides (paths, tile id, cap, step count) is
# absent here and filled in by plan.py.
_B5_TRAINER_BASE: dict[str, Any] = {
    "trainer_preset": "custom",
    "seed": 42,
    "factor": 1,
    "require_person_masks": True,
    "checkpoint_every": 5000,
    "checkpoint_keep_every": 1000000,
    "controlled_stop_after_steps": 20000,
    "sh_degree": 1,
    "sh_degree_interval": 0,
    "color_model": "sh",
    "background_color": [1.0, 1.0, 1.0],
    "pinhole_rasterize_mode": "classic",
    "pinhole_with_ut": False,
    "view_sampling_mode": "fisher_yates_without_replacement_per_epoch",
    "surface_initialization": {
        "enabled": True,
        "mode": "mipmap_k7_k30",
        "planarity_gate": 0.6,
        "normal_scale_ratio": 0.5,
    },
    "metric_scale_calibration": {
        "mode": "precomputed",
        "knn_neighbors": 7,
        "knn_reduction": "arithmetic_mean",
        "scale_multiplier": 1.0,
    },
    "rgb_l1_weight": 0.6,
    "rgb_ssim_weight": 0.4,
    "rgb_ssim_mode": "local_gaussian",
    "lidar_range_weight": 0.0,
    "lidar_range_loss_mode": "robust_log_huber",
    "lidar_log_range_huber_delta": 0.05,
    "da2_depth_weight": 0.15,
    "da2_depth_space": "compressed",
    "mono_depth_max_range_m": 30.0,
    "lidar_alpha_weight": 0.1,
    "lidar_alpha_target": 0.95,
    "lidar_alpha_dilation_radius_px": 6,
    "surface_alpha_floor_profile": True,
    "lidar_normal_alignment": {
        "enabled": True,
        "weight_align": 0.05,
        "weight_flatten": 0.05,
        "weight_tangent_isotropy": 0.1,
        "planarity_gate": 0.6,
        "flatten_mode": "tangent_ratio",
        "flatten_ratio_target": 0.33,
        "tangent_isotropy_max_ratio": 60.0,
    },
    "geometry_regularization": {
        "enabled": True,
        "opacity_sparsity_weight": 0.0,
        "scale_upper_weight": 0.0,
        "anisotropy_weight": 0.01,
        "max_world_size_m": 0.2,
        "world_shrink_factor": 0.8,
        "max_anisotropy": 80.0,
    },
    "topology_policy": {"mode": "adaptive_growth"},
    "densification_strategy": "default_3dgs",
    "densification_gradient_source": "total_loss",
    "mcmc_noise_lr": 0.0,
    "mcmc_noise_injection_stop_iter": 0,
    "mcmc_refine_start_iter": 500,
    "mcmc_refine_stop_iter": 14000,
    "mcmc_refine_every": 100,
    "learning_rates": {
        "means": 1.6e-05,
        "scales": 0.005,
        "quats": 0.001,
        "opacities": 0.05,
        "colors": 0.0025,
    },
    "means_lr_final_factor": 0.01,
    "exposure_compensation": {"enabled": True, "learning_rate": 0.005},
    "golden_evaluation": {"enabled": False},
    "final_evaluation_artifacts": False,
    "default_strategy": {
        "exact_mipmap_lifecycle": True,
        "lifecycle_execution_order": "pre_optimizer_vendor",
        "detail_split_policy": "lidar_surface_screen_detail",
        "absgrad": False,
        "grow_grad2d": 5e-05,
        "split_scale_m": 0.2,
        "prune_scale_m": 0.2,
        "prune_opa": 0.1,
        "prune_opa_late": 0.05,
        "prune_scale2d": 0.15,
        "reset_every": 300,
        "reset_opacity_cap": 0.2,
        "refine_scale2d_stop_iter": 14000,
        "refine_start_iter": 500,
        "refine_stop_iter": 14000,
        "refine_every": 100,
        "capacity_conserving_clone_opacity": False,
        "relaxed_cull_when_no_growth": True,
        "revised_opacity": True,
        "growth_metric": "footprint_weighted",
        "relaxed_cull_at_capacity": False,
        "opacity_cull_policy": "immediate",
        "vendor_cull_warmup_profile": "exact_0p10_to_0p05",
        "vendor_opacity_reset_profile": "exact_every300",
        "reset_optimizer_state": "keep",
        "reset_before_cull": True,
        "detail_split_scale_m": 0.005,
        "detail_split_screen_radius": 0.0035,
    },
    "sky_supervision": {
        "enabled": True,
        "alpha_weight": 0.5,
        "alpha_target": 0.0,
        "exclude_photometric": True,
        "exclude_mono_depth": True,
        "growth_block": True,
        "mask_erosion_px": 4,
        "require_no_lidar_within_px": 6,
    },
    "tile_ownership_masking": True,
    "tile_ownership_margin_m": 0.5,
    "tile_ownership_dilation_px": 15,
}

# What the coarse whole-scene prior changes relative to a tile arm. It is a
# tile-free run over every view: no tile inputs, no ownership, no sky term,
# a 2 m-decimated LiDAR init and an early growth stop.
_B5_COARSE_OVERRIDES: dict[str, Any] = {
    "controlled_stop_after_steps": 10000,
    "cap_max": 3000000,
    "mcmc_refine_stop_iter": 8000,
    "metric_scale_calibration": {
        "mode": "knn",
        "knn_neighbors": 7,
        "knn_reduction": "arithmetic_mean",
        "scale_multiplier": 1.0,
    },
    "surface_initialization": {
        "enabled": True,
        "mode": "planar_surfel",
        "planarity_gate": 0.6,
        "normal_scale_ratio": 0.5,
    },
    "default_strategy": {"refine_stop_iter": 8000, "refine_scale2d_stop_iter": 8000},
}
# Keys a coarse arm must not carry at all (tile-only wiring and the sky term).
_B5_COARSE_DROP: tuple[str, ...] = (
    "tile_ownership_masking",
    "tile_ownership_margin_m",
    "tile_ownership_dilation_px",
    "sky_supervision",
)

# The seed generation. The delivery backdrop for tile N renders the *other*
# tiles' checkpoints, which do not exist for a dataset nobody has trained. The
# seed pass is the recipe minus the two things that need them.
_B5_SEED_OVERRIDES: dict[str, Any] = {
    "tile_ownership_masking": False,
    "sky_supervision": {"enabled": False},
}

PROFILE_B5FILL2 = make_profile(
    name="b5fill2",
    version="2026.09.14",
    summary=(
        "house0305 delivery candidate B5fill2: four LiDAR-initialised tiles at 20k steps, "
        "cap ~1.75x initialisation, ownership masking + sky supervision + per-view stand-in "
        "backdrop, merged with a voxel-occupancy fill layer from a coarse whole-scene prior."
    ),
    runtime={
        "python": "3.12",
        "torch": "2.11.0+cu128",
        "cuda": "12.8",
        "cuda_arch_list": "12.0",
        "gsplat_version": "1.5.3",
        "gsplat_lock_relpath": "upstream/gsplat.lock.json",
        "min_vram_gib": 16.0,
        # 12.48M live gaussians at 14.0 GiB is where this class of card died.
        "max_gaussians_per_gib_vram": 781250,
        "vram_safety_factor": 0.88,
        "disk_safety_factor": 1.25,
    },
    dataset_contract={
        # What prepare() must hand the rest of the SDK. See bundle.DatasetBundle.
        "required_bundle_fields": [
            "dataset_manifest",
            "split_manifest",
            "mask_manifest",
            "mask_root",
            "person_mask_manifest",
            "person_mask_root",
            "recording_root",
            "face_cache_manifest",
            "face_cache_root",
            "renderer_mask_manifest",
            "depth_manifest",
            "depth_root",
            "face_lidar_geometry_manifest",
            "face_lidar_geometry_root",
            "mono_depth_manifest",
            "mono_depth_root",
            "lidar_cloud",
            "tile_inputs_manifest",
            "tile_inputs_root",
            "tile_geometry_manifest",
            "global_init_ply",
            "global_init_geometry",
            "pipeline_gate",
        ],
        # The trainer config keys every arm carries verbatim; prepare() fills
        # them from DatasetBundle.trainer_paths(). Listing them here is what
        # lets a dry run produce a *complete* config with visible placeholders
        # instead of one that is silently missing a manifest.
        "trainer_path_keys": [
            "dataset_manifest",
            "split_manifest",
            "mask_manifest",
            "mask_root",
            "person_mask_manifest",
            "person_mask_root",
            "recording_root",
            "face_cache_manifest",
            "face_cache_root",
            "renderer_mask_manifest",
            "depth_manifest",
            "depth_root",
            "mono_depth_manifest",
            "mono_depth_root",
            "face_lidar_geometry_manifest",
            "face_lidar_geometry_root",
            "mipmap_pipeline_gate",
        ],
        "required_derived_caches": [
            "sky_masks",
            "sky_dome",
            "global_view_backgrounds",
            "tile_ownership",
        ],
        "face_model": "face4",
        "split_policy": "rig_frame_split_v1",
        "person_masks_required": True,
    },
    tiling={
        # Steps a tile declares = epochs x its own view count; the controlled
        # stop then lands well inside it.
        "epochs_for_max_steps": 20,
        "prune_switch_fraction_of_max_steps": 0.5,
        "tile_count_policy": "from_tile_inputs_manifest",
        "reference_tile_count": 4,
    },
    trainer_base=_B5_TRAINER_BASE,
    tile_rules={
        "cap_ratio_of_initialisation": 1.756,
        "cap_round_to": 100000,
        "cap_floor_policy": "not_below_previous_generation_final_population",
        "cap_floor_headroom": 1.07,
        "cap_ceiling_policy": "vram",
        "seed_generation_overrides": _B5_SEED_OVERRIDES,
        "arm_name_pattern": "tile{tile}_{profile}_{generation}",
        "run_id_pattern": "{scene}-t{tile}-{profile}-{generation}",
    },
    coarse_prior={
        "enabled": True,
        "arm_name": "global_coarse_{profile}",
        "run_id_pattern": "{scene}-coarse-{profile}",
        "init_decimation_m": 2.0,
        "background_downsample": 4,
        "overrides": _B5_COARSE_OVERRIDES,
        "drop_keys": list(_B5_COARSE_DROP),
    },
    backdrop={
        "enabled": True,
        "sources": ["sky_dome", "other_tile_checkpoints", "coarse_prior"],
        "exclude_box_kind": "training_and_export_box",
        "exclude_margin_m": 0.0,
        "min_opacity": 0.05,
        "harmonize_exposure": True,
        "target_gain": 1.0,
        "background_rgb": [1.0, 1.0, 1.0],
        "downsample": 1,
        "decode_cache_gib": 6.0,
        "sky_dome": {
            "count": 100000,
            "radius_m": 250.0,
            "elevation_min_deg": -15.0,
            "elevation_max_deg": 88.0,
            "frame_stride": 4,
            "theta_max_deg": 80.0,
            "min_samples": 3,
            "seed": 42,
        },
    },
    merge={
        "policy": "core_owner_only",
        "tolerance_m": 1e-05,
        "harmonize_exposure": True,
        "fill": {
            "enabled": True,
            "source": "coarse_prior",
            "occupancy_voxel_m": 0.2,
            "occupancy_clearance_voxels": 0,
            "min_opacity": 0.0,
            "rule": "reject_inside_any_tile_training_and_export_box_unless_voxel_unoccupied_by_merged_tile_gaussian",
        },
    },
    export={
        "min_opacity": 0.05,
        "threshold_control": [0.0, 0.01, 0.05],
        "reimport_before_scoring": True,
    },
    battery={
        "views": 48,
        "compare_frames": 6,
        "report_alpha_coverage": True,
        "render_spec": {"pinhole_rasterize_mode": "classic", "pinhole_with_ut": False, "apply_exposure_gain": False},
    },
    acceptance={
        # Gates the report checks. Reference values are B5fill2 on house0305
        # against the competitor/R1d baselines of 2026-09-14.
        "battery_alpha_mean_min": 0.90,
        "battery_psnr_p10_min": 16.50,
        "off_surface_fraction_gt_0p2m_max": 0.06,
        "morphology_short_axis_p50_mm_max": 0.60,
        "export_gaussian_count_max": 20000000,
        "reference": {
            "battery_psnr_mean": 19.11,
            "battery_psnr_p10": 16.68,
            "battery_alpha_mean": 0.947,
            "battery_alpha_p05": 0.733,
            "lidar_range_mae": 2.46,
            "off_surface_gt_0p2_0p5_1m": [0.041, 0.009, 0.002],
            "morphology_short_axis_p50_mm": 0.466,
            "morphology_axis_ratio": 12.09,
            "export_gaussian_count": 17180000,
            "offtrajectory_psnr_18f": 16.79,
            "offtrajectory_sharpness_ours_over_ref": 0.369,
        },
    },
    cost_model={
        # seconds/step as a function of cap_max, least squares over the four
        # measured B5 tile runs; see provenance "cost_model.tile_train".
        "tile_train_seconds_per_step": {
            "intercept": 0.1410,
            "per_million_cap": 0.03782,
            "observations": [[6.0, 0.363], [8.0, 0.456], [9.9, 0.504], [11.0, 0.561]],
        },
        "sky_mask_seconds_per_face": 0.60,
        "sky_mask_bytes_per_face": 18000,
        "sky_dome_seconds": 300.0,
        "sky_dome_bytes": 12000000,
        "global_backgrounds_seconds_per_view": 0.06,
        "global_backgrounds_bytes_per_view": 1400000,
        "ownership_seconds_per_view": 0.186,
        "ownership_bytes_per_view": 96000,
        "backdrop_seconds_per_view": 0.0587,
        "backdrop_bytes_per_view": 1400000,
        "tile_run_bytes": 3221225472,
        "coarse_run_bytes": 858993459,
        "merge_seconds": 240.0,
        "merged_bytes_per_gaussian": 92,
        "ply_bytes_per_gaussian": 104,
        "export_seconds": 30.0,
        "reimport_seconds": 60.0,
        "battery_seconds": 30.0,
        "compare_seconds": 120.0,
        "offtraj_seconds": 120.0,
        "identity_seconds": 30.0,
        "report_seconds": 10.0,
        "merge_retained_fraction": 0.908,
        "export_retained_fraction": 0.683,
    },
    external_assets=(
        {
            "id": "segformer_sky",
            "purpose": "sky masks for sky_supervision",
            "model_id": "nvidia/segformer-b4-finetuned-ade-512-512",
            "revision": "2641fd1e2893964d8d473d8cf65a906cb0bff071",
            "weights_file": "pytorch_model.bin",
            "license": "NVIDIA Source Code License-NC (non-commercial)",
            "license_note": (
                "Non-commercial. The weights are used ONLY to derive supervision masks "
                "during training; neither the weights nor any derived model file are "
                "shipped in a delivery. The delivered PLY contains no SegFormer output."
            ),
            "ships_in_delivery": False,
            "runtime_device": "cpu",
        },
        {
            "id": "gsplat_extension",
            "purpose": "rasteriser",
            "license": "Apache-2.0 (gsplat) + local fisheye patch",
            "ships_in_delivery": False,
        },
    ),
    open_questions=(
        {
            "id": "backdrop-bootstrap",
            "what": (
                "The delivery backdrop for tile N renders the other tiles' checkpoints. "
                "On house0305 those were the previous generation (R1d/R1). A dataset with no "
                "previous generation needs a seed pass first, which doubles tile GPU time."
            ),
            "expressed_as": "tile_rules.seed_generation_overrides + plan generation='seed'",
            "status": "the seed overrides themselves are INFERRED, never A/B'd",
        },
        {
            "id": "cap-floor",
            "what": (
                "Tile_2's cap was raised from the 1.756x rule (5.8M) to 8.0M because the rule "
                "landed below that tile's own previous final population (7.47M). A new dataset "
                "has no previous population, so the floor cannot fire."
            ),
            "expressed_as": "tile_rules.cap_floor_policy (needs plan input prior_final_population)",
            "status": "rule is data, its input is only available on a re-run",
        },
        {
            "id": "fill-vs-sharpness",
            "what": (
                "The fill layer buys coverage (battery alpha 0.712 -> 0.947, PSNR 18.05 -> 19.11) "
                "and sells off-trajectory sharpness (0.444 -> 0.369). Both moves are far outside "
                "the re-run band. B5fill2 picks coverage; a viewer-first delivery might not."
            ),
            "expressed_as": "merge.fill.enabled plus merge.fill.occupancy_* (a monotone density knob)",
            "status": "the operating point is a judgement, not a measured optimum",
        },
        {
            "id": "pipeline-fill-passthrough",
            "what": (
                "The fill layer used to exist only on the research branch: this checkout's "
                "merge_v28_tile_checkpoints.py had no --fill-checkpoint and tools/pipeline.py "
                "deliver did not forward one. Both are now here."
            ),
            "expressed_as": (
                "merge.fill.* -> pipeline config fill_checkpoint / fill_occupancy_voxel_m / "
                "fill_occupancy_clearance_voxels / fill_min_opacity"
            ),
            "status": (
                "resolved: research merged into eng, deliver forwards the fill flags, and a "
                "fill_checkpoint naming a missing file fails config parsing "
                "(tests/test_pipeline_fill_layer.py)"
            ),
        },
        {
            "id": "prepare-step-source",
            "what": (
                "plan.py hard-codes the prepare step list (sky masks, sky dome, ownership "
                "caches). cloudstudio3dgs_sdk.ingest.plan_caches already derives the same graph "
                "from the capture bundle, with dependency bindings and staleness rules this "
                "list does not have."
            ),
            "expressed_as": "dataset_contract.required_derived_caches names them; the order is code",
            "status": "integration point: prepare()'s steps should come from plan_caches once its API settles",
        },
        {
            "id": "no-reference-model",
            "what": (
                "tools/pipeline.py requires reference_ply, reference_alignment and "
                "delivery_baselines, and its deliver stage runs a three-way compare and an "
                "off-trajectory compare against them. A first delivery of a new scene has no "
                "competitor model, so those comparisons have no meaning and the config schema "
                "still demands the paths."
            ),
            "expressed_as": "plan omits the compare steps when has_reference_model is false",
            "status": "the pipeline config schema still requires the paths; make them optional",
        },
        {
            "id": "tile-plan",
            "what": (
                "The four house0305 tiles came from a hand-checked adaptive tile plan. The rule "
                "for cutting an arbitrary scene into tiles is owned by the data-ingestion task."
            ),
            "expressed_as": "tiling.tile_count_policy = from_tile_inputs_manifest",
            "status": "delegated to prepare()",
        },
    ),
    provenance={
        "runtime": Provenance(
            "The training identity of the campaign: gsplat 1.5.3 at commit f2d1413 with the "
            "S1 fisheye keep-distortion patch, torch 2.11.0+cu128, python 3.12.",
            "upstream/gsplat.lock.json; research/quality_recovery_v2/00_runtime_identity.json",
            INHERITED,
        ),
        "runtime.max_gaussians_per_gib_vram": Provenance(
            "tile3_R1d_20k died in _grow_mipmap at 12.48M gaussians / 14.0 GiB on a 16 GiB card; "
            "the Tile_0 B5 cap was hand-lowered 12.4M -> 11.0M for the same reason.",
            "research/quality_recovery_v2/README.zh-CN.md rows 'tile3 crash' and 'Tile_0 B5'",
            MEASURED,
        ),
        "dataset_contract": Provenance(
            "The manifests every B5 arm config binds, taken from the four as-run configs.",
            "3dgs-runs/house0305_sop/tile{0,1,2,3}_B5_cap6_20k.json",
            MEASURED,
        ),
        "tiling": Provenance(
            "How a tile's step schedule is derived from its own view count, so a scene with more "
            "or fewer views per tile keeps the same number of epochs.",
            "the five as-run house0305 configs vs tile_inputs_v9/tile_inputs_manifest.json",
            MEASURED,
        ),
        "tiling.epochs_for_max_steps": Provenance(
            "max_steps is exactly 20 x the tile's own view count in all four as-run configs "
            "(2132/1829/1684/2317 views -> 42640/36580/33680/46340) and in the coarse arm "
            "(3536 train faces -> 70720).",
            "tile_inputs_v9/tile_inputs_manifest.json vs the as-run configs",
            MEASURED,
        ),
        "tiling.prune_switch_fraction_of_max_steps": Provenance(
            "default_strategy.prune_switch_step is max_steps/2 in all five as-run configs.",
            "3dgs-runs/house0305_sop/*.json",
            MEASURED,
        ),
        "trainer_base": Provenance(
            "The B5 arm configuration minus per-tile and per-dataset wiring; B5 = B3 with "
            "cap 15M -> 6M and it passed all five delivery gates.",
            "3dgs-runs/house0305_sop/tile1_B5_cap6_20k.json; README row 'B5 结果 (04:15)'",
            MEASURED,
        ),
        "trainer_base.controlled_stop_after_steps": Provenance(
            "Training longer makes it worse: dead mass rises monotonically and PSNR peaks at "
            "5-10k. 20k is a stop, not a budget; max_steps only shapes the schedule.",
            "memory longer-training-makes-it-worse; stop-on-recovery-not-flush",
            MEASURED,
        ),
        "trainer_base.default_strategy.reset_every": Provenance(
            "reset cadence 300 -> 3000 was the root cause of the cull collapse; the vendor-exact "
            "profile keeps 300 together with reset_optimizer_state=keep and reset_before_cull.",
            "memory reset-cadence-was-the-root-cause; vendor-opacity-reset-keeps-adam-state",
            MEASURED,
        ),
        "trainer_base.default_strategy.detail_split_scale_m": Provenance(
            "split at 0.2 m is structurally dead against the 0.2 m shrink ceiling; the 5 mm "
            "detail split is what actually fires and it bought real hold-out gain.",
            "memory split-is-structurally-dead; t2-merged-delivery-20260903",
            MEASURED,
        ),
        "trainer_base.geometry_regularization.anisotropy_weight": Provenance(
            "The anisotropy guard is load-bearing: at 0 the population goes flat (p99 17114); "
            "the tangent-isotropy term must stay small because it is an unbounded quadratic.",
            "memory anisotropy-guard-is-load-bearing; tangent-isotropy-unbounded-quadratic",
            MEASURED,
        ),
        "trainer_base.lidar_range_weight": Provenance(
            "A 0.5 linear range term spent 70% of the loss on occlusion-penetrating points and "
            "never converged; the delivery arms run it at 0 and lean on da2 + alpha instead.",
            "memory arm-config-overrode-trainer-default-range-term",
            MEASURED,
        ),
        "trainer_base.sky_supervision": Provenance(
            "Sky supervision with erosion 4 px and a 6 px no-LiDAR guard, alpha weight 0.5. "
            "Note the design note proposed 8 px / 24 px; the delivery arms run 4 / 6.",
            "tile1_B5_cap6_20k.json; research/quality_recovery_v2/12_sky_supervision.md",
            MEASURED,
        ),
        "trainer_base.tile_ownership_masking": Provenance(
            "Ownership masking is what removes off-surface gaussians: >0.2 m fraction fell "
            "53-73% on every tile. Dilation 15 px; widening to 40 px (B6) bought nothing and "
            "narrowing does the opposite of what it looks like it does.",
            "README rows 'Tile_0/2/3 B5' and 'B6 结果 (09:56)'",
            MEASURED,
        ),
        "tile_rules": Provenance(
            "Per-tile derivations: the capacity rule and its two clamps, plus the arm naming that "
            "keeps a retried tile a separate arm rather than an edited one.",
            "lineage blocks of tile{0,1,2,3}_B5_cap6_20k.json",
            MEASURED,
        ),
        "tile_rules.cap_ratio_of_initialisation": Provenance(
            "Tile_1's chosen cap 6.0M is 1.756x its 3,417,320-point initialisation; the other "
            "three caps were derived with that same multiple. Capacity trades directly against "
            "sharpness: 4.5M -> door ROI 0.405, 6M -> 0.501, 7.75M -> 0.528.",
            "README rows 'B4 结果', 'B5 结果', '四切片推广准备'; lineage.cap_rule in the as-run configs",
            MEASURED,
        ),
        "tile_rules.cap_floor_policy": Provenance(
            "Tile_2's 1.756x value (5.8M) was below its own previous final population (7.47M), "
            "i.e. tighter than the B4 arm that lost the door ROI; it was raised to 8.0M, "
            "which the as-run config records as 1.07x that population.",
            "lineage.cap_adjustment in tile2_B5_cap6_20k.json",
            MEASURED,
        ),
        "tile_rules.seed_generation_overrides": Provenance(
            "No seed generation was ever run: house0305 had R1d checkpoints to render backdrops "
            "from. These overrides are what the recipe reduces to without them.",
            "derived from the backdrop source list; not A/B tested",
            INFERRED,
        ),
        "coarse_prior": Provenance(
            "B0: tile-free Face4 training over every view, 1.86M-point 2 m LiDAR init, cap 3M, "
            "growth stop 8k, controlled stop 10k. It feeds both the backdrop and the fill layer.",
            "3dgs-runs/house0305_sop/house0305_global_coarse_B0_10k.json",
            MEASURED,
        ),
        "backdrop": Provenance(
            "Per-view stand-in: sky dome + the other tiles' checkpoints + the coarse prior, "
            "each exposure-harmonised, rows inside this tile's own box removed, min opacity 0.05. "
            "It is what makes ownership masking safe - the excluded pixels still get a target.",
            "tile_backgrounds_B1/Tile_1/background_manifest.json; 14_standin_backdrop_design.md",
            MEASURED,
        ),
        "backdrop.decode_cache_gib": Provenance(
            "The decode cache was unbounded: Tile_1 held 13.5 GiB and Tile_3 would have held "
            "19.4 GiB on a 32 GB host. Bounded to a 6 GiB LRU in commit 1f1b6c9.",
            "eng commit 1f1b6c9",
            MEASURED,
        ),
        "merge": Provenance(
            "Four tile checkpoints become one delivery: hard ownership arbitration, one "
            "photometric frame, then a fill layer for the volume nobody painted.",
            "delivery_B5fill2/merge_report.json",
            MEASURED,
        ),
        "merge.policy": Provenance(
            "core_owner_only gives every point one hard owner; the four core boxes partition the "
            "scene bounding box exactly.",
            "tools/merge_v28_tile_checkpoints.py; delivery_B5fill2/merge_report.json",
            MEASURED,
        ),
        "merge.fill": Provenance(
            "The coverage gap is 'nominally owned, actually unpainted' volume, so a box rule is a "
            "no-op (100% of the prior falls inside some tile box). The voxel-occupancy rule at "
            "0.2 m with clearance 0 keeps 529k of 2.40M prior rows and lifts battery alpha p05 "
            "0.189 -> 0.733, PSNR 18.05 -> 19.11, at 1.4% extra gaussians.",
            "README rows '合并填充层 (08:20)', 'B5fill (09:57)', 'B5fill2 (10:06)'; "
            "delivery_B5fill2/merge_report.json",
            MEASURED,
        ),
        "export": Provenance(
            "What the customer receives and how it is graded: one delivery threshold, a control "
            "sweep beside it, and scoring done on the re-imported PLY rather than the checkpoint.",
            "tools/pipeline.py deliver_steps",
            INHERITED,
        ),
        "export.min_opacity": Provenance(
            "0.05 is the delivery threshold; 0.0/0.01/0.05 are exported as a control so the "
            "removed count is on the record.",
            "tools/pipeline.py EXPORT_THRESHOLD_CONTROL; pipeline.house0305.machine.json",
            INHERITED,
        ),
        "export.reimport_before_scoring": Provenance(
            "The customer opens the PLY, so the PLY is what gets scored; final scores are bound "
            "to its sha256.",
            "tools/pipeline.py deliver_steps final_scores",
            INHERITED,
        ),
        "battery": Provenance(
            "The delivery battery and the render fingerprint it must use.",
            "pipeline.house0305.machine.json; delivery_eval configs",
            MEASURED,
        ),
        "battery.views": Provenance(
            "48 probe views is the delivery battery of this campaign; alpha coverage must be "
            "reported next to PSNR or a coverage gap reads as blur.",
            "pipeline.house0305.machine.json; README row 'B5 全场交付候选'",
            MEASURED,
        ),
        "battery.render_spec": Provenance(
            "classic / no-UT and no per-image gain at evaluation time; an A/B whose render "
            "fingerprints differ is not an A/B.",
            "memory eval-renderspec-and-exposure-facts",
            MEASURED,
        ),
        "acceptance": Provenance(
            "The five B5 gates plus the two the fill layer added (alpha mean and p10). Reference "
            "numbers are the B5fill2 delivery of 2026-09-14.",
            "README rows 'B5 结果', 'B5fill2 (10:06)'",
            MEASURED,
        ),
        "acceptance.battery_alpha_mean_min": Provenance(
            "B5 without fill was refused at 0.712; B5fill reached 0.921 and B5fill2 0.947 "
            "against R1d's 0.963. 0.90 is the line the fill campaign set for itself.",
            "README rows 'B5 全场交付候选', '覆盖缺口的两条修法'",
            MEASURED,
        ),
        "cost_model": Provenance(
            "Everything the dry run needs to say how long a machine will be busy and how much "
            "disk it will consume. Individual entries carry their own confidence.",
            "measured run times and du of the house0305 campaign artefacts",
            EXTRAPOLATED,
        ),
        "cost_model.tile_train_seconds_per_step": Provenance(
            "Four measured 20k-step tile runs: cap 6.0M/121 min, 8.0M/152 min, 9.9M/168 min, "
            "11.0M/187 min. Least squares gives 0.141 + 0.0378 s/step per million of cap.",
            "README rows 'B5 结果', 'Tile_2 B5', 'Tile_3 B5', 'Tile_0 B5'",
            MEASURED,
        ),
        "cost_model.backdrop_seconds_per_view": Provenance(
            "Three tiles (6133 views) rendered in 6 minutes.",
            "README row '四切片替身背景就绪 (06:28)'",
            MEASURED,
        ),
        "cost_model.ownership_seconds_per_view": Provenance(
            "Three ownership caches (6133 views) built in 19 CPU minutes.",
            "README row '四切片推广准备 (05:00)'",
            MEASURED,
        ),
        "cost_model.merge_seconds": Provenance(
            "delivery_B5fill merged on CPU in 4 minutes.",
            "README row '合并填充层 (08:20)'",
            MEASURED,
        ),
        "cost_model.backdrop_bytes_per_view": Provenance(
            "2.8/2.5/2.1/2.7 GB over 2132/1829/1684/2317 views = 1.16-1.37 MB per view.",
            "du of 3dgs-runs/house0305_sop/tile_backgrounds_B1/Tile_*",
            MEASURED,
        ),
        "cost_model.tile_run_bytes": Provenance(
            "Finished tile run directories measured 2.0 GB (4.65M final) to 2.7 GB (8.48M final); "
            "the plan budgets the 3 GB upper bound because a preflight that under-budgets disk "
            "is worse than one that over-budgets it.",
            "du of 3dgs-runs/house0305_sop/tile{0,1}_B5_cap6_20k",
            EXTRAPOLATED,
        ),
        "cost_model.merged_bytes_per_gaussian": Provenance(
            "merged.pt 2,313,173,477 B for 25.14M gaussians; the PLY 1,786,971,066 B for 17.18M.",
            "ls of 3dgs-runs/house0305_sop/delivery_B5fill2",
            MEASURED,
        ),
        "cost_model.merge_retained_fraction": Provenance(
            "27.11M source -> 24.61M after core-owner arbitration (0.908), plus 529k fill; "
            "export at 0.05 opacity kept 17.18M of 25.14M (0.683).",
            "delivery_B5fill2/merge_report.json; README row 'B5fill2 (10:06)'",
            MEASURED,
        ),
        "cost_model.sky_mask_seconds_per_face": Provenance(
            "Nobody timed the SegFormer pass; 3536 faces on CPU at b4/512 px is roughly half an "
            "hour. Budgeted, never validated.",
            "estimate",
            UNMEASURED,
        ),
        "cost_model.sky_dome_seconds": Provenance(
            "Not timed; the dome is a 100k-point fit over every 4th rig frame.",
            "estimate",
            UNMEASURED,
        ),
        "cost_model.compare_seconds": Provenance(
            "Not timed separately from the delivery it ran inside.",
            "estimate",
            UNMEASURED,
        ),
        "external_assets": Provenance(
            "The sky mask cache records the model id, revision, weights sha256 and the licence "
            "note 'NVIDIA Source Code License-NC; research supervision data only'.",
            "3dgs-datasets/house0305_sop_v9/sky_mask_train/sky_mask_train.json",
            MEASURED,
        ),
    },
)


def _derive_b5sky() -> Profile:
    """B5 without the fill layer, scored as the pair the customer receives.

    On 2026-09-15 the coverage gap the fill layer existed to close turned out to be the sky
    layer missing from the measurement: a delivery ships body + sky, nothing composited them,
    and the body's correctly transparent sky read as a coverage failure. Composited, the
    no-fill body reads alpha p05 0.898 against the 0.70 bar with sharpness 0.453 against 0.42 -
    the only configuration that passes both - and every fill variant clears coverage while
    failing sharpness, because its one real effect was painting sky into the body.

    So this profile is B5FILL2 with three changes and the acceptance rewritten to read the
    pair. Everything else is inherited verbatim so the two profiles stay comparable.
    """
    base = PROFILE_B5FILL2
    runtime = thaw(base.runtime)
    # b5fill2 wrote the card's nominal 16 GiB here. The card every arm of this campaign
    # trained on reports 15.9 GiB usable (16303 MiB), so that value refused the very hardware
    # the recipe was measured on. What the recipe actually needs is what the vram_headroom
    # check computes from the largest cap; this floor only has to admit the measured card.
    runtime["min_vram_gib"] = 15.5
    trainer_base = thaw(base.trainer_base)
    # 37.7% faster (335.2 -> 208.8 ms/step, two draws each); at 20k the result sits inside the
    # three-run B5 band on every paired ROI comparison (sign tests 31% / 53% / 47%). It does
    # move the trajectory (step-2000 populations of the two conditions do not overlap), and
    # that move produces no quality difference the protocol can see.
    trainer_base["prefetch_training_samples"] = True

    merge = thaw(base.merge)
    merge["fill"] = dict(merge["fill"])
    merge["fill"]["enabled"] = False
    merge["fill"]["why_disabled"] = (
        "the gap it filled was the sky layer missing from the measurement; with the pair "
        "scored it buys 1.7-3.1 points of alpha p05 above an already-passing 0.898 for 8-20% "
        "of novel-view sharpness"
    )

    acceptance = {
        # Every gate below reads the DELIVERED PAIR (body + sky) battery, never the body alone.
        # Reference values are the no-fill B5 delivery of 2026-09-15, composited with its
        # frozen sky layer, on the 48-view battery and the 18-frame off-trajectory strips.
        "scored_layers": "body+sky",
        "battery_alpha_p05_min": 0.70,
        # 15.048 measured; the tile-scale rerun band on p10 is about +-0.16 and the merged
        # delivery has only been measured once, so the floor sits half a dB under the reference
        # rather than at it.
        "battery_psnr_p10_min": 14.50,
        "offtrajectory_sharpness_min": 0.42,
        "morphology_short_axis_p50_mm_max": 0.60,
        "export_gaussian_count_max": 20000000,
        "reference": {
            "battery_psnr_mean": 18.031,
            "battery_psnr_p10": 15.048,
            "battery_alpha_mean": 0.978,
            "battery_alpha_p05": 0.898,
            "battery_alpha_p05_body_only": 0.189,
            "offtrajectory_psnr_18f": 16.73,
            "offtrajectory_sharpness_ours_over_ref": 0.453,
            "offtrajectory_sharpness_body_only": 0.454,
            "morphology_short_axis_p50_mm": 0.452,
            "morphology_axis_ratio": 12.07,
            "export_gaussian_count": 16719228,
            "sky_layer_gaussian_count": 100000,
        },
    }

    provenance = dict(base.provenance)
    provenance["trainer_base.prefetch_training_samples"] = Provenance(
        "Two draws per condition at 2000 steps: 334.9 / 335.6 ms per step without, 207.0 / "
        "210.5 with. A full 20k arm ran 74.2 min against 119.0 / 119.8 / 115.8 for the three "
        "B5 reruns; all three paired ROI comparisons within the rerun band.",
        "research/quality_recovery_v2/README.zh-CN.md rows of 2026-09-15 05:40 and 07:05; "
        "17_prefetch_band_run{1,2,3}.json",
        MEASURED,
    )
    provenance["merge.fill"] = Provenance(
        "With the sky layer composited the no-fill body reads alpha p05 0.898 and sharpness "
        "0.453; sharpc1 0.915 / 0.409; sharpc0 0.929 / 0.354. The fill layer clears coverage "
        "and fails sharpness in every variant because it paints sky into the body.",
        "research/quality_recovery_v2/16_fill_tradeoff_is_structural.zh-CN.md section 4",
        MEASURED,
    )
    provenance["runtime.min_vram_gib"] = Provenance(
        "The campaign's card, an RTX 5070 Ti, reports 15.9 GiB usable (16303 MiB) and trained "
        "every arm; b5fill2's nominal 16.0 refused it at preflight. 15.5 admits the measured "
        "card; the real capacity check is vram_headroom, computed from the largest planned cap.",
        "nvidia-smi on the campaign machine; cloudstudio3dgs_sdk preflight of 2026-09-18",
        MEASURED,
    )
    provenance["acceptance"] = Provenance(
        "Gates read the delivered pair. Body-only scoring counted correctly transparent sky "
        "as a coverage failure (0.189 vs 0.898 for the same model) and drove a whole line of "
        "fill-layer work at a hole that existed only in the measurement.",
        "tools/pipeline.py commit 63bec94 (pair battery, coverage.body_only / delivered_pair); "
        "delivery_B5/battery_with_sky.json",
        MEASURED,
    )

    open_questions = [dict(item) for item in base.open_questions]
    for item in open_questions:
        if item["id"] == "fill-vs-sharpness":
            item["what"] = (
                "Resolved against the fill layer. The trade it offered was 1.7-3.1 points of "
                "alpha p05 above an already-passing 0.898 for 8-20% of novel-view sharpness, "
                "and its coverage was sky painted into the body. The knobs stay in the merge "
                "tool, unused by this profile."
            )
            item["status"] = "closed 2026-09-15: fill disabled; the measurement was the defect"
    open_questions.append(
        {
            "id": "sky-layer-detail",
            "what": (
                "With the dome actually visible, off-trajectory PSNR against the competitor drops "
                "17.07 -> 16.73 because our sky layer is a flat 100k-gaussian dome where their sky "
                "has detail. Not a gate; a real quality gap in the sky layer itself."
            ),
            "expressed_as": "backdrop.sky_dome (count, radius); nothing here improves its appearance",
            "status": "open; separate work item",
        }
    )

    return make_profile(
        name="b5sky",
        version="2026.09.18",
        summary=(
            "house0305 delivery of 2026-09-15: four LiDAR-initialised tiles at 20k steps with "
            "ownership masking, sky supervision and per-view stand-in backdrops, merged with NO "
            "fill layer and shipped with the frozen sky layer. Scored and gated as that pair. "
            "Sample prefetch on (37% faster, inside the rerun band at 20k)."
        ),
        runtime=runtime,
        dataset_contract=thaw(base.dataset_contract),
        tiling=thaw(base.tiling),
        trainer_base=trainer_base,
        tile_rules=thaw(base.tile_rules),
        coarse_prior=thaw(base.coarse_prior),
        backdrop=thaw(base.backdrop),
        merge=merge,
        export=thaw(base.export),
        battery=thaw(base.battery),
        acceptance=acceptance,
        cost_model=thaw(base.cost_model),
        external_assets=thaw(base.external_assets),
        open_questions=tuple(open_questions),
        provenance=provenance,
    )


PROFILE_B5SKY = _derive_b5sky()


def _derive_b6reset() -> Profile:
    """b5sky with the opacity reset every 3000 steps instead of every 300.

    The competitor comparison of 2026-09-19 left one gap, novel-view detail at about half
    of theirs, and ladders L28-L31 on Tile_1 showed why: the recipe turns over ~18% births /
    ~15% deaths of its population every 100-step refine cycle and pushes every opacity back
    under 0.2 every 300 steps, so a gaussian lives a few hundred steps and never converges.
    Loss terms, colour degree, growth threshold and window did nothing (L28); the reset cadence
    did everything at once: door-ROI brightness-matched sharpness 0.461 -> 0.592 against the
    competitor (paired +25%, sign test 0.95, gradient agreement up so it is texture, not
    needles), off-trajectory sharpness +18% (17 of 18 strips), churn 2%, transparent share
    49% -> 26%, mid axis 2.62 -> 2.20 mm - every morphology figure toward the competitor - and
    the gain survives export at min_opacity 0.05 (4.87M gaussians, +25.3%, 42 of 44).

    Everything else is b5sky verbatim. The vendor order honours this cadence through its own
    profile name (deferred_every3000_compatibility); the exact-MipMap contract accepts it.
    Acceptance stays b5sky's: the gates were passed by b5sky and must not move because of a
    candidate. Not the default until a full four-tile delivery has been scored as the pair.
    """
    base = PROFILE_B5SKY
    trainer_base = thaw(base.trainer_base)
    strategy = dict(trainer_base["default_strategy"])
    strategy["reset_every"] = 3000
    strategy["vendor_opacity_reset_profile"] = "deferred_every3000_compatibility"
    trainer_base["default_strategy"] = strategy

    provenance = dict(base.provenance)
    provenance["trainer_base.default_strategy.reset_every"] = Provenance(
        "Tile_1 ladder L31a (base tile1_b5sky_delivery, one change): door-ROI brightness-matched "
        "sharpness ours/competitor 0.461 -> 0.592, paired +25.1% with sign test 0.95 (42 of 44), "
        "gradient energy +6.2% and agreement +6.9% (texture, not needles); off-trajectory strips "
        "+18.2% (17 of 18); tile-owned battery +0.21 / +0.09 dB; churn 17.6%/14.5% -> 1.9%/1.9% "
        "per cycle; opacity < 0.1 share 0.49 -> 0.26; mid axis 2.62 -> 2.20 mm. Exported at "
        "min_opacity 0.05 the gain is unchanged (+25.3%, 42 of 44).",
        "research/quality_recovery_v2/18_ladder28_detail_gap.zh-CN.md 4.12-4.16; ladders/L31.json; "
        "ladders/out/ladder_L31_summary.md; tile1_L31a_reset3000_20k/export_check/summary.json",
        MEASURED,
    )
    provenance["trainer_base.default_strategy.vendor_opacity_reset_profile"] = Provenance(
        "The profile name the vendor execution order derives its expected reset interval from; "
        "3000 is only admitted through it.",
        "cloudstudio_3dgs/training/trainer.py TrainerConfig.validate (pre_optimizer_vendor branch)",
        MEASURED,
    )
    open_questions = [dict(item) for item in base.open_questions]
    open_questions.append(
        {
            "id": "lifecycle-churn",
            "what": (
                "With the reset every 3000 the population still turns over 2% per cycle and 26% of "
                "it sits below opacity 0.1; the competitor carries 18%. The cull threshold (L31b) "
                "and the local coverage cull (L30c) each read sharper on their own; whether they "
                "stack on top of this cadence is unmeasured."
            ),
            "expressed_as": "trainer_base.default_strategy.reset_every / prune_opa / opacity_cull_policy",
            "status": "open; next ladder",
        }
    )
    return make_profile(
        name="b6reset",
        version="2026.09.19",
        summary=(
            "b5sky with the opacity reset every 3000 steps (was 300): the one change that moved "
            "novel-view detail toward the competitor on Tile_1 (door ROI 0.461 -> 0.592, "
            "off-trajectory +18%) with every morphology figure following. Candidate: not scored "
            "as a four-tile delivery yet."
        ),
        runtime=thaw(base.runtime),
        dataset_contract=thaw(base.dataset_contract),
        tiling=thaw(base.tiling),
        trainer_base=trainer_base,
        tile_rules=thaw(base.tile_rules),
        coarse_prior=thaw(base.coarse_prior),
        backdrop=thaw(base.backdrop),
        merge=thaw(base.merge),
        export=thaw(base.export),
        battery=thaw(base.battery),
        acceptance=thaw(base.acceptance),
        cost_model=thaw(base.cost_model),
        external_assets=thaw(base.external_assets),
        open_questions=tuple(open_questions),
        provenance=provenance,
    )


PROFILE_B6RESET = _derive_b6reset()

# The recommended recipe comes first; b5fill2 stays so the campaign's own deliveries reproduce;
# b6reset is the candidate under delivery-scale validation.
PROFILES: Mapping[str, Profile] = MappingProxyType(
    {PROFILE_B5SKY.name: PROFILE_B5SKY, PROFILE_B6RESET.name: PROFILE_B6RESET, PROFILE_B5FILL2.name: PROFILE_B5FILL2}
)
DEFAULT_PROFILE = PROFILE_B5SKY.name


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(sorted(PROFILES))}") from None
