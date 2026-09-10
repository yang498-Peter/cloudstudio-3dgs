"""One explicit render specification shared by every evaluator.

A score is only comparable with another score when both were rendered the
same way. Until now each evaluator decided the render on its own - the camera
model came from whichever dataset it happened to open, the rasterize mode
from a backend default, the SH degree from a config that said 0 - and the
report carried none of it. This module resolves every render decision from
the config and the checkpoint where the code actually defines it, records
which fields the evaluator path cannot honour (``unpropagated``), and hashes
the result so a fingerprint can sit next to every number.

Pure and torch-free: ``params`` is only inspected for tensor shapes, so a
dict of numpy arrays or anything with ``.shape`` works.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Mapping

PROPAGATED = "propagated"
UNPROPAGATED = "unpropagated"

# gsplat.rasterization() defaults (gsplat 1.5.3 signature); backend.render
# passes neither, in training and in evaluation alike.
GSPLAT_NEAR_PLANE_M = 0.01
GSPLAT_FAR_PLANE_M = 1e10

RASTERIZE_MODES = ("classic", "antialiased")
CAMERA_MODELS = ("pinhole", "fisheye")
EXPOSURE_POLICIES = ("none", "per_image_gain", "per_tile_gain", "canonical")
BACKGROUND_POLICIES = ("constant", "view_background_library")


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True)
class SpecField:
    """One resolved render decision: what is applied, where that comes from,
    and whether the evaluator path honours it."""

    value: Any
    source: str
    status: str = PROPAGATED
    note: str | None = None

    def __post_init__(self) -> None:
        if self.status not in (PROPAGATED, UNPROPAGATED):
            raise ValueError(f"unknown SpecField status {self.status!r}")


@dataclass(frozen=True)
class PinholeRenderMode:
    """What the trainer configured versus what an evaluator applies."""

    trained_rasterize_mode: str
    trained_with_ut: bool
    applied_rasterize_mode: str
    applied_with_ut: bool

    @property
    def differs(self) -> bool:
        return (
            self.trained_rasterize_mode != self.applied_rasterize_mode
            or self.trained_with_ut != self.applied_with_ut
        )


def resolve_pinhole_render_mode(
    config: Mapping[str, Any], *, honour_render_mode: bool = False
) -> PinholeRenderMode:
    """Read pinhole_rasterize_mode / pinhole_with_ut the way the trainer does.

    The trainer defaults to classic / no UT (TrainerConfig fields), validates
    the pair the same way as here, and copies them verbatim onto the backend.
    Evaluators applied the backend defaults regardless of the config until the
    ``honour_render_mode`` flag existed; with the flag off the applied pair
    stays classic / False so historical scores remain comparable, and the
    difference is recorded rather than silently dropped.
    """
    trained_mode = str(config.get("pinhole_rasterize_mode", "classic"))
    trained_ut = bool(config.get("pinhole_with_ut", False))
    if trained_mode not in RASTERIZE_MODES:
        raise ValueError("pinhole_rasterize_mode must be 'classic' or 'antialiased'")
    if trained_ut and trained_mode != "classic":
        raise ValueError("pinhole_with_ut requires classic rasterization")
    if honour_render_mode:
        applied_mode, applied_ut = trained_mode, trained_ut
    else:
        applied_mode, applied_ut = "classic", False
    return PinholeRenderMode(
        trained_rasterize_mode=trained_mode,
        trained_with_ut=trained_ut,
        applied_rasterize_mode=applied_mode,
        applied_with_ut=applied_ut,
    )


def model_sh_degree_from_params(params: Mapping[str, Any] | None) -> int | None:
    """Degree the checkpoint actually carries: sqrt(K0 + KN) - 1.

    ``None`` when no params are given or the model is not an SH model (the
    rgb_sigmoid colour model has no ``sh0``).
    """
    if params is None or "sh0" not in params or "shN" not in params:
        return None
    total = int(params["sh0"].shape[-2]) + int(params["shN"].shape[-2])
    return max(0, int(round(math.sqrt(total))) - 1)


@dataclass(frozen=True)
class RenderSpec:
    camera_model: SpecField
    intrinsics_source: SpecField
    distortion: SpecField
    pixel_centre_convention: SpecField
    resolution: SpecField
    sh_degree: SpecField
    colour_space: SpecField
    exposure_policy: SpecField
    background: SpecField
    rasterize_mode: SpecField
    with_ut: SpecField
    with_eval3d: SpecField
    clipping_near_far: SpecField
    alpha_rule: SpecField
    coordinate_frame: SpecField

    SCHEMA_VERSION: ClassVar[str] = "render-spec-1.0"

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    # -- construction -----------------------------------------------------

    @classmethod
    def from_config_and_params(
        cls,
        config: Mapping[str, Any],
        params: Mapping[str, Any] | None,
        *,
        camera_model: str | None = None,
        applied_sh_degree: int | None = None,
        applied_rasterize_mode: str | None = None,
        applied_with_ut: bool | None = None,
        honour_render_mode: bool = False,
        background_policy: str | None = None,
        background_rgb: Any | None = None,
        background_manifest: str | None = None,
        intrinsics_manifest: str | None = None,
        tile_crops: bool | None = None,
        checkpoint_meta: Mapping[str, Any] | None = None,
    ) -> "RenderSpec":
        """Resolve every field from the config and checkpoint.

        ``camera_model`` is the evaluator's dataset choice (FaceCacheDataset
        renders pinhole faces, S1TrainingDataset the raw fisheye); when not
        given it follows the trainer's rule: faces when a face cache is bound.
        ``applied_*`` are what the backend really carries (read them off the
        backend after ``_load_backend``); when absent they are derived from the
        config and ``honour_render_mode`` with the same rule ``_load_backend``
        uses. ``checkpoint_meta`` is the checkpoint payload minus tensors (its
        ``merge`` report says whether per-tile exposure gains were baked in).
        """
        if camera_model is None:
            camera_model = "pinhole" if config.get("face_cache_manifest") else "fisheye"
        if camera_model not in CAMERA_MODELS:
            raise ValueError(f"unsupported camera_model {camera_model!r}")
        fisheye = camera_model == "fisheye"

        mode = resolve_pinhole_render_mode(config, honour_render_mode=honour_render_mode)
        if applied_rasterize_mode is None:
            applied_rasterize_mode = mode.applied_rasterize_mode
        if applied_with_ut is None:
            applied_with_ut = mode.applied_with_ut
        if applied_rasterize_mode not in RASTERIZE_MODES:
            raise ValueError(f"unknown rasterize mode {applied_rasterize_mode!r}")
        applied_with_ut = bool(applied_with_ut)

        factor = int(config.get("factor", 1))
        if tile_crops is None:
            tile_crops = bool(config.get("tile_inputs_manifest"))

        # --- camera / projection ---------------------------------------
        camera = SpecField(
            value=camera_model,
            source=(
                "evaluator dataset: FaceCacheDataset samples carry camera_model="
                "'pinhole' (face_dataset.py), S1TrainingDataset 'fisheye' "
                "(dataset.py); backend.render reads sample.camera_model"
            ),
        )
        if fisheye:
            intrinsics = SpecField(
                value={
                    "kind": "dataset_manifest_intrinsic_over_factor",
                    "manifest": intrinsics_manifest or config.get("dataset_manifest"),
                    "factor": factor,
                },
                source="dataset.py S1TrainingDataset.__getitem__: K from cameras[*].intrinsic / factor",
            )
            distortion = SpecField(
                value="OPENCV_FISHEYE_k1_k4",
                source=(
                    "dataset_manifest cameras[*].distortion (OPENCV_FISHEYE enforced in "
                    "dataset.py); passed to gsplat as radial_coeffs"
                ),
            )
        else:
            intrinsics = SpecField(
                value={
                    "kind": "face_cache_K_face_minus_crop_offset",
                    "manifest": intrinsics_manifest or config.get("face_cache_manifest"),
                    "factor": factor,
                },
                source=(
                    "face_dataset.py FaceCacheDataset.__getitem__: K = face.K_face, "
                    "K[0,2]/K[1,2] minus the tile crop x/y"
                ),
            )
            distortion = SpecField(
                value="none",
                source="face samples carry radial_coeffs zeros; backend.render passes no radial for pinhole",
            )
        pixel_centre = SpecField(
            value="pixel_center_plus_half",
            source=(
                "gsplat samples pixel centres at +0.5 with K in pixel units; the face "
                "cache declares pixel_convention='pixel_center_plus_half' in face_manifest.json"
            ),
            status=UNPROPAGATED,
            note="not carried on TrainingSample; no evaluator verifies the manifest declaration",
        )
        resolution = SpecField(
            value={"factor": factor, "tile_crops": bool(tile_crops)},
            source=(
                "sample.width/height: native face size (or tile crop) for pinhole, "
                "sensor size / factor for fisheye"
            ),
        )

        # --- colour ------------------------------------------------------
        model_degree = model_sh_degree_from_params(params)
        config_degree = config.get("sh_degree")
        color_model = str(config.get("color_model", "rgb_sigmoid"))
        rendered_degree = (
            applied_sh_degree if applied_sh_degree is not None else model_degree
        )
        if color_model != "sh":
            sh = SpecField(
                value={"rendered": None, "model": None, "config": config_degree,
                       "color_model": color_model},
                source="config.color_model is not 'sh'; colours are sigmoid RGB",
            )
        elif model_degree is None:
            sh = SpecField(
                value={"rendered": rendered_degree, "model": None,
                       "config": config_degree, "color_model": color_model},
                source="backend.sh_degree from the config / caller; params not inspected",
                status=UNPROPAGATED,
                note="model degree unknown until params are given",
            )
        else:
            sh = SpecField(
                value={"rendered": rendered_degree, "model": model_degree,
                       "config": config_degree, "color_model": color_model},
                source=(
                    "model: sqrt(K0+KN)-1 from the checkpoint tensors; rendered: "
                    "backend.sh_degree after the evaluator's model-degree override"
                ),
                status=PROPAGATED if rendered_degree == model_degree else UNPROPAGATED,
                note=(
                    None
                    if rendered_degree == model_degree
                    else "rendered degree differs from the model's degree"
                ),
            )
        colour_space = SpecField(
            value="srgb8_over_255_as_linear",
            source=(
                "targets are 8-bit sRGB PNG/JPEG divided by 255; renders are raw SH RGB "
                "clamped to [0,1]; no linearisation anywhere in training or evaluation"
            ),
            status=UNPROPAGATED,
            note="no field carries it; assumed identical everywhere",
        )
        exposure = cls._resolve_exposure(config, checkpoint_meta)

        # --- compositing -------------------------------------------------
        if background_policy is None:
            background_policy = (
                "view_background_library"
                if config.get("background_image_manifest")
                else "constant"
            )
        if background_policy not in BACKGROUND_POLICIES:
            raise ValueError(f"unknown background policy {background_policy!r}")
        if background_rgb is None:
            background_rgb = config.get("background_color")
        background = SpecField(
            value={
                "policy": background_policy,
                "constant_rgb": (
                    None if background_rgb is None else [float(c) for c in background_rgb]
                ),
                "manifest": (
                    (background_manifest or config.get("background_image_manifest"))
                    if background_policy == "view_background_library"
                    else None
                ),
            },
            source=(
                "backend.render composites rgb + (1 - alpha) * background_rgb; the "
                "evaluator passes a per-view ViewBackgroundLibrary backdrop or a constant"
            ),
        )

        # --- rasterizer --------------------------------------------------
        if fisheye:
            rasterize = SpecField(
                value={"applied": "classic", "trained": "classic"},
                source="backend.render forces rasterize_mode='classic' for fisheye",
            )
            ut = SpecField(
                value={"applied": True, "trained": True},
                source="backend.render forces with_ut=True for fisheye (3DGUT)",
            )
        else:
            mode_differs = applied_rasterize_mode != mode.trained_rasterize_mode
            rasterize = SpecField(
                value={"applied": applied_rasterize_mode,
                       "trained": mode.trained_rasterize_mode},
                source=(
                    "trained: config.pinhole_rasterize_mode (trainer.py copies it onto "
                    "backend.pinhole_rasterize_mode); applied: backend attribute set by "
                    "_load_backend"
                ),
                status=UNPROPAGATED if mode_differs else PROPAGATED,
                note=(
                    "evaluator applied a different mode than training; pass "
                    "--honour-render-mode to render as trained"
                    if mode_differs
                    else None
                ),
            )
            ut_differs = applied_with_ut != mode.trained_with_ut
            ut = SpecField(
                value={"applied": applied_with_ut, "trained": mode.trained_with_ut},
                source=(
                    "trained: config.pinhole_with_ut (trainer.py copies it onto "
                    "backend.pinhole_with_ut); applied: backend attribute set by "
                    "_load_backend"
                ),
                status=UNPROPAGATED if ut_differs else PROPAGATED,
                note=(
                    "evaluator applied a different UT setting than training; pass "
                    "--honour-render-mode to render as trained"
                    if ut_differs
                    else None
                ),
            )
        eval3d_applied = fisheye or applied_with_ut
        eval3d = SpecField(
            value=eval3d_applied,
            source="backend.render: with_eval3d = (fisheye or pinhole_with_ut), same as with_ut",
        )
        clipping = SpecField(
            value=[GSPLAT_NEAR_PLANE_M, GSPLAT_FAR_PLANE_M],
            source="gsplat rasterization() defaults; backend.render passes neither near_plane nor far_plane",
            status=UNPROPAGATED,
            note="not configurable and not recorded in any training record",
        )
        alpha = SpecField(
            value={
                "composite": "rgb + (1 - alpha) * background",
                "alpha_output": "un-composited accumulated alpha",
                "global_z_order": not fisheye,
                "range_semantics": (
                    "euclidean_ray_range_m" if eval3d_applied else "pinhole_z_depth_m"
                ),
            },
            source="backend.render tail; global_z_order = camera_model != 'fisheye'",
        )
        frame = SpecField(
            value={
                "frame": "s1_local",
                "pose_convention": "c2w_opencv",
                "face_rotation": not fisheye,
            },
            source=(
                "dataset.py enforces dataset_manifest.coordinate_frame == 's1_local'; "
                "face c2w = c2w_base @ [R_face | 0] (face_dataset.py)"
            ),
        )
        return cls(
            camera_model=camera,
            intrinsics_source=intrinsics,
            distortion=distortion,
            pixel_centre_convention=pixel_centre,
            resolution=resolution,
            sh_degree=sh,
            colour_space=colour_space,
            exposure_policy=exposure,
            background=background,
            rasterize_mode=rasterize,
            with_ut=ut,
            with_eval3d=eval3d,
            clipping_near_far=clipping,
            alpha_rule=alpha,
            coordinate_frame=frame,
        )

    @staticmethod
    def _resolve_exposure(
        config: Mapping[str, Any], checkpoint_meta: Mapping[str, Any] | None
    ) -> SpecField:
        """Which exposure frame the render is in.

        Training learns one scalar gain per source image and multiplies the
        render by it for the photometric losses only; the trainer's own
        validation and every evaluator render at gain 1.0 (``canonical``).
        The tile merge with --harmonize-exposure folds each tile's median
        gain into its DC colour, so a merged checkpoint already sits in a
        per-tile frame (``per_tile_gain``) and the evaluator has nothing to
        apply. ``per_image_gain`` is reserved for an evaluator that applies
        the learned gains; none does today.
        """
        exposure_config = config.get("exposure_compensation") or {}
        training = {
            "enabled": bool(exposure_config.get("enabled", False)),
            "learning_rate": exposure_config.get("learning_rate"),
            "zero_mean_projection": bool(exposure_config.get("zero_mean_projection", False)),
            "mean_anchor_weight": float(exposure_config.get("mean_anchor_weight", 0.0)),
            "ppisp": bool((config.get("ppisp") or {}).get("enabled", False)),
        }
        merge = None if checkpoint_meta is None else checkpoint_meta.get("merge")
        baked_gains = None
        harmonized = None
        if isinstance(merge, Mapping):
            harmonized = bool(merge.get("exposure_harmonized", False))
            if harmonized:
                baked_gains = [
                    record.get("exposure_gain_applied")
                    for record in merge.get("records", [])
                ]
        if harmonized:
            return SpecField(
                value={"policy": "per_tile_gain", "training": training,
                       "baked_tile_gains": baked_gains},
                source=(
                    "checkpoint.merge.exposure_harmonized (merge_v28_tile_checkpoints.py "
                    "--harmonize-exposure bakes each tile's median gain into sh0); "
                    "evaluator renders the baked colours at gain 1.0"
                ),
            )
        if not training["enabled"] and not training["ppisp"]:
            return SpecField(
                value={"policy": "none", "training": training, "baked_tile_gains": None},
                source="config.exposure_compensation.enabled is false; nothing learned",
            )
        if checkpoint_meta is None:
            return SpecField(
                value={"policy": "canonical", "training": training, "baked_tile_gains": None},
                source=(
                    "evaluator renders at gain 1.0 like trainer validation "
                    "(golden_eval._evaluate_views passes no gain)"
                ),
                status=UNPROPAGATED,
                note=(
                    "checkpoint metadata not inspected: a merged checkpoint with baked "
                    "per-tile gains cannot be told apart from a single-tile one"
                ),
            )
        return SpecField(
            value={"policy": "canonical", "training": training, "baked_tile_gains": None},
            source=(
                "learned per-image gains live in checkpoint.auxiliary_params."
                "exposure_log_gains and are applied only inside "
                "trainer._render_supervision_loss; validation and every evaluator "
                "render at gain 1.0"
            ),
        )

    # -- serialisation ------------------------------------------------------

    def unpropagated(self) -> list[str]:
        return [
            name for name in self.field_names()
            if getattr(self, name).status == UNPROPAGATED
        ]

    def fingerprint_payload(self) -> dict[str, Any]:
        """The part of the spec that decides comparability: values and
        statuses. Sources and notes are documentation and stay out."""
        return {
            name: {"value": getattr(self, name).value, "status": getattr(self, name).status}
            for name in self.field_names()
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.fingerprint_payload())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "fields": {
                name: {
                    "value": getattr(self, name).value,
                    "source": getattr(self, name).source,
                    "status": getattr(self, name).status,
                    **(
                        {"note": getattr(self, name).note}
                        if getattr(self, name).note is not None
                        else {}
                    ),
                }
                for name in self.field_names()
            },
            "unpropagated": self.unpropagated(),
        }

    def record(self) -> dict[str, Any]:
        """What an evaluator writes next to its scores."""
        return {**self.to_dict(), "fingerprint": self.fingerprint()}

    def with_field(self, name: str, value: SpecField) -> "RenderSpec":
        if name not in self.field_names():
            raise KeyError(name)
        return replace(self, **{name: value})
