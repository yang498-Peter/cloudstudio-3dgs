"""Per-frame scalar exposure compensation for auto-exposure fisheye captures.

The S1 rig runs both fisheye cameras on independent auto exposure, so the same
static surface is observed at different brightness from frame to frame. Without
compensation the trainer averages the contradiction into washed-out texture and
absorbs shading differences into fake geometry. This module owns one
differentiable log-gain per TRAINING image, applied to the rendered RGB before
the photometric losses only; validation always renders at gain 1.0 so metrics
stay honest, and a strong zero-pull prior keeps the gains from re-encoding real
scene appearance.

Two parameterisations share the trainer contract (``gain(image_id)``,
``prior_loss()``, ``project_zero_mean()``, ``report()``, ``make_optimizer()``):

* ``per_image`` (:class:`ExposureCompensator`): one free log-gain per training
  image, learned by every tile on its own.  The WP06 audit measured that about
  half of that freedom is model residual (the same photo gets a different gain
  in each tile that trains it) rather than exposure.
* ``camera_curve`` (:class:`ExposureCurve`): one piecewise-linear log-gain
  curve per PHYSICAL camera over capture time, knots every ``knot_seconds``,
  each image reading the curve at its own timestamp.  A soft per-camera mean
  anchor keeps the curve from carrying the global brightness, an L2 penalty
  between adjacent knots keeps it smooth.  With ``frozen_curve`` the knots are
  loaded from a scene-wide estimate and not optimised, which is how every tile
  shares one correction and the cross-tile disagreement is zero by construction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXPOSURE_MODES = ("per_image", "camera_curve")
EXPOSURE_CURVE_SCHEMA_VERSION = 1
EXPOSURE_CURVE_KIND = "exposure_camera_curve"


@dataclass(frozen=True)
class ExposureCompensationConfig:
    enabled: bool = False
    learning_rate: float = 5e-3
    regularization_weight: float = 1e-2
    max_abs_log_gain: float = 0.6931471805599453  # ln(2): gain clamped to [0.5, 2]
    zero_mean_projection: bool = False
    mean_anchor_weight: float = 0.0
    mean_anchor_beta: float = 0.1
    # camera_curve parameterisation (ignored, and absent from the contract
    # dict, under the default per_image mode so existing runs stay identical).
    mode: str = "per_image"
    knot_seconds: float = 10.0
    # L2 smoothness between adjacent knots of one camera's curve; the
    # per_image zero-pull ``regularization_weight`` is not applied to knots.
    prior_weight: float = 1e-2
    # Path of a scene-wide curve (tools/fit_exposure_curve.py output); when
    # set the knots are loaded and never optimised.
    frozen_curve: str | None = None

    @property
    def is_curve(self) -> bool:
        return self.mode == "camera_curve"

    @property
    def is_frozen_curve(self) -> bool:
        return self.is_curve and self.frozen_curve is not None

    def validate(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("exposure learning_rate must be non-negative")
        if self.regularization_weight < 0.0:
            raise ValueError("exposure regularization_weight must be non-negative")
        if self.max_abs_log_gain <= 0.0:
            raise ValueError("exposure max_abs_log_gain must be positive")
        if self.mean_anchor_weight < 0.0:
            raise ValueError("exposure mean_anchor_weight must be non-negative")
        if self.mean_anchor_beta <= 0.0:
            raise ValueError("exposure mean_anchor_beta must be positive")
        if self.mode not in EXPOSURE_MODES:
            raise ValueError(
                f"exposure mode must be one of {list(EXPOSURE_MODES)}, got {self.mode!r}"
            )
        if self.mode == "per_image":
            if self.frozen_curve is not None:
                raise ValueError("exposure frozen_curve requires mode camera_curve")
            return
        if not math.isfinite(float(self.knot_seconds)) or self.knot_seconds <= 0.0:
            raise ValueError("exposure knot_seconds must be positive")
        if self.prior_weight < 0.0:
            raise ValueError("exposure prior_weight must be non-negative")
        if self.zero_mean_projection:
            raise ValueError(
                "exposure zero_mean_projection is a per_image knob; camera_curve "
                "uses the soft mean_anchor_weight"
            )
        if self.frozen_curve is not None and not Path(self.frozen_curve).is_file():
            raise ValueError(f"exposure frozen_curve is not a file: {self.frozen_curve}")

    def frozen_curve_sha256(self) -> str | None:
        if self.frozen_curve is None:
            return None
        path = Path(self.frozen_curve)
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "enabled": self.enabled,
            "learning_rate": self.learning_rate,
            "regularization_weight": self.regularization_weight,
            "max_abs_log_gain": self.max_abs_log_gain,
            "zero_mean_projection": self.zero_mean_projection,
            "mean_anchor_weight": self.mean_anchor_weight,
            "mean_anchor_beta": self.mean_anchor_beta,
        }
        if self.mode != "per_image":
            # Only non-default modes extend the contract, so the signed
            # trainer_config_sha256 of every existing per_image run is unchanged.
            value.update(
                {
                    "mode": self.mode,
                    "knot_seconds": float(self.knot_seconds),
                    "prior_weight": self.prior_weight,
                    "frozen_curve": None
                    if self.frozen_curve is None
                    else str(self.frozen_curve),
                    "frozen_curve_sha256": self.frozen_curve_sha256(),
                }
            )
        return value


class ExposureCompensator:
    """Own one clamped scalar log-gain per training image."""

    def __init__(
        self,
        image_ids: Iterable[str],
        *,
        config: ExposureCompensationConfig,
        device: str,
        group_by_image: dict[str, str] | None = None,
    ) -> None:
        import torch

        config.validate()
        if not config.enabled:
            raise ValueError(
                "ExposureCompensator cannot be constructed when compensation is disabled"
            )
        ordered = sorted(set(str(image_id) for image_id in image_ids))
        if not ordered:
            raise ValueError("exposure compensation requires at least one training image")
        self.config = config
        self.device = device
        self.index = {image_id: position for position, image_id in enumerate(ordered)}
        self.log_gains = torch.nn.Parameter(
            torch.zeros(len(ordered), dtype=torch.float32, device=device)
        )
        # Anchor groups: with independent auto-exposure per physical camera, a
        # single global anchor lets the two cameras drift in opposite
        # directions and still sum to zero, so each camera group is projected
        # to zero mean on its own.
        self.group_members: dict[str, list[int]] = {}
        if group_by_image is not None:
            missing = [image_id for image_id in ordered if image_id not in group_by_image]
            if missing:
                raise ValueError(
                    f"exposure groups missing for {len(missing)} images, e.g. {missing[0]!r}"
                )
            for image_id in ordered:
                key = str(group_by_image[image_id])
                self.group_members.setdefault(key, []).append(self.index[image_id])
        else:
            self.group_members["all"] = list(range(len(ordered)))

    def make_optimizer(self) -> Any:
        import torch

        return torch.optim.Adam(
            [{"params": [self.log_gains], "lr": self.config.learning_rate, "name": "exposure"}],
            eps=1e-15,
        )

    def gain(self, image_id: str) -> Any:
        import torch

        position = self.index.get(str(image_id))
        if position is None:
            raise KeyError(f"exposure compensation has no gain for image {image_id!r}")
        bound = self.config.max_abs_log_gain
        return torch.exp(torch.clamp(self.log_gains[position], -bound, bound))

    def prior_loss(self) -> Any:
        import torch

        loss = self.config.regularization_weight * (self.log_gains**2).mean()
        if self.config.mean_anchor_weight > 0.0:
            # Soft per-camera mean anchor (LichtFeld semantics): pull each
            # group's MEAN log gain to zero with SmoothL1 while individual
            # gains stay free. The P8 probe showed the hard zero-mean
            # projection is too aggressive - the model cannot absorb the mean
            # brightness fast enough and the unremovable residual corrupts the
            # structural supervision; the soft prior lets the gains keep
            # compensating short-term while draining the drift long-term.
            beta = self.config.mean_anchor_beta
            for members in self.group_members.values():
                mean = self.log_gains[members].mean()
                loss = (
                    loss
                    + self.config.mean_anchor_weight
                    * torch.nn.functional.smooth_l1_loss(
                        mean, mean.new_zeros(()), beta=beta
                    )
                )
        return loss

    def project_zero_mean(self) -> None:
        """Remove the dataset-mean log gain after an optimizer step.

        Per-image gains and global model brightness are jointly unobservable
        from the photometric loss alone; over a long run the gains drift bright
        while the model itself darkens, and validation (always gain 1.0) pays
        the bill. Projecting the gains onto the zero-mean subspace pins the
        global-brightness degree of freedom inside the model where validation
        can see it, while per-image differences remain free.
        """
        import torch

        if not self.config.zero_mean_projection:
            return
        with torch.no_grad():
            for members in self.group_members.values():
                subset = self.log_gains[members]
                self.log_gains[members] = subset - subset.mean()

    def report(self) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            bound = self.config.max_abs_log_gain
            clamped = torch.clamp(self.log_gains.detach(), -bound, bound)
            gains = torch.exp(clamped)
            absolute = torch.abs(clamped)
            quantiles = torch.quantile(
                absolute, torch.tensor([0.5, 0.95], device=absolute.device)
            )
        return {
            "image_count": int(self.log_gains.shape[0]),
            "mean_log_gain": float(self.log_gains.detach().mean()),
            "mean_log_gain_by_group": {
                key: float(self.log_gains.detach()[members].mean())
                for key, members in sorted(self.group_members.items())
            },
            "abs_log_gain_p50": float(quantiles[0]),
            "abs_log_gain_p95": float(quantiles[1]),
            "abs_log_gain_max": float(absolute.max()),
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "saturated_fraction": float(
                (absolute >= bound - 1e-6).float().mean()
            ),
        }


# --------------------------------------------------------------- curve maths --


def curve_knot_count(span_seconds: float, knot_seconds: float) -> int:
    """Knots at origin + k * knot_seconds covering [0, span]: the last knot is
    at or beyond the span, and there are always two so interpolation is
    defined even for a single-frame camera."""
    if not math.isfinite(span_seconds) or span_seconds < 0.0:
        raise ValueError("curve span must be finite and non-negative")
    if knot_seconds <= 0.0:
        raise ValueError("knot_seconds must be positive")
    return int(math.floor(span_seconds / knot_seconds)) + 2


def curve_interpolation_weights(
    times_s: Sequence[float], *, knot_seconds: float, knot_count: int
) -> list[tuple[int, float]]:
    """Piecewise-linear weights: ``value = (1 - w) * k[lo] + w * k[lo + 1]``.

    ``times_s`` are seconds since the curve origin. Times before the first or
    after the last knot are clamped (constant extrapolation), which is what a
    frozen scene curve does for a frame outside the range it was fitted on.
    """
    if knot_count < 2:
        raise ValueError("a curve needs at least two knots")
    out: list[tuple[int, float]] = []
    top = float(knot_count - 1)
    for time_s in times_s:
        u = float(time_s) / float(knot_seconds)
        if not math.isfinite(u):
            raise ValueError("curve evaluation time must be finite")
        u = min(max(u, 0.0), top)
        lo = min(int(math.floor(u)), knot_count - 2)
        out.append((lo, u - float(lo)))
    return out


def evaluate_curve(
    knots: Sequence[float], times_s: Sequence[float], *, knot_seconds: float
) -> list[float]:
    """Pure-python evaluation of one camera's curve (tools and tests)."""
    values: list[float] = []
    for lo, w in curve_interpolation_weights(
        times_s, knot_seconds=knot_seconds, knot_count=len(knots)
    ):
        values.append((1.0 - w) * float(knots[lo]) + w * float(knots[lo + 1]))
    return values


def build_curve_payload(
    *,
    knot_seconds: float,
    time_origin_ns: int,
    cameras: Mapping[str, Sequence[float]],
    max_abs_log_gain: float,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen-curve file: knots per physical camera on a shared time grid."""
    if not cameras:
        raise ValueError("a curve payload needs at least one camera")
    payload: dict[str, Any] = {
        "schema_version": EXPOSURE_CURVE_SCHEMA_VERSION,
        "kind": EXPOSURE_CURVE_KIND,
        "knot_seconds": float(knot_seconds),
        "time_origin_ns": int(time_origin_ns),
        "max_abs_log_gain": float(max_abs_log_gain),
        "cameras": {},
    }
    for camera, knots in sorted(cameras.items()):
        values = [float(value) for value in knots]
        if len(values) < 2:
            raise ValueError(f"camera {camera!r} curve needs at least two knots")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"camera {camera!r} curve has non-finite knots")
        payload["cameras"][str(camera)] = {
            "knot_count": len(values),
            "knot_log_gains": values,
        }
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    return payload


def load_curve_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != EXPOSURE_CURVE_KIND:
        raise ValueError(f"{path} is not an exposure camera curve")
    if payload.get("schema_version") != EXPOSURE_CURVE_SCHEMA_VERSION:
        raise ValueError(f"unsupported exposure curve schema in {path}")
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"exposure curve {path} has no cameras")
    for camera, entry in cameras.items():
        knots = entry.get("knot_log_gains") if isinstance(entry, dict) else None
        if not isinstance(knots, list) or len(knots) < 2:
            raise ValueError(f"exposure curve {path}: camera {camera!r} has no knots")
        if int(entry.get("knot_count", len(knots))) != len(knots):
            raise ValueError(f"exposure curve {path}: camera {camera!r} knot_count mismatch")
        if any(not math.isfinite(float(value)) for value in knots):
            raise ValueError(f"exposure curve {path}: camera {camera!r} has non-finite knots")
    return payload


class ExposureCurve:
    """Per physical camera, a piecewise-linear log-gain curve over capture time.

    Parameters are the knot values of every camera, concatenated in one 1-D
    tensor (``knot_log_gains``) so the checkpoint contract is a single
    auxiliary parameter like the per-image gains.  Each training image reads
    the curve of its own camera at its own timestamp; the evaluation is a
    fixed two-tap interpolation prepared at construction.
    """

    def __init__(
        self,
        image_ids: Iterable[str],
        *,
        config: ExposureCompensationConfig,
        device: str,
        camera_by_image: Mapping[str, str],
        timestamp_ns_by_image: Mapping[str, int],
    ) -> None:
        import torch

        config.validate()
        if not config.enabled:
            raise ValueError("ExposureCurve cannot be constructed when compensation is disabled")
        if not config.is_curve:
            raise ValueError("ExposureCurve requires exposure mode camera_curve")
        ordered = sorted(set(str(image_id) for image_id in image_ids))
        if not ordered:
            raise ValueError("exposure compensation requires at least one training image")
        missing = [i for i in ordered if i not in camera_by_image]
        if missing:
            raise ValueError(
                f"exposure cameras missing for {len(missing)} images, e.g. {missing[0]!r}"
            )
        missing = [i for i in ordered if i not in timestamp_ns_by_image]
        if missing:
            raise ValueError(
                f"exposure timestamps missing for {len(missing)} images, e.g. {missing[0]!r}"
            )
        self.config = config
        self.device = device
        self.index = {image_id: position for position, image_id in enumerate(ordered)}
        self.camera_by_image = {i: str(camera_by_image[i]) for i in ordered}
        self.timestamp_ns_by_image = {i: int(timestamp_ns_by_image[i]) for i in ordered}
        self.knot_seconds = float(config.knot_seconds)
        self.frozen = config.frozen_curve is not None
        self.frozen_curve_sha256: str | None = None

        cameras_present = sorted(set(self.camera_by_image.values()))
        knots_by_camera: dict[str, list[float]]
        if self.frozen:
            payload = load_curve_payload(config.frozen_curve)
            self.frozen_curve_sha256 = config.frozen_curve_sha256()
            file_knot_seconds = float(payload["knot_seconds"])
            if not math.isclose(file_knot_seconds, self.knot_seconds, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"frozen curve knot_seconds {file_knot_seconds} does not match "
                    f"config knot_seconds {self.knot_seconds}"
                )
            absent = [c for c in cameras_present if c not in payload["cameras"]]
            if absent:
                raise ValueError(f"frozen curve has no camera {absent[0]!r}")
            self.time_origin_ns = int(payload["time_origin_ns"])
            knots_by_camera = {
                camera: [float(v) for v in payload["cameras"][camera]["knot_log_gains"]]
                for camera in cameras_present
            }
        else:
            # Learnable: a shared origin at the earliest training frame, one
            # grid per camera long enough to cover that camera's last frame.
            self.time_origin_ns = min(self.timestamp_ns_by_image.values())
            knots_by_camera = {}
            for camera in cameras_present:
                last_ns = max(
                    self.timestamp_ns_by_image[i]
                    for i in ordered
                    if self.camera_by_image[i] == camera
                )
                count = curve_knot_count(
                    (last_ns - self.time_origin_ns) / 1e9, self.knot_seconds
                )
                knots_by_camera[camera] = [0.0] * count

        self.camera_slices: dict[str, tuple[int, int]] = {}
        flat: list[float] = []
        for camera in cameras_present:
            start = len(flat)
            flat.extend(knots_by_camera[camera])
            self.camera_slices[camera] = (start, len(knots_by_camera[camera]))
        self.knot_log_gains = torch.nn.Parameter(
            torch.tensor(flat, dtype=torch.float32, device=device),
            requires_grad=not self.frozen,
        )

        # Fixed interpolation taps per image (global knot indices) and, per
        # camera, the row-mean of the interpolation matrix: the camera's mean
        # log gain over its training images is ``anchor_weights . knots``.
        lo_index = [0] * len(ordered)
        hi_index = [0] * len(ordered)
        weight = [0.0] * len(ordered)
        anchor_rows: dict[str, list[float]] = {
            camera: [0.0] * len(flat) for camera in cameras_present
        }
        image_count_by_camera: dict[str, int] = {c: 0 for c in cameras_present}
        for camera in cameras_present:
            members = [i for i in ordered if self.camera_by_image[i] == camera]
            start, count = self.camera_slices[camera]
            times_s = [
                (self.timestamp_ns_by_image[i] - self.time_origin_ns) / 1e9 for i in members
            ]
            taps = curve_interpolation_weights(
                times_s, knot_seconds=self.knot_seconds, knot_count=count
            )
            image_count_by_camera[camera] = len(members)
            for image_id, (lo, w) in zip(members, taps):
                position = self.index[image_id]
                lo_index[position] = start + lo
                hi_index[position] = start + lo + 1
                weight[position] = w
                anchor_rows[camera][start + lo] += (1.0 - w) / len(members)
                anchor_rows[camera][start + lo + 1] += w / len(members)
        self._lo = torch.tensor(lo_index, dtype=torch.long, device=device)
        self._hi = torch.tensor(hi_index, dtype=torch.long, device=device)
        self._w = torch.tensor(weight, dtype=torch.float32, device=device)
        self._anchor = {
            camera: torch.tensor(row, dtype=torch.float32, device=device)
            for camera, row in anchor_rows.items()
        }
        self.image_count_by_camera = image_count_by_camera

    # ------------------------------------------------------------ contract --

    def make_optimizer(self) -> Any:
        import torch

        if self.frozen:
            return None
        return torch.optim.Adam(
            [
                {
                    "params": [self.knot_log_gains],
                    "lr": self.config.learning_rate,
                    "name": "exposure_curve",
                }
            ],
            eps=1e-15,
        )

    def log_gains_all(self) -> Any:
        """Unclamped per-image log gains in index order (differentiable)."""
        return (1.0 - self._w) * self.knot_log_gains[self._lo] + self._w * self.knot_log_gains[self._hi]

    def log_gain(self, image_id: str) -> Any:
        position = self.index.get(str(image_id))
        if position is None:
            raise KeyError(f"exposure compensation has no gain for image {image_id!r}")
        w = self._w[position]
        return (1.0 - w) * self.knot_log_gains[self._lo[position]] + w * self.knot_log_gains[
            self._hi[position]
        ]

    def gain(self, image_id: str) -> Any:
        import torch

        bound = self.config.max_abs_log_gain
        return torch.exp(torch.clamp(self.log_gain(image_id), -bound, bound))

    def log_gain_at(self, camera: str, timestamp_ns: int) -> float:
        """Evaluate a camera's curve at an arbitrary time (no grad; tools)."""
        if camera not in self.camera_slices:
            raise KeyError(f"exposure curve has no camera {camera!r}")
        start, count = self.camera_slices[camera]
        knots = self.knot_log_gains.detach()[start : start + count].tolist()
        return evaluate_curve(
            knots,
            [(int(timestamp_ns) - self.time_origin_ns) / 1e9],
            knot_seconds=self.knot_seconds,
        )[0]

    def prior_loss(self) -> Any:
        import torch

        if self.frozen:
            return self.knot_log_gains.new_zeros(())
        loss = self.knot_log_gains.new_zeros(())
        if self.config.prior_weight > 0.0:
            # L2 smoothness between adjacent knots of the same camera; the
            # mean over intervals mirrors the per_image mean-square scale.
            diffs = []
            for start, count in self.camera_slices.values():
                segment = self.knot_log_gains[start : start + count]
                diffs.append(segment[1:] - segment[:-1])
            loss = loss + self.config.prior_weight * (torch.cat(diffs) ** 2).mean()
        if self.config.mean_anchor_weight > 0.0:
            # Soft per-camera mean anchor: the camera's mean log gain over its
            # training images (a fixed linear functional of the knots) is
            # pulled to zero with SmoothL1, so the curve carries drift and
            # never the global brightness; the hard zero-mean projection is
            # deliberately unavailable in this mode (P8 probe).
            beta = self.config.mean_anchor_beta
            for row in self._anchor.values():
                mean = (row * self.knot_log_gains).sum()
                loss = loss + self.config.mean_anchor_weight * torch.nn.functional.smooth_l1_loss(
                    mean, mean.new_zeros(()), beta=beta
                )
        return loss

    def project_zero_mean(self) -> None:
        """No-op: the curve is anchored softly, never projected."""
        return None

    def mean_log_gain_by_camera(self) -> dict[str, float]:
        import torch

        with torch.no_grad():
            return {
                camera: float((row * self.knot_log_gains).sum())
                for camera, row in sorted(self._anchor.items())
            }

    def curve_payload(self, *, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
        knots = self.knot_log_gains.detach().cpu().tolist()
        return build_curve_payload(
            knot_seconds=self.knot_seconds,
            time_origin_ns=self.time_origin_ns,
            cameras={
                camera: knots[start : start + count]
                for camera, (start, count) in self.camera_slices.items()
            },
            max_abs_log_gain=self.config.max_abs_log_gain,
            provenance=provenance,
        )

    def report(self) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            bound = self.config.max_abs_log_gain
            clamped = torch.clamp(self.log_gains_all().detach(), -bound, bound)
            gains = torch.exp(clamped)
            absolute = torch.abs(clamped)
            quantiles = torch.quantile(
                absolute, torch.tensor([0.5, 0.95], device=absolute.device)
            )
            knots = self.knot_log_gains.detach()
        return {
            "mode": "camera_curve",
            "frozen": self.frozen,
            "frozen_curve": self.config.frozen_curve,
            "frozen_curve_sha256": self.frozen_curve_sha256,
            "knot_seconds": self.knot_seconds,
            "time_origin_ns": self.time_origin_ns,
            "parameter_count": int(knots.shape[0]),
            "knot_count_by_camera": {
                camera: count for camera, (_start, count) in sorted(self.camera_slices.items())
            },
            "image_count": int(clamped.shape[0]),
            "image_count_by_camera": dict(sorted(self.image_count_by_camera.items())),
            "mean_log_gain": float(clamped.mean()),
            "mean_log_gain_by_group": self.mean_log_gain_by_camera(),
            "abs_log_gain_p50": float(quantiles[0]),
            "abs_log_gain_p95": float(quantiles[1]),
            "abs_log_gain_max": float(absolute.max()),
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "saturated_fraction": float((absolute >= bound - 1e-6).float().mean()),
            "knot_abs_max": float(knots.abs().max()),
        }
