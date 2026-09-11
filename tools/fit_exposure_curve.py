#!/usr/bin/env python3
"""Fit the scene-wide per-camera exposure curve (X1) from learned tile gains.

No training and no rendering.  The WP06 audit joined the per-image log gains
of the four R1 tile checkpoints with each image's physical camera, capture
timestamp and tile (``06_photometric_consistency.csv``: one row per image x
tile that trained it).  This tool fits, per physical camera, the
piecewise-linear log-gain curve :class:`cloudstudio_3dgs.training.exposure.ExposureCurve`
evaluates at training time:

    log_gain(image i in tile T) ~= f_cam(i)(t_i) + b_T

* ``f_c`` has one knot every ``--knot-seconds`` on a grid shared by both
  cameras (origin = earliest frame in the CSV);
* ``b_T`` is one nuisance offset per tile: each tile's model frame sits at its
  own brightness (the tile median gains 0.88-0.93 the merge bakes out), which
  is not exposure and must not enter the shared curve;
* L2 smoothness ``--smoothness-weight * sum (k[j+1]-k[j])^2`` per camera;
* soft mean anchor ``--anchor-weight * (mean over the camera's images of f_c)^2``
  so each camera's curve has zero mean and the global brightness stays in the
  model (the trainer's ``mean_anchor_weight`` plays the same role);
* Huber IRLS (``--huber-delta``) so the ~1.5 % clamp-saturated gains and the
  per-tile residual outliers cannot drag the curve.  The observation is the
  applied (clamped) log gain, exactly what the trainer multiplied.

Outputs: the frozen curve file (``--out-curve``; consumed by
``exposure_compensation.frozen_curve``), a per-row CSV of gain vs fitted
(``--out-csv``) and, inside the curve file's ``provenance``, residuals per
camera / tile / environment, effective degrees of freedom (trace of the hat
matrix), a smoothness sweep with GCV, and a leave-one-tile-out check (fit on
three tiles, predict the fourth up to its own offset).

Example::

    python tools/fit_exposure_curve.py \
        --consistency-csv research/quality_recovery_v2/06_photometric_consistency.csv \
        --knot-seconds 10 --smoothness-weight 3 --anchor-weight 1e4 \
        --out-curve research/quality_recovery_v2/06_exposure_curve_scene.json \
        --out-csv research/quality_recovery_v2/06_exposure_curve_fit.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.exposure import (  # noqa: E402
    build_curve_payload,
    curve_interpolation_weights,
    curve_knot_count,
    evaluate_curve,
)

LN2 = math.log(2.0)
REQUIRED_COLUMNS = ("image_id", "camera", "tile", "timestamp_ns", "log_gain")
DEFAULT_SWEEP = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)


# ------------------------------------------------------------------ input --


def read_gain_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks columns {missing}")
        rows = []
        for raw in reader:
            rows.append(
                {
                    "image_id": str(raw["image_id"]),
                    "camera": str(raw["camera"]),
                    "tile": str(raw["tile"]),
                    "environment": str(raw.get("environment", "")),
                    "timestamp_ns": int(raw["timestamp_ns"]),
                    "log_gain": float(raw["log_gain"]),
                    "log_gain_raw": float(raw.get("log_gain_raw", raw["log_gain"])),
                    "saturated_clamp": int(float(raw.get("saturated_clamp", 0) or 0)),
                }
            )
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


# ----------------------------------------------------------------- design --


class CurveDesign:
    """Sparse-ish design of ``y = A x``: x = [knots of every camera | tile offsets]."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        knot_seconds: float,
        time_origin_ns: int | None = None,
        knot_count_by_camera: Mapping[str, int] | None = None,
    ) -> None:
        if knot_seconds <= 0.0:
            raise ValueError("knot_seconds must be positive")
        self.knot_seconds = float(knot_seconds)
        self.rows = list(rows)
        self.cameras = sorted({r["camera"] for r in self.rows})
        self.tiles = sorted({r["tile"] for r in self.rows})
        self.time_origin_ns = (
            min(int(r["timestamp_ns"]) for r in self.rows)
            if time_origin_ns is None
            else int(time_origin_ns)
        )
        self.knot_count_by_camera: dict[str, int] = {}
        for camera in self.cameras:
            if knot_count_by_camera is not None:
                self.knot_count_by_camera[camera] = int(knot_count_by_camera[camera])
                continue
            last_ns = max(int(r["timestamp_ns"]) for r in self.rows if r["camera"] == camera)
            self.knot_count_by_camera[camera] = curve_knot_count(
                (last_ns - self.time_origin_ns) / 1e9, self.knot_seconds
            )
        self.camera_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for camera in self.cameras:
            count = self.knot_count_by_camera[camera]
            self.camera_slices[camera] = (offset, count)
            offset += count
        self.curve_columns = offset
        self.tile_column = {tile: offset + k for k, tile in enumerate(self.tiles)}
        self.columns = offset + len(self.tiles)

        m = len(self.rows)
        self.A = np.zeros((m, self.columns), dtype=np.float64)
        self.y = np.array([float(r["log_gain"]) for r in self.rows], dtype=np.float64)
        self.time_s = np.array(
            [(int(r["timestamp_ns"]) - self.time_origin_ns) / 1e9 for r in self.rows]
        )
        for camera in self.cameras:
            start, count = self.camera_slices[camera]
            member = [k for k, r in enumerate(self.rows) if r["camera"] == camera]
            taps = curve_interpolation_weights(
                self.time_s[member], knot_seconds=self.knot_seconds, knot_count=count
            )
            for k, (lo, w) in zip(member, taps):
                self.A[k, start + lo] += 1.0 - w
                self.A[k, start + lo + 1] += w
        for k, r in enumerate(self.rows):
            self.A[k, self.tile_column[r["tile"]]] = 1.0

        # Anchor: the camera's mean log gain over its UNIQUE images (an image
        # trained by several tiles counts once, as it does in a tile's
        # ExposureCurve anchor).
        self.anchor_rows: dict[str, np.ndarray] = {}
        for camera in self.cameras:
            seen: dict[str, int] = {}
            for k, r in enumerate(self.rows):
                if r["camera"] == camera and r["image_id"] not in seen:
                    seen[r["image_id"]] = k
            row = np.zeros(self.columns)
            for k in seen.values():
                row[: self.curve_columns] += self.A[k, : self.curve_columns]
            row /= max(1, len(seen))
            self.anchor_rows[camera] = row

    def smoothness_matrix(self) -> np.ndarray:
        penalty = np.zeros((self.columns, self.columns))
        for start, count in self.camera_slices.values():
            for j in range(count - 1):
                d = np.zeros(self.columns)
                d[start + j] = -1.0
                d[start + j + 1] = 1.0
                penalty += np.outer(d, d)
        return penalty

    def anchor_matrix(self) -> np.ndarray:
        penalty = np.zeros((self.columns, self.columns))
        for row in self.anchor_rows.values():
            penalty += np.outer(row, row)
        return penalty

    def knots(self, x: np.ndarray) -> dict[str, list[float]]:
        return {
            camera: [float(v) for v in x[start : start + count]]
            for camera, (start, count) in self.camera_slices.items()
        }

    def tile_offsets(self, x: np.ndarray) -> dict[str, float]:
        return {tile: float(x[col]) for tile, col in self.tile_column.items()}


# -------------------------------------------------------------------- fit --


def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0.0:
        return np.ones_like(residual)
    absolute = np.abs(residual)
    return np.where(absolute <= delta, 1.0, delta / np.maximum(absolute, 1e-12))


def solve_penalised(
    design: CurveDesign,
    *,
    smoothness_weight: float,
    anchor_weight: float,
    huber_delta: float,
    iterations: int,
    row_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Penalised IRLS least squares; returns solution, weights, hat-trace edf."""
    if smoothness_weight < 0.0 or anchor_weight < 0.0:
        raise ValueError("penalty weights must be non-negative")
    if anchor_weight == 0.0 and len(design.cameras) > 0:
        # Curves and tile offsets share a common shift without the anchor.
        raise ValueError("anchor_weight must be positive: the curve mean is otherwise unidentifiable")
    A, y = design.A, design.y
    penalty = smoothness_weight * design.smoothness_matrix() + anchor_weight * design.anchor_matrix()
    base = np.ones(len(y)) if row_weights is None else np.asarray(row_weights, dtype=np.float64)
    w = base.copy()
    x = np.zeros(design.columns)
    for _ in range(max(1, iterations)):
        AtW = A.T * w
        normal = AtW @ A + penalty
        x = np.linalg.solve(normal, AtW @ y)
        residual = y - A @ x
        w = base * huber_weights(residual, huber_delta)
    AtW = A.T * w
    normal = AtW @ A + penalty
    residual = y - A @ x
    hat_diag = np.diag(np.linalg.solve(normal, AtW @ A))
    edf_by_camera = {
        camera: float(hat_diag[start : start + count].sum())
        for camera, (start, count) in design.camera_slices.items()
    }
    edf_curve = float(sum(edf_by_camera.values()))
    edf_total = float(hat_diag.sum())
    n_eff = float(w.sum())
    rss_w = float((w * residual**2).sum())
    gcv = n_eff * rss_w / max(1e-12, (n_eff - edf_total) ** 2)
    return {
        "x": x,
        "residual": residual,
        "weights": w,
        "edf_by_camera": edf_by_camera,
        "edf_curve": edf_curve,
        "edf_total": edf_total,
        "gcv": gcv,
        "rms_weighted": math.sqrt(rss_w / max(1e-12, n_eff)),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0, "rms": float("nan"), "mad": float("nan"), "p95_abs": float("nan"), "mean": float("nan")}
    return {
        "n": int(values.size),
        "rms": float(math.sqrt(float(np.mean(values**2)))),
        "mad": float(np.median(np.abs(values - np.median(values)))),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        "mean": float(values.mean()),
    }


def residual_report(design: CurveDesign, fit: Mapping[str, Any]) -> dict[str, Any]:
    residual = fit["residual"]
    x = fit["x"]
    rows = design.rows
    cams = np.array([r["camera"] for r in rows])
    tiles = np.array([r["tile"] for r in rows])
    envs = np.array([r["environment"] for r in rows])
    sat = np.array([r["saturated_clamp"] for r in rows], dtype=bool)
    # Baseline: tile offsets only (no curve) - the variance the curve explains.
    offsets_only = design.y.copy()
    for tile in design.tiles:
        member = tiles == tile
        offsets_only[member] -= design.y[member].mean()
    # Baseline: tile offset + per-camera constant.
    camera_constant = offsets_only.copy()
    for camera in design.cameras:
        member = cams == camera
        camera_constant[member] -= offsets_only[member].mean()
    report: dict[str, Any] = {
        "all": _stats(residual),
        "tile_offsets_only": _stats(offsets_only),
        "tile_offsets_plus_camera_constant": _stats(camera_constant),
        "r2_vs_tile_offsets_only": float(
            1.0 - float(np.sum(residual**2)) / max(1e-12, float(np.sum(offsets_only**2)))
        ),
        # The clamp-saturated rows carry the trainer's truncation, not the
        # curve's error; the unsaturated R2 is the honest explained fraction.
        "r2_vs_tile_offsets_only_unsaturated": float(
            1.0 - float(np.sum(residual[~sat] ** 2)) / max(1e-12, float(np.sum(offsets_only[~sat] ** 2)))
        ),
        "by_camera": {},
        "by_tile": {},
        "by_environment": {},
        "saturated_rows": {
            **_stats(residual[sat]),
            "mean_weight": float(fit["weights"][sat].mean()) if sat.any() else float("nan"),
        },
        "unsaturated_rows": _stats(residual[~sat]),
        "weights": {
            "mean": float(fit["weights"].mean()),
            "fraction_below_one": float((fit["weights"] < 1.0 - 1e-9).mean()),
        },
    }
    for camera in design.cameras:
        member = cams == camera
        report["by_camera"][camera] = {
            **_stats(residual[member]),
            "r2_vs_tile_offsets_only": float(
                1.0 - float(np.sum(residual[member] ** 2)) / max(1e-12, float(np.sum(offsets_only[member] ** 2)))
            ),
            "curve_min": float(min(design.knots(x)[camera])),
            "curve_max": float(max(design.knots(x)[camera])),
            "curve_mean_over_images": float(design.anchor_rows[camera] @ x),
        }
    for tile in design.tiles:
        report["by_tile"][tile] = _stats(residual[tiles == tile])
    for env in sorted(set(envs.tolist())):
        if env:
            report["by_environment"][env] = _stats(residual[envs == env])
    return report


def leave_one_tile_out(
    rows: Sequence[Mapping[str, Any]],
    *,
    knot_seconds: float,
    smoothness_weight: float,
    anchor_weight: float,
    huber_delta: float,
    iterations: int,
    time_origin_ns: int,
    knot_count_by_camera: Mapping[str, int],
) -> dict[str, Any]:
    """Fit on all tiles but one; predict the held-out tile's gains up to its
    own free offset.  Says whether the curve is a property of the capture
    (transfers across tiles) or of the tiles it was fitted on."""
    tiles = sorted({r["tile"] for r in rows})
    out: dict[str, Any] = {}
    for held in tiles:
        train = [r for r in rows if r["tile"] != held]
        test = [r for r in rows if r["tile"] == held]
        if not train or not test:
            continue
        design = CurveDesign(
            train,
            knot_seconds=knot_seconds,
            time_origin_ns=time_origin_ns,
            knot_count_by_camera=knot_count_by_camera,
        )
        fit = solve_penalised(
            design,
            smoothness_weight=smoothness_weight,
            anchor_weight=anchor_weight,
            huber_delta=huber_delta,
            iterations=iterations,
        )
        knots = design.knots(fit["x"])
        predicted = np.array(
            [
                evaluate_curve(
                    knots[r["camera"]],
                    [(int(r["timestamp_ns"]) - time_origin_ns) / 1e9],
                    knot_seconds=knot_seconds,
                )[0]
                for r in test
            ]
        )
        observed = np.array([float(r["log_gain"]) for r in test])
        # The held-out tile's own brightness offset is free (median, robust).
        offset = float(np.median(observed - predicted))
        residual = observed - predicted - offset
        baseline = observed - np.median(observed)
        cams = np.array([r["camera"] for r in test])
        sat = np.array([int(r.get("saturated_clamp", 0)) for r in test], dtype=bool)
        out[held] = {
            "held_out_rows": len(test),
            "offset": offset,
            "residual": _stats(residual),
            "residual_unsaturated": _stats(residual[~sat]),
            "offset_only_baseline": _stats(baseline),
            "r2_vs_offset_only": float(
                1.0 - float(np.sum(residual**2)) / max(1e-12, float(np.sum(baseline**2)))
            ),
            "r2_vs_offset_only_unsaturated": float(
                1.0 - float(np.sum(residual[~sat] ** 2)) / max(1e-12, float(np.sum(baseline[~sat] ** 2)))
            ),
            "by_camera": {
                camera: _stats(residual[cams == camera]) for camera in sorted(set(cams.tolist()))
            },
        }
    return out


def fit_exposure_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    knot_seconds: float,
    smoothness_weight: float,
    anchor_weight: float,
    huber_delta: float = 0.15,
    iterations: int = 10,
    max_abs_log_gain: float = LN2,
    sweep: Sequence[float] = DEFAULT_SWEEP,
    holdout: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Returns (curve payload with provenance, per-row records)."""
    design = CurveDesign(rows, knot_seconds=knot_seconds)
    fit = solve_penalised(
        design,
        smoothness_weight=smoothness_weight,
        anchor_weight=anchor_weight,
        huber_delta=huber_delta,
        iterations=iterations,
    )
    x = fit["x"]
    knots = design.knots(x)
    tile_offsets = design.tile_offsets(x)
    if max(abs(v) for values in knots.values() for v in values) > max_abs_log_gain:
        raise ValueError("fitted curve exceeds max_abs_log_gain; the fit is not usable frozen")

    sweep_report = []
    for weight in sweep:
        trial = solve_penalised(
            design,
            smoothness_weight=float(weight),
            anchor_weight=anchor_weight,
            huber_delta=huber_delta,
            iterations=iterations,
        )
        sweep_report.append(
            {
                "smoothness_weight": float(weight),
                "rms_weighted": trial["rms_weighted"],
                "rms": float(math.sqrt(float(np.mean(trial["residual"] ** 2)))),
                "edf_curve": trial["edf_curve"],
                "edf_by_camera": trial["edf_by_camera"],
                "gcv": trial["gcv"],
            }
        )
    holdout_report = (
        leave_one_tile_out(
            rows,
            knot_seconds=knot_seconds,
            smoothness_weight=smoothness_weight,
            anchor_weight=anchor_weight,
            huber_delta=huber_delta,
            iterations=iterations,
            time_origin_ns=design.time_origin_ns,
            knot_count_by_camera=design.knot_count_by_camera,
        )
        if holdout
        else {}
    )
    provenance = {
        "generator": "tools/fit_exposure_curve.py",
        "model": "log_gain[image, tile] = curve[camera](t_image) + offset[tile]; "
        "piecewise-linear knots every knot_seconds on a shared origin; "
        "Huber IRLS on the applied (clamped) log gain",
        "observations": len(rows),
        "unique_images": len({r["image_id"] for r in rows}),
        "tiles": design.tiles,
        "cameras": design.cameras,
        "smoothness_weight": float(smoothness_weight),
        "anchor_weight": float(anchor_weight),
        "huber_delta": float(huber_delta),
        "irls_iterations": int(iterations),
        "time_span_s": float(design.time_s.max()),
        "knot_count_by_camera": design.knot_count_by_camera,
        "parameter_count_curve": design.curve_columns,
        "tile_offsets": tile_offsets,
        "tile_offset_gains": {t: math.exp(v) for t, v in tile_offsets.items()},
        "effective_dof": {
            "curve_total": fit["edf_curve"],
            "by_camera": fit["edf_by_camera"],
            "including_tile_offsets": fit["edf_total"],
        },
        "gcv": fit["gcv"],
        "residuals": residual_report(design, fit),
        "smoothness_sweep": sweep_report,
        "leave_one_tile_out": holdout_report,
    }
    payload = build_curve_payload(
        knot_seconds=knot_seconds,
        time_origin_ns=design.time_origin_ns,
        cameras=knots,
        max_abs_log_gain=max_abs_log_gain,
        provenance=provenance,
    )
    fitted_curve = (design.A[:, : design.curve_columns] @ x[: design.curve_columns])
    records = []
    for k, r in enumerate(design.rows):
        records.append(
            {
                "image_id": r["image_id"],
                "camera": r["camera"],
                "tile": r["tile"],
                "environment": r["environment"],
                "timestamp_ns": int(r["timestamp_ns"]),
                "time_s": float(design.time_s[k]),
                "log_gain": float(r["log_gain"]),
                "log_gain_raw": float(r["log_gain_raw"]),
                "saturated_clamp": int(r["saturated_clamp"]),
                "tile_offset": float(tile_offsets[r["tile"]]),
                "curve_log_gain": float(fitted_curve[k]),
                "fitted_log_gain": float(fitted_curve[k] + tile_offsets[r["tile"]]),
                "residual": float(fit["residual"][k]),
                "huber_weight": float(fit["weights"][k]),
            }
        )
    return payload, records


# ------------------------------------------------------------------- main --


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("nothing to write")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--consistency-csv", type=Path, required=True)
    parser.add_argument("--knot-seconds", type=float, default=10.0)
    parser.add_argument("--smoothness-weight", type=float, default=3.0)
    parser.add_argument("--anchor-weight", type=float, default=1e4)
    parser.add_argument("--huber-delta", type=float, default=0.15)
    parser.add_argument("--irls-iterations", type=int, default=10)
    parser.add_argument("--no-holdout", action="store_true", help="skip the leave-one-tile-out check")
    parser.add_argument("--out-curve", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = read_gain_rows(args.consistency_csv)
    payload, records = fit_exposure_curve(
        rows,
        knot_seconds=args.knot_seconds,
        smoothness_weight=args.smoothness_weight,
        anchor_weight=args.anchor_weight,
        huber_delta=args.huber_delta,
        iterations=args.irls_iterations,
        holdout=not args.no_holdout,
    )
    payload["provenance"]["source_csv"] = str(args.consistency_csv)
    payload["provenance"]["source_csv_sha256"] = hashlib.sha256(
        args.consistency_csv.read_bytes()
    ).hexdigest()
    args.out_curve.parent.mkdir(parents=True, exist_ok=True)
    args.out_curve.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    write_csv(args.out_csv, records)

    prov = payload["provenance"]
    res = prov["residuals"]
    print(f"rows {prov['observations']}  unique images {prov['unique_images']}  span {prov['time_span_s']:.1f} s")
    print(f"knots per camera {prov['knot_count_by_camera']}  curve params {prov['parameter_count_curve']}")
    print(f"edf curve {prov['effective_dof']['curve_total']:.1f}  by camera {prov['effective_dof']['by_camera']}")
    print(f"tile offsets {prov['tile_offsets']}")
    print(
        f"residual rms {res['all']['rms']:.4f}  (tile offsets only {res['tile_offsets_only']['rms']:.4f}, "
        f"+camera constant {res['tile_offsets_plus_camera_constant']['rms']:.4f})  "
        f"R2 vs offsets {res['r2_vs_tile_offsets_only']:.3f} (unsaturated {res['r2_vs_tile_offsets_only_unsaturated']:.3f})"
    )
    for camera, entry in res["by_camera"].items():
        print(f"  {camera}: rms {entry['rms']:.4f} R2 {entry['r2_vs_tile_offsets_only']:.3f} curve [{entry['curve_min']:+.3f}, {entry['curve_max']:+.3f}] mean {entry['curve_mean_over_images']:+.5f}")
    for env, entry in res["by_environment"].items():
        print(f"  {env}: rms {entry['rms']:.4f} n {entry['n']}")
    print("sweep:")
    for entry in prov["smoothness_sweep"]:
        print(f"  lambda {entry['smoothness_weight']:>7}: rms {entry['rms']:.4f} edf {entry['edf_curve']:.1f} gcv {entry['gcv']:.5f}")
    for tile, entry in prov["leave_one_tile_out"].items():
        print(f"holdout {tile}: rms {entry['residual']['rms']:.4f} vs offset-only {entry['offset_only_baseline']['rms']:.4f} R2 {entry['r2_vs_offset_only']:.3f} (unsaturated {entry['r2_vs_offset_only_unsaturated']:.3f})")
    print(f"wrote {args.out_curve} and {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
