"""Offline QA for the LiDAR surface field, and A/B of the two floater criteria.

Two things are reported:

1. **Surface field self-statistics** — quantile distributions of planarity,
   roughness, local spacing and confidence over the LiDAR cloud. This is how you
   see what the cloud actually supports before trusting any gate tuned on it.

2. **Criterion comparison** (needs ``--checkpoint``) — the current floater
   criterion ("nearest LiDAR point is farther than tau") side by side with the
   surface-field criterion ("normal distance to the surface is farther than
   tau", plus the density-adaptive support weight). The headline number is the
   **disagreement set**: how many Gaussians the nearest-point criterion calls
   floaters while the surface criterion says they are flush against a surface.
   Those are the false positives that scan-line spacing, voxelization and
   grazing incidence manufacture out of nothing.

   Note ``d_perp <= euclidean`` holds identically, so the reverse disagreement
   at a fixed tau is always empty — it is reported anyway as an invariant check.
   The support-weight criterion is *not* bounded that way: it can flag a
   Gaussian that hugs a low-confidence (vegetation / clutter / edge) neighborhood
   which the nearest-point distance happily accepts.

   **Caveat, reported explicitly.** ``d_perp`` measures distance to the local
   plane, which is unbounded — a Gaussian metres past the *edge* of a scanned
   wall still projects onto that wall's plane and scores a small ``d_perp``.
   The ``extrapolated`` column counts disagreements whose ``d_tangent`` exceeds
   ``EXTRAPOLATION_SPACING_FACTOR`` local spacings, i.e. those that sit outside
   the sampled patch rather than between two of its samples. Any consumer
   replacing the nearest-point gate must gate on ``d_tangent`` (or the support
   weight, which folds in confidence) as well as ``d_perp``.

Usage::

    python tools/surface_field_qa.py --lidar-ply sparse_pc.ply \\
        [--checkpoint ckpt.pt] [--knn 24] [--output surface_field_qa.json]

CPU only: the checkpoint is loaded with ``map_location="cpu"``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.geometry.lidar_surface_field import (  # noqa: E402
    DEFAULT_KNN,
    build_surface_field,
    support_weight,
)
from tools.gaussian_health import (  # noqa: E402
    FLOATER_DIST_THRESHOLDS_M,
    FLOATER_MIN_OPACITY,
    _sigmoid,
    _to_numpy,
    read_ply_xyz,
)

SCHEMA_VERSION = "surface-field-qa-1.0"

FIELD_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
QUERY_PERCENTILES = (50, 75, 90, 95, 99, 99.9)
DEFAULT_SIGMA_PERP_FACTOR = 1.0
DEFAULT_SUPPORT_GATE = 0.1
# Beyond this many local spacings of tangential drift a query is no longer
# "between two samples" but off the edge of the sampled surface patch.
EXTRAPOLATION_SPACING_FACTOR = 3.0


def _percentiles(values: np.ndarray, points=FIELD_PERCENTILES) -> dict:
    if values.size == 0:
        return {f"p{str(point).replace('.', '_')}": None for point in points}
    computed = np.percentile(values, list(points))
    return {
        f"p{str(point).replace('.', '_')}": float(value)
        for point, value in zip(points, computed)
    }


def _distribution(values: np.ndarray, points=FIELD_PERCENTILES) -> dict:
    entry = _percentiles(values, points)
    entry["min"] = float(values.min()) if values.size else None
    entry["max"] = float(values.max()) if values.size else None
    entry["mean"] = float(values.mean()) if values.size else None
    return entry


def field_statistics(field) -> dict:
    """Quantile summary of the surface field itself."""
    return {
        "point_count": int(len(field)),
        "knn": int(field.knn),
        "neighbor_radius_m": float(field.neighbor_radius_m),
        "planarity": _distribution(field.planarity.astype(np.float64)),
        "roughness_m": _distribution(field.roughness.astype(np.float64)),
        "local_spacing_m": _distribution(field.local_spacing.astype(np.float64)),
        "confidence": _distribution(field.confidence.astype(np.float64)),
    }


def compare_floater_criteria(
    field,
    means: np.ndarray,
    opacity: np.ndarray,
    *,
    thresholds=FLOATER_DIST_THRESHOLDS_M,
    min_opacity: float = FLOATER_MIN_OPACITY,
    sigma_perp_factor: float = DEFAULT_SIGMA_PERP_FACTOR,
    support_gate: float = DEFAULT_SUPPORT_GATE,
) -> dict:
    """Nearest-point vs surface-normal floater criteria on the same Gaussians.

    Only *visible* Gaussians (``opacity > min_opacity``) participate, matching
    ``tools.gaussian_health``. Returns per-threshold counts plus the size and
    character of the disagreement sets.
    """
    visible = opacity > float(min_opacity)
    result = {
        "min_opacity": float(min_opacity),
        "sigma_perp_factor": float(sigma_perp_factor),
        "support_gate": float(support_gate),
        "gaussian_count": int(len(means)),
        "visible_count": int(visible.sum()),
    }
    if not visible.any():
        result["distances"] = {}
        result["thresholds"] = []
        return result

    query = field.query(means[visible])
    weights = support_weight(query, sigma_perp_factor=sigma_perp_factor)

    result["distances"] = {
        "euclidean_m": _distribution(query.euclidean, QUERY_PERCENTILES),
        "d_perp_m": _distribution(query.d_perp, QUERY_PERCENTILES),
        "d_tangent_m": _distribution(query.d_tangent, QUERY_PERCENTILES),
        "support_weight": _distribution(weights, QUERY_PERCENTILES),
        "anchor_confidence": _distribution(query.confidence, QUERY_PERCENTILES),
        "anchor_local_spacing_m": _distribution(
            query.local_spacing, QUERY_PERCENTILES
        ),
    }

    support_floater = weights < float(support_gate)
    # Outside the sampled patch rather than between two of its samples: d_perp
    # is a plane distance and does not know where the plane stops.
    extrapolated = query.d_tangent > (
        EXTRAPOLATION_SPACING_FACTOR * query.local_spacing
    )
    result["extrapolation_spacing_factor"] = float(EXTRAPOLATION_SPACING_FACTOR)
    entries = []
    for threshold in thresholds:
        nearest = query.euclidean > float(threshold)
        perp = query.d_perp > float(threshold)
        nearest_only = nearest & ~perp
        perp_only = perp & ~nearest  # invariant: always empty (d_perp <= euclidean)
        entry = {
            "threshold_m": float(threshold),
            "nearest_point_floaters": int(nearest.sum()),
            "surface_normal_floaters": int(perp.sum()),
            "agreement": int((nearest & perp).sum()),
            "nearest_only": {
                "count": int(nearest_only.sum()),
                "fraction_of_nearest": (
                    float(nearest_only.sum() / max(int(nearest.sum()), 1))
                    if nearest.any()
                    else None
                ),
                "median_d_perp_m": (
                    float(np.median(query.d_perp[nearest_only]))
                    if nearest_only.any()
                    else None
                ),
                "median_d_tangent_m": (
                    float(np.median(query.d_tangent[nearest_only]))
                    if nearest_only.any()
                    else None
                ),
                "median_support_weight": (
                    float(np.median(weights[nearest_only]))
                    if nearest_only.any()
                    else None
                ),
                "still_floater_by_support": int((nearest_only & support_floater).sum()),
                "extrapolated_past_patch_edge": int((nearest_only & extrapolated).sum()),
            },
            "perp_only_count": int(perp_only.sum()),
            "support_criterion": {
                "floaters": int(support_floater.sum()),
                "nearest_says_floater_support_says_flush": int(
                    (nearest & ~support_floater).sum()
                ),
                "nearest_says_flush_support_says_floater": int(
                    (~nearest & support_floater).sum()
                ),
            },
        }
        entries.append(entry)
    result["thresholds"] = entries
    return result


def _format(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_report(report: dict) -> str:
    lines = [f"Surface Field QA ({report['schema_version']})"]
    field = report["field"]
    lines.append(
        f"lidar_points={field['point_count']}  knn={field['knn']}  "
        f"neighbor_radius={field['neighbor_radius_m']:.4f} m  "
        f"build={report['build_seconds']:.2f} s"
    )
    lines.append("")
    lines.append("[surface field] quantiles")
    header = "  " + "metric".ljust(18) + "".join(
        f"p{str(point).replace('.', '_')}".rjust(11) for point in FIELD_PERCENTILES
    )
    lines.append(header)
    for key in ("planarity", "roughness_m", "local_spacing_m", "confidence"):
        row = field[key]
        lines.append(
            "  "
            + key.ljust(18)
            + "".join(
                _format(row[f"p{str(point).replace('.', '_')}"]).rjust(11)
                for point in FIELD_PERCENTILES
            )
        )

    comparison = report.get("comparison")
    if not comparison:
        lines.append("")
        lines.append("[criteria] skipped (no --checkpoint given)")
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"[gaussians] total={comparison['gaussian_count']}  "
        f"visible(opacity>{comparison['min_opacity']})={comparison['visible_count']}"
    )
    lines.append("")
    lines.append("[query distances] quantiles over visible gaussians")
    header = "  " + "metric".ljust(24) + "".join(
        f"p{str(point).replace('.', '_')}".rjust(11) for point in QUERY_PERCENTILES
    )
    lines.append(header)
    for key, row in comparison.get("distances", {}).items():
        lines.append(
            "  "
            + key.ljust(24)
            + "".join(
                _format(row[f"p{str(point).replace('.', '_')}"]).rjust(11)
                for point in QUERY_PERCENTILES
            )
        )

    lines.append("")
    lines.append("[criteria A/B] nearest-point distance vs surface normal distance")
    lines.append(
        "  "
        + "tau(m)".rjust(7)
        + "nearest".rjust(12)
        + "d_perp".rjust(12)
        + "both".rjust(12)
        + "nearest_only".rjust(14)
        + "perp_only".rjust(11)
        + "  (nearest_only = old criterion's false floaters)"
    )
    for entry in comparison["thresholds"]:
        lines.append(
            "  "
            + f"{entry['threshold_m']:g}".rjust(7)
            + str(entry["nearest_point_floaters"]).rjust(12)
            + str(entry["surface_normal_floaters"]).rjust(12)
            + str(entry["agreement"]).rjust(12)
            + str(entry["nearest_only"]["count"]).rjust(14)
            + str(entry["perp_only_count"]).rjust(11)
        )
    lines.append("")
    lines.append("[disagreement detail] gaussians the old criterion misclassifies")
    for entry in comparison["thresholds"]:
        only = entry["nearest_only"]
        lines.append(
            f"  tau={entry['threshold_m']:g} m: count={only['count']} "
            f"({_format(only['fraction_of_nearest'])} of nearest-point floaters)  "
            f"median d_perp={_format(only['median_d_perp_m'])} m  "
            f"median d_tangent={_format(only['median_d_tangent_m'])} m  "
            f"median support={_format(only['median_support_weight'])}"
        )
        lines.append(
            f"           of those, still floaters under the support gate "
            f"(<{comparison['support_gate']}): {only['still_floater_by_support']}"
            f"; extrapolated past the sampled patch edge "
            f"(d_tangent > {EXTRAPOLATION_SPACING_FACTOR:g} x local spacing): "
            f"{only['extrapolated_past_patch_edge']}"
        )
    lines.append("")
    support = comparison["thresholds"][0]["support_criterion"]
    lines.append(
        f"[support criterion] threshold-free floaters "
        f"(support_weight < {comparison['support_gate']}): {support['floaters']}"
    )
    for entry in comparison["thresholds"]:
        sup = entry["support_criterion"]
        lines.append(
            f"  vs nearest tau={entry['threshold_m']:g} m: "
            f"nearest-floater/support-flush={sup['nearest_says_floater_support_says_flush']}  "
            f"nearest-flush/support-floater={sup['nearest_says_flush_support_says_floater']}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LiDAR surface field statistics and nearest-point vs surface-normal "
            "floater criterion comparison."
        )
    )
    parser.add_argument("--lidar-ply", required=True, type=Path,
                        help="LiDAR initialization PLY (binary little-endian or ascii)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional torch checkpoint containing a 'params' dict")
    parser.add_argument("--knn", type=int, default=DEFAULT_KNN,
                        help=f"KNN neighborhood size (default {DEFAULT_KNN})")
    parser.add_argument("--sigma-perp-factor", type=float,
                        default=DEFAULT_SIGMA_PERP_FACTOR,
                        help="support weight tolerance multiplier (default 1.0)")
    parser.add_argument("--support-gate", type=float, default=DEFAULT_SUPPORT_GATE,
                        help="support weight below which a gaussian is a floater")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the global-spacing subsample (default 0)")
    parser.add_argument("--save-field", type=Path, default=None,
                        help="optional npz path to persist the built surface field")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional path to write the JSON report")
    args = parser.parse_args(argv)

    lidar_xyz = read_ply_xyz(args.lidar_ply)
    started = time.perf_counter()
    field = build_surface_field(lidar_xyz, knn=args.knn, seed=args.seed)
    build_seconds = time.perf_counter() - started

    report = {
        "schema_version": SCHEMA_VERSION,
        "lidar_ply": str(args.lidar_ply),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "build_seconds": float(build_seconds),
        "field": field_statistics(field),
        "comparison": None,
    }

    if args.save_field is not None:
        args.save_field.parent.mkdir(parents=True, exist_ok=True)
        field.save(args.save_field)

    if args.checkpoint is not None:
        import torch  # deferred so the numeric path stays importable without torch

        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        params = checkpoint["params"] if "params" in checkpoint else checkpoint
        means = _to_numpy(params["means"]).reshape(-1, 3)
        opacity = _sigmoid(_to_numpy(params["opacities"]).reshape(-1))
        report["comparison"] = compare_floater_criteria(
            field,
            means,
            opacity,
            sigma_perp_factor=args.sigma_perp_factor,
            support_gate=args.support_gate,
        )

    print(render_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
