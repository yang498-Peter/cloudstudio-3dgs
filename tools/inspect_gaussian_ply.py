"""Report what a Gaussian-splat PLY actually contains, ours or anyone else's.

Written to compare against a competitor's output. Before any rendering
comparison is possible, three things have to be established, and all three are
answerable from the file alone:

  format       Is it a 3DGS PLY (f_dc/f_rest/opacity/scale/rot) or a plain
               point cloud? Only the former can be rendered for a like-for-like
               picture comparison.
  structure    How many Gaussians, and how big? This campaign's own model sits
               at 390,901 Gaussians with a 6.4 cm median longest axis, and the
               measured trend is that more and smaller tracked WORSE agreement
               with the photograph - so what a better-looking model does here
               is a direct test of that finding.
  frame        Where is it in space? A competitor that ran its own SfM lands in
               a different coordinate system, and the extent/centre reported
               here is what says whether the two can be compared directly or
               need alignment first.

Scale convention matters and is reported both ways: 3DGS stores log-scale, so
the metric size is exp(scale). A file storing linear scales would read as
absurdly large under exp, which the sanity line flags rather than silently
mis-reporting.

    python tools/inspect_gaussian_ply.py --ply competitor.ply
    python tools/inspect_gaussian_ply.py --ply a.ply --compare-with b.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "gaussian-ply-inspect-1.0"
# The INRIA 3DGS PLY layout every consumer of that format writes.
GAUSSIAN_FIELDS = ("opacity", "scale_0", "rot_0", "f_dc_0")


def _read_ply(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    # The repository's own reader, so a competitor's file needs no new
    # dependency and is parsed by exactly the code our own exports go through.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gaussian_health import read_ply_records

    records = read_ply_records(path)
    names = list(records.dtype.names)
    arrays = {name: np.asarray(records[name], dtype=np.float64) for name in names}
    meta = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "count": len(records),
        "fields": names,
    }
    return arrays, meta


def _describe(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {}
    return {
        "min": float(finite.min()),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
    }


def inspect(path: Path) -> dict[str, object]:
    arrays, meta = _read_ply(path)
    present = [f for f in GAUSSIAN_FIELDS if f in arrays]
    meta["is_gaussian_splat"] = len(present) == len(GAUSSIAN_FIELDS)
    meta["gaussian_fields_present"] = present

    xyz = np.stack([arrays[axis] for axis in ("x", "y", "z")], axis=1)
    meta["centre_m"] = [float(v) for v in np.median(xyz, axis=0)]
    meta["extent_m"] = [float(v) for v in (xyz.max(axis=0) - xyz.min(axis=0))]
    # The same p95 radius the trainer uses for scene_scale, so the two numbers
    # are directly comparable when deciding whether frames match.
    radius = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)
    meta["scene_extent_p95_m"] = float(np.percentile(radius, 95))

    if not meta["is_gaussian_splat"]:
        return meta

    scale_names = sorted(n for n in arrays if n.startswith("scale_"))
    raw = np.stack([arrays[n] for n in scale_names], axis=1)
    linear = np.exp(raw)
    longest = linear.max(axis=1)
    shortest = linear.min(axis=1)
    meta["scale_fields"] = scale_names
    meta["longest_axis_m"] = _describe(longest)
    meta["shortest_axis_m"] = _describe(shortest)
    meta["aspect_ratio"] = _describe(longest / np.maximum(shortest, 1e-12))
    meta["raw_scale_stored"] = _describe(raw.reshape(-1))
    # A file storing LINEAR scales would give a metre-scale median here after
    # exp(); say so rather than reporting a wrong size.
    meta["scale_looks_logarithmic"] = bool(np.median(raw) < 0.0)

    opacity = arrays["opacity"]
    # 3DGS stores logit opacity; sigmoid brings it back to [0, 1].
    meta["opacity_stored"] = _describe(opacity)
    meta["opacity_activated"] = _describe(1.0 / (1.0 + np.exp(-opacity)))

    sh_rest = sorted(n for n in arrays if n.startswith("f_rest_"))
    meta["sh_rest_coefficients"] = len(sh_rest)
    # 3 colour channels x ((degree+1)^2 - 1) bands
    meta["sh_degree"] = {0: 0, 9: 1, 24: 2, 45: 3}.get(len(sh_rest), None)
    return meta


def _line(label: str, stats: dict[str, float], unit: str = "m") -> str:
    if not stats:
        return f"{label:<22} (none)"
    return (f"{label:<22} p05 {stats['p05']:.4f}  p50 {stats['p50']:.4f}  "
            f"p95 {stats['p95']:.4f}  max {stats['max']:.4f} {unit}")


def report(meta: dict[str, object]) -> None:
    print(f"\n{'=' * 70}\n{meta['path']}\n{'=' * 70}")
    print(f"{'gaussians':<22} {meta['count']:,}")
    print(f"{'file size':<22} {meta['size_bytes'] / 1e6:.1f} MB")
    if not meta["is_gaussian_splat"]:
        print(f"{'format':<22} NOT a 3DGS PLY - cannot be rendered for comparison")
        print(f"{'fields':<22} {', '.join(meta['fields'][:12])}"
              + (" ..." if len(meta["fields"]) > 12 else ""))
        print(f"{'present of expected':<22} {meta['gaussian_fields_present'] or 'none'}")
    else:
        print(f"{'format':<22} 3DGS PLY, SH degree "
              f"{meta['sh_degree']} ({meta['sh_rest_coefficients']} rest coeffs)")
        print(_line("longest axis", meta["longest_axis_m"]))
        print(_line("shortest axis", meta["shortest_axis_m"]))
        print(_line("aspect ratio", meta["aspect_ratio"], unit=""))
        print(_line("opacity (activated)", meta["opacity_activated"], unit=""))
        if not meta["scale_looks_logarithmic"]:
            print("WARNING: stored scales do not look logarithmic; sizes above "
                  "assume exp() and may be wrong for this file")
    centre = ", ".join(f"{v:.2f}" for v in meta["centre_m"])
    extent = " x ".join(f"{v:.1f}" for v in meta["extent_m"])
    print(f"{'centre (median xyz)':<22} ({centre}) m")
    print(f"{'bounding extent':<22} {extent} m")
    print(f"{'scene extent p95':<22} {meta['scene_extent_p95_m']:.2f} m")


def compare(a: dict[str, object], b: dict[str, object]) -> None:
    print(f"\n{'=' * 70}\ncomparison\n{'=' * 70}")
    print(f"{'':<22} {'A':>16} {'B':>16}")
    print(f"{'gaussians':<22} {a['count']:>16,} {b['count']:>16,}")
    if a["is_gaussian_splat"] and b["is_gaussian_splat"]:
        for label, key in (("longest axis p50 (m)", "longest_axis_m"),
                           ("shortest axis p50 (m)", "shortest_axis_m"),
                           ("aspect p50", "aspect_ratio")):
            print(f"{label:<22} {a[key]['p50']:>16.4f} {b[key]['p50']:>16.4f}")
    print(f"{'scene extent p95 (m)':<22} {a['scene_extent_p95_m']:>16.2f} "
          f"{b['scene_extent_p95_m']:>16.2f}")
    ratio = a["scene_extent_p95_m"] / max(b["scene_extent_p95_m"], 1e-9)
    offset = np.linalg.norm(np.array(a["centre_m"]) - np.array(b["centre_m"]))
    print(f"\nextent ratio A/B {ratio:.3f}, centre offset {offset:.2f} m")
    if abs(ratio - 1.0) < 0.02 and offset < 1.0:
        print("-> same frame and scale: renderable views transfer directly")
    elif abs(ratio - 1.0) < 0.02:
        print("-> same scale, shifted origin: a translation aligns them")
    else:
        print("-> different scale or frame: alignment required before any "
              "view-by-view picture comparison means anything")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--compare-with", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.ply.exists():
        print(f"not found: {args.ply}", file=sys.stderr)
        return 2
    first = inspect(args.ply)
    report(first)
    payload = {"schema_version": SCHEMA_VERSION, "a": first}

    if args.compare_with:
        if not args.compare_with.exists():
            print(f"not found: {args.compare_with}", file=sys.stderr)
            return 2
        second = inspect(args.compare_with)
        report(second)
        compare(first, second)
        payload["b"] = second

    if args.json:
        import json

        args.json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
