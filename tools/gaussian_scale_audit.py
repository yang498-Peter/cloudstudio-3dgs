"""Audit Gaussian extent - especially the axis nothing else measures.

`tools/gaussian_health.py` reports wall thickness, shortest-axis alignment and
floater distance. All three describe the THIN direction of a Gaussian, and on
2026-08-25 that turned out to be a blind spot with real consequences: LiDAR
normal alignment pulled the shortest axis from 3.45cm to 1.23cm while the
longest stayed at 8.5cm, so blobs became wide flat discs. Every health metric
improved, and the renders got no sharper, because what a Gaussian smears across
the image is its WIDEST axis, which nothing was measuring or constraining.

This reports all three axes, the aspect ratio, and the widest axis converted
into the pixels it covers at its own distance from the scene - the form in which
it can be compared against the size of the things it is meant to depict. Gravel
stones in the UK scene are about 3cm, roughly 5px at 5m and fl=778 px/rad, so a
Gaussian wider than that cannot represent one however many of them there are.

Run it beside any change that claims to control Gaussian size. It catches the
failure mode where a knob moves a metric without moving the mechanism: switching
`geometry_regularization` on at its defaults GREW the longest axis to 16.37cm,
and without this audit the accompanying drop in sharpness would have been read
as "scale control does not help" rather than "this control grew them".

Usage::

    python tools/gaussian_scale_audit.py --checkpoint best_golden.pt \\
        [--focal-px-per-rad 778] [--target-px 5] [--output audit.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "gaussian-scale-audit-1.0"
PERCENTILES = (5, 25, 50, 75, 95)
DEFAULT_FOCAL_PX_PER_RAD = 778.0
DEFAULT_TARGET_PX = 5.0
VISIBLE_OPACITY = 0.1


def audit(checkpoint: Path, focal_px_per_rad: float, target_px: float) -> dict:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = payload["params"]
    scales = torch.exp(params["scales"]).numpy()
    opacity = torch.sigmoid(params["opacities"]).numpy().ravel()
    means = params["means"].numpy()

    visible = opacity > VISIBLE_OPACITY
    if not visible.any():
        raise ValueError("no visible gaussians above the opacity floor")
    ordered = np.sort(scales[visible], axis=1)
    shortest, middle, longest = ordered[:, 0], ordered[:, 1], ordered[:, 2]
    aspect = longest / np.maximum(shortest, 1e-9)

    # Distance from the scene centroid stands in for typical viewing distance;
    # it is crude but consistent across models, which is what matters here.
    centre = np.median(means, axis=0)
    distance = np.clip(np.linalg.norm(means[visible] - centre, axis=1), 0.5, None)
    footprint_px = focal_px_per_rad * longest / distance

    def percentiles(values):
        return {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "total_gaussians": int(len(opacity)),
        "visible_gaussians": int(visible.sum()),
        "shortest_axis_m": percentiles(shortest),
        "middle_axis_m": percentiles(middle),
        "longest_axis_m": percentiles(longest),
        "aspect_ratio": percentiles(aspect),
        "footprint_px": percentiles(footprint_px),
        "fraction_wider_than_target": float((footprint_px > target_px).mean()),
        "target_px": target_px,
        "focal_px_per_rad": focal_px_per_rad,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--focal-px-per-rad", type=float,
                        default=DEFAULT_FOCAL_PX_PER_RAD)
    parser.add_argument("--target-px", type=float, default=DEFAULT_TARGET_PX,
                        help="size of the smallest feature that must be representable")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = audit(args.checkpoint, args.focal_px_per_rad, args.target_px)

    print(f"Gaussian scale audit ({SCHEMA_VERSION})")
    print(f"visible {report['visible_gaussians']:,} of {report['total_gaussians']:,}")
    header = "".join(f"{p:>10}%" for p in PERCENTILES)
    print(f"\n  {'metric':<16}{header}")
    for label, key, scale in (("shortest axis", "shortest_axis_m", 100.0),
                              ("middle axis", "middle_axis_m", 100.0),
                              ("longest axis", "longest_axis_m", 100.0)):
        row = "".join(f"{report[key][f'p{p}'] * scale:>10.2f}" for p in PERCENTILES)
        print(f"  {label:<16}{row}   (cm)")
    print(f"  {'aspect L/S':<16}" +
          "".join(f"{report['aspect_ratio'][f'p{p}']:>10.1f}" for p in PERCENTILES))
    print(f"  {'width':<16}" +
          "".join(f"{report['footprint_px'][f'p{p}']:>10.1f}" for p in PERCENTILES) +
          "   (px)")
    print(f"\n{report['fraction_wider_than_target'] * 100:.1f}% of visible gaussians "
          f"are wider than {report['target_px']:.0f}px - the size of the smallest "
          f"feature to be represented")

    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
