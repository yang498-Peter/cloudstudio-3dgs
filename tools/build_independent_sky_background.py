"""Build signed Face4 sky/background evidence and a standalone SH1 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.sky_background import (
    IndependentSkyConfig,
    SkyEvidenceConfig,
    build_independent_sky_initialization,
    build_sky_evidence_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--face-root", required=True, type=Path)
    parser.add_argument("--mono-manifest", required=True, type=Path)
    parser.add_argument("--mono-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--far-range-m", type=float, default=30.0)
    parser.add_argument("--minimum-world-z", type=float, default=0.0)
    parser.add_argument("--minimum-view-fraction", type=float, default=0.01)
    parser.add_argument("--sky-count", type=int, default=100_000)
    parser.add_argument("--sky-radius-m", type=float, default=100.0)
    parser.add_argument("--sky-scale-m", type=float, default=0.85)
    parser.add_argument("--sky-opacity", type=float, default=0.02)
    parser.add_argument("--skip-initialization", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    evidence = build_sky_evidence_cache(
        args.face_manifest,
        args.face_root,
        args.mono_manifest,
        args.mono_root,
        args.output,
        config=SkyEvidenceConfig(
            far_aligned_range_m=args.far_range_m,
            minimum_world_z_direction=args.minimum_world_z,
            minimum_candidate_fraction_per_view=args.minimum_view_fraction,
        ),
        force=args.force,
    )
    print(
        "sky evidence: "
        f"{evidence['summary']['accepted_view_count']}/{evidence['summary']['record_count']} views, "
        f"candidate_fraction={evidence['summary']['candidate_fraction']:.6f}, "
        f"sha256={evidence['sky_evidence_manifest_sha256']}"
    )
    if not args.skip_initialization:
        face = json.loads(args.face_manifest.read_text(encoding="utf-8"))
        centres = np.asarray(
            [np.asarray(image["c2w"], dtype=np.float64)[:3, 3] for image in face["images"]],
            dtype=np.float64,
        )
        initialization = build_independent_sky_initialization(
            evidence,
            centres.mean(axis=0),
            args.output / "sky_initialization_sh1.npz",
            config=IndependentSkyConfig(
                count=args.sky_count,
                radius_m=args.sky_radius_m,
                scale_m=args.sky_scale_m,
                opacity=args.sky_opacity,
                minimum_world_z_direction=args.minimum_world_z,
                sh_degree=1,
            ),
            force=args.force,
        )
        print(
            "sky initialization: "
            f"{initialization['gaussian_count']} Gaussians, "
            f"sha256={initialization['sky_initialization_manifest_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
