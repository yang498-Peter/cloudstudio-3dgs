#!/usr/bin/env python3
"""Generate signed, independent person masks from a dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.person_masks import PersonMaskConfig, build_person_masks
from cloudstudio_3dgs.data.torchvision_person import TorchVisionPersonSegmenter


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-mask-manifest", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--runtime-lock", type=Path, default=Path("upstream/person_mask.lock.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-max-dimension", type=int, default=800)
    parser.add_argument("--score-threshold", type=float, default=0.65)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--dilation-pixels", type=int, default=12)
    parser.add_argument("--review-frames-per-camera", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    segmenter = TorchVisionPersonSegmenter(
        args.runtime_lock,
        args.weights,
        device=args.device,
        score_threshold=args.score_threshold,
        inference_max_dimension=args.inference_max_dimension,
    )
    manifest = build_person_masks(
        _json(args.manifest),
        _json(args.base_mask_manifest),
        args.recording_root,
        args.output,
        segmenter=segmenter,
        model_identity=segmenter.model_identity,
        config=PersonMaskConfig(
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            dilation_pixels=args.dilation_pixels,
            review_frames_per_camera=args.review_frames_per_camera,
        ),
        force=args.force,
        progress=lambda done, total, image_id: print(
            f"person mask progress: {done}/{total} image_id={image_id}", flush=True
        )
        if done == 1 or done == total or done % 25 == 0
        else None,
    )
    summary = manifest["summary"]
    print(
        "person masks complete: "
        f"images={summary['image_count']} "
        f"with_person={summary['images_with_person']} "
        f"instances={summary['person_instances']} "
        f"sha256={manifest['person_mask_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
