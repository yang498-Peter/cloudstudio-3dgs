"""Command line for the ingestion layer.

    python -m cloudstudio3dgs_sdk.ingest.cli detect <dataset>
    python -m cloudstudio3dgs_sdk.ingest.cli bundle <dataset> --output <dir>
    python -m cloudstudio3dgs_sdk.ingest.cli tile --point-cloud <las> --output <plan.json>
    python -m cloudstudio3dgs_sdk.ingest.cli plan <dataset> --dataset-root ... --dry-run

``plan`` without ``--execute`` only prints; ``--execute`` runs the CPU builders
and stops at the first GPU cache, which belongs to the SDK runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .adapters import detect_adapter, load_dataset
from .bundle import describe_capabilities, write_bundle_manifest
from .caches import CacheProfile, CachePlan
from .errors import IngestError
from .tiling import TilingRule, build_slab_tile_plan, histogram_from_las, slab_split


def _write_json(path: Path, payload: dict, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to replace {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _detect(args: argparse.Namespace) -> int:
    adapter = detect_adapter(Path(args.dataset))
    print(f"{adapter.NAME} (v{adapter.VERSION}): {adapter.REQUIRES}")
    return 0


def _bundle(args: argparse.Namespace) -> int:
    bundle = load_dataset(Path(args.dataset), adapter=args.adapter)
    print(f"adapter: {bundle.adapter}")
    print(f"images: {len(bundle.images)} from {len(bundle.cameras)} cameras")
    for line in describe_capabilities(bundle):
        print("  " + line)
    for warning in bundle.warnings:
        print(f"  warning: {warning}")
    if args.output:
        path = write_bundle_manifest(
            bundle,
            Path(args.output),
            hash_images=not args.skip_image_hashes,
            force=args.force,
        )
        print(f"bundle manifest -> {path}")
    return 0


def _tile(args: argparse.Namespace) -> int:
    rule = TilingRule(
        tile_count=args.tile_count,
        axis=args.axis,
        scene_padding_fraction=args.scene_padding_fraction,
        halo_fraction_per_side=args.halo_fraction_per_side,
        overlap_margin_m=args.overlap_margin_m,
        histogram_bins=args.histogram_bins,
    )
    histogram = histogram_from_las(Path(args.point_cloud), bins=rule.histogram_bins)
    slab = slab_split(histogram, rule)
    plan = build_slab_tile_plan(slab)
    _write_json(Path(args.output), plan, force=args.force)
    print(
        f"slab tile plan: axis={slab.axis_name} tiles={rule.tile_count} "
        f"cuts={list(slab.cuts)} balance={slab.balance():.3f} "
        f"sha256={plan['tile_plan_manifest_sha256']} -> {args.output}"
    )
    print(
        "views_source=deferred: bind a projected observation table before "
        "materializing tile inputs"
    )
    return 0


def _plan(args: argparse.Namespace) -> int:
    bundle = load_dataset(Path(args.dataset), adapter=args.adapter)
    profile = CacheProfile.from_any(
        {
            "dataset_root": args.dataset_root,
            "cache_root": args.cache_root or args.dataset_root,
            "run_root": args.run_root,
            "recording_root": args.dataset,
            "source_run_dir": args.run_dir or args.dataset,
            "tile_count": args.tile_count,
        }
    )
    plan = CachePlan(bundle, profile)
    for line in plan.build(dry_run=not args.execute):
        print(line)
    if args.report:
        _write_json(Path(args.report), plan.to_dict(), force=args.force)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cloudstudio3dgs-ingest", description=__doc__)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="name the adapter that claims a dataset")
    detect.add_argument("dataset")
    detect.set_defaults(handler=_detect)

    bundle = sub.add_parser("bundle", help="load a dataset and report its capabilities")
    bundle.add_argument("dataset")
    bundle.add_argument("--adapter", default=None)
    bundle.add_argument("--output", default=None, help="write bundle_manifest.json here")
    bundle.add_argument("--skip-image-hashes", action="store_true")
    bundle.set_defaults(handler=_bundle)

    tile = sub.add_parser("tile", help="derive slab tile boxes from a point cloud")
    tile.add_argument("--point-cloud", required=True)
    tile.add_argument("--output", required=True)
    tile.add_argument("--tile-count", type=int, default=4)
    tile.add_argument("--axis", default="auto", choices=("auto", "x", "y"))
    tile.add_argument("--scene-padding-fraction", type=float, default=0.2)
    tile.add_argument("--halo-fraction-per-side", type=float, default=0.002)
    tile.add_argument("--overlap-margin-m", type=float, default=None)
    tile.add_argument("--histogram-bins", type=int, default=4096)
    tile.set_defaults(handler=_tile)

    plan = sub.add_parser("plan", help="print (or run the CPU half of) the cache plan")
    plan.add_argument("dataset")
    plan.add_argument("--adapter", default=None)
    plan.add_argument("--dataset-root", required=True)
    plan.add_argument("--cache-root", default=None)
    plan.add_argument("--run-root", required=True)
    plan.add_argument("--run-dir", default=None)
    plan.add_argument("--tile-count", type=int, default=4)
    plan.add_argument("--report", default=None, help="write the plan as JSON")
    plan.add_argument(
        "--execute",
        action="store_true",
        help="run the CPU builders; GPU caches still refuse",
    )
    plan.set_defaults(handler=_plan)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
