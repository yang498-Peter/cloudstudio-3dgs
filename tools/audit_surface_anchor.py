#!/usr/bin/env python3
"""How far is the population from the LiDAR surface it was initialized on?

Prints the far-fraction table (share of gaussians farther than 0.05 / 0.1 /
0.2 / 0.5 / 1.0 m from the nearest initialization point, for all rows and
for rows at or above an opacity floor), the z profile of near vs far rows
and, when a Tile box is known, the share outside it. This is the number the
surface-anchor prune (``surface_anchor_prune`` in the trainer config) acts
on, computed by the same function the prune uses.

    python tools/audit_surface_anchor.py --config RUN.json [--checkpoint latest.pt]
    python tools/audit_surface_anchor.py --checkpoint latest.pt --init-ply init.ply \
        [--tile-inputs-manifest tile_inputs_manifest.json --tile-id 1 | --box "[[x0,y0,z0],[x1,y1,z1]]"]
        [--opacity-floor 0.05] [--output report.json]

CPU only; the checkpoint is loaded with ``map_location="cpu"``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.surface_anchor import (  # noqa: E402
    AUDIT_DISTANCE_THRESHOLDS_M,
    audit_far_fraction,
    build_anchor_tree,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tile_box(manifest_path: Path, tile_id: int) -> list[list[float]]:
    manifest = _load_json(manifest_path)
    matches = [
        tile for tile in manifest.get("tiles", []) if int(tile.get("tile_id", -1)) == int(tile_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"tile {tile_id} is not unique in {manifest_path}")
    return matches[0]["training_and_export_box"]


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Any | None, dict[str, Any]]:
    provenance: dict[str, Any] = {}
    checkpoint = args.checkpoint
    init_ply = args.init_ply
    box = None
    if args.config is not None:
        config = _load_json(args.config)
        provenance["config"] = str(args.config)
        if checkpoint is None:
            checkpoint = Path(config["output_dir"]) / "checkpoints" / "latest.pt"
        if init_ply is None:
            init_ply = Path(config["initialization_ply"])
        if (
            args.box is None
            and args.tile_inputs_manifest is None
            and config.get("tile_inputs_manifest") is not None
            and config.get("mipmap_tile_id") is not None
        ):
            manifest_path = Path(config["tile_inputs_manifest"])
            box = _tile_box(manifest_path, int(config["mipmap_tile_id"]))
            provenance["box_source"] = f"{manifest_path} tile {config['mipmap_tile_id']}"
    if args.box is not None:
        box = json.loads(args.box)
        provenance["box_source"] = "--box"
    elif args.tile_inputs_manifest is not None:
        if args.tile_id is None:
            raise SystemExit("--tile-inputs-manifest needs --tile-id")
        box = _tile_box(args.tile_inputs_manifest, int(args.tile_id))
        provenance["box_source"] = f"{args.tile_inputs_manifest} tile {args.tile_id}"
    if checkpoint is None or init_ply is None:
        raise SystemExit("need --config or both --checkpoint and --init-ply")
    return Path(checkpoint), Path(init_ply), box, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, help="trainer config; supplies checkpoint, PLY and box")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--init-ply", type=Path)
    parser.add_argument("--tile-inputs-manifest", type=Path)
    parser.add_argument("--tile-id", type=int)
    parser.add_argument("--box", type=str, help='JSON "[[x0,y0,z0],[x1,y1,z1]]"')
    parser.add_argument("--opacity-floor", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint_path, init_ply, box, provenance = _resolve_inputs(args)

    import numpy as np
    import torch

    from cloudstudio_3dgs.training.trainer import load_initialization_ply

    started = time.perf_counter()
    xyz, _ = load_initialization_ply(init_ply)
    tree = build_anchor_tree(np.asarray(xyz, dtype=np.float64))
    tree_seconds = time.perf_counter() - started

    started = time.perf_counter()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = payload["params"]
    means = params["means"].detach().to(torch.float64).numpy()
    opacity = torch.sigmoid(params["opacities"].detach().float().flatten()).numpy()
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    report = audit_far_fraction(
        means,
        tree,
        opacity=opacity,
        opacity_floor=float(args.opacity_floor),
        thresholds_m=AUDIT_DISTANCE_THRESHOLDS_M,
        box=box,
        workers=int(args.workers),
    )
    query_seconds = time.perf_counter() - started
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(payload.get("step", 0)),
            "initialization_ply": str(init_ply),
            "box": box,
            "timing_s": {
                "tree_build": tree_seconds,
                "checkpoint_load": load_seconds,
                "query_all_thresholds": query_seconds,
            },
            **provenance,
        }
    )

    print(f"checkpoint {checkpoint_path} step {report['checkpoint_step']}")
    print(
        f"gaussians {report['gaussian_count']:,}  anchors {report['anchor_count']:,}  "
        f"visible(op>={args.opacity_floor}) {report['visible_count']:,}"
    )
    print(
        f"tree {tree_seconds:.1f}s  load {load_seconds:.1f}s  "
        f"query(k=1, no bound) {query_seconds:.1f}s"
    )
    print("threshold_m  far_all   far_visible")
    for row in report["far_fraction_table"]:
        visible = row.get("far_fraction_visible")
        print(
            f"{row['threshold_m']:>10.2f}  {row['far_fraction']:.3f}     "
            f"{'-' if visible is None else f'{visible:.3f}'}"
        )
    dq = report["distance_quantiles_m"]
    print(
        "distance p50/p90/p95/p99 m: "
        + " / ".join(f"{dq[key]:.3f}" for key in ("p50", "p90", "p95", "p99"))
    )
    near, far = report["z_quantiles_near_m"], report["z_quantiles_far_m"]
    fmt = lambda q: "/".join("-" if q[k] is None else f"{q[k]:.2f}" for k in ("p05", "p50", "p95"))
    print(f"z p05/p50/p95 near(<= {report['z_split_m']} m): {fmt(near)}   far: {fmt(far)}")
    if box is not None:
        print(
            f"outside training box: {report['outside_box_fraction']:.3f}  "
            f"(outside and far: {report['outside_box_and_far_fraction']:.3f})"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
