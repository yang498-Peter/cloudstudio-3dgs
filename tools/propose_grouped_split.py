#!/usr/bin/env python3
"""Propose a grouped research split (train / validation / test) by rig instant.

Grouping unit is the rig frame: both cameras of one capture instant and, by
construction, every Face4 face cut from either fisheye. A face can never be
held out while its sibling face or the other camera of the same instant
trains, because they share the pose estimate and most of the field of view.

Selection works on temporal blocks of consecutive rig frames rather than on
single frames, so a held-out frame is not sandwiched between two training
frames 0.3 m away. Blocks are stratified by environment (indoor / covered /
outdoor, from ``build_view_membership.py``'s CSV when given) and by 2 m
spatial cell, and blocks that revisit a place seen much earlier or later
(loop closures) are drawn deliberately, since those are the views where a
model that memorised a trajectory segment fails.

Stationary frames (rig did not move) never enter validation or test: they
are duplicates of their neighbour and would score as "held out" while being
identical to a training view.

This tool only proposes. It writes one JSON and never touches an existing
split manifest; ``manual_assignment_train_val`` inside the JSON is what
``tools/build_split_manifest.py --mode manual`` would consume, with the test
block folded into ``val`` because that builder knows two labels only.

    python tools/propose_grouped_split.py \
        --dataset-manifest .../dataset_manifest.json \
        --membership-csv research/.../01_view_membership.csv \
        --regression-split-manifest C:/Peter/3dgs-datasets/house0305_sop_v8/split_manifest.json \
        --output research/.../01_grouped_split_proposal.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIONARY_STEP_M = 0.005


def rig_records(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    images = {str(image["image_id"]): image for image in dataset["images"]}
    records = []
    for frame in sorted(dataset["rig_frames"], key=lambda item: int(item["timestamp_ns"])):
        image_ids = [str(value) for value in frame["image_ids"] if str(value) in images]
        if not image_ids:
            continue
        centres = np.array([np.asarray(images[i]["c2w"], dtype=np.float64)[:3, 3] for i in image_ids])
        records.append(
            {
                "rig_frame_id": str(frame["rig_frame_id"]),
                "timestamp_ns": int(frame["timestamp_ns"]),
                "image_ids": sorted(image_ids),
                "position_m": centres.mean(axis=0).tolist(),
            }
        )
    if len(records) < 4:
        raise ValueError("at least four rig frames are required")
    # Stationary = did not move relative to the previous OR the next frame, so
    # the first frame of a set-down run is caught as well as the rest of it.
    positions = np.array([r["position_m"] for r in records])
    still = np.linalg.norm(np.diff(positions, axis=0), axis=1) < STATIONARY_STEP_M
    for index, record in enumerate(records):
        before = index > 0 and bool(still[index - 1])
        after = index < len(still) and bool(still[index])
        record["stationary"] = before or after
    return records


def read_environment_csv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            env[str(row["rig_frame_id"])] = str(row.get("environment", "unknown") or "unknown")
    return env


def detect_loop_closures(
    records: list[dict[str, Any]], *, radius_m: float, min_gap_s: float
) -> dict[str, dict[str, Any]]:
    """A frame is a revisit when another frame lies within ``radius_m`` but at
    least ``min_gap_s`` away in time."""
    from scipy.spatial import cKDTree

    positions = np.array([r["position_m"] for r in records])
    times = np.array([r["timestamp_ns"] for r in records], dtype=np.float64) / 1e9
    tree = cKDTree(positions)
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        neighbours = tree.query_ball_point(positions[index], radius_m)
        gaps = [abs(times[j] - times[index]) for j in neighbours if j != index]
        far = [g for g in gaps if g >= min_gap_s]
        result[record["rig_frame_id"]] = {
            "revisit": bool(far),
            "max_time_gap_s": float(max(far)) if far else 0.0,
            "revisit_partner_count": len(far),
        }
    return result


def temporal_blocks(records: list[dict[str, Any]], *, block_seconds: float) -> list[list[int]]:
    """Consecutive moving frames grouped into blocks of about ``block_seconds``;
    stationary frames form their own blocks so they can be pinned to train."""
    blocks: list[list[int]] = []
    current: list[int] = []
    start_time = None
    for index, record in enumerate(records):
        t = record["timestamp_ns"] / 1e9
        if record["stationary"]:
            if current:
                blocks.append(current)
                current = []
            if blocks and all(records[j]["stationary"] for j in blocks[-1]):
                blocks[-1].append(index)
            else:
                blocks.append([index])
            start_time = None
            continue
        if current and start_time is not None and t - start_time >= block_seconds:
            blocks.append(current)
            current = []
        if not current:
            start_time = t
        current.append(index)
    if current:
        blocks.append(current)
    return blocks


def _cell(position: list[float], cell_m: float) -> tuple[int, int]:
    return (int(np.floor(position[0] / cell_m)), int(np.floor(position[1] / cell_m)))


def _rank(seed: int, label: str, block_id: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{block_id}".encode("ascii")).hexdigest()


def propose(
    records: list[dict[str, Any]],
    *,
    environment: dict[str, str] | None,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    block_seconds: float,
    cell_m: float,
    loop_radius_m: float,
    loop_min_gap_s: float,
    min_revisit_blocks: int,
) -> dict[str, Any]:
    if not 0.0 < val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0 or val_fraction + test_fraction >= 0.5:
        raise ValueError("fractions must be positive and leave the majority for training")
    environment = environment or {}
    loops = detect_loop_closures(records, radius_m=loop_radius_m, min_gap_s=loop_min_gap_s)
    blocks = temporal_blocks(records, block_seconds=block_seconds)
    block_info = []
    for number, members in enumerate(blocks):
        envs = [environment.get(records[j]["rig_frame_id"], "unknown") for j in members]
        env = max(set(envs), key=envs.count)
        block_info.append(
            {
                "block_id": f"blk_{number:03d}",
                "members": members,
                "frames": len(members),
                "environment": env,
                "cells": sorted({_cell(records[j]["position_m"], cell_m) for j in members}),
                "revisit_frames": sum(1 for j in members if loops[records[j]["rig_frame_id"]]["revisit"]),
                "stationary": all(records[j]["stationary"] for j in members),
                "start_fraction": None,
            }
        )
    t0 = records[0]["timestamp_ns"]
    t1 = records[-1]["timestamp_ns"]
    for info in block_info:
        info["start_fraction"] = round((records[info["members"][0]]["timestamp_ns"] - t0) / max(1, t1 - t0), 4)

    moving = [b for b in block_info if not b["stationary"]]
    total_moving_frames = sum(b["frames"] for b in moving)
    assignment: dict[str, str] = {}
    used_cells: set[tuple[int, int]] = set()

    def draw(label: str, fraction: float) -> list[dict[str, Any]]:
        target = round(total_moving_frames * fraction)
        chosen: list[dict[str, Any]] = []
        taken = 0
        # Per-environment quota keeps indoor and outdoor both represented.
        env_frames: dict[str, int] = {}
        for b in moving:
            env_frames[b["environment"]] = env_frames.get(b["environment"], 0) + b["frames"]
        quota = {env: max(1, round(target * n / total_moving_frames)) for env, n in env_frames.items()}
        got = {env: 0 for env in quota}
        candidates = [b for b in moving if b["block_id"] not in assignment]
        # Loop-closure blocks first (deterministic hash order among them), then the rest.
        revisit_pool = sorted((b for b in candidates if b["revisit_frames"] > 0), key=lambda b: _rank(seed, label, b["block_id"]))
        other_pool = sorted((b for b in candidates if b["revisit_frames"] == 0), key=lambda b: _rank(seed, label, b["block_id"]))
        revisit_taken = 0
        for pool, need_revisit in ((revisit_pool, True), (other_pool, False)):
            for b in pool:
                if taken >= target:
                    break
                if need_revisit and revisit_taken >= min_revisit_blocks and got[b["environment"]] >= quota[b["environment"]]:
                    continue
                if not need_revisit and got[b["environment"]] >= quota[b["environment"]]:
                    continue
                # Avoid two held-out blocks in the same spatial cell when other cells remain.
                if any(c in used_cells for c in b["cells"]) and len(pool) > 1:
                    continue
                assignment[b["block_id"]] = label
                chosen.append(b)
                taken += b["frames"]
                got[b["environment"]] += b["frames"]
                used_cells.update(b["cells"])
                if need_revisit:
                    revisit_taken += 1
        # Fill any shortfall ignoring the cell rule but keeping environment quota loosely.
        if taken < target:
            for b in sorted((b for b in candidates if b["block_id"] not in assignment), key=lambda b: _rank(seed, label + ":fill", b["block_id"])):
                if taken >= target:
                    break
                assignment[b["block_id"]] = label
                chosen.append(b)
                taken += b["frames"]
        return chosen

    val_blocks = draw("val", val_fraction)
    test_blocks = draw("test", test_fraction) if test_fraction > 0 else []
    for b in block_info:
        assignment.setdefault(b["block_id"], "train")

    frame_split: dict[str, str] = {}
    for b in block_info:
        for j in b["members"]:
            frame_split[records[j]["rig_frame_id"]] = assignment[b["block_id"]]

    from scipy.spatial import cKDTree

    train_positions = np.array([r["position_m"] for r in records if frame_split[r["rig_frame_id"]] == "train"])
    tree = cKDTree(train_positions)
    nearest: dict[str, float] = {}
    for r in records:
        if frame_split[r["rig_frame_id"]] != "train":
            nearest[r["rig_frame_id"]] = float(tree.query(np.asarray(r["position_m"]), k=1)[0])

    def group_payload(label: str) -> list[dict[str, Any]]:
        out = []
        for b in block_info:
            if assignment[b["block_id"]] != label:
                continue
            out.append(
                {
                    "block_id": b["block_id"],
                    "environment": b["environment"],
                    "start_fraction": b["start_fraction"],
                    "revisit_frames": b["revisit_frames"],
                    "rig_frames": [
                        {
                            "rig_frame_id": records[j]["rig_frame_id"],
                            "timestamp_ns": records[j]["timestamp_ns"],
                            "image_ids": records[j]["image_ids"],
                            "position_m": records[j]["position_m"],
                            "nearest_train_camera_m": nearest.get(records[j]["rig_frame_id"]),
                            "revisit": loops[records[j]["rig_frame_id"]]["revisit"],
                        }
                        for j in b["members"]
                    ],
                }
            )
        return out

    def counts(label: str) -> dict[str, Any]:
        frames = [r for r in records if frame_split[r["rig_frame_id"]] == label]
        envs: dict[str, int] = {}
        for r in frames:
            e = environment.get(r["rig_frame_id"], "unknown")
            envs[e] = envs.get(e, 0) + 1
        cells = {_cell(r["position_m"], cell_m) for r in frames}
        return {
            "rig_frames": len(frames),
            "images": sum(len(r["image_ids"]) for r in frames),
            "blocks": sum(1 for b in block_info if assignment[b["block_id"]] == label),
            "by_environment": dict(sorted(envs.items())),
            "spatial_cells": len(cells),
            "revisit_frames": sum(1 for r in frames if loops[r["rig_frame_id"]]["revisit"]),
            "stationary_frames": sum(1 for r in frames if r["stationary"]),
        }

    held = [nearest[k] for k in nearest]
    proposal = {
        "schema_version": 1,
        "algorithm_version": "grouped_rig_block_split_v1",
        "grouping": "rig_frame (both cameras + all Face4 faces of each fisheye)",
        "configuration": {
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "block_seconds": block_seconds,
            "spatial_cell_m": cell_m,
            "loop_radius_m": loop_radius_m,
            "loop_min_gap_s": loop_min_gap_s,
            "min_revisit_blocks": min_revisit_blocks,
        },
        "counts": {label: counts(label) for label in ("train", "val", "test")},
        "coverage": {
            "total_rig_frames": len(records),
            "total_spatial_cells": len({_cell(r["position_m"], cell_m) for r in records}),
            "loop_closure_frames_total": sum(1 for k in loops if loops[k]["revisit"]),
            "held_out_nearest_train_camera_m": {
                "min": float(np.min(held)) if held else None,
                "p50": float(np.percentile(held, 50)) if held else None,
                "p95": float(np.percentile(held, 95)) if held else None,
                "max": float(np.max(held)) if held else None,
                "frames_over_0_25_m": int(sum(1 for v in held if v >= 0.25)),
            },
            "held_out_start_fractions": sorted(b["start_fraction"] for b in block_info if assignment[b["block_id"]] != "train"),
        },
        "groups": {label: group_payload(label) for label in ("val", "test")},
        "rig_frame_split": frame_split,
        "image_split": {i: frame_split[r["rig_frame_id"]] for r in records for i in r["image_ids"]},
        "manual_assignment_train_val": {k: ("train" if v == "train" else "val") for k, v in frame_split.items()},
    }
    return proposal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--membership-csv", type=Path, help="environment per rig frame from build_view_membership.py")
    parser.add_argument("--regression-split-manifest", type=Path, action="append", default=[],
                        help="existing split manifests to record as regression-only sets (never modified)")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-seconds", type=float, default=3.0)
    parser.add_argument("--cell-m", type=float, default=2.0)
    parser.add_argument("--loop-radius-m", type=float, default=1.0)
    parser.add_argument("--loop-min-gap-s", type=float, default=20.0)
    parser.add_argument("--min-revisit-blocks", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.name == "split_manifest.json":
        parser.error("refusing to write a file named split_manifest.json; this tool only proposes")
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    records = rig_records(dataset)
    environment = read_environment_csv(args.membership_csv) if args.membership_csv else None
    proposal = propose(
        records,
        environment=environment,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        block_seconds=args.block_seconds,
        cell_m=args.cell_m,
        loop_radius_m=args.loop_radius_m,
        loop_min_gap_s=args.loop_min_gap_s,
        min_revisit_blocks=args.min_revisit_blocks,
    )
    proposal["dataset_manifest"] = str(args.dataset_manifest)
    proposal["dataset_manifest_declared_sha256"] = dataset.get("manifest_sha256")
    regression = []
    for path in args.regression_split_manifest:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        val_ids = [str(i) for i in manifest.get("splits", {}).get("val", [])]
        regression.append(
            {
                "path": str(path),
                "split_manifest_sha256": manifest.get("split_manifest_sha256"),
                "configuration": manifest.get("configuration"),
                "val_images": len(val_ids),
                "role": "regression_only_same_frame_set",
                "val_images_in_proposed": {
                    label: sum(1 for i in val_ids if proposal["image_split"].get(i) == label)
                    for label in ("train", "val", "test")
                },
            }
        )
    proposal["regression_sets"] = regression
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proposal, indent=1), encoding="utf-8")
    c = proposal["counts"]
    print(
        "proposal: "
        + ", ".join(f"{k} {v['rig_frames']} rig frames / {v['images']} images" for k, v in c.items())
        + f" -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
