#!/usr/bin/env python3
"""Build the production Tile plan from LiDAR visibility and accepted Face4 cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import laspy
import numpy as np
import psutil

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.adaptive_tiling import (
    AxisAlignedBox,
    build_adaptive_tile_plan,
    startup_budget_gib,
)
from cloudstudio_3dgs.pipeline.lidar_face4_observations import (
    build_lidar_face4_projected_observations,
    load_lidar_face4_projected_observations,
)
from cloudstudio_3dgs.pipeline.mipmap_gate import load_and_verify_gate


def _measured_budget() -> dict:
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.mem_get_info(0)[0] / 1024**3
    except (ImportError, RuntimeError):
        pass
    return startup_budget_gib(
        gpu0_available_gib=gpu,
        system_available_gib=psutil.virtual_memory().available / 1024**3,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--depth-manifest", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--scene-point-cloud", required=True, type=Path)
    parser.add_argument("--train-face-manifest", required=True, type=Path)
    parser.add_argument("--train-face-root", required=True, type=Path)
    parser.add_argument("--val-face-manifest", required=True, type=Path)
    parser.add_argument("--val-face-root", required=True, type=Path)
    parser.add_argument("--sky-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-raw-view", type=int, default=5_000)
    parser.add_argument("--scene-padding-fraction", type=float, default=0.2)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observation_path = args.output / "lidar_face4_projected_observations.npz"
    observation = build_lidar_face4_projected_observations(
        args.dataset_manifest,
        args.depth_manifest,
        args.depth_root,
        (
            (args.train_face_manifest, args.train_face_root),
            (args.val_face_manifest, args.val_face_root),
        ),
        observation_path,
        samples_per_raw_view=args.samples_per_raw_view,
        force=args.force,
    )
    all_table, train_table = load_lidar_face4_projected_observations(observation_path)
    gate, gate_sha = load_and_verify_gate(args.sky_gate)
    bindings = dict(gate["bindings"])
    if observation["point_cloud_sha256"] != bindings["lidar_depth_point_cloud_sha256"]:
        raise ValueError("LiDAR tile observations are bound to a different point cloud")
    header = laspy.open(args.scene_point_cloud).header
    minimum = np.asarray(header.mins, dtype=np.float64)
    maximum = np.asarray(header.maxs, dtype=np.float64)
    padding = (maximum - minimum) * args.scene_padding_fraction
    measured = _measured_budget()
    budget_gib = min(float(measured["budget_gib"]), args.budget_gib) if args.budget_gib else float(measured["budget_gib"])
    plan = build_adaptive_tile_plan(
        all_table,
        cost_table=train_table,
        root_box=AxisAlignedBox(minimum - padding, maximum + padding),
        budget_gib=budget_gib,
        source_bindings={
            "training_dataset_manifest_sha256": bindings["training_dataset_manifest_sha256"],
            "face4_train_manifest_sha256": bindings["face4_train_manifest_sha256"],
            "face4_val_manifest_sha256": bindings["face4_val_manifest_sha256"],
            "sky_train_evidence_manifest_sha256": bindings["sky_train_evidence_manifest_sha256"],
            "lidar_depth_manifest_sha256": bindings["lidar_depth_manifest_sha256"],
            "scene_point_cloud_sha256": bindings["lidar_depth_point_cloud_sha256"],
            "face4_observation_manifest_sha256": observation["face4_observation_manifest_sha256"],
            "source_sky_gate_sha256": gate_sha,
        },
    )
    for tile in plan["tiles"]:
        for view in tile["views"]:
            view["sample_id"] = observation["train_view_ids"][int(view["image_index"])]
    plan["startup_memory_budget"] = measured
    plan["effective_conservative_budget_gib"] = budget_gib
    plan["spatial_anchor_source"] = "accepted_full_lidar_depth_visibility"
    plan.pop("tile_plan_manifest_sha256", None)
    plan["tile_plan_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(plan)
    ).hexdigest()
    destination = args.output / "adaptive_tile_plan.json"
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"LiDAR Face4 observations: points={observation['point_count']}, "
        f"all={observation['all_observation_count']}, train={observation['train_observation_count']}"
    )
    print(
        f"LiDAR adaptive Tile plan: leaves={plan['leaf_count']}, "
        f"retained={plan['retained_tile_count']}, budget={budget_gib:.3f} GiB, "
        f"sha256={plan['tile_plan_manifest_sha256']} -> {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
