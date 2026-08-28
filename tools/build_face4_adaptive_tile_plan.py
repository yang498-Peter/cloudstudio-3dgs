#!/usr/bin/env python3
"""Reproject accepted AT tracks into Face4 and build a signed adaptive Tile plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.pipeline.adaptive_tiling import (
    AxisAlignedBox,
    build_adaptive_tile_plan,
    startup_budget_gib,
)
from cloudstudio_3dgs.pipeline.face4_observations import (
    build_face4_projected_observations,
    load_face4_projected_observations,
)
from cloudstudio_3dgs.pipeline.mipmap_gate import load_and_verify_gate


def _memory_budget() -> dict:
    import psutil

    gpu_available = None
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            gpu_available = free_bytes / 1024**3
    except (ImportError, RuntimeError):
        gpu_available = None
    return startup_budget_gib(
        gpu0_available_gib=gpu_available,
        system_available_gib=psutil.virtual_memory().available / 1024**3,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--train-face-manifest", required=True, type=Path)
    parser.add_argument("--train-face-root", required=True, type=Path)
    parser.add_argument("--val-face-manifest", required=True, type=Path)
    parser.add_argument("--val-face-root", required=True, type=Path)
    parser.add_argument("--sky-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-point-cloud", type=Path)
    parser.add_argument("--scene-padding-fraction", type=float, default=0.2)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--force-depth", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observation_path = args.output / "face4_projected_observations.npz"
    observation_manifest = build_face4_projected_observations(
        args.candidate_model,
        args.dataset_manifest,
        (
            (args.train_face_manifest, args.train_face_root),
            (args.val_face_manifest, args.val_face_root),
        ),
        observation_path,
        force=args.force,
    )
    all_table, train_table = load_face4_projected_observations(observation_path)
    gate, gate_sha = load_and_verify_gate(args.sky_gate)
    budget = _memory_budget()
    budget_gib = args.budget_gib
    if budget_gib is None and args.force_depth is None:
        budget_gib = float(budget["budget_gib"])
    bindings = dict(gate.get("bindings", {}))
    source_bindings = {
        "training_dataset_manifest_sha256": bindings.get("training_dataset_manifest_sha256"),
        "face4_train_manifest_sha256": bindings.get("face4_train_manifest_sha256"),
        "face4_val_manifest_sha256": bindings.get("face4_val_manifest_sha256"),
        "sky_train_evidence_manifest_sha256": bindings.get("sky_train_evidence_manifest_sha256"),
        "face4_observation_manifest_sha256": observation_manifest["face4_observation_manifest_sha256"],
        "source_sky_gate_sha256": gate_sha,
    }
    root_box = None
    if args.scene_point_cloud is not None:
        import laspy
        import numpy as np

        if not 0.0 <= args.scene_padding_fraction < 1.0:
            raise ValueError("scene padding fraction must be within [0, 1)")
        header = laspy.open(args.scene_point_cloud).header
        minimum = np.asarray(header.mins, dtype=np.float64)
        maximum = np.asarray(header.maxs, dtype=np.float64)
        padding = (maximum - minimum) * args.scene_padding_fraction
        root_box = AxisAlignedBox(minimum - padding, maximum + padding)
        source_bindings["scene_point_cloud_sha256"] = bindings.get(
            "lidar_depth_point_cloud_sha256"
        )
    plan = build_adaptive_tile_plan(
        all_table,
        cost_table=train_table,
        root_box=root_box,
        budget_gib=budget_gib,
        force_depth=args.force_depth,
        source_bindings=source_bindings,
    )
    train_view_ids = observation_manifest["train_view_ids"]
    for tile in plan["tiles"]:
        for view in tile["views"]:
            view["sample_id"] = train_view_ids[int(view["image_index"])]
    plan["startup_memory_budget"] = budget
    # Adding measured budget after construction changes the signature.
    from cloudstudio_3dgs.data.manifest import canonical_json_bytes
    import hashlib

    plan.pop("tile_plan_manifest_sha256", None)
    plan["tile_plan_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(plan)
    ).hexdigest()
    destination = args.output / "adaptive_tile_plan.json"
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Face4 observations: all={observation_manifest['all_observation_count']}, "
        f"train={observation_manifest['train_observation_count']}"
    )
    print(
        f"adaptive Tile plan: leaves={plan['leaf_count']}, "
        f"retained={plan['retained_tile_count']}, budget={budget_gib}, "
        f"sha256={plan['tile_plan_manifest_sha256']} -> {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
