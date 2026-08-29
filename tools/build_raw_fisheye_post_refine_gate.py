#!/usr/bin/env python3
"""Sign a geometry-frozen raw-fisheye appearance-refinement training arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    FIXED_TOPOLOGY_EVALUATION_READY_STATUS,
    fixed_topology_evaluation_arm_fingerprint,
    load_and_verify_gate,
    sign_gate,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--smoke-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--run-id")
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument(
        "--inherit-warm-start-auxiliary",
        action="store_true",
        help="Load auxiliary parameters such as exposure gains from an explicit warm start.",
    )
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument(
        "--freeze-opacity",
        action="store_true",
        help=(
            "Set the opacity learning rate to zero. Use this for a "
            "non-destructive colour/exposure-only refinement arm."
        ),
    )
    args = parser.parse_args()
    if args.steps < 10:
        raise ValueError("production post-refine requires at least 10 steps")
    if args.checkpoint_every is not None and not (
        1 <= args.checkpoint_every <= args.steps
    ):
        raise ValueError("checkpoint_every must be between 1 and steps")
    if args.inherit_warm_start_auxiliary and args.warm_start_checkpoint is None:
        raise ValueError(
            "inherit_warm_start_auxiliary requires an explicit warm-start checkpoint"
        )

    gate, upstream_sha = load_and_verify_gate(args.upstream_gate)
    if gate.get("status") != FIXED_TOPOLOGY_EVALUATION_READY_STATUS:
        raise ValueError("upstream gate must authorize fixed-topology evaluation")

    config = json.loads(args.smoke_config.read_text(encoding="utf-8"))
    if args.warm_start_checkpoint is not None:
        warm_start_checkpoint = args.warm_start_checkpoint.resolve()
        if not warm_start_checkpoint.is_file():
            raise FileNotFoundError(
                f"warm-start checkpoint is missing: {warm_start_checkpoint}"
            )
        config["warm_start_checkpoint"] = warm_start_checkpoint.as_posix()
    if args.inherit_warm_start_auxiliary:
        config["warm_start_fresh_auxiliary"] = []
    if args.freeze_opacity:
        config.setdefault("learning_rates", {})["opacities"] = 0.0
    config.update(
        {
            "run_id": args.run_id
            or (
                "snow-full-area-a0-raw-fisheye-appearance-"
                f"refine{args.steps}"
            ),
            "output_dir": args.output_dir.resolve().as_posix(),
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": True,
            "max_steps": args.steps,
            "checkpoint_every": args.checkpoint_every
            if args.checkpoint_every is not None
            else min(50, args.steps),
            "checkpoint_keep_every": 0,
            "mcmc_refine_start_iter": args.steps + 1,
            "mcmc_refine_stop_iter": args.steps + 2,
            "mipmap_pipeline_gate": args.output_gate.resolve().as_posix(),
        }
    )
    if config.get("topology_policy", {}).get("mode") != "strict_fixed":
        raise ValueError("post-refine must keep strict topology")
    if any(
        float(config.get("learning_rates", {}).get(name, -1.0)) != 0.0
        for name in ("means", "scales", "quats")
    ):
        raise ValueError("post-refine must freeze means, scales and quaternions")
    opacity_learning_rate = float(
        config.get("learning_rates", {}).get("opacities", -1.0)
    )
    if opacity_learning_rate < 0.0:
        raise ValueError("post-refine opacity learning rate must be non-negative")
    if not config.get("raw_fisheye_post_refine_face_manifest"):
        raise ValueError("post-refine requires signed Face4 lineage")

    fingerprint = fixed_topology_evaluation_arm_fingerprint(config)
    config_sha = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    unsigned_gate = dict(gate)
    unsigned_gate.pop("gate_manifest_sha256", None)
    evaluation = dict(unsigned_gate.get("fixed_topology_evaluation", {}))
    allowed = list(evaluation.get("allowed_arms", []))
    allowed.append(
        {
            "arm": "RAW_FISHEYE_APPEARANCE_REFINE",
            "fingerprint_sha256": fingerprint,
            "config_sha256": config_sha,
        }
    )
    evaluation["allowed_arms"] = allowed
    unsigned_gate["fixed_topology_evaluation"] = evaluation
    unsigned_gate["raw_fisheye_post_refine"] = {
        "profile": (
            "geometry_opacity_frozen_sh0_color_exposure"
            if opacity_learning_rate == 0.0
            else "geometry_frozen_sh0_opacity_color_exposure"
        ),
        "upstream_gate_sha256": upstream_sha,
        "warm_start_checkpoint": config["warm_start_checkpoint"],
        "face_lineage_manifest": config[
            "raw_fisheye_post_refine_face_manifest"
        ],
        "factor": int(config["factor"]),
        "max_steps": int(config["max_steps"]),
        "geometry_learning_rates": {
            name: float(config["learning_rates"][name])
            for name in ("means", "scales", "quats")
        },
        "opacity_learning_rate": opacity_learning_rate,
        "opacity_frozen": opacity_learning_rate == 0.0,
        "arm_fingerprint_sha256": fingerprint,
    }
    signed_gate = sign_gate(unsigned_gate)
    write_json(args.output_gate, signed_gate)
    write_json(args.output_config, config)
    print(
        f"raw-fisheye post-refine authorized: steps={args.steps}, "
        f"fingerprint={fingerprint}, gate={args.output_gate}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
