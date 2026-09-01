"""Build a signed V88 tail repair from the completed V87 Tile_0 run.

V87 reached the visual/coverage target but its unregularized scale tail grew too
wide: the median splat stayed near the competitor, while the P95 longest axis,
rendered depth and extreme anisotropy regressed.  This short continuation does
not change topology or Gaussian centres.  It progressively shrinks only splats
above 3 cm, regularizes the largest five-percent scale tail, and lets opacity,
colour and surface orientation settle for 200 steps.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import sign_gate, verify_gate
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
SOURCE = BASE / "v87_ultrasharp_detail" / "tile0_full12480"
TARGET = BASE / "v88_scale_tail_repair" / "tile0_step12680"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sign_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("config_manifest_sha256", None)
    result["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("aggressive", "balanced"),
        default="aggressive",
    )
    args = parser.parse_args()
    balanced = args.profile == "balanced"
    target = (
        BASE / "v88b_balanced_tail_repair" / "tile0_step12680"
        if balanced
        else TARGET
    )
    config = _read(SOURCE / "trainer.config.json")
    gate = _read(SOURCE / "training_gate.json")
    verify_gate(gate)
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "step_00012480.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 12480:
        raise ValueError("V88 requires the exact completed V87 step-12480 checkpoint")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    if len(source_trainer_sha) != 64:
        raise ValueError("V87 checkpoint has no signed trainer identity")

    config.update(
        {
            "run_id": (
                "snow-tile0-v88b-balanced-tail-repair-step12680"
                if balanced
                else "snow-tile0-v88-scale-tail-repair-step12680"
            ),
            "output_dir": str(target / "training_ewa"),
            "mipmap_pipeline_gate": str(target / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "max_steps": 12680,
            "checkpoint_every": 100,
            "checkpoint_keep_every": 12680,
            "view_sampling_mode": "with_replacement",
            "learning_rates": {
                "means": 0.0,
                "scales": 0.001,
                "quats": 0.0002,
                "opacities": 0.02 if balanced else 0.01,
                "colors": 0.001,
            },
            "lidar_alpha_weight": 0.10 if balanced else 0.05,
        }
    )
    config.pop("controlled_stop_after_steps", None)
    # Topology is already past its signed stop step.  Disable the exact-vendor
    # compatibility validator so this stabilization-only continuation can use
    # a tighter scale fuse without re-enabling births or culls.
    config["default_strategy"]["exact_mipmap_lifecycle"] = False
    config["default_strategy"]["lifecycle_execution_order"] = (
        "post_optimizer_gsplat"
    )
    config["default_strategy"]["cloudstudio_lifecycle_extension_profile"] = (
        "disabled"
    )
    config["default_strategy"]["vendor_capacity_cull_profile"] = "disabled"
    config["geometry_regularization"].update(
        {
            "opacity_sparsity_weight": 0.0,
            "scale_upper_weight": 0.0002 if balanced else 0.001,
            "scale_upper_tail_fraction": 0.05,
            "anisotropy_weight": 0.0002 if balanced else 0.0005,
            "max_scale_ratio_to_reference": 6.0,
            "max_anisotropy": 64.0,
            "max_world_size_m": 0.05 if balanced else 0.03,
        }
    )
    config["lidar_normal_alignment"].update(
        {
            "weight_align": 0.05,
            "weight_flatten": 0.05 if balanced else 0.10,
            "weight_point_to_plane": 0.01,
            "flatten_mode": "tangent_ratio_shortest_only",
            "flatten_ratio_target": 0.08,
        }
    )
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "v88_scale_tail_joint_quality_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})[
        "v88b_balanced_tail_repair" if balanced else "v88_scale_tail_repair"
    ] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 12480,
        "target_completed_steps": 12680,
        "source_gaussian_count": int(payload["params"]["means"].shape[0]),
        "source_findings": {
            "longest_axis_p95_m": 0.0723,
            "visible_width_gt_5px_fraction": 0.197,
            "axis_ratio_p50": 26.1,
            "axis_ratio_p95": 168.4,
            "depth_mae_m": 0.24572080387345827,
            "dead_opacity_lt_0p005_fraction": 0.2022,
            "fog_opacity_0p005_to_0p1_fraction": 0.3282,
        },
        "repair_contract": {
            "topology_events_after_source": False,
            "means_frozen": True,
            "progressive_max_world_size_m": 0.05 if balanced else 0.03,
            "scale_tail_fraction": 0.05,
            "max_anisotropy": 64.0,
            "additional_steps": 200,
        },
    }
    gate["adaptive_growth"] = {
        "profile": (
            "v88b_balanced_tail_repair_step12680"
            if balanced
            else "v88_scale_tail_repair_step12680"
        ),
        "stage": "stabilization",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": None,
        "mcmc_allowed": False,
        "capacity_cap": int(config["cap_max"]),
        "resume_checkpoint_sha256": _sha256_file(checkpoint),
        "resume_source_trainer_config_sha256": source_trainer_sha,
        "resume_allowed_lineage_differences": ["scale_calibration_sha256"],
        "warm_start_checkpoint_sha256": None,
    }
    gate = sign_gate(gate)

    _write(target / "trainer.config.json", config)
    _write(target / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(target / "trainer.config.json")
    print(target / "training_gate.json")
    print(
        "V88 tail repair: freeze means/topology, "
        f"profile={args.profile}, max_world={'5' if balanced else '3'} cm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
