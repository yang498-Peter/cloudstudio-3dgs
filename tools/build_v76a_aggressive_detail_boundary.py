"""Build V76a: aggressive but surface-safe visual detail refinement."""

from __future__ import annotations

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
SOURCE = BASE / "v75b_surface_detail_settle" / "tile0_review1002"
TARGET = BASE / "v76a_aggressive_detail" / "tile0_boundary1102"


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
    source_config = _read(SOURCE / "trainer.config.json")
    source_gate = _read(SOURCE / "training_gate.json")
    verify_gate(source_gate)
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 1002:
        raise ValueError("V76a requires the exact V75b step-1002 checkpoint")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    last_event = (
        payload.get("training_state", {})
        .get("mcmc_telemetry", {})
        .get("events", [])[-1]
    )
    if int(last_event.get("step", -1)) != 1000:
        raise ValueError("V75b did not stop after the expected step-1000 event")

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v76a-aggressive-detail-boundary1102",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 1102,
            "checkpoint_every": 1099,
            "checkpoint_keep_every": 1099,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "grow_grad2d": 0.0001,
            "detail_split_policy": "lidar_surface_screen_detail_aggressive",
            "detail_split_scale_m": 0.01,
            "detail_split_screen_radius": 0.0025,
            "revised_opacity": True,
        }
    )
    config["default_strategy"] = strategy
    proposal = dict(config["tangent_proposal"])
    proposal.update(
        {
            "tangent_sigma_factor": 0.65,
            "normal_offset_factor": 0.05,
            "thickness_factor": 0.25,
            "min_thickness_m": 0.0005,
        }
    )
    config["tangent_proposal"] = proposal
    normal = dict(config["lidar_normal_alignment"])
    normal.update(
        {
            "enabled": True,
            "weight_align": 0.02,
            "weight_flatten": 0.02,
            "weight_point_to_plane": 0.0,
            "flatten_mode": "tangent_ratio_shortest_only",
            "flatten_ratio_target": 0.1,
        }
    )
    config["lidar_normal_alignment"] = normal
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "aggressive_detail_boundary_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v76a_aggressive_detail"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 1002,
        "pre_lifecycle_checkpoint_completed_steps": 1099,
        "post_lifecycle_controlled_stop_completed_steps": 1102,
        "changes": {
            "projected_gradient_threshold": 0.0001,
            "detail_split_scale_m": 0.01,
            "detail_split_screen_radius": 0.0025,
            "tangent_sigma_factor": 0.65,
            "normal_offset_factor": 0.05,
            "newborn_min_thickness_m": 0.0005,
            "newborn_thickness_factor": 0.25,
            "shortest_axis_only_flatten_weight": 0.02,
            "shortest_axis_ratio_target": 0.1,
        },
        "safety_boundary": (
            "one lifecycle event only; stop before the step-1200 opacity "
            "Cull/reset boundary"
        ),
    }
    gate["adaptive_growth"] = {
        "profile": "v76a_aggressive_surface_detail_boundary",
        "stage": "review",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": 1102,
        "mcmc_allowed": False,
        "capacity_cap": int(config["cap_max"]),
        "resume_checkpoint_sha256": _sha256_file(checkpoint),
        "resume_source_trainer_config_sha256": source_trainer_sha,
        "resume_allowed_lineage_differences": [],
        "warm_start_checkpoint_sha256": None,
    }
    gate = sign_gate(gate)

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V76a: resume=1002 checkpoint=1099 stop=1102 aggressive detail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
