"""Build V75a: bug-fixed surface-aware detail Split from V74c step 702."""

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
SOURCE = BASE / "v74c_observation_cull_settle" / "tile0_review702"
TARGET = BASE / "v75a_surface_detail" / "tile0_boundary802"


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
    if int(payload.get("step", -1)) != 702:
        raise ValueError("V75a requires the exact V74c step-702 checkpoint")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    events = (
        payload.get("training_state", {})
        .get("mcmc_telemetry", {})
        .get("events", [])
    )
    if [int(event.get("step", -1)) for event in events] != [500, 600, 700]:
        raise ValueError("V74c lifecycle evidence is not the expected [500, 600, 700]")

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v75a-surface-detail-boundary802",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 802,
            "checkpoint_every": 799,
            "checkpoint_keep_every": 799,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "detail_split_policy": "lidar_surface_screen_detail",
            "detail_split_scale_m": 0.02,
            "detail_split_screen_radius": 0.0035,
            "revised_opacity": True,
        }
    )
    config["default_strategy"] = strategy
    config["tangent_proposal"] = {
        "enabled": True,
        "mode": "tangent",
        "planarity_gate": 0.6,
        "support_gate": 0.1,
        "support_tangent_factor": 3.0,
        "sigma_perp_factor": 1.0,
        "tangent_sigma_factor": 0.5,
        "normal_offset_factor": 0.1,
        "init_shortest_axis": True,
        "thickness_factor": 0.5,
        "min_thickness_m": 0.001,
        "reject_unsupported_births": True,
    }
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "surface_detail_boundary_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v75a_surface_detail"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 702,
        "pre_lifecycle_checkpoint_completed_steps": 799,
        "post_lifecycle_controlled_stop_completed_steps": 802,
        "algorithmic_changes": [
            "fix_split_then_clone_surface_proposal_parent_order",
            "screen_aware_detail_split",
            "lidar_tangent_surface_newborn_proposal",
            "shortest_axis_initialized_from_surface_normal",
        ],
        "unchanged_controls": [
            "projected_gradient_threshold_0p00015",
            "observation_cull_v1",
            "exact_relaxed_at_capacity",
            "all_losses_sampling_and_learning_rates",
        ],
    }
    gate["adaptive_growth"] = {
        "profile": "v75a_bugfixed_surface_detail_boundary",
        "stage": "review",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": 802,
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
    print("V75a: resume=702 checkpoint=799 stop=802 surface-detail guard enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
