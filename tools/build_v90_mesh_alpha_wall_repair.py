"""Build a bounded Tile_0 wall-coverage repair from the V88b checkpoint."""

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
SOURCE = BASE / "v88b_balanced_tail_repair" / "tile0_step12680"
TARGET = BASE / "v90_mesh_alpha_wall_repair" / "tile0_step12880"


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
    config = _read(SOURCE / "trainer.config.json")
    gate = _read(SOURCE / "training_gate.json")
    verify_gate(gate)
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "step_00012680.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 12680:
        raise ValueError("V90 requires the exact V88b step-12680 checkpoint")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    if len(source_trainer_sha) != 64:
        raise ValueError("V88b checkpoint has no signed trainer identity")

    config.update(
        {
            "run_id": "snow-tile0-v90-mesh-alpha-wall-repair-step12880",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "max_steps": 12880,
            "checkpoint_every": 100,
            "checkpoint_keep_every": 12880,
            "view_sampling_mode": "with_replacement",
            "learning_rates": {
                "means": 0.0,
                "scales": 0.0,
                "quats": 0.0,
                "opacities": 0.02,
                "colors": 0.0005,
            },
            "lidar_alpha_weight": 0.10,
            "lidar_alpha_dilation_radius_px": 3,
            "mesh_alpha_weight": 0.20,
            "mesh_alpha_target": 0.95,
            "mesh_depth_weight": 0.0,
            "mesh_normal_weight": 0.0,
            "rendered_depth_normal_consistency_weight": 0.0,
            "da2_depth_weight": 0.0,
        }
    )
    config.pop("controlled_stop_after_steps", None)
    config["geometry_regularization"].update(
        {
            "opacity_sparsity_weight": 0.0,
            "scale_upper_weight": 0.0,
            "anisotropy_weight": 0.0,
        }
    )
    config["lidar_normal_alignment"].update(
        {
            "enabled": False,
            "weight_align": 0.0,
            "weight_flatten": 0.0,
            "weight_point_to_plane": 0.0,
        }
    )
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "v90_wall_roi_black_white_alpha_evaluation",
        }
    )
    bindings = dict(gate["bindings"])
    bindings["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate["bindings"] = bindings
    gate.setdefault("evidence", {})["v90_mesh_alpha_wall_repair"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 12680,
        "target_completed_steps": 12880,
        "source_gaussian_count": int(payload["params"]["means"].shape[0]),
        "user_visual_failure": (
            "wall board black gaps, local holes and smeared patches in SuperSplat"
        ),
        "repair_contract": {
            "topology_events": False,
            "means_scales_quaternions_frozen": True,
            "opacity_sparsity_disabled": True,
            "trusted_mesh_source_types": [2, 3],
            "rejected_mesh_source_types": [4],
            "mesh_alpha_weight": 0.20,
            "mesh_alpha_target": 0.95,
            "lidar_alpha_dilation_radius_px": 3,
            "additional_steps": 200,
        },
    }
    gate["adaptive_growth"] = {
        "profile": "v90_mesh_alpha_wall_repair_step12880",
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

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V90: frozen geometry, trusted-mesh alpha coverage repair, 200 steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
