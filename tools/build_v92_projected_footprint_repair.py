"""Build V92: a 20-step projected-footprint repair from V91b.

The experiment preserves V91b topology and centres. Only splats whose current
projected radius exceeds roughly five pixels (0.0035 of the Face4 short side)
are gently shrunk, capped to about nine percent per observed step. This tests
whether projection-aware shrink can reduce V91b blur/depth error without
reopening the V88b wall holes caused by a global metric threshold.
"""

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
SOURCE = BASE / "v91b_coverage_preserving_regrowth" / "tile0_boundary12600"
TARGET = BASE / "v92_projected_footprint_repair" / "tile0_step12620"


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
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "step_00012600.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 12600:
        raise ValueError("V92 requires the exact V91b step-12600 checkpoint")
    source_count = int(payload["params"]["means"].shape[0])
    if source_count != 4_281_733:
        raise ValueError(f"unexpected V91b Gaussian count: {source_count}")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    if len(source_trainer_sha) != 64:
        raise ValueError("V91b checkpoint has no signed trainer identity")

    config.update(
        {
            "run_id": "snow-tile0-v92-projected-footprint-repair-step12620",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "max_steps": 12620,
            "checkpoint_every": 12620,
            "checkpoint_keep_every": 12620,
            "learning_rates": {
                "means": 0.0,
                "scales": 0.0,
                "quats": 0.0,
                "opacities": 0.005,
                "colors": 0.0005,
            },
            "mcmc_refine_start_iter": 12619,
            "mcmc_refine_stop_iter": 12620,
        }
    )
    config["default_strategy"]["refine_scale2d_stop_iter"] = 12620
    config["geometry_regularization"].update(
        {
            "opacity_sparsity_weight": 0.0,
            "scale_upper_weight": 0.0,
            "anisotropy_weight": 0.0,
            "screen_clip_enabled": True,
            "max_screen_fraction": 0.0035,
            "screen_clip_hardness": 1.10,
            "screen_clip_opacity_bump": 0.0,
            "max_world_size_m": 0.2,
        }
    )
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "v92_wall_door_roof_roi_joint_gate",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v92_projected_footprint_repair"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 12600,
        "target_completed_steps": 12620,
        "source_gaussian_count": source_count,
        "contract": {
            "topology_events": False,
            "means_scales_quats_lr_zero": True,
            "screen_clip_fraction": 0.0035,
            "screen_clip_hardness": 1.10,
            "max_world_size_m": 0.2,
            "additional_steps": 20,
        },
    }
    gate["adaptive_growth"] = {
        "profile": "v92_projected_footprint_repair_step12620",
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
    print("V92: 20-step projected-footprint repair, no topology or geometry LR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
