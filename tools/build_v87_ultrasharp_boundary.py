"""Build V87: two ultrasharp detail cycles from the V86b step-1002 gate."""

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
SOURCE = BASE / "v86_evidence_calibrated" / "tile0_review1002"
TARGET = BASE / "v87_ultrasharp_detail" / "tile0_boundary1202"


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
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 1002:
        raise ValueError("V87 requires the exact V86b step-1002 checkpoint")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    if len(source_trainer_sha) != 64:
        raise ValueError("V86b checkpoint has no signed trainer identity")

    config.update(
        {
            "run_id": "snow-tile0-v87-ultrasharp-detail-boundary1202",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 1202,
            "checkpoint_every": 1202,
            "checkpoint_keep_every": 12480,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "detail_split_policy": "lidar_surface_screen_detail_ultrasharp",
            "detail_split_scale_m": 0.008,
            "detail_split_screen_radius": 0.0025,
        }
    )
    config["default_strategy"] = strategy
    normal = dict(config["lidar_normal_alignment"])
    normal.update(
        {
            "weight_flatten": 0.05,
            "weight_point_to_plane": 0.01,
            "flatten_ratio_target": 0.08,
            "point_to_plane_huber_delta_m": 0.02,
        }
    )
    config["lidar_normal_alignment"] = normal
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_BOUNDARY_READY",
            "training_allowed": True,
            "next_required_stage": "v87_step1202_joint_sharpness_gate",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v87_ultrasharp_boundary"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 1002,
        "controlled_stop_completed_steps": 1202,
        "texture_density_gradient_rho": 0.42307868198276705,
        "highest_texture_quintile_delta_median": 0.0,
        "source_max_axis_p50_m": 0.009232176467776299,
        "algorithmic_changes": {
            "detail_split_scale_m": [0.01, 0.008],
            "flatten_ratio_target": [0.1, 0.08],
            "weight_flatten": [0.02, 0.05],
            "weight_point_to_plane": [0.0, 0.01],
        },
        "unchanged_controls": [
            "plain_projected_gradient_0p00015",
            "screen_radius_0p0025",
            "observation_cull_v2_conservative",
            "capacity_4278000",
            "sampling_and_photometric_losses",
            "black_surface_and_independent_sky",
        ],
    }
    adaptive = dict(gate["adaptive_growth"])
    adaptive.update(
        {
            "profile": "v87_ultrasharp_detail_boundary1202",
            "stage": "review",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 1202,
            "resume_checkpoint_sha256": _sha256_file(checkpoint),
            "resume_source_trainer_config_sha256": source_trainer_sha,
            "resume_allowed_lineage_differences": [],
        }
    )
    gate["adaptive_growth"] = adaptive
    gate = sign_gate(gate)

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V87: 8 mm detail Split + 0.08 thin-disk target, stop at 1202")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
