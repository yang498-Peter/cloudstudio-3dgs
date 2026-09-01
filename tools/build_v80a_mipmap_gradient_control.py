"""Build V80a: use the recovered MipMap gradient as the growth controller."""

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
SOURCE = BASE / "v79a_mipmap_gradient_probe" / "tile0_boundary1502"
TARGET = BASE / "v80a_mipmap_gradient_control" / "tile0_pre1600"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sign_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("config_manifest_sha256", None)
    result["config_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    config = _read(SOURCE / "trainer.config.json")
    gate = _read(SOURCE / "training_gate.json")
    verify_gate(gate)
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "step_00001499.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 1499:
        raise ValueError("V80a requires the exact V79a pre-event step-1499 checkpoint")

    config.update(
        {
            "run_id": "snow-tile0-v80a-mipmap-gradient-control-pre1600",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 1599,
            "checkpoint_every": 1599,
            "checkpoint_keep_every": 1599,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "gradient_statistics_profile": "mipmap_radius_weighted_v1",
            "grow_grad2d": 0.00015,
        }
    )
    config["default_strategy"] = strategy
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "mipmap_gradient_control_settle_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config["config_manifest_sha256"]
    gate.setdefault("evidence", {})["v80a_mipmap_gradient_control"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 1499,
        "controlled_stop_completed_steps": 1599,
        "gradient_statistics_profile": "mipmap_radius_weighted_v1",
        "grow_grad2d": 0.00015,
        "image_scale_formula": "0.5*max(1600,width,height)",
        "footprint_weight": "l2_raster_radius_px",
        "tile_outside_attenuation": 0.1,
        "purpose": "one recovered-gradient event plus a full 99-step settle window",
    }
    gate["adaptive_growth"] = dict(gate["adaptive_growth"])
    gate["adaptive_growth"].update(
        {
            "profile": "v80a_mipmap_gradient_control_pre1600",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 1599,
            "resume_checkpoint_sha256": _sha256_file(checkpoint),
            "resume_source_trainer_config_sha256": str(
                payload.get("identity", {}).get("trainer_config_sha256", "")
            ),
        }
    )
    gate = sign_gate(gate)

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V80a: recovered gradient controls step 1500, settle through step 1599")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
