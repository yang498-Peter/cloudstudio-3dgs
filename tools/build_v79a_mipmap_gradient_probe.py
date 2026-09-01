"""Build V79a: record recovered MipMap gradient statistics in parallel."""

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
SOURCE = BASE / "v78b_second_detail_cycle" / "tile0_pre1400"
TARGET = BASE / "v79a_mipmap_gradient_probe" / "tile0_boundary1502"
TILE_INPUTS = (
    BASE
    / "tile_training_inputs_lidar_4tile_v73"
    / "tile_inputs_manifest.json"
)


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
    tile_inputs = _read(TILE_INPUTS)
    tile = next(item for item in tile_inputs["tiles"] if int(item["tile_id"]) == 0)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 1399:
        raise ValueError("V79a requires the exact V78b step-1399 checkpoint")

    config.update(
        {
            "run_id": "snow-tile0-v79a-mipmap-gradient-probe-boundary1502",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 1502,
            "checkpoint_every": 1499,
            "checkpoint_keep_every": 1499,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "gradient_statistics_profile": "mipmap_radius_weighted_probe_v1",
            "gradient_tile_core_box": tile["core_box"],
            "gradient_tile_outside_attenuation": 0.1,
        }
    )
    config["default_strategy"] = strategy
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "mipmap_gradient_distribution_review",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v79a_mipmap_gradient_probe"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 1399,
        "controlled_stop_completed_steps": 1502,
        "tile_inputs_manifest_sha256": _sha256_file(TILE_INPUTS),
        "gradient_statistics_profile": "mipmap_radius_weighted_probe_v1",
        "image_scale_formula": "0.5*max(1600,width,height)",
        "footprint_weight": "l2_raster_radius_px",
        "tile_core_box": tile["core_box"],
        "tile_outside_attenuation": 0.1,
        "growth_control": "unchanged_legacy_probe_only",
    }
    gate["adaptive_growth"] = dict(gate["adaptive_growth"])
    gate["adaptive_growth"].update(
        {
            "profile": "v79a_mipmap_gradient_probe_boundary1502",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 1502,
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
    print("V79a: MipMap-equivalent gradient probe, control unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
