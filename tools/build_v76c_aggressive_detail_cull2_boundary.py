"""Build V76c: repeat step 1200 with conservative two-percent Cull."""

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
SOURCE = BASE / "v76a_aggressive_detail" / "tile0_boundary1102"
TARGET = BASE / "v76c_aggressive_detail_cull2" / "tile0_boundary1202"


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
    if int(payload.get("step", -1)) != 1102:
        raise ValueError("V76c requires the exact V76a step-1102 checkpoint")
    last_event = (
        payload.get("training_state", {})
        .get("mcmc_telemetry", {})
        .get("events", [])[-1]
    )
    if int(last_event.get("step", -1)) != 1100:
        raise ValueError("V76a did not stop after the expected step-1100 event")

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v76c-aggressive-detail-cull2-boundary1202",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 1202,
            "checkpoint_every": 1199,
            "checkpoint_keep_every": 1199,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "vendor_opacity_reset_profile": "deferred_every3000_compatibility",
            "reset_every": 3000,
            "cloudstudio_lifecycle_extension_profile": (
                "observation_cull_v2_conservative"
            ),
            "opacity_cull_max_fraction": 0.02,
        }
    )
    config["default_strategy"] = strategy
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "cull2_step1200_boundary_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v76c_cull2_boundary"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 1102,
        "post_lifecycle_controlled_stop_completed_steps": 1202,
        "changes": {
            "vendor_opacity_reset_profile": "deferred_every3000_compatibility",
            "reset_every": 3000,
            "cloudstudio_lifecycle_extension_profile": (
                "observation_cull_v2_conservative"
            ),
            "opacity_cull_max_fraction": 0.02,
        },
        "controlled_variable": (
            "repeat the V76b boundary from the unchanged V76a checkpoint; "
            "only reduce the observation-aware Cull cap from 5% to 2%"
        ),
    }
    gate["adaptive_growth"] = {
        "profile": "v76c_aggressive_surface_detail_cull2_boundary",
        "stage": "review",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": 1202,
        "mcmc_allowed": False,
        "capacity_cap": int(config["cap_max"]),
        "resume_checkpoint_sha256": _sha256_file(checkpoint),
        "resume_source_trainer_config_sha256": str(
            payload.get("identity", {}).get("trainer_config_sha256", "")
        ),
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
    print("V76c: resume=1102 stop=1202 no reset, observation Cull cap=2%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
