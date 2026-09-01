"""Resume V74a without parameter changes and isolate its step-700 event."""

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
SOURCE = BASE / "v74a_observation_cull" / "tile0_boundary602"
TARGET = BASE / "v74c_observation_cull_settle" / "tile0_review702"


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
    if int(payload.get("step", -1)) != 602:
        raise ValueError("V74c requires the exact V74a step-602 checkpoint")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    telemetry = payload.get("training_state", {}).get("mcmc_telemetry", {})
    events = telemetry.get("events", [])
    if [int(event.get("step", -1)) for event in events] != [500, 600]:
        raise ValueError("V74a lifecycle evidence is not the expected [500, 600]")
    if events[-1].get("classic_lifecycle", {}).get("opacity_reset") is not True:
        raise ValueError("V74a step-600 lifecycle did not record the required reset")

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v74c-observation-cull-settle-review702",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 702,
            "checkpoint_every": 699,
            "checkpoint_keep_every": 699,
        }
    )
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "observation_cull_settle_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v74c_settle_isolation"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 602,
        "pre_lifecycle_checkpoint_completed_steps": 699,
        "post_lifecycle_controlled_stop_completed_steps": 702,
        "parameter_changes": "none",
    }
    gate["adaptive_growth"] = {
        "profile": "v74c_observation_cull_settle_isolation",
        "stage": "review",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": 702,
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
    print("V74c: resume=602 checkpoint=699 stop=702 parameter_changes=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
