"""Build the signed V85b continuation through reset and second lifecycle.

The source is the exact V85 step-502 checkpoint.  No training parameter is
changed: step 600 captures the recovered vendor reset/lifecycle intersection,
and step 700 captures the following grow/cull event.
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
SOURCE = BASE / "v85_recovered_snow_lifecycle" / "tile0_boundary502"
TARGET = BASE / "v85b_recovered_snow_lifecycle" / "tile0_review702"


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
    if int(payload.get("step", -1)) != 502:
        raise ValueError("V85b requires the exact V85 step-502 checkpoint")
    source_trainer_sha = str(payload.get("identity", {}).get("trainer_config_sha256", ""))
    if len(source_trainer_sha) != 64:
        raise ValueError("V85 checkpoint trainer-config identity is invalid")
    telemetry = payload.get("training_state", {}).get("mcmc_telemetry", {})
    events = telemetry.get("events", [])
    if [int(event.get("step", -1)) for event in events] != [500]:
        raise ValueError("V85b requires exactly the V85 step-500 lifecycle event")

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v85b-recovered-snow-lifecycle-review702",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 702,
            "checkpoint_every": 600,
            "checkpoint_keep_every": 600,
        }
    )
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "reset_and_second_lifecycle_joint_gate",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v85b_reset_second_lifecycle"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 502,
        "source_gaussian_count": int(payload["params"]["means"].shape[0]),
        "source_first_event_added": int(telemetry.get("total_added", -1)),
        "source_first_event_pruned": int(telemetry.get("total_pruned", -1)),
        "reset_event_step": 600,
        "second_growth_cull_step": 700,
        "controlled_stop_completed_steps": 702,
        "parameter_changes": "none",
    }
    gate["adaptive_growth"] = dict(gate["adaptive_growth"])
    gate["adaptive_growth"].update(
        {
            "profile": "v85b_recovered_snow_reset_second_lifecycle",
            "stage": "review",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 702,
            "resume_checkpoint_sha256": _sha256_file(checkpoint),
            "resume_source_trainer_config_sha256": source_trainer_sha,
            "resume_allowed_lineage_differences": [],
        }
    )
    gate = sign_gate(gate)

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V85b: exact resume 502, reset 600, second lifecycle 700, stop 702")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
