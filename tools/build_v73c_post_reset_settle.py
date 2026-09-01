"""Build the signed V73c post-reset settle and lifecycle-isolation run.

V73b stopped two completed steps after the step-600 vendor opacity reset.  A
single continuation captures step 699 before the next lifecycle and step 702
after it, so opacity recovery is not confused with Split/Clone/Cull effects.
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
SOURCE = BASE / "v73b_growth_calibration" / "tile0_boundary602"
TARGET = BASE / "v73c_post_reset_settle" / "tile0_review702"


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
        raise ValueError("V73c requires the exact V73b step-602 checkpoint")
    source_trainer_sha = str(payload.get("identity", {}).get("trainer_config_sha256", ""))
    if len(source_trainer_sha) != 64:
        raise ValueError("source checkpoint trainer-config identity is invalid")
    telemetry = payload.get("training_state", {}).get("mcmc_telemetry", {})
    events = telemetry.get("events", [])
    if [int(event.get("step", -1)) for event in events] != [500, 600]:
        raise ValueError("V73b lifecycle evidence is not the expected [500, 600]")
    if events[-1].get("classic_lifecycle", {}).get("opacity_reset") is not True:
        raise ValueError("V73b step-600 lifecycle did not record the required reset")

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v73c-post-reset-settle-review702",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 702,
            # 699 is the last saved state before the step-700 lifecycle event.
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
            "next_required_stage": "post_reset_settle_and_lifecycle_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v73c_post_reset_isolation"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 602,
        "pre_lifecycle_checkpoint_completed_steps": 699,
        "post_lifecycle_controlled_stop_completed_steps": 702,
        "source_refine_event_count": int(telemetry.get("refine_event_count", -1)),
        "source_total_added": int(telemetry.get("total_added", -1)),
        "source_total_pruned": int(telemetry.get("total_pruned", -1)),
        "parameter_changes": "none",
        "purpose": "separate opacity recovery from the next lifecycle event",
    }
    gate["adaptive_growth"] = {
        "profile": "v73c_post_reset_settle_isolation",
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
    print("V73c: resume=602 checkpoint=699 stop=702 parameter_changes=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
