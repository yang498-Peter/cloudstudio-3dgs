"""Build V75b: continue the accepted V75a surface-detail policy to step 1002."""

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
SOURCE = BASE / "v75a_surface_detail" / "tile0_boundary802"
TARGET = BASE / "v75b_surface_detail_settle" / "tile0_review1002"


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
    if int(payload.get("step", -1)) != 802:
        raise ValueError("V75b requires the exact V75a step-802 checkpoint")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    last_event = (
        payload.get("training_state", {})
        .get("mcmc_telemetry", {})
        .get("events", [])[-1]
    )
    lifecycle = last_event.get("classic_lifecycle", {})
    surface_guard = lifecycle.get("surface_birth_guard", {})
    if (
        int(last_event.get("step", -1)) != 800
        or int(lifecycle.get("split_parent_count", 0)) <= 0
        or int(surface_guard.get("newborns", 0)) <= 0
    ):
        raise ValueError("V75a did not record a real guarded detail Split event")

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v75b-surface-detail-settle-review1002",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 1002,
            "checkpoint_every": 999,
            "checkpoint_keep_every": 999,
        }
    )
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "surface_detail_settle_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v75b_surface_detail_settle"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 802,
        "pre_step1000_checkpoint_completed_steps": 999,
        "post_lifecycle_controlled_stop_completed_steps": 1002,
        "parameter_changes": "none",
        "required_source_split_parent_count": int(
            lifecycle["split_parent_count"]
        ),
        "required_source_guarded_newborn_count": int(surface_guard["newborns"]),
    }
    gate["adaptive_growth"] = {
        "profile": "v75b_surface_detail_settle_isolation",
        "stage": "review",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": 1002,
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
    print("V75b: resume=802 checkpoint=999 stop=1002 parameter_changes=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
