"""Build the signed V86b Tile_0 step-1002 controlled continuation."""

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
SOURCE = BASE / "v86_evidence_calibrated" / "tile0_boundary502"
TARGET = BASE / "v86_evidence_calibrated" / "tile0_review1002"


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
    if int(payload.get("step", -1)) != 502:
        raise ValueError("V86b requires the exact V86 completed-step-502 checkpoint")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    if len(source_trainer_sha) != 64:
        raise ValueError("V86 checkpoint has no signed trainer identity")

    config.update(
        {
            "run_id": "snow-tile0-v86b-evidence-calibrated-review1002",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 1002,
            "checkpoint_every": 1002,
            "checkpoint_keep_every": 12480,
        }
    )
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "v86b_step1002_joint_quality_gate",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v86b_review1002"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 502,
        "parameter_changes": [],
        "controlled_stop_completed_steps": 1002,
        "purpose": (
            "measure five additional evidence-calibrated lifecycle events without "
            "changing loss, gradient, split, cull, reset, or capacity semantics"
        ),
    }
    adaptive = dict(gate["adaptive_growth"])
    adaptive.update(
        {
            "profile": "v86b_evidence_calibrated_review1002",
            "stage": "review",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 1002,
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
    print("V86b: exact V86 continuation, completed-step boundary 1002")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
