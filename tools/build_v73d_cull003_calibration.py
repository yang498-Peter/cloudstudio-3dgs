"""Build V73d: one-variable step-700 opacity-Cull calibration.

The run resumes the exact V73b step-602 checkpoint and differs from V73c only
by lowering both opacity-Cull thresholds from 0.05 to 0.03.  It retains the
step-699 pre-lifecycle checkpoint and stops at 702 for a controlled comparison.
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
TARGET = BASE / "v73d_cull003_calibration" / "tile0_review702"


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
        raise ValueError("V73d requires the exact V73b step-602 checkpoint")
    source_trainer_sha = str(payload.get("identity", {}).get("trainer_config_sha256", ""))
    if len(source_trainer_sha) != 64:
        raise ValueError("source checkpoint trainer-config identity is invalid")

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v73d-cull003-calibration-review702",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 702,
            "checkpoint_every": 699,
            "checkpoint_keep_every": 699,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update({"prune_opa": 0.03, "prune_opa_late": 0.03})
    config["default_strategy"] = strategy
    config = _sign_config(config)

    checkpoint_sha = _sha256_file(checkpoint)
    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "cull003_boundary_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v73d_cull003_calibration"] = {
        "source_checkpoint_sha256": checkpoint_sha,
        "source_completed_steps": 602,
        "control_run": "v73c_post_reset_settle_isolation",
        "control_prune_opacity": 0.05,
        "candidate_prune_opacity": 0.03,
        "all_other_training_parameters_unchanged": True,
        "pre_lifecycle_checkpoint_completed_steps": 699,
        "post_lifecycle_controlled_stop_completed_steps": 702,
    }
    gate["adaptive_growth"] = {
        "profile": "v73d_cull003_single_variable_calibration",
        "stage": "review",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": 702,
        "mcmc_allowed": False,
        "capacity_cap": int(config["cap_max"]),
        "resume_checkpoint_sha256": checkpoint_sha,
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
    print("V73d: resume=602 checkpoint=699 stop=702 prune_opa=0.03")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
