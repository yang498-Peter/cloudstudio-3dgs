"""Build V84 A/B: vendor all-surface opacity drain versus visible-only control.

Both arms resume the exact V83a step-3299 checkpoint, discard the inherited
gradient accumulator at step 3300, and stop at step 3399.  No topology event
can run in this interval, so the only changed training variable is the scope of
the continuous opacity-mean regularizer.
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
SOURCE = BASE / "v83a_sky_clean_mesh_guided_detail" / "tile0_pre3300"
TARGET_ROOT = BASE / "v84_opacity_scope_revisit_ab"


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


def _build_arm(
    source_config: dict,
    source_gate: dict,
    checkpoint: Path,
    *,
    arm: str,
    opacity_scope: str,
    source_trainer_config_sha256: str,
) -> Path:
    target = TARGET_ROOT / arm
    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": f"snow-tile0-v84-{arm}-opacity-scope-step3399",
            "output_dir": str(target / "training_ewa"),
            "mipmap_pipeline_gate": str(target / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 3399,
            "checkpoint_every": 3399,
            "checkpoint_keep_every": 12480,
        }
    )
    regularization = dict(config["geometry_regularization"])
    regularization["opacity_sparsity_weight"] = 0.01
    regularization["opacity_sparsity_scope"] = opacity_scope
    config["geometry_regularization"] = regularization
    strategy = dict(config["default_strategy"])
    strategy["discard_accumulated_gradient_steps"] = [3300]
    config["default_strategy"] = strategy
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "opacity_scope_alpha_population_joint_gate",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})[f"v84_{arm}_opacity_scope_revisit"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 3299,
        "opacity_sparsity_weight": 0.01,
        "opacity_sparsity_scope": opacity_scope,
        "view_sampling_mode": "fisher_yates_without_replacement_per_epoch",
        "discarded_inherited_gradient_step": 3300,
        "first_possible_next_lifecycle_step": 3400,
        "controlled_stop_completed_steps": 3399,
        "topology_events_in_probe": 0,
        "purpose": (
            "isolate whether continuous vendor-strength opacity drain is stable "
            "under the signed no-replacement and dense-geometry supervision"
        ),
    }
    gate["adaptive_growth"] = dict(gate["adaptive_growth"])
    gate["adaptive_growth"].update(
        {
            "profile": f"v84_{arm}_opacity_scope_revisit_step3399",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 3399,
            "resume_checkpoint_sha256": _sha256_file(checkpoint),
            "resume_source_trainer_config_sha256": source_trainer_config_sha256,
        }
    )
    gate = sign_gate(gate)

    _write(target / "trainer.config.json", config)
    _write(target / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    return target


def main() -> int:
    source_config = _read(SOURCE / "trainer.config.json")
    source_gate = _read(SOURCE / "training_gate.json")
    verify_gate(source_gate)
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 3299:
        raise ValueError("V84 requires the exact V83a step-3299 checkpoint")
    source_trainer_config_sha256 = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    if len(source_trainer_config_sha256) != 64:
        raise ValueError("V83a checkpoint has no valid trainer config identity")

    control = _build_arm(
        source_config,
        source_gate,
        checkpoint,
        arm="visible_control",
        opacity_scope="visible_current_view",
        source_trainer_config_sha256=source_trainer_config_sha256,
    )
    vendor = _build_arm(
        source_config,
        source_gate,
        checkpoint,
        arm="vendor_all_surface",
        opacity_scope="all",
        source_trainer_config_sha256=source_trainer_config_sha256,
    )
    print(control / "trainer.config.json")
    print(vendor / "trainer.config.json")
    print("V84: identical 100-step no-topology opacity-scope A/B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
