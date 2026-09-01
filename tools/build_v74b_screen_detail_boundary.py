"""Build V74b: V74a death control plus one screen-aware detail Split arm."""

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
TARGET = BASE / "v74b_screen_detail" / "tile0_boundary602"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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

    config = copy.deepcopy(source_config)
    config.update(
        {
            "run_id": "snow-tile0-v74b-screen-detail-boundary602",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "detail_split_policy": "lidar_surface_screen_detail",
            "detail_split_scale_m": 0.02,
            "detail_split_screen_radius": 0.0035,
            "revised_opacity": True,
        }
    )
    config["default_strategy"] = strategy
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate["next_required_stage"] = "screen_detail_boundary_evaluation"
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v74b_screen_detail"] = {
        "source_run_id": source_config["run_id"],
        "controlled_stop_after_steps": 602,
        "algorithmic_changes": ["screen_aware_detail_split_only"],
        "unchanged_death_controller": {
            "profile": "observation_cull_v1",
            "minimum_observations": 64,
            "consecutive_low_opacity_events": 2,
            "grace_after_reset_steps": 200,
            "maximum_fraction_per_event": 0.05,
        },
        "detail_split": {
            "world_scale_m": 0.02,
            "normalized_screen_radius": 0.0035,
            "revised_opacity": True,
        },
        "evidence_boundary": (
            "CloudStudio detail enhancement; all inputs, sampling, losses, "
            "gradient threshold, and Cull controller match V74a"
        ),
    }
    gate["adaptive_growth"] = {
        "profile": "v74b_observation_cull_screen_detail",
        "stage": "boundary",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": 602,
        "mcmc_allowed": False,
        "capacity_cap": int(config["cap_max"]),
        "warm_start_checkpoint_sha256": None,
    }
    gate = sign_gate(gate)

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V74b: V74a Cull + screen-aware 2 cm detail Split, stop=602")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
