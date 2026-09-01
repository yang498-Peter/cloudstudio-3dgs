"""Build the signed V74a Tile_0 observation-aware Cull boundary.

V74a starts from the same deterministic Tile_0 input as V73a and keeps the
recovered snow growth semantics unchanged.  Its only algorithmic change is a
bounded, observation-aware opacity death controller.  The controlled stop at
step 602 captures two lifecycle events without authorizing a long run.
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
SOURCE = BASE / "v73_four_tile_competitor_equivalent" / "tile0_boundary502"
TARGET = BASE / "v74a_observation_cull" / "tile0_boundary602"


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
            "run_id": "snow-tile0-v74a-observation-cull-boundary602",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "controlled_stop_after_steps": 602,
            "checkpoint_every": 500,
            "checkpoint_keep_every": 500,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            # Keep the recovered snow lifecycle and split semantics exact.
            "grow_grad2d": 0.00015,
            "detail_split_policy": "vendor_0_2m",
            "revised_opacity": False,
            "prune_opa": 0.10,
            "prune_opa_late": 0.05,
            "vendor_cull_warmup_profile": "exact_0p10_to_0p05",
            "vendor_opacity_reset_profile": "exact_every300",
            # Product arm: require persistent evidence before opacity death.
            "opacity_cull_policy": "observation_aware",
            "cloudstudio_lifecycle_extension_profile": "observation_cull_v1",
            "opacity_cull_min_observations": 64,
            "opacity_cull_consecutive_events": 2,
            "opacity_cull_grace_after_reset_steps": 200,
            "opacity_cull_max_fraction": 0.05,
            "opacity_cull_priority": "lowest_opacity",
            "opacity_cull_local_min_accumulated_alpha": 0.0,
            # This only activates when the signed absolute cap is reached.
            "vendor_capacity_cull_profile": "exact_relaxed_at_cap",
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
            "next_required_stage": "observation_cull_boundary_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v74a_observation_cull"] = {
        "source_run_id": source_config["run_id"],
        "controlled_stop_after_steps": 602,
        "algorithmic_changes": ["opacity_death_controller_only"],
        "unchanged_growth_contract": {
            "absgrad": False,
            "grow_grad2d": 0.00015,
            "growth_min_opacity": 0.15,
            "split_scale_m": 0.2,
            "detail_split_policy": "vendor_0_2m",
            "revised_opacity": False,
            "lifecycle_execution_order": "pre_optimizer_vendor",
        },
        "death_controller": {
            "policy": "observation_aware",
            "minimum_observations": 64,
            "consecutive_low_opacity_events": 2,
            "grace_after_reset_steps": 200,
            "maximum_fraction_per_event": 0.05,
            "capacity_profile": "exact_relaxed_at_cap",
        },
        "evidence_boundary": (
            "CloudStudio product enhancement; not a claim of bit-exact vendor "
            "Cull semantics"
        ),
    }
    gate["adaptive_growth"] = {
        "profile": "v74a_observation_aware_death_controller",
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
    print(
        "V74a: clean snow growth, observation-aware Cull, "
        "events=500/600, stop=602"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
