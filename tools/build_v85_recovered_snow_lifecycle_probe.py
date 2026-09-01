"""Build the V85 Snow/type-2 recovered lifecycle boundary probe.

The source is the original V73 competitor-equivalent boundary config.  V85
changes only evidence-corrected semantics that were previously mistranscribed:
the dead opacity candidate expression is not applied, split opacity is copied,
reset remains every 300 steps, and oversized rows use multiplicative shrink.
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
TARGET = BASE / "v85_recovered_snow_lifecycle" / "tile0_boundary502"


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
    config = _read(SOURCE / "trainer.config.json")
    gate = _read(SOURCE / "training_gate.json")
    verify_gate(gate)

    config.update(
        {
            "run_id": "snow-tile0-v85-recovered-snow-lifecycle-boundary502",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "controlled_stop_after_steps": 502,
            "checkpoint_every": 500,
            "checkpoint_keep_every": 12480,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "growth_min_opacity": None,
            "reset_every": 300,
            "vendor_opacity_reset_profile": "exact_every300",
            "revised_opacity": False,
            "detail_split_policy": "vendor_0_2m",
            "opacity_cull_policy": "immediate",
            "cloudstudio_lifecycle_extension_profile": "disabled",
        }
    )
    config["default_strategy"] = strategy
    regularization = dict(config["geometry_regularization"])
    regularization.update(
        {
            "opacity_sparsity_weight": 0.01,
            "opacity_sparsity_scope": "all",
            "max_world_size_m": 0.2,
        }
    )
    config["geometry_regularization"] = regularization
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "recovered_lifecycle_boundary_joint_gate",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v85_recovered_snow_lifecycle"] = {
        "dll_sha256": (
            "A910B39DBAD956DC35E9E436ACFD0FB8D92364E03BB44B0401CBEF6BCB8D492E"
        ),
        "snow_type2_split_scale_m": 0.2,
        "snow_high_grow_grad2d": 0.00015,
        "snow_type2_prune_scale2d": 0.15,
        "growth_opacity_expression_consumed": False,
        "reset_every_steps": 300,
        "split_revised_opacity": False,
        "world_shrink_factor": 0.8,
        "controlled_stop_completed_steps": 502,
        "purpose": (
            "measure first Split/Clone/Cull event after correcting only the "
            "recovered Snow/type-2 lifecycle semantics"
        ),
    }
    gate["adaptive_growth"] = dict(gate["adaptive_growth"])
    gate["adaptive_growth"].update(
        {
            "profile": "v85_recovered_snow_type2_boundary502",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 502,
        }
    )
    gate = sign_gate(gate)

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V85: Snow High/type-2 recovered first lifecycle boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
