"""Build the V73b Tile_0 growth-calibration boundary.

V73b keeps the recovered 500-step warm-up and 100-step lifecycle cadence, but
calibrates the two mechanisms that V73a proved unsafe on the real snow tile:
screen-aware detail splitting and a non-destructive first cull threshold.
It is a signed CloudStudio enhancement arm, not a claim of bit-exact vendor
behavior.
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
TARGET = BASE / "v73b_growth_calibration" / "tile0_boundary602"


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
            "run_id": "snow-tile0-v73b-growth-calibration-boundary602",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "controlled_stop_after_steps": 602,
            "checkpoint_every": 600,
            "checkpoint_keep_every": 600,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "grow_grad2d": 0.0001,
            "detail_split_policy": "lidar_surface_screen_detail",
            "detail_split_scale_m": 0.02,
            "detail_split_screen_radius": 0.0035,
            "revised_opacity": True,
            "prune_opa": 0.05,
            "prune_opa_late": 0.05,
            "vendor_cull_warmup_profile": "compatibility_uniform_0p05",
        }
    )
    config["default_strategy"] = strategy
    config = _sign_config(config)

    gate = copy.deepcopy(source_gate)
    gate.pop("gate_manifest_sha256", None)
    gate["next_required_stage"] = "growth_calibration_boundary_evaluation"
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v73b_growth_calibration"] = {
        "source_run_id": source_config["run_id"],
        "source_boundary_gaussian_count": 1806824,
        "source_clone_count": 57787,
        "source_split_parent_count": 0,
        "source_cull_count": 1102874,
        "source_cull_fraction_of_initial": 1102874 / 2851911,
        "plain_gradient_threshold": 0.0001,
        "detail_split_scale_m": 0.02,
        "detail_split_screen_radius": 0.0035,
        "early_cull_opacity": 0.05,
        "controlled_stop_after_steps": 602,
        "evidence_boundary": (
            "CloudStudio calibration arm; the recovered vendor baseline remains "
            "1.5e-4 gradient and 0.10 early opacity cull"
        ),
    }
    gate["adaptive_growth"] = {
        "profile": "v73b_screen_detail_safe_cull_calibration",
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
        "V73b: warm-up=500 lifecycle=100 boundary=602 "
        "grad=1e-4 screen-detail-split cull=0.05"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
