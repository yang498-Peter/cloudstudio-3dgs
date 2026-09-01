"""Build the signed V86 Tile_0 evidence-calibrated step-502 boundary.

V86 starts from the signed dense Tile_0 initialization.  It keeps recovered
Snow/type-2 gradient and world-scale semantics, but replaces the destructive
immediate Cull/reset interaction with the narrow quality window established by
V65h, V83, and the failed V85/V85b adversarial probe.
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
TARGET = BASE / "v86_evidence_calibrated" / "tile0_boundary502"


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
            "run_id": "snow-tile0-v86-evidence-calibrated-boundary502",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": None,
            "controlled_stop_after_steps": 502,
            "checkpoint_every": 500,
            "checkpoint_keep_every": 12480,
            # Dense metric geometry is intentionally active from step 1 in this
            # CloudStudio quality arm; this is not labelled vendor bit-exact.
            "competitor_loss_schedule_enabled": False,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "grow_grad2d": 0.00015,
            "growth_min_opacity": None,
            "split_scale_m": 0.2,
            "prune_scale_m": 0.2,
            "prune_opa": 0.05,
            "prune_opa_late": 0.05,
            "prune_scale2d": 0.15,
            "reset_every": 3000,
            "reset_opacity_cap": 0.2,
            "absgrad": False,
            "revised_opacity": False,
            "detail_split_policy": "lidar_surface_screen_detail_aggressive",
            "detail_split_scale_m": 0.01,
            "detail_split_screen_radius": 0.0025,
            "opacity_cull_policy": "observation_aware",
            "opacity_cull_min_observations": 64,
            "opacity_cull_consecutive_events": 2,
            "opacity_cull_grace_after_reset_steps": 200,
            "opacity_cull_max_fraction": 0.02,
            "opacity_cull_priority": "lowest_opacity",
            "vendor_cull_warmup_profile": "compatibility_uniform_0p05",
            "vendor_capacity_cull_profile": "exact_relaxed_at_cap",
            "vendor_opacity_reset_profile": "deferred_every3000_compatibility",
            "cloudstudio_lifecycle_extension_profile": (
                "observation_cull_v2_conservative"
            ),
            "gradient_statistics_profile": "mipmap_radius_weighted_v1",
            "discard_accumulated_gradient_steps": [],
        }
    )
    config["default_strategy"] = strategy
    regularization = dict(config["geometry_regularization"])
    regularization.update(
        {
            "opacity_sparsity_weight": 0.0025,
            "opacity_sparsity_scope": "visible_current_view",
            "max_world_size_m": 0.2,
        }
    )
    config["geometry_regularization"] = regularization
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_BOUNDARY_READY",
            "training_allowed": True,
            "next_required_stage": "v86_step502_joint_quality_gate",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v86_evidence_calibrated_boundary"] = {
        "source_route": "V65h + V83 + V85/V85b adversarial evidence",
        "vendor_bit_exact_claimed": False,
        "initialization_mode": "signed_dense_tile0_from_scratch",
        "growth_gradient_threshold": 0.00015,
        "detail_split_scale_m": 0.01,
        "detail_split_screen_radius": 0.0025,
        "opacity_cull_max_fraction_per_event": 0.02,
        "boundary_cull_profile": "signed_observation_cull_v2_conservative",
        "opacity_reset_every": 3000,
        "capacity_cap": int(config["cap_max"]),
        "controlled_stop_completed_steps": 502,
    }
    adaptive = dict(gate.get("adaptive_growth", {}))
    for key in (
        "resume_checkpoint_sha256",
        "resume_source_trainer_config_sha256",
        "resume_allowed_lineage_differences",
    ):
        adaptive.pop(key, None)
    adaptive.update(
        {
            "profile": "v86_evidence_calibrated_boundary502",
            "stage": "boundary",
            "tile_id": 0,
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 502,
            "capacity_cap": int(config["cap_max"]),
            "mcmc_allowed": False,
            "warm_start_checkpoint_sha256": None,
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
    print("V86: dense Tile_0, step-502 evidence-calibrated boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
