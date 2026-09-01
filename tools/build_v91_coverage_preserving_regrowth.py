"""Build the signed V91b Tile_0 coverage-preserving regrowth boundary.

V88b proved that shrinking the mature V87 scale tail after densification had
already stopped improves depth but exposes real wall/door/roof coverage holes.
V91b therefore resumes the higher-coverage V87 checkpoint, performs exactly one
screen-aware detail growth event at step 12500, disables opacity/scale culling
for this boundary, and settles for 100 further steps.  It is a short visual and
geometry gate, not a production long run.
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
SOURCE = BASE / "v87_ultrasharp_detail" / "tile0_full12480"
TARGET = BASE / "v91b_coverage_preserving_regrowth" / "tile0_boundary12600"


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
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "step_00012480.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 12480:
        raise ValueError("V91b requires the exact V87 step-12480 checkpoint")
    source_count = int(payload["params"]["means"].shape[0])
    if source_count != 4_191_083:
        raise ValueError(f"unexpected V87 Gaussian count: {source_count}")
    source_trainer_sha = str(
        payload.get("identity", {}).get("trainer_config_sha256", "")
    )
    if len(source_trainer_sha) != 64:
        raise ValueError("V87 checkpoint has no signed trainer identity")

    config.update(
        {
            "run_id": "snow-tile0-v91b-coverage-preserving-regrowth-boundary12600",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "max_steps": 12600,
            "checkpoint_every": 100,
            "checkpoint_keep_every": 12600,
            "view_sampling_mode": "with_replacement",
            "cap_max": 4_600_000,
            "learning_rates": {
                "means": 0.000005,
                "scales": 0.002,
                "quats": 0.0005,
                "opacities": 0.01,
                "colors": 0.001,
            },
            "mcmc_refine_start_iter": 12480,
            "mcmc_refine_every": 100,
            # The runtime gate is strict: step < stop_iter. 12501 admits only
            # the divisible step 12500 before the 12600 boundary.
            "mcmc_refine_stop_iter": 12501,
            "lidar_alpha_weight": 0.10,
            "lidar_alpha_target": 0.95,
            "lidar_alpha_dilation_radius_px": 3,
            "mesh_alpha_weight": 0.0,
            "mesh_alpha_target": 0.95,
        }
    )
    config.pop("controlled_stop_after_steps", None)

    strategy = config["default_strategy"]
    strategy.update(
        {
            "exact_mipmap_lifecycle": False,
            "lifecycle_execution_order": "post_optimizer_gsplat",
            "grow_grad2d": 0.00015,
            "growth_min_opacity": 0.10,
            "split_scale_m": 0.2,
            "prune_scale_m": 1.0,
            "prune_opa": 0.0,
            "prune_opa_late": 0.0,
            "prune_switch_step": 12600,
            "prune_scale2d": 1.0,
            "refine_scale2d_stop_iter": 12501,
            "reset_every": 1_000_000,
            "reset_opacity_cap": 0.2,
            "absgrad": False,
            "revised_opacity": False,
            "capacity_conserving_clone_opacity": False,
            "detail_split_policy": "lidar_surface_screen_detail_ultrasharp",
            "detail_split_scale_m": 0.008,
            "detail_split_screen_radius": 0.0025,
            "opacity_cull_policy": "immediate",
            "opacity_cull_min_observations": 1_000_000,
            "opacity_cull_consecutive_events": 2,
            "opacity_cull_grace_after_reset_steps": 200,
            # Cull is disabled by zero opacity threshold plus unreachable
            # world/screen thresholds. Keep the fraction field runtime-valid.
            "opacity_cull_max_fraction": 1.0,
            "vendor_capacity_cull_profile": "disabled",
            "cloudstudio_lifecycle_extension_profile": "disabled",
        }
    )

    config["geometry_regularization"].update(
        {
            "opacity_sparsity_weight": 0.0,
            "scale_upper_weight": 0.0,
            "anisotropy_weight": 0.0,
            "max_scale_ratio_to_reference": 8.0,
            "max_anisotropy": 256.0,
            # Preserve V87 low-texture coverage. High-gradient broad splats are
            # reduced by detail Split instead of a global 5 cm shrink trigger.
            "max_world_size_m": 0.2,
        }
    )
    config["lidar_normal_alignment"].update(
        {
            "weight_align": 0.02,
            "weight_flatten": 0.05,
            "weight_point_to_plane": 0.01,
            "flatten_mode": "tangent_ratio_shortest_only",
            "flatten_ratio_target": 0.08,
        }
    )
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "v91_wall_door_roof_roi_joint_gate",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v91b_coverage_preserving_regrowth"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 12480,
        "target_completed_steps": 12600,
        "source_gaussian_count": source_count,
        "contract": {
            "single_growth_event_step": 12500,
            "settle_steps_after_growth": 100,
            "capacity_cap": 4_600_000,
            "opacity_cull_disabled": True,
            "world_scale_threshold_m": 0.2,
            "detail_split_scale_m": 0.008,
            "detail_split_screen_radius": 0.0025,
            "wall_door_roof_visual_gate_required": True,
        },
    }
    gate["adaptive_growth"] = {
        "profile": "v91b_coverage_preserving_regrowth_boundary12600",
        "stage": "continuation",
        "tile_id": 0,
        "run_id": config["run_id"],
        "controlled_stop_after_steps": None,
        "mcmc_allowed": False,
        "capacity_cap": 4_600_000,
        "resume_checkpoint_sha256": _sha256_file(checkpoint),
        "resume_source_trainer_config_sha256": source_trainer_sha,
        "resume_allowed_lineage_differences": ["scale_calibration_sha256"],
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
        "V91b: V87 step12480 -> one coverage-preserving growth event at 12500 "
        "-> settle to 12600; no opacity/scale cull"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
