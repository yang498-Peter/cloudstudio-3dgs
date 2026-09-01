"""Build, but do not launch, the signed Tile_1 mesh-first mainline candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mesh_geometry import verify_mesh_geometry_manifest
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
    sign_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
VIEW_COUNT = 374
STEPS = 20 * VIEW_COUNT
FIRST_MESH_DEPTH_COMPLETED_STEP = 5 * VIEW_COUNT + 2


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _signed(payload: dict, field: str) -> dict:
    result = copy.deepcopy(payload)
    result[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale-guard",
        action="store_true",
        help="enable the competitor-aligned 0.2 m world-size safety fuse",
    )
    args = parser.parse_args()
    variant = "v63b_scaleguard" if args.scale_guard else "v63a_crossview_mesh"
    protocol = BASE / f"{variant}_boundary1872"
    config_path = protocol / f"snow_tile1_{variant}_boundary1872.config.json"
    gate_path = protocol / "training_gate.json"
    mesh_root = BASE / "d0a_mesh_face4_tile1_crossview_v63a"
    mesh_path = mesh_root / "mesh_geometry_manifest.json"
    mesh_sha = verify_mesh_geometry_manifest(_read(mesh_path))

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived["bindings"] = dict(derived["bindings"])
    derived["bindings"]["mesh_geometry_tile1_manifest_sha256"] = mesh_sha
    derived.setdefault("evidence", {})["dense_geometry_experimental_override"] = {
        "scope": "Tile_1 cross-view-filtered mesh-first boundary candidate",
        "da2_admission": "BLOCKED_BY_V62_AB",
        "mesh_topology": "OPEN3D_BPA_CANDIDATE_NOT_VENDOR_CONFIRMED",
        "cross_view_filter": "PRODUCTION_FILTER_APPLIED",
    }
    derived = sign_gate(derived)

    config = _read(
        BASE
        / "fixed_topology_evaluation_v25a"
        / "snow_tile1_fixed_topology_a0_eval_v25a.json"
    )
    config.update(
        {
            "run_id": f"snow-tile1-{variant}-boundary1872",
            "output_dir": str(
                BASE / f"training_tile1_{variant}_boundary1872"
            ),
            "mipmap_pipeline_gate": str(gate_path),
            "mesh_geometry_manifest": str(mesh_path),
            "mesh_geometry_root": str(mesh_root),
            # V62 proved that the current DA2 path dominates and degrades RGB
            # and alpha.  Keep it explicitly off while retaining the recovered
            # vendor temporal envelope for mesh depth and normal.
            "mono_depth_manifest": None,
            "mono_depth_root": None,
            "da2_depth_weight": 0.0,
            "mesh_depth_weight": 0.5,
            "mesh_normal_weight": 0.05,
            "competitor_loss_schedule_enabled": True,
            "max_steps": STEPS,
            "checkpoint_every": VIEW_COUNT,
            "checkpoint_keep_every": VIEW_COUNT,
            # The recovered schedule uses a strict ``step > 5 * V`` test.
            # With zero-based steps, step 1871 is the first mesh-depth update;
            # stop after completed step 1872 so the boundary contains it.
            "controlled_stop_after_steps": FIRST_MESH_DEPTH_COMPLETED_STEP,
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": False,
            "config_manifest_sha256": None,
            "geometry_regularization": (
                {
                    "enabled": True,
                    "opacity_sparsity_weight": 0.0,
                    "scale_upper_weight": 0.0,
                    "anisotropy_weight": 0.0,
                    "max_scale_ratio_to_reference": 8.0,
                    "max_anisotropy": 10.0,
                    "screen_clip_enabled": False,
                    "max_screen_fraction": 0.15,
                    "screen_clip_hardness": 1.5,
                    "screen_clip_opacity_bump": 0.0,
                    "max_world_size_m": 0.2,
                }
                if args.scale_guard
                else {"enabled": False}
            ),
            "fixed_topology_schedule": {
                "enabled": False,
                "phase_a_steps": 0,
                "phase_b_steps": 0,
                "phase_b_geometry_lr_scale": 1.0,
                "phase_c_geometry_lr_scale": 1.0,
                "phase_b_range_weight_scale": 1.0,
                "phase_c_range_weight_scale": 1.0,
                "phase_b_normal_weight_scale": 1.0,
                "phase_c_normal_weight_scale": 1.0,
                "audit_steps": [],
            },
        }
    )
    unsigned = copy.deepcopy(config)
    unsigned.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    plan = _signed(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": 1,
            "view_count": VIEW_COUNT,
            "steps": {"total": STEPS},
            "arms": [
                {
                    "arm": "MESH_FIRST",
                    "config_manifest_sha256": config["config_manifest_sha256"],
                }
            ],
            "training_allowed": False,
            "adaptive_growth_remains_blocked_by": [
                "mesh_first_fixed_topology_promotion_not_yet_passed",
                "cross_view_mesh_boundary_joint_gate_not_yet_passed",
                "da2_v62_ab_failed_rgb_alpha_gate",
            ],
        },
        "evaluation_plan_sha256",
    )
    readiness = _signed(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_readiness_v1",
            "status": "FIXED_TOPOLOGY_EVALUATION_PREPARED",
            "training_allowed": False,
            "adaptive_growth_allowed": False,
            "upstream_gate_manifest_sha256": derived["gate_manifest_sha256"],
            "evaluation_plan_sha256": plan["evaluation_plan_sha256"],
            "evidence": {
                "directional_pass": True,
                "phase_a_geometry_frozen": True,
                "actual_merge_contract": "retain_full_halo",
                "halo_overlap_retained": True,
                "scope": (
                    "signed cross-view mesh boundary candidate; stop immediately "
                    "after the first active mesh-depth update"
                ),
            },
        },
        "readiness_sha256",
    )
    gate = advance_fixed_topology_evaluation_gate(
        derived, readiness, plan, {"MESH_FIRST": config}
    )
    _write(protocol / "derived_upstream_gate.json", derived)
    _write(protocol / "evaluation_plan.json", plan)
    _write(protocol / "readiness.json", readiness)
    _write(config_path, config)
    _write(gate_path, gate)
    TrainerConfig.from_dict(config).validate()
    print(config_path)
    print(gate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
