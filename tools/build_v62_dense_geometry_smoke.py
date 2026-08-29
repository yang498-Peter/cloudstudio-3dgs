"""Build a signed two-step Tile_1 dense-geometry consumption smoke."""

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
from cloudstudio_3dgs.data.mesh_geometry import verify_mesh_geometry_manifest
from cloudstudio_3dgs.data.mono_depth import verify_mono_depth_manifest
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
    sign_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sign(payload: dict, field: str) -> dict:
    result = copy.deepcopy(payload)
    result[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return result


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    protocol = BASE / "v62_dense_geometry" / "smoke2"
    config_path = protocol / "snow_tile1_v62_dense_geometry_smoke2.config.json"
    gate_path = protocol / "training_gate.json"
    base_config = _read(
        BASE / "fixed_topology_evaluation_v25a" / "snow_tile1_fixed_topology_a0_eval_v25a.json"
    )
    mono_path = BASE / "d0a_da2_mesh_aligned_tile1_v62l" / "mono_depth_manifest.json"
    mesh_path = BASE / "d0a_mesh_face4_tile1_full_v62j" / "mesh_geometry_manifest.json"
    mono = _read(mono_path)
    mesh = _read(mesh_path)
    mono_sha = verify_mono_depth_manifest(mono)
    mesh_sha = verify_mesh_geometry_manifest(mesh)

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived["bindings"] = dict(derived["bindings"])
    derived["bindings"]["da2_train_manifest_sha256"] = mono_sha
    derived["bindings"]["mesh_geometry_tile1_manifest_sha256"] = mesh_sha
    derived.setdefault("evidence", {})["dense_geometry_experimental_override"] = {
        "scope": "Tile_1 fixed-topology smoke only",
        "source_da2_gate_replaced": True,
        "sky_evidence_not_reused_as_dense_geometry_evidence": True,
        "mesh_topology": "OPEN3D_BPA_CANDIDATE_NOT_VENDOR_CONFIRMED",
    }
    derived = sign_gate(derived)

    config = copy.deepcopy(base_config)
    config.update(
        {
            "run_id": "snow-tile1-v62-dense-geometry-smoke2",
            "output_dir": str(BASE / "training_tile1_v62_dense_geometry_smoke2"),
            "mipmap_pipeline_gate": str(gate_path),
            "mono_depth_manifest": str(mono_path),
            "mono_depth_root": str(BASE / "da2_face4_v23k" / "train"),
            "mesh_geometry_manifest": str(mesh_path),
            "mesh_geometry_root": str(BASE / "d0a_mesh_face4_tile1_full_v62j"),
            "da2_depth_weight": 0.5,
            "mesh_depth_weight": 0.5,
            "mesh_normal_weight": 0.05,
            "competitor_loss_schedule_enabled": False,
            "max_steps": 2,
            "checkpoint_every": 2,
            "checkpoint_keep_every": 0,
            "controlled_stop_after_steps": None,
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": False,
            "config_manifest_sha256": None,
        }
    )
    config["fixed_topology_schedule"] = {
        "enabled": True,
        "phase_a_steps": 2,
        "phase_b_steps": 0,
        "phase_b_geometry_lr_scale": 1.0,
        "phase_c_geometry_lr_scale": 1.0,
        "phase_b_range_weight_scale": 1.0,
        "phase_c_range_weight_scale": 1.0,
        "phase_b_normal_weight_scale": 1.0,
        "phase_c_normal_weight_scale": 1.0,
        "audit_steps": [1, 2],
    }
    unsigned_config = copy.deepcopy(config)
    unsigned_config.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned_config)
    ).hexdigest()

    plan = _sign(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": 1,
            "view_count": 374,
            "steps": {"total": 2},
            "arms": [
                {
                    "arm": "DENSE_GEOMETRY_SMOKE",
                    "config_manifest_sha256": config["config_manifest_sha256"],
                }
            ],
            "training_allowed": False,
            "adaptive_growth_remains_blocked_by": [
                "dense_geometry_fixed_topology_ab_not_yet_passed"
            ],
        },
        "evaluation_plan_sha256",
    )
    readiness = _sign(
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
                "scope": "two-step signed implementation smoke",
            },
        },
        "readiness_sha256",
    )
    gate = advance_fixed_topology_evaluation_gate(
        derived, readiness, plan, {"DENSE_GEOMETRY_SMOKE": config}
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
