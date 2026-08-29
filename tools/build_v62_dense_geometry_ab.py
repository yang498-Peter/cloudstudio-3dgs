"""Build a signed matched Tile_1 A/B for dense geometry supervision.

Both arms load the same Face4, DA2 and mesh sidecars and use the same seed and
fixed Gaussian topology.  Only the three dense-geometry loss weights differ.
This isolates consumption quality before topology growth/culling is allowed.
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
from cloudstudio_3dgs.data.mesh_geometry import verify_mesh_geometry_manifest
from cloudstudio_3dgs.data.mono_depth import verify_mono_depth_manifest
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    advance_fixed_topology_evaluation_gate,
    sign_gate,
    verify_gate,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
STEPS = 100


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


def _config(
    base: dict,
    *,
    arm: str,
    output_dir: Path,
    gate_path: Path,
    mono_path: Path,
    mesh_path: Path,
    dense: bool,
    early_competitor_equivalent: bool = False,
) -> dict:
    config = copy.deepcopy(base)
    config.update(
        {
            "run_id": f"snow-tile1-v62-{arm.lower()}-100",
            "output_dir": str(output_dir),
            "mipmap_pipeline_gate": str(gate_path),
            # Load identical signed sidecars in both arms.  The control weights
            # are zero, so file/view ordering and I/O cannot explain a delta.
            "mono_depth_manifest": str(mono_path),
            "mono_depth_root": str(BASE / "da2_face4_v23k" / "train"),
            "mesh_geometry_manifest": str(mesh_path),
            "mesh_geometry_root": str(BASE / "d0a_mesh_face4_tile1_full_v62j"),
            "da2_depth_weight": 0.5 if dense else 0.0,
            "mesh_depth_weight": (
                0.0 if early_competitor_equivalent else (0.5 if dense else 0.0)
            ),
            "mesh_normal_weight": 0.05 if dense else 0.0,
            # This is an isolation A/B, so all dense terms run from step one.
            # The formal 20-epoch mainline will turn the vendor schedule on
            # only after this fixed-topology gate passes.
            "competitor_loss_schedule_enabled": False,
            "max_steps": STEPS,
            "checkpoint_every": STEPS,
            "checkpoint_keep_every": 0,
            "controlled_stop_after_steps": None,
            "implementation_smoke_only": False,
            "final_evaluation_artifacts": False,
            "config_manifest_sha256": None,
        }
    )
    config["fixed_topology_schedule"] = {
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
    }
    unsigned = copy.deepcopy(config)
    unsigned.pop("config_manifest_sha256", None)
    config["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return config


def main() -> int:
    protocol = BASE / "v62_dense_geometry" / "ab100"
    gate_path = protocol / "training_gate.json"
    base = _read(
        BASE
        / "fixed_topology_evaluation_v25a"
        / "snow_tile1_fixed_topology_a0_eval_v25a.json"
    )
    mono_path = BASE / "d0a_da2_mesh_aligned_tile1_v62l" / "mono_depth_manifest.json"
    mesh_path = BASE / "d0a_mesh_face4_tile1_full_v62j" / "mesh_geometry_manifest.json"
    mono_sha = verify_mono_depth_manifest(_read(mono_path))
    mesh_sha = verify_mesh_geometry_manifest(_read(mesh_path))

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived["bindings"] = dict(derived["bindings"])
    derived["bindings"]["da2_train_manifest_sha256"] = mono_sha
    derived["bindings"]["mesh_geometry_tile1_manifest_sha256"] = mesh_sha
    derived.setdefault("evidence", {})["dense_geometry_experimental_override"] = {
        "scope": "Tile_1 matched fixed-topology A/B only",
        "source_da2_gate_replaced": True,
        "sky_evidence_not_reused_as_dense_geometry_evidence": True,
        "mesh_topology": "OPEN3D_BPA_CANDIDATE_NOT_VENDOR_CONFIRMED",
    }
    derived = sign_gate(derived)

    control = _config(
        base,
        arm="CONTROL",
        output_dir=BASE / "training_tile1_v62_control100",
        gate_path=gate_path,
        mono_path=mono_path,
        mesh_path=mesh_path,
        dense=False,
    )
    dense = _config(
        base,
        arm="DENSE",
        output_dir=BASE / "training_tile1_v62_dense100",
        gate_path=gate_path,
        mono_path=mono_path,
        mesh_path=mesh_path,
        dense=True,
    )
    early_equivalent = _config(
        base,
        arm="EARLY_EQUIVALENT",
        output_dir=BASE / "training_tile1_v62_early_equivalent100",
        gate_path=gate_path,
        mono_path=mono_path,
        mesh_path=mesh_path,
        dense=True,
        early_competitor_equivalent=True,
    )
    mesh_only = _config(
        base,
        arm="MESH_ONLY",
        output_dir=BASE / "training_tile1_v62_mesh_only100",
        gate_path=gate_path,
        mono_path=mono_path,
        mesh_path=mesh_path,
        dense=True,
    )
    mesh_only["da2_depth_weight"] = 0.0
    mesh_only.pop("config_manifest_sha256", None)
    mesh_only["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(mesh_only)
    ).hexdigest()
    configs = {
        "CONTROL": control,
        "DENSE": dense,
        "EARLY_EQUIVALENT": early_equivalent,
        "MESH_ONLY": mesh_only,
    }
    plan = _sign(
        {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "dataset": "snow-20260224",
            "tile_id": 1,
            "view_count": 374,
            "steps": {"total": STEPS},
            "arms": [
                {
                    "arm": arm,
                    "config_manifest_sha256": config["config_manifest_sha256"],
                }
                for arm, config in configs.items()
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
                # Required upstream readiness fact: the signed Phase-A
                # initialization/freeze audit has already passed.  This does
                # not claim that the present isolation A/B freezes geometry;
                # its per-arm config explicitly disables that schedule.
                "phase_a_geometry_frozen": True,
                "actual_merge_contract": "retain_full_halo",
                "halo_overlap_retained": True,
                "scope": "100-step matched dense-geometry isolation A/B",
            },
        },
        "readiness_sha256",
    )
    gate = advance_fixed_topology_evaluation_gate(derived, readiness, plan, configs)
    paths = {
        "CONTROL": protocol / "snow_tile1_v62_control100.config.json",
        "DENSE": protocol / "snow_tile1_v62_dense100.config.json",
        "EARLY_EQUIVALENT": (
            protocol / "snow_tile1_v62_early_equivalent100.config.json"
        ),
        "MESH_ONLY": protocol / "snow_tile1_v62_mesh_only100.config.json",
    }
    _write(protocol / "derived_upstream_gate.json", derived)
    _write(protocol / "evaluation_plan.json", plan)
    _write(protocol / "readiness.json", readiness)
    _write(gate_path, gate)
    for arm, path in paths.items():
        _write(path, configs[arm])
        TrainerConfig.from_dict(configs[arm]).validate()
        print(f"{arm}={path}")
    print(f"GATE={gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
