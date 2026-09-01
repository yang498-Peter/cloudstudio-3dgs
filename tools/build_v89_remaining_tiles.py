"""Build signed V89 full runs for Snow Tiles whose mesh holdout gate failed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import sign_gate, verify_gate
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
REFERENCE = BASE / "v87_ultrasharp_detail" / "tile0_full12480"


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


def _tile_record(tile_id: int) -> dict:
    payload = _read(BASE / "tile_training_inputs_lidar_4tile_v73" / "tile_inputs_manifest.json")
    for tile in payload["tiles"]:
        if int(tile["tile_id"]) == tile_id:
            return tile
    raise ValueError(f"Tile_{tile_id} is absent from the four-tile manifest")


def _point_count(tile_id: int) -> int:
    import numpy as np

    geometry = np.load(
        BASE
        / "tile_initialization_geometry_k7_k30_4tile_v73"
        / f"Tile_{tile_id}"
        / "initialization_geometry_k7_k30.npz"
    )
    for key in ("means", "points", "xyz", "scales", "scales_m", "normals"):
        if key in geometry:
            return int(geometry[key].shape[0])
    raise ValueError("initialization geometry has no count-bearing array")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True, type=int, choices=(1, 2, 3))
    args = parser.parse_args()

    tile_id = int(args.tile_id)
    tile = _tile_record(tile_id)
    view_count = int(tile["view_count"])
    initial_count = _point_count(tile_id)
    total_steps = 20 * view_count
    five_views = 5 * view_count
    ten_views = 10 * view_count
    fifteen_views = 15 * view_count
    capacity = int(math.ceil((1.5 * initial_count) / 1000.0) * 1000)
    target = BASE / "v89_lidar_reliable_full" / f"tile{tile_id}_full{total_steps}"

    config = _read(REFERENCE / "trainer.config.json")
    config.update(
        {
            "run_id": f"snow-tile{tile_id}-v89-lidar-reliable-full{total_steps}",
            "initialization_ply": str(
                BASE
                / "tile_training_inputs_lidar_4tile_v73"
                / f"Tile_{tile_id}"
                / "initialization_full_lidar.ply"
            ),
            "initialization_geometry": str(
                BASE
                / "tile_initialization_geometry_k7_k30_4tile_v73"
                / f"Tile_{tile_id}"
                / "initialization_geometry_k7_k30.npz"
            ),
            "mipmap_tile_id": tile_id,
            "face_lidar_geometry_manifest": str(
                BASE
                / "tile_face4_lidar_geometry_4tile_v73"
                / f"Tile_{tile_id}"
                / "face_lidar_geometry_manifest.json"
            ),
            "face_lidar_geometry_root": str(
                BASE / "tile_face4_lidar_geometry_4tile_v73" / f"Tile_{tile_id}"
            ),
            "mipmap_pipeline_gate": str(target / "training_gate.json"),
            "output_dir": str(target / "training_ewa"),
            "max_steps": total_steps,
            "checkpoint_every": five_views,
            "checkpoint_keep_every": total_steps,
            "cap_max": capacity,
            "lidar_range_weight": 0.05,
            "da2_depth_weight": 0.0,
            "mesh_geometry_manifest": None,
            "mesh_geometry_root": None,
            "mono_depth_manifest": None,
            "mono_depth_root": None,
            "mesh_depth_weight": 0.0,
            "mesh_normal_weight": 0.0,
        }
    )
    config.pop("resume_checkpoint", None)
    config.pop("controlled_stop_after_steps", None)
    strategy = dict(config["default_strategy"])
    strategy.update(
        {
            "prune_switch_step": ten_views,
            "refine_scale2d_stop_iter": fifteen_views,
            "gradient_tile_core_box": tile["core_box"],
        }
    )
    config["default_strategy"] = strategy
    config["mcmc_refine_stop_iter"] = fifteen_views
    config = _sign_config(config)

    gate = _read(REFERENCE / "training_gate.json")
    verify_gate(gate)
    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_EVALUATION_READY",
            "training_allowed": True,
            "next_required_stage": f"v89_tile{tile_id}_full_joint_quality_evaluation",
        }
    )
    bindings = dict(gate["bindings"])
    bindings["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    bindings.pop("adaptive_growth_warm_start_checkpoint_sha256", None)
    bindings.pop("mesh_geometry_tile0_manifest_sha256", None)
    gate["bindings"] = bindings
    gate.setdefault("evidence", {})["v89_lidar_reliable_full"] = {
        "tile_id": tile_id,
        "view_count": view_count,
        "initial_gaussian_count": initial_count,
        "capacity_cap_1p5x_rounded": capacity,
        "max_steps_20v": total_steps,
        "growth_stop_15v": fifteen_views,
        "prune_switch_10v": ten_views,
        "mesh_and_da2_disabled_reason": (
            "tile-specific spatial-block holdout mesh P95 exceeded 0.10 m"
        ),
        "fallback_geometry": "signed sparse LiDAR range weight 0.05 plus K7/K30 normals",
        "algorithm_source": "V87 ultrasharp lifecycle with tile-specific signed inputs",
    }
    adaptive = dict(gate["adaptive_growth"])
    adaptive.update(
        {
            "profile": "v89_lidar_reliable_full",
            "stage": "full",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": None,
            "resume_checkpoint_sha256": None,
            "resume_source_trainer_config_sha256": None,
            "resume_allowed_lineage_differences": [],
            "mcmc_allowed": False,
        }
    )
    gate["adaptive_growth"] = adaptive
    gate = sign_gate(gate)

    _write(target / "trainer.config.json", config)
    _write(target / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(target / "trainer.config.json")
    print(target / "training_gate.json")
    print(
        f"Tile_{tile_id}: {initial_count} -> cap {capacity}, "
        f"20V={total_steps}, mesh/DA2 fail-closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
