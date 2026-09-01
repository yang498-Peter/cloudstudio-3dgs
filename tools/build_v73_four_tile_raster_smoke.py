"""Build a signed full-resolution raster smoke for the four-Tile Snow plan.

This does not authorize quality training.  It proves that the largest forced
four-Tile leaf can load its full-resolution Face4 crops, LiDAR initialization,
and K7/K30 geometry without exceeding the local GPU/runtime envelope.
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
from cloudstudio_3dgs.pipeline.adaptive_tiling import verify_adaptive_tile_plan
from cloudstudio_3dgs.pipeline.mipmap_gate import sign_gate, verify_gate
from cloudstudio_3dgs.training.trainer import TrainerConfig


BASE = ROOT / "outputs" / "snow-20260224-full-20260825"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
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
    protocol = (
        BASE
        / "v73_four_tile_competitor_equivalent"
        / "smoke_tile0_factor1_initcap"
    )
    plan_path = BASE / "adaptive_tile_plan_lidar_visibility_4tile_v73" / "adaptive_tile_plan.json"
    tile_inputs_root = BASE / "tile_training_inputs_lidar_4tile_v73"
    geometry_root = BASE / "tile_initialization_geometry_k7_k30_4tile_v73"
    lidar_root = BASE / "tile_face4_lidar_geometry_4tile_v73" / "Tile_0"

    plan = _read(plan_path)
    plan_sha = verify_adaptive_tile_plan(plan)
    tile_inputs = _read(tile_inputs_root / "tile_inputs_manifest.json")
    tile = next(row for row in tile_inputs["tiles"] if int(row["tile_id"]) == 0)

    upstream = _read(BASE / "mipmap_upstream_data_ready_lidar_tiles_gate_v23y.json")
    verify_gate(upstream)
    derived = copy.deepcopy(upstream)
    derived.pop("gate_manifest_sha256", None)
    derived["bindings"] = dict(derived["bindings"])
    derived["bindings"]["spatial_tile_plan_manifest_sha256"] = plan_sha
    derived.setdefault("evidence", {})["v73_four_tile_compatibility"] = {
            "scope": "largest-leaf one-step full-resolution raster and memory smoke only",
        "topology": "X_to_Y_Y_four_leaves",
        "tile_id": 0,
        "tile_point_count": int(tile["initialization"]["point_count"]),
        "tile_view_count": int(tile["view_count"]),
        "quality_training_authorized": False,
    }
    derived = sign_gate(derived)
    gate_path = protocol / "upstream_smoke_gate.json"

    config = _read(
        BASE
        / "v66_competitor_strict_mesh_5v_gate"
        / "snow_tile1_v66a_black_5v.config.json"
    )
    config.update(
        {
            "run_id": "snow-tile0-v73-four-tile-raster-smoke1-factor1-initcap",
            "initialization_ply": str(
                tile_inputs_root / tile["initialization"]["path"]
            ),
            "initialization_geometry": str(
                geometry_root / "Tile_0" / "initialization_geometry_k7_k30.npz"
            ),
            "initialization_geometry_manifest": str(
                geometry_root / "tile_geometry_manifest.json"
            ),
            "mipmap_tile_id": 0,
            "tile_inputs_manifest": str(tile_inputs_root / "tile_inputs_manifest.json"),
            "tile_inputs_root": str(tile_inputs_root),
            "face_lidar_geometry_manifest": str(
                lidar_root / "face_lidar_geometry_manifest.json"
            ),
            "face_lidar_geometry_root": str(lidar_root),
            "mipmap_pipeline_gate": str(gate_path),
            "output_dir": str(protocol / "training"),
            "implementation_smoke_only": True,
            "final_evaluation_artifacts": False,
            # The fail-closed trainer contract permits exactly one factor=1
            # implementation-smoke step; this is sufficient to measure the
            # largest leaf's real raster peak without authorizing training.
            "max_steps": 1,
            "controlled_stop_after_steps": None,
            "checkpoint_every": 1,
            "checkpoint_keep_every": 1,
            "cap_max": max(3_000_000, int(tile["initialization"]["point_count"])),
            "da2_depth_weight": 0.0,
            "mono_depth_manifest": None,
            "mono_depth_root": None,
            "mesh_depth_weight": 0.0,
            "mesh_normal_weight": 0.0,
            "mesh_geometry_manifest": None,
            "mesh_geometry_root": None,
            "rendered_depth_normal_consistency_weight": 0.0,
            "competitor_loss_schedule_enabled": False,
            "bilateral_grid": {"enabled": False},
        }
    )
    config["surface_initialization"] = dict(config["surface_initialization"])
    config["surface_initialization"]["maximum_scale_m"] = 0.2
    config = _sign_config(config)
    config_path = protocol / "trainer.config.json"
    _write(gate_path, derived)
    _write(config_path, config)
    TrainerConfig.from_dict(config).validate()
    print(config_path)
    print(gate_path)
    print(f"tile_points={tile['initialization']['point_count']}")
    print(f"tile_views={tile['view_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
