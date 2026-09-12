#!/usr/bin/env python3
"""Derive and validate the B-line trainer configs (CPU only, no training).

B0  house0305_global_coarse_B0_10k.json - coarse whole-scene prior: no Tile
    inputs, every Face4 view, 1.86M-point LiDAR init (house0305_init_2m),
    planar_surfel surface initialisation with on-the-fly kNN scales, cap 3M,
    controlled stop at 10k, sky dome backdrops for whole faces.
B1  tile1_B1_standin_20k.json       - tile1_R1d_20k with the stand-in backdrop
B1  tile1_B1_standin_K2sky_20k.json - tile1_K2_sky_20k with the stand-in backdrop
    (the design note calls this B2; the file name keeps the B1 family prefix)

Every config is run through TrainerConfig.from_dict(...).validate() here.
The B1 pair points at a backdrop library that only exists after the GPU
build step, so for validation they are re-pointed at the existing dome-only
Tile_1 library (same schema) and the expected failure on the real path is
recorded verbatim. Results go to b_line_config_validation.json next to this
script.

    set PYTHONPATH=C:\\Peter\\cloudstudio-3dgs-work
    python research/quality_recovery_v2/14_standin_backdrop/make_b_line_configs.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = Path("C:/Peter/3dgs-runs/house0305_sop")
DATASETS = Path("C:/Peter/3dgs-datasets")
HERE = Path(__file__).resolve().parent

R1D = RUNS / "tile1_R1d_20k.json"
K2 = RUNS / "tile1_K2_sky_20k.json"
B0 = RUNS / "house0305_global_coarse_B0_10k.json"
B1 = RUNS / "tile1_B1_standin_20k.json"
B1_K2 = RUNS / "tile1_B1_standin_K2sky_20k.json"
STANDIN_ROOT = RUNS / "tile_backgrounds_B1" / "Tile_1"
DOME_ONLY_ROOT = RUNS / "tile_backgrounds_v9" / "Tile_1"

B0_STEPS = 10_000
B0_CAP = 3_000_000


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def count_full_face_views(base: dict) -> int:
    """Views the tile-free trainer would iterate: the Face4 cache without crops."""
    from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset

    dataset = FaceCacheDataset(
        Path(base["face_cache_manifest"]),
        Path(base["face_cache_root"]),
        verify_artifacts=False,
        dataset_manifest_path=Path(base["dataset_manifest"]),
        renderer_mask_manifest_path=Path(base["renderer_mask_manifest"]),
    )
    return len(dataset), int(dataset.filtered_empty_mask_count)


def make_b0(base: dict, view_count: int) -> dict:
    cfg = copy.deepcopy(base)
    for key in (
        "tile_inputs_manifest", "tile_inputs_root", "mipmap_tile_id",
        "initialization_geometry_manifest",
    ):
        cfg.pop(key, None)
    cfg["run_id"] = "house0305-B0-global-coarse-10k"
    cfg["output_dir"] = str(RUNS / "global_coarse_B0_10k")
    cfg["initialization_ply"] = str(DATASETS / "house0305_init_2m" / "sparse_pc.ply")
    cfg["initialization_geometry"] = str(DATASETS / "house0305_init_2m" / "lidar_init_geometry.npz")
    cfg["surface_initialization"] = {
        "enabled": True,
        "mode": "planar_surfel",
        "planarity_gate": 0.6,
        "normal_scale_ratio": 0.5,
    }
    cfg["metric_scale_calibration"] = {
        "mode": "knn",
        "knn_neighbors": 7,
        "knn_reduction": "arithmetic_mean",
        "scale_multiplier": 1.0,
    }
    # Whole faces: the dome-only library rendered per full face (downsample 4,
    # the library upsamples - correct for uncropped views).
    cfg["background_image_manifest"] = str(
        RUNS / "view_backgrounds_v9" / "view_background_manifest_train.json"
    )
    cfg["background_image_root"] = str(RUNS / "view_backgrounds_v9")
    # Epoch-permutation sampling requires max_steps == 20 view epochs; the
    # controlled stop is the real horizon.
    cfg["max_steps"] = 20 * view_count
    cfg["controlled_stop_after_steps"] = B0_STEPS
    cfg["checkpoint_every"] = 5000
    cfg["cap_max"] = B0_CAP
    # Growth ends at 8k so the last 2k steps consolidate opacity before the
    # stop; prune_switch_step must equal max_steps // 2 under the exact
    # lifecycle contract and lands after the stop (irrelevant but required).
    cfg["mcmc_refine_stop_iter"] = 8000
    strategy = dict(cfg["default_strategy"])
    strategy["refine_stop_iter"] = 8000
    strategy["refine_scale2d_stop_iter"] = 8000
    strategy["prune_switch_step"] = cfg["max_steps"] // 2
    cfg["default_strategy"] = strategy
    cfg["lineage"] = {
        "base": "tile1_R1d_20k",
        "note": (
            "B0 coarse whole-scene prior for the stand-in backdrop: tile-free "
            "Face4 training on every view, 1.86M-point LiDAR init, cap 3M, "
            "10k controlled stop, dome-only whole-face backdrops"
        ),
        "single_change": (
            "remove Tile inputs/crops; init house0305_init_2m planar_surfel + kNN "
            "scales; cap 15M->3M; growth stop 14k->8k; controlled stop 20k->10k"
        ),
    }
    return cfg


def make_b1(base: dict, *, run_id: str, output_name: str, lineage_base: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["run_id"] = run_id
    cfg["output_dir"] = str(RUNS / output_name)
    cfg["background_image_manifest"] = str(STANDIN_ROOT / "background_manifest.json")
    cfg["background_image_root"] = str(STANDIN_ROOT)
    cfg["lineage"] = {
        "base": lineage_base,
        "note": (
            "B-line: per-view backdrop = sky dome + frozen stand-in (other Tiles' "
            "R1d checkpoints and the B0 coarse prior, every gaussian inside "
            "Tile_1's training_and_export_box removed); everything else identical"
        ),
        "single_change": (
            "background_image_manifest/root tile_backgrounds_v9/Tile_1 -> "
            "tile_backgrounds_B1/Tile_1 (tools/build_standin_backgrounds.py)"
        ),
    }
    return cfg


def validate(cfg: dict) -> dict:
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    started = time.time()
    try:
        config = TrainerConfig.from_dict(copy.deepcopy(cfg))
        config.validate()
        return {"status": "PASS", "seconds": round(time.time() - started, 1)}
    except Exception as error:  # noqa: BLE001 - the message is the result
        return {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-3:],
            "seconds": round(time.time() - started, 1),
        }


def main() -> int:
    base = _load(R1D)
    k2 = _load(K2)
    results: dict = {"schema_version": 1, "kind": "b_line_config_validation_v1"}

    view_count, filtered = count_full_face_views(base)
    results["b0_full_face_view_count"] = view_count
    results["b0_filtered_empty_mask_count"] = filtered
    b0 = make_b0(base, view_count)
    _dump(B0, b0)
    results["B0"] = {"path": str(B0), "max_steps": b0["max_steps"], **validate(b0)}
    print("B0", results["B0"])

    b1 = make_b1(base, run_id="house0305-t1-B1-standin", output_name="tile1_B1_standin_20k", lineage_base="tile1_R1d_20k")
    b1_k2 = make_b1(k2, run_id="house0305-t1-B1-standin-K2sky", output_name="tile1_B1_standin_K2sky_20k", lineage_base="tile1_K2_sky_20k")
    _dump(B1, b1)
    _dump(B1_K2, b1_k2)
    for name, cfg, path in (("B1", b1, B1), ("B1_K2sky", b1_k2, B1_K2)):
        as_written = validate(cfg)
        repointed = copy.deepcopy(cfg)
        repointed["background_image_manifest"] = str(DOME_ONLY_ROOT / "background_manifest.json")
        repointed["background_image_root"] = str(DOME_ONLY_ROOT)
        with_dome_only = validate(repointed)
        results[name] = {
            "path": str(path),
            "as_written_before_gpu_build": as_written,
            "repointed_to_dome_only_library": with_dome_only,
        }
        print(name, results[name])

    # The only difference between B1 and its base must be the backdrop pair,
    # run identity and lineage.
    def diff_keys(a: dict, b: dict) -> list[str]:
        return sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))

    results["B1_vs_R1d_changed_keys"] = diff_keys(base, b1)
    results["B1_K2sky_vs_K2_changed_keys"] = diff_keys(k2, b1_k2)
    _dump(HERE / "b_line_config_validation.json", results)
    print("wrote", HERE / "b_line_config_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
