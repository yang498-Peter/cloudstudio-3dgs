"""Stage ladder arms with explicit lineage.

Every arm names the base it is derived from and the single change it makes on
top of that base; nothing is inherited implicitly, and a rejected arm can never
leak into a later one. The stager writes ``ladder_lineage.json`` next to the
configs so each result can be traced to base + diff, and the driver copies the
exact config it ran into the run directory.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

RUN = Path(r"C:\Peter\3dgs-runs\house0305_sop")
SCRATCH = Path(__file__).resolve().parent
BASE_ID = "h5v"
STOP = 3500
GATE1 = r"C:\Peter\cloudstudio-3dgs-gate1"
WORK = r"C:\Peter\cloudstudio-3dgs-work"


def _set(path, value):
    def apply(c):
        node = c
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return apply


def _many(*fns):
    def apply(c):
        for fn in fns:
            fn(c)
    return apply


# id -> (base id, description, mutator, repo)
ARMS = {
    "R1_range": ("h5v", "LiDAR range term back to the trainer default: 0.05 robust log-Huber (was 0.5 linear L1)",
                 _many(_set("lidar_range_weight", 0.05), _set("lidar_range_loss_mode", "robust_log_huber"),
                       _set("lidar_log_range_huber_delta", 0.05)), GATE1),
    "R2_noalpha": ("R1_range", "R1 + LiDAR alpha floor off",
                   _many(_set("lidar_alpha_weight", 0.0), _set("lidar_alpha_dilation_radius_px", 0),
                         _set("surface_alpha_floor_profile", False)), GATE1),
    "R3_split": ("R1_range", "R1 + adaptive detail split (0.02 m / 0.0035 screen radius, revised child opacity); alpha floor kept",
                 _many(_set("default_strategy.detail_split_policy", "lidar_surface_screen_detail"),
                       _set("default_strategy.detail_split_scale_m", 0.02),
                       _set("default_strategy.detail_split_screen_radius", 0.0035),
                       _set("default_strategy.revised_opacity", True)), GATE1),
    "R3B_split_noalpha": ("R3_split", "R3 with the alpha floor off (interaction check for R2 x split)",
                          _many(_set("lidar_alpha_weight", 0.0), _set("lidar_alpha_dilation_radius_px", 0),
                                _set("surface_alpha_floor_profile", False)), GATE1),
    "R4_flatabs": ("R3_split", "R3 + flatten absolute 1 mm (rejected: short axis 1.14 -> 3.33 mm)",
                   _many(_set("lidar_normal_alignment.flatten_mode", "absolute_m"),
                         _set("lidar_normal_alignment.flatten_target_m", 0.001)), GATE1),
    "R5_cull05": ("R3_split", "R3 + uniform 0.05 cull from the first refine",
                  _many(_set("default_strategy.vendor_cull_warmup_profile", "compatibility_uniform_0p05"),
                        _set("default_strategy.prune_opa", 0.05), _set("default_strategy.prune_opa_late", 0.05)), GATE1),
    "R5r_cull05": ("R3_split", "R5 reproduced from the explicit-lineage stager (config_as_run recorded)",
                   _many(_set("default_strategy.vendor_cull_warmup_profile", "compatibility_uniform_0p05"),
                         _set("default_strategy.prune_opa", 0.05), _set("default_strategy.prune_opa_late", 0.05)), WORK),
    "R6_opreg": ("R5_cull05", "R5 + mean-opacity drain 0.01",
                 _set("geometry_regularization.opacity_sparsity_weight", 0.01), GATE1),
    "C1_own": ("R5_cull05", "R5 + Tile ownership masking (foreign-surface pixels leave supervision)",
               _many(_set("tile_ownership_masking", True), _set("tile_ownership_margin_m", 0.5),
                     _set("tile_ownership_dilation_px", 15)), WORK),
    "C2_da2": ("R5_cull05", "R5 + monocular depth far cutoff 30 m and bounded depth space",
               _many(_set("mono_depth_max_range_m", 30.0), _set("da2_depth_space", "compressed")), WORK),
    "C3_vis": ("R5_cull05", "R5 + face LiDAR cache with hidden-point rejection (6 px cells)",
               _many(_set("face_lidar_geometry_manifest", "C:/Peter/3dgs-datasets/house0305_sop_v8/face4_lidar_train_vis6/face_lidar_geometry_manifest.json"),
                     _set("face_lidar_geometry_root", "C:/Peter/3dgs-datasets/house0305_sop_v8/face4_lidar_train_vis6")), WORK),
    "T1_split10": ("R5_cull05", "R5 + detail split gate 0.01 m (children of 20-45 mm parents can split again)",
                   _set("default_strategy.detail_split_scale_m", 0.01), WORK),
    "T2_split05": ("R5_cull05", "R5 + detail split gate 0.005 m",
                   _set("default_strategy.detail_split_scale_m", 0.005), WORK),
    "S1_sizeprior": ("R5_cull05", "R5 + size hinge at 2x the median initial size (scale_upper 0.01, ratio 2)",
                     _many(_set("geometry_regularization.scale_upper_weight", 0.01),
                           _set("geometry_regularization.max_scale_ratio_to_reference", 2.0)), WORK),
    # Size arms re-derived from the supervision-correctness winner (C2).
    "T1c_split10": ("C2_da2", "C2 + detail split gate 0.01 m",
                    _set("default_strategy.detail_split_scale_m", 0.01), WORK),
    "T2c_split05": ("C2_da2", "C2 + detail split gate 0.005 m",
                    _set("default_strategy.detail_split_scale_m", 0.005), WORK),
    "S1c_sizeprior": ("C2_da2", "C2 + size hinge at 2x the median initial size (scale_upper 0.01, ratio 2)",
                      _many(_set("geometry_regularization.scale_upper_weight", 0.01),
                            _set("geometry_regularization.max_scale_ratio_to_reference", 2.0)), WORK),
    "C4_all": ("C3_vis", "C3 + Tile ownership masking + monocular far cutoff (supervision-correctness bundle)",
               _many(_set("tile_ownership_masking", True), _set("tile_ownership_margin_m", 0.5),
                     _set("tile_ownership_dilation_px", 15),
                     _set("mono_depth_max_range_m", 30.0), _set("da2_depth_space", "compressed")), WORK),
}


def build(arm_id: str, cache: dict) -> dict:
    if arm_id == BASE_ID:
        cfg = json.loads((RUN / "tile0_h5v.json").read_text(encoding="utf-8"))
        cfg.pop("resume_checkpoint", None)
        cfg["controlled_stop_after_steps"] = STOP
        cfg["checkpoint_keep_every"] = STOP
        cfg["checkpoint_every"] = STOP
        return cfg
    if arm_id in cache:
        return copy.deepcopy(cache[arm_id])
    base_id, _, mutate, _ = ARMS[arm_id]
    cfg = build(base_id, cache)
    mutate(cfg)
    cache[arm_id] = copy.deepcopy(cfg)
    return cfg


def main() -> None:
    wanted = sys.argv[1:] or list(ARMS)
    cache: dict = {}
    lineage_path = RUN / "ladder_lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8")) if lineage_path.exists() else {}
    for arm_id in wanted:
        base_id, desc, _, repo = ARMS[arm_id]
        cfg = build(arm_id, cache)
        cfg["run_id"] = f"house0305-t0-{arm_id}"
        cfg["output_dir"] = str(RUN / f"tile0_{arm_id}")
        text = json.dumps(cfg, indent=1)
        (RUN / f"tile0_{arm_id}.json").write_text(text, encoding="utf-8")
        lineage[arm_id] = {
            "base": base_id,
            "description": desc,
            "config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "repo": repo,
        }
        cmd = "\n".join([
            "@echo off",
            r"call C:\Peter\cloudstudio-3dgs-gate1\train\env_machine_b.cmd",
            rf"cd /d {repo}",
            rf"set PYTHONPATH={repo};%PYTHONPATH%",
            "set PYTHONIOENCODING=utf-8",
            rf"set RUN={RUN}",
            r"set PY=C:\Peter\cloudstudio-3dgs\.venv-train\Scripts\python.exe",
            rf'%PY% tools/train_gsplat.py --config "%RUN%\tile0_{arm_id}.json" > "%RUN%\tile0_{arm_id}.log" 2> "%RUN%\tile0_{arm_id}.log.err"',
            rf'echo EXIT %ERRORLEVEL% >> "%RUN%\tile0_{arm_id}.log"',
            rf'copy /Y "%RUN%\tile0_{arm_id}.json" "%RUN%\tile0_{arm_id}\config_as_run.json" >nul',
            rf'%PY% "{SCRATCH}\morph_ckpt.py" "%RUN%\tile0_{arm_id}\checkpoints\latest.pt" {arm_id} > "%RUN%\tile0_{arm_id}\morph.txt" 2>&1',
            rf'%PY% tools/evaluate_probe_views.py --config "%RUN%\tile0_{arm_id}.json" --checkpoint "%RUN%\tile0_{arm_id}\checkpoints\latest.pt" --views 48 --tile-views --output "%RUN%\tile0_{arm_id}\battery_tile.json" > "%RUN%\tile0_{arm_id}\battery.log" 2>&1',
            "exit /b 0",
            "",
        ])
        (SCRATCH / f"run_{arm_id}.cmd").write_text(cmd, encoding="ascii")
        print(f"staged {arm_id} <- {base_id}: {desc}")
    lineage_path.write_text(json.dumps(lineage, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
