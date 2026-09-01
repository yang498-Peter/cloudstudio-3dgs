"""Stage the cumulative fix ladder from the tile0_h5v baseline.

Each rung adds exactly one verified change on top of the previous rung, so the
delta between consecutive rungs is attributable. Every rung is a 3,500-step
probe (one full opacity-reset cycle plus the flush at 3,101 and 400 steps of
recovery), scored on morphology and the Tile-level battery.
"""
import copy
import json
import sys
from pathlib import Path

RUN = Path(r"C:\Peter\3dgs-runs\house0305_sop")
SCRATCH = Path(__file__).resolve().parent
BASE = json.loads((RUN / "tile0_h5v.json").read_text(encoding="utf-8"))
STOP = 3500

RUNGS = [
    # id, description, mutator
    ("R1_range", "LiDAR range term back to the trainer default: 0.05 robust log-Huber (was 0.5 linear L1)",
     lambda c: c.update({"lidar_range_weight": 0.05,
                         "lidar_range_loss_mode": "robust_log_huber",
                         "lidar_log_range_huber_delta": 0.05})),
    ("R2_noalpha", "R1 + remove the one-sided LiDAR alpha floor (weight 0.5 target 0.95 dilation 6 -> off)",
     lambda c: c.update({"lidar_alpha_weight": 0.0,
                         "lidar_alpha_dilation_radius_px": 0,
                         "surface_alpha_floor_profile": False})),
    ("R3_split", "R2 + reachable split: detail split policy (0.02 m / 0.0035 screen radius, revised opacity)",
     lambda c: c["default_strategy"].update({"detail_split_policy": "lidar_surface_screen_detail",
                                             "detail_split_scale_m": 0.02,
                                             "detail_split_screen_radius": 0.0035,
                                             "revised_opacity": True})),
    ("R4_flatabs", "R3 + flatten acts on the short axis only (absolute 1 mm target instead of tangent ratio)",
     lambda c: c["lidar_normal_alignment"].update({"flatten_mode": "absolute_m",
                                                   "flatten_target_m": 0.001})),
    ("R5_cull05", "R4 + uniform 0.05 cull from the first refine (reference floor) instead of 0.10 until step 24,780",
     lambda c: c["default_strategy"].update({"vendor_cull_warmup_profile": "compatibility_uniform_0p05",
                                             "prune_opa": 0.05,
                                             "prune_opa_late": 0.05})),
    ("R6_opreg", "R5 + mean-opacity drain 0.01 (reference schedule term)",
     lambda c: c["geometry_regularization"].update({"opacity_sparsity_weight": 0.01})),
    # The next two need the fix branch (worktree) and run from there.
    ("R7_own", "R6 + Tile ownership masking (foreign-surface pixels leave supervision)",
     lambda c: c.update({"tile_ownership_masking": True,
                         "tile_ownership_margin_m": 0.5,
                         "tile_ownership_dilation_px": 15})),
    ("R8_da2", "R7 + monocular depth far cutoff 30 m and bounded depth space",
     lambda c: c.update({"mono_depth_max_range_m": 30.0,
                         "da2_depth_space": "compressed"})),
]
WORKTREE_RUNGS = {"R7_own", "R8_da2"}


def main() -> None:
    only = set(sys.argv[1:])
    cfg = copy.deepcopy(BASE)
    cfg.pop("resume_checkpoint", None)
    cfg["controlled_stop_after_steps"] = STOP
    cfg["checkpoint_keep_every"] = STOP
    cfg["checkpoint_every"] = STOP
    for rung_id, desc, mutate in RUNGS:
        mutate(cfg)
        if only and rung_id not in only:
            continue
        repo = (
            r"C:\Peter\cloudstudio-3dgs-work"
            if rung_id in WORKTREE_RUNGS
            else r"C:\Peter\cloudstudio-3dgs-gate1"
        )
        out = copy.deepcopy(cfg)
        out["run_id"] = f"house0305-t0-{rung_id}"
        out["output_dir"] = str(RUN / f"tile0_{rung_id}")
        (RUN / f"tile0_{rung_id}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        cmd = "\n".join([
            "@echo off",
            r"call C:\Peter\cloudstudio-3dgs-gate1\train\env_machine_b.cmd",
            rf"cd /d {repo}",
            rf"set PYTHONPATH={repo};%PYTHONPATH%",
            "set PYTHONIOENCODING=utf-8",
            rf"set RUN={RUN}",
            r"set PY=C:\Peter\cloudstudio-3dgs\.venv-train\Scripts\python.exe",
            rf'%PY% tools/train_gsplat.py --config "%RUN%\tile0_{rung_id}.json" > "%RUN%\tile0_{rung_id}.log" 2> "%RUN%\tile0_{rung_id}.log.err"',
            rf'echo EXIT %ERRORLEVEL% >> "%RUN%\tile0_{rung_id}.log"',
            rf'%PY% "{SCRATCH}\morph_ckpt.py" "%RUN%\tile0_{rung_id}\checkpoints\latest.pt" {rung_id} > "%RUN%\tile0_{rung_id}\morph.txt" 2>&1',
            rf'%PY% tools/evaluate_probe_views.py --config "%RUN%\tile0_{rung_id}.json" --checkpoint "%RUN%\tile0_{rung_id}\checkpoints\latest.pt" --views 48 --tile-views --output "%RUN%\tile0_{rung_id}\battery_tile.json" > "%RUN%\tile0_{rung_id}\battery.log" 2>&1',
            "exit /b 0",
            "",
        ])
        (SCRATCH / f"run_{rung_id}.cmd").write_text(cmd, encoding="ascii")
        print(f"staged {rung_id}: {desc}")


if __name__ == "__main__":
    main()
