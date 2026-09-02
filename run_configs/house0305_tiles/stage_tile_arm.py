"""Stage one arm's recipe for another Tile with a controlled stop.

Usage: stage_tile_arm.py <arm_id> <tile_id> <stop_steps>
Derives tileN_<arm>.json from tile0_<arm>.json: per-Tile paths, 20-epoch
step budget for that Tile's view count (the validator's epoch rule), the
lifecycle windows scaled to it, capacity = init x 1.34, and a controlled stop.
"""
import json
import sys
from pathlib import Path

RUN = Path(r"C:\Peter\3dgs-runs\house0305_sop")
SCRATCH = Path(__file__).resolve().parent
REPO = r"C:\Peter\cloudstudio-3dgs-work"


def main() -> None:
    arm, tile, stop = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfg = json.loads((RUN / f"tile0_{arm}.json").read_text(encoding="utf-8"))
    tiles = json.loads((RUN / "tile_inputs_multi" / "tile_inputs_manifest.json").read_text(encoding="utf-8"))
    selected = [t for t in tiles["tiles"] if int(t["tile_id"]) == tile][0]
    views = int(selected["view_count"])
    steps = 20 * views
    name = f"tile{tile}_{arm}"
    cfg.pop("resume_checkpoint", None)
    for key in ("holdout_spatial_cell_m", "holdout_guard_m"):
        cfg.pop(key, None)
    cfg["run_id"] = f"house0305-t{tile}-{arm}"
    cfg["output_dir"] = str(RUN / name)
    cfg["mipmap_tile_id"] = tile
    cfg["max_steps"] = steps
    cfg["controlled_stop_after_steps"] = stop
    cfg["checkpoint_every"] = 3500
    cfg["checkpoint_keep_every"] = 3500
    cfg["mcmc_refine_stop_iter"] = int(round(steps * 38656 / 49560))
    cfg["default_strategy"]["refine_stop_iter"] = cfg["mcmc_refine_stop_iter"]
    cfg["default_strategy"]["refine_scale2d_stop_iter"] = cfg["mcmc_refine_stop_iter"]
    cfg["default_strategy"]["prune_switch_step"] = steps // 2
    cfg["cap_max"] = int(selected["initialization"]["point_count"] * 1.34)
    for key in ("initialization_ply", "initialization_geometry", "background_image_manifest", "background_image_root"):
        cfg[key] = cfg[key].replace("Tile_0", f"Tile_{tile}")
    (RUN / f"{name}.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    cmd = "\n".join([
        "@echo off",
        r"call C:\Peter\cloudstudio-3dgs-gate1\train\env_machine_b.cmd",
        rf"cd /d {REPO}",
        rf"set PYTHONPATH={REPO};%PYTHONPATH%",
        "set PYTHONIOENCODING=utf-8",
        rf"set RUN={RUN}",
        r"set PY=C:\Peter\cloudstudio-3dgs\.venv-train\Scripts\python.exe",
        rf'%PY% tools/train_gsplat.py --config "%RUN%\{name}.json" > "%RUN%\{name}.log" 2> "%RUN%\{name}.log.err"',
        rf'echo EXIT %ERRORLEVEL% >> "%RUN%\{name}.log"',
        rf'copy /Y "%RUN%\{name}.json" "%RUN%\{name}\config_as_run.json" >nul',
        rf'%PY% "{SCRATCH}\morph_ckpt.py" "%RUN%\{name}\checkpoints\latest.pt" {name} > "%RUN%\{name}\morph.txt" 2>&1',
        "exit /b 0",
        "",
    ])
    (SCRATCH / f"run_{name}.cmd").write_text(cmd, encoding="ascii")
    print(f"staged {name}: views={views} steps={steps} stop={stop} cap={cfg['cap_max']} refine_stop={cfg['mcmc_refine_stop_iter']}")


if __name__ == "__main__":
    main()
