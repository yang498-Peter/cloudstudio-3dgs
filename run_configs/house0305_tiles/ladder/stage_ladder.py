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
    # Spatial hold-out arms: 10% of the Tile's views (whole 2 m camera cells)
    # withheld from training; scored on those views with --holdout-from.
    "C2h": ("C2_da2", "C2 with a 10% spatial hold-out (2 m cells, seed 0)",
            _many(_set("holdout_spatial_cell_m", 2.0), _set("holdout_fraction", 0.1), _set("holdout_seed", 0),
                  _set("holdout_guard_m", 1.5)), WORK),
    "R5h": ("R5_cull05", "R5 with the same 10% spatial hold-out",
            _many(_set("holdout_spatial_cell_m", 2.0), _set("holdout_fraction", 0.1), _set("holdout_seed", 0),
                  _set("holdout_guard_m", 1.5)), WORK),
    "C5_norange": ("C2_da2", "C2 without the sparse LiDAR range term (moving objects such as doors are pulled to scan-time depth)",
                   _set("lidar_range_weight", 0.0), WORK),
    "C5h_norange": ("C2h", "C2h without the sparse LiDAR range term",
                    _set("lidar_range_weight", 0.0), WORK),
    "T2h_split05": ("C2h", "C2h + detail split gate 0.005 m",
                    _set("default_strategy.detail_split_scale_m", 0.005), WORK),
    "S1h_sizeprior": ("C2h", "C2h + size hinge at 2x the median initial size",
                      _many(_set("geometry_regularization.scale_upper_weight", 0.01),
                            _set("geometry_regularization.max_scale_ratio_to_reference", 2.0)), WORK),
    # Candidate final recipe: no sparse range + 5 mm adaptive split.
    "C5T2_split05": ("C5_norange", "C5 + detail split gate 0.005 m",
                     _set("default_strategy.detail_split_scale_m", 0.005), WORK),
    "C5T2h_split05": ("C5h_norange", "C5h + detail split gate 0.005 m",
                      _set("default_strategy.detail_split_scale_m", 0.005), WORK),
    "X7_C5T2": ("C5T2_split05", "C5T2 extended to 7,000 steps (two reset cycles)",
                _many(_set("controlled_stop_after_steps", 7000), _set("checkpoint_every", 3500),
                      _set("checkpoint_keep_every", 3500)), WORK),
    "X7h_C5T2": ("C5T2h_split05", "C5T2h extended to 7,000 steps with the hold-out",
                 _many(_set("controlled_stop_after_steps", 7000), _set("checkpoint_every", 3500),
                       _set("checkpoint_keep_every", 3500)), WORK),
    "X7_T2": ("T2c_split05", "T2c (C2 + 5 mm split) extended to 7,000 steps",
              _many(_set("controlled_stop_after_steps", 7000), _set("checkpoint_every", 3500),
                    _set("checkpoint_keep_every", 3500)), WORK),
    "X7h_T2": ("T2h_split05", "T2h extended to 7,000 steps with the hold-out",
               _many(_set("controlled_stop_after_steps", 7000), _set("checkpoint_every", 3500),
                     _set("checkpoint_keep_every", 3500)), WORK),
    # Integration-compatibility A/B: the vendor-parity lifecycle (reset 300,
    # 0.2 m split/clone, no alpha floor, no sparse range, DA2 0.5 compressed)
    # versus the same with our deferred 3000-step reset. Same seed and init.
    "P0_exact300": ("h5v", "vendor-parity lifecycle profile: reset 300, 0.2 m split/clone, no alpha floor, no sparse range, DA2 0.5 compressed",
                    _many(_set("default_strategy.vendor_opacity_reset_profile", "exact_every300"),
                          _set("default_strategy.reset_every", 300),
                          _set("lidar_alpha_weight", 0.0), _set("lidar_alpha_dilation_radius_px", 0),
                          _set("surface_alpha_floor_profile", False),
                          _set("lidar_range_weight", 0.0),
                          _set("da2_depth_weight", 0.5), _set("mono_depth_max_range_m", 30.0),
                          _set("da2_depth_space", "compressed")), WORK),
    "P0_exact3000": ("P0_exact300", "same profile with the CloudStudio deferred 3000-step reset (competitor_equivalent=false)",
                     _many(_set("default_strategy.vendor_opacity_reset_profile", "deferred_every3000_compatibility"),
                           _set("default_strategy.reset_every", 3000)), WORK),
    # Initial-density A/B on the parity profile: does a sparser initial
    # population survive the vendor 300-step reset in our integration?
    "P1_init4_r300": ("P0_exact300", "parity profile, initialization thinned to 1/4 (stride 4), reset 300",
                      _set("initialization_subsample_stride", 4), WORK),
    "P1_init4_r3000": ("P0_exact3000", "parity profile, initialization thinned to 1/4, deferred reset 3000",
                       _set("initialization_subsample_stride", 4), WORK),
    "P1_init2_r300": ("P0_exact300", "parity profile, initialization thinned to 1/2, reset 300",
                      _set("initialization_subsample_stride", 2), WORK),
    # Surface-anchoring hypothesis for the reset-300 collapse: does an explicit
    # surface anchor (our alpha floor, or the sparse range prior) let the
    # population survive the vendor cadence?
    "P2_floor_r300": ("P0_exact300", "parity profile + LiDAR alpha floor (0.5, 0.95, 6 px), reset 300",
                      _many(_set("lidar_alpha_weight", 0.5), _set("lidar_alpha_dilation_radius_px", 6),
                            _set("surface_alpha_floor_profile", True)), WORK),
    "P2_range_r300": ("P0_exact300", "parity profile + sparse LiDAR range 0.05 robust, reset 300",
                      _set("lidar_range_weight", 0.05), WORK),
    # Parity target alignment: mesh depth/normal terms from the LiDAR surface
    # mesh (per-Tile half-resolution raster) and DA2 aligned to that mesh.
    "P3_mesh_r300": ("P0_exact300", "parity profile + mesh depth 0.25 / normal 0.05 + mesh-aligned DA2 0.5, reset 300",
                     _many(_set("mesh_geometry_manifest", "C:/Peter/3dgs-datasets/house0305_sop_v8/mesh_geometry_tile0_r2/mesh_geometry_manifest.json"),
                           _set("mesh_geometry_root", "C:/Peter/3dgs-datasets/house0305_sop_v8/mesh_geometry_tile0_r2"),
                           _set("mesh_depth_weight", 0.25), _set("mesh_normal_weight", 0.05),
                           _set("mono_depth_manifest", "C:/Peter/3dgs-datasets/house0305_sop_v8/da2_tile0_meshaligned_manifest.json"),
                           _set("mono_depth_root", "C:/Peter/3dgs-datasets/house0305_sop_v8/da2_train"),
                           _set("mipmap_pipeline_gate", "C:/Peter/3dgs-runs/house0305_sop/gates/gate_18_training_multi_da2mesh.json")), WORK),
    "P3_mesh_r3000": ("P3_mesh_r300", "same with the deferred 3000-step reset",
                      _many(_set("default_strategy.vendor_opacity_reset_profile", "deferred_every3000_compatibility"),
                            _set("default_strategy.reset_every", 3000)), WORK),
    # Growth-rate hypothesis for the reset-300 collapse: if the vendor's
    # projected-gradient carrier selects more parents per refine, births
    # outpace the post-reset deaths. Lower thresholds from the approved set.
    "P4_grad75_r300": ("P0_exact300", "parity profile, grow_grad2d 7.5e-5 (approved calibrated profile), reset 300",
                       _set("default_strategy.grow_grad2d", 0.000075), WORK),
    "P4_grad100_r300": ("P0_exact300", "parity profile, grow_grad2d 1e-4, reset 300",
                        _set("default_strategy.grow_grad2d", 0.0001), WORK),
    # Opacity-dynamics audit: parity profile stopped at 700 with per-row opacity
    # logits dumped every 5 steps from the post-reset state at 600 to 699 so the rows that
    # die before the 700 cull can be attributed (visibility, gradient sign, birth kind, stacking).
    "P5_trace_r300": ("P0_exact300", "parity profile, opacity trace 600-699 every 5 (one refine-free window), controlled stop 700",
                      _many(_set("opacity_trace", {"start_step": 600, "stop_step": 699, "every": 5}),
                            _set("controlled_stop_after_steps", 700)), WORK),
    # Background-blending audit (integration, not a vendor fact): the parity
    # profile composites onto a per-view photographic backdrop, so lowering an
    # opacity reveals a plausible image; a constant background makes
    # transparency expensive. Judged by the reset+101 death fraction, not PSNR.
    "P6_bgblack_r300": ("P0_exact300", "parity profile, constant black background instead of the per-view backdrop, reset 300, stop 1000",
                        _many(_set("background_image_manifest", None), _set("background_image_root", None),
                              _set("background_color", [0.0, 0.0, 0.0]),
                              _set("controlled_stop_after_steps", 1000)), WORK),
    "P6_bgwhite_r300": ("P0_exact300", "parity profile, constant white background instead of the per-view backdrop, reset 300, stop 1000",
                        _many(_set("background_image_manifest", None), _set("background_image_root", None),
                              _set("background_color", [1.0, 1.0, 1.0]),
                              _set("controlled_stop_after_steps", 1000)), WORK),
    # Reset-state audit: the library reset keeps Adam's step counter, so the
    # ~100 post-reset opacity updates run at 2-4x lr; restart the counter.
    "P7_adamstep_r300": ("P0_exact300", "parity profile, Adam step counter restarted at each opacity reset, reset 300, stop 1000",
                         _many(_set("default_strategy.reset_adam_step", True),
                               _set("controlled_stop_after_steps", 1000)), WORK),
    # Cull-threshold audit: the vendor opacity cull constant came from a
    # disassembly whose threshold mapping was never settled; this asks whether
    # the same reset dynamics collapse under the published 0.005 threshold.
    "P8_cull005_r300": ("P0_exact300", "parity profile, opacity cull 0.005 (published threshold, audit profile), reset 300, stop 1300",
                        _many(_set("default_strategy.vendor_cull_warmup_profile", "audit_uniform_0p005"),
                              _set("default_strategy.prune_opa", 0.005), _set("default_strategy.prune_opa_late", 0.005),
                              _set("controlled_stop_after_steps", 1300)), WORK),
    # Enhancement track: the 7k extension doubled the population in the second
    # reset cycle without held-out gains; stop refinement before that wave.
    "X7h_T2_stop5k": ("T2h_split05", "T2h to 7,000 steps with refinement stopped at 5,000 (no second split wave)",
                      _many(_set("controlled_stop_after_steps", 7000), _set("checkpoint_every", 3500),
                            _set("checkpoint_keep_every", 3500),
                            _set("mcmc_refine_stop_iter", 5000),
                            _set("default_strategy.refine_stop_iter", 5000),
                            _set("default_strategy.refine_scale2d_stop_iter", 5000)), WORK),
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
