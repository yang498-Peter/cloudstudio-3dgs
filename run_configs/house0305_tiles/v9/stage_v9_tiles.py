"""Stage the F6 arm: the F5 recipe (vendor reset morphology, 7.5e-5 growth,
5 mm detail split, T2h supervision bundle, compressed schedule) on the v9
all-photo dataset with occlusion-filtered LiDAR targets, one config per tile."""
import json, sys
from pathlib import Path
RUN = Path(r"C:\Peter\3dgs-runs\house0305_sop"); V8 = r"C:\Peter\3dgs-datasets\house0305_sop_v8"; V9 = r"C:\Peter\3dgs-datasets\house0305_sop_v9"
ARM = "F6_v9_allphoto_vis6_20k"
base = json.loads((RUN / "tile0_F5_keep_split05_t2h_20k.json").read_text(encoding="utf-8"))
tiles = json.loads((RUN / "tile_inputs_v9" / "tile_inputs_manifest.json").read_text(encoding="utf-8"))["tiles"]
rep = {V8 + r"\split_manifest.json": V9 + r"\split_manifest.json", V8 + r"\face4\face_manifest.json": V9 + r"\face4_train\face_manifest.json", V8 + r"\face4": V9 + r"\face4_train",
       V8 + r"\renderer_mask_train.json": V9 + r"\renderer_mask_train.json", V8 + r"\face4_lidar_train\face_lidar_geometry_manifest.json": V9 + r"\face4_lidar_train_vis6\face_lidar_geometry_manifest.json",
       V8 + r"\face4_lidar_train": V9 + r"\face4_lidar_train_vis6", V8 + r"\da2_train\mono_depth_manifest.json": V9 + r"\da2_train\mono_depth_manifest.json", V8 + r"\da2_train": V9 + r"\da2_train",
       str(RUN / "tile_inputs_multi"): str(RUN / "tile_inputs_v9"), str(RUN / "tile_geometry_multi"): str(RUN / "tile_geometry_v9"),
       str(RUN / "tile_backgrounds"): str(RUN / "tile_backgrounds_v9"), str(RUN / "gates" / "gate_17_training_multi_da2.json"): str(RUN / "gates_v9" / "gate_17_training_da2.json")}
def fix(v):
    if isinstance(v, str):
        for a, b in rep.items():
            if v == a: return b
        for a, b in rep.items():
            if v.startswith(a + "\\"): return b + v[len(a):]
    return v
for t in tiles:
    tid = int(t["tile_id"]); views = int(t["view_count"]); steps = 20 * views
    c = {k: fix(v) for k, v in base.items()}
    for k in ("resume_checkpoint", "holdout_spatial_cell_m", "holdout_guard_m", "holdout_fraction", "holdout_seed"): c.pop(k, None)
    name = f"tile{tid}_{ARM}"
    c["run_id"] = f"house0305-t{tid}-{ARM}"; c["output_dir"] = str(RUN / name); c["mipmap_tile_id"] = tid
    c["max_steps"] = steps; c["controlled_stop_after_steps"] = 20000
    c["checkpoint_every"] = 5000; c["checkpoint_keep_every"] = 1000000
    rs = 14000; c["mcmc_refine_stop_iter"] = rs; c["default_strategy"]["refine_stop_iter"] = rs; c["default_strategy"]["refine_scale2d_stop_iter"] = rs
    c["default_strategy"]["prune_switch_step"] = steps // 2
    c["cap_max"] = int(t["initialization"]["point_count"] * 1.34)
    for key in ("initialization_ply", "initialization_geometry", "background_image_manifest", "background_image_root"):
        c[key] = c[key].replace("Tile_0", f"Tile_{tid}")
    (RUN / f"{name}.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    leftovers = [k for k, v in c.items() if isinstance(v, str) and ("sop_v8\face4" in v or "tile_inputs_multi" in v or "gates\gate_" in v or "face4_lidar_train\\" in v)]
    print(f"staged {name}: views={views} max_steps={steps} stop=20000 cap={c['cap_max']} v8-leftovers={leftovers}")
