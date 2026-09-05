#!/usr/bin/env python3
"""Derive Tile_1..3 configs from a winning Tile_0 arm config.

The four-tile delivery must run the same recipe on every tile. This copies
the recipe fields (default_strategy, sh_degree, cap_max, loss weights) from
the Tile_0 arm into the existing per-tile delivery configs, which keep their
tile-specific inputs (initialisation PLY, backgrounds, tile id).

    python tools/stage_tile_arm_configs.py --tile0 RUN/tile0_C1_cap15m_20k.json \
        --template RUN/tile{t}_G9d_delivery_20k.json --tag C1d --cap 15000000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

RECIPE_KEYS = (
    "default_strategy",
    "sh_degree",
    "color_model",
    "cap_max",
    "lidar_alpha_weight",
    "lidar_alpha_target",
    "lidar_range_weight",
    "lidar_range_loss_mode",
    "lidar_normal_alignment",
    "da2_depth_weight",
    "geometry_regularization",
    "learning_rates",
    "controlled_stop_after_steps",
    "seed",
)

# Per-tile schedule values that must survive the recipe copy: max_steps is
# 20 epochs of that tile's own view count and prune_switch_step is half of it.
TILE_SPECIFIC_STRATEGY_KEYS = ("prune_switch_step",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tile0", type=Path, required=True)
    parser.add_argument("--template", required=True,
                        help="per-tile template path with {t} placeholder")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cap", type=int, help="override cap_max for every tile")
    parser.add_argument("--tiles", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    tile0 = json.loads(args.tile0.read_text(encoding="utf-8"))
    run_root = args.tile0.parent
    for t in args.tiles:
        template = Path(args.template.format(t=t))
        cfg = json.loads(template.read_text(encoding="utf-8"))
        changed = {}
        for key in RECIPE_KEYS:
            if key not in tile0 or cfg.get(key) == tile0[key]:
                continue
            value = json.loads(json.dumps(tile0[key]))
            if key == "default_strategy":
                for keep in TILE_SPECIFIC_STRATEGY_KEYS:
                    if keep in cfg.get(key, {}):
                        value[keep] = cfg[key][keep]
                if value == cfg.get(key):
                    continue
            changed[key] = (cfg.get(key), value)
            cfg[key] = value
        if args.cap is not None:
            changed["cap_max"] = (cfg.get("cap_max"), args.cap)
            cfg["cap_max"] = args.cap
        name = f"tile{t}_{args.tag}_20k"
        cfg["run_id"] = f"house0305-t{t}-{args.tag}"
        cfg["output_dir"] = str(run_root / name)
        out = run_root / f"{name}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out)
        summary = {k: v for k, v in changed.items() if k != "default_strategy"}
        if "default_strategy" in changed:
            old, new = changed["default_strategy"]
            summary["default_strategy"] = {
                k: (old.get(k), new.get(k)) for k in new if old.get(k) != new.get(k)
            }
        print(name, "tile_id", cfg.get("mipmap_tile_id"), "changes", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
