"""Promote a winning short arm to a full-length per-tile run.

Usage: stage_full_run.py <arm_tag> <tile_id>

The arms were 1800-step probes on Tile_0. A full run keeps every lifecycle
setting the arm established and changes only what is tied to run length or to
which tile is being trained.
"""
import json
import sys
from pathlib import Path

RUN = Path(r"C:\Peter\3dgs-runs\house0305_sop")
# steps = 20 x view count, per the established schedule
# max_steps is contractually 20 x view count: epoch-permutation sampling
# requires exactly twenty complete epochs, so the schedule cannot be shortened.
TILE_STEPS = {0: 49560, 1: 26940, 2: 41860}

# Stop early instead, which leaves the schedule intact. Measured: the dead-mass
# fraction climbs monotonically (0.527 at step 501 to 0.754 by 19501, no
# plateau) and windowed PSNR peaks at 16.32 around step 5-10k then declines, so
# more training accumulates near-invisible Gaussians that composite into haze.
# Running the full 20 epochs would land near 85-90% dead mass.
TILE_STOP = {0: 12000, 1: 8000, 2: 10000}
TILE_INIT = {0: 10_445_142, 1: 3_309_574, 2: 5_651_827}

arm = sys.argv[1] if len(sys.argv) > 1 else "armD"
tile = int(sys.argv[2]) if len(sys.argv) > 2 else 0

cfg = json.loads((RUN / f"tile0_{arm}.json").read_text(encoding="utf-8"))
steps = TILE_STEPS[tile]

cfg["run_id"] = f"house0305-t{tile}-full"
cfg["output_dir"] = str(RUN / f"tile{tile}_full")
cfg["mipmap_tile_id"] = tile
cfg["max_steps"] = steps
cfg["controlled_stop_after_steps"] = TILE_STOP[tile]
cfg["checkpoint_every"] = 2478
# Two retained snapshots, not four: at ~2.7 GB each on a 14M population these
# are the largest thing this job writes, and the delivery only needs latest
# plus the golden best. Retention is for post-hoc analysis.
cfg["checkpoint_keep_every"] = steps // 2

# The validator derives these from max_steps; keep them consistent or it rejects.
cfg["default_strategy"]["prune_switch_step"] = steps // 2
cfg["mcmc_refine_stop_iter"] = int(steps * 0.78)
cfg["default_strategy"]["refine_stop_iter"] = int(steps * 0.78)
cfg["default_strategy"]["refine_scale2d_stop_iter"] = int(steps * 0.78)

# Capacity: hold near init with headroom for redistribution, not for hoarding.
cfg["cap_max"] = int(TILE_INIT[tile] * 1.34)

# The arms were all Tile_0, so every per-tile artifact path in the config still
# points at Tile_0. The validator checks these against the signed manifest and
# rejects a mismatch, so repoint each one at the tile actually being trained.
for key in ("initialization_geometry", "initialization_ply"):
    if key in cfg:
        cfg[key] = str(cfg[key]).replace("Tile_0", f"Tile_{tile}")

# Backdrops cropped to this Tile's rectangle. The shared full-frame set is
# wrong for a cropped view: the library resizes rather than crops, so the loss
# composites a squashed copy of the whole frame. Half of Tile_0's views and
# most of Tile_1's are cropped, so this is the majority case, not an edge one.
tile_backgrounds = RUN / "tile_backgrounds" / f"Tile_{tile}"
cfg["background_image_manifest"] = str(tile_backgrounds / "background_manifest.json")
cfg["background_image_root"] = str(tile_backgrounds)

out = RUN / f"tile{tile}_full.json"
out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
print(f"staged {out.name}: tile={tile} steps={steps:,} cap={cfg['cap_max']:,}")
print(f"  refine_stop={cfg['mcmc_refine_stop_iter']:,} prune_switch={steps//2:,}")
print(f"  from arm={arm}")
