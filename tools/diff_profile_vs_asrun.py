#!/usr/bin/env python3
"""Diff the tile configs an SDK profile derives for house0305 against as-run tile configs.

Prints, per trainer key, where the profile's derived delivery config disagrees with the
config a tile actually trained with. Paths, ids, lineage and the scene path block are
ignored: they belong to the dataset and the run, not to the recipe. Used to turn a
research candidate (b12op05d3 / b13op05d0) into a profile and to check the result.

    python tools/diff_profile_vs_asrun.py PROFILE TILE0.json TILE1.json TILE2.json TILE3.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio3dgs_sdk.plan import DatasetSummary, TileSummary, build_plan  # noqa: E402
from cloudstudio3dgs_sdk.profile import get_profile  # noqa: E402

# house0305 v9 tile inventory with the previous populations the b5sky adopt recorded
# (sdk_house0305_b5sky/prepare/prepare_manifest.json); every later delivery inherited
# b5sky's caps, so these are what make the floor rule reproduce Tile_2's 6.8M.
TILES = ((0, 2132, 7044777, 8480130), (1, 1829, 3417320, 4654937), (2, 1684, 3309574, 6389532),
         (3, 2317, 5651827, 7587560))
# The campaign card: 16303 MiB usable.
VRAM_GIB = 16303 / 1024
IGNORED_PREFIXES = ("lineage", "output_dir", "run_id", "resume_checkpoint",
                    "warm_start_checkpoint")
IGNORED_SUFFIXES = ("_manifest", "_root", "_path", "_ply", "_checkpoint", "gsplat_lock", "recording_root",
                    "mipmap_pipeline_gate", "initialization_geometry")


def flat(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, inner in value.items():
            out.update(flat(inner, f"{prefix}.{key}" if prefix else key))
    else:
        out[prefix] = value
    return out


def ignored(key: str) -> bool:
    leaf = key.rsplit(".", 1)[-1]
    return key.startswith(IGNORED_PREFIXES) or leaf.endswith(IGNORED_SUFFIXES) or leaf in IGNORED_SUFFIXES


def main() -> int:
    profile = get_profile(sys.argv[1])
    asrun = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[2:6]]
    dataset = DatasetSummary(
        scene_tag="house0305",
        tiles=tuple(TileSummary(t, f"Tile_{t}", v, p, previous_final_population=prev) for t, v, p, prev in TILES),
        train_view_count=3536,
        global_init_point_count=1863918,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "repo" / "upstream").mkdir(parents=True)
        plan = build_plan(profile, dataset, dataset_root=root / "dataset", work_root=root / "work",
                          repo_root=root / "repo", python=Path("python.exe"), vram_gib=VRAM_GIB,
                          prior_tile_checkpoints={t: f"prior{t}.pt" for t, *_ in TILES})
    derived = {}
    for step in plan.steps:
        if step.config is not None and step.name.startswith("train_tile") and step.name.endswith("_delivery"):
            derived[int(step.name[len("train_tile"):].split("_", 1)[0])] = flat(dict(step.config))
    disagreements = {}
    for tile_id, config in enumerate(asrun):
        want, got = flat(config), derived[tile_id]
        for key in sorted(set(want) | set(got)):
            if ignored(key):
                continue
            if want.get(key, "<absent>") != got.get(key, "<absent>"):
                disagreements.setdefault(key, []).append((tile_id, got.get(key, "<absent>"), want.get(key, "<absent>")))
    for key, rows in disagreements.items():
        cells = "; ".join(f"T{t}: profile {g!r} vs as-run {w!r}" for t, g, w in rows)
        print(f"{key}: {cells}")
    print(f"{len(disagreements)} key(s) disagree")
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
