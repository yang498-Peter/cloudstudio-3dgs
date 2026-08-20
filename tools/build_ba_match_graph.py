#!/usr/bin/env python3
"""Build a deterministic training-only Rig BA match graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.ba.match_graph import (
    MatchGraphConfig,
    build_match_graph,
    write_hloc_pairs,
    write_match_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hloc-pairs", type=Path)
    parser.add_argument("--temporal-neighbors", type=int, default=4)
    parser.add_argument("--loop-max-distance-m", type=float, default=1.5)
    parser.add_argument("--loop-min-frame-gap", type=int, default=30)
    parser.add_argument("--loop-neighbors", type=int, default=4)
    args = parser.parse_args()

    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    result = build_match_graph(
        dataset,
        split,
        MatchGraphConfig(
            temporal_neighbor_rig_frames=args.temporal_neighbors,
            loop_max_distance_m=args.loop_max_distance_m,
            loop_min_frame_gap=args.loop_min_frame_gap,
            loop_neighbors_per_rig=args.loop_neighbors,
        ),
    )
    write_match_graph(args.output, result)
    if args.hloc_pairs is not None:
        write_hloc_pairs(args.hloc_pairs, result)
    print(
        f"BA match graph: rigs={result['summary']['training_rig_frames']}, "
        f"pairs={result['summary']['pair_count']}, validation images=0 -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
