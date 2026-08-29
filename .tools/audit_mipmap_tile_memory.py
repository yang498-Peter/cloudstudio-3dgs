"""Audit per-Tile RAM/VRAM behavior from a completed MipMap task.

Read-only: parses plaintext MemoryProfile rows and immutable milestone times.
The output separates the tiler's planning estimate from observed CUDA usage.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


PROFILE = re.compile(
    r"\[(?P<time>\d{4}-\d{2}-\d{2}:\d{2}\.\d{2}\.\d{2})\]"
    r"\[MemoryProfile\] \[RAM\] Avail (?P<ram_avail>[\d.]+) GB "
    r"Used (?P<ram_used>[\d.]+) GB .*?"
    r"\[G-RAM\] Avail (?P<vram_avail>[\d.]+) GB "
    r"Used (?P<vram_used>[\d.]+) GB"
)


def parse_profiles(log_path: Path) -> list[dict[str, Any]]:
    rows = []
    # The vendor file interleaves UTF-8/ASCII plaintext status with opaque
    # encrypted payload bytes. Replacement is read-only and affects only
    # undecodable payload; MemoryProfile rows themselves are ASCII.
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROFILE.search(line)
        if not match:
            continue
        rows.append(
            {
                "time": datetime.strptime(match["time"], "%Y-%m-%d:%H.%M.%S"),
                "ram_available_gib": float(match["ram_avail"]),
                "ram_used_gib": float(match["ram_used"]),
                "vram_available_gib": float(match["vram_avail"]),
                "vram_used_gib": float(match["vram_used"]),
            }
        )
    return rows


def local_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def nearest_before(rows: list[dict[str, Any]], moment: datetime) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["time"] < moment]
    return candidates[-1] if candidates else None


def nearest_after(rows: list[dict[str, Any]], moment: datetime) -> dict[str, Any] | None:
    return next((row for row in rows if row["time"] > moment), None)


def compact_profile(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "time": row["time"].isoformat(sep=" "),
        "ram_used_gib": row["ram_used_gib"],
        "vram_used_gib": row["vram_used_gib"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result_root = args.task_root / "result"
    log_path = result_root / "logs" / "log.txt"
    tile_plan_path = result_root / "task" / "tiles.json"
    plan = json.loads(tile_plan_path.read_text(encoding="utf-8"))
    planned = {tile["name"]: tile for tile in plan["tiles"]}
    profiles = parse_profiles(log_path)
    rows = []

    for tile_name in sorted(planned, key=lambda value: int(value.split("_")[-1])):
        point_cloud = result_root / "milestones" / "point_cloud" / f"{tile_name}_point_cloud.pb.bin"
        splat_dir = result_root / "milestones" / "splats" / tile_name
        level_zero = splat_dir / "gaussian_splat_level_0.pb.bin"
        levels_info = splat_dir / "levels_info.json"
        start = local_mtime(point_cloud)
        trained = local_mtime(level_zero)
        finished = local_mtime(levels_info)
        training_samples = [row for row in profiles if start <= row["time"] <= trained]
        whole_tile_samples = [row for row in profiles if start <= row["time"] <= finished]
        vram = [row["vram_used_gib"] for row in training_samples]
        ram = [row["ram_used_gib"] for row in training_samples]
        before = nearest_before(profiles, start)
        after = nearest_after(profiles, finished)
        rows.append(
            {
                "tile": tile_name,
                "planning_max_memory_gib": float(planned[tile_name]["max_memory"]),
                "point_cloud_ready": start.isoformat(sep=" "),
                "training_level_0_ready": trained.isoformat(sep=" "),
                "lod_levels_ready": finished.isoformat(sep=" "),
                "training_seconds": (trained - start).total_seconds(),
                "lod_seconds": (finished - trained).total_seconds(),
                "profile_samples_during_training": len(training_samples),
                "profile_samples_point_cloud_to_lod": len(whole_tile_samples),
                "observed_training_ram_gib": {
                    "minimum": min(ram) if ram else None,
                    "median": median(ram) if ram else None,
                    "p95": percentile(ram, 0.95),
                    "peak": max(ram) if ram else None,
                },
                "observed_training_vram_gib": {
                    "minimum": min(vram) if vram else None,
                    "median": median(vram) if vram else None,
                    "p95": percentile(vram, 0.95),
                    "peak": max(vram) if vram else None,
                },
                "profile_before_point_cloud": compact_profile(before),
                "profile_after_lod": compact_profile(after),
                "point_cloud_bytes": point_cloud.stat().st_size,
                "level_0_splat_bytes": level_zero.stat().st_size,
            }
        )

    serial_gaps = []
    for first, second in zip(rows, rows[1:]):
        first_end = datetime.fromisoformat(first["lod_levels_ready"])
        second_start = datetime.fromisoformat(second["point_cloud_ready"])
        serial_gaps.append(
            {
                "from": first["tile"],
                "to": second["tile"],
                "seconds_between_previous_lod_and_next_point_cloud": (
                    second_start - first_end
                ).total_seconds(),
                "overlap": second_start < first_end,
            }
        )

    result = {
        "task_root": str(args.task_root),
        "evidence": {
            "profile_log": str(log_path),
            "tile_plan": str(tile_plan_path),
            "profile_sample_count": len(profiles),
            "log_decode": "UTF-8 with replacement for opaque vendor payload bytes",
            "tile_phase_rule": (
                "point-cloud PB mtime -> level-0 splat PB mtime is the observed "
                "training window; level-0 -> levels_info is LOD serialization."
            ),
        },
        "important_boundary": (
            "planning_max_memory_gib is the adaptive tiler's image-load estimate; "
            "it is not a CUDA allocation limit and must not be compared as if it "
            "were nvidia-smi used memory."
        ),
        "tiles": rows,
        "serial_execution": serial_gaps,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
