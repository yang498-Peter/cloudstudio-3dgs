#!/usr/bin/env python3
"""Inspect an MVP S1 recording: verify the files a 3DGS pipeline needs are present.

Usage:
    python tools/inspect_recording.py <recording_dir>

Read-only. Reports raw-input markers, calibration summary, image counts and
solver runs (process/<run>/) with their pose/point-cloud artifacts.
See docs/S1_DATA_FORMAT.md for the format spec this checks against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAW_MARKERS = [
    "metadata.yaml", "metadata.yml", "data/data_raw.mcap", "data/data.mcap",
    "odom-realtime.csv", "colorized-realtime.las", "colorized-realtime.laz",
]

RUN_ARTIFACTS = [
    "transforms.json", "ImgPose.txt", "odom.csv",
    "colorized.las", "uncolorized.las", "colorized.laz", "uncolorized.laz",
]


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def check(path: Path) -> bool:
    ok = path.exists()
    size = f" ({fmt_size(path.stat().st_size)})" if ok and path.is_file() else ""
    print(f"  [{'x' if ok else ' '}] {path.name}{size}")
    return ok


def summarize_calibration(calib_path: Path) -> None:
    if not calib_path.exists():
        print("  [ ] info/calibration.json  MISSING — cannot build 3DGS input")
        return
    data = json.loads(calib_path.read_text(encoding="utf-8"))
    print(f"  [x] info/calibration.json  (calibrated {data.get('calibration_time', '?')})")
    for cam in data.get("cameras", []):
        intr = cam.get("intrinsic", {})
        dist = cam.get("distortion", {})
        model = dist.get("camera_model", "?")
        params = dist.get("params", {})
        print(
            f"      {cam.get('name'):>5}: {cam.get('width')}x{cam.get('height')} {model} "
            f"f=({intr.get('fl_x', 0):.1f},{intr.get('fl_y', 0):.1f}) "
            f"c=({intr.get('cx', 0):.1f},{intr.get('cy', 0):.1f}) "
            f"k=[{', '.join(f'{params.get(k, 0):.4g}' for k in ('k1', 'k2', 'k3', 'k4'))}]"
        )
        if "transform_from_lidar" not in cam:
            print("      WARNING: no transform_from_lidar (camera-LiDAR extrinsics)")


def summarize_run(run_dir: Path) -> None:
    print(f"\nsolver run: {run_dir.name}")
    have = {name: check(run_dir / name) for name in RUN_ARTIFACTS}
    tf = run_dir / "transforms.json"
    if have["transforms.json"]:
        data = json.loads(tf.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
        left = sum(1 for f in frames if str(f.get("file_path", "")).replace("\\", "/").startswith("left"))
        print(f"      frames: {len(frames)} (left {left} / right {len(frames) - left}), "
              f"keyframe thresholds: {data.get('metainfo', {})}")
    imgpose = run_dir / "ImgPose.txt"
    if have["ImgPose.txt"]:
        lines = imgpose.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"      ImgPose.txt: {max(0, len(lines) - 1)} image poses")
    geo = run_dir / "geo"
    if geo.is_dir():
        print(f"      geo/: present — ECEF/geo artifacts exist; DO NOT feed these to 3DGS training")
    und = run_dir / "undistort"
    if und.is_dir():
        n = sum(1 for _ in (und / "left").glob("*.jpg")) if (und / "left").is_dir() else 0
        print(f"      undistort/: present ({n} left images, 90-deg pinhole fallback route)")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    rec = Path(sys.argv[1])
    if not rec.is_dir():
        print(f"not a directory: {rec}")
        return 2

    print(f"recording: {rec}\n\nraw input markers:")
    marker_hits = sum(check(rec / m) for m in RAW_MARKERS)

    print("\ncalibration:")
    summarize_calibration(rec / "info" / "calibration.json")

    print("\ncamera images:")
    for side in ("left", "right"):
        d = rec / "camera" / side
        n = sum(1 for _ in d.glob("*.jpg")) if d.is_dir() else 0
        print(f"  camera/{side}: {n} jpg")

    runs = sorted(p for p in (rec / "process").glob("*") if p.is_dir()) if (rec / "process").is_dir() else []
    if not runs:
        print("\nno process/ solver runs found — run the mvps1 solver (metacam_cli) first;"
              "\n3DGS needs its ImgPose/transforms + colorized point cloud output.")
    for run in runs:
        summarize_run(run)

    print(f"\nverdict: {'looks usable' if marker_hits and runs else 'incomplete'} "
          f"({marker_hits} raw markers, {len(runs)} solver runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
