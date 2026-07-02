#!/usr/bin/env python3
"""Convert an MVP S1 solver run into a nerfstudio/gsplat-consumable dataset.

The solver already emits NeRF-style poses (transforms.json), so this is a
normalizer, not a SfM replacement:

  1. frames: fix file_path separators, point them at the raw camera/ images,
     add camera_model=OPENCV_FISHEYE, optionally rewrite transform_matrix into
     the target convention (set --pose-convention from reproject_check results).
  2. point cloud: subsample colorized.las to --init-points and write PLY with
     RGB, referenced as ply_file_path (gsplat/nerfstudio init).
  3. output dir: transforms.json + sparse_pc.ply + images/ (symlink or copy).

Pose convention: pinned on 2026-07-02 via tools/reproject_check.py + numeric
cross-check against ImgPose.txt — solver transform_matrix is camera-to-world
with OpenGL/nerfstudio axes (c2w_gl), i.e. passthrough. See docs/S1_DATA_FORMAT.md §5.

Usage:
    python converter/s1_to_nerfstudio.py --run-dir <process/run> --raw-dir <recording> \
        --out-dir <dataset_out> [--init-points 1000000] [--copy-images]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def to_nerfstudio_c2w(mat: np.ndarray, convention: str) -> np.ndarray:
    """Rewrite the solver pose matrix into nerfstudio's OpenGL c2w convention."""
    m = np.asarray(mat, dtype=np.float64)
    if convention.startswith("w2c"):
        m = np.linalg.inv(m)
    # nerfstudio expects OpenGL camera axes (x right, y up, z backward).
    # If the solver matrix is OpenCV-axes (_cv), flip the camera y/z basis vectors.
    if convention.endswith("_cv"):
        m = m @ FLIP
    return m


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.zeros(len(xyz), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    rec["xyz"], rec["rgb"] = xyz.astype(np.float32), rgb.astype(np.uint8)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        rec.tofile(fh)


def subsample_las(run_dir: Path, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    import laspy

    las_path = next(p for name in ("colorized.las", "colorized.laz", "uncolorized.las")
                    if (p := run_dir / name).exists())
    xyzs, rgbs = [], []
    with laspy.open(las_path) as reader:
        stride = max(1, reader.header.point_count // n_target)
        for chunk in reader.chunk_iterator(2_000_000):
            xyzs.append(np.column_stack([chunk.x, chunk.y, chunk.z])[::stride])
            if all(dim in chunk.point_format.dimension_names for dim in ("red", "green", "blue")):
                # LAS RGB is 16-bit
                rgbs.append((np.column_stack([chunk.red, chunk.green, chunk.blue])[::stride] >> 8))
            else:
                rgbs.append(np.full((len(xyzs[-1]), 3), 180, dtype=np.uint16))
    return np.concatenate(xyzs), np.concatenate(rgbs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--init-points", type=int, default=1_000_000)
    ap.add_argument("--pose-convention", default="c2w_gl",
                    choices=["c2w_cv", "c2w_gl", "w2c_cv", "w2c_gl"],
                    help="verified c2w_gl for solver transforms.json (2026-07-02); "
                         "override only if reproject_check says otherwise for a new SDK version")
    ap.add_argument("--copy-images", action="store_true", help="copy instead of referencing raw images")
    args = ap.parse_args()

    src = json.loads((args.run_dir / "transforms.json").read_text(encoding="utf-8"))
    out_frames = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    img_out = args.out_dir / "images"
    img_out.mkdir(exist_ok=True)

    for f in src["frames"]:
        rel = str(f["file_path"]).replace("\\", "/")
        src_img = args.raw_dir / "camera" / rel
        if not src_img.exists():
            print(f"skip missing {src_img}")
            continue
        dst_name = rel.replace("/", "_")
        if args.copy_images:
            shutil.copy2(src_img, img_out / dst_name)
        c2w = to_nerfstudio_c2w(f["transform_matrix"], args.pose_convention)
        out_frames.append({
            "file_path": f"images/{dst_name}" if args.copy_images else str(src_img),
            "w": f["w"], "h": f["h"],
            "fl_x": f["fl_x"], "fl_y": f["fl_y"], "cx": f["cx"], "cy": f["cy"],
            "k1": f["k1"], "k2": f["k2"], "k3": f["k3"], "k4": f["k4"],
            "camera_model": "OPENCV_FISHEYE",
            "transform_matrix": c2w.tolist(),
        })

    xyz, rgb = subsample_las(args.run_dir, args.init_points)
    write_ply(args.out_dir / "sparse_pc.ply", xyz, rgb)

    out = {
        "camera_model": "OPENCV_FISHEYE",
        "frames": out_frames,
        "ply_file_path": "sparse_pc.ply",
        "source": {"run_dir": str(args.run_dir), "pose_convention": args.pose_convention,
                   "solver_metainfo": src.get("metainfo", {})},
    }
    (args.out_dir / "transforms.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(out_frames)} frames + {len(xyz):,} init points -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
