#!/usr/bin/env python3
"""Convert an MVP S1 solver run into a COLMAP-format dataset for gsplat's
examples/simple_trainer.py (3DGUT route: --with_ut --with_eval3d, MCMC).

Output layout (what gsplat's colmap Parser expects):

    <out_dir>/
      images/                left_<ts>.jpg, right_<ts>.jpg  (copied keyframes)
      sparse/0/
        cameras.txt          OPENCV_FISHEYE, one camera per unique intrinsics (2: left/right)
        images.txt           world-to-camera quaternion/translation, OpenCV axes
        points3D.bin         LiDAR cloud subsample with RGB (init + no SfM tracks)

Pose math: solver transform_matrix is camera-to-world with OpenGL axes
(verified 2026-07-02, docs/S1_DATA_FORMAT.md §5); COLMAP wants world-to-camera
with OpenCV axes — see s1_common.solver_c2w_gl_to_w2c_cv.

Usage:
    python converter/s1_to_colmap.py --run-dir <process/run> --raw-dir <recording> \
        --out-dir <dataset_out> [--init-points 1000000]
"""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

import numpy as np

from s1_common import load_transforms, rotmat_to_quat_wxyz, solver_c2w_gl_to_w2c_cv, subsample_las

INTRINSIC_KEYS = ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4")


def write_points3d_bin(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """COLMAP points3D.bin with empty tracks (LiDAR points, not SfM points)."""
    rec = np.zeros(len(xyz), dtype=np.dtype(
        [("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3), ("err", "<f8"), ("tlen", "<u8")],
        align=False,
    ))
    rec["id"] = np.arange(1, len(xyz) + 1)
    rec["xyz"], rec["rgb"] = xyz, rgb
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(xyz)))
        rec.tofile(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--init-points", type=int, default=1_000_000)
    args = ap.parse_args()

    tf = load_transforms(args.run_dir)
    frames = tf["frames"]

    sparse = args.out_dir / "sparse" / "0"
    img_out = args.out_dir / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    # group frames by intrinsics -> COLMAP cameras (expect one per fisheye side)
    cameras: dict[tuple, int] = {}
    image_lines = []
    copied = 0
    for i, f in enumerate(sorted(frames, key=lambda f: str(f["file_path"])), start=1):
        rel = str(f["file_path"]).replace("\\", "/")
        src = args.raw_dir / "camera" / rel
        if not src.exists():
            print(f"skip missing image {src}")
            continue
        key = tuple(f[k] for k in INTRINSIC_KEYS)
        cam_id = cameras.setdefault(key, len(cameras) + 1)
        name = rel.replace("/", "_")
        if not (img_out / name).exists():
            shutil.copy2(src, img_out / name)
            copied += 1
        r_w2c, t_w2c = solver_c2w_gl_to_w2c_cv(f["transform_matrix"])
        qw, qx, qy, qz = rotmat_to_quat_wxyz(r_w2c)
        tx, ty, tz = t_w2c
        image_lines.append(
            f"{i} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {tx:.12g} {ty:.12g} {tz:.12g} {cam_id} {name}"
        )
        image_lines.append("")  # no 2D point observations

    cam_lines = ["# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 k3 k4"]
    for key, cam_id in sorted(cameras.items(), key=lambda kv: kv[1]):
        p = dict(zip(INTRINSIC_KEYS, key))
        cam_lines.append(
            f"{cam_id} OPENCV_FISHEYE {int(p['w'])} {int(p['h'])} "
            f"{p['fl_x']:.12g} {p['fl_y']:.12g} {p['cx']:.12g} {p['cy']:.12g} "
            f"{p['k1']:.12g} {p['k2']:.12g} {p['k3']:.12g} {p['k4']:.12g}"
        )
    (sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="ascii")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="ascii")

    xyz, rgb = subsample_las(args.run_dir, args.init_points)
    write_points3d_bin(sparse / "points3D.bin", xyz, rgb)

    n_imgs = len(image_lines) // 2
    print(f"dataset: {n_imgs} images ({copied} copied), {len(cameras)} cameras, "
          f"{len(xyz):,} init points -> {args.out_dir}")
    if len(cameras) > 2:
        print(f"NOTE: {len(cameras)} unique intrinsic sets (expected 2); "
              "per-frame intrinsics differ — check solver version behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
