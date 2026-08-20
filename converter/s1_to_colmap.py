#!/usr/bin/env python3
"""Convert an MVP S1 solver run into a COLMAP-format dataset for gsplat's
examples/simple_trainer.py (3DGUT route: --with_ut --with_eval3d, MCMC).

Output layout (what gsplat's colmap Parser expects):

    <out_dir>/
      images/                left_<ts>.jpg, right_<ts>.jpg  (copied frames)
      sparse/0/
        cameras.txt          OPENCV_FISHEYE, one camera per unique intrinsics (2: left/right)
        images.txt           world-to-camera quaternion/translation, OpenCV axes
        points3D.bin         LiDAR cloud subsample with RGB (init + no SfM tracks)

Pose math: solver transform_matrix is camera-to-world with OpenGL axes
(verified 2026-07-02, docs/S1_DATA_FORMAT.md §5); COLMAP wants world-to-camera
with OpenCV axes — see s1_common.solver_c2w_gl_to_w2c_cv.

Usage:
    python converter/s1_to_colmap.py --run-dir <process/run> --raw-dir <recording> \
        --out-dir <dataset_out> [--init-points 1000000] [--use-imgpose]

Use --use-imgpose to train from every raw camera frame that has a pose in
ImgPose.txt instead of only the solver-selected transforms.json keyframes.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

import numpy as np

from s1_common import load_transforms, rotmat_to_quat_wxyz, solver_c2w_gl_to_w2c_cv, subsample_las

INTRINSIC_KEYS = ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4")


def quat_xyzw_to_rotmat(q: list[float]) -> np.ndarray:
    """Normalized camera-to-world quaternion (x, y, z, w) -> rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0:
        raise ValueError("zero-length quaternion in ImgPose.txt")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_imgpose_frames(run_dir: Path, transforms: dict) -> list[dict]:
    """Load all c2w/OpenCV poses and attach side-specific fisheye intrinsics."""
    templates: dict[str, dict] = {}
    for frame in transforms["frames"]:
        rel = str(frame["file_path"]).replace("\\", "/")
        templates.setdefault(rel.split("/", 1)[0], frame)

    frames = []
    lines = (run_dir / "ImgPose.txt").read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 12:
            raise ValueError(f"ImgPose.txt line {line_no}: expected 12 fields, got {len(fields)}")
        rel = fields[0].replace("\\", "/")
        side = rel.split("/", 1)[0]
        if side not in templates:
            raise ValueError(f"ImgPose.txt line {line_no}: no intrinsics for camera side {side!r}")
        position = np.asarray([float(v) for v in fields[1:4]], dtype=np.float64)
        r_c2w = quat_xyzw_to_rotmat([float(v) for v in fields[7:11]])
        r_w2c = r_c2w.T
        t_w2c = -r_w2c @ position
        frame = {k: templates[side][k] for k in INTRINSIC_KEYS}
        frame.update({
            "file_path": rel,
            "_r_w2c": r_w2c,
            "_t_w2c": t_w2c,
        })
        frames.append(frame)
    return frames


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
    ap.add_argument("--raw-dir", type=Path,
                    help="recording root containing camera/; defaults to --run-dir")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--init-points", type=int, default=1_000_000)
    ap.add_argument("--image-source", choices=["auto", "camera", "undistort"], default="auto",
                    help="auto prefers raw camera/ images, then the run's 90-deg undistort/ fallback")
    ap.add_argument("--use-imgpose", action="store_true",
                    help="use every posed raw frame from ImgPose.txt (requires camera/ images)")
    args = ap.parse_args()

    tf = load_transforms(args.run_dir)
    frames = load_imgpose_frames(args.run_dir, tf) if args.use_imgpose else tf["frames"]
    raw_dir = args.raw_dir or args.run_dir

    raw_root = raw_dir / "camera"
    undistort_root = args.run_dir / "undistort"
    raw_matches = sum((raw_root / Path(str(f["file_path"]).replace("\\", "/"))).exists()
                      for f in frames)
    undistort_matches = sum((undistort_root / Path(str(f["file_path"]).replace("\\", "/"))).exists()
                            for f in frames)
    image_source = args.image_source
    if image_source == "auto":
        image_source = "camera" if raw_matches >= undistort_matches and raw_matches else "undistort"
    if args.use_imgpose and image_source != "camera":
        raise ValueError("--use-imgpose requires raw camera/ images; undistort keyframes are intentionally excluded")
    if image_source == "camera":
        image_root = raw_root
        camera_model = "OPENCV_FISHEYE"
        undistort_model = None
    else:
        image_root = undistort_root
        camera_model = "PINHOLE"
        undistort_model = tf.get("undistort_camera_model")
        if not undistort_model:
            raise ValueError("transforms.json has no undistort_camera_model metadata")
    matched = raw_matches if image_source == "camera" else undistort_matches
    if matched == 0:
        raise FileNotFoundError(f"no frame images found under {image_root}")
    print(f"image source: {image_source} ({matched}/{len(frames)} frames) -> {image_root}")

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
        src = image_root / rel
        if not src.exists():
            print(f"skip missing image {src}")
            continue
        if camera_model == "OPENCV_FISHEYE":
            key = (camera_model,) + tuple(f[k] for k in INTRINSIC_KEYS)
        else:
            intrinsic = undistort_model["intrinsic"]
            key = (
                camera_model,
                int(undistort_model["width"]), int(undistort_model["height"]),
                float(intrinsic[0][0]), float(intrinsic[1][1]),
                float(intrinsic[0][2]), float(intrinsic[1][2]),
            )
        cam_id = cameras.setdefault(key, len(cameras) + 1)
        name = rel.replace("/", "_")
        if not (img_out / name).exists():
            shutil.copy2(src, img_out / name)
            copied += 1
        if "_r_w2c" in f:
            r_w2c, t_w2c = f["_r_w2c"], f["_t_w2c"]
        else:
            r_w2c, t_w2c = solver_c2w_gl_to_w2c_cv(f["transform_matrix"])
        qw, qx, qy, qz = rotmat_to_quat_wxyz(r_w2c)
        tx, ty, tz = t_w2c
        image_lines.append(
            f"{i} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {tx:.12g} {ty:.12g} {tz:.12g} {cam_id} {name}"
        )
        image_lines.append("")  # no 2D point observations

    cam_lines = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"]
    for key, cam_id in sorted(cameras.items(), key=lambda kv: kv[1]):
        model = key[0]
        if model == "OPENCV_FISHEYE":
            p = dict(zip(INTRINSIC_KEYS, key[1:]))
            cam_lines.append(
                f"{cam_id} {model} {int(p['w'])} {int(p['h'])} "
                f"{p['fl_x']:.12g} {p['fl_y']:.12g} {p['cx']:.12g} {p['cy']:.12g} "
                f"{p['k1']:.12g} {p['k2']:.12g} {p['k3']:.12g} {p['k4']:.12g}"
            )
        else:
            _model, w, h, fx, fy, cx, cy = key
            cam_lines.append(
                f"{cam_id} {model} {w} {h} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}"
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
