"""Shared helpers for MVP S1 -> 3DGS dataset converters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# OpenGL(nerfstudio) camera axes -> OpenCV camera axes basis flip
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def load_transforms(run_dir: Path) -> dict:
    import json

    return json.loads((run_dir / "transforms.json").read_text(encoding="utf-8"))


def subsample_las(run_dir: Path, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    """Stride-subsample the solver point cloud to ~n_target points.

    Returns (xyz float64 [N,3], rgb uint8 [N,3]). Handles both plain and
    project-prefixed LAS names (newer solver runs emit <project>_colorized.las).
    """
    import laspy

    candidates = ["colorized.las", "colorized.laz", "uncolorized.las"]
    las_path = None
    for name in candidates:
        if (run_dir / name).exists():
            las_path = run_dir / name
            break
    if las_path is None:
        prefixed = sorted(run_dir.glob("*_colorized.las")) or sorted(run_dir.glob("*_uncolorized.las"))
        if prefixed:
            las_path = prefixed[0]
    if las_path is None:
        raise FileNotFoundError(f"no LAS point cloud found in {run_dir}")

    xyzs, rgbs = [], []
    with laspy.open(las_path) as reader:
        stride = max(1, reader.header.point_count // n_target)
        for chunk in reader.chunk_iterator(2_000_000):
            xyzs.append(np.column_stack([chunk.x, chunk.y, chunk.z])[::stride])
            dims = set(chunk.point_format.dimension_names)
            if {"red", "green", "blue"} <= dims:
                rgb16 = np.column_stack([chunk.red, chunk.green, chunk.blue])[::stride]
                rgbs.append((rgb16 >> 8).astype(np.uint8))
            else:
                rgbs.append(np.full((len(xyzs[-1]), 3), 180, dtype=np.uint8))
    xyz, rgb = np.concatenate(xyzs), np.concatenate(rgbs)
    print(f"point cloud: {las_path.name} -> {len(xyz):,} points (stride {stride})")
    return xyz, rgb


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


def rotmat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z), COLMAP order."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def solver_c2w_gl_to_w2c_cv(transform_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Solver transform_matrix (c2w, OpenGL axes — verified 2026-07-02) ->
    COLMAP world-to-camera (R, t) with OpenCV axes."""
    m = np.asarray(transform_matrix, dtype=np.float64)
    r_c2w_cv = m[:3, :3] @ GL_TO_CV
    cam_center = m[:3, 3]
    r_w2c = r_c2w_cv.T
    t_w2c = -r_w2c @ cam_center
    return r_w2c, t_w2c
