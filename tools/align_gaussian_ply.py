#!/usr/bin/env python3
"""Register a foreign Gaussian model onto this dataset's LiDAR frame.

The competitor's house0305 model renders beautifully but from the wrong frame:
measured on our validation views it scored agreement 0.012 - not "bad quality"
but "pointing somewhere else entirely", while its bounding extents suggested an
x/y swap against colorized.las. This solves the rigid transform, because every
comparison downstream (per-view metrics, side-by-side crops) is meaningless
until the model sits in the frame the camera poses live in.

Method, deliberately boring: both clouds are gravity-aligned metric scans of
the same building, so the transform is yaw + translation to good approximation.
A coarse search rasterizes both clouds to 2D occupancy grids and FFT
cross-correlates them per yaw step (global optimum over translation for each
yaw, no initial guess needed), the z offset comes from the median difference,
and point-to-plane-free ICP on subsamples refines the full SE(3). Gaussian
quaternions rotate along with the means (q' = r * q).

    python tools/align_gaussian_ply.py --checkpoint usa_gs.pt \
        --reference-las colorized.las --output usa_gs_aligned.pt
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CELL_M = 0.25
YAW_STEP_DEG = 1.0
# The first run stopped at 30 iterations still moving and left a 5.1 cm median
# residual, which at 1.5 m viewing distance smears a render by ~13 px - the
# comparison is meaningless at that level. Iterate to convergence and trim
# progressively tighter as the fit improves.
ICP_ITERS = 150
ICP_SUBSAMPLE = 500_000
SEED = 42


def read_las_xyz(path: Path, stride: int = 1) -> np.ndarray:
    """Minimal LAS 1.2-1.4 point reader, xyz only, honoring scale/offset."""
    with path.open("rb") as stream:
        header = stream.read(400)
        if header[:4] != b"LASF":
            raise ValueError(f"not a LAS file: {path}")
        offset_to_points = struct.unpack("<I", header[96:100])[0]
        record_length = struct.unpack("<H", header[105:107])[0]
        count = struct.unpack("<I", header[107:111])[0]
        if header[24] >= 1 and header[25] >= 4:
            count64 = struct.unpack("<Q", header[247:255])[0]
            if count64:
                count = count64
        scale = np.array(struct.unpack("<3d", header[131:155]))
        offset = np.array(struct.unpack("<3d", header[155:179]))
        stream.seek(offset_to_points)
        raw = np.frombuffer(
            stream.read(count * record_length), dtype=np.uint8
        ).reshape(count, record_length)
    xyz_int = (
        raw[::stride, :12].copy().view("<i4").reshape(-1, 3).astype(np.float64)
    )
    return xyz_int * scale + offset


def occupancy(points: np.ndarray, cell: float, size: int, centre: np.ndarray):
    grid = np.zeros((size, size), dtype=np.float32)
    ij = np.floor((points[:, :2] - centre[None, :2]) / cell).astype(np.int64) + size // 2
    keep = (ij >= 0).all(axis=1) & (ij < size).all(axis=1)
    np.add.at(grid, (ij[keep, 0], ij[keep, 1]), 1.0)
    return np.minimum(grid, 8.0)  # cap so dense walls don't dominate


def coarse_yaw_translation(model: np.ndarray, reference: np.ndarray):
    """FFT cross-correlation over a yaw sweep; returns yaw, xy shift, score."""
    span = max(
        np.ptp(reference[:, 0]), np.ptp(reference[:, 1]),
        np.ptp(model[:, 0]), np.ptp(model[:, 1]),
    )
    size = int(2 ** np.ceil(np.log2(span / CELL_M * 1.5)))
    centre_ref = np.median(reference, axis=0)
    grid_ref = occupancy(reference, CELL_M, size, centre_ref)
    fft_ref = np.fft.rfft2(grid_ref)

    centre_model = np.median(model, axis=0)
    best = (0.0, np.zeros(2), -np.inf)
    for yaw_deg in np.arange(0.0, 360.0, YAW_STEP_DEG):
        yaw = np.deg2rad(yaw_deg)
        cos, sin = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos, -sin], [sin, cos]])
        rotated = (model[:, :2] - centre_model[None, :2]) @ rot.T + centre_ref[None, :2]
        grid = occupancy(
            np.concatenate([rotated, model[:, 2:3]], axis=1), CELL_M, size, centre_ref
        )
        corr = np.fft.irfft2(np.conj(np.fft.rfft2(grid)) * fft_ref, s=grid_ref.shape)
        peak = np.unravel_index(np.argmax(corr), corr.shape)
        score = float(corr[peak])
        if score > best[2]:
            shift = np.array(
                [(peak[0] + size // 2) % size - size // 2,
                 (peak[1] + size // 2) % size - size // 2],
                dtype=np.float64) * CELL_M
            best = (yaw, shift, score)
    return best, centre_model, centre_ref


def icp_refine(model: np.ndarray, reference: np.ndarray, transform: np.ndarray):
    from scipy.spatial import cKDTree

    tree = cKDTree(reference)
    current = transform.copy()
    for iteration in range(ICP_ITERS):
        moved = model @ current[:3, :3].T + current[:3, 3]
        distance, index = tree.query(moved, workers=-1)
        # Trim non-overlapping regions; tighten as the fit improves so the
        # final iterations align on the well-matched core.
        trim = 70.0 if iteration < 20 else 60.0 if iteration < 60 else 50.0
        keep = distance < np.percentile(distance, trim)
        source = moved[keep]
        target = reference[index[keep]]
        mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - mu_s).T @ (target - mu_t))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = mu_t - rotation @ mu_s
        current = delta @ current
        step = np.linalg.norm(delta[:3, 3])
        if step < 1e-4:
            break
    moved = model @ current[:3, :3].T + current[:3, 3]
    distance, _ = tree.query(moved, workers=-1)
    return current, float(np.median(distance)), int(iteration + 1)


def rotate_quaternions(quats: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Left-multiply gaussian orientations by the transform's rotation."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_matrix(rotation)
    # Checkpoint layout is [w, x, y, z]; scipy uses [x, y, z, w].
    xyzw = np.concatenate([quats[:, 1:4], quats[:, 0:1]], axis=1)
    rotated = (r * Rotation.from_quat(xyzw)).as_quat()
    return np.concatenate([rotated[:, 3:4], rotated[:, 0:3]], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-las", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = payload["params"]
    means = params["means"].numpy().astype(np.float64)

    rng = np.random.default_rng(SEED)
    model_sub = means[rng.choice(len(means), ICP_SUBSAMPLE, replace=False)]
    reference = read_las_xyz(args.reference_las, stride=1)
    ref_sub = reference[rng.choice(len(reference), ICP_SUBSAMPLE, replace=False)]
    print(f"model {len(means):,} gaussians, reference {len(reference):,} points")

    (yaw, shift, score), centre_model, centre_ref = coarse_yaw_translation(
        model_sub, ref_sub
    )
    print(f"coarse: yaw {np.rad2deg(yaw):.1f} deg, shift {shift}, score {score:.0f}")

    cos, sin = np.cos(yaw), np.sin(yaw)
    coarse = np.eye(4)
    coarse[:2, :2] = [[cos, -sin], [sin, cos]]
    coarse[:3, 3] = np.array([
        centre_ref[0] + shift[0], centre_ref[1] + shift[1], centre_ref[2]
    ]) - coarse[:3, :3] @ centre_model
    # z from median difference after xy alignment
    coarse[2, 3] += np.median(ref_sub[:, 2]) - np.median(
        (model_sub @ coarse[:3, :3].T + coarse[:3, 3])[:, 2]
    )

    transform, residual_m, iters = icp_refine(model_sub, ref_sub, coarse)
    print(f"icp: {iters} iters, median nn distance {residual_m * 100:.1f} cm")
    if residual_m > 0.15:
        print("ALIGNMENT FAILED: residual above 15 cm - do not trust the output",
              file=sys.stderr)
        return 1

    rotation = transform[:3, :3]
    params["means"] = torch.from_numpy(
        (means @ rotation.T + transform[:3, 3]).astype(np.float32)
    )
    params["quats"] = torch.from_numpy(
        rotate_quaternions(params["quats"].numpy().astype(np.float64), rotation)
        .astype(np.float32)
    )
    payload["alignment"] = {
        "reference_las": str(args.reference_las),
        "transform": transform.tolist(),
        "median_nn_distance_m": residual_m,
        "coarse_yaw_deg": float(np.rad2deg(yaw)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"aligned checkpoint -> {args.output}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload["alignment"], indent=1), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
