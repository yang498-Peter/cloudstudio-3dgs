"""Is the poison segment a camera time-sync offset? Solve for delta-t.

The user's physical hypothesis: the two fisheyes expose independently (they
visibly differ in exposure brightness), so each camera's true pose belongs
to a slightly different trajectory time than its filename timestamp - and
during the fast close passes that shows as the un-fixable misregistration.

The test is quantitative and per-frame. If a time offset dt exists, the
measured LAS shift of frame i must equal the pixel displacement produced by
moving the camera along its OWN trajectory by dt:

    shift_i  =  J_i * dt        J_i = d(pixel)/d(t), measured numerically
                                by projecting the visible LAS points at the
                                pose interpolated a little later

so dt_i = <shift_i, J_i> / |J_i|^2 per frame, and the hypothesis stands if
dt_i is consistent within a camera (and the residual after removing J*dt
collapses). Slow segments have small |J| and say nothing - exactly why the
error only shows on the fast passes.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-gate1")

DATASETS = Path(r"C:\Peter\3dgs-datasets")
RECORDING = Path(r"C:\Peter\testdata\S1\house0305")
DEPTH_ROOT = DATASETS / "house0305_depth_ba"
SHIFTS = np.arange(-27, 28, 3)
PROBE_DT = 0.05  # seconds, for the numeric Jacobian
BANDS = ((0.30, 0.55, "control(good)"), (0.60, 0.95, "poison+edges"))
EVERY = 3


def las_sample(stride: int):
    with (RECORDING / "colorized.las").open("rb") as stream:
        header = stream.read(400)
        fmt = header[104] & 0x3F
        offset = struct.unpack("<I", header[96:100])[0]
        record = struct.unpack("<H", header[105:107])[0]
        count = struct.unpack("<I", header[107:111])[0]
        if header[24] >= 1 and header[25] >= 4:
            big = struct.unpack("<Q", header[247:255])[0]
            if big:
                count = big
        scale = np.array(struct.unpack("<3d", header[131:155]))
        shift = np.array(struct.unpack("<3d", header[155:179]))
        stream.seek(offset)
        raw = np.frombuffer(stream.read(count * record), dtype=np.uint8).reshape(
            count, record)[::stride]
    xyz = raw[:, :12].copy().view("<i4").reshape(-1, 3).astype(np.float64)
    rgb_offset = {2: 20, 3: 28, 7: 30, 8: 30}[fmt]
    rgb = raw[:, rgb_offset:rgb_offset + 6].copy().view("<u2").reshape(-1, 3)
    return xyz * scale + shift, rgb.astype(np.float32).mean(axis=1) / 65535.0


def interpolate_pose(before: np.ndarray, after: np.ndarray, alpha: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp
    rotations = Rotation.from_matrix(np.stack([before[:3, :3], after[:3, :3]]))
    rotation = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = before[:3, 3] * (1 - alpha) + after[:3, 3] * alpha
    return pose


def main() -> int:
    manifest = json.loads(
        (DATASETS / "house0305_manifest" / "dataset_manifest.json")
        .read_text(encoding="utf-8"))
    cameras = {c["camera_id"]: c for c in manifest["cameras"]}
    depth_manifest = json.loads(
        (DEPTH_ROOT / "depth_manifest.json").read_text(encoding="utf-8"))
    depth_by_id = {d["image_id"]: d for d in depth_manifest["images"]}
    xyz, gray = las_sample(60)

    # Per-camera timelines: pose neighbours come from the SAME camera so the
    # finite difference is that camera's own motion.
    per_camera: dict[str, list[dict]] = {}
    for image in manifest["images"]:
        per_camera.setdefault(str(image["camera_id"]), []).append(image)
    for images in per_camera.values():
        images.sort(key=lambda i: int(i["timestamp_ns"]))
    times = sorted(int(i["timestamp_ns"]) for i in manifest["images"])
    t0, span = times[0], times[-1] - times[0]

    print(f"{'cam':>6} {'t%':>4} {'|shift|':>8} {'|J|px/s':>8} "
          f"{'dt_ms':>7} {'resid':>6}")
    results: dict[str, list] = {"left": [], "right": []}
    for camera_id, images in per_camera.items():
        camera = cameras[camera_id]
        intr, dist = camera["intrinsic"], camera["distortion"]["params"]
        for index in range(1, len(images) - 1, EVERY):
            image = images[index]
            fraction = (int(image["timestamp_ns"]) - t0) / span
            if not any(lo <= fraction < hi for lo, hi, _ in BANDS):
                continue
            entry = depth_by_id.get(image["image_id"])
            if entry is None:
                continue
            c2w = np.asarray(image["c2w"], dtype=np.float64)
            nxt = images[index + 1]
            gap = (int(nxt["timestamp_ns"]) - int(image["timestamp_ns"])) / 1e9
            if not (0.1 < gap < 2.0):
                continue
            c2w_next = np.asarray(nxt["c2w"], dtype=np.float64)

            def project(pose, pts=xyz):
                w2c = np.linalg.inv(pose)
                local = pts @ w2c[:3, :3].T + w2c[:3, 3]
                forward = local[:, 2]
                rng = np.linalg.norm(local, axis=1)
                keep = (forward > 0.5) & (rng < 25.0)
                xy = local[keep, :2] / forward[keep, None]
                r = np.linalg.norm(xy, axis=1)
                theta = np.arctan2(r, 1.0)
                t2 = theta * theta
                fct = theta * (1 + dist["k1"] * t2 + dist["k2"] * t2**2
                               + dist["k3"] * t2**3 + dist["k4"] * t2**4)
                s = np.where(r > 1e-9, fct / np.maximum(r, 1e-9), 1.0)
                u = intr["fl_x"] * xy[:, 0] * s + intr["cx"]
                v = intr["fl_y"] * xy[:, 1] * s + intr["cy"]
                return keep, u, v, np.degrees(theta), rng[keep]

            keep, u, v, theta, rng = project(c2w)
            with np.load(DEPTH_ROOT / entry["path"]) as archive:
                height, width = (int(x) for x in archive["shape"])
                depth = np.zeros(height * width, dtype=np.float32)
                depth[archive["pixel_index"]] = archive["range_m"]
                depth = depth.reshape(height, width)
            ui = np.round(u).astype(int)
            vi = np.round(v).astype(int)
            inside = ((ui >= 32) & (ui < width - 32) & (vi >= 32)
                      & (vi < height - 32) & (theta < 80))
            ui, vi = ui[inside], vi[inside]
            visible = (depth[vi, ui] > 0) & (np.abs(depth[vi, ui] - rng[inside]) < 0.5)
            if visible.sum() < 300:
                continue
            indices = np.flatnonzero(keep)[inside][visible]
            ui, vi = ui[visible], vi[visible]
            grays = gray[indices]

            with Image.open(RECORDING / image["path"]) as handle:
                photo = np.asarray(handle.convert("L")).astype(np.float32) / 255.0
            best = (-2.0, 0, 0)
            for dy in SHIFTS:
                for dx in SHIFTS:
                    c = float(np.corrcoef(photo[vi + dy, ui + dx], grays)[0, 1])
                    if c > best[0]:
                        best = (c, dx, dy)
            measured = np.array([best[1], best[2]], dtype=np.float64)

            # Numeric d(pixel)/dt: reproject the SAME LAS points at the pose
            # a probe interval later on this camera's own trajectory.
            probe_pose = interpolate_pose(c2w, c2w_next, PROBE_DT / gap)
            keep2, u2, v2, _, _ = project(probe_pose)
            table_u = np.full(len(xyz), np.nan)
            table_v = np.full(len(xyz), np.nan)
            table_u[np.flatnonzero(keep2)] = u2
            table_v[np.flatnonzero(keep2)] = v2
            du = table_u[indices] - u.astype(np.float64)[inside][visible]
            dv = table_v[indices] - v.astype(np.float64)[inside][visible]
            good = ~np.isnan(du)
            if good.sum() < 200:
                continue
            jacobian = np.array([np.median(du[good]), np.median(dv[good])]) / PROBE_DT
            j_norm = float(np.linalg.norm(jacobian))
            if j_norm < 1.0:
                continue  # too slow here; the frame cannot vote
            dt = float(measured @ jacobian) / (j_norm ** 2)
            residual = float(np.linalg.norm(measured - jacobian * dt))
            results[camera_id].append((fraction, float(np.linalg.norm(measured)),
                                       j_norm, dt, residual))
            print(f"{camera_id:>6} {fraction*100:>3.0f}% "
                  f"{np.linalg.norm(measured):>8.1f} {j_norm:>8.1f} "
                  f"{dt*1000:>7.1f} {residual:>6.1f}")

    print("\nper-camera dt (frames with |J|>=30 px/s, i.e. fast enough to vote):")
    for camera_id, rows in results.items():
        fast = [(f, m, j, dt, r) for f, m, j, dt, r in rows if j >= 30]
        if not fast:
            print(f"  {camera_id}: no fast frames")
            continue
        dts = np.array([dt for _, _, _, dt, _ in fast]) * 1000
        residuals = np.array([r for *_, r in fast])
        shifts = np.array([m for _, m, *_ in fast])
        print(f"  {camera_id}: n={len(fast)}  dt p50 {np.median(dts):+.1f}ms "
              f"p25/p75 {np.percentile(dts,25):+.1f}/{np.percentile(dts,75):+.1f}ms"
              f"  |shift| p50 {np.median(shifts):.1f}px -> residual p50 "
              f"{np.median(residuals):.1f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
