"""Solve a per-frame along-trajectory time shift for every image - no frame dropped.

The time-sync probe proved the poison segment's misregistration is explained
by moving each camera along ITS OWN trajectory: the measured LAS shift
aligns with the velocity direction and the residual collapses to the audit
floor (6.7px -> 2.6/4.7px). The offset is not a constant calibration (the
per-frame dt spread is tens of ms), so this solves dt for EVERY image:

    dt_i  =  <measured_shift_i, J_i> / |J_i|^2      (J_i = numeric px/s)
    pose_i = interpolate(neighbour poses of the same camera, t_i + dt_i)

Slow frames (|J| below the vote floor) keep dt = 0 - they already audit at
the floor, which is exactly why the error only ever showed on fast passes.
A light median smoothing along each camera's timeline suppresses single
frame estimation noise; the fail-closed decile audit then decides whether
the corrected pose set may exist at all.
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
OUTPUT = DATASETS / "house0305_timefix"
SHIFTS = np.arange(-27, 28, 3)
PROBE_DT = 0.05
J_FLOOR = 30.0
SMOOTH_WINDOW = 5
GATE_PX = 4.0


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


class Solver:
    def __init__(self):
        self.manifest = json.loads(
            (DATASETS / "house0305_manifest" / "dataset_manifest.json")
            .read_text(encoding="utf-8"))
        self.cameras = {c["camera_id"]: c for c in self.manifest["cameras"]}
        depth_manifest = json.loads(
            (DEPTH_ROOT / "depth_manifest.json").read_text(encoding="utf-8"))
        self.depth_by_id = {d["image_id"]: d for d in depth_manifest["images"]}
        self.xyz, self.gray = las_sample(60)

    def project(self, camera, pose):
        intr, dist = camera["intrinsic"], camera["distortion"]["params"]
        w2c = np.linalg.inv(pose)
        local = self.xyz @ w2c[:3, :3].T + w2c[:3, 3]
        forward = local[:, 2]
        rng = np.linalg.norm(local, axis=1)
        keep = (forward > 0.5) & (rng < 25.0)
        xy = local[keep, :2] / forward[keep, None]
        r = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(r, 1.0)
        t2 = theta * theta
        factor = theta * (1 + dist["k1"] * t2 + dist["k2"] * t2**2
                          + dist["k3"] * t2**3 + dist["k4"] * t2**4)
        s = np.where(r > 1e-9, factor / np.maximum(r, 1e-9), 1.0)
        u = intr["fl_x"] * xy[:, 0] * s + intr["cx"]
        v = intr["fl_y"] * xy[:, 1] * s + intr["cy"]
        return keep, u, v, np.degrees(theta), rng[keep]

    def measure(self, image, pose, pose_next, gap):
        camera = self.cameras[image["camera_id"]]
        entry = self.depth_by_id.get(image["image_id"])
        if entry is None:
            return None
        keep, u, v, theta, rng = self.project(camera, pose)
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
            return None
        indices = np.flatnonzero(keep)[inside][visible]
        ui, vi = ui[visible], vi[visible]
        grays = self.gray[indices]
        with Image.open(RECORDING / image["path"]) as handle:
            photo = np.asarray(handle.convert("L")).astype(np.float32) / 255.0
        best = (-2.0, 0, 0)
        for dy in SHIFTS:
            for dx in SHIFTS:
                c = float(np.corrcoef(photo[vi + dy, ui + dx], grays)[0, 1])
                if c > best[0]:
                    best = (c, dx, dy)
        measured = np.array([best[1], best[2]], dtype=np.float64)

        probe_pose = interpolate_pose(pose, pose_next, PROBE_DT / gap)
        keep2, u2, v2, _, _ = self.project(camera, probe_pose)
        table_u = np.full(len(self.xyz), np.nan)
        table_v = np.full(len(self.xyz), np.nan)
        table_u[np.flatnonzero(keep2)] = u2
        table_v[np.flatnonzero(keep2)] = v2
        du = table_u[indices] - u.astype(np.float64)[inside][visible]
        dv = table_v[indices] - v.astype(np.float64)[inside][visible]
        good = ~np.isnan(du)
        if good.sum() < 200:
            return None
        jacobian = np.array([np.median(du[good]), np.median(dv[good])]) / PROBE_DT
        return measured, jacobian


def main() -> int:
    solver = Solver()
    per_camera: dict[str, list[dict]] = {}
    for image in solver.manifest["images"]:
        per_camera.setdefault(str(image["camera_id"]), []).append(image)
    for images in per_camera.values():
        images.sort(key=lambda i: int(i["timestamp_ns"]))

    corrected: dict[str, list] = {}
    for camera_id, images in per_camera.items():
        dts = np.zeros(len(images))
        votes = np.zeros(len(images), dtype=bool)
        for index in range(1, len(images) - 1):
            image = images[index]
            pose = np.asarray(image["c2w"], dtype=np.float64)
            nxt = images[index + 1]
            gap = (int(nxt["timestamp_ns"]) - int(image["timestamp_ns"])) / 1e9
            if not (0.1 < gap < 2.0):
                continue
            found = solver.measure(image, pose,
                                   np.asarray(nxt["c2w"], dtype=np.float64), gap)
            if found is None:
                continue
            measured, jacobian = found
            j_norm = float(np.linalg.norm(jacobian))
            if j_norm < J_FLOOR:
                continue
            dts[index] = float(measured @ jacobian) / (j_norm ** 2)
            votes[index] = True
            if index % 60 == 0:
                print(f"  {camera_id} {index}/{len(images)}: "
                      f"dt {dts[index]*1000:+.1f}ms", flush=True)
        # Median smoothing over voting neighbours; non-voters stay at zero
        # unless surrounded by voters (then they inherit the local median).
        smoothed = dts.copy()
        half = SMOOTH_WINDOW // 2
        for index in range(len(images)):
            window = [dts[j] for j in range(max(0, index - half),
                                            min(len(images), index + half + 1))
                      if votes[j]]
            if window:
                smoothed[index] = float(np.median(window))
        print(f"{camera_id}: {votes.sum()} voting frames, "
              f"dt p50 {np.median(np.abs(smoothed[votes]))*1000:.1f}ms "
              f"max {np.abs(smoothed).max()*1000:.1f}ms", flush=True)
        corrected[camera_id] = []
        for index, image in enumerate(images):
            pose = np.asarray(image["c2w"], dtype=np.float64)
            dt = float(smoothed[index])
            if abs(dt) > 1e-4:
                if dt >= 0 and index + 1 < len(images):
                    nxt = images[index + 1]
                    gap = (int(nxt["timestamp_ns"]) - int(image["timestamp_ns"])) / 1e9
                    if 0.05 < gap < 2.0:
                        pose = interpolate_pose(
                            pose, np.asarray(nxt["c2w"], dtype=np.float64), dt / gap)
                elif dt < 0 and index > 0:
                    prv = images[index - 1]
                    gap = (int(image["timestamp_ns"]) - int(prv["timestamp_ns"])) / 1e9
                    if 0.05 < gap < 2.0:
                        pose = interpolate_pose(
                            np.asarray(prv["c2w"], dtype=np.float64), pose,
                            1.0 + dt / gap)
            corrected[camera_id].append((image["image_id"], pose, dt))

    # Fail-closed audit on the corrected poses, the standard instrument.
    flat = {image_id: pose for rows in corrected.values()
            for image_id, pose, _ in rows}
    images_sorted = sorted(solver.manifest["images"],
                           key=lambda i: int(i["timestamp_ns"]))
    t0 = int(images_sorted[0]["timestamp_ns"])
    span = int(images_sorted[-1]["timestamp_ns"]) - t0
    audit = []
    for index in range(0, len(images_sorted), 12):
        image = images_sorted[index]
        pose = flat[image["image_id"]]
        found = solver.measure(image, pose, pose, 1.0)
        if found is None:
            continue
        measured, _ = found
        audit.append(((int(image["timestamp_ns"]) - t0) / span,
                      float(np.linalg.norm(measured))))
    print("\ntime-shift-corrected audit (median px per decile):")
    worst = 0.0
    for decile in range(10):
        data = [m for f, m in audit if decile / 10 <= f < (decile + 1) / 10]
        if data:
            median = float(np.median(data))
            worst = max(worst, median)
            print(f"  {decile*10:>3}-{decile*10+10:<3}%  {median:5.1f}")
    if worst > GATE_PX:
        print(f"\nGATE FAILED: worst decile {median if False else worst:.1f}px "
              f"> {GATE_PX}px - writing nothing")
        return 1
    print(f"\nGATE PASSED: worst decile {worst:.1f}px <= {GATE_PX}px")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {image_id: {"c2w": [[float(x) for x in row] for row in pose],
                          "dt_s": dt}
               for rows in corrected.values() for image_id, pose, dt in rows}
    (OUTPUT / "timefix_poses.json").write_text(
        json.dumps({"algorithm_version": "per_frame_time_shift_v1",
                    "gate_worst_decile_px": worst,
                    "poses": payload}), encoding="utf-8")
    print(f"corrected poses -> {OUTPUT / 'timefix_poses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
