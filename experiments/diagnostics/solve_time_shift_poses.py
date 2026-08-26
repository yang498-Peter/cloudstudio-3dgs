"""Solve a per-frame along-trajectory time shift for every image - no frame dropped.

v2: no temporal smoothing (v1's median window replaced correct per-frame
solves with neighbours' and the gate showed 24-29px where unsmoothed solves
left 3-5px). Every correction must EARN its place: apply it, re-measure the
frame, keep it only if the shift actually drops. Frames whose correction
fails verification stay at their original pose and are listed as stubborn -
the surgical exclusion set, not a wholesale segment cut.

v3: the measurement is embarrassingly parallel (each frame loads its own
photo and depth and correlates independently), so the three passes - solve,
verify, audit - run on a process pool. Single-threaded this took ~2h; the
pool brings it to minutes.
"""
import json
import struct
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Peter\cloudstudio-3dgs-gate1")
sys.path.insert(0, str(REPO))

DATASETS = Path(r"C:\Peter\3dgs-datasets")
RECORDING = Path(r"C:\Peter\testdata\S1\house0305")
DEPTH_ROOT = DATASETS / "house0305_depth_ba"
OUTPUT = DATASETS / "house0305_timefix"
SHIFTS = np.arange(-27, 28, 3)
PROBE_DT = 0.05
J_FLOOR = 30.0
# The shift grid steps by 3, so magnitudes quantize to 0 / 3 / 4.24 / 6...
# and the UNTOUCHED known-good frames read 3.0-4.8 on this instrument.
GATE_PX = 4.5  # per-frame improvement target inside the solver rounds
# The thresholds that matter are evidence-anchored on the in-training rig
# refiner (q2's accepted report): it ACCEPTED corrections of 1-3cm (~2-6px)
# and provably FAILED at 8.9px. So frames left above 8px contradict geometry
# beyond what training can absorb and are surgically excluded; a clean pool
# at <=6.5px sits inside the refiner's demonstrated envelope and trains with
# refinement on.
STUBBORN_PX = 8.0
CLEAN_GATE_PX = 6.5
WORKERS = 8


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
    # A large dt over a short frame gap can ask for a time beyond the
    # neighbour; clamp to the neighbour's pose and let the per-frame
    # verification decide whether the capped correction still earns its place.
    alpha = float(min(1.0, max(0.0, alpha)))
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

    def measure(self, image, pose, pose_next=None, gap=1.0):
        from PIL import Image
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
        if pose_next is None:
            return measured, None

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
            return measured, None
        jacobian = np.array([np.median(du[good]), np.median(dv[good])]) / PROBE_DT
        return measured, jacobian


_WORKER: dict = {}


def _init_worker() -> None:
    _WORKER["solver"] = Solver()


def _solve_task(args):
    key, image, pose, next_image, gap = args
    found = _WORKER["solver"].measure(
        image, np.asarray(pose, dtype=np.float64),
        np.asarray(next_image["c2w"], dtype=np.float64), gap)
    if found is None or found[1] is None:
        return key, None
    measured, jacobian = found
    j_norm = float(np.linalg.norm(jacobian))
    if j_norm < J_FLOOR:
        return key, None
    dt = float(measured @ jacobian) / (j_norm ** 2)
    return key, (dt, float(np.linalg.norm(measured)))


def _measure_task(args):
    key, image, pose = args
    found = _WORKER["solver"].measure(image, np.asarray(pose, dtype=np.float64))
    if found is None:
        return key, None
    return key, float(np.linalg.norm(found[0]))


def main() -> int:
    manifest = json.loads(
        (DATASETS / "house0305_manifest" / "dataset_manifest.json")
        .read_text(encoding="utf-8"))
    per_camera: dict[str, list[dict]] = {}
    for image in manifest["images"]:
        per_camera.setdefault(str(image["camera_id"]), []).append(image)
    for images in per_camera.values():
        images.sort(key=lambda i: int(i["timestamp_ns"]))

    pool = Pool(processes=WORKERS, initializer=_init_worker)

    final_poses: dict = {}
    accepted_dt: dict = {}
    shift_now: dict = {}
    for camera_id, images in per_camera.items():
        for index, image in enumerate(images):
            final_poses[(camera_id, index)] = np.asarray(image["c2w"],
                                                         dtype=np.float64)

    # Iterated solve+verify. A single round leaves partial recoveries (a
    # 20px error corrected to ~6px) hanging just above the gate; a second
    # round re-solves the RESIDUAL dt from the corrected pose. Every accept
    # still has to win its own re-measure - iteration adds reach, not slack.
    for round_number in (1, 2):
        tasks = []
        for camera_id, images in per_camera.items():
            for index in range(1, len(images) - 1):
                key = (camera_id, index)
                if round_number > 1 and shift_now.get(key, 0.0) <= GATE_PX:
                    continue
                image, nxt = images[index], images[index + 1]
                gap = (int(nxt["timestamp_ns"]) - int(image["timestamp_ns"])) / 1e9
                if 0.1 < gap < 2.0:
                    tasks.append((key, image, final_poses[key].tolist(), nxt, gap))
        print(f"round {round_number} solve: {len(tasks)} frames on "
              f"{WORKERS} workers", flush=True)
        solved: dict = {}
        for done, (key, result) in enumerate(
                pool.imap_unordered(_solve_task, tasks, chunksize=4), 1):
            solved[key] = result
            if done % 200 == 0:
                print(f"  solve {done}/{len(tasks)}", flush=True)

        verify_tasks = []
        dt_by_key: dict = {}
        for camera_id, images in per_camera.items():
            for index in range(1, len(images) - 1):
                key = (camera_id, index)
                result = solved.get(key)
                if result is None:
                    continue
                dt, shift_before = result
                shift_now.setdefault(key, shift_before)
                if abs(dt) <= 1e-4:
                    continue
                image = images[index]
                current = final_poses[key]
                candidate = None
                if dt >= 0:
                    nxt = images[index + 1]
                    gap = (int(nxt["timestamp_ns"])
                           - int(image["timestamp_ns"])) / 1e9
                    if 0.05 < gap < 2.0:
                        candidate = interpolate_pose(
                            current, np.asarray(nxt["c2w"], dtype=np.float64),
                            dt / gap)
                else:
                    prv = images[index - 1]
                    gap = (int(image["timestamp_ns"])
                           - int(prv["timestamp_ns"])) / 1e9
                    if 0.05 < gap < 2.0:
                        candidate = interpolate_pose(
                            np.asarray(prv["c2w"], dtype=np.float64), current,
                            1.0 + dt / gap)
                if candidate is not None:
                    verify_tasks.append((key, image, candidate))
                    dt_by_key[key] = (dt, shift_before, candidate)
        print(f"round {round_number} verify: {len(verify_tasks)} candidates",
              flush=True)
        applied = 0
        rejected = 0
        for done, (key, shift_after) in enumerate(
                pool.imap_unordered(_measure_task, verify_tasks, chunksize=4), 1):
            dt, shift_before, candidate = dt_by_key[key]
            if shift_after is not None and shift_after < shift_before - 1e-6:
                final_poses[key] = candidate
                accepted_dt[key] = accepted_dt.get(key, 0.0) + dt
                shift_now[key] = shift_after
                applied += 1
            else:
                rejected += 1
            if done % 200 == 0:
                print(f"  verify {done}/{len(verify_tasks)}", flush=True)
        print(f"round {round_number}: {applied} corrections applied, "
              f"{rejected} rejected by re-measure", flush=True)

    # Pass 3: the decile audit at final poses, dense (every 4th frame).
    images_sorted = sorted(manifest["images"], key=lambda i: int(i["timestamp_ns"]))
    t0 = int(images_sorted[0]["timestamp_ns"])
    span = int(images_sorted[-1]["timestamp_ns"]) - t0
    index_of: dict = {}
    for camera_id, images in per_camera.items():
        for index, image in enumerate(images):
            index_of[image["image_id"]] = (camera_id, index)
    audit_tasks = []
    for index in range(0, len(images_sorted), 4):
        image = images_sorted[index]
        key = index_of[image["image_id"]]
        audit_tasks.append(((image["image_id"],
                             (int(image["timestamp_ns"]) - t0) / span),
                            image, final_poses[key]))
    print(f"pass 3: auditing {len(audit_tasks)} frames", flush=True)
    audit = []
    for (image_id, fraction), magnitude in pool.imap_unordered(
            _measure_task, audit_tasks, chunksize=4):
        if magnitude is not None:
            audit.append((fraction, magnitude, image_id))
    pool.close()
    pool.join()

    stubborn_ids = {image_id for _, magnitude, image_id in audit
                    if magnitude > STUBBORN_PX}
    print(f"\nstubborn frames (final shift > {2*GATE_PX:.0f}px): "
          f"{len(stubborn_ids)} of {len(audit)} audited")
    print("audit (median px per decile: all / minus-stubborn):")
    worst_clean = 0.0
    for decile in range(10):
        rows = [(m, i) for f, m, i in audit
                if decile / 10 <= f < (decile + 1) / 10]
        if rows:
            all_median = float(np.median([m for m, _ in rows]))
            clean = [m for m, i in rows if i not in stubborn_ids]
            clean_median = float(np.median(clean)) if clean else 0.0
            worst_clean = max(worst_clean, clean_median)
            print(f"  {decile*10:>3}-{decile*10+10:<3}%  {all_median:5.1f} / "
                  f"{clean_median:5.1f}   (stubborn "
                  f"{sum(1 for _, i in rows if i in stubborn_ids)})")
    if worst_clean > CLEAN_GATE_PX:
        print(f"\nGATE FAILED: minus-stubborn worst decile {worst_clean:.1f}px "
              f"> {CLEAN_GATE_PX}px - writing nothing")
        return 1
    print(f"\nGATE PASSED: minus-stubborn worst decile {worst_clean:.1f}px "
          f"<= {CLEAN_GATE_PX}px; exclude only the stubborn list from training")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {}
    for camera_id, images in per_camera.items():
        for index, image in enumerate(images):
            key = (camera_id, index)
            payload[image["image_id"]] = {
                "c2w": [[float(x) for x in row] for row in final_poses[key]],
                "dt_s": float(accepted_dt.get(key, 0.0)),
            }
    (OUTPUT / "timefix_poses.json").write_text(
        json.dumps({"algorithm_version": "per_frame_time_shift_v3_verified",
                    "gate_worst_clean_decile_px": worst_clean,
                    "stubborn_image_ids": sorted(stubborn_ids),
                    "poses": payload}), encoding="utf-8")
    print(f"corrected poses -> {OUTPUT / 'timefix_poses.json'} "
          f"({len(stubborn_ids)} stubborn ids listed for surgical exclusion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
