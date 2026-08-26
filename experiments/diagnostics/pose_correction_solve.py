"""Closed-form per-rig pose correction from LAS shift-search peaks.

The in-training SE3 refiner fixed what it could reach and left deciles 7-9
at 9-24px: a 20px misalignment sits outside the photometric gradient's
convergence basin. This tool reaches it from OUTSIDE the render loop:

  1. for every rig frame, both cameras: project the colorized LAS cloud,
     occlusion-verify against the frame's z-buffered depth, and find the
     photo shift (coarse +-24 step 4, then fine step 1) that maximizes
     LAS-vs-photo correlation - the same instrument that diagnosed the
     drift, now used as the measurement for its cure;
  2. solve one small world-side rotation per rig (axis-angle about the rig
     centre - the SAME parameterization the trainer's refiner uses, so the
     two compose) by least squares over both cameras' measured (dx, dy),
     with the 2x3 pixel-per-radian Jacobians computed numerically - no
     sign conventions to get wrong;
  3. median-smooth the corrections along the capture timeline (drift is
     continuous; single-frame outliers are measurement noise) and zero the
     rigs whose measurement is unconfident either way;
  4. re-measure the corrected poses with the standard decile audit and
     REFUSE to write anything if any decile stays above the gate;
  5. emit a corrected dataset manifest copy (c2w updated, re-signed) plus
     re-signed mask/person/split manifests bound to the new identity.
     Depth and face caches are pose-dependent and must be rebuilt by their
     own tools afterwards - this tool only prints the exact commands.

Nothing here touches the original manifests: the corrected set is a new
directory, and training opts in by pointing its config at it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO = Path(r"C:\Peter\cloudstudio-3dgs-gate1")
sys.path.insert(0, str(REPO))

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes

DATASETS = Path(r"C:\Peter\3dgs-datasets")
RECORDING = Path(r"C:\Peter\testdata\S1\house0305")
DEPTH_ROOT = DATASETS / "house0305_depth"
GATE_PX = 4.0
COARSE = np.arange(-24, 25, 4)
FINE = np.arange(-3, 4, 1)
SMOOTH_WINDOW = 5
MIN_VISIBLE = 300
EPSILON = 1e-3  # rad, for numerical Jacobians


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


def rotation_from_axis_angle(omega: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(omega)
    skew = np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])
    a = np.sinc(theta / np.pi)
    b = 0.5 * np.sinc(theta / (2 * np.pi)) ** 2
    return np.eye(3) + a * skew + b * (skew @ skew)


def world_correction(omega: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    rotation = rotation_from_axis_angle(omega)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pivot - rotation @ pivot
    return matrix


def resign(manifest: dict, key: str) -> dict:
    manifest = dict(manifest)
    manifest.pop(key, None)
    manifest[key] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


class FrameProjector:
    """Everything needed to score one image's pose against the LiDAR map."""

    def __init__(self, image: dict, camera: dict, xyz, gray):
        from PIL import Image as PILImage
        self.intr = camera["intrinsic"]
        self.dist = camera["distortion"]["params"]
        entry = DEPTH_INDEX[image["image_id"]]
        with np.load(DEPTH_ROOT / entry["path"]) as archive:
            self.height, self.width = (int(x) for x in archive["shape"])
            depth = np.zeros(self.height * self.width, dtype=np.float32)
            depth[archive["pixel_index"]] = archive["range_m"]
            self.depth = depth.reshape(self.height, self.width)
        with PILImage.open(RECORDING / image["path"]) as handle:
            self.photo = np.asarray(handle.convert("L")).astype(np.float32) / 255.0
        self.xyz, self.gray = xyz, gray

    def project(self, c2w: np.ndarray):
        w2c = np.linalg.inv(c2w)
        local = self.xyz @ w2c[:3, :3].T + w2c[:3, 3]
        forward = local[:, 2]
        rng = np.linalg.norm(local, axis=1)
        keep = (forward > 0.5) & (rng < 25.0)
        xy = local[keep, :2] / forward[keep, None]
        r = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(r, 1.0)
        t2 = theta * theta
        d = self.dist
        factor = theta * (1 + d["k1"] * t2 + d["k2"] * t2**2
                          + d["k3"] * t2**3 + d["k4"] * t2**4)
        s = np.where(r > 1e-9, factor / np.maximum(r, 1e-9), 1.0)
        u = self.intr["fl_x"] * xy[:, 0] * s + self.intr["cx"]
        v = self.intr["fl_y"] * xy[:, 1] * s + self.intr["cy"]
        return keep, u, v, np.degrees(theta), rng

    def visible_set(self, c2w: np.ndarray):
        keep, u, v, theta, rng = self.project(c2w)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        margin = 32
        ok = ((ui >= margin) & (ui < self.width - margin)
              & (vi >= margin) & (vi < self.height - margin) & (theta < 80))
        depth_at = self.depth[np.clip(vi, 0, self.height - 1),
                              np.clip(ui, 0, self.width - 1)]
        ok &= (depth_at > 0) & (np.abs(depth_at - rng[keep]) < 0.5)
        indices = np.flatnonzero(keep)[ok]
        return indices, ui[ok], vi[ok], self.gray[indices]

    def best_shift(self, c2w: np.ndarray):
        indices, ui, vi, gray = self.visible_set(c2w)
        if len(indices) < MIN_VISIBLE:
            return None
        def corr(dx, dy):
            return float(np.corrcoef(self.photo[vi + dy, ui + dx], gray)[0, 1])
        best = (-2.0, 0, 0)
        for dy in COARSE:
            for dx in COARSE:
                c = corr(dx, dy)
                if c > best[0]:
                    best = (c, dx, dy)
        centre = best
        for dy in FINE + best[2]:
            for dx in FINE + best[1]:
                if abs(dx) > 27 or abs(dy) > 27:
                    continue
                c = corr(int(dx), int(dy))
                if c > centre[0]:
                    centre = (c, int(dx), int(dy))
        corr0 = corr(0, 0)
        return {"corr0": corr0, "corr": centre[0],
                "dx": centre[1], "dy": centre[2],
                "n": int(len(indices))}

    def mean_projection(self, c2w: np.ndarray, indices: np.ndarray):
        keep, u, v, _, _ = self.project(c2w)
        table = np.full(len(self.xyz), np.nan)
        table_v = np.full(len(self.xyz), np.nan)
        table[np.flatnonzero(keep)] = u
        table_v[np.flatnonzero(keep)] = v
        return table[indices], table_v[indices]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-root", type=Path,
                        default=DATASETS / "house0305_manifest_posefix")
    parser.add_argument("--measure-only", action="store_true",
                        help="stop after solving and the gate audit; write "
                             "nothing")
    parser.add_argument("--las-stride", type=int, default=60)
    args = parser.parse_args()

    global DEPTH_INDEX
    depth_manifest = json.loads(
        (DEPTH_ROOT / "depth_manifest.json").read_text(encoding="utf-8"))
    DEPTH_INDEX = {d["image_id"]: d for d in depth_manifest["images"]}

    manifest = json.loads(
        (DATASETS / "house0305_manifest" / "dataset_manifest.json")
        .read_text(encoding="utf-8"))
    cameras = {c["camera_id"]: c for c in manifest["cameras"]}
    xyz, gray = las_sample(args.las_stride)
    print(f"LAS sample {len(xyz):,}", flush=True)

    rigs: dict[str, list[dict]] = {}
    for image in manifest["images"]:
        rigs.setdefault(str(image["rig_frame_id"]), []).append(image)
    rig_items = sorted(
        rigs.items(),
        key=lambda kv: min(int(i["timestamp_ns"]) for i in kv[1]))
    print(f"{len(rig_items)} rig frames", flush=True)

    solved: list[np.ndarray] = []
    quality: list[float] = []
    for number, (rig_id, members) in enumerate(rig_items):
        pivot = np.mean([np.asarray(m["c2w"], dtype=np.float64)[:3, 3]
                         for m in members], axis=0)
        targets, jacobians, weights = [], [], []
        confident = True
        for member in members:
            projector = FrameProjector(member, cameras[member["camera_id"]],
                                       xyz, gray)
            c2w = np.asarray(member["c2w"], dtype=np.float64)
            found = projector.best_shift(c2w)
            if found is None:
                confident = False
                break
            if found["corr"] - found["corr0"] < 0.005 and \
                    abs(found["dx"]) + abs(found["dy"]) <= 2:
                targets.extend([0.0, 0.0])
            else:
                targets.extend([float(found["dx"]), float(found["dy"])])
            indices, ui, vi, _ = projector.visible_set(c2w)
            base_u, base_v = ui.astype(np.float64), vi.astype(np.float64)
            jacobian = np.zeros((2, 3))
            for axis in range(3):
                omega = np.zeros(3)
                omega[axis] = EPSILON
                corrected = world_correction(omega, pivot) @ c2w
                u2, v2 = projector.mean_projection(corrected, indices)
                good = ~np.isnan(u2)
                if good.sum() < MIN_VISIBLE // 2:
                    confident = False
                    break
                jacobian[0, axis] = float(np.nanmedian(u2[good] - base_u[good])) / EPSILON
                jacobian[1, axis] = float(np.nanmedian(v2[good] - base_v[good])) / EPSILON
            jacobians.append(jacobian)
            weights.append(found["n"])
        if not confident or len(jacobians) != len(members):
            solved.append(np.zeros(3))
            quality.append(0.0)
            continue
        stacked = np.vstack(jacobians)
        target = np.asarray(targets)
        omega, *_ = np.linalg.lstsq(stacked, target, rcond=None)
        solved.append(omega)
        quality.append(1.0)
        if number % 40 == 0:
            print(f"  rig {number}/{len(rig_items)}: |target| "
                  f"{np.abs(target).max():.0f}px -> omega "
                  f"{np.degrees(np.linalg.norm(omega)):.2f}deg", flush=True)

    solved_arr = np.stack(solved)
    quality_arr = np.asarray(quality)
    print(f"confident rigs: {int(quality_arr.sum())}/{len(rig_items)}")

    # Median smoothing along the timeline; unconfident rigs inherit
    # neighbours instead of injecting zeros into the walk.
    smoothed = solved_arr.copy()
    half = SMOOTH_WINDOW // 2
    for index in range(len(rig_items)):
        window = []
        for j in range(index - half, index + half + 1):
            if 0 <= j < len(rig_items) and quality_arr[j] > 0:
                window.append(solved_arr[j])
        smoothed[index] = np.median(np.stack(window), axis=0) if window else 0.0
    degrees = np.degrees(np.linalg.norm(smoothed, axis=1))
    print(f"corrections deg: p50 {np.median(degrees):.2f} "
          f"p95 {np.percentile(degrees, 95):.2f} max {degrees.max():.2f}")

    # Gate audit: the standard decile instrument, corrected poses.
    corrections = {}
    for (rig_id, members), omega in zip(rig_items, smoothed):
        pivot = np.mean([np.asarray(m["c2w"], dtype=np.float64)[:3, 3]
                         for m in members], axis=0)
        corrections[rig_id] = world_correction(omega, pivot)

    images = sorted(manifest["images"], key=lambda i: int(i["timestamp_ns"]))
    t0 = int(images[0]["timestamp_ns"])
    span = int(images[-1]["timestamp_ns"]) - t0
    audit = []
    for index in range(0, len(images), 12):
        image = images[index]
        projector = FrameProjector(image, cameras[image["camera_id"]], xyz, gray)
        c2w = corrections[str(image["rig_frame_id"])] @ np.asarray(
            image["c2w"], dtype=np.float64)
        found = projector.best_shift(c2w)
        if found is None:
            continue
        audit.append(((int(image["timestamp_ns"]) - t0) / span,
                      float(np.hypot(found["dx"], found["dy"]))))
    print("\ncorrected-pose audit (median best-shift px per decile):")
    worst = 0.0
    for decile in range(10):
        data = [m for f, m in audit if decile / 10 <= f < (decile + 1) / 10]
        if data:
            median = float(np.median(data))
            worst = max(worst, median)
            print(f"  {decile*10:>3}-{decile*10+10:<3}%  {median:5.1f}")
    if worst > GATE_PX:
        print(f"\nGATE FAILED: worst decile {worst:.1f}px > {GATE_PX}px - "
              "writing nothing")
        return 1
    print(f"\nGATE PASSED: worst decile {worst:.1f}px <= {GATE_PX}px")
    if args.measure_only:
        return 0

    # Corrected manifest chain, re-signed, in a new directory.
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    corrected = json.loads(json.dumps(manifest))
    for image in corrected["images"]:
        c2w = corrections[str(image["rig_frame_id"])] @ np.asarray(
            image["c2w"], dtype=np.float64)
        image["c2w"] = [[float(x) for x in row] for row in c2w]
    corrected["pose_correction"] = {
        "algorithm_version": "closed_form_shift_solve_v1",
        "base_manifest_sha256": manifest["manifest_sha256"],
        "gate_worst_decile_px": worst,
        "correction_deg_p50": float(np.median(degrees)),
        "correction_deg_max": float(degrees.max()),
    }
    corrected = resign(corrected, "manifest_sha256")
    (out / "dataset_manifest.json").write_text(
        json.dumps(corrected, ensure_ascii=False, indent=1), encoding="utf-8")
    new_sha = corrected["manifest_sha256"]
    print(f"corrected dataset manifest -> {out} (sha {new_sha[:12]})")

    for source, name, key in (
            (DATASETS / "house0305_masks" / "mask_manifest.json",
             "mask_manifest.json", "mask_manifest_sha256"),
            (DATASETS / "house0305_person_masks" / "person_mask_manifest.json",
             "person_mask_manifest.json", "person_mask_manifest_sha256"),
            (DATASETS / "house0305_evaluation" / "split_manifest.json",
             "split_manifest.json", "split_manifest_sha256")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["dataset_manifest_sha256"] = new_sha
        payload = resign(payload, key)
        (out / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"re-signed {name}")

    print("\nnext, pose-dependent caches (commands, not run here):")
    print(f"  build_depth_cache --manifest {out / 'dataset_manifest.json'} ...")
    print("  delete face depth npz + rebuild face manifest against the new chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
