"""How much pixel misalignment is LEFT after q2's accepted pose refinement?

The golden PSNR plateau says something still caps generalization. If the
refined poses re-run the occlusion-cleaned shift search at ~4px everywhere,
pose is fixed and the ceiling belongs to photometrics (PPISP's job) or
capacity (q4's job). If the bad deciles still read 10px+, the SE3-per-rig
parameterization is too stiff and a per-frame photometric pre-alignment
becomes the next escalation.

Delta application mirrors RigPoseRefiner.apply: world-left-multiplied
axis-angle with the rig-frame centre as pivot; delta order mirrors
FaceCacheDataset.rig_frame_ids (first-seen over the train face manifest).
"""
import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASETS = Path(r"C:\Peter\3dgs-datasets")
RECORDING = Path(r"C:\Peter\testdata\S1\house0305")
DEPTH_ROOT = DATASETS / "house0305_depth"
SEED = 42
EVERY = 12
SHIFTS = np.arange(-24, 25, 4)


def delta_matrix(delta: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    translation, omega = delta[:3], delta[3:]
    theta = np.linalg.norm(omega)
    skew = np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])
    a = np.sinc(theta / np.pi)
    b = 0.5 * np.sinc(theta / (2 * np.pi)) ** 2
    rotation = np.eye(3) + a * skew + b * (skew @ skew)
    column = translation + pivot - rotation @ pivot
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = column
    return matrix


def las_sample(path: Path, stride: int):
    with path.open("rb") as stream:
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
    return xyz * scale + shift, rgb.astype(np.float32) / 65535.0


# Deltas, in the refiner's rig order, plus per-rig pivots.
payload = torch.load(r"C:\Peter\3dgs-runs\probes\q2_full_pose\checkpoints\latest.pt",
                     map_location="cpu", weights_only=False)
deltas = payload["auxiliary_params"]["rig_pose_deltas"].detach().numpy().astype(np.float64)
del payload
face_manifest = json.loads(
    (DATASETS / "house0305_face_cache" / "face_manifest.json").read_text(encoding="utf-8"))
rig_order: list[str] = []
seen: set[str] = set()
centres_acc: dict[str, dict[str, np.ndarray]] = {}
for record in face_manifest["images"]:
    rig_id = str(record["rig_frame_id"])
    if rig_id not in seen:
        rig_order.append(rig_id)
        seen.add(rig_id)
    c2w = np.asarray(record["c2w"], dtype=np.float64)
    centres_acc.setdefault(rig_id, {})[str(record["image_id"])] = c2w[:3, 3]
assert len(rig_order) == len(deltas), (len(rig_order), len(deltas))
delta_by_rig = {rig_id: delta_matrix(deltas[i],
                                     np.mean(np.stack(list(centres_acc[rig_id].values())), axis=0))
                for i, rig_id in enumerate(rig_order)}
norms = np.linalg.norm(deltas[:, :3], axis=1)
print(f"deltas: {len(deltas)} rigs, translation p50 {np.median(norms)*100:.1f}cm "
      f"max {norms.max()*100:.1f}cm")

manifest = json.loads(
    (DATASETS / "house0305_manifest" / "dataset_manifest.json").read_text(encoding="utf-8"))
cameras = {c["camera_id"]: c for c in manifest["cameras"]}
depth_manifest = json.loads(
    (DEPTH_ROOT / "depth_manifest.json").read_text(encoding="utf-8"))
depth_by_id = {d["image_id"]: d for d in depth_manifest["images"]}

rng = np.random.default_rng(SEED)
xyz, rgb = las_sample(RECORDING / "colorized.las", stride=60)
las_gray = rgb.mean(axis=1)

images = sorted(manifest["images"], key=lambda i: int(i["timestamp_ns"]))
t0 = int(images[0]["timestamp_ns"])
span = int(images[-1]["timestamp_ns"]) - t0

rows = []
for index in range(0, len(images), EVERY):
    image = images[index]
    rig_id = str(image["rig_frame_id"])
    if rig_id not in delta_by_rig:
        continue  # val rig frame: never refined
    entry = depth_by_id.get(image["image_id"])
    if entry is None:
        continue
    camera = cameras[image["camera_id"]]
    intr, dist = camera["intrinsic"], camera["distortion"]["params"]
    for label, c2w in (("orig", np.array(image["c2w"], dtype=np.float64)),
                       ("refined", delta_by_rig[rig_id] @ np.array(image["c2w"],
                                                                   dtype=np.float64))):
        w2c = np.linalg.inv(c2w)
        local = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        forward = local[:, 2]
        rng_m = np.linalg.norm(local, axis=1)
        keep = (forward > 0.5) & (rng_m < 25.0)
        if keep.sum() < 500:
            continue
        xy = local[keep, :2] / forward[keep, None]
        r = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(r, 1.0)
        t2 = theta * theta
        factor = theta * (1 + dist["k1"] * t2 + dist["k2"] * t2**2
                          + dist["k3"] * t2**3 + dist["k4"] * t2**4)
        s = np.where(r > 1e-9, factor / np.maximum(r, 1e-9), 1.0)
        u = intr["fl_x"] * xy[:, 0] * s + intr["cx"]
        v = intr["fl_y"] * xy[:, 1] * s + intr["cy"]
        with np.load(DEPTH_ROOT / entry["path"]) as archive:
            height, width = (int(x) for x in archive["shape"])
            depth = np.zeros(height * width, dtype=np.float32)
            depth[archive["pixel_index"]] = archive["range_m"]
            depth = depth.reshape(height, width)
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        inside = ((ui >= 32) & (ui < width - 32) & (vi >= 32) & (vi < height - 32)
                  & (np.degrees(theta) < 80))
        ui, vi = ui[inside], vi[inside]
        ranges = rng_m[keep][inside]
        gray = las_gray[keep][inside]
        depth_at = depth[vi, ui]
        visible = (depth_at > 0) & (np.abs(depth_at - ranges) < 0.5)
        if visible.sum() < 300:
            continue
        ui, vi, gray = ui[visible], vi[visible], gray[visible]
        with Image.open(RECORDING / image["path"]) as handle:
            photo = np.asarray(handle.convert("L")).astype(np.float32) / 255.0
        best = (-2.0, 0, 0)
        for dy in SHIFTS:
            for dx in SHIFTS:
                c = float(np.corrcoef(photo[vi + dy, ui + dx], gray)[0, 1])
                if c > best[0]:
                    best = (c, dx, dy)
        fraction = (int(image["timestamp_ns"]) - t0) / span
        rows.append((fraction, label, float(np.hypot(best[1], best[2]))))

rows_np = {(label): [(f, m) for f, l, m in rows if l == label]
           for label in ("orig", "refined")}
print(f"\nmedian best-shift px by capture-time decile (orig -> refined):")
for decile in range(10):
    parts = []
    for label in ("orig", "refined"):
        data = [m for f, m in rows_np[label]
                if decile / 10 <= f < (decile + 1) / 10]
        parts.append(f"{np.median(data):5.1f}" if data else "  n/a")
    print(f"  {decile*10:>3}-{decile*10+10:<3}%  {parts[0]} -> {parts[1]}")
