"""Off-trajectory comparison: displace real cameras (lateral / up / yaw) and
render ours vs the aligned reference at the same novel pose. The reference
render is the pseudo ground truth (no photo exists there)."""
import sys, json, math, numpy as np
from pathlib import Path
sys.path.insert(0, "C:/Peter/cloudstudio-3dgs-work")
import torch
from PIL import Image
from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset
from tools.sharpness_metrics import _load_backend
from tools.build_three_way_compare import _load_ply_gaussians
from cloudstudio_3dgs.training.view_backgrounds import ViewBackgroundLibrary

cfg = Path(sys.argv[1]); ckpt = Path(sys.argv[2]); out = Path(sys.argv[3]); frames = int(sys.argv[4]) if len(sys.argv) > 4 else 6
raw = json.loads(cfg.read_text(encoding="utf-8"))
backend, _ = _load_backend(raw)
dataset = FaceCacheDataset(Path(raw["face_cache_manifest"]), Path(raw["face_cache_root"]), verify_artifacts=False,
                           dataset_manifest_path=Path(raw["dataset_manifest"]), renderer_mask_manifest_path=Path(raw["renderer_mask_manifest"]))
device = raw.get("device", "cuda:0")
backgrounds = ViewBackgroundLibrary(Path(raw["background_image_manifest"]), Path(raw["background_image_root"]), device=device) if raw.get("background_image_manifest") else None
td = lambda p: {k: (v.to(device) if hasattr(v, "to") else v) for k, v in p.items()}
ours = td(torch.load(ckpt, map_location="cpu", weights_only=False)["params"])
align = json.loads(Path("C:/Peter/3dgs-runs/probes/usa_gs_alignment.json").read_text())
ref = td(_load_ply_gaussians(Path("C:/baidunetdiskdownload/house/USAgs.ply"), np.asarray(align["transform"], dtype=np.float64)))

def displaced(c2w, kind):
    c2w = np.array(c2w, dtype=np.float64).copy()
    right, up = c2w[:3, 0], c2w[:3, 1]
    if kind == "lateral_1m": c2w[:3, 3] += 1.0 * right
    elif kind == "up_0.8m": c2w[:3, 3] += 0.8 * up
    elif kind == "yaw_25":
        a = math.radians(25.0); R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        c2w[:3, :3] = R @ c2w[:3, :3]
    return c2w

def render(params, sample, c2w, bg):
    with torch.no_grad():
        img, _, _, _ = backend.render(params, sample, with_range=False, background_rgb=bg, c2w_override=c2w)
    return (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

out.mkdir(parents=True, exist_ok=True)
stride = max(1, len(dataset) // frames); picks = list(range(0, len(dataset), stride))[:frames]
rows = []
for order, index in enumerate(picks):
    s = dataset[index]; photo = np.asarray(s.image, dtype=np.uint8)
    bg = backgrounds.background_for(s.image_id, height=photo.shape[0], width=photo.shape[1], torch=torch) if backgrounds else (1.0, 1.0, 1.0)
    for kind in ("lateral_1m", "up_0.8m", "yaw_25"):
        c2w = displaced(s.c2w, kind)
        a, b = render(ours, s, c2w, bg), render(ref, s, c2w, bg)
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)) / 255.0 ** 2
        psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
        strip = np.full((a.shape[0], a.shape[1] * 2 + 8, 3), 24, np.uint8); strip[:, :a.shape[1]] = a; strip[:, a.shape[1] + 8:] = b
        name = f"offtraj_{order:02d}_{kind}_{s.image_id[:12]}.png"; Image.fromarray(strip).save(out / name)
        rows.append({"file": name, "image_id": s.image_id, "kind": kind, "psnr_ours_vs_reference": round(psnr, 2)})
        print(name, "ours-vs-ref PSNR", round(psnr, 2))
(out / "offtraj_summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
