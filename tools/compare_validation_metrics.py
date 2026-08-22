"""Full-validation-set metric comparison across probe runs (offline, honest).

Computes masked PSNR/SSIM over all validation frames, LiDAR range MAE/RMSE on
COVERED supervised pixels, and reports the coverage fraction separately instead
of silently relaxing the PR-08 fail-closed gate.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from cloudstudio_3dgs.evaluation.image_metrics import masked_psnr, masked_ssim
from cloudstudio_3dgs.data.depth_cache import load_sparse_depth

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs-root", required=True, type=Path)
parser.add_argument("--runs", nargs="+", required=True)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()
PROBES = args.runs
ROOT = args.runs_root

results = {}
for probe in PROBES:
    eval_dir = ROOT / probe / "evaluation"
    ids = sorted({p.name.rsplit("_", 1)[0] for p in eval_dir.glob("*_rendered.png")})
    psnrs, ssims, maes, rmses, coverages = [], [], [], [], []
    for image_id in ids:
        ref = np.asarray(Image.open(eval_dir / f"{image_id}_reference.png"), dtype=np.float32) / 255.0
        ren = np.asarray(Image.open(eval_dir / f"{image_id}_rendered.png"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(eval_dir / f"{image_id}_mask.png")) > 127
        psnrs.append(float(masked_psnr(ren, ref, mask)))
        ssims.append(float(masked_ssim(ren, ref, mask)))
        rng_path = eval_dir / f"{image_id}_range.npy"
        npz_path = eval_dir / f"{image_id}_lidar.npz"
        if rng_path.exists() and npz_path.exists():
            rendered_range = np.load(rng_path)
            sparse = load_sparse_depth(npz_path)
            depth, confidence, valid = sparse.to_dense()
            supervised = valid & mask
            rendered_ok = np.isfinite(rendered_range) & (rendered_range > 0.0)
            covered = supervised & rendered_ok
            n_sup = int(supervised.sum())
            n_cov = int(covered.sum())
            coverages.append(n_cov / max(1, n_sup))
            if n_cov:
                err = np.abs(rendered_range[covered] - depth[covered])
                maes.append(float(err.mean()))
                rmses.append(float(np.sqrt((err**2).mean())))
    results[probe] = {
        "frames": len(ids),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_median": float(np.median(psnrs)),
        "psnr_p10": float(np.percentile(psnrs, 10)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_median": float(np.median(ssims)),
        "depth_mae_mean_m": float(np.mean(maes)) if maes else None,
        "depth_rmse_mean_m": float(np.mean(rmses)) if rmses else None,
        "depth_coverage_mean": float(np.mean(coverages)) if coverages else None,
    }

print(f"{'probe':<18} {'PSNR':>6} {'med':>6} {'p10':>6} {'SSIM':>6} {'dMAE':>6} {'dRMSE':>6} {'cover':>6}")
for probe, r in results.items():
    print(
        f"{probe:<18} {r['psnr_mean']:6.2f} {r['psnr_median']:6.2f} {r['psnr_p10']:6.2f} "
        f"{r['ssim_mean']:6.4f} "
        f"{(r['depth_mae_mean_m'] or 0):6.3f} {(r['depth_rmse_mean_m'] or 0):6.3f} "
        f"{(r['depth_coverage_mean'] or 0):6.3f}"
    )
(args.output or (ROOT / "validation_metrics_comparison.json")).write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("saved comparison json")
