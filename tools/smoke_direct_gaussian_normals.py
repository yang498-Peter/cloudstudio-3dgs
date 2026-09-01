"""CUDA smoke for direct shortest-axis Gaussian-normal raster gradients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.backend import GsplatBackend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch

    backend = GsplatBackend(
        device=args.device,
        cap_max=100,
        lock_path=ROOT / "upstream" / "gsplat.lock.json",
        densification_strategy="default_3dgs",
        default_strategy_config={"refine_start_iter": 500, "refine_stop_iter": 501},
    )
    # The installed runtime can be a 3DGUT-only locked wheel. Use its pinhole
    # compatibility path for this operator-gradient smoke instead of mutating
    # the external gsplat binary to swap in the EWA-only build.
    backend.pinhole_with_ut = True
    backend.pinhole_rasterize_mode = "classic"
    xyz = np.asarray(
        [[-0.08, -0.05, 2.0], [0.08, -0.05, 2.0], [-0.08, 0.06, 2.0], [0.08, 0.06, 2.0]],
        dtype=np.float32,
    )
    rgb = np.full((4, 3), 128, dtype=np.uint8)
    scales = np.asarray(
        [[0.025, 0.08, 0.06], [0.03, 0.09, 0.05], [0.025, 0.07, 0.09], [0.035, 0.08, 0.06]],
        dtype=np.float32,
    )
    quats = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.98, 0.0, 0.20, 0.0], [0.98, 0.20, 0.0, 0.0], [0.96, 0.0, 0.0, 0.28]],
        dtype=np.float32,
    )
    params, _optimizers, _state = backend.initialize(
        xyz,
        rgb,
        init_scales_m=scales,
        init_quaternions=quats,
        learning_rates={
            "means": 1e-5,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "colors": 2.5e-3,
        },
        color_model="rgb_sigmoid",
        sh_degree=0,
    )
    sample = SimpleNamespace(
        c2w=np.eye(4, dtype=np.float32),
        K=np.asarray([[55.0, 0.0, 31.5], [0.0, 55.0, 31.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        width=64,
        height=64,
        camera_model="pinhole",
        radial_coeffs=np.zeros(4, dtype=np.float32),
    )
    normal, valid = backend.render_gaussian_normals(params, sample)
    yy, xx = torch.meshgrid(
        torch.arange(64, device=args.device),
        torch.arange(64, device=args.device),
        indexing="ij",
    )
    target = torch.stack(
        ((xx.float() - 31.5) / 64.0, torch.zeros_like(xx, dtype=torch.float32), torch.ones_like(xx, dtype=torch.float32)),
        dim=-1,
    )
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    if not bool(valid.any()):
        raise RuntimeError("direct-normal raster produced no valid pixels")
    loss = torch.abs(normal[valid] - target[valid]).mean()
    loss.backward()
    report = {
        "status": "PASS",
        "valid_pixels": int(valid.sum().item()),
        "loss": float(loss.detach().cpu()),
        "quaternion_gradient_norm": float(params["quats"].grad.norm().detach().cpu()),
        "scale_gradient_norm": float(params["scales"].grad.norm().detach().cpu()),
        "shortest_scale_gradient_abs_max": float(
            params["scales"].grad.gather(
                1, torch.argmin(params["scales"].detach(), dim=1)[:, None]
            ).abs().max().detach().cpu()
        ),
    }
    if report["quaternion_gradient_norm"] <= 0.0:
        raise RuntimeError("direct-normal loss produced no quaternion gradient")
    if report["shortest_scale_gradient_abs_max"] <= 0.0:
        raise RuntimeError("direct-normal loss produced no shortest-scale gradient")
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
