#!/usr/bin/env python3
"""Low-VRAM real-data smoke for Face4 LiDAR loss and guarded births."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes  # noqa: E402
from cloudstudio_3dgs.geometry.lidar_surface_field import (  # noqa: E402
    build_surface_field,
)
from cloudstudio_3dgs.training.default_strategy_adapter import (  # noqa: E402
    DefaultStrategyAdapter,
)
from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset  # noqa: E402
from cloudstudio_3dgs.training.losses import (  # noqa: E402
    confidence_weighted_log_range_huber,
)
from cloudstudio_3dgs.training.tangent_proposal import (  # noqa: E402
    ProposalConfig,
    TangentProposal,
)
from cloudstudio_3dgs.training.trainer import load_initialization_ply  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--face-root", required=True, type=Path)
    parser.add_argument("--renderer-mask-manifest", required=True, type=Path)
    parser.add_argument("--lidar-geometry-manifest", required=True, type=Path)
    parser.add_argument("--lidar-geometry-root", required=True, type=Path)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument("--tile-id", type=int, default=4)
    parser.add_argument("--surface-point-limit", type=int, default=5000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    tile_inputs = _read(args.tile_inputs)
    selected = [
        tile for tile in tile_inputs["tiles"] if int(tile["tile_id"]) == args.tile_id
    ]
    if len(selected) != 1:
        raise ValueError("tile_inputs does not contain the selected Tile exactly once")
    tile = selected[0]
    lidar_manifest = _read(args.lidar_geometry_manifest)
    valid_ids = {
        record["sample_id"]
        for record in lidar_manifest["records"]
        if int(record["valid_pixels"]) > 0
    }
    dataset = FaceCacheDataset(
        args.face_manifest,
        args.face_root,
        tile_views=tile["views"],
        renderer_mask_manifest_path=args.renderer_mask_manifest,
        face_lidar_geometry_manifest_path=args.lidar_geometry_manifest,
        face_lidar_geometry_root=args.lidar_geometry_root,
    )
    sample_index = next(
        index for index, sample_id in enumerate(dataset.image_ids) if sample_id in valid_ids
    )
    sample = dataset[sample_index]
    target = torch.from_numpy(np.asarray(sample.depth_range_m)).to(
        device=device, dtype=torch.float32
    )
    confidence = torch.from_numpy(np.asarray(sample.depth_confidence)).to(
        device=device, dtype=torch.float32
    )
    mask = torch.from_numpy(np.asarray(sample.depth_mask)).to(
        device=device, dtype=torch.bool
    )
    prediction = (target.detach() * 1.01).requires_grad_(True)
    range_loss = confidence_weighted_log_range_huber(
        prediction, target, confidence, mask, delta=0.05
    )
    range_loss.backward()
    valid_gradient = prediction.grad[mask]
    if (
        not bool(torch.isfinite(range_loss).item())
        or float(range_loss.detach()) <= 0.0
    ):
        raise RuntimeError("real Face4 LiDAR range loss is not finite positive")
    if not bool((valid_gradient.abs() > 0).any().item()):
        raise RuntimeError("real Face4 LiDAR range loss produced no gradient")

    initialization = args.tile_inputs_root / Path(tile["initialization"]["path"])
    xyz, _rgb = load_initialization_ply(initialization)
    point_count = min(int(args.surface_point_limit), len(xyz))
    xyz = np.ascontiguousarray(xyz[:point_count], dtype=np.float64)
    field = build_surface_field(xyz)
    proposal = TangentProposal(
        field,
        ProposalConfig(
            enabled=True,
            planarity_gate=0.6,
            support_gate=0.1,
            tangent_sigma_factor=0.5,
            normal_offset_factor=0.1,
            init_shortest_axis=True,
            reject_unsupported_births=True,
        ),
        seed=42,
    )
    adapter = DefaultStrategyAdapter(
        scene_scale=10.0,
        refine_start_iter=500,
        refine_stop_iter=2000,
        refine_every=100,
        reset_every=300,
        grow_grad2d=0.00015,
        prune_opa=0.1,
        split_scale_m=0.2,
        prune_scale_m=0.2,
        exact_mipmap_lifecycle=True,
        growth_min_opacity=0.15,
        prune_opa_late=0.05,
        prune_switch_step=1000,
        reset_opacity_cap=0.2,
        surface_birth_proposal=proposal,
    )
    means = torch.tensor(xyz, device=device, dtype=torch.float32)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(
                torch.full((point_count, 3), 0.01, device=device).log()
            ),
            "quats": torch.nn.Parameter(
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(
                    point_count, 1
                )
            ),
            "opacities": torch.nn.Parameter(
                torch.full((point_count,), 0.3, device=device).logit()
            ),
            "colors": torch.nn.Parameter(torch.zeros(point_count, 3, device=device)),
        }
    )
    optimizers = {
        name: torch.optim.Adam([parameter], lr=1e-3)
        for name, parameter in params.items()
    }
    state = {
        "grad2d": torch.full((point_count,), 0.001, device=device),
        "count": torch.ones(point_count, device=device),
        "radii": torch.zeros(point_count, device=device),
        "scene_scale": 10.0,
    }
    clone_count, split_count = adapter._grow_mipmap(params, optimizers, state)
    if clone_count <= 0 or split_count != 0:
        raise RuntimeError("guarded real-LiDAR clone smoke did not grow as expected")
    if proposal.last_stats.get("applied") != clone_count:
        raise RuntimeError("not every accepted real-LiDAR birth used the proposal")

    report = {
        "schema_version": 1,
        "kind": "lidar_first_face4_low_vram_gpu_smoke",
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "tile_id": int(args.tile_id),
        "real_face4_range": {
            "sample_id": sample.image_id,
            "crop_shape": [int(sample.height), int(sample.width)],
            "valid_lidar_pixels": int(mask.sum().item()),
            "log_huber_loss": float(range_loss.detach().cpu()),
            "nonzero_gradient_pixels": int(
                (valid_gradient.abs() > 0).sum().item()
            ),
        },
        "guarded_births": {
            "surface_point_count": point_count,
            "eligible_clone_count": clone_count,
            "split_count": split_count,
            "proposal": proposal.last_stats,
            "final_gaussian_count": len(params["means"]),
        },
        "peak_cuda_memory_mib": int(
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        ),
        "scope": (
            "component smoke only; does not authorize long training or prove "
            "full rasterized image quality"
        ),
    }
    report["gpu_smoke_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"LiDAR-first GPU smoke PASS: range_pixels={int(mask.sum().item())}, "
        f"births={clone_count}, peak_mib={report['peak_cuda_memory_mib']}, "
        f"sha256={report['gpu_smoke_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
