#!/usr/bin/env python3
"""Build and train a tiny raw-fisheye scene with the real gsplat CUDA backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import build_per_image_masks
from cloudstudio_3dgs.data.point_cloud import write_binary_ply
from cloudstudio_3dgs.data.depth_cache import sparse_depth_npz_bytes
from cloudstudio_3dgs.evaluation.splits import SplitConfig, build_split_manifest, write_split_manifest
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap
from cloudstudio_3dgs.training.backend import GsplatBackend
from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.trainer import TrainerConfig, train


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera(side: str) -> dict:
    return {
        "camera_id": side,
        "side": side,
        "camera_type": "fisheye",
        "width": 32,
        "height": 32,
        "intrinsic": {"fl_x": 22.0, "fl_y": 22.0, "cx": 15.5, "cy": 15.5},
        "distortion": {
            "camera_model": "OPENCV_FISHEYE",
            "params": {"k1": 0.02, "k2": -0.003, "k3": 0.0002, "k4": 0.0},
        },
    }


def _pose(x: float, y: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [x, y, 0.0]
    return pose


def build_fixture(
    root: Path, backend: GsplatBackend
) -> tuple[dict, Path, Path, Path, Path, Path, Path, Path]:
    torch = backend.torch
    recording = root / "recording"
    cameras = [_camera("left"), _camera("right")]
    camera_by_id = {camera["camera_id"]: camera for camera in cameras}
    xyz = np.asarray(
        [
            [-0.35, -0.25, 2.0], [0.0, -0.25, 2.2], [0.35, -0.25, 2.0],
            [-0.35, 0.15, 2.3], [0.0, 0.15, 1.9], [0.35, 0.15, 2.3],
            [-0.18, 0.42, 2.1], [0.18, 0.42, 2.1],
        ],
        dtype=np.float32,
    )
    target_rgb = np.asarray(
        [
            [230, 40, 30], [30, 220, 50], [40, 70, 235], [240, 180, 30],
            [210, 30, 210], [20, 210, 210], [245, 120, 40], [130, 80, 245],
        ],
        dtype=np.uint8,
    )
    target_params, _, _ = backend.initialize(
        xyz,
        target_rgb,
        init_scale_m=0.16,
        learning_rates={name: 1e-4 for name in ("means", "scales", "quats", "opacities", "colors")},
    )
    target_params["opacities"].data.fill_(torch.tensor(0.85, device=backend.device).logit())

    images = []
    rig_frames = []
    sparse_depths: dict[str, SparseDepthMap] = {}
    with torch.no_grad():
        for frame_index, y in enumerate((0.0, 0.12)):
            image_ids = []
            for side, x in (("left", -0.08), ("right", 0.08)):
                image_id = f"synthetic_{side}_{frame_index:03d}"
                image_ids.append(image_id)
                c2w = _pose(x, y)
                camera = camera_by_id[side]
                intrinsic = camera["intrinsic"]
                params = camera["distortion"]["params"]
                sample = TrainingSample(
                    image_id=image_id,
                    rig_frame_id=f"rig_{frame_index:03d}",
                    camera_id=side,
                    image=np.zeros((32, 32, 3), dtype=np.uint8),
                    rgb_mask=np.ones((32, 32), dtype=bool),
                    depth_range_m=None,
                    depth_confidence=None,
                    depth_mask=None,
                    depth_cache_path=None,
                    c2w=c2w,
                    K=np.asarray(
                        [[intrinsic["fl_x"], 0.0, intrinsic["cx"]], [0.0, intrinsic["fl_y"], intrinsic["cy"]], [0.0, 0.0, 1.0]],
                        dtype=np.float32,
                    ),
                    radial_coeffs=np.asarray([params[f"k{i}"] for i in range(1, 5)], dtype=np.float32),
                    width=32,
                    height=32,
                )
                render, ray_range, alpha, _ = backend.render(
                    target_params, sample, with_range=True
                )
                valid = (
                    torch.isfinite(ray_range)
                    & (ray_range > 0.0)
                    & (alpha > 1e-4)
                ).flatten()
                valid_indexes = (
                    torch.nonzero(valid, as_tuple=False)
                    .flatten()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
                if not len(valid_indexes):
                    raise RuntimeError(f"synthetic view {image_id} has no valid ray range")
                selected = valid_indexes[:: max(1, len(valid_indexes) // 32)][:32]
                selected_ranges = ray_range.flatten()[
                    torch.as_tensor(selected, device=backend.device)
                ].cpu().numpy().astype(np.float32)
                sparse_depths[image_id] = SparseDepthMap(
                    (32, 32),
                    selected,
                    selected_ranges,
                    np.ones(len(selected), dtype=np.float32),
                    np.arange(len(selected), dtype=np.int64),
                    np.ones(len(selected), dtype=np.int32),
                )
                pixels = render.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
                relative = Path("camera") / side / f"{frame_index:03d}.png"
                path = recording / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path, format="PNG", optimize=False)
                images.append(
                    {
                        "image_id": image_id,
                        "rig_frame_id": f"rig_{frame_index:03d}",
                        "camera_id": side,
                        "side": side,
                        "timestamp_ns": 1_000_000_000 + frame_index,
                        "path_root": "recording",
                        "path": relative.as_posix(),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                        "pose_convention": "c2w_opencv",
                        "pose_source": "synthetic",
                        "c2w": c2w.tolist(),
                    }
                )
            rig_frames.append(
                {
                    "rig_frame_id": f"rig_{frame_index:03d}",
                    "timestamp_ns": 1_000_000_000 + frame_index,
                    "left_image_id": image_ids[0],
                    "right_image_id": image_ids[1],
                    "image_ids": image_ids,
                    "timestamp_delta_ns": 0,
                }
            )
    dataset = {
        "schema_version": 1,
        "coordinate_frame": "s1_local",
        "cameras": cameras,
        "images": images,
        "rig_frames": rig_frames,
    }
    dataset["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(dataset)).hexdigest()
    dataset_path = root / "dataset_manifest.json"
    dataset_path.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mask_root = root / "masks"
    masks = build_per_image_masks(dataset, mask_root)
    mask_by_id = {item["image_id"]: item for item in masks["images"]}
    depth_root = root / "depth-cache"
    depth_records = []
    for image in images:
        image_id = image["image_id"]
        payload = sparse_depth_npz_bytes(sparse_depths[image_id])
        relative = f"depth/{image_id}.npz"
        path = depth_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        depth_records.append(
            {
                "image_id": image_id,
                "camera_id": image["camera_id"],
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": [32, 32],
                "valid_pixels": len(sparse_depths[image_id].pixel_index),
                "combined_mask_sha256": mask_by_id[image_id]["combined_mask_sha256"],
            }
        )
    depth_manifest = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "mask_manifest_sha256": masks["mask_manifest_sha256"],
        "coordinate_frame": "s1_local",
        "depth_semantics": "euclidean_ray_range_m",
        "complete_dataset": True,
        "images": depth_records,
    }
    depth_manifest["depth_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(depth_manifest)
    ).hexdigest()
    depth_manifest_path = depth_root / "depth_manifest.json"
    depth_manifest_path.write_text(
        json.dumps(depth_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split = build_split_manifest(
        dataset,
        SplitConfig(mode="manual", golden_rig_frames=1),
        manual={"rig_000": "train", "rig_001": "val"},
    )
    split_path = root / "split_manifest.json"
    write_split_manifest(split_path, split)
    init_path = root / "initialization.ply"
    write_binary_ply(init_path, xyz, 255 - target_rgb)
    return (
        dataset,
        dataset_path,
        recording,
        mask_root,
        split_path,
        init_path,
        depth_manifest_path,
        depth_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gsplat-lock",
        type=Path,
        default=ROOT / "upstream" / "cloudstudio_trainer.lock.json",
    )
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"synthetic acceptance output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    backend = GsplatBackend(
        device="cuda:0",
        cap_max=64,
        lock_path=args.gsplat_lock,
        mcmc_config={"noise_injection_stop_iter": 0},
    )
    (
        _,
        dataset_path,
        recording,
        mask_root,
        split_path,
        init_path,
        depth_manifest_path,
        depth_root,
    ) = build_fixture(args.output / "fixture", backend)
    run_dir = args.output / "run"
    manifest = train(
        TrainerConfig(
            run_id="synthetic-fisheye-convergence",
            dataset_manifest=dataset_path,
            recording_root=recording,
            mask_manifest=mask_root / "mask_manifest.json",
            mask_root=mask_root,
            split_manifest=split_path,
            initialization_ply=init_path,
            output_dir=run_dir,
            gsplat_lock=args.gsplat_lock,
            depth_manifest=depth_manifest_path,
            depth_root=depth_root,
            max_steps=args.steps,
            checkpoint_every=max(1, args.steps // 2),
            factor=1,
            cap_max=64,
            init_scale_m=0.16,
            rgb_l1_weight=1.0,
            rgb_ssim_weight=0.0,
            lidar_range_weight=0.01,
            mcmc_noise_injection_stop_iter=0,
            learning_rates={
                "means": 1e-8,
                "scales": 1e-8,
                "quats": 1e-8,
                "opacities": 1e-3,
                "colors": 5e-2,
            },
        )
    )
    training = manifest["training"]
    improvement = float(training["loss_improvement_fraction"])
    acceptance = {
        "schema_version": 1,
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "initial_loss": training["initial_loss"],
        "final_loss": training["last_metrics"]["loss"],
        "best_loss": training["best_loss"],
        "loss_improvement_fraction": improvement,
        "peak_vram_bytes": training["peak_vram_bytes"],
        "final_lidar_range_l1_m": training["last_metrics"]["lidar_range_l1_m"],
        "converged": improvement >= 0.20,
    }
    acceptance_path = args.output / "synthetic_acceptance.json"
    acceptance_path.write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0 if acceptance["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
