#!/usr/bin/env python3
"""Build and train a tiny raw-fisheye scene with the real gsplat CUDA backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from cloudstudio_3dgs.training.checkpoint import compare_checkpoint_payloads
from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.runtime_evidence import (
    execute_mcmc_native_kernel_smoke,
    execute_render_scale_contract_smoke,
    sign_full_mcmc_gate_evidence,
    verify_full_mcmc_gate_evidence,
)
from cloudstudio_3dgs.training.regularization import GeometryRegularizationConfig
from cloudstudio_3dgs.training.scale_calibration import MetricScaleCalibrationConfig
from cloudstudio_3dgs.training.trainer import (
    ControlledTrainingInterruption,
    TrainerConfig,
    load_initialization_ply,
    train,
)


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


class FullMCMCAcceptanceBackend(GsplatBackend):
    """Keep one unambiguously dead Gaussian until the first relocation window."""

    def initialize(self, *args, **kwargs):
        params, optimizers, state = super().initialize(*args, **kwargs)
        params["opacities"].data[0] = self.torch.tensor(
            1e-8, device=self.device
        ).logit()
        return params, optimizers, state


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
    parser.add_argument(
        "--full-mcmc",
        action="store_true",
        help=(
            "keep upstream MCMC noise injection enabled (never stop) and size the "
            "refine window to the short run so relocation/densification actually execute; "
            "requires a gsplat build with the 3DGS and RELOC kernel groups"
        ),
    )
    parser.add_argument(
        "--resume-equivalence",
        action="store_true",
        help=(
            "run an uninterrupted full-MCMC reference plus a controlled half-run/"
            "resume and compare all checkpointed optimizer, strategy, sampler, RNG, "
            "Gaussian and telemetry state"
        ),
    )
    args = parser.parse_args()
    if args.resume_equivalence and not args.full_mcmc:
        parser.error("--resume-equivalence requires --full-mcmc")
    if args.full_mcmc and args.steps < 40:
        parser.error("--full-mcmc requires at least 40 steps to enter a refine window")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"synthetic acceptance output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.full_mcmc:
        mcmc_config = {
            "refine_start_iter": max(10, args.steps // 4),
            "refine_stop_iter": args.steps,
            "refine_every": max(10, args.steps // 8),
        }
    else:
        mcmc_config = {"noise_injection_stop_iter": 0}
    backend = GsplatBackend(
        device="cuda:0",
        cap_max=64,
        lock_path=args.gsplat_lock,
        mcmc_config=mcmc_config,
    )
    native_kernel_smoke = (
        execute_mcmc_native_kernel_smoke() if args.full_mcmc else None
    )
    render_scale_contract = (
        execute_render_scale_contract_smoke(backend) if args.full_mcmc else None
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
    init_count = 8
    if args.full_mcmc:
        # MCMC growth is n_target = int(1.05 * N): with the 8-point fixture the
        # increment truncates to zero and densification can never run. Build a
        # deterministic jittered 24-point init (targets unchanged) so 5% growth
        # is at least one Gaussian per refine step.
        base_xyz, base_rgb = load_initialization_ply(init_path)
        rng = np.random.default_rng(42)
        dense_xyz = np.concatenate(
            [base_xyz + rng.normal(0.0, 0.02, size=base_xyz.shape).astype(np.float32) for _ in range(3)]
        )
        dense_rgb = np.concatenate([base_rgb] * 3)
        init_path = args.output / "fixture" / "initialization_full_mcmc.ply"
        write_binary_ply(init_path, dense_xyz.astype(np.float32), dense_rgb)
        init_count = len(dense_xyz)
    backend_factory = FullMCMCAcceptanceBackend if args.full_mcmc else GsplatBackend

    def make_config(run_dir: Path, resume_checkpoint: Path | None = None) -> TrainerConfig:
        return TrainerConfig(
            run_id="synthetic-fisheye-convergence",
            dataset_manifest=dataset_path,
            recording_root=recording,
            mask_manifest=mask_root / "mask_manifest.json",
            mask_root=mask_root,
            split_manifest=split_path,
            initialization_ply=init_path,
            output_dir=run_dir,
            gsplat_lock=args.gsplat_lock,
            resume_checkpoint=resume_checkpoint,
            depth_manifest=depth_manifest_path,
            depth_root=depth_root,
            require_person_masks=False,
            max_steps=args.steps,
            checkpoint_every=max(1, args.steps // 2),
            factor=1,
            cap_max=64,
            init_scale_m=0.16,
            metric_scale_calibration=MetricScaleCalibrationConfig(
                mode="fixed",
                means_step_fraction=None,
                noise_std_fraction=None,
            ),
            geometry_regularization=GeometryRegularizationConfig(enabled=False),
            rgb_l1_weight=1.0,
            rgb_ssim_weight=0.0,
            lidar_range_weight=0.01,
            lidar_range_loss_mode="linear_l1",
            **(
                {
                    "mcmc_refine_start_iter": mcmc_config["refine_start_iter"],
                    "mcmc_refine_stop_iter": mcmc_config["refine_stop_iter"],
                    "mcmc_refine_every": mcmc_config["refine_every"],
                }
                if args.full_mcmc
                else {"mcmc_noise_injection_stop_iter": 0}
            ),
            learning_rates={
                # Full-MCMC mode needs a realistic means LR: noise amplitude is
                # scaler = lr * noise_lr (upstream pairs 1.6e-4 with 5e5), and a
                # frozen-geometry LR of 1e-8 silently reduces noise to nothing.
                "means": 1.6e-4 if args.full_mcmc else 1e-8,
                "scales": 1e-8,
                "quats": 1e-8,
                "opacities": 1e-3,
                "colors": 5e-2,
            },
        )

    resume_report = None
    if args.resume_equivalence:
        continuous_dir = args.output / "run_continuous"
        manifest = train(
            make_config(continuous_dir), backend_factory=backend_factory
        )
        resumed_dir = args.output / "run_resumed"
        stop_step = args.steps // 2
        try:
            train(
                make_config(resumed_dir),
                backend_factory=backend_factory,
                controlled_stop_after_steps=stop_step,
            )
        except ControlledTrainingInterruption as interruption:
            if interruption.completed_steps != stop_step:
                raise RuntimeError("controlled interruption stopped at the wrong step")
            resume_checkpoint = interruption.checkpoint_path
        else:
            raise RuntimeError("controlled interruption did not occur")
        resumed_manifest = train(
            make_config(resumed_dir, resume_checkpoint),
            backend_factory=backend_factory,
        )
        # The 3DGUT rasterization backward accumulates gradients through CUDA
        # float atomics, so summation order differs between otherwise identical
        # runs and bit-exact state equality is unattainable in principle. On
        # machine B the sole divergence across the full resumable state was a
        # 3.7e-7 max-abs drift on the (lr 1e-8, effectively frozen) quats -
        # angle noise of ~1e-7 rad. Set atol one decade above that floor;
        # genuine resume bugs (wrong RNG stream, missing optimizer state)
        # produce mismatches many orders of magnitude larger.
        resume_report = compare_checkpoint_payloads(
            continuous_dir / "checkpoints" / "latest.pt",
            resumed_dir / "checkpoints" / "latest.pt",
            atol=5e-6,
        )
        resume_report["controlled_stop_step"] = stop_step
        resume_report["continuous_run_manifest_sha256"] = manifest[
            "run_manifest_sha256"
        ]
        resume_report["resumed_run_manifest_sha256"] = resumed_manifest[
            "run_manifest_sha256"
        ]
        manifest = resumed_manifest
    else:
        manifest = train(
            make_config(args.output / "run"), backend_factory=backend_factory
        )
    training = manifest["training"]
    improvement = float(training["loss_improvement_fraction"])
    acceptance = {
        "schema_version": 1,
        "steps": args.steps,
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "mcmc_mode": "full_noise_and_refine" if args.full_mcmc else "noise_disabled",
        "initial_loss": training["initial_loss"],
        "final_loss": training["last_metrics"]["loss"],
        "best_loss": training["best_loss"],
        "loss_improvement_fraction": improvement,
        "peak_vram_bytes": training["peak_vram_bytes"],
        "final_lidar_range_l1_m": training["last_metrics"]["lidar_range_l1_m"],
        "gaussian_count": training["gaussian_count"],
        "converged": improvement >= 0.20,
    }
    if args.full_mcmc:
        telemetry = training["mcmc_telemetry"]
        operator_report = training["mcmc_operator_report"]
        count_curve = [
            {"step": 0, **telemetry["initial_snapshot"]},
            *[
                {"step": int(event["step"]) + 1, **event["after"]}
                for event in telemetry["events"]
            ],
        ]
        if count_curve[-1]["step"] != args.steps:
            count_curve.append({"step": args.steps, **telemetry["last_snapshot"]})
        acceptance["initial_gaussian_count"] = init_count
        acceptance["native_kernel_smoke"] = native_kernel_smoke
        acceptance["render_scale_contract"] = render_scale_contract
        acceptance["mcmc_operator_registration"] = operator_report["status"]
        acceptance["mcmc_noise_step_count"] = telemetry[
            "noise_injection_step_count"
        ]
        acceptance["mcmc_noise_probe_step_count"] = telemetry[
            "noise_probe_step_count"
        ]
        acceptance["mcmc_noise_nonzero_step_count"] = telemetry[
            "noise_nonzero_step_count"
        ]
        acceptance["mcmc_noise_max_abs_delta_m"] = telemetry[
            "noise_max_abs_delta_m"
        ]
        acceptance["mcmc_refine_event_count"] = telemetry["refine_event_count"]
        acceptance["mcmc_relocated_count"] = telemetry["total_relocated"]
        acceptance["mcmc_added_count"] = telemetry["total_added"]
        acceptance["mcmc_final_state_finite"] = telemetry["last_snapshot"][
            "finite"
        ]
        acceptance["mcmc_densification_ran"] = (
            int(training["gaussian_count"]) > init_count
            and int(telemetry["total_added"]) > 0
        )
        acceptance["gaussian_count_curve"] = count_curve
        acceptance["mcmc_refine_events"] = telemetry["events"]
        acceptance["converged"] = bool(
            acceptance["converged"]
            and native_kernel_smoke is not None
            and native_kernel_smoke["status"] == "PASS"
            and render_scale_contract is not None
            and render_scale_contract["status"] == "PASS"
            and acceptance["mcmc_operator_registration"] == "PASS_REGISTERED"
            and acceptance["mcmc_noise_step_count"] == args.steps
            and acceptance["mcmc_noise_nonzero_step_count"] > 0
            and acceptance["mcmc_noise_max_abs_delta_m"] > 0.0
            and acceptance["mcmc_refine_event_count"] > 0
            and acceptance["mcmc_relocated_count"] > 0
            and acceptance["mcmc_densification_ran"]
            and acceptance["mcmc_final_state_finite"]
        )
    if resume_report is not None:
        acceptance["resume_equivalence"] = resume_report
        acceptance["converged"] = bool(
            acceptance["converged"] and resume_report["status"] == "PASS"
        )
    if args.full_mcmc:
        lock = json.loads(args.gsplat_lock.read_text(encoding="utf-8"))
        runtime = {
            key: backend.runtime[key]
            for key in (
                "package",
                "version",
                "locked_commit",
                "source_kind",
                "commit",
                "clean",
            )
            if key in backend.runtime
        }
        execution_gates = {
            "covariance_forward_backward": "PASS"
            if native_kernel_smoke is not None
            and native_kernel_smoke["status"] == "PASS"
            and native_kernel_smoke["covariance_forward_finite"]
            and native_kernel_smoke["covariance_backward_finite"]
            else "FAIL",
            "mcmc_noise_nonzero": "PASS"
            if acceptance["mcmc_noise_nonzero_step_count"] > 0
            and acceptance["mcmc_noise_max_abs_delta_m"] > 0.0
            else "FAIL",
            "relocation_occurred": "PASS"
            if acceptance["mcmc_relocated_count"] > 0
            else "FAIL",
            "sample_add_occurred": "PASS"
            if acceptance["mcmc_added_count"] > 0
            else "FAIL",
            "rasterization_forward_backward": "PASS"
            if training["status"] == "COMPLETE"
            and training["completed_steps"] == args.steps
            else "FAIL",
            "metric_scale_rasterization": "PASS"
            if render_scale_contract is not None
            and render_scale_contract["status"] == "PASS"
            else "FAIL",
            "interrupted_resume_equivalence": "PASS"
            if resume_report is not None and resume_report["status"] == "PASS"
            else "NOT_RUN" if resume_report is None else "FAIL",
        }
        gate_passed = acceptance["converged"] and all(
            status == "PASS" for status in execution_gates.values()
        )
        acceptance["gate_status"] = "PASS" if gate_passed else "FAIL"
        gate_evidence = sign_full_mcmc_gate_evidence(
            {
                "schema_version": 1,
                "evidence_type": "cloudstudio_full_mcmc_gate",
                "gate_status": acceptance["gate_status"],
                "environment": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": backend.torch.__version__,
                    "torch_cuda": backend.torch.version.cuda,
                    "cuda_available": backend.torch.cuda.is_available(),
                    "gpu": backend.torch.cuda.get_device_name(0),
                },
                "lock": {
                    key: lock.get(key)
                    for key in (
                        "package",
                        "version",
                        "commit",
                        "python",
                        "torch",
                        "cuda",
                        "cuda_arch_list",
                        "patch",
                        "source_policy",
                    )
                },
                "runtime": runtime,
                "execution_gates": execution_gates,
                "native_kernel_smoke": native_kernel_smoke,
                "render_scale_contract": render_scale_contract,
                "training": {
                    key: value
                    for key, value in acceptance.items()
                    if key not in {"schema_version", "gate_status"}
                },
            }
        )
        verification = verify_full_mcmc_gate_evidence(
            gate_evidence, expected_lock_commit=str(lock["commit"])
        )
        if gate_passed and verification["status"] != "PASS":
            raise RuntimeError(
                "generated full-MCMC gate evidence failed self-verification: "
                + "; ".join(verification["errors"])
            )
        gate_evidence_path = args.output / "full_mcmc_gate_evidence.json"
        gate_evidence_path.write_text(
            json.dumps(gate_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        acceptance["gate_evidence_sha256"] = gate_evidence[
            "gate_evidence_sha256"
        ]
        acceptance["gate_evidence_verification"] = verification
    else:
        acceptance["gate_status"] = (
            "PASS_COMPATIBILITY" if acceptance["converged"] else "FAIL"
        )
    acceptance_path = args.output / "synthetic_acceptance.json"
    acceptance_path.write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0 if acceptance["gate_status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
