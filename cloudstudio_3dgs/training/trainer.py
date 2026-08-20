"""CloudStudio-owned raw-fisheye gsplat trainer with no Viewer dependency."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.image_sample import CropWindow
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.quality_report import sign_run_manifest
from cloudstudio_3dgs.training.backend import GsplatBackend
from cloudstudio_3dgs.training.checkpoint import load_checkpoint, save_checkpoint
from cloudstudio_3dgs.training.contracts import build_coordinate_transform_manifest
from cloudstudio_3dgs.training.dataset import S1TrainingDataset, TrainingSample
from cloudstudio_3dgs.training.losses import (
    confidence_weighted_range_l1,
    masked_rgb_l1,
    masked_rgb_ssim_loss,
)


@dataclass(frozen=True)
class TrainerConfig:
    run_id: str
    dataset_manifest: Path
    recording_root: Path
    mask_manifest: Path
    mask_root: Path
    split_manifest: Path
    initialization_ply: Path
    output_dir: Path
    gsplat_lock: Path
    depth_manifest: Path | None = None
    depth_root: Path | None = None
    resume_checkpoint: Path | None = None
    device: str = "cuda:0"
    seed: int = 42
    max_steps: int = 3_000
    checkpoint_every: int = 500
    factor: int = 4
    crop: CropWindow | None = None
    cap_max: int = 1_000_000
    init_scale_m: float = 0.05
    rgb_l1_weight: float = 0.8
    rgb_ssim_weight: float = 0.2
    lidar_range_weight: float = 0.05
    mcmc_refine_start_iter: int = 500
    mcmc_refine_stop_iter: int = 25_000
    mcmc_refine_every: int = 100
    mcmc_noise_injection_stop_iter: int = -1
    mcmc_noise_lr: float = 500_000.0
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {
            "means": 1.6e-4,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "colors": 2.5e-3,
        }
    )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainerConfig":
        paths = {
            key: None if value.get(key) is None else Path(value[key])
            for key in (
                "dataset_manifest",
                "recording_root",
                "mask_manifest",
                "mask_root",
                "split_manifest",
                "initialization_ply",
                "output_dir",
                "gsplat_lock",
                "depth_manifest",
                "depth_root",
                "resume_checkpoint",
            )
        }
        crop_value = value.get("crop")
        crop = None if crop_value is None else CropWindow(**crop_value)
        options = {
            key: value[key]
            for key in (
                "device",
                "seed",
                "max_steps",
                "checkpoint_every",
                "factor",
                "cap_max",
                "init_scale_m",
                "rgb_l1_weight",
                "rgb_ssim_weight",
                "lidar_range_weight",
                "mcmc_refine_start_iter",
                "mcmc_refine_stop_iter",
                "mcmc_refine_every",
                "mcmc_noise_injection_stop_iter",
                "mcmc_noise_lr",
                "learning_rates",
            )
            if key in value
        }
        return cls(run_id=str(value["run_id"]), crop=crop, **paths, **options)

    def validate(self) -> None:
        if not self.run_id or any(character in self.run_id for character in "\\/\0"):
            raise ValueError("run_id must be a non-empty path-safe name")
        if not self.device.startswith("cuda"):
            raise ValueError("3DGUT training requires an explicit CUDA device")
        if self.max_steps <= 0 or self.checkpoint_every <= 0:
            raise ValueError("max_steps and checkpoint_every must be positive")
        if self.cap_max <= 4:
            raise ValueError("cap_max must be greater than four")
        if self.init_scale_m <= 0.0:
            raise ValueError("init_scale_m must be positive")
        weights = (self.rgb_l1_weight, self.rgb_ssim_weight, self.lidar_range_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if self.rgb_l1_weight + self.rgb_ssim_weight <= 0.0:
            raise ValueError("at least one RGB loss weight must be positive")
        expected_lrs = {"means", "scales", "quats", "opacities", "colors"}
        if set(self.learning_rates) != expected_lrs:
            raise ValueError(f"learning_rates must contain exactly {sorted(expected_lrs)}")
        if any(float(value) <= 0.0 for value in self.learning_rates.values()):
            raise ValueError("all learning rates must be positive")
        if self.mcmc_refine_start_iter < 0 or self.mcmc_refine_stop_iter <= 0:
            raise ValueError("MCMC refine bounds must be non-negative/positive")
        if self.mcmc_refine_start_iter >= self.mcmc_refine_stop_iter:
            raise ValueError("MCMC refine start must be smaller than refine stop")
        if self.mcmc_refine_every <= 0 or self.mcmc_noise_lr < 0.0:
            raise ValueError("MCMC refine interval must be positive and noise LR non-negative")
        if self.mcmc_noise_injection_stop_iter < -1:
            raise ValueError("MCMC noise stop must be -1 or non-negative")
        if (self.depth_manifest is None) != (self.depth_root is None):
            raise ValueError("depth_manifest and depth_root must be provided together")
        if self.lidar_range_weight > 0.0 and self.depth_manifest is None:
            raise ValueError(
                "positive lidar_range_weight requires depth_manifest and depth_root"
            )
        required_paths = {
            "dataset_manifest": self.dataset_manifest,
            "recording_root": self.recording_root,
            "mask_manifest": self.mask_manifest,
            "mask_root": self.mask_root,
            "split_manifest": self.split_manifest,
            "initialization_ply": self.initialization_ply,
            "output_dir": self.output_dir,
            "gsplat_lock": self.gsplat_lock,
        }
        missing = [name for name, path in required_paths.items() if path is None]
        if missing:
            raise ValueError(f"trainer config is missing paths: {', '.join(sorted(missing))}")

    def contract_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm_version": "cloudstudio_gsplat_trainer_v1",
            "seed": self.seed,
            "max_steps": self.max_steps,
            "factor": self.factor,
            "crop": None
            if self.crop is None
            else {
                "x": self.crop.x,
                "y": self.crop.y,
                "width": self.crop.width,
                "height": self.crop.height,
            },
            "cap_max": self.cap_max,
            "init_scale_m": self.init_scale_m,
            "loss_weights": {
                "rgb_l1": self.rgb_l1_weight,
                "rgb_ssim": self.rgb_ssim_weight,
                "lidar_range": self.lidar_range_weight,
            },
            "learning_rates": dict(sorted(self.learning_rates.items())),
            "renderer": {
                "camera_model": "fisheye",
                "projection": "3DGUT",
                "with_ut": True,
                "with_eval3d": True,
                "range_mode": "RGB-Ed",
                "range_semantics": "euclidean_ray_range_m",
                "global_z_order": False,
                "packed": False,
            },
            "strategy": {
                "name": "MCMC",
                "refine_start_iter": self.mcmc_refine_start_iter,
                "refine_stop_iter": self.mcmc_refine_stop_iter,
                "refine_every": self.mcmc_refine_every,
                "noise_injection_stop_iter": self.mcmc_noise_injection_stop_iter,
                "noise_lr": self.mcmc_noise_lr,
            },
            "viewer": False,
        }


def load_initialization_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the canonical binary little-endian XYZ/RGB PLY from PR-04."""
    path = Path(path)
    with path.open("rb") as stream:
        header: list[str] = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            line = raw.decode("ascii").rstrip("\r\n")
            header.append(line)
            if line == "end_header":
                break
        if header[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("initialization PLY must be binary_little_endian")
        vertex_lines = [line for line in header if line.startswith("element vertex ")]
        if len(vertex_lines) != 1:
            raise ValueError("initialization PLY must contain one vertex element")
        count = int(vertex_lines[0].split()[2])
        properties = [line for line in header if line.startswith("property ")]
        if properties != [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]:
            raise ValueError("initialization PLY properties do not match canonical XYZ/RGB")
        dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)], align=False)
        records = np.fromfile(stream, dtype=dtype, count=count)
        if len(records) != count or stream.read(1):
            raise ValueError("initialization PLY payload size does not match its header")
    xyz = np.asarray(records["xyz"], dtype=np.float32).copy()
    rgb = np.asarray(records["rgb"], dtype=np.uint8).copy()
    if not len(xyz) or not np.all(np.isfinite(xyz)):
        raise ValueError("initialization PLY contains no finite points")
    if float(np.max(np.abs(xyz))) > 100_000.0:
        raise ValueError("initialization PLY is not in a safe local coordinate frame")
    return xyz, rgb


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sample(sample: TrainingSample, torch: Any, device: str) -> dict[str, Any]:
    result = {
        "rgb": torch.as_tensor(np.array(sample.image, copy=True), dtype=torch.float32, device=device) / 255.0,
        "rgb_mask": torch.as_tensor(np.array(sample.rgb_mask, copy=True), dtype=torch.bool, device=device),
    }
    if sample.depth_range_m is not None:
        result.update(
            {
                "range_m": torch.as_tensor(np.array(sample.depth_range_m, copy=True), dtype=torch.float32, device=device),
                "confidence": torch.as_tensor(np.array(sample.depth_confidence, copy=True), dtype=torch.float32, device=device),
                "depth_mask": torch.as_tensor(np.array(sample.depth_mask, copy=True), dtype=torch.bool, device=device),
            }
        )
    return result


def _save_evaluation_artifacts(
    *,
    backend: GsplatBackend,
    params: Any,
    dataset: S1TrainingDataset,
    output_dir: Path,
) -> list[dict[str, Any]]:
    torch = backend.torch
    frames: list[dict[str, Any]] = []
    artifact_root = output_dir / "evaluation"
    artifact_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            has_range = sample.depth_range_m is not None
            rendered, rendered_range, _, _ = backend.render(params, sample, with_range=has_range)
            prefix = artifact_root / sample.image_id
            reference_path = prefix.with_name(f"{sample.image_id}_reference.png")
            render_path = prefix.with_name(f"{sample.image_id}_rendered.png")
            mask_path = prefix.with_name(f"{sample.image_id}_mask.png")
            Image.fromarray(sample.image).save(reference_path, format="PNG", optimize=False)
            rendered_u8 = (
                rendered.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
            )
            Image.fromarray(rendered_u8).save(render_path, format="PNG", optimize=False)
            Image.fromarray(sample.rgb_mask.astype(np.uint8) * 255).save(
                mask_path, format="PNG", optimize=False
            )
            frame: dict[str, Any] = {
                "image_id": sample.image_id,
                "split": "val",
                "reference_rgb_path": reference_path.relative_to(output_dir).as_posix(),
                "rendered_rgb_path": render_path.relative_to(output_dir).as_posix(),
                "combined_mask_path": mask_path.relative_to(output_dir).as_posix(),
            }
            if has_range:
                assert rendered_range is not None and sample.depth_cache_path is not None
                rendered_depth_path = prefix.with_name(f"{sample.image_id}_range.npy")
                lidar_path = prefix.with_name(f"{sample.image_id}_lidar.npz")
                np.save(rendered_depth_path, rendered_range.detach().cpu().numpy().astype(np.float32))
                shutil.copyfile(sample.depth_cache_path, lidar_path)
                frame.update(
                    {
                        "rendered_depth_path": rendered_depth_path.relative_to(output_dir).as_posix(),
                        "rendered_depth_semantics": "euclidean_ray_range_m",
                        "lidar_depth_cache_path": lidar_path.relative_to(output_dir).as_posix(),
                    }
                )
            frames.append(frame)
    return frames


def train(config: TrainerConfig, *, backend_factory: Any = GsplatBackend) -> dict[str, Any]:
    """Train and write a signed run manifest. No Viewer is created or imported."""
    config.validate()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 3DGUT gsplat trainer")
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and config.resume_checkpoint is None:
        raise FileExistsError(f"training output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainset = S1TrainingDataset(
        dataset_manifest_path=config.dataset_manifest,
        recording_root=config.recording_root,
        mask_manifest_path=config.mask_manifest,
        mask_root=config.mask_root,
        split_manifest_path=config.split_manifest,
        split="train",
        depth_manifest_path=config.depth_manifest,
        depth_root=config.depth_root,
        factor=config.factor,
        crop=config.crop,
    )
    valset = S1TrainingDataset(
        dataset_manifest_path=config.dataset_manifest,
        recording_root=config.recording_root,
        mask_manifest_path=config.mask_manifest,
        mask_root=config.mask_root,
        split_manifest_path=config.split_manifest,
        split="val",
        depth_manifest_path=config.depth_manifest,
        depth_root=config.depth_root,
        factor=config.factor,
        crop=config.crop,
    )
    if trainset.dataset_sha256 != valset.dataset_sha256:
        raise ValueError("train and validation datasets have different identities")
    coordinate = build_coordinate_transform_manifest(trainset.dataset_sha256)
    _atomic_json(output_dir / "coordinate_transform_manifest.json", coordinate)
    contract = config.contract_dict()
    config_sha256 = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    backend = backend_factory(
        device=config.device,
        cap_max=config.cap_max,
        lock_path=config.gsplat_lock,
        mcmc_config={
            "refine_start_iter": config.mcmc_refine_start_iter,
            "refine_stop_iter": config.mcmc_refine_stop_iter,
            "refine_every": config.mcmc_refine_every,
            "noise_injection_stop_iter": config.mcmc_noise_injection_stop_iter,
            "noise_lr": config.mcmc_noise_lr,
        },
    )
    initialization_sha256 = _sha256_file(config.initialization_ply)
    runtime_contract = {
        key: backend.runtime.get(key)
        for key in ("package", "version", "locked_commit", "source_kind", "commit", "wheel_sha256")
        if backend.runtime.get(key) is not None
    }
    checkpoint_identity = {
        **trainset.identity,
        "coordinate_transform_sha256": coordinate["coordinate_transform_sha256"],
        "trainer_config_sha256": config_sha256,
        "initialization_ply_sha256": initialization_sha256,
        "gsplat_runtime": runtime_contract,
    }
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(config.seed)
    xyz, rgb = load_initialization_ply(config.initialization_ply)
    if len(xyz) >= config.cap_max:
        raise ValueError(
            f"initialization has {len(xyz)} Gaussians but cap_max is {config.cap_max}"
        )
    params, optimizers, strategy_state = backend.initialize(
        xyz,
        rgb,
        init_scale_m=config.init_scale_m,
        learning_rates=config.learning_rates,
    )
    completed_steps = 0
    last_metrics: dict[str, Any] = {}
    initial_loss: float | None = None
    best_loss = float("inf")
    if config.resume_checkpoint is not None:
        completed_steps, strategy_state, sampler_state, training_state = load_checkpoint(
            config.resume_checkpoint,
            expected_identity=checkpoint_identity,
            params=params,
            optimizers=optimizers,
            map_location=config.device,
        )
        sampler.set_state(sampler_state.cpu())
        last_metrics = dict(training_state["last_metrics"])
        initial_loss = float(training_state["initial_loss"])
        best_loss = float(training_state["best_loss"])
        if completed_steps >= config.max_steps:
            raise ValueError("checkpoint already reached or exceeded max_steps")

    torch.cuda.reset_peak_memory_stats(config.device)
    started = time.perf_counter()
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    for step in range(completed_steps, config.max_steps):
        index = int(torch.randint(len(trainset), (1,), generator=sampler).item())
        sample = trainset[index]
        tensors = _tensor_sample(sample, torch, config.device)
        has_range = "range_m" in tensors and config.lidar_range_weight > 0.0
        rendered, rendered_range, _, info = backend.render(params, sample, with_range=has_range)
        l1 = masked_rgb_l1(rendered, tensors["rgb"], tensors["rgb_mask"])
        ssim = masked_rgb_ssim_loss(rendered, tensors["rgb"], tensors["rgb_mask"])
        loss = config.rgb_l1_weight * l1 + config.rgb_ssim_weight * ssim
        range_loss = None
        if has_range:
            assert rendered_range is not None
            range_loss = confidence_weighted_range_l1(
                rendered_range,
                tensors["range_m"],
                tensors["confidence"],
                tensors["depth_mask"],
            )
            loss = loss + config.lidar_range_weight * range_loss
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        backend.strategy_post_step(
            params,
            optimizers,
            strategy_state,
            step=step,
            info=info,
        )
        last_metrics = {
            "loss": float(loss.detach().cpu()),
            "rgb_l1": float(l1.detach().cpu()),
            "rgb_ssim_loss": float(ssim.detach().cpu()),
            "lidar_range_l1_m": None
            if range_loss is None
            else float(range_loss.detach().cpu()),
        }
        if initial_loss is None:
            initial_loss = last_metrics["loss"]
        best_loss = min(best_loss, last_metrics["loss"])
        completed = step + 1
        if completed % config.checkpoint_every == 0 or completed == config.max_steps:
            save_checkpoint(
                checkpoint_path,
                step=completed,
                identity=checkpoint_identity,
                params=params,
                optimizers=optimizers,
                strategy_state=strategy_state,
                sampler_state=sampler.get_state(),
                training_state={
                    "last_metrics": last_metrics,
                    "initial_loss": initial_loss,
                    "best_loss": best_loss,
                },
            )

    torch.cuda.synchronize(config.device)
    duration_seconds = time.perf_counter() - started
    peak_vram_bytes = int(torch.cuda.max_memory_allocated(config.device))
    frames = _save_evaluation_artifacts(
        backend=backend,
        params=params,
        dataset=valset,
        output_dir=output_dir,
    )
    run_manifest = sign_run_manifest(
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "dataset_manifest_sha256": trainset.dataset_sha256,
            "mask_manifest_sha256": trainset.mask_sha256,
            "split_manifest_sha256": trainset.split_sha256,
            "depth_manifest_sha256": trainset.depth_sha256,
            "coordinate_transform_sha256": coordinate["coordinate_transform_sha256"],
            "trainer_config_sha256": config_sha256,
            "trainer_contract": contract,
            "gsplat_runtime": backend.runtime,
            "initialization_ply_sha256": initialization_sha256,
            "frames": frames,
            "training": {
                "status": "COMPLETE",
                "completed_steps": config.max_steps,
                "duration_seconds": duration_seconds,
                "peak_vram_bytes": peak_vram_bytes,
                "gaussian_count": len(params["means"]),
                "model_path": checkpoint_path.relative_to(output_dir).as_posix(),
                "last_metrics": last_metrics,
                "initial_loss": initial_loss,
                "best_loss": best_loss,
                "loss_improvement_fraction": None
                if initial_loss in (None, 0.0)
                else (initial_loss - last_metrics["loss"]) / initial_loss,
            },
        }
    )
    _atomic_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest


def train_from_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return train(TrainerConfig.from_dict(value))
