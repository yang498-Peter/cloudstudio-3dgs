from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import sparse_depth_npz_bytes
from cloudstudio_3dgs.data.image_sample import CropWindow
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import build_per_image_masks
from cloudstudio_3dgs.data.person_masks import PersonMaskConfig, build_person_masks
from cloudstudio_3dgs.data.point_cloud import write_binary_ply
from cloudstudio_3dgs.evaluation.splits import SplitConfig, build_split_manifest, write_split_manifest
from cloudstudio_3dgs.geometry.lidar_projection import SparseDepthMap
from cloudstudio_3dgs.training.backend import verify_gsplat_runtime
from cloudstudio_3dgs.training.checkpoint import load_checkpoint, save_checkpoint
from cloudstudio_3dgs.training.contracts import (
    build_coordinate_transform_manifest,
    verify_coordinate_transform_manifest,
)
from cloudstudio_3dgs.training.dataset import S1TrainingDataset
from cloudstudio_3dgs.training.exposure import (
    ExposureCompensationConfig,
    ExposureCompensator,
)
from cloudstudio_3dgs.training.losses import (
    confidence_weighted_range_l1,
    confidence_weighted_log_range_huber,
    global_masked_rgb_ssim_loss,
    masked_rgb_l1,
    masked_rgb_ssim_loss,
)
from cloudstudio_3dgs.training.scale_calibration import (
    MetricScaleCalibrationConfig,
    build_metric_scale_calibration,
    verify_metric_scale_calibration_report,
)
from cloudstudio_3dgs.training.regularization import (
    GeometryRegularizationConfig,
    clip_oversized_gaussians,
    geometry_regularization_terms,
)
from cloudstudio_3dgs.training.trainer import (
    TrainerConfig,
    active_sh_degree_for_step,
    appearance_learning_rates,
    load_initialization_ply,
    means_lr_for_step,
)


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_fixture(recording_root: Path) -> dict:
    cameras = []
    for side in ("left", "right"):
        cameras.append(
            {
                "camera_id": side,
                "side": side,
                "camera_type": "fisheye",
                "width": 32,
                "height": 32,
                "intrinsic": {"fl_x": 12.0, "fl_y": 14.0, "cx": 15.5, "cy": 16.5},
                "distortion": {
                    "camera_model": "OPENCV_FISHEYE",
                    "params": {"k1": 0.02, "k2": -0.003, "k3": 0.0002, "k4": 0.0},
                },
            }
        )
    images = []
    rig_frames = []
    for frame_index in range(2):
        image_ids = []
        for side in ("left", "right"):
            image_id = f"img_{side}_{frame_index:03d}"
            relative = Path("camera") / side / f"{frame_index:03d}.png"
            path = recording_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((32, 32, 3), 30 + frame_index * 60, dtype=np.uint8)
            pixels[..., 1] += 10 if side == "right" else 0
            Image.fromarray(pixels).save(path, format="PNG", optimize=False)
            image_ids.append(image_id)
            pose = np.eye(4)
            pose[0, 3] = frame_index * 0.2
            pose[1, 3] = -0.05 if side == "left" else 0.05
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
                    "c2w": pose.tolist(),
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
    manifest = {
        "schema_version": 1,
        "coordinate_frame": "s1_local",
        "cameras": cameras,
        "images": images,
        "rig_frames": rig_frames,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


def _write_depth_fixture(root: Path, dataset: dict, masks: dict) -> dict:
    mask_by_id = {item["image_id"]: item for item in masks["images"]}
    records = []
    for image in dataset["images"]:
        image_id = image["image_id"]
        sparse = SparseDepthMap(
            (32, 32),
            np.array([15 * 32 + 15, 16 * 32 + 16], dtype=np.int32),
            np.array([2.0, 3.0], dtype=np.float32),
            np.array([1.0, 0.5], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int32),
        )
        payload = sparse_depth_npz_bytes(sparse)
        relative = f"depth/{image_id}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append(
            {
                "image_id": image_id,
                "camera_id": image["camera_id"],
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": [32, 32],
                "valid_pixels": 2,
                "combined_mask_sha256": mask_by_id[image_id]["combined_mask_sha256"],
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "mask_manifest_sha256": masks["mask_manifest_sha256"],
        "coordinate_frame": "s1_local",
        "depth_semantics": "euclidean_ray_range_m",
        "images": records,
    }
    manifest["depth_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    (root / "depth_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


class TrainingDatasetTests(unittest.TestCase):
    def test_person_layer_is_signed_identity_and_tightens_rgb_and_depth(self) -> None:
        class CenterPersonSegmenter:
            def segment(self, image: np.ndarray) -> list[dict]:
                mask = np.zeros(image.shape[:2], dtype=bool)
                mask[12:20, 12:20] = True
                return [
                    {
                        "mask": mask,
                        "score": 0.99,
                        "box_xyxy": [12.0, 12.0, 20.0, 20.0],
                    }
                ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root / "recording"
            dataset = _dataset_fixture(recording)
            dataset_path = root / "dataset_manifest.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            masks = build_per_image_masks(dataset, root / "masks")
            _write_depth_fixture(root / "depth-cache", dataset, masks)
            person = build_person_masks(
                dataset,
                masks,
                recording,
                root / "person",
                segmenter=CenterPersonSegmenter(),
                model_identity={
                    "runtime": "test",
                    "version": "1",
                    "architecture": "center",
                    "weights": "fixture",
                    "weights_sha256": "a" * 64,
                    "person_class_index": 1,
                },
                config=PersonMaskConfig(dilation_pixels=0),
            )
            split = build_split_manifest(
                dataset,
                SplitConfig(mode="manual", golden_rig_frames=1),
                manual={"rig_000": "train", "rig_001": "val"},
            )
            split_path = root / "split_manifest.json"
            write_split_manifest(split_path, split)
            training = S1TrainingDataset(
                dataset_manifest_path=dataset_path,
                recording_root=recording,
                mask_manifest_path=root / "masks" / "mask_manifest.json",
                mask_root=root / "masks",
                person_mask_manifest_path=root
                / "person"
                / "person_mask_manifest.json",
                person_mask_root=root / "person",
                split_manifest_path=split_path,
                split="train",
                depth_manifest_path=root / "depth-cache" / "depth_manifest.json",
                depth_root=root / "depth-cache",
            )
            sample = training[0]

            self.assertEqual(
                training.identity["person_mask_manifest_sha256"],
                person["person_mask_manifest_sha256"],
            )
            self.assertFalse(sample.rgb_mask[16, 16])
            self.assertFalse(sample.depth_mask[16, 16])

            partial = dict(person)
            partial["images"] = person["images"][:-1]
            partial.pop("person_mask_manifest_sha256")
            partial["person_mask_manifest_sha256"] = hashlib.sha256(
                canonical_json_bytes(partial)
            ).hexdigest()
            partial_path = root / "person" / "partial.json"
            partial_path.write_text(json.dumps(partial), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "person mask manifest must cover every dataset image"
            ):
                S1TrainingDataset(
                    dataset_manifest_path=dataset_path,
                    recording_root=recording,
                    mask_manifest_path=root / "masks" / "mask_manifest.json",
                    mask_root=root / "masks",
                    person_mask_manifest_path=partial_path,
                    person_mask_root=root / "person",
                    split_manifest_path=split_path,
                    split="train",
                )

    def test_raw_fisheye_crop_masks_depth_and_intrinsics_stay_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root / "recording"
            dataset = _dataset_fixture(recording)
            dataset_path = root / "dataset_manifest.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            masks = build_per_image_masks(dataset, root / "masks")
            _write_depth_fixture(root / "depth-cache", dataset, masks)
            split = build_split_manifest(
                dataset,
                SplitConfig(mode="manual", golden_rig_frames=1),
                manual={"rig_000": "train", "rig_001": "val"},
            )
            split_path = root / "split_manifest.json"
            write_split_manifest(split_path, split)
            training = S1TrainingDataset(
                dataset_manifest_path=dataset_path,
                recording_root=recording,
                mask_manifest_path=root / "masks" / "mask_manifest.json",
                mask_root=root / "masks",
                split_manifest_path=split_path,
                split="train",
                depth_manifest_path=root / "depth-cache" / "depth_manifest.json",
                depth_root=root / "depth-cache",
                factor=2,
                crop=CropWindow(4, 6, 24, 20),
            )
            sample = training[0]

        self.assertEqual(len(training), 2)
        self.assertEqual(sample.image.shape, (10, 12, 3))
        self.assertEqual(sample.rgb_mask.shape, (10, 12))
        self.assertEqual(sample.depth_mask.shape, (10, 12))
        self.assertGreater(int(sample.rgb_mask.sum()), int(sample.depth_mask.sum()))
        np.testing.assert_allclose(sample.K[0], [6.0, 0.0, 5.75])
        np.testing.assert_allclose(sample.K[1], [0.0, 7.0, 5.25])
        np.testing.assert_allclose(sample.radial_coeffs, [0.02, -0.003, 0.0002, 0.0])
        self.assertEqual(training.identity["split"], "train")
        self.assertIsNotNone(training.identity["depth_manifest_sha256"])
        np.testing.assert_allclose(
            training.rig_frame_centers()["rig_000"], [0.0, 0.0, 0.0]
        )

    def test_source_image_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root / "recording"
            dataset = _dataset_fixture(recording)
            dataset_path = root / "dataset_manifest.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            masks = build_per_image_masks(dataset, root / "masks")
            split = build_split_manifest(
                dataset,
                SplitConfig(mode="manual", golden_rig_frames=1),
                manual={"rig_000": "train", "rig_001": "val"},
            )
            split_path = root / "split_manifest.json"
            write_split_manifest(split_path, split)
            training = S1TrainingDataset(
                dataset_manifest_path=dataset_path,
                recording_root=recording,
                mask_manifest_path=root / "masks" / "mask_manifest.json",
                mask_root=root / "masks",
                split_manifest_path=split_path,
                split="train",
            )
            first_record = training._records[0][0]
            (recording / first_record["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "source image SHA256 mismatch"):
                training[0]

    def test_partial_depth_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root / "recording"
            dataset = _dataset_fixture(recording)
            dataset_path = root / "dataset_manifest.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            masks = build_per_image_masks(dataset, root / "masks")
            depth = _write_depth_fixture(root / "depth-cache", dataset, masks)
            depth["images"] = depth["images"][:-1]
            depth.pop("depth_manifest_sha256")
            depth["depth_manifest_sha256"] = hashlib.sha256(
                canonical_json_bytes(depth)
            ).hexdigest()
            (root / "depth-cache" / "depth_manifest.json").write_text(
                json.dumps(depth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            split = build_split_manifest(
                dataset,
                SplitConfig(mode="manual", golden_rig_frames=1),
                manual={"rig_000": "train", "rig_001": "val"},
            )
            split_path = root / "split_manifest.json"
            write_split_manifest(split_path, split)

            with self.assertRaisesRegex(ValueError, "depth manifest must cover every dataset image"):
                S1TrainingDataset(
                    dataset_manifest_path=dataset_path,
                    recording_root=recording,
                    mask_manifest_path=root / "masks" / "mask_manifest.json",
                    mask_root=root / "masks",
                    split_manifest_path=split_path,
                    split="train",
                    depth_manifest_path=root / "depth-cache" / "depth_manifest.json",
                    depth_root=root / "depth-cache",
                )


class TrainingContractTests(unittest.TestCase):
    def test_production_training_fails_closed_without_person_masks(self) -> None:
        config = TrainerConfig.from_dict(
            {
                "run_id": "missing-person-mask",
                "dataset_manifest": "dataset.json",
                "recording_root": "recording",
                "mask_manifest": "masks.json",
                "mask_root": "masks",
                "split_manifest": "split.json",
                "initialization_ply": "sparse_pc.ply",
                "output_dir": "run",
                "gsplat_lock": "upstream/cloudstudio_trainer.lock.json",
                "lidar_range_weight": 0.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "requires person_mask_manifest"):
            config.validate()

    def test_checked_in_baseline_keeps_real_and_full_mcmc_gates_open(self) -> None:
        baseline = json.loads(
            (ROOT / "baselines" / "gs2_trainer.baseline.json").read_text(
                encoding="utf-8"
            )
        )
        lock_path = ROOT / "upstream" / "cloudstudio_trainer.lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_semantic_sha256 = hashlib.sha256(canonical_json_bytes(lock)).hexdigest()
        self.assertEqual(
            baseline["runtime"]["lock_semantic_sha256"], lock_semantic_sha256
        )
        self.assertFalse(baseline["runtime"]["source_patch_required"])
        self.assertGreater(
            baseline["synthetic_cuda"]["loss_improvement_fraction"], 0.20
        )
        self.assertEqual(
            baseline["acceptance"][
                "full_mcmc_noise_and_densification_windows_runtime"
            ],
            "not_run_missing_registered_cuda_operator",
        )
        self.assertEqual(
            baseline["acceptance"]["real_gs2_same_config_smoke_regression"],
            "not_run",
        )
        self.assertTrue(
            baseline["acceptance"]["positive_lidar_weight_requires_depth_inputs"]
        )
        self.assertTrue(
            baseline["acceptance"]["partial_depth_manifest_fails_closed"]
        )
        self.assertTrue(
            baseline["acceptance"]["rig_pose_shared_delta_preserves_baseline"]
        )
        self.assertEqual(
            baseline["acceptance"]["real_gs2_rig_pose_refinement_ablation"],
            "not_run_training_paused",
        )

    def test_coordinate_manifest_is_signed_identity_without_normalization(self) -> None:
        manifest = build_coordinate_transform_manifest("a" * 64)
        self.assertEqual(
            verify_coordinate_transform_manifest(manifest),
            manifest["coordinate_transform_sha256"],
        )
        self.assertFalse(manifest["normalization_applied"])
        manifest["model_frame"] = "normalized"
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_coordinate_transform_manifest(manifest)

    def test_knn_scales_and_mcmc_noise_follow_local_metric_scale(self) -> None:
        xyz = np.asarray(
            [
                [x, y, z]
                for x in (0.0, 1.0)
                for y in (0.0, 1.0)
                for z in (0.0, 1.0)
            ],
            dtype=np.float32,
        )
        policy = MetricScaleCalibrationConfig(
            mode="knn",
            knn_neighbors=3,
            means_step_fraction=0.0032,
            noise_std_fraction=0.25,
        )
        base_scales, base = build_metric_scale_calibration(
            xyz,
            policy=policy,
            fixed_scale_m=0.05,
            configured_means_lr=1.6e-4,
            configured_noise_lr=500_000.0,
        )
        large_scales, large = build_metric_scale_calibration(
            xyz * 10.0,
            policy=policy,
            fixed_scale_m=0.05,
            configured_means_lr=1.6e-4,
            configured_noise_lr=500_000.0,
        )

        np.testing.assert_allclose(base_scales, np.ones(len(xyz)), rtol=1e-6)
        np.testing.assert_allclose(large_scales, base_scales * 10.0, rtol=1e-6)
        self.assertAlmostEqual(large["reference_scale_m"], base["reference_scale_m"] * 10.0)
        self.assertAlmostEqual(
            large["effective_means_lr_m"], base["effective_means_lr_m"] * 10.0
        )
        self.assertAlmostEqual(
            large["effective_noise_lr"], base["effective_noise_lr"] / 100.0
        )
        self.assertAlmostEqual(base["nominal_noise_std_fraction"], 0.25)
        self.assertAlmostEqual(large["nominal_noise_std_fraction"], 0.25)
        self.assertAlmostEqual(
            large["nominal_noise_std_m"], base["nominal_noise_std_m"] * 10.0
        )
        self.assertEqual(len(base["scale_calibration_sha256"]), 64)

    def test_fixed_explicit_scale_mode_preserves_gate1_parameters(self) -> None:
        policy = MetricScaleCalibrationConfig(
            mode="fixed",
            means_step_fraction=None,
            noise_std_fraction=None,
        )
        scales, report = build_metric_scale_calibration(
            np.eye(4, 3, dtype=np.float32),
            policy=policy,
            fixed_scale_m=0.05,
            configured_means_lr=1.6e-4,
            configured_noise_lr=500_000.0,
        )
        np.testing.assert_array_equal(scales, np.full(4, 0.05, dtype=np.float32))
        self.assertEqual(report["effective_means_lr_m"], 1.6e-4)
        self.assertEqual(report["effective_noise_lr"], 500_000.0)
        self.assertAlmostEqual(report["nominal_noise_std_fraction"], 4.0, places=6)

    def test_checked_in_real_metric_scale_baseline_is_bound_and_signed(self) -> None:
        baseline = json.loads(
            (ROOT / "baselines" / "gs2_metric_scale_calibration.baseline.json").read_text(
                encoding="utf-8"
            )
        )
        lidar = json.loads(
            (ROOT / "baselines" / "gs2_lidar_init.baseline.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(baseline["source"]["ply_sha256"], lidar["output"]["ply_sha256"])
        self.assertEqual(baseline["source"]["point_count"], lidar["output"]["point_count"])
        self.assertEqual(
            verify_metric_scale_calibration_report(baseline["calibration"]),
            baseline["calibration"]["scale_calibration_sha256"],
        )
        self.assertEqual(baseline["acceptance"]["cpu_contract"], "PASS")
        self.assertEqual(baseline["acceptance"]["real_gpu_short_ab"], "NOT_RUN")
        tampered = json.loads(json.dumps(baseline["calibration"]))
        tampered["effective_noise_lr"] *= 2.0
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_metric_scale_calibration_report(tampered)

    def test_trainer_contract_uses_direct_fisheye_3dgut_mcmc_without_viewer(self) -> None:
        config = TrainerConfig.from_dict(
            {
                "run_id": "contract",
                "dataset_manifest": "dataset.json",
                "recording_root": "recording",
                "mask_manifest": "masks.json",
                "mask_root": "masks",
                "split_manifest": "split.json",
                "initialization_ply": "sparse_pc.ply",
                "output_dir": "run",
                "gsplat_lock": "upstream/cloudstudio_trainer.lock.json",
                "require_person_masks": False,
                "lidar_range_weight": 0.0,
            }
        )
        config.validate()
        contract = config.contract_dict()
        self.assertEqual(contract["renderer"]["camera_model"], "fisheye")
        self.assertEqual(contract["renderer"]["range_mode"], "RGB-Ed")
        self.assertEqual(contract["strategy"]["name"], "MCMC")
        self.assertEqual(contract["initialization"]["mode"], "knn")
        self.assertEqual(contract["optimizer"]["means_step_fraction"], 0.0032)
        self.assertEqual(contract["strategy"]["noise_std_fraction"], 0.25)
        self.assertEqual(
            contract["loss_contract"]["rgb_ssim"]["mode"],
            "mask_aware_local_gaussian",
        )
        self.assertEqual(
            contract["loss_contract"]["lidar_range"]["mode"],
            "robust_log_huber",
        )
        self.assertFalse(contract["rig_pose_refinement"]["enabled"])
        self.assertFalse(contract["dynamic_person_mask"]["required"])
        self.assertFalse(contract["viewer"])
        source = (ROOT / "cloudstudio_3dgs" / "training" / "trainer.py").read_text(encoding="utf-8")
        self.assertNotIn("S1_KEEP_FISHEYE", source)
        self.assertNotIn("simple_trainer", source)
        self.assertNotIn("examples.datasets", source)

    def test_rig_pose_refinement_contract_is_explicit_and_validated(self) -> None:
        config = TrainerConfig.from_dict(
            {
                "run_id": "pose-contract",
                "dataset_manifest": "dataset.json",
                "recording_root": "recording",
                "mask_manifest": "masks.json",
                "mask_root": "masks",
                "split_manifest": "split.json",
                "initialization_ply": "sparse_pc.ply",
                "output_dir": "run",
                "gsplat_lock": "upstream/cloudstudio_trainer.lock.json",
                "require_person_masks": False,
                "lidar_range_weight": 0.0,
                "rig_pose_refinement": {
                    "enabled": True,
                    "learning_rate": 0.0002,
                    "minimum_loss_improvement_fraction": 0.02,
                },
            }
        )
        config.validate()
        contract = config.contract_dict()["rig_pose_refinement"]
        self.assertTrue(contract["enabled"])
        self.assertEqual(contract["learning_rate"], 0.0002)
        self.assertEqual(contract["validation_pose_policy"], "never_optimized")

        invalid = TrainerConfig.from_dict(
            {
                "run_id": "pose-invalid",
                "dataset_manifest": "dataset.json",
                "recording_root": "recording",
                "mask_manifest": "masks.json",
                "mask_root": "masks",
                "split_manifest": "split.json",
                "initialization_ply": "sparse_pc.ply",
                "output_dir": "run",
                "gsplat_lock": "upstream/cloudstudio_trainer.lock.json",
                "require_person_masks": False,
                "lidar_range_weight": 0.0,
                "rig_pose_refinement": {"enabled": True, "maximum_translation_m": 0.0},
            }
        )
        with self.assertRaisesRegex(ValueError, "maximum_translation_m"):
            invalid.validate()

    def test_positive_lidar_weight_requires_depth_inputs(self) -> None:
        config = TrainerConfig.from_dict(
            {
                "run_id": "missing-depth",
                "dataset_manifest": "dataset.json",
                "recording_root": "recording",
                "mask_manifest": "masks.json",
                "mask_root": "masks",
                "split_manifest": "split.json",
                "initialization_ply": "sparse_pc.ply",
                "output_dir": "run",
                "gsplat_lock": "upstream/cloudstudio_trainer.lock.json",
                "require_person_masks": False,
                "lidar_range_weight": 0.05,
            }
        )
        with self.assertRaisesRegex(ValueError, "positive lidar_range_weight requires"):
            config.validate()

    def test_unpatched_lock_is_required_before_importing_runtime(self) -> None:
        lock = json.loads((ROOT / "upstream" / "cloudstudio_trainer.lock.json").read_text(encoding="utf-8"))
        self.assertIsNone(lock["patch"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patched.json"
            lock["patch"] = "local.patch"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not require"):
                verify_gsplat_runtime(path)

    def test_canonical_initialization_ply_round_trips(self) -> None:
        xyz = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]], dtype=np.float32)
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sparse_pc.ply"
            write_binary_ply(path, xyz, rgb)
            actual_xyz, actual_rgb = load_initialization_ply(path)
        np.testing.assert_array_equal(actual_xyz, xyz)
        np.testing.assert_array_equal(actual_rgb, rgb)


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class TorchTrainingContractTests(unittest.TestCase):
    def test_masked_losses_ignore_invalid_pixels_and_use_range_confidence(self) -> None:
        import torch

        target = torch.zeros((2, 2, 3))
        prediction = target.clone()
        prediction[0, 0] = 1.0
        mask = torch.tensor([[False, True], [True, True]])
        self.assertEqual(float(masked_rgb_l1(prediction, target, mask)), 0.0)
        self.assertAlmostEqual(
            float(masked_rgb_ssim_loss(target, target, mask, window_size=1)),
            0.0,
            places=6,
        )
        predicted_range = torch.tensor([[2.5, 7.0], [4.0, 1.0]])
        target_range = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        confidence = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
        depth_mask = torch.tensor([[True, True], [True, False]])
        self.assertAlmostEqual(
            float(confidence_weighted_range_l1(predicted_range, target_range, confidence, depth_mask)),
            1.0 / 3.0,
            places=6,
        )

    def test_local_masked_ssim_uses_windows_and_ignores_invalid_pixels(self) -> None:
        import torch

        target = torch.zeros((15, 15, 3), dtype=torch.float32)
        target[:, 7:, :] = 1.0
        prediction = target.clone()
        mask = torch.ones((15, 15), dtype=torch.bool)
        mask[:3, :] = False
        prediction[:3, :, :] = 100.0
        self.assertAlmostEqual(
            float(
                masked_rgb_ssim_loss(
                    prediction,
                    target,
                    mask,
                    window_size=5,
                    sigma=1.0,
                    min_valid_fraction=0.6,
                )
            ),
            0.0,
            places=6,
        )
        shifted = target.roll(shifts=2, dims=1)
        local = masked_rgb_ssim_loss(
            shifted,
            target,
            mask,
            window_size=5,
            sigma=1.0,
            min_valid_fraction=0.6,
        )
        self.assertGreater(float(local), 0.01)
        self.assertGreater(float(global_masked_rgb_ssim_loss(shifted, target, mask)), 0.0)
        sparse_mask = torch.zeros_like(mask)
        sparse_mask[7, 7] = True
        with self.assertRaisesRegex(ValueError, "no valid local windows"):
            masked_rgb_ssim_loss(
                target,
                target,
                sparse_mask,
                window_size=5,
                sigma=1.0,
                min_valid_fraction=0.8,
            )

    def test_log_range_huber_is_scale_invariant_and_robust(self) -> None:
        import math
        import torch

        target = torch.tensor([[1.0, 10.0]], dtype=torch.float32)
        prediction = target * 2.0
        confidence = torch.ones_like(target)
        mask = torch.ones_like(target, dtype=torch.bool)
        base = confidence_weighted_log_range_huber(
            prediction, target, confidence, mask, delta=0.1
        )
        scaled = confidence_weighted_log_range_huber(
            prediction * 10.0, target * 10.0, confidence, mask, delta=0.1
        )
        self.assertAlmostEqual(float(base), float(scaled), places=6)
        self.assertAlmostEqual(float(base), math.log(2.0) - 0.05, places=6)

    def test_checkpoint_resume_restores_state_and_rejects_identity_change(self) -> None:
        import torch

        params = torch.nn.ParameterDict(
            {"value": torch.nn.Parameter(torch.tensor([2.0, 3.0]))}
        )
        optimizers = {"value": torch.optim.Adam([params["value"]], lr=0.1)}
        generator = torch.Generator().manual_seed(7)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path,
                step=12,
                identity={"dataset": "a"},
                params=params,
                optimizers=optimizers,
                strategy_state={"counter": torch.tensor([3])},
                sampler_state=generator.get_state(),
                training_state={
                    "initial_loss": 1.0,
                    "best_loss": 0.5,
                    "last_metrics": {"loss": 0.6},
                },
            )
            params = torch.nn.ParameterDict(
                {"value": torch.nn.Parameter(torch.tensor([0.0]))}
            )
            optimizers = {"value": torch.optim.Adam([params["value"]], lr=0.1)}
            step, strategy, sampler_state, training_state = load_checkpoint(
                path,
                expected_identity={"dataset": "a"},
                params=params,
                optimizers=optimizers,
                map_location="cpu",
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                load_checkpoint(
                    path,
                    expected_identity={"dataset": "b"},
                    params=params,
                    optimizers=optimizers,
                    map_location="cpu",
                )
        self.assertEqual(step, 12)
        self.assertEqual(params["value"].tolist(), [2.0, 3.0])
        self.assertEqual(int(strategy["counter"].item()), 3)
        self.assertTrue(torch.equal(sampler_state, generator.get_state()))
        self.assertEqual(training_state["best_loss"], 0.5)

    def test_evaluation_saves_adjusted_lidar_supervision_not_source_cache(self) -> None:
        import torch

        from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
        from cloudstudio_3dgs.training.dataset import TrainingSample
        from cloudstudio_3dgs.training.trainer import _save_evaluation_artifacts

        sample = TrainingSample(
            image_id="factor_adjusted",
            rig_frame_id="rig",
            camera_id="left",
            image=np.full((2, 3, 3), 64, dtype=np.uint8),
            rgb_mask=np.array([[True, True, False], [True, True, True]]),
            depth_range_m=np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]], dtype=np.float32),
            depth_confidence=np.array([[1.0, 0.5, 0.0], [0.7, 0.8, 1.0]], dtype=np.float32),
            depth_mask=np.array([[True, False, True], [True, True, True]]),
            depth_cache_path=Path("source-cache-must-not-be-copied.npz"),
            c2w=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            width=3,
            height=2,
        )

        class Dataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> TrainingSample:
                self.assert_index(index)
                return sample

            @staticmethod
            def assert_index(index: int) -> None:
                if index != 0:
                    raise IndexError(index)

        class Backend:
            def __init__(self) -> None:
                self.torch = torch

            @staticmethod
            def render(params, current_sample, *, with_range):
                assert params is None and current_sample is sample and with_range
                return (
                    torch.zeros((2, 3, 3), dtype=torch.float32),
                    torch.ones((2, 3), dtype=torch.float32),
                    None,
                    None,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            frames = _save_evaluation_artifacts(
                backend=Backend(), params=None, dataset=Dataset(), output_dir=first
            )
            _save_evaluation_artifacts(
                backend=Backend(), params=None, dataset=Dataset(), output_dir=second
            )
            relative = Path(frames[0]["lidar_depth_cache_path"])
            adjusted = load_sparse_depth(first / relative)
            self.assertEqual(adjusted.shape, (2, 3))
            np.testing.assert_array_equal(adjusted.pixel_index, np.array([0, 3, 5]))
            np.testing.assert_allclose(adjusted.range_m, np.array([1.0, 4.0, 6.0]))
            np.testing.assert_allclose(adjusted.confidence, np.array([1.0, 0.7, 1.0]))
            np.testing.assert_array_equal(adjusted.source_index, np.full(3, -1))
            np.testing.assert_array_equal(adjusted.support_count, np.zeros(3))
            self.assertEqual(frames[0]["lidar_depth_valid_pixels"], 3)
            self.assertEqual(
                frames[0]["lidar_depth_cache_semantics"],
                "factor_crop_mask_adjusted_euclidean_ray_range_m",
            )
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class RenderScaleContractTests(unittest.TestCase):
    def test_australian_sh_schedule_and_means_decay_are_pure(self) -> None:
        config = TrainerConfig(
            run_id="schedule",
            dataset_manifest=Path("dataset.json"),
            recording_root=Path("recording"),
            mask_manifest=Path("masks.json"),
            mask_root=Path("masks"),
            split_manifest=Path("split.json"),
            initialization_ply=Path("init.ply"),
            output_dir=Path("output"),
            gsplat_lock=Path("lock.json"),
            require_person_masks=False,
            lidar_range_weight=0.0,
            color_model="sh",
            sh_degree=3,
            sh_degree_interval=1000,
            means_lr_final_factor=0.01,
            background_color=(1.0, 1.0, 1.0),
            exposure_compensation=ExposureCompensationConfig(enabled=True),
        )
        config.validate()
        self.assertEqual(active_sh_degree_for_step(config, 0), 0)
        self.assertEqual(active_sh_degree_for_step(config, 999), 0)
        self.assertEqual(active_sh_degree_for_step(config, 1000), 1)
        self.assertEqual(active_sh_degree_for_step(config, 2999), 2)
        self.assertEqual(active_sh_degree_for_step(config, 3000), 3)
        self.assertEqual(active_sh_degree_for_step(config, 30_000), 3)
        self.assertEqual(means_lr_for_step(1e-3, 0.01, step=0, max_steps=100), 1e-3)
        self.assertAlmostEqual(
            means_lr_for_step(1e-3, 0.01, step=100, max_steps=100), 1e-5
        )
        appearance_rates = appearance_learning_rates(
            config,
            {"means": 1e-4, "scales": 1e-3, "quats": 1e-3, "opacities": 1e-2, "colors": 2e-3},
        )
        self.assertNotIn("colors", appearance_rates)
        self.assertEqual(appearance_rates["sh0"], 2e-3)
        self.assertEqual(appearance_rates["shN"], 1e-4)
        contract = config.contract_dict()
        self.assertEqual(contract["color_model"]["mode"], "sh")
        self.assertEqual(contract["background_compositing"]["color"], [1.0, 1.0, 1.0])

    def test_australian_sh_and_background_reach_renderer(self) -> None:
        import torch

        from cloudstudio_3dgs.training.backend import GsplatBackend
        from cloudstudio_3dgs.training.dataset import TrainingSample

        captured = {}

        def rasterization(**kwargs):
            captured.update(kwargs)
            return (
                torch.zeros((1, 2, 2, 3)),
                torch.full((1, 2, 2, 1), 0.25),
                {},
            )

        class Strategy:
            @staticmethod
            def check_sanity(params, optimizers) -> None:
                if set(params) != set(optimizers):
                    raise AssertionError("parameter/optimizer key mismatch")

            @staticmethod
            def initialize_state():
                return {}

        backend = object.__new__(GsplatBackend)
        backend.torch = torch
        backend.device = "cpu"
        backend.strategy = Strategy()
        backend.rasterization = rasterization
        params, optimizers, _ = backend.initialize(
            np.eye(4, 3, dtype=np.float32),
            np.asarray(
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]],
                dtype=np.uint8,
            ),
            init_scale_m=0.05,
            learning_rates={
                name: 1e-4
                for name in ("means", "scales", "quats", "opacities", "sh0", "shN")
            },
            color_model="sh",
            sh_degree=3,
        )
        self.assertEqual(params["sh0"].shape, (4, 1, 3))
        self.assertEqual(params["shN"].shape, (4, 15, 3))
        self.assertAlmostEqual(
            optimizers["shN"].param_groups[0]["lr"],
            optimizers["sh0"].param_groups[0]["lr"],
        )
        sample = TrainingSample(
            image_id="sh_background",
            rig_frame_id="rig",
            camera_id="left",
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            rgb_mask=np.ones((2, 2), dtype=bool),
            depth_range_m=None,
            depth_confidence=None,
            depth_mask=None,
            depth_cache_path=None,
            c2w=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            width=2,
            height=2,
        )
        rgb, _, _, _ = backend.render(
            params,
            sample,
            with_range=False,
            active_sh_degree=2,
            background_rgb=(1.0, 0.5, 0.0),
        )
        self.assertEqual(captured["sh_degree"], 2)
        torch.testing.assert_close(
            captured["colors"], torch.cat((params["sh0"], params["shN"]), dim=1)
        )
        torch.testing.assert_close(
            rgb, torch.tensor([0.75, 0.375, 0.0]).expand_as(rgb)
        )

    def test_exposure_compensator_is_bounded_and_stable(self) -> None:
        import torch

        exposure = ExposureCompensator(
            ["right", "left", "left"],
            config=ExposureCompensationConfig(enabled=True),
            device="cpu",
        )
        with torch.no_grad():
            exposure.log_gains.copy_(torch.tensor([0.0, 2.0]))
        self.assertEqual(list(exposure.index), ["left", "right"])
        self.assertAlmostEqual(float(exposure.gain("left").detach()), 1.0)
        self.assertAlmostEqual(float(exposure.gain("right").detach()), 2.0)
        self.assertGreater(float(exposure.prior_loss().detach()), 0.0)

    def test_exposure_zero_mean_projection_removes_brightness_degeneracy(self) -> None:
        import torch

        anchored = ExposureCompensator(
            ["a", "b", "c", "d"],
            config=ExposureCompensationConfig(enabled=True, zero_mean_projection=True),
            device="cpu",
            group_by_image={"a": "left", "b": "left", "c": "right", "d": "right"},
        )
        with torch.no_grad():
            # Left group drifts bright, right group drifts dark: a single
            # global anchor would accept this (sum is zero-ish) even though
            # each physical camera has absorbed real scene brightness.
            anchored.log_gains.copy_(torch.tensor([0.3, 0.1, -0.25, -0.15]))
        anchored.project_zero_mean()
        gains = anchored.log_gains.detach()
        # Each camera group is centered independently; offsets inside stay.
        torch.testing.assert_close(
            gains, torch.tensor([0.1, -0.1, -0.05, 0.05]), atol=1e-6, rtol=0.0
        )
        report = anchored.report()
        self.assertAlmostEqual(report["mean_log_gain"], 0.0, places=6)
        for value in report["mean_log_gain_by_group"].values():
            self.assertAlmostEqual(value, 0.0, places=6)
        self.assertEqual(
            sorted(report["mean_log_gain_by_group"]), ["left", "right"]
        )

        # Default (off) leaves the gains untouched for run comparability.
        legacy = ExposureCompensator(
            ["a", "b"],
            config=ExposureCompensationConfig(enabled=True),
            device="cpu",
        )
        with torch.no_grad():
            legacy.log_gains.copy_(torch.tensor([0.4, 0.2]))
        legacy.project_zero_mean()
        torch.testing.assert_close(
            legacy.log_gains.detach(), torch.tensor([0.4, 0.2])
        )

    def test_decoupled_ssim_gain_moves_brightness_but_not_structure(self) -> None:
        import torch

        torch.manual_seed(7)
        target = torch.rand(24, 24, 3)
        mask = torch.ones(24, 24, dtype=torch.bool)
        dim_prediction = 0.5 * target

        coupled = masked_rgb_ssim_loss(dim_prediction, target, mask)
        gain = torch.tensor(2.0, requires_grad=True)
        decoupled = masked_rgb_ssim_loss(
            dim_prediction, target, mask, luminance_gain=gain
        )
        # The gain repairs the luminance mismatch entirely, so the decoupled
        # loss must beat the coupled one on a pure brightness error...
        self.assertLess(float(decoupled.detach()), float(coupled.detach()))
        decoupled.backward()
        self.assertIsNotNone(gain.grad)

        # ...but no gain can repair a structural error: shuffled content stays
        # bad regardless of the luminance scale.
        shuffled = target.flatten(0, 1)[torch.randperm(24 * 24)].reshape(24, 24, 3)
        for gain_value in (0.5, 1.0, 2.0):
            broken = masked_rgb_ssim_loss(
                shuffled, target, mask, luminance_gain=torch.tensor(gain_value)
            )
            self.assertGreater(float(broken.detach()), 0.5)

        # Without a gain the refactored luminance*cs form is the original SSIM.
        baseline = masked_rgb_ssim_loss(dim_prediction, target, mask, luminance_gain=None)
        torch.testing.assert_close(baseline, coupled)

    def test_clip_oversized_gaussians_shrinks_and_bumps_only_offenders(self) -> None:
        import torch

        config = GeometryRegularizationConfig(
            screen_clip_enabled=True,
            max_screen_fraction=0.15,
            screen_clip_hardness=1.5,
            screen_clip_opacity_bump=3.0,
            max_world_size_m=1.0,
        )
        params = {
            "scales": torch.tensor(
                [[-3.0, -3.0, -3.0], [1.0, 0.5, 0.0], [-2.0, -2.0, 9.0]]
            ),
            "opacities": torch.tensor([0.0, -2.0, 0.0]),
        }
        # image short side 100 px, threshold 15 px: only the second splat (60
        # px radius) is oversized; hardness caps the shrink at log(1.5).
        report = clip_oversized_gaussians(
            params,
            radii_px=torch.tensor([[4.0, 3.0], [60.0, 10.0], [1.0, 1.0]]),
            image_size_px=100,
            config=config,
        )
        self.assertEqual(report["clipped_count"], 1)
        shrink = math.log(1.5)
        # Screen clip shrinks by log(hardness), then the 1 m world fuse
        # (ln 1 = 0) clamps whatever still exceeds it, on any splat.
        torch.testing.assert_close(
            params["scales"][1], torch.tensor([0.0, 0.0, -shrink])
        )
        torch.testing.assert_close(params["scales"][0], torch.tensor([-3.0, -3.0, -3.0]))
        self.assertAlmostEqual(float(params["opacities"][1]), -2.0 + 3.0 * shrink, places=5)
        # World fuse fires for splats 1 (post-shrink axes above 0) and 2 (e^9 m).
        self.assertEqual(report["world_clamped_count"], 2)
        self.assertAlmostEqual(float(params["scales"][2].max()), 0.0)
        # Disabled config is a no-op.
        noop = clip_oversized_gaussians(
            params,
            radii_px=torch.tensor([[60.0, 60.0], [60.0, 60.0], [60.0, 60.0]]),
            image_size_px=100,
            config=GeometryRegularizationConfig(enabled=False, screen_clip_enabled=True),
        )
        self.assertEqual(noop["clipped_count"], 0)

    def test_metric_geometry_regularization_only_hits_bad_geometry(self) -> None:
        import torch

        config = GeometryRegularizationConfig(
            opacity_sparsity_weight=1.0,
            scale_upper_weight=1.0,
            anisotropy_weight=1.0,
            max_scale_ratio_to_reference=8.0,
            max_anisotropy=10.0,
        )
        params = {
            "opacities": torch.zeros(2, requires_grad=True),
            "scales": torch.tensor(
                [[0.1, 0.1, 0.1], [1.6, 0.01, 0.01]],
                dtype=torch.float32,
            ).log().requires_grad_(),
        }
        terms = geometry_regularization_terms(
            params, reference_scale_m=0.1, config=config
        )
        self.assertAlmostEqual(float(terms["opacity_sparsity"].detach()), 0.5)
        self.assertGreater(float(terms["scale_upper"].detach()), 0.0)
        self.assertGreater(float(terms["anisotropy"].detach()), 0.0)
        terms["total"].backward()
        self.assertTrue(torch.isfinite(params["opacities"].grad).all())
        self.assertTrue(torch.isfinite(params["scales"].grad).all())
        disabled = geometry_regularization_terms(
            params,
            reference_scale_m=0.1,
            config=GeometryRegularizationConfig(enabled=False),
        )
        self.assertEqual(float(disabled["total"]), 0.0)

    def test_backend_initializes_per_point_metric_knn_scales(self) -> None:
        import torch

        from cloudstudio_3dgs.training.backend import GsplatBackend

        class Strategy:
            @staticmethod
            def check_sanity(params, optimizers) -> None:
                if set(params) != set(optimizers):
                    raise AssertionError("parameter/optimizer key mismatch")

            @staticmethod
            def initialize_state():
                return {}

        backend = object.__new__(GsplatBackend)
        backend.torch = torch
        backend.device = "cpu"
        backend.strategy = Strategy()
        xyz = np.eye(4, 3, dtype=np.float32)
        rgb = np.full((4, 3), 128, dtype=np.uint8)
        scales = np.array([0.01, 0.02, 0.04, 0.08], dtype=np.float32)
        params, optimizers, state = backend.initialize(
            xyz,
            rgb,
            init_scales_m=scales,
            learning_rates={
                name: 1e-4
                for name in ("means", "scales", "quats", "opacities", "colors")
            },
        )
        torch.testing.assert_close(
            params["scales"].exp(), torch.as_tensor(scales)[:, None].repeat(1, 3)
        )
        self.assertEqual(set(params), set(optimizers))
        self.assertEqual(state, {})

    def test_backend_converts_stored_log_scales_to_linear(self) -> None:
        import torch

        from cloudstudio_3dgs.training.backend import GsplatBackend
        from cloudstudio_3dgs.training.dataset import TrainingSample

        captured = {}

        def rasterization(**kwargs):
            captured["scales"] = kwargs["scales"].detach().clone()
            return (
                torch.zeros((1, 8, 8, 3)),
                torch.zeros((1, 8, 8, 1)),
                {},
            )

        backend = object.__new__(GsplatBackend)
        backend.torch = torch
        backend.device = "cpu"
        backend.rasterization = rasterization
        params = {
            "means": torch.zeros((4, 3)),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
            "scales": torch.full((4, 3), 0.1).log(),
            "opacities": torch.full((4,), 0.5).logit(),
            "colors": torch.full((4, 3), 0.5).logit(),
        }
        sample = TrainingSample(
            image_id="scale_contract_cpu",
            rig_frame_id="rig",
            camera_id="left",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            rgb_mask=np.ones((8, 8), dtype=bool),
            depth_range_m=None,
            depth_confidence=None,
            depth_mask=None,
            depth_cache_path=None,
            c2w=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            width=8,
            height=8,
        )
        backend.render(params, sample, with_range=False)
        torch.testing.assert_close(captured["scales"], torch.full((4, 3), 0.1))

    def test_rendered_footprint_matches_linear_metric_scale(self) -> None:
        """The rasterizer must receive LINEAR metric scales, not the stored logs.

        Four opaque 0.1 m Gaussians at z=2 m under f=100 px must each cover a
        footprint of roughly f*s/z = 5 px sigma. If the log-domain parameters
        leak through (the bug that collapsed every real scene into mush while
        the small synthetic fixture kept converging), |log 0.1| = 2.3 m blobs
        cover essentially the whole 128x128 frame and this test fails.
        """
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires a CUDA device")
        from cloudstudio_3dgs.training.backend import GsplatBackend
        from cloudstudio_3dgs.training.runtime_evidence import (
            execute_render_scale_contract_smoke,
        )

        try:
            backend = GsplatBackend(
                device="cuda:0",
                cap_max=64,
                lock_path=ROOT / "upstream" / "cloudstudio_trainer.lock.json",
                mcmc_config={"noise_injection_stop_iter": 0},
            )
        except RuntimeError as exc:
            self.skipTest(f"requires the clean locked gsplat runtime: {exc}")
        report = execute_render_scale_contract_smoke(backend)
        self.assertEqual(report["status"], "PASS", report)


if __name__ == "__main__":
    unittest.main()
