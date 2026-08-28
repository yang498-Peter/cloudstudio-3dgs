from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import (
    build_depth_cache,
    load_sparse_depth,
    load_xyz_point_cloud,
    sparse_depth_npz_bytes,
    verify_depth_manifest,
)
from cloudstudio_3dgs.data.image_sample import CropWindow, load_image_sample
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import build_per_image_masks
from cloudstudio_3dgs.data.point_cloud import write_binary_ply
from cloudstudio_3dgs.geometry.kb4 import unproject_kb4
from cloudstudio_3dgs.geometry.lidar_projection import (
    DepthProjectionConfig,
    SparseDepthMap,
    project_camera_points_to_face,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec


def dataset_fixture() -> dict:
    camera = {
        "camera_id": "left",
        "side": "left",
        "camera_type": "fisheye",
        "width": 64,
        "height": 64,
        "intrinsic": {"fl_x": 20.0, "fl_y": 20.0, "cx": 31.5, "cy": 31.5},
        "distortion": {
            "camera_model": "OPENCV_FISHEYE",
            "params": {"k1": 0.02, "k2": -0.003, "k3": 0.0002, "k4": 0.0},
        },
    }
    images = []
    for index in range(2):
        images.append(
            {
                "image_id": f"img_left_{index:03d}",
                "camera_id": "left",
                "side": "left",
                "timestamp_ns": 1_000_000_000 + index,
                "path_root": "recording",
                "path": f"camera/left/{index:03d}.jpg",
                "c2w": np.eye(4).tolist(),
            }
        )
    manifest = {"schema_version": 1, "cameras": [camera], "images": images}
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


class DepthCacheTests(unittest.TestCase):
    def test_direct_face_projection_keeps_nearest_original_las_source(self) -> None:
        face = FaceSpec(
            face_id="front",
            R_face=np.eye(3),
            K_face=np.array(
                [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            width=5,
            height=5,
            half_fov_deg=60.0,
        )
        points_camera = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
                [1.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 3] = False

        depth = project_camera_points_to_face(
            points_camera,
            face,
            source_index=np.array([41, 42, 43], dtype=np.int64),
            supervision_mask=mask,
        )

        np.testing.assert_array_equal(depth.pixel_index, [12])
        np.testing.assert_allclose(depth.range_m, [2.0])
        np.testing.assert_array_equal(depth.source_index, [41])
        np.testing.assert_array_equal(depth.support_count, [2])
        self.assertGreater(float(depth.confidence[0]), 0.0)

    def test_compact_cache_reconstructs_unused_provenance_sentinels(self) -> None:
        depth = SparseDepthMap(
            shape=(4, 4),
            pixel_index=np.array([1, 7], dtype=np.int32),
            range_m=np.array([2.0, 3.0], dtype=np.float32),
            confidence=np.array([0.5, 0.75], dtype=np.float32),
            source_index=np.array([11, 12], dtype=np.int64),
            support_count=np.array([2, 3], dtype=np.int32),
        )
        compact = sparse_depth_npz_bytes(depth, include_provenance=False)
        explicit = sparse_depth_npz_bytes(depth)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "compact.npz"
            path.write_bytes(compact)
            loaded = load_sparse_depth(path)

        self.assertLess(len(compact), len(explicit))
        np.testing.assert_array_equal(loaded.pixel_index, depth.pixel_index)
        np.testing.assert_array_equal(loaded.range_m, depth.range_m)
        np.testing.assert_array_equal(loaded.confidence, depth.confidence)
        np.testing.assert_array_equal(loaded.source_index, [-1, -1])
        np.testing.assert_array_equal(loaded.support_count, [0, 0])

    def test_trainer_compact_confidence_keeps_range_exact(self) -> None:
        depth = SparseDepthMap(
            shape=(4, 4),
            pixel_index=np.array([1, 7], dtype=np.int32),
            range_m=np.array([2.125, 79.875], dtype=np.float32),
            confidence=np.array([0.321, 0.999], dtype=np.float32),
            source_index=np.array([11, 12], dtype=np.int64),
            support_count=np.array([2, 3], dtype=np.int32),
        )
        payload = sparse_depth_npz_bytes(
            depth,
            include_provenance=False,
            confidence_encoding="uint8",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "compact-u8.npz"
            path.write_bytes(payload)
            loaded = load_sparse_depth(path)
            with np.load(path, allow_pickle=False) as archive:
                names = set(archive.files)

        self.assertIn("confidence_u8", names)
        self.assertNotIn("confidence", names)
        np.testing.assert_array_equal(loaded.range_m, depth.range_m)
        np.testing.assert_allclose(loaded.confidence, depth.confidence, atol=1.0 / 255.0)
        np.testing.assert_array_equal(loaded.source_index, [-1, -1])
        np.testing.assert_array_equal(loaded.support_count, [0, 0])

    def test_ecef_scale_point_cloud_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ecef.npy"
            np.save(path, np.array([[4_000_000.0, 1_000_000.0, 5_000_000.0]]))
            with self.assertRaisesRegex(ValueError, "safe local coordinate frame"):
                load_xyz_point_cloud(path)

    def test_checked_in_depth_baseline_records_full_cache_and_keeps_training_gate_open(self) -> None:
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads(
            (root / "baselines" / "gs2_depth_cache.baseline.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(baseline["output"]["complete_dataset"])
        self.assertEqual(baseline["output"]["image_count"], 1238)
        self.assertTrue(baseline["acceptance"]["full_1238_image_cache"])
        self.assertTrue(
            baseline["acceptance"]["full_cache_workers_2_4_artifacts_byte_identical"]
        )
        self.assertEqual(
            baseline["acceptance"]["trainer_contract_train_val_load"],
            "1114_train_124_val_pass",
        )
        self.assertEqual(baseline["acceptance"]["trainer_depth_loss"], "not_run")

    def test_las_input_limit_is_deterministic(self) -> None:
        import laspy

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "points.las"
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x = np.arange(100, dtype=np.float64)
            las.y = np.zeros(100)
            las.z = np.zeros(100)
            las.write(path)
            first = load_xyz_point_cloud(path, max_points=10)
            second = load_xyz_point_cloud(path, max_points=10)

        self.assertEqual(len(first), 10)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 0], np.arange(0, 100, 10))

    def test_parallel_cache_is_byte_deterministic_and_loadable(self) -> None:
        dataset = dataset_fixture()
        camera = dataset["cameras"][0]
        pixels = np.array([[24.0, 24.0], [32.0, 32.0], [40.0, 40.0]])
        rays = unproject_kb4(
            pixels,
            camera["intrinsic"],
            camera["distortion"]["params"],
        )
        xyz = rays * np.array([[4.0], [5.0], [6.0]])
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            point_cloud = root / "sparse_pc.ply"
            write_binary_ply(point_cloud, xyz, rgb)
            masks = build_per_image_masks(dataset, root / "masks")
            first = build_depth_cache(
                dataset,
                masks,
                root / "masks",
                point_cloud,
                root / "depth-a",
                workers=1,
            )
            second = build_depth_cache(
                dataset,
                masks,
                root / "masks",
                point_cloud,
                root / "depth-b",
                workers=2,
            )
            first_manifest = (root / "depth-a" / "depth_manifest.json").read_bytes()
            second_manifest = (root / "depth-b" / "depth_manifest.json").read_bytes()
            first_files = {
                record["image_id"]: (root / "depth-a" / record["path"]).read_bytes()
                for record in first["images"]
            }
            second_files = {
                record["image_id"]: (root / "depth-b" / record["path"]).read_bytes()
                for record in second["images"]
            }
            loaded = load_sparse_depth(root / "depth-a" / first["images"][0]["path"])
            image_path = root / "image.jpg"
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image_path)
            mask_record = masks["images"][0]
            cache_path = root / "depth-a" / first["images"][0]["path"]
            sample = load_image_sample(
                image_path,
                root / "masks" / mask_record["combined_mask_path"],
                depth_path=cache_path,
                confidence_path=cache_path,
                crop=CropWindow(8, 8, 48, 48),
                factor=4,
            )

        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_files, second_files)
        self.assertTrue(first["complete_dataset"])
        self.assertEqual(first["depth_semantics"], "euclidean_ray_range_m")
        self.assertEqual(verify_depth_manifest(first), first["depth_manifest_sha256"])
        self.assertEqual(len(loaded.range_m), 3)
        self.assertEqual(loaded.shape, (64, 64))
        self.assertEqual(sample.image.shape, (12, 12, 3))
        self.assertEqual(sample.depth.shape, (12, 12))
        self.assertEqual(sample.confidence.shape, (12, 12))
        self.assertEqual(sample.mask.shape, (12, 12))
        first["images"][0]["valid_pixels"] = 0
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_depth_manifest(first)

    def test_partial_cache_is_explicit_and_config_changes_cache_key(self) -> None:
        dataset = dataset_fixture()
        camera = dataset["cameras"][0]
        ray = unproject_kb4(
            np.array([[32.0, 32.0]]),
            camera["intrinsic"],
            camera["distortion"]["params"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            point_cloud = root / "sparse_pc.ply"
            write_binary_ply(
                point_cloud,
                ray * 3.0,
                np.full((1, 3), 180, dtype=np.uint8),
            )
            masks = build_per_image_masks(dataset, root / "masks")
            partial = build_depth_cache(
                dataset,
                masks,
                root / "masks",
                point_cloud,
                root / "partial",
                max_images=1,
            )
            changed = build_depth_cache(
                dataset,
                masks,
                root / "masks",
                point_cloud,
                root / "changed",
                max_images=1,
                config=DepthProjectionConfig(max_range_m=40.0),
            )

        self.assertFalse(partial["complete_dataset"])
        self.assertEqual(partial["summary"]["image_count"], 1)
        self.assertNotEqual(partial["cache_key"], changed["cache_key"])


if __name__ == "__main__":
    unittest.main()
