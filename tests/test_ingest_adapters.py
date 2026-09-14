"""Adapter detect/load contracts and their fail-closed messages.

Every fixture is synthetic and written into a temporary directory: a miniature
S1 recording (2 rig frames + a 200-point LAS), a hand-written COLMAP binary
model, and a plain pinhole folder in both accepted JSON dialects.  CPU-only.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio3dgs_sdk.ingest.adapters import (  # noqa: E402
    adapter_by_name,
    colmap,
    detect_adapter,
    load_dataset,
    pinhole_folder,
    s1_fisheye,
)
from cloudstudio3dgs_sdk.ingest.bundle import (  # noqa: E402
    CAPABILITY_LIDAR,
    CAPABILITY_RIG,
    CAPABILITY_TIMESTAMPS,
)
from cloudstudio3dgs_sdk.ingest.errors import (  # noqa: E402
    DatasetDetectionError,
    DatasetIncompleteError,
)

IDENTITY_ROTATION = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_IMGPOSE_HEADER = "index x y z roll pitch yaw qx qy qz qw timestamp"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def write_las(path: Path, points: np.ndarray) -> Path:
    import laspy

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    data = laspy.LasData(header)
    data.x = points[:, 0]
    data.y = points[:, 1]
    data.z = points[:, 2]
    data.red = np.full(len(points), 100, dtype=np.uint16)
    data.green = np.full(len(points), 120, dtype=np.uint16)
    data.blue = np.full(len(points), 140, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write(str(path))
    return path


def make_s1_recording(root: Path, *, frames: int = 3, with_cloud: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    calibration = {
        "calibration_time": "2026-01-01_00-00-00",
        "version": "v1",
        "cameras": [
            {
                "name": side,
                "type": "fisheye",
                "width": 64,
                "height": 64,
                "intrinsic": {"fl_x": 20.0, "fl_y": 20.0, "cx": 32.0, "cy": 32.0},
                "distortion": {
                    "camera_model": "OPENCV_FISHEYE",
                    "params": {"k1": 0.01, "k2": 0.0, "k3": 0.0, "k4": 0.0},
                },
                "transform_from_lidar": {
                    "rotation": IDENTITY_ROTATION,
                    "position": [0.05 if side == "left" else -0.05, 0.0, 0.0],
                },
            }
            for side in ("left", "right")
        ],
    }
    (root / "info").mkdir(parents=True, exist_ok=True)
    (root / "info" / "calibration.json").write_text(
        json.dumps(calibration, indent=1), encoding="utf-8"
    )
    lines = [_IMGPOSE_HEADER]
    for index in range(frames):
        stamp = 1_772_726_380_000_000_000 + index * 500_000_000
        for side in ("left", "right"):
            image = root / "camera" / side / f"{stamp}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"jpeg" + bytes([index, 0 if side == "left" else 1]))
            x = 0.05 if side == "left" else -0.05
            lines.append(
                f"{side}/{stamp}.jpg {x} {float(index)} 0.0 0.0 0.0 0.0 "
                f"0.0 0.0 0.0 1.0 {stamp / 1e9:.9f}"
            )
    (root / "ImgPose.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if with_cloud:
        rng = np.random.default_rng(7)
        write_las(root / "colorized.las", rng.uniform(-5.0, 5.0, size=(200, 3)))
    return root


def _colmap_camera_bytes(camera_id: int, model_id: int, width: int, height: int, params) -> bytes:
    return struct.pack("<iiQQ", camera_id, model_id, width, height) + struct.pack(
        "<" + "d" * len(params), *params
    )


def _colmap_image_bytes(image_id: int, qvec, tvec, camera_id: int, name: str) -> bytes:
    payload = struct.pack(
        "<idddddddi", image_id, *qvec, *tvec, camera_id
    )
    payload += name.encode("utf-8") + b"\x00"
    payload += struct.pack("<Q", 0)
    return payload


def make_colmap_model(root: Path, *, images: int = 3, with_cloud: bool = False) -> Path:
    model = root / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)
    (model / "cameras.bin").write_bytes(
        struct.pack("<Q", 1)
        + _colmap_camera_bytes(1, 1, 64, 48, (30.0, 31.0, 32.0, 24.0))
    )
    image_payload = struct.pack("<Q", images)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for index in range(images):
        name = f"frame_{index:03d}.jpg"
        (images_dir / name).write_bytes(b"jpeg" + bytes([index]))
        image_payload += _colmap_image_bytes(
            index + 1, (1.0, 0.0, 0.0, 0.0), (float(index), 0.0, 0.0), 1, name
        )
    (model / "images.bin").write_bytes(image_payload)
    points = struct.pack("<Q", 2)
    for point_id in (1, 2):
        points += struct.pack(
            "<QdddBBBd", point_id, float(point_id), 0.0, 1.0, 10, 20, 30, 0.5
        )
        points += struct.pack("<Q", 0)
    (model / "points3D.bin").write_bytes(points)
    if with_cloud:
        rng = np.random.default_rng(3)
        write_las(root / "lidar.las", rng.uniform(-2.0, 2.0, size=(50, 3)))
    return root


def make_pinhole_folder(root: Path, *, dialect: str = "sdk", frames: int = 3) -> Path:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for index in range(frames):
        name = f"images/{index:04d}.png"
        (root / name).write_bytes(b"png" + bytes([index]))
        matrix = np.eye(4)
        matrix[0, 3] = float(index)
        row = {"file_path": name, "transform_matrix": matrix.tolist()}
        if dialect == "sdk":
            row["camera_id"] = "cam0"
            row["timestamp_ns"] = 1000 + index
            row["split"] = "train" if index < frames - 1 else "val"
        frame_rows.append(row)
    if dialect == "sdk":
        payload = {
            "coordinate_frame": "site_local",
            "pose_convention": "c2w_opencv",
            "cameras": [
                {
                    "camera_id": "cam0",
                    "width": 64,
                    "height": 48,
                    "fx": 30.0,
                    "fy": 31.0,
                    "cx": 32.0,
                    "cy": 24.0,
                    "model": "PINHOLE",
                }
            ],
            "frames": frame_rows,
        }
        (root / "poses.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    else:
        payload = {
            "fl_x": 30.0,
            "fl_y": 31.0,
            "cx": 32.0,
            "cy": 24.0,
            "w": 64,
            "h": 48,
            "frames": frame_rows,
        }
        (root / "transforms.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8"
        )
    return root


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


class DetectionTest(unittest.TestCase):
    def test_each_layout_is_claimed_by_exactly_one_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s1 = make_s1_recording(root / "s1")
            col = make_colmap_model(root / "colmap")
            pin = make_pinhole_folder(root / "pinhole")
            nerf = make_pinhole_folder(root / "nerf", dialect="nerfstudio")
            self.assertIs(detect_adapter(s1), s1_fisheye)
            self.assertIs(detect_adapter(col), colmap)
            self.assertIs(detect_adapter(pin), pinhole_folder)
            self.assertIs(detect_adapter(nerf), pinhole_folder)

    def test_s1_run_directory_with_transforms_is_not_a_pinhole_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1")
            (root / "transforms.json").write_text(
                json.dumps({"fl_x": 1.0, "w": 2, "h": 2, "frames": []}), encoding="utf-8"
            )
            self.assertFalse(pinhole_folder.detect(root))
            self.assertIs(detect_adapter(root), s1_fisheye)

    def test_unknown_layout_lists_what_each_adapter_wanted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DatasetDetectionError) as caught:
                detect_adapter(Path(tmp))
            message = str(caught.exception)
            for adapter in (s1_fisheye, colmap, pinhole_folder):
                self.assertIn(adapter.NAME, message)
                self.assertIn(adapter.REQUIRES[:20], message)

    def test_missing_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DatasetDetectionError):
                detect_adapter(Path(tmp) / "does-not-exist")

    def test_adapter_by_name_rejects_unknown_names(self) -> None:
        with self.assertRaises(DatasetDetectionError) as caught:
            adapter_by_name("nerfstudio")
        self.assertIn("known adapters", str(caught.exception))


# --------------------------------------------------------------------------
# S1
# --------------------------------------------------------------------------


class S1AdapterTest(unittest.TestCase):
    def test_load_produces_a_rig_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1", frames=3)
            bundle = s1_fisheye.load(root)
            self.assertEqual(bundle.adapter, "s1_fisheye")
            self.assertEqual(len(bundle.cameras), 2)
            self.assertEqual(len(bundle.images), 6)
            self.assertEqual(len(bundle.rig_frames), 3)
            self.assertIn(CAPABILITY_LIDAR, bundle.capabilities)
            self.assertIn(CAPABILITY_TIMESTAMPS, bundle.capabilities)
            self.assertIn(CAPABILITY_RIG, bundle.capabilities)
            self.assertEqual(bundle.coordinate_frame, "s1_local")
            self.assertTrue(bundle.point_cloud.path.name.endswith(".las"))

    def test_existing_signed_manifest_is_reused_byte_for_byte(self) -> None:
        from cloudstudio_3dgs.data.manifest import build_manifest, write_manifest_atomic

        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1")
            original = build_manifest(root, root)
            write_manifest_atomic(original, root, force=True)
            bundle = s1_fisheye.load(root)
            self.assertEqual(
                bundle.native_dataset_manifest["manifest_sha256"],
                original["manifest_sha256"],
            )
            # The bundle manifest records the native sha so downstream caches
            # can bind to it without re-deriving the dataset manifest.
            payload = bundle.to_manifest()
            self.assertEqual(
                payload["native_dataset_manifest_sha256"], original["manifest_sha256"]
            )

    def test_rebuilding_the_manifest_matches_the_stored_one(self) -> None:
        from cloudstudio_3dgs.data.manifest import build_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1")
            first = build_manifest(root, root)
            bundle = s1_fisheye.load(root)
            self.assertEqual(
                bundle.native_dataset_manifest["manifest_sha256"],
                first["manifest_sha256"],
            )

    def test_split_manifest_stamps_the_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1")
            bundle = s1_fisheye.load(root)
            first_frame = bundle.native_dataset_manifest["rig_frames"][0]
            split_payload = {
                "rig_frames": [
                    {
                        "rig_frame_id": first_frame["rig_frame_id"],
                        "image_ids": first_frame["image_ids"],
                        "split": "val",
                    }
                ],
                "golden_views": [],
            }
            split_path = root / "split_manifest.json"
            split_path.write_text(json.dumps(split_payload), encoding="utf-8")
            stamped = s1_fisheye.load(root, split_manifest=split_path)
            values = {image.split for image in stamped.images}
            self.assertIn("val", values)
            self.assertEqual(
                sum(1 for image in stamped.images if image.split == "val"), 2
            )

    def test_missing_pose_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1")
            (root / "ImgPose.txt").unlink()
            with self.assertRaises(DatasetIncompleteError) as caught:
                s1_fisheye.load(root)
            self.assertIn("pose file is missing", str(caught.exception))
            self.assertIn("ImgPose.txt", str(caught.exception))

    def test_missing_point_cloud_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1", with_cloud=False)
            with self.assertRaises(DatasetIncompleteError) as caught:
                s1_fisheye.load(root)
            self.assertIn("no LAS/LAZ point cloud", str(caught.exception))

    def test_empty_camera_folder_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1")
            for image in (root / "camera" / "right").iterdir():
                image.unlink()
            with self.assertRaises(DatasetIncompleteError) as caught:
                s1_fisheye.load(root)
            self.assertIn("has no images", str(caught.exception))

    def test_non_s1_path_is_refused_with_the_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DatasetIncompleteError) as caught:
                s1_fisheye.load(Path(tmp))
            self.assertIn("is not an S1 recording", str(caught.exception))


# --------------------------------------------------------------------------
# COLMAP
# --------------------------------------------------------------------------


class ColmapAdapterTest(unittest.TestCase):
    def test_load_maps_world_to_camera_onto_camera_to_world(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_colmap_model(Path(tmp) / "colmap", images=3)
            bundle = colmap.load(root)
            self.assertEqual(len(bundle.images), 3)
            self.assertEqual(len(bundle.cameras), 1)
            camera = bundle.cameras[0]
            self.assertEqual(camera.camera_model, "PINHOLE")
            self.assertAlmostEqual(camera.fx, 30.0)
            self.assertAlmostEqual(camera.cy, 24.0)
            # Identity rotation, tvec = (i, 0, 0) -> camera centre at (-i, 0, 0)
            for index, image in enumerate(bundle.images):
                self.assertAlmostEqual(image.c2w[0][3], -float(index))
                self.assertAlmostEqual(image.c2w[1][1], 1.0)

    def test_sfm_points_are_not_promoted_to_a_lidar_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_colmap_model(Path(tmp) / "colmap")
            bundle = colmap.load(root)
            self.assertIsNone(bundle.point_cloud)
            self.assertNotIn(CAPABILITY_LIDAR, bundle.capabilities)
            self.assertNotIn(CAPABILITY_TIMESTAMPS, bundle.capabilities)
            self.assertTrue(
                any(w.startswith("no_lidar_cloud_sfm_points_only:2") for w in bundle.warnings)
            )

    def test_side_car_lidar_cloud_is_picked_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_colmap_model(Path(tmp) / "colmap", with_cloud=True)
            bundle = colmap.load(root)
            self.assertIsNotNone(bundle.point_cloud)
            self.assertIn(CAPABILITY_LIDAR, bundle.capabilities)

    def test_export_sparse_point_cloud_writes_a_ply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_colmap_model(Path(tmp) / "colmap")
            destination = colmap.export_sparse_point_cloud(root, root / "sfm.ply")
            payload = destination.read_bytes()
            self.assertIn(b"element vertex 2", payload)
            self.assertEqual(len(payload.split(b"end_header\n")[1]), 2 * 15)

    def test_missing_images_fail_closed_with_a_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_colmap_model(Path(tmp) / "colmap", images=3)
            (root / "images" / "frame_001.jpg").unlink()
            with self.assertRaises(DatasetIncompleteError) as caught:
                colmap.load(root)
            self.assertIn("1 registered images are absent", str(caught.exception))
            self.assertIn("frame_001.jpg", str(caught.exception))

    def test_text_model_is_refused_with_the_conversion_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "colmap"
            (root / "sparse" / "0").mkdir(parents=True)
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                (root / "sparse" / "0" / name).write_text("# empty\n", encoding="utf-8")
            with self.assertRaises(DatasetIncompleteError) as caught:
                colmap.load(root)
            self.assertIn("model_converter", str(caught.exception))

    def test_truncated_model_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_colmap_model(Path(tmp) / "colmap")
            model = root / "sparse" / "0" / "cameras.bin"
            model.write_bytes(model.read_bytes()[:12])
            with self.assertRaises(DatasetIncompleteError) as caught:
                colmap.load(root)
            self.assertIn("truncated binary model", str(caught.exception))

    def test_unsupported_camera_model_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_colmap_model(Path(tmp) / "colmap")
            (root / "sparse" / "0" / "cameras.bin").write_bytes(
                struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 99, 64, 48)
            )
            with self.assertRaises(DatasetIncompleteError) as caught:
                colmap.load(root)
            self.assertIn("unsupported COLMAP camera model id 99", str(caught.exception))

    def test_fisheye_model_parameters_land_in_distortion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_colmap_model(Path(tmp) / "colmap", images=1)
            (root / "sparse" / "0" / "cameras.bin").write_bytes(
                struct.pack("<Q", 1)
                + _colmap_camera_bytes(
                    1, 5, 64, 48, (30.0, 31.0, 32.0, 24.0, 0.1, 0.2, 0.3, 0.4)
                )
            )
            bundle = colmap.load(root)
            camera = bundle.cameras[0]
            self.assertEqual(camera.camera_model, "OPENCV_FISHEYE")
            self.assertTrue(camera.is_fisheye)
            self.assertEqual(
                camera.distortion, {"k1": 0.1, "k2": 0.2, "k3": 0.3, "k4": 0.4}
            )


# --------------------------------------------------------------------------
# pinhole folder
# --------------------------------------------------------------------------


class PinholeFolderAdapterTest(unittest.TestCase):
    def test_sdk_dialect_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole", frames=4)
            bundle = pinhole_folder.load(root)
            self.assertEqual(len(bundle.images), 4)
            self.assertEqual(bundle.coordinate_frame, "site_local")
            self.assertIn(CAPABILITY_TIMESTAMPS, bundle.capabilities)
            self.assertEqual(
                sum(1 for image in bundle.images if image.split == "val"), 1
            )
            self.assertAlmostEqual(bundle.images[1].c2w[0][3], 1.0)

    def test_nerfstudio_dialect_converts_opengl_to_opencv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "nerf", dialect="nerfstudio")
            bundle = pinhole_folder.load(root)
            self.assertIn(
                "poses_converted_from_opengl_c2w_to_opencv_c2w", bundle.warnings
            )
            # Identity OpenGL c2w becomes diag(1, -1, -1, 1) in OpenCV axes.
            matrix = np.asarray(bundle.images[0].c2w)
            np.testing.assert_allclose(np.diag(matrix), [1.0, -1.0, -1.0, 1.0])
            self.assertNotIn(CAPABILITY_TIMESTAMPS, bundle.capabilities)
            self.assertIn("no_capture_timestamps_in_poses_json", bundle.warnings)

    def test_point_cloud_is_picked_up_when_declared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole")
            rng = np.random.default_rng(1)
            write_las(root / "cloud.las", rng.uniform(-1.0, 1.0, size=(40, 3)))
            payload = json.loads((root / "poses.json").read_text(encoding="utf-8"))
            payload["point_cloud"] = "cloud.las"
            (root / "poses.json").write_text(json.dumps(payload), encoding="utf-8")
            bundle = pinhole_folder.load(root)
            self.assertIn(CAPABILITY_LIDAR, bundle.capabilities)

    def test_missing_cloud_is_a_warning_not_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole")
            bundle = pinhole_folder.load(root)
            self.assertIsNone(bundle.point_cloud)
            self.assertIn("no_point_cloud_declared_or_found", bundle.warnings)

    def test_declared_cloud_that_is_absent_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole")
            payload = json.loads((root / "poses.json").read_text(encoding="utf-8"))
            payload["point_cloud"] = "nowhere.ply"
            (root / "poses.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DatasetIncompleteError) as caught:
                pinhole_folder.load(root)
            self.assertIn("point cloud is missing", str(caught.exception))

    def test_missing_images_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole", frames=3)
            (root / "images" / "0001.png").unlink()
            with self.assertRaises(DatasetIncompleteError) as caught:
                pinhole_folder.load(root)
            self.assertIn("1 frames name images that do not exist", str(caught.exception))

    def test_bad_transform_shape_names_the_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole")
            payload = json.loads((root / "poses.json").read_text(encoding="utf-8"))
            payload["frames"][1]["transform_matrix"] = [[1.0, 0.0], [0.0, 1.0]]
            (root / "poses.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DatasetIncompleteError) as caught:
                pinhole_folder.load(root)
            self.assertIn("frames[1] transform_matrix is", str(caught.exception))

    def test_unknown_camera_id_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole")
            payload = json.loads((root / "poses.json").read_text(encoding="utf-8"))
            payload["frames"][0]["camera_id"] = "cam9"
            (root / "poses.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DatasetIncompleteError) as caught:
                pinhole_folder.load(root)
            self.assertIn("unknown camera_id 'cam9'", str(caught.exception))

    def test_invalid_json_is_refused_with_the_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole")
            (root / "poses.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(DatasetIncompleteError) as caught:
                pinhole_folder.load(root)
            self.assertIn("not valid JSON", str(caught.exception))

    def test_unknown_dialect_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pinhole"
            root.mkdir()
            (root / "poses.json").write_text(
                json.dumps({"something": "else"}), encoding="utf-8"
            )
            with self.assertRaises(DatasetIncompleteError) as caught:
                pinhole_folder.load(root)
            self.assertIn("sdk dialect", str(caught.exception))

    def test_missing_camera_keys_are_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole")
            payload = json.loads((root / "poses.json").read_text(encoding="utf-8"))
            payload["cameras"][0].pop("fy")
            (root / "poses.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DatasetIncompleteError) as caught:
                pinhole_folder.load(root)
            self.assertIn("cameras[0] is missing required key 'fy'", str(caught.exception))


class LoadDatasetTest(unittest.TestCase):
    def test_load_dataset_detects_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_pinhole_folder(Path(tmp) / "pinhole")
            bundle = load_dataset(root)
            self.assertEqual(bundle.adapter, "pinhole_folder")

    def test_explicit_adapter_overrides_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_s1_recording(Path(tmp) / "s1")
            bundle = load_dataset(root, adapter="s1_fisheye")
            self.assertEqual(bundle.adapter, "s1_fisheye")


if __name__ == "__main__":
    unittest.main()
