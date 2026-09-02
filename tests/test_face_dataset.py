"""CPU-only synthetic tests for the fisheye face cache tool + dataset.

Builds a tiny synthetic "fisheye" (32x32, near-zero distortion) plus small
hand-planned faces, runs the cache tool's per-sample worker function directly
(no process pool), and round-trips the artifacts through FaceCacheDataset.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloudstudio_3dgs.data.face_warp import (
    build_face_warp_grid,
    warp_image_to_face,
    warp_mask_to_face,
    warp_sparse_depth_to_face,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec, face_weight
from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.face_dataset import (
    FACE_MANIFEST_NAME,
    FaceCacheDataset,
    sign_face_manifest,
    _dilate_bool,
    expand_raster,
    tile_ownership_masks,
)
from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.mono_depth import sign_mono_depth_manifest
from cloudstudio_3dgs.data.face_lidar_geometry import (
    sign_face_lidar_geometry_manifest,
)
from cloudstudio_3dgs.data.renderer_masks import sign_renderer_mask_manifest

_SPEC = importlib.util.spec_from_file_location(
    "_build_face_cache_under_test", REPO_ROOT / "tools" / "build_face_cache.py"
)
bfc = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(bfc)


# ------------------------------ synthetic camera --------------------------------

FISH_W = FISH_H = 32
FISH_K = np.array([[10.0, 0.0, 16.0], [0.0, 10.0, 16.0], [0.0, 0.0, 1.0]])
FISH_COEFFS = np.zeros(4, dtype=np.float64)
FOV_DEG = 190.0

BASE_IMAGE_ID = "img_test01"
CAMERA_ID = "cam_a"
RIG_FRAME_ID = "frame_0001"


def rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def make_face(face_id: str, R: np.ndarray, size: int, fx: float) -> FaceSpec:
    half_fov = float(np.degrees(np.arctan((size / 2.0) / fx)))
    K_face = np.array(
        [[fx, 0.0, size / 2.0], [0.0, fx, size / 2.0], [0.0, 0.0, 1.0]]
    )
    return FaceSpec(
        face_id=face_id,
        R_face=R,
        K_face=K_face,
        width=size,
        height=size,
        half_fov_deg=half_fov,
    )


FRONT = make_face("front", np.eye(3), 8, 4.0)  # half-FOV 45 deg
TILT = make_face("tilt", rot_y(np.radians(30.0)), 8, 8.0)  # half-FOV ~26.6 deg
BACK = make_face("back", rot_y(np.pi), 8, 4.0)  # entirely outside the lens FoV

BASE_C2W = np.eye(4)
BASE_C2W[:3, :3] = rot_x(0.3)
BASE_C2W[:3, 3] = [1.0, 2.0, 3.0]


def make_sample(*, with_depth: bool = True, mask_all_false: bool = False) -> TrainingSample:
    ii, jj = np.meshgrid(np.arange(FISH_H), np.arange(FISH_W), indexing="ij")
    image = np.stack(
        [(ii * 7) % 256, (jj * 7) % 256, np.full_like(ii, 128)], axis=-1
    ).astype(np.uint8)
    rgb_mask = np.zeros((FISH_H, FISH_W), dtype=bool) if mask_all_false else np.ones(
        (FISH_H, FISH_W), dtype=bool
    )
    depth = confidence = depth_mask = None
    if with_depth:
        depth = np.zeros((FISH_H, FISH_W), dtype=np.float32)
        confidence = np.zeros((FISH_H, FISH_W), dtype=np.float32)
        depth_mask = np.zeros((FISH_H, FISH_W), dtype=bool)
        depth[16, 16], confidence[16, 16], depth_mask[16, 16] = 5.0, 0.8, True
        depth[15, 16], confidence[15, 16], depth_mask[15, 16] = 6.0, 0.9, True
    return TrainingSample(
        image_id=BASE_IMAGE_ID,
        rig_frame_id=RIG_FRAME_ID,
        camera_id=CAMERA_ID,
        image=image,
        rgb_mask=rgb_mask,
        depth_range_m=depth,
        depth_confidence=confidence,
        depth_mask=depth_mask,
        depth_cache_path=None,
        c2w=BASE_C2W.astype(np.float32),
        K=FISH_K.astype(np.float32),
        radial_coeffs=FISH_COEFFS.astype(np.float32),
        width=FISH_W,
        height=FISH_H,
    )


def build_cache(root: Path) -> tuple[Path, dict, list[dict], dict]:
    """Run the tool's worker function directly and sign a manifest."""
    sample = make_sample()
    grids: dict = {}
    record, skipped = bfc.process_sample(
        sample, [FRONT, TILT, BACK], root, fov_deg=FOV_DEG, grids=grids
    )
    payload = bfc.build_manifest_payload(
        fov_deg=FOV_DEG,
        split="train",
        source_identity={"dataset_manifest_sha256": "synthetic"},
        faces_serialized={
            CAMERA_ID: [face.to_dict() for face in (FRONT, TILT, BACK)]
        },
        records=[record],
        skipped=skipped,
    )
    manifest = sign_face_manifest(payload)
    manifest_path = root / FACE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path, record, skipped, grids


class FaceCacheRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.mkdtemp(prefix="face-cache-test-")
        cls.root = Path(cls._tmp)
        cls.manifest_path, cls.record, cls.skipped, cls.grids = build_cache(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def make_dataset(self) -> FaceCacheDataset:
        return FaceCacheDataset(self.manifest_path, self.root)

    def write_renderer_manifest(self, root: Path) -> Path:
        face = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = []
        for image in face["images"]:
            for entry in image["faces"]:
                records.append(
                    {
                        "image_id": image["image_id"],
                        "camera_id": image["camera_id"],
                        "face_id": entry["face_id"],
                        "mask_path": entry["mask_path"],
                        "mask_sha256": entry["mask_sha256"],
                        "keep_pixels": int(entry["mask_true_pixels"]),
                    }
                )
        renderer = sign_renderer_mask_manifest(
            {
                "schema_version": 1,
                "kind": "face4_renderer_mask_cache",
                "split": "train",
                "source_face_manifest_sha256": face["face_manifest_sha256"],
                "policy": {
                    "profile": "mipmap_renderer_visibility_compat_v1",
                    "keep_expression": "face_cache_combined_mask != 0",
                    "competitor_reference_expression": "(seg != 255) & (seg != 33)",
                    "label_33_semantics": "UNKNOWN_NOT_INFERRED",
                },
                "masks": records,
                "summary": {
                    "face_sample_count": len(records),
                    "empty_mask_count": 0,
                    "missing_mask_count": 0,
                },
            }
        )
        path = root / "renderer_mask_manifest.json"
        path.write_text(json.dumps(renderer), encoding="utf-8")
        return path

    def copy_cache(self, destination: Path) -> Path:
        shutil.copytree(self.root, destination, dirs_exist_ok=True)
        return destination / FACE_MANIFEST_NAME

    # --------------------------- tool-side behavior ---------------------------

    def test_back_face_skipped_and_grids_cached(self) -> None:
        self.assertEqual([entry["face_id"] for entry in self.record["faces"]],
                         ["front", "tilt"])
        self.assertEqual(len(self.skipped), 1)
        self.assertEqual(self.skipped[0]["face_id"], "back")
        self.assertEqual(self.skipped[0]["reason"], "warp_all_invalid")
        # One warp grid per face was built and cached for reuse.
        self.assertEqual(set(self.grids), {"front", "tilt", "back"})

    def test_depth_presence_per_face(self) -> None:
        entries = {entry["face_id"]: entry for entry in self.record["faces"]}
        self.assertIsNotNone(entries["front"]["depth_path"])
        # The tilted face does not contain the +z depth points.
        self.assertIsNone(entries["tilt"]["depth_path"])

    def test_external_lidar_geometry_adds_depth_without_resigning_rgb_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self.copy_cache(root)
            face = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = []
            for image in face["images"]:
                for entry in image["faces"]:
                    records.append(
                        {
                            "sample_id": f"{image['image_id']}::{entry['face_id']}",
                            "image_id": image["image_id"],
                            "face_id": entry["face_id"],
                            "path": entry["depth_path"],
                            "sha256": entry["depth_sha256"],
                            "valid_pixels": (
                                0
                                if entry["depth_path"] is None
                                else int(
                                    len(
                                        load_sparse_depth(
                                            root / entry["depth_path"]
                                        ).pixel_index
                                    )
                                )
                            ),
                        }
                    )
                    entry["depth_path"] = None
                    entry["depth_sha256"] = None
            face["summary"]["with_depth_count"] = 0
            face["source_identity"]["depth_manifest_sha256"] = None
            face = sign_face_manifest(face)
            manifest_path.write_text(json.dumps(face), encoding="utf-8")
            lidar = sign_face_lidar_geometry_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_sparse_lidar_geometry",
                    "split": "train",
                    "source_face_manifest_sha256": face["face_manifest_sha256"],
                    "source_depth_manifest_sha256": "d" * 64,
                    "complete_face_cache": True,
                    "expected_face_count": len(records),
                    "records": records,
                }
            )
            lidar_path = root / "face_lidar_geometry_manifest.json"
            lidar_path.write_text(json.dumps(lidar), encoding="utf-8")
            dataset = FaceCacheDataset(
                manifest_path,
                root,
                face_lidar_geometry_manifest_path=lidar_path,
                face_lidar_geometry_root=root,
            )
            self.assertIsNotNone(dataset[0].depth_range_m)
            self.assertIsNone(dataset[1].depth_range_m)
            self.assertEqual(
                dataset.identity["face_lidar_geometry_manifest_sha256"],
                lidar["face_lidar_geometry_manifest_sha256"],
            )

    def test_idempotent_rerun_preserves_artifacts(self) -> None:
        sample = make_sample()
        record2, skipped2 = bfc.process_sample(
            sample, [FRONT, TILT, BACK], self.root, fov_deg=FOV_DEG, grids={}
        )
        self.assertEqual(len(skipped2), 1)
        for first, second in zip(self.record["faces"], record2["faces"]):
            self.assertEqual(first["rgb_sha256"], second["rgb_sha256"])
            self.assertEqual(first["mask_sha256"], second["mask_sha256"])
            self.assertEqual(first["depth_sha256"], second["depth_sha256"])
            self.assertEqual(first["mask_true_pixels"], second["mask_true_pixels"])

    def test_all_false_source_mask_skips_every_face(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record, skipped = bfc.process_sample(
                make_sample(mask_all_false=True),
                [FRONT, TILT, BACK],
                Path(tmp),
                fov_deg=FOV_DEG,
            )
        self.assertEqual(record["faces"], [])
        self.assertEqual(len(skipped), 3)
        self.assertTrue(
            all(item["reason"] in {"warp_all_invalid", "empty_mask"} for item in skipped)
        )

    # -------------------------- dataset-side behavior --------------------------

    def test_getitem_fields_and_pinhole_semantics(self) -> None:
        dataset = self.make_dataset()
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertEqual(sample.image_id, f"{BASE_IMAGE_ID}::front")
        self.assertEqual(sample.camera_id, CAMERA_ID)
        self.assertEqual(sample.rig_frame_id, RIG_FRAME_ID)
        self.assertEqual(sample.camera_model, "pinhole")
        self.assertEqual(sample.image.dtype, np.uint8)
        self.assertEqual(sample.image.shape, (8, 8, 3))
        self.assertEqual(sample.rgb_mask.dtype, np.bool_)
        self.assertEqual((sample.width, sample.height), (8, 8))
        np.testing.assert_allclose(sample.K, FRONT.K_face.astype(np.float32))
        np.testing.assert_array_equal(sample.radial_coeffs, np.zeros(4, np.float32))
        # Front face carries depth; masks compose.
        self.assertIsNotNone(sample.depth_range_m)
        self.assertIsNotNone(sample.depth_confidence)
        self.assertIsNotNone(sample.depth_mask)
        self.assertTrue(np.all(sample.rgb_mask[sample.depth_mask]))
        self.assertIsNotNone(sample.depth_cache_path)
        # Tilted face has no depth artifacts.
        tilt_sample = dataset[1]
        self.assertEqual(tilt_sample.image_id, f"{BASE_IMAGE_ID}::tilt")
        self.assertIsNone(tilt_sample.depth_range_m)
        self.assertIsNone(tilt_sample.depth_mask)
        self.assertIsNone(tilt_sample.depth_cache_path)

    def test_signed_renderer_mask_manifest_is_consumed_and_bound(self) -> None:
        renderer_path = self.write_renderer_manifest(self.root)
        dataset = FaceCacheDataset(
            self.manifest_path,
            self.root,
            renderer_mask_manifest_path=renderer_path,
        )
        self.assertEqual(
            dataset.identity["renderer_mask_manifest_sha256"],
            json.loads(renderer_path.read_text(encoding="utf-8"))[
                "renderer_mask_manifest_sha256"
            ],
        )
        self.assertTrue(np.array_equal(dataset[0].rgb_mask, self.make_dataset()[0].rgb_mask))

        renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
        renderer["source_face_manifest_sha256"] = "f" * 64
        renderer = sign_renderer_mask_manifest(renderer)
        renderer_path.write_text(json.dumps(renderer), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different Face4"):
            FaceCacheDataset(
                self.manifest_path,
                self.root,
                renderer_mask_manifest_path=renderer_path,
            )

    def test_rgb_and_mask_round_trip_match_face_warp(self) -> None:
        dataset = self.make_dataset()
        source = make_sample()
        max_theta = np.radians(FOV_DEG / 2.0)
        for index, face in ((0, FRONT), (1, TILT)):
            grid = build_face_warp_grid(
                FISH_K, FISH_COEFFS, face, max_theta_rad=max_theta
            )
            warped, _valid = warp_image_to_face(
                source.image, FISH_K, FISH_COEFFS, face,
                interpolation="bilinear", grid=grid, max_theta_rad=max_theta,
            )
            expected_rgb = np.clip(np.rint(warped), 0, 255).astype(np.uint8)
            face_mask, _ = warp_mask_to_face(
                source.rgb_mask, FISH_K, FISH_COEFFS, face,
                grid=grid, max_theta_rad=max_theta,
            )
            jj, ii = np.meshgrid(np.arange(8) + 0.5, np.arange(8) + 0.5)
            weights = face_weight(
                face, np.stack([jj.reshape(-1), ii.reshape(-1)], axis=1)
            ).reshape(8, 8)
            expected_mask = face_mask & (weights > bfc.MIN_FACE_WEIGHT)
            sample = dataset[index]
            np.testing.assert_array_equal(sample.image, expected_rgb)
            np.testing.assert_array_equal(sample.rgb_mask, expected_mask)

    def test_depth_round_trip_matches_forward_splat(self) -> None:
        dataset = self.make_dataset()
        source = make_sample()
        expected_range, expected_conf, expected_valid = warp_sparse_depth_to_face(
            source.depth_range_m,
            source.depth_confidence,
            source.depth_mask,
            FISH_K,
            FISH_COEFFS,
            FRONT,
        )
        self.assertTrue(np.any(expected_valid))
        sample = dataset[0]
        loaded_valid = sample.depth_range_m > 0.0
        np.testing.assert_array_equal(loaded_valid, expected_valid)
        np.testing.assert_allclose(
            sample.depth_range_m, expected_range.astype(np.float32), atol=1e-6
        )
        np.testing.assert_allclose(
            sample.depth_confidence, expected_conf.astype(np.float32), atol=1e-6
        )
        np.testing.assert_array_equal(
            sample.depth_mask, sample.rgb_mask & expected_valid
        )

    def test_c2w_composition(self) -> None:
        dataset = self.make_dataset()
        for index, face in ((0, FRONT), (1, TILT)):
            face_to_base = np.eye(4)
            face_to_base[:3, :3] = face.R_face
            expected = BASE_C2W @ face_to_base
            np.testing.assert_allclose(
                dataset[index].c2w, expected.astype(np.float32), atol=1e-5
            )
        # Sanity: the tilted face's forward axis in world coordinates equals
        # base rotation applied to the face center direction.
        world_forward = dataset[1].c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            world_forward,
            (BASE_C2W[:3, :3] @ TILT.center_direction_camera).astype(np.float32),
            atol=1e-5,
        )

    def test_depth_to_range_scale_math(self) -> None:
        dataset = self.make_dataset()
        sample = dataset[0]
        scale = sample.depth_to_range_scale
        self.assertEqual(scale.dtype, np.float32)
        self.assertEqual(scale.shape, (8, 8))
        jj, ii = np.meshgrid(np.arange(8) + 0.5, np.arange(8) + 0.5)
        x = (jj - 4.0) / 4.0
        y = (ii - 4.0) / 4.0
        expected = np.sqrt(1.0 + x * x + y * y)
        np.testing.assert_allclose(scale, expected.astype(np.float32), atol=1e-6)
        # Center pixels are nearly on-axis (z-depth == range); corners are not.
        self.assertLess(float(scale.min()), 1.02)
        self.assertGreaterEqual(float(scale.min()), 1.0)
        corner = float(scale[0, 0])
        self.assertAlmostEqual(corner, float(np.sqrt(1.0 + 2 * (3.5 / 4.0) ** 2)), places=5)
        self.assertGreater(corner, float(scale.min()))
        # Per-face cache returns the same array object on repeat access.
        self.assertIs(scale, dataset[0].depth_to_range_scale)

    def test_exposure_grouping_and_id_contract(self) -> None:
        dataset = self.make_dataset()
        self.assertEqual(
            dataset.image_ids,
            [f"{BASE_IMAGE_ID}::front", f"{BASE_IMAGE_ID}::tilt"],
        )
        self.assertEqual(dataset.exposure_image_ids, [BASE_IMAGE_ID])
        self.assertEqual(dataset.camera_id_by_image, {BASE_IMAGE_ID: CAMERA_ID})
        for sample_id in dataset.image_ids:
            self.assertEqual(dataset.exposure_id_for(sample_id), BASE_IMAGE_ID)
        self.assertIn("face_manifest_sha256", dataset.identity)
        self.assertEqual(dataset.identity["face_plan"], "adaptive_full_fov")
        self.assertEqual(
            dataset.identity["source_identity"],
            {"dataset_manifest_sha256": "synthetic"},
        )

    def test_tile_crop_uses_public_sample_id_and_shifts_intrinsics(self) -> None:
        crop = {
            "sample_id": f"{BASE_IMAGE_ID}::front",
            "x": 2,
            "y": 1,
            "width": 4,
            "height": 5,
        }
        full = self.make_dataset()[0]
        dataset = FaceCacheDataset(
            self.manifest_path,
            self.root,
            tile_views=[crop],
        )
        self.assertEqual(dataset.image_ids, [f"{BASE_IMAGE_ID}::front"])
        self.assertTrue(dataset.identity["tile_cropped"])
        sample = dataset[0]
        np.testing.assert_array_equal(sample.image, full.image[1:6, 2:6])
        np.testing.assert_array_equal(sample.rgb_mask, full.rgb_mask[1:6, 2:6])
        np.testing.assert_allclose(sample.K[0, 2], full.K[0, 2] - 2.0)
        np.testing.assert_allclose(sample.K[1, 2], full.K[1, 2] - 1.0)
        np.testing.assert_array_equal(
            sample.depth_to_range_scale,
            full.depth_to_range_scale[1:6, 2:6],
        )
        if full.depth_range_m is not None:
            np.testing.assert_array_equal(
                sample.depth_range_m,
                full.depth_range_m[1:6, 2:6],
            )

    def test_tile_crop_rejects_da2_filename_style_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Face4 samples"):
            FaceCacheDataset(
                self.manifest_path,
                self.root,
                tile_views=[
                    {
                        "sample_id": f"{BASE_IMAGE_ID}__front",
                        "x": 0,
                        "y": 0,
                        "width": 4,
                        "height": 4,
                    }
                ],
            )

    def test_tile_crop_shifts_intrinsics_and_consumes_aligned_da2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self.copy_cache(root)
            face_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            da2_path = root / "da2_front.npz"
            with da2_path.open("wb") as stream:
                np.savez(stream, relative_depth=np.full((4, 4), 10.0, np.float16))
            da2_sha = hashlib.sha256(da2_path.read_bytes()).hexdigest()
            mono = sign_mono_depth_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_da2_relative_depth_cache",
                    "split": "train",
                    "source_face_manifest_sha256": face_manifest[
                        "face_manifest_sha256"
                    ],
                    "dataset_manifest_sha256": "synthetic",
                    "lidar_depth_manifest_sha256": "d" * 64,
                    "complete_face_cache": True,
                    "expected_face_count": 2,
                    "records": [
                        {
                            "sample_id": f"{BASE_IMAGE_ID}__front",
                            "path": da2_path.name,
                            "sha256": da2_sha,
                            "alignment": {
                                "valid": True,
                                "scale": 2.0,
                                "shift": 1.0,
                            },
                        },
                        {
                            "sample_id": f"{BASE_IMAGE_ID}__tilt",
                            "path": da2_path.name,
                            "sha256": da2_sha,
                            "alignment": {"valid": False},
                        },
                    ],
                }
            )
            mono_path = root / "mono_depth_manifest.json"
            mono_path.write_text(json.dumps(mono), encoding="utf-8")
            dataset = FaceCacheDataset(
                manifest_path,
                root,
                tile_views=[
                    {
                        "sample_id": f"{BASE_IMAGE_ID}::front",
                        "x": 1,
                        "y": 2,
                        "width": 4,
                        "height": 3,
                    }
                ],
                mono_depth_manifest_path=mono_path,
                mono_depth_root=root,
            )
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(sample.image.shape, (3, 4, 3))
            self.assertEqual((sample.width, sample.height), (4, 3))
            self.assertAlmostEqual(float(sample.K[0, 2]), 3.0)
            self.assertAlmostEqual(float(sample.K[1, 2]), 2.0)
            np.testing.assert_allclose(sample.mono_depth_range_m, 21.0)
            self.assertTrue(np.all(sample.mono_depth_mask <= sample.rgb_mask))
            self.assertEqual(sample.depth_to_range_scale.shape, (3, 4))

    def test_mono_depth_far_cutoff_masks_saturated_targets(self) -> None:
        """Aligned monocular depth saturates in the sky and aligns to
        kilometres; a far cutoff must drop those pixels from supervision and
        must not touch anything when it is left unset."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self.copy_cache(root)
            face_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            da2_path = root / "da2_front.npz"
            relative = np.full((4, 4), 10.0, np.float16)
            relative[0, :] = 5000.0
            with da2_path.open("wb") as stream:
                np.savez(stream, relative_depth=relative)
            da2_sha = hashlib.sha256(da2_path.read_bytes()).hexdigest()
            mono = sign_mono_depth_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_da2_relative_depth_cache",
                    "split": "train",
                    "source_face_manifest_sha256": face_manifest[
                        "face_manifest_sha256"
                    ],
                    "dataset_manifest_sha256": "synthetic",
                    "lidar_depth_manifest_sha256": "d" * 64,
                    "complete_face_cache": True,
                    "expected_face_count": 2,
                    "records": [
                        {
                            "sample_id": f"{BASE_IMAGE_ID}__front",
                            "path": da2_path.name,
                            "sha256": da2_sha,
                            "alignment": {
                                "valid": True,
                                "scale": 2.0,
                                "shift": 1.0,
                            },
                        },
                        {
                            "sample_id": f"{BASE_IMAGE_ID}__tilt",
                            "path": da2_path.name,
                            "sha256": da2_sha,
                            "alignment": {"valid": False},
                        },
                    ],
                }
            )
            mono_path = root / "mono_depth_manifest.json"
            mono_path.write_text(json.dumps(mono), encoding="utf-8")

            def build(max_range):
                return FaceCacheDataset(
                    manifest_path,
                    root,
                    mono_depth_manifest_path=mono_path,
                    mono_depth_root=root,
                    mono_depth_max_range_m=max_range,
                )

            unset = build(None)[0]
            cut = build(30.0)[0]
            # Row 0 of the relative map aligns to 10,001 m, the rest to 21 m;
            # the resize to the face grid smears the saturated row, so gate on
            # the aligned range the dataset actually produced.
            expected_far = unset.mono_depth_mask & (unset.mono_depth_range_m < 30.0)
            np.testing.assert_array_equal(cut.mono_depth_mask, expected_far)
            self.assertLess(int(cut.mono_depth_mask.sum()), int(unset.mono_depth_mask.sum()))
            self.assertGreater(int(cut.mono_depth_mask.sum()), 0)
            self.assertTrue(np.array_equal(unset.mono_depth_range_m, cut.mono_depth_range_m))
            self.assertTrue(np.all(build(20.0)[0].mono_depth_mask == False))  # noqa: E712
            with self.assertRaises(ValueError):
                build(0.0)

    def test_rig_frame_surface_for_pose_refinement(self) -> None:
        """Pose refinement needs the same three members S1TrainingDataset has;
        their absence killed the first pose-refined face-cache run at step 0."""
        dataset = self.make_dataset()
        self.assertEqual(dataset.rig_frame_ids, (RIG_FRAME_ID,))
        centers = dataset.rig_frame_centers()
        self.assertEqual(set(centers), {RIG_FRAME_ID})
        np.testing.assert_allclose(centers[RIG_FRAME_ID], BASE_C2W[:3, 3])
        # One rig frame: every budget selects every face sample.
        self.assertEqual(dataset.indices_for_rig_frames(1),
                         tuple(range(len(dataset))))
        self.assertEqual(dataset.indices_for_rig_frames(5),
                         tuple(range(len(dataset))))
        with self.assertRaises(ValueError):
            dataset.indices_for_rig_frames(0)

    def test_missing_cache_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self.copy_cache(Path(tmp))
            dataset = FaceCacheDataset(manifest_path, Path(tmp))
            tilt_rgb = Path(tmp) / "faces" / f"{BASE_IMAGE_ID}_tilt_rgb.png"
            tilt_rgb.unlink()
            with self.assertRaises(FileNotFoundError):
                dataset[1]

    def test_tampered_manifest_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self.copy_cache(Path(tmp))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["fov_deg"] = 123.0  # not re-signed
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                FaceCacheDataset(manifest_path, Path(tmp))

    def test_empty_mask_entries_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self.copy_cache(Path(tmp))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["images"][0]["faces"][1]["mask_true_pixels"] = 0
            manifest = sign_face_manifest(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = FaceCacheDataset(manifest_path, Path(tmp))
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.filtered_empty_mask_count, 1)
            self.assertEqual(dataset.image_ids, [f"{BASE_IMAGE_ID}::front"])


if __name__ == "__main__":
    unittest.main()


class TileOwnershipMaskTests(unittest.TestCase):
    def test_dilate_bool_matches_brute_force_without_wraparound(self) -> None:
        rng = np.random.default_rng(3)
        mask = rng.random((13, 17)) < 0.08
        radius = 2
        expected = np.zeros_like(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                lo_i, hi_i = max(0, i - radius), min(mask.shape[0], i + radius + 1)
                lo_j, hi_j = max(0, j - radius), min(mask.shape[1], j + radius + 1)
                expected[i, j] = mask[lo_i:hi_i, lo_j:hi_j].any()
        np.testing.assert_array_equal(_dilate_bool(mask, radius), expected)
        self.assertIs(_dilate_bool(mask, 0), mask)

    def test_foreign_returns_are_removed_and_owned_neighbourhoods_protected(self) -> None:
        # Identity pose, unit-focal camera looking down +z; the box owns z < 5.
        height, width = 9, 9
        K = np.array([[4.0, 0.0, 4.5], [0.0, 4.0, 4.5], [0.0, 0.0, 1.0]], np.float32)
        c2w = np.eye(4, dtype=np.float32)
        depth_range = np.zeros((height, width), np.float32)
        depth_mask = np.zeros((height, width), bool)
        # Owned return at the centre (z-depth 2 m), foreign return in the corner (z 9 m).
        for (v, u, z) in ((4, 4, 2.0), (0, 8, 9.0)):
            x = (u + 0.5 - 4.5) / 4.0
            y = (v + 0.5 - 4.5) / 4.0
            depth_range[v, u] = z * np.sqrt(1.0 + x * x + y * y)
            depth_mask[v, u] = True
        box = np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 5.0]])
        owned, foreign_region = tile_ownership_masks(
            depth_range, depth_mask, K, c2w, box, margin_m=0.0, dilation_px=1
        )
        self.assertTrue(owned[4, 4])
        self.assertFalse(owned[0, 8])
        self.assertEqual(int(owned.sum()), 1)
        # Foreign neighbourhood is the 2x2 corner block; the owned block is untouched.
        self.assertTrue(foreign_region[0, 8] and foreign_region[1, 7])
        self.assertFalse(foreign_region[4, 4] and foreign_region[3, 3])
        self.assertEqual(int(foreign_region.sum()), 4)
        # A crop made entirely of foreign returns must not lose its RGB mask.
        all_foreign_box = np.array([[-10.0, -10.0, 20.0], [10.0, 10.0, 30.0]])
        owned_none, region_all = tile_ownership_masks(
            depth_range, depth_mask, K, c2w, all_foreign_box, margin_m=0.0, dilation_px=20
        )
        self.assertEqual(int(owned_none.sum()), 0)
        self.assertTrue(region_all.all())
        # A margin that swallows the foreign return makes everything owned.
        owned_all, region_none = tile_ownership_masks(
            depth_range, depth_mask, K, c2w, box, margin_m=5.0, dilation_px=1
        )
        self.assertEqual(int(owned_all.sum()), 2)
        self.assertFalse(region_none.any())


class RasterExpansionTests(unittest.TestCase):
    def test_expand_raster_repeats_and_trims_to_the_crop(self) -> None:
        small = np.arange(6, dtype=np.float32).reshape(2, 3)
        big = expand_raster(small, 2, (3, 5))
        self.assertEqual(big.shape, (3, 5))
        np.testing.assert_array_equal(big[0], [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(big[2], [3, 3, 4, 4, 5])
        vec = np.zeros((2, 3, 3), np.float32)
        vec[1, 2] = [0.0, 0.0, 1.0]
        self.assertEqual(expand_raster(vec, 2, (3, 5)).shape, (3, 5, 3))
        self.assertIs(expand_raster(small, 1, (2, 3)), small)
