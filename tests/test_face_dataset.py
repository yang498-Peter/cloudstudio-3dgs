"""CPU-only synthetic tests for the fisheye face cache tool + dataset.

Builds a tiny synthetic "fisheye" (32x32, near-zero distortion) plus small
hand-planned faces, runs the cache tool's per-sample worker function directly
(no process pool), and round-trips the artifacts through FaceCacheDataset.
"""

from __future__ import annotations

import importlib.util
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
)

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
