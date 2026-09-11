"""Hidden-point rejection in ``tools/build_face4_lidar_geometry.py``.

The warp path used to record ``visibility_cell_px`` in the manifest without
ever applying ``visible_point_mask``; these tests pin the filter to both
projection paths on a synthetic wall with LiDAR returns behind it.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_face4_lidar_geometry import (  # noqa: E402
    MANIFEST_NAME,
    VISIBILITY_APPLIED_IN_WARP,
    build_face4_lidar_geometry,
    face_visibility_keep_mask,
    verify_cache_visibility,
)

from cloudstudio_3dgs.data.depth_cache import (  # noqa: E402
    load_sparse_depth,
    sparse_depth_npz_bytes,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes  # noqa: E402
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec  # noqa: E402
from cloudstudio_3dgs.geometry.kb4 import unproject_kb4  # noqa: E402
from cloudstudio_3dgs.geometry.lidar_projection import (  # noqa: E402
    DepthProjectionConfig,
    SparseDepthMap,
    project_camera_points_to_face,
)
from cloudstudio_3dgs.training.face_dataset import (  # noqa: E402
    FACE_CACHE_SCHEMA_VERSION,
    sign_face_manifest,
)

WALL_M = 5.0
BEHIND_M = 8.0  # > (1 + 0.2) * 5.0 + 0.1, so the loose rule rejects it
IMAGE_ID = "img_synthetic"
CAMERA_ID = "left"
FACE_ID = "front"


def camera_fixture() -> dict:
    return {
        "camera_id": CAMERA_ID,
        "camera_type": "fisheye",
        "width": 64,
        "height": 64,
        "intrinsic": {"fl_x": 20.0, "fl_y": 20.0, "cx": 31.5, "cy": 31.5},
        "distortion": {
            "camera_model": "OPENCV_FISHEYE",
            "params": {"k1": 0.02, "k2": -0.003, "k3": 0.0002, "k4": 0.0},
        },
    }


def front_face() -> FaceSpec:
    size, fx = 64, 40.0
    return FaceSpec(
        face_id=FACE_ID,
        R_face=np.eye(3),
        K_face=np.array([[fx, 0.0, size / 2.0], [0.0, fx, size / 2.0], [0.0, 0.0, 1.0]]),
        width=size,
        height=size,
        half_fov_deg=float(np.degrees(np.arctan((size / 2.0) / fx))),
    )


def _signed(payload: dict, key: str) -> dict:
    signed = dict(payload)
    signed[key] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return signed


def wall_with_returns_behind() -> SparseDepthMap:
    """Fisheye sparse depth: a wall at 5 m on even pixels, leak-through
    returns at 8 m on the odd pixels between them (their own pixels, so the
    per-pixel z-buffer alone never removes them)."""
    height, width = 64, 64
    pixel_index: list[int] = []
    ranges: list[float] = []
    for y in range(18, 46):
        for x in range(18, 46):
            if y % 2 == 0 and x % 2 == 0:
                pixel_index.append(y * width + x)
                ranges.append(WALL_M)
            elif y % 2 == 1 and x % 2 == 1:
                pixel_index.append(y * width + x)
                ranges.append(BEHIND_M)
    order = np.argsort(pixel_index)
    index = np.asarray(pixel_index, dtype=np.int32)[order]
    count = len(index)
    return SparseDepthMap(
        (height, width),
        index,
        np.asarray(ranges, dtype=np.float32)[order],
        np.ones(count, dtype=np.float32),
        np.arange(count, dtype=np.int64),
        np.ones(count, dtype=np.int32),
    )


def write_synthetic_inputs(root: Path) -> dict[str, Path]:
    """Signed dataset / Face4 / fisheye-depth manifests for one image, one face."""
    face_root = root / "face4"
    depth_root = root / "depth"
    (face_root / "faces").mkdir(parents=True)
    depth_root.mkdir(parents=True)

    dataset = _signed(
        {
            "cameras": [camera_fixture()],
            "images": [{"image_id": IMAGE_ID, "camera_id": CAMERA_ID, "c2w": np.eye(4).tolist()}],
            "point_cloud": {"sha256": "0" * 64},
        },
        "manifest_sha256",
    )
    dataset_path = root / "dataset_manifest.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    mask = Image.fromarray(np.full((64, 64), 255, dtype=np.uint8), mode="L")
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    mask_bytes = buffer.getvalue()
    mask_rel = f"faces/{IMAGE_ID}_{FACE_ID}_mask.png"
    (face_root / mask_rel).write_bytes(mask_bytes)
    face_manifest = sign_face_manifest(
        {
            "schema_version": FACE_CACHE_SCHEMA_VERSION,
            "kind": "fisheye_face_cache",
            "split": "train",
            "source_identity": {"dataset_manifest_sha256": dataset["manifest_sha256"]},
            "cameras": {CAMERA_ID: {"faces": [front_face().to_dict()]}},
            "images": [
                {
                    "image_id": IMAGE_ID,
                    "camera_id": CAMERA_ID,
                    "faces": [
                        {
                            "face_id": FACE_ID,
                            "mask_path": mask_rel,
                            "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                        }
                    ],
                }
            ],
        }
    )
    face_manifest_path = face_root / "face_manifest.json"
    face_manifest_path.write_text(json.dumps(face_manifest), encoding="utf-8")

    depth_bytes = sparse_depth_npz_bytes(wall_with_returns_behind())
    (depth_root / f"{IMAGE_ID}.npz").write_bytes(depth_bytes)
    depth_manifest = _signed(
        {
            "algorithm_version": "kb4_ray_zbuffer_v1",
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "complete_dataset": True,
            "projection": DepthProjectionConfig().to_dict(),
            "point_cloud_sha256": "0" * 64,
            "point_cloud_points": 1,
            "images": [
                {
                    "image_id": IMAGE_ID,
                    "path": f"{IMAGE_ID}.npz",
                    "sha256": hashlib.sha256(depth_bytes).hexdigest(),
                }
            ],
        },
        "depth_manifest_sha256",
    )
    depth_manifest_path = depth_root / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps(depth_manifest), encoding="utf-8")
    return {
        "face_manifest": face_manifest_path,
        "face_root": face_root,
        "dataset_manifest": dataset_path,
        "depth_manifest": depth_manifest_path,
        "depth_root": depth_root,
    }


class FaceVisibilityKeepMaskTests(unittest.TestCase):
    def _raster(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        face_range = np.zeros((48, 48), dtype=np.float64)
        face_valid = np.zeros((48, 48), dtype=bool)
        wall = np.zeros((48, 48), dtype=bool)
        wall[::3, ::3] = True
        behind = np.zeros((48, 48), dtype=bool)
        behind[1::6, 1::6] = True
        face_range[wall] = WALL_M
        face_range[behind] = BEHIND_M
        face_valid[wall | behind] = True
        return face_range, face_valid, behind

    def test_cell_6_removes_returns_behind_a_wall_and_keeps_the_wall(self) -> None:
        face_range, face_valid, behind = self._raster()
        keep, candidates, kept = face_visibility_keep_mask(
            face_range, face_valid, DepthProjectionConfig(visibility_cell_px=6)
        )
        self.assertEqual(candidates, int(face_valid.sum()))
        self.assertEqual(kept, int((face_valid & ~behind).sum()))
        self.assertFalse(keep[behind].any())
        self.assertTrue(keep[face_valid & ~behind].all())

    def test_cell_0_keeps_everything(self) -> None:
        face_range, face_valid, _behind = self._raster()
        keep, candidates, kept = face_visibility_keep_mask(
            face_range, face_valid, DepthProjectionConfig(visibility_cell_px=0)
        )
        self.assertEqual(candidates, kept)
        np.testing.assert_array_equal(keep, face_valid)

    def test_direct_projection_reports_visibility_stats(self) -> None:
        camera = camera_fixture()
        front_pixels = np.array([[31.0, 32.0], [33.0, 32.0], [32.0, 31.0], [32.0, 33.0]])
        rays = unproject_kb4(
            np.vstack([front_pixels, [[32.0, 32.0]]]),
            camera["intrinsic"],
            camera["distortion"]["params"],
        )
        points = np.vstack([rays[:4] * 3.0, rays[4:] * 7.0])
        stats: dict[str, int] = {}
        result = project_camera_points_to_face(
            points,
            front_face(),
            config=DepthProjectionConfig(visibility_cell_px=4),
            stats=stats,
        )
        self.assertEqual(stats, {"visibility_candidates": 5, "visibility_kept": 4})
        self.assertEqual(len(result.range_m), 4)
        self.assertTrue(np.all(np.abs(result.range_m - 3.0) < 1e-3))


class WarpPathVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.mkdtemp(prefix="face4-lidar-vis-")
        cls.root = Path(cls._tmp)
        cls.inputs = write_synthetic_inputs(cls.root / "inputs")
        cls.filtered = cls._build(cls.root / "vis6", cell=6)
        cls.unfiltered = cls._build(cls.root / "vis0", cell=0)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    @classmethod
    def _build(cls, output: Path, *, cell: int) -> dict:
        overrides = (
            {"visibility_cell_px": cell, "visibility_tolerance": 0.2, "visibility_margin_m": 0.1}
            if cell > 0
            else None
        )
        return build_face4_lidar_geometry(
            face_manifest_path=cls.inputs["face_manifest"],
            face_root=cls.inputs["face_root"],
            dataset_manifest_path=cls.inputs["dataset_manifest"],
            depth_manifest_path=cls.inputs["depth_manifest"],
            depth_root=cls.inputs["depth_root"],
            output_root=output,
            projection_overrides=overrides,
        )

    def _face_ranges(self, manifest: dict, root: Path) -> np.ndarray:
        (record,) = manifest["records"]
        return load_sparse_depth(root / record["path"]).range_m

    def test_unfiltered_cache_keeps_the_returns_behind_the_wall(self) -> None:
        ranges = self._face_ranges(self.unfiltered, self.root / "vis0")
        self.assertGreater(int(np.count_nonzero(np.abs(ranges - BEHIND_M) < 1e-3)), 0)
        self.assertGreater(int(np.count_nonzero(np.abs(ranges - WALL_M) < 1e-3)), 0)
        visibility = self.unfiltered["visibility_filter"]
        self.assertFalse(visibility["applied"])
        self.assertIsNone(visibility["requested"])
        self.assertNotIn("_visibility_filtered", self.unfiltered["projection"])
        self.assertNotIn("visibility", self.unfiltered["records"][0])

    def test_cell_6_removes_the_returns_behind_the_wall(self) -> None:
        ranges = self._face_ranges(self.filtered, self.root / "vis6")
        self.assertEqual(int(np.count_nonzero(np.abs(ranges - BEHIND_M) < 1e-3)), 0)
        wall_unfiltered = int(
            np.count_nonzero(np.abs(self._face_ranges(self.unfiltered, self.root / "vis0") - WALL_M) < 1e-3)
        )
        self.assertEqual(int(np.count_nonzero(np.abs(ranges - WALL_M) < 1e-3)), wall_unfiltered)
        self.assertGreater(wall_unfiltered, 0)

    def test_manifest_records_requested_and_applied_filter(self) -> None:
        visibility = self.filtered["visibility_filter"]
        self.assertTrue(visibility["applied"])
        self.assertEqual(
            visibility["requested"],
            {"visibility_cell_px": 6, "visibility_tolerance": 0.2, "visibility_margin_m": 0.1},
        )
        self.assertEqual(visibility["applied_in"], VISIBILITY_APPLIED_IN_WARP)
        self.assertGreater(visibility["removed"], 0)
        self.assertEqual(visibility["candidates"], visibility["kept"] + visibility["removed"])
        self.assertAlmostEqual(
            visibility["removed_fraction"], visibility["removed"] / visibility["candidates"]
        )
        (record,) = self.filtered["records"]
        self.assertEqual(record["visibility"]["kept"], record["valid_pixels"])
        self.assertEqual(record["visibility"]["removed"], visibility["removed"])
        self.assertTrue(self.filtered["projection"].endswith("_visibility_filtered"))
        self.assertEqual(self.filtered["projection_config"]["visibility_cell_px"], 6)
        # The signed manifest on disk carries the same block.
        on_disk = json.loads((self.root / "vis6" / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["visibility_filter"], visibility)

    def test_verify_visibility_flags_the_unfiltered_cache_and_clears_the_filtered_one(self) -> None:
        unfiltered = verify_cache_visibility(
            self.root / "vis0",
            projection_overrides={"visibility_cell_px": 6},
        )
        self.assertGreater(unfiltered["windows_loose_violation_frac"]["max"], 0.0)
        self.assertGreater(unfiltered["rerun_removed_fraction"]["max"], 0.0)
        self.assertIsNotNone(unfiltered["manifest_visibility_filter"])
        self.assertFalse(unfiltered["manifest_visibility_filter"]["applied"])

        filtered = verify_cache_visibility(self.root / "vis6")
        self.assertEqual(filtered["windows_loose_violation_frac"]["max"], 0.0)
        self.assertEqual(filtered["rerun_removed_fraction"]["max"], 0.0)
        self.assertTrue(filtered["manifest_visibility_filter"]["applied"])
        self.assertEqual(filtered["faces"][0]["manifest_visibility"], self.filtered["records"][0]["visibility"])

        with self.assertRaises(ValueError):
            verify_cache_visibility(self.root / "vis0")


if __name__ == "__main__":
    unittest.main()
