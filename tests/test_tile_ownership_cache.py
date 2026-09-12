"""Precomputed Tile-ownership cache (data.tile_ownership_masks +
tools/build_tile_ownership_masks.py + FaceCacheDataset consumer).

CPU-only synthetic checks:

* the pair the builder stores is bit-identical to what the dataset computes
  on the fly, and a dataset consuming the cache yields the same sample;
* a missing record / artifact / crop mismatch / tampered artifact is refused;
* the manifest binding (face cache, renderer mask, box, margin, dilation)
  is enforced;
* the trainer contract gains exactly one key when the cache is consumed.
"""

from __future__ import annotations

import copy
import hashlib
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

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.tile_ownership_masks import (
    TILE_OWNERSHIP_KIND,
    TILE_OWNERSHIP_MANIFEST_SHA_KEY,
    build_tile_ownership_manifest,
    load_ownership_pair,
    ownership_record_counts,
    sign_tile_ownership_manifest,
    verify_tile_ownership_manifest,
    write_ownership_pair,
)
from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.face_dataset import (
    FACE_MANIFEST_NAME,
    FaceCacheDataset,
    sign_face_manifest,
    tile_ownership_masks,
)
from cloudstudio_3dgs.training.tile_inputs import TILE_INPUT_KIND, TILE_INPUT_SCHEMA_VERSION

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional for the CPU suite
    HAS_TORCH = False


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fixture = _load_module("_face_dataset_fixture_for_ownership", REPO_ROOT / "tests" / "test_face_dataset.py")
builder = _load_module(
    "_build_tile_ownership_masks_under_test", REPO_ROOT / "tools" / "build_tile_ownership_masks.py"
)

SAMPLE_ID = f"{fixture.BASE_IMAGE_ID}::front"
CROP = {"sample_id": SAMPLE_ID, "x": 1, "y": 1, "width": 6, "height": 6}
MARGIN_M = 0.0
DILATION_PX = 1


def _make_sample() -> TrainingSample:
    """The fixture's synthetic fisheye with more LiDAR returns spread over
    the FRONT face at distinct depths, so a box can own some and not others."""
    sample = fixture.make_sample(with_depth=True)
    depth = np.zeros_like(sample.depth_range_m)
    confidence = np.zeros_like(sample.depth_confidence)
    mask = np.zeros_like(sample.depth_mask)
    for (v, u, z) in ((16, 16, 5.0), (15, 16, 6.0), (17, 17, 2.0), (14, 18, 9.0), (18, 14, 4.0), (13, 13, 3.0)):
        depth[v, u], confidence[v, u], mask[v, u] = z, 0.9, True
    return TrainingSample(
        image_id=sample.image_id,
        rig_frame_id=sample.rig_frame_id,
        camera_id=sample.camera_id,
        image=sample.image,
        rgb_mask=sample.rgb_mask,
        depth_range_m=depth,
        depth_confidence=confidence,
        depth_mask=mask,
        depth_cache_path=None,
        c2w=sample.c2w,
        K=sample.K,
        radial_coeffs=sample.radial_coeffs,
        width=sample.width,
        height=sample.height,
    )


def _build_face_cache(root: Path) -> Path:
    bfc = fixture.bfc
    record, skipped = bfc.process_sample(
        _make_sample(), [fixture.FRONT, fixture.TILT, fixture.BACK], root, fov_deg=fixture.FOV_DEG, grids={}
    )
    payload = bfc.build_manifest_payload(
        fov_deg=fixture.FOV_DEG,
        split="train",
        source_identity={"dataset_manifest_sha256": "synthetic"},
        faces_serialized={fixture.CAMERA_ID: [face.to_dict() for face in (fixture.FRONT, fixture.TILT, fixture.BACK)]},
        records=[record],
        skipped=skipped,
    )
    manifest = sign_face_manifest(payload)
    path = root / FACE_MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _world_points(sample: TrainingSample) -> np.ndarray:
    """Unproject the sample's LiDAR returns the way tile_ownership_masks does."""
    height, width = sample.depth_mask.shape
    jj, ii = np.meshgrid(np.arange(width, dtype=np.float64) + 0.5, np.arange(height, dtype=np.float64) + 0.5)
    K = sample.K
    x = (jj - float(K[0, 2])) / float(K[0, 0])
    y = (ii - float(K[1, 2])) / float(K[1, 1])
    rng = np.asarray(sample.depth_range_m, dtype=np.float64) / np.sqrt(1.0 + x * x + y * y)
    cam = np.stack([x * rng, y * rng, rng], axis=-1)
    c2w = np.asarray(sample.c2w, dtype=np.float64)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world[sample.depth_mask]


def _tile_inputs_manifest(root: Path, box: list[list[float]], views: list[dict]) -> Path:
    payload = {
        "schema_version": TILE_INPUT_SCHEMA_VERSION,
        "kind": TILE_INPUT_KIND,
        "tile_count": 1,
        "tiles": [
            {
                "tile_id": 1,
                "name": "Tile_1",
                "core_box": box,
                "training_and_export_box": box,
                "view_count": len(views),
                "views": views,
                "initialization": {"path": "init.ply", "sha256": "0" * 64},
            }
        ],
    }
    payload["tile_inputs_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    path = root / "tile_inputs_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Scene:
    """One synthetic face cache, a Tile crop, a box owning a subset of the
    returns, and an ownership cache built by the tool's worker function."""

    def __init__(self, root: Path, *, box: list[list[float]] | None = None, margin_m: float = MARGIN_M, dilation_px: int = DILATION_PX):
        self.root = root
        self.face_root = root / "face4"
        self.face_root.mkdir()
        self.face_manifest = _build_face_cache(self.face_root)
        self.dataset_kwargs = {
            "face_cache_manifest": str(self.face_manifest),
            "face_cache_root": str(self.face_root),
            "tile_views": [dict(CROP)],
            "renderer_mask_manifest": None,
            "face_lidar_geometry_manifest": None,
            "face_lidar_geometry_root": None,
            "verify_artifacts": True,
        }
        self.plain = builder.build_dataset(self.dataset_kwargs)
        plain_sample = self.plain[0]
        points = _world_points(plain_sample)
        assert len(points) >= 2, "the synthetic crop must carry at least two returns"
        if box is None:
            anchor = points[0]
            box = [(anchor - 0.05).tolist(), (anchor + 0.05).tolist()]
        self.box = box
        self.margin_m = float(margin_m)
        self.dilation_px = int(dilation_px)
        self.tile_inputs = _tile_inputs_manifest(root, box, [dict(CROP)])
        self.tile_inputs_sha = json.loads(self.tile_inputs.read_text(encoding="utf-8"))["tile_inputs_manifest_sha256"]
        self.cache_root = root / "ownership"
        self.cache_root.mkdir()
        self.record = builder.compute_ownership_record(
            self.plain, 0, output_root=self.cache_root, box=box, margin_m=self.margin_m, dilation_px=self.dilation_px
        )
        self.manifest = self.write_manifest([self.record])

    def write_manifest(self, records: list[dict], **overrides) -> Path:
        arguments = dict(
            split="train",
            source_face_manifest_sha256=self.plain.face_manifest_sha256,
            renderer_mask_manifest_sha256=None,
            face_lidar_geometry_manifest_sha256=None,
            tile_inputs_manifest_sha256=self.tile_inputs_sha,
            tile_id=1,
            training_and_export_box=self.box,
            margin_m=self.margin_m,
            dilation_px=self.dilation_px,
            source_identity={"dataset_manifest_sha256": "synthetic"},
            records=records,
        )
        arguments.update(overrides)
        manifest = build_tile_ownership_manifest(**arguments)
        path = self.cache_root / "tile_ownership_manifest.json"
        path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return path

    def dataset(self, *, cached: bool, tile_views: list[dict] | None = None, manifest: Path | None = None, **overrides) -> FaceCacheDataset:
        kwargs = dict(
            tile_views=[dict(CROP)] if tile_views is None else tile_views,
            tile_ownership_box=self.box,
            tile_ownership_margin_m=self.margin_m,
            tile_ownership_dilation_px=self.dilation_px,
        )
        if cached:
            kwargs["tile_ownership_cache_manifest_path"] = self.manifest if manifest is None else manifest
            kwargs["tile_ownership_cache_root"] = self.cache_root
        kwargs.update(overrides)
        return FaceCacheDataset(self.face_manifest, self.face_root, **kwargs)


class OwnershipCacheEqualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="tile-ownership-cache-")
        self.scene = _Scene(Path(self._tmp))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_builder_pair_is_bit_identical_to_tile_ownership_masks(self) -> None:
        scene = self.scene
        record = scene.record
        self.assertTrue(record["ownership_applied"])
        self.assertEqual(record["crop"], {k: CROP[k] for k in ("x", "y", "width", "height")})
        plain = scene.plain[0]
        expected_owned, expected_region = tile_ownership_masks(
            plain.depth_range_m, plain.depth_mask, plain.K, plain.c2w,
            np.asarray(scene.box, dtype=np.float64), scene.margin_m, scene.dilation_px,
        )
        owned, region = load_ownership_pair(scene.cache_root, record)
        np.testing.assert_array_equal(owned, expected_owned)
        np.testing.assert_array_equal(region, expected_region)
        # The box was placed around one return: the scene exercises both
        # branches (owned and foreign) rather than passing trivially.
        self.assertGreaterEqual(int(owned.sum()), 1)
        self.assertLess(int(owned.sum()), int(plain.depth_mask.sum()))
        self.assertTrue(region.any())
        self.assertEqual(record["owned_pixels"], int(owned.sum()))
        self.assertEqual(record["foreign_region_pixels"], int(region.sum()))
        self.assertEqual(record["depth_mask_pixels"], int(plain.depth_mask.sum()))
        self.assertEqual(
            record["rgb_dropped_pixels"], int((plain.rgb_mask & region).sum())
        )

    def test_cached_dataset_yields_the_on_the_fly_sample(self) -> None:
        scene = self.scene
        on_the_fly = scene.dataset(cached=False)
        cached = scene.dataset(cached=True)
        self.assertEqual(cached.tile_ownership_cache_manifest_sha256, verify_tile_ownership_manifest(
            json.loads(scene.manifest.read_text(encoding="utf-8"))
        ))
        fly = on_the_fly[0]
        hit = cached[0]
        plain = scene.plain[0]
        for name in ("image", "rgb_mask", "depth_range_m", "depth_confidence", "depth_mask", "K", "c2w", "depth_to_range_scale"):
            np.testing.assert_array_equal(getattr(hit, name), getattr(fly, name), err_msg=name)
        self.assertEqual((hit.width, hit.height), (fly.width, fly.height))
        self.assertIsNone(hit.mono_depth_mask)
        # Ownership actually changed the supervision, so equality is not vacuous.
        self.assertLess(int(fly.rgb_mask.sum()), int(plain.rgb_mask.sum()))
        self.assertLess(int(fly.depth_mask.sum()), int(plain.depth_mask.sum()))
        # The identity is untouched: the cache is bound through the contract.
        self.assertEqual(cached.identity, on_the_fly.identity)

    def test_all_foreign_box_preserves_the_photometric_mask_in_both_paths(self) -> None:
        far_box = [[1000.0, 1000.0, 1000.0], [1001.0, 1001.0, 1001.0]]
        with tempfile.TemporaryDirectory(prefix="tile-ownership-far-") as tmp:
            scene = _Scene(Path(tmp), box=far_box, dilation_px=20)
            self.assertTrue(scene.record["rgb_mask_preserved"])
            self.assertEqual(scene.record["owned_pixels"], 0)
            self.assertEqual(scene.record["rgb_dropped_pixels"], 0)
            fly = scene.dataset(cached=False)[0]
            hit = scene.dataset(cached=True)[0]
            plain = scene.plain[0]
            np.testing.assert_array_equal(hit.rgb_mask, fly.rgb_mask)
            np.testing.assert_array_equal(fly.rgb_mask, plain.rgb_mask)
            np.testing.assert_array_equal(hit.depth_mask, fly.depth_mask)
            self.assertFalse(fly.depth_mask.any())

    def test_record_counts_follow_the_consumer_semantics(self) -> None:
        rgb = np.ones((4, 4), bool)
        depth_mask = np.zeros((4, 4), bool)
        depth_mask[0, 0] = depth_mask[3, 3] = True
        owned = np.zeros((4, 4), bool)
        owned[0, 0] = True
        region = np.zeros((4, 4), bool)
        region[2:, 2:] = True
        counts = ownership_record_counts(rgb, depth_mask, owned, region)
        self.assertEqual((counts["owned_pixels"], counts["foreign_pixels"], counts["rgb_dropped_pixels"], counts["rgb_kept_pixels"]), (1, 1, 4, 12))
        self.assertFalse(counts["rgb_mask_preserved"])
        preserved = ownership_record_counts(rgb, depth_mask, owned, np.ones((4, 4), bool))
        self.assertTrue(preserved["rgb_mask_preserved"])
        self.assertEqual((preserved["rgb_dropped_pixels"], preserved["rgb_kept_pixels"]), (0, 16))
        none = ownership_record_counts(rgb, None, None, None)
        self.assertFalse(none["ownership_applied"])
        self.assertEqual(none["rgb_kept_pixels"], 16)


class OwnershipCacheFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="tile-ownership-fail-")
        self.scene = _Scene(Path(self._tmp))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_missing_sample_record_is_refused(self) -> None:
        other = {"sample_id": f"{fixture.BASE_IMAGE_ID}::tilt", "x": 0, "y": 0, "width": 4, "height": 4}
        with self.assertRaisesRegex(ValueError, "does not cover selected Tile views"):
            self.scene.dataset(cached=True, tile_views=[other])

    def test_crop_mismatch_is_refused(self) -> None:
        shifted = dict(CROP, x=0)
        with self.assertRaisesRegex(ValueError, "crops differ"):
            self.scene.dataset(cached=True, tile_views=[shifted])

    def test_missing_or_tampered_artifact_is_refused(self) -> None:
        artifact = self.scene.cache_root / Path(*self.scene.record["path"].split("/"))
        original = artifact.read_bytes()
        # Tampered bits: SHA mismatch.
        owned, region = load_ownership_pair(self.scene.cache_root, self.scene.record)
        flipped = owned.copy()
        flipped[0, 0] = not flipped[0, 0]
        write_ownership_pair(artifact, flipped, region)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.scene.dataset(cached=True)[0]
        # Restored content, wrong pixel count in the record: refused on load.
        artifact.write_bytes(original)
        bad_record = dict(self.scene.record, owned_pixels=self.scene.record["owned_pixels"] + 1)
        with self.assertRaisesRegex(ValueError, "pixel counts"):
            load_ownership_pair(self.scene.cache_root, bad_record)
        # Gone: construction still binds, the first access fails closed.
        artifact.unlink()
        dataset = self.scene.dataset(cached=True)
        with self.assertRaisesRegex(FileNotFoundError, "tile ownership mask"):
            dataset[0]

    def test_record_claiming_support_the_sample_lacks_is_refused(self) -> None:
        scene = self.scene
        # A tilt-face crop that contains no LiDAR return: the dataset's
        # ``depth_mask.any()`` guard skips ownership there, so the record
        # carries no artifact.
        tilt_crop = {"sample_id": f"{fixture.BASE_IMAGE_ID}::tilt", "x": 2, "y": 0, "width": 6, "height": 6}
        plain_tilt = builder.build_dataset(dict(scene.dataset_kwargs, tile_views=[tilt_crop]))
        self.assertIsNotNone(plain_tilt[0].depth_mask)
        self.assertFalse(plain_tilt[0].depth_mask.any())
        honest = builder.compute_ownership_record(plain_tilt, 0, output_root=scene.cache_root, box=scene.box, margin_m=scene.margin_m, dilation_px=scene.dilation_px)
        self.assertFalse(honest["ownership_applied"])
        self.assertIsNone(honest["path"])
        self.assertEqual(honest["depth_mask_pixels"], 0)
        manifest = scene.write_manifest([scene.record, honest])
        hit = scene.dataset(cached=True, tile_views=[tilt_crop], manifest=manifest)[0]
        fly = scene.dataset(cached=False, tile_views=[tilt_crop])[0]
        np.testing.assert_array_equal(hit.rgb_mask, fly.rgb_mask)
        np.testing.assert_array_equal(hit.depth_mask, fly.depth_mask)
        self.assertFalse(hit.depth_mask.any())
        # A record that borrows the front artifact for the tilt face claims
        # LiDAR support the sample does not have.
        liar = dict(scene.record, sample_id=tilt_crop["sample_id"], face_id="tilt", crop={k: tilt_crop[k] for k in ("x", "y", "width", "height")})
        manifest = scene.write_manifest([scene.record, liar])
        with self.assertRaisesRegex(ValueError, "records LiDAR support"):
            scene.dataset(cached=True, tile_views=[tilt_crop], manifest=manifest)[0]
        # And the opposite lie: no artifact for a sample that has returns.
        denier = dict(
            honest,
            sample_id=SAMPLE_ID,
            face_id="front",
            crop={k: CROP[k] for k in ("x", "y", "width", "height")},
            width=CROP["width"],
            height=CROP["height"],
            crop_pixels=CROP["width"] * CROP["height"],
            rgb_mask_pixels=scene.record["rgb_mask_pixels"],
            rgb_kept_pixels=scene.record["rgb_mask_pixels"],
        )
        manifest = scene.write_manifest([denier])
        with self.assertRaisesRegex(ValueError, "records no LiDAR support"):
            scene.dataset(cached=True, manifest=manifest)[0]

    def test_binding_mismatches_are_refused(self) -> None:
        scene = self.scene
        with self.assertRaisesRegex(ValueError, "margin/dilation"):
            scene.dataset(cached=True, tile_ownership_margin_m=scene.margin_m + 0.5)
        with self.assertRaisesRegex(ValueError, "margin/dilation"):
            scene.dataset(cached=True, tile_ownership_dilation_px=scene.dilation_px + 1)
        other_box = [[v - 1.0 for v in scene.box[0]], [v - 1.0 for v in scene.box[1]]]
        with self.assertRaisesRegex(ValueError, "different Tile box"):
            scene.dataset(cached=True, tile_ownership_box=other_box)
        with self.assertRaisesRegex(ValueError, "requires tile_ownership_box"):
            scene.dataset(cached=True, tile_ownership_box=None)
        with self.assertRaisesRegex(ValueError, "provided together"):
            scene.dataset(cached=False, tile_ownership_cache_manifest_path=scene.manifest)
        rebound = json.loads(scene.manifest.read_text(encoding="utf-8"))
        rebound["source_face_manifest_sha256"] = "a" * 64
        rebound_path = scene.cache_root / "rebound.json"
        rebound_path.write_text(json.dumps(sign_tile_ownership_manifest(rebound)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different Face4 cache"):
            scene.dataset(cached=True, manifest=rebound_path)
        tampered = json.loads(scene.manifest.read_text(encoding="utf-8"))
        tampered["records"][0]["owned_pixels"] += 0  # unchanged content, broken signature
        tampered["tile_id"] = 2
        tampered_path = scene.cache_root / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            scene.dataset(cached=True, manifest=tampered_path)
        # Renderer-mask binding: a cache built against the face mask must
        # not serve a dataset that supervises through a renderer mask.
        with self.assertRaisesRegex(ValueError, "renderer mask"):
            scene.write_manifest([scene.record], renderer_mask_manifest_sha256="b" * 64)
            scene.dataset(cached=True)

    def test_manifest_verification_rejects_inconsistent_records(self) -> None:
        manifest = json.loads(self.scene.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["kind"], TILE_OWNERSHIP_KIND)
        self.assertEqual(verify_tile_ownership_manifest(manifest), manifest[TILE_OWNERSHIP_MANIFEST_SHA_KEY])
        broken = copy.deepcopy(manifest)
        broken["records"][0]["foreign_pixels"] += 1
        with self.assertRaisesRegex(ValueError, "LiDAR pixel counts"):
            verify_tile_ownership_manifest(sign_tile_ownership_manifest(broken))
        broken = copy.deepcopy(manifest)
        broken["records"][0]["rgb_mask_preserved"] = True
        with self.assertRaisesRegex(ValueError, "preserved mask"):
            verify_tile_ownership_manifest(sign_tile_ownership_manifest(broken))
        broken = copy.deepcopy(manifest)
        broken["summary"]["total_owned_pixels"] += 1
        with self.assertRaisesRegex(ValueError, "summary"):
            verify_tile_ownership_manifest(sign_tile_ownership_manifest(broken))
        with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
            build_tile_ownership_manifest(
                split="train", source_face_manifest_sha256="a" * 64, renderer_mask_manifest_sha256=None,
                face_lidar_geometry_manifest_sha256=None, tile_inputs_manifest_sha256="b" * 64, tile_id=1,
                training_and_export_box=[[0, 0, 0], [1, 1, 1]], margin_m=0.5, dilation_px=15,
                source_identity={}, records=[],
            )


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class OwnershipCacheTrainerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="tile-ownership-contract-")
        self.scene = _Scene(Path(self._tmp))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _config(self, **overrides):
        from cloudstudio_3dgs.training.trainer import TrainerConfig

        fields = dict(
            run_id="ownership",
            dataset_manifest=Path("d.json"),
            recording_root=Path("r"),
            mask_manifest=Path("m.json"),
            mask_root=Path("m"),
            split_manifest=Path("s.json"),
            initialization_ply=Path("i.ply"),
            output_dir=Path("o"),
            gsplat_lock=Path("l.json"),
            require_person_masks=False,
            lidar_range_weight=0.0,
            tile_inputs_manifest=self.scene.tile_inputs,
            tile_inputs_root=self.scene.root,
            mipmap_tile_id=1,
            face_cache_manifest=self.scene.face_manifest,
            face_cache_root=self.scene.face_root,
            tile_ownership_masking=True,
            tile_ownership_margin_m=self.scene.margin_m,
            tile_ownership_dilation_px=self.scene.dilation_px,
        )
        fields.update(overrides)
        return TrainerConfig(**fields)

    def test_contract_gains_only_the_cache_sha_when_the_cache_is_used(self) -> None:
        scene = self.scene
        on_the_fly = self._config()
        on_the_fly.validate()
        cached = self._config(tile_ownership_cache_manifest=scene.manifest, tile_ownership_cache_root=scene.cache_root)
        cached.validate()
        fly_contract = on_the_fly.contract_dict()
        hit_contract = cached.contract_dict()
        self.assertEqual(
            fly_contract["loss_weights"]["tile_ownership_masking"],
            {"margin_m": scene.margin_m, "dilation_px": scene.dilation_px},
        )
        expected_sha = verify_tile_ownership_manifest(json.loads(scene.manifest.read_text(encoding="utf-8")))
        self.assertEqual(
            hit_contract["loss_weights"]["tile_ownership_masking"],
            {"margin_m": scene.margin_m, "dilation_px": scene.dilation_px, "cache_manifest_sha256": expected_sha},
        )
        without = copy.deepcopy(hit_contract)
        without["loss_weights"]["tile_ownership_masking"].pop("cache_manifest_sha256")
        self.assertEqual(without, fly_contract)

    def test_validation_refusals(self) -> None:
        scene = self.scene
        with self.assertRaisesRegex(ValueError, "provided together"):
            self._config(tile_ownership_cache_manifest=scene.manifest).validate()
        with self.assertRaisesRegex(ValueError, "requires tile_ownership_masking"):
            self._config(tile_ownership_masking=False, tile_ownership_cache_manifest=scene.manifest, tile_ownership_cache_root=scene.cache_root).validate()
        with self.assertRaisesRegex(ValueError, "margin/dilation"):
            self._config(tile_ownership_margin_m=scene.margin_m + 1.0, tile_ownership_cache_manifest=scene.manifest, tile_ownership_cache_root=scene.cache_root).validate()
        with self.assertRaisesRegex(ValueError, "different Tile"):
            self._config(mipmap_tile_id=2, tile_ownership_cache_manifest=scene.manifest, tile_ownership_cache_root=scene.cache_root).validate()
        with self.assertRaisesRegex(FileNotFoundError, "cache manifest is missing"):
            self._config(tile_ownership_cache_manifest=scene.cache_root / "nope.json", tile_ownership_cache_root=scene.cache_root).validate()
        other_dir = scene.root / "other"
        other_dir.mkdir()
        # Same Tile box, one extra view: a different signed Tile inputs manifest.
        other_inputs = _tile_inputs_manifest(
            other_dir, scene.box, [dict(CROP), dict(CROP, sample_id=f"{fixture.BASE_IMAGE_ID}::tilt")]
        )
        with self.assertRaisesRegex(ValueError, "different Tile inputs"):
            self._config(tile_inputs_manifest=other_inputs, tile_ownership_cache_manifest=scene.manifest, tile_ownership_cache_root=scene.cache_root).validate()


if __name__ == "__main__":
    unittest.main()
