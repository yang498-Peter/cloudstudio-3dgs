"""DatasetBundle construction, capability derivation and manifest signing.

Synthetic fixtures only; torch-free and CPU-only.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio3dgs_sdk.ingest.bundle import (  # noqa: E402
    BUNDLE_SHA_KEY,
    CAPABILITY_FISHEYE,
    CAPABILITY_LIDAR,
    CAPABILITY_RIG,
    CAPABILITY_SPLIT,
    CAPABILITY_TIMESTAMPS,
    BundleImage,
    CameraIntrinsics,
    DatasetBundle,
    PointCloudRef,
    RigTransform,
    describe_capabilities,
    load_bundle_manifest,
    require_capabilities,
    verify_bundle_manifest,
    write_bundle_manifest,
)
from cloudstudio3dgs_sdk.ingest.errors import (  # noqa: E402
    BundleSignatureError,
    DatasetIncompleteError,
)

IDENTITY = tuple(
    tuple(1.0 if row == column else 0.0 for column in range(4)) for row in range(4)
)


def _rig(offset: float) -> RigTransform:
    return RigTransform(
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        position=(offset, 0.0, 0.0),
    )


def _camera(camera_id: str, *, model: str = "OPENCV_FISHEYE", rig: bool = True) -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_id=camera_id,
        width=64,
        height=64,
        fx=40.0,
        fy=40.0,
        cx=32.0,
        cy=32.0,
        camera_model=model,
        distortion={"k1": 0.01, "k2": 0.0, "k3": 0.0, "k4": 0.0},
        transform_from_lidar=_rig(0.05) if rig else None,
        side=camera_id,
    )


def _write_images(root: Path, names) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpeg-" + bytes([index % 251]))


def _bundle(
    root: Path,
    *,
    lidar: bool = True,
    timestamps: bool = True,
    rig: bool = True,
    split: bool = True,
) -> DatasetBundle:
    names = ["left/0.jpg", "left/1.jpg", "right/0.jpg", "right/1.jpg"]
    _write_images(root, names)
    cloud = None
    if lidar:
        cloud_path = root / "cloud.ply"
        cloud_path.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
        cloud = PointCloudRef(path=cloud_path, format="ply")
    images = []
    for index, name in enumerate(names):
        images.append(
            BundleImage(
                image_id=f"img_{index}",
                camera_id="left" if name.startswith("left") else "right",
                path=name,
                c2w=IDENTITY,
                timestamp_ns=1_000 + index if timestamps else None,
                rig_frame_id=f"rig_{index % 2}" if rig else None,
                split=("train" if index < 3 else "val") if split else None,
            )
        )
    return DatasetBundle(
        dataset_id="synthetic",
        adapter="unit_test",
        source_root=root,
        images_root=root,
        cameras=(_camera("left", rig=rig), _camera("right", rig=rig)),
        images=tuple(images),
        point_cloud=cloud,
        coordinate_frame="test_local",
    )


class BundleConstructionTest(unittest.TestCase):
    def test_capabilities_reflect_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = _bundle(root / "full")
            self.assertIn(CAPABILITY_LIDAR, full.capabilities)
            self.assertIn(CAPABILITY_TIMESTAMPS, full.capabilities)
            self.assertIn(CAPABILITY_RIG, full.capabilities)
            self.assertIn(CAPABILITY_SPLIT, full.capabilities)
            self.assertIn(CAPABILITY_FISHEYE, full.capabilities)

            bare = _bundle(
                root / "bare", lidar=False, timestamps=False, rig=False, split=False
            )
            self.assertNotIn(CAPABILITY_LIDAR, bare.capabilities)
            self.assertNotIn(CAPABILITY_TIMESTAMPS, bare.capabilities)
            self.assertNotIn(CAPABILITY_RIG, bare.capabilities)
            self.assertNotIn(CAPABILITY_SPLIT, bare.capabilities)

    def test_require_capabilities_names_what_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bare = _bundle(Path(tmp), lidar=False)
            with self.assertRaises(DatasetIncompleteError) as caught:
                require_capabilities(bare, [CAPABILITY_LIDAR], what="depth cache")
            self.assertIn(CAPABILITY_LIDAR, str(caught.exception))
            self.assertIn("depth cache", str(caught.exception))

    def test_describe_capabilities_lists_every_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lines = describe_capabilities(_bundle(Path(tmp)))
            self.assertEqual(len(lines), 6)
            self.assertTrue(any(line.startswith("[x] " + CAPABILITY_LIDAR) for line in lines))

    def test_rejects_sample_id_separator_in_image_id(self) -> None:
        with self.assertRaises(DatasetIncompleteError):
            BundleImage(image_id="a::b", camera_id="left", path="x.jpg", c2w=IDENTITY)

    def test_rejects_malformed_pose(self) -> None:
        with self.assertRaises(DatasetIncompleteError):
            BundleImage(
                image_id="a", camera_id="left", path="x.jpg", c2w=((1.0, 0.0), (0.0, 1.0))
            )

    def test_rejects_unknown_camera_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_images(root, ["a.jpg"])
            with self.assertRaises(DatasetIncompleteError) as caught:
                DatasetBundle(
                    dataset_id="x",
                    adapter="unit_test",
                    source_root=root,
                    images_root=root,
                    cameras=(_camera("left"),),
                    images=(
                        BundleImage(
                            image_id="a", camera_id="ghost", path="a.jpg", c2w=IDENTITY
                        ),
                    ),
                )
            self.assertIn("ghost", str(caught.exception))

    def test_rejects_duplicate_image_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_images(root, ["a.jpg", "b.jpg"])
            image = BundleImage(image_id="same", camera_id="left", path="a.jpg", c2w=IDENTITY)
            other = BundleImage(image_id="same", camera_id="left", path="b.jpg", c2w=IDENTITY)
            with self.assertRaises(DatasetIncompleteError):
                DatasetBundle(
                    dataset_id="x",
                    adapter="unit_test",
                    source_root=root,
                    images_root=root,
                    cameras=(_camera("left"),),
                    images=(image, other),
                )

    def test_rejects_degenerate_intrinsics(self) -> None:
        with self.assertRaises(DatasetIncompleteError):
            CameraIntrinsics(
                camera_id="left", width=0, height=64, fx=1.0, fy=1.0, cx=0.0, cy=0.0,
                camera_model="PINHOLE",
            )
        with self.assertRaises(DatasetIncompleteError):
            CameraIntrinsics(
                camera_id="left", width=64, height=64, fx=0.0, fy=1.0, cx=0.0, cy=0.0,
                camera_model="PINHOLE",
            )


class BundleManifestTest(unittest.TestCase):
    def test_sign_and_verify_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _bundle(root)
            path = write_bundle_manifest(bundle, root / "out")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                verify_bundle_manifest(payload), payload[BUNDLE_SHA_KEY]
            )
            self.assertEqual(len(payload["images"]), 4)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in payload["images"]))
            self.assertEqual(payload["point_cloud"]["format"], "ply")
            loaded = load_bundle_manifest(path, verify_artifacts=True)
            self.assertEqual(loaded[BUNDLE_SHA_KEY], payload[BUNDLE_SHA_KEY])

    def test_signature_is_stable_across_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp))
            self.assertEqual(
                bundle.to_manifest()[BUNDLE_SHA_KEY],
                bundle.to_manifest()[BUNDLE_SHA_KEY],
            )

    def test_tampered_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = _bundle(Path(tmp)).to_manifest()
            payload["coordinate_frame"] = "somewhere_else"
            with self.assertRaises(BundleSignatureError) as caught:
                verify_bundle_manifest(payload)
            self.assertIn("signature mismatch", str(caught.exception))

    def test_unsigned_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = _bundle(Path(tmp)).to_manifest()
            payload.pop(BUNDLE_SHA_KEY)
            with self.assertRaises(BundleSignatureError) as caught:
                verify_bundle_manifest(payload)
            self.assertIn("unsigned", str(caught.exception))

    def test_foreign_kind_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = _bundle(Path(tmp)).to_manifest()
            payload["kind"] = "something_else"
            # Re-sign so only the kind is wrong, not the signature.
            import hashlib

            from cloudstudio_3dgs.data.manifest import canonical_json_bytes

            unsigned = {k: v for k, v in payload.items() if k != BUNDLE_SHA_KEY}
            payload[BUNDLE_SHA_KEY] = hashlib.sha256(
                canonical_json_bytes(unsigned)
            ).hexdigest()
            with self.assertRaises(BundleSignatureError) as caught:
                verify_bundle_manifest(payload)
            self.assertIn("not a dataset bundle manifest", str(caught.exception))

    def test_artifact_verification_catches_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _bundle(root)
            payload = bundle.to_manifest()
            (root / "left" / "0.jpg").write_bytes(b"different")
            with self.assertRaises(BundleSignatureError) as caught:
                verify_bundle_manifest(payload, verify_artifacts=True)
            self.assertIn("content changed", str(caught.exception))

    def test_artifact_verification_catches_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _bundle(root)
            payload = bundle.to_manifest()
            (root / "left" / "1.jpg").unlink()
            with self.assertRaises(BundleSignatureError) as caught:
                verify_bundle_manifest(payload, verify_artifacts=True)
            self.assertIn("image is missing", str(caught.exception))

    def test_missing_image_fails_closed_at_manifest_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _bundle(root)
            (root / "right" / "0.jpg").unlink()
            with self.assertRaises(DatasetIncompleteError) as caught:
                bundle.to_manifest()
            self.assertIn("image file is missing", str(caught.exception))

    def test_write_refuses_to_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _bundle(root)
            write_bundle_manifest(bundle, root / "out")
            with self.assertRaises(FileExistsError):
                write_bundle_manifest(bundle, root / "out")
            write_bundle_manifest(bundle, root / "out", force=True)

    def test_skipping_hashes_drops_the_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = _bundle(Path(tmp)).to_manifest(hash_images=False)
            verify_bundle_manifest(payload)
            self.assertNotIn("image_content_hashes", payload["capabilities"])
            self.assertIn("image_content_hashes_not_computed", payload["warnings"])


if __name__ == "__main__":
    unittest.main()
