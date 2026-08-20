from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cloudstudio_3dgs.data.manifest import build_manifest, write_manifest_atomic


def calibration_payload() -> dict:
    cameras = []
    for side, offset in (("left", 0.0), ("right", 0.1)):
        cameras.append(
            {
                "name": side,
                "type": "fisheye",
                "width": 2912,
                "height": 2912,
                "intrinsic": {
                    "fl_x": 788.0 + offset,
                    "fl_y": 789.0 + offset,
                    "cx": 1456.0,
                    "cy": 1456.0,
                },
                "distortion": {
                    "camera_model": "OPENCV_FISHEYE",
                    "params": {"k1": 0.08, "k2": -0.01, "k3": 0.0, "k4": 0.0},
                },
                "transform_from_lidar": {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "position": [offset, 0, 0],
                },
            }
        )
    return {"version": "v2", "cameras": cameras}


class DatasetManifestTests(unittest.TestCase):
    def create_fixture(self, root: Path, *, omit_right: bool = False) -> tuple[Path, Path]:
        recording = root / "客户 数据 scene"
        run = recording / "process" / "run 01"
        (recording / "info").mkdir(parents=True)
        (recording / "camera" / "left").mkdir(parents=True)
        (recording / "camera" / "right").mkdir(parents=True)
        run.mkdir(parents=True)
        (recording / "info" / "calibration.json").write_text(
            json.dumps(calibration_payload(), ensure_ascii=False), encoding="utf-8"
        )
        left_name = "1000000000000000000.jpg"
        right_name = "1000000000000000100.jpg"
        (recording / "camera" / "left" / left_name).write_bytes(b"left-image")
        if not omit_right:
            (recording / "camera" / "right" / right_name).write_bytes(b"right-image")
        (run / "ImgPose.txt").write_text(
            "index x y z roll pitch yaw qx qy qz qw timestamp\n"
            f"left/{left_name} 1 2 3 0 0 0 0 0 0 1 1.0\n"
            f"right/{right_name} 4 5 6 0 0 0 0 0 0 1 1.0\n",
            encoding="utf-8",
        )
        (run / "colorized.las").write_bytes(b"synthetic-las")
        return recording, run

    def test_manifest_is_deterministic_and_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recording, run = self.create_fixture(Path(temporary))
            first = build_manifest(recording, run)
            second = build_manifest(recording, run)

        self.assertEqual(first, second)
        self.assertEqual(first["recording_id"], "客户 数据 scene")
        self.assertEqual(len(first["cameras"]), 2)
        self.assertEqual(len(first["images"]), 2)
        self.assertEqual(first["rig_frames"], [])
        self.assertEqual(first["warnings"], ["rig_pairing_pending_pr02"])
        for image in first["images"]:
            self.assertFalse(Path(image["path"]).is_absolute())
            self.assertNotIn("\\", image["path"])
            self.assertIsNone(image["rig_frame_id"])
        self.assertEqual(first["images"][0]["c2w"][0][3], 1.0)

    def test_missing_image_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recording, run = self.create_fixture(Path(temporary), omit_right=True)
            with self.assertRaisesRegex(FileNotFoundError, "missing camera image right/"):
                build_manifest(recording, run)

    def test_skipped_hashes_are_persisted_as_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recording, run = self.create_fixture(Path(temporary))
            manifest = build_manifest(
                recording, run, hash_images=False, hash_point_cloud=False
            )

        self.assertIn("image_content_hashes_not_computed", manifest["warnings"])
        self.assertIn("point_cloud_content_hash_not_computed", manifest["warnings"])
        self.assertEqual(manifest["images"][0]["sha256"], "not_computed")
        self.assertEqual(manifest["point_cloud"]["sha256"], "not_computed")

    def test_unposed_raw_images_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recording, run = self.create_fixture(Path(temporary))
            extra = recording / "camera" / "left" / "1000000000000000200.jpg"
            extra.write_bytes(b"unposed-image")
            manifest = build_manifest(recording, run)

        self.assertEqual(
            manifest["unposed_images"],
            ["camera/left/1000000000000000200.jpg"],
        )
        self.assertIn("unposed_camera_images:1", manifest["warnings"])

    def test_atomic_writer_rejects_nonempty_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording, run = self.create_fixture(root)
            manifest = build_manifest(recording, run)
            output = root / "输出 manifest"
            output.mkdir()
            (output / "keep.txt").write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_manifest_atomic(manifest, output)
            destination = write_manifest_atomic(manifest, output, force=True)

            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "preserve")
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), manifest)
            self.assertEqual(list(output.glob(".*.tmp")), [])

    def test_ecef_point_cloud_is_never_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recording, run = self.create_fixture(Path(temporary))
            (run / "colorized.las").unlink()
            (run / "ecef_scene_colorized.las").write_bytes(b"unsafe-ecef")

            with self.assertRaisesRegex(FileNotFoundError, "local-coordinate"):
                build_manifest(recording, run)


if __name__ == "__main__":
    unittest.main()
