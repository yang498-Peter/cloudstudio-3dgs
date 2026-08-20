from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cloudstudio_3dgs.ba.pycolmap_adapter import (
    apply_fixed_stereo_rig,
    build_training_reference_model,
    reconstruction_snapshot,
    run_bundle_adjustment_stage,
)
from cloudstudio_3dgs.ba.report import build_ba_report
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from tools import run_rig_ba


def synthetic_problem():
    import pycolmap

    frame_count = 6
    baseline_m = 0.1
    points = np.asarray(
        [[x, y, z] for x in np.linspace(-0.5, 2.0, 6) for y in (-0.4, 0.0, 0.4) for z in (4.0, 5.0)],
        dtype=np.float64,
    )
    reconstruction = pycolmap.Reconstruction()
    cameras = {}
    for camera_id, side in ((1, "left"), (2, "right")):
        camera = pycolmap.Camera.create_from_model_name(
            camera_id, "OPENCV_FISHEYE", 800.0, 800, 800
        )
        camera.params = np.asarray([800.0, 800.0, 400.0, 400.0, 0.0, 0.0, 0.0, 0.0])
        reconstruction.add_camera_with_trivial_rig(camera)
        cameras[side] = camera
    dataset_images = []
    rig_frames = []
    image_id = 1
    for frame_index in range(frame_count):
        center_x = frame_index * 0.3
        center_y = 0.12 * np.sin(frame_index * 0.9)
        angle = 0.0 if frame_index == 0 else 0.012 * np.sin(frame_index)
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        rig_center = np.asarray([center_x, center_y, 0.0])
        frame_image_ids = []
        for side, offset in (("left", 0.0), ("right", baseline_m)):
            center = rig_center + rotation.T @ np.asarray([offset, 0.0, 0.0])
            observed = []
            for point_index, point in enumerate(points):
                camera_point = point - center
                pixel = cameras[side].img_from_cam(camera_point)
                noise = np.asarray(
                    [0.12 * np.sin(point_index), 0.12 * np.cos(point_index)],
                    dtype=np.float64,
                )
                observed.append(np.asarray(pixel) + noise)
            name = f"{side}/{frame_index:03d}.jpg"
            image = pycolmap.Image(
                name=name,
                keypoints=np.asarray(observed),
                camera_id=1 if side == "left" else 2,
                image_id=image_id,
            )
            cam_from_world = np.column_stack((rotation, -rotation @ center))
            reconstruction.add_image_with_trivial_frame(
                image, pycolmap.Rigid3d(cam_from_world)
            )
            c2w = np.eye(4)
            c2w[:3, :3] = rotation.T
            c2w[:3, 3] = center
            stable_id = f"img_{side}_{frame_index:03d}"
            dataset_images.append(
                {
                    "image_id": stable_id,
                    "side": side,
                    "path": f"camera/{name}",
                    "c2w": c2w.tolist(),
                }
            )
            frame_image_ids.append(stable_id)
            image_id += 1
        rig_frames.append(
            {
                "rig_frame_id": f"rig_{frame_index:03d}",
                "timestamp_ns": 1_000_000_000 + frame_index * 100_000_000,
                "left_image_id": frame_image_ids[0],
                "right_image_id": frame_image_ids[1],
                "image_ids": frame_image_ids,
            }
        )
    for point_index, point in enumerate(points):
        track = pycolmap.Track()
        for current_image_id in range(1, image_id):
            track.add_element(current_image_id, point_index)
        reconstruction.add_point3D(point, track)
    expected = np.eye(4)
    expected[0, 3] = baseline_m
    dataset = {
        "images": dataset_images,
        "rig_frames": rig_frames,
        "rig": {"expected_right_to_left": expected.tolist()},
    }
    return reconstruction, dataset


@unittest.skipUnless(
    importlib.util.find_spec("pycolmap"),
    "optional pycolmap runtime is not installed",
)
class PycolmapRigBaTests(unittest.TestCase):
    def test_rig_ba_cli_publishes_checked_synthetic_candidate(self) -> None:
        reconstruction, dataset = synthetic_problem()
        dataset["manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(dataset)
        ).hexdigest()
        pairs = []
        for frame in dataset["rig_frames"]:
            first, second = sorted(frame["image_ids"])
            records = {image["image_id"]: image for image in dataset["images"]}
            pairs.append(
                {
                    "image_id_a": first,
                    "image_id_b": second,
                    "path_a": records[first]["path"],
                    "path_b": records[second]["path"],
                    "reasons": ["stereo"],
                    "loop_distance_m": None,
                }
            )
        graph = {
            "schema_version": 1,
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "split_image_ids": {
                "train": sorted(image["image_id"] for image in dataset["images"]),
                "val": ["validation_sentinel"],
            },
            "pairs": pairs,
            "summary": {
                "training_images": len(dataset["images"]),
                "validation_images_used": 0,
            },
        }
        graph["match_graph_sha256"] = hashlib.sha256(
            canonical_json_bytes(graph)
        ).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            reconstruction.write(model)
            triangulation_manifest = {
                "schema_version": 1,
                "inputs": {"pairs_sha256": run_rig_ba.hloc_pairs_sha256(graph)},
                "output": {
                    "sfm_model_sha256": run_rig_ba.directory_sha256(model)
                },
            }
            triangulation_manifest["triangulation_manifest_sha256"] = hashlib.sha256(
                canonical_json_bytes(triangulation_manifest)
            ).hexdigest()
            manifest_path = root / "dataset.json"
            graph_path = root / "graph.json"
            triangulation_path = root / "triangulation_runtime_manifest.json"
            manifest_path.write_text(json.dumps(dataset), encoding="utf-8")
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            triangulation_path.write_text(
                json.dumps(triangulation_manifest), encoding="utf-8"
            )
            output = root / "ba"
            argv = [
                "run_rig_ba.py",
                "--model",
                str(model),
                "--manifest",
                str(manifest_path),
                "--match-graph",
                str(graph_path),
                "--output",
                str(output),
            ]

            with patch("sys.argv", argv):
                exit_code = run_rig_ba.main()

            report = json.loads(
                (output / "report" / "ba_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(report["candidate_accepted"])
            self.assertEqual(report["position_prior"]["image_count"], 12)

    def test_training_reference_contains_only_requested_known_pose_images(self) -> None:
        reconstruction, _dataset = synthetic_problem()
        names = {"left/000.jpg", "right/000.jpg", "left/003.jpg"}

        reference = build_training_reference_model(reconstruction, names)

        self.assertEqual(reference.num_images(), 3)
        self.assertEqual(
            {image.name for image in reference.images.values()}, names
        )
        self.assertTrue(all(image.has_pose for image in reference.images.values()))

    def test_snapshot_reprojection_errors_respect_selected_training_images(self) -> None:
        reconstruction, dataset = synthetic_problem()
        apply_fixed_stereo_rig(reconstruction, dataset)
        selected = {"img_left_000", "img_right_000"}

        snapshot = reconstruction_snapshot(
            reconstruction,
            dataset,
            model_sha256="a" * 64,
            solver_success=False,
            included_image_ids=selected,
        )

        expected_observations = 2 * reconstruction.num_points3D()
        self.assertEqual(len(snapshot["reprojection_errors_px"]), expected_observations)
        self.assertEqual(len(snapshot["rig_frames"]), 1)

    def test_fixed_rig_accepts_a_complete_training_only_model(self) -> None:
        reconstruction, dataset = synthetic_problem()
        selected = {
            image_id
            for frame in dataset["rig_frames"][:3]
            for image_id in frame["image_ids"]
        }
        names = {
            image["path"].removeprefix("camera/")
            for image in dataset["images"]
            if image["image_id"] in selected
        }
        training_reference = build_training_reference_model(reconstruction, names)

        apply_fixed_stereo_rig(
            training_reference, dataset, included_image_ids=selected
        )

        self.assertEqual(training_reference.num_images(), 6)
        self.assertEqual(training_reference.num_frames(), 3)
        self.assertEqual(training_reference.num_rigs(), 1)

    def test_real_pycolmap_solver_improves_synthetic_fixed_rig(self) -> None:
        reconstruction, dataset = synthetic_problem()
        apply_fixed_stereo_rig(reconstruction, dataset)
        before = reconstruction_snapshot(
            reconstruction,
            dataset,
            model_sha256="a" * 64,
            solver_success=False,
        )
        position_priors = {
            image.image_id: np.asarray(image.projection_center(), dtype=np.float64)
            for image in reconstruction.images.values()
        }
        summary = run_bundle_adjustment_stage(
            reconstruction,
            "stage_1",
            position_priors_by_image_id=position_priors,
        )
        after = reconstruction_snapshot(
            reconstruction,
            dataset,
            model_sha256="b" * 64,
            solver_success=summary.is_solution_usable(),
        )
        report = build_ba_report(before, after, stage="stage_1")

        self.assertTrue(summary.is_solution_usable(), summary.brief_report())
        self.assertTrue(report["candidate_accepted"])
        self.assertEqual(report["gates"]["rig_baseline_fixed"]["status"], "PASS")
        self.assertEqual(report["gates"]["scene_scale_fixed"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
