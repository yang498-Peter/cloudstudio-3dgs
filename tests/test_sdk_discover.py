"""Costing a dataset nobody has prepared.

Three things are under test and they are deliberately separate:

* the derivation itself - what :mod:`cloudstudio3dgs_sdk.discover` reads off a
  capture, which of its outputs are exact and which are guesses;
* the fallback - ``--dry-run`` and ``preflight`` reaching for the derivation
  when there is no prepare manifest, instead of refusing;
* the refusal that still stands - a real run will not train from an estimated
  summary, whether it derived one itself or was handed one.

CPU only, offline, no GPU: the capture is a pinhole folder with a small LAS
cloud written into a temporary directory, and every probe the preflight needs
is injected.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cloudstudio3dgs_sdk.__main__ import EXIT_OK, EXIT_REFUSED, main
from cloudstudio3dgs_sdk.discover import (
    DEFAULT_POINT_SAMPLE_BUDGET,
    DISCOVERY_RULE_VERSION,
    MINIMUM_FACE_RECTANGLE_PIXELS,
    DiscoveryError,
    estimate_dataset_summary,
    face_plan,
    load_capture,
)
from cloudstudio3dgs_sdk.ingest.bundle import BundleImage, CameraIntrinsics, DatasetBundle
from cloudstudio3dgs_sdk.plan import DatasetSummary, build_plan
from cloudstudio3dgs_sdk.profile import PROFILE_B5FILL2
from cloudstudio3dgs_sdk.project import Project, StageRefused
from cloudstudio3dgs_sdk.requirements import PASS, WARN, GpuInfo, Probes, preflight
from tests.test_ingest_adapters import write_las
from tests.test_sdk_plan import make_repo, two_tile_dataset

# A camera looking straight down, OpenCV axes: camera +z -> world -z,
# camera +y -> world -y. det = +1, so it is a rotation and not a mirror.
LOOKING_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
# Looking straight up, away from a cloud that lies below.
LOOKING_UP = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

CAMERA_HEIGHT_M = 10.0
# 640x480 at f=500 sees +-6.4 m across and +-4.8 m down the strip from that
# height, so a camera over one tile cannot see the far end of the scene and
# the per-tile view counts actually differ.
FOCAL_PX = 500.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def scene_points(seed: int = 7, count: int = 40_000) -> np.ndarray:
    """A 40 m strip: long in x, so the slab rule cuts x, and tall enough in y
    that a tile's projection clears the minimum rectangle."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(0.0, 40.0, count),
            rng.uniform(-8.0, 8.0, count),
            rng.uniform(0.0, 4.0, count),
        ]
    )


def las_points(path: Path) -> np.ndarray:
    """The cloud as the file holds it, after LAS scale quantization."""
    import laspy

    with laspy.open(path) as reader:
        data = reader.read()
        return np.column_stack([data.x, data.y, data.z]).astype(np.float64)


def make_capture(
    root: Path,
    *,
    rotation: np.ndarray = LOOKING_DOWN,
    splits: bool = False,
    with_cloud: bool = True,
    points: np.ndarray | None = None,
) -> Path:
    """A pinhole-folder capture: poses.json, images, and a LAS cloud."""
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, x in enumerate((5.0, 15.0, 25.0, 35.0)):
        name = f"images/{index:04d}.png"
        (root / name).write_bytes(b"png" + bytes([index]))
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = (x, 0.0, CAMERA_HEIGHT_M)
        row = {
            "file_path": name,
            "camera_id": "cam0",
            "transform_matrix": matrix.tolist(),
            "timestamp_ns": 1000 + index,
        }
        if splits:
            row["split"] = "val" if index == 3 else "train"
        frames.append(row)
    payload = {
        "coordinate_frame": "site_local",
        "pose_convention": "c2w_opencv",
        "cameras": [
            {
                "camera_id": "cam0",
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "fx": FOCAL_PX,
                "fy": FOCAL_PX,
                "cx": IMAGE_WIDTH / 2.0,
                "cy": IMAGE_HEIGHT / 2.0,
                "model": "PINHOLE",
            }
        ],
        "frames": frames,
    }
    (root / "poses.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    if with_cloud:
        write_las(root / "cloud.las", scene_points() if points is None else points)
    return root


def probes(**overrides) -> Probes:
    fields = {
        "python_version": "3.12",
        "torch_version": "2.11.0+cu128",
        "gpu": GpuInfo(True, "stub card", 24.0),
        "free_disk_bytes": lambda path: 10 ** 15,
        "gsplat_extension_sha256": "ab" * 32,
        "external_asset_locator": lambda asset: (True, "stubbed"),
    }
    fields.update(overrides)
    return Probes(**fields)


class CaptureFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dataset = make_capture(self.root / "capture")
        self.work = self.root / "work"
        self.repo = make_repo(self.root / "repo")

    def estimate(self, **kwargs):
        return estimate_dataset_summary(self.dataset, PROFILE_B5FILL2, **kwargs)

    def project(self, **kwargs) -> Project:
        fields = {
            "repo_root": self.repo,
            "python": Path("python.exe"),
            "probes": probes(),
            "stream": io.StringIO(),
        }
        fields.update(kwargs)
        return Project(self.dataset, self.work, PROFILE_B5FILL2, **fields)


# --------------------------------------------------------------------------
# The derivation
# --------------------------------------------------------------------------


class EstimateTests(CaptureFixture):
    def test_the_summary_is_marked_estimated_and_says_why_field_by_field(self) -> None:
        summary = self.estimate().summary
        self.assertTrue(summary.estimated)
        joined = " ".join(summary.estimate_notes)
        for field in (
            "tile boxes",
            "init_point_count",
            "view_count",
            "train_view_count",
            "global_init_point_count",
        ):
            self.assertIn(field, joined)
        self.assertIn(DISCOVERY_RULE_VERSION, joined)

    def test_the_tile_count_comes_from_the_profile(self) -> None:
        summary = self.estimate().summary
        self.assertEqual(summary.tile_count, int(PROFILE_B5FILL2.tiling["reference_tile_count"]))
        self.assertEqual([tile.tile_id for tile in summary.tiles], [0, 1, 2, 3])
        self.assertEqual([tile.name for tile in summary.tiles], [f"Tile_{i}" for i in range(4)])

    def test_init_point_count_is_the_exact_count_inside_the_tile_box(self) -> None:
        """The one per-tile number that is a measurement, not a guess.

        Recounted here straight from the plan's boxes, because "exact" is the
        claim the module docstring makes and the claim the report rests on.
        The points are re-read from the LAS rather than taken from the array
        that was written: LAS stores scaled integers, so a point within half a
        scale unit of a box face is on a different side of it in the file than
        it was in memory, and the file is what the claim is about.
        """
        estimate = self.estimate()
        points = las_points(self.dataset / "cloud.las")
        for tile, entry in zip(estimate.summary.tiles, estimate.tile_plan["tiles"]):
            box = np.asarray(entry["training_and_export_box"], dtype=np.float64)
            inside = int(
                np.count_nonzero(np.all((points >= box[0]) & (points <= box[1]), axis=1))
            )
            self.assertEqual(tile.init_point_count, inside, entry["name"])

    def test_the_tiles_cover_the_cloud_and_overlap_only_in_the_halo(self) -> None:
        summary = self.estimate().summary
        total = sum(tile.init_point_count for tile in summary.tiles)
        self.assertEqual(summary.lidar_point_count, len(scene_points()))
        self.assertGreaterEqual(total, summary.lidar_point_count)
        self.assertLess(total, summary.lidar_point_count * 1.05)

    def test_every_tile_is_seen_by_at_least_one_face(self) -> None:
        summary = self.estimate().summary
        for tile in summary.tiles:
            self.assertGreater(tile.view_count, 0, tile.name)
        self.assertLessEqual(
            max(tile.view_count for tile in summary.tiles), summary.train_view_count
        )

    def test_cameras_facing_away_see_nothing(self) -> None:
        """The rule is geometric, not a headcount: turn the rig over and the
        per-tile view counts go to zero even though the images are still there."""
        away = make_capture(self.root / "away", rotation=LOOKING_UP)
        summary = estimate_dataset_summary(away, PROFILE_B5FILL2).summary
        self.assertEqual([tile.view_count for tile in summary.tiles], [0, 0, 0, 0])
        self.assertEqual(summary.train_view_count, 4)

    def test_a_declared_split_is_honoured_and_its_absence_is_reported(self) -> None:
        split = make_capture(self.root / "split", splits=True)
        with_split = estimate_dataset_summary(split, PROFILE_B5FILL2).summary
        self.assertEqual(with_split.train_view_count, 3)
        self.assertIn("declares a train split", " ".join(with_split.estimate_notes))

        without = self.estimate().summary
        self.assertEqual(without.train_view_count, 4)
        self.assertIn("declares no split", " ".join(without.estimate_notes))

    def test_global_init_point_count_is_declared_an_upper_bound(self) -> None:
        summary = self.estimate().summary
        self.assertEqual(summary.global_init_point_count, summary.lidar_point_count)
        self.assertIn("UPPER BOUND", " ".join(summary.estimate_notes))

    def test_a_capture_without_a_cloud_is_refused_by_name(self) -> None:
        bare = make_capture(self.root / "bare", with_cloud=False)
        with self.assertRaises(DiscoveryError) as caught:
            estimate_dataset_summary(bare, PROFILE_B5FILL2)
        self.assertIn("LiDAR", str(caught.exception))

    def test_the_estimate_is_reproducible(self) -> None:
        self.assertEqual(self.estimate().summary, self.estimate().summary)

    def test_the_sample_budget_moves_the_view_counts_but_never_the_point_counts(self) -> None:
        """The budget is a cost knob over the *view* rule only.

        init_point_count is counted over the whole stream and must be
        bit-identical at any budget; if it ever moved, "exact" would be a lie.
        The view counts may move, which is why the budget must not be tuned to
        make a number match.
        """
        dense = self.estimate(point_sample_budget=DEFAULT_POINT_SAMPLE_BUDGET * 4)
        sparse = self.estimate(point_sample_budget=200)
        self.assertEqual(
            [tile.init_point_count for tile in dense.summary.tiles],
            [tile.init_point_count for tile in sparse.summary.tiles],
        )
        self.assertEqual(dense.summary.lidar_point_count, sparse.summary.lidar_point_count)
        self.assertEqual(
            [tile.name for tile in dense.summary.tiles],
            [tile.name for tile in sparse.summary.tiles],
        )

    def test_the_estimate_renders_its_measurements_and_its_caveats(self) -> None:
        text = self.estimate().render()
        self.assertIn("ESTIMATED", text)
        self.assertIn("lidar_point_count", text)
        self.assertIn(DISCOVERY_RULE_VERSION, text)

    def test_the_estimate_serialises(self) -> None:
        payload = self.estimate().as_json()
        restored = DatasetSummary.from_json(payload["dataset"])
        self.assertTrue(restored.estimated)
        self.assertEqual(restored, self.estimate().summary)


class FacePlanTests(unittest.TestCase):
    """view_count counts the views the recipe trains on, not raw images."""

    @staticmethod
    def bundle_with(camera: CameraIntrinsics) -> DatasetBundle:
        return DatasetBundle(
            dataset_id="synthetic",
            adapter="test",
            source_root=Path("."),
            images_root=Path("."),
            cameras=(camera,),
            images=(
                BundleImage(
                    image_id="a",
                    camera_id=camera.camera_id,
                    path="a.png",
                    c2w=tuple(tuple(row) for row in np.eye(4).tolist()),
                ),
            ),
        )

    def test_a_square_fisheye_becomes_the_four_face4_views(self) -> None:
        camera = CameraIntrinsics(
            camera_id="left",
            width=2912,
            height=2912,
            fx=1000.0,
            fy=1000.0,
            cx=1456.0,
            cy=1456.0,
            camera_model="OPENCV_FISHEYE",
        )
        faces = face_plan(self.bundle_with(camera))["left"]
        self.assertEqual(len(faces), 4)
        self.assertEqual(
            sorted(face.face_id for face in faces),
            ["pitch_down_56", "pitch_up_56", "yaw_minus_35", "yaw_plus_35"],
        )

    def test_a_pinhole_camera_stays_one_view_per_image(self) -> None:
        camera = CameraIntrinsics(
            camera_id="cam0",
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            fx=FOCAL_PX,
            fy=FOCAL_PX,
            cx=320.0,
            cy=240.0,
            camera_model="PINHOLE",
        )
        faces = face_plan(self.bundle_with(camera))["cam0"]
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].face_id, "whole_image")
        self.assertEqual(faces[0].width, IMAGE_WIDTH)


class CaptureLoadTests(CaptureFixture):
    def test_planning_does_not_hash_the_capture(self) -> None:
        """A cost estimate binds nothing, so it must not pay to hash 884
        images; the adapter's own capability set is what proves it skipped."""
        bundle = load_capture(self.dataset)
        self.assertEqual(bundle.adapter, "pinhole_folder")
        self.assertTrue(all(image.sha256 is None for image in bundle.images))


# --------------------------------------------------------------------------
# The plan says which numbers it guessed
# --------------------------------------------------------------------------


class LabellingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = make_repo(self.root / "repo")

    def plan_for(self, dataset):
        return build_plan(
            PROFILE_B5FILL2,
            dataset,
            dataset_root=self.root / "dataset",
            work_root=self.root / "work",
            repo_root=self.repo,
            python=Path("python.exe"),
            prior_tile_checkpoints={0: "a.pt", 1: "b.pt"},
        )

    def test_an_estimated_plan_is_labelled_in_the_transcript(self) -> None:
        dataset = two_tile_dataset(
            estimated=True, estimate_notes=("view_count: guessed from the poses",)
        )
        text = self.plan_for(dataset).render()
        self.assertIn("[ESTIMATED]", text)
        self.assertIn("ESTIMATED DATASET", text)
        self.assertIn("A real run refuses this summary", text)
        self.assertIn("view_count: guessed from the poses", text)

    def test_a_prepared_plan_says_nothing_about_estimates(self) -> None:
        text = self.plan_for(two_tile_dataset()).render()
        self.assertNotIn("ESTIMATED", text)

    def test_an_estimated_plan_carries_a_warning_and_the_flag_in_json(self) -> None:
        plan = self.plan_for(two_tile_dataset(estimated=True))
        self.assertTrue(any("ESTIMATED" in warning for warning in plan.warnings))
        self.assertTrue(plan.as_json()["dataset"]["estimated"])

    def test_the_flag_survives_a_json_round_trip(self) -> None:
        """An estimate written out and fed back through --summary is still an
        estimate; if the flag were dropped the refusal could be edited away."""
        dataset = two_tile_dataset(estimated=True, estimate_notes=("a note",))
        restored = DatasetSummary.from_json(json.loads(json.dumps(dataset.as_json())))
        self.assertEqual(restored, dataset)

    def test_preflight_warns_about_an_estimated_summary_without_failing(self) -> None:
        report = preflight(
            self.plan_for(two_tile_dataset(estimated=True)),
            PROFILE_B5FILL2,
            repo_root=self.repo,
            probes=probes(),
        )
        self.assertEqual(report.get("dataset_summary_source").status, WARN)
        self.assertIn("ESTIMATED", report.get("dataset_summary_source").detail)
        self.assertTrue(report.ok, report.render())

    def test_preflight_says_so_when_the_numbers_were_prepared(self) -> None:
        report = preflight(
            self.plan_for(two_tile_dataset()),
            PROFILE_B5FILL2,
            repo_root=self.repo,
            probes=probes(),
        )
        check = report.get("dataset_summary_source")
        self.assertEqual(check.status, PASS)
        self.assertIn("prepare()", check.detail)


# --------------------------------------------------------------------------
# The fallback, and the refusal that survives it
# --------------------------------------------------------------------------


class FallbackTests(CaptureFixture):
    def test_planning_falls_back_instead_of_refusing(self) -> None:
        plan = self.project().plan(allow_estimate=True)
        self.assertTrue(plan.dataset.estimated)
        self.assertEqual(plan.dataset.tile_count, 4)

    def test_planning_without_the_flag_still_refuses(self) -> None:
        with self.assertRaises(StageRefused) as caught:
            self.project().plan()
        self.assertIn("no prepared dataset", str(caught.exception))

    def test_preflight_falls_back_too(self) -> None:
        report = self.project().preflight(require_gpu=True, allow_estimate=True)
        self.assertEqual(report.get("dataset_summary_source").status, WARN)

    def test_an_estimated_plan_is_never_cached_for_the_run_path(self) -> None:
        """A dry run earlier in the process must not leave a summary behind
        that a later run picks up."""
        project = self.project()
        project.plan(allow_estimate=True)
        with self.assertRaises(StageRefused):
            project.plan()
        with self.assertRaises(StageRefused):
            project.run_all(stages=("prepare",))

    def test_a_real_run_ingests_a_capture_with_no_prepare_manifest(self) -> None:
        """A real run never plans from an estimate: with no prepare manifest it ingests the
        capture, and this synthetic one (a cloud and images, no calibration) is refused with
        ingestion's own reason. Before 2026-10-06 it was refused outright, which left a fresh
        capture no way in at all."""
        project = self.project()
        with self.assertRaises(StageRefused) as caught:
            project.run_all(stages=("prepare",))
        self.assertIn("ingestion refused", str(caught.exception))
        # the step that could not read the capture is named (its log says why)
        self.assertIn("dataset_manifest failed", str(caught.exception))
        self.assertFalse(project.layout.prepare_manifest.is_file())

    def test_a_real_run_refuses_an_estimated_summary_handed_to_it(self) -> None:
        """The estimate cannot be smuggled onto the run path by injecting it."""
        summary = self.estimate().summary
        project = self.project(dataset=summary)
        with self.assertRaises(StageRefused) as caught:
            project.run_all(stages=("prepare",))
        self.assertIn("estimated dataset summary", str(caught.exception))


class CliFallbackTests(CaptureFixture):
    def run_cli(self, *argv: str) -> tuple[int, str]:
        stream = io.StringIO()
        code = main(list(argv), stream=stream)
        return code, stream.getvalue()

    def base(self) -> list[str]:
        return [
            "--dataset", str(self.dataset),
            "--work", str(self.work),
            "--profile", "b5fill2",
            "--repo-root", str(self.repo),
            "--python", "python.exe",
        ]

    def test_dry_run_costs_an_unprepared_capture_and_labels_the_estimates(self) -> None:
        code, text = self.run_cli("run", *self.base(), "--dry-run")
        self.assertEqual(code, EXIT_OK, text)
        self.assertIn("ESTIMATED DATASET", text)
        self.assertIn(str(MINIMUM_FACE_RECTANGLE_PIXELS), text)
        for stage in ("[prepare]", "[train]", "[deliver]", "[report]"):
            self.assertIn(stage, text)

    def test_dry_run_json_records_the_flag(self) -> None:
        out = self.root / "plan.json"
        code, _ = self.run_cli("run", *self.base(), "--dry-run", "--plan-json", str(out))
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(payload["dataset"]["estimated"])
        self.assertTrue(payload["dataset"]["estimate_notes"])

    def test_preflight_runs_on_an_unprepared_capture(self) -> None:
        code, text = self.run_cli("preflight", *self.base(), "--no-gpu")
        self.assertIn("dataset_summary_source", text)
        self.assertIn(code, (EXIT_OK, EXIT_REFUSED))

    def test_a_real_run_exits_refused(self) -> None:
        code, _ = self.run_cli("run", *self.base(), "--stages", "prepare")
        self.assertEqual(code, EXIT_REFUSED)

    def test_a_real_run_refuses_an_estimated_summary_file(self) -> None:
        path = self.root / "estimated_summary.json"
        path.write_text(json.dumps(self.estimate().summary.as_json()), encoding="utf-8")
        code, _ = self.run_cli(
            "run", *self.base(), "--stages", "prepare", "--summary", str(path)
        )
        self.assertEqual(code, EXIT_REFUSED)


if __name__ == "__main__":
    unittest.main()
