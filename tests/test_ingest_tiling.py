"""The slab tiling rule: determinism, balance, and schema acceptance.

Schema acceptance is checked against the trainer's own validators -
``verify_adaptive_tile_plan``, ``verify_tile_inputs_manifest`` and the
gap-free-partition rule in ``cloudstudio_3dgs.training.tile_ownership`` - all of
which are torch-free, so this runs on CPU with no CUDA context.

The house0305 comparison at the bottom is skipped unless the real LAS is on
this machine; it is a regression guard, not a unit test.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.pipeline.adaptive_tiling import (  # noqa: E402
    ProjectedObservationTable,
    verify_adaptive_tile_plan,
)
from cloudstudio_3dgs.training.tile_inputs import (  # noqa: E402
    materialize_lidar_tile_inputs,
    verify_tile_inputs_manifest,
)
from cloudstudio_3dgs.training.tile_ownership import _ordered_boxes  # noqa: E402
from cloudstudio3dgs_sdk.ingest import tiling  # noqa: E402
from cloudstudio3dgs_sdk.ingest.tiling import (  # noqa: E402
    TilingError,
    TilingRule,
    build_slab_tile_plan,
    exact_slab_counts,
    histogram_from_las,
    histogram_from_points,
    slab_split,
)

HOUSE0305_LAS = Path(r"C:\Peter\testdata\S1\house0305\colorized.las")
HOUSE0305_TILE_INPUTS = Path(
    r"C:\Peter\3dgs-runs\house0305_sop\tile_inputs_v9\tile_inputs_manifest.json"
)


def _skewed_cloud(seed: int = 11, count: int = 20_000) -> np.ndarray:
    """A long-in-X scene with two dense clusters, so equal-length != equal-count."""

    rng = np.random.default_rng(seed)
    dense = rng.normal([-18.0, 0.0, 1.0], [1.5, 4.0, 1.0], size=(count // 2, 3))
    spread = rng.uniform([-30.0, -8.0, -1.0], [30.0, 8.0, 4.0], size=(count // 2, 3))
    return np.concatenate([dense, spread], axis=0)


def _write_las(path: Path, points: np.ndarray) -> Path:
    import laspy

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    data = laspy.LasData(header)
    data.x, data.y, data.z = points[:, 0], points[:, 1], points[:, 2]
    data.red = np.full(len(points), 10, dtype=np.uint16)
    data.green = np.full(len(points), 20, dtype=np.uint16)
    data.blue = np.full(len(points), 30, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write(str(path))
    return path


def _observation_table(points: np.ndarray, *, images: int = 6) -> ProjectedObservationTable:
    """Every point seen by every image at a deterministic pixel."""

    rng = np.random.default_rng(5)
    size = np.array([640, 480], dtype=np.int64)
    observation_point = np.tile(np.arange(len(points), dtype=np.int64), images)
    observation_image = np.repeat(np.arange(images, dtype=np.int64), len(points))
    observation_xy = rng.uniform(
        [0.0, 0.0], size.astype(np.float64), size=(len(observation_point), 2)
    )
    return ProjectedObservationTable(
        points=points,
        observation_xy=observation_xy,
        observation_image=observation_image,
        observation_point=observation_point,
        image_sizes=np.tile(size, (images, 1)),
    ).validated()


class SlabRuleTest(unittest.TestCase):
    def test_split_axis_is_the_longer_horizontal_axis(self) -> None:
        points = _skewed_cloud()
        slab = slab_split(histogram_from_points(points), TilingRule(tile_count=4))
        self.assertEqual(slab.axis_name, "x")
        rotated = points[:, [1, 0, 2]]
        self.assertEqual(
            slab_split(histogram_from_points(rotated), TilingRule(tile_count=4)).axis_name,
            "y",
        )

    def test_explicit_axis_overrides_the_choice(self) -> None:
        slab = slab_split(
            histogram_from_points(_skewed_cloud()), TilingRule(tile_count=3, axis="y")
        )
        self.assertEqual(slab.axis_name, "y")

    def test_z_is_not_a_legal_axis(self) -> None:
        with self.assertRaises(TilingError):
            TilingRule(axis="z").validate()

    def test_slabs_have_near_equal_point_counts(self) -> None:
        points = _skewed_cloud()
        slab = slab_split(histogram_from_points(points), TilingRule(tile_count=4))
        counts = exact_slab_counts(points, slab.core_boxes)
        self.assertEqual(sum(counts), len(points))
        self.assertLess(max(counts) / min(counts), 1.05)
        # Equal-length slabs over the same cloud are much worse; that is the point.
        low = float(slab.root_box.minimum[slab.axis])
        high = float(slab.root_box.maximum[slab.axis])
        edges = np.linspace(low, high, 5)
        uniform = []
        for index in range(4):
            inside = (points[:, slab.axis] >= edges[index]) & (
                points[:, slab.axis] <= edges[index + 1]
            )
            uniform.append(int(np.count_nonzero(inside)))
        self.assertGreater(max(uniform) / max(min(uniform), 1), 2.0)

    def test_cuts_are_deterministic_across_chunk_orders(self) -> None:
        points = _skewed_cloud()
        rule = TilingRule(tile_count=5)
        first = slab_split(histogram_from_points(points), rule)
        shuffled = points[np.random.default_rng(99).permutation(len(points))]
        second = slab_split(histogram_from_points(shuffled), rule)
        self.assertEqual(first.cuts, second.cuts)
        self.assertEqual(first.histogram_counts, second.histogram_counts)

    def test_las_and_in_memory_histograms_agree(self) -> None:
        points = _skewed_cloud(count=4_000)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_las(Path(tmp) / "cloud.las", points)
            from_las = slab_split(histogram_from_las(path), TilingRule(tile_count=4))
            from_memory = slab_split(histogram_from_points(points), TilingRule(tile_count=4))
            for a, b in zip(from_las.cuts, from_memory.cuts):
                # LAS quantizes to the 1 mm scale factor declared above.
                self.assertAlmostEqual(a, b, places=2)

    def test_core_boxes_partition_the_root_box(self) -> None:
        slab = slab_split(histogram_from_points(_skewed_cloud()), TilingRule(tile_count=4))
        axis = slab.axis
        self.assertAlmostEqual(
            float(slab.core_boxes[0].minimum[axis]), float(slab.root_box.minimum[axis])
        )
        self.assertAlmostEqual(
            float(slab.core_boxes[-1].maximum[axis]), float(slab.root_box.maximum[axis])
        )
        for left, right in zip(slab.core_boxes, slab.core_boxes[1:]):
            self.assertAlmostEqual(
                float(left.maximum[axis]), float(right.minimum[axis])
            )

    def test_export_boxes_overlap_and_core_boxes_do_not(self) -> None:
        slab = slab_split(histogram_from_points(_skewed_cloud()), TilingRule(tile_count=4))
        axis = slab.axis
        for left, right in zip(slab.export_boxes, slab.export_boxes[1:]):
            self.assertGreater(float(left.maximum[axis]), float(right.minimum[axis]))

    def test_absolute_overlap_margin_is_honoured(self) -> None:
        rule = TilingRule(tile_count=3, overlap_margin_m=0.75)
        slab = slab_split(histogram_from_points(_skewed_cloud()), rule)
        for core, export in zip(slab.core_boxes, slab.export_boxes):
            np.testing.assert_allclose(core.minimum - export.minimum, 0.75)
            np.testing.assert_allclose(export.maximum - core.maximum, 0.75)

    def test_too_many_tiles_fails_closed(self) -> None:
        points = _skewed_cloud()
        with self.assertRaises(TilingError) as caught:
            slab_split(
                histogram_from_points(points),
                TilingRule(tile_count=60, minimum_slab_extent_m=2.0),
            )
        self.assertIn("minimum_slab_extent_m", str(caught.exception))

    def test_single_tile_is_the_whole_padded_box(self) -> None:
        slab = slab_split(histogram_from_points(_skewed_cloud()), TilingRule(tile_count=1))
        self.assertEqual(slab.cuts, ())
        np.testing.assert_allclose(slab.core_boxes[0].minimum, slab.root_box.minimum)

    def test_degenerate_inputs_are_refused(self) -> None:
        with self.assertRaises(TilingError):
            histogram_from_points(np.zeros((0, 3)))
        with self.assertRaises(TilingError):
            histogram_from_points(np.array([[np.nan, 0.0, 0.0]]))
        with self.assertRaises(TilingError):
            TilingRule(tile_count=0).validate()
        with self.assertRaises(TilingError):
            TilingRule(histogram_bins=4).validate()
        with self.assertRaises(TilingError):
            TilingRule(overlap_margin_m=-1.0).validate()


class SlabPlanSchemaTest(unittest.TestCase):
    def test_plan_passes_the_trainer_validator(self) -> None:
        slab = slab_split(histogram_from_points(_skewed_cloud()), TilingRule(tile_count=4))
        plan = build_slab_tile_plan(slab, point_cloud_sha256="c" * 64)
        self.assertEqual(verify_adaptive_tile_plan(plan), plan["tile_plan_manifest_sha256"])
        self.assertEqual(plan["leaf_count"], 4)
        self.assertEqual(plan["views_source"], "deferred")
        self.assertEqual(
            [tile["name"] for tile in plan["tiles"]],
            ["Tile_0", "Tile_1", "Tile_2", "Tile_3"],
        )

    def test_tampered_plan_is_rejected(self) -> None:
        slab = slab_split(histogram_from_points(_skewed_cloud()), TilingRule(tile_count=2))
        plan = build_slab_tile_plan(slab)
        plan["tiles"][0]["core_box"][0][0] -= 1.0
        with self.assertRaises(ValueError):
            verify_adaptive_tile_plan(plan)

    def test_plan_records_every_rule_parameter(self) -> None:
        rule = TilingRule(tile_count=3, overlap_margin_m=0.4, histogram_bins=512)
        plan = build_slab_tile_plan(slab_split(histogram_from_points(_skewed_cloud()), rule))
        config = plan["config"]
        self.assertEqual(config["rule"], tiling.TILING_RULE_VERSION)
        self.assertEqual(config["tile_count"], 3)
        self.assertEqual(config["overlap_margin_m"], 0.4)
        self.assertEqual(config["histogram_bins"], 512)
        self.assertEqual(plan["input"]["split_axis"], "x")

    def test_observations_produce_view_rectangles(self) -> None:
        points = _skewed_cloud(count=2_000)
        slab = slab_split(histogram_from_points(points), TilingRule(tile_count=2))
        table = _observation_table(points, images=4)
        plan = build_slab_tile_plan(
            slab,
            observations=table,
            view_ids=[f"img_{i}::yaw_minus_35" for i in range(4)],
        )
        self.assertEqual(plan["views_source"], "projected_observations")
        for tile in plan["tiles"]:
            self.assertGreater(tile["valid_view_count"], 0)
            self.assertEqual(len(tile["views"]), tile["valid_view_count"])
            for view in tile["views"]:
                self.assertIn("sample_id", view)
                self.assertEqual(
                    view["pixel_load"], view["width"] * view["height"]
                )
        self.assertEqual(verify_adaptive_tile_plan(plan), plan["tile_plan_manifest_sha256"])

    def test_core_boxes_are_accepted_as_an_ownership_partition(self) -> None:
        slab = slab_split(histogram_from_points(_skewed_cloud()), TilingRule(tile_count=4))
        plan = build_slab_tile_plan(slab)
        tiles = [
            {"tile_id": tile["tile_id"], "core_box": tile["core_box"]}
            for tile in plan["tiles"]
        ]
        # Raises when the boxes overlap or leave a gap.
        _ordered_boxes(tiles)

    def test_resigning_after_mutation(self) -> None:
        slab = slab_split(histogram_from_points(_skewed_cloud()), TilingRule(tile_count=2))
        plan = build_slab_tile_plan(slab)
        plan["tiles"][0]["views"] = []
        resigned = tiling.signed_plan_copy(plan)
        self.assertEqual(
            verify_adaptive_tile_plan(resigned), resigned["tile_plan_manifest_sha256"]
        )


class TileInputsEndToEndTest(unittest.TestCase):
    def test_materialized_tile_inputs_pass_the_trainer_validator(self) -> None:
        points = _skewed_cloud(count=6_000)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            las = _write_las(root / "cloud.las", points)
            digest = hashlib.sha256(las.read_bytes()).hexdigest()
            slab = slab_split(histogram_from_las(las), TilingRule(tile_count=3))
            plan = build_slab_tile_plan(
                slab,
                observations=_observation_table(points, images=3),
                view_ids=[f"img_{i}::yaw_minus_35" for i in range(3)],
                point_cloud_sha256=digest,
            )
            plan_path = root / "adaptive_tile_plan.json"
            plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")

            manifest = materialize_lidar_tile_inputs(
                plan_path,
                las,
                root / "tile_inputs",
                expected_point_cloud_sha256=digest,
            )
            self.assertEqual(
                verify_tile_inputs_manifest(
                    manifest, root=root / "tile_inputs", verify_artifacts=True
                ),
                manifest["tile_inputs_manifest_sha256"],
            )
            self.assertEqual(manifest["tile_count"], 3)
            total = sum(tile["initialization"]["point_count"] for tile in manifest["tiles"])
            # The halo duplicates a few points; nothing is lost.
            self.assertGreaterEqual(total, len(points))
            for tile in manifest["tiles"]:
                self.assertEqual(tile["view_count"], len(tile["views"]))
                self.assertEqual(
                    tile["recommended_training"]["steps"], 20 * tile["view_count"]
                )

    def test_wrong_point_cloud_sha_is_refused(self) -> None:
        points = _skewed_cloud(count=2_000)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            las = _write_las(root / "cloud.las", points)
            slab = slab_split(histogram_from_las(las), TilingRule(tile_count=2))
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(build_slab_tile_plan(slab), indent=1), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                materialize_lidar_tile_inputs(
                    plan_path,
                    las,
                    root / "tile_inputs",
                    expected_point_cloud_sha256="d" * 64,
                )


@unittest.skipUnless(
    HOUSE0305_LAS.is_file() and HOUSE0305_TILE_INPUTS.is_file(),
    "house0305 LAS and the v9 tile inputs are not on this machine",
)
class House0305ComparisonTest(unittest.TestCase):
    """Regression guard for the numbers quoted in docs/sdk_ingestion.zh-CN.md."""

    def test_four_slabs_beat_the_hand_made_boxes_on_balance(self) -> None:
        histogram = histogram_from_las(HOUSE0305_LAS)
        slab = slab_split(histogram, TilingRule(tile_count=4))
        self.assertEqual(slab.axis_name, "x")
        self.assertLess(slab.balance(), 1.05)

        reference = json.loads(HOUSE0305_TILE_INPUTS.read_text(encoding="utf-8"))
        reference_counts = [
            tile["initialization"]["point_count"] for tile in reference["tiles"]
        ]
        reference_balance = max(reference_counts) / min(reference_counts)
        self.assertGreater(reference_balance, 2.0)

        # Both partitions cover exactly the same padded root volume.
        comparison = tiling.compare_tile_boxes(
            build_slab_tile_plan(slab)["tiles"], reference["tiles"]
        )
        self.assertAlmostEqual(
            comparison["plan_total_volume_m3"],
            comparison["reference_total_volume_m3"],
            places=3,
        )


class TileCliViewBindingTest(unittest.TestCase):
    """``ingest.cli tile`` binds views from the caches, with no readiness gate.

    A capture without independent AT (raw S1Mapper poses) has no gate chain, so the
    production kd planner cannot run on it; the slab rule plus the projected LiDAR/Face4
    table is what lets such a capture be tiled at all. The projection itself is mocked:
    it needs signed dataset, depth and face manifests.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.points = _skewed_cloud(count=2_000)
        self.las = _write_las(self.root / "cloud.las", self.points)
        self.output = self.root / "tile_plan" / "adaptive_tile_plan.json"
        self.view_ids = [f"img_{i}::yaw_minus_35" for i in range(4)]

    def _run(self, table, *extra: str):
        from unittest import mock

        from cloudstudio3dgs_sdk.ingest import cli

        manifest = {
            "dataset_manifest_sha256": "d" * 64,
            "lidar_depth_manifest_sha256": "e" * 64,
            "face4_observation_manifest_sha256": "f" * 64,
            "point_cloud_sha256": "c" * 64,
            "train_view_ids": self.view_ids,
        }
        module = "cloudstudio_3dgs.pipeline.lidar_face4_observations"
        with mock.patch(f"{module}.build_lidar_face4_projected_observations", return_value=manifest) as build, \
             mock.patch(f"{module}.load_lidar_face4_projected_observations", return_value=(table, table)):
            code = cli.main([
                "tile", "--point-cloud", str(self.las), "--output", str(self.output), "--tile-count", "2",
                *extra,
            ])
        return code, build

    def _binding_args(self):
        return (
            "--dataset-manifest", str(self.root / "dataset_manifest.json"),
            "--depth-manifest", str(self.root / "depth" / "depth_manifest.json"),
            "--depth-root", str(self.root / "depth"),
            "--face", str(self.root / "face4_train" / "face_manifest.json"), str(self.root / "face4_train"),
        )

    def test_views_are_bound_from_the_caches(self) -> None:
        code, build = self._run(_observation_table(self.points, images=4), *self._binding_args())
        self.assertEqual(code, 0)
        args = build.call_args.args
        self.assertEqual(args[3], [(self.root / "face4_train" / "face_manifest.json", self.root / "face4_train")])
        self.assertEqual(args[4], self.output.parent / "lidar_face4_projected_observations.npz")
        plan = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(plan["views_source"], "projected_observations")
        self.assertEqual(plan["input"]["point_cloud_sha256"], "c" * 64)
        self.assertEqual(plan["source_bindings"]["face4_observation_manifest_sha256"], "f" * 64)
        for tile in plan["tiles"]:
            self.assertGreater(tile["valid_view_count"], 0)
            self.assertTrue({view["sample_id"] for view in tile["views"]} <= set(self.view_ids))
        self.assertEqual(verify_adaptive_tile_plan(plan), plan["tile_plan_manifest_sha256"])

    def test_partial_binding_inputs_are_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._run(_observation_table(self.points, images=4), "--dataset-manifest", str(self.root / "d.json"))
        self.assertIn("--depth-manifest", str(caught.exception))
        self.assertFalse(self.output.exists())

    def test_a_tile_no_view_sees_is_refused_before_a_plan_is_written(self) -> None:
        # only the far end (x < -25) is observed; the equal-count cut sits near x = -16.8, so
        # the other slab gets no view at all
        far = self.points[:, 0] < -25.0
        table = _observation_table(self.points[far], images=4)
        table = ProjectedObservationTable(
            points=self.points,
            observation_xy=table.observation_xy,
            observation_image=table.observation_image,
            observation_point=np.flatnonzero(far)[table.observation_point],
            image_sizes=table.image_sizes,
        ).validated()
        with self.assertRaises(SystemExit) as caught:
            self._run(table, *self._binding_args())
        self.assertIn("no training view sees", str(caught.exception))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
