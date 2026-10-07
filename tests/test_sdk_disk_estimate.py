"""Costing a capture nobody has prepared, and refusing it before prepare starts when it won't fit.

The ingest cache graph is most of the disk a fresh scene takes - house0305's caches are ~49 GiB,
house0614's (6086 images) ~340 GiB - and it used to be costed at house0305's size whatever the
capture, while the SDK plan's ingest step budgeted nothing at all. A dry run of house0614 on a
disk with 16 GB free therefore passed preflight. What is pinned:

* every cache cost scales from house0305 by its basis (images, faces, one tile's share);
* the plan's ingest step carries that estimate for an estimated summary, and only then - a
  prepared scene's plans are unchanged;
* the disk preflight refuses such a capture and says how many images would fit.
"""

from __future__ import annotations

import unittest

from cloudstudio3dgs_sdk.bundle import VALIDATION_CACHES
from cloudstudio3dgs_sdk.ingest.caches import REFERENCE_COSTS, SceneScale, estimate_ingest
from cloudstudio3dgs_sdk.plan import GIB, TileSummary
from cloudstudio3dgs_sdk.requirements import FAIL, PASS
from tests.test_sdk_plan import two_tile_dataset
from tests import test_sdk_requirements as _requirements_tests

HOUSE0305 = SceneScale(images=886, faces=3536, tiles=4)
HOUSE0614_FACES = 6086 * 4


class SceneScaleTest(unittest.TestCase):
    def test_house0305_costs_are_the_reference(self) -> None:
        for name, (minutes, gib, _) in REFERENCE_COSTS.items():
            with self.subTest(cache=name):
                self.assertAlmostEqual(HOUSE0305.minutes(name), minutes)
                self.assertAlmostEqual(HOUSE0305.gib(name), gib)

    def test_face_caches_scale_with_faces_and_depth_with_images(self) -> None:
        big = SceneScale(images=6086, faces=HOUSE0614_FACES, tiles=4)
        self.assertAlmostEqual(big.gib("face_cache"), 25.0 * HOUSE0614_FACES / 3536)
        self.assertAlmostEqual(big.gib("depth_cache"), 9.8 * 6086 / 886)
        self.assertAlmostEqual(big.gib("tile_plan"), 0.01)

    def test_a_per_tile_cache_shrinks_as_the_tiles_multiply(self) -> None:
        eight = SceneScale(images=886, faces=3536, tiles=8)
        self.assertAlmostEqual(eight.gib("view_backgrounds"), 2.5 / 2)


class EstimateIngestTest(unittest.TestCase):
    def test_house0305_comes_out_at_its_measured_size(self) -> None:
        _, gib = estimate_ingest(3536, 4, validation_caches=VALIDATION_CACHES)
        # 45.2 GiB of train-split caches the SDK plan does not budget, plus the validation copies
        self.assertAlmostEqual(gib, 49.4, delta=0.1)

    def test_house0614_comes_out_near_its_hand_estimate(self) -> None:
        minutes, gib = estimate_ingest(HOUSE0614_FACES, 4, validation_caches=VALIDATION_CACHES)
        self.assertGreater(gib, 300.0)
        self.assertLess(gib, 360.0)
        self.assertGreater(minutes / 60.0, 30.0)


class IngestStepEstimateTest(unittest.TestCase):
    """Borrows the preflight fixture (fake repo, injected probes) without re-running its tests."""

    def setUp(self) -> None:
        # a module reference, not the class: a TestCase in this namespace would be collected again
        self.fixture = _requirements_tests.PreflightTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.plan
        self.report = self.fixture.report

    def test_only_an_estimated_summary_carries_the_ingest_estimate(self) -> None:
        def ingest_bytes(dataset):
            plan = self.plan(dataset)
            return next(s for s in plan.steps if s.name == "ingest_dataset").estimate.disk_bytes

        self.assertEqual(ingest_bytes(two_tile_dataset(train_view_count=HOUSE0614_FACES)), 0)
        _, gib = estimate_ingest(HOUSE0614_FACES, 2, validation_caches=VALIDATION_CACHES)
        self.assertEqual(
            ingest_bytes(two_tile_dataset(train_view_count=HOUSE0614_FACES, estimated=True)), int(gib * GIB)
        )

    def test_a_large_capture_is_refused_with_a_subset_that_fits(self) -> None:
        dataset = two_tile_dataset(
            train_view_count=HOUSE0614_FACES,
            estimated=True,
            tiles=(
                TileSummary(0, "Tile_0", view_count=14000, init_point_count=1_000_000),
                TileSummary(1, "Tile_1", view_count=16000, init_point_count=2_000_000),
            ),
        )
        check = self.report(dataset=dataset, free_disk_bytes=lambda p: 60 * GIB).get("disk_headroom")
        self.assertEqual(check.status, FAIL)
        self.assertIn("images fit", check.remedy)
        images = int(check.remedy.split("about ")[1].split(" of its")[0])
        # the suggested subset must actually pass the same check; a subset of a capture has
        # proportionally fewer views per tile too
        share = images * 4 / HOUSE0614_FACES
        tiles = tuple(
            TileSummary(t.tile_id, t.name, max(1, int(t.view_count * share)), t.init_point_count)
            for t in dataset.tiles
        )
        subset = two_tile_dataset(train_view_count=images * 4, estimated=True, tiles=tiles)
        again = self.report(dataset=subset, free_disk_bytes=lambda p: 60 * GIB).get("disk_headroom")
        self.assertEqual(again.status, PASS, again.detail)
        self.assertGreater(images, 500)

    def test_too_little_disk_for_any_subset_says_so(self) -> None:
        dataset = two_tile_dataset(train_view_count=HOUSE0614_FACES, estimated=True)
        check = self.report(dataset=dataset, free_disk_bytes=lambda p: 1 * GIB).get("disk_headroom")
        self.assertEqual(check.status, FAIL)
        self.assertIn("does not fit", check.remedy)


if __name__ == "__main__":
    unittest.main()
