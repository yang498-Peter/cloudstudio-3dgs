"""How many tiles a capture is cut into, and refusing a tile that cannot hold its initialisation.

The profile carries a reference tile count (4, house0305). Applied to every capture it cut
house0614 (100M LiDAR points) into four tiles of ~25.7M initialisation points against 15M
caps; the trainer refuses that at startup, i.e. after a day and a half of prepare. What is
pinned:

* the tile count grows with the cloud so the cap rule fits, never below the reference, and
  house0305 stays at 4 under every profile;
* the estimate and the fresh prepare both use it, and an explicit tiling rule still wins;
* the preflight refuses a tile whose initialisation reaches its cap and names the tile count
  that would fit, and warns when a cap only throttles growth.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cloudstudio3dgs_sdk.bundle import _tile_count
from cloudstudio3dgs_sdk.ingest.tiling import TilingRule
from cloudstudio3dgs_sdk.plan import TileSummary, tile_count_for
from cloudstudio3dgs_sdk.profile import PROFILE_B5FILL2, PROFILE_B5SKY, PROFILE_B12OP05D3, PROFILES
from cloudstudio3dgs_sdk.requirements import FAIL, PASS, WARN
from tests import test_sdk_discover as _discover_tests
from tests import test_sdk_requirements as _requirements_tests
from tests.test_ingest_adapters import write_las
from tests.test_sdk_plan import two_tile_dataset

CARD_GIB = 16303 / 1024
HOUSE0305_POINTS = 18757869
HOUSE0614_POINTS = 100353516


class TileCountRuleTest(unittest.TestCase):
    def test_house0305_stays_at_four_under_every_profile(self) -> None:
        for name, profile in PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(tile_count_for(profile, HOUSE0305_POINTS, vram_gib=CARD_GIB), 4)

    def test_house0614_is_cut_so_every_tile_fits_its_cap(self) -> None:
        tiles = tile_count_for(PROFILE_B12OP05D3, HOUSE0614_POINTS)
        self.assertEqual(tiles, 19)
        # an equal share plus the halo allowance, through the cap rule, stays under the ceiling
        per_tile = HOUSE0614_POINTS * 1.05 / tiles
        self.assertLessEqual(per_tile * 1.756 * 1.5, 15_000_000)

    def test_the_card_bounds_a_profile_without_a_measured_ceiling(self) -> None:
        self.assertEqual(tile_count_for(PROFILE_B5SKY, HOUSE0614_POINTS), 4)  # nothing to bound it
        self.assertEqual(tile_count_for(PROFILE_B5SKY, HOUSE0614_POINTS, vram_gib=CARD_GIB), 17)

    def test_a_small_cloud_never_drops_below_the_reference(self) -> None:
        self.assertEqual(tile_count_for(PROFILE_B12OP05D3, 1000, vram_gib=CARD_GIB), 4)


class EstimateUsesTheRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _discover_tests.CaptureFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_the_estimate_cuts_as_many_tiles_as_the_rule_says(self) -> None:
        with mock.patch("cloudstudio3dgs_sdk.discover.tile_count_for", return_value=3) as rule:
            summary = self.fixture.estimate(vram_gib=12.0).summary
        self.assertEqual(summary.tile_count, 3)
        self.assertEqual(rule.call_args.kwargs["vram_gib"], 12.0)

    def test_an_explicit_tiling_rule_wins(self) -> None:
        with mock.patch("cloudstudio3dgs_sdk.discover.tile_count_for", return_value=3):
            summary = self.fixture.estimate(tiling_rule=TilingRule(tile_count=2)).summary
        self.assertEqual(summary.tile_count, 2)


class FreshPrepareUsesTheRuleTest(unittest.TestCase):
    def test_the_count_comes_from_the_las_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cloud = write_las(Path(tmp) / "cloud.las", np.random.default_rng(0).uniform(0, 10, (500, 3)))
            with mock.patch("cloudstudio3dgs_sdk.plan.tile_count_for", return_value=7) as rule:
                self.assertEqual(_tile_count(PROFILE_B12OP05D3, cloud, vram_gib=CARD_GIB), 7)
            self.assertEqual(rule.call_args.args[1], 500)
            self.assertEqual(rule.call_args.kwargs["vram_gib"], CARD_GIB)

    def test_an_unreadable_cloud_leaves_the_reference_in_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_tile_count(PROFILE_B12OP05D3, Path(tmp) / "missing.las", vram_gib=None))


class TileCapacityPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _requirements_tests.PreflightTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def check(self, *tiles: TileSummary, lidar: int = 0):
        dataset = two_tile_dataset(tiles=tiles, lidar_point_count=lidar)
        return self.fixture.report(dataset=dataset, vram_gib=16.0).get("tile_capacity")

    def test_ordinary_tiles_pass(self) -> None:
        self.assertEqual(
            self.check(TileSummary(0, "Tile_0", 100, 1_000_000), TileSummary(1, "Tile_1", 200, 2_000_000)).status,
            PASS,
        )

    def test_a_tile_over_its_cap_fails_and_names_the_tile_count_that_fits(self) -> None:
        check = self.check(
            TileSummary(0, "Tile_0", 100, 30_000_000), TileSummary(1, "Tile_1", 200, 2_000_000),
            lidar=32_000_000,
        )
        self.assertEqual(check.status, FAIL)
        self.assertTrue(check.required)
        self.assertIn("Tile_0 init 30.00M", check.detail)
        wanted = tile_count_for(PROFILE_B5FILL2, 32_000_000, vram_gib=24.0)  # the stub card
        self.assertIn(f"({wanted} for 32.0M points", check.remedy)

    def test_a_cap_under_the_ratio_only_warns(self) -> None:
        # 7.0M x 1.756 = 12.3M, clamped to 11.0M by a 16 GiB card
        check = self.check(TileSummary(0, "Tile_0", 100, 7_000_000), TileSummary(1, "Tile_1", 200, 2_000_000))
        self.assertEqual(check.status, WARN)
        self.assertFalse(check.required)


if __name__ == "__main__":
    unittest.main()
