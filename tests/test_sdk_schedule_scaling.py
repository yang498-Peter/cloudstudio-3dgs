"""Profiles c1 / c7 / c6 and the schedule that follows a tile's view count
(cloudstudio3dgs_sdk/profile.py, plan.py).

What is pinned:

* c1 is b12op05d3 plus exactly the efficiency knobs; c7 is c1 trained for a classic viewer;
  c6 is c7 with a lean population. b12 / b13 keep the digests they had before these existed;
* on house0305 - the tiles the schedules were measured on - every derived step is the
  measured one;
* a tile outside the measured band gets a schedule proportional to its views, bounded, on the
  100-step grid, and always under max_steps (the trainer refuses stop >= max_steps);
* a tile too small to hold any schedule is refused while the plan is being built;
* the capacity ramp ends at two thirds of the (scaled) refine window and only c6 carries it;
* the lean cap is 0.878x the LiDAR count with no floor from an earlier, fatter population.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from cloudstudio3dgs_sdk.plan import (
    DatasetSummary,
    TileSummary,
    WorkLayout,
    apply_tile_schedule,
    build_plan,
    coarse_config,
    tile_max_steps,
    tile_schedule_scale,
)
from cloudstudio3dgs_sdk.profile import (
    PROFILE_B12OP05D3,
    PROFILE_B13OP05D0,
    PROFILE_C1,
    PROFILE_C6,
    PROFILE_C7,
    PROFILES,
    thaw,
)
from tests.test_sdk_profile_b12 import CARD_GIB, HOUSE0305_B5SKY_ADOPT, _flat, _house0305

C1_DELTA = {
    "training_focal_px": "sensor",
    "prefetch_workers": 6,
    "lidar_normal_alignment.scope": "visible_current_view",
    "lidar_normal_alignment.carry_through_refine": True,
}
C7_DELTA = {
    "pinhole_rasterize_mode": "classic",
    "mcmc_refine_stop_iter": 15000,
    "default_strategy.refine_stop_iter": 15000,
    "default_strategy.refine_scale2d_stop_iter": 15000,
    "geometry_regularization.max_opacity": None,
    "geometry_regularization.min_extent_px": 0.1,
    "sample_cache_gib": 8.0,
}
C6_DELTA = {
    "initialization_subsample_stride": 3,
    "initialization_subsample_keeps_coverage": True,
    "geometry_regularization.min_extent_px": 0.447,
}


def _changed(base, new):
    a, b = _flat(thaw(base.trainer_base)), _flat(thaw(new.trainer_base))
    return {key: b.get(key) for key in set(a) | set(b) if a.get(key) != b.get(key)}


def _plan(profile, dataset=None):
    root = Path("D:/sdk-test")
    dataset = dataset or _house0305()
    return build_plan(
        profile,
        dataset,
        dataset_root=root / "dataset",
        work_root=root / "work",
        repo_root=root / "repo",
        python=Path("python.exe"),
        vram_gib=CARD_GIB,
        prior_tile_checkpoints={tile.tile_id: f"prior{tile.tile_id}.pt" for tile in dataset.tiles},
    )


def _tile_configs(profile, dataset=None):
    return {
        step.config["mipmap_tile_id"]: step.config
        for step in _plan(profile, dataset).steps
        if step.name.startswith("train_tile") and step.name.endswith("_delivery")
    }


def _dataset(*views: int) -> DatasetSummary:
    return DatasetSummary(
        scene_tag="other",
        tiles=tuple(TileSummary(index, f"Tile_{index}", count, 4_000_000) for index, count in enumerate(views)),
        train_view_count=sum(views),
        global_init_point_count=1_000_000,
    )


class RecipeDeltaTest(unittest.TestCase):
    def test_c1_is_b12_plus_the_efficiency_knobs(self) -> None:
        self.assertEqual(_changed(PROFILE_B12OP05D3, PROFILE_C1), C1_DELTA)

    def test_c7_is_c1_trained_for_a_classic_viewer(self) -> None:
        self.assertEqual(_changed(PROFILE_C1, PROFILE_C7), C7_DELTA)
        self.assertEqual(thaw(PROFILE_C1.tile_rules), thaw(PROFILE_C7.tile_rules))

    def test_c6_is_c7_with_a_lean_population(self) -> None:
        self.assertEqual(_changed(PROFILE_C7, PROFILE_C6), C6_DELTA)
        base, new = thaw(PROFILE_C7.tile_rules), thaw(PROFILE_C6.tile_rules)
        changed = {key: new.get(key) for key in set(base) | set(new) if base.get(key) != new.get(key)}
        self.assertEqual(
            changed,
            {
                "cap_multiplier": 0.5,
                "cap_ceiling": 8000000,
                "cap_floor_policy": "none",
                "capacity_ramp_end_fraction_of_refine_stop": 2.0 / 3.0,
            },
        )

    def test_the_current_best_profiles_keep_their_digests(self) -> None:
        # As they were at c9154ba, before the c profiles existed.
        self.assertEqual(
            PROFILE_B12OP05D3.profile_sha256,
            "5d3ed11f55572626f46422ebb027ed2a2bbf9062a34fcdc112b8a079f5de3f51",
        )
        self.assertEqual(
            PROFILE_B13OP05D0.profile_sha256,
            "c80da2f9350eb674779fc10582185b4204a41aadc278cb49cd2bca61b118f46a",
        )
        self.assertNotIn("schedule_scaling", PROFILE_B12OP05D3.tiling)
        self.assertNotIn("schedule_scaling", PROFILE_B13OP05D0.tiling)

    def test_all_three_are_registered_and_every_changed_knob_is_justified(self) -> None:
        self.assertIs(PROFILES["c1"], PROFILE_C1)
        self.assertIs(PROFILES["c7"], PROFILE_C7)
        self.assertIs(PROFILES["c6"], PROFILE_C6)
        for knob in (
            "trainer_base.lidar_normal_alignment.scope",
            "trainer_base.lidar_normal_alignment.carry_through_refine",
            "trainer_base.training_focal_px",
            "trainer_base.prefetch_workers",
            "tiling.schedule_scaling",
        ):
            self.assertIn(knob, PROFILE_C1.provenance)
        for knob in (
            "trainer_base.pinhole_rasterize_mode",
            "trainer_base.geometry_regularization.max_opacity",
            "trainer_base.geometry_regularization.min_extent_px",
            "trainer_base.mcmc_refine_stop_iter",
            "trainer_base.sample_cache_gib",
        ):
            self.assertIn(knob, PROFILE_C7.provenance)
        for knob in (
            "trainer_base.initialization_subsample_stride",
            "tile_rules.cap_multiplier",
            "tile_rules.capacity_ramp_end_fraction_of_refine_stop",
        ):
            self.assertIn(knob, PROFILE_C6.provenance)
        for profile in (PROFILE_C1, PROFILE_C7, PROFILE_C6):
            for knob, provenance in profile.provenance.items():
                self.assertNotIn("pending", provenance.claim, knob)


class MeasuredTilesReproduceTest(unittest.TestCase):
    def test_house0305_keeps_the_measured_schedules(self) -> None:
        reference = _tile_configs(PROFILE_B12OP05D3)
        for profile, refine_stop in ((PROFILE_C1, 21000), (PROFILE_C7, 15000), (PROFILE_C6, 15000)):
            configs = _tile_configs(profile)
            self.assertEqual(sorted(configs), [0, 1, 2, 3])
            for tile_id, config in configs.items():
                with self.subTest(profile=profile.name, tile=tile_id):
                    self.assertEqual(config["controlled_stop_after_steps"], 30000)
                    self.assertEqual(config["mcmc_refine_stop_iter"], refine_stop)
                    self.assertEqual(config["default_strategy"]["refine_stop_iter"], refine_stop)
                    self.assertEqual(config["default_strategy"]["refine_scale2d_stop_iter"], refine_stop)
                    self.assertEqual(config["max_steps"], reference[tile_id]["max_steps"])
                    self.assertIn("schedule x1.000", config["lineage"]["schedule_rule"])

    def test_c1_and_c7_keep_the_delivered_caps(self) -> None:
        reference = {tile: config["cap_max"] for tile, config in _tile_configs(PROFILE_B12OP05D3).items()}
        for profile in (PROFILE_C1, PROFILE_C7):
            self.assertEqual(
                {tile: config["cap_max"] for tile, config in _tile_configs(profile).items()}, reference
            )

    def test_the_lean_cap_is_below_the_lidar_count_and_above_the_thinned_one(self) -> None:
        caps = {tile: config["cap_max"] for tile, config in _tile_configs(PROFILE_C6).items()}
        self.assertEqual(caps, {0: 6_200_000, 1: 3_000_000, 2: 2_900_000, 3: 4_950_000})
        for tile_id, _views, points, _previous in HOUSE0305_B5SKY_ADOPT:
            self.assertLess(points / 3, caps[tile_id])
            self.assertLess(caps[tile_id], points)
        self.assertFalse([w for w in _plan(PROFILE_C6).warnings if "floor rule" in w])

    def test_only_the_lean_profile_ramps_and_it_ends_at_two_thirds_of_refinement(self) -> None:
        for profile in (PROFILE_C1, PROFILE_C7):
            for config in _tile_configs(profile).values():
                self.assertNotIn("capacity_ramp_end_step", config["default_strategy"])
        for config in _tile_configs(PROFILE_C6).values():
            self.assertEqual(config["default_strategy"]["capacity_ramp_end_step"], 10000)

    def test_profiles_without_the_rule_never_scale(self) -> None:
        self.assertEqual(tile_schedule_scale(PROFILE_B12OP05D3, 400), 1.0)
        self.assertEqual(tile_schedule_scale(PROFILE_B12OP05D3, 9000), 1.0)
        configs = _tile_configs(PROFILE_B12OP05D3, _dataset(7202, 2000))
        self.assertEqual(configs[0]["controlled_stop_after_steps"], 30000)
        self.assertNotIn("schedule_rule", configs[0]["lineage"])


class ScaleRuleTest(unittest.TestCase):
    def test_inside_the_band_the_measured_schedule_is_used(self) -> None:
        for views in (1684, 1829, 2132, 2317, int(2317 * 1.25)):
            self.assertEqual(tile_schedule_scale(PROFILE_C7, views), 1.0, views)

    def test_outside_it_the_schedule_follows_the_views_within_bounds(self) -> None:
        self.assertAlmostEqual(tile_schedule_scale(PROFILE_C7, 842), 0.5)
        self.assertAlmostEqual(tile_schedule_scale(PROFILE_C7, 3476), 3476 / 2317)
        self.assertEqual(tile_schedule_scale(PROFILE_C7, 7202), 2.0)
        self.assertEqual(tile_schedule_scale(PROFILE_C7, 10), 0.05)

    def test_a_small_tile_gets_a_schedule_the_trainer_accepts(self) -> None:
        configs = _tile_configs(PROFILE_C1, _dataset(600, 400, 2000, 2000))
        small = configs[0]
        self.assertEqual(small["max_steps"], 12000)
        self.assertEqual(small["controlled_stop_after_steps"], 10700)  # 30000 * 600 / 1684 on the grid
        self.assertEqual(small["mcmc_refine_stop_iter"], 7500)
        self.assertEqual(small["default_strategy"]["refine_stop_iter"], 7500)
        smaller = configs[1]
        self.assertEqual(smaller["max_steps"], 8000)
        self.assertEqual(smaller["controlled_stop_after_steps"], 7100)
        self.assertLess(smaller["controlled_stop_after_steps"], smaller["max_steps"])
        lean = _tile_configs(PROFILE_C6, _dataset(600, 400, 2000, 2000))[0]
        self.assertEqual(lean["mcmc_refine_stop_iter"], 5400)  # 15000 * 10700 / 30000 on the grid
        self.assertEqual(lean["default_strategy"]["capacity_ramp_end_step"], 3600)

    def test_a_crowded_tile_is_stretched_and_the_ceiling_is_announced(self) -> None:
        dataset = _dataset(7202, 3476, 2000, 2000)
        configs = _tile_configs(PROFILE_C7, dataset)
        self.assertEqual(configs[0]["controlled_stop_after_steps"], 60000)
        self.assertEqual(configs[0]["mcmc_refine_stop_iter"], 30000)
        self.assertEqual(configs[1]["controlled_stop_after_steps"], 45000)
        self.assertEqual(configs[2]["controlled_stop_after_steps"], 30000)
        warnings = [w for w in _plan(PROFILE_C7, dataset).warnings if "stretched" in w]
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 tile(s)", warnings[0])

    def test_a_tile_too_small_to_train_is_refused_while_planning(self) -> None:
        config = {
            "controlled_stop_after_steps": 30000,
            "mcmc_refine_stop_iter": 15000,
            "mcmc_refine_start_iter": 500,
            "default_strategy": {"refine_stop_iter": 15000, "refine_scale2d_stop_iter": 15000},
        }
        with self.assertRaisesRegex(ValueError, "too small to train"):
            apply_tile_schedule(PROFILE_C7, config, view_count=30, max_steps=tile_max_steps(PROFILE_C7, 30))

    def test_the_coarse_prior_fits_under_max_steps_on_a_small_capture(self) -> None:
        dataset = DatasetSummary(
            scene_tag="room",
            tiles=tuple(TileSummary(index, f"Tile_{index}", 90, 500_000) for index in range(4)),
            train_view_count=360,
            global_init_point_count=200_000,
        )
        coarse = coarse_config(
            PROFILE_C7, dataset, layout=WorkLayout(Path("D:/sdk-test/work")), bundle_paths={}
        )
        self.assertEqual(coarse["max_steps"], 7200)
        self.assertLess(coarse["controlled_stop_after_steps"], coarse["max_steps"])
        self.assertLess(coarse["mcmc_refine_stop_iter"], coarse["controlled_stop_after_steps"])

    def test_every_planned_tile_schedule_is_self_consistent(self) -> None:
        for dataset in (
            _house0305(),
            _dataset(600, 400, 2000, 2000),
            _dataset(7202, 3476, 2000, 2000),
            _dataset(120, 150, 130, 140),
        ):
            for profile in (PROFILE_C1, PROFILE_C7, PROFILE_C6):
                for tile_id, config in _tile_configs(profile, dataset).items():
                    with self.subTest(profile=profile.name, scene=dataset.scene_tag, tile=tile_id):
                        strategy = config["default_strategy"]
                        stop = config["controlled_stop_after_steps"]
                        self.assertTrue(0 < stop < config["max_steps"])
                        self.assertEqual(config["mcmc_refine_start_iter"], 500)
                        self.assertEqual(strategy["refine_stop_iter"], config["mcmc_refine_stop_iter"])
                        self.assertEqual(strategy["refine_scale2d_stop_iter"], config["mcmc_refine_stop_iter"])
                        self.assertLess(config["mcmc_refine_stop_iter"], stop)
                        ramp = strategy.get("capacity_ramp_end_step")
                        if ramp is not None:
                            self.assertTrue(strategy["refine_start_iter"] < ramp <= strategy["refine_stop_iter"])


if __name__ == "__main__":
    unittest.main()
