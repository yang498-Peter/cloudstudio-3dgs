"""Plan generation: what runs, in what order, with which caps and estimates.

The plan is the contract between "we read the recipe" and "we spent a day of
GPU". Nothing here touches a GPU, a dataset or a network; the synthetic
two-tile scene exists so the step list and the derivations can be asserted
exactly, and one test replays the real house0305 tile geometry to prove the
derivations reproduce the four as-run configs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cloudstudio3dgs_sdk.plan import (
    STAGES,
    DatasetSummary,
    TileSummary,
    WorkLayout,
    build_plan,
    coarse_config,
    prune_switch_step,
    tile_cap,
    tile_config,
    tile_max_steps,
    tile_seconds_per_step,
)
from cloudstudio3dgs_sdk.profile import PROFILE_B5FILL2, MEASURED

# Tools the plan invokes. A fake repo root has to carry all of them or the
# preflight's tools_present check fires (which is its job).
PLAN_TOOLS = (
    "build_sky_masks.py",
    "build_sky_dome.py",
    "build_tile_ownership_masks.py",
    "build_view_backgrounds.py",
    "build_standin_backgrounds.py",
    "build_three_way_compare.py",
    "build_offtrajectory_compare.py",
    "merge_v28_tile_checkpoints.py",
    "export_gaussian_ply.py",
    "import_gaussian_ply.py",
    "concat_delivery_layers.py",
    "evaluate_probe_views.py",
    "checkpoint_morphology.py",
    "freeze_run_identity.py",
    "pipeline.py",
)

# The real house0305 tiling, from tile_inputs_v9/tile_inputs_manifest.json.
HOUSE0305_TILES = (
    # tile_id, views, init points, previous final population
    (0, 2132, 7044777, 11460000),
    (1, 1829, 3417320, 3340000),
    (2, 1684, 3309574, 7470000),
    (3, 2317, 5651827, 9890000),
)
HOUSE0305_ASRUN = {
    0: {"cap": 11000000, "max_steps": 42640, "prune_switch": 21320},
    1: {"cap": 6000000, "max_steps": 36580, "prune_switch": 18290},
    2: {"cap": 8000000, "max_steps": 33680, "prune_switch": 16840},
    3: {"cap": 9900000, "max_steps": 46340, "prune_switch": 23170},
}


def make_repo(root: Path, *, fill_support: bool = True) -> Path:
    """A checkout skeleton: the tools the plan names plus the gsplat lock."""
    tools = root / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    for name in PLAN_TOOLS:
        (tools / name).write_text("# stub\n", encoding="utf-8")
    if fill_support:
        (tools / "merge_v28_tile_checkpoints.py").write_text(
            '# stub\nparser.add_argument("--fill-checkpoint")\n', encoding="utf-8"
        )
        (tools / "pipeline.py").write_text("# stub\nfill_checkpoint = True\n", encoding="utf-8")
    (root / "upstream").mkdir(parents=True, exist_ok=True)
    (root / "upstream" / "gsplat.lock.json").write_text(
        json.dumps({"schema_version": 1, "version": "1.5.3", "commit": "f2d14131483644e9"}),
        encoding="utf-8",
    )
    return root


def two_tile_dataset(**overrides) -> DatasetSummary:
    fields = {
        "scene_tag": "synth",
        "tiles": (
            TileSummary(0, "Tile_0", view_count=100, init_point_count=1_000_000),
            TileSummary(1, "Tile_1", view_count=200, init_point_count=2_000_000),
        ),
        "train_view_count": 250,
        "global_init_point_count": 500_000,
        "has_reference_model": False,
    }
    fields.update(overrides)
    return DatasetSummary(**fields)


def plan_for(dataset: DatasetSummary, root: Path, **overrides):
    kwargs = {
        "dataset_root": root / "dataset",
        "work_root": root / "work",
        "repo_root": make_repo(root / "repo"),
        "python": Path("python.exe"),
        "prior_tile_checkpoints": {tile.tile_id: f"prior{tile.tile_id}.pt" for tile in dataset.tiles},
    }
    kwargs.update(overrides)
    return build_plan(PROFILE_B5FILL2, dataset, **kwargs)


class TempRootTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class DatasetSummaryTests(TempRootTestCase):
    def test_rejects_empty_and_out_of_order_tiles(self) -> None:
        with self.assertRaises(ValueError):
            DatasetSummary("s", (), 1, 1)
        tiles = (TileSummary(1, "Tile_1", 1, 1), TileSummary(0, "Tile_0", 1, 1))
        with self.assertRaises(ValueError):
            DatasetSummary("s", tiles, 1, 1)

    def test_rejects_duplicate_tile_ids(self) -> None:
        tiles = (TileSummary(0, "a", 1, 1), TileSummary(0, "b", 1, 1))
        with self.assertRaises(ValueError):
            DatasetSummary("s", tiles, 1, 1)

    def test_json_round_trip(self) -> None:
        dataset = two_tile_dataset()
        self.assertEqual(DatasetSummary.from_json(dataset.as_json()), dataset)

    def test_reads_the_existing_tile_inputs_manifest_shape(self) -> None:
        payload = {
            "tiles": [
                {
                    "tile_id": tile_id,
                    "name": f"Tile_{tile_id}",
                    "view_count": views,
                    "initialization": {"point_count": points, "sha256": "ab" * 32},
                }
                # deliberately out of order: the reader sorts
                for tile_id, views, points, _ in reversed(HOUSE0305_TILES)
            ]
        }
        dataset = DatasetSummary.from_tile_inputs_manifest(
            payload, scene_tag="house0305", train_view_count=3536, global_init_point_count=1863918
        )
        self.assertEqual([t.tile_id for t in dataset.tiles], [0, 1, 2, 3])
        self.assertEqual(dataset.tiles[0].init_point_count, 7044777)


class DerivationTests(unittest.TestCase):
    def test_cap_is_the_measured_ratio_of_the_initialisation(self) -> None:
        tile = TileSummary(0, "Tile_0", 100, 1_000_000)
        self.assertEqual(tile_cap(PROFILE_B5FILL2, tile), 1_800_000)  # 1.756M rounded to 100k

    def test_cap_floor_fires_only_below_the_previous_population(self) -> None:
        tight = TileSummary(0, "Tile_0", 100, 1_000_000, previous_final_population=4_000_000)
        self.assertEqual(tile_cap(PROFILE_B5FILL2, tight), 4_300_000)  # 4.0M x 1.07
        # A tile whose ratio already clears its previous population is untouched.
        roomy = TileSummary(0, "Tile_0", 100, 1_000_000, previous_final_population=1_500_000)
        self.assertEqual(tile_cap(PROFILE_B5FILL2, roomy), 1_800_000)

    def test_cap_is_clamped_by_the_vram_ceiling(self) -> None:
        huge = TileSummary(0, "Tile_0", 100, 20_000_000)
        self.assertEqual(tile_cap(PROFILE_B5FILL2, huge, vram_gib=16.0), 11_000_000)
        self.assertGreater(tile_cap(PROFILE_B5FILL2, huge), 11_000_000)  # unclamped

    def test_max_steps_is_epochs_times_views(self) -> None:
        self.assertEqual(tile_max_steps(PROFILE_B5FILL2, 1829), 36580)
        self.assertEqual(prune_switch_step(PROFILE_B5FILL2, 36580), 18290)

    def test_seconds_per_step_matches_the_measured_runs(self) -> None:
        # The fit is over 121/152/168/187 min at caps 6.0/8.0/9.9/11.0M.
        for cap, minutes in ((6.0e6, 121), (8.0e6, 152), (9.9e6, 168), (11.0e6, 187)):
            predicted = tile_seconds_per_step(PROFILE_B5FILL2, int(cap)) * 20000 / 60.0
            self.assertLess(abs(predicted - minutes) / minutes, 0.05, f"cap {cap}")

    def test_house0305_derivations_reproduce_the_as_run_configs(self) -> None:
        """The recipe-as-data claim, checked against the four shipped configs."""
        for tile_id, views, points, previous in HOUSE0305_TILES:
            tile = TileSummary(tile_id, f"Tile_{tile_id}", views, points, previous_final_population=previous)
            expected = HOUSE0305_ASRUN[tile_id]
            with self.subTest(tile=tile_id):
                self.assertEqual(tile_cap(PROFILE_B5FILL2, tile, vram_gib=16.0), expected["cap"])
                steps = tile_max_steps(PROFILE_B5FILL2, views)
                self.assertEqual(steps, expected["max_steps"])
                self.assertEqual(prune_switch_step(PROFILE_B5FILL2, steps), expected["prune_switch"])
        # The coarse prior's schedule comes from the same rule over every face.
        self.assertEqual(tile_max_steps(PROFILE_B5FILL2, 3536), 70720)


class StepListTests(TempRootTestCase):
    def test_two_tile_plan_step_names(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        self.assertEqual(plan.generations, ("delivery",))
        self.assertEqual(
            [step.name for step in plan.stage_steps("prepare")],
            [
                "ingest_dataset",
                "write_arm_configs",
                "delivery_eval_config",
                "sky_masks",
                "sky_dome",
                "sky_dome_ply",
                "ownership_Tile_0",
                "ownership_Tile_1",
            ],
        )
        self.assertEqual(
            [step.name for step in plan.stage_steps("train")],
            [
                "global_view_backgrounds",
                "train_global_coarse_b5fill2",
                "backdrop_Tile_0",
                "train_tile0_b5fill2_delivery",
                "backdrop_Tile_1",
                "train_tile1_b5fill2_delivery",
            ],
        )
        self.assertEqual(
            [step.name for step in plan.stage_steps("deliver")],
            [
                "merge_tiles",
                "export_ply",
                "threshold_control",
                "reimport_ply",
                "battery",
                "pair",
                "battery_pair",
                "morphology",
                "freeze_identity",
            ],
        )
        self.assertEqual([step.name for step in plan.stage_steps("report")], ["acceptance_report"])

    def test_backdrop_precedes_the_tile_it_feeds(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        names = [step.name for step in plan.stage_steps("train")]
        self.assertLess(names.index("backdrop_Tile_1"), names.index("train_tile1_b5fill2_delivery"))
        self.assertLess(names.index("train_global_coarse_b5fill2"), names.index("backdrop_Tile_0"))

    def test_without_prior_checkpoints_a_seed_generation_is_added(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root, prior_tile_checkpoints={})
        self.assertEqual(plan.generations, ("seed", "delivery"))
        names = [step.name for step in plan.stage_steps("train")]
        self.assertIn("train_tile0_b5fill2_seed", names)
        self.assertIn("train_tile1_b5fill2_delivery", names)
        # Every seed arm runs before any backdrop, because the backdrops read them.
        self.assertLess(names.index("train_tile1_b5fill2_seed"), names.index("backdrop_Tile_0"))
        self.assertTrue(any("seed generation" in warning for warning in plan.warnings))

    def test_seed_arms_drop_ownership_and_sky(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root, prior_tile_checkpoints={})
        seed = next(s for s in plan.steps if s.name == "train_tile0_b5fill2_seed")
        delivery = next(s for s in plan.steps if s.name == "train_tile0_b5fill2_delivery")
        self.assertFalse(seed.config["tile_ownership_masking"])
        self.assertFalse(seed.config["sky_supervision"]["enabled"])
        self.assertTrue(delivery.config["tile_ownership_masking"])
        self.assertTrue(delivery.config["sky_supervision"]["enabled"])
        self.assertNotIn("tile_ownership_cache_manifest", seed.config)

    def test_compare_steps_only_exist_with_a_reference_model(self) -> None:
        without = plan_for(two_tile_dataset(), self.root)
        self.assertNotIn("compare_matched", [s.name for s in without.steps])
        self.assertTrue(any("reference" in w for w in without.warnings))
        with_ref = plan_for(two_tile_dataset(has_reference_model=True), self.root)
        self.assertIn("compare_matched", [s.name for s in with_ref.steps])
        self.assertIn("offtrajectory", [s.name for s in with_ref.steps])

    def test_stage_subset_is_honoured(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root, stages=("prepare", "train"))
        self.assertEqual(plan.stage_steps("deliver"), ())
        self.assertEqual(plan.stage_steps("report"), ())
        self.assertTrue(plan.stage_steps("prepare"))

    def test_unknown_stage_lookup_raises(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        with self.assertRaises(KeyError):
            plan.stage_steps("polish")


class ConfigTests(TempRootTestCase):
    def test_tile_config_carries_the_derivations_and_the_cache_paths(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        step = next(s for s in plan.steps if s.name == "train_tile1_b5fill2_delivery")
        config = step.config
        self.assertEqual(config["mipmap_tile_id"], 1)
        self.assertEqual(config["cap_max"], tile_cap(PROFILE_B5FILL2, plan.dataset.tiles[1]))
        self.assertEqual(config["max_steps"], 20 * 200)
        self.assertEqual(config["default_strategy"]["prune_switch_step"], 2000)
        self.assertEqual(config["controlled_stop_after_steps"], 20000)
        self.assertTrue(config["tile_ownership_cache_manifest"].endswith("tile_ownership_manifest.json"))
        self.assertTrue(config["sky_supervision"]["mask_manifest"].endswith("sky_mask_train.json"))
        self.assertIn("backdrops", config["background_image_root"])
        self.assertEqual(config["lineage"]["profile_sha256"], PROFILE_B5FILL2.profile_sha256)

    def test_tile_config_does_not_mutate_the_profile(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        before = PROFILE_B5FILL2.profile_sha256
        step = next(s for s in plan.steps if s.name == "train_tile0_b5fill2_delivery")
        dict(step.config)["cap_max"] = 1
        self.assertEqual(PROFILE_B5FILL2.profile_sha256, before)
        self.assertNotIn("cap_max", PROFILE_B5FILL2.trainer_base)

    def test_coarse_config_drops_the_tile_only_wiring(self) -> None:
        dataset = two_tile_dataset()
        layout = WorkLayout(self.root / "work")
        config = coarse_config(PROFILE_B5FILL2, dataset, layout=layout, bundle_paths={})
        for key in ("tile_ownership_masking", "sky_supervision", "tile_inputs_manifest", "mipmap_tile_id"):
            self.assertNotIn(key, config)
        self.assertEqual(config["cap_max"], 3_000_000)
        self.assertEqual(config["controlled_stop_after_steps"], 10000)
        self.assertEqual(config["max_steps"], 20 * 250)
        self.assertEqual(config["default_strategy"]["refine_stop_iter"], 8000)
        # the override is a merge, not a replacement
        self.assertEqual(config["default_strategy"]["reset_every"], 300)
        self.assertEqual(config["surface_initialization"]["mode"], "planar_surfel")

    def test_tile_config_uses_bundle_paths_when_prepare_has_run(self) -> None:
        dataset = two_tile_dataset()
        layout = WorkLayout(self.root / "work")
        config = tile_config(
            PROFILE_B5FILL2,
            dataset,
            dataset.tiles[0],
            generation="delivery",
            layout=layout,
            bundle_paths={"dataset_manifest": "D:/scene/dataset_manifest.json"},
            cap_max=1_800_000,
        )
        self.assertEqual(config["dataset_manifest"], "D:/scene/dataset_manifest.json")
        self.assertTrue(config["split_manifest"].startswith("<prepare:"))


class EstimateTests(TempRootTestCase):
    def test_every_step_declares_an_estimate_and_its_confidence(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        for step in plan.steps:
            with self.subTest(step=step.name):
                self.assertGreaterEqual(step.estimate.seconds, 0.0)
                self.assertGreaterEqual(step.estimate.disk_bytes, 0)
                self.assertTrue(step.estimate.basis)
                self.assertIn(
                    step.estimate.confidence,
                    ("measured", "extrapolated", "inferred", "unmeasured", "inherited"),
                )

    def test_totals_add_up_and_inherit_the_worst_confidence(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        train = plan.total("train")
        self.assertAlmostEqual(
            train.seconds, sum(step.estimate.seconds for step in plan.stage_steps("train")), places=3
        )
        grand = plan.total()
        self.assertGreater(grand.seconds, train.seconds)
        self.assertNotEqual(grand.confidence, MEASURED)  # some knob is unmeasured

    def test_training_dominates_the_estimate(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        tile_step = next(s for s in plan.steps if s.name == "train_tile1_b5fill2_delivery")
        # 20k controlled stop at the tile's cap, through the measured fit.
        cap = tile_cap(PROFILE_B5FILL2, plan.dataset.tiles[1])
        self.assertAlmostEqual(tile_step.estimate.seconds, 20000 * tile_seconds_per_step(PROFILE_B5FILL2, cap))

    def test_render_shows_stages_totals_and_caps(self) -> None:
        text = plan_for(two_tile_dataset(), self.root).render()
        for fragment in ("[prepare]", "[train]", "[deliver]", "[report]", "total", "cap ", "profile_sha256"):
            self.assertIn(fragment, text)

    def test_render_surfaces_blocking_steps(self) -> None:
        repo = make_repo(self.root / "norepo", fill_support=False)
        plan = plan_for(two_tile_dataset(), self.root, repo_root=repo)
        merge = next(step for step in plan.steps if step.name == "merge_tiles")
        self.assertIn("--fill-checkpoint", merge.blocking)
        self.assertIn("BLOCKED merge_tiles", plan.render())
        self.assertEqual(plan.blocking_steps(), (merge,))

    def test_no_blocking_when_the_checkout_supports_the_fill(self) -> None:
        plan = plan_for(two_tile_dataset(), self.root)
        self.assertEqual(plan.blocking_steps(), ())


class PlanIdentityTests(TempRootTestCase):
    def test_plan_sha_changes_with_the_dataset(self) -> None:
        first = plan_for(two_tile_dataset(), self.root)
        bigger = two_tile_dataset(
            tiles=(
                TileSummary(0, "Tile_0", 100, 1_000_000),
                TileSummary(1, "Tile_1", 200, 3_000_000),
            )
        )
        second = plan_for(bigger, self.root)
        self.assertNotEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(first.profile_sha256, second.profile_sha256)

    def test_plan_sha_is_stable_for_the_same_inputs(self) -> None:
        dataset = two_tile_dataset()
        self.assertEqual(plan_for(dataset, self.root).plan_sha256, plan_for(dataset, self.root).plan_sha256)

    def test_plan_json_is_serialisable_and_complete(self) -> None:
        payload = plan_for(two_tile_dataset(), self.root).as_json()
        text = json.dumps(payload)
        self.assertIn("merge_tiles", text)
        self.assertEqual(len(payload["steps"]), sum(len(plan_for(two_tile_dataset(), self.root).stage_steps(s)) for s in STAGES))
        self.assertEqual(sorted(payload["tile_caps"]), ["0", "1"])


if __name__ == "__main__":
    unittest.main()
