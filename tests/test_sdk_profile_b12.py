"""The current-best recipe as an SDK profile (b12op05d3, b13op05d0) and what it needs.

What is pinned:

* the earlier profiles keep their digests. An adopted work root records the profile sha256 and
  refuses a process whose profile hashes differently; house0305's b5sky adopt recorded
  95bdb17f..., so adding profiles must not move it;
* b12op05d3 is b6reset plus exactly the delivered ladder's changes (b7 -> b12), and b13op05d0
  differs from it only in the alpha dilation;
* the cap rule reproduces the delivered caps (15.0 / 9.0 / 10.2 / 14.85M) from the house0305
  inventory and the previous populations the b5sky adopt recorded, and a fresh scene gets
  ratio x multiplier under the measured ceiling and the card's;
* the refined sky label is a prepare step and is the mask the trainer reads.

The key-for-key comparison with the four as-run tile configs needs the run tree, so it is a
tool, not a test: ``tools/diff_profile_vs_asrun.py`` (0 disagreements for b12 and b13 on
2026-10-07).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from cloudstudio3dgs_sdk.plan import DatasetSummary, TileSummary, WorkLayout, build_plan, tile_cap
from cloudstudio3dgs_sdk.profile import (
    DEFAULT_PROFILE,
    PROFILE_B5FILL2,
    PROFILE_B5SKY,
    PROFILE_B6RESET,
    PROFILE_B12OP05D3,
    PROFILE_B13OP05D0,
    PROFILES,
    thaw,
)

CARD_GIB = 16303 / 1024  # the campaign card, RTX 5070 Ti

# tile_id, views, init points, previous final population as the b5sky adopt recorded it
# (sdk_house0305_b5sky/prepare/prepare_manifest.json). Every later delivery inherited those
# caps, so this is the inventory the delivered caps derive from.
HOUSE0305_B5SKY_ADOPT = (
    (0, 2132, 7044777, 8480130),
    (1, 1829, 3417320, 4654937),
    (2, 1684, 3309574, 6389532),
    (3, 2317, 5651827, 7587560),
)

# The ladder b6reset -> b12op05d3, as the four as-run b12 tile configs carry it.
B12_DELTA = {
    "controlled_stop_after_steps": 30000,
    "sh_degree": 0,
    "pinhole_rasterize_mode": "antialiased",
    "mcmc_refine_stop_iter": 21000,
    "lidar_alpha_dilation_radius_px": 3,
    "lidar_alpha_support_mode": "strict_visibility",
    "lidar_alpha_exclude_sky_label": True,
    "default_strategy.refine_stop_iter": 21000,
    "default_strategy.refine_scale2d_stop_iter": 21000,
    "default_strategy.prune_opa": 0.04,
    "default_strategy.prune_opa_late": 0.04,
    "default_strategy.vendor_cull_warmup_profile": "calibrated_uniform_0p04",
    "geometry_regularization.max_opacity": 0.5,
    "sky_supervision.mask_erosion_px": 24,
}


def _flat(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, inner in value.items():
            out.update(_flat(inner, f"{prefix}.{key}" if prefix else key))
    else:
        out[prefix] = value
    return out


def _house0305(previous=True) -> DatasetSummary:
    return DatasetSummary(
        scene_tag="house0305",
        tiles=tuple(
            TileSummary(t, f"Tile_{t}", v, p, previous_final_population=prev if previous else None)
            for t, v, p, prev in HOUSE0305_B5SKY_ADOPT
        ),
        train_view_count=3536,
        global_init_point_count=1863918,
    )


def _plan(profile, *, bundle_paths=None, previous=True):
    root = Path("D:/sdk-test")
    return build_plan(
        profile,
        _house0305(previous),
        dataset_root=root / "dataset",
        work_root=root / "work",
        repo_root=root / "repo",
        python=Path("python.exe"),
        vram_gib=CARD_GIB,
        bundle_paths=bundle_paths,
        prior_tile_checkpoints={t: f"prior{t}.pt" for t, *_ in HOUSE0305_B5SKY_ADOPT},
    )


class EarlierProfilesTest(unittest.TestCase):
    def test_earlier_profiles_keep_their_digest(self) -> None:
        pinned = (
            (PROFILE_B5SKY, "95bdb17f8c1c116ae06bb057876b1267622738a0b914ae9132b582f9aa9d0881"),
            (PROFILE_B6RESET, "c4617a58bb2c020bb659647522e28766c48ee3025ce9c734b8745dd164a6dd65"),
            (PROFILE_B5FILL2, "a8e2d7129f40cfedb83d24a7ee7c5d1033731413c260901bcf1dfa1e736254e6"),
        )
        for profile, digest in pinned:
            with self.subTest(profile=profile.name):
                self.assertEqual(profile.profile_sha256, digest)

    def test_the_default_does_not_move_to_a_recipe_that_fails_a_gate(self) -> None:
        # b12's export is 34.6M gaussians against the 20M gate it inherits
        self.assertEqual(DEFAULT_PROFILE, "b5sky")
        self.assertEqual(PROFILE_B12OP05D3.acceptance["export_gaussian_count_max"], 20000000)
        ids = {item["id"] for item in PROFILE_B12OP05D3.open_questions}
        self.assertIn("export-count-gate", ids)


class RecipeDeltaTest(unittest.TestCase):
    def test_b12_is_b6reset_plus_the_delivered_ladder(self) -> None:
        before = _flat(thaw(PROFILE_B6RESET.trainer_base))
        after = _flat(thaw(PROFILE_B12OP05D3.trainer_base))
        changed = {key: after.get(key, "<absent>") for key in set(before) | set(after) if before.get(key) != after.get(key)}
        self.assertEqual(changed, B12_DELTA)

    def test_b13_differs_from_b12_only_in_the_dilation(self) -> None:
        b12 = _flat(thaw(PROFILE_B12OP05D3.trainer_base))
        b13 = _flat(thaw(PROFILE_B13OP05D0.trainer_base))
        self.assertEqual({k for k in b12 if b12[k] != b13[k]}, {"lidar_alpha_dilation_radius_px"})
        self.assertEqual(b13["lidar_alpha_dilation_radius_px"], 0)
        for section in ("tile_rules", "dataset_contract", "runtime", "acceptance", "cost_model"):
            self.assertEqual(PROFILE_B13OP05D0.section(section), PROFILE_B12OP05D3.section(section), section)

    def test_every_changed_knob_carries_its_own_justification(self) -> None:
        for key in B12_DELTA:
            with self.subTest(knob=key):
                self.assertIn(f"trainer_base.{key}", PROFILE_B12OP05D3.provenance)
        for key in ("tile_rules.cap_multiplier", "tile_rules.cap_ceiling", "dataset_contract.sky_mask_refinement",
                    "runtime.max_gaussians_per_gib_vram", "cost_model.tile_train_seconds_per_step"):
            with self.subTest(knob=key):
                self.assertIsNot(PROFILE_B12OP05D3.why(key), PROFILE_B6RESET.why(key))

    def test_both_are_registered(self) -> None:
        self.assertIs(PROFILES["b12op05d3"], PROFILE_B12OP05D3)
        self.assertIs(PROFILES["b13op05d0"], PROFILE_B13OP05D0)


class CapRuleTest(unittest.TestCase):
    def test_the_delivered_caps_reproduce(self) -> None:
        caps = {t.tile_id: tile_cap(PROFILE_B12OP05D3, t, vram_gib=CARD_GIB) for t in _house0305().tiles}
        self.assertEqual(caps, {0: 15_000_000, 1: 9_000_000, 2: 10_200_000, 3: 14_850_000})

    def test_b6reset_caps_and_where_tile2_came_from(self) -> None:
        # Tile_2: 1.756 x 3.31M = 5.81M is below the 6.39M it reached before, so the floor
        # takes 6.39M x 1.07 = 6.84M -> 6.8M; Tile_0 is the old SH1 VRAM clamp.
        caps = {t.tile_id: tile_cap(PROFILE_B6RESET, t, vram_gib=CARD_GIB) for t in _house0305().tiles}
        self.assertEqual(caps, {0: 10_900_000, 1: 6_000_000, 2: 6_800_000, 3: 9_900_000})

    def test_a_fresh_tile_gets_ratio_times_multiplier(self) -> None:
        tile = TileSummary(0, "Tile_0", 1000, 2_000_000)
        # 1.756 x 2.0M = 3.512M -> 3.5M, x1.5, not rounded again
        self.assertEqual(tile_cap(PROFILE_B12OP05D3, tile, vram_gib=CARD_GIB), 5_250_000)

    def test_the_measured_ceiling_holds_on_the_campaign_card(self) -> None:
        tile = TileSummary(0, "Tile_0", 1000, 10_000_000)
        self.assertEqual(tile_cap(PROFILE_B12OP05D3, tile), 15_000_000)
        self.assertEqual(tile_cap(PROFILE_B12OP05D3, tile, vram_gib=CARD_GIB), 15_000_000)

    def test_a_smaller_card_clamps_below_the_ceiling(self) -> None:
        tile = TileSummary(0, "Tile_0", 1000, 4_000_000)
        # 1.1M/GiB x 8 GiB x 0.88 = 7.744M -> 7.7M; the rule asks 7.0M x 1.5 = 10.5M
        self.assertEqual(tile_cap(PROFILE_B12OP05D3, tile, vram_gib=8.0), 7_700_000)

    def test_warnings_name_the_floor_and_the_ceiling_only_where_they_fire(self) -> None:
        warnings = _plan(PROFILE_B12OP05D3).warnings
        floor = [w for w in warnings if "floor rule" in w]
        ceiling = [w for w in warnings if "measured ceiling" in w]
        self.assertEqual([w.split(":")[0] for w in floor], ["Tile_2"])
        self.assertEqual([w.split(":")[0] for w in ceiling], ["Tile_0"])
        self.assertFalse([w for w in warnings if "VRAM ceiling" in w])

    def test_the_tile_config_records_the_rule_it_used(self) -> None:
        config = next(s.config for s in _plan(PROFILE_B12OP05D3).steps if s.name == "train_tile2_b12op05d3_delivery")
        self.assertEqual(config["cap_max"], 10_200_000)
        self.assertIn("x1.5", config["lineage"]["cap_rule"])
        self.assertIn("floor 6389532", config["lineage"]["cap_rule"])


class ViewsPerTileTest(unittest.TestCase):
    def test_house0305_is_inside_the_measured_range(self) -> None:
        self.assertFalse([w for w in _plan(PROFILE_B12OP05D3).warnings if "views-per-tile" in w])

    def test_a_crowded_tile_is_named_with_its_visit_count(self) -> None:
        dataset = DatasetSummary(
            scene_tag="big",
            tiles=(TileSummary(0, "Tile_0", 7202, 5_000_000), TileSummary(1, "Tile_1", 2000, 5_000_000)),
            train_view_count=24344,
            global_init_point_count=1_000_000,
        )
        root = Path("D:/sdk-test")
        plan = build_plan(PROFILE_B12OP05D3, dataset, dataset_root=root / "d", work_root=root / "w",
                          repo_root=root / "r", python=Path("python.exe"),
                          prior_tile_checkpoints={0: "a.pt", 1: "b.pt"})
        crowded = [w for w in plan.warnings if "views-per-tile" in w]
        self.assertEqual(len(crowded), 1)
        self.assertIn("1 tile(s)", crowded[0])
        self.assertIn("~4.2 times", crowded[0])

    def test_profiles_without_a_measured_range_say_nothing(self) -> None:
        self.assertNotIn("measured_views_per_tile", PROFILE_B6RESET.tiling)


class RefinedSkyLabelPlanTest(unittest.TestCase):
    def test_a_fresh_plan_builds_the_raw_label_then_refines_it_into_the_trainer_mask(self) -> None:
        plan = _plan(PROFILE_B12OP05D3)
        layout = WorkLayout(Path("D:/sdk-test/work"))
        names = [s.name for s in plan.steps]
        self.assertLess(names.index("sky_masks"), names.index("sky_masks_refined"))
        raw = next(s for s in plan.steps if s.name == "sky_masks")
        refined = next(s for s in plan.steps if s.name == "sky_masks_refined")
        self.assertEqual(raw.outputs, (str(layout.sky_mask_raw_manifest),))
        self.assertEqual(refined.outputs, (str(layout.sky_mask_manifest),))
        command = list(refined.command)
        self.assertTrue(command[1].endswith("refine_sky_masks.py"))
        self.assertEqual(command[command.index("--source-manifest") + 1], str(layout.sky_mask_raw_manifest))
        for flag, value in (("--dark-ratio", "0.75"), ("--edge-ratio", "0.1"), ("--dilate-px", "1")):
            self.assertEqual(command[command.index(flag) + 1], value)
        config = next(s.config for s in plan.steps if s.name == "train_tile0_b12op05d3_delivery")
        self.assertEqual(config["sky_supervision"]["mask_manifest"], str(layout.sky_mask_manifest))
        self.assertEqual(config["sky_supervision"]["mask_erosion_px"], 24)

    def test_a_supplied_refined_label_needs_no_raw_label(self) -> None:
        adopted = "C:/data/sky_mask_train_pr/sky_mask_train.json"
        plan = _plan(PROFILE_B12OP05D3, bundle_paths={"sky_mask_manifest": adopted,
                                                     "sky_mask_root": "C:/data/sky_mask_train_pr"})
        names = [s.name for s in plan.steps]
        self.assertNotIn("sky_masks", names)
        refined = next(s for s in plan.steps if s.name == "sky_masks_refined")
        self.assertEqual(refined.outputs, (adopted,))
        config = next(s.config for s in plan.steps if s.name == "train_tile1_b12op05d3_delivery")
        self.assertEqual(config["sky_supervision"]["mask_manifest"], adopted)

    def test_profiles_without_refinement_keep_the_single_sky_step(self) -> None:
        plan = _plan(PROFILE_B6RESET)
        layout = WorkLayout(Path("D:/sdk-test/work"))
        names = [s.name for s in plan.steps]
        self.assertNotIn("sky_masks_refined", names)
        raw = next(s for s in plan.steps if s.name == "sky_masks")
        self.assertEqual(raw.outputs, (str(layout.sky_mask_manifest),))


if __name__ == "__main__":
    unittest.main()
