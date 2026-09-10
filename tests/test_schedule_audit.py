from __future__ import annotations

import re
import unittest
from pathlib import Path

from cloudstudio_3dgs.training.schedule_audit import (
    ADAPTER_DEFAULTS,
    TRAINER_DEFAULTS,
    diff_configs,
    lifecycle_events,
    means_lr_for_step,
    resolved_schedule,
    summarize_events,
)

ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = ROOT / "cloudstudio_3dgs" / "training" / "trainer.py"
ADAPTER_SOURCE = ROOT / "cloudstudio_3dgs" / "training" / "default_strategy_adapter.py"


def _synthetic_config(**overrides: object) -> dict:
    config = {
        "run_id": "synthetic",
        "max_steps": 1000,
        "controlled_stop_after_steps": 400,
        "sh_degree": 2,
        "sh_degree_interval": 100,
        "color_model": "sh",
        "view_sampling_mode": "fisher_yates_without_replacement_per_epoch",
        "learning_rates": {"means": 1e-2, "scales": 0.005, "quats": 0.001, "opacities": 0.05, "colors": 0.0025},
        "means_lr_final_factor": 0.01,
        "metric_scale_calibration": {"mode": "precomputed", "means_step_fraction": None},
        "mcmc_refine_start_iter": 100,
        "mcmc_refine_stop_iter": 700,
        "mcmc_refine_every": 100,
        "default_strategy": {
            "exact_mipmap_lifecycle": True,
            "lifecycle_execution_order": "pre_optimizer_vendor",
            "refine_start_iter": 100,
            "refine_stop_iter": 700,
            "refine_every": 100,
            "refine_scale2d_stop_iter": 700,
            "reset_every": 300,
            "prune_opa": 0.1,
            "prune_opa_late": 0.05,
            "prune_switch_step": 500,
            "vendor_cull_warmup_profile": "exact_0p10_to_0p05",
            "vendor_opacity_reset_profile": "exact_every300",
        },
    }
    config.update(overrides)
    return config


class MeansLrScheduleTests(unittest.TestCase):
    def test_formula_matches_trainer_source_text(self) -> None:
        # trainer.py imports torch transitively, so parity is nailed against
        # the source text of means_lr_for_step rather than by calling it.
        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "return float(base_learning_rate * (final_factor ** (step / max(1, max_steps))))",
            source,
        )
        self.assertIn(
            "max_steps=config.max_steps,",
            source[source.index("decayed = means_lr_for_step(") :][:400],
            "trainer must still use config.max_steps as the LR denominator",
        )

    def test_lr_at_last_executed_step_uses_max_steps_denominator(self) -> None:
        schedule = resolved_schedule(_synthetic_config(), view_count=50)
        nominal = schedule["means_lr"]["nominal"]
        self.assertEqual(schedule["steps"]["last_executed_step"], 399)
        self.assertAlmostEqual(nominal["start"], 1e-2)
        self.assertAlmostEqual(nominal["last_executed"], 1e-2 * 0.01 ** (399 / 1000))
        self.assertAlmostEqual(nominal["declared_final"], 1e-4)
        self.assertAlmostEqual(
            schedule["means_lr"]["nominal_last_executed_over_declared_final"],
            0.01 ** (399 / 1000) / 0.01,
        )
        self.assertFalse(
            next(c for c in schedule["consistency_checks"] if c["name"] == "controlled_stop_reaches_declared_final_lr")["ok"]
        )

    def test_effective_lr_follows_means_step_fraction(self) -> None:
        config = _synthetic_config(metric_scale_calibration={"mode": "precomputed"})
        schedule = resolved_schedule(config, view_count=50, reference_scale_m=0.01)
        effective = schedule["means_lr"]["effective"]
        self.assertAlmostEqual(effective["base"], 0.01 * TRAINER_DEFAULTS["means_step_fraction"])
        self.assertAlmostEqual(effective["last_executed"], effective["base"] * 0.01 ** (399 / 1000))
        without_reference = resolved_schedule(config, view_count=50)
        self.assertIsNone(without_reference["means_lr"]["effective"])
        self.assertIn("learning_rates_means_is_the_optimizer_base", [m["name"] for m in without_reference["mismatches"]])

    def test_post_refine_geometry_scale_applies_after_refine_stop(self) -> None:
        config = _synthetic_config(post_refine_geometry_lr_scale=0.5, controlled_stop_after_steps=900)
        schedule = resolved_schedule(config, view_count=50)
        nominal = schedule["means_lr"]["nominal"]
        self.assertAlmostEqual(nominal["refine_stop"], 0.5 * means_lr_for_step(1e-2, 0.01, step=700, max_steps=1000))
        self.assertAlmostEqual(nominal["last_executed"], 0.5 * means_lr_for_step(1e-2, 0.01, step=899, max_steps=1000))

    def test_pure_formula_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            means_lr_for_step(1e-2, 0.0, step=1, max_steps=10)
        with self.assertRaises(ValueError):
            means_lr_for_step(1e-2, 0.5, step=-1, max_steps=10)
        self.assertEqual(means_lr_for_step(0.0, 0.5, step=1, max_steps=10), 0.0)


class LateThresholdTests(unittest.TestCase):
    def test_never_reached_when_switch_after_refine_stop(self) -> None:
        strategy = dict(_synthetic_config()["default_strategy"], prune_switch_step=800)
        schedule = resolved_schedule(_synthetic_config(default_strategy=strategy, controlled_stop_after_steps=950), view_count=50)
        late = schedule["event_summary"]["late_threshold"]
        self.assertFalse(late["ever_applied"])
        self.assertIsNone(late["first_step"])
        names = [m["name"] for m in schedule["mismatches"]]
        self.assertIn("late_opacity_threshold_reachable", names)
        self.assertIn("prune_switch_step_is_half_max_steps", names)
        self.assertTrue(all(e["cull_opacity_threshold"] == 0.1 for e in schedule["events"]))

    def test_reached_when_switch_inside_refine_window(self) -> None:
        schedule = resolved_schedule(_synthetic_config(), view_count=50)
        late = schedule["event_summary"]["late_threshold"]
        self.assertFalse(late["ever_applied"], "stop at 400 precedes switch at 500")
        longer = resolved_schedule(_synthetic_config(controlled_stop_after_steps=700), view_count=50)
        late = longer["event_summary"]["late_threshold"]
        self.assertTrue(late["ever_applied"])
        self.assertEqual(late["first_step"], 500)
        self.assertEqual(late["event_count"], 2)  # steps 500 and 600
        self.assertNotIn("late_opacity_threshold_reachable", [m["name"] for m in longer["mismatches"]])

    def test_post_refine_cull_can_reach_late_threshold(self) -> None:
        strategy = dict(
            _synthetic_config()["default_strategy"],
            prune_switch_step=800,
            post_refine_cull_every=50,
            post_refine_cull_until=900,
        )
        schedule = resolved_schedule(_synthetic_config(default_strategy=strategy, controlled_stop_after_steps=950), view_count=50)
        late = schedule["event_summary"]["late_threshold"]
        self.assertTrue(late["ever_applied"])
        self.assertEqual(late["first_step"], 800)
        post = [e for e in schedule["events"] if e["kind"] == "post_refine_cull"]
        self.assertEqual([e["step"] for e in post], [700, 750, 800, 850, 900])
        self.assertTrue(all(not e["grow"] and not e["reset"] for e in post))


class DuplicateFieldTests(unittest.TestCase):
    def test_refine_stop_mismatch_is_flagged(self) -> None:
        config = _synthetic_config()
        config["default_strategy"]["refine_stop_iter"] = 800
        schedule = resolved_schedule(config, view_count=50)
        mismatch = next(m for m in schedule["mismatches"] if m["name"] == "refine_stop_iter")
        self.assertEqual(mismatch["fields"]["mcmc_refine_stop_iter"], 700)
        self.assertEqual(mismatch["fields"]["default_strategy.refine_stop_iter"], 800)
        # The adapter (and therefore the event table) follows default_strategy.
        self.assertEqual(schedule["lifecycle"]["refine_stop_iter"], 800)
        scale2d = next(c for c in schedule["consistency_checks"] if c["name"] == "refine_scale2d_stop_iter_matches_refine_stop")
        self.assertTrue(scale2d["ok"])

    def test_refine_every_mismatch_and_reset_profile_mismatch(self) -> None:
        config = _synthetic_config()
        config["default_strategy"]["refine_every"] = 200
        config["default_strategy"]["reset_every"] = 3000
        schedule = resolved_schedule(config, view_count=50)
        names = {m["name"] for m in schedule["mismatches"]}
        self.assertIn("refine_every", names)
        self.assertIn("reset_every_matches_vendor_opacity_reset_profile", names)

    def test_consistent_config_has_no_duplicate_mismatch(self) -> None:
        schedule = resolved_schedule(_synthetic_config(), view_count=50)
        duplicate_mismatches = [m for m in schedule["mismatches"] if m["kind"] == "duplicate_field"]
        self.assertEqual(duplicate_mismatches, [])

    def test_nested_absent_falls_back_to_top_level_like_backend_setdefault(self) -> None:
        config = _synthetic_config()
        for key in ("refine_start_iter", "refine_stop_iter", "refine_every"):
            config["default_strategy"].pop(key)
        schedule = resolved_schedule(config, view_count=50)
        self.assertEqual(schedule["lifecycle"]["refine_stop_iter"], 700)
        self.assertEqual(schedule["lifecycle"]["refine_every"], 100)
        self.assertNotIn("refine_stop_iter", [m["name"] for m in schedule["mismatches"]])

    def test_view_epoch_budget_check(self) -> None:
        schedule = resolved_schedule(_synthetic_config(), view_count=50)
        check = next(c for c in schedule["consistency_checks"] if c["name"] == "max_steps_is_20_view_epochs")
        self.assertTrue(check["ok"])
        self.assertAlmostEqual(schedule["views"]["average_visits_per_image"], 8.0)
        off = resolved_schedule(_synthetic_config(), view_count=51)
        self.assertIn("max_steps_is_20_view_epochs", [m["name"] for m in off["mismatches"]])

    def test_diff_configs_reports_leaf_paths(self) -> None:
        repo = {"a": 1, "nested": {"x": 1, "y": 2}, "only_repo": True}
        as_run = {"a": 2, "nested": {"x": 1, "y": 3}, "only_run": True}
        paths = {d["path"]: d["state"] for d in diff_configs(repo, as_run)}
        self.assertEqual(
            paths,
            {"a": "differs", "nested.y": "differs", "only_repo": "only_in_repo", "only_run": "only_in_as_run"},
        )
        self.assertEqual(diff_configs(repo, repo), [])


class EventCountTests(unittest.TestCase):
    def test_small_schedule_counts(self) -> None:
        events = lifecycle_events(
            stop_step=1500,
            refine_start_iter=100,
            refine_stop_iter=1000,
            refine_every=100,
            reset_every=300,
            exact_mipmap_lifecycle=True,
            prune_opa=0.1,
            prune_opa_late=0.05,
            prune_switch_step=750,
            post_refine_cull_every=None,
            post_refine_cull_until=None,
        )
        summary = summarize_events(events)
        self.assertEqual([e["step"] for e in events], [100, 200, 300, 400, 500, 600, 700, 800, 900])
        self.assertEqual(summary["grow"]["count"], 9)
        self.assertEqual(summary["cull"]["count"], 9)
        self.assertEqual(summary["reset"]["count"], 3)
        self.assertEqual([e["step"] for e in events if e["reset"]], [300, 600, 900])
        self.assertEqual(summary["grow"]["last_step"], 900)
        self.assertEqual(summary["late_threshold"]["first_step"], 800)
        self.assertEqual(summary["post_refine_cull"]["count"], 0)

    def test_non_exact_lifecycle_excludes_refine_start_step(self) -> None:
        events = lifecycle_events(
            stop_step=1000,
            refine_start_iter=100,
            refine_stop_iter=500,
            refine_every=100,
            reset_every=3000,
            exact_mipmap_lifecycle=False,
            prune_opa=ADAPTER_DEFAULTS["prune_opa"],
            prune_opa_late=None,
            prune_switch_step=None,
            post_refine_cull_every=None,
            post_refine_cull_until=None,
        )
        self.assertEqual([e["step"] for e in events], [200, 300, 400])
        self.assertTrue(all(e["threshold_phase"] == "early" for e in events))

    def test_opportunity_after_last_birth(self) -> None:
        schedule = resolved_schedule(_synthetic_config(controlled_stop_after_steps=950), view_count=50)
        opportunity = schedule["opportunity_after_last_birth"]
        self.assertEqual(opportunity["last_growth_step"], 600)
        self.assertEqual(opportunity["optimizer_steps_after_last_growth"], 350)
        self.assertAlmostEqual(opportunity["average_visits_per_image_after_last_growth"], 7.0)
        self.assertEqual(opportunity["last_reset_step"], 600)

    def test_sh_schedule_transitions(self) -> None:
        schedule = resolved_schedule(_synthetic_config(), view_count=50)
        sh = schedule["sh_degree"]
        self.assertTrue(sh["progressive"])
        self.assertEqual([t["step"] for t in sh["transitions"]], [0, 100, 200])
        self.assertEqual(sh["active_degree_at_last_executed_step"], 2)
        flat = resolved_schedule(_synthetic_config(sh_degree_interval=0, sh_degree=1), view_count=50)["sh_degree"]
        self.assertFalse(flat["progressive"])
        self.assertEqual(flat["active_degree_at_last_executed_step"], 1)


class SourceParityTests(unittest.TestCase):
    """Nail the transcribed adapter/trainer logic against the source text."""

    def test_adapter_early_return_and_refine_predicate(self) -> None:
        source = ADAPTER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("if step >= self.refine_stop_iter:\n            self._post_refine_cull(params, optimizers, state, step)\n            return", source)
        self.assertIn("reset = step % int(self.inner.reset_every) == 0", source)
        self.assertIn("if step >= int(self.prune_switch_step)", source)
        self.assertRegex(source, re.compile(r"step < self\.refine_stop_iter\s+and step >= self\.refine_start_iter\s+and step % self\.refine_every == 0"))

    def test_trainer_defaults_still_match(self) -> None:
        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        for field, expected in (
            ("mcmc_refine_start_iter", "500"),
            ("mcmc_refine_stop_iter", "25_000"),
            ("mcmc_refine_every", "100"),
            ("sh_degree_interval", "1000"),
            ("means_lr_final_factor", "1.0"),
            ("post_refine_geometry_lr_scale", "1.0"),
        ):
            self.assertRegex(source, re.compile(rf"^\s+{field}: [^=]+= {re.escape(expected)}\n", re.MULTILINE), field)
        self.assertIn('"prune_switch_step": self.max_steps // 2,', source)
        self.assertEqual(TRAINER_DEFAULTS["mcmc_refine_stop_iter"], 25_000)


if __name__ == "__main__":
    unittest.main()
