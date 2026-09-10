"""Research schedule contract: pure resolver, trainer wiring, gate policy, generated S0/S1 arms."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unittest
from pathlib import Path

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    ORDERED_STAGES,
    RESEARCH_SCHEDULE_CONTRACT_GATE_POLICY,
    UPSTREAM_DATA_READY_STATUS,
    advance_adaptive_growth_gate,
    sign_gate,
)
from cloudstudio_3dgs.training.schedule_audit import (
    RESEARCH_CONTRACT_DECLARABLE_FIELDS,
    RESEARCH_REFINE_STOP_MAX_FRACTION,
    RESEARCH_SCHEDULE_CONTRACT_V1,
    RESEARCH_SCHEDULE_CONTRACTS,
    TRAINER_DEFAULTS,
    research_schedule_contract,
    resolved_schedule,
    validate_research_schedule_contract,
)
try:
    from cloudstudio_3dgs.training.trainer import TrainerConfig
except ImportError:  # torch is an optional training dependency
    TrainerConfig = None

ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = ROOT / "cloudstudio_3dgs" / "training" / "trainer.py"
ARM_DIR = ROOT / "run_configs" / "house0305_tiles" / "v9"
BASE_ARM = ARM_DIR / "tile0_R1_range0_20k.json"
S0_ARM = ARM_DIR / "tile0_S0_control.json"
S1_ARM = ARM_DIR / "tile0_S1_rescaled20k.json"
# Runtime facts of the tile0 R1 run (research/quality_recovery_v2/02_schedule_audit.json).
TILE0_VIEW_COUNT = 2132
TILE0_REFERENCE_SCALE_M = 0.007767821662127972
# Mismatches the historical control reproduces (02_schedule_audit.json).
S0_EXPECTED_MISMATCHES = [
    "controlled_stop_reaches_declared_final_lr",
    "learning_rates_means_is_the_optimizer_base",
    "late_opacity_threshold_reachable",
]


def _contract_config(**overrides: object) -> dict:
    """Synthetic H=1000 config that satisfies research_rescaled_horizon_v1."""
    config = {
        "run_id": "synthetic-contract",
        "max_steps": 1000,
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
        "schedule_contract": RESEARCH_SCHEDULE_CONTRACT_V1,
    }
    config.update(overrides)
    return config


def _strategy(**overrides: object) -> dict:
    return dict(_contract_config()["default_strategy"], **overrides)


def _violation_names(config: dict) -> list[str]:
    return [check["name"] for check in research_schedule_contract(config)["violations"]]


def _trainer_dict(**overrides: object) -> dict:
    """Minimal TrainerConfig input carrying the contract; no files, no exact lifecycle."""
    config = {
        "run_id": "contract-trainer",
        "dataset_manifest": "dataset.json",
        "recording_root": "recording",
        "mask_manifest": "masks.json",
        "mask_root": "masks",
        "split_manifest": "split.json",
        "initialization_ply": "sparse_pc.ply",
        "output_dir": "run",
        "gsplat_lock": "upstream/cloudstudio_trainer.lock.json",
        "require_person_masks": False,
        "lidar_range_weight": 0.0,
        "max_steps": 1000,
        "checkpoint_every": 500,
        "sh_degree": 2,
        "sh_degree_interval": 100,
        "color_model": "sh",
        "view_sampling_mode": "fisher_yates_without_replacement_per_epoch",
        "learning_rates": {"means": 1e-2, "scales": 0.005, "quats": 0.001, "opacities": 0.05, "colors": 0.0025},
        "means_lr_final_factor": 0.01,
        "metric_scale_calibration": {"mode": "knn", "means_step_fraction": None, "noise_std_fraction": None},
        "mcmc_refine_start_iter": 100,
        "mcmc_refine_stop_iter": 700,
        "mcmc_refine_every": 100,
        "densification_strategy": "default_3dgs",
        "topology_policy": {"mode": "adaptive_growth"},
        "default_strategy": {
            "refine_start_iter": 100,
            "refine_stop_iter": 700,
            "refine_every": 100,
            "refine_scale2d_stop_iter": 700,
            "reset_every": 300,
            "prune_opa": 0.1,
            "prune_opa_late": 0.05,
            "prune_switch_step": 500,
        },
        "schedule_contract": RESEARCH_SCHEDULE_CONTRACT_V1,
    }
    config.update(overrides)
    return config


class ContractAcceptedTests(unittest.TestCase):
    def test_consistent_config_resolves_with_no_violations(self) -> None:
        record = validate_research_schedule_contract(_contract_config(), training_view_count=50, holdout_view_count=5)
        self.assertEqual(record["name"], RESEARCH_SCHEDULE_CONTRACT_V1)
        self.assertFalse(record["competitor_parity"])
        self.assertEqual(record["horizon_steps"], 1000)
        self.assertEqual(record["stop_step"], 1000)
        self.assertEqual(record["violations"], [])
        self.assertEqual(
            record["horizon_fractions"],
            {
                "refine_start": 0.1,
                "refine_stop": 0.7,
                "refine_stop_max": RESEARCH_REFINE_STOP_MAX_FRACTION,
                "refine_scale2d_stop": 0.7,
                "prune_switch": 0.5,
                "reset_every": 0.3,
                "sh_full_degree_step": 0.2,
            },
        )
        self.assertEqual(record["multiples_of_refine_every"], {"refine_start": 1.0, "refine_stop": 7.0, "reset_every": 3.0})
        self.assertEqual(record["means_lr"]["authoritative_field"], "learning_rates.means")
        self.assertAlmostEqual(record["means_lr"]["declared_final"], 1e-4)
        self.assertAlmostEqual(record["means_lr"]["at_last_executed_step"], 1e-2 * 0.01 ** (999 / 1000))
        self.assertTrue(record["late_threshold"]["ever_applied"])
        self.assertEqual(record["late_threshold"]["first_step"], 500)
        self.assertEqual(record["views"]["training_view_count"], 50)
        self.assertAlmostEqual(record["views"]["epochs_over_training_views"], 20.0)
        self.assertAlmostEqual(record["views"]["epochs_over_tile_views"], 1000 / 55)
        self.assertEqual(record["views"]["parity_20_epoch_max_steps"], 1100)
        self.assertEqual(sorted(record["resolved_fields"]), sorted(RESEARCH_CONTRACT_DECLARABLE_FIELDS))
        # The inherited resolved_schedule checks travel with the record; the
        # parity epoch rule is the one check the contract replaces.
        inherited = {check["name"] for check in record["checks"] if check["kind"] != "research_schedule_contract"}
        self.assertIn("prune_switch_step_is_half_max_steps", inherited)
        self.assertIn("late_opacity_threshold_reachable", inherited)
        self.assertIn("reset_cadence_aligns_with_refine_cadence", inherited)
        self.assertNotIn("max_steps_is_20_view_epochs", inherited)

    def test_controlled_stop_may_restate_the_horizon(self) -> None:
        record = validate_research_schedule_contract(_contract_config(controlled_stop_after_steps=1000))
        self.assertEqual(record["stop_step"], 1000)
        self.assertEqual(record["controlled_stop_after_steps"], 1000)

    def test_declared_fields_are_verified_against_resolved(self) -> None:
        declared = {"resolved": {"horizon_steps": 1000, "refine_stop_iter": 700, "prune_switch_step": 500, "means_lr_base": 1e-2}}
        record = validate_research_schedule_contract(_contract_config(schedule_contract_fields=declared))
        self.assertIn("declared_fields_match_resolved", [c["name"] for c in record["checks"]])
        wrong = {"resolved": {"refine_stop_iter": 600}}
        self.assertEqual(_violation_names(_contract_config(schedule_contract_fields=wrong)), ["declared_fields_match_resolved"])
        with self.assertRaisesRegex(ValueError, "undeclarable"):
            research_schedule_contract(_contract_config(schedule_contract_fields={"resolved": {"cap_max": 1}}))

    def test_late_threshold_reachable_through_post_refine_cull(self) -> None:
        # refine_stop at 0.4 H puts the H/2 switch outside the growth window;
        # only an enabled post-refine cull makes prune_opa_late reachable.
        strategy = _strategy(refine_stop_iter=400, refine_scale2d_stop_iter=400)
        config = _contract_config(default_strategy=strategy, mcmc_refine_stop_iter=400)
        self.assertEqual(_violation_names(config), ["late_opacity_threshold_reachable"])
        strategy.update(post_refine_cull_every=100, post_refine_cull_until=900)
        record = validate_research_schedule_contract(_contract_config(default_strategy=strategy, mcmc_refine_stop_iter=400))
        self.assertEqual(record["late_threshold"]["first_step"], 500)
        self.assertGreater(record["event_summary"]["post_refine_cull"]["count"], 0)


class ContractRejectedTests(unittest.TestCase):
    def test_missing_or_unknown_contract_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "declares no schedule_contract"):
            research_schedule_contract(_contract_config(schedule_contract=None))
        with self.assertRaisesRegex(ValueError, "unknown schedule_contract"):
            research_schedule_contract(_contract_config(schedule_contract="competitor_parity_20_epochs"))
        self.assertEqual(RESEARCH_SCHEDULE_CONTRACTS, (RESEARCH_SCHEDULE_CONTRACT_V1,))

    def test_truncating_controlled_stop_is_a_violation(self) -> None:
        names = _violation_names(_contract_config(controlled_stop_after_steps=400))
        self.assertIn("stop_step_is_horizon", names)
        self.assertIn("controlled_stop_reaches_declared_final_lr", names)
        with self.assertRaisesRegex(ValueError, "stop_step_is_horizon"):
            validate_research_schedule_contract(_contract_config(controlled_stop_after_steps=400))

    def test_refine_stop_beyond_settling_fraction(self) -> None:
        strategy = _strategy(refine_stop_iter=800, refine_scale2d_stop_iter=800)
        names = _violation_names(_contract_config(default_strategy=strategy, mcmc_refine_stop_iter=800))
        self.assertEqual(names, ["refine_window_inside_horizon"])
        # A config may declare a different ceiling, but only a valid one.
        relaxed = _contract_config(
            default_strategy=strategy, mcmc_refine_stop_iter=800, schedule_contract_fields={"refine_stop_fraction_max": 0.8}
        )
        self.assertEqual(validate_research_schedule_contract(relaxed)["horizon_fractions"]["refine_stop_max"], 0.8)
        with self.assertRaisesRegex(ValueError, "refine_stop_fraction_max"):
            research_schedule_contract(_contract_config(schedule_contract_fields={"refine_stop_fraction_max": 1.5}))

    def test_refine_bounds_must_align_to_refine_every(self) -> None:
        strategy = _strategy(refine_stop_iter=650, refine_scale2d_stop_iter=650)
        names = _violation_names(_contract_config(default_strategy=strategy, mcmc_refine_stop_iter=650))
        self.assertEqual(names, ["refine_window_is_refine_every_aligned"])

    def test_scale2d_stop_must_be_declared_and_equal(self) -> None:
        self.assertEqual(
            _violation_names(_contract_config(default_strategy=_strategy(refine_scale2d_stop_iter=600))),
            ["refine_scale2d_stop_iter_matches_refine_stop", "refine_scale2d_stop_declared_and_equal"],
        )
        strategy = _strategy()
        strategy.pop("refine_scale2d_stop_iter")
        self.assertEqual(_violation_names(_contract_config(default_strategy=strategy)), ["refine_scale2d_stop_declared_and_equal"])

    def test_prune_switch_and_late_threshold_must_be_declared(self) -> None:
        # The trainer's exact-lifecycle rule (switch == H // 2) is inherited unchanged.
        self.assertEqual(
            _violation_names(_contract_config(default_strategy=_strategy(prune_switch_step=400))),
            ["prune_switch_step_is_half_max_steps"],
        )
        strategy = _strategy()
        strategy.pop("prune_switch_step")
        strategy.pop("prune_opa_late")
        # Dropping the late threshold also breaks the inherited vendor cull profile check.
        self.assertEqual(
            _violation_names(_contract_config(default_strategy=strategy)),
            ["opacity_thresholds_match_vendor_cull_warmup_profile", "late_threshold_declared"],
        )

    def test_reset_cadence_must_be_a_refine_multiple_and_fire(self) -> None:
        self.assertEqual(
            _violation_names(_contract_config(default_strategy=_strategy(reset_every=250, vendor_opacity_reset_profile="deferred_every3000_compatibility"))),
            ["reset_every_matches_vendor_opacity_reset_profile", "reset_cadence_aligns_with_refine_cadence"],
        )
        self.assertEqual(
            _violation_names(_contract_config(default_strategy=_strategy(reset_every=3000, vendor_opacity_reset_profile="deferred_every3000_compatibility"))),
            ["reset_fires_inside_refine_window"],
        )

    def test_sh_full_degree_must_arrive_before_horizon(self) -> None:
        self.assertEqual(_violation_names(_contract_config(sh_degree_interval=600)), ["sh_full_degree_reached_before_horizon"])
        record = validate_research_schedule_contract(_contract_config(sh_degree_interval=0))
        self.assertEqual(record["horizon_fractions"]["sh_full_degree_step"], 0.0)

    def test_means_step_fraction_must_be_explicitly_null(self) -> None:
        absent = _contract_config(metric_scale_calibration={"mode": "precomputed"})
        self.assertEqual(
            _violation_names(absent),
            ["learning_rates_means_is_the_optimizer_base", "means_step_fraction_explicitly_null"],
        )
        self.assertEqual(
            _violation_names(_contract_config(metric_scale_calibration={"mode": "precomputed", "means_step_fraction": 0.0032})),
            ["learning_rates_means_is_the_optimizer_base", "means_step_fraction_explicitly_null"],
        )
        self.assertEqual(TRAINER_DEFAULTS["means_step_fraction"], 0.0032, "the default the contract must override")


@unittest.skipUnless(TrainerConfig is not None, "torch is an optional training dependency")
class DefaultBehaviourUnchangedTests(unittest.TestCase):
    def test_resolved_schedule_without_contract_keeps_parity_epoch_check(self) -> None:
        config = _contract_config()
        config.pop("schedule_contract")
        schedule = resolved_schedule(config, view_count=50)
        names = [check["name"] for check in schedule["consistency_checks"]]
        self.assertIn("max_steps_is_20_view_epochs", names)
        self.assertNotIn("max_steps_is_contract_horizon", names)
        with_contract = resolved_schedule(_contract_config(), view_count=50)
        names = [check["name"] for check in with_contract["consistency_checks"]]
        self.assertNotIn("max_steps_is_20_view_epochs", names)
        self.assertIn("max_steps_is_contract_horizon", names)
        self.assertEqual(with_contract["mismatches"], [])

    def test_trainer_without_contract_is_untouched(self) -> None:
        plain = _trainer_dict()
        plain.pop("schedule_contract")
        config = TrainerConfig.from_dict(plain)
        config.validate()
        self.assertIsNone(config.schedule_contract)
        self.assertIsNone(config.schedule_contract_fields)
        self.assertNotIn("schedule_contract", config.contract_dict())
        with self.assertRaisesRegex(ValueError, "between zero and max_steps"):
            TrainerConfig.from_dict(dict(plain, controlled_stop_after_steps=1000)).validate()
        with self.assertRaisesRegex(ValueError, "requires a named schedule_contract"):
            TrainerConfig.from_dict(dict(plain, schedule_contract_fields={"arm": "S1"})).validate()

    def test_parity_pre_flight_and_lifecycle_rules_survive_in_source(self) -> None:
        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "and config.max_steps != 20 * (len(trainset) + holdout_view_count)",
            source,
            "the 20-epoch parity pre-flight must remain the default branch",
        )
        self.assertRegex(
            source,
            re.compile(r"if config\.schedule_contract is not None:.*?schedule_contract_as_run\.json.*?\n    elif \(\n        config\.view_sampling_mode", re.DOTALL),
            "the contract branch must be the explicit alternative to the parity pre-flight",
        )
        self.assertIn('"prune_switch_step": self.max_steps // 2,', source)
        self.assertIn('"refine_scale2d_stop_iter": self.mcmc_refine_stop_iter,', source)


@unittest.skipUnless(TrainerConfig is not None, "torch is an optional training dependency")
class TrainerWiringTests(unittest.TestCase):
    def test_validate_resolves_and_signs_the_contract(self) -> None:
        config = TrainerConfig.from_dict(_trainer_dict())
        config.validate()
        audit = config.schedule_audit_dict()
        self.assertIsNone(audit["metric_scale_calibration"]["means_step_fraction"])
        self.assertIn("means_step_fraction", audit["metric_scale_calibration"])
        record = config.contract_dict()["schedule_contract"]
        self.assertEqual(record["name"], RESEARCH_SCHEDULE_CONTRACT_V1)
        self.assertEqual(record["violations"], [])
        self.assertEqual(record["late_threshold"]["first_step"], 500)
        self.assertIsNone(record["views"]["training_view_count"], "static resolution carries no runtime facts")
        # contract_dict is what the run manifest signs: it must be canonical JSON.
        canonical_json_bytes(record)

    def test_validate_rejects_a_violated_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "research schedule contract .* violated: .*means_step_fraction_explicitly_null"):
            TrainerConfig.from_dict(_trainer_dict(metric_scale_calibration={"mode": "knn"})).validate()
        with self.assertRaisesRegex(ValueError, "stop_step_is_horizon"):
            TrainerConfig.from_dict(_trainer_dict(controlled_stop_after_steps=400)).validate()

    def test_controlled_stop_equal_to_horizon_is_accepted_only_under_contract(self) -> None:
        config = TrainerConfig.from_dict(_trainer_dict(controlled_stop_after_steps=1000))
        config.validate()
        self.assertEqual(config.contract_dict()["schedule_contract"]["controlled_stop_after_steps"], 1000)

    def test_declared_fields_pass_through_to_the_trainer(self) -> None:
        declared = {"resolved": {"horizon_steps": 1000, "prune_switch_step": 500}}
        config = TrainerConfig.from_dict(_trainer_dict(schedule_contract_fields=declared))
        config.validate()
        self.assertEqual(config.schedule_contract_fields, declared)
        wrong = {"resolved": {"horizon_steps": 2000}}
        with self.assertRaisesRegex(ValueError, "declared_fields_match_resolved"):
            TrainerConfig.from_dict(_trainer_dict(schedule_contract_fields=wrong)).validate()


class GatePolicyTests(unittest.TestCase):
    @staticmethod
    def _upstream_data_gate() -> dict:
        sha = {name: value * 64 for name, value in (("dataset", "1"), ("split", "2"), ("face", "3"), ("mask", "4"), ("da2", "5"), ("tile", "6"))}
        return sign_gate(
            {
                "schema_version": GATE_SCHEMA_VERSION,
                "profile": GATE_PROFILE,
                "status": UPSTREAM_DATA_READY_STATUS,
                "training_allowed": False,
                "completed_stages": list(ORDERED_STAGES[:15]),
                "next_required_stage": ORDERED_STAGES[15],
                "blocking_reasons": ["training implementation is not aligned"],
                "bindings": {
                    "training_dataset_manifest_sha256": sha["dataset"],
                    "split_manifest_sha256": sha["split"],
                    "face4_train_manifest_sha256": sha["face"],
                    "face4_val_manifest_sha256": sha["face"],
                    "training_circle_mask_manifest_sha256": sha["mask"],
                    "da2_train_manifest_sha256": sha["da2"],
                    "spatial_tile_plan_manifest_sha256": sha["tile"],
                },
            }
        )

    @staticmethod
    def _signed(config: dict) -> dict:
        signed = copy.deepcopy(config)
        signed.pop("config_manifest_sha256", None)
        signed["config_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(signed)).hexdigest()
        return signed

    def test_adaptive_growth_gate_refuses_to_sign_a_research_contract(self) -> None:
        parity = {
            "run_id": "v26-boundary",
            "mipmap_tile_id": 1,
            "topology_policy": {"mode": "adaptive_growth"},
            "densification_strategy": "default_3dgs",
            "densification_gradient_source": "rgb_only",
            "mcmc_refine_start_iter": 500,
            "mcmc_refine_every": 100,
            "mcmc_refine_stop_iter": 5610,
            "mcmc_noise_injection_stop_iter": 0,
            "mcmc_noise_lr": 0.0,
            "max_steps": 7480,
            "controlled_stop_after_steps": 502,
            "factor": 1,
            "cap_max": 2200000,
            "sh_degree": 0,
            "da2_depth_weight": 0.0,
            "error_weighted_sampling": {"enabled": False},
            "default_strategy": {
                "exact_mipmap_lifecycle": True,
                "grow_grad2d": 0.00015,
                "growth_min_opacity": 0.15,
                "split_scale_m": 0.2,
                "prune_scale_m": 0.2,
                "prune_opa": 0.1,
                "prune_opa_late": 0.05,
                "prune_switch_step": 3740,
                "prune_scale2d": 0.15,
                "reset_every": 300,
                "reset_opacity_cap": 0.2,
                "absgrad": True,
                "revised_opacity": True,
            },
            "tangent_proposal": {"enabled": True, "reject_unsupported_births": True},
            "geometry_regularization": {
                "enabled": True,
                "opacity_sparsity_weight": 1e-4,
                "scale_upper_weight": 1e-4,
                "anisotropy_weight": 1e-4,
                "max_scale_ratio_to_reference": 8.0,
                "max_anisotropy": 10.0,
            },
        }
        gate = advance_adaptive_growth_gate(self._upstream_data_gate(), self._signed(parity), stage="boundary")
        self.assertTrue(gate["training_allowed"])
        research = dict(parity, schedule_contract=RESEARCH_SCHEDULE_CONTRACT_V1)
        with self.assertRaisesRegex(ValueError, "competitor-parity schedules only"):
            advance_adaptive_growth_gate(self._upstream_data_gate(), self._signed(research), stage="boundary")
        # Signature still verified first: a tampered contract config is a signature error, not a policy one.
        tampered = self._signed(research)
        tampered["max_steps"] = 7481
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            advance_adaptive_growth_gate(self._upstream_data_gate(), tampered, stage="boundary")

    def test_gate_policy_names_every_contract(self) -> None:
        self.assertEqual(set(RESEARCH_SCHEDULE_CONTRACT_GATE_POLICY), set(RESEARCH_SCHEDULE_CONTRACTS))
        self.assertEqual(RESEARCH_SCHEDULE_CONTRACT_GATE_POLICY[RESEARCH_SCHEDULE_CONTRACT_V1], "not_applicable_refused_by_adaptive_growth_gate")


@unittest.skipUnless(TrainerConfig is not None, "torch is an optional training dependency")
class GeneratedArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = json.loads(BASE_ARM.read_text(encoding="utf-8"))
        cls.s0 = json.loads(S0_ARM.read_text(encoding="utf-8"))
        cls.s1 = json.loads(S1_ARM.read_text(encoding="utf-8"))

    def test_s0_is_the_base_relabelled(self) -> None:
        expected = dict(self.base, run_id="house0305-t0-S0-control", output_dir=r"C:\Peter\3dgs-runs\house0305_sop\tile0_S0_control")
        self.assertEqual(self.s0, expected)
        self.assertNotIn("schedule_contract", self.s0)

    def test_s0_reproduces_the_audited_mismatches(self) -> None:
        schedule = resolved_schedule(self.s0, TILE0_VIEW_COUNT, reference_scale_m=TILE0_REFERENCE_SCALE_M)
        self.assertEqual([m["name"] for m in schedule["mismatches"]], S0_EXPECTED_MISMATCHES)
        self.assertEqual(schedule["steps"]["stop_step"], 20000)
        self.assertEqual(schedule["steps"]["max_steps"], 42640)
        self.assertAlmostEqual(schedule["means_lr"]["nominal_last_executed_over_declared_final"], 11.533486, places=5)
        self.assertFalse(schedule["event_summary"]["late_threshold"]["ever_applied"])

    def test_s1_passes_the_contract_and_the_audit(self) -> None:
        self.assertEqual(self.s1["schedule_contract"], RESEARCH_SCHEDULE_CONTRACT_V1)
        record = validate_research_schedule_contract(self.s1, training_view_count=TILE0_VIEW_COUNT, holdout_view_count=0)
        self.assertEqual(record["violations"], [])
        self.assertEqual(record["horizon_steps"], 20000)
        self.assertEqual(record["late_threshold"]["first_step"], 10000)
        self.assertAlmostEqual(record["views"]["epochs_over_training_views"], 20000 / TILE0_VIEW_COUNT)
        self.assertEqual(record["views"]["parity_20_epoch_max_steps"], 42640)
        schedule = resolved_schedule(self.s1, TILE0_VIEW_COUNT)
        self.assertEqual(schedule["mismatches"], [])
        self.assertAlmostEqual(schedule["means_lr"]["nominal_last_executed_over_declared_final"], 1.0, places=3)
        self.assertEqual(schedule["means_lr"]["effective_note"], "means_step_fraction is null: optimizer uses learning_rates.means")
        self.assertEqual(schedule["event_summary"]["grow"]["last_step"], 13900)
        self.assertEqual(schedule["event_summary"]["reset"]["count"], 45)

    def test_s1_moves_exactly_the_linked_fields(self) -> None:
        fields = self.s1["schedule_contract_fields"]
        self.assertEqual(
            fields["changed_fields"],
            [
                "controlled_stop_after_steps",
                "default_strategy.prune_switch_step",
                "learning_rates.means",
                "max_steps",
                "metric_scale_calibration.means_step_fraction",
                "output_dir",
                "run_id",
            ],
        )
        self.assertTrue(set(fields["changed_fields"]) <= set(fields["linked_fields"]))
        for path in ("default_strategy.refine_stop_iter", "default_strategy.refine_scale2d_stop_iter", "default_strategy.reset_every", "sh_degree_interval"):
            self.assertIn(path, fields["linked_fields"])
            self.assertFalse(fields["moved_from_base"][path]["changed"], path)
        self.assertEqual(self.s1["max_steps"], 20000)
        self.assertNotIn("controlled_stop_after_steps", self.s1)
        self.assertEqual(self.s1["default_strategy"]["prune_switch_step"], 10000)
        self.assertEqual(self.s1["default_strategy"]["refine_stop_iter"], 14000)
        self.assertEqual(self.s1["mcmc_refine_stop_iter"], 14000)
        self.assertIsNone(self.s1["metric_scale_calibration"]["means_step_fraction"])
        self.assertAlmostEqual(self.s1["learning_rates"]["means"], 0.0032 * TILE0_REFERENCE_SCALE_M)
        self.assertEqual(fields["provenance"]["base_config_sha256"], hashlib.sha256(BASE_ARM.read_bytes()).hexdigest())
        self.assertEqual(fields["resolved"]["prune_switch_step"], 10000)
        # Everything outside the linked fields is byte-identical to the base.
        stripped = {k: v for k, v in self.s1.items() if k not in {"schedule_contract", "schedule_contract_fields"}}
        differing = set()
        for key in set(stripped) | set(self.base):
            if stripped.get(key, "<absent>") != self.base.get(key, "<absent>"):
                differing.add(key)
        self.assertEqual(
            differing,
            {"run_id", "output_dir", "max_steps", "controlled_stop_after_steps", "default_strategy", "learning_rates", "metric_scale_calibration"},
        )

    def test_s1_loads_through_the_trainer_when_its_inputs_are_present(self) -> None:
        # Full TrainerConfig.validate() touches the signed manifests the arm
        # references (renderer masks, geometry, tile inputs, gate); it is
        # machine-conditional evidence, skipped where those inputs are absent.
        required = [self.s1[key] for key in ("renderer_mask_manifest", "face_cache_manifest", "initialization_geometry_manifest", "tile_inputs_manifest", "mipmap_pipeline_gate", "dataset_manifest")]
        if not all(Path(path).is_file() for path in required):
            self.skipTest("tile0 R1 inputs are not present on this machine")
        config = TrainerConfig.from_dict(self.s1)
        config.validate()
        record = config.contract_dict()["schedule_contract"]
        self.assertEqual(record["violations"], [])
        self.assertEqual(record["resolved_fields"], self.s1["schedule_contract_fields"]["resolved"])
        control = TrainerConfig.from_dict(self.s0)
        control.validate()
        self.assertNotIn("schedule_contract", control.contract_dict())


if __name__ == "__main__":
    unittest.main()
