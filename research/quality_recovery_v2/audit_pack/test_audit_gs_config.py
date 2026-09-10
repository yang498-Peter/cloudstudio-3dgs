"""Tests for the independent audit helper, NOT the repository's training suite."""
import copy
import unittest
from audit_gs_config import audit_config

BASE = {
    "run_id": "documented-tile0-key-fields-NOT-as-run-snapshot",
    "max_steps": 42640,
    "controlled_stop_after_steps": 20000,
    "learning_rates": {"means": 1.6e-5},
    "means_lr_final_factor": 0.01,
    "mcmc_refine_stop_iter": 14000,
    "densification_gradient_source": "total_loss",
    "exposure_compensation": {"enabled": True},
    "default_strategy": {
        "exact_mipmap_lifecycle": True, "refine_start_iter": 500,
        "refine_stop_iter": 14000, "refine_every": 100,
        "prune_switch_step": 21320, "reset_every": 300
    }
}

class AuditTests(unittest.TestCase):
    def test_tile0_schedule(self):
        r = audit_config(BASE)
        self.assertFalse(r["late_prune_threshold_reachable"])
        self.assertEqual(r["last_regular_refine_step"], 13900)
        self.assertEqual(r["regular_refine_event_count"], 135)
        self.assertAlmostEqual(r["nominal_lr_multiple_of_terminal_target"], 11.533486201476643)
        self.assertTrue({"S001", "S002", "G001", "E001"}.issubset({w["code"] for w in r["warnings"]}))

    def test_tile1_unreachable_despite_stop_after_switch(self):
        c = copy.deepcopy(BASE)
        c["max_steps"] = 36580
        c["default_strategy"]["prune_switch_step"] = 18290
        r = audit_config(c)
        self.assertFalse(r["late_prune_threshold_reachable"])
        self.assertAlmostEqual(r["nominal_lr_multiple_of_terminal_target"], 8.064193911650367)

    def test_real_20k_schedule(self):
        c = copy.deepcopy(BASE)
        c["max_steps"] = 20000
        c["default_strategy"]["prune_switch_step"] = 10000
        r = audit_config(c)
        self.assertTrue(r["late_prune_threshold_reachable"])
        self.assertLess(r["nominal_lr_multiple_of_terminal_target"], 1.001)
        self.assertNotIn("S001", {w["code"] for w in r["warnings"]})

    def test_post_refine_window(self):
        c = copy.deepcopy(BASE)
        c["controlled_stop_after_steps"] = 30000
        c["default_strategy"]["post_refine_cull_every"] = 100
        c["default_strategy"]["post_refine_cull_until"] = 23000
        self.assertTrue(audit_config(c)["late_prune_threshold_reachable"])
        c["default_strategy"]["post_refine_cull_until"] = 15000
        self.assertFalse(audit_config(c)["late_prune_threshold_reachable"])

    def test_duplicate_key_disagreement(self):
        c = copy.deepcopy(BASE)
        c["mcmc_refine_stop_iter"] = 21000
        self.assertIn("S003", {w["code"] for w in audit_config(c)["warnings"]})

    def test_invalid_horizon(self):
        c = copy.deepcopy(BASE)
        c["max_steps"] = 0
        with self.assertRaises(ValueError):
            audit_config(c)

    def test_inputs_not_mutated(self):
        c = copy.deepcopy(BASE)
        before = copy.deepcopy(c)
        audit_config(c)
        self.assertEqual(c, before)

    def test_normal_strategy_does_not_claim_exact_cull_semantics(self):
        c = copy.deepcopy(BASE)
        c["default_strategy"]["exact_mipmap_lifecycle"] = False
        self.assertIsNone(audit_config(c)["late_prune_threshold_reachable"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
