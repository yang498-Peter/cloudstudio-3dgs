from __future__ import annotations

import unittest

from cloudstudio_3dgs.training.topology_policy import (
    FixedTopologyScheduleConfig,
    TopologyPolicyConfig,
    topology_count_transition,
)
from cloudstudio_3dgs.training.trainer import TrainerConfig


def _trainer_config(**overrides):
    value = {
        "run_id": "fixed-topology-contract",
        "trainer_preset": "custom",
        "dataset_manifest": "dataset.json",
        "recording_root": "recording",
        "mask_manifest": "masks.json",
        "mask_root": "masks",
        "split_manifest": "split.json",
        "initialization_ply": "init.ply",
        "output_dir": "run",
        "gsplat_lock": "upstream/gsplat.lock.json",
        "require_person_masks": False,
        "lidar_range_weight": 0.0,
        "max_steps": 20,
        "topology_policy": {"mode": "strict_fixed"},
        "fixed_topology_schedule": {
            "enabled": True,
            "phase_a_steps": 5,
            "phase_b_steps": 10,
            "phase_b_geometry_lr_scale": 0.25,
            "phase_c_geometry_lr_scale": 0.10,
            "audit_steps": [1, 5, 15, 20],
        },
    }
    value.update(overrides)
    return TrainerConfig.from_dict(value)


class TopologyPolicyTests(unittest.TestCase):
    def test_trainer_contract_records_inactive_strategy_and_three_phases(self) -> None:
        config = _trainer_config()
        config.validate()
        contract = config.contract_dict()
        self.assertEqual(
            contract["algorithm_version"],
            "cloudstudio_gsplat_trainer_v8_fixed_topology",
        )
        self.assertFalse(contract["strategy"]["active"])
        self.assertEqual(contract["topology_policy"]["mode"], "strict_fixed")
        self.assertEqual(
            contract["fixed_topology_schedule"]["audit_steps"], [1, 5, 15, 20]
        )

    def test_trainer_rejects_birth_controls_in_fixed_topology(self) -> None:
        config = _trainer_config(tangent_proposal={"enabled": True})
        with self.assertRaisesRegex(ValueError, "no-birth topology"):
            config.validate()

    def test_strict_fixed_has_no_strategy_or_prune(self) -> None:
        policy = TopologyPolicyConfig(mode="strict_fixed")
        policy.validate(max_steps=100)
        self.assertFalse(policy.strategy_enabled)
        topology_count_transition(
            mode=policy.mode, before_count=10, after_count=10, prune_due=False
        )
        with self.assertRaisesRegex(RuntimeError, "changed Gaussian count"):
            topology_count_transition(
                mode=policy.mode, before_count=10, after_count=9, prune_due=False
            )

    def test_opacity_prune_only_allows_one_decrease_and_never_growth(self) -> None:
        policy = TopologyPolicyConfig(
            mode="opacity_prune_only",
            opacity_prune_step=20,
            opacity_prune_threshold=0.02,
        )
        policy.validate(max_steps=100)
        topology_count_transition(
            mode=policy.mode, before_count=10, after_count=7, prune_due=True
        )
        with self.assertRaisesRegex(RuntimeError, "created Gaussians"):
            topology_count_transition(
                mode=policy.mode, before_count=10, after_count=11, prune_due=True
            )
        with self.assertRaisesRegex(RuntimeError, "outside its prune step"):
            topology_count_transition(
                mode=policy.mode, before_count=10, after_count=9, prune_due=False
            )

    def test_phase_schedule_freezes_geometry_then_enables_gap_diagnostic(self) -> None:
        schedule = FixedTopologyScheduleConfig(
            enabled=True,
            phase_a_steps=5,
            phase_b_steps=10,
            phase_b_geometry_lr_scale=0.25,
            phase_c_geometry_lr_scale=0.10,
            audit_steps=(1, 5, 15, 20),
        )
        schedule.validate(max_steps=20, topology_mode="strict_fixed")
        self.assertEqual(schedule.phase_for_step(0)["geometry_lr_scale"], 0.0)
        self.assertEqual(schedule.phase_for_step(5)["name"], "B_LIDAR_GEOMETRY")
        self.assertEqual(schedule.phase_for_step(15)["name"], "C_GAP_DIAGNOSTIC")
        self.assertTrue(schedule.phase_for_step(15)["emit_gap_analysis"])

    def test_schedule_rejects_growth_and_unsorted_audit_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "no-birth"):
            FixedTopologyScheduleConfig(enabled=True).validate(
                max_steps=10, topology_mode="adaptive_growth"
            )
        with self.assertRaisesRegex(ValueError, "unique, sorted"):
            FixedTopologyScheduleConfig(
                enabled=True, audit_steps=(2, 1)
            ).validate(max_steps=10, topology_mode="strict_fixed")


if __name__ == "__main__":
    unittest.main()
