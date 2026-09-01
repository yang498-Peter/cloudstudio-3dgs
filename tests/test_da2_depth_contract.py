"""The monocular-depth supervision knobs must validate and must only change a
config's signed contract when a config actually asks for them, so every
existing signed profile keeps its signature."""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional for the CPU suite
    HAS_TORCH = False

from cloudstudio_3dgs.training.trainer import TrainerConfig


def _base(**overrides):
    fields = dict(
        run_id="da2",
        dataset_manifest=Path("d.json"),
        recording_root=Path("r"),
        mask_manifest=Path("m.json"),
        mask_root=Path("m"),
        split_manifest=Path("s.json"),
        initialization_ply=Path("i.ply"),
        output_dir=Path("o"),
        gsplat_lock=Path("l.json"),
        require_person_masks=False,
        lidar_range_weight=0.0,
    )
    fields.update(overrides)
    return TrainerConfig(**fields)


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class Da2DepthContractTests(unittest.TestCase):
    def test_defaults_leave_the_contract_unchanged(self) -> None:
        config = _base(da2_depth_weight=0.15)
        config.validate()
        self.assertIsNone(config.mono_depth_max_range_m)
        self.assertEqual(config.da2_depth_space, "linear")
        self.assertNotIn("da2_depth_contract", config.contract_dict()["loss_weights"])

    def test_knobs_enter_the_contract_only_when_da2_is_active(self) -> None:
        active = _base(
            da2_depth_weight=0.15,
            mono_depth_max_range_m=30.0,
            da2_depth_space="compressed",
        )
        active.validate()
        self.assertEqual(
            active.contract_dict()["loss_weights"]["da2_depth_contract"],
            {"max_range_m": 30.0, "space": "compressed"},
        )
        inactive = _base(mono_depth_max_range_m=30.0, da2_depth_space="compressed")
        inactive.validate()
        self.assertNotIn(
            "da2_depth_contract", inactive.contract_dict()["loss_weights"]
        )

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "da2_depth_space"):
            _base(da2_depth_space="metres").validate()
        with self.assertRaisesRegex(ValueError, "mono_depth_max_range_m"):
            _base(mono_depth_max_range_m=0.0).validate()


if __name__ == "__main__":
    unittest.main()
