"""A thinned initialization that keeps its coverage: config rules and contract (CPU only).

The scale growth itself runs inside ``train()`` on the signed per-point geometry; what is
pinned here is that the knob cannot be set without a stride, that it is recorded, and that
leaving it off keeps a recorded arm's contract byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import unittest

AS_RUN = pathlib.Path(r"C:/Peter/3dgs-runs/house0305_sop/tile1_B5_cap6_20k/config_as_run.json")


@unittest.skipUnless(AS_RUN.is_file(), "needs the recorded tile1_B5_cap6_20k run directory")
class ThinnedInitializationContractTest(unittest.TestCase):
    def _config(self, **overrides):
        from cloudstudio_3dgs.training.trainer import TrainerConfig

        raw = json.loads(AS_RUN.read_text(encoding="utf-8"))
        raw.update(overrides)
        return TrainerConfig.from_dict(raw)

    def test_off_keeps_the_recorded_contract(self) -> None:
        from cloudstudio_3dgs.data.manifest import canonical_json_bytes

        config = self._config()
        config.validate()
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(config.contract_dict())).hexdigest(),
            "7d4139bc7450b3d43929180031841399f2f132a0bcdd22d355bc2c5c99a1c971",
        )

    def test_coverage_needs_a_stride(self) -> None:
        with self.assertRaises(ValueError):
            self._config(initialization_subsample_keeps_coverage=True).validate()

    def test_stride_and_coverage_are_recorded(self) -> None:
        config = self._config(
            initialization_subsample_stride=3, initialization_subsample_keeps_coverage=True
        )
        config.validate()
        initialization = config.contract_dict()["initialization"]
        self.assertEqual(initialization["subsample_stride"], 3)
        self.assertTrue(initialization["subsample_keeps_coverage"])
        plain = self._config(initialization_subsample_stride=3).contract_dict()["initialization"]
        self.assertNotIn("subsample_keeps_coverage", plain)


if __name__ == "__main__":
    unittest.main()
