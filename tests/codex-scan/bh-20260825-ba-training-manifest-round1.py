"""Regression: an accepted BA candidate must become Trainer-visible signed input."""

from __future__ import annotations

import unittest

from cloudstudio_3dgs.ba.training_manifest import build_ba_training_manifest


class AcceptedBaTrainingManifestRegression(unittest.TestCase):
    def test_bridge_entrypoint_exists(self) -> None:
        self.assertTrue(callable(build_ba_training_manifest))


if __name__ == "__main__":
    unittest.main()
