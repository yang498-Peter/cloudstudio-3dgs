from __future__ import annotations

import hashlib
import json
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReproducibleBaselineTests(unittest.TestCase):
    def test_locked_patch_hash_and_encoding(self) -> None:
        lock = json.loads(
            (ROOT / "upstream" / "gsplat.lock.json").read_text(encoding="utf-8")
        )
        patch_bytes = (ROOT / lock["patch"]).read_bytes()

        self.assertFalse(patch_bytes.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf")))
        self.assertNotIn(b"\r\n", patch_bytes)
        self.assertTrue(patch_bytes.decode("utf-8").startswith("diff --git "))
        self.assertEqual(hashlib.sha256(patch_bytes).hexdigest(), lock["patch_sha256"])
        self.assertRegex(lock["commit"], r"^[0-9a-f]{40}$")

    def test_smoke_baseline_fails_closed_on_point_budget(self) -> None:
        with (ROOT / "configs" / "smoke_8gb.toml").open("rb") as fh:
            config = tomllib.load(fh)
        baseline = json.loads(
            (ROOT / "baselines" / "gs2_smoke.baseline.json").read_text(encoding="utf-8")
        )

        self.assertFalse(config["normalize_world_space"])
        self.assertGreater(baseline["dataset"]["expected_init_points"], config["cap_max"])
        self.assertIn("blocked_by_init_point_budget", baseline["training"]["status"])

    def test_uv_lock_matches_supported_python(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as fh:
            project = tomllib.load(fh)
        with (ROOT / "uv.lock").open("rb") as fh:
            lock = tomllib.load(fh)

        self.assertEqual(project["project"]["requires-python"], ">=3.12,<3.13")
        self.assertEqual(lock["requires-python"], "==3.12.*")


if __name__ == "__main__":
    unittest.main()
