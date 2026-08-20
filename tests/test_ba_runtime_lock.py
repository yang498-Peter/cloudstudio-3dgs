from __future__ import annotations

import json
import unittest
from pathlib import Path

from cloudstudio_3dgs.ba.runtime_lock import (
    load_runtime_lock,
    runtime_lock_sha256,
    verify_signed_runtime_manifest,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


class BaRuntimeLockTests(unittest.TestCase):
    def test_runtime_manifest_signature_detects_tampering(self) -> None:
        import hashlib

        manifest = {"schema_version": 1, "value": 4}
        manifest["runtime_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest()
        self.assertEqual(
            verify_signed_runtime_manifest(
                manifest, "runtime_manifest_sha256"
            ),
            manifest["runtime_manifest_sha256"],
        )
        manifest["value"] = 5
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_signed_runtime_manifest(manifest, "runtime_manifest_sha256")

    def test_repository_lock_has_exact_vcs_commits_and_licenses(self) -> None:
        path = Path(__file__).resolve().parents[1] / "upstream" / "rig_ba.lock.json"
        lock = load_runtime_lock(path)

        self.assertEqual(len(runtime_lock_sha256(lock)), 64)
        self.assertEqual(lock["components"]["hloc"]["license"], "Apache-2.0")
        self.assertEqual(
            lock["components"]["lightglue"]["license"], "Apache-2.0"
        )
        self.assertEqual(lock["components"]["aliked"]["license"], "BSD-3-Clause")
        self.assertFalse(lock["runtime_policy"]["skip_geometric_verification"])

    def test_checked_in_real_baseline_keeps_real_ba_gate_open(self) -> None:
        path = Path(__file__).resolve().parents[1] / "baselines" / "gs2_rig_ba.baseline.json"
        baseline = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(baseline["real_match_graph"]["validation_images_used"], 0)
        self.assertTrue(baseline["synthetic_pycolmap_stage_1"]["candidate_accepted"])
        self.assertEqual(
            baseline["acceptance"][
                "real_reprojection_p50_improvement_at_least_30_percent"
            ],
            "not_run",
        )
        self.assertFalse(baseline["acceptance"]["real_candidate_accepted"])


if __name__ == "__main__":
    unittest.main()
