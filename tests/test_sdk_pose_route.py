"""The readiness gate is owed exactly when the trainer asks for it.

The gate chain starts from an independent AT report (tools/build_mipmap_frontend_gate.py), and
TrainerConfig.validate requires ``mipmap_pipeline_gate`` only for data whose dataset manifest
carries independent-AT lineage. The SDK used to require a gate for every capture, which left a
capture trained on its own S1Mapper poses (house0614 before any AT) with no way in. Pinned here:

* a dataset manifest with no AT lineage builds with no gate, records ``raw_capture_poses`` and
  hands the trainer a JSON null gate;
* AT lineage, or a manifest that cannot be read, still refuses without a gate;
* a prepare manifest may carry an empty gate path only when its scene records the raw route.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from cloudstudio3dgs_sdk.adopt import verify_prepare_manifest
from cloudstudio3dgs_sdk.bundle import POSE_ROUTE_AT, POSE_ROUTE_RAW, load_dataset_bundle
from cloudstudio3dgs_sdk.ingest.errors import DatasetIncompleteError
from cloudstudio3dgs_sdk.profile import PROFILE_B5SKY
from cloudstudio3dgs_sdk.project import StageRefused
from cloudstudio_3dgs.pipeline.mipmap_gate import INDEPENDENT_AT_ALGORITHM

from tests.test_sdk_bundle_build import FakePlan, _bundle, _graph


class PoseRouteBuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()
        self.work = self.root / "work"
        self.addCleanup(self._tmp.cleanup)

    def _runner(self, command):
        if "build_lidar_init.py" in command[1]:
            out = pathlib.Path(command[command.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "sparse_pc.ply").write_bytes(b"ply")
            (out / "lidar_init_geometry.npz").write_bytes(b"npz")
        return 0

    def _build(self, *, lineage, gate=None, write_manifest=True):
        specs = _graph(self.work)
        if write_manifest:
            manifest = self.work / "dataset" / "dataset_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            payload = {"manifest_sha256": "a" * 64, "images": []}
            if lineage:
                payload["training_lineage"] = {"independent_at_algorithm_version": INDEPENDENT_AT_ALGORITHM}
            manifest.write_text(json.dumps(payload), encoding="utf-8")
        plan = FakePlan(specs, present=[s.name for s in specs])
        with mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", lambda path, **kw: _bundle(self.root)), \
             mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", return_value=plan), \
             mock.patch("cloudstudio_3dgs.pipeline.mipmap_gate.load_and_verify_gate", return_value=({}, "sha")):
            return load_dataset_bundle(
                self.root / "capture", PROFILE_B5SKY, self.work, python="python",
                repo_root=self.root / "repo", pipeline_gate=gate, runner=self._runner,
            )

    def test_raw_capture_poses_build_with_no_gate(self):
        scene = self._build(lineage=False)
        self.assertIsNone(scene.pipeline_gate)
        self.assertEqual(scene.pose_route, POSE_ROUTE_RAW)
        self.assertIsNone(scene.trainer_paths()["mipmap_pipeline_gate"])
        self.assertEqual(scene.as_json()["pose_route"], POSE_ROUTE_RAW)

    def test_independent_at_data_still_needs_a_gate(self):
        with self.assertRaises(DatasetIncompleteError) as caught:
            self._build(lineage=True)
        self.assertIn("--pipeline-gate", str(caught.exception))
        scene = self._build(lineage=True, gate=self.root / "gate.json")
        self.assertEqual(scene.pose_route, POSE_ROUTE_AT)
        self.assertEqual(scene.trainer_paths()["mipmap_pipeline_gate"], str(self.root / "gate.json"))

    def test_an_unreadable_manifest_counts_as_at_data(self):
        with self.assertRaises(DatasetIncompleteError):
            self._build(lineage=False, write_manifest=False)


class PrepareManifestGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.file = self.root / "dataset_manifest.json"
        self.file.write_text("{}", encoding="utf-8")

    def _payload(self, route):
        return {
            "dataset": {},
            "trainer_paths": {"dataset_manifest": str(self.file), "mipmap_pipeline_gate": None},
            "derived_paths": {},
            "digests": {},
            "scene": {"pose_route": route},
        }

    def test_an_empty_gate_path_is_refused_unless_the_scene_records_the_raw_route(self):
        with self.assertRaises(StageRefused) as caught:
            verify_prepare_manifest(self._payload(POSE_ROUTE_AT), manifest_path=self.root / "m.json")
        self.assertIn("mipmap_pipeline_gate is empty", str(caught.exception))
        # On the raw route the gate is not a missing file; whatever the manifest is refused for
        # next, it is not that.
        try:
            verify_prepare_manifest(self._payload(POSE_ROUTE_RAW), manifest_path=self.root / "m.json")
        except StageRefused as error:
            self.assertNotIn("mipmap_pipeline_gate", str(error))


if __name__ == "__main__":
    unittest.main()
