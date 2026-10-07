"""The fresh-dataset build path (cloudstudio3dgs_sdk/bundle.py), CPU only.

``load_dataset_bundle`` used to be a stub, so the SDK could prepare no dataset at all. It now
runs the ingest layer's cache graph, CPU half only, and projects the result onto the trainer's
path contract. What is pinned, against a fake adapter and a fake cache plan so nothing touches
disk or a GPU:

* a capture with no LiDAR cloud is refused by name - the recipe cannot run without one;
* CPU caches are built in dependency order and each build is asked for by name;
* the first GPU cache whose inputs are ready stops the run with its exact command, and caches
  that do not depend on it are still built first;
* without a pipeline gate the build refuses and names the gate tools, because forging the
  readiness contract is not this module's job;
* with everything present, every ingest cache lands on the right PreparedScene field, tile
  ownership is keyed by tile id, and the global initialisation is built at the profile's
  decimation.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import cloudstudio3dgs_sdk.bundle as bundle_module
from cloudstudio3dgs_sdk.bundle import PreparedScene, load_dataset_bundle
from cloudstudio3dgs_sdk.ingest.caches import CPU, GPU, STATUS_MISSING, STATUS_PRESENT
from cloudstudio3dgs_sdk.ingest.errors import DatasetIncompleteError, GpuStepRequired
from cloudstudio3dgs_sdk.profile import PROFILE_B5SKY, PROFILE_B12OP05D3


class FakeSpec:
    def __init__(self, name, device, root, manifest, *, depends_on=(), tile_id=None):
        self.name = name
        self.device = device
        self.root = root
        self.manifest = manifest
        self.command = ("python", f"build_{name}.py", "--output", str(manifest))
        self.depends_on = tuple(depends_on)
        self.tile_id = tile_id


class FakePlan:
    """Statuses come back in the order given; ``build`` records what was asked for."""

    def __init__(self, statuses, present=()):
        self._statuses = statuses
        self.present = set(present)
        self.built = []

    def statuses(self):
        out = []
        for spec in self._statuses:
            done = spec.name in self.present or spec.name in self.built
            out.append(
                SimpleNamespace(
                    spec=spec,
                    status=STATUS_PRESENT if done else STATUS_MISSING,
                    reason="present" if done else "missing",
                    must_build=not done,
                )
            )
        return out

    def build(self, *, dry_run, only, runner):
        assert dry_run is False
        for name in only:
            self.built.append(name)
        return []


def _graph(work):
    caches = work / "caches"
    runs = work / "runs"
    dataset = work / "dataset"
    specs = [
        FakeSpec("dataset_manifest", CPU, dataset, dataset / "dataset_manifest.json"),
        FakeSpec("mask_manifest", CPU, dataset / "masks", dataset / "masks" / "mask_manifest.json",
                 depends_on=("dataset_manifest",)),
        FakeSpec("person_mask_manifest", GPU, dataset / "person_masks",
                 dataset / "person_masks" / "person_mask_manifest.json",
                 depends_on=("mask_manifest",)),
        FakeSpec("depth_cache", CPU, dataset / "depth", dataset / "depth" / "depth_manifest.json",
                 depends_on=("dataset_manifest",)),
        FakeSpec("split_manifest", CPU, caches, caches / "split_manifest.json",
                 depends_on=("dataset_manifest",)),
        FakeSpec("face_cache", CPU, caches / "face4_train", caches / "face4_train" / "face_manifest.json",
                 depends_on=("dataset_manifest",)),
        FakeSpec("renderer_mask", CPU, caches / "face4_train", caches / "renderer_mask_train.json",
                 depends_on=("face_cache",)),
        FakeSpec("face_lidar_geometry", CPU, caches / "face4_lidar_train_vis6",
                 caches / "face4_lidar_train_vis6" / "face_lidar_geometry_manifest.json",
                 depends_on=("face_cache", "depth_cache")),
        FakeSpec("mono_depth", GPU, caches / "da2_train", caches / "da2_train" / "mono_depth_manifest.json",
                 depends_on=("face_cache",)),
        FakeSpec("sky_masks", CPU, caches / "sky_mask_train", caches / "sky_mask_train" / "sky_mask_train.json",
                 depends_on=("face_cache",)),
        FakeSpec("tile_plan", CPU, runs / "tile_plan", runs / "tile_plan" / "adaptive_tile_plan.json"),
        FakeSpec("tile_inputs", CPU, runs / "tile_inputs", runs / "tile_inputs" / "tile_inputs_manifest.json",
                 depends_on=("tile_plan",)),
        FakeSpec("tile_geometry", CPU, runs / "tile_geometry",
                 runs / "tile_geometry" / "tile_geometry_manifest.json", depends_on=("tile_inputs",)),
        FakeSpec("tile_ownership_0", CPU, runs / "tile_ownership" / "Tile_0",
                 runs / "tile_ownership" / "Tile_0" / "tile_ownership_manifest.json",
                 depends_on=("tile_inputs", "face_lidar_geometry"), tile_id=0),
        FakeSpec("tile_ownership_1", CPU, runs / "tile_ownership" / "Tile_1",
                 runs / "tile_ownership" / "Tile_1" / "tile_ownership_manifest.json",
                 depends_on=("tile_inputs", "face_lidar_geometry"), tile_id=1),
    ]
    return specs


def _bundle(root, *, cloud=True):
    return SimpleNamespace(
        adapter="fake",
        images=(1, 2, 3),
        point_cloud=SimpleNamespace(path=root / "cloud.las") if cloud else None,
        source_root=root,
        dataset_id="scene_x",
    )


class LoadDatasetBundleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()
        self.work = self.root / "work"
        self.addCleanup(self._tmp.cleanup)
        self.calls = []

    def _runner(self, command):
        self.calls.append(tuple(command))
        # the global init builder writes its two outputs
        if "build_lidar_init.py" in command[1]:
            out = pathlib.Path(command[command.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "sparse_pc.ply").write_bytes(b"ply")
            (out / "lidar_init_geometry.npz").write_bytes(b"npz")
        return 0

    def _run(self, plan, *, cloud=True, gate=None, **extra):
        self.load_calls = []

        def fake_load(path, *, adapter=None, **kwargs):
            self.load_calls.append((path, adapter, kwargs))
            return _bundle(self.root, cloud=cloud)

        with mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", fake_load), \
             mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", return_value=plan), \
             mock.patch("cloudstudio_3dgs.pipeline.mipmap_gate.load_and_verify_gate", return_value=({}, "sha")):
            return load_dataset_bundle(
                self.root / "capture", PROFILE_B5SKY, self.work,
                python="python", repo_root=self.root / "repo", pipeline_gate=gate, runner=self._runner,
                **extra,
            )

    def test_the_adapter_is_detected_unless_named_and_run_dir_is_passed_only_when_given(self):
        specs = _graph(self.work)
        plan = FakePlan(specs, present=[s.name for s in specs])
        self._run(plan, gate=self.root / "gate.json")
        self.assertEqual(self.load_calls, [(self.root / "capture", None, {})],
                         "no run_dir kwarg at all when none was given: adapters without one must not see it")
        self._run(plan, gate=self.root / "gate.json", adapter="s1_fisheye", run_dir=self.root / "processed")
        self.assertEqual(self.load_calls, [(self.root / "capture", "s1_fisheye", {"run_dir": self.root / "processed"})])

    def test_the_validation_caches_the_battery_reads_are_built_too(self):
        # evaluate_probe_views reads face4_val / renderer_mask_val / face4_lidar_val_* derived by
        # name from the training caches; prepare used to build the train split only, so a fresh
        # capture reached the battery with no validation caches (project.py refuses that up front).
        train = FakePlan(_graph(self.work), present=[s.name for s in _graph(self.work)])
        val_specs = _graph(self.work)
        # in the val graph the shared caches are present; the split-specific ones are not
        shared = [s.name for s in val_specs if s.name not in ("face_cache", "renderer_mask", "face_lidar_geometry")]
        val = FakePlan(val_specs, present=shared)
        calls = []

        def fake_plan_caches(bundle, profile, **kwargs):
            calls.append(kwargs.get("split", "train"))
            return val if kwargs.get("split") == "val" else train

        with mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", lambda path, **kw: _bundle(self.root)), \
             mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", fake_plan_caches), \
             mock.patch("cloudstudio_3dgs.pipeline.mipmap_gate.load_and_verify_gate", return_value=({}, "sha")):
            load_dataset_bundle(
                self.root / "capture", PROFILE_B5SKY, self.work, python="python",
                repo_root=self.root / "repo", pipeline_gate=self.root / "gate.json", runner=self._runner,
            )
        self.assertEqual(calls, ["train", "val"])
        self.assertEqual(train.built, [])
        self.assertEqual(sorted(val.built), ["face_cache", "face_lidar_geometry", "renderer_mask"])
        # dependency order inside the val graph
        self.assertLess(val.built.index("face_cache"), val.built.index("renderer_mask"))

    def test_a_split_capture_reads_poses_and_lidar_from_the_run_dir(self):
        # house0614 ships the recording (camera/, info/) and the S1Mapper output (ImgPose.txt,
        # colorized.las) as two folders. The cache graph and the global initialisation must read
        # the processed folder, not the recording, or the LiDAR init points at a folder with no
        # cloud in it.
        specs = _graph(self.work)
        plan = FakePlan(specs, present=[s.name for s in specs])
        processed = self.root / "processed"

        def fake_load(path, *, adapter=None, **kwargs):
            return _bundle(self.root / "capture")

        with mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", return_value=plan) as plan_caches, \
             mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", fake_load), \
             mock.patch("cloudstudio_3dgs.pipeline.mipmap_gate.load_and_verify_gate", return_value=({}, "sha")):
            load_dataset_bundle(
                self.root / "capture", PROFILE_B5SKY, self.work,
                python="python", repo_root=self.root / "repo", pipeline_gate=self.root / "gate.json",
                runner=self._runner, adapter="s1_fisheye", run_dir=processed,
            )
        kwargs = plan_caches.call_args.kwargs
        self.assertEqual(kwargs["recording_root"], self.root / "capture")
        self.assertEqual(kwargs["source_run_dir"], processed)
        init = [c for c in self.calls if "build_lidar_init.py" in c[1]]
        self.assertEqual(len(init), 1)
        self.assertEqual(init[0][init[0].index("--run") + 1], str(processed))

    def test_a_capture_without_a_cloud_is_refused_by_name(self):
        plan = FakePlan(_graph(self.work))
        with self.assertRaises(DatasetIncompleteError) as caught:
            self._run(plan, cloud=False)
        self.assertIn("no LiDAR point cloud", str(caught.exception))
        self.assertEqual(plan.built, [], "nothing may be built for a capture that cannot run")

    def test_cpu_caches_build_in_dependency_order_until_a_gpu_cache_is_next(self):
        plan = FakePlan(_graph(self.work))
        with self.assertRaises(GpuStepRequired) as caught:
            self._run(plan, gate=self.root / "gate.json")
        # person masks (GPU) depend on mask_manifest; everything else that does not depend on a
        # GPU cache was built before stopping
        built = plan.built
        self.assertIn("dataset_manifest", built)
        self.assertIn("mask_manifest", built)
        self.assertIn("face_cache", built)
        self.assertIn("tile_plan", built)
        self.assertIn("tile_inputs", built)
        self.assertLess(built.index("dataset_manifest"), built.index("mask_manifest"))
        self.assertLess(built.index("face_cache"), built.index("sky_masks"))
        self.assertNotIn("person_mask_manifest", built)
        self.assertNotIn("mono_depth", built)
        # the first GPU cache whose inputs were ready is the one named, with its command
        self.assertIn("person_mask_manifest", str(caught.exception))
        self.assertIn("build_person_mask_manifest.py", str(caught.exception))

    def test_without_a_gate_the_build_refuses_and_names_the_gate_tools(self):
        plan = FakePlan(_graph(self.work), present=[s.name for s in _graph(self.work)])
        with self.assertRaises(DatasetIncompleteError) as caught:
            self._run(plan, gate=None)
        self.assertIn("advance_mipmap_da2_gate.py", str(caught.exception))
        self.assertIn("--pipeline-gate", str(caught.exception))

    def test_a_complete_graph_projects_onto_the_trainer_path_contract(self):
        specs = _graph(self.work)
        plan = FakePlan(specs, present=[s.name for s in specs])
        scene = self._run(plan, gate=self.root / "gate.json")
        self.assertIsInstance(scene, PreparedScene)
        by_name = {s.name: s for s in specs}
        self.assertEqual(scene.dataset_manifest, by_name["dataset_manifest"].manifest)
        self.assertEqual(scene.mask_root, by_name["mask_manifest"].root)
        self.assertEqual(scene.face_cache_manifest, by_name["face_cache"].manifest)
        self.assertEqual(scene.face_cache_root, by_name["face_cache"].root)
        self.assertEqual(scene.mono_depth_manifest, by_name["mono_depth"].manifest)
        self.assertEqual(scene.tile_inputs_root, by_name["tile_inputs"].root)
        self.assertEqual(scene.tile_geometry_manifest, by_name["tile_geometry"].manifest)
        self.assertEqual(scene.lidar_cloud, self.root / "cloud.las")
        self.assertEqual(scene.pipeline_gate, self.root / "gate.json")
        self.assertEqual(scene.scene_tag, "scene_x")
        self.assertEqual(sorted(scene.caches.tile_ownership), [0, 1])
        self.assertEqual(scene.caches.tile_ownership[1][0], by_name["tile_ownership_1"].manifest)
        self.assertEqual(scene.caches.sky_mask_manifest, by_name["sky_masks"].manifest)
        # the global initialisation was built at the profile's decimation
        init_calls = [c for c in self.calls if "build_lidar_init.py" in c[1]]
        self.assertEqual(len(init_calls), 1)
        self.assertIn("--voxel-size", init_calls[0])
        self.assertEqual(init_calls[0][init_calls[0].index("--voxel-size") + 1],
                         str(float(PROFILE_B5SKY.coarse_prior["init_decimation_m"])))
        self.assertEqual(scene.global_init_ply, self.work / "caches" / "global_init" / "sparse_pc.ply")

    def test_a_refining_profile_trains_against_the_refined_label(self):
        caches = self.work / "caches"
        specs = _graph(self.work) + [
            FakeSpec("sky_masks_refined", CPU, caches / "sky_mask_train_refined",
                     caches / "sky_mask_train_refined" / "sky_mask_train.json",
                     depends_on=("sky_masks", "face_cache")),
        ]
        plan = FakePlan(specs, present=[s.name for s in specs])

        def fake_load(path, *, adapter=None, **kwargs):
            return _bundle(self.root)

        with mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", fake_load), \
             mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", return_value=plan) as planned, \
             mock.patch("cloudstudio_3dgs.pipeline.mipmap_gate.load_and_verify_gate", return_value=({}, "sha")):
            scene = load_dataset_bundle(
                self.root / "capture", PROFILE_B12OP05D3, self.work,
                python="python", repo_root=self.root / "repo", pipeline_gate=self.root / "gate.json",
                runner=self._runner,
            )
        self.assertEqual(scene.caches.sky_mask_manifest, specs[-1].manifest)
        self.assertEqual(scene.caches.sky_mask_root, specs[-1].root)
        # the cache layer is told the profile's refinement; it does not read Profile itself
        for call in planned.call_args_list:
            self.assertEqual(call.kwargs["sky_mask_refinement"],
                             PROFILE_B12OP05D3.dataset_contract["sky_mask_refinement"])

    def test_an_existing_global_init_is_not_rebuilt(self):
        specs = _graph(self.work)
        plan = FakePlan(specs, present=[s.name for s in specs])
        init = self.work / "caches" / "global_init"
        init.mkdir(parents=True)
        (init / "sparse_pc.ply").write_bytes(b"ply")
        (init / "lidar_init_geometry.npz").write_bytes(b"npz")
        self._run(plan, gate=self.root / "gate.json")
        self.assertEqual([c for c in self.calls if "build_lidar_init.py" in c[1]], [])


if __name__ == "__main__":
    unittest.main()
