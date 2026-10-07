"""prepare runs the GPU caches of a fresh capture itself (W4).

Ingestion builds the CPU half of the cache graph and stops at the first GPU cache that is due
(person masks, Depth Anything V2). It used to stop the whole prepare stage there and print the
command for a person to run. On a CUDA host the SDK now runs it under the work root's GPU
lease and calls back in; what is pinned:

* each due GPU cache runs once, under the lease, then ingestion is called again;
* without a GPU, with a weight path the host never named, with the lease held elsewhere, or
  with a cache that ran and is still due, prepare refuses by name instead of looping;
* weight paths come from the caller or the environment, never from the profile;
* ingestion leaves ownership masks and per-view backgrounds to the plan (the ingest versions
  carried a ``<trainer_config>`` placeholder / an input nothing produced) and passes the
  weight paths to the cache graph.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cloudstudio3dgs_sdk.bundle import load_dataset_bundle
from cloudstudio3dgs_sdk.ingest.errors import GpuStepRequired
from cloudstudio3dgs_sdk.project import GPU_CACHE_ASSET_ENV, StageFailed, StageRefused
from cloudstudio3dgs_sdk.requirements import GpuInfo, Probes
from tests import test_sdk_project as _project_tests

DA2_COMMAND = ("python", "tools/build_da2_face_cache.py", "--checkpoint", "C:/models/da2.pth", "--output", "x")


class _Fixture(_project_tests.ProjectFixture):
    def ingest(self, outcomes, **project_kwargs):
        """Drive ``ingest_fresh`` with a fake ingestion that raises/returns ``outcomes`` in turn."""
        calls = []
        queue = list(outcomes)

        def fake_load(*args, **kwargs):
            calls.append(kwargs)
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        project = self.project(dataset=None, **project_kwargs)
        with mock.patch("cloudstudio3dgs_sdk.project.load_dataset_bundle", fake_load), \
             mock.patch("cloudstudio3dgs_sdk.project.summarize_prepared_scene", return_value=self.dataset):
            result = project.ingest_fresh()
        return project, result, calls


class GpuCacheRunTest(_Fixture):
    def test_a_due_gpu_cache_runs_under_the_lease_and_ingestion_resumes(self) -> None:
        scene = _project_tests.fake_scene(self.dataset_root)
        due = GpuStepRequired("mono_depth needs the GPU", cache="mono_depth", command=DA2_COMMAND)
        project, result, calls = self.ingest([due, scene])
        self.assertEqual(self.runner.calls, ["gpu_cache_mono_depth"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result, self.dataset)
        self.assertTrue(project.layout.prepare_manifest.is_file())
        # the lease was released
        self.assertFalse((project.layout.runs / "gpu.lock").exists())

    def test_two_gpu_caches_run_one_after_the_other(self) -> None:
        scene = _project_tests.fake_scene(self.dataset_root)
        person = GpuStepRequired("p", cache="person_mask_manifest", command=("python", "person.py"))
        depth = GpuStepRequired("d", cache="mono_depth", command=DA2_COMMAND)
        self.ingest([person, depth, scene])
        self.assertEqual(self.runner.calls, ["gpu_cache_person_mask_manifest", "gpu_cache_mono_depth"])

    def test_no_gpu_refuses_with_the_command(self) -> None:
        no_card = Probes("3.12", None, GpuInfo(False, detail="no card"), lambda p: 10 ** 15)
        due = GpuStepRequired("mono_depth needs the GPU", cache="mono_depth", command=DA2_COMMAND)
        with self.assertRaises(StageRefused) as caught:
            self.ingest([due], probes=no_card)
        self.assertIn("build_da2_face_cache.py", str(caught.exception))
        self.assertEqual(self.runner.calls, [])

    def test_a_cache_that_ran_and_is_still_due_is_not_run_again(self) -> None:
        due = GpuStepRequired("mono_depth needs the GPU", cache="mono_depth", command=DA2_COMMAND)
        again = GpuStepRequired("mono_depth needs the GPU", cache="mono_depth", command=DA2_COMMAND)
        with self.assertRaises(StageRefused) as caught:
            self.ingest([due, again])
        self.assertIn("ran but its output still does not verify", str(caught.exception))
        self.assertEqual(self.runner.calls, ["gpu_cache_mono_depth"])

    def test_an_unnamed_weight_path_is_refused_with_the_flag_and_variable(self) -> None:
        command = ("python", "tools/build_person_masks.py", "--weights", "<person_weights>")
        due = GpuStepRequired("p", cache="person_mask_manifest", command=command)
        with self.assertRaises(StageRefused) as caught:
            self.ingest([due])
        self.assertIn("--person-weights", str(caught.exception))
        self.assertIn("CS3DGS_PERSON_WEIGHTS", str(caught.exception))
        self.assertEqual(self.runner.calls, [])

    def test_a_failed_gpu_cache_stops_prepare(self) -> None:
        self.runner.fail_on.add("gpu_cache_mono_depth")
        due = GpuStepRequired("mono_depth needs the GPU", cache="mono_depth", command=DA2_COMMAND)
        with self.assertRaises(StageFailed):
            self.ingest([due])

    def test_a_held_lease_is_refused(self) -> None:
        project = self.project(dataset=None)
        lock = project.layout.runs / "gpu.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": os.getpid(), "token": "someone-else"}), encoding="utf-8")
        due = GpuStepRequired("mono_depth needs the GPU", cache="mono_depth", command=DA2_COMMAND)
        with self.assertRaises(StageRefused) as caught:
            self.ingest([due])
        self.assertIn("another job holds it", str(caught.exception))


class AssetPathTest(_Fixture):
    def test_explicit_paths_win_over_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {"CS3DGS_DA2_CHECKPOINT": "C:/env/da2.pth", "CS3DGS_PERSON_WEIGHTS": "C:/env/p.pth"}):
            project = self.project(assets={"da2_checkpoint": "C:/arg/da2.pth"})
            assets = project.gpu_cache_assets()
        self.assertEqual(assets["da2_checkpoint"], Path("C:/arg/da2.pth"))
        self.assertEqual(assets["person_weights"], Path("C:/env/p.pth"))
        self.assertNotIn("da2_model_source", assets)

    def test_every_variable_names_a_cache_profile_field(self) -> None:
        from cloudstudio3dgs_sdk.ingest.caches import CacheProfile

        for field in GPU_CACHE_ASSET_ENV:
            self.assertIn(field, CacheProfile.__dataclass_fields__)

    def test_ingestion_receives_the_assets(self) -> None:
        scene = _project_tests.fake_scene(self.dataset_root)
        _, _, calls = self.ingest([scene], assets={"person_weights": "C:/w/p.pth"})
        self.assertEqual(calls[0]["assets"]["person_weights"], Path("C:/w/p.pth"))


class IngestionGraphTest(_project_tests.ProjectFixture):
    def test_the_plan_owns_ownership_and_backgrounds_and_the_graph_gets_the_weights(self) -> None:
        planned = mock.MagicMock(return_value=SimpleNamespace(statuses=lambda: []))
        bundle = SimpleNamespace(adapter="fake", images=(), point_cloud=None, source_root=self.root, dataset_id="x")
        with mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", return_value=bundle), \
             mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", planned):
            try:
                load_dataset_bundle(self.dataset_root, _project_tests.PROFILE_B5FILL2, self.work,
                                    python="python", repo_root=self.repo,
                                    assets={"person_weights": "C:/w/p.pth", "da2_checkpoint": None})
            except Exception:  # noqa: BLE001 - the fake capture has no cloud; only the call matters
                pass
        # no cloud: refused before the graph is planned
        self.assertFalse(planned.called)

        cloud = SimpleNamespace(path=self.root / "missing.las")
        bundle = SimpleNamespace(adapter="fake", images=(), point_cloud=cloud, source_root=self.root, dataset_id="x")
        with mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", return_value=bundle), \
             mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", planned):
            try:
                load_dataset_bundle(self.dataset_root, _project_tests.PROFILE_B5FILL2, self.work,
                                    python="python", repo_root=self.repo,
                                    assets={"person_weights": "C:/w/p.pth", "da2_checkpoint": None})
            except Exception:  # noqa: BLE001 - nothing is built; only the graph's arguments matter
                pass
        kwargs = planned.call_args_list[0].kwargs
        self.assertIs(kwargs["tile_ownership"], False)
        self.assertIs(kwargs["view_backgrounds"], False)
        self.assertEqual(kwargs["person_weights"], Path("C:/w/p.pth"))
        self.assertNotIn("da2_checkpoint", kwargs)

    def test_the_gpu_refusal_carries_the_cache_and_its_command(self) -> None:
        from tests.test_sdk_bundle_build import FakePlan, _bundle, _graph

        work = self.work
        plan = FakePlan(_graph(work))
        with mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", lambda *a, **k: _bundle(self.root)), \
             mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", return_value=plan):
            with self.assertRaises(GpuStepRequired) as caught:
                load_dataset_bundle(self.root / "capture", _project_tests.PROFILE_B5FILL2, work,
                                    python="python", repo_root=self.repo, runner=lambda command: 0)
        self.assertIn(caught.exception.cache, ("person_mask_manifest", "mono_depth"))
        self.assertTrue(caught.exception.command)


class GpuCacheAssetPreflightTest(_project_tests.ProjectFixture):
    def checks(self, profile, assets):
        from cloudstudio3dgs_sdk.requirements import _gpu_cache_asset_checks

        return {row.name: row for row in _gpu_cache_asset_checks(profile, assets)}

    def test_an_unnamed_or_missing_weight_fails_with_its_flag(self) -> None:
        rows = self.checks(SimpleNamespace(external_assets=()), {"da2_checkpoint": self.root / "gone.pth"})
        self.assertEqual(rows["gpu_cache_asset_person_weights"].status, "FAIL")
        self.assertIn("--person-weights", rows["gpu_cache_asset_person_weights"].remedy)
        self.assertEqual(rows["gpu_cache_asset_da2_checkpoint"].status, "FAIL")
        self.assertIn("does not exist", rows["gpu_cache_asset_da2_checkpoint"].detail)

    def test_a_pinned_weight_must_hash_to_the_pin(self) -> None:
        import hashlib

        weights = self.root / "p.pth"
        weights.write_bytes(b"weights")
        pin = hashlib.sha256(b"weights").hexdigest()
        profile = SimpleNamespace(external_assets=(
            {"cache_field": "person_weights", "sha256": pin, "title": "Mask R-CNN", "license": "BSD"},
        ))
        self.assertEqual(self.checks(profile, {"person_weights": weights})["gpu_cache_asset_person_weights"].status, "PASS")
        weights.write_bytes(b"other weights")
        row = self.checks(profile, {"person_weights": weights})["gpu_cache_asset_person_weights"]
        self.assertEqual(row.status, "FAIL")
        self.assertIn("the profile pins", row.detail)

    def test_the_b12_profile_pins_both_weight_files(self) -> None:
        from cloudstudio3dgs_sdk.profile import PROFILE_B12OP05D3

        pinned = {a["cache_field"]: a["sha256"] for a in PROFILE_B12OP05D3.external_assets if a.get("cache_field")}
        self.assertEqual(sorted(pinned), ["da2_checkpoint", "person_weights"])
        self.assertFalse(any(a.get("ships_in_delivery") for a in PROFILE_B12OP05D3.external_assets))

    def test_only_a_fresh_capture_owes_the_weights(self) -> None:
        fresh = self.project().preflight(require_gpu=False)
        self.assertIn("gpu_cache_asset_person_weights", {c.name for c in fresh.checks})
        project = self.project()
        self.seed_prepare_manifest(project)
        prepared = project.preflight(require_gpu=False)
        self.assertNotIn("gpu_cache_asset_person_weights", {c.name for c in prepared.checks})


class EnvScriptTest(_project_tests.ProjectFixture):
    def test_the_pipeline_config_carries_the_env_script(self) -> None:
        project = self.project(env_script=self.root / "env.cmd")
        self.seed_prepare_manifest(project)
        config = json.loads(project.write_pipeline_config(project.plan()).read_text(encoding="utf-8"))
        self.assertEqual(config["env_script"], str(self.root / "env.cmd"))
        bare = self.project()
        self.assertNotIn("env_script", json.loads(bare.write_pipeline_config(bare.plan()).read_text(encoding="utf-8")))

    def test_the_cli_loads_it_before_the_preflight_and_from_the_environment(self) -> None:
        from cloudstudio3dgs_sdk import __main__ as cli

        argv = ["preflight", "--dataset", str(self.dataset_root), "--work", str(self.work)]
        for extra, environ in ((["--env-script", "C:/env/a.cmd"], {}), ([], {"CS3DGS_ENV_SCRIPT": "C:/env/b.cmd"})):
            with mock.patch.dict(os.environ, environ), \
                 mock.patch.object(cli, "_load_env_script") as load, \
                 mock.patch.object(cli, "_project", side_effect=RuntimeError("stop after loading")):
                with self.assertRaises(RuntimeError):
                    cli.main(argv + extra, stream=open(os.devnull, "w", encoding="utf-8"))
            self.assertEqual(load.call_args.args[0], Path(extra[1] if extra else environ["CS3DGS_ENV_SCRIPT"]))
