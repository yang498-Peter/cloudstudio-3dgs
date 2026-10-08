"""Stage state, resume and the fail-closed chain.

No GPU, no network, no trainer: the runner is a stub that creates whatever
``--output`` it is handed, so what is under test is the state machine - a
completed stage is skipped, an upstream artefact that moved is refused, a
changed profile is refused, and a step this checkout cannot run is refused
before anything starts.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from cloudstudio3dgs_sdk.bundle import DerivedCaches, PreparedScene
from cloudstudio3dgs_sdk.ingest.errors import GpuStepRequired
from cloudstudio3dgs_sdk.plan import PlannedStep
from cloudstudio3dgs_sdk.profile import PROFILE_B5FILL2, PROFILE_B5SKY, Provenance, make_profile
from cloudstudio3dgs_sdk.project import (
    COMPLETE,
    FAILED,
    RUNNING,
    Project,
    StageRefused,
    StageState,
    derived_paths_from_scene,
    digest,
    digest_matches,
)
from cloudstudio3dgs_sdk.requirements import GpuInfo, Probes
from tests.test_sdk_plan import make_repo, two_tile_dataset

PLY_STUB = b"ply\nformat binary_little_endian 1.0\nelement vertex 7\nend_header\n"


class RecordingRunner:
    """Creates the outputs a real tool would create; records what it ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_on: set[str] = set()

    def __call__(self, step: PlannedStep, *, log: Path) -> int:
        self.calls.append(step.name)
        if step.name in self.fail_on:
            return 3
        for index, token in enumerate(step.command):
            if token in ("--output", "--output-checkpoint", "--output-report", "--output-root"):
                self._make(Path(step.command[index + 1]))
        for output in step.outputs:
            self._make(Path(output))
        return 0

    @staticmethod
    def _make(path: Path) -> None:
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".ply":
                path.write_bytes(PLY_STUB)
            elif path.suffix == ".json":
                path.write_text(json.dumps({"stub": path.name}), encoding="utf-8")
            else:
                path.write_bytes(b"stub")
        else:
            path.mkdir(parents=True, exist_ok=True)


def good_probes(*, free_bytes: int = 10 ** 15) -> Probes:
    return Probes(
        python_version="3.12",
        torch_version="2.11.0+cu128",
        gpu=GpuInfo(True, "stub", 24.0),
        free_disk_bytes=lambda path: free_bytes,
        gsplat_extension_sha256="ab" * 32,
        external_asset_locator=lambda asset: (True, "stubbed"),
    )


def fake_scene(root: Path) -> PreparedScene:
    """A prepared scene whose files exist: adopting a manifest verifies them."""
    fields = {}
    root.mkdir(parents=True, exist_ok=True)
    for name in PreparedScene.__dataclass_fields__:
        if name in ("scene_tag", "caches", "pose_route"):
            continue
        if name == "dataset_root" or name.endswith("_root"):
            target = root / name
            target.mkdir(parents=True, exist_ok=True)
        else:
            target = root / f"{name}.json"
            target.write_text(json.dumps({"stub": name}), encoding="utf-8")
        fields[name] = target
    return PreparedScene(scene_tag="synth", caches=DerivedCaches(), **fields)


class ProjectFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = make_repo(self.root / "repo")
        self.work = self.root / "work"
        self.dataset_root = self.root / "dataset"
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.runner = RecordingRunner()
        self.dataset = two_tile_dataset()

    def project(self, **overrides) -> Project:
        kwargs = {
            "repo_root": self.repo,
            "python": Path("python.exe"),
            "runner": self.runner,
            "probes": good_probes(),
            "dataset": self.dataset,
            "stream": open(os.devnull, "w", encoding="utf-8"),
        }
        kwargs.update(overrides)
        stream = kwargs["stream"]
        self.addCleanup(stream.close)
        profile = kwargs.pop("profile", PROFILE_B5FILL2)
        return Project(self.dataset_root, self.work, profile, **kwargs)

    def seed_prepare_manifest(self, project: Project | None = None, *, priors: bool = True) -> Path:
        project = project or self.project()
        scene = fake_scene(self.dataset_root)
        return project.write_prepare_manifest(
            scene,
            self.dataset,
            prior_tile_checkpoints={0: "prior0.pt", 1: "prior1.pt"} if priors else None,
        )


class DigestTests(ProjectFixture):
    def test_small_files_get_a_real_sha(self) -> None:
        target = self.root / "small.json"
        target.write_text("{}", encoding="utf-8")
        record = digest(target)
        self.assertEqual(record["digest_kind"], "sha256")
        self.assertTrue(digest_matches(record)[0])
        target.write_text('{"x": 1}', encoding="utf-8")
        ok, detail = digest_matches(record)
        self.assertFalse(ok)
        self.assertIn("sha256", detail)

    def test_large_files_fall_back_to_size_and_mtime(self) -> None:
        target = self.root / "big.pt"
        target.write_bytes(b"0" * 2048)
        record = digest(target, max_bytes=1024)
        self.assertEqual(record["digest_kind"], "size_mtime")
        self.assertNotIn("sha256", record)
        self.assertTrue(digest_matches(record, max_bytes=1024)[0])
        target.write_bytes(b"0" * 4096)
        self.assertFalse(digest_matches(record, max_bytes=1024)[0])

    def test_missing_and_empty_records(self) -> None:
        self.assertEqual(digest_matches(None), (False, "no digest recorded"))
        self.assertFalse(digest_matches({"path": str(self.root / "gone.json")})[0])


class StageStateTests(ProjectFixture):
    def test_sidecar_shape_matches_the_pipeline_job_convention(self) -> None:
        path = self.work / "sdk_state" / "stage_prepare.json"
        state = StageState(path, stage="prepare")
        state.set(RUNNING, "started")
        state.set(COMPLETE, "done", outputs={})
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["job"], "sdk_stage")
        self.assertEqual(payload["name"], "prepare")
        self.assertEqual(payload["state"], COMPLETE)
        self.assertEqual([entry["state"] for entry in payload["history"]], [RUNNING, COMPLETE])
        self.assertIn("updated_at", payload)

    def test_unknown_state_is_rejected(self) -> None:
        state = StageState(self.work / "sdk_state" / "stage_train.json", stage="train")
        with self.assertRaises(ValueError):
            state.set("ALMOST", "no")

    def test_a_corrupt_sidecar_is_ignored_not_trusted(self) -> None:
        path = self.work / "sdk_state" / "stage_prepare.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json", encoding="utf-8")
        self.assertIsNone(StageState(path, stage="prepare").state)


class PrepareTests(ProjectFixture):
    def test_ingestion_is_delegated_with_the_bundle_signature(self) -> None:
        """No manifest: prepare calls the ingestion glue with python and repo root."""
        seen: dict[str, object] = {}
        scene = fake_scene(self.dataset_root)

        def fake_load(dataset_root, profile, work_root, *, python, repo_root, adapter, run_dir, pipeline_gate,
                      vram_gib, assets, pose_route, log):
            seen.update(
                dataset_root=dataset_root, profile=profile, work_root=work_root, python=python,
                repo_root=repo_root, adapter=adapter, run_dir=run_dir, pipeline_gate=pipeline_gate,
                vram_gib=vram_gib, pose_route=pose_route,
            )
            log("[prepare] fake adapter says hello")
            return scene

        with mock.patch("cloudstudio3dgs_sdk.project.load_dataset_bundle", fake_load):
            project = self.project()
            result = project.prepare()
        self.assertEqual(result.action, "ran")
        self.assertIn("ingest_dataset", result.steps_run)
        self.assertEqual(seen["dataset_root"], self.dataset_root)
        self.assertEqual(seen["work_root"], self.work)
        self.assertEqual(seen["python"], Path("python.exe"))
        self.assertEqual(seen["repo_root"], self.repo)
        self.assertIs(seen["profile"], PROFILE_B5FILL2)
        # Nothing fresh-dataset-specific was given, so nothing is invented for the adapter.
        self.assertIsNone(seen["adapter"])
        self.assertIsNone(seen["run_dir"])
        self.assertIsNone(seen["pipeline_gate"])
        # the card size reaches the tile-count rule
        self.assertEqual(seen["vram_gib"], project.vram_gib)
        # the library default stays the capture's own poses; the CLI defaults to the AT route
        self.assertEqual(seen["pose_route"], "raw_capture_poses")
        payload = json.loads((self.work / "prepare" / "prepare_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["trainer_paths"], scene.trainer_paths())
        self.assertIn("derived_paths", payload)

    def test_fresh_dataset_inputs_reach_the_ingestion_glue(self) -> None:
        """--adapter / --run-dir / --pipeline-gate are the SDK's whole fresh-dataset surface;
        a Project that took them must hand them on, or a split capture can never be prepared."""
        seen: dict[str, object] = {}
        scene = fake_scene(self.dataset_root)

        def fake_load(dataset_root, profile, work_root, **kwargs):
            seen.update(kwargs)
            return scene

        with mock.patch("cloudstudio3dgs_sdk.project.load_dataset_bundle", fake_load):
            project = self.project(
                adapter="s1_fisheye",
                run_dir=self.dataset_root / "processed",
                pipeline_gate=self.work / "gate" / "pipeline_gate.json",
            )
            project.prepare()
        self.assertEqual(seen["adapter"], "s1_fisheye")
        self.assertEqual(seen["run_dir"], self.dataset_root / "processed")
        self.assertEqual(seen["pipeline_gate"], self.work / "gate" / "pipeline_gate.json")

    def test_a_gpu_step_required_by_ingestion_is_a_refusal_with_the_command(self) -> None:
        error = GpuStepRequired("view_backgrounds_0 needs CUDA")
        error.command = ["python.exe", "tools/build_view_backgrounds.py", "--config", "x.json"]

        def fake_load(*args, **kwargs):
            raise error

        with mock.patch("cloudstudio3dgs_sdk.project.load_dataset_bundle", fake_load):
            project = self.project()
            with self.assertRaises(StageRefused) as caught:
                project.prepare()
        message = str(caught.exception)
        self.assertIn("needs a GPU", message)
        self.assertIn("build_view_backgrounds.py --config x.json", message)
        self.assertEqual(self.project().stage_state("prepare").state, FAILED)

    def test_an_existing_manifest_is_adopted(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        result = project.prepare()
        self.assertEqual(result.action, "ran")
        # Adopting is skipping: the manifest is already the step's output.
        self.assertIn("ingest_dataset", result.steps_skipped)
        self.assertIn("write_arm_configs", result.steps_run)
        self.assertIn("delivery_eval_config", result.steps_run)
        self.assertEqual(self.project().stage_state("prepare").state, COMPLETE)

    def test_an_existing_manifest_is_verified_not_trusted(self) -> None:
        project = self.project()
        manifest = self.seed_prepare_manifest(project)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        Path(payload["trainer_paths"]["split_manifest"]).unlink()
        with self.assertRaises(StageRefused) as caught:
            project.prepare()
        self.assertIn("split_manifest", str(caught.exception))
        self.assertEqual(self.project().stage_state("prepare").state, FAILED)
        self.assertEqual(self.runner.calls, [])

    def test_delivery_eval_config_is_the_tile0_config_with_its_identity_changed(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.prepare()
        eval_config = json.loads((self.work / "delivery_eval.json").read_text(encoding="utf-8"))
        tile0 = json.loads((self.work / "runs" / "tile0_b5fill2_delivery.json").read_text(encoding="utf-8"))
        self.assertEqual(eval_config["run_id"], "synth-delivery-eval")
        self.assertNotEqual(eval_config["output_dir"], tile0["output_dir"])
        for key in (
            "background_image_manifest", "background_image_root", "dataset_manifest", "device",
            "face_cache_manifest", "face_cache_root", "face_lidar_geometry_manifest",
            "face_lidar_geometry_root", "mipmap_tile_id", "renderer_mask_manifest", "tile_inputs_manifest",
            "tile_ownership_dilation_px", "tile_ownership_margin_m", "sh_degree",
        ):
            self.assertEqual(eval_config[key], tile0[key], key)
        pipeline = json.loads((self.work / "pipeline.json").read_text(encoding="utf-8"))
        self.assertEqual(pipeline["delivery_eval_config"], str(self.work / "delivery_eval.json"))
        self.assertEqual(pipeline["sky_ply"], str(self.work / "caches" / "sky_dome.ply"))

    def test_sky_dome_ply_skips_when_its_output_exists_but_derived_configs_are_refreshed(self) -> None:
        """A built artefact is skipped when present; the evaluator config and arm configs
        are derived from the manifest and rewritten every time, because a resumed run once
        scored against a stale evaluator config the manifest no longer described."""
        project = self.project()
        self.seed_prepare_manifest(project)
        (self.work / "caches").mkdir(parents=True, exist_ok=True)
        (self.work / "caches" / "sky_dome.ply").write_bytes(PLY_STUB)
        (self.work / "delivery_eval.json").write_text("{}", encoding="utf-8")
        result = project.prepare()
        self.assertIn("sky_dome_ply", result.steps_skipped)
        self.assertIn("delivery_eval_config", result.steps_run)
        self.assertIn("write_arm_configs", result.steps_run)
        self.assertNotEqual((self.work / "delivery_eval.json").read_text(encoding="utf-8"), "{}", "the stale stub was replaced")
        self.assertNotIn("sky_dome_ply", self.runner.calls)
        self.assertIn("sky_dome", self.runner.calls)

    def test_prepare_writes_every_arm_config_and_the_pipeline_config(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.prepare()
        runs = self.work / "runs"
        self.assertTrue((runs / "tile0_b5fill2_delivery.json").is_file())
        self.assertTrue((runs / "tile1_b5fill2_delivery.json").is_file())
        self.assertTrue((runs / "global_coarse_b5fill2.json").is_file())
        pipeline = json.loads((self.work / "pipeline.json").read_text(encoding="utf-8"))
        self.assertEqual(pipeline["run_root"], str(runs))
        self.assertEqual(pipeline["battery_views"], 48)
        # tools/pipeline.py validates that the pattern carries both {tile} and {tag}; the literal
        # profile-name form this used to assert was refused by the pipeline on the first real
        # SDK run, before any training. With {tag} = profile it expands to the same arm names.
        self.assertEqual(pipeline["delivery_tile_arm_pattern"], "tile{tile}_{tag}_delivery")

    def test_a_completed_stage_is_skipped(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.prepare()
        calls = len(self.runner.calls)
        again = self.project().prepare()
        self.assertEqual(again.action, "skipped")
        self.assertEqual(len(self.runner.calls), calls)

    def test_force_re_runs_a_completed_stage(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.prepare()
        calls = len(self.runner.calls)
        again = self.project().prepare(force=True)
        self.assertEqual(again.action, "ran")
        self.assertGreater(len(self.runner.calls), calls)

    def test_a_deleted_output_re_opens_the_stage(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.prepare()
        (self.work / "caches" / "sky_masks" / "sky_mask_train.json").unlink()
        current, why = self.project()._stage_is_current("prepare")
        self.assertFalse(current)
        self.assertIn("no longer matches", why)


class FreshCaptureTests(ProjectFixture):
    """A capture nobody prepared: no prepare manifest and no injected summary.

    The plan is built from the tile inventory, which only ingestion produces, so prepare
    used to refuse before it could ingest anything (tests hid it by injecting dataset=).
    Ingestion now runs first and the summary is measured from what it built.
    """

    def _scene(self, *, views=(40, 30)) -> PreparedScene:
        import dataclasses

        import laspy
        import numpy as np

        scene = fake_scene(self.dataset_root)
        tiles = [
            {"tile_id": i, "name": f"Tile_{i}", "view_count": v,
             "initialization": {"point_count": 1000 + i, "sha256": f"{i}" * 64}}
            for i, v in enumerate(views)
        ]
        Path(scene.tile_inputs_manifest).write_text(json.dumps({"tiles": tiles}), encoding="utf-8")
        Path(scene.face_cache_manifest).write_text(
            json.dumps({"images": [{"faces": [0, 1, 2, 3]}, {"faces": [0, 1, 2, 3]}]}), encoding="utf-8"
        )
        Path(scene.global_init_ply).write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 1234\nproperty float x\nend_header\n"
        )
        las = self.dataset_root / "cloud.las"
        header = laspy.LasHeader(point_format=2, version="1.2")
        data = laspy.LasData(header)
        data.x, data.y, data.z = np.arange(50.0), np.zeros(50), np.zeros(50)
        data.write(str(las))
        return dataclasses.replace(scene, lidar_cloud=las)

    def test_a_fresh_capture_is_ingested_before_the_first_plan(self) -> None:
        scene = self._scene()
        with mock.patch("cloudstudio3dgs_sdk.project.load_dataset_bundle", lambda *a, **k: scene):
            project = self.project(dataset=None)
            project.prepare()
        payload = json.loads((self.work / "prepare" / "prepare_manifest.json").read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        self.assertEqual([tile["view_count"] for tile in dataset["tiles"]], [40, 30])
        self.assertEqual(dataset["train_view_count"], 8)
        self.assertEqual(dataset["global_init_point_count"], 1234)
        self.assertEqual(dataset["lidar_point_count"], 50)
        self.assertFalse(dataset.get("estimated", False))
        # the arm configs were written from that summary
        self.assertTrue(any(self.work.joinpath("runs").glob("*.json")))

    def test_a_tile_no_view_sees_refuses_before_any_arm_config(self) -> None:
        scene = self._scene(views=(40, 0))
        with mock.patch("cloudstudio3dgs_sdk.project.load_dataset_bundle", lambda *a, **k: scene):
            project = self.project(dataset=None)
            with self.assertRaises(StageRefused) as caught:
                project.prepare()
        self.assertIn("no training view sees Tile_1", str(caught.exception))
        self.assertFalse((self.work / "prepare" / "prepare_manifest.json").exists())


class FailClosedTests(ProjectFixture):
    def _complete_prepare(self) -> Project:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.prepare()
        return project

    def test_train_refuses_before_prepare(self) -> None:
        project = self.project()
        with self.assertRaises(StageRefused) as caught:
            project.train()
        self.assertIn("needs prepare COMPLETE", str(caught.exception))

    def test_train_refuses_when_a_prepare_output_changed(self) -> None:
        self._complete_prepare()
        cache = self.work / "caches" / "sky_masks" / "sky_mask_train.json"
        cache.write_text('{"tampered": true}', encoding="utf-8")
        with self.assertRaises(StageRefused) as caught:
            self.project().train()
        message = str(caught.exception)
        self.assertIn("changed since it was recorded", message)
        self.assertIn("sky_mask_train.json", message)

    def test_train_refuses_when_a_prepare_output_vanished(self) -> None:
        self._complete_prepare()
        (self.work / "caches" / "sky_dome.pt").unlink()
        with self.assertRaises(StageRefused) as caught:
            self.project().train()
        self.assertIn("missing", str(caught.exception))

    def test_a_changed_profile_refuses_to_resume(self) -> None:
        self._complete_prepare()
        payload = PROFILE_B5FILL2.as_dict()
        provenance = dict(PROFILE_B5FILL2.provenance)
        payload.pop("provenance")
        payload["trainer_base"]["cap_max_note"] = "edited"
        other = make_profile(provenance=provenance, **payload)
        with self.assertRaises(StageRefused) as caught:
            self.project(profile=other).train()
        self.assertIn("different recipe", str(caught.exception))

    def test_a_changed_profile_does_not_silently_skip_a_done_stage(self) -> None:
        self._complete_prepare()
        payload = PROFILE_B5FILL2.as_dict()
        provenance = dict(PROFILE_B5FILL2.provenance)
        payload.pop("provenance")
        payload["version"] = "9999.01.01"
        other = make_profile(provenance=provenance, **payload)
        current, why = self.project(profile=other)._stage_is_current("prepare")
        self.assertFalse(current)
        self.assertIn("different profile sha256", why)

    def test_a_blocking_step_is_caught_at_the_first_gpu_stage(self) -> None:
        """The merge gap costs nothing if it surfaces before the training does."""
        from cloudstudio3dgs_sdk.requirements import PreflightFailed

        project = self.project(repo_root=make_repo(self.root / "norepo", fill_support=False))
        self.seed_prepare_manifest(project)
        project.prepare()
        with self.assertRaises(PreflightFailed) as caught:
            project.train()
        self.assertIn("--fill-checkpoint", str(caught.exception))

    def test_a_blocking_step_still_refuses_if_the_preflight_is_bypassed(self) -> None:
        class NoPreflight(Project):
            def preflight(self, *, require_gpu: bool = True):
                report = super().preflight(require_gpu=require_gpu)
                return type(report)(
                    tuple(c for c in report.checks if c.name != "checkout_supports_profile"),
                    report.profile_sha256,
                    report.plan_sha256,
                )

        project = self.project(repo_root=make_repo(self.root / "norepo", fill_support=False))
        self.seed_prepare_manifest(project)
        project.prepare()
        bypass = NoPreflight(
            self.dataset_root,
            self.work,
            PROFILE_B5FILL2,
            repo_root=self.root / "norepo",
            python=Path("python.exe"),
            runner=self.runner,
            probes=good_probes(),
            stream=open(os.devnull, "w", encoding="utf-8"),
        )
        self.addCleanup(bypass.stream.close)
        bypass.train()
        with self.assertRaises(StageRefused) as caught:
            bypass.deliver()
        self.assertIn("--fill-checkpoint", str(caught.exception))
        self.assertEqual(bypass.stage_state("deliver").state, FAILED)

    def test_preflight_failure_stops_a_gpu_stage(self) -> None:
        project = self.project(probes=Probes("3.12", None, GpuInfo(False, detail="no card"), lambda p: 10 ** 15))
        self.seed_prepare_manifest(project)
        project.prepare()  # prepare is CPU-only and does not preflight
        from cloudstudio3dgs_sdk.requirements import PreflightFailed

        with self.assertRaises(PreflightFailed) as caught:
            project.train()
        self.assertIn("gpu", str(caught.exception))


class FullRunTests(ProjectFixture):
    def test_run_all_walks_the_four_stages(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        results = project.run_all()
        self.assertEqual([r.stage for r in results], ["prepare", "train", "deliver", "report"])
        self.assertTrue(all(r.ok for r in results))
        report = self.work / "report" / "b5fill2_report.json"
        self.assertTrue(report.is_file())
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(payload["profile"]["profile_sha256"], PROFILE_B5FILL2.profile_sha256)
        self.assertTrue(any(g["status"] == "UNVERIFIED" for g in payload["gates"]))
        self.assertIn("tile_rules.seed_generation_overrides", payload["unmeasured_knobs"])
        self.assertTrue((self.work / "report" / "b5fill2_report.md").is_file())

    def test_a_second_run_all_skips_everything(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.run_all()
        self.runner.calls.clear()
        results = self.project().run_all()
        self.assertEqual({r.action for r in results}, {"skipped"})
        self.assertEqual(self.runner.calls, [])

    def test_stage_subset_runs_only_those_stages(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        results = project.run_all(stages=("prepare",))
        self.assertEqual([r.stage for r in results], ["prepare"])
        self.assertIsNone(self.project().stage_state("train").state)

    def test_unknown_stage_is_rejected_before_anything_runs(self) -> None:
        project = self.project()
        with self.assertRaises(KeyError):
            project.run_all(stages=("prepare", "polish"))
        self.assertEqual(self.runner.calls, [])

    def test_dry_run_executes_nothing(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        results = project.run_all(dry_run=True)
        self.assertEqual({r.action for r in results}, {"planned"})
        self.assertEqual(self.runner.calls, [])
        self.assertIsNone(self.project().stage_state("prepare").state)

    def test_a_failing_step_marks_the_stage_failed(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        self.runner.fail_on = {"sky_dome"}
        with self.assertRaises(Exception):
            project.prepare()
        state = self.project().stage_state("prepare")
        self.assertEqual(state.state, FAILED)
        self.assertIn("sky_dome", state.reason)

    def test_threshold_control_records_every_variant(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        project.run_all()
        record = json.loads(
            (self.work / "runs" / "delivery_b5fill2" / "threshold_control" / "threshold_control.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual([v["min_opacity"] for v in record["variants"]], [0.0, 0.01, 0.05])
        self.assertEqual(record["delivery_min_opacity"], 0.05)
        for variant in record["variants"]:
            self.assertEqual(variant["vertex_count"], 7)
            self.assertEqual(variant["removed_vs_zero"], 0)

    def test_the_delivered_pair_is_scored_and_the_report_reads_it(self) -> None:
        """Both batteries are kept; the alpha/PSNR gates read the pair, morphology the body."""

        class ScoringRunner(RecordingRunner):
            def __call__(self, step: PlannedStep, *, log: Path) -> int:
                code = super().__call__(step, log=log)
                if step.name == "battery":
                    Path(step.outputs[0]).write_text(
                        json.dumps({"alpha_p05": 0.19, "alpha_mean": 0.61, "psnr_p10": 16.4, "psnr_mean": 18.9, "views": 48}),
                        encoding="utf-8",
                    )
                elif step.name == "battery_pair":
                    Path(step.outputs[0]).write_text(
                        json.dumps({"alpha_p05": 0.90, "alpha_mean": 0.95, "psnr_p10": 16.7, "psnr_mean": 19.1, "views": 48}),
                        encoding="utf-8",
                    )
                elif step.name == "morphology":
                    Path(step.outputs[0]).write_text(
                        json.dumps({"stats": {"short_p50_mm": 0.47, "max_min_p50": 12.0}}), encoding="utf-8"
                    )
                return code

        self.runner = ScoringRunner()
        project = self.project()
        self.seed_prepare_manifest(project)
        results = project.run_all()
        self.assertTrue(all(r.ok for r in results))
        names = self.runner.calls
        self.assertLess(names.index("reimport_ply"), names.index("pair"))
        self.assertLess(names.index("pair"), names.index("battery_pair"))
        pair = next(s for s in project.plan().steps if s.name == "pair")
        self.assertIn(str(self.work / "caches" / "sky_dome.pt"), pair.command)
        self.assertTrue((self.work / "runs" / "delivery_b5fill2" / "delivery_pair.pt").is_file())
        report = json.loads((self.work / "report" / "b5fill2_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["coverage"]["body_only"]["layers"], "body")
        self.assertEqual(report["coverage"]["delivered_pair"]["layers"], "body+sky")
        self.assertEqual(report["coverage"]["body_only"]["alpha_p05"], 0.19)
        self.assertEqual(report["coverage"]["delivered_pair"]["alpha_p05"], 0.90)
        self.assertEqual(report["gate"]["reads"], "delivered_pair")
        self.assertEqual(report["gate"]["alpha_p05"], 0.90)
        self.assertEqual(report["gate"]["body_only_alpha_p05"], 0.19)
        self.assertEqual(report["measured"]["battery_psnr_p10"], 16.7)
        self.assertEqual(report["measured"]["body_only_psnr_p10"], 16.4)
        self.assertEqual(report["measured"]["morphology_short_axis_p50_mm"], 0.47)
        self.assertEqual(report["measured"]["export_gaussian_count"], 7)
        gates = {g["gate"]: g for g in report["gates"]}
        for key in ("battery_alpha_p05_min", "battery_psnr_p10_min", "offtrajectory_sharpness_min",
                    "export_gaussian_count_max", "morphology_short_axis_p50_mm_max"):
            self.assertIn(key, gates)
        self.assertEqual(gates["battery_psnr_p10_min"]["reads"], "delivered_pair")
        self.assertEqual(gates["battery_psnr_p10_min"]["measured"], 16.7)
        self.assertEqual(gates["battery_alpha_p05_min"]["measured"], 0.90)
        self.assertEqual(gates["morphology_short_axis_p50_mm_max"]["reads"], "body_only")
        self.assertEqual(gates["export_gaussian_count_max"]["reads"], "body_only")
        # a gate whose profile threshold or measurement is absent stays UNVERIFIED
        self.assertEqual(gates["offtrajectory_sharpness_min"]["status"], "UNVERIFIED")
        for key, gate in gates.items():
            threshold = PROFILE_B5FILL2.acceptance.get(key)
            if threshold is None or gate["measured"] is None:
                self.assertEqual(gate["status"], "UNVERIFIED", key)
            else:
                self.assertIn(gate["status"], ("PASS", "FAIL"), key)
        markdown = (self.work / "report" / "b5fill2_report.md").read_text(encoding="utf-8")
        self.assertIn("delivered_pair", markdown)
        self.assertIn("body_only", markdown)

    def test_dataset_summary_requires_prepare_or_an_injection(self) -> None:
        project = Project(self.dataset_root, self.work, PROFILE_B5FILL2, repo_root=self.repo)
        with self.assertRaises(StageRefused) as caught:
            project.dataset_summary()
        self.assertIn("prepare_manifest.json", str(caught.exception))

    def test_prepare_manifest_round_trips_through_the_project(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        fresh = Project(self.dataset_root, self.work, PROFILE_B5FILL2, repo_root=self.repo, python=Path("python.exe"))
        self.assertEqual(fresh.dataset_summary(), self.dataset)
        self.assertEqual(fresh.prior_tile_checkpoints(), {0: "prior0.pt", 1: "prior1.pt"})
        self.assertEqual(fresh.plan().generations, ("delivery",))


class ValidationCacheTests(ProjectFixture):
    """prepare refuses when the battery's validation caches, derived by name from the evaluator
    config, do not exist - instead of the battery finding out after every tile has trained."""

    def scene_on_v9_style_caches(self):
        import dataclasses

        scene = fake_scene(self.dataset_root)
        train_root = self.dataset_root / "face4_train"
        train_root.mkdir(parents=True, exist_ok=True)
        manifest = train_root / "face_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return dataclasses.replace(scene, face_cache_manifest=manifest, face_cache_root=train_root)

    def test_prepare_refuses_naming_the_missing_validation_cache(self) -> None:
        project = self.project()
        project.write_prepare_manifest(self.scene_on_v9_style_caches(), self.dataset)
        with self.assertRaises(StageRefused) as caught:
            project.prepare()
        message = str(caught.exception)
        self.assertIn("validation caches", message)
        self.assertIn(str(self.dataset_root / "face4_val"), message)
        self.assertNotIn("face4_val_train", message, "the v9 spelling must not be mangled")
        self.assertFalse((self.work / "delivery_eval.json").exists(), "nothing is written on refusal")
        self.assertEqual(project.stage_state("prepare").state, FAILED)

    def test_prepare_passes_once_the_validation_cache_exists(self) -> None:
        project = self.project()
        project.write_prepare_manifest(self.scene_on_v9_style_caches(), self.dataset)
        val_root = self.dataset_root / "face4_val"
        val_root.mkdir(parents=True, exist_ok=True)
        (val_root / "face_manifest.json").write_text("{}", encoding="utf-8")
        result = project.prepare()
        self.assertIn("delivery_eval_config", result.steps_run)
        written = json.loads((self.work / "delivery_eval.json").read_text(encoding="utf-8"))
        self.assertEqual(written["face_cache_manifest"], str(self.dataset_root / "face4_train" / "face_manifest.json"))


class ReferenceStripsTests(ProjectFixture):
    """An adopted competitor model adds the strips to deliver; the sharpness gate reads them.

    Pinned on b5sky, the profile that carries the sharpness gate. The first SDK delivery of house0305 adopted the reference model and still reported the
    off-trajectory gate UNVERIFIED: the plan only looked at the summary's flag, and the strip
    steps it would have planned had no arguments. Adding the strips after a delivery is
    complete must also re-run deliver for just those steps, and then the report.
    """

    class StripRunner(RecordingRunner):
        def __call__(self, step: PlannedStep, *, log: Path) -> int:
            code = super().__call__(step, log=log)
            if step.name == "offtrajectory_score":
                rows = [{"file": f"offtraj_{i}.png", "psnr_q": 17.0, "sharp_ratio": ratio} for i, ratio in enumerate((0.40, 0.46, 0.52))]
                Path(step.outputs[0]).write_text(json.dumps({"b5sky": rows}), encoding="utf-8")
            return code

    def reference_paths(self) -> dict[str, str]:
        ply = self.root / "competitor" / "ref.ply"
        align = self.root / "competitor" / "align.json"
        ply.parent.mkdir(parents=True, exist_ok=True)
        ply.write_bytes(PLY_STUB)
        align.write_text("{}", encoding="utf-8")
        return {"reference_ply": str(ply), "reference_alignment": str(align)}

    def test_adopted_reference_plans_the_strips_with_real_arguments(self) -> None:
        self.runner = self.StripRunner()
        project = self.project(profile=PROFILE_B5SKY)
        scene = fake_scene(self.dataset_root)
        project.write_prepare_manifest(
            scene, self.dataset,
            derived_paths={**derived_paths_from_scene(scene, repo_root=self.repo), **self.reference_paths()},
        )
        plan = project.plan()
        self.assertFalse(any("no reference" in w for w in plan.warnings))
        by_name = {s.name: s for s in plan.steps}
        offtraj = by_name["offtrajectory"]
        self.assertIn("build_offtrajectory_compare.py", offtraj.command[1])
        self.assertIn(str(self.work / "delivery_eval.json"), offtraj.command)
        self.assertIn(str(self.work / "runs" / "delivery_b5sky" / "reimported.pt"), offtraj.command)
        self.assertIn("--reference-ply", offtraj.command)
        self.assertIn(str(self.root / "competitor" / "ref.ply"), offtraj.command)
        compare = by_name["compare_matched"]
        self.assertIn("--frames", compare.command)
        self.assertIn("--reference-alignment", compare.command)
        score = by_name["offtrajectory_score"]
        self.assertIn("score_offtrajectory_strips.py", score.command[1])
        self.assertIn(f"b5sky={self.work / 'runs' / 'delivery_b5sky' / 'offtrajectory'}", score.command)
        results = project.run_all()
        self.assertTrue(all(r.ok for r in results))
        report = json.loads((self.work / "report" / "b5sky_report.json").read_text(encoding="utf-8"))
        gates = {g["gate"]: g for g in report["gates"]}
        self.assertEqual(gates["offtrajectory_sharpness_min"]["measured"], 0.46, "median sharp_ratio of the rows")
        self.assertEqual(gates["offtrajectory_sharpness_min"]["status"], "PASS")
        self.assertEqual(report["measured"]["offtrajectory_strip_count"], 3)

    def test_a_reference_adopted_after_delivery_reruns_only_the_strips_and_then_the_report(self) -> None:
        self.runner = self.StripRunner()
        project = self.project(profile=PROFILE_B5SKY)
        project.write_prepare_manifest(fake_scene(self.dataset_root), self.dataset)
        first = project.run_all()
        self.assertEqual([r.action for r in first], ["ran"] * 4)
        gates = {g["gate"]: g for g in json.loads((self.work / "report" / "b5sky_report.json").read_text(encoding="utf-8"))["gates"]}
        self.assertEqual(gates["offtrajectory_sharpness_min"]["status"], "UNVERIFIED")
        # the reference arrives: the manifest changes, so prepare re-runs (cheaply), train
        # stays as it was, deliver owes exactly the strips, and the report follows
        self.runner.calls.clear()
        project = self.project(profile=PROFILE_B5SKY)
        scene = fake_scene(self.dataset_root)
        project.write_prepare_manifest(
            scene, self.dataset,
            derived_paths={**derived_paths_from_scene(scene, repo_root=self.repo), **self.reference_paths()},
        )
        second = project.run_all()
        by_stage = {r.stage: r for r in second}
        self.assertEqual(by_stage["deliver"].action, "ran")
        self.assertEqual(set(by_stage["deliver"].steps_run), {"compare_matched", "offtrajectory", "offtrajectory_score"})
        self.assertNotIn("merge_tiles", self.runner.calls)
        self.assertNotIn("battery", self.runner.calls)
        self.assertEqual(by_stage["report"].action, "ran")
        gates = {g["gate"]: g for g in json.loads((self.work / "report" / "b5sky_report.json").read_text(encoding="utf-8"))["gates"]}
        self.assertEqual(gates["offtrajectory_sharpness_min"]["status"], "PASS")
        # and a third run has nothing left to do
        third = self.project(profile=PROFILE_B5SKY).run_all()
        self.assertEqual({r.action for r in third}, {"skipped"})


if __name__ == "__main__":
    unittest.main()
