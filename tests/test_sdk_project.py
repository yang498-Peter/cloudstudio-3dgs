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

from cloudstudio3dgs_sdk.bundle import INGEST_PACKAGE, DerivedCaches, PreparedScene
from cloudstudio3dgs_sdk.plan import PlannedStep
from cloudstudio3dgs_sdk.profile import PROFILE_B5FILL2, Provenance, make_profile
from cloudstudio3dgs_sdk.project import (
    COMPLETE,
    FAILED,
    RUNNING,
    Project,
    StageRefused,
    StageState,
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
    fields = {}
    for name in PreparedScene.__dataclass_fields__:
        if name in ("scene_tag", "caches"):
            continue
        fields[name] = root / f"{name}.json"
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
    def test_ingestion_is_delegated_and_says_so(self) -> None:
        project = self.project()
        with self.assertRaises(NotImplementedError) as caught:
            project.prepare()
        message = str(caught.exception)
        self.assertIn(INGEST_PACKAGE, message)
        self.assertIn("prepare_manifest.json", message)

    def test_an_existing_manifest_is_adopted(self) -> None:
        project = self.project()
        self.seed_prepare_manifest(project)
        result = project.prepare()
        self.assertEqual(result.action, "ran")
        # Adopting is skipping: the manifest is already the step's output.
        self.assertIn("ingest_dataset", result.steps_skipped)
        self.assertIn("write_arm_configs", result.steps_run)
        self.assertEqual(self.project().stage_state("prepare").state, COMPLETE)

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
        self.assertEqual(pipeline["delivery_tile_arm_pattern"], "tile{tile}_b5fill2_delivery")

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


if __name__ == "__main__":
    unittest.main()
