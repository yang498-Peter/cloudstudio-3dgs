"""The independent-AT pose route (W10): raw tier -> AT -> training manifest -> signed gate chain.

The quality of every house0305 delivery rests on independent-AT poses, and the trainer refuses
AT-lineage fisheye data without a signed readiness gate. Neither existed for a new capture: the
chain was a page of hand-run commands and the gate a file somebody produced. What is pinned:

* the AT route rewires the ingest graph without touching the raw route: the capture's manifest,
  masks, person masks and split move to a raw tier, the AT steps read them, and
  ``dataset_manifest`` becomes the AT-published training manifest every other cache binds to;
* the AT solver runs with the convergence settings house0305 needed (80 outer iterations at
  1e-6; the defaults did not converge), and a new AT report makes the training tier stale;
* the time-sync audit refuses a non-zero offset, the one value the frontend gate admits;
* the full-resolution smoke is a config the trainer's smoke rules admit;
* the gate chain runs in the order house0305 was gated in, stops for the GPU smoke, resumes,
  and restarts from gate_10 when any input moves;
* the SDK tile plan carries the bindings the real surface tile gate compares (test_ingest_tiling).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cloudstudio3dgs_sdk.ingest import at_steps
from cloudstudio3dgs_sdk.ingest.caches import (
    STATUS_PRESENT,
    STATUS_STALE,
    CachePlan,
    CachePlanError,
    estimate_ingest,
)
from cloudstudio3dgs_sdk.ingest.errors import DatasetIncompleteError, GpuStepRequired
from cloudstudio3dgs_sdk.ingest.gates import (
    GATE_FILES,
    SURFACE_DEFERRAL_REASON,
    GateInputs,
    build_gate_chain,
    gate_commands,
)
from tests import test_ingest_caches as _caches_tests

AT = "independent_at"


def _plan(root: Path, **overrides) -> CachePlan:
    profile = _caches_tests._profile(root, pose_route=AT, tile_ownership=False, view_backgrounds=False, **overrides)
    return CachePlan(_caches_tests._bundle(root), profile)


class AtRouteGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.plan = _plan(self.root)
        self.order = [spec.name for spec in self.plan]

    def test_the_raw_route_is_untouched(self) -> None:
        raw = CachePlan(_caches_tests._bundle(self.root), _caches_tests._profile(self.root))
        self.assertFalse([spec.name for spec in raw if spec.name.startswith(("raw_", "at_", "timesync"))])

    def test_an_unknown_route_is_refused(self) -> None:
        profile = _caches_tests._profile(self.root, pose_route="guess")
        with self.assertRaises(CachePlanError):
            CachePlan(_caches_tests._bundle(self.root), profile)

    def test_the_chain_runs_raw_tier_then_at_then_the_training_tier(self) -> None:
        position = {name: index for index, name in enumerate(self.order)}
        chain = ["raw_dataset_manifest", "raw_mask_manifest", "raw_person_mask_manifest", "at_features",
                 "at_triangulation", "at_solve", "dataset_manifest", "mask_manifest", "person_mask_manifest",
                 "face_cache", "tile_plan"]
        self.assertEqual(sorted(chain, key=position.__getitem__), chain)
        self.assertLess(position["at_features_raw"], position["at_features"])
        self.assertLess(position["timesync"], position["dataset_manifest"])

    def test_the_raw_tier_lives_beside_the_training_tier(self) -> None:
        raw = self.plan.spec("raw_dataset_manifest")
        self.assertEqual(raw.manifest, self.root / "dataset_raw" / "dataset_manifest.json")
        self.assertIn(str(self.root / "dataset_raw"), raw.command)
        person = self.plan.spec("raw_person_mask_manifest")
        self.assertEqual(person.manifest, self.root / "dataset_raw" / "person_masks" / "person_mask_manifest.json")
        self.assertEqual(person.device, "gpu")
        self.assertEqual({b.depends_on for b in person.bindings}, {"raw_dataset_manifest", "raw_mask_manifest"})

    def test_the_training_manifest_is_published_from_the_at(self) -> None:
        spec = self.plan.spec("dataset_manifest")
        command = list(spec.command)
        self.assertTrue(command[1].endswith("build_ba_training_manifest.py"))
        self.assertEqual(command[command.index("--manifest") + 1], str(self.root / "dataset_raw" / "dataset_manifest.json"))
        self.assertEqual(command[command.index("--split-manifest") + 1], str(self.root / "dataset_raw" / "split_manifest.json"))
        self.assertEqual(spec.manifest, self.root / "dataset" / "dataset_manifest.json")

    def test_the_solver_runs_with_the_settings_house0305_needed(self) -> None:
        command = list(self.plan.spec("at_solve").command)
        self.assertEqual(command[command.index("--intrinsic-outer-iterations") + 1], "200")
        self.assertEqual(command[command.index("--intrinsic-convergence-tol") + 1], "1e-6")
        self.assertIn("--triangulation-runtime-manifest", command)  # the gate checks triangulation_identity

    def test_features_are_filtered_by_the_raw_masks_and_read_the_camera_folder(self) -> None:
        spec = self.plan.spec("at_features")
        command = list(spec.command)
        self.assertEqual(command[command.index("--image-dir") + 1], str(self.root / "recording" / "camera"))
        self.assertEqual(
            {b.manifest_key: b.depends_on for b in spec.bindings},
            {
                "feature_filter.dataset_manifest_sha256": "raw_dataset_manifest",
                "feature_filter.mask_manifest_sha256": "raw_mask_manifest",
                "feature_filter.person_mask_manifest_sha256": "raw_person_mask_manifest",
            },
        )

    def test_training_person_masks_are_a_cpu_rebind(self) -> None:
        spec = self.plan.spec("person_mask_manifest")
        self.assertEqual(spec.device, "cpu")
        self.assertIn("rebind_person_mask_base.py", " ".join(spec.command))
        self.assertIn("fresh", spec.command)

    def test_a_new_at_report_makes_the_training_manifest_stale(self) -> None:
        write = _caches_tests._write
        raw = write(self.plan.spec("raw_dataset_manifest").manifest, {"images": []}, "manifest_sha256")
        split = write(self.plan.spec("raw_split_manifest").manifest, {"dataset_manifest_sha256": raw}, "split_manifest_sha256")
        report = write(self.plan.spec("at_solve").manifest, {"dataset_manifest_sha256": raw, "v": 1}, "report_sha256")
        write(
            self.plan.spec("dataset_manifest").manifest,
            {"training_lineage": {"base_dataset_manifest_sha256": raw, "independent_at_report_sha256": report,
                                  "split_manifest_sha256": split}},
            "manifest_sha256",
        )
        self.plan.invalidate()
        self.assertEqual(self.plan.status_of(self.plan.spec("dataset_manifest")).status, STATUS_PRESENT)
        write(self.plan.spec("at_solve").manifest, {"dataset_manifest_sha256": raw, "v": 2}, "report_sha256")
        self.plan.invalidate()
        status = self.plan.status_of(self.plan.spec("dataset_manifest"))
        self.assertEqual(status.status, STATUS_STALE)
        self.assertIn("independent_at_report_sha256", status.reason)

    def test_an_unconverged_at_report_is_not_a_built_cache(self) -> None:
        # the UK capture's first full solve: signed, usable, 80 iterations, not converged
        write = _caches_tests._write
        raw = write(self.plan.spec("raw_dataset_manifest").manifest, {"images": []}, "manifest_sha256")
        tri = write(self.plan.spec("at_triangulation").manifest, {"inputs": {}}, "triangulation_manifest_sha256")
        body = {"dataset_manifest_sha256": raw, "triangulation_identity": {"triangulation_manifest_sha256": tri},
                "solver_usable": True, "solver_converged": False, "intrinsic_outer_converged": False}
        write(self.plan.spec("at_solve").manifest, body, "report_sha256")
        self.plan.invalidate()
        status = self.plan.status_of(self.plan.spec("at_solve"))
        self.assertEqual(status.status, STATUS_STALE)
        self.assertIn("solver_converged", status.reason)
        body.update(solver_converged=True, intrinsic_outer_converged=True)
        write(self.plan.spec("at_solve").manifest, body, "report_sha256")
        self.plan.invalidate()
        self.assertEqual(self.plan.status_of(self.plan.spec("at_solve")).status, STATUS_PRESENT)

    def test_a_refused_time_sync_is_not_a_built_cache(self) -> None:
        write = _caches_tests._write
        raw = write(self.plan.spec("raw_dataset_manifest").manifest, {"images": []}, "manifest_sha256")
        write(self.plan.spec("timesync").manifest, {"base_dataset_manifest_sha256": raw, "accepted": False},
              "sdk_step_manifest_sha256")
        self.plan.invalidate()
        self.assertEqual(self.plan.status_of(self.plan.spec("timesync")).status, STATUS_STALE)

    def test_the_estimate_adds_the_at_route_only_when_asked(self) -> None:
        _, raw = estimate_ingest(3536, 4, validation_caches=("face_cache",))
        _, at = estimate_ingest(3536, 4, validation_caches=("face_cache",), pose_route=AT)
        self.assertGreater(at - raw, 4.0)  # features, triangulation, AT, raw tier


class AtStepsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_step_manifests_are_signed(self) -> None:
        signed = at_steps.sign_step_manifest({"kind": "x", "value": 1})
        self.assertEqual(at_steps.verify_step_manifest(signed), signed[at_steps.STEP_SHA_KEY])
        signed["value"] = 2
        with self.assertRaises(ValueError):
            at_steps.verify_step_manifest(signed)

    def test_fresh_clears_only_inside_the_work_root(self) -> None:
        output = self.root / "work" / "caches" / "triangulation"
        output.mkdir(parents=True)
        (output / "stale.db").write_bytes(b"x")
        with mock.patch.object(at_steps, "_run", return_value=0) as run:
            self.assertEqual(at_steps.fresh(output, self.root / "work", ["python", "tool.py"]), 0)
        self.assertFalse(output.exists())
        run.assert_called_once()
        with self.assertRaises(SystemExit):
            at_steps.fresh(self.root / "elsewhere", self.root / "work", ["python", "tool.py"])

    def _audit(self, best: float) -> tuple[int, dict]:
        raw = self.root / "raw.json"
        raw.write_text(json.dumps({"manifest_sha256": "a" * 64}), encoding="utf-8")
        model = self.root / "model.json"
        model.write_text(json.dumps(at_steps.sign_step_manifest(
            {"dataset_manifest_sha256": "a" * 64, "config": "c.json", "checkpoint": "m.pt"})), encoding="utf-8")
        output = self.root / "timesync"

        def fake_run(command):
            report = Path(command[command.index("--output") + 1])
            report.write_text(json.dumps({"base_dataset_manifest_sha256": "a" * 64, "best_offset_ms": best}),
                              encoding="utf-8")
            return 0

        args = at_steps.build_parser().parse_args([
            "timesync-audit", "--model-manifest", str(model), "--dataset-manifest", str(raw), "--output", str(output)])
        with mock.patch.object(at_steps, "_run", side_effect=fake_run):
            code = at_steps.timesync_audit(args)
        step = json.loads((output / at_steps.TIMESYNC_STEP_MANIFEST).read_text(encoding="utf-8"))
        return code, step

    def test_a_zero_offset_passes_and_is_signed(self) -> None:
        code, step = self._audit(0.0)
        self.assertEqual(code, 0)
        self.assertTrue(step["accepted"])
        at_steps.verify_step_manifest(step)

    def test_a_clock_offset_is_refused(self) -> None:
        code, step = self._audit(10.0)
        self.assertEqual(code, 3)
        self.assertFalse(step["accepted"])

    def test_the_smoke_config_is_one_strict_fixed_full_resolution_step(self) -> None:
        inputs = self.root / "tile_inputs_manifest.json"
        inputs.write_text(json.dumps({"tiles": [{"tile_id": 0, "initialization": {
            "path": "Tile_0/initialization_full_lidar.ply", "point_count": 1000}}]}), encoding="utf-8")
        geometry = self.root / "geometry" / "tile_geometry_manifest.json"
        geometry.parent.mkdir()
        geometry.write_text(json.dumps({"tiles": [{"tile_id": 0, "geometry": {
            "path": "Tile_0/initialization_geometry_k7_k30.npz"}}]}), encoding="utf-8")
        argv = ["pipeline-smoke", "--tile-inputs", str(inputs), "--tile-geometry-manifest", str(geometry),
                "--cap-max", "2000", "--output", str(self.root / "smoke")]
        for flag in ("gate", "tile-inputs-root", "dataset-manifest", "split-manifest", "mask-manifest", "mask-root",
                     "person-mask-manifest", "person-mask-root", "recording-root", "face-cache-manifest",
                     "face-cache-root", "renderer-mask-manifest", "depth-manifest", "depth-root",
                     "face-lidar-geometry-manifest", "face-lidar-geometry-root", "gsplat-lock"):
            argv += [f"--{flag}", str(self.root / flag)]
        config = at_steps.smoke_config(at_steps.build_parser().parse_args(argv))
        # trainer.py implementation_smoke_only: <=2 steps; factor 1 only with 1 strict-fixed step;
        # no evaluation; checkpoint only at the last step; no densification before the end
        self.assertTrue(config["implementation_smoke_only"])
        self.assertEqual((config["factor"], config["max_steps"], config["checkpoint_every"]), (1, 1, 1))
        self.assertEqual(config["topology_policy"], {"mode": "strict_fixed"})
        self.assertFalse(config["golden_evaluation"]["enabled"])
        self.assertFalse(config["final_evaluation_artifacts"])
        # gate_15 has no DA2 binding: a mono-depth key would be refused at start-up
        self.assertFalse([key for key in config if key.startswith("mono_depth")])
        self.assertEqual(config["mipmap_pipeline_gate"], str(self.root / "gate"))
        self.assertEqual(config["initialization_ply"], str(self.root / "tile-inputs-root" / "Tile_0" / "initialization_full_lidar.ply"))
        self.assertEqual(config["initialization_geometry"],
                         str(geometry.parent / "Tile_0" / "initialization_geometry_k7_k30.npz"))
        self.assertGreater(config["cap_max"], 1000)  # init < cap or the trainer refuses


def _inputs(root: Path) -> GateInputs:
    paths = {}
    for name in GateInputs.__dataclass_fields__:
        path = root / "inputs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        paths[name] = path
    return GateInputs(**paths)


class GateChainTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.inputs = _inputs(self.root)
        self.gates = self.root / "gates"
        self.calls: list[str] = []

    def _run(self, command) -> int:
        output = Path(command[command.index("--output") + 1])
        self.calls.append(output.name)
        output.write_text("{}", encoding="utf-8")
        return 0

    def _build(self):
        return build_gate_chain(self.inputs, self.gates, python="py", repo_root=self.root, cap_max=99, run=self._run)

    def _smoke_ran(self) -> None:
        smoke = self.gates / "pipeline_smoke"
        smoke.mkdir(parents=True, exist_ok=True)
        (smoke / "pipeline_smoke.json").write_text("{}", encoding="utf-8")

    def test_the_chain_stops_for_the_smoke_then_finishes(self) -> None:
        with self.assertRaises(GpuStepRequired) as caught:
            self._build()
        self.assertEqual(self.calls, list(GATE_FILES[:5]))
        self.assertEqual(caught.exception.cache, "pipeline_smoke")
        command = list(caught.exception.command)
        self.assertEqual(command[command.index("--gate") + 1], str(self.gates / "gate_15_upstream.json"))
        self.assertEqual(command[command.index("--cap-max") + 1], "99")
        self._smoke_ran()
        self.assertEqual(self._build(), self.gates / "gate_17_training.json")
        self.assertEqual(self.calls, list(GATE_FILES))

    def test_a_complete_chain_is_not_rebuilt(self) -> None:
        with self.assertRaises(GpuStepRequired):
            self._build()
        self._smoke_ran()
        self._build()
        self.calls.clear()
        self._build()
        self.assertEqual(self.calls, [])

    def test_a_moved_input_restarts_the_chain_from_gate_10(self) -> None:
        with self.assertRaises(GpuStepRequired):
            self._build()
        self._smoke_ran()
        self._build()
        self.inputs.at_report.write_text(json.dumps({"moved": True}), encoding="utf-8")
        self.calls.clear()
        with self.assertRaises(GpuStepRequired):  # the smoke ran under the old gate_15
            self._build()
        self.assertEqual(self.calls, list(GATE_FILES[:5]))

    def test_a_missing_input_or_a_refusing_tool_is_named(self) -> None:
        self.inputs.da2_val.unlink()
        with self.assertRaises(DatasetIncompleteError) as caught:
            self._build()
        self.assertIn("da2_val", str(caught.exception))
        self.inputs.da2_val.write_text("{}", encoding="utf-8")
        with self.assertRaises(DatasetIncompleteError) as caught:
            build_gate_chain(self.inputs, self.gates, python="py", repo_root=self.root, cap_max=1, run=lambda c: 1)
        self.assertIn("gate_10_frontend.json refused", str(caught.exception))

    def test_the_commands_follow_the_house0305_surface_route(self) -> None:
        commands = dict(gate_commands(self.inputs, self.gates, python="py", repo_root=self.root))
        frontend = list(commands["gate_10_frontend.json"])
        self.assertEqual(frontend[frontend.index("--time-sync-report") + 1], str(self.inputs.time_sync_report))
        self.assertEqual(sum(1 for part in frontend if part.startswith("--")), 15)
        tile = list(commands["gate_15_upstream.json"])
        self.assertEqual(tile[tile.index("--lidar-depth-gate") + 1], str(self.gates / "gate_12_lidar_depth.json"))
        self.assertEqual(tile[tile.index("--deferral-reason") + 1], SURFACE_DEFERRAL_REASON)
        promote = list(commands["gate_16_surface_frozen.json"])
        self.assertEqual(promote[promote.index("--fullres-smoke-manifest") + 1],
                         str(self.gates / "pipeline_smoke" / "run" / "run_manifest.json"))
        bind = list(commands["gate_17_training.json"])
        self.assertTrue(bind[1].endswith("bind_monocular_depth_gate.py"))
        self.assertEqual(bind[bind.index("--val-da2") + 1], str(self.inputs.da2_val))


class BundleAtRouteTest(unittest.TestCase):
    def test_the_bundle_builds_the_gate_and_trains_under_it(self) -> None:
        from cloudstudio3dgs_sdk import bundle as bundle_module
        from tests.test_sdk_bundle_build import FakePlan, FakeSpec, _bundle, _graph

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            work = root / "work"
            specs = _graph(work)
            plan = FakePlan(specs, present=[s.name for s in specs])
            init = work / "caches" / "global_init"
            init.mkdir(parents=True)
            (init / "sparse_pc.ply").write_bytes(b"ply")
            (init / "lidar_init_geometry.npz").write_bytes(b"npz")
            built = {}

            def fake_chain(inputs, gates_dir, **kwargs):
                built.update(gates_dir=gates_dir, **kwargs)
                return gates_dir / "gate_17_training.json"

            with mock.patch("cloudstudio3dgs_sdk.ingest.load_dataset", lambda *a, **k: _bundle(root)), \
                 mock.patch("cloudstudio3dgs_sdk.ingest.plan_caches", return_value=plan) as planned, \
                 mock.patch("cloudstudio3dgs_sdk.ingest.gates.build_gate_chain", fake_chain), \
                 mock.patch("cloudstudio3dgs_sdk.ingest.gates.gate_inputs_from_specs",
                            return_value=SimpleNamespace(tile_inputs=root / "ti.json")), \
                 mock.patch.object(bundle_module, "_smoke_cap", return_value=123), \
                 mock.patch("cloudstudio_3dgs.pipeline.mipmap_gate.load_and_verify_gate", return_value=({}, "sha")):
                scene = bundle_module.load_dataset_bundle(
                    root / "capture", SimpleNamespace(coarse_prior={"init_decimation_m": 2.0}),
                    work, python="python", repo_root=root / "repo", runner=lambda c: 0, pose_route=AT,
                )
            self.assertEqual(scene.pipeline_gate, work / "runs" / "gates" / "gate_17_training.json")
            self.assertEqual(scene.pose_route, AT)
            self.assertEqual(built["cap_max"], 123)
            for call in planned.call_args_list:
                self.assertEqual(call.kwargs["pose_route"], AT)


class CliPoseRouteTest(unittest.TestCase):
    def test_the_cli_defaults_to_the_at_route(self) -> None:
        from cloudstudio3dgs_sdk import __main__ as cli

        args = cli.build_parser().parse_args(["run", "--dataset", "d", "--work", "w"])
        self.assertEqual(args.pose_route, AT)
        args = cli.build_parser().parse_args(["run", "--dataset", "d", "--work", "w", "--pose-route", "raw_capture_poses"])
        with mock.patch.object(cli, "Project") as project:
            cli._project(args, open(os.devnull, "w", encoding="utf-8"))
        self.assertEqual(project.call_args.kwargs["pose_route"], "raw_capture_poses")


if __name__ == "__main__":
    unittest.main()


class AtRoutePreflightTest(unittest.TestCase):
    def test_a_runtime_off_the_lock_fails_and_missing_weights_warn(self) -> None:
        from cloudstudio3dgs_sdk import requirements

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(requirements, "_at_runtime_evidence", side_effect=RuntimeError("hloc commit x != locked y")), \
             mock.patch.object(requirements, "_torch_hub_checkpoints", return_value=Path(tmp)):
            rows = {row.name: row for row in requirements._at_route_checks(Path(tmp))}
        self.assertEqual(rows["at_runtime_lock"].status, "FAIL")
        self.assertIn("locked", rows["at_runtime_lock"].detail)
        self.assertEqual(rows["at_feature_weights"].status, "WARN")
        self.assertFalse(rows["at_feature_weights"].required)

    def test_only_a_fresh_capture_on_the_at_route_is_checked(self) -> None:
        from tests import test_sdk_project as _project_tests

        fixture = _project_tests.ProjectFixture()
        fixture.setUp()
        try:
            with mock.patch("cloudstudio3dgs_sdk.requirements._at_route_checks", return_value=[]) as checks:
                fixture.project(pose_route=AT).preflight(require_gpu=False)
                self.assertTrue(checks.called)
                checks.reset_mock()
                fixture.project().preflight(require_gpu=False)
                self.assertFalse(checks.called)
        finally:
            fixture.doCleanups()
