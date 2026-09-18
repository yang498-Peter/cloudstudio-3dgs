"""Adopting an already-prepared scene from its as-run trainer configs.

A synthetic scene is laid out the way house0305 is on disk: signed tile
inputs, tile geometry, ownership, backdrop, view-background, sky-mask and
Face4 manifests, the artefacts they hash, and one as-run config per tile plus
the coarse prior's. The configs are produced by the SDK's own ``tile_config``
so the round trip can be asserted exactly: adopt, plan, and the config the
plan writes must equal the as-run one except for its run identity.

Every refusal is tested with a real missing file or a real sha mismatch,
never a mocked check.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from cloudstudio3dgs_sdk.__main__ import main
from cloudstudio3dgs_sdk.adopt import adopt_scene, self_sha256, verify_prepare_manifest
from cloudstudio3dgs_sdk.plan import (
    DERIVED_SCENE_KEYS,
    DERIVED_TILE_KEYS,
    DatasetSummary,
    TileSummary,
    WorkLayout,
    coarse_config,
    tile_cap,
    tile_config,
    tile_key,
)
from cloudstudio3dgs_sdk.profile import PROFILE_B5FILL2
from cloudstudio3dgs_sdk.project import Project, StageRefused
from tests.test_sdk_plan import make_repo
from tools.pipeline import file_sha256


def ply_bytes(rows: int, salt: bytes = b"") -> bytes:
    return b"ply\nformat binary_little_endian 1.0\nelement vertex %d\nend_header\n" % rows + salt


def write_signed(path: Path, payload: dict, key: str) -> str:
    payload = dict(payload)
    payload.pop(key, None)
    payload[key] = self_sha256(payload, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload[key]


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


TILES = ((0, "Tile_0", 100, 5), (1, "Tile_1", 200, 9))


class AsRunScene:
    """house0305's on-disk shape in miniature, with every sha consistent."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset = root / "dataset"
        self.runs = root / "asrun"
        self.repo = make_repo(root / "repo")
        self.paths: dict[str, str] = {}
        self._build()

    # -- construction -------------------------------------------------------

    def _build(self) -> None:
        for key in PROFILE_B5FILL2.dataset_contract["trainer_path_keys"]:
            if key.endswith("_root"):
                target = self.dataset / key
                target.mkdir(parents=True, exist_ok=True)
            else:
                target = self.dataset / f"{key}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({"stub": key}), encoding="utf-8")
            self.paths[key] = str(target)
        face = {
            "schema_version": 1,
            "kind": "fisheye_face_cache",
            "split": "train",
            "images": [
                {"image_id": f"img{i}", "faces": [{"face_id": f"face{j}"} for j in range(4)]} for i in range(3)
            ],
            "summary": {"face_sample_count": 12, "image_count": 3},
        }
        self.face_sha = write_signed(Path(self.paths["face_cache_manifest"]), face, "face_manifest_sha256")

        las = write_bytes(self.dataset / "cloud.las", b"LASF" + b"\x00" * 64)
        inputs_root = self.runs / "tile_inputs"
        entries = []
        for tile_id, name, views, points in TILES:
            ply = write_bytes(inputs_root / name / "initialization_full_lidar.ply", ply_bytes(points, name.encode()))
            entries.append(
                {
                    "tile_id": tile_id,
                    "name": name,
                    "view_count": views,
                    "initialization": {
                        "kind": "full_lidar_roi_with_tile_halo",
                        "path": f"{name}/initialization_full_lidar.ply",
                        "point_count": points,
                        "sha256": file_sha256(ply),
                        "bytes": ply.stat().st_size,
                    },
                }
            )
            self.paths[tile_key(tile_id, "initialization_ply")] = str(ply)
        tile_inputs = {
            "schema_version": 1,
            "kind": "lidar_adaptive_tile_training_inputs_v1",
            "source_point_cloud": {"path": str(las), "sha256": file_sha256(las)},
            "tile_count": len(TILES),
            "tiles": entries,
        }
        self.tile_inputs_manifest = inputs_root / "tile_inputs_manifest.json"
        self.tile_inputs_sha = write_signed(self.tile_inputs_manifest, tile_inputs, "tile_inputs_manifest_sha256")
        self.paths["tile_inputs_manifest"] = str(self.tile_inputs_manifest)
        self.paths["tile_inputs_root"] = str(inputs_root)
        self.paths["lidar_cloud"] = str(las)

        geometry_root = self.runs / "tile_geometry"
        geo_entries = []
        for tile_id, name, _, points in TILES:
            npz = write_bytes(geometry_root / name / "initialization_geometry_k7_k30.npz", b"NPZ" + name.encode())
            geo_entries.append(
                {
                    "tile_id": tile_id,
                    "name": name,
                    "initialization_ply_sha256": entries[tile_id]["initialization"]["sha256"],
                    "point_count": points,
                    "geometry": {"path": f"{name}/initialization_geometry_k7_k30.npz", "sha256": file_sha256(npz)},
                }
            )
            self.paths[tile_key(tile_id, "initialization_geometry")] = str(npz)
        self.geometry_manifest = geometry_root / "tile_geometry_manifest.json"
        write_signed(
            self.geometry_manifest,
            {"schema_version": 1, "tile_inputs_manifest_sha256": self.tile_inputs_sha, "tiles": geo_entries},
            "tile_geometry_manifest_sha256",
        )
        self.paths["tile_geometry_manifest"] = str(self.geometry_manifest)
        self.paths["gsplat_lock"] = str(self.repo / "upstream" / "gsplat.lock.json")

        global_ply = write_bytes(self.dataset / "init_2m" / "sparse_pc.ply", ply_bytes(42))
        global_npz = write_bytes(self.dataset / "init_2m" / "lidar_init_geometry.npz", b"NPZ-global")
        self.paths["global_init_ply"] = str(global_ply)
        self.paths["global_init_geometry"] = str(global_npz)

        sky_root = self.dataset / "sky_mask_train"
        masks = []
        for i in range(3):
            for j in range(4):
                rel = f"faces/img{i}_face{j}_sky.png"
                mask = write_bytes(sky_root / rel, b"PNG" + f"{i}{j}".encode())
                masks.append({"image_id": f"img{i}", "face_id": f"face{j}", "mask_path": rel, "mask_sha256": file_sha256(mask)})
        self.sky_manifest = sky_root / "sky_mask_train.json"
        write_signed(
            self.sky_manifest,
            {"schema_version": 1, "kind": "face4_sky_mask_cache", "source_face_manifest_sha256": self.face_sha, "masks": masks},
            "sky_mask_manifest_sha256",
        )
        self.paths["sky_mask_manifest"] = str(self.sky_manifest)
        self.paths["sky_mask_root"] = str(sky_root)

        self.dome = write_bytes(self.runs / "probes" / "sky_dome.pt", b"DOME" * 16)
        self.dome_sha = file_sha256(self.dome)
        self.paths["sky_dome_checkpoint"] = str(self.dome)
        self.sky_ply = write_bytes(self.runs / "exports" / "sky.ply", ply_bytes(7))
        self.paths["sky_dome_ply"] = str(self.sky_ply)

        for tile_id, name, views, _ in TILES:
            own_root = self.runs / "tile_ownership" / name
            own = own_root / "tile_ownership_manifest.json"
            write_signed(
                own,
                {
                    "schema_version": 1,
                    "kind": "face4_tile_ownership_mask_cache",
                    "tile_id": tile_id,
                    "tile_inputs_manifest_sha256": self.tile_inputs_sha,
                    "source_face_manifest_sha256": self.face_sha,
                    "records": [{"sample_id": f"s{k}"} for k in range(views // 50)],
                },
                "tile_ownership_manifest_sha256",
            )
            self.paths[tile_key(tile_id, "ownership_manifest")] = str(own)
            self.paths[tile_key(tile_id, "ownership_root")] = str(own_root)
            bd_root = self.runs / "tile_backgrounds" / name
            bd = bd_root / "background_manifest.json"
            write_signed(
                bd,
                {
                    "schema_version": 1,
                    "split": "train",
                    "tile_id": tile_id,
                    "dome_source": str(self.dome),
                    "dome_sha256": self.dome_sha,
                    "source_tile_inputs_manifest_sha256": self.tile_inputs_sha,
                    "standin": {"schema_version": 1, "dome_count": 7},
                    "views": {},
                },
                "manifest_sha256",
            )
            self.paths[tile_key(tile_id, "backdrop_manifest")] = str(bd)
            self.paths[tile_key(tile_id, "backdrop_root")] = str(bd_root)
        vb_root = self.runs / "view_backgrounds"
        vb = vb_root / "view_background_manifest_train.json"
        write_signed(
            vb,
            {"schema_version": 1, "split": "train", "dome_source": str(self.dome), "dome_sha256": self.dome_sha, "views": {}},
            "manifest_sha256",
        )
        self.paths["global_view_backgrounds_manifest"] = str(vb)
        self.paths["global_view_backgrounds_root"] = str(vb_root)

        # The as-run configs: the SDK's own derivation against these paths,
        # with the run identity a human would have given them.
        summary = DatasetSummary(
            scene_tag="synth",
            tiles=tuple(TileSummary(t, n, v, p) for t, n, v, p in TILES),
            train_view_count=12,
            global_init_point_count=42,
        )
        layout = WorkLayout(self.runs)
        self.tile_configs: dict[int, Path] = {}
        for tile in summary.tiles:
            config = tile_config(
                PROFILE_B5FILL2,
                summary,
                tile,
                generation="delivery",
                layout=layout,
                bundle_paths=self.paths,
                cap_max=tile_cap(PROFILE_B5FILL2, tile),
            )
            config["run_id"] = f"synth-t{tile.tile_id}-B5-cap6"
            config["output_dir"] = str(self.runs / f"tile{tile.tile_id}_B5_cap6_20k")
            config["lineage"] = {"base": "hand-written", "single_change": "none"}
            target = self.runs / f"tile{tile.tile_id}_B5_cap6_20k" / "config_as_run.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(config, indent=1), encoding="utf-8")
            self.tile_configs[tile.tile_id] = target
        coarse = coarse_config(PROFILE_B5FILL2, summary, layout=layout, bundle_paths=self.paths)
        coarse["run_id"] = "synth-B0-global-coarse-10k"
        coarse["output_dir"] = str(self.runs / "global_coarse_B0_10k")
        coarse["lineage"] = {"base": "hand-written", "single_change": "none"}
        self.coarse_config = self.runs / "synth_global_coarse_B0_10k.json"
        self.coarse_config.write_text(json.dumps(coarse, indent=1), encoding="utf-8")

    # -- helpers ------------------------------------------------------------

    def adopt(self, **overrides):
        kwargs = {
            "profile": PROFILE_B5FILL2,
            "sky_ply": self.sky_ply,
        }
        kwargs.update(overrides)
        return adopt_scene([self.tile_configs[0], self.tile_configs[1]], self.coarse_config, **kwargs)

    def project(self, work: Path, **overrides) -> Project:
        kwargs = {"repo_root": self.repo, "python": Path("python.exe"), "stream": io.StringIO()}
        kwargs.update(overrides)
        return Project(self.dataset, work, PROFILE_B5FILL2, **kwargs)


class AdoptFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.scene = AsRunScene(self.root)
        self.work = self.root / "work"

    def adopted_project(self) -> Project:
        adopted = self.scene.adopt()
        project = self.scene.project(self.work)
        project.write_prepare_manifest(
            adopted.scene, adopted.dataset, prior_tile_checkpoints=adopted.prior_tile_checkpoints,
            derived_paths=adopted.derived_paths, digests=adopted.digests, adopted=adopted.as_json(),
        )
        return project


class AdoptSucceedsTests(AdoptFixture):
    def test_without_telemetry_no_tile_gets_a_cap_floor_and_the_note_says_so(self) -> None:
        # The floor rule needs the previous generation's final population. The fixture plants
        # no trainer telemetry, so adopt must leave the floor unset rather than guess.
        adopted = self.scene.adopt()
        self.assertTrue(all(t.previous_final_population is None for t in adopted.dataset.tiles))
        self.assertTrue(any("no trainer telemetry" in note for note in adopted.notes), adopted.notes)

    def test_final_population_is_read_from_trainer_telemetry_when_present(self) -> None:
        # house0305's Tile_2 as-run cap was 8.0M, hand-set from an earlier generation; the rule
        # applied to the adopted arm's own final population gives 6.8M. adopt must read the
        # exact count from monitor/progress.jsonl, not load a 2 GB checkpoint to count rows.
        telemetry = self.scene.runs / "tile1_B5_cap6_20k" / "monitor" / "progress.jsonl"
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        telemetry.write_text(
            '{"completed_steps": 10, "gaussian_count": 5}\n'
            '{"completed_steps": 20000, "gaussian_count": 777}\n',
            encoding="utf-8",
        )
        adopted = self.scene.adopt()
        by_id = {t.tile_id: t for t in adopted.dataset.tiles}
        self.assertEqual(by_id[1].previous_final_population, 777)
        self.assertIsNone(by_id[0].previous_final_population)
        self.assertTrue(any("tile 1=777" in note for note in adopted.notes), adopted.notes)

    def test_dataset_summary_is_read_from_the_manifests(self) -> None:
        adopted = self.scene.adopt()
        dataset = adopted.dataset
        self.assertEqual(dataset.scene_tag, "synth")
        self.assertEqual([(t.tile_id, t.name, t.view_count, t.init_point_count) for t in dataset.tiles],
                         [(0, "Tile_0", 100, 5), (1, "Tile_1", 200, 9)])
        self.assertEqual(dataset.train_view_count, 12)
        self.assertEqual(dataset.global_init_point_count, 42)
        self.assertEqual(dataset.lidar_point_count, 0)
        self.assertTrue(any("lidar_point_count is 0" in note for note in adopted.notes))
        self.assertFalse(dataset.estimated)

    def test_every_derived_key_is_bound_and_verified(self) -> None:
        adopted = self.scene.adopt()
        for key in DERIVED_SCENE_KEYS:
            if key == "delivery_eval_config":
                continue
            self.assertIn(key, adopted.derived_paths, key)
        for tile_id, _, _, _ in TILES:
            for suffix in DERIVED_TILE_KEYS:
                self.assertIn(tile_key(tile_id, suffix), adopted.derived_paths)
        self.assertEqual(adopted.derived_paths, {k: v for k, v in self.scene.paths.items() if k in adopted.derived_paths})
        # every hashed artefact carries a sha256 digest
        for key in ("tile_inputs_manifest", "lidar_cloud", "sky_dome_checkpoint", "sky_dome_ply",
                    "tile1_initialization_ply", "tile1_initialization_geometry", "tile1_ownership_manifest",
                    "tile1_backdrop_manifest", "global_view_backgrounds_manifest", "sky_mask_manifest",
                    "face_cache_manifest", "gsplat_lock", "global_init_ply"):
            self.assertEqual(adopted.digests[key]["digest_kind"], "sha256", key)
        self.assertTrue(any("12 mask files" in line for line in adopted.verified))

    def test_bundle_paths_merges_trainer_and_derived_paths(self) -> None:
        project = self.adopted_project()
        paths = project.bundle_paths()
        self.assertEqual(paths["dataset_manifest"], self.scene.paths["dataset_manifest"])
        self.assertEqual(paths["tile1_ownership_root"], self.scene.paths["tile1_ownership_root"])
        self.assertEqual(paths["sky_dome_checkpoint"], str(self.scene.dome))
        self.assertEqual(paths["gsplat_lock"], self.scene.paths["gsplat_lock"])

    def test_a_plan_from_the_adopted_manifest_has_no_placeholders(self) -> None:
        project = self.adopted_project()
        plan = project.plan()
        self.assertEqual(plan.placeholders(), ())
        self.assertNotIn("<prepare:", plan.render())
        self.assertNotIn("<prepare:", json.dumps(plan.as_json()))

    def test_cache_building_steps_are_skippable_on_an_adopted_scene(self) -> None:
        project = self.adopted_project()
        plan = project.plan()
        skippable = {step.name for step in plan.skippable_steps()}
        for name in ("sky_masks", "sky_dome", "sky_dome_ply", "ownership_Tile_0", "ownership_Tile_1",
                     "global_view_backgrounds", "backdrop_Tile_0", "backdrop_Tile_1"):
            self.assertIn(name, skippable, name)
        transcript = plan.render()
        self.assertIn("sky_dome_ply", transcript)
        self.assertIn("skip: outputs present", transcript)
        result = project.prepare()
        self.assertEqual(result.action, "ran")
        self.assertEqual(sorted(result.steps_run), ["delivery_eval_config", "write_arm_configs"])
        for name in ("ingest_dataset", "sky_masks", "sky_dome", "sky_dome_ply", "ownership_Tile_0", "ownership_Tile_1"):
            self.assertIn(name, result.steps_skipped, name)

    def test_force_never_rebuilds_an_adopted_artefact_in_place(self) -> None:
        """The adopted caches are somebody else's files; --force re-runs only what this work root built."""

        class Runner:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def __call__(self, step, *, log):
                self.calls.append(step.name)
                for output in step.outputs:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    Path(output).write_bytes(b"rebuilt")
                return 0

        runner = Runner()
        adopted = self.scene.adopt()
        project = self.scene.project(self.work, runner=runner)
        project.write_prepare_manifest(
            adopted.scene, adopted.dataset, derived_paths=adopted.derived_paths, digests=adopted.digests,
        )
        project.prepare()
        before = {key: Path(path).read_bytes() for key, path in self.scene.paths.items() if Path(path).is_file()}
        result = project.prepare(force=True)
        self.assertEqual(result.action, "ran")
        # ingest_dataset "runs" under force but only re-verifies the manifest.
        self.assertEqual(sorted(result.steps_run), ["delivery_eval_config", "ingest_dataset", "write_arm_configs"])
        for name in ("sky_masks", "sky_dome", "sky_dome_ply", "ownership_Tile_0", "ownership_Tile_1"):
            self.assertIn(name, result.steps_skipped, name)
            self.assertNotIn(name, runner.calls, name)
        after = {key: Path(path).read_bytes() for key, path in self.scene.paths.items() if Path(path).is_file()}
        self.assertEqual(before, after)

    def test_the_written_tile_config_equals_the_as_run_config_except_identity(self) -> None:
        project = self.adopted_project()
        project.prepare()
        written = json.loads((self.work / "runs" / "tile1_b5fill2_delivery.json").read_text(encoding="utf-8"))
        asrun = json.loads(self.scene.tile_configs[1].read_text(encoding="utf-8"))
        differing = sorted(key for key in set(written) | set(asrun) if written.get(key) != asrun.get(key))
        self.assertEqual(differing, ["lineage", "output_dir", "run_id"])
        coarse = json.loads((self.work / "runs" / "global_coarse_b5fill2.json").read_text(encoding="utf-8"))
        asrun_coarse = json.loads(self.scene.coarse_config.read_text(encoding="utf-8"))
        differing = sorted(key for key in set(coarse) | set(asrun_coarse) if coarse.get(key) != asrun_coarse.get(key))
        self.assertEqual(differing, ["lineage", "output_dir", "run_id"])

    def test_pipeline_config_points_at_the_adopted_sky_ply_and_eval_config(self) -> None:
        project = self.adopted_project()
        project.prepare()
        pipeline = json.loads((self.work / "pipeline.json").read_text(encoding="utf-8"))
        self.assertEqual(pipeline["sky_ply"], str(self.scene.sky_ply))
        self.assertEqual(pipeline["tile_inputs_manifest"], str(self.scene.tile_inputs_manifest))
        self.assertEqual(pipeline["delivery_eval_config"], str(self.work / "delivery_eval.json"))
        self.assertTrue((self.work / "delivery_eval.json").is_file())

    def test_as_run_checkpoints_become_prior_tile_checkpoints(self) -> None:
        adopted = self.scene.adopt()
        self.assertEqual(adopted.prior_tile_checkpoints, {})
        for tile_id, target in self.scene.tile_configs.items():
            write_bytes(target.parent / "checkpoints" / "latest.pt", b"CKPT")
        adopted = self.scene.adopt()
        self.assertEqual(sorted(adopted.prior_tile_checkpoints), [0, 1])
        project = self.scene.project(self.work)
        project.write_prepare_manifest(
            adopted.scene, adopted.dataset, prior_tile_checkpoints=adopted.prior_tile_checkpoints,
            derived_paths=adopted.derived_paths, digests=adopted.digests,
        )
        plan = project.plan()
        self.assertEqual(plan.generations, ("delivery",))
        backdrop = next(s for s in plan.steps if s.name == "backdrop_Tile_0")
        self.assertIn(adopted.prior_tile_checkpoints[1], backdrop.command)

    def test_without_a_sky_ply_the_export_step_runs_into_the_layout(self) -> None:
        adopted = self.scene.adopt(sky_ply=None)
        self.assertNotIn("sky_dome_ply", adopted.derived_paths)
        project = self.scene.project(self.work)
        project.write_prepare_manifest(adopted.scene, adopted.dataset, derived_paths=adopted.derived_paths)
        step = next(s for s in project.plan().steps if s.name == "sky_dome_ply")
        self.assertEqual(step.outputs, (str(self.work / "caches" / "sky_dome.ply"),))
        self.assertIn(str(self.scene.dome), step.command)

    def test_re_verification_passes_then_refuses_when_an_artefact_moves(self) -> None:
        project = self.adopted_project()
        payload = json.loads(project.layout.prepare_manifest.read_text(encoding="utf-8"))
        checked = verify_prepare_manifest(payload)
        self.assertIn("digest:sky_dome_checkpoint", checked)
        self.scene.dome.write_bytes(b"DOME" * 15)
        with self.assertRaises(StageRefused) as caught:
            verify_prepare_manifest(payload)
        self.assertIn("sky_dome_checkpoint", str(caught.exception))
        # and prepare() itself, adopting the manifest, refuses the same way
        with self.assertRaises(StageRefused):
            project.prepare()


class AdoptFailsClosedTests(AdoptFixture):
    def test_a_missing_file_is_refused_by_name(self) -> None:
        npz = Path(self.scene.paths["tile1_initialization_geometry"])
        npz.unlink()
        with self.assertRaises(StageRefused) as caught:
            self.scene.adopt()
        self.assertIn(str(npz), str(caught.exception))

    def test_a_missing_sky_mask_file_is_refused_by_name(self) -> None:
        mask = Path(self.scene.paths["sky_mask_root"]) / "faces" / "img2_face3_sky.png"
        mask.unlink()
        with self.assertRaises(StageRefused) as caught:
            self.scene.adopt()
        self.assertIn(str(mask), str(caught.exception))

    def test_a_sha_mismatch_is_refused_by_name(self) -> None:
        ply = Path(self.scene.paths["tile1_initialization_ply"])
        ply.write_bytes(ply_bytes(9, b"rewritten"))
        with self.assertRaises(StageRefused) as caught:
            self.scene.adopt()
        message = str(caught.exception)
        self.assertIn(str(ply), message)
        self.assertIn("sha256", message)

    def test_a_dome_that_is_not_the_one_the_backdrops_rendered_is_refused(self) -> None:
        other = write_bytes(self.root / "other_dome.pt", b"OTHER")
        with self.assertRaises(StageRefused) as caught:
            self.scene.adopt(sky_dome=other)
        self.assertIn(str(other), str(caught.exception))
        self.assertIn("dome_sha256", str(caught.exception))

    def test_an_edited_signed_manifest_is_refused(self) -> None:
        own = Path(self.scene.paths["tile0_ownership_manifest"])
        payload = json.loads(own.read_text(encoding="utf-8"))
        payload["tile_id"] = 0
        payload["records"].append({"sample_id": "smuggled"})
        own.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(StageRefused) as caught:
            self.scene.adopt()
        self.assertIn(str(own), str(caught.exception))
        self.assertIn("tile_ownership_manifest_sha256", str(caught.exception))

    def test_a_sky_ply_with_the_wrong_row_count_is_refused(self) -> None:
        wrong = write_bytes(self.root / "wrong_sky.ply", ply_bytes(8))
        with self.assertRaises(StageRefused) as caught:
            self.scene.adopt(sky_ply=wrong)
        self.assertIn("8 rows", str(caught.exception))

    def test_a_missing_tile_config_is_refused(self) -> None:
        with self.assertRaises(StageRefused) as caught:
            adopt_scene([self.scene.tile_configs[0]], self.scene.coarse_config, profile=PROFILE_B5FILL2)
        self.assertIn("[1]", str(caught.exception))

    def test_configs_that_disagree_on_the_dataset_are_refused(self) -> None:
        config = json.loads(self.scene.tile_configs[1].read_text(encoding="utf-8"))
        config["split_manifest"] = str(self.root / "elsewhere" / "split_manifest.json")
        self.scene.tile_configs[1].write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(StageRefused) as caught:
            self.scene.adopt()
        self.assertIn("split_manifest", str(caught.exception))
        self.assertIn("disagree", str(caught.exception))

    def test_a_config_whose_init_is_not_the_manifests_is_refused(self) -> None:
        config = json.loads(self.scene.tile_configs[0].read_text(encoding="utf-8"))
        config["initialization_ply"] = self.scene.paths["tile1_initialization_ply"]
        self.scene.tile_configs[0].write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(StageRefused) as caught:
            self.scene.adopt()
        self.assertIn("Tile_0", str(caught.exception))
        self.assertIn("initialization_ply", str(caught.exception))


class AdoptCliTests(AdoptFixture):
    def run_cli(self, *argv: str) -> tuple[int, str]:
        stream = io.StringIO()
        code = main(list(argv), stream=stream)
        return code, stream.getvalue()

    def test_adopt_then_dry_run_has_no_placeholders_and_marks_skips(self) -> None:
        code, text = self.run_cli(
            "adopt",
            "--work", str(self.work),
            "--tile-config", str(self.scene.tile_configs[0]),
            "--tile-config", str(self.scene.tile_configs[1]),
            "--coarse-config", str(self.scene.coarse_config),
            "--sky-ply", str(self.scene.sky_ply),
            "--repo-root", str(self.scene.repo),
            "--python", "python.exe",
        )
        self.assertEqual(code, 0, text)
        self.assertIn("adopted synth: 2 tiles, 12 training faces", text)
        self.assertTrue((self.work / "prepare" / "prepare_manifest.json").is_file())
        code, text = self.run_cli(
            "run",
            "--dataset", str(self.scene.dataset),
            "--work", str(self.work),
            "--dry-run",
            "--repo-root", str(self.scene.repo),
            "--python", "python.exe",
        )
        self.assertEqual(code, 0, text)
        self.assertNotIn("<prepare:", text)
        self.assertNotIn("ESTIMATED", text)
        for name in ("sky_masks", "sky_dome", "sky_dome_ply", "ownership_Tile_1"):
            line = next(line for line in text.splitlines() if f" {name} " in line)
            self.assertIn("skip: outputs present", line, line)

    def test_adopt_refuses_with_exit_1_naming_the_file(self) -> None:
        lock = Path(self.scene.paths["gsplat_lock"])
        lock.unlink()
        import contextlib

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = self.run_cli(
                "adopt",
                "--work", str(self.work),
                "--tile-config", str(self.scene.tile_configs[0]),
                "--tile-config", str(self.scene.tile_configs[1]),
                "--coarse-config", str(self.scene.coarse_config),
            )
        self.assertEqual(code, 1)
        self.assertIn(str(lock), err.getvalue())
        self.assertFalse((self.work / "prepare" / "prepare_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
