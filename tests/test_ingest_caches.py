"""CachePlan: ordering, sha-binding skip logic, GPU refusal, capability gating.

No builder is ever executed: ``build(dry_run=False)`` is driven through an
injected ``runner`` that writes the manifest the real tool would have written.
CPU-only, torch-free.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes  # noqa: E402
from cloudstudio3dgs_sdk.ingest.bundle import (  # noqa: E402
    BundleImage,
    CameraIntrinsics,
    DatasetBundle,
    PointCloudRef,
    RigTransform,
)
from cloudstudio3dgs_sdk.ingest.caches import (  # noqa: E402
    CPU,
    GPU,
    STATUS_BLOCKED,
    STATUS_MISSING,
    STATUS_PRESENT,
    STATUS_STALE,
    CachePlan,
    CachePlanError,
    CacheProfile,
    build_cache_specs,
    plan_caches,
)
from cloudstudio3dgs_sdk.ingest.errors import (  # noqa: E402
    DatasetIncompleteError,
    GpuStepRequired,
)

IDENTITY = tuple(
    tuple(1.0 if row == column else 0.0 for column in range(4)) for row in range(4)
)


def _sign(payload: dict, key: str) -> dict:
    body = {k: v for k, v in payload.items() if k != key}
    body[key] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def _write(path: Path, payload: dict, key: str) -> str:
    signed = _sign(payload, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signed, indent=1), encoding="utf-8")
    return signed[key]


def _bundle(root: Path, *, lidar: bool = True) -> DatasetBundle:
    images_root = root / "recording"
    (images_root / "camera" / "left").mkdir(parents=True, exist_ok=True)
    image = images_root / "camera" / "left" / "0.jpg"
    image.write_bytes(b"jpeg")
    cloud = None
    if lidar:
        cloud_path = root / "recording" / "colorized.las"
        cloud_path.write_bytes(b"LASF")
        cloud = PointCloudRef(path=cloud_path, format="las", sha256="a" * 64)
    return DatasetBundle(
        dataset_id="synthetic",
        adapter="unit_test",
        source_root=images_root,
        images_root=images_root,
        cameras=(
            CameraIntrinsics(
                camera_id="left",
                width=64,
                height=64,
                fx=20.0,
                fy=20.0,
                cx=32.0,
                cy=32.0,
                camera_model="OPENCV_FISHEYE",
                transform_from_lidar=RigTransform(
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)
                ),
            ),
            CameraIntrinsics(
                camera_id="right",
                width=64,
                height=64,
                fx=20.0,
                fy=20.0,
                cx=32.0,
                cy=32.0,
                camera_model="OPENCV_FISHEYE",
                transform_from_lidar=RigTransform(
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.1, 0.0, 0.0)
                ),
            ),
        ),
        images=(
            BundleImage(
                image_id="img_0",
                camera_id="left",
                path="camera/left/0.jpg",
                c2w=IDENTITY,
                timestamp_ns=1,
            ),
        ),
        point_cloud=cloud,
        coordinate_frame="s1_local",
    )


def _profile(root: Path, **overrides) -> CacheProfile:
    values = dict(
        dataset_root=root / "dataset",
        cache_root=root / "cache",
        run_root=root / "run",
        recording_root=root / "recording",
        source_run_dir=root / "recording",
        tile_count=2,
        person_weights=root / "weights.pth",
        da2_model_source=root / "da2",
        da2_checkpoint=root / "da2" / "model.pth",
        trainer_config=root / "config.json",
    )
    values.update(overrides)
    return CacheProfile(**values)


class CacheInventoryTest(unittest.TestCase):
    def test_inventory_covers_every_cache_the_recipe_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_cache_specs(_bundle(root), _profile(root))
            names = {spec.name for spec in specs}
            for expected in (
                "dataset_manifest",
                "mask_manifest",
                "person_mask_manifest",
                "depth_cache",
                "split_manifest",
                "face_cache",
                "renderer_mask",
                "face_lidar_geometry",
                "mono_depth",
                "sky_masks",
                "tile_plan",
                "tile_inputs",
                "tile_geometry",
                "tile_ownership_0",
                "tile_ownership_1",
                "view_backgrounds_0",
                "view_backgrounds_1",
            ):
                self.assertIn(expected, names)

    def test_gpu_flags_match_the_builders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = {s.name: s for s in build_cache_specs(_bundle(root), _profile(root))}
            self.assertEqual(specs["mono_depth"].device, GPU)
            self.assertEqual(specs["person_mask_manifest"].device, GPU)
            self.assertEqual(specs["view_backgrounds_0"].device, GPU)
            # SegFormer sky masks run on CPU by design; only --allow-cuda changes that.
            self.assertEqual(specs["sky_masks"].device, CPU)
            self.assertEqual(specs["face_lidar_geometry"].device, CPU)
            self.assertEqual(specs["tile_geometry"].device, CPU)

    def test_every_cache_states_a_cost_and_its_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for spec in build_cache_specs(_bundle(root), _profile(root)):
                self.assertGreater(spec.estimated_minutes, 0.0, spec.name)
                self.assertTrue(
                    spec.cost_basis.startswith(("measured", "estimated")), spec.name
                )

    def test_optional_stages_can_be_switched_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _profile(
                root,
                person_masks=False,
                mono_depth=False,
                sky_masks=False,
                view_backgrounds=False,
            )
            names = {s.name for s in build_cache_specs(_bundle(root), profile)}
            self.assertNotIn("mono_depth", names)
            self.assertNotIn("sky_masks", names)
            self.assertNotIn("person_mask_manifest", names)
            self.assertFalse(any(n.startswith("view_backgrounds") for n in names))

    def test_profile_from_mapping_and_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {
                "dataset_root": root / "d",
                "cache_root": root / "c",
                "run_root": root / "r",
                "recording_root": root / "rec",
                "source_run_dir": root / "rec",
                "unknown_key": 1,
            }
            profile = CacheProfile.from_any(mapping)
            self.assertEqual(profile.dataset_root, root / "d")

            class Foreign:
                dataset_root = root / "d"
                cache_root = root / "c"
                run_root = root / "r"
                recording_root = root / "rec"
                source_run_dir = root / "rec"
                tile_count = 3

            self.assertEqual(CacheProfile.from_any(Foreign()).tile_count, 3)

    def test_profile_without_roots_fails_closed(self) -> None:
        with self.assertRaises(CachePlanError) as caught:
            CacheProfile.from_any({"tile_count": 4})
        self.assertIn("missing required roots", str(caught.exception))


class CacheOrderingTest(unittest.TestCase):
    def test_dependencies_precede_dependents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            position = {spec.name: index for index, spec in enumerate(plan)}
            for spec in plan:
                for dependency in spec.depends_on:
                    self.assertIn(dependency, position, spec.name)
                    self.assertLess(position[dependency], position[spec.name], spec.name)

    def test_face_cache_precedes_everything_face_shaped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            position = {spec.name: index for index, spec in enumerate(plan)}
            for consumer in ("renderer_mask", "face_lidar_geometry", "mono_depth", "sky_masks"):
                self.assertLess(position["face_cache"], position[consumer])

    def test_face_cache_asks_for_the_mipmap_face4_layout(self) -> None:
        # build_face_cache.py defaults to adaptive_full_fov; the recipe's caches are the four
        # MipMap-aligned faces (house0305 v9: --face-plan mipmap_face4, 884 images -> 3536 faces)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = list(CachePlan(_bundle(root), _profile(root)).spec("face_cache").command)
            self.assertEqual(command[command.index("--face-plan") + 1], "mipmap_face4")


class CacheStatusTest(unittest.TestCase):
    def _plan(self, root: Path, **kwargs) -> CachePlan:
        return CachePlan(_bundle(root, **kwargs), _profile(root))

    def test_nothing_on_disk_is_all_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            statuses = {s.spec.name: s for s in plan.statuses()}
            self.assertEqual(statuses["dataset_manifest"].status, STATUS_MISSING)
            self.assertEqual(len(plan.pending()), len(plan.specs))
            self.assertGreater(plan.total_minutes(), 0.0)

    def test_matching_bindings_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            dataset_sha = _write(
                plan.spec("dataset_manifest").manifest,
                {"schema_version": 1, "images": []},
                "manifest_sha256",
            )
            _write(
                plan.spec("mask_manifest").manifest,
                {"schema_version": 1, "dataset_manifest_sha256": dataset_sha},
                "mask_manifest_sha256",
            )
            plan.invalidate()
            statuses = {s.spec.name: s for s in plan.statuses()}
            self.assertEqual(statuses["dataset_manifest"].status, STATUS_PRESENT)
            self.assertEqual(statuses["mask_manifest"].status, STATUS_PRESENT)
            self.assertNotIn(
                "mask_manifest", {s.spec.name for s in plan.pending()}
            )

    def test_changed_upstream_sha_makes_the_cache_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            _write(
                plan.spec("dataset_manifest").manifest,
                {"schema_version": 1, "images": []},
                "manifest_sha256",
            )
            _write(
                plan.spec("mask_manifest").manifest,
                {"schema_version": 1, "dataset_manifest_sha256": "b" * 64},
                "mask_manifest_sha256",
            )
            plan.invalidate()
            status = plan.status_of(plan.spec("mask_manifest"))
            self.assertEqual(status.status, STATUS_STALE)
            self.assertIn("dataset_manifest_sha256", status.reason)

    def test_nested_binding_keys_are_followed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            shas = {}
            for name, payload, key in (
                ("dataset_manifest", {"images": []}, "manifest_sha256"),
                ("mask_manifest", {}, "mask_manifest_sha256"),
                ("person_mask_manifest", {}, "person_mask_manifest_sha256"),
                ("depth_cache", {}, "depth_manifest_sha256"),
                ("split_manifest", {}, "split_manifest_sha256"),
            ):
                shas[name] = _write(plan.spec(name).manifest, dict(payload), key)
            _write(
                plan.spec("face_cache").manifest,
                {
                    "kind": "fisheye_face_cache",
                    "source_identity": {
                        "dataset_manifest_sha256": shas["dataset_manifest"],
                        "mask_manifest_sha256": shas["mask_manifest"],
                        "person_mask_manifest_sha256": shas["person_mask_manifest"],
                        "depth_manifest_sha256": shas["depth_cache"],
                        "split_manifest_sha256": shas["split_manifest"],
                    },
                },
                "face_manifest_sha256",
            )
            plan.invalidate()
            self.assertEqual(
                plan.status_of(plan.spec("face_cache")).status, STATUS_PRESENT
            )

    def test_optional_binding_may_be_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            shas = {}
            for name, key in (
                ("dataset_manifest", "manifest_sha256"),
                ("mask_manifest", "mask_manifest_sha256"),
                ("depth_cache", "depth_manifest_sha256"),
                ("split_manifest", "split_manifest_sha256"),
            ):
                shas[name] = _write(plan.spec(name).manifest, {}, key)
            _write(
                plan.spec("face_cache").manifest,
                {
                    "source_identity": {
                        "dataset_manifest_sha256": shas["dataset_manifest"],
                        "mask_manifest_sha256": shas["mask_manifest"],
                        "depth_manifest_sha256": shas["depth_cache"],
                        "split_manifest_sha256": shas["split_manifest"],
                    }
                },
                "face_manifest_sha256",
            )
            plan.invalidate()
            self.assertEqual(
                plan.status_of(plan.spec("face_cache")).status, STATUS_PRESENT
            )

    def test_unsigned_manifest_is_stale_not_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            path = plan.spec("dataset_manifest").manifest
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            plan.invalidate()
            status = plan.status_of(plan.spec("dataset_manifest"))
            self.assertEqual(status.status, STATUS_STALE)
            self.assertIn("manifest_sha256", status.reason)

    def test_caches_that_need_lidar_are_blocked_without_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp), lidar=False)
            statuses = {s.spec.name: s for s in plan.statuses()}
            for blocked in ("depth_cache", "face_lidar_geometry", "tile_plan", "tile_inputs"):
                self.assertEqual(statuses[blocked].status, STATUS_BLOCKED, blocked)
                self.assertIn("lidar_point_cloud", statuses[blocked].reason)
            self.assertEqual(statuses["mask_manifest"].status, STATUS_MISSING)


class CacheBuildTest(unittest.TestCase):
    def test_dry_run_prints_every_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            lines = plan.build(dry_run=True)
            text = "\n".join(lines)
            self.assertIn("build_face_cache.py", text)
            self.assertIn("build_da2_face_cache.py", text)
            self.assertIn("[BUILD] dataset_manifest (cpu)", text)
            self.assertIn("to build:", text)

    def test_dry_run_never_touches_the_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            before = sorted(p.name for p in root.iterdir())
            plan.build(dry_run=True)
            self.assertEqual(before, sorted(p.name for p in root.iterdir()))

    def test_gpu_step_is_refused_with_the_runner_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            with self.assertRaises(GpuStepRequired) as caught:
                plan.build(dry_run=False, only=["mono_depth"], runner=lambda cmd: 0)
            message = str(caught.exception)
            self.assertIn("GPU step, run through the SDK runner", message)
            self.assertIn("build_da2_face_cache.py", message)

    def test_blocked_cache_raises_naming_the_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root, lidar=False), _profile(root))
            with self.assertRaises(DatasetIncompleteError) as caught:
                plan.build(dry_run=False, only=["depth_cache"], runner=lambda cmd: 0)
            self.assertIn("lidar_point_cloud", str(caught.exception))

    def test_cpu_step_runs_and_rebinds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            calls: list[list[str]] = []

            def runner(command):
                calls.append(list(command))
                _write(
                    plan.spec("dataset_manifest").manifest,
                    {"schema_version": 1, "images": []},
                    "manifest_sha256",
                )
                return 0

            report = plan.build(
                dry_run=False, only=["dataset_manifest"], runner=runner
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("cloudstudio_3dgs.data.manifest", " ".join(calls[0]))
            self.assertTrue(any(line.startswith("build dataset_manifest") for line in report))
            self.assertTrue(any("manifest_sha256=" in line for line in report))
            self.assertEqual(
                plan.status_of(plan.spec("dataset_manifest")).status, STATUS_PRESENT
            )

    def test_already_present_cache_is_skipped_on_a_real_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            _write(
                plan.spec("dataset_manifest").manifest,
                {"schema_version": 1, "images": []},
                "manifest_sha256",
            )
            plan.invalidate()

            def runner(command):
                raise AssertionError("the builder must not run for a present cache")

            report = plan.build(dry_run=False, only=["dataset_manifest"], runner=runner)
            self.assertTrue(report[0].startswith("skip  dataset_manifest"))

    def test_failing_builder_raises_with_the_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            with self.assertRaises(CachePlanError) as caught:
                plan.build(dry_run=False, only=["dataset_manifest"], runner=lambda cmd: 3)
            self.assertIn("exit code 3", str(caught.exception))

    def test_builder_that_writes_nothing_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CachePlan(_bundle(root), _profile(root))
            with self.assertRaises(CachePlanError) as caught:
                plan.build(dry_run=False, only=["dataset_manifest"], runner=lambda cmd: 0)
            self.assertIn("does not verify", str(caught.exception))

    def test_unfilled_profile_placeholders_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _profile(root, trainer_config=None)
            plan = CachePlan(_bundle(root), profile)
            with self.assertRaises(CachePlanError) as caught:
                plan.build(
                    dry_run=False, only=["tile_ownership_0"], runner=lambda cmd: 0
                )
            self.assertIn("<trainer_config>", str(caught.exception))

    def test_to_dict_is_json_serialisable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = plan_caches(_bundle(root), _profile(root)).to_dict()
            json.dumps(payload)
            self.assertEqual(payload["adapter"], "unit_test")
            self.assertTrue(all("command" in row for row in payload["caches"]))


class RefinedSkyLabelTest(unittest.TestCase):
    """The b12 recipe supervises against a photometrically refined sky label."""

    REFINEMENT = {"dark_ratio": 0.75, "edge_ratio": 0.10, "dilate_px": 1}

    def _plan(self, root: Path, **overrides) -> CachePlan:
        return CachePlan(_bundle(root), _profile(root, sky_mask_refinement=self.REFINEMENT, **overrides))

    def test_without_refinement_there_is_no_refined_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = {s.name for s in build_cache_specs(_bundle(root), _profile(root))}
            self.assertNotIn("sky_masks_refined", names)

    def test_refinement_reads_the_raw_label_and_writes_its_own_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            raw, refined = plan.spec("sky_masks"), plan.spec("sky_masks_refined")
            self.assertEqual(set(refined.depends_on), {"sky_masks", "face_cache"})
            self.assertNotEqual(refined.root, raw.root)
            command = list(refined.command)
            self.assertEqual(command[command.index("--source-manifest") + 1], str(raw.manifest))
            self.assertEqual(command[command.index("--output-root") + 1], str(refined.root))
            self.assertEqual(command[command.index("--dark-ratio") + 1], "0.75")
            self.assertEqual(refined.device, CPU)

    def test_switching_sky_masks_off_drops_the_refinement_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            names = {s.name for s in self._plan(Path(tmp), sky_masks=False)}
            self.assertNotIn("sky_masks", names)
            self.assertNotIn("sky_masks_refined", names)

    def test_a_new_raw_label_makes_the_refined_cache_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            face = _write(plan.spec("face_cache").manifest, {"kind": "fisheye_face_cache"}, "face_manifest_sha256")
            raw = _write(plan.spec("sky_masks").manifest, {"source_face_manifest_sha256": face, "v": 1},
                         "sky_mask_manifest_sha256")
            _write(
                plan.spec("sky_masks_refined").manifest,
                {"source_face_manifest_sha256": face,
                 "rule": {"refinement": {"source_sky_mask_manifest_sha256": raw}}},
                "sky_mask_manifest_sha256",
            )
            plan.invalidate()
            self.assertEqual(plan.status_of(plan.spec("sky_masks_refined")).status, STATUS_PRESENT)
            _write(plan.spec("sky_masks").manifest, {"source_face_manifest_sha256": face, "v": 2},
                   "sky_mask_manifest_sha256")
            plan.invalidate()
            status = plan.status_of(plan.spec("sky_masks_refined"))
            self.assertEqual(status.status, STATUS_STALE)
            self.assertIn("source_sky_mask_manifest_sha256", status.reason)


class IngestCliTest(unittest.TestCase):
    def test_plan_subcommand_prints_without_touching_the_builders(self) -> None:
        import contextlib
        import io

        from cloudstudio3dgs_sdk.ingest import cli
        from cloudstudio3dgs_sdk.ingest.adapters import pinhole_folder  # noqa: F401

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir(parents=True)
            (images / "0.png").write_bytes(b"png")
            (root / "poses.json").write_text(
                json.dumps(
                    {
                        "coordinate_frame": "site",
                        "cameras": [
                            {
                                "camera_id": "cam0",
                                "width": 8,
                                "height": 8,
                                "fx": 4.0,
                                "fy": 4.0,
                                "cx": 4.0,
                                "cy": 4.0,
                            }
                        ],
                        "frames": [
                            {
                                "file_path": "images/0.png",
                                "camera_id": "cam0",
                                "transform_matrix": [
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = cli.main(
                    [
                        "plan",
                        str(root),
                        "--dataset-root",
                        str(root / "dataset"),
                        "--run-root",
                        str(root / "run"),
                        "--report",
                        str(root / "plan.json"),
                    ]
                )
            self.assertEqual(code, 0)
            text = buffer.getvalue()
            self.assertIn("[BUILD] dataset_manifest (cpu)", text)
            # No LiDAR in this dataset: the LiDAR-bound caches must be blocked.
            self.assertIn(STATUS_BLOCKED, text)
            report = json.loads((root / "plan.json").read_text(encoding="utf-8"))
            blocked = {
                row["name"] for row in report["caches"] if row["status"] == STATUS_BLOCKED
            }
            self.assertIn("depth_cache", blocked)
            self.assertIn("tile_inputs", blocked)

    def test_detect_subcommand_reports_the_failure(self) -> None:
        import contextlib
        import io

        from cloudstudio3dgs_sdk.ingest import cli

        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                code = cli.main(["detect", tmp])
            self.assertEqual(code, 2)
            self.assertIn("no adapter recognises", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
