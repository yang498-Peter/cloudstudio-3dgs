"""Preflight failure paths.

Every probe is injected, so these run on any host: no GPU, no weights, no
network. What is under test is that each way a host can be wrong produces a
required FAIL with a remedy, and that a required FAIL stops the run.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cloudstudio3dgs_sdk.plan import GIB, TileSummary, build_plan
from cloudstudio3dgs_sdk.profile import PROFILE_B5FILL2
from cloudstudio3dgs_sdk.requirements import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    GpuInfo,
    PreflightFailed,
    Probes,
    preflight,
)
from tests.test_sdk_plan import make_repo, two_tile_dataset


def probes(**overrides) -> Probes:
    fields = {
        "python_version": "3.12",
        "torch_version": "2.11.0+cu128",
        "gpu": GpuInfo(True, "stub card", 24.0),
        "free_disk_bytes": lambda path: 10 ** 15,
        "gsplat_extension_sha256": "ab" * 32,
        "external_asset_locator": lambda asset: (True, "stubbed"),
    }
    fields.update(overrides)
    return Probes(**fields)


class PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = make_repo(self.root / "repo")

    def plan(self, dataset=None, *, repo=None, vram_gib=None):
        return build_plan(
            PROFILE_B5FILL2,
            dataset or two_tile_dataset(),
            dataset_root=self.root / "dataset",
            work_root=self.root / "work",
            repo_root=repo or self.repo,
            python=Path("python.exe"),
            prior_tile_checkpoints={0: "a.pt", 1: "b.pt"},
            vram_gib=vram_gib,
        )

    def report(self, **overrides):
        plan_kwargs = {key: overrides.pop(key) for key in ("dataset", "repo", "vram_gib") if key in overrides}
        require_gpu = overrides.pop("require_gpu", True)
        return preflight(
            self.plan(**plan_kwargs),
            PROFILE_B5FILL2,
            repo_root=plan_kwargs.get("repo") or self.repo,
            probes=probes(**overrides),
            require_gpu=require_gpu,
        )

    # -- the happy path exists so the failures mean something ------------

    def test_a_correct_host_passes(self) -> None:
        report = self.report()
        self.assertTrue(report.ok, report.render())
        report.raise_for_status()
        self.assertEqual(report.profile_sha256, PROFILE_B5FILL2.profile_sha256)

    # -- python / torch ---------------------------------------------------

    def test_wrong_python_fails(self) -> None:
        report = self.report(python_version="3.11")
        self.assertEqual(report.get("python_version").status, FAIL)
        self.assertFalse(report.ok)
        self.assertIn("3.12", report.get("python_version").remedy)

    def test_missing_torch_fails_only_when_a_gpu_is_required(self) -> None:
        self.assertEqual(self.report(torch_version=None).get("torch_version").status, FAIL)
        relaxed = self.report(torch_version=None, gpu=GpuInfo(False, detail="cpu host"), require_gpu=False)
        self.assertEqual(relaxed.get("torch_version").status, SKIP)
        self.assertTrue(relaxed.ok)

    def test_a_different_torch_warns_but_does_not_block(self) -> None:
        report = self.report(torch_version="2.9.0+cu121")
        check = report.get("torch_version")
        self.assertEqual(check.status, WARN)
        self.assertFalse(check.required)
        self.assertTrue(report.ok)

    # -- gsplat -----------------------------------------------------------

    def test_missing_gsplat_lock_fails(self) -> None:
        bare = make_repo(self.root / "bare")
        (bare / "upstream" / "gsplat.lock.json").unlink()
        report = preflight(self.plan(repo=bare), PROFILE_B5FILL2, repo_root=bare, probes=probes())
        self.assertEqual(report.get("gsplat_lock").status, FAIL)

    def test_a_lock_at_another_version_fails(self) -> None:
        other = make_repo(self.root / "other")
        (other / "upstream" / "gsplat.lock.json").write_text(
            json.dumps({"version": "1.4.0", "commit": "deadbeef"}), encoding="utf-8"
        )
        report = preflight(self.plan(repo=other), PROFILE_B5FILL2, repo_root=other, probes=probes())
        self.assertEqual(report.get("gsplat_lock").status, FAIL)
        self.assertIn("1.5.3", report.get("gsplat_lock").detail)

    def test_an_uncompiled_extension_fails_and_names_the_abbrev_trap(self) -> None:
        report = self.report(gsplat_extension_sha256=None, gsplat_import_error="no module")
        check = report.get("gsplat_extension")
        self.assertEqual(check.status, FAIL)
        self.assertIn("abbrev", check.remedy)

    # -- GPU and VRAM -----------------------------------------------------

    def test_no_cuda_device_fails(self) -> None:
        report = self.report(gpu=GpuInfo(False, detail="torch.cuda.is_available() is False"))
        self.assertEqual(report.get("gpu").status, FAIL)
        self.assertIn("is False", report.get("gpu").detail)

    def test_a_prepare_only_host_may_skip_the_gpu(self) -> None:
        report = self.report(gpu=GpuInfo(False, detail="cpu host"), require_gpu=False)
        self.assertEqual(report.get("gpu").status, SKIP)
        self.assertTrue(report.ok)

    def test_a_card_below_the_minimum_fails(self) -> None:
        report = self.report(gpu=GpuInfo(True, "small card", 8.0))
        self.assertEqual(report.get("gpu").status, FAIL)
        self.assertIn("16", report.get("gpu").detail)

    def test_a_cap_the_card_cannot_hold_fails(self) -> None:
        """The campaign lost a run to exactly this; it must fail at startup."""
        big = two_tile_dataset(
            tiles=(
                TileSummary(0, "Tile_0", 100, 1_000_000),
                TileSummary(1, "Tile_1", 200, 40_000_000),
            )
        )
        report = self.report(dataset=big, gpu=GpuInfo(True, "16 GiB card", 16.0))
        check = report.get("vram_headroom")
        self.assertEqual(check.status, FAIL)
        self.assertIn("11.00M this card can hold", check.detail)
        self.assertIn("re-plan", check.remedy)

    def test_planning_with_vram_clamps_the_cap_and_passes(self) -> None:
        big = two_tile_dataset(
            tiles=(
                TileSummary(0, "Tile_0", 100, 1_000_000),
                TileSummary(1, "Tile_1", 200, 40_000_000),
            )
        )
        report = self.report(dataset=big, vram_gib=16.0, gpu=GpuInfo(True, "16 GiB card", 16.0))
        self.assertEqual(report.get("vram_headroom").status, PASS)
        self.assertTrue(any(c.name.startswith("plan_warning") and "clamped" in c.detail for c in report.checks))

    # -- disk -------------------------------------------------------------

    def test_insufficient_disk_fails_with_both_numbers(self) -> None:
        report = self.report(free_disk_bytes=lambda path: 3 * GIB)
        check = report.get("disk_headroom")
        self.assertEqual(check.status, FAIL)
        self.assertIn("3.0 GB free", check.detail)
        self.assertIn("still to run need", check.detail)

    def test_disk_check_applies_the_safety_factor(self) -> None:
        plan = self.plan()
        needed = plan.total().disk_bytes
        self.assertEqual(plan.pending().disk_bytes, needed, "nothing exists yet: everything is pending")
        just_under = int(needed * 1.24)
        self.assertEqual(self.report(free_disk_bytes=lambda p: just_under).get("disk_headroom").status, FAIL)
        just_over = int(needed * 1.26)
        self.assertEqual(self.report(free_disk_bytes=lambda p: just_over).get("disk_headroom").status, PASS)

    def test_a_resumed_run_is_charged_only_for_the_steps_still_to_run(self) -> None:
        """The first SDK delivery of house0305 trained for seven hours, then the deliver stage's
        preflight asked for the whole plan's disk again and refused. Outputs that exist are
        skipped by the driver, so they cost nothing more."""
        plan = self.plan()
        whole = plan.total().disk_bytes
        heavy = max((s for s in plan.steps if s.outputs), key=lambda s: s.estimate.disk_bytes)
        self.assertGreater(heavy.estimate.disk_bytes, 0)
        for output in heavy.outputs:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_bytes(b"done")
        pending = plan.pending().disk_bytes
        self.assertEqual(pending, whole - heavy.estimate.disk_bytes)
        check = self.report(free_disk_bytes=lambda p: int(pending * 1.26)).get("disk_headroom")
        self.assertEqual(check.status, PASS, check.detail)
        self.assertIn("whole plan", check.detail)
        self.assertEqual(self.report(free_disk_bytes=lambda p: int(pending * 1.24)).get("disk_headroom").status, FAIL)

    # -- external assets ---------------------------------------------------

    def test_missing_sky_weights_fail_and_state_the_licence(self) -> None:
        report = self.report(external_asset_locator=lambda asset: (False, "not on this host"))
        check = report.get("asset_segformer_sky")
        self.assertEqual(check.status, FAIL)
        self.assertIn("nvidia/segformer-b4-finetuned-ade-512-512", check.detail)
        self.assertIn("NVIDIA Source Code License-NC", check.detail)
        self.assertIn("nor any derived model file are shipped in a delivery", check.detail)
        self.assertIn("supervision masks", check.detail)
        self.assertIn("2641fd1e", check.remedy)

    def test_present_sky_weights_still_state_the_licence(self) -> None:
        check = self.report().get("asset_segformer_sky")
        self.assertEqual(check.status, PASS)
        self.assertIn("non-commercial", check.detail.lower())

    # -- this checkout ------------------------------------------------------

    def test_a_checkout_without_the_fill_merge_fails(self) -> None:
        crippled = make_repo(self.root / "crippled", fill_support=False)
        report = preflight(self.plan(repo=crippled), PROFILE_B5FILL2, repo_root=crippled, probes=probes())
        check = report.get("checkout_supports_profile")
        self.assertEqual(check.status, FAIL)
        self.assertIn("--fill-checkpoint", check.detail)
        self.assertFalse(report.ok)

    def test_a_missing_tool_fails(self) -> None:
        crippled = make_repo(self.root / "notools")
        (crippled / "tools" / "build_sky_dome.py").unlink()
        report = preflight(self.plan(repo=crippled), PROFILE_B5FILL2, repo_root=crippled, probes=probes())
        check = report.get("tools_present")
        self.assertEqual(check.status, FAIL)
        self.assertIn("build_sky_dome.py", check.detail)

    # -- reporting ----------------------------------------------------------

    def test_unmeasured_knobs_warn_without_blocking(self) -> None:
        check = self.report().get("profile_confidence")
        self.assertEqual(check.status, WARN)
        self.assertFalse(check.required)
        self.assertIn("seed_generation_overrides", check.detail)

    def test_raise_for_status_names_every_failure(self) -> None:
        report = self.report(python_version="3.10", gpu=GpuInfo(False, detail="no card"))
        with self.assertRaises(PreflightFailed) as caught:
            report.raise_for_status()
        message = str(caught.exception)
        self.assertIn("python_version", message)
        self.assertIn("gpu", message)

    def test_render_and_json_cover_every_check(self) -> None:
        report = self.report(python_version="3.10")
        text = report.render()
        self.assertIn("preflight FAIL", text)
        self.assertIn("python_version", text)
        payload = report.as_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(len(payload["checks"]), len(report.checks))
        json.dumps(payload)

    def test_unknown_check_lookup_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.report().get("no_such_check")


if __name__ == "__main__":
    unittest.main()
