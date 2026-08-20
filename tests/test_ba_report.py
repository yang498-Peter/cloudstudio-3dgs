from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cloudstudio_3dgs.ba.report import (
    build_ba_report,
    stage_options,
    verify_ba_report,
    write_ba_report,
)


def snapshot(*, after: bool, scale: float = 1.0, focal_scale: float = 1.0) -> dict:
    frames = {}
    baseline = np.eye(4)
    baseline[0, 3] = 0.1
    for index in range(6):
        frames[f"rig_{index:03d}"] = {
            "timestamp_ns": 1_000_000_000 + index * 100_000_000,
            "center_m": [index * scale, 0.0, 0.0],
            "right_to_left": baseline.tolist(),
        }
    cameras = {
        side: {
            "fl_x": 800.0 * focal_scale,
            "fl_y": 805.0 * focal_scale,
            "cx": 400.0,
            "cy": 401.0,
            "k1": 0.01,
            "k2": -0.001,
            "k3": 0.0,
            "k4": 0.0,
        }
        for side in ("left", "right")
    }
    return {
        "model_sha256": ("b" if after else "a") * 64,
        "solver_success": after,
        "reprojection_errors_px": [0.8, 0.9, 1.0, 1.1]
        if after
        else [1.8, 1.9, 2.0, 2.1],
        "rig_frames": frames,
        "cameras": cameras,
    }


class BaReportTests(unittest.TestCase):
    def test_stage_two_accepts_bounded_focal_refinement_and_fixed_rig(self) -> None:
        before = snapshot(after=False)
        after = snapshot(after=True, focal_scale=1.01)
        report = build_ba_report(before, after, stage="stage_2")

        self.assertTrue(report["candidate_accepted"])
        self.assertEqual(report["published_model"], "after")
        self.assertEqual(verify_ba_report(report), report["ba_report_sha256"])
        self.assertGreater(
            report["gates"]["reprojection_p50_improvement"]["improvement_fraction"],
            0.3,
        )
        self.assertFalse(report["stage_options"]["refine_sensor_from_rig"])

    def test_scale_drift_or_unapproved_intrinsics_reject_candidate(self) -> None:
        before = snapshot(after=False)
        scaled = snapshot(after=True, scale=1.02)
        scale_report = build_ba_report(before, scaled, stage="stage_1")
        self.assertFalse(scale_report["candidate_accepted"])
        self.assertEqual(scale_report["gates"]["scene_scale_fixed"]["status"], "FAIL")

        focal = snapshot(after=True, focal_scale=1.01)
        focal_report = build_ba_report(before, focal, stage="stage_1")
        self.assertEqual(
            focal_report["gates"]["camera_parameter_bounds"]["status"], "FAIL"
        )
        self.assertEqual(focal_report["published_model"], "before")

    def test_stage_three_allows_k1_k2_but_rejects_k3_k4(self) -> None:
        before = snapshot(after=False)
        allowed = snapshot(after=True)
        for camera in allowed["cameras"].values():
            camera["k1"] += 0.005
            camera["k2"] -= 0.003
        self.assertTrue(
            build_ba_report(before, allowed, stage="stage_3")["candidate_accepted"]
        )

        forbidden = snapshot(after=True)
        forbidden["cameras"]["left"]["k3"] = 0.001
        report = build_ba_report(before, forbidden, stage="stage_3")
        self.assertFalse(report["candidate_accepted"])
        self.assertEqual(
            report["gates"]["camera_parameter_bounds"]["status"], "FAIL"
        )

    def test_reports_are_byte_deterministic(self) -> None:
        report = build_ba_report(
            snapshot(after=False), snapshot(after=True), stage="stage_1"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_ba_report(root / "a", report)
            write_ba_report(root / "b", report)
            for name in ("ba_report.json", "ba_report.html"):
                self.assertEqual((root / "a" / name).read_bytes(), (root / "b" / name).read_bytes())

    def test_stage_configuration_rejects_unknown_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage_1"):
            stage_options("stage_4")


if __name__ == "__main__":
    unittest.main()
