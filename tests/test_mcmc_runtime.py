from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional training dependency
    torch = None

from cloudstudio_3dgs.training.runtime_evidence import (
    REQUIRED_MCMC_BUILD_FEATURES,
    REQUIRED_MCMC_NATIVE_OPS,
    append_mcmc_telemetry,
    build_mcmc_runtime_report,
    build_mcmc_step_event,
    initialize_mcmc_telemetry,
    require_finite_training_tensors,
    require_full_mcmc_runtime,
    snapshot_gaussians,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]


class MCMCRuntimeReportTests(unittest.TestCase):
    def _report(self, *, missing_op: str | None = None) -> dict:
        registered = set(REQUIRED_MCMC_NATIVE_OPS)
        if missing_op is not None:
            registered.remove(missing_op)
        return build_mcmc_runtime_report(
            build_config={name: True for name in REQUIRED_MCMC_BUILD_FEATURES},
            registered_ops=registered,
            cuda_available=True,
            cuda_device_name="fixture-gpu",
            source_runtime={
                "locked_commit": "a" * 40,
                "commit": "a" * 40,
                "clean": True,
            },
            sample_add_available=True,
        )

    def test_complete_operator_inventory_passes_registration_gate(self) -> None:
        report = self._report()
        self.assertEqual(report["status"], "PASS_REGISTERED")
        self.assertEqual(report["missing_native_ops"], [])
        self.assertEqual(report["missing_build_features"], [])
        require_full_mcmc_runtime(report)

    def test_missing_covariance_operator_fails_closed(self) -> None:
        report = self._report(missing_op="quat_scale_to_covar_preci_fwd")
        self.assertEqual(report["status"], "FAIL_INCOMPLETE_RUNTIME")
        self.assertIn("quat_scale_to_covar_preci_fwd", report["missing_native_ops"])
        with self.assertRaisesRegex(RuntimeError, "quat_scale_to_covar_preci_fwd"):
            require_full_mcmc_runtime(report)

    def test_checked_in_runtime_baseline_is_signed_and_gate_passed(self) -> None:
        # Machine B (RTX 5070 Ti, full-kernel clean build of the locked commit)
        # produced the real GPU execution evidence: noise injected on every
        # step, five refine events adding Gaussians, a kill-based interrupted
        # resume within 5e-3 relative loss tolerance, and a completed 3000-step
        # real-data training run. Relocation was invoked but relocated zero
        # Gaussians because none went dead, and the real-data run is runtime
        # evidence only - its visual quality is explicitly not accepted.
        baseline = json.loads(
            (ROOT / "baselines" / "full_mcmc_runtime.baseline.json").read_text(
                encoding="utf-8"
            )
        )
        expected = baseline.pop("runtime_evidence_sha256")
        actual = hashlib.sha256(canonical_json_bytes(baseline)).hexdigest()
        self.assertEqual(actual, expected)
        self.assertEqual(baseline["gate_status"], "PASS")
        gates = baseline["execution_gates"]
        self.assertEqual(gates["interrupted_resume_equivalence"], "PASS")
        self.assertEqual(gates["sample_add_occurred"], "PASS")
        self.assertEqual(gates["mcmc_noise_nonzero"], "PASS")
        self.assertEqual(gates["real_gpu_training"], "PASS_RUNTIME_ONLY")
        evidence = baseline["execution_evidence"]
        self.assertGreaterEqual(evidence["synthetic_full_mcmc"]["total_added"], 1)
        self.assertTrue(evidence["interrupted_resume"]["kill_based_interruption"])
        self.assertLessEqual(
            evidence["interrupted_resume"]["loss_relative_difference"], 5e-3
        )


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class MCMCTelemetryTests(unittest.TestCase):
    def _params(self) -> dict:
        return {
            "means": torch.nn.Parameter(torch.zeros((4, 3))),
            "scales": torch.nn.Parameter(
                torch.tensor(
                    [
                        [-2.0, -2.0, -2.0],
                        [-1.0, -1.0, -1.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                    ]
                )
            ),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.tensor([0.001, 0.2, 0.5, 0.9]))
            ),
        }

    def test_refine_event_tracks_relocation_add_and_distributions(self) -> None:
        before = snapshot_gaussians(self._params(), min_opacity=0.005)
        after_params = self._params()
        for name, parameter in list(after_params.items()):
            after_params[name] = torch.nn.Parameter(
                torch.cat([parameter.detach(), parameter.detach()[:1]], dim=0)
            )
        after = snapshot_gaussians(after_params, min_opacity=0.005)
        event = build_mcmc_step_event(
            step=600,
            before=before,
            after=after,
            refine_start_iter=500,
            refine_stop_iter=25_000,
            refine_every=100,
            noise_injection_stop_iter=-1,
        )
        self.assertTrue(event["refine_triggered"])
        self.assertTrue(event["noise_injection_invoked"])
        self.assertEqual(event["relocated_count"], 1)
        self.assertEqual(event["new_gaussian_count"], 1)

        telemetry = initialize_mcmc_telemetry(before)
        append_mcmc_telemetry(telemetry, event)
        self.assertEqual(telemetry["refine_event_count"], 1)
        self.assertEqual(telemetry["total_relocated"], 1)
        self.assertEqual(telemetry["total_added"], 1)
        self.assertEqual(telemetry["noise_injection_step_count"], 1)
        self.assertEqual(telemetry["last_snapshot"]["gaussian_count"], 5)

    def test_non_finite_parameter_or_gradient_fails_closed(self) -> None:
        params = self._params()
        params["means"].grad = torch.zeros_like(params["means"])
        params["means"].grad[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "gradients.means"):
            require_finite_training_tensors(
                params=params,
                loss=torch.tensor(1.0),
                stage="after_backward",
                check_gradients=True,
            )

    def test_snapshot_marks_exponentiated_scale_overflow_non_finite(self) -> None:
        params = self._params()
        params["scales"].data[0, 0] = 1000.0
        snapshot = snapshot_gaussians(params, min_opacity=0.005)
        self.assertFalse(snapshot["finite"])
        self.assertIsNone(snapshot["scale_m"])


if __name__ == "__main__":
    unittest.main()
