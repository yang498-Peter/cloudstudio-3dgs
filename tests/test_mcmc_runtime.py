from __future__ import annotations

import copy
import hashlib
import json
import tempfile
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
    sign_full_mcmc_gate_evidence,
    snapshot_gaussians,
    verify_full_mcmc_gate_evidence,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.checkpoint import (
    compare_checkpoint_payloads,
    save_checkpoint,
)
from cloudstudio_3dgs.training.backend import GsplatBackend
from tools.promote_full_mcmc_gate import promote_full_mcmc_gate_baseline


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

    def test_checked_in_runtime_baseline_is_signed_and_fail_closed_or_full_pass(self) -> None:
        baseline = json.loads(
            (ROOT / "baselines" / "full_mcmc_runtime.baseline.json").read_text(
                encoding="utf-8"
            )
        )
        if baseline.get("evidence_type") == "cloudstudio_full_mcmc_gate":
            lock = json.loads(
                (ROOT / "upstream" / "cloudstudio_trainer.lock.json").read_text(
                    encoding="utf-8"
                )
            )
            report = verify_full_mcmc_gate_evidence(
                baseline, expected_lock_commit=str(lock["commit"])
            )
            self.assertEqual(report["status"], "PASS", report["errors"])
        else:
            expected = baseline.pop("runtime_evidence_sha256")
            actual = hashlib.sha256(canonical_json_bytes(baseline)).hexdigest()
            self.assertEqual(actual, expected)
            self.assertEqual(baseline["gate_status"], "FAIL")
            gates = baseline["execution_gates"]
            execution = baseline.get("execution_evidence")
            if execution is None:
                self.assertEqual(
                    gates["interrupted_resume_equivalence"], "NOT_RUN"
                )
            else:
                self.assertEqual(gates["interrupted_resume_equivalence"], "PASS")
                self.assertEqual(gates["sample_add_occurred"], "PASS")
                self.assertNotEqual(gates["mcmc_noise_nonzero"], "PASS")
                self.assertEqual(gates["real_gpu_training"], "PASS_RUNTIME_ONLY")
                self.assertNotEqual(gates["relocation_occurred"], "PASS")
                synthetic = execution["synthetic_full_mcmc"]
                self.assertGreaterEqual(synthetic["total_added"], 1)
                self.assertEqual(synthetic["total_relocated"], 0)
                self.assertTrue(
                    execution["interrupted_resume"]["kill_based_interruption"]
                )
                self.assertEqual(execution["full_state_resume"]["status"], "PASS")
                self.assertEqual(
                    execution["full_state_resume"]["mismatch_count"], 0
                )

    def _passing_gate_evidence(self) -> dict:
        commit = "a" * 40
        return {
            "schema_version": 1,
            "evidence_type": "cloudstudio_full_mcmc_gate",
            "gate_status": "PASS",
            "environment": {
                "cuda_available": True,
                "gpu": "fixture-gpu",
                "python": "3.12.9",
                "torch": "2.11.0+cu128",
                "torch_cuda": "12.8",
            },
            "lock": {"commit": commit},
            "runtime": {
                "locked_commit": commit,
                "commit": commit,
                "clean": True,
            },
            "execution_gates": {
                "covariance_forward_backward": "PASS",
                "mcmc_noise_nonzero": "PASS",
                "relocation_occurred": "PASS",
                "sample_add_occurred": "PASS",
                "rasterization_forward_backward": "PASS",
                "metric_scale_rasterization": "PASS",
                "interrupted_resume_equivalence": "PASS",
            },
            "render_scale_contract": {
                "status": "PASS",
                "covered_pixels": 512,
                "minimum_covered_pixels": 40,
                "maximum_covered_pixels": 4000,
                "alpha_finite": True,
            },
            "native_kernel_smoke": {
                "status": "PASS",
                "covariance_forward_finite": True,
                "covariance_backward_finite": True,
                "fused_perturb_finite": True,
                "fused_perturb_max_abs_delta": 0.01,
            },
            "training": {
                "steps": 80,
                "run_manifest_sha256": "b" * 64,
                "mcmc_operator_registration": "PASS_REGISTERED",
                "initial_loss": 1.0,
                "final_loss": 0.1,
                "loss_improvement_fraction": 0.9,
                "initial_gaussian_count": 24,
                "gaussian_count": 29,
                "mcmc_noise_step_count": 80,
                "mcmc_noise_nonzero_step_count": 76,
                "mcmc_noise_max_abs_delta_m": 0.002,
                "mcmc_refine_event_count": 4,
                "mcmc_relocated_count": 1,
                "mcmc_added_count": 5,
                "mcmc_final_state_finite": True,
                "gaussian_count_curve": [
                    {"step": -1, "gaussian_count": 24},
                    {"step": 80, "gaussian_count": 29},
                ],
                "resume_equivalence": {
                    "status": "PASS",
                    "mismatch_count": 0,
                    "reference_step": 80,
                    "resumed_step": 80,
                    "reference_gaussian_count": 29,
                    "resumed_gaussian_count": 29,
                },
            },
        }

    def test_signed_full_gate_evidence_passes_and_tampering_fails(self) -> None:
        signed = sign_full_mcmc_gate_evidence(self._passing_gate_evidence())
        report = verify_full_mcmc_gate_evidence(
            signed, expected_lock_commit="a" * 40
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["errors"], [])

        signed["training"]["mcmc_added_count"] = 0
        tampered = verify_full_mcmc_gate_evidence(
            signed, expected_lock_commit="a" * 40
        )
        self.assertEqual(tampered["status"], "FAIL")
        self.assertIn("gate evidence signature mismatch", tampered["errors"])

    def test_full_gate_rejects_configured_but_zero_noise(self) -> None:
        evidence = self._passing_gate_evidence()
        evidence["training"]["mcmc_noise_nonzero_step_count"] = 0
        evidence["training"]["mcmc_noise_max_abs_delta_m"] = 0.0
        signed = sign_full_mcmc_gate_evidence(evidence)
        report = verify_full_mcmc_gate_evidence(
            signed, expected_lock_commit="a" * 40
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "training did not observe nonzero MCMC position noise", report["errors"]
        )

    def test_full_gate_requires_metric_scale_rasterization_smoke(self) -> None:
        evidence = self._passing_gate_evidence()
        evidence.pop("render_scale_contract")
        evidence["execution_gates"]["metric_scale_rasterization"] = "NOT_RUN"
        signed = sign_full_mcmc_gate_evidence(evidence)
        report = verify_full_mcmc_gate_evidence(
            signed, expected_lock_commit="a" * 40
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("metric scale rasterization smoke did not pass", report["errors"])

    def test_only_verified_gate_evidence_can_be_promoted_atomically(self) -> None:
        signed = sign_full_mcmc_gate_evidence(self._passing_gate_evidence())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "full_mcmc_runtime.baseline.json"
            promoted = promote_full_mcmc_gate_baseline(
                signed,
                output,
                expected_lock_commit="a" * 40,
            )
            stored = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(promoted["status"], "PASS")
            self.assertEqual(stored, signed)

            tampered = dict(signed)
            tampered["gate_status"] = "FAIL"
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                promote_full_mcmc_gate_baseline(
                    tampered,
                    root / "tampered.json",
                    expected_lock_commit="a" * 40,
                )
            self.assertFalse((root / "tampered.json").exists())

            with self.assertRaisesRegex(FileExistsError, "already contains PASS"):
                promote_full_mcmc_gate_baseline(
                    signed,
                    output,
                    expected_lock_commit="a" * 40,
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

        noise_only = build_mcmc_step_event(
            step=601,
            before=None,
            after=None,
            refine_start_iter=500,
            refine_stop_iter=25_000,
            refine_every=100,
            noise_injection_stop_iter=-1,
            noise_position_delta_max_m=0.002,
        )
        append_mcmc_telemetry(telemetry, noise_only)
        self.assertEqual(telemetry["noise_nonzero_step_count"], 1)
        self.assertAlmostEqual(telemetry["noise_max_abs_delta_m"], 0.002)

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

    def test_backend_observes_real_non_refine_noise_delta_once(self) -> None:
        class NoiseStrategy:
            min_opacity = 0.005
            refine_start_iter = 10
            refine_stop_iter = 100
            refine_every = 10
            noise_injection_stop_iter = -1

            @staticmethod
            def step_post_backward(*, params, **_kwargs) -> None:
                params["means"].data.add_(0.002)

        backend = object.__new__(GsplatBackend)
        backend.strategy = NoiseStrategy()
        params = {
            "means": torch.nn.Parameter(torch.zeros((4, 3))),
        }
        optimizers = {
            "means": type("Optimizer", (), {"param_groups": [{"lr": 1.0}]})()
        }
        strategy_state = {}
        first = backend.strategy_post_step(
            params, optimizers, strategy_state, step=1, info={}
        )
        second = backend.strategy_post_step(
            params, optimizers, strategy_state, step=2, info={}
        )
        self.assertAlmostEqual(first["noise_position_delta_max_m"], 0.002)
        self.assertIsNone(second["noise_position_delta_max_m"])
        self.assertTrue(strategy_state["_cloudstudio_noise_probe_observed"])

    def test_snapshot_marks_exponentiated_scale_overflow_non_finite(self) -> None:
        params = self._params()
        params["scales"].data[0, 0] = 1000.0
        snapshot = snapshot_gaussians(params, min_opacity=0.005)
        self.assertFalse(snapshot["finite"])
        self.assertIsNone(snapshot["scale_m"])

    def test_checkpoint_equivalence_compares_full_resumable_state(self) -> None:
        params = torch.nn.ParameterDict(
            {"means": torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]]))}
        )
        optimizers = {"means": torch.optim.Adam([params["means"]], lr=0.1)}
        generator = torch.Generator().manual_seed(17)
        state = {
            "last_metrics": {"loss": 0.5},
            "initial_loss": 1.0,
            "best_loss": 0.5,
            "mcmc_telemetry": {"total_added": 2},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pt"
            resumed = root / "resumed.pt"
            changed = root / "changed.pt"
            common = {
                "step": 20,
                "identity": {"dataset": "a"},
                "params": params,
                "optimizers": optimizers,
                "strategy_state": {"counter": torch.tensor([3])},
                "sampler_state": generator.get_state(),
                "training_state": state,
            }
            save_checkpoint(reference, **common)
            save_checkpoint(resumed, **common)
            passing = compare_checkpoint_payloads(reference, resumed)
            params["means"].data[0, 0] += 0.1
            save_checkpoint(changed, **common)
            failing = compare_checkpoint_payloads(reference, changed)

        self.assertEqual(passing["status"], "PASS")
        self.assertEqual(passing["mismatch_count"], 0)
        self.assertEqual(failing["status"], "FAIL")
        self.assertTrue(
            any("params.means" in item for item in failing["mismatches"])
        )

    def test_checkpoint_equivalence_normalizes_only_valid_evaluation_self_hashes(self) -> None:
        import hashlib
        import torch

        from cloudstudio_3dgs.data.manifest import canonical_json_bytes

        def signed_evaluation(psnr: float) -> dict:
            record = {
                "algorithm_version": "golden_validation_v2",
                "completed_steps": 80,
                "summary": {"psnr_db_mean": psnr, "ssim_mean": 0.8},
            }
            record["golden_evaluation_sha256"] = hashlib.sha256(
                canonical_json_bytes(record)
            ).hexdigest()
            return record

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.pt"
            resumed_path = root / "resumed.pt"
            base = {
                "step": 80,
                "params": {"means": torch.zeros((1, 3))},
                "training_state": {
                    "golden_evaluation": {
                        "history": [signed_evaluation(20.0)],
                    }
                },
            }
            resumed = copy.deepcopy(base)
            resumed["training_state"]["golden_evaluation"]["history"] = [
                signed_evaluation(20.0 + 2e-7)
            ]
            torch.save(base, reference_path)
            torch.save(resumed, resumed_path)
            passing = compare_checkpoint_payloads(
                reference_path, resumed_path, atol=5e-6, rtol=1e-6
            )
            self.assertEqual(passing["status"], "PASS")
            self.assertEqual(
                passing["tolerance_normalized_evaluation_hash_count"], 1
            )

            resumed["training_state"]["golden_evaluation"]["history"][0][
                "golden_evaluation_sha256"
            ] = "0" * 64
            torch.save(resumed, resumed_path)
            failing = compare_checkpoint_payloads(
                reference_path, resumed_path, atol=5e-6, rtol=1e-6
            )
            self.assertEqual(failing["status"], "FAIL")
            self.assertTrue(
                any("signed evaluation hash mismatch" in item for item in failing["mismatches"])
            )


if __name__ == "__main__":
    unittest.main()
