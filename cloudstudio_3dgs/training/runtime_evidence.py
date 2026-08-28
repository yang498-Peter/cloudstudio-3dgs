"""Fail-closed runtime evidence and telemetry for full gsplat MCMC."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Iterable, Mapping

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


REQUIRED_MCMC_BUILD_FEATURES = ("3dgs", "3dgut", "reloc")
REQUIRED_MCMC_NATIVE_OPS = (
    "mcmc_perturb_positions",
    "projection_ut_3dgs_fused",
    "quat_scale_to_covar_preci_bwd",
    "quat_scale_to_covar_preci_fwd",
    "rasterize_to_pixels_from_world_3dgs",
    "relocation",
)
FULL_MCMC_GATE_HASH_FIELD = "gate_evidence_sha256"
FULL_MCMC_EXECUTION_GATES = (
    "covariance_forward_backward",
    "mcmc_noise_nonzero",
    "relocation_occurred",
    "sample_add_occurred",
    "rasterization_forward_backward",
    "metric_scale_rasterization",
    "interrupted_resume_equivalence",
)


def sign_full_mcmc_gate_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached canonical evidence payload with a tamper checksum."""
    if FULL_MCMC_GATE_HASH_FIELD in evidence:
        raise ValueError("full-MCMC gate evidence is already signed")
    signed = copy.deepcopy(dict(evidence))
    signed[FULL_MCMC_GATE_HASH_FIELD] = hashlib.sha256(
        canonical_json_bytes(signed)
    ).hexdigest()
    return signed


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def verify_full_mcmc_gate_evidence(
    evidence: Mapping[str, Any], *, expected_lock_commit: str | None = None
) -> dict[str, Any]:
    """Fail closed unless one signed payload proves every Gate 1 exit condition."""
    errors: list[str] = []
    unsigned = copy.deepcopy(dict(evidence))
    expected_signature = unsigned.pop(FULL_MCMC_GATE_HASH_FIELD, None)
    actual_signature = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    signature_valid = expected_signature == actual_signature
    if not signature_valid:
        errors.append("gate evidence signature mismatch")
    if unsigned.get("schema_version") != 1:
        errors.append("unsupported gate evidence schema")
    if unsigned.get("evidence_type") != "cloudstudio_full_mcmc_gate":
        errors.append("unexpected gate evidence type")
    if unsigned.get("gate_status") != "PASS":
        errors.append("gate status is not PASS")

    environment = unsigned.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("cuda_available") is not True
    ):
        errors.append("CUDA environment evidence is missing")
    else:
        for field in ("gpu", "python", "torch", "torch_cuda"):
            if not isinstance(environment.get(field), str) or not environment[field]:
                errors.append(f"environment.{field} is missing")

    lock = unsigned.get("lock")
    runtime = unsigned.get("runtime")
    lock_commit = lock.get("commit") if isinstance(lock, dict) else None
    if not isinstance(lock_commit, str) or len(lock_commit) != 40:
        errors.append("locked gsplat commit is invalid")
    if expected_lock_commit is not None and lock_commit != expected_lock_commit:
        errors.append("gate evidence does not match the expected gsplat lock")
    if not isinstance(runtime, dict) or runtime.get("clean") is not True:
        errors.append("gsplat runtime is not a clean checkout")
    elif (
        runtime.get("locked_commit") != lock_commit
        or runtime.get("commit") != lock_commit
    ):
        errors.append("gsplat runtime commit does not match the lock")

    execution_gates = unsigned.get("execution_gates")
    if not isinstance(execution_gates, dict):
        errors.append("execution gate report is missing")
    else:
        for name in FULL_MCMC_EXECUTION_GATES:
            if execution_gates.get(name) != "PASS":
                errors.append(f"execution gate {name} is not PASS")

    smoke = unsigned.get("native_kernel_smoke")
    if not isinstance(smoke, dict) or smoke.get("status") != "PASS":
        errors.append("native kernel smoke did not pass")
    else:
        for field in (
            "covariance_forward_finite",
            "covariance_backward_finite",
            "fused_perturb_finite",
        ):
            if smoke.get(field) is not True:
                errors.append(f"native kernel smoke {field} is not true")
        perturb_delta = smoke.get("fused_perturb_max_abs_delta")
        if not _finite_number(perturb_delta) or float(perturb_delta) <= 0.0:
            errors.append("native fused perturbation did not move positions")

    scale_smoke = unsigned.get("render_scale_contract")
    if not isinstance(scale_smoke, dict) or scale_smoke.get("status") != "PASS":
        errors.append("metric scale rasterization smoke did not pass")
    else:
        covered = scale_smoke.get("covered_pixels")
        minimum = scale_smoke.get("minimum_covered_pixels")
        maximum = scale_smoke.get("maximum_covered_pixels")
        if (
            scale_smoke.get("alpha_finite") is not True
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (covered, minimum, maximum)
            )
            or not minimum <= covered <= maximum
        ):
            errors.append("metric scale rasterization footprint is invalid")

    training = unsigned.get("training")
    if not isinstance(training, dict):
        errors.append("training acceptance evidence is missing")
    else:
        steps = training.get("steps")
        steps_valid = (
            isinstance(steps, int) and not isinstance(steps, bool) and steps > 0
        )
        if not steps_valid:
            errors.append("training step count is invalid")
        run_hash = training.get("run_manifest_sha256")
        if not isinstance(run_hash, str) or len(run_hash) != 64:
            errors.append("run manifest hash is invalid")
        if training.get("mcmc_operator_registration") != "PASS_REGISTERED":
            errors.append("MCMC operator registration did not pass")
        initial_loss = training.get("initial_loss")
        final_loss = training.get("final_loss")
        improvement = training.get("loss_improvement_fraction")
        if not all(
            _finite_number(value)
            for value in (initial_loss, final_loss, improvement)
        ):
            errors.append("training loss evidence is not finite")
        elif float(final_loss) >= float(initial_loss) or float(improvement) < 0.20:
            errors.append("synthetic full-MCMC training did not converge")

        initial_count = training.get("initial_gaussian_count")
        final_count = training.get("gaussian_count")
        added_count = training.get("mcmc_added_count")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (initial_count, final_count, added_count)
        ):
            errors.append("Gaussian count evidence is invalid")
        elif (
            initial_count <= 0
            or added_count <= 0
            or final_count != initial_count + added_count
        ):
            errors.append("Gaussian add/count evidence is inconsistent")

        noise_steps = training.get("mcmc_noise_step_count")
        nonzero_noise_steps = training.get("mcmc_noise_nonzero_step_count")
        noise_delta = training.get("mcmc_noise_max_abs_delta_m")
        if steps_valid and noise_steps != steps:
            errors.append("MCMC noise was not invoked on every configured step")
        if (
            not isinstance(nonzero_noise_steps, int)
            or isinstance(nonzero_noise_steps, bool)
            or nonzero_noise_steps <= 0
            or not _finite_number(noise_delta)
            or float(noise_delta) <= 0.0
        ):
            errors.append("training did not observe nonzero MCMC position noise")
        refine_count = training.get("mcmc_refine_event_count")
        relocated_count = training.get("mcmc_relocated_count")
        if not isinstance(refine_count, int) or refine_count <= 0:
            errors.append("MCMC refine window did not execute")
        if not isinstance(relocated_count, int) or relocated_count <= 0:
            errors.append("MCMC relocation did not occur")
        if training.get("mcmc_final_state_finite") is not True:
            errors.append("final Gaussian state is not finite")

        curve = training.get("gaussian_count_curve")
        if not isinstance(curve, list) or len(curve) < 2:
            errors.append("Gaussian count curve is incomplete")
        else:
            curve_steps = [
                item.get("step") for item in curve if isinstance(item, dict)
            ]
            curve_counts = [
                item.get("gaussian_count") for item in curve if isinstance(item, dict)
            ]
            curve_valid = (
                len(curve_steps) == len(curve)
                and all(isinstance(value, int) for value in curve_steps + curve_counts)
                and all(
                    left < right
                    for left, right in zip(curve_steps, curve_steps[1:])
                )
                and all(
                    left <= right
                    for left, right in zip(curve_counts, curve_counts[1:])
                )
                and curve_counts[0] == initial_count
                and curve_counts[-1] == final_count
            )
            if not curve_valid:
                errors.append("Gaussian count curve is inconsistent")

        resume = training.get("resume_equivalence")
        if not isinstance(resume, dict) or resume.get("status") != "PASS":
            errors.append("interrupted resume equivalence did not pass")
        else:
            if resume.get("mismatch_count") != 0:
                errors.append("resumed checkpoint has state mismatches")
            if steps_valid and (
                resume.get("reference_step") != steps
                or resume.get("resumed_step") != steps
            ):
                errors.append("resumed checkpoint step does not match the reference")
            if (
                resume.get("reference_gaussian_count") != final_count
                or resume.get("resumed_gaussian_count") != final_count
            ):
                errors.append("resumed Gaussian identity/count does not match")

    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "signature_valid": signature_valid,
        "errors": errors,
    }


def build_mcmc_runtime_report(
    *,
    build_config: Mapping[str, Any],
    registered_ops: Iterable[str],
    cuda_available: bool,
    cuda_device_name: str | None,
    source_runtime: Mapping[str, Any],
    sample_add_available: bool,
) -> dict[str, Any]:
    """Build a deterministic registration report without claiming kernel execution."""
    registered = set(registered_ops)
    feature_status = {
        name: bool(build_config.get(name, False))
        for name in REQUIRED_MCMC_BUILD_FEATURES
    }
    operator_status = {
        name: name in registered for name in REQUIRED_MCMC_NATIVE_OPS
    }
    missing_features = sorted(
        name for name, available in feature_status.items() if not available
    )
    missing_ops = sorted(
        name for name, available in operator_status.items() if not available
    )
    locked_commit = source_runtime.get("locked_commit")
    actual_commit = source_runtime.get("commit")
    provenance_ok = (
        source_runtime.get("clean") is True
        and isinstance(locked_commit, str)
        and actual_commit == locked_commit
    )
    complete = (
        bool(cuda_available)
        and provenance_ok
        and bool(sample_add_available)
        and not missing_features
        and not missing_ops
    )
    return {
        "schema_version": 1,
        "evidence_scope": "operator_registration_only",
        "status": "PASS_REGISTERED" if complete else "FAIL_INCOMPLETE_RUNTIME",
        "cuda_available": bool(cuda_available),
        "cuda_device_name": cuda_device_name,
        "clean_locked_source": provenance_ok,
        "build_features": feature_status,
        "native_ops": operator_status,
        "sample_add_python_api": bool(sample_add_available),
        "missing_build_features": missing_features,
        "missing_native_ops": missing_ops,
        "limitations": [
            "registration does not prove forward or backward kernel execution",
            "registration does not prove relocation, sampling, or noise occurred",
            "registration does not prove interrupted-resume equivalence",
        ],
    }


def require_full_mcmc_runtime(report: Mapping[str, Any]) -> None:
    """Reject an incomplete or provenance-unbound MCMC runtime."""
    if report.get("status") == "PASS_REGISTERED":
        return
    reasons: list[str] = []
    if not report.get("cuda_available"):
        reasons.append("CUDA unavailable")
    if not report.get("clean_locked_source"):
        reasons.append("source is not the clean locked commit")
    if not report.get("sample_add_python_api"):
        reasons.append("sample_add Python API missing")
    reasons.extend(
        f"build feature {name} missing"
        for name in report.get("missing_build_features", [])
    )
    reasons.extend(
        f"native op {name} missing" for name in report.get("missing_native_ops", [])
    )
    raise RuntimeError("full MCMC runtime is incomplete: " + "; ".join(reasons))


def audit_loaded_mcmc_runtime(source_runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect the currently loaded gsplat extension after provenance verification."""
    import torch
    from gsplat.cuda._backend import _C
    from gsplat.strategy.ops import sample_add

    if _C is None:
        build_config: Mapping[str, Any] = {}
    else:
        build_config = _C.build_config()
    registered_ops = {
        qualified.split("::", 1)[1]
        for qualified in torch._C._dispatch_get_all_op_names()
        if qualified.startswith("gsplat::")
    }
    device_name = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return build_mcmc_runtime_report(
        build_config=build_config,
        registered_ops=registered_ops,
        cuda_available=torch.cuda.is_available(),
        cuda_device_name=device_name,
        source_runtime=source_runtime,
        sample_add_available=callable(sample_add),
    )


def execute_mcmc_native_kernel_smoke(device: str = "cuda:0") -> dict[str, Any]:
    """Execute covariance forward/backward and fused position perturbation."""
    import torch
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("native MCMC kernel smoke requires an explicit CUDA device")
    quats = torch.zeros((4, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)
    scales = torch.full(
        (4, 3), 0.1, dtype=torch.float32, device=device, requires_grad=True
    )
    covariances, _ = quat_scale_to_covar_preci(
        quats,
        scales,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )
    assert covariances is not None
    covariance_loss = covariances.square().sum()
    covariance_loss.backward()
    covariance_forward_finite = bool(torch.isfinite(covariances).all().item())
    covariance_backward_finite = bool(
        quats.grad is not None
        and scales.grad is not None
        and torch.isfinite(quats.grad).all().item()
        and torch.isfinite(scales.grad).all().item()
    )

    positions = torch.zeros((4, 3), dtype=torch.float32, device=device)
    log_scales = torch.full(
        (4, 3), float(torch.tensor(0.1).log()), dtype=torch.float32, device=device
    )
    opacity_logits = torch.full(
        (4,), float(torch.tensor(0.001).logit()), dtype=torch.float32, device=device
    )
    noise = torch.ones_like(positions)
    torch.ops.gsplat.mcmc_perturb_positions(
        positions,
        quats.detach(),
        log_scales,
        opacity_logits,
        noise,
        1.0,
    )
    torch.cuda.synchronize(device)
    perturb_max_abs_delta = float(positions.abs().max().cpu())
    perturb_finite = bool(torch.isfinite(positions).all().item())
    passed = (
        covariance_forward_finite
        and covariance_backward_finite
        and perturb_finite
        and perturb_max_abs_delta > 0.0
    )
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "device": str(device),
        "covariance_forward_finite": covariance_forward_finite,
        "covariance_backward_finite": covariance_backward_finite,
        "fused_perturb_finite": perturb_finite,
        "fused_perturb_max_abs_delta": perturb_max_abs_delta,
    }


def execute_render_scale_contract_smoke(backend: Any) -> dict[str, Any]:
    """Render a bounded metric-scale footprint with the real backend."""
    import numpy as np
    import torch

    from cloudstudio_3dgs.training.dataset import TrainingSample

    expected_scale_m = 0.1
    xyz = np.asarray(
        [
            [-0.6, -0.6, 2.0],
            [0.6, -0.6, 2.0],
            [-0.6, 0.6, 2.0],
            [0.6, 0.6, 2.0],
        ],
        dtype=np.float32,
    )
    rgb = np.full((4, 3), 255, dtype=np.uint8)
    params, _, _ = backend.initialize(
        xyz,
        rgb,
        init_scale_m=expected_scale_m,
        learning_rates={
            name: 1e-4
            for name in ("means", "scales", "quats", "opacities", "colors")
        },
    )
    params["opacities"].data.fill_(
        torch.tensor(0.999, device=backend.device).logit()
    )
    sample = TrainingSample(
        image_id="metric_scale_rasterization_smoke",
        rig_frame_id="metric_scale_rasterization_smoke",
        camera_id="left",
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        rgb_mask=np.ones((128, 128), dtype=bool),
        depth_range_m=None,
        depth_confidence=None,
        depth_mask=None,
        depth_cache_path=None,
        c2w=np.eye(4, dtype=np.float32),
        K=np.asarray(
            [
                [100.0, 0.0, 63.5],
                [0.0, 100.0, 63.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        radial_coeffs=np.zeros(4, dtype=np.float32),
        width=128,
        height=128,
    )
    with torch.no_grad():
        _, _, alpha, _ = backend.render(params, sample, with_range=False)
    alpha_finite = bool(torch.isfinite(alpha).all().item())
    covered = int((alpha.squeeze() > 0.5).sum().item())
    minimum = 40
    maximum = 4000
    passed = alpha_finite and minimum <= covered <= maximum
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "expected_linear_scale_m": expected_scale_m,
        "covered_pixels": covered,
        "minimum_covered_pixels": minimum,
        "maximum_covered_pixels": maximum,
        "alpha_finite": alpha_finite,
    }


# torch.quantile refuses inputs above 2**24 elements. A 15.9M-Gaussian model
# carries 47.7M scale values, so the telemetry that merely DESCRIBES a run
# aborted it before step one - which is what killed the H3 arm.
_QUANTILE_LIMIT = 2 ** 24


def _quantiles(tensor: Any) -> dict[str, float]:
    import torch

    flattened = tensor.detach().float().flatten()
    # min and max come from the full tensor: exact, cheap, and they are what a
    # reader checks for degenerate geometry.
    minimum = float(flattened.min().cpu())
    maximum = float(flattened.max().cpu())
    if flattened.numel() > _QUANTILE_LIMIT:
        # Deterministic stride rather than random sampling: identical evidence
        # must come out of an identical model. Parameter tensors carry no
        # meaningful ordering, so a strided subsample is unbiased here.
        stride = flattened.numel() // _QUANTILE_LIMIT + 1
        flattened = flattened[::stride]
    values = torch.quantile(
        flattened,
        torch.tensor([0.5, 0.95], device=flattened.device),
    )
    return {
        "min": minimum,
        "p50": float(values[0].cpu()),
        "p95": float(values[1].cpu()),
        "max": maximum,
    }


def snapshot_gaussians(
    params: Mapping[str, Any], *, min_opacity: float
) -> dict[str, Any]:
    """Capture bounded Gaussian distribution evidence at a refine boundary."""
    import torch

    required = {"means", "scales", "opacities"}
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(f"Gaussian snapshot is missing parameters: {', '.join(missing)}")
    count = int(len(params["means"]))
    if count <= 0:
        raise ValueError("Gaussian snapshot requires at least one Gaussian")
    opacities = torch.sigmoid(params["opacities"].detach().flatten())
    scales = torch.exp(params["scales"].detach())
    finite = (
        all(
            bool(torch.isfinite(params[name].detach()).all().item())
            for name in required
        )
        and bool(torch.isfinite(opacities).all().item())
        and bool(torch.isfinite(scales).all().item())
    )
    return {
        "gaussian_count": count,
        "dead_gaussian_count": int((opacities <= float(min_opacity)).sum().item()),
        "finite": finite,
        "opacity": _quantiles(opacities) if finite else None,
        "scale_m": _quantiles(scales) if finite else None,
    }


def _refine_triggered(
    *,
    step: int,
    refine_start_iter: int,
    refine_stop_iter: int,
    refine_every: int,
    refine_start_inclusive: bool = False,
) -> bool:
    return (
        step < refine_stop_iter
        and (
            step >= refine_start_iter
            if refine_start_inclusive
            else step > refine_start_iter
        )
        and step % refine_every == 0
    )


def build_mcmc_step_event(
    *,
    step: int,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    refine_start_iter: int,
    refine_stop_iter: int,
    refine_every: int,
    noise_injection_stop_iter: int,
    noise_position_delta_max_m: float | None = None,
    strategy_prunes: bool = False,
    refine_start_inclusive: bool = False,
) -> dict[str, Any]:
    """Describe one completed strategy call without overstating visual quality."""
    refine = _refine_triggered(
        step=step,
        refine_start_iter=refine_start_iter,
        refine_stop_iter=refine_stop_iter,
        refine_every=refine_every,
        refine_start_inclusive=refine_start_inclusive,
    )
    if refine and (before is None or after is None):
        raise ValueError("refine telemetry requires before and after snapshots")
    if not refine and (before is not None or after is not None):
        raise ValueError("non-refine telemetry must not carry distribution snapshots")
    noise = noise_injection_stop_iter < 0 or step < noise_injection_stop_iter
    if noise_position_delta_max_m is not None:
        noise_position_delta_max_m = float(noise_position_delta_max_m)
        if (
            not noise
            or not math.isfinite(noise_position_delta_max_m)
            or noise_position_delta_max_m < 0.0
        ):
            raise ValueError("noise position delta requires a finite active-noise step")
    relocated = 0
    added = 0
    pruned = 0
    if refine:
        assert before is not None and after is not None
        relocated = int(before["dead_gaussian_count"])
        added = int(after["gaussian_count"]) - int(before["gaussian_count"])
        if added < 0:
            # MCMC relocates dead Gaussians and appends new ones but never
            # removes any, so a falling count means the strategy misbehaved.
            # Classic 3DGS densification prunes low-opacity and oversized splats
            # by design, and asserting MCMC's invariant against it would reject
            # correct behaviour - so the caller states which semantics apply
            # rather than the check being relaxed for everyone.
            if not strategy_prunes:
                raise RuntimeError(
                    "MCMC strategy unexpectedly reduced Gaussian count"
                )
            pruned = -added
            added = 0
    return {
        "step": int(step),
        "refine_triggered": refine,
        "noise_injection_invoked": noise,
        "relocated_count": relocated,
        "new_gaussian_count": added,
        "pruned_gaussian_count": pruned,
        "noise_position_delta_max_m": noise_position_delta_max_m,
        "before": before,
        "after": after,
    }


def initialize_mcmc_telemetry(initial: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "initial_snapshot": dict(initial),
        "last_snapshot": dict(initial),
        "refine_event_count": 0,
        "noise_injection_step_count": 0,
        "noise_probe_step_count": 0,
        "noise_nonzero_step_count": 0,
        "noise_max_abs_delta_m": 0.0,
        "total_relocated": 0,
        "total_added": 0,
        "events": [],
    }


def append_mcmc_telemetry(
    telemetry: dict[str, Any], event: Mapping[str, Any]
) -> None:
    """Append only refine-boundary detail while keeping noise evidence compact."""
    if event.get("noise_injection_invoked"):
        telemetry["noise_injection_step_count"] += 1
    noise_delta = event.get("noise_position_delta_max_m")
    if noise_delta is not None:
        telemetry["noise_probe_step_count"] += 1
        telemetry["noise_max_abs_delta_m"] = max(
            float(telemetry["noise_max_abs_delta_m"]), float(noise_delta)
        )
        if float(noise_delta) > 0.0:
            telemetry["noise_nonzero_step_count"] += 1
    if not event.get("refine_triggered"):
        return
    telemetry["refine_event_count"] += 1
    telemetry["total_relocated"] += int(event["relocated_count"])
    telemetry["total_added"] += int(event["new_gaussian_count"])
    telemetry["last_snapshot"] = dict(event["after"])
    telemetry["events"].append(dict(event))


def require_finite_training_tensors(
    *,
    params: Mapping[str, Any],
    loss: Any | None,
    stage: str,
    check_gradients: bool,
    check_parameters: bool = True,
) -> None:
    """Fail before an invalid state can replace the latest good checkpoint."""
    import torch

    invalid: list[str] = []
    if loss is not None and not bool(torch.isfinite(loss.detach()).all().item()):
        invalid.append("loss")
    for name, parameter in params.items():
        if check_parameters and not bool(
            torch.isfinite(parameter.detach()).all().item()
        ):
            invalid.append(f"params.{name}")
        gradient = parameter.grad
        if (
            check_gradients
            and gradient is not None
            and not bool(torch.isfinite(gradient.detach()).all().item())
        ):
            invalid.append(f"gradients.{name}")
    if invalid:
        raise FloatingPointError(
            f"non-finite training tensor at {stage}: {', '.join(sorted(invalid))}"
        )
