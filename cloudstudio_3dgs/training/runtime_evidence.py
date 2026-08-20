"""Fail-closed runtime evidence and telemetry for full gsplat MCMC."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


REQUIRED_MCMC_BUILD_FEATURES = ("3dgs", "3dgut", "reloc")
REQUIRED_MCMC_NATIVE_OPS = (
    "mcmc_perturb_positions",
    "projection_ut_3dgs_fused",
    "quat_scale_to_covar_preci_bwd",
    "quat_scale_to_covar_preci_fwd",
    "rasterize_to_pixels_from_world_3dgs",
    "relocation",
)


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


def _quantiles(tensor: Any) -> dict[str, float]:
    import torch

    flattened = tensor.detach().float().flatten()
    values = torch.quantile(
        flattened,
        torch.tensor([0.0, 0.5, 0.95, 1.0], device=flattened.device),
    )
    return {
        "min": float(values[0].cpu()),
        "p50": float(values[1].cpu()),
        "p95": float(values[2].cpu()),
        "max": float(values[3].cpu()),
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
    *, step: int, refine_start_iter: int, refine_stop_iter: int, refine_every: int
) -> bool:
    return (
        step < refine_stop_iter
        and step > refine_start_iter
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
) -> dict[str, Any]:
    """Describe one completed strategy call without overstating visual quality."""
    refine = _refine_triggered(
        step=step,
        refine_start_iter=refine_start_iter,
        refine_stop_iter=refine_stop_iter,
        refine_every=refine_every,
    )
    if refine and (before is None or after is None):
        raise ValueError("refine telemetry requires before and after snapshots")
    if not refine and (before is not None or after is not None):
        raise ValueError("non-refine telemetry must not carry distribution snapshots")
    noise = noise_injection_stop_iter < 0 or step < noise_injection_stop_iter
    relocated = 0
    added = 0
    if refine:
        assert before is not None and after is not None
        relocated = int(before["dead_gaussian_count"])
        added = int(after["gaussian_count"]) - int(before["gaussian_count"])
        if added < 0:
            raise RuntimeError("MCMC strategy unexpectedly reduced Gaussian count")
    return {
        "step": int(step),
        "refine_triggered": refine,
        "noise_injection_invoked": noise,
        "relocated_count": relocated,
        "new_gaussian_count": added,
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
