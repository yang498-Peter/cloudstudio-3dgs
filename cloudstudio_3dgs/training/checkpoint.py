"""Atomic, identity-bound training checkpoints."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


_TOLERANCE_NORMALIZED_EVALUATION_HASHES = {
    "golden_evaluation_sha256",
    "full_evaluation_sha256",
}


def compare_checkpoint_payloads(
    reference_path: Path,
    resumed_path: Path,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
    max_reported_mismatches: int = 64,
) -> dict[str, Any]:
    """Compare every state required for deterministic interrupted resume."""
    import torch

    if atol < 0.0 or rtol < 0.0:
        raise ValueError("checkpoint comparison tolerances must be non-negative")
    reference = torch.load(
        Path(reference_path), map_location="cpu", weights_only=False
    )
    resumed = torch.load(Path(resumed_path), map_location="cpu", weights_only=False)
    mismatches: list[str] = []
    mismatch_count = 0
    max_abs_error = 0.0
    normalized_evaluation_hash_paths: set[str] = set()

    def record(message: str) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if len(mismatches) < max_reported_mismatches:
            mismatches.append(message)

    def validated_evaluation_hashes(value: Any, path: str) -> set[str]:
        validated: set[str] = set()
        if isinstance(value, dict):
            for hash_key in _TOLERANCE_NORMALIZED_EVALUATION_HASHES:
                if hash_key not in value:
                    continue
                expected = str(value[hash_key])
                unsigned = dict(value)
                unsigned.pop(hash_key, None)
                actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
                hash_path = f"{path}.{hash_key}"
                if actual != expected:
                    record(
                        f"{hash_path}: signed evaluation hash mismatch "
                        f"(expected {expected}, computed {actual})"
                    )
                else:
                    validated.add(hash_path)
            for key, child in value.items():
                validated.update(validated_evaluation_hashes(child, f"{path}.{key}"))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                validated.update(validated_evaluation_hashes(child, f"{path}[{index}]"))
        return validated

    reference_validated_hashes = validated_evaluation_hashes(reference, "checkpoint")
    resumed_validated_hashes = validated_evaluation_hashes(resumed, "checkpoint")

    def compare(left: Any, right: Any, path: str) -> None:
        nonlocal max_abs_error
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                record(f"{path}: tensor/type mismatch")
                return
            if left.shape != right.shape:
                record(f"{path}: shape {tuple(left.shape)} != {tuple(right.shape)}")
                return
            if left.dtype != right.dtype:
                record(f"{path}: dtype {left.dtype} != {right.dtype}")
                return
            if left.is_floating_point() or left.is_complex():
                error = 0.0
                if left.numel():
                    error = float((left - right).abs().max().item())
                    max_abs_error = max(max_abs_error, error)
                if not torch.allclose(left, right, atol=atol, rtol=rtol):
                    record(f"{path}: floating tensor differs (max_abs={error:.9g})")
            elif not torch.equal(left, right):
                record(f"{path}: tensor differs")
            return
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                record(f"{path}: dict/type mismatch")
                return
            left_keys = set(left)
            right_keys = set(right)
            for key in sorted(left_keys - right_keys, key=repr):
                record(f"{path}.{key}: missing from resumed checkpoint")
            for key in sorted(right_keys - left_keys, key=repr):
                record(f"{path}.{key}: unexpected in resumed checkpoint")
            for key in sorted(left_keys & right_keys, key=repr):
                child_path = f"{path}.{key}"
                if (
                    key in _TOLERANCE_NORMALIZED_EVALUATION_HASHES
                    and child_path.startswith(
                        "checkpoint.training_state.golden_evaluation."
                    )
                    and child_path in reference_validated_hashes
                    and child_path in resumed_validated_hashes
                ):
                    normalized_evaluation_hash_paths.add(child_path)
                    continue
                compare(left[key], right[key], child_path)
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if not isinstance(left, (list, tuple)) or not isinstance(
                right, (list, tuple)
            ):
                record(f"{path}: sequence/type mismatch")
                return
            if len(left) != len(right):
                record(f"{path}: length {len(left)} != {len(right)}")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, float) or isinstance(right, float):
            try:
                left_float = float(left)
                right_float = float(right)
            except (TypeError, ValueError):
                record(f"{path}: float/type mismatch")
                return
            error = abs(left_float - right_float)
            max_abs_error = max(max_abs_error, error)
            if not math.isclose(left_float, right_float, abs_tol=atol, rel_tol=rtol):
                record(f"{path}: {left_float} != {right_float}")
            return
        if left != right:
            record(f"{path}: {left!r} != {right!r}")

    compare(reference, resumed, "checkpoint")
    reference_params = reference.get("params", {})
    resumed_params = resumed.get("params", {})
    return {
        "schema_version": 1,
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "atol": float(atol),
        "rtol": float(rtol),
        "reference_step": reference.get("step"),
        "resumed_step": resumed.get("step"),
        "reference_gaussian_count": None
        if "means" not in reference_params
        else int(len(reference_params["means"])),
        "resumed_gaussian_count": None
        if "means" not in resumed_params
        else int(len(resumed_params["means"])),
        "max_abs_error": max_abs_error,
        "mismatch_count": mismatch_count,
        "mismatches": mismatches,
        "tolerance_normalized_evaluation_hash_count": len(
            normalized_evaluation_hash_paths
        ),
        "tolerance_normalized_evaluation_hash_paths": sorted(
            normalized_evaluation_hash_paths
        ),
        "compared_state": [
            "parameters_and_gaussian_order",
            "optimizer_state",
            "MCMC_strategy_state",
            "sampler_state",
            "training_telemetry",
            "auxiliary_state",
            "CPU_and_CUDA_RNG_state",
            "signed_evaluation_history_semantics",
        ],
    }


def save_checkpoint(
    path: Path,
    *,
    step: int,
    identity: dict[str, Any],
    params: Any,
    optimizers: dict[str, Any],
    strategy_state: Any,
    sampler_state: Any,
    training_state: dict[str, Any],
    auxiliary_params: dict[str, Any] | None = None,
    auxiliary_optimizers: dict[str, Any] | None = None,
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "step": int(step),
        "identity": identity,
        "params": params.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "strategy_state": strategy_state,
        "sampler_state": sampler_state,
        "training_state": training_state,
        "auxiliary_params": {
            name: value.detach().clone()
            for name, value in ({} if auxiliary_params is None else auxiliary_params).items()
        },
        "auxiliary_optimizers": {
            name: optimizer.state_dict()
            for name, optimizer in (
                {} if auxiliary_optimizers is None else auxiliary_optimizers
            ).items()
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path,
    *,
    expected_identity: dict[str, Any],
    params: Any,
    optimizers: dict[str, Any],
    map_location: str,
    auxiliary_params: dict[str, Any] | None = None,
    auxiliary_optimizers: dict[str, Any] | None = None,
) -> tuple[int, Any, Any, dict[str, Any]]:
    import torch

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("identity") != expected_identity:
        raise ValueError("checkpoint identity does not match this training run")
    checkpoint_params = payload["params"]
    if set(checkpoint_params) != set(params):
        raise ValueError("checkpoint parameter names do not match the trainer")
    for name, value in checkpoint_params.items():
        if params[name].shape != value.shape:
            replacement = torch.nn.Parameter(value.to(map_location))
            params[name] = replacement
            optimizers[name].param_groups[0]["params"] = [replacement]
    params.load_state_dict(checkpoint_params)
    for name, optimizer in optimizers.items():
        if name not in payload["optimizers"]:
            raise ValueError(f"checkpoint has no optimizer state for {name}")
        optimizer.load_state_dict(payload["optimizers"][name])
    expected_auxiliary = {} if auxiliary_params is None else auxiliary_params
    checkpoint_auxiliary = payload.get("auxiliary_params", {})
    if set(checkpoint_auxiliary) != set(expected_auxiliary):
        raise ValueError("checkpoint auxiliary parameter names do not match the trainer")
    with torch.no_grad():
        for name, parameter in expected_auxiliary.items():
            value = checkpoint_auxiliary[name]
            if parameter.shape != value.shape:
                raise ValueError(f"checkpoint auxiliary parameter shape mismatch for {name}")
            parameter.copy_(value.to(map_location))
    expected_auxiliary_optimizers = (
        {} if auxiliary_optimizers is None else auxiliary_optimizers
    )
    checkpoint_auxiliary_optimizers = payload.get("auxiliary_optimizers", {})
    if set(checkpoint_auxiliary_optimizers) != set(expected_auxiliary_optimizers):
        raise ValueError("checkpoint auxiliary optimizer names do not match the trainer")
    for name, optimizer in expected_auxiliary_optimizers.items():
        optimizer.load_state_dict(checkpoint_auxiliary_optimizers[name])
    torch.set_rng_state(payload["torch_rng_state"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        # torch.load with a CUDA map_location moves the saved RNG blobs onto the
        # device, but set_rng_state_all requires CPU ByteTensors.
        torch.cuda.set_rng_state_all(
            [state.cpu().to(torch.uint8) for state in payload["cuda_rng_state"]]
        )
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("checkpoint has no training metric state")
    return (
        int(payload["step"]),
        payload["strategy_state"],
        payload["sampler_state"],
        training_state,
    )
