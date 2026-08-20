"""Atomic, identity-bound training checkpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


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
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("checkpoint has no training metric state")
    return (
        int(payload["step"]),
        payload["strategy_state"],
        payload["sampler_state"],
        training_state,
    )
