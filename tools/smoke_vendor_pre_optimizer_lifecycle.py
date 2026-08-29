#!/usr/bin/env python3
"""CUDA smoke for the recovered pre-optimizer classic 3DGS lifecycle.

This is intentionally synthetic: it isolates the dangerous contract that a
Split -> Clone -> Cull event replaces Parameter rows before Adam consumes the
current backward pass.  The real raster/data-path check remains the signed
Tile_1 step-502 run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.default_strategy_adapter import (
    DefaultStrategyAdapter,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    cuda_index = 0 if device.index is None else device.index

    torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats()

    adapter = DefaultStrategyAdapter(
        scene_scale=10.0,
        refine_start_iter=500,
        refine_stop_iter=2000,
        refine_every=100,
        reset_every=300,
        grow_grad2d=0.00015,
        prune_opa=0.1,
        split_scale_m=0.2,
        prune_scale_m=0.2,
        prune_scale2d=0.15,
        refine_scale2d_stop_iter=2000,
        exact_mipmap_lifecycle=True,
        growth_min_opacity=0.15,
        prune_opa_late=0.05,
        prune_switch_step=1000,
        reset_opacity_cap=0.2,
        revised_opacity=False,
        lifecycle_execution_order="pre_optimizer_vendor",
    )
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    device=device,
                )
            ),
            "scales": torch.nn.Parameter(
                torch.tensor(
                    [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1]],
                    device=device,
                ).log()
            ),
            "quats": torch.nn.Parameter(
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]] * 3,
                    device=device,
                )
            ),
            "opacities": torch.nn.Parameter(
                torch.tensor([0.2, 0.2, 0.05], device=device).logit()
            ),
            "colors": torch.nn.Parameter(torch.zeros(3, 3, device=device)),
        }
    )
    optimizers = {
        name: torch.optim.Adam([parameter], lr=1e-3)
        for name, parameter in params.items()
    }

    for parameter in params.values():
        parameter.grad = torch.ones_like(parameter)
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    row_gradient = torch.tensor([1.0, 2.0, 3.0], device=device)
    for parameter in params.values():
        view = (3,) + (1,) * (parameter.ndim - 1)
        parameter.grad = row_gradient.view(view).expand_as(parameter).clone()
    state = {
        "grad2d": torch.tensor([0.001, 0.001, 0.0], device=device),
        "count": torch.ones(3, device=device),
        "radii": torch.zeros(3, device=device),
        "scene_scale": 10.0,
    }
    preserved = adapter._preserve_current_step_gradients(params, state)
    clone_count, split_count = adapter._grow_mipmap(params, optimizers, state)
    cull_count = adapter._prune_mipmap(params, optimizers, state, step=500)
    adapter._restore_current_step_gradients(params, state, preserved)

    expected = torch.tensor([1.0, 2.0, 2.0, 1.0], device=device)
    actual = params["means"].grad[:, 0]
    if not torch.equal(actual, expected):
        raise RuntimeError(f"gradient remap mismatch: {actual.tolist()}")
    before_step = params["means"].detach().clone()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if not torch.all(params["means"].detach() != before_step):
        raise RuntimeError("Adam did not consume every remapped means gradient")

    means_state = optimizers["means"].state[params["means"]]
    if means_state["exp_avg"].shape[0] != 4:
        raise RuntimeError("Adam moment rows did not follow topology")
    if not torch.all(means_state["exp_avg"].abs().sum(dim=1) > 0.0):
        raise RuntimeError("one or more Adam moment rows remained empty")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "status": "PASS",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(cuda_index)
            if device.type == "cuda"
            else "cpu"
        ),
        "before_count": 3,
        "clone_count": clone_count,
        "split_parent_count": split_count,
        "cull_count": cull_count,
        "after_count": len(params["means"]),
        "gradient_provenance": actual.detach().cpu().tolist(),
        "adam_moment_rows": int(means_state["exp_avg"].shape[0]),
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else 0
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
