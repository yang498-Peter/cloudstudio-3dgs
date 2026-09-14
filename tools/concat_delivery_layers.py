"""Join a delivery's body and sky layers into the single model a customer receives.

A delivery ships as two files: the body PLY and a frozen sky PLY, which ``tools/pipeline.py``
copies side by side at publish time. Nothing composited them, so every quality number scored
the body alone against photographs that contain sky - and the body's sky is *correctly*
transparent, because the sky layer supplies it. The coverage metric read that correct
transparency as a failure.

Measured on house0305's 48 battery views, body alone against body plus sky:

    no fill    alpha p05  0.189 -> 0.898
    sharpc1    alpha p05  0.722 -> 0.915
    sharpc0    alpha p05  0.825 -> 0.929

while PSNR moved by at most 0.018, because the evaluator already composites a per-view backdrop
behind the render. Only alpha was penalised, and alpha is what the gate read. A whole line of
work went into filling a hole that existed only in how the body was scored.

This makes the pair evaluable so a gate can read the product instead of half of it. Plain
concatenation: both inputs are final, so there is no ownership, occupancy or exposure rule to
apply. The one subtlety is spherical harmonics - see ``concat_params``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

PARAMETER_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def concat_params(body: dict, sky: dict) -> dict:
    """Concatenate two parameter dicts, zero-padding the sky layer's SH bands if needed.

    The frozen sky layer is DC-only (``shN`` with zero bands) while a body carries SH1. Padding
    the sky's rest coefficients with ZERO is exact rather than invented: a spherical harmonic
    sum whose rest terms are zero equals its DC term, so a padded row renders identically to the
    degree-0 row it came from. Padding the other way would truncate coefficients that carry real
    colour, so that direction is refused.
    """
    merged = {}
    for key in PARAMETER_KEYS:
        if key not in body:
            raise ValueError(f"body checkpoint has no {key}")
        if key not in sky:
            raise ValueError(f"sky checkpoint has no {key}")
        left = body[key].detach().cpu()
        right = sky[key].detach().cpu()
        if left.shape[1:] != right.shape[1:]:
            if key != "shN" or left.ndim != 3 or right.ndim != 3 or left.shape[2] != right.shape[2]:
                raise ValueError(
                    f"{key} band layout differs and is not a zero-paddable shN: "
                    f"body {tuple(left.shape[1:])} vs sky {tuple(right.shape[1:])}"
                )
            if right.shape[1] > left.shape[1]:
                raise ValueError(
                    f"the sky layer carries more SH bands than the body "
                    f"({right.shape[1]} > {left.shape[1]}); padding would truncate real coefficients"
                )
            pad = torch.zeros(
                (right.shape[0], left.shape[1] - right.shape[1], right.shape[2]),
                dtype=right.dtype,
            )
            right = torch.cat([right, pad], dim=1)
        merged[key] = torch.cat([left, right], dim=0)
    return merged


def _params_of(blob: dict, what: str) -> dict:
    params = blob.get("params") or blob.get("splats")
    if not isinstance(params, dict) or "means" not in params:
        raise ValueError(f"{what} checkpoint carries no params dict")
    return params


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--body", type=pathlib.Path, required=True)
    parser.add_argument("--sky", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    body = torch.load(args.body, map_location="cpu", weights_only=False)
    sky = torch.load(args.sky, map_location="cpu", weights_only=False)
    body_params = _params_of(body, "body")
    sky_params = _params_of(sky, "sky")

    out = dict(body)
    out["params"] = concat_params(body_params, sky_params)
    # Optimizer and strategy state describe the body's own training; neither survives a join,
    # and carrying them would let a resume mistake the pair for a trainable model.
    for key in ("optimizers", "strategy_state", "auxiliary_optimizers"):
        out.pop(key, None)
    out["delivery_layers"] = {
        "body": {"path": str(args.body), "gaussian_count": int(len(body_params["means"]))},
        "sky": {"path": str(args.sky), "gaussian_count": int(len(sky_params["means"]))},
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(out, tmp)
    tmp.replace(args.output)
    print(
        "body %d + sky %d = %d gaussians -> %s"
        % (len(body_params["means"]), len(sky_params["means"]),
           len(out["params"]["means"]), args.output)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
