#!/usr/bin/env python3
"""Concatenate compatible Gaussian checkpoint layers without retraining geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


PARAMETER_NAMES = ("means", "quats", "scales", "opacities", "sh0", "shN")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_params(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"checkpoint has no params mapping: {path}")
    missing = [name for name in PARAMETER_NAMES if name not in params]
    if missing:
        raise ValueError(f"checkpoint is missing {missing}: {path}")
    count = int(params["means"].shape[0])
    for name in PARAMETER_NAMES:
        if int(params[name].shape[0]) != count:
            raise ValueError(f"parameter {name} has inconsistent count in {path}")
    return payload, params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    base_payload, base = load_params(args.base)
    layer_payload, layer = load_params(args.layer)
    for name in PARAMETER_NAMES:
        if base[name].shape[1:] != layer[name].shape[1:]:
            raise ValueError(
                f"parameter {name} shape mismatch: "
                f"{tuple(base[name].shape)} vs {tuple(layer[name].shape)}"
            )
        if base[name].dtype != layer[name].dtype:
            layer[name] = layer[name].to(dtype=base[name].dtype)

    output_payload = dict(base_payload)
    output_payload["params"] = {
        name: torch.cat((base[name], layer[name]), dim=0)
        for name in PARAMETER_NAMES
    }
    output_payload["composed_layers"] = [
        {
            "kind": "base_surface",
            "path": args.base.resolve().as_posix(),
            "sha256": sha256(args.base),
            "gaussian_count": int(base["means"].shape[0]),
        },
        {
            "kind": "independent_sky",
            "path": args.layer.resolve().as_posix(),
            "sha256": sha256(args.layer),
            "gaussian_count": int(layer["means"].shape[0]),
            "metadata": layer_payload.get("sky_dome"),
        },
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(args.output)

    report = {
        "schema_version": 1,
        "kind": "gaussian_checkpoint_layer_composition_v1",
        "status": "PASS",
        "base_checkpoint": args.base.resolve().as_posix(),
        "base_checkpoint_sha256": output_payload["composed_layers"][0]["sha256"],
        "base_gaussian_count": int(base["means"].shape[0]),
        "layer_checkpoint": args.layer.resolve().as_posix(),
        "layer_checkpoint_sha256": output_payload["composed_layers"][1]["sha256"],
        "layer_gaussian_count": int(layer["means"].shape[0]),
        "total_gaussian_count": int(output_payload["params"]["means"].shape[0]),
        "output_checkpoint": args.output.resolve().as_posix(),
        "output_checkpoint_sha256": sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"composed {report['base_gaussian_count']:,} + "
        f"{report['layer_gaussian_count']:,} = "
        f"{report['total_gaussian_count']:,} Gaussians -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
