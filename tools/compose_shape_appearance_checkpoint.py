#!/usr/bin/env python3
"""Compose verified Gaussian shape with verified appearance/coverage parameters."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-checkpoint", required=True, type=Path)
    parser.add_argument("--appearance-checkpoint", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    shape_payload = torch.load(
        args.shape_checkpoint, map_location="cpu", weights_only=False
    )
    appearance_payload = torch.load(
        args.appearance_checkpoint, map_location="cpu", weights_only=False
    )
    shape = shape_payload.get("params") or shape_payload.get("splats")
    appearance = appearance_payload.get("params") or appearance_payload.get("splats")
    if not isinstance(shape, dict) or not isinstance(appearance, dict):
        raise ValueError("both checkpoints must contain Gaussian params")

    required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    if not required.issubset(shape) or not required.issubset(appearance):
        raise ValueError("checkpoint Gaussian parameter layout is incomplete")
    if tuple(shape) != tuple(appearance):
        raise ValueError("checkpoint Gaussian parameter order differs")
    if any(shape[key].shape != appearance[key].shape for key in required):
        raise ValueError("checkpoint Gaussian parameter shapes differ")
    if not torch.equal(shape["means"], appearance["means"]):
        maximum = float((shape["means"] - appearance["means"]).abs().max())
        raise ValueError(f"Gaussian row identity differs (max mean delta {maximum})")
    if not torch.equal(shape["shN"], appearance["shN"]):
        raise ValueError("SH-rest layout/content differs")

    params = {
        "means": appearance["means"].detach().cpu().clone(),
        "opacities": appearance["opacities"].detach().cpu().clone(),
        "quats": shape["quats"].detach().cpu().clone(),
        "scales": shape["scales"].detach().cpu().clone(),
        "sh0": appearance["sh0"].detach().cpu().clone(),
        "shN": appearance["shN"].detach().cpu().clone(),
    }
    shape_sha = _sha256(args.shape_checkpoint)
    appearance_sha = _sha256(args.appearance_checkpoint)
    identity = copy.deepcopy(appearance_payload.get("identity", {}))
    identity.update(
        {
            "kind": "gaussian_shape_appearance_composition_v1",
            "shape_checkpoint_sha256": shape_sha,
            "appearance_checkpoint_sha256": appearance_sha,
            "row_identity": "exact_means_tensor_equality",
            "parameter_sources": {
                "means": "appearance",
                "opacities": "appearance",
                "sh0": "appearance",
                "shN": "appearance",
                "scales": "shape",
                "quats": "shape",
            },
        }
    )
    auxiliary_params = {
        key: value.detach().cpu().clone()
        for key, value in appearance_payload.get("auxiliary_params", {}).items()
    }
    output = {
        "schema_version": 1,
        "step": int(appearance_payload.get("step", 0)),
        "identity": identity,
        "params": params,
        "auxiliary_params": auxiliary_params,
    }
    _save_checkpoint(args.output_checkpoint, output)
    report = {
        "schema_version": 1,
        "kind": "gaussian_shape_appearance_composition_report_v1",
        "status": "PASS",
        "gaussian_count": int(params["means"].shape[0]),
        "row_identity": "exact_means_tensor_equality",
        "shape_checkpoint": args.shape_checkpoint.resolve().as_posix(),
        "shape_checkpoint_sha256": shape_sha,
        "appearance_checkpoint": args.appearance_checkpoint.resolve().as_posix(),
        "appearance_checkpoint_sha256": appearance_sha,
        "output_checkpoint": args.output_checkpoint.resolve().as_posix(),
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "parameter_sources": identity["parameter_sources"],
        "auxiliary_parameter_sources": {
            key: "appearance" for key in sorted(auxiliary_params)
        },
    }
    _write_json(args.output_report, report)
    print(
        f"composed {report['gaussian_count']} Gaussians -> {args.output_checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
