from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_contract(values: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a dictionary")
    contract: dict[str, dict[str, Any]] = {}
    for name, value in sorted(values.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{label} {name!r} is not a tensor")
        contract[str(name)] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return contract


def _require_equal(source: dict[str, Any], target: dict[str, Any], key: str) -> None:
    if source.get(key) != target.get(key):
        raise ValueError(f"sampling rebind lineage mismatch for {key}")


def rebind_checkpoint_sampling_identity(
    source_checkpoint: Path,
    target_lineage_checkpoint: Path,
    output_checkpoint: Path,
    report_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Create a warm-start-only checkpoint with a verified sampling identity.

    This is intentionally narrower than editing Trainer's lineage gate.  The
    source model parameters are preserved, while the target checkpoint only
    supplies a face-cache identity after all base manifests, coordinates,
    initialization, runtime, parameter shapes and auxiliary shapes agree.
    """

    source_checkpoint = Path(source_checkpoint).resolve()
    target_lineage_checkpoint = Path(target_lineage_checkpoint).resolve()
    output_checkpoint = Path(output_checkpoint).resolve()
    report_path = Path(report_path).resolve()
    for path, label in (
        (source_checkpoint, "source checkpoint"),
        (target_lineage_checkpoint, "target lineage checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    for path in (output_checkpoint, report_path):
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}")

    target = torch.load(target_lineage_checkpoint, map_location="cpu", weights_only=False)
    if target.get("schema_version") != 1:
        raise ValueError("unsupported target checkpoint schema")
    target_identity = target.get("identity")
    if not isinstance(target_identity, dict):
        raise ValueError("target checkpoint has no identity")
    target_identity = copy.deepcopy(target_identity)
    target_source_identity = target_identity.get("source_identity")
    if not isinstance(target_source_identity, dict):
        raise ValueError("target checkpoint is not bound to a derived sampling identity")
    if not target_identity.get("face_manifest_sha256"):
        raise ValueError("target checkpoint has no face manifest identity")
    target_param_contract = _tensor_contract(target.get("params"), "target params")
    target_aux_contract = _tensor_contract(
        target.get("auxiliary_params", {}), "target auxiliary params"
    )
    target_step = int(target.get("step", 0))
    del target

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("schema_version") != 1:
        raise ValueError("unsupported source checkpoint schema")
    source_identity = source.get("identity")
    if not isinstance(source_identity, dict):
        raise ValueError("source checkpoint has no identity")
    if "source_identity" in source_identity:
        raise ValueError("source checkpoint is already bound to derived sampling")

    for key, expected in target_source_identity.items():
        if source_identity.get(key) != expected:
            raise ValueError(f"sampling rebind base identity mismatch for {key}")
    for key in (
        "coordinate_transform_sha256",
        "initialization_ply_sha256",
        "initialization_geometry_sha256",
        "surface_initialization_sha256",
        "gsplat_runtime",
    ):
        _require_equal(source_identity, target_identity, key)

    source_param_contract = _tensor_contract(source.get("params"), "source params")
    source_aux_contract = _tensor_contract(
        source.get("auxiliary_params", {}), "source auxiliary params"
    )
    if source_param_contract != target_param_contract:
        raise ValueError("sampling rebind parameter contract mismatch")
    if source_aux_contract != target_aux_contract:
        raise ValueError("sampling rebind auxiliary parameter contract mismatch")

    source_sha256 = _sha256_file(source_checkpoint)
    target_sha256 = _sha256_file(target_lineage_checkpoint)
    source_step = int(source.get("step", 0))
    derived = {
        "schema_version": 1,
        "step": source_step,
        "identity": target_identity,
        "params": source["params"],
        "auxiliary_params": source.get("auxiliary_params", {}),
        "derived_warm_start_only": True,
        "resume_supported": False,
        "sampling_identity_rebind": {
            "source_checkpoint_sha256": source_sha256,
            "target_lineage_checkpoint_sha256": target_sha256,
            "target_face_manifest_sha256": target_identity["face_manifest_sha256"],
        },
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_checkpoint.parent,
        prefix=f".{output_checkpoint.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(derived, temporary_path)
        os.replace(temporary_path, output_checkpoint)
    finally:
        temporary_path.unlink(missing_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sampling_identity_rebind",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_step": source_step,
        "target_lineage_checkpoint": str(target_lineage_checkpoint),
        "target_lineage_checkpoint_sha256": target_sha256,
        "target_step": target_step,
        "target_face_manifest_sha256": target_identity["face_manifest_sha256"],
        "parameter_contract": source_param_contract,
        "auxiliary_parameter_contract": source_aux_contract,
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": _sha256_file(output_checkpoint),
        "output_checkpoint_bytes": output_checkpoint.stat().st_size,
        "warm_start_supported": True,
        "resume_supported": False,
    }
    report["sampling_identity_rebind_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

