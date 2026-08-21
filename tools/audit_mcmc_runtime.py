#!/usr/bin/env python3
"""Write deterministic full-MCMC registration evidence for one locked GPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.backend import verify_gsplat_runtime
from cloudstudio_3dgs.training.runtime_evidence import (
    audit_loaded_mcmc_runtime,
    execute_mcmc_native_kernel_smoke,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sanitize_runtime(runtime: dict) -> dict:
    allowed = {
        "package",
        "version",
        "locked_commit",
        "source_kind",
        "commit",
        "clean",
    }
    return {key: runtime[key] for key in sorted(allowed) if key in runtime}


def build_evidence(lock_path: Path, *, execute_kernels: bool = False) -> dict:
    import torch

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    provenance_error = None
    try:
        runtime = verify_gsplat_runtime(lock_path)
    except Exception as exc:  # evidence must still explain a fail-closed result
        provenance_error = f"{type(exc).__name__}: {exc}"
        runtime = {
            "package": "gsplat",
            "version": None,
            "locked_commit": str(lock.get("commit")),
            "commit": None,
            "clean": False,
        }
    report = audit_loaded_mcmc_runtime(runtime)
    registration_passed = report["status"] == "PASS_REGISTERED"
    evidence = {
        "schema_version": 1,
        "evidence_type": "cloudstudio_full_mcmc_runtime_registration",
        "gate_status": "PASS_REGISTRATION_ONLY" if registration_passed else "FAIL",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": report["cuda_device_name"],
        },
        "lock": {
            key: lock.get(key)
            for key in (
                "package",
                "version",
                "commit",
                "python",
                "torch",
                "cuda",
                "cuda_arch_list",
                "patch",
                "source_policy",
            )
        },
        "runtime": _sanitize_runtime(runtime),
        "provenance_error": provenance_error,
        "operator_registration": report,
        "execution_gates": {
            "covariance_forward_backward": "NOT_RUN",
            "mcmc_noise_nonzero": "NOT_RUN",
            "relocation_occurred": "NOT_RUN",
            "sample_add_occurred": "NOT_RUN",
            "rasterization_forward_backward": "NOT_RUN",
            "interrupted_resume_equivalence": "NOT_RUN",
            "real_gpu_training": "NOT_RUN",
        },
    }
    if execute_kernels:
        if registration_passed:
            try:
                kernel_smoke = execute_mcmc_native_kernel_smoke()
            except Exception as exc:
                kernel_smoke = {
                    "schema_version": 1,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            kernel_smoke = {
                "schema_version": 1,
                "status": "NOT_RUN_INCOMPLETE_RUNTIME",
            }
        evidence["native_kernel_smoke"] = kernel_smoke
        evidence["gate_status"] = (
            "PASS_KERNEL_SMOKE" if kernel_smoke["status"] == "PASS" else "FAIL"
        )
    evidence["runtime_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(evidence)
    ).hexdigest()
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gsplat-lock",
        type=Path,
        default=ROOT / "upstream" / "cloudstudio_trainer.lock.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute-kernels", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"runtime evidence already exists: {args.output}")
    evidence = build_evidence(
        args.gsplat_lock, execute_kernels=args.execute_kernels
    )
    _atomic_json(args.output, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if str(evidence["gate_status"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
