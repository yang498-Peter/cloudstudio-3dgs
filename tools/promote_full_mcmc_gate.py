#!/usr/bin/env python3
"""Atomically promote one verified full-MCMC Gate 1 payload to the baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.runtime_evidence import (
    verify_full_mcmc_gate_evidence,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def promote_full_mcmc_gate_baseline(
    evidence: Mapping[str, Any],
    output: Path,
    *,
    expected_lock_commit: str,
    replace_pass: bool = False,
) -> dict[str, Any]:
    """Verify first, then atomically replace only a non-PASS baseline."""
    report = verify_full_mcmc_gate_evidence(
        evidence, expected_lock_commit=expected_lock_commit
    )
    if report["status"] != "PASS":
        raise ValueError(
            "full-MCMC gate evidence rejected: " + "; ".join(report["errors"])
        )
    output = Path(output)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("gate_status") == "PASS" and not replace_pass:
            raise FileExistsError(
                f"baseline already contains PASS evidence: {output}; "
                "use --replace-pass only for an explicitly reviewed replacement"
            )
    _atomic_json(output, evidence)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baselines" / "full_mcmc_runtime.baseline.json",
    )
    parser.add_argument(
        "--gsplat-lock",
        type=Path,
        default=ROOT / "upstream" / "cloudstudio_trainer.lock.json",
    )
    parser.add_argument("--replace-pass", action="store_true")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    lock = json.loads(args.gsplat_lock.read_text(encoding="utf-8"))
    report = promote_full_mcmc_gate_baseline(
        evidence,
        args.output,
        expected_lock_commit=str(lock["commit"]),
        replace_pass=args.replace_pass,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
