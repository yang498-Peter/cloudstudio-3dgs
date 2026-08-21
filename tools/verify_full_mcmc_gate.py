#!/usr/bin/env python3
"""Verify one signed full-MCMC Gate 1 evidence payload against the repo lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.runtime_evidence import (
    verify_full_mcmc_gate_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--gsplat-lock",
        type=Path,
        default=ROOT / "upstream" / "cloudstudio_trainer.lock.json",
    )
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    lock = json.loads(args.gsplat_lock.read_text(encoding="utf-8"))
    report = verify_full_mcmc_gate_evidence(
        evidence, expected_lock_commit=str(lock["commit"])
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
