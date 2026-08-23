#!/usr/bin/env python3
"""Verify completed Gate 2 Trainer A/B runs and sign their metric report."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.ab_results import (
    build_trainer_ab_report,
    verify_trainer_ab_report,
)


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing A/B report: {path}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-lpips-not-run", action="store_true")
    parser.add_argument("--minimum-full-evals", type=int, default=2)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    report = build_trainer_ab_report(
        matrix,
        matrix_root=args.matrix.parent,
        require_lpips=not args.allow_lpips_not_run,
        minimum_periodic_full_evaluations=args.minimum_full_evals,
    )
    verify_trainer_ab_report(report, matrix)
    _write_atomic(args.output, report)
    print(
        f"A/B status={report['gate2_quality_candidate']['status']}, "
        f"sha256={report['ab_report_sha256']} -> {args.output}"
    )
    return 0 if report["gate2_quality_candidate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
