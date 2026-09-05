#!/usr/bin/env python3
"""Assemble the first-batch evidence registry from the artifacts on disk.

Every claim in the ledger points at a file with a hash and a verdict that a
script computed, not a sentence someone remembered. Re-running this after any
artifact changes keeps the registry honest; editing it by hand does not.

    python tools/build_evidence_registry.py --root research/quality_recovery_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(path: Path, verdict: str, claim: str, **detail: Any) -> dict[str, Any]:
    record = {
        "artifact": str(path).replace("\\", "/"),
        "sha256": _sha(path) if path.exists() else None,
        "verdict": verdict,
        "claim": claim,
    }
    record.update(detail)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("research/quality_recovery_v1"))
    args = parser.parse_args()
    root = args.root
    entries: list[dict[str, Any]] = []

    # WP00 identity
    for path in sorted((root / "identity").glob("*.json")):
        record = _load(path)
        run = record.get("run") or {}
        checkpoint = run.get("checkpoint") or record.get("checkpoint") or {}
        payload = checkpoint.get("payload") or {}
        missing = [
            key for key, value in (run.get("inputs") or {}).items() if value.get("missing")
        ]
        entries.append(
            _entry(
                path,
                "FAIL" if missing else "PASS",
                "identity frozen: config_as_run, inputs, checkpoint, extension hashed",
                wp="WP00",
                gaussian_count=payload.get("gaussian_count"),
                sh_rest_coeffs=payload.get("sh_rest_coeffs"),
                cap_max=(run.get("resolved") or {}).get("cap_max"),
                missing_inputs=missing,
                extension_sha256=((record.get("runtime") or {}).get("gsplat") or {})
                .get("extension", {})
                .get("sha256"),
            )
        )
    dag = root / "cache_dependency_dag.json"
    if dag.exists():
        loaded = _load(dag)
        entries.append(
            _entry(
                dag,
                "PASS",
                "cache dependency DAG generated from frozen identities",
                wp="WP00",
                nodes=len(loaded["nodes"]),
                edges=len(loaded["edges"]),
                protected_roots=len(loaded["protected_roots"]),
            )
        )

    # WP00 CI channels
    for name, channel in (
        ("collection-cpu-local.json", "cpu"),
        ("collection-torch-cpu.json", "torch-cpu"),
        ("collection-cuda.json", "cuda"),
    ):
        path = root / "ci" / name
        if not path.exists():
            entries.append(
                {
                    "artifact": f"{root}/ci/{name}".replace("\\", "/"),
                    "sha256": None,
                    "verdict": "NOT_RUN",
                    "claim": f"{channel} test channel audited",
                    "wp": "WP00",
                }
            )
            continue
        summary = _load(path)
        entries.append(
            _entry(
                path,
                "PASS" if not summary["violations"] else "FAIL",
                f"{channel} test channel audited by tests/check_collection.py",
                wp="WP00",
                collected=summary["collected"],
                passed=summary["passed"],
                skipped=summary["skipped"],
                failed=summary["failed"],
                errored=summary["errored"],
                optional_not_run=summary["optional_not_run"],
                violations=summary["violations"],
            )
        )

    # WP01 roundtrip
    report_path = root / "wp01_roundtrip" / "report.json"
    if report_path.exists():
        report = _load(report_path)
        verdict = report["verdict"]
        entries.append(
            _entry(
                report_path,
                "PASS" if all(v == "PASS" for v in verdict.values()) else "FAIL",
                "checkpoint -> PLY -> checkpoint -> same-backend render roundtrip with controls",
                wp="WP01",
                sub_verdicts=verdict,
                gaussian_count=report["gaussian_count"],
                sh_degree=report["sh_degree"],
                tensor_max_abs={
                    k: v.get("max_abs")
                    for k, v in report["tensor_roundtrip"].items()
                    if isinstance(v, dict)
                },
                render_max_abs=[f["max_abs"] for f in report["render_roundtrip"]["frames"]],
                render_psnr=[f["psnr"] for f in report["render_roundtrip"]["frames"]],
                controls={
                    k: {"max_abs": v["max_abs"], "psnr": v["psnr"]}
                    for k, v in report["controls"].items()
                    if isinstance(v, dict)
                },
            )
        )
    else:
        entries.append(
            {
                "artifact": str(report_path).replace("\\", "/"),
                "sha256": None,
                "verdict": "NOT_RUN",
                "claim": "roundtrip",
                "wp": "WP01",
            }
        )

    registry = {
        "schema_version": 1,
        "entries": entries,
        "counts": {
            verdict: sum(1 for e in entries if e["verdict"] == verdict)
            for verdict in ("PASS", "FAIL", "NOT_RUN")
        },
    }
    out = root / "evidence_registry.json"
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, out)
    print(json.dumps(registry["counts"]))
    for entry in entries:
        print(f"  {entry['verdict']:8s} {entry.get('wp','')} {entry['claim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
