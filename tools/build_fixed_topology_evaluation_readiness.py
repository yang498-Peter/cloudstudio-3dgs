#!/usr/bin/env python3
"""Bind fixed-topology evidence without silently authorizing a long run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.evaluation.quality_report import verify_run_manifest
from cloudstudio_3dgs.pipeline.mipmap_gate import load_and_verify_gate


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_embedded_hash(value: dict[str, Any], field: str) -> str:
    expected = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if expected != actual:
        raise ValueError(f"{field} mismatch")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-gate", required=True, type=Path)
    parser.add_argument("--accuracy-coverage", required=True, type=Path)
    parser.add_argument("--ownership", required=True, type=Path)
    parser.add_argument("--directional-smoke", required=True, type=Path)
    parser.add_argument("--fullres-smoke", required=True, type=Path)
    parser.add_argument("--evaluation-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    upstream, upstream_sha = load_and_verify_gate(args.upstream_gate)
    if upstream.get("status") != "UPSTREAM_DATA_READY":
        raise ValueError("fixed-topology evaluation requires UPSTREAM_DATA_READY")
    accuracy = _read(args.accuracy_coverage)
    accuracy_sha = _verify_embedded_hash(accuracy, "audit_sha256")
    if accuracy.get("status") != "ACCURACY_COVERAGE_READY":
        raise ValueError("accuracy times coverage audit is not ready")
    ownership = _read(args.ownership)
    ownership_sha = _verify_embedded_hash(ownership, "ownership_contract_sha256")
    if ownership.get("export_scope") != "core_owner_only":
        raise ValueError("ownership contract is not core-only")
    directional = _read(args.directional_smoke)
    directional_sha = verify_run_manifest(directional)
    fullres = _read(args.fullres_smoke)
    fullres_sha = verify_run_manifest(fullres)
    for name, run in (("directional", directional), ("fullres", fullres)):
        training = run.get("training", {})
        if training.get("status") != "IMPLEMENTATION_SMOKE_COMPLETE":
            raise ValueError(f"{name} smoke did not complete")
        topology = training.get("topology", {})
        if topology.get("policy", {}).get("mode") != "strict_fixed":
            raise ValueError(f"{name} smoke was not strict_fixed")
        if topology.get("initial_gaussian_count") != topology.get(
            "final_gaussian_count"
        ):
            raise ValueError(f"{name} smoke changed topology")
    directional_audits = directional["training"]["optimization_phases"]["audits"]
    if len(directional_audits) < 2:
        raise ValueError("directional smoke lacks Phase A and B audits")
    if any(
        not audit.get("range_directionality", {}).get("directional_pass", False)
        for audit in directional_audits
    ):
        raise ValueError("directional range smoke failed")
    if directional_audits[0]["parameter_updates"]["means"]["changed_count"] != 0:
        raise ValueError("Phase A changed means")
    if directional_audits[-1]["point_to_plane_drift"]["max_m"] > 0.01:
        raise ValueError("Phase B point-to-plane drift exceeded 1 cm")
    if fullres["trainer_contract"]["factor"] != 1:
        raise ValueError("full-resolution smoke did not use factor=1")

    plan = _read(args.evaluation_plan)
    plan_sha = _verify_embedded_hash(plan, "evaluation_plan_sha256")
    unsigned = {
        "schema_version": 1,
        "kind": "fixed_topology_evaluation_readiness_v1",
        "status": "FIXED_TOPOLOGY_EVALUATION_PREPARED",
        "tile_id": 1,
        "upstream_gate_manifest_sha256": upstream_sha,
        "accuracy_coverage_audit_sha256": accuracy_sha,
        "core_ownership_contract_sha256": ownership_sha,
        "directional_smoke_run_manifest_sha256": directional_sha,
        "fullres_smoke_run_manifest_sha256": fullres_sha,
        "evaluation_plan_sha256": plan_sha,
        "evidence": {
            "range_semantics": "euclidean_ray_range_m",
            "accuracy_max_m": accuracy["accuracy_m"]["max"],
            "coverage_fraction": accuracy["coverage_fraction"],
            "per_view_coverage_p5": accuracy["per_view_coverage"]["p5"],
            "directional_pass": True,
            "phase_a_geometry_frozen": True,
            "phase_b_point_to_plane_max_m": directional_audits[-1][
                "point_to_plane_drift"
            ]["max_m"],
            "fullres_peak_vram_bytes": fullres["training"]["peak_vram_bytes"],
            "core_only_merge_contract": True,
        },
        "training_allowed": False,
        "reason": "prepared evidence requires explicit promotion before 7480-step A0/A1",
        "adaptive_growth_allowed": False,
    }
    result = dict(unsigned)
    result["readiness_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
