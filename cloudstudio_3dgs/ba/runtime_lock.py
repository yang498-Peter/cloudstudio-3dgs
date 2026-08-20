"""Validation and provenance evidence for the optional Rig BA runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def load_runtime_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("Rig BA runtime lock has an unsupported schema")
    components = lock.get("components")
    if not isinstance(components, dict):
        raise ValueError("Rig BA runtime lock has no components")
    for name in ("hloc", "lightglue", "aliked", "pycolmap"):
        component = components.get(name)
        if not isinstance(component, dict):
            raise ValueError(f"Rig BA runtime lock is missing {name}")
        if not component.get("repo") or not component.get("license"):
            raise ValueError(f"Rig BA runtime lock has incomplete {name} provenance")
        commit = component.get("commit")
        if commit is not None and (
            len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit)
        ):
            raise ValueError(f"Rig BA runtime lock has an invalid {name} commit")
    return lock


def runtime_lock_sha256(lock: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(lock)).hexdigest()


def verify_signed_runtime_manifest(
    manifest: dict[str, Any], signature_field: str
) -> str:
    expected = str(manifest.get(signature_field, ""))
    if not expected:
        raise ValueError(f"runtime manifest has no {signature_field}")
    unsigned = dict(manifest)
    unsigned.pop(signature_field, None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"runtime manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    return actual


def _installed_vcs_commit(distribution: importlib.metadata.Distribution) -> str | None:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    metadata = json.loads(raw)
    return metadata.get("vcs_info", {}).get("commit_id")


def collect_runtime_evidence(
    lock: dict[str, Any], *, allow_unverified_vcs: bool = False
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    for name in ("hloc", "lightglue", "pycolmap"):
        component = lock["components"][name]
        distribution_name = str(component["distribution"])
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"locked Rig BA distribution is not installed: {distribution_name}"
            ) from exc
        installed_version = distribution.version
        installed_commit = _installed_vcs_commit(distribution)
        expected_commit = component.get("commit")
        expected_version = component.get("version")
        status = "PASS"
        if expected_version and installed_version != expected_version:
            failures.append(
                f"{name} version {installed_version} != locked {expected_version}"
            )
            status = "FAIL"
        if expected_commit and installed_commit != expected_commit:
            status = "UNVERIFIED" if installed_commit is None else "FAIL"
            if status == "FAIL" or not allow_unverified_vcs:
                failures.append(
                    f"{name} commit {installed_commit or 'unknown'} != locked {expected_commit}"
                )
        evidence[name] = {
            "distribution": distribution_name,
            "installed_version": installed_version,
            "installed_vcs_commit": installed_commit,
            "expected_version": expected_version,
            "expected_vcs_commit": expected_commit,
            "status": status,
        }
    if failures:
        raise RuntimeError("; ".join(failures))
    return evidence
