"""Fail-closed output policy for resumable HLoc feature and match artifacts."""

from __future__ import annotations

from pathlib import Path


FEATURES_NAME = "features-aliked-n16.h5"
MATCHES_NAME = "matches-aliked-lightglue.h5"
RUNTIME_MANIFEST_NAME = "feature_runtime_manifest.json"
PARTIAL_ARTIFACT_NAMES = frozenset({FEATURES_NAME, MATCHES_NAME})


def prepare_hloc_output(
    output_dir: Path,
    *,
    overwrite: bool = False,
    resume: bool = False,
) -> bool:
    """Validate an HLoc output directory and return the upstream overwrite flag."""
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"HLoc output is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = {path.name: path for path in output_dir.iterdir()}
    if not entries:
        return overwrite
    if overwrite:
        return True
    if not resume:
        raise FileExistsError(
            f"HLoc output is not empty: {output_dir}; pass --resume or --overwrite"
        )
    if RUNTIME_MANIFEST_NAME in entries:
        raise FileExistsError(
            f"HLoc output is already signed complete: {entries[RUNTIME_MANIFEST_NAME]}"
        )
    unknown = sorted(set(entries) - PARTIAL_ARTIFACT_NAMES)
    if unknown:
        raise ValueError(f"HLoc resume output contains unknown artifacts: {unknown[:4]}")
    for name, path in entries.items():
        if not path.is_file():
            raise ValueError(f"HLoc resume artifact is not a file: {path}")
    if MATCHES_NAME in entries and FEATURES_NAME not in entries:
        raise ValueError("HLoc matches cannot be resumed without the feature artifact")
    return False
