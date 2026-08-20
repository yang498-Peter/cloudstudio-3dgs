#!/usr/bin/env python3
"""Extract ALIKED features and LightGlue matches for a checked Rig match graph."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.ba.runtime_lock import (
    collect_runtime_evidence,
    load_runtime_lock,
    runtime_lock_sha256,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        default=REPOSITORY_ROOT / "upstream" / "rig_ba.lock.json",
    )
    parser.add_argument(
        "--allow-unverified-vcs",
        action="store_true",
        help="allow wheel installs without a provable VCS commit; evidence remains UNVERIFIED",
    )
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"HLoc image directory does not exist: {args.image_dir}")
    if not args.pairs.is_file():
        raise FileNotFoundError(f"HLoc pairs file does not exist: {args.pairs}")
    if not args.runtime_lock.is_file():
        raise FileNotFoundError(f"Rig BA runtime lock does not exist: {args.runtime_lock}")
    if args.output.resolve().is_relative_to(args.image_dir.resolve()):
        raise ValueError("HLoc output cannot be inside the immutable image directory")
    lock = load_runtime_lock(args.runtime_lock)
    runtime_evidence = collect_runtime_evidence(
        lock, allow_unverified_vcs=args.allow_unverified_vcs
    )
    try:
        import torch
        from hloc import extract_features, match_features
    except ImportError as exc:
        raise RuntimeError(
            "ALIKED+LightGlue requires the optional HLoc runtime; "
            "install the repository-locked runtime without replacing CUDA PyTorch"
        ) from exc
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but PyTorch cannot access CUDA")
    if "aliked-n16" not in extract_features.confs:
        raise RuntimeError("installed HLoc has no aliked-n16 extractor configuration")
    if "aliked+lightglue" not in match_features.confs:
        raise RuntimeError("installed HLoc has no aliked+lightglue matcher configuration")
    if args.output.exists():
        if not args.output.is_dir():
            raise NotADirectoryError(f"HLoc output is not a directory: {args.output}")
        if any(args.output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"HLoc output is not empty: {args.output}; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    pair_lines = [
        line.split() for line in args.pairs.read_text(encoding="utf-8").splitlines()
    ]
    if not pair_lines or any(len(pair) != 2 for pair in pair_lines):
        raise ValueError("HLoc pairs must contain exactly two image names per line")
    names = sorted({name for pair in pair_lines for name in pair})
    if any(not (args.image_dir / name).is_file() for name in names):
        missing = [name for name in names if not (args.image_dir / name).is_file()]
        raise FileNotFoundError(f"HLoc image list has missing files: {missing[:4]}")
    features_path = args.output / "features-aliked-n16.h5"
    matches_path = args.output / "matches-aliked-lightglue.h5"
    extract_features.main(
        extract_features.confs["aliked-n16"],
        args.image_dir,
        image_list=names,
        feature_path=features_path,
        overwrite=args.overwrite,
    )
    match_features.main(
        match_features.confs["aliked+lightglue"],
        args.pairs,
        features_path,
        matches=matches_path,
        overwrite=args.overwrite,
    )
    runtime = {
        "schema_version": 1,
        "extractor": "aliked-n16",
        "matcher": "aliked+lightglue",
        "image_count": len(names),
        "pair_file_sha256": sha256_file(args.pairs),
        "features_sha256": sha256_file(features_path),
        "matches_sha256": sha256_file(matches_path),
        "cuda_used": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "hloc_version": importlib.metadata.version("hloc"),
        "runtime_lock_sha256": runtime_lock_sha256(lock),
        "runtime": runtime_evidence,
    }
    runtime["runtime_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(runtime)
    ).hexdigest()
    atomic_write(
        args.output / "feature_runtime_manifest.json",
        (json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    print(
        f"HLoc ALIKED+LightGlue: images={len(names)}, cuda={runtime['cuda_used']} "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
