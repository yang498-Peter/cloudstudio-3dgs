#!/usr/bin/env python3
"""Write a signed subset of a dataset manifest by uniform temporal subsampling.

The subset keeps the parent manifest's schema: every top-level key is copied,
only ``images`` / ``rig_frames`` / ``splits`` are filtered, and the result is
re-signed with the same ``manifest_sha256`` rule as ``build_manifest``. When
the parent is rig-structured (``rig_frames`` present) rig frames are kept or
dropped whole so left/right pairs never split; images outside any rig frame
form their own single-image units.

Selection is deterministic: units are ordered by timestamp, ``k`` of ``n``
units are taken at a constant stride and the seed only chooses the phase
inside the first stride. A subset therefore covers the full capture span
instead of a contiguous chunk, which is what a scale-down validation needs.

    python tools/subset_dataset_manifest.py \
        --manifest C:/data/house0614_manifest/dataset_manifest.json \
        --count 900 --seed 0 --output-dir C:/data/house0614_subset900
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import verify_dataset_manifest

MANIFEST_NAME = "dataset_manifest.json"
REPORT_NAME = "report.json"
ALGORITHM_VERSION = "uniform_temporal_subset_v1"
PARENT_HASH_KEY = "derived:parent-dataset-manifest"


def _sign(manifest: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    unsigned["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return unsigned


def _units(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Group images into selection units ordered by capture time."""
    images = list(manifest.get("images", []))
    by_id = {str(image["image_id"]): image for image in images}
    if len(by_id) != len(images):
        raise ValueError("dataset manifest contains duplicate image IDs")
    rig_frames = list(manifest.get("rig_frames") or [])
    units: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for frame in rig_frames:
        ids = [str(value) for value in frame.get("image_ids", [])]
        members = [by_id[value] for value in ids if value in by_id]
        if not members:
            continue
        for value in ids:
            if value in claimed:
                raise ValueError(f"image {value} belongs to more than one rig frame")
            claimed.add(value)
        units.append(
            {
                "unit_id": str(frame["rig_frame_id"]),
                "kind": "rig_frame",
                "timestamp_ns": int(frame.get("timestamp_ns", members[0]["timestamp_ns"])),
                "image_ids": [str(member["image_id"]) for member in members],
            }
        )
    for image in images:
        image_id = str(image["image_id"])
        if image_id in claimed:
            continue
        units.append(
            {
                "unit_id": image_id,
                "kind": "image",
                "timestamp_ns": int(image["timestamp_ns"]),
                "image_ids": [image_id],
            }
        )
    units.sort(key=lambda unit: (unit["timestamp_ns"], unit["unit_id"]))
    structure = "rig_frames" if rig_frames else "images"
    return units, structure


def uniform_indices(total: int, keep: int, seed: int) -> tuple[list[int], float]:
    """Pick ``keep`` of ``total`` positions at a constant stride.

    The phase is drawn from the seed so different seeds give different but
    equally uniform subsets; ``keep >= total`` returns every position.
    """
    if total <= 0:
        return [], 0.0
    if keep >= total:
        return list(range(total)), 0.0
    if keep <= 0:
        return [], 0.0
    phase = random.Random(int(seed)).random()
    stride = total / keep
    chosen: list[int] = []
    for index in range(keep):
        position = int((index + phase) * stride)
        chosen.append(min(position, total - 1))
    # A stride of at least one keeps positions distinct; guard anyway so the
    # count contract holds even under floating-point rounding at the end.
    unique = sorted(set(chosen))
    if len(unique) != keep:
        raise AssertionError(f"uniform stride produced {len(unique)} of {keep} positions")
    return unique, phase


def _gap_stats(timestamps_ns: list[int]) -> dict[str, float | None]:
    if len(timestamps_ns) < 2:
        return {"median_s": None, "p95_s": None, "max_s": None}
    gaps = sorted(
        (b - a) / 1e9 for a, b in zip(timestamps_ns[:-1], timestamps_ns[1:])
    )
    def pick(fraction: float) -> float:
        index = min(len(gaps) - 1, int(round(fraction * (len(gaps) - 1))))
        return float(gaps[index])
    return {"median_s": pick(0.5), "p95_s": pick(0.95), "max_s": float(gaps[-1])}


def subset_manifest(
    manifest: dict[str, Any], *, count: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(subset_manifest, report)``; both are plain JSON-ready dicts."""
    if count <= 0:
        raise ValueError("count must be positive")
    parent_sha = verify_dataset_manifest(manifest)
    units, structure = _units(manifest)
    total_images = sum(len(unit["image_ids"]) for unit in units)
    if total_images == 0:
        raise ValueError("dataset manifest contains no posed images")
    # Units carry a near-constant number of images (2 per rig frame), so the
    # unit budget is the image budget scaled by the mean unit size.
    keep_units = max(1, int(round(count * len(units) / total_images)))
    keep_units = min(keep_units, len(units))
    kept_positions, phase = uniform_indices(len(units), keep_units, seed)
    kept_set = set(kept_positions)
    kept_units = [units[index] for index in kept_positions]
    dropped_units = [unit for index, unit in enumerate(units) if index not in kept_set]
    kept_ids = {image_id for unit in kept_units for image_id in unit["image_ids"]}

    subset = dict(manifest)
    subset["images"] = [
        image for image in manifest.get("images", []) if str(image["image_id"]) in kept_ids
    ]
    kept_unit_ids = {unit["unit_id"] for unit in kept_units if unit["kind"] == "rig_frame"}
    subset["rig_frames"] = [
        frame
        for frame in manifest.get("rig_frames") or []
        if str(frame["rig_frame_id"]) in kept_unit_ids
    ]
    subset["splits"] = {
        name: [value for value in ids if value in kept_ids]
        for name, ids in (manifest.get("splits") or {}).items()
    }
    diagnostics = dict(manifest.get("rig_diagnostics") or {})
    if "pair_count" in diagnostics:
        diagnostics["parent_pair_count"] = diagnostics["pair_count"]
        diagnostics["pair_count"] = len(subset["rig_frames"])
    subset["rig_diagnostics"] = diagnostics
    source_hashes = dict(manifest.get("source_hashes") or {})
    source_hashes[PARENT_HASH_KEY] = parent_sha
    subset["source_hashes"] = source_hashes
    warning = (
        f"subset_dataset:{ALGORITHM_VERSION}:seed={seed}:"
        f"{len(kept_units)}/{len(units)} {structure} units,"
        f"{len(kept_ids)}/{total_images} images"
    )
    subset["warnings"] = list(manifest.get("warnings") or []) + [warning]
    subset = _sign(subset)

    per_camera_total: dict[str, int] = {}
    per_camera_kept: dict[str, int] = {}
    for image in manifest.get("images", []):
        camera = str(image["camera_id"])
        per_camera_total[camera] = per_camera_total.get(camera, 0) + 1
        if str(image["image_id"]) in kept_ids:
            per_camera_kept[camera] = per_camera_kept.get(camera, 0) + 1
    all_stamps = [unit["timestamp_ns"] for unit in units]
    kept_stamps = [unit["timestamp_ns"] for unit in kept_units]
    report = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "parent_manifest_sha256": parent_sha,
        "subset_manifest_sha256": subset["manifest_sha256"],
        "parent_recording_id": manifest.get("recording_id"),
        "selection": {
            "mode": "uniform_temporal",
            "unit_structure": structure,
            "requested_image_count": count,
            "seed": seed,
            "phase": phase,
            "stride_units": len(units) / keep_units,
            "unit_count": len(units),
            "kept_unit_count": len(kept_units),
            "dropped_unit_count": len(dropped_units),
        },
        "images": {
            "parent_posed": total_images,
            "kept": len(kept_ids),
            "dropped": total_images - len(kept_ids),
            "per_camera_parent": per_camera_total,
            "per_camera_kept": per_camera_kept,
            "unposed_carried_unchanged": list(manifest.get("unposed_images") or []),
        },
        "time_coverage": {
            "parent_span_s": (all_stamps[-1] - all_stamps[0]) / 1e9 if all_stamps else 0.0,
            "kept_span_s": (kept_stamps[-1] - kept_stamps[0]) / 1e9 if kept_stamps else 0.0,
            "parent_gap": _gap_stats(all_stamps),
            "kept_gap": _gap_stats(kept_stamps),
        },
        "rig_frames": {
            "parent": len(manifest.get("rig_frames") or []),
            "kept": len(subset["rig_frames"]),
            "kept_whole": True,
            "diagnostic_statistics": "copied from parent; only pair_count re-counted",
        },
        "drop_reason": "uniform_temporal_not_selected",
        "kept_units": [
            {
                "unit_id": unit["unit_id"],
                "kind": unit["kind"],
                "timestamp_ns": unit["timestamp_ns"],
                "position": position,
                "image_ids": unit["image_ids"],
            }
            for position, unit in zip(kept_positions, kept_units)
        ],
        "dropped_unit_ids": [unit["unit_id"] for unit in dropped_units],
    }
    return subset, report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", required=True, type=Path, help="parent dataset_manifest.json")
    parser.add_argument("--count", required=True, type=int, help="target number of posed images")
    parser.add_argument("--seed", type=int, default=0, help="chooses the phase inside the first stride")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=f"directory receiving {MANIFEST_NAME} and {REPORT_NAME}",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_out = args.output_dir / MANIFEST_NAME
    report_out = args.output_dir / REPORT_NAME
    if not args.force and (manifest_out.exists() or report_out.exists()):
        parser.error(f"{args.output_dir} already holds outputs; pass --force to overwrite")
    subset, report = subset_manifest(manifest, count=args.count, seed=args.seed)
    report["parent_manifest_path"] = str(args.manifest.resolve())
    _write_json(manifest_out, subset)
    _write_json(report_out, report)
    verify_dataset_manifest(json.loads(manifest_out.read_text(encoding="utf-8")))
    selection = report["selection"]
    print(
        f"subset: {report['images']['kept']}/{report['images']['parent_posed']} images, "
        f"{selection['kept_unit_count']}/{selection['unit_count']} {selection['unit_structure']} units, "
        f"seed={args.seed} sha256={subset['manifest_sha256']} -> {manifest_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
