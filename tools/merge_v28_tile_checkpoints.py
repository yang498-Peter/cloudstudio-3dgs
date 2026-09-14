#!/usr/bin/env python3
"""Merge trained snow Tiles using a signed core-only or halo policy.

``--fill-checkpoint`` optionally adds a fill layer: a coarse whole-scene
prior that carries only what no delivery Tile claimed. Without it the merged
tensors are bit-identical to before and the report keeps its previous key
set (the .pt container itself is not byte-reproducible - torch.save writes a
different archive for the same payload on every call).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest
from cloudstudio_3dgs.training.tile_ownership import assign_core_owners


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_tile_checkpoint(value: str) -> tuple[int, Path]:
    tile, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("tile checkpoint must use TILE_ID=PATH")
    try:
        tile_id = int(tile)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Tile id must be an integer") from exc
    return tile_id, Path(path)


_SH_C0 = 0.28209479177387814


def inside_any_box_mask(
    means: np.ndarray, boxes: np.ndarray, *, tolerance_m: float = 0.0
) -> np.ndarray:
    """Rows whose centre lies inside at least one of the delivery Tile boxes."""

    xyz = np.asarray(means, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("means must be an [N, 3] array")
    bounds = np.asarray(boxes, dtype=np.float64)
    if bounds.ndim != 3 or bounds.shape[1:] != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("boxes must be a finite [T, 2, 3] array")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("every box must have positive extent")
    inside = np.zeros(len(xyz), dtype=bool)
    for box in bounds:
        inside |= np.all(
            (xyz >= box[0] - float(tolerance_m)) & (xyz <= box[1] + float(tolerance_m)),
            axis=1,
        )
    return inside


def unclaimed_voxel_mask(
    fill_means: np.ndarray,
    tile_means: np.ndarray,
    *,
    voxel_m: float,
    clearance_voxels: int = 1,
) -> np.ndarray:
    """Fill rows no retained delivery gaussian sits near, at voxel granularity.

    The Tile core boxes are a checked gap-free partition of the scene box and
    every export box contains its core, so "inside no Tile box" is the empty
    set on a real plan and the box rule alone can never carry a fill layer.
    What the delivery lacks is content, not space: a Tile told not to
    supervise the pixels it does not own grows nothing there, so the merge is
    transparent inside a box that formally owns it. This is the finer claim
    test - a fill row survives only when its own voxel, and every voxel
    within ``clearance_voxels`` of it in Chebyshev distance, holds no
    retained delivery gaussian.
    """

    voxel = float(voxel_m)
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("voxel_m must be a positive finite length")
    clearance = int(clearance_voxels)
    if clearance < 0:
        raise ValueError("clearance_voxels must be non-negative")
    fill = np.asarray(fill_means, dtype=np.float64)
    tile = np.asarray(tile_means, dtype=np.float64)
    if fill.ndim != 2 or fill.shape[1] != 3 or tile.ndim != 2 or tile.shape[1] != 3:
        raise ValueError("means must be [N, 3] arrays")
    if len(fill) == 0:
        return np.zeros(0, dtype=bool)
    if len(tile) == 0:
        return np.ones(len(fill), dtype=bool)
    origin = np.minimum(fill.min(axis=0), tile.min(axis=0))
    fill_index = np.floor((fill - origin) / voxel).astype(np.int64)
    tile_index = np.floor((tile - origin) / voxel).astype(np.int64)
    low = np.minimum(fill_index.min(axis=0), tile_index.min(axis=0)) - clearance
    high = np.maximum(fill_index.max(axis=0), tile_index.max(axis=0)) + clearance
    span = (high - low + 1).astype(np.int64)
    if float(span[0]) * float(span[1]) * float(span[2]) > float(2**62):
        raise ValueError("voxel_m is too small for this scene's extent")
    multiplier = np.array([1, span[0], span[0] * span[1]], dtype=np.int64)
    fill_index -= low
    tile_index -= low
    # Every index sits at least ``clearance`` voxels inside the padded grid,
    # so a neighbour shift stays on its own axis and the flat key arithmetic
    # cannot wrap into the next row.
    claimed_keys = np.unique(tile_index @ multiplier)
    offsets = np.arange(-clearance, clearance + 1, dtype=np.int64)
    shifts = np.unique(
        [
            int(dx * multiplier[0] + dy * multiplier[1] + dz * multiplier[2])
            for dx in offsets
            for dy in offsets
            for dz in offsets
        ]
    )
    keys = np.unique(np.concatenate([claimed_keys + shift for shift in shifts]))
    query = fill_index @ multiplier
    position = np.clip(np.searchsorted(keys, query), 0, len(keys) - 1)
    return keys[position] != query


def _median_exposure_gain(payload: dict, *, what: str) -> float:
    log_gains = (payload.get("auxiliary_params") or {}).get("exposure_log_gains")
    if log_gains is None:
        raise ValueError(f"{what} carries no exposure gains to harmonize")
    return float(torch.exp(log_gains.detach().float()).median())


def _bake_exposure_gain(params: dict, gain: float) -> dict:
    # rgb = sh0 * C0 + 0.5, and the target is rgb * gain, so the DC
    # band carries the scale and the shifted grey point together.
    baked = dict(params)
    sh0 = params["sh0"].detach().float()
    baked["sh0"] = (sh0 * gain + (gain - 1.0) * 0.5 / _SH_C0).to(params["sh0"].dtype)
    return baked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-inputs", required=True, type=Path)
    parser.add_argument("--tile-inputs-root", required=True, type=Path)
    parser.add_argument(
        "--tile-checkpoint",
        required=True,
        action="append",
        type=_parse_tile_checkpoint,
    )
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument(
        "--merge-policy",
        choices=("core_owner_only", "retain_full_halo"),
        default="core_owner_only",
        help=(
            "core_owner_only keeps one hard owner per point; retain_full_halo "
            "keeps every trained Tile row, matching the observed MipMap export policy"
        ),
    )
    parser.add_argument("--tolerance-m", type=float, default=1e-5)
    parser.add_argument(
        "--fill-checkpoint",
        action="append",
        type=Path,
        default=None,
        help=(
            "repeatable coarse prior carrying the pixels no delivery Tile "
            "claims; rows inside any Tile training_and_export_box are dropped. "
            "Omitted (the default) the merge is the merge it always was"
        ),
    )
    parser.add_argument(
        "--fill-occupancy-voxel-m",
        type=float,
        default=None,
        help=(
            "keep a fill row inside a Tile box when that Tile grew nothing "
            "there: no retained delivery gaussian in its voxel or within "
            "--fill-occupancy-clearance-voxels of it"
        ),
    )
    parser.add_argument("--fill-occupancy-clearance-voxels", type=int, default=1)
    parser.add_argument(
        "--fill-min-opacity",
        type=float,
        default=0.0,
        help=(
            "drop fill rows whose sigmoid(opacity) is below this floor; the "
            "default 0.0 is the floor this merge applies to Tile rows"
        ),
    )
    parser.add_argument(
        "--harmonize-exposure",
        action="store_true",
        help=(
            "bake each tile's own learned exposure into its colours before "
            "concatenating, so every tile lands in one photometric frame"
        ),
    )
    args = parser.parse_args()

    manifest = json.loads(args.tile_inputs.read_text(encoding="utf-8"))
    manifest_sha = verify_tile_inputs_manifest(
        manifest, root=args.tile_inputs_root, verify_artifacts=True
    )
    tiles = sorted(manifest["tiles"], key=lambda item: int(item["tile_id"]))
    checkpoints = dict(args.tile_checkpoint)
    expected_ids = {int(tile["tile_id"]) for tile in tiles}
    if set(checkpoints) != expected_ids:
        raise ValueError(
            f"checkpoints must cover exactly {sorted(expected_ids)}, got {sorted(checkpoints)}"
        )

    boxes = np.asarray([tile["core_box"] for tile in tiles], dtype=np.float64)
    global_min = boxes[:, 0].min(axis=0)
    global_max = boxes[:, 1].max(axis=0)
    merged: dict[str, list[torch.Tensor]] = {}
    parameter_keys: tuple[str, ...] | None = None
    coordinate_sha: str | None = None
    first_identity: dict | None = None
    records: list[dict] = []
    completed_steps: list[int] = []

    for tile in tiles:
        tile_id = int(tile["tile_id"])
        checkpoint_path = checkpoints[tile_id]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Tile_{tile_id} checkpoint is missing: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        params = payload.get("params") or payload.get("splats")
        if not isinstance(params, dict) or "means" not in params:
            raise ValueError(f"Tile_{tile_id} checkpoint has no params")
        tile_gain = None
        if args.harmonize_exposure:
            # Every tile learns its OWN gain for the same shared boundary
            # photo, and a tile's colours are only correct once its own gain
            # is applied. Concatenating without them leaves each tile in a
            # different photometric frame, which reads as rectangular
            # brightness patches with hard axis-aligned edges - the most
            # visible seam artifact available. Folding a tile's own gain into
            # its DC colour returns every tile to the photograph's frame.
            tile_gain = _median_exposure_gain(payload, what=f"Tile_{tile_id}")
            params = _bake_exposure_gain(params, tile_gain)
        keys = tuple(sorted(params))
        if parameter_keys is None:
            parameter_keys = keys
        elif keys != parameter_keys:
            raise ValueError("Tile checkpoints have different Gaussian parameter layouts")
        identity = payload.get("identity", {})
        current_coordinate_sha = str(identity.get("coordinate_transform_sha256", ""))
        if coordinate_sha is None:
            coordinate_sha = current_coordinate_sha
            first_identity = copy.deepcopy(identity)
        elif current_coordinate_sha != coordinate_sha:
            raise ValueError("Tile checkpoints use different coordinate transforms")

        means = params["means"].detach().cpu().numpy().astype(np.float64, copy=False)
        inside_global = np.all(
            (means >= global_min - args.tolerance_m)
            & (means <= global_max + args.tolerance_m),
            axis=1,
        )
        owners = np.full(len(means), -1, dtype=np.int64)
        owners[inside_global] = assign_core_owners(
            means[inside_global], tiles, tolerance_m=args.tolerance_m
        )
        core_keep = owners == tile_id
        keep = (
            core_keep
            if args.merge_policy == "core_owner_only"
            else np.ones(len(means), dtype=bool)
        )
        keep_tensor = torch.from_numpy(keep)
        for key in parameter_keys:
            value = params[key].detach().cpu()
            if value.shape[0] != len(means):
                raise ValueError(f"Tile_{tile_id} parameter {key} has another row count")
            merged.setdefault(key, []).append(value[keep_tensor])
        opacity = torch.sigmoid(params["opacities"].detach().cpu().reshape(-1))
        records.append(
            {
                "tile_id": tile_id,
                "checkpoint": checkpoint_path.resolve().as_posix(),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "completed_steps": int(payload.get("step", -1)),
                "exposure_gain_applied": tile_gain,
                "input_gaussian_count": int(len(means)),
                "core_gaussian_count": int(np.count_nonzero(core_keep)),
                "retained_gaussian_count": int(np.count_nonzero(keep)),
                "discarded_by_merge_policy_count": int(np.count_nonzero(~keep)),
                "retained_dead_opacity_below_0_005_count": int(
                    torch.count_nonzero(opacity[keep_tensor] < 0.005).item()
                ),
            }
        )
        completed_steps.append(int(payload.get("step", -1)))
        del payload, params, means, owners, core_keep, keep, keep_tensor, opacity

    assert parameter_keys is not None and first_identity is not None
    combined = {key: torch.cat(merged[key], dim=0) for key in parameter_keys}
    merged.clear()
    tile_total = int(combined["means"].shape[0])
    source_total = int(sum(record["input_gaussian_count"] for record in records))

    # The fill layer: a coarse whole-scene prior carrying only what the
    # delivery Tiles never claimed. Ownership masking told each Tile not to
    # supervise the pixels it does not own and a per-view stand-in backdrop
    # covered them during training; the stand-in does not ship, so the merge
    # is transparent there and the evaluation background shows through. The
    # fill rows are the same photometric frame as the Tiles and never sit
    # where a Tile already put a gaussian.
    fill_records: list[dict] = []
    fill_total = 0
    fill_paths = list(args.fill_checkpoint or [])
    if fill_paths:
        export_boxes = np.asarray(
            [tile["training_and_export_box"] for tile in tiles], dtype=np.float64
        )
        tile_means = combined["means"].detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        for index, fill_path in enumerate(fill_paths):
            if not fill_path.is_file():
                raise FileNotFoundError(f"fill checkpoint is missing: {fill_path}")
            payload = torch.load(fill_path, map_location="cpu", weights_only=False)
            params = payload.get("params") or payload.get("splats")
            if not isinstance(params, dict) or "means" not in params:
                raise ValueError(f"fill checkpoint has no params: {fill_path}")
            if tuple(sorted(params)) != parameter_keys:
                raise ValueError(
                    f"fill checkpoint has another Gaussian parameter layout: {fill_path}"
                )
            fill_identity = payload.get("identity", {})
            if str(fill_identity.get("coordinate_transform_sha256", "")) != coordinate_sha:
                raise ValueError(
                    f"fill checkpoint uses another coordinate transform: {fill_path}"
                )
            fill_gain = None
            if args.harmonize_exposure:
                fill_gain = _median_exposure_gain(payload, what=f"fill {fill_path}")
                params = _bake_exposure_gain(params, fill_gain)
            means = params["means"].detach().cpu().numpy().astype(np.float64, copy=False)
            claimed = inside_any_box_mask(
                means, export_boxes, tolerance_m=args.tolerance_m
            )
            rejected_box = int(np.count_nonzero(claimed))
            if args.fill_occupancy_voxel_m is not None and rejected_box:
                free = unclaimed_voxel_mask(
                    means[claimed],
                    tile_means,
                    voxel_m=args.fill_occupancy_voxel_m,
                    clearance_voxels=args.fill_occupancy_clearance_voxels,
                )
                claimed_indices = np.flatnonzero(claimed)
                claimed[claimed_indices[free]] = False
            keep = ~claimed
            rejected_occupied = int(np.count_nonzero(claimed))
            opacity = torch.sigmoid(params["opacities"].detach().cpu().reshape(-1))
            rejected_opacity = 0
            if args.fill_min_opacity > 0.0:
                dead = (opacity < float(args.fill_min_opacity)).numpy()
                rejected_opacity = int(np.count_nonzero(dead & keep))
                keep &= ~dead
            keep_tensor = torch.from_numpy(keep)
            kept = int(np.count_nonzero(keep))
            for key in parameter_keys:
                value = params[key].detach().cpu()
                if value.shape[0] != len(means):
                    raise ValueError(
                        f"fill parameter {key} has another row count: {fill_path}"
                    )
                if value.shape[1:] != combined[key].shape[1:]:
                    raise ValueError(
                        f"fill parameter {key} has another band layout: {fill_path}"
                    )
                merged.setdefault(key, []).append(value[keep_tensor])
            retained_means = means[keep]
            fill_records.append(
                {
                    "checkpoint": fill_path.resolve().as_posix(),
                    "checkpoint_sha256": _sha256(fill_path),
                    "completed_steps": int(payload.get("step", -1)),
                    "exposure_gain_applied": fill_gain,
                    "input_gaussian_count": int(len(means)),
                    "rejected_inside_tile_box_count": rejected_box,
                    "rejected_as_delivery_occupied_count": rejected_occupied,
                    "rejected_by_opacity_floor_count": rejected_opacity,
                    "retained_gaussian_count": kept,
                    "retained_bounds": (
                        [retained_means.min(axis=0).tolist(), retained_means.max(axis=0).tolist()]
                        if kept
                        else None
                    ),
                }
            )
            fill_total += kept
            if (
                args.fill_occupancy_voxel_m is not None
                and kept
                and index + 1 < len(fill_paths)
            ):
                # A later fill source must not stack on top of an earlier
                # one either; the rows already accepted now claim their space.
                tile_means = np.concatenate([tile_means, retained_means], axis=0)
            del payload, params, means, claimed, keep, keep_tensor, opacity, retained_means
        if fill_total:
            for key in parameter_keys:
                combined[key] = torch.cat([combined[key]] + merged[key], dim=0)
        merged.clear()
        del tile_means

    total = int(combined["means"].shape[0])
    report = {
        "schema_version": 1,
        "kind": "snow_v28_tile_checkpoint_merge_v2",
        "status": "PASS",
        "merge_policy": args.merge_policy,
        "tile_inputs_manifest_sha256": manifest_sha,
        "coordinate_transform_sha256": coordinate_sha,
        "tile_count": len(tiles),
        "source_gaussian_count": source_total,
        "merged_gaussian_count": total,
        "discarded_by_merge_policy_count": source_total - tile_total,
        "exposure_harmonized": bool(args.harmonize_exposure),
        "shared_boundary_rule": (
            "minimum_tile_id"
            if args.merge_policy == "core_owner_only"
            else "retain_every_tile_training_and_export_halo_without_deduplication"
        ),
        "records": records,
    }
    if fill_paths:
        report["tile_gaussian_count"] = tile_total
        report["fill_gaussian_count"] = fill_total
        report["fill_source_count"] = len(fill_records)
        report["fill_exclusion"] = {
            "box_kind": "training_and_export_box",
            "tolerance_m": float(args.tolerance_m),
            "occupancy_voxel_m": (
                None
                if args.fill_occupancy_voxel_m is None
                else float(args.fill_occupancy_voxel_m)
            ),
            "occupancy_clearance_voxels": (
                None
                if args.fill_occupancy_voxel_m is None
                else int(args.fill_occupancy_clearance_voxels)
            ),
            "min_opacity": float(args.fill_min_opacity),
            "rule": (
                "reject_inside_any_tile_training_and_export_box"
                if args.fill_occupancy_voxel_m is None
                else "reject_inside_any_tile_training_and_export_box_unless_voxel_unoccupied_by_merged_tile_gaussian"
            ),
        }
        report["fill_sources"] = fill_records
    report["merge_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    identity = copy.deepcopy(first_identity)
    identity.update(
        {
            "kind": "snow_v28_merged_tile_checkpoint_v2",
            "merge_policy": args.merge_policy,
            "tile_inputs_manifest_sha256": manifest_sha,
            "merge_report_sha256": report["merge_report_sha256"],
            "source_tile_checkpoint_sha256": {
                str(record["tile_id"]): record["checkpoint_sha256"]
                for record in records
            },
        }
    )
    if fill_paths:
        identity["fill_checkpoint_sha256"] = [
            record["checkpoint_sha256"] for record in fill_records
        ]
    checkpoint = {
        "schema_version": 1,
        "step": max(completed_steps),
        "params": combined,
        "identity": identity,
        "merge": report,
    }
    _save(args.output_checkpoint, checkpoint)
    report["output_checkpoint"] = args.output_checkpoint.resolve().as_posix()
    report["output_checkpoint_sha256"] = _sha256(args.output_checkpoint)
    # Bind output metadata in a second outer signature without changing the
    # embedded merge identity already stored in the immutable checkpoint.
    report["delivery_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    _write_json(args.output_report, report)
    fill_note = f" plus {fill_total} fill Gaussians" if fill_paths else ""
    print(
        f"merged {tile_total}/{source_total} Gaussians with {args.merge_policy}"
        f"{fill_note} -> {args.output_checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
