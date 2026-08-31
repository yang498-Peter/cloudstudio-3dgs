#!/usr/bin/env python3
"""Merge five trained snow Tiles using a signed core-only or halo policy."""

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
            log_gains = (payload.get("auxiliary_params") or {}).get(
                "exposure_log_gains"
            )
            if log_gains is None:
                raise ValueError(
                    f"Tile_{tile_id} carries no exposure gains to harmonize"
                )
            tile_gain = float(torch.exp(log_gains.detach().float()).median())
            # rgb = sh0 * C0 + 0.5, and the target is rgb * gain, so the DC
            # band carries the scale and the shifted grey point together.
            params = dict(params)
            sh0 = params["sh0"].detach().float()
            params["sh0"] = (
                sh0 * tile_gain + (tile_gain - 1.0) * 0.5 / _SH_C0
            ).to(params["sh0"].dtype)
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
    total = int(combined["means"].shape[0])
    source_total = int(sum(record["input_gaussian_count"] for record in records))
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
        "discarded_by_merge_policy_count": source_total - total,
        "exposure_harmonized": bool(args.harmonize_exposure),
        "shared_boundary_rule": (
            "minimum_tile_id"
            if args.merge_policy == "core_owner_only"
            else "retain_every_tile_training_and_export_halo_without_deduplication"
        ),
        "records": records,
    }
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
    print(
        f"merged {total}/{source_total} Gaussians with {args.merge_policy} "
        f"-> {args.output_checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
