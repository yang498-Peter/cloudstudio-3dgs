#!/usr/bin/env python3
"""Derive a face-excluded *variant* of a production Tile inputs manifest and
the trainer config that trains on it with everything else untouched.

Where ``tools/build_diagnostic_set.py`` restricts a Tile to a handful of
selected parent images (a region diagnostic), this tool keeps **every** parent
image of the Tile and only drops the given face ids (e.g. ``pitch_up_56``)
from the Tile's ``views``.  A parent image that would be left with zero Tile
faces is dropped from the parent set and reported; images with >= 1 remaining
face stay.  The derived manifests are produced by the same
``derive_tile_inputs_manifest`` / ``derive_tile_geometry_manifest`` functions
the diagnostic presets use, so they are signed with the repository's canonical
signing, the geometry manifest is bound to the derived inputs sha, and the
Tile's initialization PLY / geometry npz are referenced verbatim (no copies):
``TrainerConfig.validate``'s binding checks pass with ``tile_inputs_root``
left at the production root.

Written into ``--out-dir``::

    tile_inputs_manifest.json    one-Tile, signed; ``diagnostic`` carries the
                                 exclusion (face ids, counts before/after,
                                 dropped parents) and the variant label
    tile_geometry_manifest.json  one-Tile, bound to the derived inputs sha,
                                 geometry npz path relative to --out-dir
    variant.json                 counts, shas, ROI-id presence, schedule
                                 pre-flight (see below)

With ``--base-config`` the production arm config is copied to
``--out-config`` with only ``run_id``, ``output_dir``,
``tile_inputs_manifest``, ``initialization_geometry_manifest`` and a
``lineage`` block changed (asserted); ``--copy-config-to`` drops a
byte-identical copy into a second directory (the repo's run_configs).

Schedule pre-flight.  The production Tile arms run without a
``schedule_contract``; for them ``trainer.train`` requires
``max_steps == 20 x training view count`` (fisher_yates_without_replacement
_per_epoch + adaptive_growth), and ``TrainerConfig.validate`` requires
``default_strategy.prune_switch_step == max_steps // 2`` for the exact MipMap
lifecycle.  Dropping faces changes the view count, so a config that keeps the
base schedule verbatim passes ``validate()`` but is rejected by ``train()``
at start-up.  This tool never rewrites the schedule: it records the pre-flight
outcome and the parity-implied values in ``variant.json`` and
``lineage.schedule`` and prints a warning, leaving the decision (re-derive
``max_steps``/``prune_switch_step`` for the new view count, or declare a
research schedule contract) to the operator.

Example::

    python tools/derive_tile_inputs_variant.py \
        --base-config C:/Peter/3dgs-runs/house0305_sop/tile1_R1d_20k.json \
        --exclude-faces pitch_up_56 --label F3_nopitchup \
        --out-dir C:/Peter/3dgs-runs/house0305_sop/tile_inputs_v9_F3 \
        --run-id house0305-t1-F3-nopitchup \
        --output-dir C:/Peter/3dgs-runs/house0305_sop/tile1_F3_nopitchup_20k \
        --out-config C:/Peter/3dgs-runs/house0305_sop/tile1_F3_nopitchup_20k.json \
        --copy-config-to run_configs/house0305_tiles/v9 \
        --roi-ids C:/Peter/3dgs-runs/house0305_sop/diag_v2/indoor_door_leaf_Tile_1/DIAG_40/roi_compare_ids_f3.json \
        --validate
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_diagnostic_set import (  # noqa: E402
    _relative_posix,
    derive_tile_geometry_manifest,
    derive_tile_inputs_manifest,
    face_counts,
    parse_face_list,
    split_sample_id,
)
from cloudstudio_3dgs.training.mipmap_tile_geometry import verify_tile_geometry_manifest  # noqa: E402
from cloudstudio_3dgs.training.schedule_audit import MIPMAP_EPOCHS_PER_RUN, resolved_schedule  # noqa: E402
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest  # noqa: E402

GENERATOR = "tools/derive_tile_inputs_variant.py"
VARIANT_KIND = "production_tile_face_exclusion_variant_v1"
# The only top-level config keys a variant may change; anything else is a
# recipe/schedule change and is refused so the arm stays a pure view-set arm.
CONFIG_ALLOWED_DIFF = frozenset(
    {"run_id", "output_dir", "tile_inputs_manifest", "initialization_geometry_manifest", "lineage"}
)
# Same list make_diagnostic_arm_config.py records: the Tile artefacts the
# variant references verbatim through the production root.
VERBATIM_FIELDS = [
    "tile_inputs_root",
    "initialization_ply",
    "initialization_geometry",
    "background_image_manifest",
    "background_image_root",
    "face_cache_manifest",
    "mipmap_tile_id",
]
_MISSING = object()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: Mapping[str, Any], *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------------
# pure: manifests
# ----------------------------------------------------------------------------


def plan_face_exclusion(tile: Mapping[str, Any], exclude_face_ids: Sequence[str]) -> dict[str, Any]:
    """Which parent images of ``tile`` keep >= 1 face once ``exclude_face_ids``
    are dropped (first-seen order), which are emptied, and the full-Tile
    counts before anything is removed."""
    excluded = {str(f) for f in exclude_face_ids}
    faces_by_image: dict[str, list[str]] = {}
    for view in tile["views"]:
        image_id, face_id = split_sample_id(view["sample_id"])
        faces_by_image.setdefault(image_id, []).append(face_id)
    kept = [image for image, faces in faces_by_image.items() if any(f not in excluded for f in faces)]
    kept_set = set(kept)
    dropped = [image for image in faces_by_image if image not in kept_set]
    return {
        "parent_image_ids": kept,
        "parent_images_dropped": dropped,
        "parent_image_count_before": len(faces_by_image),
        "tile_view_count_before": len(tile["views"]),
        "face_counts_before": face_counts(tile["views"]),
    }


def derive_variant_manifests(
    base_inputs: Mapping[str, Any],
    base_geometry: Mapping[str, Any],
    *,
    tile_name: str,
    exclude_face_ids: Sequence[str],
    geometry_path: str,
    label: str,
    note: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Pure: production inputs + geometry -> (derived inputs, derived geometry
    bound to them, summary).  ``geometry_path`` is where the derived geometry
    manifest will find the Tile's npz, relative to its own directory."""
    excluded = [str(f) for f in exclude_face_ids]
    if not excluded:
        raise ValueError("a variant needs at least one face id to exclude")
    if not label:
        raise ValueError("a variant needs a label")
    base_inputs_sha = verify_tile_inputs_manifest(dict(base_inputs))
    verify_tile_geometry_manifest(dict(base_geometry))
    if base_geometry.get("tile_inputs_manifest_sha256") != base_inputs_sha:
        raise ValueError("base Tile geometry manifest is bound to different Tile inputs than the manifest given")
    matches = [t for t in base_inputs["tiles"] if t.get("name") == tile_name]
    if len(matches) != 1:
        raise ValueError(f"base Tile inputs do not contain a unique Tile named {tile_name!r}")
    tile = matches[0]
    plan = plan_face_exclusion(tile, excluded)
    if not plan["parent_image_ids"]:
        raise ValueError(f"excluding faces {excluded} leaves {tile_name} with no parent image")
    provenance: dict[str, Any] = {
        "kind": VARIANT_KIND,
        "variant": str(label),
        "variant_generator": GENERATOR,
        "variant_rule": (
            "every parent image of the Tile kept; the listed face ids dropped from the Tile views; "
            "a parent left with zero faces is dropped and listed"
        ),
        "parent_image_count_before_exclusion": plan["parent_image_count_before"],
        "parent_images_dropped_by_exclusion": list(plan["parent_images_dropped"]),
        "tile_view_count_before_exclusion": plan["tile_view_count_before"],
        "tile_face_counts_before_exclusion": plan["face_counts_before"],
    }
    if note:
        provenance["note"] = str(note)
    inputs = derive_tile_inputs_manifest(
        base_inputs,
        tile_name=tile_name,
        image_ids=plan["parent_image_ids"],
        provenance=provenance,
        exclude_face_ids=excluded,
    )
    geometry = derive_tile_geometry_manifest(
        base_geometry,
        tile_id=int(tile["tile_id"]),
        geometry_path=str(geometry_path),
        tile_inputs_manifest_sha256=inputs["tile_inputs_manifest_sha256"],
        provenance=provenance,
    )
    views = inputs["tiles"][0]["views"]
    summary = {
        "kind": VARIANT_KIND,
        "generator": GENERATOR,
        "variant": str(label),
        "tile": tile_name,
        "tile_id": int(tile["tile_id"]),
        "excluded_face_ids": excluded,
        "view_count_before": plan["tile_view_count_before"],
        "view_count_after": len(views),
        "views_removed": plan["tile_view_count_before"] - len(views),
        "face_counts_before": plan["face_counts_before"],
        "face_counts_after": face_counts(views),
        "parent_image_count_before": plan["parent_image_count_before"],
        "parent_image_count_after": len(plan["parent_image_ids"]),
        "parent_images_dropped": list(plan["parent_images_dropped"]),
        "tile_inputs_manifest_sha256": inputs["tile_inputs_manifest_sha256"],
        "tile_geometry_manifest_sha256": geometry["tile_geometry_manifest_sha256"],
        "derived_from_tile_inputs_manifest_sha256": base_inputs_sha,
        "derived_from_tile_geometry_manifest_sha256": base_geometry["tile_geometry_manifest_sha256"],
        "initialization_point_count": int(tile["initialization"]["point_count"]),
    }
    if note:
        summary["note"] = str(note)
    return inputs, geometry, summary


def roi_presence(views: Sequence[Mapping[str, Any]], sample_ids: Sequence[str]) -> dict[str, Any]:
    """Which of ``sample_ids`` are Tile views of the derived manifest."""
    present = {str(v["sample_id"]) for v in views}
    wanted = [str(s) for s in sample_ids]
    missing = [s for s in wanted if s not in present]
    return {"count": len(wanted), "present": len(wanted) - len(missing), "missing": missing, "all_present": not missing}


# ----------------------------------------------------------------------------
# pure: config
# ----------------------------------------------------------------------------


def schedule_preflight(config: Mapping[str, Any], view_count: int) -> dict[str, Any]:
    """The trainer's start-up rule for this config at ``view_count`` training
    views, resolved through ``schedule_audit.resolved_schedule`` (the same
    check the audit CLI reports).  Never changes the config."""
    schedule = resolved_schedule(dict(config), int(view_count))
    names = {"max_steps_is_20_view_epochs", "max_steps_is_contract_horizon"}
    checks = [c for c in schedule["consistency_checks"] if c["name"] in names]
    check = checks[0] if checks else None
    max_steps = int(config["max_steps"])
    parity_max_steps = MIPMAP_EPOCHS_PER_RUN * int(view_count)
    exact_lifecycle = bool((config.get("default_strategy") or {}).get("exact_mipmap_lifecycle"))
    stop = config.get("controlled_stop_after_steps")
    record: dict[str, Any] = {
        "policy": "schedule kept identical to the base config; not rewritten by the tool",
        "schedule_contract": config.get("schedule_contract"),
        "rule": (
            "trainer.train: max_steps == 20 x training view count for fisher_yates_without_replacement_per_epoch "
            "+ adaptive_growth without a schedule_contract; TrainerConfig.validate: prune_switch_step == max_steps // 2 "
            "for exact_mipmap_lifecycle"
        ),
        "check": check["name"] if check else None,
        "ok": bool(check["ok"]) if check else True,
        "max_steps": max_steps,
        "training_view_count": int(view_count),
        "configured_epochs": max_steps / int(view_count),
        "controlled_stop_after_steps": stop,
        "executed_epochs_at_stop": (int(stop) if stop is not None else max_steps) / int(view_count),
        "parity_max_steps": parity_max_steps,
        "parity_prune_switch_step": parity_max_steps // 2 if exact_lifecycle else None,
        "current_prune_switch_step": (config.get("default_strategy") or {}).get("prune_switch_step"),
    }
    if check is not None and not check["ok"]:
        record["trainer_error"] = "MipMap epoch-permutation sampling requires exactly 20 complete view epochs"
        record["resolution_options"] = [
            f"parity: max_steps {max_steps} -> {parity_max_steps} and default_strategy.prune_switch_step -> "
            f"{parity_max_steps // 2} (changes the means-LR decay denominator and the late-cull switch; controlled stop unchanged)",
            "contract: declare schedule_contract research_rescaled_horizon_v1 with max_steps as the horizon H "
            "(controlled stop may only restate H; linked fields validated as fractions of H)",
        ]
    return record


def top_level_diff(base: Mapping[str, Any], derived: Mapping[str, Any]) -> set[str]:
    return {k for k in set(base) | set(derived) if base.get(k, _MISSING) != derived.get(k, _MISSING)}


def derive_variant_config(
    base: Mapping[str, Any],
    *,
    run_id: str,
    output_dir: str,
    tile_inputs_manifest: str,
    tile_geometry_manifest: str,
    summary: Mapping[str, Any],
    base_config_path: str,
    base_config_sha256: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Pure: base arm config -> variant config.  Only ``run_id``,
    ``output_dir``, the two manifest paths and ``lineage`` may differ
    (asserted); the schedule is left verbatim and its pre-flight outcome at
    the new view count is recorded under ``lineage.schedule``."""
    if int(base.get("mipmap_tile_id", -1)) != int(summary["tile_id"]):
        raise ValueError(
            f"base config trains Tile {base.get('mipmap_tile_id')} but the variant manifests are for Tile {summary['tile_id']}"
        )
    config = copy.deepcopy(dict(base))
    rebound: dict[str, dict[str, Any]] = {}
    for key, value in (
        ("tile_inputs_manifest", str(tile_inputs_manifest)),
        ("initialization_geometry_manifest", str(tile_geometry_manifest)),
    ):
        rebound[key] = {"base": config.get(key), "variant": value}
        config[key] = value
    config["run_id"] = str(run_id)
    config["output_dir"] = str(output_dir)
    faces = ",".join(summary["excluded_face_ids"])
    lineage: dict[str, Any] = {
        "base": Path(base_config_path).stem,
        "base_run_id": base.get("run_id"),
        "base_config": str(base_config_path),
        "base_config_sha256": str(base_config_sha256),
        "generator": GENERATOR,
        "single_change": (
            f"view set only: every {faces} face of {summary['tile']} dropped from the Tile inputs "
            f"({summary['view_count_before']} -> {summary['view_count_after']} views, "
            f"{summary['parent_image_count_after']} of {summary['parent_image_count_before']} parent images kept); "
            "recipe, schedule, cap and every other field identical to the base"
        ),
        "data_variant": dict(summary),
        "rebound_fields": rebound,
        "verbatim_fields": list(VERBATIM_FIELDS),
        "schedule": schedule_preflight(config, int(summary["view_count_after"])),
    }
    if "lineage" in base:
        lineage["base_lineage"] = copy.deepcopy(base["lineage"])
    if note:
        lineage["note"] = str(note)
    config["lineage"] = lineage
    changed = top_level_diff(base, config)
    unexpected = changed - CONFIG_ALLOWED_DIFF
    if unexpected:
        raise AssertionError(f"variant config changes fields outside the allowed set: {sorted(unexpected)}")
    return config


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True, help="production arm config (tile_inputs_manifest, initialization_geometry_manifest, mipmap_tile_id are read from it)")
    parser.add_argument("--tile-inputs-manifest", type=Path, default=None, help="override the base config's tile_inputs_manifest")
    parser.add_argument("--tile-geometry-manifest", type=Path, default=None, help="override the base config's initialization_geometry_manifest")
    parser.add_argument("--tile-id", type=int, default=None, help="override the base config's mipmap_tile_id")
    parser.add_argument("--exclude-faces", required=True, metavar="FACE[,FACE]", help="face ids dropped from every parent image of the Tile")
    parser.add_argument("--label", required=True, help="variant label recorded in the manifests / lineage, e.g. F3_nopitchup")
    parser.add_argument("--out-dir", type=Path, required=True, help="where the derived manifests + variant.json go (new directory)")
    parser.add_argument("--run-id", default=None, help="run_id of the variant config (with --out-config)")
    parser.add_argument("--output-dir", default=None, help="output_dir of the variant config (with --out-config)")
    parser.add_argument("--out-config", type=Path, default=None, help="write the variant config here")
    parser.add_argument("--copy-config-to", type=Path, default=None, help="directory that gets a byte-identical copy of --out-config")
    parser.add_argument("--roi-ids", type=Path, default=None, help="JSON list of sample ids; their presence in the variant view set is reported")
    parser.add_argument("--note", default=None, help="free-text line recorded in the manifests' diagnostic block and the config lineage")
    parser.add_argument("--validate", action="store_true", help="TrainerConfig.from_dict(...).validate() the written config (training venv, hashes the PLY / npz)")
    parser.add_argument("--overwrite", action="store_true", help="allow --out-dir / --out-config to exist already")
    args = parser.parse_args(argv)

    base_config = _load(args.base_config)
    base_config_sha = _sha256_file(args.base_config)
    inputs_path = args.tile_inputs_manifest or Path(str(base_config["tile_inputs_manifest"]))
    geometry_manifest_path = args.tile_geometry_manifest or Path(str(base_config["initialization_geometry_manifest"]))
    tile_id = args.tile_id if args.tile_id is not None else int(base_config["mipmap_tile_id"])
    excluded = parse_face_list(args.exclude_faces)
    if not excluded:
        raise SystemExit("--exclude-faces lists no face id")
    if args.out_config is not None and (not args.run_id or not args.output_dir):
        raise SystemExit("--out-config needs --run-id and --output-dir")
    out_dir = args.out_dir
    for target in (out_dir / "tile_inputs_manifest.json", out_dir / "tile_geometry_manifest.json", args.out_config):
        if target is not None and target.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {target} (pass --overwrite)")

    base_inputs = _load(inputs_path)
    base_geometry = _load(geometry_manifest_path)
    tiles = [t for t in base_inputs["tiles"] if int(t["tile_id"]) == int(tile_id)]
    if len(tiles) != 1:
        raise SystemExit(f"Tile inputs do not contain a unique Tile {tile_id}")
    tile_name = str(tiles[0]["name"])
    geometry_tile = [t for t in base_geometry["tiles"] if int(t["tile_id"]) == int(tile_id)]
    if len(geometry_tile) != 1:
        raise SystemExit(f"Tile geometry manifest does not contain a unique Tile {tile_id}")
    geometry_npz = (geometry_manifest_path.parent / geometry_tile[0]["geometry"]["path"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry_rel = _relative_posix(geometry_npz, out_dir.resolve())

    inputs, geometry, summary = derive_variant_manifests(
        base_inputs, base_geometry, tile_name=tile_name, exclude_face_ids=excluded,
        geometry_path=geometry_rel, label=args.label, note=args.note,
    )
    inputs_out = out_dir / "tile_inputs_manifest.json"
    geometry_out = out_dir / "tile_geometry_manifest.json"
    _dump(inputs_out, inputs, indent=2)
    _dump(geometry_out, geometry, indent=2)
    # Verify exactly as the trainer will (artefact hashes included): inputs
    # against the production root, geometry against the new directory.
    tile_inputs_root = Path(str(base_config.get("tile_inputs_root") or inputs_path.parent))
    verify_tile_inputs_manifest(_load(inputs_out), root=tile_inputs_root, verify_artifacts=True)
    verify_tile_geometry_manifest(_load(geometry_out), root=out_dir, verify_artifacts=True)
    summary["tile_inputs_manifest"] = str(inputs_out)
    summary["tile_geometry_manifest"] = str(geometry_out)
    summary["tile_inputs_root"] = str(tile_inputs_root)
    summary["derived_from"] = {
        "tile_inputs_manifest": str(inputs_path),
        "tile_geometry_manifest": str(geometry_manifest_path),
        "base_config": str(args.base_config),
        "base_config_sha256": base_config_sha,
    }
    print(
        f"{tile_name}: {summary['view_count_before']} -> {summary['view_count_after']} views "
        f"(-{summary['views_removed']}), parents {summary['parent_image_count_before']} -> {summary['parent_image_count_after']} "
        f"(dropped {len(summary['parent_images_dropped'])}); faces {summary['face_counts_before']} -> {summary['face_counts_after']}"
    )
    print(f"  tile_inputs_manifest   {inputs_out}  sha256 {summary['tile_inputs_manifest_sha256']}")
    print(f"  tile_geometry_manifest {geometry_out}  sha256 {summary['tile_geometry_manifest_sha256']}  (npz {geometry_rel})")

    if args.roi_ids is not None:
        roi = roi_presence(inputs["tiles"][0]["views"], _load_list(args.roi_ids))
        roi["path"] = str(args.roi_ids)
        summary["roi_ids"] = roi
        print(f"  roi ids {roi['present']}/{roi['count']} present" + (f", missing {roi['missing']}" if roi["missing"] else ""))

    failures: list[str] = []
    if args.out_config is not None:
        config = derive_variant_config(
            base_config,
            run_id=str(args.run_id),
            output_dir=str(args.output_dir),
            tile_inputs_manifest=str(inputs_out),
            tile_geometry_manifest=str(geometry_out),
            summary=summary,
            base_config_path=str(args.base_config),
            base_config_sha256=base_config_sha,
            note=args.note,
        )
        _dump(args.out_config, config, indent=1)
        summary["config"] = {
            "path": str(args.out_config),
            "sha256": _sha256_file(args.out_config),
            "run_id": config["run_id"],
            "output_dir": config["output_dir"],
            "changed_top_level_fields": sorted(top_level_diff(base_config, config)),
        }
        summary["schedule_preflight"] = config["lineage"]["schedule"]
        print(f"  config {args.out_config}  changed fields {summary['config']['changed_top_level_fields']}")
        preflight = config["lineage"]["schedule"]
        if preflight["ok"]:
            print(f"  schedule pre-flight OK ({preflight['check']}): max_steps {preflight['max_steps']} at {preflight['training_view_count']} views")
        else:
            print(
                f"  WARNING schedule pre-flight would REJECT this config at train start: max_steps {preflight['max_steps']} != "
                f"{MIPMAP_EPOCHS_PER_RUN} x {preflight['training_view_count']} views = {preflight['parity_max_steps']} "
                f"(trainer: '{preflight['trainer_error']}'); schedule deliberately left identical to the base - see lineage.schedule"
            )
        if args.copy_config_to is not None:
            args.copy_config_to.mkdir(parents=True, exist_ok=True)
            copied = args.copy_config_to / args.out_config.name
            shutil.copyfile(args.out_config, copied)
            if _sha256_file(copied) != summary["config"]["sha256"]:
                raise AssertionError(f"copy differs: {copied}")
            summary["config"]["copy"] = str(copied)
            print(f"  copied to {copied}")
        if args.validate:
            from cloudstudio_3dgs.training.trainer import TrainerConfig

            try:
                TrainerConfig.from_dict(_load(args.out_config)).validate()
                summary["config"]["validate"] = "OK"
                print(f"  validate: OK ({args.out_config})")
            except Exception as error:  # report verbatim, do not weaken
                summary["config"]["validate"] = f"{type(error).__name__}: {error}"
                failures.append(summary["config"]["validate"])
                print(f"  validate: FAIL ({args.out_config})\n    {type(error).__name__}: {error}")
    _dump(out_dir / "variant.json", summary, indent=2)
    print(f"  variant.json {out_dir / 'variant.json'}")
    return 1 if failures else 0


def _load_list(path: Path) -> list[str]:
    payload = _load(path)
    if isinstance(payload, dict):
        payload = payload.get("sample_ids") or payload.get("ids") or []
    if not isinstance(payload, list):
        raise SystemExit(f"{path} is not a JSON list of sample ids")
    return [str(s) for s in payload]


if __name__ == "__main__":
    raise SystemExit(main())
