#!/usr/bin/env python3
"""Turn a WP03 diagnostic view set into ordinary trainer arm configs.

From a base arm config (the Tile's R1 recipe) and a preset directory written
by ``tools/build_diagnostic_set.py`` (derived Tile inputs / geometry manifests
+ ``selection.json``) this writes an arm config that

* trains only the diagnostic views: ``tile_inputs_manifest`` points at the
  derived manifest (``tile_inputs_root`` stays the Tile's own root so the
  initialization PLY / backgrounds / face caches are referenced verbatim) and
  ``initialization_geometry_manifest`` at the derived geometry manifest bound
  to it; ``mipmap_tile_id``, PLY, geometry npz and backdrops are unchanged;
* runs a short horizon H under ``schedule_contract: research_rescaled_horizon_v1``
  via ``tools/make_rescaled_schedule_config.rescale_to_horizon`` (growth window
  = the base arm's executed fraction of H snapped to ``refine_every``,
  ``prune_switch_step = H // 2``, reset cadence unchanged, means LR frozen to
  the base arm's effective optimizer base);
* applies the variant: ``R1`` (recipe as is), ``G0`` (lifecycle order
  ``post_optimizer_gsplat``, growth signal ``total_loss``), ``G1`` (same order,
  growth signal ``rgb_only``);
* is named ``diag_<region>_<count>_<label>`` (run_id, output_dir leaf, file)
  and carries a ``diag`` block: region, view ids, base config sha256, derived
  manifest sha256s.

``--validate`` loads each config through ``TrainerConfig.from_dict(...).validate()``
(needs the training venv: torch, and it hashes the Tile PLY / geometry npz);
a failure is printed verbatim and the exit code is non-zero.  ``--eval``
additionally writes ``<region>_eval.json``: the DIAG arm's inputs relabelled
for the existing evaluators (``build_three_way_compare.py`` /
``build_offtrajectory_compare.py`` read ``tile_inputs_manifest`` +
``mipmap_tile_id`` to pick views) plus a ``diag_eval`` block that maps the
region ROI into each compare-strip panel.

Example::

    python tools/make_diagnostic_arm_config.py \
        --base run_configs/house0305_tiles/v9/tile1_R1d_20k.json \
        --diag-root C:/Peter/3dgs-runs/house0305_sop/diag_v2/indoor_door_leaf_Tile_1 \
        --out-dir run_configs/house0305_tiles/diag_v2 --horizon 3000 --validate --eval
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from cloudstudio_3dgs.training.schedule_audit import (  # noqa: E402
    resolved_schedule,
    validate_research_schedule_contract,
)
from make_rescaled_schedule_config import rescale_to_horizon  # noqa: E402

GENERATOR = "tools/make_diagnostic_arm_config.py"
DEFAULT_HORIZON = 3000
VARIANTS: dict[str, dict[str, Any]] = {
    # label -> field overrides relative to the base recipe
    "R1": {},
    "G0": {"default_strategy.lifecycle_execution_order": "post_optimizer_gsplat", "densification_gradient_source": "total_loss"},
    "G1": {"default_strategy.lifecycle_execution_order": "post_optimizer_gsplat", "densification_gradient_source": "rgb_only"},
}
# Which presets get which variants (task brief §6 / §13): U0 and U1 on the
# recipe only; the coverage set also carries the growth-signal pair.
DEFAULT_PLAN: dict[str, tuple[str, ...]] = {"U0": ("R1",), "U1": ("R1",), "DIAG": ("R1", "G0", "G1")}
STRIP_GAP_PX = 8  # tools/build_three_way_compare.py / build_offtrajectory_compare.py


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diag_run_id(region: str, count: int, label: str) -> str:
    return f"diag_{region}_{int(count)}_{label}"


def _set_path(config: dict[str, Any], dotted: str, value: Any) -> tuple[Any, Any]:
    container = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        container = container.setdefault(part, {})
    before = container.get(parts[-1], "<absent>")
    container[parts[-1]] = value
    return before, value


def build_diagnostic_arm(
    base: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    diag_tile_inputs_manifest: str,
    diag_tile_geometry_manifest: str,
    horizon: int,
    label: str,
    output_dir: str,
    reference_scale_m: float | None,
    base_config_path: str,
    base_config_sha256: str,
    checkpoint_every: int | None = None,
    refine_stop_fraction: float | None = None,
) -> dict[str, Any]:
    """Pure: base config dict + selection -> diagnostic arm config dict.

    The returned config already passed ``validate_research_schedule_contract``
    (torch-free); ``TrainerConfig.validate`` is the caller's job.
    """
    if label not in VARIANTS:
        raise ValueError(f"unknown variant {label!r}; expected one of {sorted(VARIANTS)}")
    region = str(selection["region"]["label"])
    tile_id = int(selection["region"]["tile_id"])
    if int(base.get("mipmap_tile_id", -1)) != tile_id:
        raise ValueError(
            f"base config trains Tile {base.get('mipmap_tile_id')} but the selection is for Tile {tile_id}"
        )
    count = int(selection["count"])
    run_id = diag_run_id(region, count, label)

    staged = copy.deepcopy(dict(base))
    rebinding: dict[str, dict[str, Any]] = {}
    for key, value in (
        ("tile_inputs_manifest", diag_tile_inputs_manifest),
        ("initialization_geometry_manifest", diag_tile_geometry_manifest),
    ):
        before, after = _set_path(staged, key, value)
        rebinding[key] = {"base": before, "diag": after}
    # PLY / geometry npz / backgrounds / tile_inputs_root deliberately verbatim.
    provenance = {
        "base_config": base_config_path,
        "base_config_sha256": base_config_sha256,
        "base_run_id": base.get("run_id"),
        "generator": GENERATOR,
        "reference_scale_m": reference_scale_m,
        "diagnostic_preset": selection.get("preset"),
        "diagnostic_region": region,
    }
    arm = rescale_to_horizon(
        staged,
        horizon=int(horizon),
        run_id=run_id,
        output_stem=run_id,
        reference_scale_m=reference_scale_m,
        refine_stop_fraction=refine_stop_fraction,
        keep_controlled_stop=False,
        provenance=provenance,
    )
    arm["output_dir"] = str(output_dir)
    arm["schedule_contract_fields"]["arm"] = label
    arm["schedule_contract_fields"]["role"] = "wp03_diagnostic_short_horizon"
    arm["schedule_contract_fields"]["moved_from_base"]["output_dir"]["s1"] = str(output_dir)

    variant_changes: dict[str, dict[str, Any]] = {}
    for dotted, value in VARIANTS[label].items():
        before, after = _set_path(arm, dotted, value)
        variant_changes[dotted] = {"base": before, "diag": after}

    if checkpoint_every is not None:
        before, after = _set_path(arm, "checkpoint_every", int(checkpoint_every))
        variant_changes["checkpoint_every"] = {"base": before, "diag": after}

    arm["diag"] = {
        "schema_version": 1,
        "generator": GENERATOR,
        "region": selection["region"],
        "preset": selection.get("preset"),
        "count": count,
        "variant": label,
        "variant_fields": variant_changes,
        "horizon_steps": int(horizon),
        "base_config": base_config_path,
        "base_config_sha256": base_config_sha256,
        "tile_inputs_manifest": diag_tile_inputs_manifest,
        "tile_inputs_manifest_sha256": selection.get("tile_inputs_manifest_sha256"),
        "tile_geometry_manifest": diag_tile_geometry_manifest,
        "tile_geometry_manifest_sha256": selection.get("tile_geometry_manifest_sha256"),
        "rebound_fields": rebinding,
        "verbatim_fields": ["tile_inputs_root", "initialization_ply", "initialization_geometry", "background_image_manifest", "background_image_root", "face_cache_manifest", "mipmap_tile_id"],
        "parent_image_ids": [str(r["image_id"]) for r in selection.get("selected", [])],
        "view_sample_ids": list(selection.get("view_sample_ids", [])),
        "view_count": int(selection.get("view_count", 0)),
        "selection_policy": selection.get("policy"),
        "selected_coverage": [
            {
                key: row.get(key)
                for key in (
                    "image_id", "camera_id", "rig_frame_id", "support_fraction", "n_effective", "n_samples",
                    "range_min_m", "range_median_m", "sharpness_lapvar_median", "luma_median", "selection_rank",
                )
            }
            for row in selection.get("selected", [])
        ],
        "notes": [
            "views outside the selection never receive gradients; opacity resets + culls will remove unseen Gaussians of the full-Tile initialization over the run (expected, the arm is a region diagnostic, not a delivery)",
            "H = max_steps; controlled_stop_after_steps removed so the run completes and writes run_manifest.json",
        ],
    }
    # Re-validate after the variant edits (they do not touch schedule fields,
    # but the contract must hold on the file that is written).
    validate_research_schedule_contract(arm)
    return arm


def strip_layout_note(view_count_hint: int | None = None) -> dict[str, Any]:
    return {
        "three_way_compare": (
            "tools/build_three_way_compare.py writes compare_<k>_<sample_id[:18]>.png = [photo | ours | reference] at the "
            f"Tile-crop size of that sample, {STRIP_GAP_PX}px dark gaps; panel i starts at x = i * (crop.width + {STRIP_GAP_PX}). "
            "Views are picked by stride over the diagnostic manifest's Tile views (face-manifest order), so pass --frames >= view_count to get every view."
        ),
        "offtrajectory_compare": (
            "tools/build_offtrajectory_compare.py writes offtraj_<k>_<kind>_<image_id[:12]>.png = [ours | reference] at a displaced pose; "
            "the ROI box does not apply there (the camera moved) - score those strips whole."
        ),
        "roi_scoring": (
            "For a compare strip of sample_id S with crop (w, h): ROI in panel i = [x0 + i*(w+8), x1 + i*(w+8)) x [y0, y1) using "
            "diag_eval.roi_in_crops[S].roi; crop the same box from photo/ours/reference and score (PSNR after per-channel affine match, "
            "Laplacian-variance ratio ours/photo) on the box only; views whose roi is null do not see the region in that face."
        ),
        "roi_semantics": "roi.x0/y0/x1/y1 are array indices inside the Tile crop = face pixel index (rint(c - 0.5)) minus crop.x/crop.y; box = 2-98 percentile of the projected region samples.",
        "frames_hint": view_count_hint,
    }


def build_eval_config(
    diag_arm: Mapping[str, Any], *, selection: Mapping[str, Any], output_dir: str
) -> dict[str, Any]:
    """The region's evaluator config: same inputs/views as the DIAG arm, no
    schedule contract (evaluators do not validate), plus the ROI mapping."""
    region = str(selection["region"]["label"])
    config = copy.deepcopy(dict(diag_arm))
    config["run_id"] = f"diag_{region}_eval"
    config["output_dir"] = str(output_dir)
    config.pop("diag", None)
    config["diag_eval"] = {
        "schema_version": 1,
        "generator": GENERATOR,
        "region": selection["region"],
        "evaluates_arms": [diag_run_id(region, int(c), v) for c, vs in ((1, ("R1",)), (5, ("R1",)), (int(selection["count"]), DEFAULT_PLAN["DIAG"])) for v in vs],
        "view_set": {"preset": selection.get("preset"), "count": int(selection["count"]), "view_count": int(selection.get("view_count", 0)),
                     "tile_inputs_manifest": config.get("tile_inputs_manifest")},
        "roi_in_crops": {entry["sample_id"]: entry for entry in selection.get("roi_in_crops", [])},
        "strip_layout": strip_layout_note(int(selection.get("view_count", 0))),
        "usage": [
            "python tools/build_three_way_compare.py --config <this file> --checkpoint <arm>/checkpoints/latest.pt --output <arm>/compare --frames <view_count> [--reference-ply ... --reference-alignment ...]",
            "python tools/build_offtrajectory_compare.py <this file> <arm>/checkpoints/latest.pt <arm>/offtraj <frames>",
            "python tools/score_compare_strips.py <arm>/compare   (whole-panel scores; ROI-only scores use diag_eval.roi_in_crops)",
        ],
        "caveat": "the U0/U1 arms trained on a subset of these views; scores on views they did not train on are novel-view scores for them and training-view scores for the DIAG arms (report separately)",
    }
    return config


def _default_reference_scale(base: Mapping[str, Any]) -> tuple[float | None, str]:
    """The base arm's runtime median Gaussian scale, if its run left a report."""
    output_dir = base.get("output_dir")
    if output_dir:
        report = Path(str(output_dir)) / "surface_initialization_report.json"
        if report.is_file():
            payload = _load(report)
            value = payload.get("tangent_scale_median")
            if value is not None:
                return float(value), f"{report}:tangent_scale_median"
    return None, "not found (pass --reference-scale-m)"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="base arm config (the Tile's recipe)")
    parser.add_argument("--diag-root", type=Path, required=True, help="<out-root>/<region> written by build_diagnostic_set.py")
    parser.add_argument("--presets", nargs="+", default=None, help="preset dirs to use (default: every *_N dir with a selection.json)")
    parser.add_argument("--variants", nargs="+", default=None, help="restrict variants (default plan: U0/U1 -> R1; DIAG -> R1 G0 G1)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--refine-stop-fraction", type=float, default=None)
    parser.add_argument("--reference-scale-m", type=float, default=None, help="base arm runtime median scale; default reads the base run's surface_initialization_report.json")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="default: min(base checkpoint_every, horizon)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "run_configs" / "house0305_tiles" / "diag_v2")
    parser.add_argument("--runs-root", type=Path, default=None, help="where the arms write outputs (default <diag-root>/runs)")
    parser.add_argument("--validate", action="store_true", help="TrainerConfig.from_dict(...).validate() every config (needs torch + data)")
    parser.add_argument("--eval", action="store_true", help="also write <region>_eval.json from the largest preset")
    args = parser.parse_args(argv)

    base = _load(args.base)
    base_sha = _sha256_bytes(args.base)
    base_abs = args.base.resolve()
    base_rel = base_abs.relative_to(ROOT).as_posix() if base_abs.is_relative_to(ROOT) else str(base_abs)
    reference_scale, reference_source = (args.reference_scale_m, "--reference-scale-m") if args.reference_scale_m is not None else _default_reference_scale(base)
    print(f"reference_scale_m = {reference_scale} ({reference_source})")
    checkpoint_every = args.checkpoint_every
    if checkpoint_every is None:
        checkpoint_every = min(int(base.get("checkpoint_every", args.horizon)), int(args.horizon))

    preset_dirs = [args.diag_root / p for p in args.presets] if args.presets else sorted(
        p for p in args.diag_root.iterdir() if p.is_dir() and (p / "selection.json").is_file()
    )
    if not preset_dirs:
        raise SystemExit(f"no preset directories with selection.json under {args.diag_root}")
    runs_root = args.runs_root or (args.diag_root / "runs")

    written: list[Path] = []
    failures: list[tuple[Path, str]] = []
    largest: tuple[int, dict[str, Any], dict[str, Any]] | None = None
    for preset_dir in preset_dirs:
        selection = _load(preset_dir / "selection.json")
        preset = str(selection["preset"])
        variants = tuple(args.variants) if args.variants else DEFAULT_PLAN.get(preset, ("R1",))
        for label in variants:
            run_id = diag_run_id(selection["region"]["label"], int(selection["count"]), label)
            arm = build_diagnostic_arm(
                base,
                selection=selection,
                diag_tile_inputs_manifest=str(preset_dir / "tile_inputs_manifest.json"),
                diag_tile_geometry_manifest=str(preset_dir / "tile_geometry_manifest.json"),
                horizon=args.horizon,
                label=label,
                output_dir=str(runs_root / run_id),
                reference_scale_m=reference_scale,
                base_config_path=base_rel,
                base_config_sha256=base_sha,
                checkpoint_every=checkpoint_every,
                refine_stop_fraction=args.refine_stop_fraction,
            )
            out = args.out_dir / f"{run_id}.json"
            _dump(out, arm)
            written.append(out)
            schedule = resolved_schedule(arm, int(selection.get("view_count", 0)) or None, reference_scale_m=reference_scale)
            lifecycle = schedule["lifecycle"]
            print(
                f"{run_id}: views={selection.get('view_count')} H={schedule['steps']['max_steps']} refine=[{lifecycle['refine_start_iter']},{lifecycle['refine_stop_iter']}) "
                f"every={lifecycle['refine_every']} reset_every={lifecycle['reset_every']} prune_switch={lifecycle['prune_switch_step']} "
                f"grow={schedule['event_summary']['grow']['count']} resets={schedule['event_summary']['reset']['count']} late_first={schedule['event_summary']['late_threshold']['first_step']} "
                f"epochs={schedule['views']['configured_epochs']} means_lr={arm['learning_rates']['means']:.4g} mismatches={[m['name'] for m in schedule['mismatches']]}"
            )
            if args.validate:
                from cloudstudio_3dgs.training.trainer import TrainerConfig

                try:
                    TrainerConfig.from_dict(_load(out)).validate()
                    print(f"  validate: OK ({out})")
                except Exception as error:  # report verbatim, do not weaken
                    failures.append((out, f"{type(error).__name__}: {error}"))
                    print(f"  validate: FAIL ({out})\n    {type(error).__name__}: {error}")
            if label == "R1" and (largest is None or int(selection["count"]) > largest[0]):
                largest = (int(selection["count"]), selection, arm)

    if args.eval and largest is not None:
        _count, selection, arm = largest
        region = selection["region"]["label"]
        eval_config = build_eval_config(arm, selection=selection, output_dir=str(runs_root / f"diag_{region}_eval"))
        out = args.out_dir / f"{region}_eval.json"
        _dump(out, eval_config)
        written.append(out)
        print(f"eval config: {out} ({len(eval_config['diag_eval']['roi_in_crops'])} Tile views with ROI boxes)")

    print(f"wrote {len(written)} files under {args.out_dir}")
    if failures:
        print("VALIDATION FAILURES:")
        for path, message in failures:
            print(f"  {path}\n    {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
