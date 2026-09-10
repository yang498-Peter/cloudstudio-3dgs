#!/usr/bin/env python3
"""Audit the schedule one or more trainer configs actually execute.

Pure CPU, no torch.  For every ``--config`` the tool resolves the means-LR
curve, the SH schedule, every grow/cull/reset event, the opacity thresholds,
the supervision weights and the optimisation opportunity after the last
birth, then flags duplicated fields that disagree.  When the config's
``output_dir`` holds a ``config_as_run.json`` the as-run copy is audited too
and every leaf difference is listed.  Runtime facts are read when readable:
the Tile view count from ``tile_inputs_manifest`` and, for precomputed
metric scales, the reference scale from ``surface_initialization_report.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.schedule_audit import diff_configs, resolved_schedule


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tile_view_count(config: dict[str, Any]) -> tuple[int | None, str]:
    manifest_path = config.get("tile_inputs_manifest")
    tile_id = config.get("mipmap_tile_id")
    if manifest_path is None or tile_id is None:
        return None, "config has no tile_inputs_manifest / mipmap_tile_id"
    path = Path(manifest_path)
    if not path.is_file():
        return None, f"tile inputs manifest not readable: {path}"
    manifest = _load_json(path)
    matches = [tile for tile in manifest.get("tiles", []) if int(tile.get("tile_id", -1)) == int(tile_id)]
    if len(matches) != 1:
        return None, f"tile {tile_id} not unique in {path}"
    tile = matches[0]
    count = int(tile.get("view_count", len(tile.get("views", []))))
    return count, f"{path} tile {tile_id}"


def _training_view_count(output_dir: Path | None, tile_views: int | None) -> tuple[int | None, str]:
    if output_dir is None:
        return tile_views, "no output_dir"
    holdout = output_dir / "holdout_views.json"
    if holdout.is_file():
        exposure = _load_json(holdout).get("exposure") or {}
        actual = exposure.get("actual_train_view_count")
        if actual is not None:
            return int(actual), f"{holdout}"
    return tile_views, "no holdout_views.json; training views == tile views"


def _reference_scale_m(config: dict[str, Any], output_dir: Path | None) -> tuple[float | None, str]:
    """Runtime median Gaussian scale, only recoverable for precomputed scales.

    In ``precomputed`` mode ``build_metric_scale_calibration`` takes the median
    of the tangent axis of the exact surface scales, which the trainer signs
    as ``tangent_scale_median`` in ``surface_initialization_report.json``.
    Other modes derive the reference from KNN spacing that is not persisted.
    """
    mode = (config.get("metric_scale_calibration") or {}).get("mode")
    if mode != "precomputed":
        return None, f"metric_scale_calibration.mode={mode!r}: reference scale not persisted"
    if output_dir is None:
        return None, "no output_dir"
    report = output_dir / "surface_initialization_report.json"
    if not report.is_file():
        return None, f"not readable: {report}"
    value = _load_json(report).get("tangent_scale_median")
    if value is None:
        return None, f"{report} lacks tangent_scale_median"
    return float(value), f"{report}:tangent_scale_median"


def audit_config(config_path: Path, as_run_override: Path | None) -> dict[str, Any]:
    repo_config = _load_json(config_path)
    output_dir = Path(repo_config["output_dir"]) if repo_config.get("output_dir") else None
    as_run_path = as_run_override
    if as_run_path is None and output_dir is not None:
        candidate = output_dir / "config_as_run.json"
        as_run_path = candidate if candidate.is_file() else None

    tile_views, view_source = _tile_view_count(repo_config)
    training_views, training_source = _training_view_count(output_dir, tile_views)
    reference_scale, reference_source = _reference_scale_m(repo_config, output_dir)

    result: dict[str, Any] = {
        "config_path": str(config_path),
        "config_as_run_path": None if as_run_path is None else str(as_run_path),
        "runtime_facts": {
            "tile_view_count": tile_views,
            "tile_view_count_source": view_source,
            "training_view_count": training_views,
            "training_view_count_source": training_source,
            "reference_scale_m": reference_scale,
            "reference_scale_source": reference_source,
        },
        "repo": resolved_schedule(
            repo_config,
            tile_views,
            reference_scale_m=reference_scale,
            training_view_count=training_views,
        ),
        "as_run": None,
        "repo_vs_as_run_differences": None,
    }
    if as_run_path is not None:
        as_run_config = _load_json(as_run_path)
        result["as_run"] = resolved_schedule(
            as_run_config,
            tile_views,
            reference_scale_m=reference_scale,
            training_view_count=training_views,
        )
        result["repo_vs_as_run_differences"] = diff_configs(repo_config, as_run_config)
    return result


def _events_rows(name: str, source: str, schedule: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "config": name,
            "source": source,
            "step": event["step"],
            "kind": event["kind"],
            "grow": int(event["grow"]),
            "cull": int(event["cull"]),
            "reset": int(event["reset"]),
            "cull_opacity_threshold": event["cull_opacity_threshold"],
            "threshold_phase": event["threshold_phase"],
        }
        for event in schedule["events"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", required=True, type=Path, help="trainer config JSON (repeatable)")
    parser.add_argument(
        "--config-as-run",
        action="append",
        type=Path,
        default=[],
        help="explicit config_as_run.json, positionally paired with --config; defaults to <output_dir>/config_as_run.json",
    )
    parser.add_argument("--output", required=True, type=Path, help="audit JSON to write")
    parser.add_argument("--events-csv", type=Path, help="one row per lifecycle event")
    args = parser.parse_args(argv)
    if args.config_as_run and len(args.config_as_run) != len(args.config):
        parser.error("--config-as-run must be given once per --config or not at all")

    audits = []
    rows: list[dict[str, Any]] = []
    for index, config_path in enumerate(args.config):
        override = args.config_as_run[index] if args.config_as_run else None
        audit = audit_config(config_path, override)
        audits.append(audit)
        name = config_path.stem
        rows.extend(_events_rows(name, "repo", audit["repo"]))
        if audit["as_run"] is not None and audit["repo_vs_as_run_differences"]:
            rows.extend(_events_rows(name, "as_run", audit["as_run"]))

    document = {
        "schema_version": 1,
        "kind": "cloudstudio_training_schedule_audit_collection",
        "configs": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.events_csv is not None:
        args.events_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.events_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["config", "source", "step", "kind", "grow", "cull", "reset", "cull_opacity_threshold", "threshold_phase"],
            )
            writer.writeheader()
            writer.writerows(rows)

    for audit in audits:
        repo = audit["repo"]
        lr = repo["means_lr"]
        print(f"== {audit['config_path']}")
        print(
            f"   steps: max={repo['steps']['max_steps']} stop={repo['steps']['stop_step']} "
            f"views={repo['views']['tile_view_count']} visits/image={repo['views']['average_visits_per_image']}"
        )
        print(
            f"   means LR nominal last={lr['nominal']['last_executed']:.4e} final={lr['nominal']['declared_final']:.4e} "
            f"ratio={lr['nominal_last_executed_over_declared_final']}"
        )
        if lr["effective"] is not None:
            print(f"   means LR effective base={lr['effective']['base']:.4e} last={lr['effective']['last_executed']:.4e}")
        summary = repo["event_summary"]
        print(
            f"   events: grow={summary['grow']['count']} cull={summary['cull']['count']} reset={summary['reset']['count']} "
            f"late_threshold_applied={summary['late_threshold']['ever_applied']}"
        )
        for mismatch in repo["mismatches"]:
            print(f"   MISMATCH {mismatch['name']}: {mismatch['fields']}")
        differences = audit["repo_vs_as_run_differences"]
        if differences is None:
            print("   as-run: not found")
        else:
            print(f"   as-run differences: {len(differences)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
