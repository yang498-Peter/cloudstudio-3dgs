"""Wrapped steps of the independent-AT pose route that are not a single tool call.

The pose route (raw S1 poses -> independent AT -> training manifest -> signed gate chain) is
mostly plain tools, each run as one cache of the ingest graph. Four steps need more than an
argv, so they live here and the graph calls them as ``python -m
cloudstudio3dgs_sdk.ingest.at_steps <step> ...``:

``fresh``
    Run a tool that refuses a non-empty output directory (triangulation, person-mask rebind)
    after clearing what an earlier, failed attempt left there. Only directories inside the
    work root are ever cleared; they are this SDK's own derived caches.
``timesync-model``
    The time-sync audit renders a *trained* model over camera-time offsets, and the frontend
    gate refuses a capture without the audit. A short whole-scene run on the raw poses
    (house0614 runbook A1: 3000 steps at factor 4, 1M cap, no depth) is the model; no gate is
    owed because the raw manifest has no AT lineage.
``timesync-audit``
    ``tools/audit_camera_time_sync.py`` against that model, then a signed step manifest. A
    non-zero best offset means the camera clocks are off; the frontend gate would refuse it,
    so this step refuses first and says so.
``pipeline-smoke``
    The one-step, factor-1, strict-fixed Tile_0 run the surface training gate is promoted
    from: it measures peak VRAM at full resolution under the tile gate.

Each writes a manifest signed with ``sdk_step_manifest_sha256`` so the graph's status logic
(present / stale / missing) applies to it like to any other cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP_SHA_KEY = "sdk_step_manifest_sha256"
TIMESYNC_MODEL_MANIFEST = "timesync_model.json"
TIMESYNC_STEP_MANIFEST = "time_sync_step.json"
TIMESYNC_REPORT = "time_sync_report.json"
SMOKE_STEP_MANIFEST = "pipeline_smoke.json"
TIMESYNC_OFFSETS_MS = (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0)

# The time-sync model: house0614 runbook A1 (C:/Peter/3dgs-runs/house0614_smoke_f4_config.json),
# with the depth keys dropped and the range term off, as that runbook says - a raw capture has
# no depth cache yet and the trainer wants both depth keys or neither.
TIMESYNC_MODEL_RECIPE: Mapping[str, Any] = {
    "device": "cuda:0",
    "seed": 42,
    "factor": 4,
    "max_steps": 3000,
    "checkpoint_every": 3000,
    "cap_max": 1_000_000,
    "init_scale_m": 0.05,
    "lidar_range_weight": 0.0,
    "rgb_l1_weight": 0.8,
    "rgb_ssim_weight": 0.2,
    "mcmc_noise_injection_stop_iter": -1,
    "mcmc_refine_every": 100,
    "mcmc_refine_start_iter": 500,
    "mcmc_refine_stop_iter": 3000,
}

# The full-resolution smoke: the non-path keys of run_configs/house0305_tiles/v9/tile0_smoke_v9.json,
# the configuration house0305's gate_16 was promoted from. One strict-fixed step trains nothing;
# the knobs only have to be a combination the trainer accepts, and this one is proven.
SMOKE_RECIPE: Mapping[str, Any] = {
    "trainer_preset": "custom",
    "seed": 42,
    "device": "cuda:0",
    "factor": 1,
    "max_steps": 1,
    "checkpoint_every": 1,
    "implementation_smoke_only": True,
    "topology_policy": {"mode": "strict_fixed"},
    "golden_evaluation": {"enabled": False},
    "final_evaluation_artifacts": False,
    "background_color": [1.0, 1.0, 1.0],
    "color_model": "sh",
    "sh_degree": 0,
    "sh_degree_interval": 0,
    "require_person_masks": True,
    "pinhole_rasterize_mode": "classic",
    "pinhole_with_ut": False,
    "view_sampling_mode": "fisher_yates_without_replacement_per_epoch",
    "densification_strategy": "default_3dgs",
    "densification_gradient_source": "total_loss",
    "default_strategy": {
        "exact_mipmap_lifecycle": False,
        "lifecycle_execution_order": "post_optimizer_gsplat",
    },
    "surface_initialization": {
        "enabled": True,
        "mode": "mipmap_k7_k30",
        "planarity_gate": 0.6,
        "normal_scale_ratio": 0.5,
    },
    "metric_scale_calibration": {
        "mode": "precomputed",
        "knn_neighbors": 7,
        "knn_reduction": "arithmetic_mean",
        "scale_multiplier": 1.0,
    },
    "learning_rates": {"means": 0.00016, "scales": 0.005, "quats": 0.001, "opacities": 0.05, "colors": 0.0025},
    "means_lr_final_factor": 0.01,
    "exposure_compensation": {"enabled": True, "learning_rate": 0.005},
    "rgb_l1_weight": 0.6,
    "rgb_ssim_weight": 0.4,
    "rgb_ssim_mode": "local_gaussian",
    "lidar_range_weight": 0.5,
    "lidar_range_loss_mode": "linear_l1",
    "lidar_log_range_huber_delta": 0.05,
    "lidar_alpha_weight": 0.3,
    "lidar_alpha_target": 0.95,
    "lidar_alpha_dilation_radius_px": 6,
    "surface_alpha_floor_profile": True,
    "mcmc_noise_lr": 0.0,
    "mcmc_noise_injection_stop_iter": 0,
    "mcmc_refine_every": 100,
    "mcmc_refine_start_iter": 500,
    "mcmc_refine_stop_iter": 38656,
}


# --------------------------------------------------------------------------
# signing and small helpers
# --------------------------------------------------------------------------


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_step_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != STEP_SHA_KEY}
    body[STEP_SHA_KEY] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def verify_step_manifest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != STEP_SHA_KEY}
    expected = hashlib.sha256(_canonical(body)).hexdigest()
    if payload.get(STEP_SHA_KEY) != expected:
        raise ValueError("SDK step manifest signature does not match its content")
    return expected


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha(path: Path, key: str) -> str:
    return str(json.loads(Path(path).read_text(encoding="utf-8")).get(key, ""))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _run(command: Sequence[str]) -> int:
    print("$ " + " ".join(str(part) for part in command), flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), environment.get("PYTHONPATH", "")) if part
    )
    return subprocess.run([str(part) for part in command], cwd=str(REPO_ROOT), env=environment, check=False).returncode


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def fresh(output: Path, work_root: Path, command: Sequence[str]) -> int:
    """Clear an earlier attempt's output (inside the work root only), then run ``command``."""
    if not _inside(output, work_root):
        raise SystemExit(f"refusing to clear {output}: it is not inside the work root {work_root}")
    if output.exists():
        shutil.rmtree(output) if output.is_dir() else output.unlink()
    return _run(command)


def timesync_model_config(args: argparse.Namespace, init_ply: Path) -> dict[str, Any]:
    config = dict(TIMESYNC_MODEL_RECIPE)
    config.update(
        {
            "run_id": "sdk-timesync-model",
            "output_dir": str(args.output / "run"),
            "dataset_manifest": str(args.dataset_manifest),
            "split_manifest": str(args.split_manifest),
            "mask_manifest": str(args.mask_manifest),
            "mask_root": str(args.mask_root),
            "person_mask_manifest": str(args.person_mask_manifest),
            "person_mask_root": str(args.person_mask_root),
            "recording_root": str(args.recording_root),
            "initialization_ply": str(init_ply),
            "gsplat_lock": str(args.gsplat_lock),
        }
    )
    return config


def timesync_model(args: argparse.Namespace) -> int:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    init_dir = output / "init"
    init_ply = init_dir / "sparse_pc.ply"
    if not init_ply.is_file():
        code = _run(
            (
                sys.executable, str(REPO_ROOT / "tools" / "build_lidar_init.py"),
                "--run", str(args.run_dir), "--output", str(init_dir),
                "--voxel-size", str(args.init_voxel_m), "--with-pca", "--seed", "42",
            )
        )
        if code != 0 or not init_ply.is_file():
            print(f"timesync-model: build_lidar_init.py exited {code}", file=sys.stderr)
            return code or 1
    config_path = output / "timesync_model_config.json"
    _write_json(config_path, timesync_model_config(args, init_ply))
    checkpoint = output / "run" / "checkpoints" / "latest.pt"
    code = _run((sys.executable, str(REPO_ROOT / "tools" / "train_gsplat.py"), "--config", str(config_path)))
    if code != 0 or not checkpoint.is_file():
        print(f"timesync-model: train_gsplat.py exited {code}", file=sys.stderr)
        return code or 1
    _write_json(
        output / TIMESYNC_MODEL_MANIFEST,
        sign_step_manifest(
            {
                "kind": "sdk_timesync_model",
                "schema_version": 1,
                "dataset_manifest_sha256": _manifest_sha(args.dataset_manifest, "manifest_sha256"),
                "config": str(config_path),
                "config_sha256": file_sha256(config_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "recipe": dict(TIMESYNC_MODEL_RECIPE),
                "source": "house0614 runbook A1: a short raw-pose whole-scene run, rendered by the audit",
            }
        ),
    )
    return 0


def timesync_audit(args: argparse.Namespace) -> int:
    model = json.loads(Path(args.model_manifest).read_text(encoding="utf-8"))
    verify_step_manifest(model)
    base_sha = _manifest_sha(args.dataset_manifest, "manifest_sha256")
    if model.get("dataset_manifest_sha256") != base_sha:
        print("timesync-audit: the model was trained on a different raw manifest", file=sys.stderr)
        return 1
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / TIMESYNC_REPORT
    command = [
        sys.executable, str(REPO_ROOT / "tools" / "audit_camera_time_sync.py"),
        "--config", str(model["config"]),
        "--checkpoint", str(model["checkpoint"]),
        "--base-dataset-manifest", str(args.dataset_manifest),
        "--offset-ms", *[str(value) for value in args.offset_ms],
        "--factor", str(args.factor),
        "--maximum-rig-frames", str(args.maximum_rig_frames),
        "--output", str(report_path),
    ]
    code = _run(command)
    if code != 0 or not report_path.is_file():
        print(f"timesync-audit: audit_camera_time_sync.py exited {code}", file=sys.stderr)
        return code or 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = float(report.get("best_offset_ms", float("inf")))
    accepted = abs(best) <= 1e-9 and report.get("base_dataset_manifest_sha256") == base_sha
    _write_json(
        output / TIMESYNC_STEP_MANIFEST,
        sign_step_manifest(
            {
                "kind": "sdk_time_sync",
                "schema_version": 1,
                "base_dataset_manifest_sha256": base_sha,
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
                "best_offset_ms": best,
                "accepted": accepted,
                "factor": args.factor,
                "maximum_rig_frames": args.maximum_rig_frames,
            }
        ),
    )
    if not accepted:
        print(
            f"timesync-audit: the camera clocks are off by {best:g} ms against the poses (significant at "
            "the audit's alpha). The frontend gate refuses this capture; fix the image timestamps and "
            "re-ingest.",
            file=sys.stderr,
        )
        return 3
    return 0


def _tile0_paths(tile_inputs: Path, tile_inputs_root: Path, geometry_manifest: Path) -> dict[str, str]:
    inputs = json.loads(Path(tile_inputs).read_text(encoding="utf-8"))
    geometry = json.loads(Path(geometry_manifest).read_text(encoding="utf-8"))
    tile = next(entry for entry in inputs["tiles"] if int(entry["tile_id"]) == 0)
    block = next(entry for entry in geometry["tiles"] if int(entry["tile_id"]) == 0)
    return {
        "initialization_ply": str(Path(tile_inputs_root) / str(tile["initialization"]["path"])),
        "initialization_count": int(tile["initialization"].get("point_count", 0) or 0),
        "initialization_geometry": str(Path(geometry_manifest).parent / str(block["geometry"]["path"])),
    }


def smoke_config(args: argparse.Namespace) -> dict[str, Any]:
    tile = _tile0_paths(args.tile_inputs, args.tile_inputs_root, args.tile_geometry_manifest)
    config = json.loads(json.dumps(SMOKE_RECIPE))
    config.update(
        {
            "run_id": "sdk-pipeline-smoke-tile0",
            "output_dir": str(args.output / "run"),
            "mipmap_tile_id": 0,
            "cap_max": int(args.cap_max),
            "mipmap_pipeline_gate": str(args.gate),
            "dataset_manifest": str(args.dataset_manifest),
            "split_manifest": str(args.split_manifest),
            "mask_manifest": str(args.mask_manifest),
            "mask_root": str(args.mask_root),
            "person_mask_manifest": str(args.person_mask_manifest),
            "person_mask_root": str(args.person_mask_root),
            "recording_root": str(args.recording_root),
            "face_cache_manifest": str(args.face_cache_manifest),
            "face_cache_root": str(args.face_cache_root),
            "renderer_mask_manifest": str(args.renderer_mask_manifest),
            "depth_manifest": str(args.depth_manifest),
            "depth_root": str(args.depth_root),
            "face_lidar_geometry_manifest": str(args.face_lidar_geometry_manifest),
            "face_lidar_geometry_root": str(args.face_lidar_geometry_root),
            "tile_inputs_manifest": str(args.tile_inputs),
            "tile_inputs_root": str(args.tile_inputs_root),
            "initialization_ply": tile["initialization_ply"],
            "initialization_geometry": tile["initialization_geometry"],
            "initialization_geometry_manifest": str(args.tile_geometry_manifest),
            "gsplat_lock": str(args.gsplat_lock),
        }
    )
    return config


def pipeline_smoke(args: argparse.Namespace) -> int:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "pipeline_smoke_config.json"
    _write_json(config_path, smoke_config(args))
    run_manifest = output / "run" / "run_manifest.json"
    if run_manifest.exists():
        run_manifest.unlink()
    code = _run((sys.executable, str(REPO_ROOT / "tools" / "train_gsplat.py"), "--config", str(config_path)))
    if code != 0 or not run_manifest.is_file():
        print(f"pipeline-smoke: train_gsplat.py exited {code}", file=sys.stderr)
        return code or 1
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    _write_json(
        output / SMOKE_STEP_MANIFEST,
        sign_step_manifest(
            {
                "kind": "sdk_pipeline_smoke",
                "schema_version": 1,
                "gate": str(args.gate),
                "gate_sha256": _manifest_sha(args.gate, "gate_manifest_sha256"),
                "run_manifest": str(run_manifest),
                "run_manifest_sha256": file_sha256(run_manifest),
                "peak_vram_bytes": (payload.get("training") or {}).get("peak_vram_bytes"),
                "gaussian_count": (payload.get("training") or {}).get("gaussian_count"),
            }
        ),
    )
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _paths(parser: argparse.ArgumentParser, *names: str) -> None:
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cloudstudio3dgs_sdk.ingest.at_steps", description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)

    run_fresh = sub.add_parser("fresh", help="clear an earlier attempt's output dir, then run a command")
    _paths(run_fresh, "output", "work-root")
    run_fresh.add_argument("command", nargs=argparse.REMAINDER)

    model = sub.add_parser("timesync-model", help="short raw-pose whole-scene run the time-sync audit renders")
    _paths(model, "dataset-manifest", "split-manifest", "mask-manifest", "mask-root", "person-mask-manifest",
           "person-mask-root", "recording-root", "run-dir", "gsplat-lock", "output")
    model.add_argument("--init-voxel-m", type=float, default=0.2)

    audit = sub.add_parser("timesync-audit", help="camera-time offset sweep against the time-sync model")
    _paths(audit, "model-manifest", "dataset-manifest", "output")
    audit.add_argument("--offset-ms", type=float, nargs="+", default=list(TIMESYNC_OFFSETS_MS))
    audit.add_argument("--factor", type=int, choices=(1, 2, 4), default=4)
    audit.add_argument("--maximum-rig-frames", type=int, default=40)

    smoke = sub.add_parser("pipeline-smoke", help="one strict-fixed factor-1 Tile_0 step under the tile gate")
    _paths(smoke, "gate", "tile-inputs", "tile-inputs-root", "tile-geometry-manifest", "dataset-manifest",
           "split-manifest", "mask-manifest", "mask-root", "person-mask-manifest", "person-mask-root",
           "recording-root", "face-cache-manifest", "face-cache-root", "renderer-mask-manifest",
           "depth-manifest", "depth-root", "face-lidar-geometry-manifest", "face-lidar-geometry-root",
           "gsplat-lock", "output")
    smoke.add_argument("--cap-max", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.step == "fresh":
        command = [part for part in args.command if part != "--"]
        if not command:
            raise SystemExit("fresh: no command given")
        return fresh(args.output, args.work_root, command)
    if args.step == "timesync-model":
        return timesync_model(args)
    if args.step == "timesync-audit":
        return timesync_audit(args)
    return pipeline_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
