"""The signed readiness gate chain the trainer requires for independent-AT data.

``trainer.py`` refuses fisheye data with independent-AT lineage unless ``mipmap_pipeline_gate``
names a gate it accepts. house0305 v9 reached ``SURFACE_FROZEN_TRAINING_READY`` through this
sequence, every step a repo tool (2026-10-08 survey of ``house0305_sop/gates_v9``):

    gate_10  build_mipmap_frontend_gate.py      raw + AT + training-tier evidence
    gate_11  advance_mipmap_renderer_mask_gate  renderer masks (train/val)
    gate_12  advance_mipmap_lidar_depth_gate    LiDAR depth of the training manifest
    gate_15  advance_mipmap_tile_gate           surface route from gate_12 + the tile plan
    core     build_tile_core_ownership_contract
    smoke    one strict-fixed factor-1 Tile_0 step under gate_15 (at_steps pipeline-smoke)
    gate_16  promote_surface_frozen_training_gate
    gate_17  bind_monocular_depth_gate          binds DA2 train/val; the gate configs name

The DA2 gate (13) is a dead end on the surface route and the sky gate (14) is not run: the
surface route defers both, with the reason house0305 recorded. A changed input invalidates
the whole chain: the gate files are rebuilt from gate_10, never patched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .errors import DatasetIncompleteError, GpuStepRequired

SURFACE_DEFERRAL_REASON = (
    "surface route: per-view prerendered backdrops stand in for the independent sky model, "
    "so that stage stays deferred"
)
CHAIN_INPUTS = "chain_inputs.json"
GATE_FILES = (
    "gate_10_frontend.json",
    "gate_11_renderer_mask.json",
    "gate_12_lidar_depth.json",
    "gate_15_upstream.json",
    "core_ownership.json",
    "gate_16_surface_frozen.json",
    "gate_17_training.json",
)
TRAINING_GATE = GATE_FILES[-1]
SMOKE_DIR = "pipeline_smoke"


@dataclass(frozen=True)
class GateInputs:
    """Every artefact the chain reads, by the role the gate tools give it."""

    raw_dataset: Path
    time_sync_report: Path
    raw_circle_mask: Path
    raw_person_mask: Path
    feature_runtime: Path
    triangulation_runtime: Path
    at_report: Path
    candidate_model: Path
    training_dataset: Path
    training_circle_mask: Path
    training_circle_root: Path
    training_person_mask: Path
    training_person_root: Path
    split: Path
    face4_train: Path
    face4_train_root: Path
    face4_val: Path
    renderer_train: Path
    renderer_val: Path
    depth: Path
    depth_root: Path
    face_lidar_geometry: Path
    face_lidar_geometry_root: Path
    tile_plan: Path
    tile_inputs: Path
    tile_inputs_root: Path
    tile_geometry: Path
    da2_train: Path
    da2_val: Path
    recording_root: Path
    gsplat_lock: Path

    def fingerprint(self) -> dict[str, str]:
        """sha256 of every input file; directories by the sha their producer recorded."""
        rows: dict[str, str] = {}
        for field in fields(self):
            path = Path(getattr(self, field.name))
            if path.is_file():
                rows[field.name] = _file_sha(path)
            elif path.is_dir():
                rows[field.name] = f"dir:{path}"
            else:
                rows[field.name] = "missing"
        return rows


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool(repo_root: Path, name: str) -> str:
    return str(Path(repo_root) / "tools" / name)


def smoke_command(inputs: GateInputs, gates_dir: Path, *, python: str, cap_max: int) -> tuple[str, ...]:
    return (
        python, "-m", "cloudstudio3dgs_sdk.ingest.at_steps", "pipeline-smoke",
        "--gate", str(gates_dir / "gate_15_upstream.json"),
        "--tile-inputs", str(inputs.tile_inputs),
        "--tile-inputs-root", str(inputs.tile_inputs_root),
        "--tile-geometry-manifest", str(inputs.tile_geometry),
        "--dataset-manifest", str(inputs.training_dataset),
        "--split-manifest", str(inputs.split),
        "--mask-manifest", str(inputs.training_circle_mask),
        "--mask-root", str(inputs.training_circle_root),
        "--person-mask-manifest", str(inputs.training_person_mask),
        "--person-mask-root", str(inputs.training_person_root),
        "--recording-root", str(inputs.recording_root),
        "--face-cache-manifest", str(inputs.face4_train),
        "--face-cache-root", str(inputs.face4_train_root),
        "--renderer-mask-manifest", str(inputs.renderer_train),
        "--depth-manifest", str(inputs.depth),
        "--depth-root", str(inputs.depth_root),
        "--face-lidar-geometry-manifest", str(inputs.face_lidar_geometry),
        "--face-lidar-geometry-root", str(inputs.face_lidar_geometry_root),
        "--gsplat-lock", str(inputs.gsplat_lock),
        "--cap-max", str(int(cap_max)),
        "--output", str(gates_dir / SMOKE_DIR),
    )


def gate_commands(inputs: GateInputs, gates_dir: Path, *, python: str, repo_root: Path) -> list[tuple[str, tuple[str, ...]]]:
    """(output file, command) for every gate step except the smoke, in order."""
    g = {name: str(gates_dir / name) for name in GATE_FILES}
    smoke_manifest = str(gates_dir / SMOKE_DIR / "run" / "run_manifest.json")
    return [
        (GATE_FILES[0], (
            python, _tool(repo_root, "build_mipmap_frontend_gate.py"),
            "--raw-dataset", str(inputs.raw_dataset),
            "--time-sync-report", str(inputs.time_sync_report),
            "--raw-circle-mask", str(inputs.raw_circle_mask),
            "--raw-person-mask", str(inputs.raw_person_mask),
            "--feature-runtime", str(inputs.feature_runtime),
            "--triangulation-runtime", str(inputs.triangulation_runtime),
            "--at-report", str(inputs.at_report),
            "--candidate-model", str(inputs.candidate_model),
            "--training-dataset", str(inputs.training_dataset),
            "--training-circle-mask", str(inputs.training_circle_mask),
            "--training-person-mask", str(inputs.training_person_mask),
            "--split-manifest", str(inputs.split),
            "--face4-train", str(inputs.face4_train),
            "--face4-val", str(inputs.face4_val),
            "--output", g[GATE_FILES[0]],
        )),
        (GATE_FILES[1], (
            python, _tool(repo_root, "advance_mipmap_renderer_mask_gate.py"),
            "--frontend-gate", g[GATE_FILES[0]],
            "--train-renderer-mask", str(inputs.renderer_train),
            "--val-renderer-mask", str(inputs.renderer_val),
            "--output", g[GATE_FILES[1]],
        )),
        (GATE_FILES[2], (
            python, _tool(repo_root, "advance_mipmap_lidar_depth_gate.py"),
            "--renderer-gate", g[GATE_FILES[1]],
            "--depth-manifest", str(inputs.depth),
            "--output", g[GATE_FILES[2]],
        )),
        (GATE_FILES[3], (
            python, _tool(repo_root, "advance_mipmap_tile_gate.py"),
            "--lidar-depth-gate", g[GATE_FILES[2]],
            "--tile-plan", str(inputs.tile_plan),
            "--deferral-reason", SURFACE_DEFERRAL_REASON,
            "--output", g[GATE_FILES[3]],
        )),
        (GATE_FILES[4], (
            python, _tool(repo_root, "build_tile_core_ownership_contract.py"),
            "--tile-inputs", str(inputs.tile_inputs),
            "--tile-inputs-root", str(inputs.tile_inputs_root),
            "--output", g[GATE_FILES[4]],
        )),
        (GATE_FILES[5], (
            python, _tool(repo_root, "promote_surface_frozen_training_gate.py"),
            "--upstream-gate", g[GATE_FILES[3]],
            "--ownership-contract", g[GATE_FILES[4]],
            "--fullres-smoke-manifest", smoke_manifest,
            "--tile-plan", str(inputs.tile_plan),
            "--output", g[GATE_FILES[5]],
        )),
        (GATE_FILES[6], (
            python, _tool(repo_root, "bind_monocular_depth_gate.py"),
            "--training-gate", g[GATE_FILES[5]],
            "--train-da2", str(inputs.da2_train),
            "--val-da2", str(inputs.da2_val),
            "--output", g[GATE_FILES[6]],
        )),
    ]


def build_gate_chain(
    inputs: GateInputs,
    gates_dir: Path,
    *,
    python: str | None = None,
    repo_root: Path,
    cap_max: int,
    run: Callable[[Sequence[str]], int],
    say: Callable[[str], None] = lambda line: None,
) -> Path:
    """Build (or confirm) the chain; return the training gate the arm configs name.

    Raises :class:`GpuStepRequired` for the smoke the first time through; the caller runs it
    under its GPU lease and calls back in.
    """
    python = python or sys.executable
    gates_dir = Path(gates_dir)
    missing = [field.name for field in fields(inputs) if not Path(getattr(inputs, field.name)).exists()]
    if missing:
        raise DatasetIncompleteError("the gate chain is missing inputs: " + ", ".join(missing))

    fingerprint = inputs.fingerprint()
    record = gates_dir / CHAIN_INPUTS
    previous = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else None
    if previous != fingerprint and gates_dir.exists():
        # An input moved: every gate binds to its upstream, so the chain restarts at gate_10.
        say("[prepare] gate chain: inputs changed, rebuilding from gate_10")
        shutil.rmtree(gates_dir)
    gates_dir.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(fingerprint, indent=1, sort_keys=True), encoding="utf-8")

    smoke_done = (gates_dir / SMOKE_DIR / "pipeline_smoke.json").is_file()
    for output, command in gate_commands(inputs, gates_dir, python=python, repo_root=repo_root):
        target = gates_dir / output
        if target.is_file():
            say(f"[prepare] gate chain: {output} present")
            continue
        if output == GATE_FILES[5] and not smoke_done:
            raise GpuStepRequired(
                "pipeline_smoke needs the GPU: one strict-fixed factor-1 Tile_0 step under gate_15",
                cache="pipeline_smoke",
                command=smoke_command(inputs, gates_dir, python=python, cap_max=cap_max),
            )
        say(f"[prepare] gate chain: building {output}")
        code = run(command)
        if code != 0 or not target.is_file():
            raise DatasetIncompleteError(
                f"gate chain: {output} refused (exit {code}). Command: " + " ".join(command)
            )
    return gates_dir / TRAINING_GATE


def gate_inputs_from_specs(
    specs: Mapping[str, object],
    val_specs: Mapping[str, object],
    *,
    recording_root: Path,
    gsplat_lock: Path,
) -> GateInputs:
    """Map the ingest graph's caches onto the roles the gate tools read."""

    def manifest(name: str, table: Mapping[str, object] = specs) -> Path:
        return Path(getattr(table[name], "manifest"))

    def root(name: str, table: Mapping[str, object] = specs) -> Path:
        return Path(getattr(table[name], "root"))

    timesync_root = root("timesync")
    return GateInputs(
        raw_dataset=manifest("raw_dataset_manifest"),
        time_sync_report=timesync_root / "time_sync_report.json",
        raw_circle_mask=manifest("raw_mask_manifest"),
        raw_person_mask=manifest("raw_person_mask_manifest"),
        feature_runtime=manifest("at_features"),
        triangulation_runtime=manifest("at_triangulation"),
        at_report=manifest("at_solve"),
        candidate_model=root("at_solve") / "candidate_model",
        training_dataset=manifest("dataset_manifest"),
        training_circle_mask=manifest("mask_manifest"),
        training_circle_root=root("mask_manifest"),
        training_person_mask=manifest("person_mask_manifest"),
        training_person_root=root("person_mask_manifest"),
        split=manifest("split_manifest"),
        face4_train=manifest("face_cache"),
        face4_train_root=root("face_cache"),
        face4_val=manifest("face_cache", val_specs),
        renderer_train=manifest("renderer_mask"),
        renderer_val=manifest("renderer_mask", val_specs),
        depth=manifest("depth_cache"),
        depth_root=root("depth_cache"),
        face_lidar_geometry=manifest("face_lidar_geometry"),
        face_lidar_geometry_root=root("face_lidar_geometry"),
        tile_plan=manifest("tile_plan"),
        tile_inputs=manifest("tile_inputs"),
        tile_inputs_root=root("tile_inputs"),
        tile_geometry=manifest("tile_geometry"),
        da2_train=manifest("mono_depth"),
        da2_val=manifest("mono_depth", val_specs),
        recording_root=Path(recording_root),
        gsplat_lock=Path(gsplat_lock),
    )
