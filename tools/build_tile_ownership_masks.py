#!/usr/bin/env python3
"""Precompute the Tile-ownership mask pair for every view of one Tile.

``FaceCacheDataset`` with ``tile_ownership_masking`` removes, per cropped
Tile view, the dilated neighbourhood of LiDAR returns that lie outside the
Tile's ``training_and_export_box`` (plus margin) from the photometric / DA2
masks and keeps only the owned returns in the range mask. On the fly that is
a full-resolution dilation on the CPU for every sample every step (3.2x
slower training at DIAG scale). This tool computes the pair once per view and
stores it under ``--output-root`` with a signed manifest
(:mod:`cloudstudio_3dgs.data.tile_ownership_masks`).

Bit-identity with the on-the-fly path: the worker builds the SAME
``FaceCacheDataset`` the trainer builds for the config (Face4 cache, renderer
mask manifest, Face4 LiDAR geometry, Tile crops) but WITHOUT the ownership
box, takes ``sample = dataset[index]`` and calls
``tile_ownership_masks(sample.depth_range_m, sample.depth_mask, sample.K,
sample.c2w, box, margin, dilation)`` - the very objects ``__getitem__`` passes
when the box is set (nothing touches them between the crop and the call).
Views whose cropped depth mask is empty get a record without an artifact,
mirroring the ``depth_mask.any()`` guard of the dataset.

Resumable: ``records.jsonl`` under the output root is appended per view; a
view is reused when its record carries the current binding fingerprint and
its ``.npz`` SHA still matches. Multiprocess (Windows spawn-safe); never
imports the GPU trainer path (``TrainerConfig.from_dict`` only resolves the
config's paths and knob defaults).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST_NAME = "tile_ownership_manifest.json"
RECORDS_NAME = "records.jsonl"


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _lower_priority() -> None:
    """Run below normal so a concurrent trainer's data loading is never starved."""
    if os.name != "nt":
        try:
            os.nice(10)
        except OSError:
            pass
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    # Without explicit types the pseudo-handle (-1) is truncated to a 32-bit
    # int and SetPriorityClass fails silently, leaving the process at Normal.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    below_normal = 0x00004000
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal):
        _log(f"warning: could not lower process priority (error {ctypes.get_last_error()})")


# ------------------------------------------------------------- dataset ----


def build_dataset(kwargs: dict[str, Any]):
    """The trainer's Tile dataset minus the ownership box (and minus the
    supervision sources that do not feed the pair: DA2, mesh, sky, sensor
    coordinates)."""
    from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset

    def path_or_none(key: str) -> Path | None:
        value = kwargs.get(key)
        return None if value is None else Path(value)

    return FaceCacheDataset(
        face_manifest_path=Path(kwargs["face_cache_manifest"]),
        cache_root=Path(kwargs["face_cache_root"]),
        tile_views=kwargs["tile_views"],
        renderer_mask_manifest_path=path_or_none("renderer_mask_manifest"),
        face_lidar_geometry_manifest_path=path_or_none("face_lidar_geometry_manifest"),
        face_lidar_geometry_root=path_or_none("face_lidar_geometry_root"),
        verify_artifacts=bool(kwargs.get("verify_artifacts", True)),
    )


def compute_ownership_record(
    dataset: Any,
    index: int,
    *,
    output_root: Path,
    box: Any,
    margin_m: float,
    dilation_px: int,
) -> dict[str, Any]:
    """Compute, store and describe the pair of one dataset sample."""
    import numpy as np

    from cloudstudio_3dgs.data.tile_ownership_masks import (
        ownership_record_counts,
        tile_ownership_mask_path_for,
        write_ownership_pair,
    )
    from cloudstudio_3dgs.training.face_dataset import (
        SAMPLE_ID_SEPARATOR,
        tile_ownership_masks,
    )

    sample = dataset[index]
    crop = dataset.tile_crop(index)
    image_id, face_id = str(sample.image_id).rsplit(SAMPLE_ID_SEPARATOR, 1)
    relative = tile_ownership_mask_path_for(image_id, face_id)
    depth_range = sample.depth_range_m
    depth_mask = sample.depth_mask
    if depth_range is None or depth_mask is None or not bool(depth_mask.any()):
        counts = ownership_record_counts(sample.rgb_mask, depth_mask, None, None)
        path = sha = None
    else:
        owned, foreign_region = tile_ownership_masks(
            depth_range,
            depth_mask,
            sample.K,
            sample.c2w,
            np.asarray(box, dtype=np.float64),
            float(margin_m),
            int(dilation_px),
        )
        counts = ownership_record_counts(sample.rgb_mask, depth_mask, owned, foreign_region)
        sha = write_ownership_pair(
            output_root / Path(*relative.split("/")), owned, foreign_region
        )
        path = relative
    return {
        "sample_id": str(sample.image_id),
        "image_id": image_id,
        "camera_id": str(sample.camera_id),
        "face_id": face_id,
        "crop": crop,
        "width": int(sample.width),
        "height": int(sample.height),
        "path": path,
        "sha256": sha,
        **counts,
    }


# ------------------------------------------------------- multiprocessing ----

_WORKER: dict[str, Any] = {}


def _worker_init(
    dataset_kwargs: dict[str, Any],
    output_root: str,
    box: list[list[float]],
    margin_m: float,
    dilation_px: int,
    fingerprint: str,
    below_normal: bool,
) -> None:
    if below_normal:
        _lower_priority()
    _WORKER["dataset"] = build_dataset(dataset_kwargs)
    _WORKER["output_root"] = Path(output_root)
    _WORKER["box"] = box
    _WORKER["margin_m"] = float(margin_m)
    _WORKER["dilation_px"] = int(dilation_px)
    _WORKER["fingerprint"] = fingerprint


def _worker_process(index: int) -> dict[str, Any]:
    t0 = time.time()
    record = compute_ownership_record(
        _WORKER["dataset"],
        index,
        output_root=_WORKER["output_root"],
        box=_WORKER["box"],
        margin_m=_WORKER["margin_m"],
        dilation_px=_WORKER["dilation_px"],
    )
    record["rule_fingerprint"] = _WORKER["fingerprint"]
    record["_index"] = int(index)
    record["_seconds"] = time.time() - t0
    return record


# ---------------------------------------------------------------- resume ----


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[str(record["sample_id"])] = record
    return records


def _reusable(record: dict[str, Any] | None, output_root: Path, fingerprint: str) -> bool:
    if record is None or record.get("rule_fingerprint") != fingerprint:
        return False
    if not record.get("ownership_applied"):
        return record.get("path") is None and record.get("sha256") is None
    if not record.get("path"):
        return False
    artifact = output_root / Path(*str(record["path"]).split("/"))
    return artifact.is_file() and _sha256_file(artifact) == str(record["sha256"])


def binding_fingerprint(binding: dict[str, Any]) -> str:
    from cloudstudio_3dgs.data.manifest import canonical_json_bytes

    return hashlib.sha256(canonical_json_bytes(binding)).hexdigest()


def _percentiles(values: list[float]) -> str:
    import numpy as np

    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"p05 {np.percentile(arr, 5):.3f} p50 {np.median(arr):.3f} "
        f"p95 {np.percentile(arr, 95):.3f} mean {arr.mean():.3f}"
    )


# ------------------------------------------------------------------ main ----


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path, help="trainer config JSON (paths, tile id, knobs)")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tile-id", type=int, default=None, help="override the config's mipmap_tile_id")
    parser.add_argument("--margin-m", type=float, default=None, help="override tile_ownership_margin_m")
    parser.add_argument("--dilation-px", type=int, default=None, help="override tile_ownership_dilation_px")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N views (no manifest)")
    parser.add_argument("--select", default=None, help="comma list of image_id::face_id (no manifest)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-verify-artifacts", action="store_true", help="skip Face4 artifact SHA checks in the workers")
    parser.add_argument("--no-below-normal-priority", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not args.no_below_normal_priority:
        _lower_priority()

    from cloudstudio_3dgs.data.tile_ownership_masks import (
        TILE_OWNERSHIP_ENCODING,
        TILE_OWNERSHIP_FUNCTION,
        build_tile_ownership_manifest,
        load_tile_ownership_manifest,
    )
    from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    config_path = args.config.resolve()
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    config = TrainerConfig.from_dict(config_dict)
    if config.face_cache_manifest is None or config.face_cache_root is None:
        raise SystemExit("config has no face cache (Tile ownership needs Face4 training)")
    if config.tile_inputs_manifest is None or config.tile_inputs_root is None:
        raise SystemExit("config has no Tile inputs")
    tile_id = int(config.mipmap_tile_id if args.tile_id is None else args.tile_id)
    margin_m = float(config.tile_ownership_margin_m if args.margin_m is None else args.margin_m)
    dilation_px = int(config.tile_ownership_dilation_px if args.dilation_px is None else args.dilation_px)
    if margin_m < 0.0 or dilation_px < 0:
        raise SystemExit("margin and dilation must be non-negative")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _log(f"pid {os.getpid()}  config {config_path}  output {output_root}")

    tile_inputs = json.loads(config.tile_inputs_manifest.read_text(encoding="utf-8"))
    tile_inputs_sha = verify_tile_inputs_manifest(tile_inputs)
    selected = [t for t in tile_inputs["tiles"] if int(t["tile_id"]) == tile_id]
    if len(selected) != 1:
        raise SystemExit(f"Tile inputs do not contain a unique Tile {tile_id}")
    tile = selected[0]
    tile_views = list(tile["views"])
    box = [[float(v) for v in tile["training_and_export_box"][0]], [float(v) for v in tile["training_and_export_box"][1]]]
    _log(
        f"Tile {tile_id} ({tile.get('name')}): {len(tile_views)} views, box {box}, "
        f"tile inputs sha {tile_inputs_sha[:12]}"
    )

    dataset_kwargs = {
        "face_cache_manifest": str(config.face_cache_manifest),
        "face_cache_root": str(config.face_cache_root),
        "tile_views": tile_views,
        "renderer_mask_manifest": None if config.renderer_mask_manifest is None else str(config.renderer_mask_manifest),
        "face_lidar_geometry_manifest": None if config.face_lidar_geometry_manifest is None else str(config.face_lidar_geometry_manifest),
        "face_lidar_geometry_root": None if config.face_lidar_geometry_root is None else str(config.face_lidar_geometry_root),
        "verify_artifacts": not args.no_verify_artifacts,
    }
    t_dataset = time.time()
    dataset = build_dataset(dataset_kwargs)
    split = str(dataset.manifest.get("split", ""))
    binding = {
        "function": TILE_OWNERSHIP_FUNCTION,
        "encoding": TILE_OWNERSHIP_ENCODING,
        "source_face_manifest_sha256": dataset.face_manifest_sha256,
        "renderer_mask_manifest_sha256": dataset.renderer_mask_manifest_sha256,
        "face_lidar_geometry_manifest_sha256": dataset.face_lidar_geometry_manifest_sha256,
        "tile_inputs_manifest_sha256": tile_inputs_sha,
        "tile_id": tile_id,
        "training_and_export_box": box,
        "margin_m": margin_m,
        "dilation_px": dilation_px,
    }
    fingerprint = binding_fingerprint(binding)
    _log(
        f"dataset ready in {time.time() - t_dataset:.1f}s: {len(dataset)} samples, split={split}, "
        f"face sha {dataset.face_manifest_sha256[:12]}, renderer sha "
        f"{(dataset.renderer_mask_manifest_sha256 or 'none')[:12]}, lidar sha "
        f"{(dataset.face_lidar_geometry_manifest_sha256 or 'none')[:12]}, "
        f"margin {margin_m} m, dilation {dilation_px} px, fingerprint {fingerprint[:12]}"
    )

    sample_ids = dataset.sample_ids()
    indices = list(range(len(dataset)))
    partial = False
    if args.select:
        wanted = {item.strip() for item in args.select.split(",") if item.strip()}
        indices = [i for i in indices if sample_ids[i] in wanted]
        if len(indices) != len(wanted):
            raise SystemExit("--select names samples that are not in this Tile")
        partial = True
    if args.limit is not None:
        indices = indices[: max(0, int(args.limit))]
        partial = True

    records_path = output_root / RECORDS_NAME
    existing = {} if args.no_resume else _load_records(records_path)
    todo = [i for i in indices if not _reusable(existing.get(sample_ids[i]), output_root, fingerprint)]
    reused = len(indices) - len(todo)
    _log(f"{len(indices)} views selected ({'partial' if partial else 'full Tile'}): {len(todo)} to compute, {reused} reused")

    t_loop = time.time()
    seconds: list[float] = []
    done = 0
    if todo:
        workers = max(1, min(int(args.workers), len(todo)))
        init_args = (
            dataset_kwargs,
            str(output_root),
            box,
            margin_m,
            dilation_px,
            fingerprint,
            not args.no_below_normal_priority,
        )
        if workers == 1:
            _worker_init(*init_args)
            results = map(_worker_process, todo)
            pool = None
        else:
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=workers, initializer=_worker_init, initargs=init_args)
            results = pool.imap(_worker_process, todo, chunksize=max(1, int(args.chunksize)))
        try:
            with records_path.open("a", encoding="utf-8", newline="\n") as stream:
                for record in results:
                    seconds.append(float(record.pop("_seconds")))
                    record.pop("_index", None)
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    existing[str(record["sample_id"])] = record
                    done += 1
                    if done % args.log_every == 0 or done == 1 or done == len(todo):
                        elapsed = time.time() - t_loop
                        rate = done / elapsed if elapsed > 0 else 0.0
                        remaining = (len(todo) - done) / rate / 60 if rate > 0 else float("nan")
                        _log(
                            f"{done}/{len(todo)} computed  last {record['sample_id']} "
                            f"owned {100 * record['lidar_owned_fraction']:.1f}% of returns, "
                            f"rgb dropped {100 * record['rgb_dropped_fraction']:.1f}%  "
                            f"worker {seconds[-1]:.2f}s  wall {rate:.2f}/s  ETA {remaining:.1f} min"
                        )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    _log(
        f"loop done in {(time.time() - t_loop) / 60:.2f} min: computed {done}, reused {reused}"
        + (f", mean worker {sum(seconds) / len(seconds):.2f}s/view" if seconds else "")
    )

    if partial:
        _log("partial selection: no manifest written")
        return 0

    records: list[dict[str, Any]] = []
    for i in indices:
        record = existing.get(sample_ids[i])
        if not _reusable(record, output_root, fingerprint):
            raise SystemExit(f"{sample_ids[i]} has no verified ownership pair after the loop")
        records.append({k: v for k, v in record.items() if k != "rule_fingerprint"})
    manifest = build_tile_ownership_manifest(
        split=split,
        source_face_manifest_sha256=dataset.face_manifest_sha256,
        renderer_mask_manifest_sha256=dataset.renderer_mask_manifest_sha256,
        face_lidar_geometry_manifest_sha256=dataset.face_lidar_geometry_manifest_sha256,
        tile_inputs_manifest_sha256=tile_inputs_sha,
        tile_id=tile_id,
        training_and_export_box=box,
        margin_m=margin_m,
        dilation_px=dilation_px,
        source_identity={
            **dict(dataset.manifest.get("source_identity", {})),
            "config": str(config_path),
            "run_id": str(config.run_id),
        },
        records=records,
    )
    manifest_path = output_root / MANIFEST_NAME
    _atomic_json(manifest_path, manifest)
    reloaded = load_tile_ownership_manifest(
        manifest_path, expected_face_manifest_sha256=dataset.face_manifest_sha256
    )
    summary = reloaded["summary"]
    _log(
        f"manifest written: {manifest_path} sha={reloaded['tile_ownership_manifest_sha256'][:12]} "
        f"samples={summary['sample_count']} images={summary['image_count']} "
        f"with_ownership={summary['samples_with_ownership']} "
        f"rgb_mask_preserved={summary['samples_with_rgb_mask_preserved']}"
    )
    _log(
        f"pooled: rgb dropped {100 * summary['pooled_rgb_dropped_fraction']:.2f}% of supervised pixels, "
        f"LiDAR owned {100 * summary['pooled_lidar_owned_fraction']:.2f}% of returns; "
        f"per-view medians: rgb dropped {100 * summary['median_rgb_dropped_fraction']:.2f}%, "
        f"owned {100 * summary['median_lidar_owned_fraction']:.2f}%"
    )
    by_face: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["ownership_applied"]:
            by_face.setdefault(str(record["face_id"]), []).append(record)
    for face_id, face_records in sorted(by_face.items()):
        _log(
            f"  {face_id:14s} n={len(face_records):4d}  rgb dropped "
            f"{_percentiles([r['rgb_dropped_fraction'] for r in face_records])}  |  owned "
            f"{_percentiles([r['lidar_owned_fraction'] for r in face_records])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
