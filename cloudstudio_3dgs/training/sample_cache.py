"""Decode a training sample once per run, not once per visit.

A tile shows each of its ~2000 views about fifteen times. Every visit used to decode the same
four PNGs and two depth archives from scratch and reduce them to the training resolution
again - 300-450 ms of CPU per sample. With a lean population the render takes 40 ms, so the
step was bound by that decode: 17% GPU utilisation on tile1_c4_30k with six decode threads.

``SampleCache`` writes each sample, as the dataset returned it, to an uncompressed archive in
the run directory the first time it is fetched and reads it back afterwards (a few ms). The
cache is lossless - a cached sample compares equal, array for array, to a freshly decoded one
(tests/test_sample_cache.py) - so it changes wall clock only.

Storage is kept small without losing a bit:

* boolean masks are bit-packed;
* sparse LiDAR range / confidence are stored at their non-zero pixels when everything else
  is exactly zero (it is, for a reduced sample), dense otherwise;
* the z-to-range factor of a reduced sample is a pure function of its intrinsics and is
  recomputed on load; an unreduced sample stores it.

The cache belongs to one run: ``identity`` (dataset identity, view list, reduction) is written
next to it, and a directory with another identity is emptied rather than trusted. A sample the
cache cannot represent exactly (mesh supervision, sensor pixel coordinates) is simply not
cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.sample_resolution import _range_scale

SCHEMA_VERSION = 1
#: The cache stops growing before the volume has less than this free: it is an
#: optimisation, and a full system disk takes everything else on the machine with it.
DEFAULT_MIN_FREE_BYTES = 12 * 1024**3
IDENTITY_NAME = "identity.json"
#: Fields the archive reproduces. A sample carrying anything else that is not None is not cached.
_UNSUPPORTED = (
    "sensor_pixel_coords",
    "mesh_depth_range_m",
    "mesh_normal_camera",
    "mesh_confidence",
    "mesh_depth_mask",
)
_BOOL_FIELDS = ("rgb_mask", "depth_mask", "mono_depth_mask", "sky_mask")
_PATH_FIELDS = ("depth_cache_path", "mono_depth_cache_path", "mesh_geometry_cache_path")


def identity_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, dtype=bool).reshape(-1))


def _unpack(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(bits, count=shape[0] * shape[1]).astype(bool).reshape(shape)


def encode_sample(sample: TrainingSample) -> dict[str, np.ndarray] | None:
    """The arrays to store for ``sample``, or None when it cannot be stored exactly."""
    if any(getattr(sample, name, None) is not None for name in _UNSUPPORTED):
        return None
    if sample.camera_model != "pinhole":
        return None
    shape = (int(sample.height), int(sample.width))
    image = np.asarray(sample.image)
    if image.dtype != np.uint8 or image.shape[:2] != shape:
        return None
    meta = {
        "image_id": sample.image_id,
        "rig_frame_id": sample.rig_frame_id,
        "camera_id": sample.camera_id,
        "camera_model": sample.camera_model,
        "height": shape[0],
        "width": shape[1],
        "sensor_resolution": None if sample.sensor_resolution is None else list(sample.sensor_resolution),
        "resolution_reduction": float(sample.resolution_reduction),
        "paths": {
            name: None if getattr(sample, name) is None else str(getattr(sample, name))
            for name in _PATH_FIELDS
        },
        "present": [],
    }
    arrays: dict[str, np.ndarray] = {
        "image": image,
        "c2w": np.asarray(sample.c2w),
        "K": np.asarray(sample.K),
        "radial_coeffs": np.asarray(sample.radial_coeffs),
    }
    for name in _BOOL_FIELDS:
        value = getattr(sample, name)
        if value is None:
            continue
        value = np.asarray(value)
        if value.dtype != np.bool_ or value.shape != shape:
            return None
        arrays[f"bits_{name}"] = _pack(value)
        meta["present"].append(name)
    if sample.depth_range_m is not None:
        depth_range = np.asarray(sample.depth_range_m)
        confidence = np.asarray(sample.depth_confidence)
        if depth_range.dtype != np.float32 or confidence.dtype != np.float32:
            return None
        support = (depth_range != 0) | (confidence != 0)
        flat = np.flatnonzero(support)
        if flat.size * 12 < depth_range.size * 8:
            arrays["depth_index"] = flat.astype(np.int32)
            arrays["depth_range_values"] = depth_range.reshape(-1)[flat]
            arrays["depth_confidence_values"] = confidence.reshape(-1)[flat]
            meta["depth"] = "sparse"
        else:
            arrays["depth_range_m"] = depth_range
            arrays["depth_confidence"] = confidence
            meta["depth"] = "dense"
    if sample.mono_depth_range_m is not None:
        mono = np.asarray(sample.mono_depth_range_m)
        if mono.dtype != np.float32:
            return None
        arrays["mono_depth_range_m"] = mono
        meta["present"].append("mono_depth_range_m")
    if sample.depth_to_range_scale is not None:
        scale = np.asarray(sample.depth_to_range_scale)
        if sample.resolution_reduction > 1.0 and np.array_equal(
            scale, _range_scale(np.asarray(sample.K), shape)
        ):
            meta["range_scale"] = "from_intrinsics"
        else:
            arrays["depth_to_range_scale"] = scale
            meta["range_scale"] = "stored"
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    return arrays


def decode_sample(arrays: Any) -> TrainingSample:
    meta = json.loads(bytes(np.asarray(arrays["meta"])).decode("utf-8"))
    shape = (int(meta["height"]), int(meta["width"]))
    present = set(meta["present"])
    values: dict[str, Any] = {name: None for name in (*_BOOL_FIELDS, *_UNSUPPORTED)}
    for name in _BOOL_FIELDS:
        if name in present:
            values[name] = _unpack(np.asarray(arrays[f"bits_{name}"]), shape)
    depth_range = depth_confidence = None
    if meta.get("depth") == "sparse":
        index = np.asarray(arrays["depth_index"]).astype(np.int64)
        depth_range = np.zeros(shape, dtype=np.float32)
        depth_confidence = np.zeros(shape, dtype=np.float32)
        depth_range.reshape(-1)[index] = np.asarray(arrays["depth_range_values"])
        depth_confidence.reshape(-1)[index] = np.asarray(arrays["depth_confidence_values"])
    elif meta.get("depth") == "dense":
        depth_range = np.array(arrays["depth_range_m"])
        depth_confidence = np.array(arrays["depth_confidence"])
    K = np.array(arrays["K"])
    range_scale = None
    if meta.get("range_scale") == "from_intrinsics":
        range_scale = _range_scale(K, shape)
    elif meta.get("range_scale") == "stored":
        range_scale = np.array(arrays["depth_to_range_scale"])
    paths = {name: None if value is None else Path(value) for name, value in meta["paths"].items()}
    known = {field.name for field in fields(TrainingSample)}
    payload = dict(
        image_id=meta["image_id"],
        rig_frame_id=meta["rig_frame_id"],
        camera_id=meta["camera_id"],
        image=np.array(arrays["image"]),
        rgb_mask=values["rgb_mask"],
        depth_range_m=depth_range,
        depth_confidence=depth_confidence,
        depth_mask=values["depth_mask"],
        depth_cache_path=paths["depth_cache_path"],
        c2w=np.array(arrays["c2w"]),
        K=K,
        radial_coeffs=np.array(arrays["radial_coeffs"]),
        width=shape[1],
        height=shape[0],
        camera_model=meta["camera_model"],
        depth_to_range_scale=range_scale,
        sensor_resolution=None if meta["sensor_resolution"] is None else tuple(meta["sensor_resolution"]),
        mono_depth_range_m=(
            np.array(arrays["mono_depth_range_m"]) if "mono_depth_range_m" in present else None
        ),
        mono_depth_mask=values["mono_depth_mask"],
        mono_depth_cache_path=paths["mono_depth_cache_path"],
        mesh_geometry_cache_path=paths["mesh_geometry_cache_path"],
        sky_mask=values["sky_mask"],
        resolution_reduction=float(meta["resolution_reduction"]),
    )
    return TrainingSample(**{name: value for name, value in payload.items() if name in known})


class SampleCache:
    """Per-run store of decoded samples under ``root``, bounded by ``budget_bytes``.

    Thread-safe for the lookahead prefetcher: an archive is written to a temporary name and
    renamed, so a reader sees a whole file or none.
    """

    def __init__(
        self,
        root: Path,
        *,
        identity: Any,
        budget_bytes: int,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
        free_bytes: Any = None,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("sample cache budget must be positive")
        self.root = Path(root)
        self.budget_bytes = int(budget_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self._free_bytes = free_bytes or (lambda: shutil.disk_usage(self.root).free)
        self.identity = identity_sha256({"schema": SCHEMA_VERSION, "identity": identity})
        self._lock = threading.Lock()
        self._bytes = 0
        self._full = False
        self.hits = 0
        self.stored = 0
        self.skipped = 0
        self.read_errors = 0
        marker = self.root / IDENTITY_NAME
        recorded = None
        if marker.is_file():
            try:
                recorded = json.loads(marker.read_text(encoding="utf-8")).get("identity")
            except (OSError, ValueError):
                recorded = None
        if self.root.exists() and recorded != self.identity:
            # Another run's samples, or a half-written directory: never trusted.
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if recorded != self.identity:
            marker.write_text(
                json.dumps({"schema_version": SCHEMA_VERSION, "identity": self.identity}),
                encoding="utf-8",
            )
        existing = [path for path in self.root.glob("*.npz") if ".tmp" not in path.name]
        for leftover in self.root.glob("*.tmp.npz"):
            leftover.unlink(missing_ok=True)
        self._bytes = sum(path.stat().st_size for path in existing)
        # One writer per index: on Windows a rename onto a file another thread is reading
        # fails, so an index is claimed before it is written and never written twice.
        self._claimed = {int(path.stem) for path in existing}

    def _path(self, index: int) -> Path:
        return self.root / f"{int(index):07d}.npz"

    def load(self, index: int) -> TrainingSample | None:
        path = self._path(index)
        if not path.is_file():
            return None
        sample = None
        for attempt in range(3):
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    sample = decode_sample(arrays)
                break
            except OSError:
                # A scanner or indexer can hold a freshly written archive for a moment on
                # Windows; give it two short chances to let go.
                time.sleep(0.02 * (attempt + 1))
        if sample is None:
            # The cache is an optimisation: a read that keeps failing is a miss, and the
            # sample is decoded as if it had never been stored.
            with self._lock:
                self.read_errors += 1
            return None
        with self._lock:
            self.hits += 1
        return sample

    def store(self, index: int, sample: TrainingSample) -> bool:
        with self._lock:
            if self._full or int(index) in self._claimed:
                return False
            self._claimed.add(int(index))
        arrays = encode_sample(sample)
        if arrays is None:
            with self._lock:
                self.skipped += 1
            return False
        path = self._path(index)
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
        try:
            np.savez(temporary, **arrays)
            size = temporary.stat().st_size
        except OSError:
            # Disk full or the file is held: stop writing, never fail the run.
            temporary.unlink(missing_ok=True)
            with self._lock:
                self._full = True
            return False
        with self._lock:
            if (
                self._bytes + size > self.budget_bytes
                or self._free_bytes() < self.min_free_bytes
            ):
                self._full = True
                keep = False
            else:
                self._bytes += size
                self.stored += 1
                keep = True
        if keep:
            try:
                os.replace(temporary, path)
            except OSError:
                keep = False
                with self._lock:
                    self._bytes -= size
                    self.stored -= 1
        if not keep:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return keep

    def fetch(self, index: int, decode: Any) -> TrainingSample:
        """The sample for ``index``: from the cache, else decoded and stored."""
        cached = self.load(index)
        if cached is not None:
            return cached
        sample = decode(index)
        self.store(index, sample)
        return sample

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "stored": self.stored,
                "skipped": self.skipped,
                "read_errors": self.read_errors,
                "bytes": self._bytes,
                "full": int(self._full),
            }

    def remove(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
