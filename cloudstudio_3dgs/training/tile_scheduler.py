"""Strictly serial tile-training lifecycle and CUDA cache policy."""

from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.adaptive_tiling import verify_adaptive_tile_plan


@dataclass(frozen=True)
class TileMemoryPolicy:
    empty_cache_interval_steps: int = 2
    synchronize_before_empty_cache: bool = True
    collect_python_garbage_after_tile: bool = True

    def validate(self) -> None:
        if self.empty_cache_interval_steps <= 0:
            raise ValueError("CUDA cache interval must be positive")


def release_cuda_cache(
    cuda: Any,
    *,
    synchronize: bool,
) -> None:
    """Release cached blocks without assuming CUDA is present."""

    if cuda is None or not bool(cuda.is_available()):
        return
    if synchronize:
        cuda.synchronize()
    cuda.empty_cache()


def maybe_release_cuda_cache(
    cuda: Any,
    *,
    step: int,
    policy: TileMemoryPolicy,
) -> bool:
    """Apply the recovered even-step cache policy and report whether it ran."""

    policy.validate()
    if step < 0:
        raise ValueError("training step must be non-negative")
    if step % policy.empty_cache_interval_steps:
        return False
    release_cuda_cache(
        cuda,
        synchronize=policy.synchronize_before_empty_cache,
    )
    return cuda is not None and bool(cuda.is_available())


def run_tiles_serially(
    tile_plan: dict[str, Any],
    train_tile: Callable[[dict[str, Any], Callable[[int], bool]], dict[str, Any]],
    *,
    cuda: Any = None,
    policy: TileMemoryPolicy = TileMemoryPolicy(),
    include_discarded: bool = False,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Train one tile at a time and return a signed lifecycle trace.

    The callback receives the tile record and a step hook.  It must call the
    hook after each completed optimizer step; the hook owns the periodic CUDA
    cache release.  Only one callback invocation can be active at a time.
    """

    plan_sha = verify_adaptive_tile_plan(tile_plan)
    policy.validate()
    events: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    active_tile: int | None = None
    started = monotonic()
    for tile in tile_plan["tiles"]:
        if tile.get("low_support_discarded") and not include_discarded:
            events.append(
                {
                    "event": "tile_skipped_low_support",
                    "tile_id": int(tile["tile_id"]),
                }
            )
            continue
        if active_tile is not None:
            raise RuntimeError("strict serial tile contract was violated")
        tile_id = int(tile["tile_id"])
        active_tile = tile_id
        tile_started = monotonic()
        events.append({"event": "tile_start", "tile_id": tile_id})

        def step_hook(step: int) -> bool:
            released = maybe_release_cuda_cache(
                cuda,
                step=step,
                policy=policy,
            )
            if released:
                events.append(
                    {
                        "event": "cuda_cache_release",
                        "tile_id": tile_id,
                        "step": int(step),
                    }
                )
            return released

        try:
            result = train_tile(tile, step_hook)
            if not isinstance(result, dict):
                raise TypeError("tile trainer callback must return a dictionary")
            results.append({"tile_id": tile_id, **result})
            events.append(
                {
                    "event": "tile_complete",
                    "tile_id": tile_id,
                    "elapsed_seconds": monotonic() - tile_started,
                }
            )
        finally:
            active_tile = None
            if policy.collect_python_garbage_after_tile:
                gc.collect()
            release_cuda_cache(
                cuda,
                synchronize=policy.synchronize_before_empty_cache,
            )
            events.append({"event": "post_tile_cache_release", "tile_id": tile_id})

    payload = {
        "schema_version": 1,
        "kind": "strict_serial_tile_training_trace_v1",
        "tile_plan_manifest_sha256": plan_sha,
        "policy": {
            "empty_cache_interval_steps": policy.empty_cache_interval_steps,
            "synchronize_before_empty_cache": policy.synchronize_before_empty_cache,
            "collect_python_garbage_after_tile": policy.collect_python_garbage_after_tile,
        },
        "strict_serial_execution": True,
        "completed_tile_count": len(results),
        "elapsed_seconds": monotonic() - started,
        "events": events,
        "results": results,
    }
    payload["tile_training_trace_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def assert_serial_trace(trace: dict[str, Any]) -> None:
    """Fail if a lifecycle trace overlaps tiles or omits post-tile release."""

    active: int | None = None
    completed: set[int] = set()
    released: set[int] = set()
    for event in trace.get("events", []):
        tile_id = int(event["tile_id"])
        if event["event"] == "tile_start":
            if active is not None:
                raise ValueError("tile lifecycle trace contains overlapping tile starts")
            active = tile_id
        elif event["event"] == "tile_complete":
            if active != tile_id:
                raise ValueError("tile completed while another tile was active")
            active = None
            completed.add(tile_id)
        elif event["event"] == "post_tile_cache_release":
            released.add(tile_id)
    if active is not None:
        raise ValueError("tile lifecycle trace ended with an active tile")
    if not completed.issubset(released):
        raise ValueError("one or more completed tiles lack a post-tile cache release")
