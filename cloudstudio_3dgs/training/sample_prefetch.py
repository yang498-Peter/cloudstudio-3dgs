"""One-deep background fetch for training samples.

Measured on ``tile1_B5_cap6_20k``: a step costs about 350 ms and is flat against gaussian
count, and ``trainset[index]`` alone costs about 161 ms of it - six artifacts per view
(face RGB PNG, renderer mask, sky mask, LiDAR sparse depth, DA2 depth, ownership pair)
decoded from scratch on the training thread, between the previous step's last device
readback and this step's render. Nothing about it overlaps the GPU, which sits at 36 mean
utilisation while it happens.

Under ``fisher_yates_without_replacement_per_epoch`` the next index is a pure function of
the step, the seed and the dataset length, so the fetch can start one step early. PIL
decode and zlib inflate both release the GIL, so a single worker thread is enough.

What makes this safe to run with bit-identical results:

* the worker is the ONLY caller of ``dataset[index]``, so the dataset's own memo caches
  (verified paths, per-face warp grids) are never touched by two threads;
* ``__getitem__`` is a pure function of the index - its caches are write-once and keyed,
  so it does not matter which step first touches one;
* a fetch that raises is not raised on the prefetching step. The exception is held and
  re-raised from ``get`` for that index, which is the step that would have raised it
  without prefetching. A run that stops before consuming the sample never sees it.

Prefetching does not touch the sampling order, the RNG, the revisit bookkeeping or any
tensor; only the wall-clock moment of decoding moves.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional


class SamplePrefetcher:
    """Fetch one dataset sample ahead on a worker thread.

    The caller drives it: ``get(index)`` for the sample this step needs, then
    ``prime(next_index)`` to start the next one. A ``prime`` for an index that is already
    in flight or already fetched is free; a ``prime`` for a different index discards the
    unclaimed one.
    """

    def __init__(self, fetch: Callable[[int], Any], *, name: str = "sample-prefetch") -> None:
        self._fetch = fetch
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._requested: Optional[int] = None
        self._result: Optional[tuple[int, Any, Optional[BaseException]]] = None
        self._closed = False
        self.hits = 0
        self.misses = 0
        self.discarded = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- worker -----------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._wake:
                while not self._closed and (self._requested is None or self._result is not None):
                    self._wake.wait()
                if self._closed:
                    return
                index = self._requested
                self._requested = None
            assert index is not None
            sample: Any = None
            error: Optional[BaseException] = None
            try:
                sample = self._fetch(index)
            except BaseException as exc:  # re-raised from get() for this index
                error = exc
            with self._wake:
                # A close() or a prime() for a different index while this fetch was in
                # flight makes the result unwanted; drop it rather than block the slot.
                if self._closed:
                    return
                if self._requested is not None and self._requested != index:
                    self.discarded += 1
                else:
                    self._result = (index, sample, error)
                self._wake.notify_all()

    # -- caller -----------------------------------------------------------------

    def prime(self, index: int) -> None:
        """Start fetching ``index`` unless it is already in flight or already fetched."""
        with self._wake:
            if self._closed:
                return
            if self._result is not None and self._result[0] == index:
                return
            if self._requested == index:
                return
            if self._result is not None:
                # An unclaimed sample for some other index: the caller changed its mind.
                self._result = None
                self.discarded += 1
            self._requested = index
            self._wake.notify_all()

    def get(self, index: int) -> Any:
        """Return the sample for ``index``, waiting for the worker if it is not ready."""
        with self._wake:
            if self._closed:
                raise RuntimeError("prefetcher is closed")
            ready = self._result is not None and self._result[0] == index
            if ready:
                self.hits += 1
            else:
                self.misses += 1
                if self._result is not None:
                    self._result = None
                    self.discarded += 1
                if self._requested != index:
                    self._requested = index
                    self._wake.notify_all()
            while self._result is None or self._result[0] != index:
                if self._closed:
                    raise RuntimeError("prefetcher is closed")
                self._wake.wait()
            _, sample, error = self._result
            self._result = None
            self._wake.notify_all()
        if error is not None:
            raise error
        return sample

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._result = None
            self._requested = None
            self._wake.notify_all()
        self._thread.join(timeout=30.0)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "discarded": self.discarded}

    def __enter__(self) -> "SamplePrefetcher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class EpochOrderCache:
    """Epoch permutations for the step being run and the one being fetched ahead.

    The loop used to hold a single epoch's order and reshuffle whenever the epoch
    changed. Looking one step ahead crosses the epoch boundary one step early, which
    would make that single slot thrash. Holding the current and previous epoch costs one
    extra permutation and keeps the shuffle count at one per epoch.

    The permutation is a pure function of ``(length, seed, epoch)``, so which step first
    asks for it cannot change what it contains.
    """

    def __init__(self, order_fn: Callable[..., tuple[int, ...]], length: int, seed: int) -> None:
        if length <= 0:
            raise ValueError("dataset length must be positive")
        self._order_fn = order_fn
        self._length = int(length)
        self._seed = int(seed)
        self._orders: dict[int, tuple[int, ...]] = {}
        self.shuffles = 0

    def order(self, epoch: int) -> tuple[int, ...]:
        order = self._orders.get(epoch)
        if order is None:
            order = tuple(self._order_fn(self._length, seed=self._seed, epoch=epoch))
            self._orders[epoch] = order
            self.shuffles += 1
            for stale in [key for key in self._orders if key < epoch - 1]:
                del self._orders[stale]
        return order

    def index_for_step(self, step: int) -> int:
        return self.order(step // self._length)[step % self._length]
