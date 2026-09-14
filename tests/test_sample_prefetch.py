"""One-deep sample prefetcher (cloudstudio_3dgs/training/sample_prefetch.py), CPU only.

What is pinned:

* the sample handed back for an index is exactly what the fetch function returns, in the
  order the caller asks for it, whether or not the prime guessed right;
* only the worker thread ever calls the fetch function, so a dataset's write-once memo
  caches never see two threads;
* a fetch that raises does not raise on the step that primed it - the exception is
  re-raised from ``get`` for that index, and a run that never asks for the index never
  sees it;
* a prime for a different index discards the unclaimed one instead of deadlocking;
* the sequence of fetched indices under a realistic drive loop matches the synchronous
  sequence exactly.
"""

from __future__ import annotations

import pathlib
import threading
import time
import unittest

from cloudstudio_3dgs.training.sample_prefetch import SamplePrefetcher


class Recorder:
    def __init__(self, delay: float = 0.0, fail_on: set[int] | None = None):
        self.delay = delay
        self.fail_on = fail_on or set()
        self.calls: list[int] = []
        self.threads: set[str] = set()
        self._lock = threading.Lock()

    def __call__(self, index: int):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.calls.append(index)
            self.threads.add(threading.current_thread().name)
        if index in self.fail_on:
            raise ValueError("artifact check failed for index %d" % index)
        return ("sample", index)


class SamplePrefetchTest(unittest.TestCase):
    def test_drive_loop_matches_the_synchronous_sequence(self):
        order = [5, 2, 9, 2, 0, 7]
        rec = Recorder(delay=0.002)
        got = []
        with SamplePrefetcher(rec) as pf:
            pf.prime(order[0])
            for i, index in enumerate(order):
                got.append(pf.get(index))
                if i + 1 < len(order):
                    pf.prime(order[i + 1])
        self.assertEqual(got, [("sample", i) for i in order])
        self.assertEqual(rec.calls, order)
        self.assertEqual(len(rec.threads), 1)
        self.assertNotIn(threading.current_thread().name, rec.threads)

    def test_a_wrong_guess_still_returns_the_right_sample(self):
        rec = Recorder(delay=0.002)
        with SamplePrefetcher(rec) as pf:
            pf.prime(11)          # guess
            self.assertEqual(pf.get(4), ("sample", 4))   # caller wanted something else
            self.assertEqual(pf.get(11), ("sample", 11))
        self.assertIn(4, rec.calls)
        self.assertIn(11, rec.calls)

    def test_get_without_any_prime_still_works(self):
        rec = Recorder()
        with SamplePrefetcher(rec) as pf:
            self.assertEqual(pf.get(3), ("sample", 3))
            self.assertEqual(pf.stats()["hits"], 0)
            self.assertEqual(pf.stats()["misses"], 1)

    def test_hits_are_counted_when_the_guess_is_right(self):
        rec = Recorder(delay=0.002)
        with SamplePrefetcher(rec) as pf:
            pf.prime(1)
            # give the worker time to finish so the get is a genuine hit
            for _ in range(200):
                if pf._result is not None:
                    break
                time.sleep(0.002)
            pf.get(1)
            self.assertEqual(pf.stats()["hits"], 1)
            self.assertEqual(pf.stats()["misses"], 0)

    def test_a_failing_prefetch_raises_only_when_that_index_is_asked_for(self):
        rec = Recorder(delay=0.001, fail_on={7})
        with SamplePrefetcher(rec) as pf:
            pf.prime(7)
            # the step that primed index 7 keeps running: asking for 6 must not raise
            self.assertEqual(pf.get(6), ("sample", 6))
            pf.prime(7)
            with self.assertRaises(ValueError):
                pf.get(7)

    def test_a_failing_prefetch_is_never_seen_if_the_run_stops_first(self):
        rec = Recorder(delay=0.001, fail_on={7})
        pf = SamplePrefetcher(rec)
        try:
            self.assertEqual(pf.get(6), ("sample", 6))
            pf.prime(7)          # the run stops here; index 7 is never consumed
            time.sleep(0.05)
        finally:
            pf.close()           # must not raise
        self.assertIn(7, rec.calls)

    def test_repeated_primes_for_the_same_index_fetch_once(self):
        rec = Recorder(delay=0.002)
        with SamplePrefetcher(rec) as pf:
            pf.prime(8)
            pf.prime(8)
            pf.prime(8)
            self.assertEqual(pf.get(8), ("sample", 8))
        self.assertEqual(rec.calls.count(8), 1)

    def test_close_is_idempotent_and_blocks_further_use(self):
        pf = SamplePrefetcher(Recorder())
        pf.close()
        pf.close()
        with self.assertRaises(RuntimeError):
            pf.get(0)


class EpochOrderCacheTest(unittest.TestCase):
    def test_sequence_matches_shuffling_each_epoch_directly(self):
        from cloudstudio_3dgs.training.sample_prefetch import EpochOrderCache
        from cloudstudio_3dgs.training.trainer import fisher_yates_epoch_order

        length, seed = 7, 42
        cache = EpochOrderCache(fisher_yates_epoch_order, length, seed)
        direct = [
            fisher_yates_epoch_order(length, seed=seed, epoch=e) for e in range(4)
        ]
        expected = [index for order in direct for index in order]
        got = [cache.index_for_step(s) for s in range(length * 4)]
        self.assertEqual(got, expected)

    def test_looking_one_step_past_an_epoch_boundary_does_not_thrash(self):
        from cloudstudio_3dgs.training.sample_prefetch import EpochOrderCache
        from cloudstudio_3dgs.training.trainer import fisher_yates_epoch_order

        length = 5
        cache = EpochOrderCache(fisher_yates_epoch_order, length, 42)
        # the drive loop: index for this step, then the index for the next one
        for step in range(length * 3):
            cache.index_for_step(step)
            cache.index_for_step(step + 1)
        # one permutation per epoch touched (epochs 0..3), not one per step
        self.assertEqual(cache.shuffles, 4)

    def test_a_zero_length_dataset_is_rejected(self):
        from cloudstudio_3dgs.training.sample_prefetch import EpochOrderCache
        from cloudstudio_3dgs.training.trainer import fisher_yates_epoch_order

        with self.assertRaises(ValueError):
            EpochOrderCache(fisher_yates_epoch_order, 0, 42)


# The contract sha below is bound to one real as-run config, so these cases only run on a
# machine that still has that run directory. Everything they pin is about the contract, not
# about the machine.
AS_RUN = pathlib.Path(
    r"C:/Peter/3dgs-runs/house0305_sop/tile1_B5_cap6_20k/config_as_run.json"
)


@unittest.skipUnless(AS_RUN.is_file(), "needs the recorded tile1_B5_cap6_20k run directory")
class PrefetchConfigContractTest(unittest.TestCase):
    def _config(self, **overrides):
        import json
        from cloudstudio_3dgs.training.trainer import TrainerConfig

        raw = json.loads(AS_RUN.read_text(encoding="utf-8"))
        raw.update(overrides)
        return TrainerConfig.from_dict(raw)

    def test_the_contract_is_byte_identical_when_prefetch_is_off(self):
        import hashlib
        from cloudstudio_3dgs.data.manifest import canonical_json_bytes

        config = self._config()
        config.validate()
        self.assertFalse(config.prefetch_training_samples)
        contract = config.contract_dict()
        self.assertNotIn("prefetch_training_samples", contract["view_sampling"])
        # the recorded delivery arm's contract, recomputed before this field existed
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(contract)).hexdigest(),
            "7d4139bc7450b3d43929180031841399f2f132a0bcdd22d355bc2c5c99a1c971",
        )

    def test_enabling_prefetch_adds_exactly_one_contract_key(self):
        config = self._config(prefetch_training_samples=True)
        config.validate()
        contract = config.contract_dict()
        self.assertTrue(contract["view_sampling"]["prefetch_training_samples"])
        baseline = self._config().contract_dict()
        self.assertEqual(
            set(contract["view_sampling"]) - set(baseline["view_sampling"]),
            {"prefetch_training_samples"},
        )

    def test_prefetch_is_refused_for_sampling_with_replacement(self):
        config = self._config(
            prefetch_training_samples=True, view_sampling_mode="with_replacement"
        )
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
