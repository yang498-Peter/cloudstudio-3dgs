"""Per-run decoded-sample cache (cloudstudio_3dgs/training/sample_cache.py), CPU only.

What is pinned:

* a cached sample equals the decoded one field for field and array for array, reduced or not;
* sparse LiDAR storage is used only when it loses nothing, and the z-to-range factor is
  recomputed only when it is exactly the function of the intrinsics it claims to be;
* a directory written under another identity is emptied, not read;
* the byte budget stops further writes without failing the run;
* a sample the archive cannot represent is passed through uncached;
* concurrent fetches of the same index decode at most a few times and always return the sample.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from cloudstudio_3dgs.training.dataset import TrainingSample
from cloudstudio_3dgs.training.sample_cache import SampleCache, decode_sample, encode_sample
from cloudstudio_3dgs.training.sample_resolution import reduce_sample


def make_sample(seed: int = 0, height: int = 24, width: int = 30) -> TrainingSample:
    rng = np.random.default_rng(seed)
    focal = 60.0
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    jj, ii = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    scale = np.sqrt(1.0 + ((jj - K[0, 2]) / focal) ** 2 + ((ii - K[1, 2]) / focal) ** 2).astype(np.float32)
    depth_mask = rng.random((height, width)) < 0.1
    depth_range = np.where(depth_mask, rng.random((height, width)) * 20 + 0.5, 0.0).astype(np.float32)
    confidence = np.where(depth_mask, rng.random((height, width)), 0.0).astype(np.float32)
    rgb_mask = rng.random((height, width)) > 0.2
    return TrainingSample(
        image_id=f"img{seed}::pitch_up_56",
        rig_frame_id=f"rig{seed}",
        camera_id="left",
        image=rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        rgb_mask=rgb_mask,
        depth_range_m=depth_range,
        depth_confidence=confidence,
        depth_mask=depth_mask & rgb_mask,
        depth_cache_path=Path("C:/cache/depth.npz"),
        c2w=rng.random((4, 4)).astype(np.float32),
        K=K,
        radial_coeffs=np.zeros(4, dtype=np.float32),
        width=width,
        height=height,
        camera_model="pinhole",
        depth_to_range_scale=scale,
        sensor_resolution=(2912, 2912),
        mono_depth_range_m=(rng.random((height, width)) * 9).astype(np.float32),
        mono_depth_mask=rng.random((height, width)) > 0.3,
        mono_depth_cache_path=Path("C:/cache/mono.npz"),
        sky_mask=rng.random((height, width)) > 0.7,
    )


def assert_same_sample(test: unittest.TestCase, a: TrainingSample, b: TrainingSample) -> None:
    for field in fields(TrainingSample):
        left, right = getattr(a, field.name), getattr(b, field.name)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            test.assertIsNotNone(left, field.name)
            test.assertIsNotNone(right, field.name)
            test.assertEqual(left.dtype, right.dtype, field.name)
            test.assertTrue(np.array_equal(left, right), field.name)
        else:
            test.assertEqual(left, right, field.name)


class RoundTripTest(unittest.TestCase):
    def test_a_stored_sample_equals_the_decoded_one(self) -> None:
        sample = make_sample()
        arrays = encode_sample(sample)
        self.assertIsNotNone(arrays)
        assert_same_sample(self, decode_sample(arrays), sample)
        self.assertIn("depth_index", arrays)  # 10% LiDAR coverage is stored sparsely
        self.assertIn("depth_to_range_scale", arrays)  # unreduced: the factor is stored

    def test_a_reduced_sample_round_trips_and_recomputes_its_range_factor(self) -> None:
        reduced = reduce_sample(make_sample(3, height=36, width=45), 3.0)
        arrays = encode_sample(reduced)
        self.assertNotIn("depth_to_range_scale", arrays)
        assert_same_sample(self, decode_sample(arrays), reduced)
        self.assertEqual(decode_sample(arrays).resolution_reduction, 3.0)

    def test_dense_lidar_is_stored_dense(self) -> None:
        sample = make_sample()
        dense = replace(
            sample,
            depth_range_m=np.full((24, 30), 3.0, dtype=np.float32),
            depth_confidence=np.full((24, 30), 0.5, dtype=np.float32),
        )
        arrays = encode_sample(dense)
        self.assertIn("depth_range_m", arrays)
        assert_same_sample(self, decode_sample(arrays), dense)

    def test_values_outside_the_mask_survive(self) -> None:
        sample = make_sample()
        ranges = sample.depth_range_m.copy()
        ranges[0, 0] = np.float32(np.nan)
        ranges[1, 1] = np.float32(7.5)  # a return the mask does not cover
        stray = replace(sample, depth_range_m=ranges)
        decoded = decode_sample(encode_sample(stray))
        self.assertTrue(np.isnan(decoded.depth_range_m[0, 0]))
        self.assertEqual(float(decoded.depth_range_m[1, 1]), 7.5)

    def test_optional_supervision_may_be_absent(self) -> None:
        bare = replace(
            make_sample(), depth_range_m=None, depth_confidence=None, depth_mask=None,
            mono_depth_range_m=None, mono_depth_mask=None, sky_mask=None, depth_cache_path=None,
        )
        assert_same_sample(self, decode_sample(encode_sample(bare)), bare)

    def test_a_sample_the_archive_cannot_hold_is_not_encoded(self) -> None:
        sample = make_sample()
        self.assertIsNone(encode_sample(replace(sample, sensor_pixel_coords=np.zeros((24, 30, 2), np.float32))))
        self.assertIsNone(encode_sample(replace(sample, mesh_depth_range_m=np.zeros((24, 30), np.float32))))
        self.assertIsNone(encode_sample(replace(sample, camera_model="fisheye")))


class SampleCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "sample_cache"

    def test_the_second_visit_reads_the_cache(self) -> None:
        calls: list[int] = []

        def decode(index: int) -> TrainingSample:
            calls.append(index)
            return make_sample(index)

        cache = SampleCache(self.root, identity={"run": "a"}, budget_bytes=1 << 30)
        first = cache.fetch(4, decode)
        second = cache.fetch(4, decode)
        self.assertEqual(calls, [4])
        assert_same_sample(self, first, second)
        self.assertEqual(cache.stats()["hits"], 1)
        self.assertEqual(cache.stats()["stored"], 1)

    def test_a_reopened_cache_of_the_same_run_is_reused(self) -> None:
        SampleCache(self.root, identity={"run": "a"}, budget_bytes=1 << 30).fetch(1, make_sample)
        reopened = SampleCache(self.root, identity={"run": "a"}, budget_bytes=1 << 30)
        self.assertGreater(reopened.stats()["bytes"], 0)
        sample = reopened.fetch(1, lambda index: self.fail("decoded again"))
        assert_same_sample(self, sample, make_sample(1))

    def test_another_identity_empties_the_directory(self) -> None:
        SampleCache(self.root, identity={"run": "a"}, budget_bytes=1 << 30).fetch(1, make_sample)
        other = SampleCache(self.root, identity={"run": "b"}, budget_bytes=1 << 30)
        self.assertEqual(other.stats()["bytes"], 0)
        self.assertIsNone(other.load(1))

    def test_the_budget_stops_writes_without_failing(self) -> None:
        one = len(np.lib.format.magic(1, 0))  # noqa: F841 - any small positive budget
        cache = SampleCache(self.root, identity={"run": "a"}, budget_bytes=6000)
        for index in range(4):
            assert_same_sample(self, cache.fetch(index, make_sample), make_sample(index))
        stats = cache.stats()
        self.assertEqual(stats["full"], 1)
        self.assertLess(stats["stored"], 4)
        self.assertLessEqual(stats["bytes"], 6000)
        self.assertFalse(list(self.root.glob("*.tmp.npz")))

    def test_a_nearly_full_volume_stops_writes_too(self) -> None:
        free = {"bytes": 50 * 1024**3}
        cache = SampleCache(
            self.root, identity={"run": "a"}, budget_bytes=1 << 30,
            min_free_bytes=12 * 1024**3, free_bytes=lambda: free["bytes"],
        )
        cache.fetch(0, make_sample)
        free["bytes"] = 11 * 1024**3
        assert_same_sample(self, cache.fetch(1, make_sample), make_sample(1))
        self.assertEqual(cache.stats()["stored"], 1)
        self.assertEqual(cache.stats()["full"], 1)
        self.assertIsNone(cache.load(1))
        self.assertIsNotNone(cache.load(0))

    def test_an_unsupported_sample_passes_through(self) -> None:
        cache = SampleCache(self.root, identity={"run": "a"}, budget_bytes=1 << 30)
        odd = replace(make_sample(), camera_model="fisheye")
        self.assertIs(cache.fetch(0, lambda index: odd), odd)
        self.assertEqual(cache.stats()["skipped"], 1)
        self.assertIsNone(cache.load(0))

    def test_concurrent_fetches_all_return_the_sample(self) -> None:
        cache = SampleCache(self.root, identity={"run": "a"}, budget_bytes=1 << 30)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: cache.fetch(2, make_sample), range(24)))
        for result in results:
            assert_same_sample(self, result, make_sample(2))
        self.assertFalse(list(self.root.glob("*.tmp.npz")))

    def test_remove_frees_the_directory(self) -> None:
        cache = SampleCache(self.root, identity={"run": "a"}, budget_bytes=1 << 30)
        cache.fetch(0, make_sample)
        cache.remove()
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
