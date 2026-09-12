from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cloudstudio_3dgs.training.view_backgrounds import (
    ViewBackgroundLibrary,
    write_view_background_manifest,
)


class ViewBackgroundTests(unittest.TestCase):
    def test_roundtrip_resize_and_fail_closed_missing_view(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stored = np.zeros((8, 6, 3), dtype=np.uint8)
            stored[:, :3] = (255, 0, 0)
            Image.fromarray(stored).save(root / "a.png")
            manifest = root / "manifest.json"
            write_view_background_manifest(
                manifest,
                views={"view::a": {"file": "a.png", "height": 8, "width": 6}},
                metadata={"split": "train", "downsample": 4},
            )

            library = ViewBackgroundLibrary(manifest, root, device="cpu")
            background = library.background_for(
                "view::a", height=32, width=24, torch=torch
            )
            self.assertEqual(tuple(background.shape), (32, 24, 3))
            self.assertGreater(float(background[:, :6, 0].mean()), 0.9)
            self.assertLess(float(background[:, -6:, 0].mean()), 0.1)

            with self.assertRaises(ValueError):
                library.background_for("view::b", height=8, width=6, torch=torch)

    def test_tampered_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            write_view_background_manifest(
                manifest,
                views={"v": {"file": "v.png", "height": 2, "width": 2}},
                metadata={"split": "train"},
            )
            text = manifest.read_text(encoding="utf-8").replace('"train"', '"val"')
            manifest.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                ViewBackgroundLibrary(manifest, root, device="cpu")


if __name__ == "__main__":
    unittest.main()


class BoundedCacheTests(unittest.TestCase):
    """The stand-in libraries are ~10 MB per view; an unbounded cache would hold 19 GiB."""

    def _library(self, tmp, budget):
        import json
        import numpy as np
        from PIL import Image
        from cloudstudio_3dgs.training.view_backgrounds import ViewBackgroundLibrary, _sha256_bytes

        root = Path(tmp)
        views = {}
        for i in range(6):
            name = f"v{i}.png"
            Image.fromarray(np.full((32, 32, 3), i * 10, np.uint8)).save(root / name)
            views[f"img_{i}"] = {"file": name, "height": 32, "width": 32}
        body = {"schema_version": 1, "views": views}
        body["manifest_sha256"] = _sha256_bytes(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        (root / "m.json").write_text(json.dumps(body), encoding="utf-8")
        return ViewBackgroundLibrary(root / "m.json", root, device="cpu", cache_budget_bytes=budget)

    def test_cache_is_evicted_once_the_budget_is_exceeded(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            one = 32 * 32 * 3
            library = self._library(tmp, budget=2 * one)
            for i in range(4):
                library.background_for(f"img_{i}", height=32, width=32, torch=torch)
            self.assertLessEqual(library._cache_bytes, 2 * one)
            self.assertLessEqual(len(library._cache), 2)
            # the most recent view is still cached, the oldest is gone
            self.assertIn("img_3", library._cache)
            self.assertNotIn("img_0", library._cache)

    def test_zero_budget_disables_caching_but_still_serves(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            library = self._library(tmp, budget=0)
            first = library.background_for("img_1", height=32, width=32, torch=torch)
            second = library.background_for("img_1", height=32, width=32, torch=torch)
            self.assertEqual(len(library._cache), 0)
            self.assertTrue(bool((first == second).all()))
