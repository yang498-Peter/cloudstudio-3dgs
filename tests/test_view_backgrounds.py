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
