from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tools.export_gaussian_ply import export_checkpoint_ply


def _vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(4096).split(b"end_header\n", 1)[0].decode("ascii")
    line = next(item for item in header.splitlines() if item.startswith("element vertex "))
    return int(line.rsplit(" ", 1)[1])


class ExportGaussianPlyLayerTests(unittest.TestCase):
    def make_checkpoint(self, path: Path, *, with_sky_metadata: bool = True) -> None:
        count = 5
        payload = {
            "params": {
                "means": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
                "scales": torch.zeros((count, 3)),
                "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
                "opacities": torch.zeros((count,)),
                "sh0": torch.zeros((count, 1, 3)),
                "shN": torch.zeros((count, 0, 3)),
            }
        }
        if with_sky_metadata:
            payload["sky_layer"] = {
                "sky_gaussian_start": 3,
                "sky_gaussian_count": 2,
            }
        torch.save(payload, path)

    def test_exports_surface_sky_and_combined_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            self.make_checkpoint(checkpoint)
            expected = {"surface": 3, "sky": 2, "all": 5}
            for layer, count in expected.items():
                output = root / f"{layer}.ply"
                report = export_checkpoint_ply(checkpoint, output, layer=layer)
                self.assertEqual(report["layer"], layer)
                self.assertEqual(report["gaussians_written"], count)
                self.assertEqual(_vertex_count(output), count)

    def test_partition_export_fails_without_signed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            self.make_checkpoint(checkpoint, with_sky_metadata=False)
            with self.assertRaisesRegex(ValueError, "sky_layer metadata"):
                export_checkpoint_ply(checkpoint, root / "sky.ply", layer="sky")

    def test_partition_boundary_must_match_parameter_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            self.make_checkpoint(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["sky_layer"]["sky_gaussian_count"] = 3
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "boundary is inconsistent"):
                export_checkpoint_ply(checkpoint, root / "surface.ply", layer="surface")


if __name__ == "__main__":
    unittest.main()
