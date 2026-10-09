from __future__ import annotations

import tempfile
import unittest
from unittest import mock
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from tools.export_gaussian_ply import export_checkpoint_ply
from tools.gaussian_health import read_ply_records


def _vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(4096).split(b"end_header\n", 1)[0].decode("ascii")
    line = next(item for item in header.splitlines() if item.startswith("element vertex "))
    return int(line.rsplit(" ", 1)[1])


class ExportGaussianPlyLayerTests(unittest.TestCase):
    def make_checkpoint(self, path: Path, *, with_sky_metadata: bool = True) -> None:
        count = 5
        payload = {
            "identity": {"dataset_manifest_sha256": "dataset-sha"},
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

    def make_sky_report(self, path: Path, *, total: int = 5) -> None:
        report = {
            "dataset_manifest_sha256": "dataset-sha",
            "sky_gaussian_start": 3,
            "sky_gaussian_count": 2,
            "total_gaussian_count": total,
        }
        report["sky_layer_report_sha256"] = hashlib.sha256(
            canonical_json_bytes(report)
        ).hexdigest()
        path.write_text(json.dumps(report), encoding="utf-8")

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

    def test_signed_augmentation_report_recovers_warm_start_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            report_path = root / "sky.json"
            self.make_checkpoint(checkpoint, with_sky_metadata=False)
            self.make_sky_report(report_path)
            result = export_checkpoint_ply(
                checkpoint,
                root / "sky.ply",
                layer="sky",
                sky_layer_report=report_path,
            )
            self.assertEqual(result["gaussians_written"], 2)

    def test_tampered_or_wrong_count_sky_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            report_path = root / "sky.json"
            self.make_checkpoint(checkpoint, with_sky_metadata=False)
            self.make_sky_report(report_path, total=6)
            with self.assertRaisesRegex(ValueError, "total does not match"):
                export_checkpoint_ply(
                    checkpoint,
                    root / "sky.ply",
                    layer="sky",
                    sky_layer_report=report_path,
                )
            self.make_sky_report(report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["sky_gaussian_start"] = 2
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                export_checkpoint_ply(
                    checkpoint,
                    root / "sky.ply",
                    layer="sky",
                    sky_layer_report=report_path,
                )

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

    def test_surface_scale_cap_does_not_shrink_independent_sky(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            output = root / "capped.ply"
            self.make_checkpoint(checkpoint)
            report = export_checkpoint_ply(
                checkpoint,
                output,
                surface_gaussian_count=3,
                max_surface_scale_m=0.08,
            )
            records = read_ply_records(output)
            self.assertEqual(report["surface_scale_values_capped"], 9)
            self.assertAlmostEqual(float(records["scale_0"][0]), math.log(0.08), places=6)
            self.assertEqual(float(records["scale_0"][3]), 0.0)



class CountOnlyTests(unittest.TestCase):
    """--count-thresholds must count exactly the rows an export at that threshold writes."""

    def make_checkpoint(self, path: Path) -> None:
        count = 9
        # Opacities straddling the control thresholds, including values on a threshold.
        probabilities = torch.tensor([0.001, 0.009, 0.01, 0.02, 0.049, 0.05, 0.3, 0.9, 0.999])
        payload = {
            "identity": {"dataset_manifest_sha256": "dataset-sha"},
            "sky_layer": {"sky_gaussian_start": 6, "sky_gaussian_count": 3},
            "params": {
                "means": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
                "scales": torch.zeros((count, 3)),
                "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
                "opacities": torch.logit(probabilities),
                "sh0": torch.zeros((count, 1, 3)),
                "shN": torch.zeros((count, 0, 3)),
            },
        }
        torch.save(payload, path)

    def test_counts_equal_the_vertex_counts_of_real_exports(self) -> None:
        from tools.export_gaussian_ply import count_checkpoint_rows

        thresholds = [0.0, 0.01, 0.05]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            self.make_checkpoint(checkpoint)
            for layer in ("all", "surface", "sky"):
                counted = count_checkpoint_rows(checkpoint, thresholds, layer=layer)
                self.assertEqual(counted["gaussians_total"], 9)
                for threshold, row in zip(thresholds, counted["counts"]):
                    output = root / f"{layer}_{threshold:g}.ply"
                    report = export_checkpoint_ply(checkpoint, output, layer=layer, min_opacity=threshold)
                    self.assertEqual(row["min_opacity"], threshold)
                    self.assertEqual(row["gaussians_written"], report["gaussians_written"], (layer, threshold))
                    self.assertEqual(row["gaussians_written"], _vertex_count(output), (layer, threshold))

    def test_the_cli_writes_counts_and_no_ply(self) -> None:
        from tools import export_gaussian_ply

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            self.make_checkpoint(checkpoint)
            output = root / "counts" / "threshold_counts.json"
            argv = ["export_gaussian_ply.py", "--checkpoint", str(checkpoint), "--output", str(output),
                    "--count-thresholds", "0", "0.01", "0.05"]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                self.assertEqual(export_gaussian_ply.main(), 0)
            counted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([row["min_opacity"] for row in counted["counts"]], [0.0, 0.01, 0.05])
            self.assertEqual(counted["counts"][0]["gaussians_written"], 9)
            self.assertEqual(list(root.rglob("*.ply")), [])

if __name__ == "__main__":
    unittest.main()
