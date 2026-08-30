from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.sky_layer import (
    SkyLayerConfig,
    augment_checkpoint_with_sky,
    build_sky_layer,
)


class SkyLayerTests(unittest.TestCase):
    def test_fibonacci_cap_is_deterministic_and_above_requested_horizon(self) -> None:
        config = SkyLayerConfig(count=100, radius_m=10.0, min_world_z_direction=0.2)
        first = build_sky_layer(np.array([1.0, 2.0, 3.0]), config)
        second = build_sky_layer(np.array([1.0, 2.0, 3.0]), config)
        np.testing.assert_array_equal(first["means"], second["means"])
        directions = (first["means"] - np.array([1.0, 2.0, 3.0])) / 10.0
        self.assertGreater(float(directions[:, 2].min()), 0.2)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)

    def test_checkpoint_augmentation_preserves_surface_and_records_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "coordinate_frame": "s1_local",
                "images": [
                    {"c2w": np.eye(4).tolist()},
                    {"c2w": (np.eye(4) + np.array(
                        [[0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 6], [0, 0, 0, 0]],
                        dtype=float,
                    )).tolist()},
                ],
            }
            manifest["manifest_sha256"] = hashlib.sha256(
                canonical_json_bytes(manifest)
            ).hexdigest()
            manifest_path = root / "dataset.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            params = {
                "means": torch.zeros((2, 3)),
                "scales": torch.zeros((2, 3)),
                "quats": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
                "opacities": torch.zeros((2,)),
                "sh0": torch.zeros((2, 1, 3)),
                "shN": torch.zeros((2, 8, 3)),
            }
            source_path = root / "source.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "step": 7,
                    "identity": {"dataset_manifest_sha256": manifest["manifest_sha256"]},
                    "params": params,
                    "auxiliary_params": {"exposure_log_gains": torch.zeros(2)},
                },
                source_path,
            )
            output_path = root / "sky.pt"
            report_path = root / "sky.json"
            report = augment_checkpoint_with_sky(
                source_path,
                manifest_path,
                output_path,
                report_path,
                SkyLayerConfig(count=100, radius_m=10.0, scale_m=0.5),
            )
            payload = torch.load(output_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["step"], 7)
            self.assertTrue(payload["derived_warm_start_only"])
            self.assertEqual(tuple(payload["params"]["means"].shape), (102, 3))
            self.assertEqual(tuple(payload["params"]["shN"].shape), (102, 8, 3))
            torch.testing.assert_close(payload["params"]["means"][:2], params["means"])
            self.assertEqual(report["sky_gaussian_start"], 2)
            self.assertEqual(report["sky_gaussian_count"], 100)
            self.assertEqual(report["total_gaussian_count"], 102)
            self.assertTrue(report["warm_start_supported"])
            self.assertFalse(report["resume_supported"])

    def test_prebaked_dome_is_appended_verbatim_with_rezeroed_higher_bands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "coordinate_frame": "s1_local",
                "images": [{"c2w": np.eye(4).tolist()}],
            }
            manifest["manifest_sha256"] = hashlib.sha256(
                canonical_json_bytes(manifest)
            ).hexdigest()
            manifest_path = root / "dataset.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            params = {
                "means": torch.zeros((2, 3)),
                "scales": torch.zeros((2, 3)),
                "quats": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
                "opacities": torch.zeros((2,)),
                "sh0": torch.zeros((2, 1, 3)),
                "shN": torch.zeros((2, 8, 3)),
            }
            source_path = root / "source.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "step": 7,
                    "identity": {"dataset_manifest_sha256": manifest["manifest_sha256"]},
                    "params": params,
                    "auxiliary_params": {},
                },
                source_path,
            )

            # A baked dome carries its own colours and opacities but only SH0;
            # the destination model's higher bands must come back zeroed, and
            # every baked value must survive the append bit for bit.
            dome_count = 150
            dome = {
                "means": torch.full((dome_count, 3), 25.0),
                "scales": torch.full((dome_count, 3), 0.44),
                "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(dome_count, 1),
                "opacities": torch.full((dome_count,), 2.5),
                "sh0": torch.full((dome_count, 1, 3), 0.9),
                "shN": torch.zeros((dome_count, 0, 3)),
            }
            dome_path = root / "dome.pt"
            torch.save({"params": dome}, dome_path)

            output_path = root / "sky.pt"
            report_path = root / "sky.json"
            report = augment_checkpoint_with_sky(
                source_path,
                manifest_path,
                output_path,
                report_path,
                SkyLayerConfig(count=100),
                prebaked_dome=dome_path,
            )
            payload = torch.load(output_path, map_location="cpu", weights_only=False)
            self.assertEqual(tuple(payload["params"]["means"].shape), (152, 3))
            self.assertEqual(tuple(payload["params"]["shN"].shape), (152, 8, 3))
            torch.testing.assert_close(payload["params"]["means"][2:], dome["means"])
            torch.testing.assert_close(
                payload["params"]["opacities"][2:], dome["opacities"]
            )
            self.assertEqual(
                float(payload["params"]["shN"][2:].abs().max()), 0.0
            )
            self.assertEqual(report["sky_gaussian_count"], dome_count)
            self.assertEqual(report["kind"], "prebaked_photo_dome")
            self.assertIn("prebaked_dome_sha256", report)


if __name__ == "__main__":
    unittest.main()
