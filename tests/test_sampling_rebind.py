from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.training.sampling_rebind import (
    rebind_checkpoint_sampling_identity,
)


class SamplingIdentityRebindTests(unittest.TestCase):
    def _checkpoint(self, path: Path, identity: dict, *, rows: int = 2) -> None:
        torch.save(
            {
                "schema_version": 1,
                "step": 7,
                "identity": identity,
                "params": {
                    "means": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
                    "opacities": torch.zeros(rows),
                },
                "auxiliary_params": {"exposure_log_gains": torch.zeros(2)},
            },
            path,
        )

    def test_rebind_preserves_parameters_and_signs_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "dataset_manifest_sha256": "dataset",
                "mask_manifest_sha256": "mask",
                "coordinate_transform_sha256": "coordinate",
                "initialization_ply_sha256": "initialization",
                "initialization_geometry_sha256": "geometry",
                "surface_initialization_sha256": "surface",
                "gsplat_runtime": {"commit": "locked"},
                "split": "train",
                "factor": 1,
                "crop": None,
            }
            source_path = root / "source.pt"
            target_path = root / "target.pt"
            output_path = root / "output.pt"
            report_path = root / "report.json"
            self._checkpoint(source_path, base)
            target_identity = {
                "face_manifest_sha256": "face",
                "face_plan": "mipmap_face4",
                "source_identity": {
                    key: base[key]
                    for key in ("dataset_manifest_sha256", "mask_manifest_sha256", "split", "factor", "crop")
                },
                **{
                    key: base[key]
                    for key in (
                        "coordinate_transform_sha256",
                        "initialization_ply_sha256",
                        "initialization_geometry_sha256",
                        "surface_initialization_sha256",
                        "gsplat_runtime",
                    )
                },
            }
            self._checkpoint(target_path, target_identity)

            report = rebind_checkpoint_sampling_identity(
                source_path, target_path, output_path, report_path
            )
            output = torch.load(output_path, map_location="cpu", weights_only=False)
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            torch.testing.assert_close(output["params"]["means"], source["params"]["means"])
            self.assertEqual(output["identity"], target_identity)
            self.assertTrue(output["derived_warm_start_only"])
            self.assertFalse(output["resume_supported"])
            unsigned = dict(report)
            signature = unsigned.pop("sampling_identity_rebind_sha256")
            self.assertEqual(signature, hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)

    def test_rebind_rejects_base_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = {
                "coordinate_transform_sha256": "coordinate",
                "initialization_ply_sha256": "initialization",
                "initialization_geometry_sha256": "geometry",
                "surface_initialization_sha256": "surface",
                "gsplat_runtime": {"commit": "locked"},
            }
            source_path = root / "source.pt"
            target_path = root / "target.pt"
            self._checkpoint(source_path, {"dataset_manifest_sha256": "a", **shared})
            self._checkpoint(
                target_path,
                {
                    "face_manifest_sha256": "face",
                    "source_identity": {"dataset_manifest_sha256": "b"},
                    **shared,
                },
            )
            with self.assertRaisesRegex(ValueError, "base identity mismatch"):
                rebind_checkpoint_sampling_identity(
                    source_path,
                    target_path,
                    root / "output.pt",
                    root / "report.json",
                )


if __name__ == "__main__":
    unittest.main()
