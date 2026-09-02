"""The monocular-depth supervision knobs must validate and must only change a
config's signed contract when a config actually asks for them, so every
existing signed profile keeps its signature."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional for the CPU suite
    HAS_TORCH = False

from cloudstudio_3dgs.training.trainer import TrainerConfig


def _da2_sources(root: Path) -> dict:
    # contract_dict reads and verifies the signed manifest, so a real one is needed.
    from cloudstudio_3dgs.data.mono_depth import sign_mono_depth_manifest

    manifest = root / "da2.json"
    manifest.write_text(
        json.dumps(
            sign_mono_depth_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_da2_relative_depth_cache",
                    "split": "train",
                    "source_face_manifest_sha256": "f" * 64,
                    "dataset_manifest_sha256": "synthetic",
                    "lidar_depth_manifest_sha256": "d" * 64,
                    "complete_face_cache": True,
                    "expected_face_count": 0,
                    "records": [],
                }
            )
        ),
        encoding="utf-8",
    )
    return dict(mono_depth_manifest=manifest, mono_depth_root=root)


def _base(**overrides):
    fields = dict(
        run_id="da2",
        dataset_manifest=Path("d.json"),
        recording_root=Path("r"),
        mask_manifest=Path("m.json"),
        mask_root=Path("m"),
        split_manifest=Path("s.json"),
        initialization_ply=Path("i.ply"),
        output_dir=Path("o"),
        gsplat_lock=Path("l.json"),
        require_person_masks=False,
        lidar_range_weight=0.0,
    )
    fields.update(overrides)
    return TrainerConfig(**fields)


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class Da2DepthContractTests(unittest.TestCase):
    def test_defaults_leave_the_contract_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _base(da2_depth_weight=0.15, **_da2_sources(Path(tmp)))
            config.validate()
            self.assertIsNone(config.mono_depth_max_range_m)
            self.assertEqual(config.da2_depth_space, "linear")
            self.assertNotIn(
                "da2_depth_contract", config.contract_dict()["loss_weights"]
            )

    def test_knobs_enter_the_contract_only_when_da2_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            active = _base(
                da2_depth_weight=0.15,
                mono_depth_max_range_m=30.0,
                da2_depth_space="compressed",
                **_da2_sources(Path(tmp)),
            )
            active.validate()
            self.assertEqual(
                active.contract_dict()["loss_weights"]["da2_depth_contract"],
                {"max_range_m": 30.0, "space": "compressed"},
            )
        inactive = _base(mono_depth_max_range_m=30.0, da2_depth_space="compressed")
        inactive.validate()
        self.assertNotIn(
            "da2_depth_contract", inactive.contract_dict()["loss_weights"]
        )

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "da2_depth_space"):
            _base(da2_depth_space="metres").validate()
        with self.assertRaisesRegex(ValueError, "mono_depth_max_range_m"):
            _base(mono_depth_max_range_m=0.0).validate()


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class TileOwnershipContractTests(unittest.TestCase):
    def test_off_by_default_and_absent_from_the_contract(self) -> None:
        config = _base()
        config.validate()
        self.assertFalse(config.tile_ownership_masking)
        self.assertNotIn("tile_ownership_masking", config.contract_dict()["loss_weights"])

    def test_requires_tile_inputs_and_face_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "tile_ownership_masking"):
            _base(tile_ownership_masking=True).validate()

    def test_knob_values_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiles = root / "tiles.json"
            tiles.write_text("{}", encoding="utf-8")
            faces = root / "faces.json"
            faces.write_text("{}", encoding="utf-8")
            for bad in (dict(tile_ownership_margin_m=-1.0), dict(tile_ownership_dilation_px=-3)):
                with self.assertRaisesRegex(ValueError, "tile_ownership"):
                    _base(
                        tile_ownership_masking=True,
                        tile_inputs_manifest=tiles,
                        tile_inputs_root=root,
                        face_cache_manifest=faces,
                        face_cache_root=root,
                        **bad,
                    ).validate()


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class InitializationStrideContractTests(unittest.TestCase):
    def test_default_is_as_signed_and_absent_from_the_contract(self) -> None:
        config = _base()
        config.validate()
        self.assertEqual(config.initialization_subsample_stride, 1)
        self.assertNotIn("subsample_stride", config.contract_dict()["initialization"])

    def test_stride_enters_the_contract_and_is_validated(self) -> None:
        config = _base(initialization_subsample_stride=4)
        config.validate()
        self.assertEqual(config.contract_dict()["initialization"]["subsample_stride"], 4)
        with self.assertRaisesRegex(ValueError, "initialization_subsample_stride"):
            _base(initialization_subsample_stride=0).validate()


if __name__ == "__main__":
    unittest.main()
