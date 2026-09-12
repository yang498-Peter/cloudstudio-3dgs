"""Stand-in backdrop builder (tools/build_standin_backgrounds.py), CPU only.

Synthetic gaussians and a fake backend stand in for the checkpoints and the
CUDA rasterizer. What is pinned:

* row selection: the selected Tile's box (with margin) removes exactly the
  rows inside it, the opacity floor acts on sigmoid(opacity), the optional
  anchor rule removes rows far from every LiDAR point, and the counts add up;
* layer concatenation zero-pads a DC-only dome to the stand-in's SH width;
* exposure harmonisation matches the merge tool's DC formula;
* the rendered library is written at each view's own crop size, the manifest
  loads through ViewBackgroundLibrary (signed, fail-closed for a missing
  view) and carries the ``standin`` provenance block;
* build_standin composes dome + filtered sources and reports per source.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - CPU channel without torch
    HAS_TORCH = False

from tools.build_standin_backgrounds import (  # noqa: E402
    SH_C0,
    bake_exposure_gain,
    build_standin,
    checkpoint_layer,
    concat_layers,
    inside_box_mask,
    median_exposure_gain,
    render_backdrops,
    select_standin_rows,
    write_standin_manifest,
)

BOX = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]


def _layer(means: np.ndarray, *, opacity_logit: float = 4.0, sh_bands: int = 3):
    count = len(means)
    return {
        "means": torch.as_tensor(np.asarray(means, dtype=np.float32)),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        "scales": torch.full((count, 3), -3.0),
        "opacities": torch.full((count,), float(opacity_logit)),
        "sh0": torch.full((count, 1, 3), 0.5),
        "shN": torch.full((count, sh_bands, 3), 0.25),
    }


def _payload(layer, *, log_gains=None, step=20000):
    payload = {"params": layer, "step": step}
    if log_gains is not None:
        payload["auxiliary_params"] = {"exposure_log_gains": torch.tensor(log_gains)}
    return payload


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class RowSelectionTests(unittest.TestCase):
    def test_box_opacity_and_anchor_rules_attribute_every_removed_row(self) -> None:
        means = np.array(
            [
                [0.0, 0.0, 0.0],   # inside box -> removed by box
                [0.9, 0.9, 0.9],   # inside box -> removed by box
                [1.05, 0.0, 0.0],  # just outside; inside with margin 0.1; anchored (1,0,0)
                [3.0, 0.0, 0.0],   # outside, alive, anchored (anchor at 3,0,0)
                [5.0, 0.0, 0.0],   # outside, alive, no anchor within 0.5 -> anchor rule
                [7.0, 0.0, 0.0],   # outside, dead (opacity)
            ],
            dtype=np.float32,
        )
        layer = _layer(means)
        layer["opacities"][5] = -6.0  # sigmoid ~ 0.0025 < 0.05
        anchors = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]])

        filtered, stats = select_standin_rows(
            layer, torch=torch, exclude_box=BOX, min_opacity=0.05,
            anchors=anchors, max_anchor_distance_m=0.5,
        )
        self.assertEqual(stats["input_count"], 6)
        self.assertEqual(stats["removed_inside_box"], 2)
        self.assertEqual(stats["removed_opacity"], 1)
        self.assertEqual(stats["removed_anchor"], 1)
        self.assertEqual(stats["kept_count"], 2)
        self.assertEqual(
            stats["removed_inside_box"] + stats["removed_opacity"] + stats["removed_anchor"] + stats["kept_count"],
            stats["input_count"],
        )
        kept = filtered["means"].numpy()
        np.testing.assert_allclose(kept, [[1.05, 0.0, 0.0], [3.0, 0.0, 0.0]])
        for key, value in filtered.items():
            self.assertEqual(int(value.shape[0]), 2, key)

        with_margin, stats_margin = select_standin_rows(
            layer, torch=torch, exclude_box=BOX, exclude_margin_m=0.1, min_opacity=0.0,
        )
        self.assertEqual(stats_margin["removed_inside_box"], 3)
        self.assertEqual(stats_margin["removed_opacity"], 0)
        self.assertEqual(int(with_margin["means"].shape[0]), 3)

        untouched, stats_none = select_standin_rows(layer, torch=torch, exclude_box=None)
        self.assertEqual(stats_none["kept_count"], 6)
        self.assertTrue(torch.equal(untouched["means"], layer["means"]))

    def test_inside_box_mask_rejects_degenerate_boxes(self) -> None:
        with self.assertRaises(ValueError):
            inside_box_mask(np.zeros((1, 3)), [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        mask = inside_box_mask(np.array([[1.0, 1.0, 1.0], [1.0001, 0.0, 0.0]]), BOX)
        self.assertEqual(mask.tolist(), [True, False])

    def test_anchor_rule_requires_anchor_points(self) -> None:
        with self.assertRaises(ValueError):
            select_standin_rows(
                _layer(np.zeros((2, 3))), torch=torch, exclude_box=None,
                max_anchor_distance_m=0.3,
            )


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class LayerTests(unittest.TestCase):
    def test_checkpoint_layer_accepts_params_or_splats_and_materialises_shN(self) -> None:
        dome = _layer(np.zeros((4, 3)), sh_bands=3)
        del dome["shN"]
        from_params = checkpoint_layer({"params": dome}, torch=torch)
        self.assertEqual(tuple(from_params["shN"].shape), (4, 0, 3))
        from_splats = checkpoint_layer({"splats": _layer(np.zeros((2, 3)))}, torch=torch)
        self.assertEqual(tuple(from_splats["shN"].shape), (2, 3, 3))
        with self.assertRaises(ValueError):
            checkpoint_layer({"step": 1}, torch=torch)
        broken = _layer(np.zeros((3, 3)))
        broken["scales"] = broken["scales"][:2]
        with self.assertRaises(ValueError):
            checkpoint_layer({"params": broken}, torch=torch)

    def test_concat_pads_dc_only_layer_to_the_widest_band(self) -> None:
        dome = _layer(np.zeros((3, 3)), sh_bands=0)
        tile = _layer(np.ones((2, 3)), sh_bands=3)
        merged = concat_layers([dome, tile], torch=torch)
        self.assertEqual(tuple(merged["shN"].shape), (5, 3, 3))
        self.assertTrue(torch.all(merged["shN"][:3] == 0.0))
        self.assertTrue(torch.all(merged["shN"][3:] == 0.25))
        self.assertEqual(tuple(merged["means"].shape), (5, 3))
        with self.assertRaises(ValueError):
            concat_layers([], torch=torch)

    def test_exposure_bake_matches_merge_tool_dc_formula(self) -> None:
        layer = _layer(np.zeros((2, 3)))
        payload = _payload(layer, log_gains=[np.log(0.8), np.log(0.9), np.log(1.0)])
        gain = median_exposure_gain(payload, torch=torch)
        self.assertAlmostEqual(gain, 0.9, places=6)
        baked = bake_exposure_gain(layer, gain)
        rgb_before = SH_C0 * layer["sh0"] + 0.5
        rgb_after = SH_C0 * baked["sh0"] + 0.5
        torch.testing.assert_close(rgb_after, rgb_before * gain)
        torch.testing.assert_close(baked["shN"], layer["shN"] * gain)
        self.assertIsNone(median_exposure_gain({"params": layer}, torch=torch))
        with self.assertRaises(ValueError):
            bake_exposure_gain(layer, 0.0)


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class BuildStandinTests(unittest.TestCase):
    def test_build_composes_dome_and_filtered_sources_with_provenance(self) -> None:
        dome = _layer(np.full((5, 3), 100.0), sh_bands=0)
        tile_a = _layer(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0]]))
        tile_a["opacities"][2] = -6.0
        tile_b = _layer(np.array([[0.5, 0.5, 0.5], [4.0, 0.0, 0.0]]))
        payloads = {
            Path("dome.pt"): {"params": dome, "sky_dome": {"count": 5}},
            Path("a.pt"): _payload(tile_a, log_gains=[np.log(0.5)]),
            Path("b.pt"): _payload(tile_b, log_gains=[0.0]),
        }
        params, provenance = build_standin(
            torch=torch,
            dome_path=Path("dome.pt"),
            standin_paths=[Path("a.pt"), Path("b.pt")],
            exclude_box=BOX,
            exclude_box_kind="training_and_export_box",
            exclude_margin_m=0.0,
            min_opacity=0.05,
            anchors=None,
            anchor_source=None,
            max_anchor_distance_m=None,
            harmonize_exposure=True,
            target_gain=1.0,
            load=lambda path: payloads[path],
        )
        # dome 5 + a: (2,0,0) + b: (4,0,0) = 7 rows
        self.assertEqual(int(params["means"].shape[0]), 7)
        self.assertEqual(provenance["dome_count"], 5)
        self.assertEqual(provenance["standin_count"], 2)
        self.assertEqual(provenance["rendered_gaussian_count"], 7)
        a, b = provenance["sources"]
        self.assertEqual((a["removed_inside_box"], a["removed_opacity"], a["kept_count"]), (1, 1, 1))
        self.assertEqual((b["removed_inside_box"], b["removed_opacity"], b["kept_count"]), (1, 0, 1))
        self.assertAlmostEqual(a["exposure_gain_applied"], 0.5)
        self.assertAlmostEqual(b["exposure_gain_applied"], 1.0)
        self.assertEqual(provenance["exclusion"]["box"], BOX)
        self.assertEqual(provenance["exposure"]["frame"], "photo")
        # a's kept row carries the baked gain 0.5: rgb 0.5*C0+0.5 -> half
        kept_a = params["sh0"][5]
        expected = (0.5 * SH_C0 + 0.5) * 0.5
        self.assertAlmostEqual(float(SH_C0 * kept_a[0, 0] + 0.5), expected, places=6)

        with self.assertRaises(ValueError):
            build_standin(
                torch=torch, dome_path=Path("dome.pt"), standin_paths=[Path("dome.pt")],
                exclude_box=None, exclude_box_kind="none", exclude_margin_m=0.0,
                min_opacity=0.0, anchors=None, anchor_source=None,
                max_anchor_distance_m=None, harmonize_exposure=True, target_gain=1.0,
                load=lambda path: payloads[path],
            )


@unittest.skipUnless(HAS_TORCH, "torch is an optional training dependency")
class RenderAndManifestTests(unittest.TestCase):
    def test_library_is_stored_at_crop_size_signed_and_fail_closed(self) -> None:
        from cloudstudio_3dgs.training.view_backgrounds import ViewBackgroundLibrary

        class FakeBackend:
            calls: list[tuple[str, int, int, tuple]] = []

            @staticmethod
            def render(params, sample, *, with_range, background_rgb):
                FakeBackend.calls.append((sample.image_id, sample.height, sample.width, background_rgb))
                image = torch.zeros((sample.height, sample.width, 3))
                image[:, : sample.width // 2, 0] = 1.0  # left half red
                return image, None, None, {}

        samples = [
            SimpleNamespace(image_id="img_a::yaw_minus_35", height=8, width=6),
            SimpleNamespace(image_id="img_b::pitch_up_56", height=4, width=10),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Tile_1"
            views = render_backdrops(
                backend=FakeBackend(), torch=torch, params={}, samples=samples,
                output_root=root, background_rgb=(1.0, 1.0, 1.0), save_threads=2,
                progress=lambda text: None,
            )
            self.assertEqual(views["img_a::yaw_minus_35"], {"file": "img_a__yaw_minus_35.png", "height": 8, "width": 6})
            self.assertEqual(views["img_b::pitch_up_56"], {"file": "img_b__pitch_up_56.png", "height": 4, "width": 10})
            self.assertEqual([call[3] for call in FakeBackend.calls], [(1.0, 1.0, 1.0)] * 2)

            signature = write_standin_manifest(
                root / "background_manifest.json",
                views=views, tile_id=1, tile_inputs_manifest_sha256="f" * 64,
                dome_path=Path("dome.pt"), dome_sha256="d" * 64,
                background_rgb=(1.0, 1.0, 1.0), downsample=1,
                provenance={"schema_version": 1, "standin_count": 2, "sources": []},
            )
            self.assertEqual(len(signature), 64)
            manifest = json.loads((root / "background_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["standin"]["standin_count"], 2)
            self.assertEqual(manifest["render_resolution"], "tile_crop")
            self.assertEqual(manifest["tile_id"], 1)

            library = ViewBackgroundLibrary(root / "background_manifest.json", root, device="cpu")
            served = library.background_for("img_a::yaw_minus_35", height=8, width=6, torch=torch)
            self.assertEqual(tuple(served.shape), (8, 6, 3))
            self.assertGreater(float(served[:, :3, 0].mean()), 0.99)
            self.assertLess(float(served[:, 3:, 0].mean()), 0.01)
            with self.assertRaises(ValueError):
                library.background_for("img_c::yaw_minus_35", height=8, width=6, torch=torch)

            text = (root / "background_manifest.json").read_text(encoding="utf-8")
            (root / "background_manifest.json").write_text(
                text.replace('"standin_count": 2', '"standin_count": 3'), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                ViewBackgroundLibrary(root / "background_manifest.json", root, device="cpu")

    def test_render_refuses_a_size_mismatch_and_duplicate_ids(self) -> None:
        class WrongSizeBackend:
            @staticmethod
            def render(params, sample, *, with_range, background_rgb):
                return torch.zeros((sample.height + 1, sample.width, 3)), None, None, {}

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                render_backdrops(
                    backend=WrongSizeBackend(), torch=torch, params={},
                    samples=[SimpleNamespace(image_id="x::f", height=2, width=2)],
                    output_root=Path(temporary), background_rgb=(1.0, 1.0, 1.0),
                    progress=lambda text: None,
                )

        class OkBackend:
            @staticmethod
            def render(params, sample, *, with_range, background_rgb):
                return torch.zeros((sample.height, sample.width, 3)), None, None, {}

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                render_backdrops(
                    backend=OkBackend(), torch=torch, params={},
                    samples=[SimpleNamespace(image_id="x::f", height=2, width=2)] * 2,
                    output_root=Path(temporary), background_rgb=(1.0, 1.0, 1.0),
                    progress=lambda text: None,
                )


if __name__ == "__main__":
    unittest.main()
