"""Tile checkpoint merge (tools/merge_v28_tile_checkpoints.py), CPU only.

Synthetic two-Tile checkpoints and a synthetic signed tile inputs manifest.
What is pinned:

* the fill layer is dropped wherever a delivery Tile's
  ``training_and_export_box`` claims the space, and survives outside every
  box;
* the optional voxel rule rescues fill rows inside a box that no retained
  delivery gaussian occupies, and rejects the ones that sit on top of one;
* the fill opacity floor acts on ``sigmoid(opacity)``;
* a merge with no fill row - either because no fill checkpoint was given, or
  because every fill row was rejected - writes bit-identical tensors and a
  report that differs only in the fill keys and the signatures bound to
  them, so the default path is unchanged;
* the report gains ``fill_*`` keys only when a fill checkpoint is given, and
  stays signed either way.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - CPU channel without torch
    HAS_TORCH = False

from cloudstudio_3dgs.data.manifest import canonical_json_bytes  # noqa: E402
from cloudstudio_3dgs.training.tile_inputs import (  # noqa: E402
    TILE_INPUT_KIND,
    TILE_INPUT_SCHEMA_VERSION,
)
from tools.merge_v28_tile_checkpoints import (  # noqa: E402
    inside_any_box_mask,
    main,
    unclaimed_voxel_mask,
)

COORDINATE_SHA = "a" * 64
CORE_BOXES = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
EXPORT_BOXES = (
    [[-0.1, -0.1, -0.1], [1.1, 1.1, 1.1]],
    [[0.9, -0.1, -0.1], [2.1, 1.1, 1.1]],
)
# Two Tile gaussians per Tile, both well inside their own core.
TILE_MEANS = {
    0: [[0.25, 0.25, 0.25], [0.35, 0.25, 0.25]],
    1: [[1.25, 0.25, 0.25], [1.35, 0.25, 0.25]],
}


def _write_tile_inputs(root: Path) -> Path:
    tiles = []
    for tile_id, (core, export) in enumerate(zip(CORE_BOXES, EXPORT_BOXES)):
        artifact = root / f"Tile_{tile_id}_init.ply"
        artifact.write_bytes(f"synthetic init {tile_id}\n".encode("utf-8"))
        tiles.append(
            {
                "tile_id": tile_id,
                "name": f"Tile_{tile_id}",
                "core_box": core,
                "training_and_export_box": export,
                "view_count": 0,
                "views": [],
                "initialization": {
                    "path": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                },
            }
        )
    payload = {
        "schema_version": TILE_INPUT_SCHEMA_VERSION,
        "kind": TILE_INPUT_KIND,
        "tile_count": len(tiles),
        "tiles": tiles,
    }
    payload["tile_inputs_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    path = root / "tile_inputs_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _params(means: list[list[float]], opacities: list[float] | None = None) -> dict:
    count = len(means)
    values = torch.arange(count, dtype=torch.float32).reshape(count, 1, 1)
    return {
        "means": torch.tensor(means, dtype=torch.float32),
        "quats": torch.zeros((count, 4), dtype=torch.float32) + values.reshape(count, 1),
        "scales": torch.full((count, 3), -2.0, dtype=torch.float32),
        "opacities": torch.tensor(
            [2.0] * count if opacities is None else opacities, dtype=torch.float32
        ),
        "sh0": values.expand(count, 1, 3).clone() * 0.1,
        "shN": values.expand(count, 3, 3).clone() * 0.01,
    }


def _write_checkpoint(path: Path, means: list[list[float]], *, step: int, opacities=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "step": step,
            "params": _params(means, opacities),
            "identity": {"coordinate_transform_sha256": COORDINATE_SHA},
            "auxiliary_params": {"exposure_log_gains": torch.zeros(3)},
        },
        path,
    )
    return path


def _run(tile_inputs: Path, output: Path, *extra: str) -> dict:
    argv = [
        "merge_v28_tile_checkpoints.py",
        "--tile-inputs",
        str(tile_inputs),
        "--tile-inputs-root",
        str(tile_inputs.parent),
        "--output-checkpoint",
        str(output / "merged.pt"),
        "--output-report",
        str(output / "merge_report.json"),
        "--harmonize-exposure",
    ]
    for tile_id in sorted(TILE_MEANS):
        argv += [
            "--tile-checkpoint",
            f"{tile_id}={tile_inputs.parent / f'tile{tile_id}.pt'}",
        ]
    argv += list(extra)
    original = sys.argv
    sys.argv = argv
    try:
        assert main() == 0
    finally:
        sys.argv = original
    return json.loads((output / "merge_report.json").read_text(encoding="utf-8"))


@unittest.skipUnless(HAS_TORCH, "torch is required")
class FillRowSelectionTests(unittest.TestCase):
    def test_box_mask_marks_rows_inside_any_tile_box(self) -> None:
        boxes = np.asarray(EXPORT_BOXES, dtype=np.float64)
        means = np.asarray(
            [[0.5, 0.5, 0.5], [2.05, 0.5, 0.5], [5.0, 5.0, 5.0], [-0.05, 0.0, 0.0]],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            inside_any_box_mask(means, boxes), [True, True, False, True]
        )

    def test_voxel_rule_keeps_rows_no_delivery_gaussian_occupies(self) -> None:
        tile = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
        fill = np.asarray(
            [[0.02, 0.0, 0.0], [0.52, 0.0, 0.0], [0.25, 0.0, 0.0], [3.0, 3.0, 3.0]],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            unclaimed_voxel_mask(fill, tile, voxel_m=0.1, clearance_voxels=0),
            [False, False, True, True],
        )
        # One voxel of clearance also rejects the neighbours of an occupied
        # voxel, so a row 0.05 m from a delivery gaussian no longer survives.
        np.testing.assert_array_equal(
            unclaimed_voxel_mask(
                np.asarray([[0.15, 0.0, 0.0], [3.0, 3.0, 3.0]], dtype=np.float64),
                tile,
                voxel_m=0.1,
                clearance_voxels=1,
            ),
            [False, True],
        )

    def test_voxel_rule_rejects_degenerate_parameters(self) -> None:
        points = np.zeros((1, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            unclaimed_voxel_mask(points, points, voxel_m=0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            unclaimed_voxel_mask(points, points, voxel_m=0.1, clearance_voxels=-1)


@unittest.skipUnless(HAS_TORCH, "torch is required")
class MergeFillLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.tile_inputs = _write_tile_inputs(self.root)
        for tile_id, means in TILE_MEANS.items():
            _write_checkpoint(self.root / f"tile{tile_id}.pt", means, step=100 + tile_id)
        # One fill row inside both Tile boxes on top of a Tile gaussian, one
        # inside a Tile box where no Tile grew anything, one outside every
        # box, and one outside with a dead opacity.
        self.fill = _write_checkpoint(
            self.root / "fill.pt",
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
            step=50,
            opacities=[2.0, 2.0, 2.0, -4.0],
        )
        self.addCleanup(self._temporary.cleanup)

    def _output(self, name: str) -> Path:
        path = self.root / name
        path.mkdir()
        return path

    def test_box_rule_drops_rows_inside_a_tile_and_keeps_the_rest(self) -> None:
        report = _run(self.tile_inputs, self._output("boxed"), "--fill-checkpoint", str(self.fill))
        source = report["fill_sources"][0]
        self.assertEqual(source["input_gaussian_count"], 4)
        self.assertEqual(source["rejected_inside_tile_box_count"], 2)
        self.assertEqual(source["retained_gaussian_count"], 2)
        self.assertEqual(report["fill_gaussian_count"], 2)
        self.assertEqual(report["tile_gaussian_count"], 4)
        self.assertEqual(report["merged_gaussian_count"], 6)
        self.assertEqual(report["discarded_by_merge_policy_count"], 0)
        self.assertEqual(source["checkpoint_sha256"], _sha256(self.fill))
        self.assertEqual(source["completed_steps"], 50)
        self.assertEqual(source["exposure_gain_applied"], 1.0)
        merged = torch.load(
            self.root / "boxed" / "merged.pt", map_location="cpu", weights_only=False
        )
        kept = merged["params"]["means"].numpy()
        np.testing.assert_allclose(kept[4:], [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])

    def test_voxel_rule_rescues_the_row_no_tile_grew_into(self) -> None:
        report = _run(
            self.tile_inputs,
            self._output("voxel"),
            "--fill-checkpoint",
            str(self.fill),
            "--fill-occupancy-voxel-m",
            "0.1",
            "--fill-occupancy-clearance-voxels",
            "1",
        )
        source = report["fill_sources"][0]
        self.assertEqual(source["rejected_inside_tile_box_count"], 2)
        self.assertEqual(source["rejected_as_delivery_occupied_count"], 1)
        self.assertEqual(source["retained_gaussian_count"], 3)
        merged = torch.load(
            self.root / "voxel" / "merged.pt", map_location="cpu", weights_only=False
        )
        np.testing.assert_allclose(
            merged["params"]["means"].numpy()[4:],
            [[0.75, 0.75, 0.75], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
        )
        self.assertEqual(report["fill_exclusion"]["occupancy_voxel_m"], 0.1)
        self.assertEqual(report["fill_exclusion"]["occupancy_clearance_voxels"], 1)

    def test_opacity_floor_drops_dead_fill_rows(self) -> None:
        report = _run(
            self.tile_inputs,
            self._output("floor"),
            "--fill-checkpoint",
            str(self.fill),
            "--fill-min-opacity",
            "0.5",
        )
        source = report["fill_sources"][0]
        self.assertEqual(source["rejected_by_opacity_floor_count"], 1)
        self.assertEqual(source["retained_gaussian_count"], 1)
        self.assertEqual(report["fill_exclusion"]["min_opacity"], 0.5)

    def test_without_a_fill_source_the_report_has_no_fill_keys(self) -> None:
        report = _run(self.tile_inputs, self._output("plain"))
        for key in (
            "fill_sources",
            "fill_gaussian_count",
            "fill_source_count",
            "fill_exclusion",
            "tile_gaussian_count",
        ):
            self.assertNotIn(key, report)
        self.assertEqual(len(report["merge_report_sha256"]), 64)
        self.assertEqual(len(report["delivery_report_sha256"]), 64)
        checkpoint = torch.load(
            self.root / "plain" / "merged.pt", map_location="cpu", weights_only=False
        )
        self.assertNotIn("fill_checkpoint_sha256", checkpoint["identity"])

    def test_a_fill_source_that_keeps_no_row_changes_no_tensor(self) -> None:
        plain = _run(self.tile_inputs, self._output("base"))
        # Every row of this fill source is inside a Tile box, so the merge
        # must be the same merge - the same bytes, and a report that differs
        # only in the fill keys and the two signatures bound to them.
        inside = _write_checkpoint(
            self.root / "inside.pt", [[0.25, 0.25, 0.25], [1.25, 0.25, 0.25]], step=50
        )
        filled = _run(
            self.tile_inputs, self._output("empty_fill"), "--fill-checkpoint", str(inside)
        )
        self.assertEqual(filled["fill_gaussian_count"], 0)
        base = torch.load(
            self.root / "base" / "merged.pt", map_location="cpu", weights_only=False
        )
        same = torch.load(
            self.root / "empty_fill" / "merged.pt", map_location="cpu", weights_only=False
        )
        self.assertEqual(set(base["params"]), set(same["params"]))
        for key, value in base["params"].items():
            self.assertTrue(torch.equal(value, same["params"][key]), key)
        self.assertEqual(base["step"], same["step"])
        volatile = {
            "fill_sources",
            "fill_gaussian_count",
            "fill_source_count",
            "fill_exclusion",
            "tile_gaussian_count",
            "merge_report_sha256",
            "delivery_report_sha256",
            "output_checkpoint",
            "output_checkpoint_sha256",
        }
        self.assertEqual(
            {k: v for k, v in plain.items() if k not in volatile},
            {k: v for k, v in filled.items() if k not in volatile},
        )

    def test_a_second_fill_source_does_not_stack_on_the_first(self) -> None:
        twin = _write_checkpoint(
            self.root / "twin.pt", [[0.75, 0.75, 0.75], [7.0, 7.0, 7.0]], step=50
        )
        report = _run(
            self.tile_inputs,
            self._output("two"),
            "--fill-checkpoint",
            str(self.fill),
            "--fill-checkpoint",
            str(twin),
            "--fill-occupancy-voxel-m",
            "0.1",
        )
        self.assertEqual(report["fill_source_count"], 2)
        # The first source already took [0.75, 0.75, 0.75]; the twin only
        # contributes the row outside every Tile box.
        self.assertEqual(report["fill_sources"][0]["retained_gaussian_count"], 3)
        self.assertEqual(report["fill_sources"][1]["retained_gaussian_count"], 1)
        self.assertEqual(report["fill_gaussian_count"], 4)
        merged = torch.load(
            self.root / "two" / "merged.pt", map_location="cpu", weights_only=False
        )
        np.testing.assert_allclose(
            merged["params"]["means"].numpy()[-1], [7.0, 7.0, 7.0]
        )

    def test_a_fill_source_from_another_coordinate_frame_is_refused(self) -> None:
        stranger = self.root / "stranger.pt"
        payload = torch.load(self.fill, map_location="cpu", weights_only=False)
        payload["identity"] = {"coordinate_transform_sha256": "b" * 64}
        torch.save(payload, stranger)
        with self.assertRaisesRegex(ValueError, "another coordinate transform"):
            _run(
                self.tile_inputs,
                self._output("stranger"),
                "--fill-checkpoint",
                str(stranger),
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
