"""Tile coverage audit (tools/audit_competitor_tile_coverage.py), CPU only.

Synthetic PLY + a signed two-Tile manifest: point classification against the
boxes, the rigid alignment convention (model -> reference, as
tools/align_gaussian_ply.py stores it), the opacity subset, the LiDAR-support
subset, and the PNG side product.
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

from cloudstudio_3dgs.data.manifest import canonical_json_bytes  # noqa: E402
from cloudstudio_3dgs.training.tile_inputs import (  # noqa: E402
    TILE_INPUT_KIND,
    TILE_INPUT_SCHEMA_VERSION,
)
from tools.audit_competitor_tile_coverage import (  # noqa: E402
    apply_transform,
    audit,
    classify_points,
    horizontal_distance_beyond_union,
    lidar_support_mask,
    render_topdown,
    tile_boxes,
)

BOX_A = [[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
BOX_B = [[1.0, 0.0, 0.0], [2.0, 2.0, 1.0]]


def _write_manifest(root: Path) -> Path:
    payload = {
        "schema_version": TILE_INPUT_SCHEMA_VERSION,
        "kind": TILE_INPUT_KIND,
        "tile_count": 2,
        "tiles": [
            {
                "tile_id": tile_id, "name": name, "core_box": box,
                "training_and_export_box": box, "view_count": 0, "views": [],
                "initialization": {"path": "init.ply", "sha256": "0" * 64},
            }
            for tile_id, name, box in ((1, "Tile_1", BOX_B), (0, "Tile_0", BOX_A))
        ],
    }
    payload["tile_inputs_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    path = root / "tile_inputs_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_ply(path: Path, xyz: np.ndarray, opacity: np.ndarray | None = None) -> None:
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if opacity is not None:
        fields.append(("opacity", "<f4"))
    records = np.zeros(len(xyz), dtype=np.dtype(fields))
    records["x"], records["y"], records["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if opacity is not None:
        records["opacity"] = opacity
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property float {name}" for name, _ in fields]
    header.append("end_header")
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        stream.write(records.tobytes())


class CoverageTests(unittest.TestCase):
    def test_classification_orders_boxes_by_tile_id_and_measures_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = json.loads(_write_manifest(Path(temporary)).read_text(encoding="utf-8"))
        boxes = tile_boxes(manifest, "core_box")
        self.assertEqual([tile_id for tile_id, _, _ in boxes], [0, 1])
        points = np.array([
            [0.5, 1.0, 0.5],   # Tile_0
            [1.0, 1.0, 0.5],   # shared face -> lower id wins
            [1.5, 1.0, 0.5],   # Tile_1
            [5.0, 1.0, 0.5],   # outside, 3 m beyond in x
            [0.5, 1.0, 9.0],   # over the footprint, above
        ])
        owner = classify_points(points, boxes)
        self.assertEqual(owner.tolist(), [0, 0, 1, -1, -1])
        beyond = horizontal_distance_beyond_union(points, boxes)
        np.testing.assert_allclose(beyond, [0.0, 0.0, 0.0, 3.0, 0.0])
        with self.assertRaises(ValueError):
            tile_boxes(manifest, "halo_box")

    def test_transform_is_model_to_reference(self) -> None:
        transform = np.eye(4)
        transform[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        transform[:3, 3] = [10.0, 0.0, 0.0]
        moved = apply_transform(np.array([[1.0, 0.0, 0.0]]), transform)
        np.testing.assert_allclose(moved, [[10.0, 1.0, 0.0]])
        np.testing.assert_allclose(apply_transform(np.array([[1.0, 2.0, 3.0]]), None), [[1.0, 2.0, 3.0]])

    def test_lidar_support_mask_is_a_bounded_query(self) -> None:
        anchors = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        supported = lidar_support_mask(
            np.array([[0.3, 0.0, 0.0], [5.0, 0.0, 0.0], [10.4, 0.0, 0.0]]), anchors, max_distance_m=0.5,
        )
        self.assertEqual(supported.tolist(), [True, False, True])

    def test_audit_end_to_end_with_opacity_anchor_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_manifest(root)
            # model frame = reference frame shifted by +100 in x
            xyz_model = np.array([
                [100.5, 1.0, 0.5],  # Tile_0, near anchor
                [101.5, 1.0, 0.5],  # Tile_1, no anchor near
                [105.0, 1.0, 0.5],  # outside, dead
                [100.5, 1.0, 5.0],  # over footprint above, alive, no anchor
            ])
            opacity = np.array([4.0, 4.0, -6.0, 4.0], dtype=np.float32)
            ply = root / "model.ply"
            _write_ply(ply, xyz_model, opacity)
            anchors = root / "anchors.ply"
            _write_ply(anchors, np.array([[0.5, 1.0, 0.5], [1.5, 1.0, 0.0]]))
            alignment = root / "alignment.json"
            transform = np.eye(4)
            transform[0, 3] = -100.0
            alignment.write_text(json.dumps({"transform": transform.tolist()}), encoding="utf-8")

            report, xyz, owner, supported = audit(
                ply=ply, tile_inputs=manifest, transform_json=alignment,
                box_kind="training_and_export_box", opacity_floor=0.05, label="synthetic",
                anchor_ply=anchors, anchor_distance_m=0.6,
            )
            self.assertTrue(report["transform_applied"])
            self.assertEqual(owner.tolist(), [0, 1, -1, -1])
            by_name = {subset["subset"]: subset for subset in report["subsets"]}
            self.assertEqual(by_name["all"]["outside_union_count"], 2)
            self.assertEqual(by_name["all"]["per_tile_count"], {"Tile_0": 1, "Tile_1": 1})
            self.assertEqual(by_name["all"]["outside_over_footprint_above_boxes_count"], 1)
            self.assertEqual(by_name["all"]["outside_by_horizontal_distance"][1]["count"], 1)  # 3 m band
            self.assertEqual(by_name["opacity_ge_0.05"]["count"], 3)
            self.assertEqual(by_name["opacity_ge_0.05"]["outside_union_count"], 1)
            # anchor within 0.6 m: row0 (0 m), row1 (0.5 m below the anchor) -> supported
            self.assertEqual(supported.tolist(), [True, True, False, False])
            self.assertEqual(by_name["lidar_unsupported_gt_0.6m"]["count"], 2)
            self.assertEqual(by_name["lidar_unsupported_gt_0.6m_opacity_ge_0.05"]["count"], 1)
            self.assertEqual(report["lidar_unsupported_location"]["over_las_footprint_count"], 1)
            self.assertEqual(report["lidar_unsupported_location"]["beyond_las_xy_extent_count"], 1)
            self.assertEqual(report["anchor"]["point_count"], 2)

            png = root / "topdown.png"
            render_topdown(
                xyz, owner, tile_boxes(json.loads(manifest.read_text(encoding="utf-8")), "core_box"),
                png, cell_m=0.5, title="t", highlight=~supported,
            )
            self.assertGreater(png.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
