from __future__ import annotations

import unittest

import numpy as np

from cloudstudio_3dgs.training.tile_ownership import (
    assign_core_owners,
    build_core_ownership_contract,
)


TILES = [
    {"tile_id": 0, "core_box": [[0, 0, 0], [1, 2, 1]]},
    {"tile_id": 1, "core_box": [[1, 0, 0], [2, 1, 1]]},
    {"tile_id": 2, "core_box": [[1, 1, 0], [2, 2, 1]]},
]


class TileOwnershipTests(unittest.TestCase):
    def test_shared_boundaries_have_one_deterministic_owner(self) -> None:
        points = np.asarray(
            [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 0.5], [1, 1, 0.5]],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(assign_core_owners(points, TILES), [0, 1, 2, 0])

    def test_outside_points_and_positive_volume_overlap_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "no core owner"):
            assign_core_owners(np.asarray([[3, 0, 0]], dtype=np.float64), TILES)
        overlap = [
            {"tile_id": 0, "core_box": [[0, 0, 0], [1.1, 1, 1]]},
            {"tile_id": 1, "core_box": [[1, 0, 0], [2, 1, 1]]},
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            assign_core_owners(np.asarray([[0.5, 0.5, 0.5]]), overlap)

    def test_contract_forbids_direct_halo_concatenation(self) -> None:
        contract = build_core_ownership_contract(
            tiles=TILES, tile_inputs_manifest_sha256="a" * 64
        )
        self.assertEqual(contract["export_scope"], "core_owner_only")
        self.assertFalse(contract["direct_halo_concatenation_allowed"])
        self.assertEqual(len(contract["ownership_contract_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
