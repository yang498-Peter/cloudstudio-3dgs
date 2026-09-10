"""Per-tile arm overrides for a delivery.

tile3 of the R1d delivery crashed under cap 15M and was retried as a new arm
at cap 13M (the frozen-config rule forbids reusing the name). The delivery
must be able to merge tiles 1-2 from the tag pattern and tile 3 from the
retried arm without renaming or copying checkpoints.
"""

from __future__ import annotations

import unittest

from tools.pipeline import PipelineError, parse_tile_overrides, resolve_delivery_tile_arms


class _Cfg:
    delivery_tiles = [1, 2, 3]

    @staticmethod
    def delivery_tile_arm(tag: str, tile: int) -> str:
        return f"tile{tile}_{tag}_20k"


class TileOverrideTests(unittest.TestCase):
    def test_parse_accepts_tile_equals_arm(self):
        self.assertEqual(parse_tile_overrides(["3=tile3_R1d_cap13m_20k"]), {3: "tile3_R1d_cap13m_20k"})
        self.assertEqual(parse_tile_overrides([]), {})

    def test_parse_rejects_malformed_and_duplicates(self):
        for bad in (["3"], ["x=arm"], ["3="], ["3=a", "3=b"]):
            with self.assertRaises(PipelineError):
                parse_tile_overrides(bad)

    def test_override_replaces_only_the_named_tile(self):
        arms = resolve_delivery_tile_arms(_Cfg(), "R1d", {3: "tile3_R1d_cap13m_20k"})
        self.assertEqual(arms, {1: "tile1_R1d_20k", 2: "tile2_R1d_20k", 3: "tile3_R1d_cap13m_20k"})

    def test_override_for_a_non_delivery_tile_is_refused(self):
        with self.assertRaises(PipelineError):
            resolve_delivery_tile_arms(_Cfg(), "R1d", {7: "tile7_x"})


if __name__ == "__main__":
    unittest.main()
