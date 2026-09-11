"""Face hold-out: whole Face4 faces leave the Tile view set but stay in the epoch basis."""

from __future__ import annotations

import unittest

from cloudstudio_3dgs.training.holdout import select_face_holdout


class FaceHoldoutTests(unittest.TestCase):
    def test_faces_are_split_in_order_and_everything_else_is_kept(self):
        views = [{"sample_id": f"img_{i}::{face}", "crop": i} for i in range(3) for face in ("pitch_up_56", "yaw_minus_35")]
        kept, held = select_face_holdout(views, ["pitch_up_56"])
        self.assertEqual([v["sample_id"] for v in held], [f"img_{i}::pitch_up_56" for i in range(3)])
        self.assertEqual([v["sample_id"] for v in kept], [f"img_{i}::yaw_minus_35" for i in range(3)])
        self.assertEqual(kept[1]["crop"], 1)

    def test_holding_out_every_view_is_refused(self):
        views = [{"sample_id": "img_0::pitch_up_56"}]
        with self.assertRaises(ValueError):
            select_face_holdout(views, ["pitch_up_56"])

    def test_config_contract_key_only_when_faces_are_held_out(self):
        from cloudstudio_3dgs.training.trainer import TrainerConfig

        base = TrainerConfig.from_dict({"run_id": "a", "output_dir": "C:/tmp/a", "max_steps": 100})
        self.assertEqual(base.holdout_face_ids, ())
        contract = base.contract_dict() if hasattr(base, "contract_dict") else None
        if contract is not None:
            self.assertNotIn("face_holdout", contract.get("view_sampling", {}))
        cfg = TrainerConfig.from_dict({"run_id": "a", "output_dir": "C:/tmp/a", "max_steps": 100, "holdout_face_ids": ["pitch_up_56"]})
        self.assertEqual(cfg.holdout_face_ids, ("pitch_up_56",))
        with self.assertRaises(ValueError):
            cfg.validate()  # needs Tile inputs and a face cache


if __name__ == "__main__":
    unittest.main()
