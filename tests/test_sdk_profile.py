"""The profile contract: frozen, hashed, path-free and fully justified.

A profile is the only thing standing between "the recipe we measured" and
"whatever was in the config directory that week". These tests pin the three
properties that claim depends on: it cannot be edited in place, its sha256
moves when any value moves, and every section says who measured it.
"""

from __future__ import annotations

import unittest

from cloudstudio3dgs_sdk.profile import (
    CONFIDENCE_LEVELS,
    MEASURED,
    PROFILE_B5FILL2,
    PROFILES,
    Profile,
    Provenance,
    UNMEASURED,
    freeze,
    get_profile,
    make_profile,
    thaw,
)


def _variant(**overrides) -> Profile:
    """The same profile with one section replaced; for sha comparisons."""
    payload = PROFILE_B5FILL2.as_dict()
    provenance = dict(PROFILE_B5FILL2.provenance)
    payload.pop("provenance")
    payload.update(overrides)
    return make_profile(provenance=provenance, **payload)


class ProfileImmutabilityTests(unittest.TestCase):
    def test_attributes_cannot_be_rebound(self) -> None:
        with self.assertRaises(Exception):
            PROFILE_B5FILL2.name = "other"  # type: ignore[misc]

    def test_top_level_section_cannot_be_edited_in_place(self) -> None:
        with self.assertRaises(TypeError):
            PROFILE_B5FILL2.trainer_base["cap_max"] = 1  # type: ignore[index]

    def test_nested_block_cannot_be_edited_in_place(self) -> None:
        with self.assertRaises(TypeError):
            PROFILE_B5FILL2.trainer_base["default_strategy"]["reset_every"] = 3000  # type: ignore[index]
        with self.assertRaises(TypeError):
            PROFILE_B5FILL2.merge["fill"]["occupancy_clearance_voxels"] = 2  # type: ignore[index]

    def test_lists_are_tuples(self) -> None:
        self.assertIsInstance(PROFILE_B5FILL2.trainer_base["background_color"], tuple)
        self.assertIsInstance(PROFILE_B5FILL2.export["threshold_control"], tuple)
        with self.assertRaises(AttributeError):
            PROFILE_B5FILL2.export["threshold_control"].append(0.1)  # type: ignore[attr-defined]

    def test_as_dict_is_a_detached_copy(self) -> None:
        before = PROFILE_B5FILL2.profile_sha256
        payload = PROFILE_B5FILL2.as_dict()
        payload["trainer_base"]["cap_max"] = 999
        payload["merge"]["fill"]["occupancy_voxel_m"] = 99.0
        self.assertEqual(PROFILE_B5FILL2.profile_sha256, before)
        self.assertNotIn("cap_max", PROFILE_B5FILL2.trainer_base)

    def test_freeze_rejects_non_json(self) -> None:
        with self.assertRaises(TypeError):
            freeze({"when": object()})

    def test_freeze_thaw_round_trip(self) -> None:
        original = {"a": [1, {"b": 2}], "c": None}
        self.assertEqual(thaw(freeze(original)), original)


class ProfileShaTests(unittest.TestCase):
    def test_sha_is_deterministic(self) -> None:
        self.assertEqual(PROFILE_B5FILL2.profile_sha256, PROFILE_B5FILL2.profile_sha256)
        self.assertEqual(len(PROFILE_B5FILL2.profile_sha256), 64)

    def test_sha_moves_when_a_knob_moves(self) -> None:
        trainer = PROFILE_B5FILL2.as_dict()["trainer_base"]
        trainer["default_strategy"]["reset_every"] = 3000
        changed = _variant(trainer_base=trainer)
        self.assertNotEqual(changed.profile_sha256, PROFILE_B5FILL2.profile_sha256)

    def test_sha_moves_when_only_the_justification_changes(self) -> None:
        provenance = dict(PROFILE_B5FILL2.provenance)
        provenance["trainer_base"] = Provenance("rewritten claim", "somewhere", MEASURED)
        payload = PROFILE_B5FILL2.as_dict()
        payload.pop("provenance")
        changed = make_profile(provenance=provenance, **payload)
        self.assertNotEqual(changed.profile_sha256, PROFILE_B5FILL2.profile_sha256)

    def test_identity_carries_name_version_and_sha(self) -> None:
        identity = PROFILE_B5FILL2.identity()
        self.assertEqual(identity["profile"], "b5fill2")
        self.assertEqual(identity["profile_sha256"], PROFILE_B5FILL2.profile_sha256)


class ProvenanceTests(unittest.TestCase):
    def test_every_section_has_a_justification(self) -> None:
        for section in Profile.SECTIONS:
            with self.subTest(section=section):
                self.assertIsInstance(PROFILE_B5FILL2.why(section), Provenance)

    def test_load_bearing_knobs_are_justified_individually(self) -> None:
        # Each of these was a campaign in its own right; the group-level
        # justification is not enough to reconstruct why it holds its value.
        for knob in (
            "trainer_base.default_strategy.reset_every",
            "trainer_base.default_strategy.detail_split_scale_m",
            "trainer_base.geometry_regularization.anisotropy_weight",
            "trainer_base.lidar_range_weight",
            "trainer_base.sky_supervision",
            "trainer_base.tile_ownership_masking",
            "trainer_base.controlled_stop_after_steps",
            "tile_rules.cap_ratio_of_initialisation",
            "merge.fill",
            "runtime.max_gaussians_per_gib_vram",
        ):
            with self.subTest(knob=knob):
                self.assertIn(knob, PROFILE_B5FILL2.provenance)

    def test_why_falls_back_to_the_nearest_prefix(self) -> None:
        nested = PROFILE_B5FILL2.why("merge.fill.occupancy_voxel_m")
        self.assertIs(nested, PROFILE_B5FILL2.provenance["merge.fill"])

    def test_why_raises_for_an_unknown_section(self) -> None:
        with self.assertRaises(KeyError):
            PROFILE_B5FILL2.why("not_a_section")

    def test_unmeasured_knobs_are_reported(self) -> None:
        unmeasured = PROFILE_B5FILL2.unmeasured_knobs()
        self.assertIn("tile_rules.seed_generation_overrides", unmeasured)
        self.assertIn("cost_model.sky_mask_seconds_per_face", unmeasured)
        for knob in unmeasured:
            self.assertIn(PROFILE_B5FILL2.why(knob).confidence, ("inferred", UNMEASURED))

    def test_confidence_values_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            Provenance("claim", "source", "probably")
        for level in CONFIDENCE_LEVELS:
            Provenance("claim", "source", level)

    def test_every_provenance_names_a_source(self) -> None:
        for key, why in PROFILE_B5FILL2.provenance.items():
            with self.subTest(key=key):
                self.assertTrue(why.source.strip())
                self.assertTrue(why.claim.strip())


class ProfileShapeTests(unittest.TestCase):
    def test_profile_holds_no_absolute_paths(self) -> None:
        """Paths belong to a dataset, not to a recipe."""

        def walk(value, trail: str):
            if isinstance(value, str):
                self.assertFalse(
                    value[1:3] == ":\\" or value[1:3] == ":/" or value.startswith(("/", "\\\\")),
                    f"{trail} looks like an absolute path: {value!r}",
                )
            elif isinstance(value, dict):
                for key, item in value.items():
                    walk(item, f"{trail}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    walk(item, f"{trail}[{index}]")

        walk(PROFILE_B5FILL2.as_dict(), "profile")

    def test_external_assets_never_ship(self) -> None:
        for asset in PROFILE_B5FILL2.external_assets:
            with self.subTest(asset=asset["id"]):
                self.assertFalse(asset["ships_in_delivery"])
        segformer = next(a for a in PROFILE_B5FILL2.external_assets if a["id"] == "segformer_sky")
        self.assertIn("non-commercial", segformer["license"].lower())
        self.assertIn("supervision masks", segformer["license_note"])

    def test_open_questions_are_structured(self) -> None:
        self.assertTrue(PROFILE_B5FILL2.open_questions)
        ids = {item["id"] for item in PROFILE_B5FILL2.open_questions}
        self.assertIn("backdrop-bootstrap", ids)
        self.assertIn("pipeline-fill-passthrough", ids)
        for item in PROFILE_B5FILL2.open_questions:
            for key in ("id", "what", "expressed_as", "status"):
                self.assertIn(key, item)

    def test_registry_lookup(self) -> None:
        self.assertIs(get_profile("b5fill2"), PROFILE_B5FILL2)
        with self.assertRaises(KeyError):
            get_profile("nope")
        with self.assertRaises(TypeError):
            PROFILES["extra"] = PROFILE_B5FILL2  # type: ignore[index]

    def test_make_profile_rejects_unknown_fields(self) -> None:
        payload = PROFILE_B5FILL2.as_dict()
        provenance = dict(PROFILE_B5FILL2.provenance)
        payload.pop("provenance")
        payload["surprise"] = 1
        with self.assertRaises(TypeError):
            make_profile(provenance=provenance, **payload)

    def test_make_profile_requires_provenance_objects(self) -> None:
        payload = PROFILE_B5FILL2.as_dict()
        payload.pop("provenance")
        with self.assertRaises(TypeError):
            make_profile(provenance={"runtime": "a string"}, **payload)


if __name__ == "__main__":
    unittest.main()
